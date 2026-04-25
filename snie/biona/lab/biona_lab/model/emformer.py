"""Biona Emformer — streaming speech recognition model.

CRITICAL CONTRACT — tensor names and shape constants must match exactly:
  biona/axon/core/include/biona/core/onnx_contract.hpp

Any deviation causes a silent runtime failure in Biona Axon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

# ══ ONNX CONTRACT CONSTANTS — must match onnx_contract.hpp ══
MEL_BANDS       = 80    # ONNX_MEL_BANDS  / AUDIO_MEL_BANDS
EMBEDDING_DIM   = 512   # ONNX_EMBEDDING_DIM
MEMORY_VECTORS  = 4     # ONNX_MEMORY_VECTORS
LC_FRAMES       = 64    # ONNX_LEFT_CONTEXT_FRAMES
CHUNK_FRAMES    = 2     # ONNX_CHUNK_FRAMES
OPSET           = 17    # ONNX_OPSET_VERSION

# Input / output tensor names (must match string constants in onnx_contract.hpp)
ONNX_INPUT_NAMES  = ["chunk", "memory", "left_context_key", "left_context_val"]
ONNX_OUTPUT_NAMES = ["logits", "memory_out", "lc_key_out", "lc_val_out", "embedding"]
# ════════════════════════════════════════════════════════════

# Fixed Emformer architecture constants
_NUM_HEADS   = 8
_FFN_DIM     = 2048
_NUM_LAYERS  = 20
_DROPOUT     = 0.1

# torchaudio Emformer exposes left-context state as (key, val) per layer.
# Each has shape [B, LC_FRAMES, head_dim] where head_dim = encoder_dim / num_heads.
# encoder_dim for torchaudio Emformer with input_dim=80 equals input_dim (no projection
# inside the encoder) — confirmed by torchaudio source.
_ENCODER_DIM = MEL_BANDS   # torchaudio Emformer output dim == input_dim


class BionaEmformer(nn.Module):
    """
    Streaming Emformer encoder with CTC head and acoustic embedding projection.

    forward(chunk, memory, left_context_key, left_context_val)
        -> (logits, memory_out, lc_key_out, lc_val_out, embedding)
    """

    def __init__(self, vocab_size: int = 29) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        self.encoder = torchaudio.models.Emformer(
            input_dim=MEL_BANDS,
            num_heads=_NUM_HEADS,
            ffn_dim=_FFN_DIM,
            num_layers=_NUM_LAYERS,
            segment_length=CHUNK_FRAMES,
            left_context_length=LC_FRAMES,
            right_context_length=0,       # zero look-ahead
            memory_size=MEMORY_VECTORS,
            dropout=_DROPOUT,
        )

        # Acoustic embedding: encoder_dim → EMBEDDING_DIM
        self.embedding_proj = nn.Linear(_ENCODER_DIM, EMBEDDING_DIM)

        # CTC output head: encoder_dim → vocab_size
        self.ctc_head = nn.Linear(_ENCODER_DIM, vocab_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        chunk: torch.Tensor,              # [B, CHUNK_FRAMES, MEL_BANDS]
        memory: torch.Tensor,             # [B, MEMORY_VECTORS, _ENCODER_DIM]
        left_context_key: torch.Tensor,   # [n_layers, B, LC_FRAMES, head_dim]
        left_context_val: torch.Tensor,   # [n_layers, B, LC_FRAMES, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits        : [CHUNK_FRAMES, B, vocab_size]
        memory_out    : [B, MEMORY_VECTORS, _ENCODER_DIM]
        lc_key_out    : [n_layers, B, LC_FRAMES, head_dim]
        lc_val_out    : [n_layers, B, LC_FRAMES, head_dim]
        embedding     : [B, EMBEDDING_DIM]
        """
        B = chunk.size(0)

        # torchaudio Emformer streaming forward expects:
        #   input        : [B, T, input_dim]
        #   lengths      : [B]  (all == T for fixed-size chunks)
        #   state        : list of [memory, lc_key, lc_val] per layer,
        #                  OR the flattened state list torchaudio uses internally.
        #
        # torchaudio Emformer.infer() is the streaming API:
        #   infer(input, lengths, state) -> (output, lengths, state)
        # where state is List[List[Tensor]].

        lengths = torch.full((B,), CHUNK_FRAMES, dtype=torch.int32, device=chunk.device)

        # Rebuild torchaudio state list from flat contract tensors
        state = self._contract_to_state(memory, left_context_key, left_context_val)

        # [B, T, encoder_dim], lengths, new_state
        enc_out, _, new_state = self.encoder.infer(chunk, lengths, state)

        # CTC logits: [T, B, vocab_size]
        logits = self.ctc_head(enc_out).permute(1, 0, 2)

        # Embedding: mean-pool over time → [B, EMBEDDING_DIM]
        embedding = self.embedding_proj(enc_out.mean(dim=1))

        # Unpack new state back to contract tensors
        memory_out, lc_key_out, lc_val_out = self._state_to_contract(new_state, B, chunk.device)

        return logits, memory_out, lc_key_out, lc_val_out, embedding

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_initial_state(
        self, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return zero-initialised (memory, left_context_key, left_context_val)
        with shapes matching the ONNX contract.
        """
        head_dim = _ENCODER_DIM // _NUM_HEADS
        memory = torch.zeros(batch_size, MEMORY_VECTORS, _ENCODER_DIM)
        lc_key = torch.zeros(_NUM_LAYERS, batch_size, LC_FRAMES, head_dim)
        lc_val = torch.zeros(_NUM_LAYERS, batch_size, LC_FRAMES, head_dim)
        return memory, lc_key, lc_val

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, vocab_size: int = 29) -> "BionaEmformer":
        """Construct model with the fixed Biona architecture config."""
        return cls(vocab_size=vocab_size)

    # ------------------------------------------------------------------
    # Internal state conversion helpers
    # ------------------------------------------------------------------

    def _contract_to_state(
        self,
        memory: torch.Tensor,
        lc_key: torch.Tensor,
        lc_val: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """
        Convert flat contract tensors to the nested state list that
        torchaudio.models.Emformer.infer() expects.

        torchaudio Emformer state format (from source):
            List[ [memory_i, lc_key_i, lc_val_i] ] for each layer i
        where:
            memory_i  : [B, memory_size, encoder_dim]
            lc_key_i  : [B, LC_FRAMES, head_dim]
            lc_val_i  : [B, LC_FRAMES, head_dim]
        """
        state: list[list[torch.Tensor]] = []
        for i in range(_NUM_LAYERS):
            # memory is shared across layers in our contract tensor;
            # torchaudio maintains per-layer memory internally.
            state.append([
                memory,          # [B, MEMORY_VECTORS, _ENCODER_DIM]
                lc_key[i],       # [B, LC_FRAMES, head_dim]
                lc_val[i],       # [B, LC_FRAMES, head_dim]
            ])
        return state

    @staticmethod
    def _state_to_contract(
        state: list[list[torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert torchaudio nested state back to flat contract tensors.
        Use the last layer's memory as the memory_out (representative).
        """
        head_dim = _ENCODER_DIM // _NUM_HEADS

        lc_keys = []
        lc_vals = []
        memory_out = None

        for layer_state in state:
            mem_i, lc_k_i, lc_v_i = layer_state
            memory_out = mem_i          # overwrite each time; keep last layer
            lc_keys.append(lc_k_i)     # [B, LC_FRAMES, head_dim]
            lc_vals.append(lc_v_i)

        # Stack layers: [n_layers, B, LC_FRAMES, head_dim]
        lc_key_out = torch.stack(lc_keys, dim=0)
        lc_val_out = torch.stack(lc_vals, dim=0)

        return memory_out, lc_key_out, lc_val_out
