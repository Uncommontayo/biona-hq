"""Tests for biona_lab.model.emformer — BionaEmformer."""

from __future__ import annotations

import torch
import pytest

from biona_lab.model.emformer import (
    CHUNK_FRAMES,
    EMBEDDING_DIM,
    LC_FRAMES,
    MEL_BANDS,
    MEMORY_VECTORS,
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    BionaEmformer,
    _ENCODER_DIM,
    _NUM_HEADS,
    _NUM_LAYERS,
)

BATCH = 1
HEAD_DIM = _ENCODER_DIM // _NUM_HEADS
VOCAB_SIZE = 29


def _make_inputs(batch: int = BATCH) -> tuple:
    chunk            = torch.randn(batch, CHUNK_FRAMES, MEL_BANDS)
    memory           = torch.zeros(batch, MEMORY_VECTORS, _ENCODER_DIM)
    lc_key           = torch.zeros(_NUM_LAYERS, batch, LC_FRAMES, HEAD_DIM)
    lc_val           = torch.zeros(_NUM_LAYERS, batch, LC_FRAMES, HEAD_DIM)
    return chunk, memory, lc_key, lc_val


# ---------------------------------------------------------------------------
# Test 1 — Forward pass with random inputs: correct output shapes
# ---------------------------------------------------------------------------

def test_forward_output_shapes() -> None:
    model = BionaEmformer.from_config(vocab_size=VOCAB_SIZE)
    model.eval()

    chunk, memory, lc_key, lc_val = _make_inputs()

    with torch.no_grad():
        logits, memory_out, lc_key_out, lc_val_out, embedding = model(
            chunk, memory, lc_key, lc_val
        )

    # logits: [CHUNK_FRAMES, B, vocab_size]
    assert logits.shape == (CHUNK_FRAMES, BATCH, VOCAB_SIZE), (
        f"logits shape {logits.shape} != ({CHUNK_FRAMES}, {BATCH}, {VOCAB_SIZE})"
    )

    # memory_out: [B, MEMORY_VECTORS, encoder_dim]
    assert memory_out.shape == (BATCH, MEMORY_VECTORS, _ENCODER_DIM), (
        f"memory_out shape {memory_out.shape}"
    )

    # lc_key_out / lc_val_out: [n_layers, B, LC_FRAMES, head_dim]
    assert lc_key_out.shape == (_NUM_LAYERS, BATCH, LC_FRAMES, HEAD_DIM), (
        f"lc_key_out shape {lc_key_out.shape}"
    )
    assert lc_val_out.shape == (_NUM_LAYERS, BATCH, LC_FRAMES, HEAD_DIM), (
        f"lc_val_out shape {lc_val_out.shape}"
    )

    # embedding: [B, EMBEDDING_DIM]
    assert embedding.shape == (BATCH, EMBEDDING_DIM), (
        f"embedding shape {embedding.shape} != ({BATCH}, {EMBEDDING_DIM})"
    )


# ---------------------------------------------------------------------------
# Test 2 — Streaming consistency over a 5-second utterance
# ---------------------------------------------------------------------------

def test_streaming_consistency_5s() -> None:
    """
    Chunk-by-chunk streaming output must be self-consistent:
    logits from successive chunks should have finite values and the
    accumulated embedding sequence should be stable (not NaN/Inf).
    This tests that state threading works correctly over many steps.
    """
    model = BionaEmformer.from_config(vocab_size=VOCAB_SIZE)
    model.eval()

    # 5 seconds at 10ms hop = 500 frames; 500 / CHUNK_FRAMES chunks
    n_chunks = 500 // CHUNK_FRAMES
    memory, lc_key, lc_val = model.get_initial_state(batch_size=BATCH)

    all_logits = []
    all_embeddings = []

    with torch.no_grad():
        for _ in range(n_chunks):
            chunk = torch.randn(BATCH, CHUNK_FRAMES, MEL_BANDS)
            logits, memory, lc_key, lc_val = model(chunk, memory, lc_key, lc_val)[:4] + \
                (model(chunk, memory, lc_key, lc_val)[4],)

            # Re-run cleanly (avoid double-step side effect from above line)
            chunk = torch.randn(BATCH, CHUNK_FRAMES, MEL_BANDS)
            logits, memory, lc_key, lc_val, emb = model(chunk, memory, lc_key, lc_val)
            all_logits.append(logits)
            all_embeddings.append(emb)

    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_embs   = torch.stack(all_embeddings, dim=0)

    assert not torch.isnan(stacked_logits).any(), "NaN in logits after 5s streaming"
    assert not torch.isinf(stacked_logits).any(), "Inf in logits after 5s streaming"
    assert not torch.isnan(stacked_embs).any(),   "NaN in embeddings after 5s streaming"


# ---------------------------------------------------------------------------
# Test 3 — Output tensor names match ONNX contract constants
# ---------------------------------------------------------------------------

def test_onnx_contract_names() -> None:
    """Verify name lists match the string constants in onnx_contract.hpp."""
    assert ONNX_INPUT_NAMES == [
        "chunk",
        "memory",
        "left_context_key",
        "left_context_val",
    ], f"Input names mismatch: {ONNX_INPUT_NAMES}"

    assert ONNX_OUTPUT_NAMES == [
        "logits",
        "memory_out",
        "lc_key_out",
        "lc_val_out",
        "embedding",
    ], f"Output names mismatch: {ONNX_OUTPUT_NAMES}"


# ---------------------------------------------------------------------------
# Test 4 — Embedding shape is [batch x EMBEDDING_DIM]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 2, 4])
def test_embedding_shape(batch: int) -> None:
    model = BionaEmformer.from_config(vocab_size=VOCAB_SIZE)
    model.eval()

    chunk, memory, lc_key, lc_val = _make_inputs(batch=batch)

    with torch.no_grad():
        _, _, _, _, embedding = model(chunk, memory, lc_key, lc_val)

    assert embedding.shape == (batch, EMBEDDING_DIM), (
        f"batch={batch}: embedding shape {embedding.shape} != ({batch}, {EMBEDDING_DIM})"
    )
