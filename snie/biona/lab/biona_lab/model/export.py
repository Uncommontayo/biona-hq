"""ONNX export and encryption for Biona Lab.

THIS IS THE MOST CRITICAL FILE IN BIONA LAB.
The exported model must satisfy the ONNX contract in:
  biona/axon/core/include/biona/core/onnx_contract.hpp

Bundle layout (must match EncryptedModelBundle in Biona Axon):
  magic(4) + version(4) + iv(12) + tag(16) + hmac(32) + ciphertext
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import os
import struct
import time
from pathlib import Path

import torch

from biona_lab.model.emformer import (
    CHUNK_FRAMES,
    EMBEDDING_DIM,
    LC_FRAMES,
    MEL_BANDS,
    MEMORY_VECTORS,
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    OPSET,
    BionaEmformer,
    _ENCODER_DIM,
    _NUM_HEADS,
    _NUM_LAYERS,
)

logger = logging.getLogger(__name__)

# Bundle format constants
_BUNDLE_MAGIC   = b"SNIE"          # 4 bytes
_BUNDLE_VERSION = 1                # uint32 LE
_AES_KEY_BYTES  = 32               # AES-256
_GCM_IV_BYTES   = 12
_GCM_TAG_BYTES  = 16
_HMAC_BYTES     = 32               # HMAC-SHA256

HEAD_DIM = _ENCODER_DIM // _NUM_HEADS


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: BionaEmformer,
    output_path: Path,
    opset: int = OPSET,
) -> Path:
    """
    Export streaming BionaEmformer to ONNX.

    Tensor names and shapes match onnx_contract.hpp exactly.
    Returns the path to the exported .onnx file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy inputs matching the ONNX contract
    chunk            = torch.zeros(1, CHUNK_FRAMES, MEL_BANDS)           # [B, 2, 80]
    memory           = torch.zeros(1, MEMORY_VECTORS, _ENCODER_DIM)      # [B, 4, 80]
    left_context_key = torch.zeros(_NUM_LAYERS, 1, LC_FRAMES, HEAD_DIM)  # [20, B, 64, 10]
    left_context_val = torch.zeros(_NUM_LAYERS, 1, LC_FRAMES, HEAD_DIM)

    logger.info("Exporting ONNX model to %s (opset %d)…", output_path, opset)

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(chunk, memory, left_context_key, left_context_val),
            f=str(output_path),
            input_names=ONNX_INPUT_NAMES,
            output_names=ONNX_OUTPUT_NAMES,
            dynamic_axes={
                "chunk":     {0: "batch"},
                "logits":    {0: "batch", 1: "T"},
                "embedding": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
        )

    logger.info("ONNX export complete: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx_export(onnx_path: Path) -> bool:
    """
    Load the exported model with onnxruntime, run one inference,
    and verify all output tensor names and shapes match the contract.

    Returns True on success, raises RuntimeError on any mismatch.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        logger.warning(
            "onnxruntime not installed — skipping ONNX verification. "
            "Install with: pip install onnxruntime"
        )
        return True

    logger.info("Verifying ONNX export: %s", onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Check input names
    actual_inputs = [inp.name for inp in sess.get_inputs()]
    for expected in ONNX_INPUT_NAMES:
        if expected not in actual_inputs:
            raise RuntimeError(
                f"ONNX contract violation: expected input '{expected}' "
                f"not found. Got: {actual_inputs}"
            )

    # Check output names
    actual_outputs = [out.name for out in sess.get_outputs()]
    for expected in ONNX_OUTPUT_NAMES:
        if expected not in actual_outputs:
            raise RuntimeError(
                f"ONNX contract violation: expected output '{expected}' "
                f"not found. Got: {actual_outputs}"
            )

    # Run one inference with dummy data
    import numpy as np
    feeds = {
        "chunk":            np.zeros((1, CHUNK_FRAMES, MEL_BANDS),        dtype=np.float32),
        "memory":           np.zeros((1, MEMORY_VECTORS, _ENCODER_DIM),   dtype=np.float32),
        "left_context_key": np.zeros((_NUM_LAYERS, 1, LC_FRAMES, HEAD_DIM), dtype=np.float32),
        "left_context_val": np.zeros((_NUM_LAYERS, 1, LC_FRAMES, HEAD_DIM), dtype=np.float32),
    }
    outputs = sess.run(None, feeds)
    output_map = dict(zip(actual_outputs, outputs))

    # Shape checks
    logits    = output_map["logits"]
    memory_o  = output_map["memory_out"]
    lc_key_o  = output_map["lc_key_out"]
    lc_val_o  = output_map["lc_val_out"]
    embedding = output_map["embedding"]

    def _check(name, arr, expected_shape):
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"ONNX contract violation: '{name}' shape {arr.shape} "
                f"!= expected {expected_shape}"
            )

    _check("logits",        logits,    (CHUNK_FRAMES, 1, 29))  # 29 = default vocab
    _check("memory_out",    memory_o,  (1, MEMORY_VECTORS, _ENCODER_DIM))
    _check("lc_key_out",    lc_key_o,  (_NUM_LAYERS, 1, LC_FRAMES, HEAD_DIM))
    _check("lc_val_out",    lc_val_o,  (_NUM_LAYERS, 1, LC_FRAMES, HEAD_DIM))
    _check("embedding",     embedding, (1, EMBEDDING_DIM))

    logger.info("ONNX verification passed — all tensor names and shapes correct.")
    return True


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

def encrypt_bundle(
    onnx_path: Path,
    output_path: Path,
    encryption_key_hex: str,
    hmac_key_hex: str,
) -> Path:
    """
    Encrypt an ONNX file into a .snie bundle using AES-256-GCM.

    Bundle layout (matches EncryptedModelBundle in Biona Axon):
      magic(4) | version(4 LE uint32) | iv(12) | tag(16) | hmac(32) | ciphertext

    Returns output_path.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    except ImportError:
        raise ImportError(
            "cryptography package required for bundle encryption. "
            "Install with: pip install cryptography"
        )

    enc_key = bytes.fromhex(encryption_key_hex)
    hmac_key = bytes.fromhex(hmac_key_hex)

    if len(enc_key) != _AES_KEY_BYTES:
        raise ValueError(f"Encryption key must be {_AES_KEY_BYTES} bytes (256-bit)")
    if len(hmac_key) != _AES_KEY_BYTES:
        raise ValueError(f"HMAC key must be {_AES_KEY_BYTES} bytes (256-bit)")

    plaintext = Path(onnx_path).read_bytes()

    # Generate a random 12-byte IV
    iv = os.urandom(_GCM_IV_BYTES)

    # AES-256-GCM encrypt (returns ciphertext + 16-byte tag appended)
    aesgcm = AESGCM(enc_key)
    ct_with_tag = aesgcm.encrypt(iv, plaintext, None)
    ciphertext = ct_with_tag[:-_GCM_TAG_BYTES]
    tag        = ct_with_tag[-_GCM_TAG_BYTES:]

    # HMAC-SHA256 over (magic + version_bytes + iv + tag + ciphertext)
    version_bytes = struct.pack("<I", _BUNDLE_VERSION)
    mac_data = _BUNDLE_MAGIC + version_bytes + iv + tag + ciphertext
    mac = _hmac.new(hmac_key, mac_data, hashlib.sha256).digest()

    # Write bundle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(_BUNDLE_MAGIC)
        fh.write(version_bytes)
        fh.write(iv)
        fh.write(tag)
        fh.write(mac)
        fh.write(ciphertext)

    logger.info(
        "Bundle encrypted: %s (%d bytes plaintext → %d bytes bundle)",
        output_path, len(plaintext), output_path.stat().st_size,
    )
    return output_path


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()
