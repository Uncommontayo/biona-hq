"""CLI: export a trained BionaEmformer checkpoint to an encrypted .snie bundle.

Usage:
    python scripts/export_model.py \\
        --checkpoint /models/snie/checkpoint_00200000.pt \\
        --output /models/snie/snie_v1.0.0.snie

Requires environment variables:
    BIONA_MODEL_ENCRYPTION_KEY  — hex-encoded 256-bit AES key
    BIONA_MODEL_HMAC_KEY        — hex-encoded 256-bit HMAC key
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

from biona_lab.model.emformer import BionaEmformer
from biona_lab.model.export import (
    _sha256_file,
    encrypt_bundle,
    export_to_onnx,
    verify_onnx_export,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and encrypt a Biona Emformer checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_XXXXXXXX.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the encrypted .snie bundle",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Require encryption keys from environment
    enc_key_hex  = os.environ.get("BIONA_MODEL_ENCRYPTION_KEY", "")
    hmac_key_hex = os.environ.get("BIONA_MODEL_HMAC_KEY", "")
    if not enc_key_hex or not hmac_key_hex:
        logger.error(
            "BIONA_MODEL_ENCRYPTION_KEY and BIONA_MODEL_HMAC_KEY must be set."
        )
        sys.exit(1)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    # ── Step 1: Load checkpoint ──────────────────────────────────────
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    vocab_size = cfg.get("vocab_size", 29)
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", float("inf"))

    model = BionaEmformer.from_config(vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded model — step=%d, val_loss=%.4f, vocab_size=%d", step, val_loss, vocab_size)

    # ── Step 2: Export to ONNX ───────────────────────────────────────
    onnx_path = args.output.with_suffix(".onnx")
    export_to_onnx(model, onnx_path, opset=args.opset)

    # ── Step 3: Verify export — hard fail if contract violated ───────
    logger.info("Verifying ONNX export against contract…")
    try:
        verify_onnx_export(onnx_path)
    except RuntimeError as exc:
        logger.error("ONNX verification FAILED: %s", exc)
        sys.exit(1)

    # ── Step 4: Encrypt bundle ───────────────────────────────────────
    bundle_path = args.output
    encrypt_bundle(onnx_path, bundle_path, enc_key_hex, hmac_key_hex)

    # ── Step 5: Write export manifest ───────────────────────────────
    manifest = {
        "onnx_sha256":   _sha256_file(onnx_path),
        "bundle_sha256": _sha256_file(bundle_path),
        "model_config":  cfg,
        "checkpoint":    str(checkpoint_path),
        "step":          step,
        "val_loss":      val_loss,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = bundle_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Export manifest: %s", manifest_path)
    logger.info("Bundle: %s", bundle_path)
    logger.info("Export complete.")


if __name__ == "__main__":
    main()
