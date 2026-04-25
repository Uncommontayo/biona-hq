"""CTC training loop for the Biona Emformer model.

Entry point:
    python -m biona_lab.training.train [--config train_config.json]
    python -m biona_lab.training.train --resume checkpoint_00050000.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from biona_lab.model.emformer import BionaEmformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

# 29-token vocab: blank(0), a-z(1-26), apostrophe(27), space(28)
_BLANK_IDX = 0
_CHARS = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz") + ["'", " "]

def _text_to_indices(text: str) -> list[int]:
    out = []
    for ch in text:
        if ch in "abcdefghijklmnopqrstuvwxyz":
            out.append(ord(ch) - ord("a") + 1)
        elif ch == "'":
            out.append(27)
        elif ch == " ":
            out.append(28)
        # silently skip anything outside vocab
    return out


def _greedy_decode(log_probs: torch.Tensor) -> str:
    """
    Greedy CTC decode from log-probabilities [T, vocab_size].
    Collapses repeated tokens and removes blanks.
    """
    ids = log_probs.argmax(dim=-1).tolist()
    tokens: list[int] = []
    prev = _BLANK_IDX
    for t in ids:
        if t != prev and t != _BLANK_IDX:
            tokens.append(t)
        prev = t
    return "".join(
        _CHARS[t] if t < len(_CHARS) else "" for t in tokens
    )


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def _wer(ref: str, hyp: str) -> float:
    """Word error rate between two strings."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # Dynamic programming edit distance
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r):
        nd = [i + 1] + [0] * len(h)
        for j, hw in enumerate(h):
            nd[j + 1] = min(
                d[j + 1] + 1,
                nd[j] + 1,
                d[j] + (0 if rw == hw else 1),
            )
        d = nd
    return d[len(h)] / len(r)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    warmup_steps: int = 5000
    max_steps: int = 200_000
    grad_clip: float = 1.0
    checkpoint_every: int = 5000
    eval_every: int = 2500
    batch_size: int = 32
    device: str = "cuda"
    amp: bool = True
    vocab_size: int = 29
    shard_dir: str = field(
        default_factory=lambda: os.environ.get("SHARD_OUTPUT_DIR", "/data/shards")
    )
    model_output_dir: str = field(
        default_factory=lambda: os.environ.get("MODEL_OUTPUT_DIR", "/models/snie")
    )

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _decode_sample(sample: dict) -> dict | None:
    """Decode a raw WebDataset sample into tensors."""
    try:
        import io
        features = np.load(io.BytesIO(sample["audio.npy"]))   # [T x 80]
        transcript = sample["text.txt"].decode("utf-8").strip()
        return {"features": features, "transcript": transcript}
    except Exception as exc:
        logger.debug("Bad sample %s: %s", sample.get("__key__"), exc)
        return None


def _collate(batch: list[dict]) -> dict:
    """Dynamic batching: pad features and encode transcripts."""
    # Sort by descending length for CTC (not strictly required but good practice)
    batch = sorted(batch, key=lambda x: x["features"].shape[0], reverse=True)

    input_lengths = torch.tensor([b["features"].shape[0] for b in batch], dtype=torch.long)
    max_t = int(input_lengths.max())

    features = torch.zeros(len(batch), max_t, 80, dtype=torch.float32)
    for i, b in enumerate(batch):
        t = b["features"].shape[0]
        features[i, :t] = torch.from_numpy(b["features"])

    target_seqs = [_text_to_indices(b["transcript"]) for b in batch]
    target_lengths = torch.tensor([len(s) for s in target_seqs], dtype=torch.long)
    # Concatenate all targets for CTCLoss
    targets = torch.tensor(
        [idx for seq in target_seqs for idx in seq], dtype=torch.long
    )

    return {
        "features": features,               # [B, T_max, 80]
        "input_lengths": input_lengths,     # [B]
        "targets": targets,                 # [sum(target_lengths)]
        "target_lengths": target_lengths,   # [B]
    }


def _build_dataloader(
    shard_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
) -> wds.WebLoader:
    pattern = str(Path(shard_dir) / "shard-*.tar")
    dataset = (
        wds.WebDataset(pattern, resampled=shuffle)
        .shuffle(10000 if shuffle else 0)
        .map(_decode_sample)
        .select(lambda x: x is not None)
        .batched(batch_size, collation_fn=_collate, partial=False)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=4)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _build_lr_lambda(warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

_MAX_CHECKPOINTS = 3


def _save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    val_loss: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "val_loss": val_loss,
        },
        path,
    )
    logger.info("Checkpoint saved: %s", path)

    # Prune old checkpoints — keep last _MAX_CHECKPOINTS
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    for old in checkpoints[:-_MAX_CHECKPOINTS]:
        old.unlink()
        logger.debug("Removed old checkpoint: %s", old)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    val_loss = ckpt.get("val_loss", float("inf"))
    logger.info("Resumed from checkpoint %s at step %d (val_loss=%.4f)", path, step, val_loss)
    return step, val_loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: BionaEmformer,
    config: TrainConfig,
    device: torch.device,
    ctc_loss: nn.CTCLoss,
    step: int,
) -> tuple[float, float]:
    """Run one pass over dev shard. Returns (val_loss, val_wer)."""
    model.eval()
    val_loader = _build_dataloader(config.shard_dir, "dev", config.batch_size, shuffle=False)

    total_loss = 0.0
    total_wer = 0.0
    n_batches = 0
    max_val_batches = 50  # cap validation length for speed

    memory, lc_key, lc_val = model.get_initial_state(batch_size=1)
    memory = memory.to(device)
    lc_key = lc_key.to(device)
    lc_val = lc_val.to(device)

    for batch in val_loader:
        if n_batches >= max_val_batches:
            break

        features = batch["features"].to(device)          # [B, T, 80]
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        B = features.size(0)

        # Run chunk-by-chunk over the utterance
        all_logits: list[torch.Tensor] = []
        mem, lck, lcv = model.get_initial_state(batch_size=B)
        mem, lck, lcv = mem.to(device), lck.to(device), lcv.to(device)

        from biona_lab.model.emformer import CHUNK_FRAMES
        T = features.size(1)
        for start in range(0, T, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, T)
            chunk = features[:, start:end, :]
            if chunk.size(1) < CHUNK_FRAMES:
                # Pad last chunk
                pad = torch.zeros(B, CHUNK_FRAMES - chunk.size(1), 80, device=device)
                chunk = torch.cat([chunk, pad], dim=1)
            logits, mem, lck, lcv, _ = model(chunk, mem, lck, lcv)
            all_logits.append(logits)  # [CHUNK_FRAMES, B, vocab_size]

        log_probs = torch.log_softmax(
            torch.cat(all_logits, dim=0), dim=-1
        )  # [T_out, B, vocab]

        out_lengths = torch.full((B,), log_probs.size(0), dtype=torch.long, device=device)
        loss = ctc_loss(log_probs, targets, out_lengths, target_lengths)
        total_loss += loss.item()

        # WER on first item in batch (representative)
        hyp = _greedy_decode(log_probs[:, 0, :])
        ref_len = int(target_lengths[0].item())
        ref_start = int(target_lengths[:0].sum().item()) if n_batches > 0 else 0
        ref_ids = targets[ref_start : ref_start + ref_len].tolist()
        ref = "".join(_CHARS[i] if i < len(_CHARS) else "" for i in ref_ids)
        total_wer += _wer(ref, hyp)

        n_batches += 1

    model.train()
    if n_batches == 0:
        return float("inf"), float("inf")
    return total_loss / n_batches, total_wer / n_batches


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: TrainConfig, resume_path: Path | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Optional wandb
    try:
        import wandb  # type: ignore
        wandb.init(project="biona-lab", config=asdict(config))
        use_wandb = True
        logger.info("wandb enabled")
    except ImportError:
        use_wandb = False

    device_str = config.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    model = BionaEmformer.from_config(vocab_size=config.vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = LambdaLR(
        optimizer,
        _build_lr_lambda(config.warmup_steps, config.max_steps),
    )
    ctc_loss = nn.CTCLoss(blank=_BLANK_IDX, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device_str == "cuda")

    output_dir = Path(config.model_output_dir)

    start_step = 0
    last_val_loss = float("inf")
    if resume_path is not None:
        start_step, last_val_loss = _load_checkpoint(resume_path, model, optimizer)
        # Fast-forward scheduler
        for _ in range(start_step):
            scheduler.step()

    train_loader = _build_dataloader(config.shard_dir, "train", config.batch_size, shuffle=True)
    loader_iter: Iterator = iter(train_loader)

    model.train()
    step = start_step
    t0 = time.monotonic()

    logger.info(
        "Training start — device=%s, amp=%s, steps=%d→%d",
        device_str, config.amp, start_step, config.max_steps,
    )

    while step < config.max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        B = features.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.amp and device_str == "cuda"):
            # Chunk-by-chunk forward through the utterance
            from biona_lab.model.emformer import CHUNK_FRAMES
            all_logits: list[torch.Tensor] = []
            mem, lck, lcv = model.get_initial_state(batch_size=B)
            mem, lck, lcv = mem.to(device), lck.to(device), lcv.to(device)

            T = features.size(1)
            for start in range(0, T, CHUNK_FRAMES):
                end = min(start + CHUNK_FRAMES, T)
                chunk = features[:, start:end, :]
                if chunk.size(1) < CHUNK_FRAMES:
                    pad = torch.zeros(B, CHUNK_FRAMES - chunk.size(1), 80, device=device)
                    chunk = torch.cat([chunk, pad], dim=1)
                logits, mem, lck, lcv, _ = model(chunk, mem, lck, lcv)
                all_logits.append(logits)

            log_probs = torch.log_softmax(
                torch.cat(all_logits, dim=0), dim=-1
            )  # [T_out, B, vocab]
            out_lengths = torch.full(
                (B,), log_probs.size(0), dtype=torch.long, device=device
            )
            loss = ctc_loss(log_probs, targets, out_lengths, target_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        # Logging
        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.monotonic() - t0
            logger.info(
                "step=%d loss=%.4f lr=%.2e grad_norm=%.3f elapsed=%.0fs",
                step, loss.item(), lr, grad_norm, elapsed,
            )
            if use_wandb:
                wandb.log({"train_loss": loss.item(), "lr": lr, "grad_norm": grad_norm}, step=step)

        # Validation
        if step % config.eval_every == 0:
            val_loss, val_wer = _validate(model, config, device, ctc_loss, step)
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "EVAL step=%d train_loss=%.4f val_loss=%.4f val_wer=%.4f lr=%.2e",
                step, loss.item(), val_loss, val_wer, lr,
            )
            if use_wandb:
                wandb.log({"val_loss": val_loss, "val_wer": val_wer}, step=step)
            last_val_loss = val_loss

        # Checkpoint
        if step % config.checkpoint_every == 0:
            _save_checkpoint(output_dir, model, optimizer, step, config, last_val_loss)

    # Final checkpoint
    _save_checkpoint(output_dir, model, optimizer, step, config, last_val_loss)
    logger.info("Training complete at step %d.", step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Biona Emformer with CTC loss.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to train_config.json (uses defaults if omitted)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint .pt file to resume from",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
