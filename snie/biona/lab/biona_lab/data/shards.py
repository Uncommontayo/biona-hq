"""WebDataset shard writer for Biona Lab.

Packages processed LibriSpeech samples (log-mel features + transcripts)
into WebDataset tar shards with a manifest for reproducible training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import webdataset as wds
from tqdm import tqdm

from biona_lab.data.features import LogMelExtractor
from biona_lab.data.librispeech import LibriSpeechLoader, segment

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class ShardWriter:
    """
    Writes WebDataset shards from a LibriSpeech dataset.

    Each shard is a tar archive containing samples with:
      - audio.npy   — log-mel features [T x 80] float32
      - text.txt    — normalised transcript
      - meta.json   — metadata including SHA256 of features
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size_mb: int = 512,
        max_samples_per_shard: int = 10000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.max_samples_per_shard = max_samples_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        loader: LibriSpeechLoader,
        extractor: LogMelExtractor,
    ) -> None:
        """
        Full pipeline: load → segment → extract features → write shards.
        Logs tqdm progress with samples/sec, hours processed, shards written.
        Writes manifest.json on completion.
        """
        shard_infos: list[dict] = []
        total_samples = 0
        total_duration_s = 0.0

        shard_index = 0
        current_shard_path = self._shard_path(shard_index)
        current_shard_samples = 0
        current_shard_bytes = 0

        # Stream all records (may be large — don't materialise)
        record_iter = loader.load()

        pbar = tqdm(
            desc="Writing shards",
            unit="samples",
            dynamic_ncols=True,
        )

        t0 = time.monotonic()
        tar_writer = wds.TarWriter(str(current_shard_path))

        try:
            for record in record_iter:
                # Segment long utterances
                segments = segment(record)

                for seg in segments:
                    try:
                        features, audio_bytes = self._extract_features(seg, extractor)
                    except Exception as exc:
                        logger.warning("Skipping sample %s: %s", seg.get("audio_path"), exc)
                        continue

                    sample_key = self._make_key(seg)
                    meta = self._make_meta(seg, audio_bytes)

                    self._write_sample(tar_writer, sample_key, features, seg["transcript"], meta)

                    sample_bytes = len(audio_bytes)
                    current_shard_samples += 1
                    current_shard_bytes += sample_bytes
                    total_samples += 1
                    total_duration_s += seg["duration_s"]

                    # Roll over shard if size or count limit hit
                    if (
                        current_shard_bytes >= self.shard_size_bytes
                        or current_shard_samples >= self.max_samples_per_shard
                    ):
                        tar_writer.close()
                        sha = _sha256_file(current_shard_path)
                        shard_infos.append({
                            "filename": current_shard_path.name,
                            "sha256": sha,
                            "n_samples": current_shard_samples,
                        })
                        shard_index += 1
                        current_shard_path = self._shard_path(shard_index)
                        current_shard_samples = 0
                        current_shard_bytes = 0
                        tar_writer = wds.TarWriter(str(current_shard_path))

                    elapsed = time.monotonic() - t0
                    hours = total_duration_s / 3600.0
                    pbar.set_postfix(
                        hours=f"{hours:.2f}h",
                        shards=shard_index + 1,
                        refresh=False,
                    )
                    pbar.update(1)

        finally:
            tar_writer.close()
            pbar.close()

        # Finalise last shard if it has samples
        if current_shard_samples > 0:
            sha = _sha256_file(current_shard_path)
            shard_infos.append({
                "filename": current_shard_path.name,
                "sha256": sha,
                "n_samples": current_shard_samples,
            })
        else:
            # Remove empty trailing shard
            current_shard_path.unlink(missing_ok=True)

        self._write_manifest(shard_infos, total_samples, total_duration_s)
        logger.info(
            "Done. %d samples, %.2f hours, %d shards written to %s",
            total_samples,
            total_duration_s / 3600.0,
            len(shard_infos),
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_path(self, index: int) -> Path:
        return self.output_dir / f"shard-{index:05d}.tar"

    @staticmethod
    def _extract_features(
        record: dict,
        extractor: LogMelExtractor,
    ) -> tuple[np.ndarray, bytes]:
        """Load audio segment, extract features, return (array, bytes)."""
        import soundfile as sf

        audio_path = record["audio_path"]
        start_s = record.get("segment_start_s")
        end_s = record.get("segment_end_s")

        y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)

        if start_s is not None and end_s is not None:
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            y = y[start_sample:end_sample]

        features = extractor.extract(y, sr)  # [T x 80] float32

        # Serialise to bytes for hashing and writing
        buf = io.BytesIO()
        np.save(buf, features)
        audio_bytes = buf.getvalue()

        return features, audio_bytes

    @staticmethod
    def _make_key(record: dict) -> str:
        """Unique sample key: speaker_chapter_uttid[_segidx]."""
        audio_path = Path(record["audio_path"])
        base = audio_path.stem  # e.g. 1234-56789-0001
        start_s = record.get("segment_start_s")
        if start_s is not None:
            # Encode segment start as milliseconds to keep key filesystem-safe
            base = f"{base}_{int(start_s * 1000):07d}ms"
        return base

    @staticmethod
    def _make_meta(record: dict, audio_bytes: bytes) -> dict:
        source = record.get("source", "librispeech")
        return {
            "duration_s": round(record["duration_s"], 4),
            "source": source,
            "speaker_id": record["speaker_id"],
            "shard_hash": _sha256_bytes(audio_bytes),
            "schema_version": _SCHEMA_VERSION,
        }

    @staticmethod
    def _write_sample(
        writer: wds.TarWriter,
        key: str,
        features: np.ndarray,
        transcript: str,
        meta: dict,
    ) -> None:
        """Write one sample to the open TarWriter."""
        buf = io.BytesIO()
        np.save(buf, features)

        writer.write({
            "__key__": key,
            "audio.npy": buf.getvalue(),
            "text.txt": transcript.encode("utf-8"),
            "meta.json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        })

    def _write_manifest(
        self,
        shard_infos: list[dict],
        total_samples: int,
        total_duration_s: float,
    ) -> None:
        manifest = {
            "total_samples": total_samples,
            "total_hours": round(total_duration_s / 3600.0, 4),
            "shard_files": shard_infos,
            "schema_version": _SCHEMA_VERSION,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        fingerprint = _sha256_file(manifest_path)
        logger.info("Dataset fingerprint (manifest SHA256): %s", fingerprint)
        logger.info("Manifest written to %s", manifest_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write WebDataset shards from LibriSpeech data."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100"],
        help="LibriSpeech splits to process (default: train-clean-100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SHARD_OUTPUT_DIR", "/data/shards")),
        help="Directory to write shards and manifest.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("LIBRISPEECH_DATA_DIR", "/data/librispeech")),
        help="Root directory of extracted LibriSpeech data",
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=512,
        help="Maximum shard size in MB (default: 512)",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=10000,
        help="Maximum samples per shard (default: 10000)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    loader = LibriSpeechLoader(data_dir=args.data_dir, splits=args.splits)
    extractor = LogMelExtractor()
    writer = ShardWriter(
        output_dir=args.output_dir,
        shard_size_mb=args.shard_size_mb,
        max_samples_per_shard=args.max_samples_per_shard,
    )
    writer.write(loader, extractor)


if __name__ == "__main__":
    main()
