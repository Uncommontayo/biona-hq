"""LibriSpeech dataset loader for Biona Lab training pipeline."""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_SPLITS = {
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
}

_SEGMENT_MIN_DURATION_S = 0.5
_SIGMA_FILTER_THRESHOLD = 3.0


def normalize_transcript(text: str) -> str:
    """Lowercase, remove punctuation except apostrophes, collapse whitespace."""
    text = text.lower()
    # Remove all punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment(record: dict, max_duration_s: float = 30.0) -> list[dict]:
    """
    Split long utterances at silence boundaries.
    Returns a list of segment dicts (may be just [record] if short enough).
    """
    duration = record["duration_s"]
    if duration <= max_duration_s:
        return [record]

    audio_path = record["audio_path"]
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    # Get non-silent intervals
    intervals = librosa.effects.split(y, top_db=30)

    segments = []
    for start_sample, end_sample in intervals:
        start_s = start_sample / sr
        end_s = end_sample / sr
        seg_duration = end_s - start_s

        if seg_duration < _SEGMENT_MIN_DURATION_S:
            continue
        if seg_duration > max_duration_s:
            # Sub-split at max_duration_s boundaries
            cursor = start_s
            while cursor < end_s:
                chunk_end = min(cursor + max_duration_s, end_s)
                if chunk_end - cursor >= _SEGMENT_MIN_DURATION_S:
                    seg = dict(record)
                    seg["segment_start_s"] = cursor
                    seg["segment_end_s"] = chunk_end
                    seg["duration_s"] = chunk_end - cursor
                    segments.append(seg)
                cursor = chunk_end
        else:
            seg = dict(record)
            seg["segment_start_s"] = start_s
            seg["segment_end_s"] = end_s
            seg["duration_s"] = seg_duration
            segments.append(seg)

    return segments if segments else [record]


class LibriSpeechLoader:
    """
    Loads LibriSpeech data from a local directory.

    Yields records: {
        "audio_path": Path,
        "transcript": str,
        "speaker_id": str,
        "chapter_id": str,
        "duration_s": float,
        "source": str,
    }
    """

    def __init__(
        self,
        data_dir: Path,
        splits: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.splits = splits or ["train-clean-100"]

        for split in self.splits:
            if split not in SUPPORTED_SPLITS:
                raise ValueError(
                    f"Unsupported split '{split}'. "
                    f"Choose from: {sorted(SUPPORTED_SPLITS)}"
                )

    def load(self) -> Iterator[dict]:
        """Yield validated, normalised records from all configured splits."""
        all_records: list[dict] = []

        for split in self.splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                logger.warning("Split directory not found, skipping: %s", split_dir)
                continue

            source = "librispeech-clean" if "clean" in split else "librispeech-other"
            records = list(self._load_split(split_dir, source))
            all_records.extend(records)
            logger.info(
                "Split '%s': loaded %d raw records before sigma filter", split, len(records)
            )

        if not all_records:
            logger.warning("No records loaded from any split.")
            return

        # Sigma filter: discard entries where duration/token-count ratio is an outlier
        valid_records, discarded = self._sigma_filter(all_records)
        logger.info(
            "Discarded %d entries (%g sigma filter)", discarded, _SIGMA_FILTER_THRESHOLD
        )

        # Per-split statistics
        self._log_statistics(valid_records)

        yield from valid_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_split(self, split_dir: Path, source: str) -> Iterator[dict]:
        """Walk a split directory and yield raw records."""
        # LibriSpeech layout: <split>/<speaker_id>/<chapter_id>/<files>
        for trans_file in split_dir.rglob("*.trans.txt"):
            chapter_dir = trans_file.parent
            parts = chapter_dir.relative_to(split_dir).parts
            if len(parts) < 2:
                continue
            speaker_id, chapter_id = parts[0], parts[1]

            transcripts = self._parse_trans_file(trans_file)
            for utt_id, raw_text in transcripts.items():
                audio_path = chapter_dir / f"{utt_id}.flac"
                if not audio_path.exists():
                    logger.debug("Audio file missing, skipping: %s", audio_path)
                    continue

                try:
                    duration_s = self._get_duration(audio_path)
                except Exception as exc:
                    logger.debug("Cannot read audio %s: %s", audio_path, exc)
                    continue

                transcript = normalize_transcript(raw_text)
                yield {
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "duration_s": duration_s,
                    "source": source,
                }

    @staticmethod
    def _parse_trans_file(trans_file: Path) -> dict[str, str]:
        """Parse a .trans.txt file into {utterance_id: transcript}."""
        transcripts: dict[str, str] = {}
        with trans_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
        return transcripts

    @staticmethod
    def _get_duration(audio_path: Path) -> float:
        """Return audio duration in seconds without loading the full waveform."""
        info = sf.info(str(audio_path))
        return info.frames / info.samplerate

    @staticmethod
    def _sigma_filter(
        records: list[dict],
        threshold: float = _SIGMA_FILTER_THRESHOLD,
    ) -> tuple[list[dict], int]:
        """
        Remove records whose duration/token-count ratio deviates > threshold sigma
        from the dataset mean.
        """
        ratios = np.array(
            [
                r["duration_s"] / max(len(r["transcript"].split()), 1)
                for r in records
            ],
            dtype=np.float64,
        )
        mean = ratios.mean()
        std = ratios.std()

        if std == 0:
            return records, 0

        valid = [r for r, ratio in zip(records, ratios) if abs(ratio - mean) <= threshold * std]
        discarded = len(records) - len(valid)
        return valid, discarded

    @staticmethod
    def _log_statistics(records: list[dict]) -> None:
        """Log dataset statistics (hours, speakers, vocab size, duration stats)."""
        if not records:
            return

        total_hours = sum(r["duration_s"] for r in records) / 3600.0
        speakers = {r["speaker_id"] for r in records}
        vocab: set[str] = set()
        durations = []
        for r in records:
            vocab.update(r["transcript"].split())
            durations.append(r["duration_s"])

        dur_arr = np.array(durations)
        logger.info(
            "Dataset stats — total: %.1f h | speakers: %d | vocab: %d words | "
            "duration mean: %.2f s ± %.2f s",
            total_hours,
            len(speakers),
            len(vocab),
            dur_arr.mean(),
            dur_arr.std(),
        )
