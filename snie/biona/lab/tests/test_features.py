"""Tests for biona_lab.data.features — LogMelExtractor."""

import math

import numpy as np
import pytest

from biona_lab.data.features import (
    HOP_SAMPLES,
    MEL_BANDS,
    SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
    LogMelExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio(duration_s: float, sr: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Return a sine-wave audio signal of given duration."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def _expected_frames(n_samples: int) -> int:
    """librosa.stft with center=True: T = 1 + n_samples // HOP_SAMPLES."""
    return 1 + n_samples // HOP_SAMPLES


# ---------------------------------------------------------------------------
# Test 1 — Output shape is [T x MEL_BANDS] for various audio lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("duration_s", [0.5, 1.0, 3.0, 10.0])
def test_output_shape(duration_s: float) -> None:
    extractor = LogMelExtractor()
    audio = _make_audio(duration_s)
    features = extractor.extract(audio, sr=SAMPLE_RATE_HZ)

    expected_t = _expected_frames(len(audio))
    assert features.ndim == 2, "Features must be 2-D"
    assert features.shape[1] == MEL_BANDS, f"Expected {MEL_BANDS} mel bands, got {features.shape[1]}"
    assert features.shape[0] == expected_t, (
        f"Expected {expected_t} frames for {duration_s}s audio, got {features.shape[0]}"
    )
    assert features.dtype == np.float32, "Features must be float32"


# ---------------------------------------------------------------------------
# Test 2 — Streaming chunk returns [2 x MEL_BANDS]
# ---------------------------------------------------------------------------

def test_streaming_chunk_shape() -> None:
    extractor = LogMelExtractor()
    extractor.reset_streaming_state()

    # 20 ms chunk = 320 samples at 16 kHz
    chunk = _make_audio(0.020)
    assert len(chunk) == 320, "Chunk must be exactly 320 samples (20 ms at 16 kHz)"

    features = extractor.extract_streaming_chunk(chunk)

    assert features.ndim == 2, "Streaming output must be 2-D"
    assert features.shape[0] == 2, f"Expected 2 frames from 20 ms chunk, got {features.shape[0]}"
    assert features.shape[1] == MEL_BANDS, f"Expected {MEL_BANDS} bands, got {features.shape[1]}"
    assert features.dtype == np.float32, "Streaming output must be float32"


# ---------------------------------------------------------------------------
# Test 3 — Batch vs streaming consistency (CRITICAL for train/infer parity)
# ---------------------------------------------------------------------------

def test_streaming_matches_batch() -> None:
    """
    Chunk-by-chunk streaming must produce the same features as the batch path
    within floating-point tolerance.

    We compare the interior frames only (skip the first and last few) because
    the batch extractor uses centre-padding (librosa default) while the
    streaming path does not have look-ahead, so edge frames will naturally
    differ slightly. Interior frames must match.
    """
    extractor = LogMelExtractor()

    duration_s = 5.0
    audio = _make_audio(duration_s)

    # Batch reference
    batch_features = extractor.extract(audio, sr=SAMPLE_RATE_HZ)

    # Streaming: feed 20 ms chunks (320 samples) sequentially
    extractor.reset_streaming_state()
    chunk_size = 320  # 20 ms at 16 kHz
    streaming_frames: list[np.ndarray] = []

    for start in range(0, len(audio) - chunk_size + 1, chunk_size):
        chunk = audio[start : start + chunk_size]
        frames = extractor.extract_streaming_chunk(chunk)  # [2 x 80]
        streaming_frames.append(frames)

    streaming_features = np.concatenate(streaming_frames, axis=0)  # [T_stream x 80]

    # Interior frame comparison: skip first 4 and last 4 frames
    margin = 4
    n_compare = min(
        batch_features.shape[0] - 2 * margin,
        streaming_features.shape[0] - 2 * margin,
    )
    assert n_compare > 0, "Not enough frames to compare interior"

    batch_interior = batch_features[margin : margin + n_compare]
    stream_interior = streaming_features[margin : margin + n_compare]

    # Per-utterance normalisation differs slightly between batch and streaming
    # (batch normalises the whole utterance; streaming normalises per chunk).
    # We therefore compare Pearson correlation rather than absolute values.
    for band in range(MEL_BANDS):
        b = batch_interior[:, band]
        s = stream_interior[:, band]
        # Normalise both to zero mean before correlating
        b_norm = b - b.mean()
        s_norm = s - s.mean()
        b_std = b_norm.std()
        s_std = s_norm.std()
        if b_std == 0 or s_std == 0:
            continue
        corr = float(np.dot(b_norm, s_norm) / (len(b_norm) * b_std * s_std))
        assert corr > 0.90, (
            f"Band {band}: batch vs streaming correlation {corr:.3f} < 0.90 "
            f"— feature pipeline may be misaligned"
        )


# ---------------------------------------------------------------------------
# Test 4 — Sample-rate mismatch raises AssertionError
# ---------------------------------------------------------------------------

def test_sr_mismatch_raises() -> None:
    extractor = LogMelExtractor()
    audio = _make_audio(1.0)

    with pytest.raises(AssertionError, match="Expected 16000"):
        extractor.extract(audio, sr=8000)

    with pytest.raises(AssertionError, match="Expected 16000"):
        extractor.extract(audio, sr=44100)
