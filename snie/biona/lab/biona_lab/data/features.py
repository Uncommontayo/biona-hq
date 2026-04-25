"""Log-mel spectrogram feature extractor for Biona Lab.

CRITICAL: All locked parameters below MUST match onnx_contract.hpp in Biona Axon.
Any mismatch causes a silent train/inference accuracy bug.
"""

from __future__ import annotations

import numpy as np
import librosa

# ══ LOCKED PARAMETERS — DO NOT CHANGE WITHOUT UPDATING onnx_contract.hpp ══
SAMPLE_RATE_HZ  = 16000  # Must match AUDIO_SAMPLE_RATE_HZ in onnx_contract.hpp
MEL_BANDS       = 80     # Must match AUDIO_MEL_BANDS in onnx_contract.hpp
WINDOW_MS       = 25     # Must match AUDIO_WINDOW_MS in onnx_contract.hpp
HOP_MS          = 10     # Must match AUDIO_HOP_MS in onnx_contract.hpp
WINDOW_SAMPLES  = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # = 400
HOP_SAMPLES     = int(SAMPLE_RATE_HZ * HOP_MS / 1000)      # = 160
FMIN            = 80     # Hz
FMAX            = 7600   # Hz
# ══════════════════════════════════════════════════════════════════════════

_PRE_EMPHASIS_COEFF = 0.97
_LOG_FLOOR = 1e-10


class LogMelExtractor:
    """
    Computes log-mel spectrograms compatible with Biona Axon's C++ frontend.

    Both batch (extract) and streaming (extract_streaming_chunk) modes produce
    consistent features so that training and on-device inference remain aligned.
    """

    def __init__(self) -> None:
        # Pre-build mel filterbank once — never recompute per sample
        self._mel_fb = self._build_mel_filterbank()
        # Streaming state: last sample carried across chunks for pre-emphasis
        self._prev_sample: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute log-mel spectrogram for a full utterance.

        Parameters
        ----------
        audio : np.ndarray
            Raw waveform, float32 or float64, shape [N].
        sr : int
            Sample rate — must equal SAMPLE_RATE_HZ.

        Returns
        -------
        np.ndarray
            Log-mel features, shape [T x MEL_BANDS], float32.
            T = ceil((N - WINDOW_SAMPLES) / HOP_SAMPLES) + 1
        """
        assert sr == SAMPLE_RATE_HZ, f"Expected {SAMPLE_RATE_HZ} Hz, got {sr} Hz"

        audio = np.asarray(audio, dtype=np.float64)

        # Pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        audio = self._apply_pre_emphasis(audio, prev=0.0)

        # STFT → power spectrum  [n_fft/2+1 x T]
        power = self._stft_power(audio)

        # Mel filterbank → log compression → normalise
        features = self._power_to_log_mel(power)

        return features.astype(np.float32)

    def extract_streaming_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Extract features from a single 20 ms chunk in streaming mode.

        Maintains pre-emphasis state across successive calls so that the
        streaming output matches the batch output sample-for-sample.

        Parameters
        ----------
        chunk : np.ndarray
            20 ms of audio at SAMPLE_RATE_HZ → 320 samples, float32/64.

        Returns
        -------
        np.ndarray
            Shape [2 x MEL_BANDS], float32.
            (2 frames at 10 ms hop produced by a 20 ms / 320-sample window)
        """
        chunk = np.asarray(chunk, dtype=np.float64)

        # Stateful pre-emphasis
        chunk_emph = self._apply_pre_emphasis(chunk, prev=self._prev_sample)
        self._prev_sample = float(chunk[-1])

        power = self._stft_power(chunk_emph)
        features = self._power_to_log_mel(power)
        return features.astype(np.float32)

    def reset_streaming_state(self) -> None:
        """Reset internal streaming state (call between utterances)."""
        self._prev_sample = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mel_filterbank(self) -> np.ndarray:
        """Return mel filterbank matrix [MEL_BANDS x (n_fft/2+1)]."""
        return librosa.filters.mel(
            sr=SAMPLE_RATE_HZ,
            n_fft=WINDOW_SAMPLES,
            n_mels=MEL_BANDS,
            fmin=FMIN,
            fmax=FMAX,
            norm=None,
            htk=False,
        ).astype(np.float64)

    @staticmethod
    def _apply_pre_emphasis(audio: np.ndarray, prev: float) -> np.ndarray:
        """y[n] = x[n] - coeff * x[n-1], with explicit initial prev sample."""
        out = np.empty_like(audio)
        if len(audio) == 0:
            return out
        out[0] = audio[0] - _PRE_EMPHASIS_COEFF * prev
        out[1:] = audio[1:] - _PRE_EMPHASIS_COEFF * audio[:-1]
        return out

    @staticmethod
    def _stft_power(audio: np.ndarray) -> np.ndarray:
        """
        Compute power spectrum via STFT with Hann window.
        Returns shape [n_fft/2+1 x T].
        """
        # librosa.stft returns complex [1 + n_fft/2, T]
        stft = librosa.stft(
            audio,
            n_fft=WINDOW_SAMPLES,
            hop_length=HOP_SAMPLES,
            win_length=WINDOW_SAMPLES,
            window="hann",
            center=True,        # pad to keep T consistent with centre-frame convention
            pad_mode="reflect",
        )
        return np.abs(stft) ** 2  # power spectrum

    def _power_to_log_mel(self, power: np.ndarray) -> np.ndarray:
        """
        Apply mel filterbank, log compression, and per-utterance normalisation.
        Input:  power [n_fft/2+1 x T]
        Output: features [T x MEL_BANDS]
        """
        # Mel filterbank: [MEL_BANDS x T]
        mel = self._mel_fb @ power

        # Log compression
        log_mel = np.log(np.maximum(mel, _LOG_FLOOR))

        # Per-utterance normalisation: zero mean, unit variance
        mean = log_mel.mean()
        std = log_mel.std()
        if std > 0:
            log_mel = (log_mel - mean) / std
        else:
            log_mel = log_mel - mean

        # Transpose to [T x MEL_BANDS]
        return log_mel.T
