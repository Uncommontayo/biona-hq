#pragma once

#include "biona/core/interfaces/vad.hpp"

namespace biona {

/**
 * @brief Energy-based VAD fallback for clean acoustic environments.
 *
 * Computes RMS energy of the input chunk and compares against a configurable
 * threshold (default: -40 dBFS).
 *
 * No external dependencies. Completes in < 0.1ms per chunk.
 * No dynamic allocation in isSpeech() or confidence().
 *
 * Thread safety: single-threaded only.
 */
class EnergyVAD final : public VAD {
public:
    /**
     * @param cfg              VAD configuration (hangover_ms and reset_silence_ms apply).
     * @param threshold_dbfs   Speech/silence threshold in dBFS (default: -40.0f).
     */
    explicit EnergyVAD(const VADConfig& cfg, float threshold_dbfs = -40.0f);

    bool  isSpeech  (const AudioChunk& chunk) noexcept override;
    float confidence(const AudioChunk& chunk) noexcept override;
    void  reset     ()                        noexcept override;

private:
    int32_t hangover_ms_;
    int32_t reset_silence_ms_;
    float   threshold_linear_;   // pre-converted from dBFS

    int32_t frames_since_speech_  = 0;
    int32_t frames_since_silence_ = 0;

    static constexpr int32_t CHUNK_MS = 20;

    [[nodiscard]] float computeRMS(const AudioChunk& chunk) noexcept;
};

} // namespace biona
