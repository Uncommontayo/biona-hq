#pragma once

#include "biona/core/interfaces/vad.hpp"

#include <fvad.h>  // libfvad (WebRTC VAD C port)

namespace biona {

/**
 * @brief WebRTC VAD implementation using libfvad.
 *
 * Implements hangover logic:
 *   - Returns true for hangover_ms after the last positive detection.
 *     This captures trailing phonemes that fall below the detection threshold.
 *   - Resets internal state after reset_silence_ms of continuous silence.
 *
 * Performance: isSpeech() completes in < 2ms. No dynamic allocation in hot path.
 *
 * Thread safety: single-threaded. Do NOT share across threads.
 */
class WebRTCVAD final : public VAD {
public:
    explicit WebRTCVAD(const VADConfig& cfg);
    ~WebRTCVAD() override;

    bool  isSpeech  (const AudioChunk& chunk) noexcept override;
    float confidence(const AudioChunk& chunk) noexcept override;
    void  reset     ()                        noexcept override;

private:
    Fvad*   handle_         = nullptr;
    int32_t sensitivity_;
    int32_t hangover_ms_;
    int32_t reset_silence_ms_;
    int32_t sample_rate_    = 16000;

    // Hangover state (number of frames since last positive detection)
    int32_t frames_since_speech_ = 0;
    int32_t frames_since_silence_= 0;

    // Frames per ms (computed from sample rate and chunk size)
    static constexpr int32_t CHUNK_MS = 20;

    [[nodiscard]] bool  runFvad(const AudioChunk& chunk) noexcept;
};

} // namespace biona
