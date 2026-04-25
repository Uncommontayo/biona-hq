#include "webrtc_vad.hpp"
#include "biona/core/errors.hpp"

#include <cassert>
#include <cmath>

namespace biona {

WebRTCVAD::WebRTCVAD(const VADConfig& cfg)
    : sensitivity_(cfg.sensitivity)
    , hangover_ms_(cfg.hangover_ms)
    , reset_silence_ms_(cfg.reset_silence_ms)
{
    handle_ = fvad_new();
    throwIf(handle_ == nullptr, BionaError::VADInitFailed,
            "fvad_new() failed — out of memory");

    int rc = fvad_set_mode(handle_, sensitivity_);
    throwIf(rc != 0, BionaError::VADInitFailed,
            "fvad_set_mode() failed — invalid sensitivity value");

    rc = fvad_set_sample_rate(handle_, sample_rate_);
    throwIf(rc != 0, BionaError::VADInitFailed,
            "fvad_set_sample_rate() failed — unsupported sample rate");
}

WebRTCVAD::~WebRTCVAD() {
    if (handle_) {
        fvad_free(handle_);
        handle_ = nullptr;
    }
}

bool WebRTCVAD::runFvad(const AudioChunk& chunk) noexcept {
    // fvad_process expects exactly 10ms, 20ms, or 30ms of audio
    // at the configured sample rate.
    const size_t expected = static_cast<size_t>(sample_rate_ * CHUNK_MS / 1000);
    assert(chunk.n_samples == expected && "chunk.n_samples must equal sample_rate*chunk_ms/1000");
    (void)expected;

    int result = fvad_process(handle_,
                              chunk.data,
                              chunk.n_samples);
    return result == 1;
}

bool WebRTCVAD::isSpeech(const AudioChunk& chunk) noexcept {
    bool active = runFvad(chunk);

    if (active) {
        frames_since_speech_  = 0;
        frames_since_silence_ = 0;
        return true;
    }

    frames_since_speech_++;
    frames_since_silence_++;

    // Reset state if we've been silent long enough
    const int32_t reset_frames = reset_silence_ms_ / CHUNK_MS;
    if (frames_since_silence_ >= reset_frames) {
        fvad_reset(handle_);
        frames_since_speech_  = 0;
        frames_since_silence_ = 0;
    }

    // Hangover: continue reporting speech for hangover_ms after last detection
    const int32_t hangover_frames = hangover_ms_ / CHUNK_MS;
    return frames_since_speech_ < hangover_frames;
}

float WebRTCVAD::confidence(const AudioChunk& chunk) noexcept {
    bool active = runFvad(chunk);
    if (active) return 1.0f;

    const int32_t hangover_frames = hangover_ms_ / CHUNK_MS;
    if (frames_since_speech_ >= hangover_frames) return 0.0f;

    // Linear decay through hangover window
    float decay = 1.0f - static_cast<float>(frames_since_speech_) /
                          static_cast<float>(hangover_frames);
    return decay;
}

void WebRTCVAD::reset() noexcept {
    if (handle_) fvad_reset(handle_);
    frames_since_speech_  = 0;
    frames_since_silence_ = 0;
}

} // namespace biona
