#include "energy_vad.hpp"

#include <cassert>
#include <cmath>

namespace biona {

EnergyVAD::EnergyVAD(const VADConfig& cfg, float threshold_dbfs)
    : hangover_ms_(cfg.hangover_ms)
    , reset_silence_ms_(cfg.reset_silence_ms)
{
    // Convert dBFS threshold to linear amplitude fraction of int16 range
    // int16 full-scale = 32768
    threshold_linear_ = 32768.0f * std::pow(10.0f, threshold_dbfs / 20.0f);
}

float EnergyVAD::computeRMS(const AudioChunk& chunk) noexcept {
    if (chunk.n_samples == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < chunk.n_samples; ++i) {
        double s = static_cast<double>(chunk.data[i]);
        sum += s * s;
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(chunk.n_samples)));
}

bool EnergyVAD::isSpeech(const AudioChunk& chunk) noexcept {
    float rms = computeRMS(chunk);
    bool active = (rms >= threshold_linear_);

    if (active) {
        frames_since_speech_  = 0;
        frames_since_silence_ = 0;
        return true;
    }

    frames_since_speech_++;
    frames_since_silence_++;

    const int32_t reset_frames = reset_silence_ms_ / CHUNK_MS;
    if (frames_since_silence_ >= reset_frames) {
        frames_since_speech_  = 0;
        frames_since_silence_ = 0;
    }

    const int32_t hangover_frames = hangover_ms_ / CHUNK_MS;
    return frames_since_speech_ < hangover_frames;
}

float EnergyVAD::confidence(const AudioChunk& chunk) noexcept {
    float rms = computeRMS(chunk);
    if (rms >= threshold_linear_) return 1.0f;

    const int32_t hangover_frames = hangover_ms_ / CHUNK_MS;
    if (frames_since_speech_ >= hangover_frames) return 0.0f;

    float decay = 1.0f - static_cast<float>(frames_since_speech_) /
                          static_cast<float>(hangover_frames);
    return decay;
}

void EnergyVAD::reset() noexcept {
    frames_since_speech_  = 0;
    frames_since_silence_ = 0;
}

} // namespace biona
