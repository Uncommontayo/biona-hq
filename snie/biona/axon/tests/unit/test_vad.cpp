#include "biona/core/interfaces/vad.hpp"
#include "biona/core/errors.hpp"
#include "biona/core/types.hpp"

#include <cassert>
#include <cmath>
#include <vector>

// ── Audio generation helpers ─────────────────────────────────────────────────

static constexpr int    SAMPLE_RATE = 16000;
static constexpr int    CHUNK_MS    = 20;
static constexpr size_t CHUNK_SAMPLES = static_cast<size_t>(SAMPLE_RATE * CHUNK_MS / 1000); // 320

/// Generate a chunk of silence (all zeros).
static std::vector<int16_t> makeSilence() {
    return std::vector<int16_t>(CHUNK_SAMPLES, 0);
}

/// Generate a chunk of a sine wave at @p freq_hz — speech-like at 300Hz.
static std::vector<int16_t> makeSine(float freq_hz, float amplitude = 16000.0f) {
    std::vector<int16_t> buf(CHUNK_SAMPLES);
    for (size_t i = 0; i < CHUNK_SAMPLES; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(SAMPLE_RATE);
        buf[i] = static_cast<int16_t>(amplitude * std::sin(2.0f * 3.14159265f * freq_hz * t));
    }
    return buf;
}

static biona::AudioChunk makeChunk(const std::vector<int16_t>& buf, int64_t ts = 0) {
    return biona::AudioChunk{buf.data(), buf.size(), ts};
}

// ── EnergyVAD tests (no libfvad dependency) ──────────────────────────────────

static void test_energy_vad_silence() {
    biona::VADConfig cfg;
    cfg.type         = biona::VADType::ENERGY;
    cfg.hangover_ms  = 200;
    auto vad = biona::createVAD(cfg);

    auto silence = makeSilence();
    for (int i = 0; i < 20; ++i) {
        auto chunk = makeChunk(silence, i * CHUNK_MS);
        assert(!vad->isSpeech(chunk) && "Silence should not trigger VAD");
    }
}

static void test_energy_vad_sine_speech() {
    biona::VADConfig cfg;
    cfg.type         = biona::VADType::ENERGY;
    cfg.hangover_ms  = 200;
    auto vad = biona::createVAD(cfg);

    auto sine = makeSine(300.0f, 20000.0f); // Strong 300Hz tone
    bool any_speech = false;
    for (int i = 0; i < 10; ++i) {
        auto chunk = makeChunk(sine, i * CHUNK_MS);
        if (vad->isSpeech(chunk)) any_speech = true;
    }
    assert(any_speech && "300Hz sine wave should trigger EnergyVAD");
}

static void test_energy_vad_hangover() {
    biona::VADConfig cfg;
    cfg.type             = biona::VADType::ENERGY;
    cfg.hangover_ms      = 200;  // 10 frames
    cfg.reset_silence_ms = 2000;
    auto vad = biona::createVAD(cfg);

    // Feed speech to trigger VAD
    auto sine = makeSine(300.0f, 20000.0f);
    for (int i = 0; i < 5; ++i) {
        auto chunk = makeChunk(sine, i * CHUNK_MS);
        vad->isSpeech(chunk);
    }

    // Switch to silence — should remain true for hangover_ms (10 frames)
    auto silence = makeSilence();
    int true_count = 0;
    for (int i = 0; i < 15; ++i) {
        auto chunk = makeChunk(silence, (5 + i) * CHUNK_MS);
        if (vad->isSpeech(chunk)) true_count++;
    }
    // Should get approximately hangover_ms/CHUNK_MS = 10 true returns
    assert(true_count >= 8 && "Hangover should keep VAD active after speech ends");
    assert(true_count <= 11 && "Hangover should not extend indefinitely");
}

static void test_silero_throws() {
    biona::VADConfig cfg;
    cfg.type = biona::VADType::SILERO;
    bool threw = false;
    try {
        biona::createVAD(cfg);
    } catch (const biona::BionaException& e) {
        assert(e.code() == biona::BionaError::UnsupportedRuntime);
        threw = true;
    }
    assert(threw && "SILERO VAD must throw UnsupportedRuntime");
}

int main() {
    test_energy_vad_silence();
    test_energy_vad_sine_speech();
    test_energy_vad_hangover();
    test_silero_throws();
    return 0;
}
