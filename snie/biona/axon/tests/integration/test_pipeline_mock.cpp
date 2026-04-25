#include "biona/core/interfaces/inference_engine.hpp"
#include "biona/core/interfaces/vad.hpp"
#include "biona/core/types.hpp"
#include "biona/core/signal.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// MockInferenceEngine — returns fixed 512-dim embedding and "hello"
// ---------------------------------------------------------------------------

class MockInferenceEngine final : public biona::InferenceEngine {
public:
    bool initialize(const biona::ModelConfig&) override {
        initialized_ = true;
        return true;
    }
    biona::InferenceResult run(const biona::AudioFeatures&) override {
        ++call_count_;
        biona::InferenceResult r;
        r.embedding.assign(512, 0.42f);
        r.text        = "hello";
        r.latency_ms  = 5;
        r.vad_triggered = true;
        return r;
    }
    void reset() override { call_count_ = 0; }
    biona::EngineInfo info() const override {
        return {"mock", "1.0", true};
    }
    int call_count_ = 0;
    bool initialized_ = false;
};

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

static constexpr int    SAMPLE_RATE   = 16000;
static constexpr int    CHUNK_MS      = 20;
static constexpr size_t CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS / 1000; // 320

static std::vector<int16_t> makeSilence() {
    return std::vector<int16_t>(CHUNK_SAMPLES, 0);
}

static std::vector<int16_t> makeSine(float amp = 20000.0f) {
    std::vector<int16_t> buf(CHUNK_SAMPLES);
    for (size_t i = 0; i < CHUNK_SAMPLES; ++i) {
        float t  = static_cast<float>(i) / SAMPLE_RATE;
        buf[i]   = static_cast<int16_t>(amp * std::sin(2.0f * 3.14159265f * 300.0f * t));
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_silence_gated_by_vad() {
    biona::VADConfig cfg;
    cfg.type         = biona::VADType::ENERGY;
    cfg.hangover_ms  = 0; // No hangover for this test
    auto vad = biona::createVAD(cfg);

    auto engine = std::make_unique<MockInferenceEngine>();
    engine->initialize({});

    auto silence = makeSilence();
    for (int i = 0; i < 10; ++i) {
        biona::AudioChunk chunk{silence.data(), silence.size(), i * CHUNK_MS};
        if (!vad->isSpeech(chunk)) {
            // VAD gated — skip inference (correct behaviour)
        } else {
            engine->run(biona::AudioFeatures{});
        }
    }
    assert(engine->call_count_ == 0 && "Silence must not trigger inference");
}

static void test_speech_produces_inference_and_signal() {
    biona::VADConfig cfg;
    cfg.type         = biona::VADType::ENERGY;
    cfg.hangover_ms  = 200;
    auto vad = biona::createVAD(cfg);

    auto engine = std::make_unique<MockInferenceEngine>();
    engine->initialize({});

    auto sine = makeSine();
    biona::Signal last_signal;

    for (int i = 0; i < 10; ++i) {
        biona::AudioChunk chunk{sine.data(), sine.size(), i * CHUNK_MS};
        if (vad->isSpeech(chunk)) {
            biona::InferenceResult res = engine->run(biona::AudioFeatures{});

            // Assemble signal manually
            last_signal.text      = res.text;
            last_signal.embedding = res.embedding;
            last_signal.latency_ms= res.latency_ms;
        }
    }

    assert(engine->call_count_ > 0   && "Speech must trigger inference");
    assert(last_signal.embedding.size() == 512 && "Embedding must be 512-dim");
    assert(last_signal.text == "hello"          && "Text must be 'hello'");
}

int main() {
    test_silence_gated_by_vad();
    test_speech_produces_inference_and_signal();
    return 0;
}
