#include "biona/core/signal.hpp"

#include <cassert>
#include <string>

// ── helpers ─────────────────────────────────────────────────────────────────

static bool contains(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}

// ── tests ────────────────────────────────────────────────────────────────────

static void test_signal_embedding_only() {
    biona::Signal sig;
    sig.text      = "hello";
    sig.embedding.assign(512, 0.5f);
    sig.timestamp_ms = 1000;
    sig.latency_ms   = 12;

    assert(!sig.emotion.has_value()  && "emotion should be null");
    assert(!sig.speaker.has_value()  && "speaker should be null");
    assert(!sig.intent.has_value()   && "intent should be null");
    assert(sig.embedding.size() == 512 && "embedding must be 512-dim");
}

static void test_attach_emotion_does_not_change_core_fields() {
    biona::Signal sig;
    sig.text      = "hello";
    sig.embedding.assign(512, 1.0f);
    sig.timestamp_ms = 2000;
    sig.latency_ms   = 15;

    biona::EmotionSignal emo{0.8f, 0.6f, 0.9f};
    sig.emotion = emo;

    // Core fields unchanged
    assert(sig.text           == "hello");
    assert(sig.embedding.size()== 512);
    assert(sig.timestamp_ms   == 2000);
    assert(sig.latency_ms     == 15);
    assert(!sig.speaker.has_value());
    assert(!sig.intent.has_value());
    assert(sig.emotion.has_value());
    assert(sig.emotion->valence == 0.8f);
}

static void test_to_json_contains_schema_version() {
    biona::Signal sig;
    sig.text      = "world";
    sig.embedding.assign(512, 0.0f);
    sig.timestamp_ms = 500;
    sig.latency_ms   = 8;

    std::string json = sig.to_json();
    assert(contains(json, "schema_version") && "JSON must contain schema_version");
    assert(contains(json, "1.0")            && "schema_version must be 1.0");
    assert(contains(json, "\"text\"")       && "JSON must contain text field");
    assert(contains(json, "\"embedding\"")  && "JSON must contain embedding field");
}

static void test_to_json_omits_null_optional_fields() {
    biona::Signal sig;
    sig.embedding.assign(512, 0.0f);

    std::string json = sig.to_json();
    // Optional fields must be omitted when null (not null-valued)
    assert(!contains(json, "\"emotion\"") && "null emotion must be omitted from JSON");
    assert(!contains(json, "\"speaker\"") && "null speaker must be omitted from JSON");
    assert(!contains(json, "\"intent\"")  && "null intent must be omitted from JSON");
}

static void test_to_json_includes_present_optional_fields() {
    biona::Signal sig;
    sig.embedding.assign(512, 0.0f);
    sig.emotion  = biona::EmotionSignal{0.5f, 0.5f, 0.5f};
    sig.speaker  = biona::SpeakerSignal{"speaker_001", 0.95f};

    std::string json = sig.to_json();
    assert(contains(json, "\"emotion\"")     && "present emotion must appear in JSON");
    assert(contains(json, "\"speaker\"")     && "present speaker must appear in JSON");
    assert(contains(json, "speaker_001")     && "speaker_id must appear in JSON");
    assert(!contains(json, "\"intent\"")     && "null intent must be omitted");
}

int main() {
    test_signal_embedding_only();
    test_attach_emotion_does_not_change_core_fields();
    test_to_json_contains_schema_version();
    test_to_json_omits_null_optional_fields();
    test_to_json_includes_present_optional_fields();
    return 0;
}
