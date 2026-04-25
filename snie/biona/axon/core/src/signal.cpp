#include "biona/core/signal.hpp"

#include <sstream>

namespace biona {

namespace {

// Minimal JSON helpers — no external dependency required
std::string jsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else { out += c; }
    }
    out += '"';
    return out;
}

std::string jsonFloatArray(const std::vector<float>& v) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) oss << ',';
        oss << v[i];
    }
    oss << ']';
    return oss.str();
}

} // anonymous namespace

std::string Signal::to_json() const {
    std::ostringstream j;
    j << '{';
    j << "\"schema_version\":" << jsonString(schema_version) << ',';
    j << "\"text\":"           << jsonString(text)           << ',';
    j << "\"timestamp_ms\":"   << timestamp_ms               << ',';
    j << "\"latency_ms\":"     << latency_ms                 << ',';
    j << "\"embedding\":"      << jsonFloatArray(embedding);

    if (emotion) {
        j << ",\"emotion\":{"
          << "\"valence\":"   << emotion->valence   << ','
          << "\"arousal\":"   << emotion->arousal   << ','
          << "\"stability\":" << emotion->stability
          << '}';
    }
    if (speaker) {
        j << ",\"speaker\":{"
          << "\"speaker_id\":" << jsonString(speaker->speaker_id) << ','
          << "\"confidence\":"  << speaker->confidence
          << '}';
    }
    if (intent) {
        j << ",\"intent\":{"
          << "\"intent_label\":" << jsonString(intent->intent_label) << ','
          << "\"confidence\":"   << intent->confidence
          << '}';
    }

    j << '}';
    return j.str();
}

} // namespace biona
