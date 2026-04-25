#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace biona {

/**
 * @brief Emotion signal — continuous valence/arousal/stability scores.
 *
 * Produced asynchronously by the speech emotion recognition (SER) model.
 * All values are normalised floats in the range [0.0, 1.0].
 */
struct EmotionSignal {
    float valence;   ///< Emotional positivity (0 = negative, 1 = positive)
    float arousal;   ///< Energy / activation level (0 = calm, 1 = excited)
    float stability; ///< Consistency over the utterance (0 = variable, 1 = stable)
};

/**
 * @brief Speaker identity signal.
 *
 * Populated by the speaker diarisation / verification module.
 */
struct SpeakerSignal {
    std::string speaker_id; ///< Opaque speaker identifier string
    float       confidence; ///< Match confidence in [0.0, 1.0]
};

/**
 * @brief Intent classification signal.
 *
 * Populated by the lightweight on-device intent classifier.
 */
struct IntentSignal {
    std::string intent_label; ///< Intent label from the classifier vocabulary
    float       confidence;   ///< Classification confidence in [0.0, 1.0]
};

/**
 * @brief The stable signal envelope — the public output contract of Biona Axon.
 *
 * Schema stability rules (MUST be respected across all versions):
 *   - Adding a new std::optional<T> field is the ONLY allowed schema change.
 *   - Never add required (non-optional) fields after v1.0.
 *   - Never remove existing fields.
 *   - schema_version must be bumped on any breaking change.
 *
 * The embedding field is ALWAYS populated (512-dimensional float vector).
 * Optional signal fields are null when not computed (e.g. async not enabled).
 */
struct Signal {
    std::string        schema_version = "1.0"; ///< Schema version identifier
    std::string        text;                   ///< CTC-decoded transcript
    int64_t            timestamp_ms = 0;        ///< Timestamp of the originating audio chunk
    int32_t            latency_ms   = 0;        ///< End-to-end inference latency

    /// Always populated — 512-dim embedding from the streaming Emformer
    std::vector<float> embedding;

    /// Null when SER is disabled or async result has not yet arrived
    std::optional<EmotionSignal>  emotion;

    /// Null when speaker diarisation is disabled
    std::optional<SpeakerSignal>  speaker;

    /// Null when intent classification is disabled
    std::optional<IntentSignal>   intent;

    /**
     * @brief Serialise this signal to a JSON string.
     *
     * The output always contains: schema_version, text, timestamp_ms,
     * latency_ms, and embedding. Optional fields are omitted when null.
     *
     * @return UTF-8 encoded JSON string.
     */
    [[nodiscard]] std::string to_json() const;
};

} // namespace biona
