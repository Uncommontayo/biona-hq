#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace biona {

/**
 * @brief A raw audio chunk from the microphone or file source.
 *
 * The data pointer is NOT owned by this struct; the caller retains
 * ownership and must ensure the buffer remains valid for the duration
 * of any call that consumes this chunk.
 */
struct AudioChunk {
    const int16_t* data;        ///< Pointer to PCM samples (16-bit signed)
    size_t         n_samples;   ///< Number of samples in this chunk
    int64_t        timestamp_ms;///< Wall-clock timestamp of the first sample (ms)
};

/**
 * @brief Log-mel spectrogram features extracted from one audio chunk.
 *
 * Contains exactly 80 mel bands matching the Frontend spec:
 *   25ms window, 10ms hop, 16kHz sample rate.
 */
struct AudioFeatures {
    std::array<float, 80> mel_bands; ///< Log-mel energies for 80 frequency bands
    size_t                n_frames;  ///< Number of frames represented (typically 2 for 20ms)
};

/**
 * @brief Configuration for loading and running an inference engine.
 */
struct ModelConfig {
    std::string model_path;             ///< Path to the encrypted .biona bundle
    std::string key_id;                 ///< Key identifier for SecretManager lookup
    int32_t     sample_rate_hz = 16000; ///< Expected audio sample rate (must be 16000)
    int32_t     chunk_size_ms  = 20;    ///< Audio chunk duration in milliseconds
};

/**
 * @brief Metadata describing a loaded inference engine backend.
 */
struct EngineInfo {
    std::string runtime_name;       ///< e.g. "onnxruntime", "tflite", "coreml"
    std::string runtime_version;    ///< Runtime version string
    bool        supports_streaming; ///< True if stateful streaming is supported
};

/**
 * @brief Result produced by one inference call.
 *
 * Contains the decoded transcript, speaker embedding, timing, and VAD flag.
 * The embedding is always 512-dimensional.
 */
struct InferenceResult {
    std::vector<float> embedding;   ///< 512-dim speaker/speech embedding
    std::string        text;        ///< CTC-decoded transcript for this chunk
    int32_t            latency_ms;  ///< Wall-clock inference latency in ms
    bool               vad_triggered; ///< True if VAD indicated speech for this chunk
};

} // namespace biona
