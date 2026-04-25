#pragma once

#include "biona/core/types.hpp"

#include <functional>
#include <memory>
#include <unordered_map>

namespace biona {

/**
 * @brief Abstract base class for all inference engine backends.
 *
 * Threading guarantees:
 *   - One instance is bound to a single thread. Do NOT share instances
 *     across threads.
 *   - initialize() must be called exactly once before any call to run().
 *   - reset() and run() must NOT be called concurrently.
 *   - info() is safe to call from any thread after initialize() returns.
 *
 * Hot-path contract:
 *   - run() is called on every 20ms audio chunk that passes VAD.
 *   - run() MUST NOT allocate heap memory (all buffers pre-allocated in initialize()).
 *   - run() latency budget: < 15ms on the target device.
 */
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    /**
     * @brief Initialise the engine with the given model configuration.
     *
     * Loads and decrypts the model, allocates all working buffers,
     * and validates the ONNX contract.
     *
     * @param cfg  Model and audio configuration.
     * @return true on success.
     * @throws BionaException on any initialisation failure.
     */
    virtual bool initialize(const ModelConfig& cfg) = 0;

    /**
     * @brief Run inference on one frame of audio features.
     *
     * @pre initialize() has been called and returned true.
     * @param features  Log-mel spectrogram for a single 20ms chunk.
     * @return InferenceResult containing transcript text, 512-dim embedding,
     *         and measured latency.
     *
     * Hot path — no heap allocation permitted inside this method.
     */
    virtual InferenceResult run(const AudioFeatures& features) = 0;

    /**
     * @brief Reset streaming state between utterances.
     *
     * Zeroes the streaming memory tensors (memory bank, left-context key/val)
     * so the next call to run() starts fresh.
     *
     * Must NOT be called concurrently with run().
     */
    virtual void reset() = 0;

    /**
     * @brief Return static metadata about this engine backend.
     *
     * Safe to call from any thread after initialize() has returned.
     */
    virtual EngineInfo info() const = 0;

    // Non-copyable, non-movable — engines hold hardware/driver resources.
    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&)                 = delete;
    InferenceEngine& operator=(InferenceEngine&&)      = delete;

protected:
    InferenceEngine() = default;
};

// ---------------------------------------------------------------------------
// Factory infrastructure
// ---------------------------------------------------------------------------

/**
 * @brief Supported inference engine backends.
 */
enum class EngineType {
    ONNX,    ///< ONNX Runtime (cross-platform)
    TFLITE,  ///< TensorFlow Lite (Android / Linux)
    COREML,  ///< Core ML (iOS / macOS)
};

/// Callable that creates and returns a new InferenceEngine instance.
using EngineCreator = std::function<std::unique_ptr<InferenceEngine>()>;

/**
 * @brief Registry and factory for InferenceEngine backends.
 *
 * Each engine module registers itself at static-init time via registerEngine().
 * create() instantiates and initialises an engine of the requested type.
 *
 * Thread safety: registerEngine() must be called before any concurrent access.
 *               create() is safe to call from multiple threads concurrently.
 */
class InferenceEngineFactory {
public:
    /**
     * @brief Create and initialise an engine of the given type.
     *
     * @param type  Requested backend.
     * @param cfg   Model configuration forwarded to initialize().
     * @return Owning pointer to the initialised engine.
     * @throws BionaException(UnsupportedRuntime) if type is not registered.
     * @throws BionaException(InitFailed) if initialize() fails.
     */
    static std::unique_ptr<InferenceEngine> create(EngineType type,
                                                   const ModelConfig& cfg);

    /**
     * @brief Register a creator function for an engine type.
     *
     * Typically called once at static-init time from each engine module.
     *
     * @param type     The engine type this creator produces.
     * @param creator  Callable returning a new (uninitialised) engine.
     */
    static void registerEngine(EngineType type, EngineCreator creator);

private:
    /// Returns the singleton registry map.
    static std::unordered_map<EngineType, EngineCreator>& registry();
};

} // namespace biona
