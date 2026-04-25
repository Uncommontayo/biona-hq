#pragma once

/**
 * @file onnx_engine.hpp
 * @brief ONNX Runtime inference engine for streaming Emformer ASR.
 *
 * Threading: OnnxEngine is NOT thread-safe.
 *   - One OnnxEngine instance must be created per thread.
 *   - initialize(), run(), and reset() must NOT be called concurrently.
 *   - info() is safe to call after initialize() returns.
 *
 * Memory: ALL input and output tensors are pre-allocated in initialize().
 *         run() performs ZERO heap allocations.
 *
 * The engine uses IOBinding for zero-copy tensor reuse between calls.
 */

#include "biona/core/interfaces/inference_engine.hpp"
#include "biona/core/types.hpp"
#include "biona/core/onnx_contract.hpp"
#include "biona/security/model_loader.hpp"
#include "biona/security/secret_manager.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace biona {

/**
 * @brief ONNX Runtime implementation of InferenceEngine.
 *
 * Implements stateful streaming inference using the Emformer architecture.
 * Streaming state (memory bank, left-context key/value) is maintained
 * across calls to run() and zeroed by reset().
 */
class OnnxEngine final : public InferenceEngine {
public:
    OnnxEngine() = default;
    ~OnnxEngine() override = default;

    bool          initialize(const ModelConfig& cfg) override;
    InferenceResult run(const AudioFeatures& features) override;
    void          reset() override;
    EngineInfo    info() const override;

private:
    // ORT session objects
    Ort::Env            env_{ORT_LOGGING_LEVEL_WARNING, "BionaAxon"};
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session>    session_;
    std::unique_ptr<Ort::IoBinding>  binding_;
    Ort::MemoryInfo  mem_info_{Ort::MemoryInfo::CreateCpu(
                                   OrtArenaAllocator, OrtMemTypeDefault)};

    // Pre-allocated input tensors (zero alloc in run())
    std::vector<float> input_chunk_;       // [1, CHUNK_FRAMES, MEL_BANDS]
    std::vector<float> input_memory_;      // [1, MEMORY_VECTORS, MEL_BANDS]
    std::vector<float> input_lc_key_;      // [1, LC_FRAMES, MEL_BANDS]
    std::vector<float> input_lc_val_;      // [1, LC_FRAMES, MEL_BANDS]

    // Pre-allocated output buffers
    std::vector<float> output_logits_;     // [CHUNK_FRAMES, 1, vocab_size]
    std::vector<float> output_memory_;     // [1, MEMORY_VECTORS, MEL_BANDS]
    std::vector<float> output_lc_key_;     // [1, LC_FRAMES, MEL_BANDS]
    std::vector<float> output_lc_val_;     // [1, LC_FRAMES, MEL_BANDS]
    std::vector<float> output_embedding_;  // [1, EMBEDDING_DIM]

    int32_t vocab_size_ = 0;
    bool    initialized_ = false;

    // CTC greedy decode: logits [T x vocab_size] → string
    std::string ctcGreedyDecode(const std::vector<float>& logits,
                                int32_t n_frames,
                                int32_t vocab) const;
};

} // namespace biona
