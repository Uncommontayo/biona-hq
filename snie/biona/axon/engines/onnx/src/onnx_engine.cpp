#include "biona/axons/onnx_engine.hpp"
#include "biona/core/errors.hpp"
#include "biona/core/onnx_contract.hpp"
#include "biona/security/safe_log.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace biona {

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

bool OnnxEngine::initialize(const ModelConfig& cfg) {
    // 1. Load and decrypt model
    ModelLoader loader;
    // SecretManager is created from the key_id field (type string)
    auto secrets = createSecretManager("env");
    SecureBuffer model_bytes = loader.load(cfg.model_path, *secrets);

    // 2. Session options
    session_opts_.SetIntraOpNumThreads(1);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3. Load session from in-memory buffer (never from file path)
    session_ = std::make_unique<Ort::Session>(
        env_,
        model_bytes.data(),
        model_bytes.size(),
        session_opts_);

    // 4. Validate tensor names against contract
    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> actual_inputs, actual_outputs;
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        actual_inputs.emplace_back(session_->GetInputNameAllocated(i, alloc).get());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        actual_outputs.emplace_back(session_->GetOutputNameAllocated(i, alloc).get());
    }
    throwIf(!ONNXContractValidator::validate(actual_inputs, actual_outputs),
            BionaError::InitFailed,
            "ONNX model tensor names do not match the contract in onnx_contract.hpp");

    // Determine vocab size from the logits output shape
    auto logits_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto shape = logits_info.GetShape();
    // Expected shape: [CHUNK_FRAMES, 1, vocab_size]
    throwIf(shape.size() < 3, BionaError::InitFailed, "Unexpected logits tensor rank");
    vocab_size_ = static_cast<int32_t>(shape[2]);

    // 5. Pre-allocate all input/output buffers
    input_chunk_  .assign(ONNX_CHUNK_FRAMES  * ONNX_MEL_BANDS,           0.0f);
    input_memory_ .assign(ONNX_MEMORY_VECTORS* ONNX_MEL_BANDS,           0.0f);
    input_lc_key_ .assign(ONNX_LEFT_CONTEXT_FRAMES * ONNX_MEL_BANDS,     0.0f);
    input_lc_val_ .assign(ONNX_LEFT_CONTEXT_FRAMES * ONNX_MEL_BANDS,     0.0f);

    output_logits_   .assign(ONNX_CHUNK_FRAMES * vocab_size_,             0.0f);
    output_memory_   .assign(ONNX_MEMORY_VECTORS * ONNX_MEL_BANDS,        0.0f);
    output_lc_key_   .assign(ONNX_LEFT_CONTEXT_FRAMES * ONNX_MEL_BANDS,   0.0f);
    output_lc_val_   .assign(ONNX_LEFT_CONTEXT_FRAMES * ONNX_MEL_BANDS,   0.0f);
    output_embedding_.assign(ONNX_EMBEDDING_DIM,                          0.0f);

    // 6. Create IOBinding
    binding_ = std::make_unique<Ort::IoBinding>(*session_);

    initialized_ = true;
    SafeLog::event(BIONA_EVENT_ENGINE_INITIALIZED);
    return true;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

InferenceResult OnnxEngine::run(const AudioFeatures& features) {
    assert(initialized_ && "OnnxEngine::run() called before initialize()");

    auto t0 = std::chrono::steady_clock::now();

    // 1. Copy features into pre-allocated input chunk tensor (no new allocation)
    static_assert(sizeof(features.mel_bands) == ONNX_MEL_BANDS * sizeof(float));
    std::copy(features.mel_bands.begin(), features.mel_bands.end(),
              input_chunk_.begin());

    // 2. Build input tensor shapes and bind via IOBinding
    std::array<int64_t, 3> chunk_shape  = {1, ONNX_CHUNK_FRAMES,          ONNX_MEL_BANDS};
    std::array<int64_t, 3> mem_shape    = {1, ONNX_MEMORY_VECTORS,        ONNX_MEL_BANDS};
    std::array<int64_t, 3> lc_shape     = {1, ONNX_LEFT_CONTEXT_FRAMES,   ONNX_MEL_BANDS};

    auto make_input = [&](std::vector<float>& buf, const int64_t* shape, size_t rank)
        -> Ort::Value {
        return Ort::Value::CreateTensor<float>(
            mem_info_, buf.data(), buf.size(), shape, rank);
    };

    Ort::Value t_chunk  = make_input(input_chunk_,  chunk_shape.data(), 3);
    Ort::Value t_mem    = make_input(input_memory_,  mem_shape.data(),   3);
    Ort::Value t_lc_key = make_input(input_lc_key_,  lc_shape.data(),    3);
    Ort::Value t_lc_val = make_input(input_lc_val_,  lc_shape.data(),    3);

    binding_->BindInput(ONNX_INPUT_CHUNK.data(),   t_chunk);
    binding_->BindInput(ONNX_INPUT_MEMORY.data(),  t_mem);
    binding_->BindInput(ONNX_INPUT_LC_KEY.data(),  t_lc_key);
    binding_->BindInput(ONNX_INPUT_LC_VAL.data(),  t_lc_val);

    // Output binding (pre-allocated buffers)
    std::array<int64_t, 3> logits_shape = {ONNX_CHUNK_FRAMES, 1, vocab_size_};
    std::array<int64_t, 2> emb_shape    = {1, ONNX_EMBEDDING_DIM};

    auto bind_output = [&](std::vector<float>& buf,
                           const std::string_view& name,
                           const int64_t* shape, size_t rank) {
        Ort::Value t = Ort::Value::CreateTensor<float>(
            mem_info_, buf.data(), buf.size(), shape, rank);
        binding_->BindOutput(name.data(), t);
    };

    bind_output(output_logits_,    ONNX_OUTPUT_LOGITS,    logits_shape.data(), 3);
    bind_output(output_memory_,    ONNX_OUTPUT_MEMORY,    mem_shape.data(),    3);
    bind_output(output_lc_key_,    ONNX_OUTPUT_LC_KEY,    lc_shape.data(),     3);
    bind_output(output_lc_val_,    ONNX_OUTPUT_LC_VAL,    lc_shape.data(),     3);
    bind_output(output_embedding_, ONNX_OUTPUT_EMBEDDING, emb_shape.data(),    2);

    // 3. Run
    session_->Run(Ort::RunOptions{nullptr}, *binding_);

    // 4. CTC greedy decode logits → text
    std::string text = ctcGreedyDecode(output_logits_, ONNX_CHUNK_FRAMES, vocab_size_);

    // 5. Copy embedding
    std::vector<float> embedding(output_embedding_.begin(), output_embedding_.end());

    // 6. Update streaming state tensors (copy outputs → inputs for next call)
    std::copy(output_memory_.begin(), output_memory_.end(), input_memory_.begin());
    std::copy(output_lc_key_.begin(), output_lc_key_.end(), input_lc_key_.begin());
    std::copy(output_lc_val_.begin(), output_lc_val_.end(), input_lc_val_.begin());

    // 7. Measure latency
    auto t1 = std::chrono::steady_clock::now();
    int32_t latency_ms = static_cast<int32_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    SafeLog::metric(BIONA_METRIC_INFERENCE_LATENCY_MS, static_cast<double>(latency_ms));

    return InferenceResult{
        std::move(embedding),
        std::move(text),
        latency_ms,
        /*vad_triggered=*/true  // VAD already gated upstream
    };
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void OnnxEngine::reset() {
    std::fill(input_memory_.begin(),  input_memory_.end(),  0.0f);
    std::fill(input_lc_key_.begin(),  input_lc_key_.end(),  0.0f);
    std::fill(input_lc_val_.begin(),  input_lc_val_.end(),  0.0f);
    SafeLog::event(BIONA_EVENT_ENGINE_RESET);
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

EngineInfo OnnxEngine::info() const {
    return EngineInfo{
        "onnxruntime",
        OrtGetApiBase()->GetVersionString(),
        /*supports_streaming=*/true
    };
}

// ---------------------------------------------------------------------------
// CTC greedy decoder
// ---------------------------------------------------------------------------

std::string OnnxEngine::ctcGreedyDecode(const std::vector<float>& logits,
                                         int32_t n_frames,
                                         int32_t vocab) const {
    // logits layout: [n_frames, 1, vocab]
    // Blank token = index 0

    std::string result;
    int32_t prev_token = 0; // blank

    for (int32_t t = 0; t < n_frames; ++t) {
        const float* row = logits.data() + t * vocab;

        // Argmax
        int32_t best = 0;
        float   best_val = row[0];
        for (int32_t v = 1; v < vocab; ++v) {
            if (row[v] > best_val) {
                best_val = row[v];
                best     = v;
            }
        }

        // Collapse repeats and remove blank (token 0)
        if (best != prev_token && best != 0) {
            // Map token index to character (assumes simple char vocabulary)
            // In a real deployment this mapping comes from the tokenizer;
            // here we emit the raw token index as a placeholder character.
            result += static_cast<char>(best < 128 ? best : '?');
        }
        prev_token = best;
    }

    return result;
}

} // namespace biona
