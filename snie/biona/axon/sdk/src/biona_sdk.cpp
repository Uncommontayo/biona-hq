#include "biona/biona.h"

#include "biona/core/errors.hpp"
#include "biona/core/interfaces/inference_engine.hpp"
#include "biona/core/interfaces/vad.hpp"
#include "biona/core/interfaces/frontend.hpp"
#include "biona/core/interfaces/signal_layer.hpp"
#include "biona/core/signal_thread_pool.hpp"
#include "biona/core/types.hpp"
#include "biona/security/safe_log.hpp"
#include "biona/security/secret_manager.hpp"
#include "biona/axons/onnx_engine.hpp"

#include <cstring>
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// Internal context struct
// ---------------------------------------------------------------------------

struct EngineContext {
    std::unique_ptr<biona::InferenceEngine>  engine;
    std::unique_ptr<biona::VAD>              vad;
    // Frontend and SignalLayer are interfaces — concrete impls added when available
    // std::unique_ptr<biona::Frontend>       frontend;
    // std::unique_ptr<biona::SignalLayer>    signal_layer;
    std::unique_ptr<biona::SignalThreadPool> thread_pool;

    // Callback registration
    BIONA_SignalCallback callback  = nullptr;
    void*               user_data = nullptr;

    // Persistent output buffers (valid until next ProcessChunk call)
    std::string         last_text;
    std::vector<float>  last_embedding;

    int64_t chunk_id = 0;
};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

static BIONA_Error toC(biona::BionaError code) {
    switch (code) {
        case biona::BionaError::OK:                 return BIONA_OK;
        case biona::BionaError::InitFailed:         return BIONA_ERR_INIT_FAILED;
        case biona::BionaError::ModelDecryptFailed: return BIONA_ERR_DECRYPT_FAILED;
        case biona::BionaError::InvalidAudio:       return BIONA_ERR_INVALID_AUDIO;
        case biona::BionaError::InferenceFailed:    return BIONA_ERR_INFERENCE_FAILED;
        case biona::BionaError::OOM:                return BIONA_ERR_OOM;
        default:                                    return BIONA_ERR_INIT_FAILED;
    }
}

// ---------------------------------------------------------------------------
// BIONA_Create
// ---------------------------------------------------------------------------

extern "C" BIONA_Error BIONA_Create(const BIONA_Config* cfg, BIONA_Engine* out_engine) {
    if (!cfg || !out_engine) return BIONA_ERR_INIT_FAILED;

    try {
        biona::SafeLog::configure(
            static_cast<biona::LogLevel>(cfg->log_level));

        auto ctx = std::make_unique<EngineContext>();

        // Build ModelConfig
        biona::ModelConfig model_cfg;
        model_cfg.model_path     = cfg->model_bundle_path ? cfg->model_bundle_path : "";
        model_cfg.key_id         = cfg->secret_manager_type ? cfg->secret_manager_type : "env";
        model_cfg.sample_rate_hz = cfg->sample_rate_hz;
        model_cfg.chunk_size_ms  = cfg->chunk_size_ms;

        // Validate sample rate
        biona::throwIf(cfg->sample_rate_hz != 16000,
                       biona::BionaError::InvalidAudio,
                       "sample_rate_hz must be 16000");

        // Inference engine (ONNX)
        ctx->engine = std::make_unique<biona::OnnxEngine>();
        ctx->engine->initialize(model_cfg);

        // VAD (WebRTC)
        biona::VADConfig vad_cfg;
        vad_cfg.type = biona::VADType::WEBRTC;
        ctx->vad = biona::createVAD(vad_cfg);

        // Async thread pool (if enabled)
        if (cfg->enable_async) {
            ctx->thread_pool = std::make_unique<biona::SignalThreadPool>(2, 500);
            ctx->thread_pool->setCallback(
                [out = ctx.get()](int64_t chunk_id, const biona::Signal& sig) {
                    if (!out->callback) return;
                    BIONA_AsyncSignal async{};
                    async.chunk_id   = chunk_id;
                    async.confidence = 1.0f;
                    if (sig.emotion) {
                        async.signal_type = 0;
                        async.values[0]   = sig.emotion->valence;
                        async.values[1]   = sig.emotion->arousal;
                        async.values[2]   = sig.emotion->stability;
                        out->callback(&async, out->user_data);
                    }
                    if (sig.speaker) {
                        async.signal_type = 1;
                        async.values[0]   = sig.speaker->confidence;
                        async.confidence  = sig.speaker->confidence;
                        out->callback(&async, out->user_data);
                    }
                    if (sig.intent) {
                        async.signal_type = 2;
                        async.values[0]   = sig.intent->confidence;
                        async.confidence  = sig.intent->confidence;
                        out->callback(&async, out->user_data);
                    }
                });
        }

        *out_engine = ctx.release();
        return BIONA_OK;

    } catch (const biona::BionaException& e) {
        biona::SafeLog::error("BIONA_Create", e.what());
        return toC(e.code());
    } catch (const std::exception& e) {
        biona::SafeLog::error("BIONA_Create", e.what());
        return BIONA_ERR_INIT_FAILED;
    } catch (...) {
        return BIONA_ERR_INIT_FAILED;
    }
}

// ---------------------------------------------------------------------------
// BIONA_Destroy
// ---------------------------------------------------------------------------

extern "C" void BIONA_Destroy(BIONA_Engine engine) {
    if (engine) {
        delete static_cast<EngineContext*>(engine);
    }
}

// ---------------------------------------------------------------------------
// BIONA_ProcessChunk
// ---------------------------------------------------------------------------

extern "C" BIONA_Error BIONA_ProcessChunk(BIONA_Engine   engine,
                                          const int16_t* pcm,
                                          size_t         n_samples,
                                          BIONA_Result*  out_result) {
    if (!engine || !pcm || !out_result) return BIONA_ERR_INVALID_AUDIO;

    auto* ctx = static_cast<EngineContext*>(engine);

    try {
        int64_t ts_ms = ctx->chunk_id * 20; // 20ms per chunk

        biona::AudioChunk chunk{pcm, n_samples, ts_ms};

        // VAD gate
        if (!ctx->vad->isSpeech(chunk)) {
            out_result->text          = "";
            out_result->timestamp_ms  = ts_ms;
            out_result->latency_ms    = 0;
            out_result->embedding     = nullptr;
            out_result->embedding_len = 0;
            ++ctx->chunk_id;
            return BIONA_OK;
        }

        // Feature extraction — placeholder (real Frontend added later)
        biona::AudioFeatures features{};
        features.n_frames = 2;
        // In a full impl: ctx->frontend->extract(chunk, features);

        // Inference
        biona::InferenceResult result = ctx->engine->run(features);

        // Persist output strings/vectors (valid until next call)
        ctx->last_text      = result.text;
        ctx->last_embedding = result.embedding;

        out_result->text          = ctx->last_text.c_str();
        out_result->timestamp_ms  = ts_ms;
        out_result->latency_ms    = result.latency_ms;
        out_result->embedding     = ctx->last_embedding.data();
        out_result->embedding_len = static_cast<int32_t>(ctx->last_embedding.size());

        // Enqueue async signal task if pool is active
        if (ctx->thread_pool) {
            biona::SignalTask task;
            task.chunk_id = ctx->chunk_id;
            task.result   = result;
            ctx->thread_pool->enqueue(std::move(task));
        }

        ++ctx->chunk_id;
        return BIONA_OK;

    } catch (const biona::BionaException& e) {
        biona::SafeLog::error("ProcessChunk", e.what());
        return toC(e.code());
    } catch (const std::exception& e) {
        biona::SafeLog::error("ProcessChunk", e.what());
        return BIONA_ERR_INFERENCE_FAILED;
    } catch (...) {
        return BIONA_ERR_INFERENCE_FAILED;
    }
}

// ---------------------------------------------------------------------------
// BIONA_RegisterSignalCallback
// ---------------------------------------------------------------------------

extern "C" BIONA_Error BIONA_RegisterSignalCallback(BIONA_Engine        engine,
                                                    BIONA_SignalCallback cb,
                                                    void*               user_data) {
    if (!engine) return BIONA_ERR_INIT_FAILED;
    auto* ctx       = static_cast<EngineContext*>(engine);
    ctx->callback   = cb;
    ctx->user_data  = user_data;
    return BIONA_OK;
}

// ---------------------------------------------------------------------------
// BIONA_ErrorString
// ---------------------------------------------------------------------------

extern "C" const char* BIONA_ErrorString(BIONA_Error err) {
    switch (err) {
        case BIONA_OK:                    return "OK";
        case BIONA_ERR_INIT_FAILED:       return "Initialisation failed";
        case BIONA_ERR_DECRYPT_FAILED:    return "Model decryption failed";
        case BIONA_ERR_INVALID_AUDIO:     return "Invalid audio input";
        case BIONA_ERR_INFERENCE_FAILED:  return "Inference failed";
        case BIONA_ERR_OOM:               return "Out of memory";
        default:                          return "Unknown error";
    }
}
