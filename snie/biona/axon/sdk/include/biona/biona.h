/**
 * @file biona.h
 * @brief Biona Axon public C API (C99 compatible).
 *
 * This is the ONLY surface that external developers and platform adapters touch.
 * All types and functions use the BIONA_ prefix and are declared extern "C".
 *
 * Thread safety:
 *   - BIONA_ProcessChunk is NOT thread-safe per engine handle.
 *   - Create one BIONA_Engine per thread if parallel inference is needed.
 *   - BIONA_Create / BIONA_Destroy must not be called concurrently on the
 *     same handle.
 */

#ifndef BIONA_H
#define BIONA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque engine handle
// ---------------------------------------------------------------------------

typedef void* BIONA_Engine;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

typedef struct {
    const char* model_bundle_path;    /**< Path to the .biona encrypted bundle */
    int         sample_rate_hz;       /**< Must be 16000 */
    int         chunk_size_ms;        /**< Recommended: 20 */
    int         enable_async;         /**< 1 = enable SER/speaker/intent, 0 = sync only */
    int         log_level;            /**< 0=PERF_ONLY  1=SYSTEM  2=DEBUG */
    const char* secret_manager_type;  /**< "env" | "keychain" | "keystore" | "tpm" */
} BIONA_Config;

// ---------------------------------------------------------------------------
// Result (synchronous)
// ---------------------------------------------------------------------------

typedef struct {
    const char*  text;          /**< Null-terminated transcript. Owned by SDK.
                                     Valid until the next BIONA_ProcessChunk call. */
    int64_t      timestamp_ms;  /**< Timestamp of the originating chunk */
    int32_t      latency_ms;    /**< End-to-end inference latency in ms */
    const float* embedding;     /**< 512 floats. Owned by SDK.
                                     Valid until the next BIONA_ProcessChunk call. */
    int32_t      embedding_len; /**< Always 512 */
} BIONA_Result;

// ---------------------------------------------------------------------------
// Async signal
// ---------------------------------------------------------------------------

typedef struct {
    int64_t chunk_id;       /**< Chunk sequence number this signal belongs to */
    int     signal_type;    /**< 0=EMOTION  1=SPEAKER  2=INTENT */
    float   values[8];      /**< Signal-specific payload:
                                   EMOTION: values[0]=valence, [1]=arousal, [2]=stability
                                   SPEAKER: values[0]=confidence
                                   INTENT:  values[0]=confidence                      */
    float   confidence;     /**< Overall confidence score */
} BIONA_AsyncSignal;

typedef void (*BIONA_SignalCallback)(const BIONA_AsyncSignal* signal, void* user_data);

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

typedef enum {
    BIONA_OK                   = 0,
    BIONA_ERR_INIT_FAILED      = 1,
    BIONA_ERR_DECRYPT_FAILED   = 2,
    BIONA_ERR_INVALID_AUDIO    = 3,
    BIONA_ERR_INFERENCE_FAILED = 4,
    BIONA_ERR_OOM              = 5,
} BIONA_Error;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/**
 * @brief Create and initialise a Biona engine instance.
 *
 * @param cfg         Engine configuration.
 * @param out_engine  On success: set to the new engine handle.
 * @return BIONA_OK on success, or an error code.
 */
BIONA_Error BIONA_Create(const BIONA_Config* cfg, BIONA_Engine* out_engine);

/**
 * @brief Destroy a Biona engine instance and free all resources.
 *
 * @param engine  Handle returned by BIONA_Create.
 */
void BIONA_Destroy(BIONA_Engine engine);

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/**
 * @brief Process one 20ms audio chunk.
 *
 * NOT thread-safe. One call at a time per engine handle.
 * Must return within the latency budget (p99 < 150ms per the Biona spec §12).
 *
 * @param engine    Engine handle.
 * @param pcm       PCM samples (16-bit signed, 16kHz).
 * @param n_samples Number of samples (must be sample_rate_hz * chunk_size_ms / 1000).
 * @param out_result On BIONA_OK: populated with text, embedding, and latency.
 *                   The text and embedding pointers are valid until the next call.
 * @return BIONA_OK on success.
 */
BIONA_Error BIONA_ProcessChunk(BIONA_Engine     engine,
                               const int16_t*   pcm,
                               size_t           n_samples,
                               BIONA_Result*    out_result);

// ---------------------------------------------------------------------------
// Async signal delivery
// ---------------------------------------------------------------------------

/**
 * @brief Register a callback for asynchronous optional signals.
 *
 * The callback is invoked from a worker thread when a SER, speaker, or
 * intent signal is ready. The callback must be thread-safe.
 *
 * @param engine     Engine handle.
 * @param cb         Callback function.
 * @param user_data  Opaque pointer forwarded to each callback invocation.
 * @return BIONA_OK on success.
 */
BIONA_Error BIONA_RegisterSignalCallback(BIONA_Engine        engine,
                                         BIONA_SignalCallback cb,
                                         void*               user_data);

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/**
 * @brief Return a human-readable string for a BIONA_Error code.
 *
 * The returned pointer points to a static string. Never free it.
 */
const char* BIONA_ErrorString(BIONA_Error err);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* BIONA_H */
