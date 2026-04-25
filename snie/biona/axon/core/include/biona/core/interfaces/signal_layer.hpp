#pragma once

#include "biona/core/signal.hpp"
#include "biona/core/types.hpp"

namespace biona {

/**
 * @brief Abstract interface for assembling the Signal envelope.
 *
 * SignalLayer combines the synchronous InferenceResult (text + embedding)
 * with optional async signals (emotion, speaker, intent) into the final
 * Signal struct that is delivered to the caller.
 *
 * Threading guarantees:
 *   - assemble() is called from the main pipeline thread only.
 *   - attachAsyncSignal() may be called from worker threads.
 *     Implementations must protect shared state with a mutex or lock-free
 *     structure (e.g. the SPSC queue or a concurrent map).
 *   - assemble() and attachAsyncSignal() must NOT be called concurrently
 *     for the same chunk_id.
 */
class SignalLayer {
public:
    virtual ~SignalLayer() = default;

    /**
     * @brief Assemble a Signal from a synchronous inference result.
     *
     * Populates schema_version, text, timestamp_ms, latency_ms, and embedding
     * from @p sync_result. Optional signals are null at this point and may
     * be populated later via attachAsyncSignal().
     *
     * Called from the main pipeline thread.
     *
     * @param sync_result  The InferenceResult from OnnxEngine::run().
     * @return Assembled Signal with optional fields null.
     */
    virtual Signal assemble(const InferenceResult& sync_result) = 0;

    /**
     * @brief Attach an async EmotionSignal to a previously assembled chunk.
     *
     * Thread-safe — may be called from any worker thread.
     *
     * @param chunk_id  The chunk this signal corresponds to.
     * @param emotion   Completed emotion signal to attach.
     */
    virtual void attachAsyncSignal(int64_t chunk_id, const EmotionSignal& emotion) = 0;

    /**
     * @brief Attach an async SpeakerSignal to a previously assembled chunk.
     *
     * Thread-safe — may be called from any worker thread.
     *
     * @param chunk_id  The chunk this signal corresponds to.
     * @param speaker   Completed speaker signal to attach.
     */
    virtual void attachAsyncSignal(int64_t chunk_id, const SpeakerSignal& speaker) = 0;

    /**
     * @brief Attach an async IntentSignal to a previously assembled chunk.
     *
     * Thread-safe — may be called from any worker thread.
     *
     * @param chunk_id  The chunk this signal corresponds to.
     * @param intent    Completed intent signal to attach.
     */
    virtual void attachAsyncSignal(int64_t chunk_id, const IntentSignal& intent) = 0;

protected:
    SignalLayer() = default;
};

} // namespace biona
