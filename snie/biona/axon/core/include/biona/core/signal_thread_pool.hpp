#pragma once

/**
 * @file signal_thread_pool.hpp
 * @brief Fixed thread pool for computing optional async signals (SER, speaker ID, intent).
 *
 * The main pipeline thread pushes SignalTask items into the SPSC queue via enqueue().
 * Worker threads pop tasks, compute optional signals, and deliver results via
 * the registered AsyncSignalCallback.
 *
 * Design notes:
 *   - enqueue() is called from the hot-path (main thread). No heap allocation.
 *   - If the queue is full, enqueue() returns false and logs BIONA_METRIC_QUEUE_OVERFLOW.
 *   - If a task exceeds timeout_ms, a null Signal is delivered and
 *     BIONA_METRIC_SIGNAL_TIMEOUT is logged.
 *   - Workers check running_ with memory_order_relaxed — clean shutdown within 100ms.
 */

#include "biona/core/types.hpp"
#include "biona/core/signal.hpp"
#include "biona/core/spsc_queue.hpp"
// NOTE: safe_log.hpp is included in the .cpp only — security/ must not be
//       a compile-time dependency of core/ headers.

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace biona {

/**
 * @brief Task envelope for the async signal thread pool.
 */
struct SignalTask {
    int64_t                               chunk_id;
    InferenceResult                       result;
    std::chrono::steady_clock::time_point enqueued_at;
};

/// Callback type invoked when an async signal is ready.
using AsyncSignalCallback = std::function<void(int64_t chunk_id, const Signal&)>;

/**
 * @brief Fixed thread pool for async optional signal computation.
 *
 * Thread safety:
 *   - enqueue() is safe to call from the main thread only (SPSC constraint).
 *   - setCallback() must be called before any enqueue() call.
 *   - Destructor joins all worker threads cleanly.
 */
class SignalThreadPool {
public:
    /**
     * @brief Construct the thread pool and start worker threads.
     *
     * @param n_threads  Number of worker threads (typically 1–4).
     * @param timeout_ms Maximum ms a task may wait before being timed out.
     */
    explicit SignalThreadPool(size_t n_threads, int32_t timeout_ms = 500);

    /**
     * @brief Stop workers and join all threads.
     *
     * Workers will drain remaining tasks or time out within timeout_ms,
     * then exit within 100ms of running_ being set to false.
     */
    ~SignalThreadPool();

    // Non-copyable, non-movable
    SignalThreadPool(const SignalThreadPool&)            = delete;
    SignalThreadPool& operator=(const SignalThreadPool&) = delete;

    /**
     * @brief Enqueue a task for async processing.
     *
     * Hot path — no heap allocation. Called from the main thread only.
     *
     * @param task  Task to enqueue (moved in).
     * @return true if enqueued; false if queue is full (overflow logged).
     */
    bool enqueue(SignalTask&& task) noexcept;

    /**
     * @brief Register the callback invoked when async signals complete.
     *
     * Must be called before enqueue(). The callback is invoked from a worker thread.
     */
    void setCallback(AsyncSignalCallback cb);

private:
    static constexpr size_t QUEUE_CAPACITY = 64;

    SPSCQueue<SignalTask, QUEUE_CAPACITY> queue_;
    std::vector<std::thread>             workers_;
    std::atomic<bool>                    running_{true};
    AsyncSignalCallback                  callback_;
    std::mutex                           callback_mutex_;
    int32_t                              timeout_ms_;

    void workerLoop();
};

} // namespace biona
