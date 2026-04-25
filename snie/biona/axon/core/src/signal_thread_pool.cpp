#include "biona/core/signal_thread_pool.hpp"
#include "biona/security/safe_log.hpp"

#include <chrono>
#include <thread>

namespace biona {

SignalThreadPool::SignalThreadPool(size_t n_threads, int32_t timeout_ms)
    : timeout_ms_(timeout_ms)
{
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this]{ workerLoop(); });
    }
}

SignalThreadPool::~SignalThreadPool() {
    running_.store(false, std::memory_order_relaxed);
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

bool SignalThreadPool::enqueue(SignalTask&& task) noexcept {
    task.enqueued_at = std::chrono::steady_clock::now();
    if (!queue_.push(std::move(task))) {
        SafeLog::metric(BIONA_METRIC_QUEUE_OVERFLOW, 1.0);
        return false;
    }
    return true;
}

void SignalThreadPool::setCallback(AsyncSignalCallback cb) {
    std::lock_guard<std::mutex> lk(callback_mutex_);
    callback_ = std::move(cb);
}

void SignalThreadPool::workerLoop() {
    using Clock = std::chrono::steady_clock;

    while (running_.load(std::memory_order_relaxed)) {
        SignalTask task;
        if (!queue_.pop(task)) {
            // Brief yield to avoid spinning at 100% CPU on an empty queue
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        // Check timeout
        auto now = Clock::now();
        auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - task.enqueued_at).count();

        if (age_ms > static_cast<long long>(timeout_ms_)) {
            SafeLog::metric(BIONA_METRIC_SIGNAL_TIMEOUT, 1.0);
            // Deliver a null/empty signal to unblock the caller
            Signal null_signal;
            null_signal.text         = task.result.text;
            null_signal.embedding    = task.result.embedding;
            null_signal.latency_ms   = task.result.latency_ms;
            // Optional fields remain null (timed out)

            std::lock_guard<std::mutex> lk(callback_mutex_);
            if (callback_) callback_(task.chunk_id, null_signal);
            continue;
        }

        // In a full implementation each worker would own an inference model
        // instance for computing SER / speaker / intent signals here.
        // For now we assemble and deliver a Signal with embedding only.
        Signal sig;
        sig.text       = task.result.text;
        sig.embedding  = task.result.embedding;
        sig.latency_ms = task.result.latency_ms;
        // Optional emotion / speaker / intent filled by the worker's model here

        std::lock_guard<std::mutex> lk(callback_mutex_);
        if (callback_) callback_(task.chunk_id, sig);
    }
}

} // namespace biona
