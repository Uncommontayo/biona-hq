#pragma once

/**
 * @file spsc_queue.hpp
 * @brief Lock-free single-producer single-consumer ring queue.
 *
 * Design notes:
 *   - Uses acquire/release memory ordering — never seq_cst in push/pop.
 *   - Cache-line aligned head/tail counters to prevent false sharing.
 *   - Zero heap allocation after construction.
 *   - Capacity must be a power of two (enforced by static_assert).
 *
 * Thread safety:
 *   - Exactly ONE producer thread may call push().
 *   - Exactly ONE consumer thread may call pop().
 *   - push() and pop() may run concurrently with each other.
 *   - empty() and full() are approximate — only safe within the same thread.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace biona {

template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0,
        "SPSCQueue Capacity must be a power of two");
    static_assert(std::is_move_constructible_v<T> || std::is_copy_constructible_v<T>,
        "SPSCQueue element type must be movable or copyable");

public:
    SPSCQueue() = default;

    // Non-copyable, non-movable (atomics cannot be relocated safely)
    SPSCQueue(const SPSCQueue&)            = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    /**
     * @brief Push an item onto the queue (producer side).
     *
     * Called from the SINGLE producer thread only. No heap allocation.
     *
     * @param item  Item to enqueue (moved in).
     * @return true if enqueued successfully; false if the queue is full.
     *         The caller MUST NOT block — drop or log the overflow.
     */
    bool push(T&& item) noexcept {
        const size_t wr = wr_.load(std::memory_order_relaxed);
        const size_t next_wr = (wr + 1) & mask_;

        if (next_wr == rd_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }

        buf_[wr] = std::move(item);
        wr_.store(next_wr, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pop an item from the queue (consumer side).
     *
     * Called from the SINGLE consumer thread only. No heap allocation.
     *
     * @param out  Destination for the dequeued item.
     * @return true if an item was dequeued; false if the queue was empty.
     */
    bool pop(T& out) noexcept {
        const size_t rd = rd_.load(std::memory_order_relaxed);

        if (rd == wr_.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }

        out = std::move(buf_[rd]);
        rd_.store((rd + 1) & mask_, std::memory_order_release);
        return true;
    }

    /**
     * @brief Approximate check — only reliable from within a single thread.
     */
    bool empty() const noexcept {
        return rd_.load(std::memory_order_acquire) ==
               wr_.load(std::memory_order_acquire);
    }

    /**
     * @brief Approximate check — only reliable from within a single thread.
     */
    bool full() const noexcept {
        size_t wr   = wr_.load(std::memory_order_acquire);
        size_t next = (wr + 1) & mask_;
        return next == rd_.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t mask_ = Capacity - 1;

    alignas(64) std::array<T, Capacity> buf_{};
    alignas(64) std::atomic<size_t>     rd_{0};
    alignas(64) std::atomic<size_t>     wr_{0};
};

} // namespace biona
