#include "biona/core/spsc_queue.hpp"

#include <cassert>
#include <atomic>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

// ── single-threaded correctness ───────────────────────────────────────────

static void test_push_pop_basic() {
    biona::SPSCQueue<int32_t, 8> q;

    assert(q.empty() && "new queue must be empty");
    assert(!q.full() && "new queue must not be full");

    { int32_t v = 42; assert(q.push(std::move(v))); }
    assert(!q.empty());

    int32_t val = 0;
    assert(q.pop(val) && val == 42);
    assert(q.empty());
}

static void test_overflow_returns_false() {
    biona::SPSCQueue<int32_t, 4> q; // capacity 4 (ring uses N-1 = 3 slots)

    // Fill to capacity
    int32_t pushed = 0;
    while (q.push(int32_t(pushed))) ++pushed;

    // Next push must fail (queue full)
    { int32_t v = 999; assert(!q.push(std::move(v)) && "push on full queue must return false"); }
    assert(q.full());
}

static void test_pop_empty_returns_false() {
    biona::SPSCQueue<int32_t, 8> q;
    int32_t val = 0;
    assert(!q.pop(val) && "pop on empty queue must return false");
}

// ── producer / consumer thread test (10k items) ───────────────────────────

static void test_producer_consumer_10k() {
    static constexpr size_t N = 10000;
    biona::SPSCQueue<int32_t, 1024> q;

    std::atomic<int32_t> received_sum{0};
    std::atomic<bool>    done{false};

    // Consumer thread
    std::thread consumer([&]{
        int32_t count = 0;
        int32_t sum   = 0;
        while (count < static_cast<int32_t>(N)) {
            int32_t v = 0;
            if (q.pop(v)) { // NOLINT
                sum += v;
                ++count;
            }
        }
        received_sum.store(sum, std::memory_order_relaxed);
        done.store(true, std::memory_order_release);
    });

    // Producer: push 0..N-1
    for (int32_t i = 0; i < static_cast<int32_t>(N); ++i) {
        while (!q.push(int32_t(i))) { /* spin — unlikely with 1024 capacity */ }
    }

    consumer.join();

    // Verify sum 0+1+...+(N-1) = N*(N-1)/2
    int64_t expected = static_cast<int64_t>(N) * (N - 1) / 2;
    assert(received_sum.load() == static_cast<int32_t>(expected)
           && "All items must arrive in order with correct values");
    assert(done.load(std::memory_order_acquire));
}

int main() {
    test_push_pop_basic();
    test_overflow_returns_false();
    test_pop_empty_returns_false();
    test_producer_consumer_10k();
    return 0;
}
