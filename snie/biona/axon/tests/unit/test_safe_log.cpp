#include "biona/security/safe_log.hpp"

#include <cassert>
#include <sstream>
#include <string>

// ── helpers ─────────────────────────────────────────────────────────────────

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// ── tests ────────────────────────────────────────────────────────────────────

static void test_metric_emitted() {
    std::ostringstream oss;
    biona::SafeLog::configure(biona::LogLevel::PERF_ONLY, &oss);
    biona::SafeLog::metric(biona::BIONA_METRIC_INFERENCE_LATENCY_MS, 12.5);
    const std::string out = oss.str();
    assert(contains(out, "[METRIC]") && "metric prefix missing");
    assert(contains(out, "inference.latency_ms") && "metric name missing");
    assert(contains(out, "12.5") || contains(out, "12.500") && "metric value missing");
}

static void test_event_emitted() {
    std::ostringstream oss;
    biona::SafeLog::configure(biona::LogLevel::PERF_ONLY, &oss);
    biona::SafeLog::event(biona::BIONA_EVENT_ENGINE_INITIALIZED);
    const std::string out = oss.str();
    assert(contains(out, "[EVENT]") && "event prefix missing");
    assert(contains(out, "engine.initialized") && "event name missing");
}

static void test_error_emitted_at_system_level() {
    std::ostringstream oss;
    biona::SafeLog::configure(biona::LogLevel::SYSTEM_STATE, &oss);
    biona::SafeLog::error("OOM", "mlock failed for model buffer");
    const std::string out = oss.str();
    assert(contains(out, "[ERROR]") && "error prefix missing");
    assert(contains(out, "OOM") && "error code missing");
    assert(contains(out, "mlock failed") && "error detail missing");
}

static void test_error_suppressed_at_perf_only() {
    std::ostringstream oss;
    biona::SafeLog::configure(biona::LogLevel::PERF_ONLY, &oss);
    biona::SafeLog::error("TestError", "should not appear");
    const std::string out = oss.str();
    assert(!contains(out, "[ERROR]") && "error should be suppressed at PERF_ONLY");
}

#ifndef NDEBUG
static void test_debug_emitted_in_debug_build() {
    std::ostringstream oss;
    biona::SafeLog::configure(biona::LogLevel::DEBUG_ONLY, &oss);
    biona::SafeLog::debug("TestTag", "debug message ", 42);
    const std::string out = oss.str();
    assert(contains(out, "[DEBUG]") && "debug prefix missing");
    assert(contains(out, "TestTag") && "debug tag missing");
    assert(contains(out, "debug message") && "debug body missing");
    assert(contains(out, "42") && "debug value missing");
}
#endif

int main() {
    test_metric_emitted();
    test_event_emitted();
    test_error_emitted_at_system_level();
    test_error_suppressed_at_perf_only();
#ifndef NDEBUG
    test_debug_emitted_in_debug_build();
#endif
    return 0;
}
