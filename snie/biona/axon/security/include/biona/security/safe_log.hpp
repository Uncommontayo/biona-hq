#pragma once

/**
 * @file safe_log.hpp
 * @brief Privacy-preserving structured logger for Biona Axon.
 *
 * SECURITY CONTRACT — enforced at design time:
 *   NEVER LOG:  audio samples, transcript text, embedding vectors,
 *               user identifiers, session IDs, or any user-derived data.
 *   ALWAYS LOG: performance metrics (latency, memory), system state
 *               (engine init, model load), error codes and detail strings.
 *
 * The API is intentionally narrow:
 *   - metric()  accepts a name + numeric value only.
 *   - event()   accepts a pre-approved event name only.
 *   - error()   accepts an error code string + a detail string (no user data).
 *   - debug()   is completely stripped in NDEBUG builds at compile time.
 *
 * All methods are static and thread-safe.
 */

#include <iostream>
#include <string_view>

namespace biona {

// ---------------------------------------------------------------------------
// Log level
// ---------------------------------------------------------------------------

/**
 * @brief Controls which log calls are emitted.
 *
 * PERF_ONLY    — emit only metric() and event() calls.
 * SYSTEM_STATE — emit metric(), event(), and error() calls.
 * DEBUG_ONLY   — emit all calls (only available in debug builds).
 */
enum class LogLevel {
    PERF_ONLY    = 0,
    SYSTEM_STATE = 1,
    DEBUG_ONLY   = 2,
};

// ---------------------------------------------------------------------------
// Pre-approved metric name constants
// ---------------------------------------------------------------------------

/// Inference latency for one 20ms chunk (ms).
inline constexpr std::string_view BIONA_METRIC_INFERENCE_LATENCY_MS = "inference.latency_ms";

/// Number of frames classified as speech by VAD since last reset.
inline constexpr std::string_view BIONA_METRIC_VAD_SPEECH_FRAMES    = "vad.speech_frames";

/// Number of frames classified as silence by VAD since last reset.
inline constexpr std::string_view BIONA_METRIC_VAD_SILENCE_FRAMES   = "vad.silence_frames";

/// Incremented each time a SignalTask is dropped because the queue is full.
inline constexpr std::string_view BIONA_METRIC_QUEUE_OVERFLOW        = "signal.queue_overflow";

/// Incremented each time an async signal exceeds the timeout budget.
inline constexpr std::string_view BIONA_METRIC_SIGNAL_TIMEOUT        = "signal.timeout";

/// Current resident set size of the process (KB).
inline constexpr std::string_view BIONA_METRIC_MEMORY_RSS_KB         = "memory.rss_kb";

// ---------------------------------------------------------------------------
// Pre-approved event name constants
// ---------------------------------------------------------------------------

inline constexpr std::string_view BIONA_EVENT_ENGINE_INITIALIZED  = "engine.initialized";
inline constexpr std::string_view BIONA_EVENT_MODEL_LOAD_SUCCESS   = "model.load_success";
inline constexpr std::string_view BIONA_EVENT_MODEL_LOAD_FAILED    = "model.load_failed";
inline constexpr std::string_view BIONA_EVENT_ENGINE_RESET         = "engine.reset";

// ---------------------------------------------------------------------------
// SafeLog
// ---------------------------------------------------------------------------

/**
 * @brief Structured, privacy-preserving logger (static API, singleton backend).
 *
 * Thread-safety: all static methods are safe to call from multiple threads.
 * An internal mutex serialises writes to the configured output stream.
 *
 * configure() must be called once at startup before any log call.
 * If configure() is never called, SYSTEM_STATE level is used and output
 * goes to std::cerr.
 */
class SafeLog {
public:
    SafeLog() = delete;

    /**
     * @brief Configure the logger backend.
     *
     * @param level  Minimum log level to emit.
     * @param out    Output stream. Must remain valid for the lifetime of the
     *               process. Defaults to std::cerr.
     */
    static void configure(LogLevel level, std::ostream* out = &std::cerr);

    /**
     * @brief Emit a structured numeric metric.
     *
     * Format: [METRIC] <name>=<value>
     *
     * @param name   Pre-approved metric name (use BIONA_METRIC_* constants).
     * @param value  Numeric measurement. Never pass user-derived data here.
     */
    static void metric(std::string_view name, double value) noexcept;

    /**
     * @brief Emit a discrete system event.
     *
     * Format: [EVENT] <name>
     *
     * @param name  Pre-approved event name (use BIONA_EVENT_* constants).
     */
    static void event(std::string_view name) noexcept;

    /**
     * @brief Emit a structured error.
     *
     * Format: [ERROR] code=<code> detail=<detail>
     *
     * @param code    Short error code string (e.g. "OOM", "InitFailed").
     * @param detail  Human-readable detail. MUST NOT contain user data,
     *                audio content, secrets, or PII.
     */
    static void error(std::string_view code, std::string_view detail) noexcept;

    // ── Debug logging — stripped entirely in NDEBUG builds ─────────────────
#ifndef NDEBUG
    /**
     * @brief Debug log — variadic, tag-scoped.
     *
     * @warning Only available in debug builds. Calls are no-ops in release.
     * @param tag   Short subsystem tag (e.g. "OnnxEngine", "VAD").
     * @param args  Values to print. MUST NOT include user data or secrets.
     */
    template<typename... Args>
    static void debug(std::string_view tag, Args&&... args);
#else
    template<typename... Args>
    static void debug(std::string_view, Args&&...) noexcept {}
#endif
};

} // namespace biona

// ---------------------------------------------------------------------------
// debug() template implementation (debug builds only)
// ---------------------------------------------------------------------------
#ifndef NDEBUG
#include <mutex>
#include <sstream>

namespace biona {
namespace detail {
    void debugWrite(std::string_view tag, const std::string& msg) noexcept;
} // namespace detail

template<typename... Args>
void SafeLog::debug(std::string_view tag, Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    detail::debugWrite(tag, oss.str());
}

} // namespace biona
#endif // !NDEBUG
