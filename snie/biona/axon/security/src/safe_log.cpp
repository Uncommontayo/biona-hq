#include "biona/security/safe_log.hpp"

#include <iostream>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>

namespace biona {

namespace {

struct LoggerState {
    std::mutex   mu;
    LogLevel     level = LogLevel::SYSTEM_STATE;
    std::ostream* out  = &std::cerr;
};

LoggerState& state() {
    static LoggerState s;
    return s;
}

void write(const char* prefix, std::string_view body) noexcept {
    try {
        auto& s = state();
        std::lock_guard<std::mutex> lk(s.mu);
        *s.out << prefix << body << '\n';
        s.out->flush();
    } catch (...) {
        // Never propagate — logging must not crash the engine
    }
}

} // anonymous namespace

void SafeLog::configure(LogLevel level, std::ostream* out) {
    auto& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.level = level;
    s.out   = (out != nullptr) ? out : &std::cerr;
}

void SafeLog::metric(std::string_view name, double value) noexcept {
    if (static_cast<int>(state().level) < static_cast<int>(LogLevel::PERF_ONLY)) return;
    try {
        std::string msg;
        msg.reserve(64);
        msg += "[METRIC] ";
        msg += name;
        msg += '=';
        msg += std::to_string(value);
        write("", msg);
    } catch (...) {}
}

void SafeLog::event(std::string_view name) noexcept {
    if (static_cast<int>(state().level) < static_cast<int>(LogLevel::PERF_ONLY)) return;
    try {
        std::string msg;
        msg.reserve(32);
        msg += "[EVENT] ";
        msg += name;
        write("", msg);
    } catch (...) {}
}

void SafeLog::error(std::string_view code, std::string_view detail) noexcept {
    if (static_cast<int>(state().level) < static_cast<int>(LogLevel::SYSTEM_STATE)) return;
    try {
        std::string msg;
        msg.reserve(64);
        msg += "[ERROR] code=";
        msg += code;
        msg += " detail=";
        msg += detail;
        write("", msg);
    } catch (...) {}
}

// ── debug helpers (debug builds only) ──────────────────────────────────────
#ifndef NDEBUG
namespace detail {

void debugWrite(std::string_view tag, const std::string& body) noexcept {
    if (static_cast<int>(state().level) < static_cast<int>(LogLevel::DEBUG_ONLY)) return;
    try {
        std::string msg;
        msg.reserve(tag.size() + body.size() + 12);
        msg += "[DEBUG][";
        msg += tag;
        msg += "] ";
        msg += body;
        write("", msg);
    } catch (...) {}
}

} // namespace detail
#endif // !NDEBUG

} // namespace biona
