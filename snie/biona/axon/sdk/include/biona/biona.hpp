#pragma once

/**
 * @file biona.hpp
 * @brief C++ convenience wrapper around the Biona C ABI.
 *
 * Provides RAII ownership of the BIONA_Engine handle and type-safe
 * callback registration.
 */

#include "biona.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace biona {

/**
 * @brief C++ exception wrapping a BIONA_Error.
 */
class BionaApiError : public std::runtime_error {
public:
    explicit BionaApiError(BIONA_Error err)
        : std::runtime_error(BIONA_ErrorString(err))
        , code_(err)
    {}
    BIONA_Error code() const noexcept { return code_; }
private:
    BIONA_Error code_;
};

/**
 * @brief RAII wrapper for a BIONA_Engine handle.
 *
 * Non-copyable. Movable.
 */
class Engine {
public:
    explicit Engine(const BIONA_Config& cfg) {
        BIONA_Engine h = nullptr;
        BIONA_Error err = BIONA_Create(&cfg, &h);
        if (err != BIONA_OK) throw BionaApiError(err);
        handle_ = h;
    }

    ~Engine() {
        if (handle_) BIONA_Destroy(handle_);
    }

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }
    Engine& operator=(Engine&& o) noexcept {
        if (this != &o) {
            if (handle_) BIONA_Destroy(handle_);
            handle_ = o.handle_;
            o.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Process one audio chunk.
     *
     * @return BIONA_Result populated with text, embedding, and latency.
     *         Pointers are valid until the next call.
     */
    BIONA_Result processChunk(const int16_t* pcm, size_t n_samples) {
        BIONA_Result result{};
        BIONA_Error err = BIONA_ProcessChunk(handle_, pcm, n_samples, &result);
        if (err != BIONA_OK) throw BionaApiError(err);
        return result;
    }

    /**
     * @brief Register an async signal callback.
     */
    void registerSignalCallback(BIONA_SignalCallback cb, void* user_data = nullptr) {
        BIONA_Error err = BIONA_RegisterSignalCallback(handle_, cb, user_data);
        if (err != BIONA_OK) throw BionaApiError(err);
    }

    BIONA_Engine handle() const noexcept { return handle_; }

private:
    BIONA_Engine handle_ = nullptr;
};

} // namespace biona
