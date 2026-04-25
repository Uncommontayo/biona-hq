#pragma once

/**
 * @file secure_buffer.hpp
 * @brief RAII wrapper for sensitive in-memory data (model plaintext, key material).
 *
 * SecureBuffer:
 *   - Calls mlock() immediately on allocation to prevent the OS paging the
 *     buffer to swap (which would leave plaintext on disk).
 *   - Calls explicit_bzero / SecureZeroMemory on destruction so the wipe
 *     cannot be optimised away by the compiler.
 *   - Calls munlock() on destruction.
 *   - Non-copyable. Movable.
 *
 * RULE: Any buffer that holds decrypted model bytes MUST be a SecureBuffer.
 *       Never copy the contents into a plain std::vector<uint8_t>.
 */

#include "biona/core/errors.hpp"
#include "biona/security/safe_log.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

namespace biona {

/**
 * @brief Locked, zeroing RAII buffer for sensitive binary data.
 */
class SecureBuffer {
public:
    SecureBuffer() = default;

    /**
     * @brief Allocate a SecureBuffer of @p n bytes.
     *
     * Calls mlock() immediately. Throws BionaException(OOM) on failure.
     *
     * @param n  Number of bytes to allocate.
     */
    explicit SecureBuffer(size_t n) {
        data_.resize(n);
        if (n > 0) {
            lockMemory();
        }
    }

    ~SecureBuffer() {
        wipeAndUnlock();
    }

    // Non-copyable
    SecureBuffer(const SecureBuffer&)            = delete;
    SecureBuffer& operator=(const SecureBuffer&) = delete;

    // Movable
    SecureBuffer(SecureBuffer&& o) noexcept
        : data_(std::move(o.data_))
        , locked_(o.locked_)
    {
        o.locked_ = false;
    }

    SecureBuffer& operator=(SecureBuffer&& o) noexcept {
        if (this != &o) {
            wipeAndUnlock();
            data_     = std::move(o.data_);
            locked_   = o.locked_;
            o.locked_ = false;
        }
        return *this;
    }

    [[nodiscard]] uint8_t*       data()       noexcept { return data_.data(); }
    [[nodiscard]] const uint8_t* data() const noexcept { return data_.data(); }
    [[nodiscard]] size_t         size() const noexcept { return data_.size(); }
    [[nodiscard]] bool           empty()const noexcept { return data_.empty(); }

private:
    std::vector<uint8_t> data_;
    bool                 locked_ = false;

    void lockMemory() {
#if defined(_WIN32)
        BOOL ok = VirtualLock(data_.data(), data_.size());
        if (!ok) {
#else
        int rc = mlock(data_.data(), data_.size());
        if (rc != 0) {
#endif
            SafeLog::event(BIONA_EVENT_MODEL_LOAD_FAILED);
            throw BionaException(BionaError::OOM,
                "SecureBuffer: mlock() failed — cannot lock model plaintext in RAM");
        }
        locked_ = true;
    }

    void wipeAndUnlock() noexcept {
        if (!data_.empty()) {
#if defined(_WIN32)
            SecureZeroMemory(data_.data(), data_.size());
            if (locked_) VirtualUnlock(data_.data(), data_.size());
#elif defined(__STDC_LIB_EXT1__)
            memset_s(data_.data(), data_.size(), 0, data_.size());
            if (locked_) munlock(data_.data(), data_.size());
#else
            volatile uint8_t* p = data_.data();
            for (size_t i = 0; i < data_.size(); ++i) p[i] = 0;
            if (locked_) munlock(data_.data(), data_.size());
#endif
            locked_ = false;
        }
    }
};

} // namespace biona
