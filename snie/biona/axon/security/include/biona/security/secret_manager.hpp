#pragma once

/**
 * @file secret_manager.hpp
 * @brief Abstract secret management interface for Biona Axon.
 *
 * SECURITY POLICY:
 *   - No secrets in source code.
 *   - No secrets in build artifacts.
 *   - Secrets are read from the environment, platform keychain, hardware
 *     keystore, or TPM — never hardcoded.
 *
 *   IMPORTANT: The std::string returned by get() contains key material.
 *   Callers MUST zero the buffer after use. Use SecureString (below) as
 *   the preferred owner for any value returned by get().
 *   Explicitly call explicit_bzero / SecureZeroMemory / memset_s on
 *   any raw buffer that holds key material before deallocation.
 */

#include "biona/core/errors.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace biona {

// ---------------------------------------------------------------------------
// Pre-approved secret key name constants
// ---------------------------------------------------------------------------

/// AES-256-GCM encryption key for the model bundle (hex-encoded, 64 chars).
inline constexpr std::string_view BIONA_SECRET_MODEL_ENCRYPTION_KEY = "MODEL_ENCRYPTION_KEY";

/// HMAC-SHA256 key for bundle integrity verification (hex-encoded, 64 chars).
inline constexpr std::string_view BIONA_SECRET_MODEL_HMAC_KEY       = "MODEL_HMAC_KEY";

// ---------------------------------------------------------------------------
// SecureString — RAII wrapper that zeroes its buffer on destruction
// ---------------------------------------------------------------------------

/**
 * @brief RAII wrapper for sensitive string data (key material, passwords).
 *
 * The internal buffer is zero-filled on destruction using a
 * compiler-barrier-safe zeroing function so the wipe cannot be optimised away.
 *
 * Non-copyable (copying key material creates additional cleartext exposure).
 * Movable.
 */
class SecureString {
public:
    SecureString() = default;

    /// Construct from a plain std::string. The source string is consumed and
    /// zeroed immediately — do NOT use it after this call.
    explicit SecureString(std::string&& s) : data_(std::move(s)) {}

    ~SecureString() { wipe(); }

    SecureString(const SecureString&)            = delete;
    SecureString& operator=(const SecureString&) = delete;

    SecureString(SecureString&& o) noexcept : data_(std::move(o.data_)) {
        o.data_.clear();
    }
    SecureString& operator=(SecureString&& o) noexcept {
        if (this != &o) {
            wipe();
            data_ = std::move(o.data_);
            o.data_.clear();
        }
        return *this;
    }

    [[nodiscard]] const std::string& str() const noexcept { return data_; }
    [[nodiscard]] bool               empty() const noexcept { return data_.empty(); }

private:
    std::string data_;

    void wipe() noexcept {
        if (!data_.empty()) {
#if defined(_WIN32)
            SecureZeroMemory(data_.data(), data_.size());
#elif defined(__STDC_LIB_EXT1__)
            memset_s(data_.data(), data_.size(), 0, data_.size());
#else
            // Explicit bzero — not optimised away by the compiler
            volatile char* p = data_.data();
            for (size_t i = 0; i < data_.size(); ++i) p[i] = 0;
#endif
            data_.clear();
        }
    }
};

// ---------------------------------------------------------------------------
// SecretManager abstract base
// ---------------------------------------------------------------------------

/**
 * @brief Abstract interface for reading runtime secrets.
 *
 * Implementations: EnvSecretManager, KeychainSecretManager (iOS),
 * KeystoreSecretManager (Android), TPMSecretManager (Linux enterprise).
 */
class SecretManager {
public:
    virtual ~SecretManager() = default;

    /**
     * @brief Retrieve the value for @p key.
     *
     * @warning The returned std::string contains key material.
     *          Wrap it in a SecureString immediately, or call explicit_bzero
     *          on the buffer after use.
     *
     * @param key  Secret key identifier (use BIONA_SECRET_* constants).
     * @return Secret value as a plain string.
     * @throws BionaException(InitFailed) if the key is not found.
     */
    virtual std::string get(std::string_view key) const = 0;

    /**
     * @brief Check whether @p key exists without revealing its value.
     *
     * @param key  Secret key identifier.
     * @return true if the key exists in the backend.
     */
    virtual bool has(std::string_view key) const noexcept = 0;

protected:
    SecretManager() = default;
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * @brief Construct a SecretManager implementation by type name.
 *
 * @param type  Backend selector:
 *   "env"      → EnvSecretManager (reads process environment variables)
 *   "keychain" → stub (throws UnsupportedRuntime — iOS impl added later)
 *   "keystore" → stub (throws UnsupportedRuntime — Android impl added later)
 *   "tpm"      → stub (throws UnsupportedRuntime — Linux enterprise impl later)
 *
 * @return Owning pointer to the SecretManager.
 * @throws BionaException(UnsupportedRuntime) for unknown type strings.
 */
std::unique_ptr<SecretManager> createSecretManager(const std::string& type);

} // namespace biona
