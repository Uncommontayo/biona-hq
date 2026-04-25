#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace biona {

/**
 * @brief Enumeration of all Biona engine error codes.
 *
 * Used both in BionaException and in the C ABI (BIONA_Error).
 */
enum class BionaError : int32_t {
    OK                  = 0, ///< No error
    InitFailed          = 1, ///< Engine or subsystem initialisation failed
    ModelDecryptFailed  = 2, ///< HMAC or AES-GCM decryption verification failed
    InvalidAudio        = 3, ///< Audio input is malformed or wrong format
    InferenceFailed     = 4, ///< ONNX / TFLite / CoreML inference call failed
    VADInitFailed       = 5, ///< VAD subsystem failed to initialise
    OOM                 = 6, ///< Memory allocation or mlock() failed
    UnsupportedRuntime  = 7, ///< Requested engine type is not supported on this platform
};

/**
 * @brief Exception thrown by Biona subsystems on unrecoverable errors.
 *
 * Carries a BionaError code alongside the human-readable message from
 * std::runtime_error.  Callers at the C ABI boundary must catch this and
 * translate to the corresponding BIONA_Error value.
 */
class BionaException : public std::runtime_error {
public:
    /**
     * @param code    The structured error code.
     * @param message Human-readable detail string. Must NOT contain user data,
     *                audio content, secrets, or PII.
     */
    explicit BionaException(BionaError code, std::string_view message)
        : std::runtime_error(std::string(message))
        , code_(code)
    {}

    /// Returns the structured error code.
    [[nodiscard]] BionaError code() const noexcept { return code_; }

private:
    BionaError code_;
};

/**
 * @brief Throws a BionaException if @p condition is true.
 *
 * Intended for precondition checks throughout the engine. Example:
 * @code
 *   throwIf(cfg.sample_rate_hz != 16000,
 *           BionaError::InvalidAudio,
 *           "sample_rate_hz must be 16000");
 * @endcode
 *
 * @param condition If true, the exception is thrown.
 * @param code      Error code to attach.
 * @param msg       Detail message. Must NOT contain user data or secrets.
 */
inline void throwIf(bool condition, BionaError code, std::string_view msg) {
    if (condition) {
        throw BionaException(code, msg);
    }
}

} // namespace biona
