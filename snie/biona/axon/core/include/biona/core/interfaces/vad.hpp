#pragma once

#include "biona/core/types.hpp"

#include <cstdint>
#include <memory>

namespace biona {

/**
 * @brief Abstract Voice Activity Detection interface.
 *
 * Threading guarantees:
 *   - All methods are single-threaded. VAD is called exclusively from the
 *     main pipeline thread. Do NOT call any method concurrently.
 *   - isSpeech() and confidence() are marked noexcept; they must not throw.
 *
 * Hot-path contract:
 *   - isSpeech() is called on every 20ms chunk regardless of content.
 *   - isSpeech() MUST complete in < 2ms.
 *   - No heap allocation permitted inside isSpeech() or confidence().
 */
class VAD {
public:
    virtual ~VAD() = default;

    /**
     * @brief Classify one audio chunk as speech or non-speech.
     *
     * Hot path — noexcept, no dynamic allocation.
     *
     * @param chunk  20ms PCM audio chunk at 16kHz.
     * @return true if the chunk (or its hangover window) contains speech.
     */
    virtual bool isSpeech(const AudioChunk& chunk) noexcept = 0;

    /**
     * @brief Return a speech confidence score for the given chunk.
     *
     * Hot path — noexcept, no dynamic allocation.
     *
     * @param chunk  20ms PCM audio chunk at 16kHz.
     * @return Confidence in [0.0, 1.0]. May be interpolated with hangover decay.
     */
    virtual float confidence(const AudioChunk& chunk) noexcept = 0;

    /**
     * @brief Reset all internal state (hangover counter, WebRTC state machine).
     *
     * Call between utterances or after a long silence. noexcept.
     */
    virtual void reset() noexcept = 0;

protected:
    VAD() = default;
};

// ---------------------------------------------------------------------------
// VAD configuration
// ---------------------------------------------------------------------------

/**
 * @brief Supported VAD algorithm implementations.
 */
enum class VADType {
    WEBRTC, ///< WebRTC VAD (libfvad) — default, best accuracy
    SILERO, ///< Silero VAD — not yet implemented
    ENERGY, ///< Simple RMS energy threshold — fallback for clean environments
};

/**
 * @brief Configuration for VAD construction.
 */
struct VADConfig {
    VADType type            = VADType::WEBRTC; ///< VAD algorithm to use
    int32_t sensitivity     = 2;  ///< 0=low, 1=medium, 2=aggressive (WebRTC aggressiveness)
    int32_t hangover_ms     = 200; ///< Continue reporting speech this long after last positive
    int32_t reset_silence_ms= 500; ///< Reset internal state after this much continuous silence
};

/**
 * @brief Factory function — construct a VAD from configuration.
 *
 * @param cfg  VAD type and tuning parameters.
 * @return Owning pointer to the constructed VAD.
 * @throws BionaException(VADInitFailed) on construction failure.
 * @throws BionaException(UnsupportedRuntime) for VADType::SILERO.
 */
std::unique_ptr<VAD> createVAD(const VADConfig& cfg);

} // namespace biona
