#pragma once

#include "biona/core/types.hpp"

namespace biona {

/**
 * @brief Abstract interface for audio feature extraction (log-mel spectrogram).
 *
 * Specification (must match onnx_contract.hpp constants exactly):
 *   - 80 mel filter banks
 *   - 25ms analysis window
 *   - 10ms hop size
 *   - 16kHz input sample rate
 *   - 20ms input chunk → 2 output frames
 *
 * Threading guarantees:
 *   - extract() is called exclusively from the main pipeline thread.
 *   - Do NOT share a Frontend instance across threads.
 *
 * Hot-path contract:
 *   - extract() is called on every chunk that passes VAD.
 *   - extract() MUST NOT perform dynamic allocation — output is written into
 *     the caller-provided @p out buffer (pre-allocated).
 *   - extract() is noexcept; failures should set n_frames = 0.
 */
class Frontend {
public:
    virtual ~Frontend() = default;

    /**
     * @brief Extract log-mel spectrogram from a raw audio chunk.
     *
     * Writes results into @p out. The caller must pre-allocate @p out.
     *
     * Hot path — noexcept, no dynamic allocation.
     *
     * @param chunk  20ms PCM audio at 16kHz (320 samples).
     * @param out    Output buffer. On success: out.mel_bands is populated
     *               and out.n_frames is set to the number of output frames
     *               (typically 2 for a 20ms chunk). On failure: out.n_frames = 0.
     */
    virtual void extract(const AudioChunk& chunk,
                         AudioFeatures&    out) noexcept = 0;

protected:
    Frontend() = default;
};

} // namespace biona
