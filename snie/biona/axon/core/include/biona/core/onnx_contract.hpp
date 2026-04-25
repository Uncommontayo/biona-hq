#pragma once

/**
 * @file onnx_contract.hpp
 * @brief Single source of truth for the streaming Emformer ONNX model interface.
 *
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  THIS FILE IS THE AXON↔LAB CONTRACT.  DO NOT CHANGE WITHOUT VERSIONING. ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 *
 * Biona Axon (C++ runtime) consumes the ONNX model whose tensor names and
 * shapes are defined here.  Biona Lab (PyTorch training pipeline) MUST produce
 * an ONNX export that exactly matches every name and dimension defined below.
 *
 * Rules for changes:
 *   1. Any modification to a tensor name or shape constant requires:
 *      a. A version bump in this file (update ONNX_CONTRACT_VERSION).
 *      b. A corresponding update to Biona Lab's torch.onnx.export call.
 *      c. A full model retrain + re-export + re-encrypt.
 *   2. The source of truth is THIS FILE (Axon).
 *      If a mismatch is discovered, Biona Lab is updated to match Axon —
 *      never the other way around.
 *   3. Run the VALIDATION prompt after any change to confirm consistency.
 *
 * How Biona Lab must export:
 *   torch.onnx.export(
 *       model,
 *       (chunk, memory, lc_key, lc_val),
 *       "model.onnx",
 *       input_names  = [ONNX_INPUT_CHUNK, ONNX_INPUT_MEMORY,
 *                       ONNX_INPUT_LC_KEY, ONNX_INPUT_LC_VAL],
 *       output_names = [ONNX_OUTPUT_LOGITS, ONNX_OUTPUT_MEMORY,
 *                       ONNX_OUTPUT_LC_KEY, ONNX_OUTPUT_LC_VAL,
 *                       ONNX_OUTPUT_EMBEDDING],
 *       opset_version = ONNX_OPSET_VERSION,
 *       ...
 *   )
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace biona {

// ---------------------------------------------------------------------------
// Contract version
// ---------------------------------------------------------------------------

/// Increment when any constant below changes.
inline constexpr std::string_view ONNX_CONTRACT_VERSION = "1.0";

// ---------------------------------------------------------------------------
// Input tensor names
// ---------------------------------------------------------------------------

/// Current audio features chunk: shape [1, ONNX_CHUNK_FRAMES, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_INPUT_CHUNK   = "chunk";

/// Memory bank from previous chunk: shape [1, ONNX_MEMORY_VECTORS, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_INPUT_MEMORY  = "memory";

/// Left-context attention key cache: shape [1, ONNX_LEFT_CONTEXT_FRAMES, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_INPUT_LC_KEY  = "left_context_key";

/// Left-context attention value cache: shape [1, ONNX_LEFT_CONTEXT_FRAMES, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_INPUT_LC_VAL  = "left_context_val";

// ---------------------------------------------------------------------------
// Output tensor names
// ---------------------------------------------------------------------------

/// CTC logits: shape [ONNX_CHUNK_FRAMES, 1, vocab_size]
inline constexpr std::string_view ONNX_OUTPUT_LOGITS   = "logits";

/// Updated memory bank: shape [1, ONNX_MEMORY_VECTORS, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_OUTPUT_MEMORY   = "memory_out";

/// Updated left-context key: shape [1, ONNX_LEFT_CONTEXT_FRAMES, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_OUTPUT_LC_KEY   = "lc_key_out";

/// Updated left-context value: shape [1, ONNX_LEFT_CONTEXT_FRAMES, ONNX_MEL_BANDS]
inline constexpr std::string_view ONNX_OUTPUT_LC_VAL   = "lc_val_out";

/// 512-dim speaker/speech embedding: shape [1, ONNX_EMBEDDING_DIM]
inline constexpr std::string_view ONNX_OUTPUT_EMBEDDING = "embedding";

// ---------------------------------------------------------------------------
// Shape constants
// ---------------------------------------------------------------------------

/// Number of mel filter banks — must match Frontend spec exactly.
inline constexpr int32_t ONNX_MEL_BANDS           = 80;

/// Dimension of the output embedding vector.
inline constexpr int32_t ONNX_EMBEDDING_DIM        = 512;

/// Number of memory summary vectors per chunk in the Emformer memory bank.
inline constexpr int32_t ONNX_MEMORY_VECTORS       = 4;

/// Number of left-context frames (640ms at 10ms hop).
inline constexpr int32_t ONNX_LEFT_CONTEXT_FRAMES  = 64;

/// Number of input frames per chunk (20ms at 10ms hop = 2 frames).
inline constexpr int32_t ONNX_CHUNK_FRAMES         = 2;

/// ONNX opset version used during export.
inline constexpr int32_t ONNX_OPSET_VERSION        = 17;

// ---------------------------------------------------------------------------
// Audio processing constants
// ---------------------------------------------------------------------------

/// Expected audio sample rate in Hz. MUST be 16000.
inline constexpr int32_t AUDIO_SAMPLE_RATE_HZ = 16000;

/// Audio chunk duration in milliseconds.
inline constexpr int32_t AUDIO_CHUNK_MS       = 20;

/// Number of mel filter banks (mirrors ONNX_MEL_BANDS for use in Frontend).
inline constexpr int32_t AUDIO_MEL_BANDS      = 80;

/// Analysis window duration in milliseconds.
inline constexpr int32_t AUDIO_WINDOW_MS      = 25;

/// Hop size (frame shift) in milliseconds.
inline constexpr int32_t AUDIO_HOP_MS         = 10;

// ---------------------------------------------------------------------------
// Compile-time consistency checks
// ---------------------------------------------------------------------------

static_assert(ONNX_MEL_BANDS  == AUDIO_MEL_BANDS,
    "ONNX_MEL_BANDS and AUDIO_MEL_BANDS must be equal");
static_assert(AUDIO_CHUNK_MS  % AUDIO_HOP_MS == 0,
    "AUDIO_CHUNK_MS must be an integer multiple of AUDIO_HOP_MS");
static_assert(AUDIO_CHUNK_MS / AUDIO_HOP_MS == ONNX_CHUNK_FRAMES,
    "ONNX_CHUNK_FRAMES must equal AUDIO_CHUNK_MS / AUDIO_HOP_MS");

// ---------------------------------------------------------------------------
// Runtime contract validator
// ---------------------------------------------------------------------------

/**
 * @brief Validates a loaded ONNX model's tensor names against this contract.
 *
 * Call immediately after loading the ONNX session to detect mismatches
 * between the compiled contract and the model file.
 *
 * Logs a clear error message for each mismatched or missing tensor name.
 *
 * @param actual_input_names   Input tensor names reported by the ONNX session.
 * @param actual_output_names  Output tensor names reported by the ONNX session.
 * @return true if all expected names are present; false on any mismatch.
 */
struct ONNXContractValidator {
    static bool validate(const std::vector<std::string>& actual_input_names,
                         const std::vector<std::string>& actual_output_names);
};

} // namespace biona
