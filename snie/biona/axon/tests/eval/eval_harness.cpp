/**
 * @file eval_harness.cpp
 * @brief Latency measurement tool for Biona Axon.
 *
 * Usage:
 *   eval_harness <wav_dir> <model_bundle> [--p99-limit-ms N]
 *
 * Reads all .wav files (16kHz mono PCM) from <wav_dir>, feeds each 20ms chunk
 * through BIONA_ProcessChunk, records per-chunk latency, then outputs a JSON
 * report with p50, p95, p99, max, and total chunk count.
 *
 * Exits with code 1 if p99_ms > 150ms (hard gate from Biona spec §12).
 */

#include "biona/biona.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal WAV loader (PCM 16-bit mono 16kHz only)
// ---------------------------------------------------------------------------

static std::vector<int16_t> loadWav(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << '\n';
        return {};
    }

    // Skip 44-byte canonical WAV header
    char header[44];
    f.read(header, 44);
    if (!f || f.gcount() < 44) return {};

    // Read remaining bytes as int16
    std::vector<int16_t> samples;
    int16_t s;
    while (f.read(reinterpret_cast<char*>(&s), 2)) {
        samples.push_back(s);
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------

static double percentile(std::vector<int32_t>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(sorted.size()));
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return static_cast<double>(sorted[idx]);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: eval_harness <wav_dir> <model_bundle> [--p99-limit-ms N]\n";
        return 1;
    }

    const std::string wav_dir     = argv[1];
    const std::string model_path  = argv[2];
    int32_t p99_limit_ms          = 150; // default Biona spec §12 gate

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--p99-limit-ms" && i + 1 < argc) {
            p99_limit_ms = std::atoi(argv[++i]);
        }
    }

    // Create engine
    BIONA_Config cfg{};
    cfg.model_bundle_path   = model_path.c_str();
    cfg.sample_rate_hz      = 16000;
    cfg.chunk_size_ms       = 20;
    cfg.enable_async        = 0;
    cfg.log_level           = 0; // PERF_ONLY
    cfg.secret_manager_type = "env";

    BIONA_Engine engine = nullptr;
    BIONA_Error err = BIONA_Create(&cfg, &engine);
    if (err != BIONA_OK) {
        std::cerr << "BIONA_Create failed: " << BIONA_ErrorString(err) << '\n';
        return 1;
    }

    static constexpr size_t CHUNK_SAMPLES = 16000 * 20 / 1000; // 320

    std::vector<int32_t> latencies;
    latencies.reserve(10000);

    for (const auto& entry : fs::directory_iterator(wav_dir)) {
        if (entry.path().extension() != ".wav") continue;

        auto samples = loadWav(entry.path());
        if (samples.empty()) continue;

        for (size_t offset = 0; offset + CHUNK_SAMPLES <= samples.size();
             offset += CHUNK_SAMPLES) {

            auto t0 = std::chrono::steady_clock::now();

            BIONA_Result result{};
            BIONA_Error chunk_err = BIONA_ProcessChunk(
                engine,
                samples.data() + offset,
                CHUNK_SAMPLES,
                &result);

            auto t1 = std::chrono::steady_clock::now();
            int32_t wall_ms = static_cast<int32_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            if (chunk_err == BIONA_OK) {
                latencies.push_back(wall_ms);
            }
        }
    }

    BIONA_Destroy(engine);

    if (latencies.empty()) {
        std::cerr << "No chunks processed — no .wav files found in: " << wav_dir << '\n';
        return 1;
    }

    std::sort(latencies.begin(), latencies.end());

    double p50  = percentile(latencies, 50.0);
    double p95  = percentile(latencies, 95.0);
    double p99  = percentile(latencies, 99.0);
    double pmax = static_cast<double>(latencies.back());
    size_t n    = latencies.size();

    // Output JSON report
    std::cout << "{\n"
              << "  \"p50_ms\": "    << p50      << ",\n"
              << "  \"p95_ms\": "    << p95      << ",\n"
              << "  \"p99_ms\": "    << p99      << ",\n"
              << "  \"max_ms\": "    << pmax     << ",\n"
              << "  \"n_chunks\": "  << n        << ",\n"
              << "  \"model_path\": \"" << model_path << "\"\n"
              << "}\n";

    if (p99 > static_cast<double>(p99_limit_ms)) {
        std::cerr << "FAIL: p99 latency " << p99
                  << "ms exceeds gate of " << p99_limit_ms << "ms\n";
        return 1;
    }

    std::cout << "PASS: p99 " << p99 << "ms <= " << p99_limit_ms << "ms\n";
    return 0;
}
