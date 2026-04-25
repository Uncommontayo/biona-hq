# Biona Axon — C++ On-Device Engine
## Claude CLI Prompts · 12 Prompts

> **How to use**
> Paste each prompt into Claude CLI in VS Code **in order**, one at a time.
> Run these in one VS Code terminal while Biona Lab runs in a separate terminal.
> Complete the **SETUP** prompt first if you haven't already (run once, not per track).

---

## SETUP — Run This Once (Both Tracks)

```
Create a root project directory called `snie/` with two top-level subdirectories:
- `biona/axon/`   → Biona Axon: C++17 on-device inference engine
- `biona/lab/` → Biona Lab: Python training and labeling pipeline

Inside each, create a `.gitkeep` file so the directories are tracked by git.
Then create a root `README.md` with the following content:

# Biona — Biona Speech Intelligence Engine

## Repository Structure
- `axon/`   — Biona Axon: On-device C++17 inference engine
- `lab/` — Server-side Python training pipeline (Biona Lab)

## Biona Laboundary Rule
**Biona Axon runs on-device. Biona Lab runs on servers. These are separate systems.**
LLM inference, cloud APIs, and large-batch GPU processing belong in Biona Lab ONLY.
The ONNX model produced by Biona Lab is the only artifact that crosses the boundary.

Also create a root `.gitignore` with entries for: build/, __pycache__/, *.pyc,
.env, *.onnx, *.tflite, *.mlmodel, *.key, *.enc, node_modules/, .DS_Store
```


---

---

# BIONA AXON — C++ On-Device Engine
## Axon: Run these prompts in order in `biona/axon/`

---

### A-1 · CMake Project Scaffold

```
You are a senior C++ systems engineer. Create a production-grade CMake project
scaffold for an on-device Biona Speech Intelligence Engine called Biona.

Work inside the directory: biona/axon/

Create the following directory structure and all required CMakeLists.txt files:

biona/axon/
├── CMakeLists.txt              ← Root CMake (project entry point)
├── cmake/
│   ├── toolchains/
│   │   ├── linux-x86_64.cmake  ← Linux toolchain
│   │   ├── android-arm64.cmake ← Android NDK toolchain
│   │   └── ios-arm64.cmake     ← iOS Xcode toolchain
│   ├── FindONNXRuntime.cmake   ← ONNX Runtime finder module
│   └── CompilerFlags.cmake     ← Hardened compiler flags
├── core/
│   ├── CMakeLists.txt
│   └── include/biona/core/      ← (empty, headers added in A-2)
├── engines/
│   ├── CMakeLists.txt
│   ├── onnx/CMakeLists.txt
│   ├── tflite/CMakeLists.txt
│   └── coreml/CMakeLists.txt
├── security/
│   ├── CMakeLists.txt
│   └── include/biona/security/
├── adapters/
│   ├── CMakeLists.txt
│   └── include/biona/adapters/
├── sdk/
│   ├── CMakeLists.txt
│   └── include/biona/
└── tests/
    ├── CMakeLists.txt
    └── unit/

Requirements for CMakeLists.txt files:
- C++17 standard enforced globally
- cmake_minimum_required(VERSION 3.25)
- Each subdirectory is a separate static library target
- core/ must have ZERO dependencies on engines/, security/, adapters/, or sdk/
- Strict compiler flags in CompilerFlags.cmake:
  -Wall -Wextra -Wpedantic -Werror -fstack-protector-strong
  -D_FORTIFY_SOURCE=2 (release only) -fPIE
- ONNX Runtime linked only in engines/onnx/ — never in core/
- Add a top-level BUILD_TESTING option (default ON)
- Add platform detection: BIONA_PLATFORM_LINUX, BIONA_PLATFORM_ANDROID, BIONA_PLATFORM_IOS
- Android toolchain must reference NDK via ANDROID_NDK env variable
- iOS toolchain must set CMAKE_SYSTEM_NAME=iOS and use Xcode generator

Generate all files with full content. No placeholders.
```

---

### A-2 · Core Type Definitions

```
You are a senior C++17 systems engineer. Add the core type definitions to
biona/axon/core/include/biona/core/

Create the following header files with full production-ready content:

1. types.hpp — Fundamental types used across the entire engine:
   - AudioChunk struct: { const int16_t* data; size_t n_samples; int64_t timestamp_ms; }
   - AudioFeatures struct: { std::array<float, 80> mel_bands; size_t n_frames; }
     (80 mel bands, matches Frontend spec exactly)
   - ModelConfig struct: { std::string model_path; std::string key_id; int sample_rate_hz = 16000; int chunk_size_ms = 20; }
   - EngineInfo struct: { std::string runtime_name; std::string runtime_version; bool supports_streaming; }
   - InferenceResult struct: { std::vector<float> embedding; std::string text; int32_t latency_ms; bool vad_triggered; }
   - SignalTask struct: { InferenceResult result; int64_t chunk_id; }
   - All structs must be trivially copyable where possible
   - Use fixed-size types (int32_t, int64_t, float) — no int, long, double

2. signal.hpp — The stable signal envelope schema (the public contract):
   - Signal struct with fields:
     std::string schema_version = "1.0"
     std::string text
     int64_t timestamp_ms
     int32_t latency_ms
     std::vector<float> embedding   ← always populated, 512-dim
     std::optional<EmotionSignal> emotion    ← null = not computed
     std::optional<SpeakerSignal> speaker    ← null = not computed
     std::optional<IntentSignal> intent      ← null = not computed
   - EmotionSignal struct: { float valence; float arousal; float stability; }
   - SpeakerSignal struct: { std::string speaker_id; float confidence; }
   - IntentSignal struct: { std::string intent_label; float confidence; }
   - RULE: Adding a new optional signal field is the ONLY allowed schema change.
     Never add required fields. Never remove fields.
   - Include a to_json() method returning std::string (use nlohmann/json or manual)

3. errors.hpp — Error types:
   - enum class BionaError: OK, InitFailed, ModelDecryptFailed, InvalidAudio,
     InferenceFailed, VADInitFailed, OOM, UnsupportedRuntime
   - BionaException class (std::runtime_error subclass) carrying BionaError code
   - Helper: void throwIf(bool condition, BionaError code, std::string_view msg)

All headers must:
- Use #pragma once
- Include Doxygen comments on every public type and field
- Have zero external dependencies beyond C++17 stdlib and std::optional
```

---

### A-3 · Abstract Interfaces

```
You are a senior C++17 systems engineer building an on-device speech engine.
The core type headers already exist at biona/axon/core/include/biona/core/

Create the following pure abstract interface headers in
biona/axon/core/include/biona/core/interfaces/

1. inference_engine.hpp — Abstract inference engine:
   class InferenceEngine {
   public:
     virtual ~InferenceEngine() = default;
     virtual bool initialize(const ModelConfig& cfg) = 0;
     virtual InferenceResult run(const AudioFeatures& features) = 0;
     virtual void reset() = 0;          // Reset streaming state between utterances
     virtual EngineInfo info() const = 0;
     // Non-copyable, non-movable (engines hold hardware resources)
     InferenceEngine(const InferenceEngine&) = delete;
     InferenceEngine& operator=(const InferenceEngine&) = delete;
   };

   Also define:
   - using EngineCreator = std::function<std::unique_ptr<InferenceEngine>()>;
   - enum class EngineType { ONNX, TFLITE, COREML };
   - InferenceEngineFactory class with:
     * static std::unique_ptr<InferenceEngine> create(EngineType, const ModelConfig&)
     * static void registerEngine(EngineType, EngineCreator)
     * Private: static std::unordered_map<EngineType, EngineCreator>& registry()

2. vad.hpp — Abstract VAD interface:
   class VAD {
   public:
     virtual ~VAD() = default;
     virtual bool isSpeech(const AudioChunk& chunk) noexcept = 0;
     virtual float confidence(const AudioChunk& chunk) noexcept = 0;
     virtual void reset() noexcept = 0;
   };
   enum class VADType { WEBRTC, SILERO, ENERGY };
   struct VADConfig {
     VADType type = VADType::WEBRTC;
     int sensitivity = 2;          // 0=low, 1=medium, 2=aggressive (WebRTC)
     int hangover_ms = 200;        // Continue after last positive detection
     int reset_silence_ms = 500;   // Reset state after this much silence
   };
   std::unique_ptr<VAD> createVAD(const VADConfig& cfg);

3. signal_layer.hpp — Signal assembly interface:
   class SignalLayer {
   public:
     virtual ~SignalLayer() = default;
     virtual Signal assemble(const InferenceResult& sync_result) = 0;
     virtual void attachAsyncSignal(int64_t chunk_id, const EmotionSignal&) = 0;
     virtual void attachAsyncSignal(int64_t chunk_id, const SpeakerSignal&) = 0;
     virtual void attachAsyncSignal(int64_t chunk_id, const IntentSignal&) = 0;
   };

4. frontend.hpp — Feature extraction interface:
   class Frontend {
   public:
     virtual ~Frontend() = default;
     // Extract log-mel spectrogram from raw audio chunk
     // SPEC: 80 mel bands, 25ms window, 10ms hop, 16kHz sample rate
     // NO dynamic allocation. Pre-allocated output buffer.
     virtual void extract(const AudioChunk& chunk, AudioFeatures& out) noexcept = 0;
   };

All interfaces must:
- Include detailed Doxygen comments explaining threading guarantees
- Note which methods are safe to call from the hot path (noexcept)
- Note which methods must NOT be called concurrently
```

---

### A-4 · ONNX Contract Header (The A↔B Bridge)

```
You are a senior C++17 systems engineer. Create a critical contract header that
defines the exact ONNX model interface that both:
- Biona Axon (C++ runtime) expects to consume
- Biona Lab (PyTorch training) must produce on export

Create: biona/axon/core/include/biona/core/onnx_contract.hpp

This header is the single source of truth for the streaming Emformer ONNX interface.

It must define as constexpr string_view constants:

INPUT TENSOR NAMES:
  ONNX_INPUT_CHUNK          = "chunk"
  ONNX_INPUT_MEMORY         = "memory"
  ONNX_INPUT_LC_KEY         = "left_context_key"
  ONNX_INPUT_LC_VAL         = "left_context_val"

OUTPUT TENSOR NAMES:
  ONNX_OUTPUT_LOGITS        = "logits"
  ONNX_OUTPUT_MEMORY        = "memory_out"
  ONNX_OUTPUT_LC_KEY        = "lc_key_out"
  ONNX_OUTPUT_LC_VAL        = "lc_val_out"
  ONNX_OUTPUT_EMBEDDING     = "embedding"

SHAPE CONSTANTS (as constexpr int):
  ONNX_MEL_BANDS            = 80
  ONNX_EMBEDDING_DIM        = 512
  ONNX_MEMORY_VECTORS       = 4       // Memory bank summary vectors per chunk
  ONNX_LEFT_CONTEXT_FRAMES  = 64      // 640ms at 10ms hop
  ONNX_CHUNK_FRAMES         = 2       // 20ms chunk at 10ms hop
  ONNX_OPSET_VERSION        = 17

AUDIO PROCESSING CONSTANTS:
  AUDIO_SAMPLE_RATE_HZ      = 16000
  AUDIO_CHUNK_MS            = 20
  AUDIO_MEL_BANDS           = 80
  AUDIO_WINDOW_MS           = 25
  AUDIO_HOP_MS              = 10

Also include a struct ONNXContractValidator with a static method:
  static bool validate(const std::vector<std::string>& actual_input_names,
                       const std::vector<std::string>& actual_output_names);
This method checks the loaded model's tensor names against the contract and
logs a clear error message for each mismatch.

Add a large Doxygen block at the top explaining:
- This file is the ONLY place these constants should be defined
- Biona Lab's torch.onnx.export call must match these names exactly
- Any change here requires a version bump and retraining
```

---

### A-5 · SafeLog & Telemetry

```
You are a senior C++17 systems engineer building a security-first speech engine.
Existing headers: biona/axon/core/include/biona/core/

Create biona/axon/security/include/biona/security/safe_log.hpp
and     biona/axon/security/src/safe_log.cpp

STRICT SECURITY RULES (enforce at compile time where possible):
- NEVER log: audio samples, transcript text, embedding vectors, user identifiers
- ALWAYS log: performance metrics, system state, error codes

Design requirements:

1. enum class LogLevel { PERF_ONLY = 0, SYSTEM_STATE = 1, DEBUG_ONLY = 2 };

2. class SafeLog (static methods only, singleton logger backend):
   // Structured metric — name + numeric value ONLY. No free text.
   static void metric(std::string_view name, double value) noexcept;

   // Discrete event — name only
   static void event(std::string_view name) noexcept;

   // Error — code string + detail string. Detail must never contain user data.
   static void error(std::string_view code, std::string_view detail) noexcept;

   // Configure backend (call once at startup)
   static void configure(LogLevel level, std::ostream* out = &std::cerr);

   // Debug — completely stripped in NDEBUG builds at compile time
   #ifndef NDEBUG
   template<typename... Args>
   static void debug(std::string_view tag, Args&&... args);
   #else
   template<typename... Args>
   static void debug(std::string_view, Args&&...) noexcept {}
   #endif

3. Define these pre-approved metric name constants as constexpr string_view:
   BIONA_METRIC_INFERENCE_LATENCY_MS    = "inference.latency_ms"
   BIONA_METRIC_VAD_SPEECH_FRAMES       = "vad.speech_frames"
   BIONA_METRIC_VAD_SILENCE_FRAMES      = "vad.silence_frames"
   BIONA_METRIC_QUEUE_OVERFLOW          = "signal.queue_overflow"
   BIONA_METRIC_SIGNAL_TIMEOUT          = "signal.timeout"
   BIONA_METRIC_MEMORY_RSS_KB           = "memory.rss_kb"

   BIONA_EVENT_ENGINE_INITIALIZED       = "engine.initialized"
   BIONA_EVENT_MODEL_LOAD_SUCCESS       = "model.load_success"
   BIONA_EVENT_MODEL_LOAD_FAILED        = "model.load_failed"
   BIONA_EVENT_ENGINE_RESET             = "engine.reset"

4. Thread safety: all static methods must be safe to call from multiple threads.
   Use a mutex internally for the output stream.

5. Implementation must never call printf, puts, cout directly —
   route everything through the configured ostream backend.

Include unit tests in biona/axon/tests/unit/test_safe_log.cpp using
a simple assert-based or catch2 framework that verifies:
- Metrics are emitted correctly
- Debug logs are suppressed when NDEBUG is defined (test this via mock)
```

---

### A-6 · SecretManager

```
You are a senior C++17 systems engineer. Create the secret management abstraction
for the Biona engine. No secrets in source code. No secrets in build artifacts.

Create:
  biona/axon/security/include/biona/security/secret_manager.hpp
  biona/axon/security/src/secret_manager_env.cpp   ← environment var impl
  biona/axon/security/src/secret_manager_factory.cpp

1. Abstract class SecretManager:
   class SecretManager {
   public:
     virtual ~SecretManager() = default;
     // Returns secret value for key. Throws BionaException(InitFailed) if not found.
     virtual std::string get(std::string_view key) const = 0;
     // Returns true if key exists without revealing value
     virtual bool has(std::string_view key) const noexcept = 0;
   };

2. class EnvSecretManager : public SecretManager
   - Reads from process environment variables via getenv()
   - Optional key prefix (e.g., prefix="BIONA_" means key "MODEL_KEY" → env var "BIONA_MODEL_KEY")
   - Throws if env var not found

3. Factory function:
   std::unique_ptr<SecretManager> createSecretManager(const std::string& type);
   // type: "env" → EnvSecretManager
   // type: "keychain" → stub that throws "not implemented" (iOS impl added later)
   // type: "keystore" → stub that throws "not implemented" (Android impl added later)
   // type: "tpm" → stub that throws "not implemented" (Linux enterprise impl later)

4. Define these pre-approved secret key name constants:
   BIONA_SECRET_MODEL_ENCRYPTION_KEY = "MODEL_ENCRYPTION_KEY"
   BIONA_SECRET_MODEL_HMAC_KEY       = "MODEL_HMAC_KEY"

5. SECURITY: The get() return value (std::string) will contain key material.
   Document clearly that callers must call explicit_bzero or equivalent
   on the string buffer after use. Provide a SecureString RAII wrapper
   that zero-fills its buffer on destruction.

No unit tests for this module — secret material must not appear in test output.
Instead, create a test that only verifies has() returns true for existing vars
and false for missing vars, without printing any values.
```

---

### A-7 · ModelLoader with AES-256-GCM

```
You are a senior C++17 systems engineer. Create a secure model loader that
decrypts ONNX model files in memory only. Model plaintext must NEVER touch disk.

Existing: biona/axon/security/include/biona/security/
           safe_log.hpp, secret_manager.hpp

Create:
  biona/axon/security/include/biona/security/model_loader.hpp
  biona/axon/security/src/model_loader.cpp
  biona/axon/security/include/biona/security/secure_buffer.hpp

1. class SecureBuffer — RAII wrapper for sensitive memory:
   - Holds std::vector<uint8_t> internally
   - Destructor calls explicit_bzero (Linux) / SecureZeroMemory (Windows) / memset_s (POSIX)
   - Calls mlock() on construction to prevent OS paging to swap
   - Calls munlock() on destruction
   - Non-copyable. Movable.
   - data() and size() accessors
   - On mlock() failure: log BIONA_EVENT_MODEL_LOAD_FAILED and throw BionaException(OOM)

2. struct EncryptedModelBundle:
   - magic: std::array<uint8_t, 4>   = {0x42, 0x49, 0x4F, 0x4E  // "BION"
   - version: uint32_t               = 1
   - iv: std::array<uint8_t, 12>     // AES-GCM 96-bit IV
   - tag: std::array<uint8_t, 16>    // AES-GCM 128-bit auth tag
   - hmac: std::array<uint8_t, 32>   // HMAC-SHA256 over (iv + tag + ciphertext)
   - ciphertext: std::vector<uint8_t>
   Provide: static EncryptedModelBundle fromFile(const std::filesystem::path&);

3. class ModelLoader:
   // Primary method: load, verify HMAC, decrypt into locked memory
   SecureBuffer load(const std::filesystem::path& bundle_path,
                     const SecretManager& secrets);
   private:
     // 1. Read and parse EncryptedModelBundle
     // 2. Derive key from BIONA_SECRET_MODEL_ENCRYPTION_KEY via HKDF-SHA256
     // 3. Verify HMAC-SHA256 BEFORE any decryption attempt
     // 4. AES-256-GCM decrypt — verify auth tag before returning ANY bytes
     // 5. Return SecureBuffer with plaintext ONNX bytes
     SecureBuffer decryptAESGCM(const EncryptedModelBundle&, const std::string& key_hex);

4. Use OpenSSL (libssl/libcrypto) for AES-GCM and HMAC.
   Add find_package(OpenSSL REQUIRED) to security/CMakeLists.txt.

5. Implement a companion CLI tool: snie/tools/encrypt_model/
   A small C++ program that takes a plaintext .onnx file + key env var
   and produces an encrypted .biona bundle. This is used at model packaging time.

CRITICAL RULES in comments:
- VERIFY HMAC BEFORE DECRYPTION — document why (prevents padding oracle attacks)
- VERIFY GCM TAG BEFORE USING ANY PLAINTEXT BYTES
- mlock() the output buffer immediately after allocation
- Never log key material, IV, or any portion of plaintext
```

---

### A-8 · WebRTC VAD Integration

```
You are a senior C++17 systems engineer. Implement the WebRTC VAD adapter
for the Biona engine.

Existing interface: biona/axon/core/include/biona/core/interfaces/vad.hpp

Create:
  biona/axon/core/src/vad/webrtc_vad.hpp
  biona/axon/core/src/vad/webrtc_vad.cpp
  biona/axon/core/src/vad/energy_vad.hpp   ← fallback
  biona/axon/core/src/vad/energy_vad.cpp
  biona/axon/core/src/vad/vad_factory.cpp  ← implements createVAD()

For WebRTCVAD:
- Include WebRTC VAD via: find_package or FetchContent from
  https://github.com/dpirch/libfvad (C port of WebRTC VAD, Apache 2.0)
- class WebRTCVAD : public VAD
- Constructor: takes VADConfig, initializes fvad handle
- isSpeech(): calls fvad_process() — must complete in < 2ms
- Implement hangover logic:
  * Keep a frame counter since last positive detection
  * Return true for hangover_ms after last positive (captures trailing phonemes)
  * Reset internal state after reset_silence_ms of continuous silence
- confidence(): return fvad confidence approximation (0.0 or 1.0 from WebRTC,
  interpolated with hangover decay)
- reset(): reset fvad state + hangover counter

For EnergyVAD (fallback for clean environments):
- Compute RMS energy of chunk
- Compare against configurable threshold (default: -40 dBFS)
- No external dependency
- < 0.1ms per chunk

For createVAD() factory:
- VADType::WEBRTC → WebRTCVAD
- VADType::ENERGY → EnergyVAD
- VADType::SILERO → throw BionaException(UnsupportedRuntime, "Silero VAD not yet implemented")

PERFORMANCE RULES:
- No dynamic allocation in isSpeech() or confidence() — pre-allocate all buffers
- No mutex in hot path — VAD is single-threaded (called from main pipeline thread only)
- Assert that chunk.n_samples == (sample_rate * chunk_ms / 1000) at debug time

Include unit tests in biona/axon/tests/unit/test_vad.cpp:
- Test with synthetic silence (all zeros) → expect false
- Test with synthetic speech-like signal (sine wave 300Hz) → expect true (after warmup)
- Test hangover: speech then silence → should remain true for hangover_ms
```

---

### A-9 · OnnxEngine Implementation

```
You are a senior C++17 systems engineer. Implement the ONNX Runtime inference
engine for streaming Emformer ASR.

Existing files:
  biona/axon/core/include/biona/core/interfaces/inference_engine.hpp
  biona/axon/core/include/biona/core/onnx_contract.hpp
  biona/axon/security/include/biona/security/model_loader.hpp
  biona/axon/security/include/biona/security/safe_log.hpp

Create:
  biona/axon/engines/onnx/include/biona/axons/onnx_engine.hpp
  biona/axon/engines/onnx/src/onnx_engine.cpp

class OnnxEngine : public InferenceEngine:

initialize(const ModelConfig& cfg):
  1. Use ModelLoader to decrypt model into SecureBuffer
  2. Create Ort::Env and Ort::SessionOptions
  3. Load session from in-memory buffer (not from file path — use CreateFromArray)
  4. Validate loaded model tensor names against ONNXContractValidator
  5. Create IOBinding for zero-copy tensor reuse
  6. Pre-allocate ALL input/output tensors to their maximum sizes — NO allocation in run()
  7. Initialize streaming state tensors (memory, left_context_key, left_context_val)
     to zeros with correct shapes from onnx_contract.hpp constants
  8. Log BIONA_EVENT_ENGINE_INITIALIZED on success

run(const AudioFeatures& features):
  1. Copy features into pre-allocated input tensor (no new allocation)
  2. Bind input tensors via IOBinding
  3. Run session
  4. Extract logits → CTC greedy decode → text string
  5. Extract embedding tensor (512-dim float vector)
  6. Update streaming state tensors (copy memory_out → memory, etc.)
  7. Measure wall-clock latency, log BIONA_METRIC_INFERENCE_LATENCY_MS
  8. Return InferenceResult { embedding, text, latency_ms }

reset():
  Zero-fill all streaming state tensors (memory, left_context_key, left_context_val)

CTC Greedy Decoder (implement inline):
  - Take logits [T x vocab_size]
  - Argmax per frame
  - Collapse repeats and remove blank token (token 0)
  - Return decoded string

Threading: OnnxEngine is NOT thread-safe. One instance per thread.
Document this in the header clearly.

Add to engines/onnx/CMakeLists.txt:
  find_package(OnnxRuntime REQUIRED)
  target_link_libraries(biona_onnx_engine PRIVATE OnnxRuntime::OnnxRuntime)
```

---

### A-10 · SPSC Queue & Async Signal Thread Pool

```
You are a senior C++17 systems engineer. Implement the async signal processing
infrastructure for optional signals (SER, speaker ID, intent).

Create:
  biona/axon/core/include/biona/core/spsc_queue.hpp
  biona/axon/core/include/biona/core/signal_thread_pool.hpp
  biona/axon/core/src/signal_thread_pool.cpp

1. SPSCQueue<T, Capacity> — Lock-free single-producer single-consumer ring queue:
   template<typename T, size_t Capacity>
   class SPSCQueue {
     alignas(64) std::array<T, Capacity> buf_;  // cache-line aligned
     alignas(64) std::atomic<size_t> rd_{0};
     alignas(64) std::atomic<size_t> wr_{0};
   public:
     // Returns false if full — caller must NOT block
     bool push(T&& item) noexcept;
     // Returns false if empty
     bool pop(T& out) noexcept;
     bool empty() const noexcept;
     bool full() const noexcept;
   };
   - Use std::memory_order_acquire/release — never seq_cst in push/pop
   - No mutex, no condition variable, no heap allocation after construction
   - Static assert that T is trivially copyable or movable

2. struct SignalTask:
   int64_t chunk_id;
   InferenceResult result;
   std::chrono::steady_clock::time_point enqueued_at;

3. using AsyncSignalCallback = std::function<void(int64_t chunk_id, const Signal&)>;

4. class SignalThreadPool:
   // Fixed thread pool for computing optional signals in background
   explicit SignalThreadPool(size_t n_threads, int timeout_ms = 500);
   ~SignalThreadPool();  // Joins all threads cleanly

   // Called from main thread — push task onto queue
   // Returns false if queue full (overflow) — emits BIONA_METRIC_QUEUE_OVERFLOW
   bool enqueue(SignalTask&& task) noexcept;

   // Register callback for when async signals complete
   void setCallback(AsyncSignalCallback cb);

   private:
     static constexpr size_t QUEUE_CAPACITY = 64;
     SPSCQueue<SignalTask, QUEUE_CAPACITY> queue_;
     std::vector<std::thread> workers_;
     std::atomic<bool> running_{true};
     AsyncSignalCallback callback_;
     std::mutex callback_mutex_;
     int timeout_ms_;

     void workerLoop();  // Each thread runs this
     // timeout check: if task older than timeout_ms_, deliver null signal
     // and log BIONA_METRIC_SIGNAL_TIMEOUT

RULES:
- No heap allocation in enqueue() — it is called from the hot path
- Each worker thread owns its own inference model instance (no sharing)
- Workers must check running_ atomically — clean shutdown within 100ms
- Use std::memory_order_relaxed for running_ reads in hot loop
```

---

### A-11 · C ABI SDK

```
You are a senior C++17 systems engineer. Create the public C ABI for the
Biona engine. This is the only surface that external developers touch.

Create:
  biona/axon/sdk/include/biona/biona.h        ← Public C header (extern "C")
  biona/axon/sdk/src/biona_sdk.cpp            ← C++ implementation of C API
  biona/axon/sdk/include/biona/biona.hpp       ← Optional C++ convenience wrapper

The C header must be pure C (compilable as C99):

typedef void* BIONA_Engine;

typedef struct {
  const char* model_bundle_path;
  int         sample_rate_hz;     // Must be 16000
  int         chunk_size_ms;      // Recommended: 20
  int         enable_async;       // 1 = enable SER/speaker/intent, 0 = sync only
  int         log_level;          // 0=PERF_ONLY 1=SYSTEM 2=DEBUG
  const char* secret_manager_type; // "env" | "keychain" | "keystore" | "tpm"
} BIONA_Config;

typedef struct {
  const char*  text;              // Null-terminated transcript. Owned by SDK.
  int64_t      timestamp_ms;
  int32_t      latency_ms;
  const float* embedding;         // 512 floats. Owned by SDK. Valid until next call.
  int32_t      embedding_len;
} BIONA_Result;

typedef struct {
  int64_t chunk_id;
  int     signal_type;            // 0=EMOTION 1=SPEAKER 2=INTENT
  float   values[8];              // Signal-specific payload
  float   confidence;
} BIONA_AsyncSignal;

typedef void (*BIONA_SignalCallback)(const BIONA_AsyncSignal* signal, void* user_data);

typedef enum {
  BIONA_OK                   = 0,
  BIONA_ERR_INIT_FAILED      = 1,
  BIONA_ERR_DECRYPT_FAILED   = 2,
  BIONA_ERR_INVALID_AUDIO    = 3,
  BIONA_ERR_INFERENCE_FAILED = 4,
  BIONA_ERR_OOM              = 5,
} BIONA_Error;

// Lifecycle
BIONA_Error BIONA_Create (const BIONA_Config* cfg, BIONA_Engine* out_engine);
void       BIONA_Destroy(BIONA_Engine engine);

// Inference — must return within latency budget
BIONA_Error BIONA_ProcessChunk(BIONA_Engine engine,
                              const int16_t* pcm, size_t n_samples,
                              BIONA_Result*   out_result);

// Async signal delivery
BIONA_Error BIONA_RegisterSignalCallback(BIONA_Engine engine,
                                        BIONA_SignalCallback cb,
                                        void* user_data);

// Utility
const char* BIONA_ErrorString(BIONA_Error err);

Implementation in biona_sdk.cpp:
- BIONA_Engine is a raw pointer to an internal EngineContext struct
- EngineContext holds: unique_ptr<InferenceEngine>, unique_ptr<VAD>,
  unique_ptr<Frontend>, unique_ptr<SignalLayer>, unique_ptr<SignalThreadPool>
- BIONA_ProcessChunk runs: VAD → (if speech) Frontend → Engine → SignalLayer → result
- All exceptions caught at API boundary, translated to BIONA_Error codes
- Thread safety: BIONA_ProcessChunk is NOT thread-safe per engine handle.
  Document this. Users create one handle per thread.
```

---

### A-12 · Integration Test & Eval Harness Stub

```
You are a senior C++17 systems engineer. Create the test infrastructure for
the Biona engine.

Create:
  biona/axon/tests/unit/test_signal_schema.cpp
  biona/axon/tests/unit/test_spsc_queue.cpp
  biona/axon/tests/integration/test_pipeline_mock.cpp
  biona/axon/tests/eval/eval_harness.cpp
  biona/axon/tests/CMakeLists.txt

1. test_signal_schema.cpp — Schema stability tests:
   - Verify Signal struct can be constructed with embedding only (all others null)
   - Verify adding emotion does not change text or embedding fields
   - Verify to_json() produces valid JSON with schema_version field
   - Use simple assert() — no external test framework required

2. test_spsc_queue.cpp — Lock-free queue correctness:
   - Single-threaded push/pop correctness
   - Overflow handling (push returns false when full, does not block)
   - Producer/consumer thread test: 10k items, verify all received in order
   - Verify no heap allocation occurs after construction (use custom allocator hook)

3. test_pipeline_mock.cpp — Full pipeline with mock engine:
   - Create a MockInferenceEngine that returns a fixed 512-dim embedding and "hello"
   - Wire: VAD (energy) → Frontend (mock) → MockEngine → SignalLayer
   - Feed 10 audio chunks of silence → verify VAD gates them (no inference)
   - Feed 10 chunks of 300Hz sine wave → verify inference is called, Signal produced
   - Verify Signal.embedding.size() == 512
   - Verify Signal.text == "hello"

4. eval_harness.cpp — Latency measurement tool:
   - Reads a directory of .wav files (16kHz mono PCM)
   - Runs BIONA_ProcessChunk on each 20ms chunk
   - Records per-chunk latency
   - Outputs JSON report: { p50_ms, p95_ms, p99_ms, max_ms, n_chunks, model_path }
   - Fails with exit code 1 if p99_ms > 150 (hard gate from Section 12 of Biona spec)

Update tests/CMakeLists.txt to build all test targets.
```

---

---

## VALIDATION — Run After Both Tracks Complete

```
You are a senior C++17 systems engineer and ML engineer performing a
cross-system integration check.

Verify that the following constants are IDENTICAL between Biona Axon and Biona Lab.
Read the actual files and compare — do not assume they match.

Biona Axon constants (in biona/axon/core/include/biona/core/onnx_contract.hpp):
  ONNX_MEL_BANDS, ONNX_CHUNK_FRAMES, ONNX_EMBEDDING_DIM,
  ONNX_MEMORY_VECTORS, ONNX_LEFT_CONTEXT_FRAMES,
  AUDIO_SAMPLE_RATE_HZ, AUDIO_WINDOW_MS, AUDIO_HOP_MS
  Input/output tensor names

Biona Lab constants (in biona/lab/biona_lab/data/features.py and
                        biona/lab/biona_lab/model/emformer.py):
  SAMPLE_RATE_HZ, MEL_BANDS, WINDOW_MS, HOP_MS,
  Emformer: input_dim, segment_length, left_context_length, memory_size,
  ONNX export: input_names, output_names

For each constant pair, print: [MATCH] or [MISMATCH: A={value} B={value}]

If ANY mismatch exists: print a clear error explaining which file to update
and what value it should be changed to. The source of truth is Axon's
onnx_contract.hpp — Biona Lab must match it, not the other way around.
```

---

*End of Biona Claude CLI Implementation Prompts v1.0*
*Run prompts A-1 through A-12 in Axon, B-1 through B-8 in Lab.*
*Run the VALIDATION prompt after both tracks are complete.*
