# Biona Lab — Python Training Pipeline
## Claude CLI Prompts · 8 Prompts

> **How to use**
> Paste each prompt into Claude CLI in VS Code **in order**, one at a time.
> Run these in a separate VS Code terminal from Biona Axon.
> Complete the **SETUP** prompt first if you haven't already (run once, not per track).
> **Important:** Do not run B-5 (Emformer) or B-7 (ONNX Export) until
> A-4 (ONNX Contract Header) is complete — Biona Lab must match Biona Axon's contract.

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
- `axon/`   — On-device C++17 inference engine (Biona Axon)
- `lab/` — Biona Lab: Server-side Python training pipeline

## System Boundary Rule
**Biona Axon runs on-device. Biona Lab runs on servers. These are separate systems.**
LLM inference, cloud APIs, and large-batch GPU processing belong in Biona Lab ONLY.
The ONNX model produced by Biona Lab is the only artifact that crosses the boundary.

Also create a root `.gitignore` with entries for: build/, __pycache__/, *.pyc,
.env, *.onnx, *.tflite, *.mlmodel, *.key, *.enc, node_modules/, .DS_Store
```


---

---

# BIONA LAB — Python Training Pipeline
## Lab: Run these prompts in order in `biona/lab/`

---

### B-1 · Python Project Scaffold

```
You are a senior ML engineer. Create a production-grade Python project scaffold
for the Biona speech model training pipeline.

Work inside: biona/lab/

Create the following structure:

biona/lab/
├── pyproject.toml              ← Modern Python packaging
├── requirements.txt            ← Pinned dependencies
├── .env.example                ← Environment variable template (no secrets)
├── Makefile                    ← Common commands: train, label, export, eval
├── biona_lab/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── librispeech.py      ← Dataset loader (added in B-2)
│   │   ├── features.py         ← Log-mel extraction (added in B-3)
│   │   └── shards.py           ← WebDataset writer (added in B-4)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── emformer.py         ← Emformer definition (added in B-5)
│   │   └── export.py           ← ONNX export (added in B-7)
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py            ← Training loop (added in B-6)
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── semantic_labeler.py ← LLM labeling pipeline (added in B-8)
│   └── eval/
│       ├── __init__.py
│       └── wer.py              ← WER computation against eval harness
├── scripts/
│   ├── download_librispeech.sh
│   ├── run_labeling.py
│   └── export_model.py
└── tests/
    ├── test_features.py
    └── test_schema.py

pyproject.toml must specify:
  - python >= 3.11
  - package name: biona-lab
  - tool.ruff for linting

requirements.txt must pin these versions:
  torch==2.3.1
  torchaudio==2.3.1
  webdataset==0.2.86
  openai==1.30.0
  jsonschema==4.22.0
  librosa==0.10.2
  soundfile==0.12.1
  numpy==1.26.4
  tqdm==4.66.4
  pytest==8.2.2

.env.example must contain (with placeholder values):
  BIONA_MODEL_ENCRYPTION_KEY=<hex-encoded-256-bit-key>
  BIONA_MODEL_HMAC_KEY=<hex-encoded-256-bit-key>
  OPENAI_API_KEY=<your-openai-key>
  LIBRISPEECH_DATA_DIR=/data/librispeech
  SHARD_OUTPUT_DIR=/data/shards
  MODEL_OUTPUT_DIR=/models/snie

Makefile targets:
  make download    → runs scripts/download_librispeech.sh
  make shards      → python -m biona_lab.data.shards
  make train       → python -m biona_lab.training.train
  make label       → python scripts/run_labeling.py
  make export      → python scripts/export_model.py
  make test        → pytest tests/
```

---

### B-2 · LibriSpeech Data Loader

```
You are a senior ML engineer building a speech training pipeline.
Project structure already exists at biona/lab/

Create: biona/lab/biona_lab/data/librispeech.py

Implement a LibriSpeech dataset loader with these requirements:

1. class LibriSpeechLoader:
   __init__(self, data_dir: Path, splits: list[str] = ["train-clean-100"])
   Supported splits: train-clean-100, train-clean-360, train-other-500,
                     dev-clean, dev-other, test-clean, test-other

   load() -> Iterator[dict]:
     Yields records: { "audio_path": Path, "transcript": str,
                       "speaker_id": str, "chapter_id": str,
                       "duration_s": float, "source": str }

2. Validation on load:
   - Verify audio file exists and is readable
   - Parse transcript from .trans.txt files
   - Compute audio duration using soundfile (no loading full audio)
   - Discard pairs where duration/token count ratio > 3 sigma from dataset mean
     (prevents misaligned or corrupted entries from poisoning training)
   - Log discarded count at the end: "Discarded N entries (K sigma filter)"

3. Segmentation for long utterances:
   def segment(record: dict, max_duration_s: float = 30.0) -> list[dict]:
     If audio is longer than max_duration_s, split at silence boundaries
     Use librosa.effects.split() with top_db=30 for silence detection
     Each segment gets: { ...record, "segment_start_s": float, "segment_end_s": float }
     Segments shorter than 0.5s are discarded

4. Normalization:
   def normalize_transcript(text: str) -> str:
     Lowercase, remove punctuation except apostrophes,
     collapse whitespace, strip leading/trailing spaces

5. Dataset statistics on load (log, don't store):
   - Total hours of audio per split
   - Speaker count
   - Vocabulary size (unique words)
   - Mean / std of duration distribution

Also create: biona/lab/scripts/download_librispeech.sh
  A bash script that downloads and extracts LibriSpeech tar.gz files
  from openslr.org for the specified splits.
  Usage: bash download_librispeech.sh train-clean-100 /data/librispeech
```

---

### B-3 · Log-Mel Feature Extractor

```
You are a senior ML engineer. Create the log-mel spectrogram feature extractor.

CRITICAL: These parameters MUST match the C++ Frontend spec in Biona Axon exactly.
Any mismatch causes a silent train/inference accuracy bug.

Create: biona/lab/biona_lab/data/features.py

# ══ LOCKED PARAMETERS — DO NOT CHANGE WITHOUT UPDATING onnx_contract.hpp ══
SAMPLE_RATE_HZ  = 16000  # Must match AUDIO_SAMPLE_RATE_HZ in onnx_contract.hpp
MEL_BANDS       = 80     # Must match AUDIO_MEL_BANDS in onnx_contract.hpp
WINDOW_MS       = 25     # Must match AUDIO_WINDOW_MS in onnx_contract.hpp
HOP_MS          = 10     # Must match AUDIO_HOP_MS in onnx_contract.hpp
WINDOW_SAMPLES  = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # = 400
HOP_SAMPLES     = int(SAMPLE_RATE_HZ * HOP_MS / 1000)      # = 160
FMIN            = 80     # Hz
FMAX            = 7600   # Hz
# ══════════════════════════════════════════════════════════════════════════

class LogMelExtractor:
  def __init__(self):
    # Pre-build mel filterbank once — do not recompute per sample
    self._mel_fb = self._build_mel_filterbank()

  def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns log-mel spectrogram of shape [T x MEL_BANDS] as float32.
    T = number of frames = ceil((len(audio) - WINDOW_SAMPLES) / HOP_SAMPLES) + 1
    """
    assert sr == SAMPLE_RATE_HZ, f"Expected {SAMPLE_RATE_HZ}Hz, got {sr}Hz"
    # Resample if needed (should not happen in practice)
    # Apply pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
    # STFT with Hann window
    # Apply mel filterbank
    # Log compression: log(max(spectrogram, 1e-10))
    # Per-utterance normalization: zero mean, unit variance
    # Return float32 array [T x 80]

  def extract_streaming_chunk(self, chunk: np.ndarray) -> np.ndarray:
    """
    Extract features from a single 20ms chunk (streaming mode).
    Maintains internal state (prev_sample for pre-emphasis).
    Returns [2 x 80] (2 frames at 10ms hop from 20ms chunk).
    """

  def _build_mel_filterbank(self) -> np.ndarray:
    # Use librosa.filters.mel() with exact parameters above

Also create: biona/lab/tests/test_features.py
  Tests:
  1. Output shape is [T x 80] for various audio lengths
  2. Streaming chunk returns [2 x 80]
  3. Full utterance and streaming chunk-by-chunk produce consistent features
     (within floating point tolerance) — this is CRITICAL for train/infer parity
  4. Assert sr mismatch raises AssertionError
```

---

### B-4 · WebDataset Shard Writer

```
You are a senior ML engineer. Create the WebDataset shard writer that
packages the processed training data.

Create: biona/lab/biona_lab/data/shards.py

class ShardWriter:
  def __init__(self, output_dir: Path, shard_size_mb: int = 512,
               max_samples_per_shard: int = 10000):

  def write(self, loader: LibriSpeechLoader, extractor: LogMelExtractor):
    """
    Pipeline: load → segment → normalize → extract features → write shards
    Shows tqdm progress bar with: samples/sec, hours processed, shards written
    """

  def _write_sample(self, writer, record: dict, features: np.ndarray):
    """
    Each WebDataset entry contains:
      "__key__":     unique sample ID (speaker_chapter_utterance_segment)
      "audio.npy":   log-mel features [T x 80] float32 as bytes
      "text.txt":    normalized transcript
      "meta.json":   {
        "duration_s": float,
        "source": str,       // "librispeech-clean" | "librispeech-other"
        "speaker_id": str,
        "shard_hash": str,   // SHA256 of this sample's audio.npy
        "schema_version": "1.0"
      }
    """

After writing all shards:
  - Write a manifest.json in output_dir: { total_samples, total_hours,
    shard_files: [{ filename, sha256, n_samples }], schema_version }
  - The manifest SHA256 is the dataset fingerprint used in eval reports

Also implement: python -m biona_lab.data.shards as CLI:
  python -m biona_lab.data.shards \
    --splits train-clean-100 train-clean-360 \
    --output-dir $SHARD_OUTPUT_DIR \
    --shard-size-mb 512
```

---

### B-5 · Emformer Model Definition

```
You are a senior ML engineer specializing in streaming speech recognition.
Create the Emformer model that matches the C++ ONNX contract exactly.

CRITICAL CONTRACT — must match biona/axon/core/include/biona/core/onnx_contract.hpp:
  Input names:   chunk, memory, left_context_key, left_context_val
  Output names:  logits, memory_out, lc_key_out, lc_val_out, embedding
  MEL_BANDS      = 80
  EMBEDDING_DIM  = 512
  MEMORY_VECTORS = 4
  LC_FRAMES      = 64
  CHUNK_FRAMES   = 2
  OPSET          = 17

Create: biona/lab/biona_lab/model/emformer.py

class BionaEmformer(nn.Module):
  Use torchaudio.models.Emformer as the backbone encoder.
  Add:
  - A linear projection layer: encoder_dim → EMBEDDING_DIM (512)
    for producing the acoustic embedding
  - A CTC output head: encoder_dim → vocab_size
  - vocab_size defaults to 29 (26 letters + apostrophe + space + blank)

  forward(chunk, memory, left_context_key, left_context_val):
    Returns: logits, memory_out, lc_key_out, lc_val_out, embedding

  Emformer configuration:
    input_dim = 80            # mel bands
    num_heads = 8
    ffn_dim = 2048
    num_layers = 20
    segment_length = 2        # chunk frames (20ms)
    left_context_length = 64  # LC_FRAMES
    right_context_length = 0  # Zero look-ahead (minimum latency)
    memory_size = 4           # MEMORY_VECTORS
    dropout = 0.1

  @torch.no_grad()
  def get_initial_state(self, batch_size: int = 1) -> tuple:
    Returns (memory, left_context_key, left_context_val) all zeros
    with correct shapes for the ONNX contract.

  @classmethod
  def from_config(cls, vocab_size: int = 29) -> "BionaEmformer":
    Factory with the above fixed config.

Also create: biona/lab/tests/test_model.py
  Tests:
  1. Model forward pass with random inputs — correct output shapes
  2. Streaming consistency: chunk-by-chunk output matches batch output
     (within tolerance) for a 5-second utterance
  3. Verify output tensor names match ONNX contract constants (before export)
  4. Embedding shape is [batch x 512]
```

---

### B-6 · CTC Training Loop

```
You are a senior ML engineer. Create the CTC training loop for the Biona
Emformer model.

Create: biona/lab/biona_lab/training/train.py

Requirements:

1. Data loading:
   - Load WebDataset shards from SHARD_OUTPUT_DIR
   - Dynamic batching: group by similar lengths to minimize padding
   - Batch size: 32 (configurable)
   - Use WebDataset's shuffle buffer (10000 samples)
   - Prefetch 4 batches

2. Training configuration (dataclass):
   @dataclass
   class TrainConfig:
     learning_rate: float = 3e-4
     warmup_steps: int = 5000
     max_steps: int = 200000
     grad_clip: float = 1.0
     checkpoint_every: int = 5000
     eval_every: int = 2500
     batch_size: int = 32
     device: str = "cuda"  # or "cpu"
     amp: bool = True       # Automatic mixed precision
     vocab_size: int = 29

3. Loss: torch.nn.CTCLoss(blank=0, zero_infinity=True)
   Input: logits [T x B x vocab_size], targets, input_lengths, target_lengths

4. Optimizer: AdamW with cosine LR schedule + warmup

5. Checkpointing:
   Save: model state_dict, optimizer state_dict, step, config, val_loss
   Filename: checkpoint_{step:08d}.pt
   Keep last 3 checkpoints only

6. Validation every eval_every steps:
   - Compute CTC loss on dev-clean shard
   - Compute WER using greedy decoder
   - Log: step, train_loss, val_loss, val_wer, lr, grad_norm

7. Logging: use Python logging module only. No wandb/tensorboard dependency
   (optional: check for wandb availability and use if present)

8. CLI entry point:
   python -m biona_lab.training.train --config train_config.json
   Or: python -m biona_lab.training.train (uses defaults)

9. Resume from checkpoint:
   python -m biona_lab.training.train --resume checkpoint_00050000.pt
```

---

### B-7 · ONNX Export — The A↔B Bridge

```
You are a senior ML engineer. Create the ONNX export script that produces
the model bundle consumed by Biona Axon's C++ runtime.

THIS IS THE MOST CRITICAL FILE IN SYSTEM B — it must match the ONNX contract
defined in biona/axon/core/include/biona/core/onnx_contract.hpp exactly.

Create: biona/lab/biona_lab/model/export.py

def export_to_onnx(model: BionaEmformer, output_path: Path,
                   opset: int = 17) -> Path:
  """
  Export streaming Emformer to ONNX with correct tensor names and shapes.
  Returns path to exported .onnx file.
  """

  # Create dummy inputs matching the contract:
  chunk              = torch.zeros(1, 2, 80)           # [B, CHUNK_FRAMES, MEL_BANDS]
  memory             = torch.zeros(1, 4, 512)          # [B, MEMORY_VECTORS, d_model]
  left_context_key   = torch.zeros(20, 1, 64, 64)     # [n_layers, B, LC_FRAMES, head_dim]
  left_context_val   = torch.zeros(20, 1, 64, 64)

  torch.onnx.export(
    model,
    args=(chunk, memory, left_context_key, left_context_val),
    f=str(output_path),
    input_names=["chunk", "memory", "left_context_key", "left_context_val"],
    output_names=["logits", "memory_out", "lc_key_out", "lc_val_out", "embedding"],
    dynamic_axes={
      "chunk":            {0: "batch"},
      "logits":           {0: "batch", 1: "T"},
      "embedding":        {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
  )

def verify_onnx_export(onnx_path: Path) -> bool:
  """
  Load with onnxruntime and run one inference. Verify output tensor names
  and shapes match the contract exactly. Return True on success.
  """

def encrypt_bundle(onnx_path: Path, output_path: Path,
                   encryption_key_hex: str, hmac_key_hex: str) -> Path:
  """
  Encrypt the ONNX file into a .biona bundle using AES-256-GCM.
  Bundle format matches the EncryptedModelBundle struct in Biona Axon.
  Layout: magic(4) + version(4) + iv(12) + tag(16) + hmac(32) + ciphertext
  """

Also create: biona/lab/scripts/export_model.py
CLI that:
  1. Loads checkpoint
  2. Calls export_to_onnx()
  3. Calls verify_onnx_export() — fails hard if verification fails
  4. Calls encrypt_bundle() using keys from environment variables
  5. Writes manifest: { onnx_sha256, bundle_sha256, model_config, timestamp }

Usage:
  python scripts/export_model.py \
    --checkpoint /models/snie/checkpoint_00200000.pt \
    --output /models/snie/snie_v1.0.0.snie
```

---

### B-8 · Semantic Labeling Pipeline

```
You are a senior ML engineer. Create the LLM-based semantic labeling pipeline.

IMPORTANT: This pipeline runs at TRAINING TIME ONLY on servers.
It NEVER runs on-device. Cloud LLM inference is appropriate here.

Create: biona/lab/biona_lab/labeling/semantic_labeler.py

LABEL SCHEMA (validate every LLM output against this):
{
  "intent":          string (from fixed intent taxonomy),
  "topic":           list[string],
  "entities":        list[{"text": string, "type": string}],
  "domain":          "medical" | "legal" | "finance" | "general",
  "confidence":      float (0.0 - 1.0),
  "labeler_model":   string,
  "labeler_version": string,
  "schema_version":  "1.0"
}

Intent taxonomy (use exactly these values):
  schedule_meeting, request_info, complaint, confirm, cancel,
  provide_info, greeting, farewell, clarify, other

1. class PIIRedactor:
   def redact(self, text: str) -> str:
     Use regex to redact: phone numbers, email addresses, SSNs,
     credit card patterns. Replace with [REDACTED_TYPE].
     MUST run before any text is sent to cloud LLM.

2. class SemanticLabeler:
   def __init__(self, model: str = "gpt-4o", max_retries: int = 3):

   def label(self, transcript: str) -> dict:
     1. Redact PII first (mandatory)
     2. Call LLM with structured output (response_format={"type": "json_object"})
     3. Validate response against JSON schema (jsonschema.validate)
     4. Retry up to max_retries on schema violation
     5. Return validated label dict
     6. Raise LabelingError after max_retries exhausted

   SYSTEM PROMPT (include verbatim):
   """You are a semantic labeling assistant for speech data.
   Given a transcript, extract: intent (from the fixed taxonomy),
   topics (free list), named entities with types, and domain.
   Respond ONLY with valid JSON matching the exact schema provided.
   Do not add any fields not in the schema. Do not explain your reasoning."""

3. class BatchLabeler:
   def label_dataset(self, shard_dir: Path, output_dir: Path,
                     sample_rate: float = 1.0):
     - Load samples from WebDataset shards
     - Label each transcript (or sample_rate fraction for testing)
     - Write output as a parallel .jsonl file per shard
     - Human review sample: randomly flag 5% for review
       (write to review_queue.jsonl)
     - Progress: tqdm with samples/sec and estimated time remaining
     - On failure: log and continue (never abort entire batch for one failure)
     - Final report: total labeled, failed, flagged for review

4. schema validation function:
   LABEL_SCHEMA = { ... }  ← jsonschema dict matching the schema above
   def validate_label(label: dict) -> bool

CLI: python scripts/run_labeling.py --shards /data/shards --output /data/labels
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
and what value it should be changed to. The source of truth is Biona Axon's
onnx_contract.hpp — Biona Lab must match it, not the other way around.
```

---

*End of Biona Claude CLI Implementation Prompts v1.0*
*Run prompts A-1 through A-12 in Axon, B-1 through B-8 in Lab.*
*Run the VALIDATION prompt after both tracks are complete.*
