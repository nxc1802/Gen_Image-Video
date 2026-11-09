# Factory v4 Lite — Unified Kaggle Dual-T4 Benchmark

## 1. Executive Overview

This document specifies the unified benchmark and reproducibility architecture for Factory v4 Lite on the Kaggle Dual Tesla T4 GPU platform, per Master Plan §6 and §7.

Factory v4 Lite evaluates 100% of declared capabilities across all 11 foundation models using native inference engines and authentic weights. The suite produces verifiable multi-modal artifacts (PNG images, WAV audio, MP4 video, structured JSON, embedding vectors), sampled GPU VRAM telemetry time series, and canonical reports.

### Mandatory Integrity Contract

```text
REAL BACKEND AVAILABLE   -> execute authentic inference with real weights
REAL BACKEND UNAVAILABLE -> explicit BACKEND_UNAVAILABLE or HARDWARE_BLOCKED status
NEVER                    -> mock PNG containers, mock MP4 headers, pseudo embeddings,
                            canned transcripts, sine-wave TTS, or fabricated telemetry
```

Zero simulation-as-certification, zero fake fallbacks, zero synthetic passes.

---

## 2. Kaggle Dual-T4 Hardware Architecture & Memory Placement

The target hardware profile is a Kaggle Dual NVIDIA Tesla T4 environment consisting of two independent accelerators without a unified virtual address space:

- **GPU 0 Partition** (Allocatable ceiling: **14.2 GB / 14,540 MB**):
  - Workloads: `qwen-text`, `qwen-vlm`, `qwen-decision`, `qwen-image-2.1`, `qwen-speech`.
- **GPU 1 Partition** (Allocatable ceiling: **14.2 GB / 14,540 MB**):
  - Workloads: `wan-2.2`, `minimax-h3`, `ace-step-1.5`, `yue2`, `qwen-embedding`, `enhancement-suite-v1`.

### Invariance Rules

1. **Zero Shared 32 GB Assumption**: Each accelerator has an absolute physical ceiling of 15.0 GB (15,360 MB) and an allocatable ceiling of 14.2 GB (14,540 MB), leaving >= 820 MB for CUDA context, activations, and runtime allocations.
2. **Mutual Exclusion**: Exactly 1 active generative workload per GPU device at any given moment.
3. **Hardware Fail-Closed**: Any model variant or request requiring > 14.2 GB on a single GPU is marked `HARDWARE_BLOCKED` rather than producing synthetic passes.

---

## 3. Unified Kaggle Bootstrap Engine

Deployment and environment initialization are consolidated into a single reproducible bootstrap script:

```bash
# Verify environment without modifying files (dry run)
python -m deployments.kaggle.bootstrap --check-only

# Complete bootstrap with dependency installation and scratch initialization
python -m deployments.kaggle.bootstrap --install-missing
```

### Bootstrap Capabilities

- **Hardware Topology Detection**: Identifies host platform, GPU count, device names, and per-device VRAM bounds.
- **Kaggle Dataset Mounting**: Discovers weight directories under `/kaggle/input` and establishes symlinks into `/Volumes/WorkSpace/Factory/Gen_Image:Video/weights` or local runtime paths.
- **Dependency Audit**: Validates required packages (`fastapi`, `torch`, `pydantic`, `pyyaml`, `pynvml`, etc.).
- **Ephemeral Storage Setup**: Creates `/tmp/factory_lite/artifacts/` with verified write permissions.
- **Authentication Setup**: Configures `FACTORY_API_KEY` for secure local IPC.

---

## 4. Unified Multi-Modal Benchmark Suite

The unified benchmark runner (`benchmarks/runner.py`) executes declared test cases from `benchmarks/cases.yaml` using the FastAPI TestClient lifecycle or live HTTP endpoints.

### Execution Modes

1. **`smoke`**: Fast feasibility evaluation (positive representative cases + all 9 negative security/contract rejection cases).
2. **`full`**: Exhaustive evaluation across all 91 declared test cases (82 positive + 9 negative).
3. **`perf`**: Multi-iteration latency and peak VRAM profiling (median latency recorded over 3 iterations).

### CLI Commands

```bash
# Execute smoke suite
python -m benchmarks.runner --mode smoke

# Execute full 91-case suite
python -m benchmarks.runner --mode full

# Execute performance profiling
python -m benchmarks.runner --mode perf

# Targeted evaluation by model
python -m benchmarks.runner --mode smoke --model qwen-text

# Targeted evaluation by case ID
python -m benchmarks.runner --mode smoke --case-id BM-TXT-01
```

---

## 5. Status Semantics

Every benchmark test case yields one of the following explicit status values:

| Status | Definition |
|---|---|
| `PASS` | Authentic model inference succeeded: backend returned non-empty bytes, binary container was parsed and validated (magic header, dimensions, duration, checksum), and sampled NVML trace was recorded. Negative rejection test cases pass when the API correctly rejects invalid inputs with HTTP 400, 401, 404, or 422. |
| `BACKEND_UNAVAILABLE` | Declared capability whose native backend engine or live worker is not actively running/reachable. No synthetic fallbacks, no mock passes. |
| `HARDWARE_BLOCKED` | Workload or model variant exceeds the 14.2 GB per-GPU allocatable ceiling on Dual-T4, or execution attempted on a non-GPU host. |
| `FAIL` | Model execution completed but produced an invalid artifact (e.g. corrupted header, wrong dimensions, or out-of-tolerance duration). |
| `ERROR` | Unexpected unhandled exception or system fault during test execution. |
| `NOT_IMPLEMENTED` | Feature explicitly declared unbacked in model manifest. |
| `NOT_RUN` | Case was not scheduled for execution. |

---

## 6. Model Manifest Coverage (11 Models, 91 Total Cases)

The benchmark catalog evaluates **91 test cases** total: **82 positive capability cases** and **9 negative contract rejection cases**.

### Foundation Model Test Matrix

| Model ID | Service ID | Target GPU | Min VRAM | Declared Capabilities | Positive Cases |
|---|---|:---:|:---:|---|:---:|
| `qwen-text` | `text-service` | GPU 0 | 12.0 GB | Chat, generation, reasoning, creative, code | 5 |
| `qwen-vlm` | `vision-service` | GPU 0 | 14.0 GB | OCR, VQA, captioning, document analysis, grounding | 5 |
| `qwen-decision` | `decision-service` | GPU 0 | 6.0 GB | Tool calling, schema validation, JSON mode | 3 |
| `qwen-embedding` | `embed-service` | GPU 1 | 4.0 GB | Symmetric & asymmetric dense text embeddings | 2 |
| `qwen-speech` | `speech-service` | GPU 0 | 8.0 GB | Base TTS, voice cloning, multi-speaker | 3 |
| `qwen-image-2.1` | `image-service` | GPU 0 | 14.0 GB | Standard & Turbo T2I, I2I, edit, inpaint, controlnet | 12 |
| `wan-2.2` | `video-service` | GPU 1 | 14.0 GB | 720p T2V, I2V, cinematic, portrait | 4 |
| `minimax-h3` | `video-service` | GPU 1 | 14.0 GB | T2V, I2V, prompt extend, reference conditioning, high FPS | 5 |
| `ace-step-1.5` | `music-service` | GPU 1 | 6.0 GB | Full, vocal, instrumental, and 30 genre variations | 33 |
| `yue2` | `music-service` | GPU 1 | 12.0 GB | Dual-track, vocal, instrumental, pop, rock, ballad | 6 |
| `enhancement-suite-v1` | `enhancement-service` | GPU 1 | 6.0 GB | Upscale 4x, denoise, color grade, frame interpolation | 4 |
| **Total Positive Cases** | | | | | **82** |

### Negative Rejection Test Cases (Fail-Closed Verification)

| Case ID | Rejection Scenario | Expected Status | Contract / Invariant |
|---|---|:---:|---|
| `BM-NEG-01` | Invalid model variant name | HTTP 400 | Rejects undeclared variants fail-closed |
| `BM-NEG-02` | Unsupported capability request | HTTP 400 | Rejects unverified capabilities |
| `BM-NEG-03` | Unregistered model ID | HTTP 404 | Returns 404 for nonexistent models |
| `BM-NEG-04` | Image dimensions exceeding bounds | HTTP 400 | Enforces max resolution constraints |
| `BM-NEG-05` | Negative audio duration | HTTP 400 | Enforces valid positive durations |
| `BM-NEG-06` | Excessive video frame count | HTTP 400 | Enforces video length boundaries |
| `BM-NEG-07` | Missing or invalid API key | HTTP 401 | Enforces single Bearer token auth |
| `BM-NEG-08` | Empty prompt submission | HTTP 400 | Validates non-empty input payloads |
| `BM-NEG-09` | Malformed JSON payload body | HTTP 422 | FastAPI request body schema validation |

---

## 7. Artifact & Metric Verification Engine (`benchmarks/validators.py`)

All outputs are rigorously inspected by binary format parsers:

1. **PNG Image Validation**:
   - Magic bytes verification: `\x89PNG\r\n\x1a\n`.
   - `IHDR` chunk parsing: exact width, height, bit depth (8/16-bit), and color type.
   - Minimum byte threshold verification.
2. **WAV Audio Validation**:
   - `RIFF` + `WAVE` format header parsing.
   - `fmt ` chunk inspection: channel count (mono/stereo), sample rate (e.g. 24 kHz, 44.1 kHz), bit depth (16/24/32-bit).
   - `data` chunk length inspection and duration bounds checking (`min_duration <= duration <= max_duration`).
3. **MP4 Video Validation**:
   - Binary container inspection: `ftyp` box verification (`isom`, `mp42`, `qt  `).
   - `moov` / `mvhd` box timescale and duration parsing.
   - `tkhd` track box dimension parsing (width, height).
4. **Structured JSON & Decision Validation**:
   - Schema enforcement: required keys, allowed choices, score ranges, predicate booleans.
5. **Embedding Vector Validation**:
   - Dimension matching against model specification (e.g. 1024 or 1536).
   - Numerical type checking (disallowing booleans or non-numbers).
   - Sum of squares and unit L2 norm calculation with explicit `OverflowError`, `ValueError`, and `TypeError` protection.
   - Non-zero norm and finite float validation (rejecting NaN, Inf, and all-zero vectors).
6. **Symbolic Score Validation**:
   - YuE2 ABC notation header inspection (`X:`, `T:`, `M:`, `L:`, `K:`).
7. **Sampled NVML VRAM Telemetry**:
   - Background sampling thread via `pynvml` at 100 ms intervals across the execution window.
   - Peak VRAM tracking per discrete device (GPU 0 and GPU 1).
   - SHA-256 cryptographic hash computation for all generated artifacts.

---

## 8. Benchmark Deliverables Schema

Each benchmark execution creates a timestamped run directory in `benchmarks/results/run_<timestamp>/`:

- **`results.json`**: Complete execution traces including hardware environment, git commit, status breakdown, model matrix, and case-by-case execution records.
- **`metrics.csv`**: Tabular metric records: `case_id`, `model`, `service`, `task`, `variant`, `status`, `latency_ms`, `peak_vram_mb`, `artifact_bytes`, `sha256`.
- **`summary.md`**: Executive markdown report with status summary, model capability matrix, and artifact verification details.

---

## 9. Studio Web Operations Console Integration

The Studio Web console (`factory/apps/studio-web`) integrates directly with the Control Plane API:

- **Operations Console (`OperationsConsoleView.tsx`)**: Real-time overview of GPU memory allocation, active worker leases, and system health.
- **Model Registry (`ModelRegistryView.tsx`)**: Authoritative capability browser showing supported tasks, verified variants, and memory requirements.
- **Job Dashboard (`JobDashboardView.tsx`)**: Live job queue monitor, status updates, and artifact download links.
- **Streaming Inspector (`StreamingInspectorView.tsx`)**: Real-time SSE/WebSocket token and event stream inspector with explicit labeling distinguishing live telemetry from simulated demo streams.
- **API Explorer (`ApiExplorerView.tsx`)**: Interactive API playground and multi-language code generation (cURL, Python, TypeScript).

The Control Plane exposes the canonical application instance via `factory_control_api:app`:

```bash
uv run uvicorn factory_control_api:app --host 0.0.0.0 --port 8000
```
