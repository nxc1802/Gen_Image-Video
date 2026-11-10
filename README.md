# Factory v4 Lite — Personal AI Production Platform

Factory v4 Lite is a lightweight, inference-first Personal AI Production Platform delivering authentic model inference through a verified API and Studio Web Operations Console, backed by a unified benchmark suite on Kaggle Dual T4 hardware.

By default, Factory v4 Lite operates in a zero-external-dependency mode: in-memory job queuing, discrete Dual T4 GPU scheduling, streaming events, ephemeral scratch disk artifact delivery, and single API key Bearer authentication. For production and hybrid cloud environments, hardened extension modes are fully supported: S3/MinIO object storage, PostgreSQL/SQLite durable stores with Alembic schema migrations, and standardized JWT authentication.

> **Validation Status:** Architecture, security, contract, and benchmark test suites are automated and 100% green (925+ unit, contract, and integration tests passing). Models adhere strictly to capability truth: unverified or unavailable capabilities fail closed with explicit non-success statuses (`BACKEND_UNAVAILABLE`, `HARDWARE_BLOCKED`, or 503 HTTP status) instead of returning mock or synthetic outputs. Negative API contract test passes are strictly isolated from model inference pass rates, ensuring zero false certifications.

---

## Runtime Architecture

- **Control Plane (`factory_control_api:app`)** — Lightweight FastAPI Control Plane in `factory/apps/control-api/`; in-memory queue, ephemeral scratch storage, discrete Dual T4 scheduler.
- **Studio Web (`factory/apps/studio-web/`)** — Next.js 15 / React 19 Operations Console, API Explorer, Job Dashboard, Model Registry, and Streaming Inspector.
- **Services (`services/*`)**:
  - `services/text/` — Native vLLM OpenAI-compatible text serving (Qwen 2.5 Text).
  - `services/vlm/` — Native vLLM multimodal serving with SSRF containment (Qwen 2.5 VLM).
  - `services/decision/` — Native vLLM Structured Decisions API (Qwen 2.5 Decision).
  - `services/embedding/` — Native vLLM embedding and rerank workers (Qwen 2.5 Embedding).
  - `services/speech/` — Qwen3-ASR on vLLM + Qwen3-TTS on vLLM-Omni; zero fabricated inference.
  - `services/image/` and `services/video/` — Native ComfyUI / Diffusers workers with chunked streaming artifact delivery (Qwen-Image 2.1, Wan 2.2, MiniMax H3).
  - `services/music/` — Native ACE-Step 1.5 and YuE2 workers with ABC notation verification.
  - `services/enhancement/` — Enhancement suite; unbacked or offline backends fail closed with 503.
- **Manifests (`manifests/`)** — Authoritative model, service, capability, and hardware source of truth across all 11 foundation models and explicit fast-inference variants (12 model manifests total).
- **Unified Kaggle Deployment (`deployments/kaggle/`)** — Single reproducible bootstrap engine (`deployments.kaggle.bootstrap`).
- **Unified Benchmark Suite (`benchmarks/`)** — Multi-modal runner, test cases, and binary container validators (`benchmarks.runner`, `benchmarks.validators`).

---

## Safety and Correctness Invariants

Factory v4 Lite strictly adheres to:

```text
REAL BACKEND AVAILABLE   -> execute genuine inference with authentic weights
REAL BACKEND UNAVAILABLE -> 503 BACKEND_UNAVAILABLE / SERVICE_UNAVAILABLE
NEVER                    -> mock PNG containers, mock MP4 headers, pseudo embeddings,
                            canned transcripts, sine-wave TTS, or fabricated telemetry
```

1. **Zero External Infrastructure by Default**: Operates completely self-contained in Lite mode (in-memory queue/db, ephemeral scratch disk). Production extension modes (PostgreSQL/SQLite durable storage with Alembic migrations, S3 bucket storage, and JWT auth) can be enabled via configuration flags.
2. **Authentication Security**: Constant-time verification using `FACTORY_API_KEY` via `Authorization: Bearer <key>`, or standardized JWT tokens (HS256/RS256) with audience and issuer verification.
3. **Artifact Delivery & Scratch Retention**: Generated media resides in the scratch directory (`SCRATCH_ARTIFACTS_DIR`, default `/tmp/factory_lite/artifacts/{job_id}/`) and is retained until TTL expiration (`ARTIFACT_TTL_SECONDS=3600` by default), or persisted to S3 object storage when `STORAGE_BACKEND=s3`. Immediate unlinking on download can be enabled with `ARTIFACT_CLEANUP_ON_DOWNLOAD=true`.
4. **Discrete Dual-T4 GPU Memory Partitioning**:
   - 2x 16 GB Tesla T4 GPUs with **14.2 GB (14,540 MB)** max allocatable VRAM each (leaving >= 820 MB CUDA headroom).
   - Zero assumed 32 GB shared memory pool; workloads must fit single-GPU partitions:
     - **GPU 0**: Text, VLM, Decision, Image, Speech.
     - **GPU 1**: Video, Music, Embedding, Enhancement.
   - Strict 1-job-per-GPU concurrency mutual exclusion lock.

---

## Status Semantics

The platform and benchmark suite enforce authentic, fail-closed status semantics:

| Status | Meaning |
|---|---|
| `PASS` | Authentic model inference verified against genuine checkpoint weights, valid artifact container (MIME, dimensions, duration, sha256 checksum), and sampled GPU VRAM telemetry. |
| `BACKEND_UNAVAILABLE` | Declared capability whose native backend engine or live worker is not actively running/reachable. No synthetic fallbacks, no mock passes. |
| `HARDWARE_BLOCKED` | Workload or model variant exceeds the 14.2 GB per-GPU allocatable ceiling on Dual-T4. |
| `FAIL` | Model execution completed but returned invalid artifact headers, empty bytes, or out-of-bounds parameters. |
| `ERROR` | Unexpected unhandled exception or system fault during execution. |
| Rejection Cases | Missing auth (401), invalid variant (400), unsupported capability (400), unregistered model (404), or schema violations (422) fail closed as expected. |

---

## Single Working Deploy Path

### 1. Launch Control Plane API

```bash
# Set environment variables
export FACTORY_API_KEY="sk-factory-prod-secret-token"

# Start FastAPI server via the canonical entrypoint
uv run uvicorn factory_control_api.server:app --host 0.0.0.0 --port 8000
# (or package entrypoint alias: uv run uvicorn factory_control_api:app --host 0.0.0.0 --port 8000)
```

### 2. Launch Studio Web Operations Console

Studio Web provides an interactive operations console, model registry browser, job dashboard, and real-time streaming inspector.

```bash
cd factory/apps/studio-web

# Development mode
npm run dev

# Production build and start
npm run build
npm run start
```

Studio Web is accessible at `http://localhost:3000` and communicates with the Control Plane at `http://localhost:8000` using the configured Bearer token.

---

## Unified Kaggle Dual-T4 Deployment & Benchmark

### Pre-Flight Bootstrap Verification

Run the unified bootstrap script to verify GPU topology, Python dependencies, dataset mounts, and scratch disk configuration without modifying files:

```bash
# Check-only dry run
python -m deployments.kaggle.bootstrap --check-only

# Or install missing dependencies automatically
python -m deployments.kaggle.bootstrap --install-missing
```

### Unified Multi-Modal Benchmark Suite

Execute the benchmark suite across all 11 model manifests directly via the unified runner:

```bash
# Smoke mode: quick feasibility and negative rejection evaluation
python -m benchmarks.runner --mode smoke

# Full mode: exhaustive evaluation across all 91 declared cases
python -m benchmarks.runner --mode full

# Perf mode: multi-iteration latency and VRAM profiling
python -m benchmarks.runner --mode perf

# Targeted evaluation by model
python -m benchmarks.runner --mode smoke --model qwen-text
```

Benchmark deliverables are generated in `benchmarks/results/run_<timestamp>/`:
- `results.json`: Canonical execution records and hardware profiles.
- `metrics.csv`: Tabular metric traces (latency, VRAM peak, artifact SHA-256).
- `summary.md`: Markdown summary report with status breakdown and capability matrix.

### Current Benchmark Verification & Model Truth Status

- **Benchmark Run**: `run_20261109_173256` (commit `6fbfd6bfc8d60d42357876884ecc15d2d4ec4fea`)
- **Total Evaluated Cases**: 91 cases across 11 foundation models
- **Negative Contract Rejection Cases**: 9 total, **9 pass (100.0% pass rate)**. All negative scenarios (`BM-NEG-01` to `BM-NEG-09`) fail closed cleanly with expected HTTP error codes (`400`, `401`, `404`, `422`).
- **Model Inference Cases**: 82 total, **0 pass (0.0% pass rate)** on local offline environment (`BACKEND_UNAVAILABLE: 82`, `HARDWARE_BLOCKED: 0`, `FAILED: 0`).
- **Hardware Status Transparency**: When evaluated on hosts without attached physical NVIDIA Tesla T4 GPUs (e.g. Darwin arm64 development host), jobs time out gracefully in queue (`QUEUE_TIMEOUT`) and are transparently recorded as `BACKEND_UNAVAILABLE` rather than fabricating simulated passes.
- **Zero False Certifications**: Contract rejection passes are strictly segregated from model inference metrics; no negative test latencies contaminate model inference performance statistics.

---

## Verification & Testing

```bash
# 1. Manifest Validation (all 11 models, 0 errors)
python -m model_registry.cli validate --path manifests

# 2. Benchmark Runner Unit Tests
pytest tests/unit/benchmarks/test_unified_benchmark_runner.py -v

# 3. Minimal Preflight Suite
pytest tests/test_preflight.py -v

# 4. Full Regression Test Suite
pytest tests/test_preflight.py tests/unit -q

# 5. Studio Web Build
cd factory/apps/studio-web && npm run build
```
