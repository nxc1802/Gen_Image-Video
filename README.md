# Factory v4 Lite — Personal AI Production Platform

> **Upgrade authority:** [Master Plan](docs/MASTER_PLAN.md) governs architecture and phase order. The [Phase 0 truth baseline](validation/baseline_matrix.md) supersedes historical certification claims. **No model/capability pair is currently production-certified under the Master Plan's V3 + V4 + semantic gates.**

Factory is being refactored into an OpenAI-compatible, single-host inference platform targeting Kaggle Dual T4. Phases 0–7 passed their exit gates before the ongoing native integration refinements; current-source regressions are tracked in the upgrade status; all eight engines now select separate Manifest-owned adapters; the [upgrade status](docs/UPGRADE_STATUS.md) records current limits and reproduction commands. The [application API guide](docs/OPENAI_API.md) records the supported OpenAI SDK subset, tested with official Python SDK 3.49.0 over local HTTP. Positive SDK acceptance is software-only; production inference remains evidence-locked. Phase 8 Native V3 is in progress: Wave A passes 10/10 real semantic golden cases across Decision/Embedding/Rerank/Text; Wave B passes 8/8 real semantic cases across ASR/VLM/TTS/voice cloning; Wave C passes 8/8 Standard/Turbo image cases; Wave D Video is in progress (2/13 native golden pairs PASS); Waves E–F remain pending. No capability is production-ready before Full Factory E2E. The sections below describe the legacy implementation while migration proceeds through the Master Plan's exit gates.

Explicit deployment examples are in [deployments/examples](deployments/examples). Use `factory deployment init/add/show/validate` to configure model/backend/GPU placement. `factory serve` now starts only the selected providers after full preflight, checks current-session process/health/GPU identity and shuts down owned processes. The Phase 4 live gate verified pinned ASR load/health/shutdown on Kaggle Dual-T4. Phase 5 also ran an isolated exact-checkpoint RealESRGAN x4 backend probe on GPU0 (512×512 photo → 2048×2048 PNG); Enhancement remains DISABLED without V3/semantic/Full Factory E2E certification. Healthy backends alone do not authorize public inference. `/v1/models` lists active session models only; `/v1/registry/models` is the separate catalog.

The current core uses local asynchronous jobs, fixed deployment GPU bindings, one scratch artifact store and a single Factory API key. It has no Redis/Postgres/S3 or worker/lease requirement. The [Control Plane guide](docs/CONTROL_PLANE.md) explains cancellation and migration boundaries; legacy infrastructure integrations are isolated from production startup and retained for explicit migration tests.

> **Validation Status:** Software tests verify contracts and implementation behavior. They do not certify model inference. Historical speech runners contain synthetic fallback paths; their outputs and GPU allocations are excluded from certification. Static legacy readiness flags do not attest to the current deployment. See [baseline.json](validation/baseline.json) for the frozen claims and audit decisions.

---

## Runtime Architecture

- **Control Plane (`factory_control_api:app`)** — Single-host FastAPI composition with canonical Engine dispatch, local async jobs and scratch artifacts; worker leases and database/storage gateways are absent from default startup.
- **Studio Web (`factory/apps/studio-web/`)** — Next.js 15 / React 19 Operations Console, API Explorer, Job Dashboard, Model Registry, and Streaming Inspector.
- **Services (`services/*`)**:
  - `services/text/` — Legacy vLLM launcher, currently blocked for the manifest-selected Qwen3.8 GGUF/llama.cpp backend pending lifecycle migration.
  - `services/vlm/` — Legacy vLLM multimodal service with SSRF containment; canonical model identity is Qwen2-VL-7B AWQ.
  - `services/decision/` — Canonical DecisionEngine with a manifest-selected adapter; software-only validation so far, without production authorization.
  - `services/embedding/` — Legacy vector workers; canonical Embedding and Reranker have separate pinned Qwen3 identities.
  - `services/speech/` — Separate pinned Qwen3-ASR and Qwen3-TTS models; the legacy launcher refuses incompatible backend selections pending migration.
  - `services/image/` and `services/video/` — Native ComfyUI / Diffusers workers with chunked streaming artifact delivery (Qwen-Image 2.1, Wan 2.2, MiniMax H3).
  - `services/music/` — Legacy music workers retained for migration. ACE-Step bundles have separate IDs; YuE2 is blocked on the selected BF16/T4 runtime.
  - `services/enhancement/` — Enhancement suite; unbacked or offline backends fail closed with 503.
- **Manifests (`manifests/`)** — Canonical catalog of 20 pinned model identities; model aliases, checkpoint files and requirements come from ModelManifest. Inspect with `factory models inspect MODEL`.
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

1. **Local Core Runtime**: In-memory jobs and local scratch artifacts require no database, broker or object store. Old infrastructure settings belong to retained compatibility code; core startup does not initialize those systems.
2. **Authentication**: Constant-time verification using `FACTORY_API_KEY` via `Authorization: Bearer <key>`. JWT remains an explicit optional extra; worker/RBAC gateways are outside core composition.
3. **Artifact Lifecycle**: Core generated media uses `SCRATCH_ARTIFACTS_DIR` and TTL/quota sweeping. Downloads require owned completed jobs and matching stored hashes; cancelled or incomplete jobs publish no content.
4. **Explicit Dual-T4 GPU Placement**:
   - Deployment YAML assigns devices to each model before serving; subsystem names never select a GPU.
   - Preflight checks each device's declared VRAM budget independently (14.2 GB allocatable per T4).
   - Multiple resident models may share one GPU when preflight permits; execution locks remain per GPU.
   - A dual-GPU entry uses explicit devices `[0, 1]` and exclusive access. Runtime load/VRAM observations remain distinct from Native V3/semantic/Full Factory E2E authorization.

---

## Status Semantics

The platform and benchmark suite enforce authentic, fail-closed status semantics:

| Status | Meaning |
|---|---|
| `PASS` | A named gate passed within its recorded scope; software/API/lifecycle PASS does not certify neural inference or production readiness. Production capability authorization requires exact-checkpoint Native V3, semantic validation and Full Factory E2E. |
| `BACKEND_UNAVAILABLE` | Declared capability whose native backend engine or live worker is not actively running/reachable. No synthetic fallbacks, no mock passes. |
| `HARDWARE_BLOCKED` | Workload or model variant exceeds the 14.2 GB per-GPU allocatable ceiling on Dual-T4. |
| `FAIL` | Model execution completed but returned invalid artifact headers, empty bytes, or out-of-bounds parameters. |
| `ERROR` | Unexpected unhandled exception or system fault during execution. |
| Rejection Cases | Missing auth (401), invalid variant (400), unsupported capability (400), unregistered model (404), or schema violations (422) fail closed as expected. |

---

## Retained deployment and Studio notes

The following sections describe the pre-refactor interfaces. Use `factory deployment ...` and `factory serve` for the canonical runtime; the old generic job/worker gateways are retained only in the non-production migration host. Studio consumption moves to the new API in Phase 10.

### Historical launch path

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
