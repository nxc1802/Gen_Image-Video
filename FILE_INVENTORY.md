# Factory v4 Lite — Repository File & Dependency Inventory

**Milestone:** Phase 0 (Freeze, Inventory, and Baseline Recovery)  
**Date:** 2026-11-08  
**Baseline Git Commit:** `145a16f218dbe828032c74ae8a8293dbf5946219`  
**Baseline Recovery Tag:** `baseline-v4-pre-simplification`  
**Baseline Recovery Branch:** `recovery/baseline-145a16f`  
**Plan Adoption Commit:** `d2cfbb3` (`chore(baseline): adopt simplification and kaggle benchmark plan with recovery tag`)  
**Authoritative Plan:** `Factory_v4_Lite_Simplification_Correctness_Kaggle_Benchmark_Plan.md`  

---

## 1. Executive Summary & Baseline Reference

Before executing any file or code deletions in Phase 1 (Aggressive repository and runtime simplification), this inventory establishes an exhaustive classification of all files, directories, dependencies, and historical benchmark artifacts across the repository.

### 1.1 Baseline Anchors
- **Baseline Commit**: `145a16f218dbe828032c74ae8a8293dbf5946219`
- **Immutable Tag**: `baseline-v4-pre-simplification` (verified via `git rev-parse baseline-v4-pre-simplification`)
- **Recovery Branch**: `recovery/baseline-145a16f` (verified via `git rev-parse recovery/baseline-145a16f`)
- **Purpose**: Guarantees that any legacy code, historical benchmark evidence, or omitted tests can be recovered with zero loss of provenance at any time via `git checkout baseline-v4-pre-simplification`.

---

## 2. Four-Tier Classification Taxonomy

Every file and directory in the repository is assigned exactly one of four operational classifications:

1. **`REQUIRED_RUNTIME`**: Essential for local personal AI platform runtime, the FastAPI control plane, Studio Web UI, model registry, service adapters, workflows, contracts, and core configuration. Must NOT be deleted or moved.
2. **`REQUIRED_BENCHMARK`**: Essential for Kaggle Dual T4 benchmark execution, including benchmark configurations (`evaluation_cases.yaml`), benchmark harness (`live_runner.py`), validators, report generation, and Kaggle API communication (`runner.py`).
3. **`REDUNDANT`**: Obsolete, duplicate, or historical material eligible for immediate deletion in Phase 1 without impacting personal production or benchmark operations.
4. **`CONDITIONAL`**: Files that have external or cross-subsystem references requiring explicit prerequisite steps (such as test adjustments, documentation updates, or archive backups) before removal.

---

## 3. Master Directory & File Inventory

### 3.1 Repository Root Files

| Path | Size (Bytes) | Category | Description & Justification | Action in Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| `Factory_v4_Lite_Simplification_Correctness_Kaggle_Benchmark_Plan.md` | ~23,188 | `REQUIRED_RUNTIME` | Authoritative 5-phase execution plan for simplification, correctness, and Kaggle benchmark. | Retain as primary plan |
| `README.md` | ~4,200 | `REQUIRED_RUNTIME` | Canonical system documentation, architecture overview, quickstart. | Retain; remediate C4 launch entrypoint |
| `.env.example` | ~2,100 | `REQUIRED_RUNTIME` | Environment configuration template for server, auth, and storage. | Retain; remediate C3 `STORAGE_BACKEND` |
| `pyproject.toml` | ~3,800 | `REQUIRED_RUNTIME` | Workspace package configuration, dependencies, and build settings. | Retain; remove legacy path exclusions in Phase 1 |
| `uv.lock` | ~180,000 | `REQUIRED_RUNTIME` | Resolved dependency lockfile ensuring reproducible Python runtime. | Retain |
| `pytest.ini` | ~1,200 | `REQUIRED_RUNTIME` | Test runner configuration, markers, and execution options. | Retain; update test discovery paths in Phase 1 |
| `.gitignore` | ~1,500 | `REQUIRED_RUNTIME` | Git ignore rules for virtual environments, build outputs, and `.agents`. | Retain; clean legacy ignore patterns |
| `Refactor.md` | 29,244 | `REDUNDANT` | Superseded bilingual (Vietnamese/English) initial specification notes from v3.5. | Delete |
| `PROJECT.md` | 14,891 | `REDUNDANT` | Historical enterprise multi-tier specification (Postgres, Redis, MinIO, Celery). | Delete |
| `ORIGINAL_REQUEST.md` (root) | 12,934 | `REDUNDANT` | Historical 2026-10-23 refactor request. (Current authoritative request is preserved in `.agents/teamwork/ORIGINAL_REQUEST.md`). | Delete |
| `TEST_INFRA.md` | 14,009 | `REDUNDANT` | Historical 4-tier opaque-box test framework specification. | Delete |
| `TEST_READY.md` | 7,524 | `REDUNDANT` | Stale E2E test matrix documentation for earlier milestone. | Delete |
| `LIVE_VALIDATION.md` | 6,207 | `CONDITIONAL` | Superseded live validation guidelines. **Safeguard:** Referenced by 6 model manifests (`qwen-vlm`, `wan-2.2`, `qwen-decision`, `qwen-text`, `qwen-image-2.1`, `qwen-embedding`). | Update 6 model manifests to reference `BENCHMARK.md`, then delete |
| `bug_audit.md` | 22,786 | `REDUNDANT` | Historical audit record for FIX-01 through FIX-20. | Delete |
| `benchmark_acceptance_report.md` | 34,563 | `REDUNDANT` | Stale Phase 6 163-case acceptance report from `p5_matrix_live_04`. | Delete (preserved in baseline tag & `benchmarks/reports/`) |

---

### 3.2 CI/CD Workflows (`.github/`)

| Path | Size (Bytes) | Category | Description & Justification | Action in Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| `.github/workflows/ci.yml` | 4,128 | `REDUNDANT` | GitHub Actions workflow for Next.js build, uv compileall, and pytest matrix. Irrelevant for personal production. | Delete |
| `.github/workflows/security.yml` | 5,649 | `CONDITIONAL` | Security auditing workflow. **BREAKING SAFEGUARD:** Explicitly read by `tests/unit/services/test_p0_fixes_14_to_20.py:164`. | Adjust/retire test assertion before deleting |
| `.github/dependabot.yml` | 1,002 | `REDUNDANT` | Dependabot automated dependency update configuration. | Delete |
| `.github/CODEOWNERS` | 904 | `REDUNDANT` | Repository code ownership assignments. | Delete |

---

### 3.3 Legacy Subsystem (`legacy/`)

- **Total Scope:** 645 files, **205 MB**.
- **Isolation Status:** Verified 100% decoupled from active codebase via `tests/security/test_no_legacy_imports.py` (passes 2/2 tests).
- **Classification:** `REDUNDANT` (all preserved under `baseline-v4-pre-simplification`).

| Subtree | Files | Size | Content Summary | Action in Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| `legacy/v3/ui/` | ~310 | ~25 MB | Old Next.js v3.5 frontend (`package.json`, `src/`, `tailwind.config.ts`). Superseded by `factory/apps/studio-web`. | Delete |
| `legacy/v3/all-in-one/` | ~120 | ~110 MB | Monolithic server (`main.py`, `app.py`, `schemas.py`), Colab/Kaggle notebooks (`run_downloader.ipynb`, `run_wheels_builder.ipynb`, `run_speech.ipynb`, `run_vlm.ipynb`), audio/vlm/visual engines. | Delete |
| `legacy/v3/artifacts/` | 32 | ~60 MB | Obsolete test PNG, MP4, and WAV media artifacts and benchmark outputs. | Delete |
| `legacy/v3/pipelines/` | 7 | ~2 MB | Legacy pipeline chain scripts (`run_dragon_pipeline.py`, `auto_run_chain.py`, etc.). | Delete |
| `legacy/v3/benchmarks/` | 6 | ~1.5 MB | Historical benchmark scripts (`test_ref2va_turbo.py`, etc.). | Delete |
| `legacy/v3/scripts/` | 7 | ~1 MB | Standalone scene generation scripts. | Delete |
| `legacy/v3/docs/` | 6 | ~500 KB | Historical documentation (`BENCHMARK_VISUAL_REPORT.md`, `USAGE_GUIDE.md`). | Delete |
| `legacy/v3/.env.archived` | 1 | ~5 KB | Archived environment variables for v3. | Delete |

---

### 3.4 Documentation & Infrastructure

| Path | Size | Category | Description | Action in Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| `docs/operations/DISASTER_RECOVERY.md` | ~6 KB | `REDUNDANT` | Enterprise disaster recovery procedure. | Delete |
| `docs/operations/ROLLBACK_SOP.md` | ~5 KB | `REDUNDANT` | Enterprise rollback standard operating procedure. | Delete |
| `docs/operations/BRANCH_PROTECTION.md` | ~4 KB | `REDUNDANT` | GitHub branch protection rules guide. | Delete |
| `docs/api/`, `docs/architecture/`, `docs/models/` | 0 | `REDUNDANT` | Empty placeholder directories. | Prune |
| `infrastructure/ngrok/ngrok.yml` | ~2 KB | `REDUNDANT` | Ngrok tunnel definition for multi-service enterprise setup. | Delete |
| `infrastructure/ngrok/traffic-policy.yml` | ~3 KB | `REDUNDANT` | Ngrok traffic policies. | Delete |
| `infrastructure/ngrok/traffic-policy-raw.yml` | ~3 KB | `REDUNDANT` | Raw traffic policy template. | Delete |

---

### 3.5 Core Application & Packages (`factory/` & `packages/`)

| Path | Category | Description & Justification |
| :--- | :--- | :--- |
| `factory/apps/control-api/` | `REQUIRED_RUNTIME` | Core FastAPI Control Plane application (`src/main.py`, `src/api/routers/`, `src/queue/`, `src/storage/`, `src/assets/`). Implements in-memory GPU scheduling, job queue, scratch disk storage, and REST API. |
| `factory/apps/studio-web/` | `REQUIRED_RUNTIME` | Next.js 15 personal production UI (`src/app/`, `src/components/`, `src/context/`, `src/hooks/`). Full typecheck and build pass cleanly. |
| `packages/contracts/` | `REQUIRED_RUNTIME` | Authoritative Pydantic schemas, manifest models (`ModelManifest`, `ServiceManifest`, `HardwareTopologyManifest`), and API request/response contracts. |
| `packages/auth/` | `REQUIRED_RUNTIME` | Single bearer token validation (`FACTORY_API_KEY`), dynamic credential resolution, and sanitized exceptions. |
| `packages/model-registry/` | `REQUIRED_RUNTIME` | YAML manifest loader, strict schema validator, and CLI engine (`factory-manifest validate`). |
| `packages/client-sdk/` | `REQUIRED_RUNTIME` | Python client SDK for Factory v4 Lite API interaction; lightweight TypeScript types. |

---

### 3.6 Services & Workflows (`services/` & `workflows/`)

| Path | Category | Description & Justification | Action in Phase 1 |
| :--- | :--- | :--- | :--- |
| `services/text/` | `REQUIRED_RUNTIME` | Text generation adapter and vLLM launcher for `qwen-text`. | Retain |
| `services/vision/` | `REQUIRED_RUNTIME` | Vision-language adapter and vLLM launcher for `qwen-vlm`. | Retain |
| `services/decision/` | `REQUIRED_RUNTIME` | Structured routing and predicate evaluation adapter for `qwen-decision`. | Retain |
| `services/embedding/` | `REQUIRED_RUNTIME` | Dense vector embedding and reranking adapter for `qwen-embedding`. | Retain |
| `services/speech/` | `REQUIRED_RUNTIME` | Audio speech-to-text and text-to-speech adapter for `qwen-speech`. | Retain |
| `services/image/` | `REQUIRED_RUNTIME` | ComfyUI adapter and client for `qwen-image-2.1`. | Retain |
| `services/video/` | `REQUIRED_RUNTIME` | ComfyUI adapter and client for `wan-2.2` and `minimax-h3`. | Retain |
| `services/music/` | `REQUIRED_RUNTIME` | Native ACE-Step and YuE2 client adapters for `ace-step-1.5` and `yue2`. | Retain |
| `services/enhancement/` | `REQUIRED_RUNTIME` | Post-processing enhancement adapter (declares explicit `BACKEND_UNAVAILABLE`). | Retain |
| `services/jobs/` | `REQUIRED_RUNTIME` | Job gateway service adapter. | Retain |
| `services/rerank/` | `REQUIRED_RUNTIME` | Dedicated rerank adapter. | Retain |
| `services/*/README.md` (7 files) | `REDUNDANT` | 0-byte placeholder README files. | Delete |
| `workflows/image/` | `REQUIRED_RUNTIME` | ComfyUI workflow JSON graphs and execution chains for image generation. | Retain |
| `workflows/video/` | `REQUIRED_RUNTIME` | ComfyUI workflow JSON graphs and execution chains for video generation. | Retain |
| `workflows/*/README.md` (2 files) | `REDUNDANT` | 0-byte placeholder README files. | Delete |

---

### 3.7 Model & Hardware Manifests (`manifests/`)

All manifests are `REQUIRED_RUNTIME` and validate with 0 errors via `.venv/bin/python -m model_registry.cli validate --path manifests`.

- **Model Manifests (11 files in `manifests/models/`)**:
  1. `qwen-text.yaml`: `qwen-text` (Qwen2.5-7B-Instruct-AWQ, vLLM, GPU 0)
  2. `qwen-vlm.yaml`: `qwen-vlm` (Qwen2-VL-7B-Instruct-AWQ, vLLM, GPU 0)
  3. `qwen-decision.yaml`: `qwen-decision` (Qwen2.5-3B-Instruct, vLLM, GPU 0)
  4. `qwen-embedding.yaml`: `qwen-embedding` (Qwen3-Embedding-0.6B + Reranker, GPU 1)
  5. `qwen-speech.yaml`: `qwen-speech` (Qwen3-ASR-1.7B + TTS, Dual-GPU Split)
  6. `qwen-image-2.1.yaml`: `qwen-image-2.1` (Q4_K_M GGUF, ComfyUI, GPU 0)
  7. `wan-2.2.yaml`: `wan-2.2` (Wan2.2-TI2V-5B-Diffusers 4-bit, ComfyUI, GPU 1)
  8. `minimax-h3.yaml`: `minimax-h3` (Hailuo-H3-Diffusers 4-bit, ComfyUI, GPU 1)
  9. `ace-step-1.5.yaml`: `ace-step-1.5` (ACE-Step-1.5 2B/XL, Native API, GPU 1)
  10. `yue2.yaml`: `yue2` (YuE-s1-7B-anneal-en-cot, Native API, GPU 1)
  11. `enhancement-suite-v1.yaml`: `enhancement-suite-v1` (Real-ESRGAN/RIFE, GPU 1, 0% coverage, explicit unavailable)
- **Service Manifests (11 files in `manifests/services/`)**: `decision.yaml`, `embedding.yaml`, `enhancement.yaml`, `image.yaml`, `jobs.yaml`, `music.yaml`, `rerank.yaml`, `speech.yaml`, `text.yaml`, `video.yaml`, `vision.yaml`.
- **Hardware Topologies (3 files in `manifests/hardware/`)**: `kaggle-dual-t4.yaml` (canonical target), `colab-single-t4.yaml`, `local-apple-silicon.yaml`.
- **Capabilities (12 files in `manifests/capabilities/`)**: Complete set of capability schemas.

---

### 3.8 Deployments (`deployments/`)

| Path | Size / Count | Category | Description & Justification | Action in Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| `deployments/kaggle/runner.py` | 17,432 bytes | `REQUIRED_BENCHMARK` | Kaggle REST API v1 client (`KaggleAPIClient`), kernel push, log streaming, and artifact retrieval. | Retain; integrate with bootstrap |
| `deployments/kaggle/prepare_datasets.py` | 18,506 bytes | `REDUNDANT` | Automated kernel script generator for downloading weights. | Consolidate into `bootstrap.py` |
| `deployments/kaggle/certify_kernel.py` | 78,917 bytes | `REDUNDANT` | Monolithic dual-T4 in-kernel certification script. | Superseded by unified benchmark runner |
| `deployments/kaggle/session1_visual_pipeline.py` | 78,913 bytes | `REDUNDANT` | Session 1 runner (Visual stack). | Consolidate into `bootstrap.py` |
| `deployments/kaggle/session2_audio_pipeline.py` | 25,075 bytes | `REDUNDANT` | Session 2 runner (Audio stack). | Consolidate into `bootstrap.py` |
| `deployments/kaggle/monitor_both_sessions.py` | 8,495 bytes | `REDUNDANT` | Multi-session concurrent orchestrator. | Consolidate into `bootstrap.py` |
| `deployments/kaggle/smoke_runner.py` | 8,775 bytes | `REDUNDANT` | In-kernel smoke test script. | Consolidate into unified `benchmarks/runner.py` |
| `deployments/kaggle/live_gpu_test.py` | 19,749 bytes | `REDUNDANT` | Standalone GPU harness (contains C10 synthetic tensor tests). | Delete (replaced by genuine benchmark runner) |
| `deployments/kaggle/run_high_quality_test.py` | 13,742 bytes | `REDUNDANT` | Standalone diffusion test. | Consolidate / Delete |
| `deployments/kaggle/run_studio_quality_pipeline.py` | 19,189 bytes | `REDUNDANT` | Standalone multi-model test. | Consolidate / Delete |
| `deployments/kaggle/audio_outputs/` | **319.52 MB** (14 files) | `REDUNDANT` | Stale audio artifacts (`vieneu_v3_turbo_vietnamese.wav`, `qwen3_tts_english.wav`, etc.). | Delete from git tracking (preserved in baseline tag) |
| `deployments/kaggle/visual_outputs/` | **4.63 MB** (9 files) | `REDUNDANT` | Stale video and image artifacts. | Delete from git tracking (preserved in baseline tag) |
| `deployments/kaggle/high_quality_outputs/` | **1.07 MB** (6 files) | `REDUNDANT` | Stale video and audio artifacts. | Delete from git tracking (preserved in baseline tag) |
| `deployments/kaggle/outputs/` | **0.05 MB** (4 files) | `REDUNDANT` | Stale certification outputs. | Delete from git tracking (preserved in baseline tag) |
| `deployments/kaggle/certification_results.json` | 45,794 bytes | `REDUNDANT` | Stale Kaggle certification dump. | Retain as historical evidence in baseline tag |
| `deployments/kaggle/live_hardware_attested_report.json` | 4,240 bytes | `REDUNDANT` | Stale attestation report containing C10 synthetic pass. | Retain as historical evidence in baseline tag |
| `deployments/local/`, `deployments/colab/` | 0 | `REDUNDANT` | Empty placeholder directories. | Prune |

---

### 3.9 Test Suites (`tests/`)

Total current scope: **106 test files**, **1,559 test functions**.

| Subtree | Files | Tests | Category | Role & Action in Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| `tests/e2e/test_factory_lite_e2e.py` | 1 | 26 | `REQUIRED_RUNTIME` | **Primary Preflight Suite**: Tests in-memory FastAPI app, auth, job submission, queue, SSE progress, and scratch disk. Verified 26/26 passed in 1.10s. |
| `tests/security/` | 6 | 46 | `REQUIRED_RUNTIME` | Security invariants: Bearer auth enforcement, error sanitization, legacy import isolation (`test_no_legacy_imports.py`), SSRF protection, proxy elimination. Retain. |
| `tests/contract/` | 2 | 8 | `REQUIRED_RUNTIME` | Schema compliance and manifest contract validation. Retain. |
| `tests/unit/` | 39 | 469 | `CONDITIONAL` | Unit tests for storage, routers, queue, watchdog. Selected critical assertions will be consolidated into the 10-15 scenario preflight suite. Note: `test_p0_fixes_14_to_20.py` has the `security.yml` safeguard. |
| `tests/adversarial/` | 30 | 657 | `REDUNDANT` | Historical milestone challenger suites (M1–M5, P0–P6) with massive parameterization (e.g., 222 items in acestep, 237 in yue2). Contains C9 hardcoded Mac path in `test_adversarial_p4_yue2.py:974`. Safe to prune in Phase 1. |
| `tests/e2e/tier1_feature_coverage/` | 15 | 195 | `REDUNDANT` | Historical 4-tier opaque test suite. Superseded by `test_factory_lite_e2e.py`. Safe to prune. |
| `tests/e2e/tier2_boundaries_corners/` | 9 | 103 | `REDUNDANT` | Historical boundary test suite. Superseded by preflight suite. Safe to prune. |
| `tests/e2e/tier3_cross_feature/` | 2 | 25 | `REDUNDANT` | Historical cross-feature test suite. Safe to prune. |
| `tests/e2e/tier4_real_world_scenarios/` | 2 | 30 | `REDUNDANT` | Historical scenario test suite. Safe to prune. |
| `tests/integration/`, `tests/load/` | 0 | 0 | `REDUNDANT` | Empty directories. Prune in Phase 1. |

---

## 4. Historical Benchmark Evidence Inventory & Provenance

Full provenance of all historical benchmark executions prior to simplification is documented below:

### 4.1 Historical Reports (`benchmarks/reports/`)

| Report File | Size (Bytes) | Summary of Historical Evidence | Provenance Notes |
| :--- | :--- | :--- | :--- |
| `final_report.md` | 34,563 | Master 163-case acceptance report (`p5_matrix_live_04`). Certified 51 PASS cases, 80 HARDWARE_BLOCKED, 32 NOT_IMPLEMENTED. | Documented C10 issue: certified PASS cases included synthetic tensor execution. Preserved for provenance. |
| `capability_matrix.md` | 29,677 | Matrix of declared vs verified capabilities across all 11 foundation models and 163 evaluation cases. | Establishes the historical capability baseline. |
| `model_comparison.md` | 13,456 | Cross-model comparative analysis (latency, memory footprint, output quality across visual and audio models). | Retain for historical performance reference. |
| `failure_analysis.md` | 11,130 | Detailed breakdown of the 80 hardware-blocked cases (Dual T4 VRAM limits) and 32 not-implemented cases. | Crucial reference for Phase 3 blocking statuses. |
| `latency_vram.md` | 6,141 | Latency and peak VRAM breakdown table across all active foundation models. | Historical benchmark telemetry. |
| `executive_summary.md` | 5,442 | Executive overview of Phase 5 benchmark results and hardware feasibility conclusions. | High-level summary of pre-simplification status. |

### 4.2 Historical Run Results (`benchmarks/results/`)

- **Active Runs with Artifacts**:
  1. `benchmarks/results/p5_matrix_live_01/`: Contains `failures.json`, `metadata.json`, `metrics.csv`, `results.jsonl` (Run timestamp: 2026-11-08T18:41:20Z, 163 cases).
  2. `benchmarks/results/p5_matrix_live_02/`: Contains `failures.json`, `metadata.json`, `metrics.csv`, `results.jsonl` (Run timestamp: 2026-11-08T18:55:04Z, 163 cases).
  3. `benchmarks/results/p5_matrix_live_03/`: Contains `failures.json`, `metadata.json`, `metrics.csv`, `results.jsonl` (Run timestamp: 2026-11-08T19:08:12Z, 163 cases).
  4. `benchmarks/results/p5_matrix_live_04/`: Contains `failures.json`, `metadata.json`, `metrics.csv`, `results.jsonl` (Run timestamp: 2026-11-08T19:28:45Z, final certified run, 163 cases).
- **Empty / Incomplete Run Directories (Eligible for cleanup)**:
  `audit_test_r21/`, `run_20261108_183829/`, `run_20261108_183840/`, `run_20261108_183903/`, `run_20261108_185257/`, `run_20261108_185258/`, `run_20261108_185617/`, `run_20261108_185628/`, `run_20261108_185737/`, `run_20261108_185810/`, `run_20261108_185828/`, `run_20261108_185849/`, `run_20261108_185952/`, `run_20261108_190501/`, `run_20261108_191521/`, `run_20261108_192357/`, `run_20261108_192432/`, `run_20261108_193311/`.

---

## 5. Dependency Trace Safeguards & Breaking Change Analysis

Before Phase 1 deletions proceed, the following 6 specific dependency traces MUST be addressed:

### 5.1 Safeguard 1: Test Dependency on `.github/workflows/security.yml`
- **Location:** `tests/unit/services/test_p0_fixes_14_to_20.py` line 164
- **Code:**
  ```python
  ci_security = Path(".github/workflows/security.yml").read_text(encoding="utf-8")
  assert "uv audit ||" not in ci_security
  assert "npm audit --audit-level=critical" in ci_security
  ```
- **Hazard:** Directly deleting `.github/workflows/security.yml` causes `FileNotFoundError` during pytest.
- **Remediation:** Remove or retire `test_p0_fixes_14_to_20.py` (which is part of the historical unit test suite) concurrently when deleting `.github/workflows/`.

### 5.2 Safeguard 2: Model Manifest Citations of `LIVE_VALIDATION.md`
- **Location:** 6 model manifests:
  - `manifests/models/qwen-text.yaml:33`
  - `manifests/models/qwen-vlm.yaml:33`
  - `manifests/models/qwen-decision.yaml:31`
  - `manifests/models/qwen-embedding.yaml:34`
  - `manifests/models/qwen-image-2.1.yaml:55`
  - `manifests/models/wan-2.2.yaml:35`
- **Hazard:** Deleting `LIVE_VALIDATION.md` leaves orphaned documentation pointers in model manifests.
- **Remediation:** Update `validation_notes` in these 6 YAML manifests to cite `BENCHMARK.md Section 4` prior to deleting `LIVE_VALIDATION.md`.

### 5.3 Safeguard 3: Legacy Subsystem Isolation
- **Verification:** Verified via `.venv/bin/pytest tests/security/test_no_legacy_imports.py -v`.
- **Finding:** No active module in `factory/`, `packages/`, `services/`, `workflows/`, or `infrastructure/` imports from `legacy/` or `kaggle/`.
- **Remediation:** `legacy/` can be deleted cleanly from the working branch. When deleting, remove `"legacy"` exclusions from `pyproject.toml`, `pytest.ini`, and `.gitignore`.

### 5.4 Safeguard 4: Storage Configuration Scheme (C3)
- **Location:** `.env.example:56` (`STORAGE_BACKEND=ephemeral_disk`) vs `factory/apps/control-api/src/storage/config.py:46` (`{"local", "s3"}`).
- **Hazard:** Launching with `.env.example` crashes with `RuntimeError: Unsupported STORAGE_BACKEND 'ephemeral_disk'`.
- **Remediation:** Update `config.py` to accept `ephemeral_disk` as an alias for `local`, or update `.env.example` to `STORAGE_BACKEND=local`.

### 5.5 Safeguard 5: Launch Entrypoint Inconsistency (C4)
- **Location:** `README.md:49` specifies `uv run uvicorn factory_control_api.server:app`. No `server.py` exists in `src/factory_control_api/`.
- **Hazard:** Produces `ModuleNotFoundError: No module named 'factory_control_api.server'`.
- **Remediation:** Add `server.py` in `factory/apps/control-api/src/factory_control_api/` re-exporting `app`, and update `README.md` to reference `factory_control_api:app`.

### 5.6 Safeguard 6: Hardcoded Local Mac Filesystem Paths (C9)
- **Location:** `tests/adversarial/test_adversarial_p4_yue2.py:974` (`Path("/Volumes/WorkSpace/Factory/Gen_Image:Video/manifests/models/yue2.yaml")`).
- **Hazard:** Fails immediately on Linux or Kaggle cloud environments.
- **Remediation:** When pruning historical adversarial tests in Phase 1, ensure any retained test uses `Path(__file__).resolve().parents[...]`.

---

## 6. Actionable Roadmap for Phase 1 Simplification

1. **Step 1: Manifest Citation Decoupling**
   - Update 6 model manifests (`qwen-text`, `qwen-vlm`, `qwen-decision`, `qwen-embedding`, `qwen-image-2.1`, `wan-2.2`) to reference `BENCHMARK.md` instead of `LIVE_VALIDATION.md`.
2. **Step 2: Test Safeguard Decoupling**
   - Retire or adjust `tests/unit/services/test_p0_fixes_14_to_20.py:164` to remove the string check on `.github/workflows/security.yml`.
3. **Step 3: Redundant File Removal**
   - Delete `.github/workflows/` (`ci.yml`, `security.yml`, `dependabot.yml`, `CODEOWNERS`).
   - Delete `legacy/` (entire 205 MB tree).
   - Delete obsolete root documentation (`Refactor.md`, `PROJECT.md`, root `ORIGINAL_REQUEST.md`, `TEST_INFRA.md`, `TEST_READY.md`, `LIVE_VALIDATION.md`, `bug_audit.md`, `benchmark_acceptance_report.md`).
   - Delete `docs/operations/` and prune empty directories (`docs/api/`, `docs/architecture/`, `docs/models/`, `deployments/local/`, `deployments/colab/`, `tests/integration/`, `tests/load/`).
   - Delete 9 0-byte README files in `services/` and `workflows/`.
   - Remove large binary outputs from `deployments/kaggle/*_outputs/` and prune empty run folders in `benchmarks/results/`.
4. **Step 4: Kaggle Deployment Consolidation**
   - Consolidate disparate Kaggle scripts (`session1`, `session2`, `run_high_quality_test`, `run_studio_quality_pipeline`, `monitor_both_sessions`, `certify_kernel.py`) into a single `deployments/kaggle/bootstrap.py` while retaining `KaggleAPIClient` in `runner.py`.
5. **Step 5: Preflight Suite Finalization**
   - Validate that the minimal 10–15 scenario preflight suite executes in under 5 seconds with zero external service dependencies.
