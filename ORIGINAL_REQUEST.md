# Original User Request

## 2026-10-23T21:25:06Z

Refactor Kaggle AI Studio from v3.5 to Factory v4: a production-grade, decoupled AI Factory where a dedicated Control Plane manages authentication, job lifecycle, service/model registry, storage, and telemetry without inference, while individual AI subsystems (Text, VLM, Decision, Embedding, Speech, Image, Video, Music, Enhancement) run natively on their optimal serving frameworks (vLLM, ComfyUI, ACE-Step 1.5) behind fixed ngrok static domains with end-to-end security and a two-tier queue architecture.

Working directory: /Volumes/WorkSpace/Factory/Gen_Image:Video
Integrity mode: development
Reference specification: /Volumes/WorkSpace/Factory/Gen_Image:Video/Refactor.md

## Requirements

### R1. Architecture Decoupling & Manifest-Driven Foundation (Wave 0 & 1)
- Archive legacy v3.5 runtime into `legacy/v3/`.
- Establish the new repository directory tree (`factory/apps/control-api/`, `factory/apps/studio-web/`, `services/`, `workflows/`, `packages/`, `infrastructure/`, `deployments/`, `manifests/`, `tests/`, `benchmarks/`, `docs/`).
- Eliminate the monolithic inference gateway, global `STUDIO_MODE`, and duplicated config files.
- Implement strict declarative schemas for `ServiceManifest`, `ModelManifest`, and `CapabilityManifest`. A model family is only marked `FULL` if 100% of required capabilities are backed by real workflows/checkpoints.

### R2. Edge Networking & Mandatory End-to-End Security (Wave 1 & 2)
- Replace ephemeral Cloudflare Quick Tunnels with fixed/reserved ngrok static domains per subsystem (`text.*`, `image.*`, `video.*`, `speech.*`, `music.*`, `jobs.*`, `studio.*`).
- Enforce edge security policies (JWT verification, rate limiting, IP restrictions for admin routes).
- Enforce mandatory Bearer token authentication across all application endpoints with no unauthenticated public bypass (`API_KEY=""` prohibited).
- Eliminate arbitrary proxying (`x-target-base-url` removed) and prevent credential forwarding to fallback domains.
- Implement an Asset Service for SSRF protection and media validation before ingestion into models.
- Sanitize all public error responses (`{ "error": { "code": "...", "request_id": "..." } }`), logging raw stack traces internally.

### R3. Two-Tier Queue & Factory Job Lifecycle (Wave 3)
- Tier 1: Retain native engine concurrency (vLLM continuous batching scheduler, ComfyUI `/prompt` queue/WS, ACE-Step async task system).
- Tier 2: Build a production Factory Job management layer using PostgreSQL (job state, metadata, worker leases, retries) and Redis (real-time progress, pub/sub events, short-lived locks, heartbeats).
- Support standard lifecycle states: `CREATED` → `QUEUED` → `ADMITTED` → `RUNNING` → `STREAMING` → `SUCCEEDED` / `FAILED` / `CANCELLED` / `RETRYING`.
- Standardize multi-modal streaming: SSE for token/reasoning generation, WebSocket/chunked PCM for audio, SSE/WS for generation progress.
- Implement S3-compatible object storage with signed URLs for large artifacts, deprecating inline base64 for media.
- Standardize OpenTelemetry metrics and distributed request tracing (`request_id` → `job_id` → `worker_id` → `model_id` → `artifact_id`).

### R4. Core AI Subsystem Migration (Wave 4 & 5)
- **Text & VLM**: Native vLLM deployment for Qwen models supporting OpenAI Chat Completions, Responses API, structured JSON output, tool calling, and multimodal inputs.
- **Decision Engine**: Dedicated vLLM service for `/v1/decisions` supporting predicate, choice, and score routing with accuracy and calibration validation (no fake heuristic outputs).
- **Speech**: Official vLLM serving for Qwen3-ASR (streaming transcription, timestamps, forced alignment) and vLLM-Omni for Qwen3-TTS (streaming speech, voice cloning, reference audio).
- **Embedding & Rerank**: vLLM/TEI serving real Qwen text & multimodal vectors; fail gracefully with `503 MODEL_UNAVAILABLE` rather than generating deterministic fake vectors.
- **Image**: Standalone headless ComfyUI on GPU 0 with a versioned Workflow Registry for Qwen-Image 2.1 (T2I, I2I, inpainting via mask, outpainting, instruction edit, reference conditioning, ControlNet).
- **Video**: Standalone headless ComfyUI on GPU 1 with Wan2.2 and MiniMax H3 workflows (T2V, I2V, First/Last Frame, reference conditioning, chaining, video extension, and native stereo Audio VAE).
- **Music**: Standalone ACE-Step 1.5 async REST server integration (text-to-music, lyrics, audio-to-audio).
- **Enhancement**: Dedicated post-processing service for image/video upscaling, frame interpolation, and audio enhancement.

### R5. Factory Studio Frontend & Verification Suite (Wave 6, 7 & 8)
- Overhaul Next.js 15 / React 19 Studio Web into a dynamic Operations Console & API Explorer that reads schemas and capability manifests dynamically (no hardcoded model dropdowns).
- Build live Telemetry & Streaming Inspectors (SSE/WS event loggers, queue monitors, GPU utilization gauges).
- Implement interactive Job Dashboard (retry, cancel, artifact preview, code generator for curl, Python, TS).
- Complete automated end-to-end contract test suite (QA verification for every capability, fault injection, worker reboot recovery, and load testing with 100+ concurrent requests).

## Acceptance Criteria

### Architecture & Foundation
- [ ] 0 monolithic inference server files remaining in production runtime.
- [ ] 0 global `STUDIO_MODE` switches; every subsystem is independently deployable and scalable.
- [ ] 0 duplicated configuration files (`models.yaml` consolidated into modular manifests).

### Security & Networking
- [ ] 100% of endpoints require valid Bearer authentication; unauthenticated requests receive `401 Unauthorized`.
- [ ] 0 arbitrary proxy endpoints (`x-target-base-url` removed from client and server).
- [ ] 0 credential-forwarding fallbacks across different upstream domains.
- [ ] 100% of subsystem URLs remain static and unchanged after 10 consecutive Kaggle/worker reboots.
- [ ] 0 internal stack traces or absolute filesystem paths exposed in API error responses.

### Queue & Reliability
- [ ] System accepts 100 concurrent job submissions without crashing or dropping requests.
- [ ] Long-running jobs support cancellation, retries, and state recovery upon worker crash.
- [ ] Media artifacts larger than 1MB are stored in S3-compatible storage with signed download URLs.

### AI Engine Integrity & Streaming
- [ ] 100% of model families marked `FULL` pass automated contract tests for all declared capabilities.
- [ ] 0 fake, mock, or simulated vectors returned when an embedding/rerank model fails (must return `503 MODEL_UNAVAILABLE`).
- [ ] Token streaming (SSE), audio streaming (chunked/WS), and progress streaming (SSE/WS) operate end-to-end.

### Operations & UI
- [ ] Factory Studio dynamically renders controls, parameters, and forms from backend manifests.
- [ ] Distributed tracing correlates `request_id` across the control plane, worker, and final artifact.


## Follow-up — 2026-10-24T04:28:22Z

The quota window has fully reset and the user has requested to continue reconnecting the multi-agent system. Please resume Milestone 2 Iteration 2 remediation and verification.


## Follow-up — 2026-10-25T08:32:22Z

Harden and complete the Factory v4 AI inference platform codebase by fixing packaging and CI blockers, resolving runtime contract mismatches between Studio Web and Control Plane, establishing worker lease authorization, removing fabricated telemetry, and ensuring 100% test pass rate.

Working directory: /Volumes/WorkSpace/Factory/Gen_Image:Video
Integrity mode: development

## Requirements

### R1. Packaging & Workspace Build Alignment (P0)
- Configure Hatchling build wheel targets in `packages/auth/pyproject.toml`, `packages/client-sdk/python/pyproject.toml` (with entrypoint in `src/factory_client/__init__.py`), and `factory/apps/control-api/pyproject.toml`.
- Provide `factory_control_api` package entrypoint in `factory/apps/control-api/src/factory_control_api/__init__.py` exporting `app`.
- Ensure `uv sync --all-packages` executes cleanly and builds wheels without error.

### R2. Canonical Task ID Resolution & Contract Alignment (P0)
- In `packages/contracts/src/factory_contracts/tasks.py`, implement `normalize_task(task: str) -> str` and `CANONICAL_TASK_MAP` mapping capability IDs (`image.t2i`, `video.t2v`, `image.i2i`, `image.inpaint`, etc.) and short aliases to canonical names (`text_to_image`, `text_to_video`, etc.).
- Ensure `JobLifecycleManager.submit_job` normalizes `req.task` so queued jobs in the database use canonical task names.
- Update `services/image/worker.py` and `services/video/worker.py` `supported_tasks` declarations and `workflows/image/registry.py` and `workflows/video/registry.py` `TASK_ALIASES` to support both canonical task names and capability IDs.

### R3. Job Dashboard List Endpoint (P0)
- Add `list_jobs(limit: int = 50, offset: int = 0, service: Optional[str] = None, state: Optional[str] = None) -> List[FactoryJob]` to `BaseJobRepository`.
- Implement `list_jobs` in `SqliteJobRepository` and `PostgresJobRepository` (ordered by `created_at DESC`).
- Expose `manager.list_jobs` in `JobLifecycleManager`.
- Implement `GET /v1/jobs` in `factory/apps/control-api/src/api/routers/jobs.py` matching Studio Web client contract (`apiClient.listJobs`).

### R4. Asset Upload API Router (P0)
- Create `factory/apps/control-api/src/api/routers/assets.py` mounting `POST /v1/assets/upload` and `GET /v1/assets/{asset_id}/download`.
- `POST /v1/assets/upload` accepts `multipart/form-data` file uploads, validates via `AssetService.ingest_bytes()`, stores content via active `StorageProvider`, and returns artifact metadata with `s3_url` and `download_url`.
- Include `assets.router` in `factory/apps/control-api/src/main.py`.

### R5. Worker Lease Token & Ownership Authorization (P0)
- Extend `ProgressUpdateRequest`, `JobCompletionRequest`, and `JobFailureRequest` in `factory_contracts/jobs/worker_contract.py` to include `worker_id: Optional[str]` and `lease_token: Optional[str]`.
- Enforce in `JobLifecycleManager.update_progress`, `complete_job`, and `fail_job` that the caller's `worker_id` and `lease_token` match the active job lease; return HTTP 403 `UnauthorizedError` if unauthorized.
- Enforce in `LeaseCoordinator.process_heartbeat` that `req.lease_token` matches active lease before extending.
- Update worker daemons (`services/image/worker.py`, `services/video/worker.py`, `services/music/worker.py`) to pass active `worker_id` and `lease_token`.

### R6. Real Telemetry & Stale Registry Elimination (P1)
- Remove hardcoded `11.2` and `13.8` VRAM numbers from `TelemetryMetrics` and `inspector.py`. Default initial VRAM values to `null`.
- In `LeaseCoordinator.process_heartbeat`, update real VRAM in `metrics.set_gpu_vram` when worker heartbeat reports `vram_used_mb`.
- In `SqliteJobRepository` and `PostgresJobRepository` `upsert_worker_heartbeat`, replace `COALESCE(?, current_job_id)` with explicit assignment `current_job_id = $7` so when worker finishes a job and sends `active_job_id=None`, stale job IDs are cleared. Also update worker capabilities on heartbeat.

### R7. Auth Configuration Alignment & S3 Fail-Fast (P1)
- In `packages/auth`, add `jwt_algorithm` (from `JWT_ALGORITHM`, default `HS256`) and `jwt_audience` to `AuthConfig`. Pass both to `decode_jwt_token`.
- In `S3StorageProvider._ensure_bucket_exists`, fail fast with an exception if running in production and bucket is inaccessible.

### R8. GitHub Actions CI Configuration (P1)
- Create `.github/workflows/ci.yml` running two matrix jobs:
  1. `studio-web`: checkout, setup-node, npm ci, typecheck, lint, next build.
  2. `backend`: checkout, setup-python (3.11 & 3.12), install uv, `uv sync --all-packages`, `factory-manifest validate`, `pytest tests/unit tests/security tests/contract tests/adversarial tests/e2e workflows services -q`.

## Acceptance Criteria

### Automated Verification
- [ ] `uv sync --all-packages` exits with code 0.
- [ ] `.venv/bin/factory-manifest validate` exits with code 0 and reports 0 errors.
- [ ] All 1,065+ existing tests pass without regressions:
  - `tests/unit`: 83+ passed
  - `tests/security`: 116+ passed
  - `tests/contract`: 4+ passed
  - `tests/adversarial`: 394+ passed
  - `tests/e2e`: 468+ passed
  - `workflows` & `services`: 61+ passed
- [ ] New unit tests pass:
  - `tests/unit/api/test_jobs_list.py`: verifies `GET /v1/jobs` pagination, limit, and status filters.
  - `tests/unit/api/test_assets_upload.py`: verifies `POST /v1/assets/upload` multipart upload and error cases.
  - `tests/unit/queue/test_worker_lease_auth.py`: verifies unauthorized worker cannot progress/complete/fail another worker's lease.
  - `tests/unit/contracts/test_task_normalization.py`: verifies canonical task normalization.
- [ ] Studio Web production build succeeds: `npm run build` in `factory/apps/studio-web`.
