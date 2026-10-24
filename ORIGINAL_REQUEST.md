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
