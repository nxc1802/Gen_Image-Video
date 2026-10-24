# Project: Factory v4 Refactoring (Kaggle AI Studio)

## Architecture
Factory v4 refactors the monolithic Kaggle AI Studio v3.5 into a decoupled, production-grade AI Factory where a dedicated **Control Plane** (`factory/apps/control-api/`) manages authentication, job lifecycle, service/model registry, storage, and telemetry without inference, while individual AI subsystems run natively on their optimal serving frameworks behind fixed ngrok static domains with end-to-end security and a two-tier queue architecture.

### Decoupled Subsystems & Frameworks:
- **Control Plane** (`factory/apps/control-api/`): FastAPI, PostgreSQL (job state, leases, retries), Redis (events, pub/sub, locks), MinIO/S3 (signed URLs), OpenTelemetry.
- **Studio Web** (`factory/apps/studio-web/`): Next.js 15 / React 19 dynamic Operations Console & API Explorer reading backend manifests.
- **Text & VLM** (`services/text/`, `services/vlm/`): Native vLLM for Qwen models (OpenAI Chat Completions, Responses API, structured output, tool calling).
- **Decision Engine** (`services/decision/`): Dedicated vLLM service for `/v1/decisions` (predicate, choice, score routing).
- **Speech** (`services/speech/`): Official vLLM Qwen3-ASR (streaming transcription, forced alignment) & vLLM-Omni Qwen3-TTS (streaming speech, voice cloning).
- **Embedding & Rerank** (`services/embedding/`): Real vLLM/TEI Qwen embeddings; fail-fast with `503 MODEL_UNAVAILABLE` (zero fake vectors).
- **Image Subsystem** (`services/image/`, `workflows/image/`): Headless ComfyUI on GPU 0 with versioned Workflow Registry for Qwen-Image 2.1 family.
- **Video Subsystem** (`services/video/`, `workflows/video/`): Headless ComfyUI on GPU 1 with Wan2.2 & MiniMax H3 (T2V, I2V, FL2V, Video Extension, Audio VAE).
- **Music Subsystem** (`services/music/`): Standalone ACE-Step 1.5 async REST server integration.
- **Enhancement Subsystem** (`services/enhancement/`): Post-processing service for image/video upscaling, frame interpolation, audio enhancement.

### Data Flow & Network Ingress:
- All external client and Studio Web requests route over **fixed ngrok static domains** (e.g. `jobs.*`, `text.*`, `image.*`, `studio.*`).
- Ingress enforced by edge Traffic Policies: mandatory Bearer token authentication, rate limiting, and zero unauthenticated bypass.
- Media inputs are validated and sanitized against SSRF via an internal **Asset Service** before ingestion by model workers.
- Large artifacts (>1MB) are uploaded to S3-compatible storage with signed URLs; inline base64 is deprecated.

---

## Feature Inventory

Every feature discovered in the Survey Phase is assigned to an implementation milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Job Submission | Submit asynchronous generative job to Factory queue | M2 | Refactor.md §13 |
| 2 | Job Status Query | Retrieve detailed status, progress, and artifacts of a job | M2 | Refactor.md §13 |
| 3 | Job Cancellation | Cancel an active, queued, or streaming job | M2 | Refactor.md §13 |
| 4 | Job Retry | Re-enqueue a failed or cancelled job with exponential backoff | M2 | Refactor.md §13 |
| 5 | Job SSE Events | Stream live progress events for a job over SSE | M2 | Refactor.md §13, §15 |
| 6 | Job WebSocket | Bidirectional WebSocket for job events, streaming, and control | M2 | Refactor.md §13 |
| 7 | Service Manifest Discovery | Discover all registered services and their static URLs | M1 | Refactor.md §17 |
| 8 | Model Manifest Registry | Retrieve all models, hardware requirements, and capabilities | M1 | Refactor.md §17 |
| 9 | Asset Service Upload / SSRF Validation | Ingest external URL or upload media, validate, upload to S3 | M1 | Refactor.md §11 |
| 10 | Text Chat Completions | OpenAI-compatible chat completion with continuous batching | M3 | Refactor.md §3, §71 |
| 11 | Text Responses API | OpenAI-compatible Responses API for complex reasoning | M3 | Refactor.md §3, §71 |
| 12 | Decision Predicate | Binary True/False decision with calibrated probability score | M3 | Refactor.md §4 |
| 13 | Decision Choice | Multi-class routing and selection among discrete choices | M3 | Refactor.md §4 |
| 14 | Decision Score | Continuous scalar quality/priority evaluation | M3 | Refactor.md §4 |
| 15 | Speech Audio Transcription | Transcribe audio file with language detection & timestamps | M3 | Refactor.md §6 |
| 16 | Speech Streaming ASR | Real-time speech-to-text over WebSocket | M3 | Refactor.md §6 |
| 17 | Speech Forced Alignment | Align transcript words to exact audio timestamps | M3 | Refactor.md §6 |
| 18 | Speech Synthesis (TTS) | OpenAI-compatible speech synthesis with voice cloning | M3 | Refactor.md §6 |
| 19 | Vector Embeddings | Generate dense vector embeddings for text / multimodal (real vectors only) | M3 | Refactor.md §3, §74 |
| 20 | Document Rerank | Cross-encoder relevance ranking of documents (real models only) | M3 | Refactor.md §3, §75 |
| 21 | Image Text-to-Image | Generate image from text prompt via ComfyUI (GPU 0) | M3 | Refactor.md §7 |
| 22 | Image Image-to-Image | Transform source image conditioned on text prompt | M3 | Refactor.md §7 |
| 23 | Image Inpainting | Fill masked region of source image with prompt | M3 | Refactor.md §7 |
| 24 | Image Outpainting | Expand canvas boundaries of source image | M3 | Refactor.md §7 |
| 25 | Image Instruction Edit | Modify image via natural language edit instruction | M3 | Refactor.md §7 |
| 26 | Image Reference Conditioning | Guide character/style consistency using reference images | M3 | Refactor.md §7 |
| 27 | Image ControlNet | Guide generation using pose, depth, or canny edges | M3 | Refactor.md §7 |
| 28 | Video Text-to-Video | Generate video from prompt using Wan2.2 / MiniMax H3 | M3 | Refactor.md §8 |
| 29 | Video Image-to-Video | Animate static image into video clip | M3 | Refactor.md §8 |
| 30 | Video First & Last Frame (FL2V) | Generate video interpolating between two keyframes | M3 | Refactor.md §8 |
| 31 | Video Extension & Chaining | Extend existing video clip continuously | M3 | Refactor.md §8 |
| 32 | Video with Audio | Generate synchronized video with stereo audio track (Audio VAE) | M3 | Refactor.md §8 |
| 33 | Music Text-to-Music | Generate music track from prompt via ACE-Step 1.5 | M3 | Refactor.md §5 |
| 34 | Music Lyrics-to-Music | Generate vocal/instrumental song conditioned on lyrics | M3 | Refactor.md §5 |
| 35 | Music Audio-to-Audio | Remix or style transfer an existing audio track | M3 | Refactor.md §5 |
| 36 | Enhancement Image Super-Resolution | Upscale image 2x/4x using Real-ESRGAN / BSRGAN | M3 | Refactor.md §9 |
| 37 | Enhancement Video Super-Resolution | Upscale video frames to high resolution | M3 | Refactor.md §9 |
| 38 | Enhancement Frame Interpolation | Double or quadruple video frame rate via RIFE | M3 | Refactor.md §9 |
| 39 | Enhancement Audio Enhancement | Denoise, balance, and clarify audio track | M3 | Refactor.md §9 |
| 40 | Studio Dynamic Schema Form | Auto-render parameter controls from model manifests | M4 | Refactor.md §17, §19 |
| 41 | Studio Streaming Inspector | Visual viewer for live SSE tokens and ComfyUI WS steps | M4 | Refactor.md §19 |
| 42 | Studio Code Generator | Export request payload into executable client code | M4 | Refactor.md §19 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Foundation & Security (Waves 0, 1, 2) | Archive legacy v3.5 into `legacy/v3/`, establish Factory v4 directory tree, implement declarative schemas (`ServiceManifest`, `ModelManifest`, `CapabilityManifest`, Unified Error Contract), edge ngrok static domain definitions, mandatory Bearer token auth across all endpoints with zero bypass, remove `x-target-base-url` & credential forwarding, Asset Service SSRF protection, sanitized error responses. | none | DONE |
| M2 | Two-Tier Queue & Factory Job Lifecycle (Wave 3) | Tier 1 native engine concurrency, Tier 2 PostgreSQL + Redis factory job manager, state machine (`CREATED` -> `QUEUED` -> `ADMITTED` -> `RUNNING` -> `STREAMING` -> `SUCCEEDED` / `FAILED` / `CANCELLED` / `RETRYING`), worker leasing, heartbeats, dead-job recovery, multi-modal streaming (SSE tokens, WS audio, SSE/WS progress), S3 storage with signed URLs, OpenTelemetry tracing. | M1 | DONE |
| M3 | Core AI Subsystem Migration (Waves 4 & 5) | Text & VLM vLLM service, dedicated vLLM `/v1/decisions` service, Speech Qwen3-ASR & Qwen3-TTS services, Embedding & Rerank real vLLM/TEI service (503 fail-fast, zero fake vectors), ComfyUI GPU 0 Image daemon & workflow registry, ComfyUI GPU 1 Video daemon & workflow registry, ACE-Step 1.5 music REST integration, Media Enhancement service. | M1, M2 | DONE |
| M4 | Factory Studio Web Dynamic Console (Wave 6) | Overhaul Next.js 15 / React 19 Studio Web into dynamic Operations Console & API Explorer reading backend manifests, live Telemetry & Streaming Inspectors, Job Dashboard, API Code Generator. | M1, M2, M3 | IN_PROGRESS |
| M5 | Final Milestone: E2E Test Suite Pass & Adversarial Hardening (Waves 7 & 8) | Phase 1: 100% pass of E2E Test Suite published by E2E Testing Track (Tiers 1-4). Phase 2: Adversarial Coverage Hardening (Tier 5 white-box challenger loop). 100 concurrent requests stress test and worker reboot recovery verification. | M1, M2, M3, M4, TEST_READY | PLANNED |

---

## Interface Contracts

### Control Plane (`factory/apps/control-api`) ↔ Subsystem Workers (`services/*`)
- **Worker Registration**: `POST /v1/registry/workers` (registers `worker_id`, `subsystem`, `supported_models`, `capabilities`, `internal_url`, `ngrok_domain`).
- **Worker Leasing**: `POST /v1/jobs/lease` (worker requests next job matching capabilities; leases with TTL 300s).
- **Worker Heartbeat**: `POST /v1/workers/{worker_id}/heartbeat` (sent every 30s; worker lease extended).
- **Progress Reporting**: `POST /v1/jobs/{job_id}/progress` (worker updates step, percent, message, preview frames).
- **Job Completion**: `POST /v1/jobs/{job_id}/complete` (worker provides artifact paths, sha256, duration, token counts).
- **Job Failure**: `POST /v1/jobs/{job_id}/fail` (worker reports sanitized error code and message).

### Public Client / Studio Web ↔ Factory Control Plane
- **Authentication**: `Authorization: Bearer <API_KEY>` (mandatory on 100% of non-public endpoints).
- **Error Format**:
  ```json
  {
    "error": {
      "code": "STRING_ENUM",
      "message": "Human readable message",
      "request_id": "req-uuid-v4",
      "details": {}
    }
  }
  ```
- **Job Submission Contract**:
  ```json
  {
    "service": "text | image | video | speech | music | decision | embedding | enhancement",
    "model": "model-family-id",
    "task": "task-name",
    "inputs": {},
    "priority": 1,
    "idempotency_key": "optional-uuid"
  }
  ```

### Asset Service ↔ Storage (`MinIO / S3`)
- **Validation**: Strict URL check against loopback (`127.0.0.1`, `::1`), private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and cloud metadata (`169.254.169.254`).
- **Artifact Retrieval**: Returns presigned S3 GET URL with 24h expiration for any artifact > 1MB.

---

## Code Layout

```text
/Volumes/WorkSpace/Project/Gen_Image:Video/
├── factory/
│   ├── apps/
│   │   ├── control-api/             # Factory Control Plane API (FastAPI)
│   │   │   ├── src/
│   │   │   │   ├── auth/            # Mandatory Bearer token & JWT middleware
│   │   │   │   ├── registry/        # Service, Model, and Capability registry
│   │   │   │   ├── queue/           # PostgreSQL & Redis job queue & state machine
│   │   │   │   ├── assets/          # SSRF-safe Asset Service
│   │   │   │   ├── storage/         # S3/MinIO client & signed URL generator
│   │   │   │   ├── telemetry/       # OpenTelemetry request tracing & metrics
│   │   │   │   └── api/             # FastAPI routers (/v1/jobs, /v1/models, etc.)
│   │   │   ├── pyproject.toml
│   │   │   └── Dockerfile
│   │   └── studio-web/              # Next.js 15 / React 19 Operations Console
│   │       ├── src/
│   │       │   ├── app/             # App router pages (dashboard, explorer, inspector)
│   │       │   ├── components/      # Dynamic schema form generators & meters
│   │       │   └── lib/             # API clients (NO x-target-base-url)
│   │       └── package.json
│   ├── services/                    # Standalone Subsystem Daemons & Launchers
│   │   ├── text/                    # vLLM Text & VLM launcher
│   │   ├── decision/                # Dedicated vLLM /v1/decisions service
│   │   ├── speech/                  # Qwen3-ASR & Qwen3-TTS launchers
│   │   ├── embedding/               # vLLM/TEI embedding & rerank launcher
│   │   ├── image/                   # ComfyUI GPU 0 headless daemon
│   │   ├── video/                   # ComfyUI GPU 1 headless daemon
│   │   ├── music/                   # ACE-Step 1.5 async REST launcher
│   │   └── enhancement/             # Image/Video/Audio post-processing daemon
│   ├── workflows/                   # Versioned ComfyUI JSON workflows
│   │   ├── image/                   # T2I, I2I, Inpaint, Outpaint, Edit, ControlNet
│   │   └── video/                   # T2V, I2V, FL2V, Video Extension, Audio VAE
│   ├── packages/                    # Shared Python & TypeScript packages
│   │   ├── contracts/               # Pydantic schemas (Manifests, Errors, Jobs)
│   │   ├── model-registry/          # Manifest loader & validator
│   │   └── client-sdk/              # Factory Python & TS client SDK
│   ├── infrastructure/              # ngrok, Redis, Postgres DDL, S3, Docker
│   ├── deployments/                 # Kaggle dual-T4, Colab, Local profiles
│   ├── manifests/                   # YAML definitions for Services, Models, Hardware
│   │   ├── services/
│   │   ├── models/
│   │   └── hardware/
│   └── tests/                       # Automated Verification Suites
│       ├── unit/
│       ├── contract/
│       ├── integration/
│       ├── e2e/
│       ├── load/
│       └── security/
├── docs/                            # Architecture, API & operational runbooks
└── legacy/                          # Archived Legacy v3.5
    └── v3/                          # Frozen v3.5 codebase (all-in-one, old UI, tests)
```
