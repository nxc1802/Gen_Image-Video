# Factory v4 Architecture Overview

## 1. System Vision & Architecture Principles

Factory v4 is an inference-first Personal AI Production Platform providing authentic multi-modal AI generation and serving across 12 declarative model manifests. It is designed to run seamlessly on Kaggle Dual NVIDIA Tesla T4 (2x 16GB GDDR6) hardware, as well as local development environments.

### Core Architectural Pillars
1. **Capability Truth & Zero Fabricated Outputs**:
   - The platform strictly enforces truth in capability states. If a model engine or backend daemon is unreachable, jobs fail closed with `BACKEND_UNAVAILABLE` (HTTP 503).
   - Mock headers, dummy image containers, simulated waveforms, or synthetic latencies are strictly prohibited.
2. **Dual-Mode Operation**:
   - **Lite Mode (Default)**: Completely self-contained with zero external dependencies (no PostgreSQL server, no Redis broker, no S3/MinIO service, and no Alembic migrations required). Uses an in-memory queue and an ephemeral local scratch disk.
   - **Hardened Production Mode**: First-class enterprise extension modes supporting PostgreSQL/SQLite durable persistence, automated Alembic migrations (`FACTORY_AUTO_MIGRATE=true`), S3-compatible object storage (`STORAGE_BACKEND=s3`), and standardized JWT authentication (`JWT_SECRET_KEY`, `JWT_AUDIENCE`, `JWT_ISSUER`).
3. **Discrete Dual-T4 Hardware Partitioning**:
   - Two discrete Tesla T4 GPUs with **14.2 GB (14,540 MB)** allocatable ceiling each (leaving >= 820 MB CUDA headroom).
   - Strict avoidance of shared-memory illusions; models are routed to dedicated devices:
     - **GPU 0**: `text-service` (Qwen 2.5 Text), `vision-service` (Qwen 2.5 VLM), `decision-service` (Qwen 2.5 Decision), `image-service` (Qwen-Image 2.1 & Turbo), `speech-service` (Qwen3-ASR & TTS).
     - **GPU 1**: `video-service` (Wan 2.2, MiniMax H3), `music-service` (ACE-Step 1.5, YuE2), `embed-service` (Qwen-Embedding & Rerank), `enhancement-service` (Enhancement Suite).
     - **Dual GPU Split**: Qwen3.8-27B (Dual T4 tensor split across GPUs 0 and 1 via atomic dual acquisition).

---

## 2. High-Level Subsystem Topology

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Studio Web Operations Console                        │
│                   (Next.js 15 / React 19 @ Port 3000)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / SSE / WebSocket
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Factory Control Plane API Server                     │
│                  (FastAPI / Python 3.12 @ Port 8000)                   │
├───────────────────────────────────┬────────────────────────────────────┤
│  Authentication & Security        │  Job Lifecycle & SSoT Queue       │
│  - Bearer API Key Verification    │  - Memory / PostgreSQL Store       │
│  - Standardized JWT (HS256/RS256) │  - GPU Partitioned Scheduling      │
│  - Trusted Proxies Rate Limiting  │  - Atomic Worker Lease Broker      │
├───────────────────────────────────┴────────────────────────────────────┤
│  Storage & Delivery Layer                                              │
│  - Ephemeral Scratch Disk (SCRATCH_ARTIFACTS_DIR) / Signed Tokens      │
│  - S3 / MinIO Object Storage Provider with Presigned URLs              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Internal IPC / HTTP
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│     Discrete GPU 0 (cuda:0)       │ │     Discrete GPU 1 (cuda:1)       │
├───────────────────────────────────┤ ├───────────────────────────────────┤
│ - Text Service (Port 8080)        │ │ - Video Service (Port 8189)       │
│ - VLM Service (Port 8081)         │ │ - Music Service (Ports 8008/8009) │
│ - Decision Service (Port 8082)    │ │ - Embedding Service (Port 8083)   │
│ - Image Service (Port 8188)       │ │ - Enhancement Service             │
│ - Speech Service (Ports 8290/8291)│ │                                   │
└───────────────────────────────────┘ └───────────────────────────────────┘
```

---

## 3. Storage Architecture & Artifact Boundaries

Factory v4 abstracts artifact persistence via `StorageProvider`:

1. **Local Scratch Disk Provider (`LocalStorageProvider`)**:
   - Default for Lite Mode.
   - Storage directory configured via `SCRATCH_ARTIFACTS_DIR` (default `/tmp/factory_lite/artifacts`).
   - Atomic disk writes: generated media is written to a temporary `.tmp_*` file and renamed atomically (`os.replace`) to prevent corrupt or partial reads during streaming.
   - Access security: media downloads and chunked video previews use signed time-limited tokens (`LOCAL_STORAGE_SECRET_KEY`) with HMAC-SHA256 signatures, or standard Bearer authorization.
   - Lifecycle retention: background sweeper unlinks expired jobs based on `ARTIFACT_TTL_SECONDS` (default: 3600s). Immediate unlinking on completed download can be activated via `ARTIFACT_CLEANUP_ON_DOWNLOAD=true`.

2. **S3 Object Storage Provider (`S3StorageProvider`)**:
   - Production extension mode (`STORAGE_BACKEND=s3`).
   - Validates bucket existence on startup (`_ensure_bucket_exists`).
   - Generates presigned download and upload URLs using `S3_PUBLIC_ENDPOINT_URL` (for external client routing) and `S3_ENDPOINT_URL` (for internal container networking).

---

## 4. Queue Scheduling & Worker Lease Protocol

1. **Single Source of Truth (SSoT)**:
   - All state transitions (`PENDING` -> `RUNNING` -> `SUCCEEDED` / `FAILED` / `CANCELLED`) are managed centrally by the Control Plane queue repository.
2. **Atomic Worker Lease Protocol**:
   - Workers claim jobs via `POST /v1/jobs/lease` and receive a unique cryptographically random `lease_token`.
   - Heartbeats (`POST /v1/jobs/{job_id}/heartbeat`) and progress updates (`POST /v1/jobs/{job_id}/progress`) strictly require matching `worker_id` and `lease_token`. Unauthorized or expired leases return HTTP 403.
3. **Atomic Multi-GPU Locking**:
   - Models requiring multi-GPU acceleration (e.g. `qwen3.8-27b` tensor split) acquire `[0, 1]` simultaneously via `acquire_gpus([0, 1], job_id)` or none.
4. **Terminal SUCCEEDED Invariant**:
   - No job transitions to `SUCCEEDED` prior to output schema validation and artifact persistence completion.
