# Factory v4 Lite — Personal AI API Provider

Factory v4 Lite is an inference-first Personal AI API Provider refactored from Kaggle AI Studio v3.5. The Control Plane provides single API key authentication, in-memory job queuing, discrete Dual T4 GPU scheduling, streaming events, and ephemeral scratch disk artifact delivery. External enterprise infrastructure dependencies (PostgreSQL, Redis, S3/MinIO, Alembic, complex JWT RBAC) have been completely eliminated.

> **Validation status:** Architecture, security, and contract test suites are automated and 100% green without external dependencies. Models retain `validation_status: pending_live` until verified on live GPU hardware via `deployments.kaggle.smoke_runner`. Model manifests are authoritative: unverified or unavailable capabilities fail closed with 503 instead of returning mock or synthetic outputs.

## Runtime Architecture

- `factory/apps/control-api/` — Lightweight FastAPI Control Plane; in-memory queue, ephemeral scratch storage, Dual T4 scheduler.
- `factory/apps/studio-web/` — Next.js 15 / React 19 API Explorer, Job Dashboard and streaming inspector.
- `services/text/` — Native vLLM OpenAI-compatible text serving.
- `services/vlm/` — Native vLLM multimodal serving with SSRF containment.
- `services/decision/` — Native vLLM Structured Decisions API.
- `services/embedding/` — Native vLLM embedding and rerank workers.
- `services/speech/` — Qwen3-ASR on vLLM + Qwen3-TTS on vLLM-Omni; zero fabricated inference.
- `services/image/` and `services/video/` — Native ComfyUI workers with chunked streaming artifact delivery.
- `services/music/` — Native ACE-Step 1.5 worker.
- `services/enhancement/` — Real enhancement workers; unavailable backends fail with 503.
- `manifests/` — Service/model/capability/hardware source of truth.
- `deployments/kaggle/` — Kaggle Dual-T4 deployment and 4-stage real inference smoke runner (`deployments.kaggle.smoke_runner`).

## Safety and Correctness Invariants

Factory v4 Lite strictly adheres to:

```text
REAL BACKEND AVAILABLE   -> execute genuine inference
REAL BACKEND UNAVAILABLE -> 503 MODEL_UNAVAILABLE / SERVICE_UNAVAILABLE
NEVER                    -> mock PNG containers, mock MP4 headers, pseudo embeddings,
                            canned transcripts, sine-wave TTS, or fabricated telemetry
```

- **Zero External Infrastructure**: No PostgreSQL, SQLite server, Redis broker, S3/MinIO bucket, or Alembic migrations required.
- **Single API Key Security**: Constant-time verification using `FACTORY_API_KEY` via `Authorization: Bearer <key>`.
- **Ephemeral Artifacts**: All generated media resides exclusively in `/tmp/factory_lite/artifacts/{job_id}/` and is automatically unlinked upon completed client download or TTL expiration.
- **Dual T4 GPU Scheduling**: Strict 1-job-per-GPU concurrency lock per discrete GPU device (GPU 0: Text/VLM/Image/Speech, GPU 1: Video/Music/Embedding).

## Single Working Deploy Path

Factory v4 Lite requires zero external infrastructure (no PostgreSQL, Redis, or S3).
Launch the Control Plane with:

```bash
# 1. Install dependencies
uv sync --all-packages

# 2. Set API Key and launch server
export FACTORY_API_KEY="sk-factory-prod-secret-token"
uv run uvicorn factory_control_api.server:app --host 0.0.0.0 --port 8000
```

## Quick Verification

```bash
uv sync --all-packages
uv run python -m compileall -q factory packages services tests
uv run factory-manifest validate --path manifests
uv run pytest tests/unit tests/security tests/contract tests/adversarial tests/e2e -q

cd factory/apps/studio-web
npm run build
```
