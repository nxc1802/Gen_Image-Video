# Factory v4 — Decoupled AI Factory

Factory v4 is a production-oriented refactor of Kaggle AI Studio v3.5. The Control Plane owns authentication, durable jobs, worker leasing, storage, registry, streaming events and telemetry; model inference is delegated to native serving frameworks instead of being reimplemented inside one monolithic FastAPI server.

> **Validation status:** architecture and contract tests are automated, but the repository is **not declared production-certified until the live GPU validation gates in `LIVE_VALIDATION.md` pass**. Model manifests are authoritative: capabilities marked `partial` or `false` must not be presented by clients as available.

## Runtime architecture

- `factory/apps/control-api/` — FastAPI Control Plane only; no model weights.
- `factory/apps/studio-web/` — Next.js 15 / React 19 API Explorer, Job Dashboard and streaming inspector.
- `services/text/` — native vLLM OpenAI-compatible serving.
- `services/vlm/` — native vLLM multimodal serving with media-domain SSRF allowlisting.
- `services/decision/` — native vLLM Decisions API.
- `services/embedding/` — native vLLM embedding and rerank workers.
- `services/speech/` — Qwen3-ASR on vLLM + Qwen3-TTS on vLLM-Omni; compatibility gateway contains no fabricated inference.
- `services/image/` and `services/video/` — native ComfyUI workers.
- `services/music/` — native ACE-Step 1.5 REST API.
- `services/enhancement/` — strict adapters to real enhancement workers; unavailable backends fail with 503 rather than creating synthetic media.
- `manifests/` — service/model/capability/hardware source of truth.
- `infrastructure/ngrok/` — fixed domains and edge policies.
- `legacy/v3/` — archived runtime; excluded from Factory v4 execution.

## Safety and correctness invariants

Factory v4 follows a strict rule:

```text
REAL BACKEND AVAILABLE -> execute real inference
REAL BACKEND UNAVAILABLE -> 503 MODEL_UNAVAILABLE
NEVER -> pseudo embeddings, lexical rerank masquerading as AI, canned transcripts,
         sine-wave TTS/music, synthetic enhancement outputs or invented decision scores
```

Production additionally requires PostgreSQL, Redis and S3/MinIO. SQLite, the in-memory broker and local artifact storage are development/test-only. Raw native APIs that do not provide Factory Bearer authentication (for example ComfyUI) must be protected by the ngrok raw-API edge policy.

## Quick checks

```bash
uv sync --all-packages
uv run python -m compileall -q factory packages services
uv run factory-manifest validate --path manifests
uv run pytest tests/e2e -q

cd factory/apps/studio-web
npm ci
npm run build
```

The default `tests/e2e` mode is a deterministic contract simulation. Set `FACTORY_E2E_LIVE=1` only when a real Factory deployment is available. Passing simulation tests is not evidence that GPU inference or external infrastructure is healthy.

See `PROJECT.md`, `Refactor.md`, `TEST_READY.md` and `LIVE_VALIDATION.md` for the implementation and release gates.
