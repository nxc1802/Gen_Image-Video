# Single-host Control Plane

Phase 7 follows [MASTER_PLAN.md](MASTER_PLAN.md). The canonical production entrypoint remains `factory serve`, exporting the same `factory_control_api:app` / `factory_control_api.server:app`. It composes the explicit DeploymentRegistry/BackendSession, canonical Engines/Adapters, local asynchronous media jobs and scratch artifacts. Public API details are in [OPENAI_API.md](OPENAI_API.md).

## Core composition

Core startup requires no Redis, PostgreSQL, SQLite migrations, S3, worker registration, lease ownership or enterprise role hierarchy. Job metadata is in memory; artifacts live under `SCRATCH_ARTIFACTS_DIR` with TTL/quota sweeping. Authentication defaults to the existing single Factory API key. Physical GPU assignment remains fixed by deployment YAML/CLI; shared sync, async and streaming execution use the same per-session device locks.

Default startup initializes:

- Auth configuration and Manifest catalog snapshot.
- Explicit selected deployment bindings and, in production, prepared/launched/healthy owned backend processes.
- One scratch manager and one local job manager.
- Request IDs, tracing, errors, body limits, timeouts and rate limits.

It does not initialize a queue broker, distributed scheduler, lease coordinator, worker watchdog, old JobLifecycleManager, old ModelRuntimeManager, repository/migrations, storage gateway or S3 provider. No heartbeat loop is required for in-process jobs. Backend health/process ownership remains necessary for the genuine external SDK daemons.

The default app mounts the OpenAI/Factory inference routes, catalog/runtime diagnostics, health and local job management. It does not mount legacy worker/asset/storage/Studio gateways. There is no generic `POST /v1/jobs` inference gateway; applications submit through the corresponding OpenAI route or the model/capability-specific music/enhancement extensions. `/v1/registry/runtime` reports actual core composition and optional infrastructure imports; it never certifies inference readiness.

## Job lifecycle and cancellation

| Operation | Meaning |
|---|---|
| `GET /v1/jobs` | Owner-scoped retained jobs, up to the requested bounded limit; no `service` field. |
| `GET /v1/jobs/{id}` | Canonical state/progress/error/timestamps, with no backend credential or asset path. |
| `POST /v1/jobs/{id}/cancel` | Idempotent cancellation. Waiting jobs become cancelled without calling the native inference hook. Running jobs become cancelling until the actual native call returns. |
| `GET /v1/jobs/{id}/events` | SSE snapshots of actual state transitions; no broker or fabricated inference percentages. |
| `GET /v1/jobs/{id}/content` | Owned completed bytes verified against the stored SHA-256; incomplete/cancelled jobs have no downloadable content. |

Video/music/enhancement status/content routes remain available. The retained OpenAI Video object has no cancelled enum: video retrieval represents terminal cancellation as `failed` with `JOB_CANCELLED`; canonical `/v1/jobs` management exposes `cancelled`. Pending native work remains `cancelling` on the canonical route, rather than claiming that the GPU has stopped.

Waiting for a GPU lock is asynchronous and does not consume an executor thread. Device locks are acquired in a stable order; cancelling while partially acquiring a dual-device reservation releases the acquired subset. Synchronous engines and streaming requests use the same physical-device locks, so cancellation cannot let another co-resident engine overlap an unfinished native call.

Cancelling an asyncio task cannot hard-preempt a Python SDK/thread or an already submitted native graph. Factory drains the outstanding call, discards its result and keeps the GPU reservation until it actually ends. Repeated cancellation cannot release that reservation early. No alternate backend or model is started. During shutdown, admission closes, jobs receive cancellation, owned backends stop to interrupt outstanding RPCs, jobs drain, and scratch sweeping stops. A submission still undergoing admission cannot create a new job after shutdown begins.

Progress is 0 until actual completion, then 100; queue/start/cancel/failure are real events, not invented native progress. Maximum outstanding/retained jobs remain 32/256. Restart loses job metadata; TTL expires metadata/content, and quota/TTL sweeping can remove artifacts. Retrieval fails if bytes are missing or changed. These local ephemeral semantics are explicit and are not disguised as durable distributed execution.

## Retained migration code

`factory/apps/control-api/src/legacy.py` holds the former lifecycle and gateway composition for explicit historical contract fixtures and Studio migration. It is not imported by core startup or selected by `factory serve`, and it rejects production startup. Its old queues, leases, repositories, storage endpoints and Studio code remain available to the migration tests; no large deletion was performed.

Historical tests now import that migration app explicitly. Runner tests inject it only into their test fixtures; production entrypoints are not redirected to legacy behavior. The frozen baseline and old artifacts keep their original provenance and cannot certify the new runtime. Studio's current legacy route parity test checks the retained host; migration to the new core API remains Phase 10.

Infrastructure libraries moved to optional extras:

- `factory-control-api[legacy]`: retained S3/database/migration dependencies and legacy JWT extra.
- `factory-auth[jwt]`: explicit optional JWT/crypto dependency. API-key-only startup imports neither PyJWT nor crypto.
- The workspace development group includes the legacy extra to run historical software tests; it is not a production/core requirement.

Legacy `FACTORY_DB_BACKEND`, `DATABASE_URL`, `REDIS_URL`, `STORAGE_BACKEND` and S3 settings do not select infrastructure for the new core. Core artifacts always use its local scratch configuration. Retained compatibility modules own the former infrastructure settings. No unavailable service is probed and then silently substituted during core inference.

## Verification and limits

The executable Phase 7 gate creates a fresh Python environment and installs only the core editable packages with their default dependencies. In the measured run this requires **30 packages**, with no Redis/SQL/S3 SDK/PyJWT/crypto installation. The runtime probe blocks imports of those optional libraries and denies network connections except its own HTTP listener, while unreachable Redis/Postgres/S3 environment URLs are present. It starts the actual core app, verifies unvalidated requests fail closed, then exercises the real local job/Engine/storage plumbing using an explicitly injected test authorization and Adapter double. No fixture hook/config flag exists in production modules.

Eleven local lifecycle contracts cover waiting/running/repeated cancellation, shared/partial dual-device locks, shutdown order, late-admission rejection, owner isolation, TTL/hash checks, failed output, HTTP status/events and rejection of the retained legacy host in production. The official SDK regression gate still applies. These are **software-only** checks, not GPU inference or semantic evidence.

```bash
.venv/bin/python -m pytest tests/contract/test_local_control_plane.py -q -o addopts=''
.venv/bin/python benchmarks/control_plane_gate.py
```

The gate also requires current-source prior phase gates and an actual Kaggle core process reporting no initialized legacy components or imported infrastructure. ASR lifecycle evidence proves exact checkpoint load/health/GPU residency/shutdown only. The isolated disabled Enhancement probe is not V3/semantic/Full Factory E2E certification. Subsequent Phase 8 native semantic evidence is tracked separately in [UPGRADE_STATUS.md](UPGRADE_STATUS.md) and the [native matrix](../validation/native_matrix.md); it does not authorize production through this software gate. **No model/capability is production-ready.**
