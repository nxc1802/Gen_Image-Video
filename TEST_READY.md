# Factory v4 Lite: E2E Test Suite Specification & Coverage Matrix

**Suite Location**: `tests/e2e/test_factory_lite_e2e.py`  
**Target Architecture**: Factory v4 Lite In-Memory Personal AI API Provider  
**Author**: `teamwork_test_writer_lite_1`  
**Status**: Published & Ready for Milestone Verification  

---

## 1. Overview & Architecture

The Factory v4 Lite E2E test suite (`tests/e2e/test_factory_lite_e2e.py`) is an opaque-box, requirement-driven test suite designed according to the authoritative project requirements (`ORIGINAL_REQUEST.md` and `PROJECT.md`).

### Zero External Service Dependencies
- **No PostgreSQL or SQLite server**: Uses FastAPI `TestClient` targeting the in-memory Control Plane application (`app`).
- **No Redis Broker or Pub/Sub**: Operates with in-process memory queue / scheduler and pub/sub streams.
- **No S3 / MinIO**: Directly validates local scratch disk artifact delivery and automatic cleanup (`/tmp/factory_lite/artifacts/`).
- **No Alembic**: Zero database migration dependencies.

---

## 2. Test Runner Commands

All tests execute via the workspace virtual environment (`.venv`):

### Run Entire Factory Lite E2E Suite
```bash
.venv/bin/pytest tests/e2e/test_factory_lite_e2e.py -v
```

### Run Tier 1: Core Feature Coverage
```bash
.venv/bin/pytest tests/e2e/test_factory_lite_e2e.py -k "TestTier1CoreFeatureCoverage" -v
```

### Run Tier 2: Boundary & Corner Cases
```bash
.venv/bin/pytest tests/e2e/test_factory_lite_e2e.py -k "TestTier2BoundaryCornerCases" -v
```

### Run Tier 3: Cross-Feature Combinations
```bash
.venv/bin/pytest tests/e2e/test_factory_lite_e2e.py -k "TestTier3CrossFeatureCombinations" -v
```

### Run Tier 4: Real-world Provider Scenarios
```bash
.venv/bin/pytest tests/e2e/test_factory_lite_e2e.py -k "TestTier4RealWorldProviderScenarios" -v
```

---

## 3. 4-Tier Test Matrix & Requirement Mapping

| Tier | Test Method | Covered Requirements | Expected Output Source | Current Baseline Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | `test_t1_health_check_public` | `GET /health` public endpoint (R2) | `PROJECT.md § Control Plane ↔ Client` | **PASS** |
| **Tier 1** | `test_t1_model_discovery` | Dynamic model manifest discovery `GET /v1/models` (R2) | `manifests/models/*.yaml`, `PROJECT.md` | **PASS** |
| **Tier 1** | `test_t1_single_api_key_auth_enforcement` | Single `FACTORY_API_KEY` Bearer auth (R1, R2) | `ORIGINAL_REQUEST.md R1`, `PROJECT.md` | **PASS** |
| **Tier 1** | `test_t1_job_submission` | Inference job creation `POST /v1/jobs` (R2) | `PROJECT.md § Control Plane ↔ Client` | **PASS** |
| **Tier 1** | `test_t1_job_status_lookup` | In-memory job state query `GET /v1/jobs/{id}` (R2) | `PROJECT.md § Control Plane ↔ Client` | **PASS** |
| **Tier 1** | `test_t1_job_events_stream` | SSE progress event streaming `GET /v1/jobs/{id}/events` (R2) | `ORIGINAL_REQUEST.md R2`, `PROJECT.md` | **PASS** |
| **Tier 1** | `test_t1_job_cancellation` | Clean in-memory cancellation `DELETE /v1/jobs/{id}` (R2) | `PROJECT.md § Control Plane ↔ Client` | **PASS** |
| **Tier 1** | `test_t1_job_result_decision_json` | Direct JSON result delivery for Decision/Embedding (R2) | `ORIGINAL_REQUEST.md R2`, `PROJECT.md` | **PENDING (M2)** |
| **Tier 1** | `test_t1_job_result_image_binary` | Direct PNG/JPEG binary result for Image generation (R2) | `ORIGINAL_REQUEST.md R2`, `PROJECT.md` | **PENDING (M2)** |
| **Tier 1** | `test_t1_job_result_audio_binary` | Direct WAV binary result for Speech/Music generation (R2) | `ORIGINAL_REQUEST.md R2`, `PROJECT.md` | **PENDING (M2)** |
| **Tier 1** | `test_t1_job_result_video_binary` | Direct MP4 binary result for Video generation (R2) | `ORIGINAL_REQUEST.md R2`, `PROJECT.md` | **PENDING (M2)** |
| **Tier 1** | `test_t1_job_result_text_stream` | Direct text/SSE stream result for Text/VLM (R2) | `ORIGINAL_REQUEST.md R2`, `PROJECT.md` | **PENDING (M2)** |
| **Tier 2** | `test_t2_invalid_payload_schema` | Rejection of malformed / incomplete job payloads (R2) | `JobSubmissionRequest` schema, HTTP 422/400 | **PASS** |
| **Tier 2** | `test_t2_unknown_model_rejected` | Rejection of non-existent model families (R2) | `model_registry`, HTTP 400 | **PASS** |
| **Tier 2** | `test_t2_unsupported_capability_rejected` | Rejection of tasks not supported by target model (R2) | `CapabilityManifest`, HTTP 400 | **PASS** |
| **Tier 2** | `test_t2_payload_exceeding_1mb_rejected` | Rejection of inline payloads > 1MB (R2) | `threshold.py`, HTTP 413 | **PASS** |
| **Tier 2** | `test_t2_invalid_api_key_variations` | Rejection of malformed/adversarial Bearer tokens (R1) | Constant-time digest comparison, HTTP 401 | **PASS** |
| **Tier 2** | `test_t2_nonexistent_job_id_lookup_result_cancel` | Lookup, delete, and result on missing ID (R2) | In-memory store contract, HTTP 404 | **PASS** |
| **Tier 2** | `test_t2_double_cancel_handling` | Clean idempotent cancellation handling (R2) | `JobStateMachine`, HTTP 200/400 | **PASS** |
| **Tier 2** | `test_t2_cancel_succeeded_job_rejected` | Guard against cancelling terminal jobs (R2) | `JobStateMachine`, HTTP 400 | **PASS** |
| **Tier 2** | `test_t2_error_sanitization_integrity` | Adversarial error sanitization (zero stack leaks) | Security contract, 0 internal path leaks | **PASS** |
| **Tier 3** | `test_t3_submit_stream_download_disk_cleanup` | Submit -> Stream -> Scratch disk output -> Download -> Auto cleanup (R2) | Background task + scratch disk removal | **PENDING (M2)** |
| **Tier 3** | `test_t3_idempotent_job_submission_consistency` | Duplicate submission idempotency (R2) | `InMemoryJobStore` idempotency map | **PASS** |
| **Tier 4** | `test_t4_concurrent_job_submissions_dual_t4_gpus` | Concurrent burst across Dual T4 GPU partitions (R3) | GPU 0 (Image/Text/Decision) vs GPU 1 (Video/Music/Embed) | **PASS** |
| **Tier 4** | `test_t4_worker_disconnect_and_timeout_deterministic_failure` | Deterministic worker disconnect/timeout (R3) | Deterministic failure without duplicate execution | **PASS** |
| **Tier 4** | `test_t4_worker_http_status_reporting_without_lease_token` | Clean HTTP worker progress/complete reporting (R1, R3) | Zero cryptographic lease tokens | **PASS** |

---

## 4. Discovered Implementation Defects & Escalations

During initial execution of the newly authored test suite against the live codebase, the following issues were discovered and escalated:

### Defect 1: Unresolved Imports in `factory/apps/control-api/src/queue/__init__.py`
- **Location**: `factory/apps/control-api/src/queue/__init__.py:9`
- **Symptom**: `ModuleNotFoundError: No module named 'factory.apps.control_api.src.queue.reaper'`
- **Root Cause**: Implementing agent `teamwork_worker_m1_1` deleted `reaper.py`, `postgres.py`, `sqlite.py`, and `redis_broker.py`, but `src/queue/__init__.py` still contains imports for `DeadJobReaper`, `SqliteJobRepository`, `PostgresJobRepository`, and `RedisQueueBroker`.
- **Remediation Required**: Implementing agent must remove these deleted module references from `src/queue/__init__.py`.

### Defect 2: Missing Route `GET /v1/jobs/{id}/result`
- **Location**: `factory/apps/control-api/src/api/routers/jobs.py`
- **Symptom**: `GET /v1/jobs/{id}/result` returns 404 Not Found.
- **Root Cause**: Milestone 2 (Phase 2 API & Artifact Delivery) has not yet been implemented.
- **Remediation Required**: Implementing agent for Milestone 2 must implement `GET /v1/jobs/{id}/result` supporting direct JSON, binary image/audio/video, and text streaming with automatic scratch disk cleanup.
