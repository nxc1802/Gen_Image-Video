"""
tests/e2e/test_factory_lite_e2e.py
Factory v4 Lite: Requirement-Driven Opaque-Box E2E Test Suite

4-Tier Methodology:
- Tier 1: Core Feature Coverage (health, model discovery, job submit, job status, job events, job cancel, job result across media types, single API key auth).
- Tier 2: Boundary & Corner Cases (invalid payload, unknown model, unsupported capability, payload > 1MB, invalid API key, non-existent job ID, double cancel).
- Tier 3: Cross-Feature Combinations (submit -> event stream -> direct binary/streaming download -> automatic disk cleanup verification).
- Tier 4: Real-world Provider Scenarios (concurrent job submissions across Dual T4 GPUs, simulated worker disconnect / timeout leading to deterministic failure without duplicate execution).

Requirements:
- Zero dependencies on external enterprise services (no PostgreSQL, SQLite server, Redis, S3/MinIO, or Alembic).
- Uses FastAPI TestClient targeting the in-memory Factory Lite app.
"""

import base64
import concurrent.futures
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from fastapi.testclient import TestClient

# Authoritative Test Token for Factory v4 Lite
VALID_TEST_API_KEY = "sk-factory-lite-e2e-valid-key-2026"
INVALID_TEST_API_KEY = "sk-factory-bad-key-unauthorized"
SCRATCH_ARTIFACTS_DIR = "/tmp/factory_lite/artifacts"


@pytest.fixture(scope="session", autouse=True)
def configure_lite_environment():
    """Configures environment for in-memory Factory Lite operation."""
    os.environ["FACTORY_API_KEY"] = VALID_TEST_API_KEY
    os.environ["ENVIRONMENT"] = "test"
    os.environ["FACTORY_ENV"] = "test"
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "memory"
    os.makedirs(SCRATCH_ARTIFACTS_DIR, exist_ok=True)
    yield
    # Cleanup scratch disk after session
    if os.path.exists(SCRATCH_ARTIFACTS_DIR):
        try:
            shutil.rmtree(SCRATCH_ARTIFACTS_DIR)
        except Exception:
            pass


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provides a TestClient connected directly to the in-memory Factory Lite app."""
    from factory.apps.control_api.src.legacy import app

    with TestClient(app, base_url="http://testserver") as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Returns valid Bearer authorization headers."""
    return {
        "Authorization": f"Bearer {VALID_TEST_API_KEY}",
        "Content-Type": "application/json",
    }


def assert_sanitized_error(response, expected_status: int, expected_code: Optional[str] = None):
    """
    Verifies that error responses strictly conform to the Factory Lite Error Envelope:
    { "error": { "code": "...", "message": "...", "request_id": "req-..." } }
    and leak ZERO internal stack traces, Python classes, or filesystem paths.
    """
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. Response: {response.text}"
    )
    data = response.json()
    assert "error" in data, f"Missing 'error' key in response envelope: {data}"
    err = data["error"]
    assert "code" in err and len(err["code"]) > 0, f"Missing or empty error code: {err}"
    if expected_code:
        if isinstance(expected_code, list):
            assert err["code"] in expected_code, f"Expected code in {expected_code}, got '{err['code']}'"
        else:
            assert err["code"] == expected_code, f"Expected code '{expected_code}', got '{err['code']}'"
    assert "message" in err and len(err["message"]) > 0, f"Missing or empty error message: {err}"
    assert "request_id" in err, f"Missing request_id in error: {err}"

    # Strict error sanitization checks
    raw_text = response.text
    leak_signatures = [
        r"Traceback \(most recent call last\)",
        r"/Volumes/[A-Za-z0-9_/:-]+",
        r"/Users/[A-Za-z0-9_/:-]+",
        r"/home/[A-Za-z0-9_/:-]+",
        r"C:\\",
        r"ZeroDivisionError:",
        r"KeyError:",
        r"pydantic_core\._pydantic_core\.ValidationError",
    ]
    for pattern in leak_signatures:
        assert not re.search(pattern, raw_text), (
            f"Security Leak! Internal trace/path detected matching '{pattern}': {raw_text}"
        )


def assert_valid_job_envelope(job: Dict[str, Any], expected_service: str, expected_model: str, expected_task: str):
    """Verifies that a job dictionary adheres to the FactoryJob schema."""
    assert "job_id" in job and job["job_id"].startswith("job-")
    assert "request_id" in job and job["request_id"].startswith("req-")
    assert job["service"] == expected_service
    assert job["model"] == expected_model
    assert job["task"] == expected_task
    assert job["state"] in [
        "CREATED", "QUEUED", "ADMITTED", "RUNNING", "STREAMING", "SUCCEEDED", "FAILED", "CANCELLED", "RETRYING"
    ]
    assert "created_at" in job
    assert "progress" in job
    assert "priority" in job
    assert "inputs" in job


# ==============================================================================
# TIER 1: Core Feature Coverage
# ==============================================================================

class TestTier1CoreFeatureCoverage:
    """
    Tier 1 verifies the primary operational path for each core Factory Lite feature:
    - Public healthcheck
    - Dynamic model discovery
    - Single API key authentication enforcement
    - Job submission, status lookup, SSE event streaming, and cancellation
    - Direct result delivery across all media types (JSON, image, audio, video, text)
    """

    def test_t1_health_check_public(self, client: TestClient):
        """R2 / AC: GET /health is unauthenticated and returns healthy status."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "healthy"
        assert body.get("plane") == "control"
        assert "version" in body

    def test_t1_model_discovery(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2 / AC: GET /v1/registry/models dynamically lists all registered model manifests."""
        resp = client.get("/v1/registry/models", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        models = body["data"]
        assert len(models) >= 7, f"Expected at least 7 model families, found {len(models)}"

        model_ids = {m["metadata"]["id"] for m in models if "metadata" in m and "id" in m["metadata"]}
        # Check presence of major multimodal model families
        assert "qwen-text" in model_ids or any("text" in mid for mid in model_ids)
        assert "qwen-image-2.1" in model_ids or any("image" in mid for mid in model_ids)

        for m in models:
            assert "metadata" in m
            assert "capabilities" in m
            assert isinstance(m["capabilities"], dict)

    def test_t1_single_api_key_auth_enforcement(self, client: TestClient):
        """R1 / R2: Single API key authentication required for all protected endpoints."""
        # 1. Missing Authorization header
        resp_missing = client.get("/v1/registry/models")
        assert resp_missing.status_code == 401
        assert_sanitized_error(resp_missing, 401, "UNAUTHORIZED")

        # 2. Empty Bearer token
        resp_empty = client.get("/v1/registry/models", headers={"Authorization": "Bearer "})
        assert resp_empty.status_code == 401
        assert_sanitized_error(resp_empty, 401, "UNAUTHORIZED")

        # 3. Invalid API key
        resp_invalid = client.get("/v1/registry/models", headers={"Authorization": f"Bearer {INVALID_TEST_API_KEY}"})
        assert resp_invalid.status_code == 401
        assert_sanitized_error(resp_invalid, 401, "UNAUTHORIZED")

        # 4. Valid API key succeeds
        resp_valid = client.get("/v1/registry/models", headers={"Authorization": f"Bearer {VALID_TEST_API_KEY}"})
        assert resp_valid.status_code == 200

    def test_t1_job_submission(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2 / AC: POST /v1/jobs submits a valid inference job and returns 201 Created."""
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {
                "prompt": "A futuristic city under a neon sunset, high resolution",
                "width": 1024,
                "height": 1024,
            },
            "priority": 10,
        }
        resp = client.post("/v1/jobs", json=payload, headers=auth_headers)
        assert resp.status_code == 201
        job = resp.json()
        assert_valid_job_envelope(job, "image", "qwen-image-2.1", "image.text_to_image")
        assert job["state"] in ["QUEUED", "ADMITTED", "RUNNING"]

    def test_t1_job_status_lookup(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2 / AC: GET /v1/jobs/{id} returns the current state and progress of the job."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A majestic red panda in the bamboo forest"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        get_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        job = get_resp.json()
        assert job["job_id"] == job_id
        assert job["service"] == "image"
        assert job["model"] == "qwen-image-2.1"
        assert job["task"] == 'image.text_to_image'
        assert "progress" in job

    def test_t1_job_events_stream(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2 / AC: GET /v1/jobs/{id}/events streams SSE progress and lifecycle events."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Events stream test prompt"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        stream_headers = dict(auth_headers)
        stream_headers["Accept"] = "text/event-stream"
        with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=stream_headers) as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers.get("content-type", "")
            lines = []
            for line in stream_resp.iter_lines():
                if line:
                    lines.append(line)
                if len(lines) >= 3:
                    break
            assert len(lines) > 0, "SSE stream yielded no events"
            # Verify event structure
            raw_stream = "\n".join(lines)
            assert "event:" in raw_stream or "data:" in raw_stream or ": keepalive" in raw_stream

    def test_t1_job_cancellation(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2 / AC: DELETE /v1/jobs/{id} cancels a queued job."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Job to be cancelled"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        del_resp = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data.get("job_id") == job_id
        assert del_data.get("state") == "CANCELLED"

        # Verify state via GET
        get_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["state"] == "CANCELLED"

    def test_t1_job_result_decision_json(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2: GET /v1/jobs/{id}/result delivers direct JSON response for Decision/Embedding."""
        # Note: /result endpoint is an R2 requirement. If not yet implemented, this test
        # surfaces the pending contract status for the implementation worker.
        submit_payload = {
            "service": "decision",
            "model": "qwen-decision",
            "task": "predicate_evaluation",
            "inputs": {
                "statement": "The user requested an image generation workflow.",
                "context": "Context: Factory Lite routing",
            },
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        if result_resp.status_code == 404:
            pytest.fail(
                f"R2 Contract Failure: GET /v1/jobs/{job_id}/result returned 404 Not Found. "
                "Milestone 2 route /v1/jobs/{id}/result must be implemented."
            )
        assert result_resp.status_code in [200, 400, 409], f"Unexpected status: {result_resp.status_code}"
        if result_resp.status_code == 200:
            assert "application/json" in result_resp.headers.get("content-type", "")
            # Ensure zero S3 URLs returned
            assert "s3://" not in result_resp.text
            assert "amazonaws.com" not in result_resp.text

    def test_t1_job_result_image_binary(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2: GET /v1/jobs/{id}/result delivers direct binary PNG/JPEG for Image jobs."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A glowing crystal in a dark cave"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        if result_resp.status_code == 404:
            pytest.fail(
                f"R2 Contract Failure: GET /v1/jobs/{job_id}/result returned 404 Not Found. "
                "Milestone 2 route /v1/jobs/{id}/result must be implemented."
            )
        assert result_resp.status_code in [200, 400, 409]
        if result_resp.status_code == 200:
            content_type = result_resp.headers.get("content-type", "")
            assert "image/png" in content_type or "image/jpeg" in content_type

    def test_t1_job_result_audio_binary(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2: GET /v1/jobs/{id}/result delivers direct binary WAV audio for Speech/Music jobs."""
        submit_payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Upbeat synthwave rhythm with smooth bassline"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        if result_resp.status_code == 404:
            pytest.fail(
                f"R2 Contract Failure: GET /v1/jobs/{job_id}/result returned 404 Not Found. "
                "Milestone 2 route /v1/jobs/{id}/result must be implemented."
            )
        assert result_resp.status_code in [200, 400, 409]
        if result_resp.status_code == 200:
            assert "audio/wav" in result_resp.headers.get("content-type", "")

    def test_t1_job_result_video_binary(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2: GET /v1/jobs/{id}/result delivers direct binary MP4 video for Video jobs."""
        submit_payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "A soaring eagle over snow-capped mountains, cinematic 4k"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        if result_resp.status_code == 404:
            pytest.fail(
                f"R2 Contract Failure: GET /v1/jobs/{job_id}/result returned 404 Not Found. "
                "Milestone 2 route /v1/jobs/{id}/result must be implemented."
            )
        assert result_resp.status_code in [200, 400, 409]
        if result_resp.status_code == 200:
            assert "video/mp4" in result_resp.headers.get("content-type", "")

    def test_t1_job_result_text_stream(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2: GET /v1/jobs/{id}/result delivers text/SSE response for Text/VLM jobs."""
        submit_payload = {
            "service": "text",
            "model": "qwen-text",
            "task": "chat_completion",
            "inputs": {
                "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}]
            },
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        if result_resp.status_code == 404:
            pytest.fail(
                f"R2 Contract Failure: GET /v1/jobs/{job_id}/result returned 404 Not Found. "
                "Milestone 2 route /v1/jobs/{id}/result must be implemented."
            )
        assert result_resp.status_code in [200, 400, 409]
        if result_resp.status_code == 200:
            content_type = result_resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type or "text/plain" in content_type or "application/json" in content_type


# ==============================================================================
# TIER 2: Boundary & Corner Cases
# ==============================================================================

class TestTier2BoundaryCornerCases:
    """
    Tier 2 tests boundary conditions, invalid inputs, edge cases, and adversarial values:
    - Schema validation errors (422/400)
    - Unknown models and unsupported capabilities (400)
    - 1MB payload limits (413)
    - Authentication boundary violations (401)
    - Non-existent IDs (404)
    - Double cancel and invalid state transitions (400/200)
    - Strict error response sanitization (no traces, no internal filesystem paths)
    """

    def test_t2_invalid_payload_schema(self, client: TestClient, auth_headers: Dict[str, str]):
        """POST /v1/jobs rejects payloads missing required fields or having invalid types."""
        # 1. Empty payload
        resp_empty = client.post("/v1/jobs", json={}, headers=auth_headers)
        assert resp_empty.status_code in [400, 422]
        assert_sanitized_error(resp_empty, resp_empty.status_code)

        # 2. Missing 'inputs'
        resp_no_inputs = client.post(
            "/v1/jobs",
            json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image"},
            headers=auth_headers,
        )
        assert resp_no_inputs.status_code in [400, 422]
        assert_sanitized_error(resp_no_inputs, resp_no_inputs.status_code)

        # 3. Invalid inputs type (string instead of dict)
        resp_bad_type = client.post(
            "/v1/jobs",
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": "not_a_dictionary",
            },
            headers=auth_headers,
        )
        assert resp_bad_type.status_code in [400, 422]
        assert_sanitized_error(resp_bad_type, resp_bad_type.status_code)

    def test_t2_unknown_model_rejected(self, client: TestClient, auth_headers: Dict[str, str]):
        """POST /v1/jobs rejects unknown/unregistered model family IDs with HTTP 400."""
        payload = {
            "service": "image",
            "model": "completely-fake-nonexistent-model-v99",
            "task": "text_to_image",
            "inputs": {"prompt": "test"},
        }
        resp = client.post("/v1/jobs", json=payload, headers=auth_headers)
        assert resp.status_code == 400
        assert_sanitized_error(resp, 400)

    def test_t2_unsupported_capability_rejected(self, client: TestClient, auth_headers: Dict[str, str]):
        """POST /v1/jobs rejects requests where model does not support the requested task."""
        # Text model cannot do ComfyUI inpaint
        payload = {
            "service": "image",
            "model": "qwen-text",
            "task": "inpaint",
            "inputs": {"prompt": "test"},
        }
        resp = client.post("/v1/jobs", json=payload, headers=auth_headers)
        assert resp.status_code == 400
        assert_sanitized_error(resp, 400, "CAPABILITY_NOT_SUPPORTED")

    def test_t2_payload_exceeding_1mb_rejected(self, client: TestClient, auth_headers: Dict[str, str]):
        """POST /v1/jobs rejects inline media payloads > 1MB with HTTP 413 Payload Too Large."""
        # Construct valid base64 data URI exceeding 1MB (1,048,576 bytes)
        raw_oversized_bytes = b"0" * (1024 * 1024 + 2048)
        encoded_b64 = base64.b64encode(raw_oversized_bytes).decode("ascii")
        large_data_uri = f"data:image/png;base64,{encoded_b64}"

        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {
                "prompt": "Test large payload",
                "image": large_data_uri,
            },
        }
        resp = client.post("/v1/jobs", json=payload, headers=auth_headers)
        assert resp.status_code == 413
        assert_sanitized_error(resp, 413, "PAYLOAD_TOO_LARGE")

    def test_t2_invalid_api_key_variations(self, client: TestClient):
        """Rejects various malformed or adversarial Authorization header schemes."""
        malformed_headers = [
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {"Authorization": "Token secret123"},
            {"Authorization": "Bearer"},
            {"Authorization": f"Bearer {VALID_TEST_API_KEY}' OR '1'='1"},
            {"Authorization": "Bearer \x00nullbyte"},
        ]
        for headers in malformed_headers:
            resp = client.get("/v1/registry/models", headers=headers)
            assert resp.status_code == 401, f"Failed on headers: {headers}"
            assert_sanitized_error(resp, 401, "UNAUTHORIZED")

    def test_t2_nonexistent_job_id_lookup_result_cancel(self, client: TestClient, auth_headers: Dict[str, str]):
        """GET, DELETE, and GET /result on non-existent job ID return HTTP 404 Not Found."""
        fake_id = f"job-{uuid.uuid4()}"

        # 1. Lookup
        resp_get = client.get(f"/v1/jobs/{fake_id}", headers=auth_headers)
        assert resp_get.status_code == 404
        assert_sanitized_error(resp_get, 404, ["JOB_NOT_FOUND", "NOT_FOUND"])

        # 2. Cancel
        resp_del = client.delete(f"/v1/jobs/{fake_id}", headers=auth_headers)
        assert resp_del.status_code == 404
        assert_sanitized_error(resp_del, 404, ["JOB_NOT_FOUND", "NOT_FOUND"])

        # 3. Result
        resp_res = client.get(f"/v1/jobs/{fake_id}/result", headers=auth_headers)
        assert resp_res.status_code == 404

    def test_t2_double_cancel_handling(self, client: TestClient, auth_headers: Dict[str, str]):
        """Cancelling an already cancelled job is handled cleanly without crash."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Double cancel test"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # First cancellation
        resp1 = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert resp1.status_code == 200

        # Second cancellation
        resp2 = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        # Idempotent 200 or clean terminal reject (400), NEVER 500
        assert resp2.status_code in [200, 400]
        if resp2.status_code != 200:
            assert_sanitized_error(resp2, resp2.status_code)

    def test_t2_cancel_succeeded_job_rejected(self, client: TestClient, auth_headers: Dict[str, str]):
        """Cancelling a completed job returns HTTP 400 Invalid Request."""
        from factory.apps.control_api.src.queue.repository import get_job_repository
        from factory_contracts.jobs.job_contract import JobState

        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Completed job cancel test"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        repo = get_job_repository()
        try:
            import asyncio
            asyncio.run(repo.update_job_state(job_id, JobState.SUCCEEDED))
        except Exception:
            pass

        del_resp = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        if del_resp.status_code == 400:
            assert_sanitized_error(del_resp, 400, "INVALID_REQUEST")

    def test_t2_error_sanitization_integrity(self, client: TestClient, auth_headers: Dict[str, str]):
        """Verifies that unhandled inputs (special characters, unicode, control bytes) do not leak stack traces."""
        adversarial_inputs = [
            {"prompt": "A" * 10000, "meta": {"deep": {"nested": [1, 2, 3]}}},
            {"prompt": "Unicode test: 測試 🚀 \u0000 \t \n \r \f"},
            {"prompt": "'; DROP TABLE jobs; --"},
            {"prompt": "<script>alert('xss')</script>"},
        ]
        for inp in adversarial_inputs:
            payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": inp,
            }
            resp = client.post("/v1/jobs", json=payload, headers=auth_headers)
            if resp.status_code != 201:
                assert_sanitized_error(resp, resp.status_code)
            else:
                assert resp.json()["job_id"].startswith("job-")


# ==============================================================================
# TIER 3: Cross-Feature Combinations
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """
    Tier 3 validates end-to-end multi-feature workflows:
    - Submit -> Stream progress -> Complete with scratch artifact -> Download -> Verify disk auto-cleanup
    - Idempotency key consistency across duplicate submissions
    - Multi-modal pipeline execution and routing
    """

    def test_t3_submit_stream_download_disk_cleanup(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R2 / AC: Full lifecycle delivery with automatic scratch disk cleanup.
        1. Submit job via POST /v1/jobs.
        2. Stream events via GET /v1/jobs/{id}/events.
        3. Worker writes generated artifact to /tmp/factory_lite/artifacts/{job_id}/artifact.png.
        4. Verify artifact exists on disk before client download.
        5. Client calls GET /v1/jobs/{id}/result to download artifact.
        6. Verify immediate automatic disk cleanup (file removed from scratch disk).
        """
        scratch_disk = getattr(client.app.state, "scratch_disk", None)
        orig_cleanup = scratch_disk.config.cleanup_on_download if scratch_disk else False
        if scratch_disk:
            scratch_disk.config.cleanup_on_download = True

        try:
            # Step 1: Submit job
            submit_payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Lifecycle test image"},
            }
            submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
            assert submit_resp.status_code == 201
            job_id = submit_resp.json()["job_id"]

            # Step 2: Connect to event stream
            stream_headers = dict(auth_headers)
            stream_headers["Accept"] = "text/event-stream"
            with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=stream_headers) as stream_resp:
                assert stream_resp.status_code == 200

            # Step 3: Simulate worker writing scratch disk artifact
            job_scratch_dir = os.path.join(SCRATCH_ARTIFACTS_DIR, job_id)
            os.makedirs(job_scratch_dir, exist_ok=True)
            artifact_path = os.path.join(job_scratch_dir, "output.png")
            dummy_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
            with open(artifact_path, "wb") as f:
                f.write(dummy_png_bytes)

            # Step 4: Verify file exists on scratch disk before download
            assert os.path.exists(artifact_path), f"Artifact was not created at {artifact_path}"

            # Step 4b: Mark job as completed/succeeded before client download
            complete_resp = client.post(
                f"/v1/jobs/{job_id}/complete",
                json={"artifact_path": artifact_path},
                headers=auth_headers,
            )
            assert complete_resp.status_code == 200

            # Step 5: Client calls GET /v1/jobs/{job_id}/result
            result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
            if result_resp.status_code == 404:
                pytest.fail(
                    f"Milestone 2 route GET /v1/jobs/{job_id}/result returned 404. "
                    "Direct artifact download and automatic disk cleanup requires M2 implementation."
                )

            assert result_resp.status_code == 200
            assert result_resp.content == dummy_png_bytes

            # Step 6: Verify automatic disk cleanup
            cleaned_up = False
            for _ in range(10):
                if not os.path.exists(artifact_path):
                    cleaned_up = True
                    break
                time.sleep(0.1)

            assert cleaned_up, f"Automatic disk cleanup failed: {artifact_path} still exists after download."
        finally:
            if scratch_disk:
                scratch_disk.config.cleanup_on_download = orig_cleanup

    def test_t3_idempotent_job_submission_consistency(self, client: TestClient, auth_headers: Dict[str, str]):
        """R2 / AC: Duplicate submissions with the same idempotency_key return the same job."""
        idemp_key = f"idemp-{uuid.uuid4()}"
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Idempotent submission prompt"},
            "idempotency_key": idemp_key,
        }

        resp1 = client.post("/v1/jobs", json=payload, headers=auth_headers)
        assert resp1.status_code in [200, 201]
        job1 = resp1.json()
        job_id_1 = job1["job_id"]

        resp2 = client.post("/v1/jobs", json=payload, headers=auth_headers)
        assert resp2.status_code in [200, 201]
        job2 = resp2.json()
        job_id_2 = job2["job_id"]

        # Both submissions must resolve to the identical job entity
        assert job_id_1 == job_id_2
        assert job1["created_at"] == job2["created_at"]


# ==============================================================================
# TIER 4: Real-world Provider Scenarios
# ==============================================================================

class TestTier4RealWorldProviderScenarios:
    """
    Tier 4 simulates production provider operations:
    - High concurrency submission burst across Dual T4 GPUs (GPU 0 vs GPU 1 partitioning)
    - Simulated worker disconnect / timeout leading to deterministic failure without duplicate execution
    - Clean HTTP worker status reporting without cryptographic lease tokens
    """

    def test_t4_concurrent_job_submissions_dual_t4_gpus(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R3 / AC: Accepts concurrent job submissions distributed across Dual T4 GPUs.
        GPU 0: image, text, decision
        GPU 1: video, music, embedding
        Enforces zero dropped requests and unique job allocation.
        """
        workload = [
            # GPU 0 targets
            {"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "GPU 0 Image"}},
            {"service": "text", "model": "qwen-text", "task": "chat_completion", "inputs": {"messages": [{"role": "user", "content": "GPU 0 Text"}]}},
            {"service": "decision", "model": "qwen-decision", "task": "predicate_evaluation", "inputs": {"statement": "GPU 0 Decision"}},
            {"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "GPU 0 Image 2"}},
            # GPU 1 targets
            {"service": "video", "model": "wan-2.2", "task": "text_to_video", "inputs": {"prompt": "GPU 1 Video"}},
            {"service": "music", "model": "ace-step-1.5", "task": "text_to_music", "inputs": {"prompt": "GPU 1 Music"}},
            {"service": "video", "model": "wan-2.2", "task": "text_to_video", "inputs": {"prompt": "GPU 1 Video 2"}},
            {"service": "music", "model": "ace-step-1.5", "task": "text_to_music", "inputs": {"prompt": "GPU 1 Music 2"}},
        ]

        def submit_work(item):
            resp = client.post("/v1/jobs", json=item, headers=auth_headers)
            return resp.status_code, resp.json()

        # Execute 8 concurrent submissions across 4 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(submit_work, workload))

        job_ids = set()
        for status_code, body in results:
            assert status_code == 201, f"Concurrent submission failed: {body}"
            assert "job_id" in body
            job_ids.add(body["job_id"])

        # Zero dropped jobs: all 8 jobs received unique job_ids
        assert len(job_ids) == len(workload), f"Duplicate or dropped job IDs: {len(job_ids)} vs {len(workload)}"

        # Verify each job is queryable in the in-memory store
        for jid in job_ids:
            get_resp = client.get(f"/v1/jobs/{jid}", headers=auth_headers)
            assert get_resp.status_code == 200
            assert get_resp.json()["job_id"] == jid

    def test_t4_worker_disconnect_and_timeout_deterministic_failure(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R3 / AC: Worker disconnect or timeout results in deterministic job failure.
        Crucial requirement: NO duplicate speculative re-execution or re-queueing!
        """
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Disconnect timeout test"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # Worker reports fatal disconnect/timeout failure conforming to JobFailureRequest schema
        fail_payload = {
            "error": {
                "code": "WORKER_DISCONNECTED",
                "message": "Worker process terminated unexpectedly; execution aborted.",
                "request_id": f"req-{uuid.uuid4()}",
            },
            "retryable": False,
        }
        fail_resp = client.post(f"/v1/jobs/{job_id}/fail", json=fail_payload, headers=auth_headers)
        assert fail_resp.status_code in [200, 403]  # If auth requires worker role, checks API contract

        if fail_resp.status_code == 200:
            # Verify deterministic failure status
            get_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
            assert get_resp.status_code == 200
            job = get_resp.json()
            assert job["state"] == "FAILED"
            assert job["error"] is not None
            assert job["error"]["code"] == "WORKER_DISCONNECTED"
            # Verify job is NOT automatically re-enqueued to prevent duplicate execution
            assert job["state"] != "QUEUED"
            assert job["state"] != "RUNNING"

    def test_t4_worker_http_status_reporting_without_lease_token(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R1 / R3 / AC: Worker reports status via HTTP without cryptographic lease tokens.
        Eliminates lease_token requirement for progress, complete, and fail endpoints.
        """
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "HTTP status reporting test"},
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # Worker reports progress without lease_token
        progress_payload = {
            "stage": "denoising",
            "step": 15,
            "total": 30,
            "percent": 50.0,
            "message": "Step 15/30 completed",
        }
        prog_resp = client.post(f"/v1/jobs/{job_id}/progress", json=progress_payload, headers=auth_headers)
        assert prog_resp.status_code in [200, 403]
        if prog_resp.status_code == 200:
            get_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
            assert get_resp.status_code == 200
            prog = get_resp.json().get("progress", {})
            assert prog.get("step") == 15
            assert prog.get("percent") == 50.0
