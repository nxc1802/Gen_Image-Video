"""
tests/adversarial/test_adversarial_m2_delivery_gate.py
Adversarial Stress & Empirical Challenge Suite for Milestone M2:
Phase 2 API & Artifact Delivery Gate Verification.

Challenge Vectors:
1. Multi-Modality Result Verification (Decision JSON, Embedding JSON, Image PNG/JPEG,
   Audio WAV/MP3, Video MP4, Text SSE/Plain/JSON) - zero S3 URLs.
2. Concurrent Downloads & Race Condition Stress (Rapid parallel GETs on same artifact,
   zero 500s, BackgroundTask unlinking race safety, zero residual disk footprint).
3. Edge Case States & Error Mapping:
   - Missing job ID -> 404 UnifiedError
   - Cancelled job -> 409 Conflict (JOB_CANCELLED)
   - Queued/Running/Streaming job -> 409 Conflict with progress
   - Failed job: WORKER_DISCONNECTED -> 502, GATEWAY_TIMEOUT/TIMEOUT -> 504,
     MODEL/SERVICE_UNAVAILABLE -> 503, other -> 400
   - Succeeded media job with missing artifact -> 404 UnifiedError
4. Security & Boundary Probes (Missing/Invalid Bearer token -> 401, path traversal in job_id -> sanitized rejection).
5. Scratch Disk Sweeper & LRU Quota Bounds under pressure.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from factory_control_api import app
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
)
from factory_contracts.jobs.worker_contract import JobCompletionRequest, JobFailureRequest
from factory_contracts.errors import ErrorDetail, ErrorCode
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.storage.scratch_disk import (
    ScratchDiskConfig,
    ScratchDiskManager,
)

TEST_API_KEY = "test-sk-adversarial-challenger-m2-2026"
SCRATCH_DIR = "/tmp/factory_lite/artifacts"


@pytest.fixture(autouse=True)
def configure_test_env(monkeypatch):
    """Sets required environment variables for Factory Lite in-memory mode."""
    monkeypatch.setenv("FACTORY_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("FACTORY_ENV", "test")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "memory")
    monkeypatch.setenv("SCRATCH_ARTIFACTS_DIR", SCRATCH_DIR)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    yield
    # Cleanup scratch directory after each test
    if os.path.exists(SCRATCH_DIR):
        try:
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
        except Exception:
            pass


@pytest.fixture
def client() -> TestClient:
    """Provides a TestClient connected to the FastAPI app within lifespan."""
    with TestClient(app, base_url="http://testserver") as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {TEST_API_KEY}",
        "Content-Type": "application/json",
    }


def assert_sanitized_error(response, expected_status: int, expected_code: Optional[str] = None):
    """Verifies sanitized UnifiedErrorResponse envelope and zero internal path leaks."""
    assert response.status_code == expected_status, (
        f"Expected HTTP {expected_status}, got {response.status_code}. Response: {response.text}"
    )
    data = response.json()
    assert "error" in data, f"Missing 'error' key in response envelope: {data}"
    err = data["error"]
    assert "code" in err and len(err["code"]) > 0, f"Missing or empty error code: {err}"
    if expected_code:
        assert err["code"] == expected_code, f"Expected code '{expected_code}', got '{err['code']}'"
    assert "message" in err and len(err["message"]) > 0
    assert "request_id" in err

    # Verify zero stack trace or internal filesystem path leaks
    raw_text = response.text
    assert "Traceback" not in raw_text
    assert "/Volumes/" not in raw_text
    assert "/Users/" not in raw_text


# ==============================================================================
# VECTOR 1: Multi-Modality Delivery Across All Modalities
# ==============================================================================

class TestVector1MultiModalityDelivery:
    """Stress-tests and verifies result delivery across all media modalities with zero S3 URLs."""

    def test_decision_json_modality(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        Decision jobs return direct JSON with choices and zero S3 references.
        Adversarial probe: Verifies whether JobCompletionRequest(result=...) payload is preserved
        or dropped in favor of hardcoded/simulated fallback.
        """
        submit_payload = {
            "service": "decision",
            "model": "qwen-decision",
            "task": "predicate_evaluation",
            "inputs": {
                "statement": "The model is certified on Dual T4 GPUs.",
                "context": "Empirical certification results.",
            },
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # Worker completes decision job with in-memory result
        manager: JobLifecycleManager = app.state.job_manager
        decision_result = {
            "statement": "The model is certified on Dual T4 GPUs.",
            "decision": True,
            "confidence": 0.985,
            "choices": [{"index": 0, "decision": True, "confidence": 0.985}],
        }
        asyncio.run(
            manager.complete_job(
                job_id=job_id,
                req=JobCompletionRequest(result=decision_result),
                caller_role="admin",
            )
        )

        # Query GET /v1/jobs/{job_id}/result
        res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert res.status_code == 200
        assert "application/json" in res.headers.get("content-type", "")
        body = res.json()
        assert body["job_id"] == job_id

        # Strict Zero S3 Check
        assert "s3://" not in res.text
        assert "amazonaws.com" not in res.text

        # Real result verification (probing whether complete_job dropped result)
        assert "decision" in body, (
            f"Defect: complete_job discarded worker result! Server returned fallback choices: {body}"
        )
        assert body["decision"] is True
        assert body["confidence"] == 0.985

    def test_embedding_json_modality(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        Embedding jobs return direct JSON with unit vector and zero S3 references.
        Adversarial probe: Verifies whether real vector passed to complete_job is returned,
        or if the server returns fake zero vectors ([0.0] * 1024).
        """
        submit_payload = {
            "service": "embedding",
            "model": "qwen-embedding",
            "task": "text_embedding",
            "inputs": {"text": "Empirical challenge for embedding delivery"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # Worker completes embedding job with 1024-dim vector
        manager: JobLifecycleManager = app.state.job_manager
        vector = [0.03125] * 1024
        embed_result = {
            "embedding": vector,
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
        asyncio.run(
            manager.complete_job(
                job_id=job_id,
                req=JobCompletionRequest(result=embed_result),
                caller_role="admin",
            )
        )

        res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert res.status_code == 200
        assert "application/json" in res.headers.get("content-type", "")
        body = res.json()
        assert body["job_id"] == job_id

        # Strict Zero S3 Check
        assert "s3://" not in res.text
        assert "amazonaws.com" not in res.text

        # Real vector verification (probing against fake zero-vector fallback)
        assert body["embedding"] == vector, (
            f"Defect: complete_job discarded worker embedding! Returned fake/zero vector instead of real payload."
        )

    def test_image_png_and_jpeg_modalities(self, client: TestClient, auth_headers: Dict[str, str]):
        """Image jobs return direct binary image/png and image/jpeg with inline Content-Disposition."""
        for ext, mime in [("png", "image/png"), ("jpg", "image/jpeg")]:
            submit_payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": f"Test image generation {ext}"},
            }
            resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
            assert resp.status_code == 201
            job_id = resp.json()["job_id"]

            job_dir = Path(SCRATCH_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            dummy_data = f"DUMMY_IMAGE_{ext.upper()}_DATA_{uuid.uuid4()}".encode("utf-8")
            file_path = job_dir / f"generated_output.{ext}"
            file_path.write_bytes(dummy_data)

            comp_resp = client.post(
                f"/v1/jobs/{job_id}/complete",
                json={"artifact_path": str(file_path)},
                headers=auth_headers,
            )
            assert comp_resp.status_code == 200

            res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
            assert res.status_code == 200
            assert mime in res.headers.get("content-type", "")
            assert "inline" in res.headers.get("content-disposition", "")
            assert res.content == dummy_data
            assert "s3://" not in res.headers.get("content-disposition", "")

    def test_audio_wav_and_mp3_modalities(self, client: TestClient, auth_headers: Dict[str, str]):
        """Speech/Music jobs return direct binary audio/wav and audio/mpeg."""
        for ext, mime in [("wav", "audio/wav"), ("mp3", "audio/mpeg")]:
            submit_payload = {
                "service": "music",
                "model": "ace-step-1.5",
                "task": "text_to_music",
                "inputs": {"prompt": f"Test music generation {ext}"},
            }
            resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
            assert resp.status_code == 201
            job_id = resp.json()["job_id"]

            job_dir = Path(SCRATCH_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            dummy_audio = f"RIFF_HEADER_DUMMY_AUDIO_{ext}_{uuid.uuid4()}".encode("utf-8")
            file_path = job_dir / f"track.{ext}"
            file_path.write_bytes(dummy_audio)

            comp_resp = client.post(
                f"/v1/jobs/{job_id}/complete",
                json={"artifact_path": str(file_path)},
                headers=auth_headers,
            )
            assert comp_resp.status_code == 200

            res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
            assert res.status_code == 200
            assert mime in res.headers.get("content-type", "")
            assert "inline" in res.headers.get("content-disposition", "")
            assert res.content == dummy_audio

    def test_video_mp4_modality(self, client: TestClient, auth_headers: Dict[str, str]):
        """Video jobs return video/mp4 with Accept-Ranges: bytes and attachment Content-Disposition."""
        submit_payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "A flowing waterfall under sunset"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        job_dir = Path(SCRATCH_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dummy_video = b"\x00\x00\x00\x18ftypmp42" + os.urandom(2048)
        file_path = job_dir / "render.mp4"
        file_path.write_bytes(dummy_video)

        comp_resp = client.post(
            f"/v1/jobs/{job_id}/complete",
            json={"artifact_path": str(file_path)},
            headers=auth_headers,
        )
        assert comp_resp.status_code == 200

        res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert res.status_code == 200
        assert "video/mp4" in res.headers.get("content-type", "")
        assert res.headers.get("accept-ranges") == "bytes"
        assert "attachment" in res.headers.get("content-disposition", "")
        assert f"{job_id}.mp4" in res.headers.get("content-disposition", "")
        assert res.content == dummy_video

    def test_text_vlm_streaming_and_plain_modalities(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        Text/VLM supports JSON, text/plain, and text/event-stream without S3 URLs.
        Adversarial probe: Verifies whether text result passed to complete_job is returned,
        or if empty string is returned due to dropped job.result.
        """
        submit_payload = {
            "service": "text",
            "model": "qwen-text",
            "task": "chat_completion",
            "inputs": {"prompt": "Hello world"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        manager: JobLifecycleManager = app.state.job_manager
        text_result = {"text": "Generated response from Qwen Text.", "usage": {"completion_tokens": 6}}
        asyncio.run(
            manager.complete_job(
                job_id=job_id,
                req=JobCompletionRequest(result=text_result),
                caller_role="admin",
            )
        )

        # 1a. Request as text/plain
        plain_headers = dict(auth_headers)
        plain_headers["Accept"] = "text/plain"
        res_plain = client.get(f"/v1/jobs/{job_id}/result", headers=plain_headers)
        assert res_plain.status_code == 200
        assert "text/plain" in res_plain.headers.get("content-type", "")
        assert res_plain.text == "Generated response from Qwen Text.", (
            f"Defect: complete_job discarded worker text result! Returned empty string instead: '{res_plain.text}'"
        )

        # 1b. Request as application/json
        json_headers = dict(auth_headers)
        json_headers["Accept"] = "application/json"
        res_json = client.get(f"/v1/jobs/{job_id}/result", headers=json_headers)
        assert res_json.status_code == 200
        assert "application/json" in res_json.headers.get("content-type", "")
        assert res_json.json()["text"] == "Generated response from Qwen Text."
        assert res_json.json()["job_id"] == job_id

        # 1c. Zero S3 URLs
        assert "s3://" not in res_plain.text
        assert "s3://" not in res_json.text


# ==============================================================================
# VECTOR 2: Concurrent Downloads & Race Condition Testing
# ==============================================================================

class TestVector2ConcurrencyAndRaceConditions:
    """Stress-tests rapid parallel GET requests on the same artifact and validates background cleanup safety."""

    def test_rapid_concurrent_downloads_completed_job(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        Adversarially fires 50 concurrent GET requests against a single media artifact of a SUCCEEDED job.
        Verifies:
        - 0 status code 500 (Internal Server Error)
        - 0 unhandled exceptions or crashes
        - Every response is either HTTP 200 (downloaded) or HTTP 404 (already unlinked)
        - Scratch disk directory is cleanly unlinked after all requests complete (0 residual files).
        """
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Concurrency race test image"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        job_dir = Path(SCRATCH_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        artifact_bytes = b"CONCURRENCY_TEST_BINARY_BYTES_" + os.urandom(1024)
        file_path = job_dir / "target.png"
        file_path.write_bytes(artifact_bytes)

        # Transition job to SUCCEEDED
        manager: JobLifecycleManager = app.state.job_manager
        asyncio.run(
            manager.complete_job(
                job_id=job_id,
                req=JobCompletionRequest(artifacts=[]),
                caller_role="admin",
            )
        )

        num_requests = 50
        status_codes = []
        errors = []

        def _fire_request(i: int):
            try:
                r = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
                return r.status_code, r.text
            except Exception as e:
                return 500, str(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(_fire_request, i) for i in range(num_requests)]
            for future in futures:
                code, text = future.result()
                status_codes.append(code)
                if code == 500:
                    errors.append(text)

        # Critical assertions:
        assert len(errors) == 0, f"Encountered 500 Internal Server Errors: {errors}"
        assert all(c in (200, 404) for c in status_codes), f"Unexpected status codes: {set(status_codes)}"
        assert 200 in status_codes, "At least one request should have received 200 OK"

        # Give background task up to 1 second to complete unlink
        cleaned_up = False
        for _ in range(10):
            if not file_path.exists():
                cleaned_up = True
                break
            time.sleep(0.1)

        assert cleaned_up, f"Residual artifact file was not unlinked: {file_path}"
        assert not job_dir.exists(), f"Residual job directory was not removed: {job_dir}"

        # Subsequent GET after cleanup must return clean 404 UnifiedError
        subsequent_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert_sanitized_error(subsequent_resp, 404, "NOT_FOUND")

    def test_multiple_jobs_concurrent_lifecycle_burst(self, client: TestClient, auth_headers: Dict[str, str]):
        """Submits 20 distinct jobs in burst, writes artifacts, and downloads concurrently."""
        num_jobs = 20
        job_ids = []

        # 1. Burst Submit
        for i in range(num_jobs):
            payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": f"Burst job {i}"},
            }
            res = client.post("/v1/jobs", json=payload, headers=auth_headers)
            assert res.status_code == 201
            job_ids.append(res.json()["job_id"])

        # 2. Write unique artifact for each job and complete
        manager: JobLifecycleManager = app.state.job_manager
        for j_id in job_ids:
            j_dir = Path(SCRATCH_DIR) / j_id
            j_dir.mkdir(parents=True, exist_ok=True)
            (j_dir / "out.png").write_bytes(f"DATA_{j_id}".encode("utf-8"))
            asyncio.run(
                manager.complete_job(
                    job_id=j_id,
                    req=JobCompletionRequest(artifacts=[]),
                    caller_role="admin",
                )
            )

        # 3. Download all in parallel
        def _download_job(j_id: str):
            r = client.get(f"/v1/jobs/{j_id}/result", headers=auth_headers)
            return j_id, r.status_code, r.content

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(_download_job, job_ids))

        for j_id, code, content in results:
            assert code == 200, f"Job {j_id} download failed with status {code}"
            assert content == f"DATA_{j_id}".encode("utf-8")

        # 4. Verify all files cleaned up
        time.sleep(0.5)
        for j_id in job_ids:
            assert not (Path(SCRATCH_DIR) / j_id).exists(), f"Job directory {j_id} not cleaned up."


# ==============================================================================
# VECTOR 3: Edge Case States & Error Mapping
# ==============================================================================

class TestVector3EdgeCaseStatesAndErrorMapping:
    """Stress-tests all non-standard, in-progress, cancelled, and failed job states on /result."""

    def test_nonexistent_job_id_returns_404_unified_error(self, client: TestClient, auth_headers: Dict[str, str]):
        """GET /v1/jobs/{non_existent}/result returns 404 UnifiedError."""
        fake_id = f"job-nonexistent-{uuid.uuid4()}"
        res = client.get(f"/v1/jobs/{fake_id}/result", headers=auth_headers)
        assert_sanitized_error(res, 404, "NOT_FOUND")

    def test_cancelled_job_returns_409_conflict(self, client: TestClient, auth_headers: Dict[str, str]):
        """GET /v1/jobs/{job_id}/result on a cancelled job returns HTTP 409 Conflict."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "To be cancelled"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # Cancel the job
        del_res = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert del_res.status_code == 200

        # Query result
        res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert_sanitized_error(res, 409, "JOB_CANCELLED")
        assert "cancelled" in res.json()["error"]["message"].lower()

    def test_queued_and_running_job_returns_409_with_progress(self, client: TestClient, auth_headers: Dict[str, str]):
        """GET /v1/jobs/{job_id}/result on in-progress job returns HTTP 409 Conflict with progress payload."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "In-progress test"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # 1. State: QUEUED
        res_queued = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert res_queued.status_code == 409
        body_queued = res_queued.json()
        assert body_queued["job_id"] == job_id
        assert body_queued["state"] in ("QUEUED", "ADMITTED", "RUNNING")
        assert "progress" in body_queued

        # 2. State: RUNNING with progress
        manager: JobLifecycleManager = app.state.job_manager
        job = asyncio.run(manager.get_job(job_id))
        job.state = JobState.RUNNING
        job.progress = JobProgress(stage="sampling", step=15, total=30, percent=50.0)
        asyncio.run(manager.repository.update_job(job))

        res_running = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert res_running.status_code == 409
        body_running = res_running.json()
        assert body_running["state"] == "RUNNING"
        assert body_running["progress"]["percent"] == 50.0

    def test_failed_job_error_mapping(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        Verifies specific error code mappings for failed jobs:
        - WORKER_DISCONNECTED -> HTTP 502 Bad Gateway
        - GATEWAY_TIMEOUT / TIMEOUT -> HTTP 504 Gateway Timeout
        - MODEL_UNAVAILABLE / SERVICE_UNAVAILABLE -> HTTP 503 Service Unavailable
        - INVALID_REQUEST / BAD_REQUEST -> HTTP 400 Bad Request
        """
        cases = [
            (ErrorCode.WORKER_DISCONNECTED, 502, "WORKER_DISCONNECTED"),
            (ErrorCode.GATEWAY_TIMEOUT, 504, "GATEWAY_TIMEOUT"),
            (ErrorCode.TIMEOUT, 504, "GATEWAY_TIMEOUT"),
            (ErrorCode.MODEL_UNAVAILABLE, 503, "MODEL_UNAVAILABLE"),
            (ErrorCode.SERVICE_UNAVAILABLE, 503, "SERVICE_UNAVAILABLE"),
            (ErrorCode.INVALID_REQUEST, 400, "INVALID_REQUEST"),
        ]

        manager: JobLifecycleManager = app.state.job_manager

        for err_code_enum, expected_status, expected_err_code in cases:
            # Submit fresh job
            submit_payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": f"Failure probe {err_code_enum.value}"},
            }
            resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
            assert resp.status_code == 201
            job_id = resp.json()["job_id"]

            # Fail the job via lifecycle manager
            err_detail = ErrorDetail(
                code=err_code_enum,
                message=f"Simulated failure with {err_code_enum.value}",
                request_id="req-test-fail-123",
            )
            asyncio.run(
                manager.fail_job(
                    job_id=job_id,
                    req=JobFailureRequest(error=err_detail, retryable=False),
                    caller_role="admin",
                )
            )

            # Query /result
            res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
            assert_sanitized_error(res, expected_status, expected_err_code)

    def test_succeeded_media_job_missing_scratch_artifact(self, client: TestClient, auth_headers: Dict[str, str]):
        """If a media job is SUCCEEDED but the scratch artifact file is missing, return HTTP 404 UnifiedError."""
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Missing artifact test"},
        }
        resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # Complete job with empty artifacts (no file on disk)
        manager: JobLifecycleManager = app.state.job_manager
        asyncio.run(
            manager.complete_job(
                job_id=job_id,
                req=JobCompletionRequest(artifacts=[]),
                caller_role="admin",
            )
        )

        res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert_sanitized_error(res, 404, "NOT_FOUND")


# ==============================================================================
# VECTOR 4: Security & Boundary Probes on /result
# ==============================================================================

class TestVector4SecurityAndBoundaryProbes:
    """Security probes verifying authentication enforcement and directory traversal prevention."""

    def test_unauthenticated_request_rejected(self, client: TestClient):
        """GET /v1/jobs/{id}/result without Authorization header is rejected with 401."""
        res = client.get("/v1/jobs/job-12345/result")
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

    def test_invalid_bearer_token_rejected(self, client: TestClient):
        """GET /v1/jobs/{id}/result with invalid token is rejected with 401."""
        headers = {"Authorization": "Bearer bad-token-12345"}
        res = client.get("/v1/jobs/job-12345/result", headers=headers)
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

    def test_path_traversal_in_job_id_safely_contained(self, client: TestClient, auth_headers: Dict[str, str]):
        """Adversarial path traversal in job_id must not leak filesystem or cause 500 error."""
        traversal_vectors = [
            "../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "....//....//etc/shadow",
            "job-1/../../../etc",
        ]
        for vec in traversal_vectors:
            res = client.get(f"/v1/jobs/{vec}/result", headers=auth_headers)
            assert res.status_code in (400, 404, 422), f"Unexpected status {res.status_code} for vector '{vec}'"
            assert "root:" not in res.text
            assert "passwd" not in res.text
            assert_sanitized_error(res, res.status_code)


# ==============================================================================
# VECTOR 5: Scratch Disk Sweeper & Quota Stress
# ==============================================================================

class TestVector5ScratchDiskSweeperStress:
    """Empirical verification of ScratchDiskManager TTL sweep and LRU quota enforcement."""

    def test_ttl_eviction_sweeps_expired_files_only(self):
        """TTL sweeper purges files older than ttl_seconds and retains fresh files."""
        cfg = ScratchDiskConfig(root_dir=SCRATCH_DIR, ttl_seconds=2, sweep_interval_seconds=1)
        mgr = ScratchDiskManager(config=cfg)

        old_dir = Path(SCRATCH_DIR) / "job-old"
        fresh_dir = Path(SCRATCH_DIR) / "job-fresh"
        old_dir.mkdir(parents=True, exist_ok=True)
        fresh_dir.mkdir(parents=True, exist_ok=True)

        old_file = old_dir / "old_art.bin"
        fresh_file = fresh_dir / "fresh_art.bin"
        old_file.write_bytes(b"EXPIRED_DATA" * 50)
        fresh_file.write_bytes(b"FRESH_DATA" * 50)

        # Artificially age old_file by 10 seconds
        past_time = time.time() - 10
        os.utime(old_file, (past_time, past_time))

        stats = mgr.sweep_once()
        assert stats["evicted_ttl"] >= 1
        assert not old_file.exists(), "Old file was not evicted by TTL sweep"
        assert fresh_file.exists(), "Fresh file was prematurely evicted"

    def test_lru_quota_eviction_under_disk_pressure(self):
        """When total storage exceeds max_storage_bytes, oldest files are purged down to 80%."""
        # Minimum quota is 1MB (1048576 bytes)
        quota = 1048576
        cfg = ScratchDiskConfig(
            root_dir=SCRATCH_DIR,
            ttl_seconds=3600,
            max_storage_bytes=quota,
        )
        mgr = ScratchDiskManager(config=cfg)

        # Write 3 files of 500 KB each = 1.5 MB (exceeds 1 MB quota)
        now = time.time()
        file_size = 500 * 1024
        for i in range(3):
            j_dir = Path(SCRATCH_DIR) / f"job-quota-{i}"
            j_dir.mkdir(parents=True, exist_ok=True)
            f_path = j_dir / f"file_{i}.bin"
            f_path.write_bytes(b"X" * file_size)
            # Stagger timestamps: 0 is oldest, 2 is newest
            os.utime(f_path, (now + i * 10, now + i * 10))

        stats = mgr.sweep_once()
        assert stats["evicted_lru"] >= 1
        assert stats["remaining_bytes"] <= int(quota * 0.8)
        # Verify file_0 (oldest) was evicted
        assert not (Path(SCRATCH_DIR) / "job-quota-0" / "file_0.bin").exists()
        # Verify file_2 (newest) remains
        assert (Path(SCRATCH_DIR) / "job-quota-2" / "file_2.bin").exists()
