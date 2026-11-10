"""
tests/test_preflight.py
Factory v4 Lite: Consolidated Minimal Preflight Test Suite (PRE-01 through PRE-15).

Authoritative Coverage (FILE_INVENTORY.md §3.1 & Simplification Plan §4.3):
- PRE-01 (Startup): Health check GET /health unauthenticated returns 200 OK.
- PRE-02 (Manifest Discovery): GET /v1/models loads and validates all 11 model manifests dynamically.
- PRE-03 (Auth Enforcement): Authorization: Bearer <FACTORY_API_KEY> strictly enforced (401 on missing/bad).
- PRE-04 (Job Submission): POST /v1/jobs accepts valid generative payloads and normalizes task contracts.
- PRE-05 (Fail-Closed Rejection): Rejects unknown models and unsupported capabilities with non-success status.
- PRE-06 (Job Lifecycle): In-memory queue state transitions (CREATED -> QUEUED -> RUNNING -> SUCCEEDED/FAILED).
- PRE-07 (Progress Streaming): GET /v1/jobs/{id}/events streams SSE progress chunks with correct headers.
- PRE-08 (Cancellation & Idempotency): DELETE /v1/jobs/{id} cleanly cancels jobs; duplicate cancel handled cleanly.
- PRE-09 (Worker Lease & Watchdog): Worker acquires lease; unrenewed lease triggers watchdog deadline reaping.
- PRE-10 (GPU Allocation & Bounds): Scheduler enforces discrete GPU 0 vs GPU 1 partition affinity & mutual exclusion.
- PRE-11 (Artifact Verification): Validates output presence, non-empty bytes, valid MIME, and real SHA-256 hash.
- PRE-12 (Scratch Disk Storage): Ephemeral scratch storage under /tmp/factory_lite/artifacts/ with cleanup.
- PRE-13 (SSRF & Boundary Containment): Blocks private/loopback URLs, enforces 1 MB inline and 50 MB upload limits.
- PRE-14 (Error Sanitization): Strict error envelope (code, message, request_id); zero trace/path leaks.
- PRE-15 (Backend Unavailable Probe): When backend is offline, returns 503 / BACKEND_UNAVAILABLE; zero mock fallback.
"""

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Dict, Generator, Optional
import uuid

import pytest
from fastapi.testclient import TestClient

from factory_auth import verify_bearer_token
from factory_contracts.errors import (
    ErrorCode,
    ErrorDetail,
    ModelUnavailableError,
    PayloadTooLargeError,
    SSRFBlockedError,
    UnsupportedCapabilityError,
    classify_worker_error,
)
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.queue.state_machine import (
    InvalidStateTransitionError,
    JobStateMachine,
)
from factory_contracts.jobs.job_contract import FactoryJob, JobProgress, JobState
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    WorkerLeaseRequest,
)
from factory_contracts.tasks import normalize_model, normalize_task
from model_registry import ManifestLoader

from factory.apps.control_api.src.assets.service import MAX_PAYLOAD_SIZE, AssetService
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.main import app
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.policy import SUBSYSTEM_GPU_MAP
from factory.apps.control_api.src.queue.repository import get_job_repository
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog
from factory.apps.control_api.src.storage.scratch_disk import (
    ScratchDiskConfig,
    ScratchDiskManager,
)
from services.enhancement.backend import call_backend
from services.text.src.multimodal import validate_url_security

# Authoritative Test Constants
VALID_API_KEY = "sk-factory-preflight-valid-key-2026"
INVALID_API_KEY = "sk-factory-invalid-unauthorized-key"
SCRATCH_ARTIFACTS_DIR = "/tmp/factory_lite/artifacts"


@pytest.fixture(scope="session", autouse=True)
def configure_preflight_environment():
    """Configures environment for in-memory Factory Lite preflight verification."""
    os.environ["FACTORY_API_KEY"] = VALID_API_KEY
    os.environ["ENVIRONMENT"] = "test"
    os.environ["FACTORY_ENV"] = "test"
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "memory"
    os.makedirs(SCRATCH_ARTIFACTS_DIR, exist_ok=True)
    yield
    if os.path.exists(SCRATCH_ARTIFACTS_DIR):
        try:
            shutil.rmtree(SCRATCH_ARTIFACTS_DIR)
        except Exception:
            pass


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provides a TestClient targeting the in-memory Factory Lite app."""
    with TestClient(app, base_url="http://testserver") as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Returns valid Bearer authorization headers."""
    return {
        "Authorization": f"Bearer {VALID_API_KEY}",
        "Content-Type": "application/json",
    }


def assert_sanitized_error(response, expected_status: int, expected_code: Optional[str] = None):
    """
    Verifies error response strictly adheres to the error envelope:
    { "error": { "code": "...", "message": "...", "request_id": "req-..." } }
    and contains zero internal stack traces or filesystem leaks.
    """
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. Response: {response.text}"
    )
    data = response.json()
    assert "error" in data, f"Missing 'error' in response: {data}"
    err = data["error"]
    assert "code" in err and len(err["code"]) > 0, f"Missing or empty error code: {err}"
    if expected_code:
        if isinstance(expected_code, list):
            assert err["code"] in expected_code, f"Expected code in {expected_code}, got '{err['code']}'"
        else:
            assert err["code"] == expected_code, f"Expected code '{expected_code}', got '{err['code']}'"
    assert "message" in err and len(err["message"]) > 0, f"Missing or empty error message: {err}"
    assert "request_id" in err, f"Missing request_id in error: {err}"

    # Verify zero leaks of stack traces or local paths
    raw_text = response.text
    for leak_pattern in [
        r"Traceback \(most recent call last\)",
        r"/Volumes/[A-Za-z0-9_/:-]+",
        r"/Users/[A-Za-z0-9_/:-]+",
        r"/home/[A-Za-z0-9_/:-]+",
        r"ZeroDivisionError:",
        r"KeyError:",
    ]:
        assert not re.search(leak_pattern, raw_text), f"Internal trace leak detected: {leak_pattern}"


# ==============================================================================
# PRE-01 through PRE-15 Scenario Implementations
# ==============================================================================

def test_pre_01_startup_health_check(client: TestClient):
    """PRE-01 (Startup): GET /health unauthenticated returns 200 OK and control plane status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "healthy"
    assert body.get("plane") == "control"
    assert "version" in body


def test_pre_02_manifest_discovery(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-02 (Manifest Discovery): GET /v1/models loads and validates all 11 model manifests dynamically."""
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    models = body["data"]

    # All 11 registered foundation models must be discoverable
    expected_models = {
        "qwen-text",
        "qwen-vlm",
        "qwen-decision",
        "qwen-embedding",
        "qwen-speech",
        "qwen-image-2.1",
        "wan-2.2",
        "minimax-h3",
        "ace-step-1.5",
        "yue2",
        "enhancement-suite-v1",
    }
    discovered_ids = {
        m["metadata"]["id"] for m in models
        if isinstance(m, dict) and "metadata" in m and "id" in m["metadata"]
    }
    assert expected_models.issubset(discovered_ids), (
        f"Missing model manifests in discovery: {expected_models - discovered_ids}"
    )

    # Validate schema registry loads cleanly from disk
    manifests_dir = Path("manifests")
    if manifests_dir.exists():
        loader = ManifestLoader(manifests_dir)
        models_dict = loader.load_models()
        services_dict = loader.load_services()
        hw_dict = loader.load_hardware_topologies()
        cap_dict = loader.load_capabilities()
        assert len(models_dict) in (11, 12)
        assert len(services_dict) == 11
        assert len(hw_dict) == 3
        assert len(cap_dict) == 12


def test_pre_03_auth_enforcement(client: TestClient):
    """PRE-03 (Auth Enforcement): Authorization: Bearer <FACTORY_API_KEY> strictly enforced (401 on bad/missing)."""
    # 1. Missing header
    r_missing = client.get("/v1/models")
    assert r_missing.status_code == 401
    assert_sanitized_error(r_missing, 401, "UNAUTHORIZED")

    # 2. Empty Bearer
    r_empty = client.get("/v1/models", headers={"Authorization": "Bearer "})
    assert r_empty.status_code == 401
    assert_sanitized_error(r_empty, 401, "UNAUTHORIZED")

    # 3. Invalid token
    r_invalid = client.get("/v1/models", headers={"Authorization": f"Bearer {INVALID_API_KEY}"})
    assert r_invalid.status_code == 401
    assert_sanitized_error(r_invalid, 401, "UNAUTHORIZED")

    # 4. Valid token succeeds
    r_valid = client.get("/v1/models", headers={"Authorization": f"Bearer {VALID_API_KEY}"})
    assert r_valid.status_code == 200


def test_pre_04_job_submission_and_normalization(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-04 (Job Submission): POST /v1/jobs accepts valid generative payloads and normalizes task contracts."""
    payload = {
        "service": "image",
        "model": "qwen-image-2.1",
        "task": "image.t2i",  # Alias format
        "inputs": {"prompt": "A serene mountain lake at sunrise, highly detailed"},
        "priority": 10,
    }
    resp = client.post("/v1/jobs", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    job = resp.json()
    assert job["job_id"].startswith("job-")
    assert job["request_id"].startswith("req-")
    assert job["service"] == "image"
    assert job["model"] == "qwen-image-2.1"
    # Canonical normalized task contract
    assert job["task"] == "text_to_image"
    assert job["state"] == "QUEUED"


def test_pre_05_fail_closed_rejection(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-05 (Fail-Closed Rejection): Rejects unknown models and unsupported capabilities with non-success status."""
    # 1. Unknown model family
    bad_model_payload = {
        "service": "image",
        "model": "non-existent-hallucinated-model",
        "task": "text_to_image",
        "inputs": {"prompt": "test"},
    }
    r_bad_model = client.post("/v1/jobs", json=bad_model_payload, headers=auth_headers)
    assert r_bad_model.status_code == 400
    assert_sanitized_error(r_bad_model, 400)

    # 2. Unsupported capability (e.g. text model asked to inpaint image)
    unsupported_payload = {
        "service": "image",
        "model": "qwen-text",
        "task": "inpaint",
        "inputs": {"prompt": "test"},
    }
    r_unsupported = client.post("/v1/jobs", json=unsupported_payload, headers=auth_headers)
    assert r_unsupported.status_code == 400
    assert_sanitized_error(r_unsupported, 400, "CAPABILITY_NOT_SUPPORTED")


def test_pre_06_job_lifecycle_state_transitions():
    """PRE-06 (Job Lifecycle): In-memory queue state transitions (CREATED -> QUEUED -> RUNNING -> SUCCEEDED/FAILED)."""
    # Test valid progression
    assert JobStateMachine.can_transition(JobState.CREATED, JobState.QUEUED) is True
    assert JobStateMachine.can_transition(JobState.QUEUED, JobState.RUNNING) is True
    assert JobStateMachine.can_transition(JobState.RUNNING, JobState.SUCCEEDED) is True

    # Test failure path
    assert JobStateMachine.can_transition(JobState.RUNNING, JobState.FAILED) is True

    # Test cancellation path
    assert JobStateMachine.can_transition(JobState.QUEUED, JobState.CANCELLED) is True
    assert JobStateMachine.can_transition(JobState.SUCCEEDED, JobState.RUNNING) is False

    # Test invalid transitions raise InvalidStateTransitionError
    with pytest.raises(InvalidStateTransitionError):
        JobStateMachine.validate_transition(JobState.SUCCEEDED, JobState.RUNNING, "test-job")

    with pytest.raises(InvalidStateTransitionError):
        JobStateMachine.validate_transition(JobState.CANCELLED, JobState.SUCCEEDED, "test-job")


def test_pre_07_progress_streaming_sse(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-07 (Progress Streaming): GET /v1/jobs/{id}/events streams SSE progress chunks with correct event headers."""
    # Submit job
    submit_resp = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Progress stream preflight test"},
        },
        headers=auth_headers,
    )
    assert submit_resp.status_code == 201
    job_id = submit_resp.json()["job_id"]

    stream_headers = dict(auth_headers)
    stream_headers["Accept"] = "text/event-stream"
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=stream_headers) as sse_stream:
        assert sse_stream.status_code == 200
        assert "text/event-stream" in sse_stream.headers.get("content-type", "")
        lines = []
        for line in sse_stream.iter_lines():
            if line:
                lines.append(line)
            if len(lines) >= 3:
                break
        assert len(lines) > 0, "No SSE events streamed from endpoint"
        content = "\n".join(lines)
        assert ("event:" in content or "data:" in content or ": keepalive" in content)


def test_pre_08_cancellation_and_idempotency(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-08 (Cancellation & Idempotency): DELETE /v1/jobs/{id} cleanly cancels active/queued jobs; duplicate cancel is idempotent."""
    idemp_key = f"idemp-preflight-{uuid.uuid4()}"
    payload = {
        "service": "image",
        "model": "qwen-image-2.1",
        "task": "text_to_image",
        "inputs": {"prompt": "Cancellation & idempotency test"},
        "idempotency_key": idemp_key,
    }

    # 1. First submission
    r1 = client.post("/v1/jobs", json=payload, headers=auth_headers)
    assert r1.status_code == 201
    job_id_1 = r1.json()["job_id"]

    # 2. Second submission with same idempotency key returns identical job
    r2 = client.post("/v1/jobs", json=payload, headers=auth_headers)
    assert r2.status_code in [200, 201]
    assert r2.json()["job_id"] == job_id_1

    # 3. Cancel job
    del_resp1 = client.delete(f"/v1/jobs/{job_id_1}", headers=auth_headers)
    assert del_resp1.status_code == 200
    assert del_resp1.json()["state"] == "CANCELLED"

    # 4. Duplicate cancellation handled cleanly (no 500 internal error)
    del_resp2 = client.delete(f"/v1/jobs/{job_id_1}", headers=auth_headers)
    assert del_resp2.status_code in [200, 400]
    if del_resp2.status_code != 200:
        assert_sanitized_error(del_resp2, del_resp2.status_code)


@pytest.mark.asyncio
async def test_pre_09_worker_lease_and_watchdog(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-09 (Worker Lease & Watchdog): Worker acquires job lease; unrenewed lease triggers watchdog deadline reaping."""
    # Submit job
    submit_resp = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Lease and watchdog preflight test"},
        },
        headers=auth_headers,
    )
    assert submit_resp.status_code == 201
    job_id = submit_resp.json()["job_id"]

    repo = get_job_repository()
    scheduler = InMemoryScheduler()
    stream_manager = getattr(client.app.state, "stream_manager", None)
    manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_manager)
    coordinator = LeaseCoordinator(repository=repo, broker=scheduler)

    # Worker leases the job
    worker_id = f"worker-test-{uuid.uuid4()}"
    lease_req = WorkerLeaseRequest(
        worker_id=worker_id,
        subsystem="image",
        supported_models=["qwen-image-2.1"],
        supported_tasks=["text_to_image"],
        gpu_device=0,
    )
    lease_resp = await coordinator.lease_job(lease_req)
    assert lease_resp.job is not None
    leased_job_id = lease_resp.job.job_id

    # Simulate expired worker heartbeat (timeout threshold exceeded)
    await asyncio.sleep(0.05)
    reaped_count = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=0.01,  # Expired immediately
        default_job_timeout_seconds=300.0,
    )
    assert reaped_count >= 1

    # Verify job state reaped to FAILED with WORKER_DISCONNECTED
    reaped_job = await repo.get_job(leased_job_id)
    assert reaped_job.state == JobState.FAILED
    assert reaped_job.error is not None
    assert reaped_job.error.code == ErrorCode.WORKER_DISCONNECTED.value or "DISCONNECTED" in reaped_job.error.code
    await scheduler.close()


def test_pre_10_gpu_allocation_and_bounds():
    """PRE-10 (GPU Allocation & Bounds): Scheduler enforces discrete GPU 0 vs GPU 1 partition affinity & mutual exclusion."""
    scheduler = InMemoryScheduler()

    # 1. Verify subsystem partition affinity mapping
    assert SUBSYSTEM_GPU_MAP["image"] == 0
    assert SUBSYSTEM_GPU_MAP["text"] == 0
    assert SUBSYSTEM_GPU_MAP["decision"] == 0
    assert SUBSYSTEM_GPU_MAP["video"] == 1
    assert SUBSYSTEM_GPU_MAP["music"] == 1
    assert SUBSYSTEM_GPU_MAP["embedding"] == 1

    # 2. Acquire GPU 0 for job A
    job_a = f"job-a-{uuid.uuid4()}"
    assert scheduler.acquire_gpu(0, job_a) is True
    assert scheduler.is_gpu_busy(0) is True

    # 3. Mutual exclusion: concurrent attempt to acquire GPU 0 for job B MUST fail
    job_b = f"job-b-{uuid.uuid4()}"
    assert scheduler.acquire_gpu(0, job_b) is False

    # 4. Discrete Dual-GPU isolation: GPU 0 busy does NOT prevent acquiring GPU 1
    job_c = f"job-c-{uuid.uuid4()}"
    assert scheduler.acquire_gpu(1, job_c) is True
    assert scheduler.is_gpu_busy(1) is True

    # 5. Release GPU 0 and verify job B can now acquire it
    assert scheduler.release_gpu(0, job_a) is True
    assert scheduler.is_gpu_busy(0) is False
    assert scheduler.acquire_gpu(0, job_b) is True
    assert scheduler.is_gpu_busy(0) is True

    scheduler.release_gpu(0, job_b)
    scheduler.release_gpu(1, job_c)


@pytest.mark.asyncio
async def test_pre_11_artifact_verification(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-11 (Artifact Verification): Validates output presence, non-empty bytes, valid MIME, and real SHA-256 hash."""
    repo = get_job_repository()
    stream_manager = getattr(client.app.state, "stream_manager", None)
    scheduler = InMemoryScheduler()
    manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_manager)

    # Submit job
    submit_resp = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Artifact validation test"},
        },
        headers=auth_headers,
    )
    job_id = submit_resp.json()["job_id"]

    # Test 1: Complete with missing artifact path fails closed with ARTIFACT_MISSING
    non_existent_file = f"/tmp/factory_lite/artifacts/{job_id}/missing.png"
    failed_job = await manager.complete_job(
        job_id=job_id,
        req=JobCompletionRequest(artifact_path=non_existent_file),
        caller_role="admin",
    )
    assert failed_job.state == JobState.FAILED
    assert failed_job.error is not None
    assert failed_job.error.code == ErrorCode.ARTIFACT_MISSING

    # Test 2: Complete with genuine artifact computes valid sha256, mime, size
    valid_submit = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Genuine artifact test"},
        },
        headers=auth_headers,
    )
    valid_job_id = valid_submit.json()["job_id"]

    valid_dir = os.path.join(SCRATCH_ARTIFACTS_DIR, valid_job_id)
    os.makedirs(valid_dir, exist_ok=True)
    real_artifact_path = os.path.join(valid_dir, "output.png")
    dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    with open(real_artifact_path, "wb") as f:
        f.write(dummy_bytes)

    expected_sha = hashlib.sha256(dummy_bytes).hexdigest()
    succeeded_job = await manager.complete_job(
        job_id=valid_job_id,
        req=JobCompletionRequest(artifact_path=real_artifact_path),
        caller_role="admin",
    )
    assert succeeded_job.state == JobState.SUCCEEDED
    assert len(succeeded_job.artifacts) >= 1
    art = succeeded_job.artifacts[0]
    assert art.sha256 == expected_sha
    assert art.size_bytes == len(dummy_bytes)
    assert "image/png" in art.mime_type
    await scheduler.close()


def test_pre_12_scratch_disk_storage(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-12 (Scratch Disk Storage): Ephemeral scratch storage delivery under /tmp/factory_lite/artifacts/ with unlinking and cleanup."""
    scratch_disk = getattr(client.app.state, "scratch_disk", None)
    assert scratch_disk is not None
    orig_cleanup = scratch_disk.config.cleanup_on_download
    scratch_disk.config.cleanup_on_download = True

    try:
        # Submit job
        submit_resp = client.post(
            "/v1/jobs",
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Scratch disk test"},
            },
            headers=auth_headers,
        )
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # Write artifact to scratch disk
        job_dir = os.path.join(SCRATCH_ARTIFACTS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        art_file = os.path.join(job_dir, "render.png")
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        with open(art_file, "wb") as f:
            f.write(png_data)

        # Mark complete
        comp_resp = client.post(
            f"/v1/jobs/{job_id}/complete",
            json={"artifact_path": art_file},
            headers=auth_headers,
        )
        assert comp_resp.status_code == 200

        # Download artifact
        res_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert res_resp.status_code == 200
        assert res_resp.content == png_data

        # Verify automatic download unlinking
        cleaned = False
        for _ in range(10):
            if not os.path.exists(art_file):
                cleaned = True
                break
            time.sleep(0.05)
        assert cleaned is True, f"Artifact was not cleaned up after download: {art_file}"

        # Verify TTL cleanup logic
        stale_dir = os.path.join(SCRATCH_ARTIFACTS_DIR, f"job-stale-{uuid.uuid4()}")
        os.makedirs(stale_dir, exist_ok=True)
        stale_file = os.path.join(stale_dir, "stale.png")
        with open(stale_file, "wb") as f:
            f.write(b"stale")
        # Run cleanup via sweep_once with ttl_seconds=0
        orig_ttl = scratch_disk.config.ttl_seconds
        try:
            scratch_disk.config.ttl_seconds = 0
            sweep_stats = scratch_disk.sweep_once()
            assert sweep_stats["evicted_ttl"] >= 1 or sweep_stats["pruned_dirs"] >= 1
        finally:
            scratch_disk.config.ttl_seconds = orig_ttl
    finally:
        scratch_disk.config.cleanup_on_download = orig_cleanup


def test_pre_13_ssrf_and_boundary_containment(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-13 (SSRF & Boundary Containment): Blocks private/loopback URLs, enforces 1 MB inline and 50 MB upload limits."""
    # 1. SSRF prevention
    blocked_urls = [
        "http://127.0.0.1:8080/admin",
        "http://localhost:5000/keys",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/private",
        "http://192.168.1.1/router",
        "file:///etc/passwd",
        "ftp://example.com/file",
    ]
    for url in blocked_urls:
        with pytest.raises(SSRFBlockedError):
            validate_target_url(url)

    # 2. Inline payload limit > 1 MB rejected with 413
    raw_oversized_bytes = b"0" * (1024 * 1024 + 2048)
    encoded_b64 = base64.b64encode(raw_oversized_bytes).decode("ascii")
    large_data_uri = f"data:image/png;base64,{encoded_b64}"
    r_inline = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Large inline prompt", "image": large_data_uri},
        },
        headers=auth_headers,
    )
    assert r_inline.status_code == 413
    assert_sanitized_error(r_inline, 413, "PAYLOAD_TOO_LARGE")

    # 3. Upload endpoint limit > 50 MB rejected
    asset_service = AssetService(max_size_bytes=MAX_PAYLOAD_SIZE)
    oversized_bytes = b"0" * (MAX_PAYLOAD_SIZE + 1024)
    with pytest.raises(PayloadTooLargeError):
        asset_service.ingest_bytes(oversized_bytes)


def test_pre_14_error_sanitization(client: TestClient, auth_headers: Dict[str, str]):
    """PRE-14 (Error Sanitization): Strict error envelope (code, message, request_id); zero trace/path leaks."""
    # Trigger 400 Bad Request
    r_bad = client.post("/v1/jobs", json={"service": "image"}, headers=auth_headers)
    assert r_bad.status_code in [400, 422]
    assert_sanitized_error(r_bad, r_bad.status_code)

    # Trigger 404 Not Found
    r_404 = client.get(f"/v1/jobs/job-{uuid.uuid4()}", headers=auth_headers)
    assert r_404.status_code == 404
    assert_sanitized_error(r_404, 404, ["JOB_NOT_FOUND", "NOT_FOUND"])

    # Trigger 401 Unauthorized
    r_401 = client.get("/v1/models")
    assert r_401.status_code == 401
    assert_sanitized_error(r_401, 401, "UNAUTHORIZED")


def test_pre_15_backend_unavailable_probe():
    """PRE-15 (Backend Unavailable Probe): When backend is offline, returns explicit 503 / BACKEND_UNAVAILABLE; zero mock fallback."""
    # 1. Unconfigured enhancement backend raises ModelUnavailableError (HTTP 503)
    orig_env = os.environ.get("IMAGE_SR_BACKEND_URL")
    if "IMAGE_SR_BACKEND_URL" in os.environ:
        del os.environ["IMAGE_SR_BACKEND_URL"]

    try:
        with pytest.raises(ModelUnavailableError) as exc_info:
            call_backend("IMAGE_SR_BACKEND_URL", "/v1/image_super_resolution", {"image_url": "https://example.com/test.png"})
        assert exc_info.value.status_code == 503
        assert "not configured" in str(exc_info.value.message)
    finally:
        if orig_env is not None:
            os.environ["IMAGE_SR_BACKEND_URL"] = orig_env

    # 2. Connection error mapping maps to BACKEND_UNAVAILABLE with retryable=True
    err_code, err_msg, retryable = classify_worker_error(
        ConnectionRefusedError("Connection refused by backend at port 8000"),
        subsystem="vllm",
        stage="chat_completion",
    )
    assert err_code == ErrorCode.BACKEND_UNAVAILABLE
    assert retryable is True
    assert "unavailable or unreachable" in err_msg
