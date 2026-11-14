"""
tests/unit/api/test_artifact_concurrency_p04.py
Unit and regression test suite for Requirement P0.4:
Artifact Delivery and Concurrent Downloads.

Validates:
1. Concurrent downloads (10-50 parallel clients) succeed with 100% 200 OK and zero HTTP 500 race conditions.
2. Artifact retention across downloads (cleanup_on_download=False default) and retry downloads.
3. Missing artifact returns standardized HTTP 404 with error code ARTIFACT_NOT_FOUND.
4. Atomic artifact writes (NamedTemporaryFile + fsync + os.replace) leave zero residual .tmp_ files.
5. Periodic sweeper only prunes files after TTL expiration.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import time
from typing import Dict
import uuid

import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from fastapi.testclient import TestClient

from factory.apps.control_api.src.legacy import app
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    JobSubmissionRequest,
)
from factory_contracts.jobs.worker_contract import JobCompletionRequest
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.storage.scratch_disk import (
    ScratchDiskConfig,
    ScratchDiskManager,
    get_scratch_disk,
)

TEST_API_KEY = "test-sk-p04-concurrency-2026"
SCRATCH_DIR = "/tmp/factory_lite/artifacts_p04"


@pytest.fixture(autouse=True)
def configure_test_env(monkeypatch):
    """Configures clean test environment for P0.4 artifact concurrency testing."""
    monkeypatch.setenv("FACTORY_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("FACTORY_ENV", "test")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "memory")
    monkeypatch.setenv("SCRATCH_ARTIFACTS_DIR", SCRATCH_DIR)
    monkeypatch.setenv("ARTIFACT_CLEANUP_ON_DOWNLOAD", "false")
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    yield
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


def test_p04_concurrent_downloads_10_requests_succeed_200(client: TestClient, auth_headers: Dict[str, str]):
    """Verifies that 20 concurrent downloads of the same artifact succeed with 200 OK and no 500 races."""
    # 1. Submit an image job
    submit_res = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Majestic Vietnamese Imperial Dragon"},
        },
        headers=auth_headers,
    )
    assert submit_res.status_code == 201
    job_id = submit_res.json()["job_id"]

    # 2. Persist artifact on scratch disk
    manager: JobLifecycleManager = app.state.job_manager
    scratch_disk: ScratchDiskManager = app.state.scratch_disk
    artifact_bytes = b"\x89PNG\r\n\x1a\n" + b"CONCURRENT_P04_TEST_IMAGE_PAYLOAD" * 50
    filename = "master_image.png"

    art_ref = asyncio.run(
        scratch_disk.save_artifact(
            job_id=job_id,
            filename=filename,
            data=artifact_bytes,
            mime_type="image/png",
        )
    )

    # 3. Complete job with artifact
    asyncio.run(
        manager.complete_job(
            job_id=job_id,
            req=JobCompletionRequest(artifacts=[art_ref]),
            caller_role="admin",
        )
    )

    # 4. Fire 20 concurrent download requests
    num_requests = 20

    def _download():
        resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        return resp.status_code, resp.content

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: _download(), range(num_requests)))

    status_codes = [code for code, _ in results]
    assert all(c == 200 for c in status_codes), f"Expected all 200 OK, got: {set(status_codes)}"
    assert all(content == artifact_bytes for _, content in results)

    # 5. Verify file and directory still exist on scratch disk (retained)
    artifact_path = scratch_disk.get_artifact_path(job_id, filename)
    assert artifact_path.exists(), "Artifact was prematurely deleted after concurrent downloads"


def test_p04_artifact_retention_and_retry_download(client: TestClient, auth_headers: Dict[str, str]):
    """Verifies artifact is retained through retention window and retries return identical content."""
    # 1. Submit job
    submit_res = client.post(
        "/v1/jobs",
        json={
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Ha Long Bay Sunset Melody"},
        },
        headers=auth_headers,
    )
    assert submit_res.status_code == 201
    job_id = submit_res.json()["job_id"]

    # 2. Write artifact
    manager: JobLifecycleManager = app.state.job_manager
    scratch_disk: ScratchDiskManager = app.state.scratch_disk
    audio_bytes = b"RIFF" + b"WAVE_P04_AUDIO_SAMPLE" * 40
    filename = "speech_output.wav"

    art_ref = asyncio.run(
        scratch_disk.save_artifact(
            job_id=job_id,
            filename=filename,
            data=audio_bytes,
            mime_type="audio/wav",
        )
    )

    # 3. Complete job
    asyncio.run(
        manager.complete_job(
            job_id=job_id,
            req=JobCompletionRequest(artifacts=[art_ref]),
            caller_role="admin",
        )
    )

    # 4. Initial download
    res1 = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
    assert res1.status_code == 200
    assert res1.content == audio_bytes

    # Verify file remains on disk
    file_path = scratch_disk.get_artifact_path(job_id, filename)
    assert file_path.exists()

    # 5. Sequential retry downloads (e.g. 3 retries)
    for retry_idx in range(3):
        retry_res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert retry_res.status_code == 200, f"Retry {retry_idx + 1} failed with status {retry_res.status_code}"
        assert retry_res.content == audio_bytes, f"Retry {retry_idx + 1} content mismatch"


def test_p04_missing_artifact_returns_controlled_404_artifact_not_found(client: TestClient, auth_headers: Dict[str, str]):
    """Verifies that SUCCEEDED media job with missing file returns HTTP 404 ARTIFACT_NOT_FOUND."""
    # 1. Submit job
    submit_res = client.post(
        "/v1/jobs",
        json={
            "service": "video",
            "model": "minimax-h3",
            "task": "text_to_video",
            "inputs": {"prompt": "Golden dragon flying over karst peaks"},
        },
        headers=auth_headers,
    )
    assert submit_res.status_code == 201
    job_id = submit_res.json()["job_id"]

    # 2. Complete job with empty artifacts (no physical file created)
    manager: JobLifecycleManager = app.state.job_manager
    asyncio.run(
        manager.complete_job(
            job_id=job_id,
            req=JobCompletionRequest(artifacts=[]),
            caller_role="admin",
        )
    )

    # 3. Request result
    res = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "ARTIFACT_NOT_FOUND"
    assert "Artifact file not found" in data["error"]["message"]


def test_p04_atomic_write_publish():
    """Verifies atomic write publish via NamedTemporaryFile leaves no temporary .tmp_ files."""
    scratch = ScratchDiskManager(ScratchDiskConfig(root_dir=SCRATCH_DIR))
    job_id = f"job-atomic-{uuid.uuid4().hex[:8]}"
    filename = "render_atomic.mp4"
    data = b"\x00\x00\x00\x18ftypisom" + b"TEST_ATOMIC_PAYLOAD" * 100

    art_ref = asyncio.run(
        scratch.save_artifact(
            job_id=job_id,
            filename=filename,
            data=data,
            mime_type="video/mp4",
        )
    )

    target_path = scratch.get_artifact_path(job_id, filename)
    assert target_path.exists()
    assert target_path.read_bytes() == data

    # Verify no .tmp_ files remain in job directory
    job_dir = scratch.get_job_dir(job_id)
    tmp_files = list(job_dir.glob(".tmp_*"))
    assert len(tmp_files) == 0, f"Leftover temporary files found: {tmp_files}"


def test_p04_sweeper_evicts_only_after_ttl_expiration():
    """Verifies background sweeper preserves fresh files and only evicts files older than TTL."""
    # Configure 2 second TTL
    short_config = ScratchDiskConfig(
        root_dir=SCRATCH_DIR,
        ttl_seconds=2,
        sweep_interval_seconds=1,
    )
    scratch = ScratchDiskManager(short_config)

    job_id_fresh = "job-fresh-01"
    job_id_expired = "job-expired-01"

    asyncio.run(scratch.save_artifact(job_id_fresh, "fresh.png", b"FRESH_DATA"))
    asyncio.run(scratch.save_artifact(job_id_expired, "expired.png", b"EXPIRED_DATA"))

    fresh_path = scratch.get_artifact_path(job_id_fresh, "fresh.png")
    expired_path = scratch.get_artifact_path(job_id_expired, "expired.png")

    assert fresh_path.exists()
    assert expired_path.exists()

    # Manually backdate the mtime/atime of the expired file to 10 seconds ago
    old_time = time.time() - 10
    os.utime(str(expired_path), (old_time, old_time))
    os.utime(str(scratch.get_job_dir(job_id_expired)), (old_time, old_time))

    # Run sweep_once
    scratch.sweep_once()

    # Expired file should be gone, fresh file should still exist
    assert not expired_path.exists(), "Expired artifact was not evicted by sweeper"
    assert fresh_path.exists(), "Fresh artifact was incorrectly evicted by sweeper"
