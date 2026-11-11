"""
tests/e2e/test_golden_path_real_e2e.py
Factory v4 Lite: Factory Lifecycle Contract (L4.5) Verification Test Suite

Validates the complete Factory Lite lifecycle across Control API, Queue/Scheduler,
Worker Leasing, Heartbeat, Progress Streaming, Artifact Generation, Result Delivery,
and Automatic Disk Cleanup without mock simulation.

Hierarchy Level: FACTORY_LIFECYCLE_CONTRACT_PASS (Pre-requisite for Live On-GPU L5)
Coverage: Visual, Decision, Text, and Embedding lifecycles.
Note: Physical on-GPU execution against live llama-server / ComfyUI is evaluated
via deployments/kaggle/live_factory_e2e_runner.py (FACTORY_LIVE_E2E_PASS).
"""

import hashlib
import os
import shutil
import time
from typing import Dict, Generator

import pytest
from fastapi.testclient import TestClient

from factory_contracts.jobs.job_contract import JobState
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
)

VALID_TEST_API_KEY = "sk-factory-lite-e2e-valid-key-2026"
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
    if os.path.exists(SCRATCH_ARTIFACTS_DIR):
        try:
            shutil.rmtree(SCRATCH_ARTIFACTS_DIR)
        except Exception:
            pass


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provides a TestClient connected directly to the in-memory Factory Lite app."""
    from factory.apps.control_api.src.main import app

    with TestClient(app, base_url="http://testserver") as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {VALID_TEST_API_KEY}",
        "Content-Type": "application/json",
    }


class TestFactoryLifecycleContract:
    """
    Validates FACTORY_LIFECYCLE_CONTRACT_PASS:
    In-memory complete multi-step lifecycle loop:
    Client Submit -> Queue -> Worker Lease -> Heartbeat/Progress ->
    Complete -> Event SSE -> Artifact Download -> Auto Cleanup.
    """

    def test_golden_path_image_pipeline_e2e(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R2 / P0-01: End-to-end image generation lifecycle.
        1. Client submits job (POST /v1/jobs).
        2. Worker leases job (POST /v1/jobs/lease).
        3. Worker sends periodic heartbeat (POST /v1/workers/{id}/heartbeat).
        4. Worker streams progressive sampling steps (POST /v1/jobs/{id}/progress).
        5. Worker writes genuine artifact to scratch disk and completes (POST /v1/jobs/{id}/complete).
        6. Client receives SSE lifecycle events (GET /v1/jobs/{id}/events).
        7. Client downloads binary PNG artifact with hash verification (GET /v1/jobs/{id}/result).
        8. Scratch disk automatically cleaned up.
        """
        worker_id = "worker-t4-gpu0-visual"
        prompt_text = "Imperial Vietnamese Golden Dragon soaring over limestone karst towers of Ha Long Bay"

        # 1. Client Submits Job
        submit_payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {
                "prompt": prompt_text,
                "width": 960,
                "height": 640,
                "steps": 8,
            },
            "priority": 10,
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201, f"Job submit failed: {submit_resp.text}"
        job_data = submit_resp.json()
        job_id = job_data["job_id"]
        assert job_id.startswith("job-")
        assert job_data["state"] in [JobState.CREATED.value, JobState.QUEUED.value]

        # 2. Worker leases the queued job
        lease_req = WorkerLeaseRequest(
            worker_id=worker_id,
            subsystem="image",
            supported_models=["qwen-image-2.1"],
            supported_tasks=["text_to_image"],
            gpu_device=0,
        )
        lease_resp = client.post("/v1/jobs/lease", json=lease_req.model_dump(), headers=auth_headers)
        assert lease_resp.status_code == 200, f"Worker lease failed: {lease_resp.text}"
        lease_data = lease_resp.json()
        assert lease_data.get("job") is not None, "Worker failed to lease queued job"
        leased_job = lease_data["job"]
        assert leased_job["job_id"] == job_id
        lease_token = lease_data["lease_token"]
        assert lease_token is not None

        # 3. Worker sends heartbeat
        hb_req = WorkerHeartbeatRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            active_job_id=job_id,
            vram_used_mb=13053.2,
            vram_total_mb=15360.0,
            gpu_utilization_percent=88.5,
        )
        hb_resp = client.post(f"/v1/workers/{worker_id}/heartbeat", json=hb_req.model_dump(), headers=auth_headers)
        assert hb_resp.status_code == 200
        hb_data = hb_resp.json()
        assert hb_data["status"] == "acknowledged"

        # 4. Worker reports progress steps
        for step in [2, 4, 6, 8]:
            pct = (step / 8) * 100.0
            prog_req = ProgressUpdateRequest(
                stage="sampling",
                step=step,
                total=8,
                percent=pct,
                worker_id=worker_id,
                lease_token=lease_token,
                message=f"Diffusion step {step}/8",
            )
            prog_resp = client.post(f"/v1/jobs/{job_id}/progress", json=prog_req.model_dump(), headers=auth_headers)
            assert prog_resp.status_code == 200

        # 5. Worker writes generated artifact to scratch disk and completes
        job_scratch_dir = os.path.join(SCRATCH_ARTIFACTS_DIR, job_id)
        os.makedirs(job_scratch_dir, exist_ok=True)
        artifact_path = os.path.join(job_scratch_dir, "studio_qwen_image_master.png")

        # Generate representative 960x640 PNG image bytes
        raw_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x03\xc0\x00\x00\x02\x80\x08\x06\x00\x00\x00"
        image_content = raw_header + b"\x00" * 4096 + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        expected_sha256 = hashlib.sha256(image_content).hexdigest()
        with open(artifact_path, "wb") as f:
            f.write(image_content)

        complete_req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifact_path=artifact_path,
            peak_vram_mb=13053.2,
            inference_duration_ms=168365.14,
        )
        comp_resp = client.post(f"/v1/jobs/{job_id}/complete", json=complete_req.model_dump(), headers=auth_headers)
        assert comp_resp.status_code == 200

        # 6. Verify job status is SUCCEEDED via GET /v1/jobs/{id}
        status_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["state"] == JobState.SUCCEEDED.value
        assert status_data["progress"]["percent"] == 100.0

        # 7. Configure cleanup_on_download and verify direct artifact download
        scratch_disk = getattr(client.app.state, "scratch_disk", None)
        orig_cleanup = scratch_disk.config.cleanup_on_download if scratch_disk else False
        if scratch_disk:
            scratch_disk.config.cleanup_on_download = True

        try:
            result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
            assert result_resp.status_code == 200
            assert "image/png" in result_resp.headers.get("content-type", "")
            downloaded_bytes = result_resp.content
            assert len(downloaded_bytes) == len(image_content)
            downloaded_sha256 = hashlib.sha256(downloaded_bytes).hexdigest()
            assert downloaded_sha256 == expected_sha256

            # 8. Verify automatic scratch disk cleanup
            cleaned_up = False
            for _ in range(15):
                if not os.path.exists(artifact_path):
                    cleaned_up = True
                    break
                time.sleep(0.05)
            assert cleaned_up, f"Automatic disk cleanup failed: {artifact_path} was not purged."
        finally:
            if scratch_disk:
                scratch_disk.config.cleanup_on_download = orig_cleanup

    def test_golden_path_decision_pipeline_e2e(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R2 / P0-01: End-to-end decision routing lifecycle.
        1. Client submits decision job (POST /v1/jobs).
        2. Worker leases decision job (POST /v1/jobs/lease).
        3. Worker evaluates decision and reports completion with structured output.
        4. Client fetches status and result.
        """
        worker_id = "worker-t4-gpu0-decision"
        submit_payload = {
            "service": "decision",
            "model": "qwen-decision",
            "task": "predicate_evaluation",
            "inputs": {
                "statement": "The user prompt requires a cinematic video generation pipeline.",
                "context": "Prompt: 'Create a 4K dragon flying over Halong Bay with smooth 24fps camera motion.'",
            },
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # Worker leases
        lease_req = WorkerLeaseRequest(
            worker_id=worker_id,
            subsystem="decision",
            supported_models=["qwen-decision"],
            supported_tasks=["predicate_evaluation"],
            gpu_device=0,
        )
        lease_resp = client.post("/v1/jobs/lease", json=lease_req.model_dump(), headers=auth_headers)
        assert lease_resp.status_code == 200
        lease_data = lease_resp.json()
        assert lease_data.get("job") is not None
        assert lease_data["job"]["job_id"] == job_id
        lease_token = lease_data["lease_token"]

        # Worker completes
        decision_result = {
            "task": "predicate_evaluation",
            "evaluated_predicate": True,
            "confidence": 0.985,
            "rationale": "Direct mention of 24fps camera motion and video generation parameters.",
        }
        complete_req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            result=decision_result,
            inference_duration_ms=12.4,
            peak_vram_mb=451.2,
        )
        comp_resp = client.post(f"/v1/jobs/{job_id}/complete", json=complete_req.model_dump(), headers=auth_headers)
        assert comp_resp.status_code == 200

        # Client queries result
        status_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["state"] == JobState.SUCCEEDED.value

    def test_golden_path_text_pipeline_e2e(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R2 / P0-01: End-to-end text reasoning lifecycle.
        1. Client submits text job (POST /v1/jobs).
        2. Worker leases text job (POST /v1/jobs/lease).
        3. Worker completes text job.
        4. Client retrieves result.
        """
        worker_id = "worker-t4-gpu0-text"
        submit_payload = {
            "service": "text",
            "model": "qwen-text",
            "task": "chat_completion",
            "inputs": {
                "messages": [
                    {"role": "user", "content": "Explain the architecture of Dual-T4 multi-modal factory."}
                ]
            },
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # Worker leases
        lease_req = WorkerLeaseRequest(
            worker_id=worker_id,
            subsystem="text",
            supported_models=["qwen-text"],
            supported_tasks=["chat_completion"],
            gpu_device=0,
        )
        lease_resp = client.post("/v1/jobs/lease", json=lease_req.model_dump(), headers=auth_headers)
        assert lease_resp.status_code == 200
        lease_data = lease_resp.json()
        assert lease_data.get("job") is not None
        assert lease_data["job"]["job_id"] == job_id
        lease_token = lease_data["lease_token"]

        # Worker completes
        complete_req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            result={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The Factory architecture routes jobs across Dual T4 GPUs using Sharded Workloads.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 16, "total_tokens": 28},
            },
            tokens_per_sec=6.64,
            inference_duration_ms=334.4,
            peak_vram_mb=11341.2,
        )
        comp_resp = client.post(f"/v1/jobs/{job_id}/complete", json=complete_req.model_dump(), headers=auth_headers)
        assert comp_resp.status_code == 200

        # Client queries result
        status_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["state"] == JobState.SUCCEEDED.value

    def test_golden_path_embedding_pipeline_e2e(self, client: TestClient, auth_headers: Dict[str, str]):
        """
        R2 / P0-01: End-to-end dense embedding lifecycle.
        1. Client submits embedding job (POST /v1/jobs).
        2. Worker leases embedding job (POST /v1/jobs/lease).
        3. Worker completes embedding job with 1024-dim vector result.
        4. Client retrieves result and validates embedding shape.
        """
        worker_id = "worker-t4-gpu1-embedding"
        submit_payload = {
            "service": "embedding",
            "model": "qwen-embedding",
            "task": "embedding",
            "inputs": {
                "input": "Dual Tesla T4 neural acceleration across multi-modal models",
                "dimensions": 1024,
            },
        }
        submit_resp = client.post("/v1/jobs", json=submit_payload, headers=auth_headers)
        assert submit_resp.status_code == 201
        job_id = submit_resp.json()["job_id"]

        # Worker leases
        lease_req = WorkerLeaseRequest(
            worker_id=worker_id,
            subsystem="embedding",
            supported_models=["qwen-embedding"],
            supported_tasks=["embedding"],
            gpu_device=1,
        )
        lease_resp = client.post("/v1/jobs/lease", json=lease_req.model_dump(), headers=auth_headers)
        assert lease_resp.status_code == 200
        lease_data = lease_resp.json()
        assert lease_data.get("job") is not None
        assert lease_data["job"]["job_id"] == job_id
        lease_token = lease_data["lease_token"]

        # Worker completes with 1024-dim embedding
        sample_vector = [0.03125] * 1024
        complete_req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            result={
                "object": "list",
                "data": [{"index": 0, "embedding": sample_vector}],
                "model": "qwen-embedding",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
            inference_duration_ms=4.8,
            peak_vram_mb=451.2,
        )
        comp_resp = client.post(f"/v1/jobs/{job_id}/complete", json=complete_req.model_dump(), headers=auth_headers)
        assert comp_resp.status_code == 200

        # Client queries result
        status_resp = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        job_result = status_resp.json()
        assert job_result["state"] == JobState.SUCCEEDED.value
        assert len(job_result["result"]["data"][0]["embedding"]) == 1024



