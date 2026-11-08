"""
tests/adversarial/test_adversarial_m3_challenger_verification.py
Adversarial empirical stress tests for Milestone M3 (GPU Runtime Simplification):
1. Deterministic Failure & Watchdog:
   - Abruptly disconnected worker (heartbeat stops > 60s) -> transitions to FAILED with code="WORKER_DISCONNECTED".
   - GET /v1/jobs/{id}/result returns HTTP 502 Bad Gateway with sanitized error envelope.
   - Job is NEVER re-enqueued (state != "QUEUED"), guaranteeing zero duplicate inference execution.
   - GPU partition is immediately released upon failure.
   - Disconnected worker cannot resurrect the job via late progress/complete requests.
   - Streaming job disconnect handled cleanly.
   - Execution timeout (> 300s) fails with code="TIMEOUT" and GET /v1/jobs/{id}/result returns HTTP 504.
2. VRAM Pre-Admission Guard:
   - Submit job via HTTP POST /v1/jobs with model requiring > 14.2 GB VRAM -> rejected immediately with HTTP 400 INVALID_REQUEST.
   - Boundary tests: 14.2 GB admitted vs 14.21 GB rejected.
   - Rejected job is never admitted to queue or job store.
3. Dual-GPU Concurrent Disconnect Stress:
   - Both GPU 0 and GPU 1 workers disconnect simultaneously -> both fail deterministically and both partitions release.
"""
from datetime import datetime, timedelta, timezone
import os
from unittest.mock import MagicMock
import pytest

TEST_API_KEY = "sk-test-challenger-m3-key-2026"
os.environ["FACTORY_API_KEY"] = TEST_API_KEY
os.environ["ENVIRONMENT"] = "test"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["FACTORY_DB_BACKEND"] = "memory"

from starlette.testclient import TestClient

from factory_auth.config import AuthConfig
from factory_contracts.jobs.job_contract import (
    JobState,
    JobSubmissionRequest,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    ProgressUpdateRequest,
    WorkerLeaseRequest,
)
from factory_contracts.manifests.model import (
    AllocationPolicy,
    ModelHardwareSpec,
    ModelLimitsSpec,
    ModelManifest,
    ModelMetadata,
    ModelStatus,
    ModelValidationStatus,
    ModelWeightFormat,
    ModelWeightsSpec,
    PrecisionType,
    QuantizationType,
)
import asyncio
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog, watchdog_loop
from factory.apps.control_api.src.main import app as control_api_app
from model_registry.loader import ManifestLoader

AUTH_HEADERS = {
    "Authorization": f"Bearer {TEST_API_KEY}",
    "Content-Type": "application/json",
}


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("FACTORY_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "memory")
    control_api_app.state.auth_config = AuthConfig(api_key=TEST_API_KEY, jwt_secret=None, environment="test")


def _create_mock_heavy_model(model_id: str, vram_gb: float) -> ModelManifest:
    return ModelManifest(
        apiVersion="factory.ai/v1",
        kind="ModelManifest",
        metadata=ModelMetadata(
            id=model_id,
            family="heavy-llm",
            display_name=f"Heavy Model {vram_gb}GB",
            service_id="text-service",
            version="1.0.0",
        ),
        weights=ModelWeightsSpec(
            primary_checkpoint=f"meta/{model_id}",
            format=ModelWeightFormat.SAFETENSORS,
            quantization=QuantizationType.AWQ,
            default_precision=PrecisionType.FP16,
        ),
        hardware=ModelHardwareSpec(
            min_vram_gb=vram_gb,
            context_window=4096,
            preferred_gpu="t4",
            allocation_policy=AllocationPolicy.SINGLE_GPU,
        ),
        capabilities={"chat_completion": True},
        status=ModelStatus.FULL,
        coverage_percent=100.0,
        missing_capabilities=[],
        validation_status=ModelValidationStatus.PENDING_LIVE,
        validation_notes="Challenger test model.",
        limits=ModelLimitsSpec(),
    )


# ==============================================================================
# CHALLENGE SUITE 1: Deterministic Failure on Disconnected Worker & 502 Result
# ==============================================================================

@pytest.mark.asyncio
async def test_adversarial_worker_disconnect_watchdog_deterministic_502_and_gpu_release():
    """
    Challenge 1:
    - Worker leases job on GPU 0.
    - Worker heartbeat stops (> 60s).
    - Watchdog marks job FAILED with code="WORKER_DISCONNECTED".
    - GET /v1/jobs/{id}/result returns HTTP 502 Bad Gateway.
    - Job is NEVER re-enqueued (state != QUEUED), guaranteeing zero duplicate inference execution.
    - GPU partition is immediately released.
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        # 1. Submit job targeting GPU 0 (image service)
        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Adversarial disconnect prompt"},
                priority=10,
            ),
            request_id="req-challenger-dc-01",
        )
        assert job.state == JobState.QUEUED

        # 2. Worker leases job on GPU 0
        lease = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-abrupt-dc-01",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
            )
        )
        assert lease.job is not None
        assert lease.job.job_id == job.job_id
        assert scheduler.is_gpu_busy(0) is True

        # Verify job is in RUNNING state
        running_job = await repo.get_job(job.job_id)
        assert running_job.state == JobState.RUNNING

        # 3. Simulate abrupt worker death: backdate heartbeat to 70 seconds ago (> 60s threshold)
        stale_hb = (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat()
        await repo.upsert_worker_heartbeat(
            worker_id="worker-abrupt-dc-01",
            active_job_id=job.job_id,
            subsystem="image",
            gpu_device=0,
        )
        async with repo._lock:
            repo._workers["worker-abrupt-dc-01"]["last_heartbeat"] = stale_hb

        # 4. Trigger Watchdog
        failed_count = await check_active_jobs_watchdog(
            repository=repo,
            scheduler=scheduler,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
        )
        assert failed_count == 1

        # 5. Verify job is in terminal FAILED state with code="WORKER_DISCONNECTED"
        failed_job = await repo.get_job(job.job_id)
        assert failed_job.state == JobState.FAILED
        assert failed_job.error is not None
        assert failed_job.error.code == "WORKER_DISCONNECTED"

        # 6. Verify zero duplicate re-queueing invariant: state != QUEUED, not eligible for lease
        assert failed_job.state != JobState.QUEUED
        assert failed_job.worker is None
        eligible_jobs = await repo.find_eligible_jobs_for_worker(
            subsystem="image",
            supported_models=["qwen-image-2.1"],
            supported_tasks=["text_to_image"],
        )
        assert all(j.job_id != job.job_id for j in eligible_jobs)

        # 7. Verify GPU partition was immediately released
        assert scheduler.is_gpu_busy(0) is False

        # 8. Verify GET /v1/jobs/{id}/result returns HTTP 502 Bad Gateway
        res_result = client.get(f"/v1/jobs/{job.job_id}/result", headers=AUTH_HEADERS)
        assert res_result.status_code == 502, f"Expected 502, got {res_result.status_code}: {res_result.text}"
        data = res_result.json()
        assert "error" in data
        assert data["error"]["code"] == "WORKER_DISCONNECTED"
        assert "Worker heartbeat timed out" in data["error"]["message"]

        # 9. Verify subsequent job on GPU 0 can be leased immediately
        job_next, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Next image on freed GPU 0"},
            ),
            request_id="req-challenger-next",
        )
        lease_next = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-healthy-02",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
            )
        )
        assert lease_next.job is not None
        assert lease_next.job.job_id == job_next.job_id
        assert scheduler.is_gpu_busy(0) is True


@pytest.mark.asyncio
async def test_adversarial_streaming_worker_disconnect_watchdog():
    """
    Challenge 2:
    Verify that an in-progress job in STREAMING state that abruptly disconnects
    is also caught by watchdog, transitions to FAILED (WORKER_DISCONNECTED),
    releases GPU, and returns HTTP 502 on GET /v1/jobs/{id}/result.
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="text",
                model="qwen-text",
                task="chat_completion",
                inputs={"messages": [{"role": "user", "content": "Stream me a story"}]},
            ),
            request_id="req-stream-dc-01",
        )
        lease = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-stream-dc",
                subsystem="text",
                supported_models=["qwen-text"],
                supported_tasks=["chat_completion"],
                gpu_device=0,
            )
        )
        assert lease.job is not None
        assert scheduler.is_gpu_busy(0) is True

        # Transition to STREAMING state
        async with repo._lock:
            repo._jobs[job.job_id].state = JobState.STREAMING

        streaming_job = await repo.get_job(job.job_id)
        assert streaming_job.state == JobState.STREAMING

        # Backdate worker heartbeat to 80s ago
        stale_hb = (datetime.now(timezone.utc) - timedelta(seconds=80)).isoformat()
        await repo.upsert_worker_heartbeat(
            worker_id="worker-stream-dc",
            active_job_id=job.job_id,
            subsystem="text",
            gpu_device=0,
        )
        async with repo._lock:
            repo._workers["worker-stream-dc"]["last_heartbeat"] = stale_hb

        # Run watchdog
        failed_count = await check_active_jobs_watchdog(
            repository=repo,
            scheduler=scheduler,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
        )
        assert failed_count == 1

        # Assertions
        failed_job = await repo.get_job(job.job_id)
        assert failed_job.state == JobState.FAILED
        assert failed_job.error.code == "WORKER_DISCONNECTED"
        assert scheduler.is_gpu_busy(0) is False

        res_result = client.get(f"/v1/jobs/{job.job_id}/result", headers=AUTH_HEADERS)
        assert res_result.status_code == 502
        assert res_result.json()["error"]["code"] == "WORKER_DISCONNECTED"


@pytest.mark.asyncio
async def test_adversarial_dead_worker_cannot_resurrect_failed_job():
    """
    Challenge 3:
    Stress-test zero duplicate inference guarantee:
    After watchdog marks job FAILED, if the disconnected worker recovers and tries
    to report progress or complete the job, it must be rejected or no-oped,
    preventing state resurrection or duplicate completion.
    """
    with TestClient(control_api_app, base_url="http://testserver"):
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Zombie worker challenge"},
            ),
            request_id="req-zombie-01",
        )
        await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="zombie-worker",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
            )
        )

        # Backdate heartbeat and trigger watchdog
        stale_hb = (datetime.now(timezone.utc) - timedelta(seconds=75)).isoformat()
        await repo.upsert_worker_heartbeat(
            worker_id="zombie-worker",
            active_job_id=job.job_id,
            subsystem="image",
            gpu_device=0,
        )
        async with repo._lock:
            repo._workers["zombie-worker"]["last_heartbeat"] = stale_hb

        await check_active_jobs_watchdog(
            repository=repo,
            scheduler=scheduler,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
        )

        job_after_wd = await repo.get_job(job.job_id)
        assert job_after_wd.state == JobState.FAILED

        # 1. Zombie worker attempts progress update -> Must not mutate state
        prog_res = await manager.update_progress(
            job.job_id,
            ProgressUpdateRequest(
                stage="sampling",
                step=10,
                total=20,
                percent=50.0,
                worker_id="zombie-worker",
            ),
        )
        assert prog_res.state == JobState.FAILED

        # 2. Zombie worker attempts job completion -> Must fail transition or authorization
        with pytest.raises(Exception):
            await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    worker_id="zombie-worker",
                    inference_duration_ms=1000.0,
                    result={"res": "late"},
                ),
            )

        # State remains strictly FAILED
        final_job = await repo.get_job(job.job_id)
        assert final_job.state == JobState.FAILED
        assert final_job.error.code == "WORKER_DISCONNECTED"


@pytest.mark.asyncio
async def test_adversarial_watchdog_execution_timeout_504_gateway_timeout():
    """
    Challenge 4:
    Verify that when job execution exceeds timeout (> 300s),
    watchdog transitions job to FAILED with code="TIMEOUT", releases GPU partition,
    and GET /v1/jobs/{id}/result returns HTTP 504 Gateway Timeout.
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="video",
                model="wan-2.2",
                task="text_to_video",
                inputs={"prompt": "Long running video job"},
            ),
            request_id="req-timeout-video",
        )
        await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-slow-video",
                subsystem="video",
                supported_models=["wan-2.2"],
                supported_tasks=["text_to_video"],
                gpu_device=1,
            )
        )
        assert scheduler.is_gpu_busy(1) is True

        # Heartbeat is recent (worker is alive), but job execution started 950s ago (> 900s task timeout)
        now = datetime.now(timezone.utc)
        recent_hb = now.isoformat()
        old_started = now - timedelta(seconds=950)

        await repo.upsert_worker_heartbeat(
            worker_id="worker-slow-video",
            active_job_id=job.job_id,
            subsystem="video",
            gpu_device=1,
        )
        async with repo._lock:
            repo._workers["worker-slow-video"]["last_heartbeat"] = recent_hb
            repo._jobs[job.job_id].started_at = old_started

        # Run watchdog with 300s job timeout
        failed_count = await check_active_jobs_watchdog(
            repository=repo,
            scheduler=scheduler,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
            default_job_timeout_seconds=300.0,
        )
        assert failed_count == 1

        # Check job state and error code
        timeout_job = await repo.get_job(job.job_id)
        assert timeout_job.state == JobState.FAILED
        assert timeout_job.error.code == "TIMEOUT"

        # Check GPU 1 is released
        assert scheduler.is_gpu_busy(1) is False

        # Check GET /v1/jobs/{id}/result returns HTTP 504
        res_result = client.get(f"/v1/jobs/{job.job_id}/result", headers=AUTH_HEADERS)
        assert res_result.status_code == 504
        data = res_result.json()
        assert data["error"]["code"] == "GATEWAY_TIMEOUT"
        assert "exceeded timeout threshold" in data["error"]["message"]


@pytest.mark.asyncio
async def test_adversarial_dual_gpu_concurrent_disconnect_watchdog():
    """
    Challenge 5:
    Both GPU 0 and GPU 1 workers experience concurrent disconnect.
    Watchdog sweep must deterministically fail both jobs, free both discrete GPU partitions,
    and return 502 for both results.
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        # Submit GPU 0 and GPU 1 jobs
        job_gpu0, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Dual crash GPU 0"},
            ),
            request_id="req-dual-gpu0",
        )
        job_gpu1, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="video",
                model="wan-2.2",
                task="text_to_video",
                inputs={"prompt": "Dual crash GPU 1"},
            ),
            request_id="req-dual-gpu1",
        )

        await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="crash-worker-gpu0",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
            )
        )
        await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="crash-worker-gpu1",
                subsystem="video",
                supported_models=["wan-2.2"],
                supported_tasks=["text_to_video"],
                gpu_device=1,
            )
        )
        assert scheduler.is_gpu_busy(0) is True
        assert scheduler.is_gpu_busy(1) is True

        # Backdate both heartbeats to 90s ago
        stale_hb = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()
        await repo.upsert_worker_heartbeat(
            worker_id="crash-worker-gpu0",
            active_job_id=job_gpu0.job_id,
            subsystem="image",
            gpu_device=0,
        )
        await repo.upsert_worker_heartbeat(
            worker_id="crash-worker-gpu1",
            active_job_id=job_gpu1.job_id,
            subsystem="video",
            gpu_device=1,
        )
        async with repo._lock:
            repo._workers["crash-worker-gpu0"]["last_heartbeat"] = stale_hb
            repo._workers["crash-worker-gpu1"]["last_heartbeat"] = stale_hb

        # Run watchdog sweep
        failed_count = await check_active_jobs_watchdog(
            repository=repo,
            scheduler=scheduler,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
        )
        assert failed_count == 2

        # Both discrete GPUs released
        assert scheduler.is_gpu_busy(0) is False
        assert scheduler.is_gpu_busy(1) is False

        # Both return 502 Bad Gateway
        res0 = client.get(f"/v1/jobs/{job_gpu0.job_id}/result", headers=AUTH_HEADERS)
        res1 = client.get(f"/v1/jobs/{job_gpu1.job_id}/result", headers=AUTH_HEADERS)
        assert res0.status_code == 502
        assert res1.status_code == 502


# ==============================================================================
# CHALLENGE SUITE 2: VRAM Pre-Admission Guard via HTTP POST /v1/jobs
# ==============================================================================

def test_adversarial_vram_pre_admission_rejection_over_14_2_gb_http():
    """
    Challenge 6:
    Verify via genuine HTTP POST /v1/jobs that any model requiring > 14.2 GB VRAM
    is rejected immediately at pre-admission with HTTP 400 and error code 'INVALID_REQUEST',
    and that no job is created in the repository.
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        heavy_model = _create_mock_heavy_model("heavy-model-32b", 16.0)

        # Mock ManifestLoader on app state
        real_loader = getattr(control_api_app.state, "manifest_loader", None)
        mock_loader = MagicMock(spec=ManifestLoader)
        base_models = real_loader.load_models() if real_loader else {}
        merged_models = dict(base_models)
        merged_models["heavy-model-32b"] = heavy_model
        mock_loader.load_models.return_value = merged_models
        control_api_app.state.manifest_loader = mock_loader

        payload = {
            "service": "text",
            "model": "heavy-model-32b",
            "task": "chat_completion",
            "inputs": {"messages": [{"role": "user", "content": "Run on 16GB VRAM"}]},
        }
        res = client.post("/v1/jobs", json=payload, headers=AUTH_HEADERS)
        assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
        data = res.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_REQUEST"
        assert "Requested model exceeds max allocatable VRAM of 14.2 GB" in data["error"]["message"]

        # Verify job was never enqueued
        repo = getattr(control_api_app.state, "repository", None)
        if repo and hasattr(repo, "_jobs"):
            assert all(j.model != "heavy-model-32b" for j in repo._jobs.values())


def test_adversarial_vram_pre_admission_boundary_conditions():
    """
    Challenge 7:
    Boundary condition testing for VRAM Pre-Admission Guard:
    - Model with min_vram_gb = 14.2 -> Permitted (HTTP 201 Created)
    - Model with min_vram_gb = 14.21 -> Rejected (HTTP 400 INVALID_REQUEST)
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        model_14_2 = _create_mock_heavy_model("model-boundary-14-2", 14.2)
        model_14_21 = _create_mock_heavy_model("model-boundary-14-21", 14.21)

        real_loader = getattr(control_api_app.state, "manifest_loader", None)
        mock_loader = MagicMock(spec=ManifestLoader)
        base_models = real_loader.load_models() if real_loader else {}
        merged = dict(base_models)
        merged["model-boundary-14-2"] = model_14_2
        merged["model-boundary-14-21"] = model_14_21
        mock_loader.load_models.return_value = merged
        control_api_app.state.manifest_loader = mock_loader

        # 1. 14.21 GB must be rejected
        res_reject = client.post(
            "/v1/jobs",
            json={
                "service": "text",
                "model": "model-boundary-14-21",
                "task": "chat_completion",
                "inputs": {"messages": [{"role": "user", "content": "Boundary test"}]},
            },
            headers=AUTH_HEADERS,
        )
        assert res_reject.status_code == 400
        assert res_reject.json()["error"]["code"] == "INVALID_REQUEST"
        assert "Requested model exceeds max allocatable VRAM of 14.2 GB" in res_reject.json()["error"]["message"]

        # 2. 14.2 GB must pass pre-admission guard
        res_admit = client.post(
            "/v1/jobs",
            json={
                "service": "text",
                "model": "model-boundary-14-2",
                "task": "chat_completion",
                "inputs": {"messages": [{"role": "user", "content": "Boundary test"}]},
            },
            headers=AUTH_HEADERS,
        )
        assert res_admit.status_code == 201, f"Expected 201, got {res_admit.status_code}: {res_admit.text}"
        assert "job_id" in res_admit.json()


@pytest.mark.asyncio
async def test_adversarial_watchdog_autonomous_background_loop_sweep():
    """
    Challenge 8:
    Verify the actual asynchronous watchdog_loop background task automatically
    sweeps and fails disconnected workers without manual intervention.
    """
    with TestClient(control_api_app, base_url="http://testserver") as client:
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Background loop test"},
            ),
            request_id="req-bg-loop-01",
        )
        await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-bg-01",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
            )
        )
        assert scheduler.is_gpu_busy(0) is True

        # Backdate heartbeat to 70s ago
        stale_hb = (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat()
        await repo.upsert_worker_heartbeat(
            worker_id="worker-bg-01",
            active_job_id=job.job_id,
            subsystem="image",
            gpu_device=0,
        )
        async with repo._lock:
            repo._workers["worker-bg-01"]["last_heartbeat"] = stale_hb

        # Start background watchdog loop running with 0.1s interval
        task = asyncio.create_task(
            watchdog_loop(control_api_app, interval_seconds=0.1, heartbeat_timeout_seconds=60.0)
        )
        try:
            # Allow loop to execute a sweep
            await asyncio.sleep(0.3)

            # Job must be transitioned to FAILED autonomously
            updated_job = await repo.get_job(job.job_id)
            assert updated_job.state == JobState.FAILED
            assert updated_job.error.code == "WORKER_DISCONNECTED"
            assert scheduler.is_gpu_busy(0) is False

            # Result endpoint returns 502
            res = client.get(f"/v1/jobs/{job.job_id}/result", headers=AUTH_HEADERS)
            assert res.status_code == 502
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_adversarial_healthy_worker_heartbeat_keeps_job_alive():
    """
    Challenge 9:
    Verify that an active worker sending periodic heartbeats is NOT terminated by watchdog,
    and GPU remains safely locked for inference.
    """
    with TestClient(control_api_app, base_url="http://testserver"):
        repo = control_api_app.state.repository
        scheduler = control_api_app.state.scheduler
        manager = control_api_app.state.job_manager
        coordinator = control_api_app.state.lease_coordinator

        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Healthy worker prompt"},
            ),
            request_id="req-healthy-01",
        )
        await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-alive-01",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
                gpu_device=0,
            )
        )
        assert scheduler.is_gpu_busy(0) is True

        # Send fresh heartbeat
        await repo.upsert_worker_heartbeat(
            worker_id="worker-alive-01",
            active_job_id=job.job_id,
            subsystem="image",
            gpu_device=0,
        )

        # Run watchdog with 60s timeout
        failed_count = await check_active_jobs_watchdog(
            repository=repo,
            scheduler=scheduler,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
        )
        assert failed_count == 0

        # Job is still RUNNING, GPU still busy
        job_active = await repo.get_job(job.job_id)
        assert job_active.state == JobState.RUNNING
        assert scheduler.is_gpu_busy(0) is True

