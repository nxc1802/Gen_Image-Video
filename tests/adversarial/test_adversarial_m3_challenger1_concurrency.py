"""
tests/adversarial/test_adversarial_m3_challenger1_concurrency.py
Adversarial Concurrency and Dual T4 Partitioning Stress Harness for Milestone M3.

Empirically tests and stress-probes:
1. GPU 0 strict single-concurrency isolation and worker mutual exclusion via HTTP API.
2. Discrete GPU 1 independence under active GPU 0 load.
3. GPU 0 release upon completion and immediate re-acquisition by waiting workers.
4. High-contention burst with 50 concurrent workers: zero double-leases, bounded max_concurrent_services <= 1.
5. Automatic GPU partition release upon cancellation (DELETE / POST cancel) and failure (POST fail).
6. Watchdog disconnect timeout release of GPU partition.
7. VRAM pre-admission guard boundary conditions (14.2 GB threshold).
"""
import asyncio
from datetime import datetime, timedelta, timezone
import os
import pytest
import httpx

from factory_auth.config import AuthConfig
from factory_contracts.jobs.job_contract import JobState
from factory_contracts.errors import InvalidRequestError
from factory.apps.control_api.src.main import app as control_plane_app, lifespan
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog
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
from model_registry.loader import ManifestLoader

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}

os.environ["FACTORY_API_KEY"] = AUTH_TOKEN
os.environ["STORAGE_BACKEND"] = "local"
os.environ["FACTORY_DB_BACKEND"] = "memory"


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("FACTORY_API_KEY", AUTH_TOKEN)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "memory")
    control_plane_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")




@pytest.mark.asyncio
async def test_gpu0_concurrency_isolation_and_gpu1_independence_http():
    """
    1. Submit concurrent jobs to GPU 0 (image, text).
    2. Verify Worker 1 leases job 1, GPU 0 is marked busy, and Worker 2 receives job=None while job 1 is running.
    3. Verify discrete GPU 1 independence: while GPU 0 is busy, Worker 3 can lease a GPU 1 job (video/music) without being blocked by GPU 0.
    4. When job 1 completes, verify GPU 0 is freed and Worker 2 can now acquire the next GPU 0 job.
    """
    async with lifespan(control_plane_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers=AUTH_HEADERS,
            timeout=30.0,
        ) as client:
            scheduler = control_plane_app.state.scheduler

            # Ensure initial clean state
            assert scheduler.is_gpu_busy(0) is False
            assert scheduler.is_gpu_busy(1) is False

            # Step 1: Submit 2 jobs to GPU 0 (Image service)
            sub1 = await client.post("/v1/jobs", json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "GPU 0 Job 1"},
                "priority": 20,
            })
            assert sub1.status_code == 201
            job1_id = sub1.json()["job_id"]

            sub2 = await client.post("/v1/jobs", json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "GPU 0 Job 2"},
                "priority": 10,
            })
            assert sub2.status_code == 201
            job2_id = sub2.json()["job_id"]

            # Also submit a job for GPU 1 (Video service)
            sub3 = await client.post("/v1/jobs", json={
                "service": "video",
                "model": "wan-2.2",
                "task": "text_to_video",
                "inputs": {"prompt": "GPU 1 Job 3", "fps": 24},
                "priority": 15,
            })
            assert sub3.status_code == 201
            job3_id = sub3.json()["job_id"]

            # Step 2: Worker 1 leases Job 1 on GPU 0
            lease1_res = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-gpu0-01",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert lease1_res.status_code == 200
            lease1 = lease1_res.json()
            assert lease1["job"] is not None
            assert lease1["job"]["job_id"] == job1_id
            assert scheduler.is_gpu_busy(0) is True

            # Step 2b: Worker 2 attempts to lease a job on GPU 0 while Worker 1 is running
            lease2_res = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-gpu0-02",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert lease2_res.status_code == 200
            lease2 = lease2_res.json()
            # Concurrency bound strictly enforced: Worker 2 receives job=None
            assert lease2["job"] is None

            # Step 3: Verify discrete GPU 1 independence
            # Worker 3 can lease GPU 1 job without being blocked by GPU 0
            lease3_res = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-gpu1-01",
                "subsystem": "video",
                "supported_models": ["wan-2.2"],
                "supported_tasks": ["text_to_video"],
                "gpu_device": 1,
            })
            assert lease3_res.status_code == 200
            lease3 = lease3_res.json()
            assert lease3["job"] is not None
            assert lease3["job"]["job_id"] == job3_id
            # Both GPUs are now busy independently
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is True

            # Step 4: Worker 1 completes Job 1
            comp1_res = await client.post(f"/v1/jobs/{job1_id}/complete", json={
                "worker_id": "worker-gpu0-01",
                "inference_duration_ms": 1200.0,
                "result": {"output": "image_done.png"},
            })
            assert comp1_res.status_code == 200
            # GPU 0 is freed, while GPU 1 remains busy
            assert scheduler.is_gpu_busy(0) is False
            assert scheduler.is_gpu_busy(1) is True

            # Step 4b: Worker 2 can now acquire the next GPU 0 job
            lease2_retry_res = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-gpu0-02",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert lease2_retry_res.status_code == 200
            lease2_retry = lease2_retry_res.json()
            assert lease2_retry["job"] is not None
            assert lease2_retry["job"]["job_id"] == job2_id
            assert scheduler.is_gpu_busy(0) is True

            # Clean up: complete remaining jobs
            comp3_res = await client.post(f"/v1/jobs/{job3_id}/complete", json={
                "worker_id": "worker-gpu1-01",
                "inference_duration_ms": 3000.0,
                "result": {"output": "video_done.mp4"},
            })
            assert comp3_res.status_code == 200
            assert scheduler.is_gpu_busy(1) is False

            comp2_res = await client.post(f"/v1/jobs/{job2_id}/complete", json={
                "worker_id": "worker-gpu0-02",
                "inference_duration_ms": 1100.0,
                "result": {"output": "image2_done.png"},
            })
            assert comp2_res.status_code == 200
            assert scheduler.is_gpu_busy(0) is False


@pytest.mark.asyncio
async def test_high_contention_burst_50_workers_mutual_exclusion():
    """
    Stress-test with 50 concurrent workers competing for 30 jobs across GPU 0 and GPU 1.
    Verifies:
    1. Zero double-leasing under high concurrency.
    2. Concurrency bound per GPU (at no instant is more than 1 job active per GPU).
    3. Complete throughput without deadlocks.
    """
    async with lifespan(control_plane_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers=AUTH_HEADERS,
            timeout=60.0,
        ) as client:
            scheduler = control_plane_app.state.scheduler

            # Submit 15 jobs for GPU 0 and 15 jobs for GPU 1
            submitted_jobs_gpu0 = []
            submitted_jobs_gpu1 = []
            for i in range(15):
                r0 = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": f"Stress Image {i}"},
                    "priority": 10 + i,
                })
                assert r0.status_code == 201
                submitted_jobs_gpu0.append(r0.json()["job_id"])

                r1 = await client.post("/v1/jobs", json={
                    "service": "video",
                    "model": "wan-2.2",
                    "task": "text_to_video",
                    "inputs": {"prompt": f"Stress Video {i}"},
                    "priority": 10 + i,
                })
                assert r1.status_code == 201
                submitted_jobs_gpu1.append(r1.json()["job_id"])

            leased_jobs = set()
            active_jobs_per_gpu = {0: 0, 1: 0}
            lock = asyncio.Lock()
            max_observed_concurrency = {0: 0, 1: 0}

            async def stress_worker(worker_id: str, gpu_dev: int, service: str, model: str, task: str):
                completed = 0
                while completed < 3:
                    res = await client.post("/v1/jobs/lease", json={
                        "worker_id": worker_id,
                        "subsystem": service,
                        "supported_models": [model],
                        "supported_tasks": [task],
                        "gpu_device": gpu_dev,
                        "lease_duration_seconds": 30,
                    })
                    assert res.status_code == 200
                    data = res.json()
                    job = data.get("job")
                    if not job:
                        await asyncio.sleep(0.005)
                        # Check if all submitted jobs for this GPU have been processed
                        async with lock:
                            target_set = set(submitted_jobs_gpu0 if gpu_dev == 0 else submitted_jobs_gpu1)
                            if target_set.issubset(leased_jobs):
                                break
                        continue

                    jid = job["job_id"]
                    async with lock:
                        assert jid not in leased_jobs, f"Double lease detected on {jid} by {worker_id}!"
                        leased_jobs.add(jid)
                        active_jobs_per_gpu[gpu_dev] += 1
                        if active_jobs_per_gpu[gpu_dev] > max_observed_concurrency[gpu_dev]:
                            max_observed_concurrency[gpu_dev] = active_jobs_per_gpu[gpu_dev]
                        # Crucial invariant: Concurrency must NEVER exceed 1 per GPU
                        assert active_jobs_per_gpu[gpu_dev] <= 1, f"Concurrency violation on GPU {gpu_dev}!"

                    # Simulate brief GPU execution
                    await asyncio.sleep(0.01)

                    # Complete job
                    c_res = await client.post(f"/v1/jobs/{jid}/complete", json={
                        "worker_id": worker_id,
                        "inference_duration_ms": 50.0,
                        "result": {"status": "ok"},
                    })
                    assert c_res.status_code == 200

                    async with lock:
                        active_jobs_per_gpu[gpu_dev] -= 1
                    completed += 1

            # Launch 25 workers for GPU 0 and 25 workers for GPU 1 concurrently
            workers = []
            for i in range(25):
                workers.append(stress_worker(f"w-gpu0-{i}", 0, "image", "qwen-image-2.1", "text_to_image"))
                workers.append(stress_worker(f"w-gpu1-{i}", 1, "video", "wan-2.2", "text_to_video"))

            await asyncio.gather(*workers)

            # Assert all 30 jobs were leased and completed
            assert len(leased_jobs) == 30
            assert max_observed_concurrency[0] == 1, "GPU 0 concurrency exceeded 1!"
            assert max_observed_concurrency[1] == 1, "GPU 1 concurrency exceeded 1!"
            assert scheduler.is_gpu_busy(0) is False
            assert scheduler.is_gpu_busy(1) is False


@pytest.mark.asyncio
async def test_gpu_partition_release_on_cancellation():
    """
    Verify that cancelling a running job releases the GPU partition immediately
    and allows subsequent jobs to be leased without blockage.
    """
    async with lifespan(control_plane_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers=AUTH_HEADERS,
            timeout=30.0,
        ) as client:
            scheduler = control_plane_app.state.scheduler

            # 1. Submit two jobs to GPU 0
            s1 = await client.post("/v1/jobs", json={
                "service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Cancel test 1"}
            })
            j1 = s1.json()["job_id"]
            s2 = await client.post("/v1/jobs", json={
                "service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Cancel test 2"}
            })
            j2 = s2.json()["job_id"]

            # 2. Worker 1 leases job 1
            l1 = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-cancel-1",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert l1.json()["job"]["job_id"] == j1
            assert scheduler.is_gpu_busy(0) is True

            # 3. Cancel job 1 via DELETE /v1/jobs/{j1}
            del_res = await client.delete(f"/v1/jobs/{j1}")
            assert del_res.status_code == 200
            assert scheduler.is_gpu_busy(0) is False

            # 4. Worker 2 can now immediately lease job 2
            l2 = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-cancel-2",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert l2.json()["job"]["job_id"] == j2
            assert scheduler.is_gpu_busy(0) is True

            # Clean up
            await client.post(f"/v1/jobs/{j2}/complete", json={
                "worker_id": "worker-cancel-2",
                "inference_duration_ms": 100.0,
                "result": {"status": "ok"},
            })
            assert scheduler.is_gpu_busy(0) is False


@pytest.mark.asyncio
async def test_gpu_partition_release_on_failure():
    """
    Verify that failing a running job releases the GPU partition immediately.
    """
    async with lifespan(control_plane_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers=AUTH_HEADERS,
            timeout=30.0,
        ) as client:
            scheduler = control_plane_app.state.scheduler

            s1 = await client.post("/v1/jobs", json={
                "service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Fail test"}
            })
            j1 = s1.json()["job_id"]

            l1 = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-fail-1",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert l1.json()["job"]["job_id"] == j1
            assert scheduler.is_gpu_busy(0) is True

            # Fail job
            fail_res = await client.post(f"/v1/jobs/{j1}/fail", json={
                "worker_id": "worker-fail-1",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Simulated CUDA error",
                    "request_id": "req-fail-test",
                },
                "retryable": False,
            })
            assert fail_res.status_code == 200
            assert scheduler.is_gpu_busy(0) is False


@pytest.mark.asyncio
async def test_watchdog_disconnect_releases_gpu():
    """
    Verify watchdog marks disconnected worker as FAILED with WORKER_DISCONNECTED
    and immediately releases the GPU partition.
    """
    async with lifespan(control_plane_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers=AUTH_HEADERS,
            timeout=30.0,
        ) as client:
            scheduler = control_plane_app.state.scheduler
            repo = control_plane_app.state.repository
            manager = control_plane_app.state.job_manager

            sub = await client.post("/v1/jobs", json={
                "service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "WD test"}
            })
            jid = sub.json()["job_id"]

            l = await client.post("/v1/jobs/lease", json={
                "worker_id": "worker-crashed-t4",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
            })
            assert l.json()["job"]["job_id"] == jid
            assert scheduler.is_gpu_busy(0) is True

            # Simulate worker disconnection: backdate heartbeat to 70s ago
            stale_iso = (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat()
            async with repo._lock:
                repo._workers["worker-crashed-t4"]["last_heartbeat"] = stale_iso

            # Run watchdog
            failed_count = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
            )
            assert failed_count == 1
            assert scheduler.is_gpu_busy(0) is False

            # Verify terminal state
            job = await repo.get_job(jid)
            assert job.state == JobState.FAILED
            assert job.error.code == "WORKER_DISCONNECTED"


def test_vram_pre_admission_boundary():
    """
    Verify boundary conditions for VRAM pre-admission guard:
    - min_vram_gb <= 14.2 is accepted
    - min_vram_gb > 14.2 is rejected with HTTP 400
    """
    from factory.apps.control_api.src.api.routers.jobs import _validate_job_capability
    from unittest.mock import MagicMock
    from factory_contracts.jobs.job_contract import JobSubmissionRequest

    def make_manifest(vram_gb: float):
        return ModelManifest(
            apiVersion="factory.ai/v1",
            kind="ModelManifest",
            metadata=ModelMetadata(
                id=f"test-model-{vram_gb}gb",
                family="test-family",
                display_name=f"Test Model {vram_gb}GB",
                service_id="text-service",
                version="1.0.0",
            ),
            weights=ModelWeightsSpec(
                primary_checkpoint="meta/test",
                format=ModelWeightFormat.SAFETENSORS,
                quantization=QuantizationType.AWQ,
                default_precision=PrecisionType.FP16,
            ),
            hardware=ModelHardwareSpec(
                min_vram_gb=vram_gb,
                context_window=2048,
                preferred_gpu="t4",
                allocation_policy=AllocationPolicy.SINGLE_GPU,
            ),
            capabilities={"chat_completion": True},
            status=ModelStatus.FULL,
            coverage_percent=100.0,
            missing_capabilities=[],
            validation_status=ModelValidationStatus.PENDING_LIVE,
            validation_notes="Boundary test",
            limits=ModelLimitsSpec(),
        )

    # 1. Exactly 14.2 GB should pass validation without error
    manifest_14_2 = make_manifest(14.2)
    mock_request = MagicMock()
    mock_loader = MagicMock(spec=ManifestLoader)
    mock_loader.load_models.return_value = {"test-model-14.2gb": manifest_14_2}
    mock_request.app.state.manifest_loader = mock_loader

    req_pass = JobSubmissionRequest(
        service="text",
        model="test-model-14.2gb",
        task="chat_completion",
        inputs={"prompt": "test"},
    )
    # Should not raise
    _validate_job_capability(req_pass, mock_request)

    # 2. 14.3 GB should raise InvalidRequestError
    manifest_14_3 = make_manifest(14.3)
    mock_loader.load_models.return_value = {"test-model-14.3gb": manifest_14_3}
    req_fail = JobSubmissionRequest(
        service="text",
        model="test-model-14.3gb",
        task="chat_completion",
        inputs={"prompt": "test"},
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        _validate_job_capability(req_fail, mock_request)
    assert exc_info.value.status_code == 400
    assert "Requested model exceeds max allocatable VRAM of 14.2 GB" in str(exc_info.value.message)
