"""
tests/unit/queue/test_gpu_concurrency.py
Unit tests verifying Milestone M3 (GPU Runtime Simplification):
1. Strict GPU Concurrency Bound (max 1 concurrent job per discrete GPU).
2. Discrete Dual T4 GPU 0 vs GPU 1 Independence.
3. VRAM Pre-Admission Rejection for models > 14.2 GB (HTTP 400 INVALID_REQUEST).
4. Deterministic Watchdog Disconnect Failure (no duplicate re-queueing to QUEUED, instant GPU release).
5. Worker HTTP Status Reporting without mandatory lease tokens.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from starlette.testclient import TestClient

from factory_contracts.errors import InvalidRequestError, UnauthorizedError
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
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
from factory.apps.control_api.src.queue.broker import get_queue_broker
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository import InMemoryJobStore
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.legacy import app as control_api_app


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-admin-key"}


@pytest.mark.asyncio
class TestGPUConcurrency:
    async def _setup(self):
        repo = InMemoryJobStore(db_path=":memory:")
        await repo.initialize()
        scheduler = InMemoryScheduler()
        await scheduler.initialize()
        stream_mgr = StreamManager(repository=repo)
        manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=scheduler)
        return repo, scheduler, stream_mgr, manager, coordinator

    async def test_concurrency_bound_one_per_gpu(self):
        """
        Enforce max_concurrent_services: 1 per GPU ID.
        While worker 1 is running on GPU 0, worker 2 receives job=None.
        When worker 1 completes, GPU 0 is released and worker 2 receives the second job.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            # 1. Submit 2 jobs targeted for GPU 0 (image service)
            job1, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "First image on GPU 0"},
                    priority=20,
                ),
                request_id="req-gpu0-job1",
            )
            job2, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Second image on GPU 0"},
                    priority=10,
                ),
                request_id="req-gpu0-job2",
            )

            # 2. Worker 1 leases a job on GPU 0
            lease1 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu0-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease1.job is not None
            assert lease1.job.job_id == job1.job_id
            assert scheduler.is_gpu_busy(0) is True

            # 3. Worker 2 attempts to lease a job on GPU 0 while Worker 1 is running
            lease2 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu0-02",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            # Concurrency bound = 1 must strictly hold: worker 2 receives None
            assert lease2.job is None

            # 4. Worker 1 completes the job
            await manager.complete_job(
                job1.job_id,
                JobCompletionRequest(
                    worker_id="worker-gpu0-01",
                    inference_duration_ms=1500.0,
                    result={"status": "done"},
                ),
                caller_role="admin",
            )
            # GPU 0 is now released
            assert scheduler.is_gpu_busy(0) is False

            # 5. Worker 2 can now lease Job 2
            lease2_retry = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu0-02",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease2_retry.job is not None
            assert lease2_retry.job.job_id == job2.job_id
            assert scheduler.is_gpu_busy(0) is True
        finally:
            await repo.close()
            await scheduler.close()

    async def test_discrete_gpu0_vs_gpu1_independence(self):
        """
        Verify GPU 0 and GPU 1 partitions are completely independent.
        A busy GPU 0 does not block GPU 1 job dispatch.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            # 1. Submit Job 1 on GPU 0 (image) and Job 2 on GPU 1 (video)
            job_gpu0, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Image on GPU 0"},
                ),
                request_id="req-indep-gpu0",
            )
            job_gpu1, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="video",
                    model="wan-2.2",
                    task="text_to_video",
                    inputs={"prompt": "Video on GPU 1"},
                ),
                request_id="req-indep-gpu1",
            )

            # 2. Lease GPU 0 job
            lease_gpu0 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu0",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease_gpu0.job is not None
            assert lease_gpu0.job.job_id == job_gpu0.job_id
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is False

            # 3. Lease GPU 1 job while GPU 0 is busy
            lease_gpu1 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu1",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                    gpu_device=1,
                )
            )
            assert lease_gpu1.job is not None
            assert lease_gpu1.job.job_id == job_gpu1.job_id
            # Both discrete GPUs now actively running independent workloads
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is True

            # 4. Complete GPU 1 job: GPU 1 releases while GPU 0 remains busy
            await manager.complete_job(
                job_gpu1.job_id,
                JobCompletionRequest(
                    worker_id="worker-gpu1",
                    inference_duration_ms=4500.0,
                    result={"video_url": "test.mp4"},
                ),
                caller_role="admin",
            )
            assert scheduler.is_gpu_busy(1) is False
            assert scheduler.is_gpu_busy(0) is True

            # 5. Complete GPU 0 job: GPU 0 releases
            await manager.complete_job(
                job_gpu0.job_id,
                JobCompletionRequest(
                    worker_id="worker-gpu0",
                    inference_duration_ms=1200.0,
                    result={"image_url": "test.png"},
                ),
                caller_role="admin",
            )
            assert scheduler.is_gpu_busy(0) is False
            assert scheduler.is_gpu_busy(1) is False
        finally:
            await repo.close()
            await scheduler.close()

    async def test_deterministic_disconnect_failure_no_requeue(self):
        """
        Verify watchdog marks disconnected workers (> 60s heartbeat) as FAILED
        with WORKER_DISCONNECTED (HTTP 502 semantics), releases the GPU partition,
        and never re-enqueues the job to QUEUED.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Watchdog test image"},
                ),
                request_id="req-watchdog-test",
            )

            # Worker leases job
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-crash-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease.job is not None
            assert scheduler.is_gpu_busy(0) is True

            # Simulate worker heartbeat 65 seconds in the past
            stale_time = (datetime.now(timezone.utc) - timedelta(seconds=65)).isoformat()
            await repo.upsert_worker_heartbeat(
                worker_id="worker-crash-01",
                active_job_id=job.job_id,
                subsystem="image",
                gpu_device=0,
            )
            # Manually backdate heartbeat in repo
            async with repo._lock:
                repo._workers["worker-crash-01"]["last_heartbeat"] = stale_time

            # Run watchdog check
            failed_count = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
            )
            assert failed_count == 1

            # Verify job is FAILED with code WORKER_DISCONNECTED
            terminal_job = await repo.get_job(job.job_id)
            assert terminal_job.state == JobState.FAILED
            assert terminal_job.error is not None
            assert terminal_job.error.code == "WORKER_DISCONNECTED"
            assert terminal_job.state != JobState.QUEUED

            # Verify GPU partition was released immediately
            assert scheduler.is_gpu_busy(0) is False

            # Verify subsequent worker can immediately lease on GPU 0
            job_next, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Next image"},
                ),
                request_id="req-watchdog-next",
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
        finally:
            await repo.close()
            await scheduler.close()

    async def test_worker_http_reporting_without_lease_token(self):
        """
        Verify worker reports without lease_token are rejected with HTTP 403 under P0.6,
        while admin callers are permitted.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "No token workflow"},
                ),
                request_id="req-notoken",
            )

            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-notoken-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease.job is not None

            # 1. Non-admin progress without lease_token must be rejected with HTTP 403
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=10,
                        total=20,
                        percent=50.0,
                        worker_id="worker-notoken-01",
                        lease_token=None,  # Explicitly omitted
                    ),
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403

            # 2. Non-admin complete without lease_token must be rejected with HTTP 403
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        worker_id="worker-notoken-01",
                        lease_token=None,  # Explicitly omitted
                        inference_duration_ms=2100.0,
                        result={"image": "success"},
                    ),
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403

            # 3. Admin override is permitted
            admin_progress = await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="sampling",
                    step=10,
                    total=20,
                    percent=50.0,
                    worker_id="worker-notoken-01",
                    lease_token=None,
                ),
                caller_role="admin",
            )
            assert admin_progress.progress.percent == 50.0

            admin_complete = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    worker_id="worker-notoken-01",
                    lease_token=None,
                    inference_duration_ms=2100.0,
                    result={"image": "success"},
                ),
                caller_role="admin",
            )
            assert admin_complete.state == JobState.SUCCEEDED
            assert scheduler.is_gpu_busy(0) is False
        finally:
            await repo.close()
            await scheduler.close()

    async def test_watchdog_handles_none_leased_at_gracefully(self):
        """
        Verify watchdog safely falls back to job.created_at when worker.leased_at is None.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Timestamp fallback test"},
                ),
                request_id="req-ts-fallback",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-none-leased",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease.job is not None

            # Manually simulate leased_at=None on worker in repo
            async with repo._lock:
                repo._jobs[job.job_id].worker.leased_at = None
                repo._jobs[job.job_id].started_at = None

            # Should not raise AttributeError: 'NoneType' object has no attribute 'tzinfo'
            failed_count = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
            )
            assert isinstance(failed_count, int)
        finally:
            await repo.close()
            await scheduler.close()

    async def test_lease_job_releases_gpu_on_repository_exception(self):
        """
        Verify lease_job releases the GPU partition if an exception is raised during lease acquisition.
        """
        repo, scheduler, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Exception safety test"},
                ),
                request_id="req-exc-safety",
            )

            # Patch acquire_lease to raise an unexpected RuntimeError
            with patch.object(repo, "acquire_lease", side_effect=RuntimeError("Simulated failure")):
                with pytest.raises(RuntimeError, match="Simulated failure"):
                    await coordinator.lease_job(
                        WorkerLeaseRequest(
                            worker_id="worker-exc-01",
                            subsystem="image",
                            supported_models=["qwen-image-2.1"],
                            supported_tasks=["text_to_image"],
                            gpu_device=0,
                        )
                    )

            # GPU 0 must NOT remain leaked/locked
            assert scheduler.is_gpu_busy(0) is False
        finally:
            await repo.close()
            await scheduler.close()


def test_vram_pre_admission_rejection_over_14_2_gb():
    """
    Verify submission of a model requiring > 14.2 GB VRAM is rejected at pre-admission
    with HTTP 400 and message 'Requested model exceeds max allocatable VRAM of 14.2 GB'.
    """
    from factory.apps.control_api.src.api.routers.jobs import _validate_job_capability
    from model_registry.loader import ManifestLoader

    # Create dummy model requiring 16.0 GB VRAM
    heavy_model = ModelManifest(
        apiVersion="factory.ai/v1",
        kind="ModelManifest",
        metadata=ModelMetadata(
            id="heavy-model-32b",
            family="heavy-llm",
            display_name="Heavy Model 32B",
            service_id="text-service",
            version="1.0.0",
        ),
        weights=ModelWeightsSpec(
            primary_checkpoint="meta/heavy-32b",
            format=ModelWeightFormat.SAFETENSORS,
            quantization=QuantizationType.AWQ,
            default_precision=PrecisionType.FP16,
        ),
        hardware=ModelHardwareSpec(
            min_vram_gb=16.0,  # Exceeds 14.2 GB limit!
            context_window=4096,
            preferred_gpu="t4",
            allocation_policy=AllocationPolicy.SINGLE_GPU,
        ),
        capabilities={"chat_completion": True},
        status=ModelStatus.FULL,
        coverage_percent=100.0,
        missing_capabilities=[],
        validation_status=ModelValidationStatus.PENDING_LIVE,
        validation_notes="Heavy model test.",
        limits=ModelLimitsSpec(),
    )

    req = JobSubmissionRequest(
        service="text",
        model="heavy-model-32b",
        task="chat_completion",
        inputs={"prompt": "Hello"},
    )

    mock_request = MagicMock()
    mock_loader = MagicMock(spec=ManifestLoader)
    mock_loader.load_models.return_value = {"heavy-model-32b": heavy_model}
    mock_request.app.state.manifest_loader = mock_loader

    with pytest.raises(InvalidRequestError) as exc_info:
        _validate_job_capability(req, mock_request)

    assert exc_info.value.status_code == 400
    assert "Requested model exceeds max allocatable VRAM of 14.2 GB" in str(exc_info.value.message)
