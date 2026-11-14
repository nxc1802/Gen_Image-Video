"""
tests/unit/queue/test_terminal_succeeded_guard.py
Unit tests verifying terminal state guards in JobLifecycleManager (Requirement R2, P2.3):
- No job enters terminal SUCCEEDED state prior to output schema validation and artifact persistence
- Missing/empty artifact on media job causes direct transition to FAILED (never touches SUCCEEDED)
- Non-media jobs fail with INVALID_REQUEST if req.result is missing or empty
- Multi-GPU releases [0, 1] on timeout or worker disconnect
"""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]

from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobProgress,
    JobState,
    JobTelemetry,
    ArtifactReference,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
)
from factory_contracts.errors import ErrorCode
from factory.apps.control_api.src.queue.broker import InMemoryScheduler
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskConfig, ScratchDiskManager


class TestTerminalSucceededGuard:
    async def _setup_queue(self, tmp_path):
        repo = InMemoryJobStore()
        await repo.initialize()
        broker = InMemoryScheduler()
        await broker.initialize()
        stream_mgr = StreamManager(repository=repo)
        scratch = ScratchDiskManager(ScratchDiskConfig(root_dir=str(tmp_path / "scratch")))
        await scratch.initialize()

        manager = JobLifecycleManager(
            repository=repo,
            broker=broker,
            stream_manager=stream_mgr,
        )
        return repo, broker, manager, scratch

    @pytest.mark.asyncio
    async def test_job_never_transitions_to_succeeded_on_artifact_failure(self, tmp_path):
        repo, broker, manager, scratch = await self._setup_queue(tmp_path)

        job_id = "job-11111111-2222-3333-4444-555555555555"
        lease_token = "lease-token-valid-12345"
        worker_id = "worker-01"

        # Register worker & lease
        job = FactoryJob(
            job_id=job_id,
            request_id="req-11111111-2222-3333-4444-555555555555",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test prompt"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="generating", percent=50.0),
            worker=WorkerLease(
                worker_id=worker_id,
                gpu_device=0,
                lease_token=lease_token,
            ),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        # Attempt to complete job pointing to a non-existent artifact path
        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifact_path="/tmp/non_existent_artifact_path.png",
            inference_duration_ms=1200.0,
        )

        completed_job = await manager.complete_job(job_id, req, caller_worker_id=worker_id)

        # Assert job transitioned directly to FAILED and never became SUCCEEDED
        assert completed_job.state == JobState.FAILED
        assert completed_job.error is not None
        assert completed_job.error.code == ErrorCode.ARTIFACT_MISSING
        assert not completed_job.artifacts

        # Verify repository state
        persisted = await repo.get_job(job_id)
        assert persisted.state == JobState.FAILED

        # Verify GPU was released
        assert not broker.is_gpu_busy(0)

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_non_media_job_fails_when_result_payload_missing(self, tmp_path):
        repo, broker, manager, scratch = await self._setup_queue(tmp_path)

        job_id = "job-33333333-4444-5555-6666-777777777777"
        lease_token = "lease-token-valid-67890"
        worker_id = "worker-text-01"

        job = FactoryJob(
            job_id=job_id,
            request_id="req-33333333-4444-5555-6666-777777777777",
            service="text",
            model="qwen-3.8-27b",
            task="text_generation",
            inputs={"prompt": "Hello"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="generating", percent=50.0),
            worker=WorkerLease(
                worker_id=worker_id,
                gpu_device=0,
                lease_token=lease_token,
            ),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        # Complete non-media job with result=None / empty
        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            result=None,
            inference_duration_ms=450.0,
        )

        failed_job = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert failed_job.state == JobState.FAILED
        assert failed_job.error is not None
        assert failed_job.error.code == ErrorCode.INVALID_REQUEST
        assert "without result payload" in failed_job.error.message

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_job_succeeds_only_after_artifact_persistence_verified(self, tmp_path):
        repo, broker, manager, scratch = await self._setup_queue(tmp_path)

        job_id = "job-44444444-5555-6666-7777-888888888888"
        lease_token = "lease-token-valid-abcde"
        worker_id = "worker-image-01"

        # Write a real on-disk artifact
        payload = b"GENUINE_IMAGE_PIXELS_12345"
        art_path = scratch.get_artifact_path(job_id, "output.png")
        art_path.parent.mkdir(parents=True, exist_ok=True)
        art_path.write_bytes(payload)

        job = FactoryJob(
            job_id=job_id,
            request_id="req-44444444-5555-6666-7777-888888888888",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "mountains at sunset"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="generating", percent=80.0),
            worker=WorkerLease(
                worker_id=worker_id,
                gpu_device=0,
                lease_token=lease_token,
            ),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifact_path=str(art_path),
            inference_duration_ms=2100.0,
            fps=12.5,
        )

        succeeded_job = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert succeeded_job.state == JobState.SUCCEEDED
        assert succeeded_job.completed_at is not None
        assert len(succeeded_job.artifacts) == 1
        assert succeeded_job.artifacts[0].size_bytes == len(payload)
        assert len(succeeded_job.artifacts[0].sha256) == 64
        assert succeeded_job.telemetry.inference_duration_ms == 2100.0

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_watchdog_multi_gpu_release_on_timeout(self, tmp_path):
        repo, broker, manager, scratch = await self._setup_queue(tmp_path)

        job_id = "job-55555555-6666-7777-8888-999999999999"
        # Acquire dual GPUs [0, 1]
        broker.acquire_gpus([0, 1], job_id)
        assert broker.is_gpu_busy(0)
        assert broker.is_gpu_busy(1)

        # Stale job started 400 seconds ago
        past_time = datetime.now(timezone.utc).timestamp() - 400
        job = FactoryJob(
            job_id=job_id,
            request_id="req-55555555-6666-7777-8888-999999999999",
            service="text",
            model="qwen-3.8-27b",
            task="text_generation",
            inputs={"prompt": "long running text"},
            state=JobState.RUNNING,
            started_at=datetime.fromtimestamp(past_time, tz=timezone.utc),
            worker=WorkerLease(
                worker_id="worker-dual-gpu",
                gpu_device=0,
                lease_token="lease-dual-gpu",
                leased_at=datetime.fromtimestamp(past_time, tz=timezone.utc),
            ),
        )
        await repo.create_job(job)

        # Run watchdog check
        failed_count = await check_active_jobs_watchdog(
            repository=repo,
            scheduler=broker,
            job_manager=manager,
            heartbeat_timeout_seconds=60.0,
            default_job_timeout_seconds=300.0,
        )

        assert failed_count == 1
        # Verify BOTH GPU 0 and GPU 1 were released
        assert not broker.is_gpu_busy(0), "GPU 0 was not released!"
        assert not broker.is_gpu_busy(1), "GPU 1 was not released!"

        await scratch.shutdown()
        await broker.close()
        await repo.close()
