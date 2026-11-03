"""
tests/unit/queue/test_worker_lease_auth.py
Unit tests verifying worker lease ownership, cryptographic token verification,
HTTP 403 UnauthorizedError enforcement, and stale registry clearing.
"""
from datetime import datetime, timezone
import pytest

from factory_contracts.errors import ErrorDetail, UnauthorizedError
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
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
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository import InMemoryJobStore as SqliteJobRepository
from factory.apps.control_api.src.streaming.manager import StreamManager


@pytest.mark.asyncio
class TestWorkerLeaseAuth:
    async def _setup(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)
        return repo, broker, stream_mgr, manager, coordinator

    async def test_authorized_worker_can_progress_and_complete(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            sub_req = JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Authorized worker workflow"},
            )
            job, _ = await manager.submit_job(sub_req, request_id="req-test-auth")

            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-auth-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            assert lease_resp.job is not None
            assert lease_resp.lease_token is not None
            token = lease_resp.lease_token

            # 1. Progress update with matching worker_id and lease_token
            prog_res = await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="sampling",
                    step=5,
                    total=20,
                    percent=25.0,
                    message="Sampling 5/20",
                    worker_id="worker-auth-01",
                    lease_token=token,
                ),
            )
            assert prog_res.progress.step == 5
            assert prog_res.progress.percent == 25.0

            # 2. Complete job with matching worker_id and lease_token
            comp_res = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    artifacts=[
                        ArtifactReference(
                            artifact_id="art-auth-01",
                            type="image",
                            mime_type="image/png",
                            size_bytes=4096,
                            storage_path=f"image/{job.job_id}/out.png",
                            url="https://s3.ai.example.com/art.png",
                            download_url="https://s3.ai.example.com/art.png",
                            sha256="a" * 64,
                        )
                    ],
                    inference_duration_ms=1200.0,
                    worker_id="worker-auth-01",
                    lease_token=token,
                ),
            )
            assert comp_res.state == JobState.SUCCEEDED
            assert comp_res.worker is None
            assert len(comp_res.artifacts) == 1
        finally:
            await repo.close()

    async def test_authorized_worker_can_fail_job(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Failing job"},
                ),
                request_id="req-test-auth-fail",
            )

            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-auth-02",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease_resp.lease_token

            fail_res = await manager.fail_job(
                job.job_id,
                JobFailureRequest(
                    error=ErrorDetail(code="INTERNAL_ERROR", message="CUDA out of memory", request_id=job.request_id),
                    retryable=False,
                    worker_id="worker-auth-02",
                    lease_token=token,
                ),
            )
            assert fail_res.state == JobState.FAILED
            assert fail_res.worker is None
        finally:
            await repo.close()

    async def test_unauthorized_worker_rejected_on_progress(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Progress unauthorized check"},
                ),
                request_id="req-test-auth-prog",
            )

            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-legit",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease_resp.lease_token

            # 1. Mismatched worker_id
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=1,
                        total=10,
                        percent=10.0,
                        worker_id="worker-imposter",
                        lease_token=token,
                    ),
                )
            assert exc_info.value.status_code == 403
            assert exc_info.value.code == "UNAUTHORIZED"

            # 2. Mismatched lease_token
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=1,
                        total=10,
                        percent=10.0,
                        worker_id="worker-legit",
                        lease_token="forged-token",
                    ),
                )
            assert exc_info.value.status_code == 403

            # 3. Missing tokens
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=1,
                        total=10,
                        percent=10.0,
                    ),
                )
            assert exc_info.value.status_code == 403
        finally:
            await repo.close()

    async def test_unauthorized_worker_rejected_on_complete(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Complete unauthorized check"},
                ),
                request_id="req-test-auth-comp",
            )

            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-legit-2",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease_resp.lease_token

            artifact = ArtifactReference(
                artifact_id="art-test",
                type="image",
                mime_type="image/png",
                size_bytes=100,
                storage_path=f"image/{job.job_id}/out.png",
                url="https://s3.ai.example.com/art.png",
                download_url="https://s3.ai.example.com/art.png",
                sha256="b" * 64,
            )

            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="worker-hacker",
                        lease_token=token,
                    ),
                )
            assert exc_info.value.status_code == 403

            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="worker-legit-2",
                        lease_token="invalid-token",
                    ),
                )
            assert exc_info.value.status_code == 403
        finally:
            await repo.close()

    async def test_unauthorized_worker_rejected_on_fail(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Fail unauthorized check"},
                ),
                request_id="req-test-auth-failcheck",
            )

            await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-legit-3",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.fail_job(
                    job.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(code="INTERNAL_ERROR", message="Attacker trying to fail job", request_id="req-1"),
                        worker_id="worker-other",
                        lease_token="wrong-token",
                    ),
                )
            assert exc_info.value.status_code == 403
        finally:
            await repo.close()

    async def test_heartbeat_lease_extension_requires_valid_token(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Heartbeat token check"},
                ),
                request_id="req-test-auth-hb",
            )

            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-hb-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            assert lease_resp.lease_token is not None

            # 1. Heartbeat with wrong lease_token does not extend lease
            hb_fail = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-hb-01",
                    active_job_id=job.job_id,
                    lease_token="invalid-token-1234",
                    gpu_utilization_percent=50.0,
                    vram_used_mb=8000.0,
                )
            )
            assert hb_fail.lease_extended_until is None

            # 2. Heartbeat with missing lease_token does not extend lease
            hb_none = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-hb-01",
                    active_job_id=job.job_id,
                    lease_token=None,
                )
            )
            assert hb_none.lease_extended_until is None

            # 3. Heartbeat with correct token extends lease
            hb_ok = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-hb-01",
                    active_job_id=job.job_id,
                    lease_token=lease_resp.lease_token,
                    gpu_utilization_percent=50.0,
                    vram_used_mb=8000.0,
                )
            )
            assert hb_ok.lease_extended_until is not None
        finally:
            await repo.close()

    async def test_stale_job_id_cleared_on_idle_heartbeat(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            worker_id = "worker-idle-test-01"

            # Heartbeat while busy with job-abc
            await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id=worker_id,
                    active_job_id="job-abc",
                    gpu_utilization_percent=75.0,
                    vram_used_mb=9000.0,
                )
            )
            reg_busy = await repo.get_worker(worker_id)
            assert reg_busy is not None
            assert reg_busy["current_job_id"] == "job-abc"

            # Worker finishes job and sends heartbeat with active_job_id=None
            await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id=worker_id,
                    active_job_id=None,
                    gpu_utilization_percent=5.0,
                    vram_used_mb=1000.0,
                )
            )
            reg_idle = await repo.get_worker(worker_id)
            assert reg_idle is not None
            assert reg_idle["current_job_id"] is None
        finally:
            await repo.close()

    async def test_unleased_job_backward_compatibility(self):
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Unleased job test"},
                ),
                request_id="req-test-auth-unleased",
            )
            assert job.worker is None

            # Unleased modification without tokens is permitted (for backward compatibility)
            updated = await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="sampling",
                    step=1,
                    total=10,
                    percent=10.0,
                ),
            )
            assert updated.progress.percent == 10.0

            # But claiming a worker_id on an unleased job is rejected
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=2,
                        total=10,
                        percent=20.0,
                        worker_id="fake-worker",
                    ),
                )
            assert exc_info.value.status_code == 403

            # And calling with caller_worker_id on unleased job is rejected even if worker_id is omitted
            with pytest.raises(UnauthorizedError) as exc_caller:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=3,
                        total=10,
                        percent=30.0,
                    ),
                    caller_worker_id="authenticated-worker-id",
                )
            assert exc_caller.value.status_code == 403
        finally:
            await repo.close()

    async def test_caller_worker_id_mismatch_rejected(self):
        """Caller worker identity from token claims must match worker_id in request."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Caller binding test"},
                ),
                request_id="req-caller-bind",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-real",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease.lease_token

            # Caller token claim is 'worker-spoof', but request body claims 'worker-real'
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=1,
                        total=10,
                        percent=10.0,
                        worker_id="worker-real",
                        lease_token=token,
                    ),
                    caller_worker_id="worker-spoof",
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403

            # Admin caller can manage any worker
            prog = await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="sampling",
                    step=1,
                    total=10,
                    percent=10.0,
                    worker_id="worker-real",
                    lease_token=token,
                ),
                caller_worker_id="admin-user",
                caller_role="admin",
            )
            assert prog.progress.percent == 10.0
        finally:
            await repo.close()

    async def test_expired_lease_blocks_progress_and_completion(self):
        """Worker with an expired lease cannot mutate job state."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            from datetime import timedelta
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Expiry test"},
                ),
                request_id="req-expiry-test",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-expiry",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    lease_duration_seconds=300,
                )
            )
            token = lease.lease_token

            # Manually expire the lease in the repository
            past = datetime.now(timezone.utc) - timedelta(seconds=10)
            await repo.extend_lease(worker_id="worker-expiry", job_id=job.job_id, new_expires_at=past)

            # Progress must be rejected due to expiration
            with pytest.raises(UnauthorizedError) as exc_p:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=1,
                        total=10,
                        percent=10.0,
                        worker_id="worker-expiry",
                        lease_token=token,
                    ),
                )
            assert exc_p.value.status_code == 403
            assert "expired" in exc_p.value.message.lower()

            # Completion must also be rejected due to expiration
            artifact = ArtifactReference(
                artifact_id="art-expired",
                type="image",
                mime_type="image/png",
                size_bytes=100,
                storage_path=f"image/{job.job_id}/out.png",
                url="https://s3.ai.example.com/art.png",
                download_url="https://s3.ai.example.com/art.png",
                sha256="d" * 64,
            )
            with pytest.raises(UnauthorizedError) as exc_c:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="worker-expiry",
                        lease_token=token,
                    ),
                )
            assert exc_c.value.status_code == 403
            assert "expired" in exc_c.value.message.lower()
        finally:
            await repo.close()

    async def test_worker_capability_matching_enforces_registered_manifests(self):
        """Worker cannot lease jobs outside its verified registered manifest capabilities."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            # 1. Submit a video job
            job_video, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="video",
                    model="wan-2.2",
                    task="text_to_video",
                    inputs={"prompt": "Video generation"},
                ),
                request_id="req-video-cap",
            )

            # 2. An image worker attempts to claim it supports wan-2.2
            rogue_lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="image-worker-rogue",
                    subsystem="image",
                    supported_models=["qwen-image-2.1", "wan-2.2"],  # wan-2.2 is not in image manifest!
                    supported_tasks=["text_to_image", "text_to_video"],
                )
            )
            # Must NOT lease the video job to the image worker
            assert rogue_lease.job is None

            # 3. Legitimate video worker can lease it
            legit_video_lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="video-worker-legit",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                )
            )
            assert legit_video_lease.job is not None
            assert legit_video_lease.job.job_id == job_video.job_id
        finally:
            await repo.close()
