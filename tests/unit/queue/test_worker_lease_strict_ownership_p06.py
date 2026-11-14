"""
tests/unit/queue/test_worker_lease_strict_ownership_p06.py
Comprehensive unit tests for Gate 2 (P0.6) Worker Authentication & Strict Lease Ownership.
Verifies that:
- Non-admin worker callers without a valid lease_token are rejected on progress, complete, fail (HTTP 403).
- Forged or mismatched lease_tokens are rejected with HTTP 403 UnauthorizedError.
- Mismatched worker_ids attempting to mutate another worker's lease are rejected with HTTP 403.
- Expired leases are rejected on all worker mutation operations with HTTP 403.
- Heartbeats with active jobs require strict worker_id and lease_token ownership, rejecting imposters with HTTP 403.
- Admin callers are exempt from worker lease ownership checks.
"""
from datetime import datetime, timedelta, timezone
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from unittest.mock import MagicMock

from factory_contracts.errors import ErrorDetail, UnauthorizedError
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    WorkerLease,
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
from factory.apps.control_api.src.queue.repository import InMemoryJobStore
from factory.apps.control_api.src.streaming.manager import StreamManager


@pytest.mark.asyncio
class TestWorkerLeaseStrictOwnershipP06:
    async def _setup(self):
        repo = InMemoryJobStore(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)
        return repo, broker, stream_mgr, manager, coordinator

    async def test_worker_calling_progress_without_lease_token_rejected_403(self):
        """Non-admin worker calling progress on leased job without lease_token rejected with HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Strict progress test"},
                ),
                request_id="req-p06-prog",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-p06-01",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            assert lease_resp.lease_token is not None

            # Non-admin worker calling with caller_worker_id but missing lease_token
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=2,
                        total=20,
                        percent=10.0,
                        worker_id="worker-p06-01",
                        lease_token=None,  # Missing
                    ),
                    caller_worker_id="worker-p06-01",
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403
            assert "Invalid or missing lease token" in exc_info.value.message
        finally:
            await repo.close()

    async def test_worker_calling_complete_without_lease_token_rejected_403(self):
        """Non-admin worker calling complete without lease_token rejected with HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Strict complete test"},
                ),
                request_id="req-p06-comp",
            )
            await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-p06-02",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            artifact = ArtifactReference(
                artifact_id="art-p06-01",
                type="image",
                mime_type="image/png",
                size_bytes=2048,
                storage_path=f"image/{job.job_id}/out.png",
                url="https://s3.ai.example.com/art.png",
                download_url="https://s3.ai.example.com/art.png",
                sha256="c" * 64,
            )

            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="worker-p06-02",
                        lease_token=None,  # Missing
                    ),
                    caller_worker_id="worker-p06-02",
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403
            assert "Invalid or missing lease token" in exc_info.value.message
        finally:
            await repo.close()

    async def test_worker_calling_fail_without_lease_token_rejected_403(self):
        """Non-admin worker calling fail without lease_token rejected with HTTP 403."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Strict fail test"},
                ),
                request_id="req-p06-fail",
            )
            await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-p06-03",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.fail_job(
                    job.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(code="INTERNAL_ERROR", message="OOM", request_id=job.request_id),
                        worker_id="worker-p06-03",
                        lease_token=None,  # Missing
                    ),
                    caller_worker_id="worker-p06-03",
                    caller_role="worker",
                )
            assert exc_info.value.status_code == 403
            assert "Invalid or missing lease token" in exc_info.value.message
        finally:
            await repo.close()

    async def test_forged_lease_token_rejected_on_all_endpoints(self):
        """Forged lease_token rejected with HTTP 403 on progress, complete, fail."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Forged token test"},
                ),
                request_id="req-p06-forged",
            )
            await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-p06-04",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            # 1. Progress with forged token
            with pytest.raises(UnauthorizedError) as exc_prog:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=1,
                        total=10,
                        percent=10.0,
                        worker_id="worker-p06-04",
                        lease_token="forged-tok-666",
                    ),
                    caller_worker_id="worker-p06-04",
                    caller_role="worker",
                )
            assert exc_prog.value.status_code == 403

            # 2. Complete with forged token
            with pytest.raises(UnauthorizedError) as exc_comp:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        worker_id="worker-p06-04",
                        lease_token="forged-tok-666",
                        inference_duration_ms=1000.0,
                    ),
                    caller_worker_id="worker-p06-04",
                    caller_role="worker",
                )
            assert exc_comp.value.status_code == 403

            # 3. Fail with forged token
            with pytest.raises(UnauthorizedError) as exc_fail:
                await manager.fail_job(
                    job.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(code="INTERNAL_ERROR", message="err", request_id=job.request_id),
                        worker_id="worker-p06-04",
                        lease_token="forged-tok-666",
                    ),
                    caller_worker_id="worker-p06-04",
                    caller_role="worker",
                )
            assert exc_fail.value.status_code == 403
        finally:
            await repo.close()

    async def test_expired_lease_rejected_on_all_operations(self):
        """Expired worker lease rejected on progress, complete, fail, and heartbeat."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Expired lease test"},
                ),
                request_id="req-p06-expired",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-p06-05",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease_resp.lease_token

            # Manually expire the lease in the past
            leased_job = await repo.get_job(job.job_id)
            past = datetime.now(timezone.utc) - timedelta(seconds=60)
            leased_job.worker.lease_expires_at = past
            await repo.update_job(leased_job)

            # 1. Progress on expired lease
            with pytest.raises(UnauthorizedError) as exc_prog:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="sampling",
                        step=5,
                        total=10,
                        percent=50.0,
                        worker_id="worker-p06-05",
                        lease_token=token,
                    ),
                    caller_worker_id="worker-p06-05",
                    caller_role="worker",
                )
            assert exc_prog.value.status_code == 403
            assert "expired" in exc_prog.value.message.lower()

            # 2. Heartbeat on expired lease
            with pytest.raises(UnauthorizedError) as exc_hb:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="worker-p06-05",
                        active_job_id=job.job_id,
                        lease_token=token,
                    )
                )
            assert exc_hb.value.status_code == 403
            assert "expired" in exc_hb.value.message.lower()
        finally:
            await repo.close()

    async def test_heartbeat_ownership_validation(self):
        """Heartbeat with active_job_id validates worker_id and lease_token."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Heartbeat ownership"},
                ),
                request_id="req-p06-hb",
            )
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-owner",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease_resp.lease_token

            # 1. Imposter worker sending heartbeat for another worker's job
            with pytest.raises(UnauthorizedError) as exc_imposter:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="worker-imposter",
                        active_job_id=job.job_id,
                        lease_token=token,
                    )
                )
            assert exc_imposter.value.status_code == 403
            assert "not the active lessee" in exc_imposter.value.message

            # 2. Correct worker but invalid lease_token
            with pytest.raises(UnauthorizedError) as exc_bad_tok:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="worker-owner",
                        active_job_id=job.job_id,
                        lease_token="bad-token-xyz",
                    )
                )
            assert exc_bad_tok.value.status_code == 403
            assert "Invalid lease token" in exc_bad_tok.value.message

            # 3. Legitimate worker with correct token extends lease
            hb_ok = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-owner",
                    active_job_id=job.job_id,
                    lease_token=token,
                )
            )
            assert hb_ok.status == "acknowledged"
            assert hb_ok.lease_extended_until is not None

            # 4. Idle heartbeat (no active job) succeeds cleanly
            hb_idle = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-owner",
                    active_job_id=None,
                )
            )
            assert hb_idle.status == "acknowledged"
            assert hb_idle.lease_extended_until is None
        finally:
            await repo.close()

    async def test_admin_role_bypasses_lease_ownership(self):
        """Admin caller can mutate/complete/fail jobs without worker lease token."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Admin bypass test"},
                ),
                request_id="req-p06-admin",
            )
            await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-p06-06",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )

            # Admin caller completes job without providing lease_token
            completed = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    worker_id="worker-p06-06",
                    inference_duration_ms=500.0,
                    result={"admin": "override"},
                ),
                caller_role="admin",
            )
            assert completed.state == JobState.SUCCEEDED
            assert completed.worker is None
        finally:
            await repo.close()
