"""
tests/security/test_p2_worker_lease_adversarial_probes.py
Empirical Adversarial Stress Harness for Phase 2:
1. Worker Lease Validation & Mutation Attacks (P0.6):
   - Mismatched / forged lease token on active job -> HTTP 403 UnauthorizedError.
   - Missing lease token on active job -> HTTP 403 UnauthorizedError.
   - Unleased job mutation attempts -> HTTP 403 UnauthorizedError (never dynamically leased).
   - Forged token on terminal states (SUCCEEDED, FAILED, CANCELLED) -> HTTP 403 or 409 (never silent 200).
   - HTTP endpoint identity spoofing & forbidden role attacks.
2. Worker Idle State Clearing Probes (P1.5):
   - Job completion clears worker current_job_id to None in repository.
   - Job failure clears worker current_job_id to None in repository.
   - Job cancellation clears worker current_job_id to None in repository.
   - Watchdog heartbeat timeout sweep clears worker current_job_id to None and frees GPU partition.
3. Dual T4 Multi-GPU Concurrency & Mutual Exclusion (P1.5):
   - dual_gpu_split (qwen-text) fails atomically when GPU 0 is busy, leaving GPU 1 completely free.
   - dual_gpu_split (qwen-text) fails atomically when GPU 1 is busy, leaving GPU 0 completely free.
   - Successful dual_gpu_split lease marks both GPU 0 and GPU 1 busy, mutually excluding single-GPU jobs.
   - Release frees both GPUs simultaneously and allows independent single-GPU jobs to proceed.
"""
from datetime import datetime, timedelta, timezone
import json
import os
import time
import uuid
import pytest
import jwt
from starlette.testclient import TestClient

from factory_contracts.errors import (
    ErrorDetail,
    ForbiddenError,
    InvalidRequestError,
    UnauthorizedError,
    FactoryError,
)
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
from factory_contracts.manifests.model import AllocationPolicy
from factory.apps.control_api.src.queue.repository import InMemoryJobStore
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog
from factory.apps.control_api.src.main import app as control_api_app
from factory_auth.config import AuthConfig


# ==============================================================================
# Helper Fixtures & Setup
# ==============================================================================

JWT_SECRET = "adversarial-probe-jwt-secret-key-32chars"


def make_worker_jwt(worker_id: str, role: str = "worker", exp_seconds: int = 3600) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": worker_id,
        "worker_id": worker_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=exp_seconds)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def build_test_environment():
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    await scheduler.initialize()
    stream_mgr = StreamManager(repository=repo)
    manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)
    coordinator = LeaseCoordinator(repository=repo, broker=scheduler)
    return repo, scheduler, stream_mgr, manager, coordinator


# ==============================================================================
# Group 1: Adversarial Lease Mutation Probes (P0.6)
# ==============================================================================

@pytest.mark.asyncio
class TestAdversarialLeaseMutationsP06:
    """Stress-tests unauthorized, forged, unleased, and terminal state mutations."""

    async def test_probe_mismatched_lease_token_on_active_job(self):
        """Mismatched lease_token on active job must raise HTTP 403 UnauthorizedError across all mutation endpoints."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Adversarial token test"},
                ),
                request_id="req-probe-mismatch",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker-01",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None
            assert lease.lease_token is not None
            genuine_token = lease.lease_token
            wrong_token = f"forged-token-{uuid.uuid4()}"

            # 1. Progress with wrong token
            with pytest.raises(UnauthorizedError) as exc_prog:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="generating",
                        step=5,
                        total=20,
                        percent=25.0,
                        worker_id="legit-worker-01",
                        lease_token=wrong_token,
                    ),
                    caller_worker_id="legit-worker-01",
                    caller_role="worker",
                )
            assert exc_prog.value.status_code == 403
            assert "Invalid or missing lease token" in exc_prog.value.message

            # 2. Complete with wrong token
            with pytest.raises(UnauthorizedError) as exc_comp:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        worker_id="legit-worker-01",
                        lease_token=wrong_token,
                        result={"text": "hacked"},
                    ),
                    caller_worker_id="legit-worker-01",
                    caller_role="worker",
                )
            assert exc_comp.value.status_code == 403
            assert "Invalid or missing lease token" in exc_comp.value.message

            # 3. Fail with wrong token
            with pytest.raises(UnauthorizedError) as exc_fail:
                await manager.fail_job(
                    job.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(code="INTERNAL_ERROR", message="boom", request_id=job.request_id),
                        worker_id="legit-worker-01",
                        lease_token=wrong_token,
                    ),
                    caller_worker_id="legit-worker-01",
                    caller_role="worker",
                )
            assert exc_fail.value.status_code == 403
            assert "Invalid or missing lease token" in exc_fail.value.message

            # 4. Heartbeat with wrong token
            with pytest.raises(UnauthorizedError) as exc_hb:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="legit-worker-01",
                        active_job_id=job.job_id,
                        lease_token=wrong_token,
                    )
                )
            assert exc_hb.value.status_code == 403
            assert "Invalid lease token" in exc_hb.value.message

            # Verify genuine worker with genuine token still succeeds
            ok_progress = await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="generating",
                    step=10,
                    total=20,
                    percent=50.0,
                    worker_id="legit-worker-01",
                    lease_token=genuine_token,
                ),
                caller_worker_id="legit-worker-01",
                caller_role="worker",
            )
            assert ok_progress.progress.percent == 50.0
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_missing_lease_token_on_leased_job(self):
        """Worker calling progress, complete, fail with missing lease_token (None or empty) must return HTTP 403."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Missing token probe"},
                ),
                request_id="req-probe-missing",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-missing-tok",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None

            # Test None and empty string
            for bad_tok in (None, ""):
                # 1. Progress
                with pytest.raises(UnauthorizedError) as exc_prog:
                    await manager.update_progress(
                        job.job_id,
                        ProgressUpdateRequest(
                            stage="generating",
                            step=1,
                            total=10,
                            percent=10.0,
                            worker_id="worker-missing-tok",
                            lease_token=bad_tok,
                        ),
                        caller_worker_id="worker-missing-tok",
                        caller_role="worker",
                    )
                assert exc_prog.value.status_code == 403

                # 2. Complete
                with pytest.raises(UnauthorizedError) as exc_comp:
                    await manager.complete_job(
                        job.job_id,
                        JobCompletionRequest(
                            worker_id="worker-missing-tok",
                            lease_token=bad_tok,
                            result={"text": "none"},
                        ),
                        caller_worker_id="worker-missing-tok",
                        caller_role="worker",
                    )
                assert exc_comp.value.status_code == 403

                # 3. Fail
                with pytest.raises(UnauthorizedError) as exc_fail:
                    await manager.fail_job(
                        job.job_id,
                        JobFailureRequest(
                            error=ErrorDetail(code="INTERNAL_ERROR", message="err", request_id=job.request_id),
                            worker_id="worker-missing-tok",
                            lease_token=bad_tok,
                        ),
                        caller_worker_id="worker-missing-tok",
                        caller_role="worker",
                    )
                assert exc_fail.value.status_code == 403
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_unleased_job_mutation_rejected_no_dynamic_lease(self):
        """Worker calling progress, complete, fail on an unleased (QUEUED) job must return HTTP 403, never dynamically lease."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Unleased job mutation"},
                ),
                request_id="req-probe-unleased",
            )
            assert job.state == JobState.QUEUED
            assert job.worker is None

            # 1. Progress on unleased job
            with pytest.raises(UnauthorizedError) as exc_prog:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="generating",
                        step=1,
                        total=10,
                        percent=10.0,
                        worker_id="predator-worker",
                        lease_token="invented-token",
                    ),
                    caller_worker_id="predator-worker",
                    caller_role="worker",
                )
            assert exc_prog.value.status_code == 403

            # 2. Complete on unleased job
            with pytest.raises(UnauthorizedError) as exc_comp:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        worker_id="predator-worker",
                        lease_token="invented-token",
                        result={"text": "fake"},
                    ),
                    caller_worker_id="predator-worker",
                    caller_role="worker",
                )
            assert exc_comp.value.status_code == 403

            # 3. Fail on unleased job
            with pytest.raises(UnauthorizedError) as exc_fail:
                await manager.fail_job(
                    job.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(code="INTERNAL_ERROR", message="fake", request_id=job.request_id),
                        worker_id="predator-worker",
                        lease_token="invented-token",
                    ),
                    caller_worker_id="predator-worker",
                    caller_role="worker",
                )
            assert exc_fail.value.status_code == 403

            # Verify job is still QUEUED and was NEVER dynamically leased
            unmutated = await repo.get_job(job.job_id)
            assert unmutated.state == JobState.QUEUED
            assert unmutated.worker is None
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_forged_or_stale_token_on_terminal_jobs(self):
        """Mutation attempts on terminal jobs (SUCCEEDED, FAILED, CANCELLED) must return HTTP 403 or 409, never silent 200."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            # Case A: SUCCEEDED job
            job_succeeded, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Succeeded terminal probe"},
                ),
                request_id="req-probe-succ",
            )
            lease_succ = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-succ",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            tok_succ = lease_succ.lease_token
            # Complete the job legally
            await manager.complete_job(
                job_succeeded.job_id,
                JobCompletionRequest(
                    worker_id="worker-succ",
                    lease_token=tok_succ,
                    result={"content": "original-output"},
                ),
                caller_worker_id="worker-succ",
                caller_role="worker",
            )
            terminal_succ = await repo.get_job(job_succeeded.job_id)
            assert terminal_succ.state == JobState.SUCCEEDED
            assert terminal_succ.worker is None

            # Attempt to mutate SUCCEEDED job with forged/stale token
            with pytest.raises((UnauthorizedError, FactoryError)) as exc_succ_prog:
                await manager.update_progress(
                    job_succeeded.job_id,
                    ProgressUpdateRequest(
                        stage="tampering",
                        step=99,
                        total=100,
                        percent=99.0,
                        worker_id="worker-succ",
                        lease_token=tok_succ,
                    ),
                    caller_worker_id="worker-succ",
                    caller_role="worker",
                )
            assert exc_succ_prog.value.status_code in (403, 409)

            # Case B: FAILED job
            job_failed, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Failed terminal probe"},
                ),
                request_id="req-probe-fail",
            )
            lease_fail = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-fail",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            tok_fail = lease_fail.lease_token
            await manager.fail_job(
                job_failed.job_id,
                JobFailureRequest(
                    error=ErrorDetail(code="INTERNAL_ERROR", message="engine died", request_id=job_failed.request_id),
                    worker_id="worker-fail",
                    lease_token=tok_fail,
                    retryable=False,
                ),
                caller_worker_id="worker-fail",
                caller_role="worker",
            )
            terminal_fail = await repo.get_job(job_failed.job_id)
            assert terminal_fail.state == JobState.FAILED

            # Attempt to mutate FAILED job
            with pytest.raises((UnauthorizedError, FactoryError)) as exc_fail_prog:
                await manager.update_progress(
                    job_failed.job_id,
                    ProgressUpdateRequest(
                        stage="tampering",
                        step=50,
                        total=100,
                        percent=50.0,
                        worker_id="worker-fail",
                        lease_token=tok_fail,
                    ),
                    caller_worker_id="worker-fail",
                    caller_role="worker",
                )
            assert exc_fail_prog.value.status_code in (403, 409)

            # Case C: CANCELLED job
            job_cancelled, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Cancelled terminal probe"},
                ),
                request_id="req-probe-canc",
            )
            lease_canc = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-canc",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            tok_canc = lease_canc.lease_token
            await manager.cancel_job(job_cancelled.job_id)
            terminal_canc = await repo.get_job(job_cancelled.job_id)
            assert terminal_canc.state == JobState.CANCELLED

            # Attempt to mutate CANCELLED job
            with pytest.raises((UnauthorizedError, FactoryError)) as exc_canc_prog:
                await manager.update_progress(
                    job_cancelled.job_id,
                    ProgressUpdateRequest(
                        stage="tampering",
                        step=50,
                        total=100,
                        percent=50.0,
                        worker_id="worker-canc",
                        lease_token=tok_canc,
                    ),
                    caller_worker_id="worker-canc",
                    caller_role="worker",
                )
            assert exc_canc_prog.value.status_code in (403, 409)

            # Ensure original payloads and terminal states were never modified
            assert (await repo.get_job(job_succeeded.job_id)).state == JobState.SUCCEEDED
            assert (await repo.get_job(job_failed.job_id)).state == JobState.FAILED
            assert (await repo.get_job(job_cancelled.job_id)).state == JobState.CANCELLED
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_stolen_token_cross_worker_hijack_blocked(self):
        """Attacker possessing a stolen lease_token cannot mutate another worker's leased job."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Victim job payload"},
                ),
                request_id="req-probe-hijack",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="victim-worker-01",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None
            stolen_token = lease.lease_token

            # Scenario A: Attacker authenticates as 'attacker-worker', but uses stolen token
            with pytest.raises(UnauthorizedError) as exc_a:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="hijacking",
                        step=5,
                        total=10,
                        percent=50.0,
                        worker_id="attacker-worker",
                        lease_token=stolen_token,
                    ),
                    caller_worker_id="attacker-worker",
                    caller_role="worker",
                )
            assert exc_a.value.status_code == 403
            assert "not authorized to update job" in exc_a.value.message

            # Scenario B: Attacker authenticates as 'attacker-worker', but claims worker_id='victim-worker-01' in body
            with pytest.raises(UnauthorizedError) as exc_b:
                await manager.update_progress(
                    job.job_id,
                    ProgressUpdateRequest(
                        stage="hijacking",
                        step=5,
                        total=10,
                        percent=50.0,
                        worker_id="victim-worker-01",
                        lease_token=stolen_token,
                    ),
                    caller_worker_id="attacker-worker",
                    caller_role="worker",
                )
            assert exc_b.value.status_code == 403
            assert "does not match request worker" in exc_b.value.message

            # Scenario C: Attacker attempts to complete the victim's job
            with pytest.raises(UnauthorizedError) as exc_c:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        worker_id="attacker-worker",
                        lease_token=stolen_token,
                        result={"text": "hijacked output"},
                    ),
                    caller_worker_id="attacker-worker",
                    caller_role="worker",
                )
            assert exc_c.value.status_code == 403

            # Scenario D: Attacker attempts heartbeat on victim's job
            with pytest.raises(UnauthorizedError) as exc_d:
                await coordinator.process_heartbeat(
                    WorkerHeartbeatRequest(
                        worker_id="attacker-worker",
                        active_job_id=job.job_id,
                        lease_token=stolen_token,
                    )
                )
            assert exc_d.value.status_code == 403
            assert "not the active lessee" in exc_d.value.message
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_concurrent_worker_lease_collision_exactly_one_wins(self):
        """When 10 workers concurrently attempt to lease a single queued job, exactly 1 wins and 9 receive None."""
        import asyncio
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "High contention job"},
                ),
                request_id="req-probe-contention",
            )

            # 10 workers simultaneously attempt to lease
            async def try_lease(worker_idx: int):
                return await coordinator.lease_job(
                    WorkerLeaseRequest(
                        worker_id=f"contending-worker-{worker_idx:02d}",
                        subsystem="text",
                        supported_models=["qwen-text"],
                        supported_tasks=["chat_completion"],
                    )
                )

            results = await asyncio.gather(*[try_lease(i) for i in range(10)])

            winners = [r for r in results if r.job is not None]
            losers = [r for r in results if r.job is None]

            # Exactly ONE worker won the lease
            assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"
            assert len(losers) == 9, f"Expected 9 losers, got {len(losers)}"
            assert winners[0].job.job_id == job.job_id
            assert winners[0].lease_token is not None
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_heartbeat_on_cancelled_job_signals_cancel_and_does_not_extend(self):
        """Worker heartbeat on a CANCELLED job must return cancel_requested=True and lease_extended_until=None."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Cancellation heartbeat probe"},
                ),
                request_id="req-probe-canc-hb",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-canc-hb",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None

            # Cancel job mid-flight
            await manager.cancel_job(job.job_id)

            # Worker sends heartbeat
            hb_resp = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-canc-hb",
                    active_job_id=job.job_id,
                    lease_token=lease.lease_token,
                )
            )
            assert hb_resp.cancel_requested is True
            assert hb_resp.lease_extended_until is None
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_terminal_job_double_complete_and_double_fail_idempotence(self):
        """Completing an already SUCCEEDED job is idempotent without corrupting data or throwing."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Idempotent completion probe"},
                ),
                request_id="req-probe-idemp",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-idemp",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            tok = lease.lease_token

            # First completion
            c1 = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    worker_id="worker-idemp",
                    lease_token=tok,
                    result={"content": "original"},
                ),
                caller_worker_id="worker-idemp",
                caller_role="worker",
            )
            assert c1.state == JobState.SUCCEEDED
            assert c1.result == {"content": "original"}

            # Admin or system idempotent re-check
            c2 = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    worker_id="worker-idemp",
                    result={"content": "attempted-overwrite"},
                ),
                caller_role="admin",
            )
            assert c2.state == JobState.SUCCEEDED
            assert c2.result == {"content": "original"}  # Preserved original output!
        finally:
            await repo.close()
            await scheduler.close()


# ==============================================================================
# Group 2: Worker Idle State Clearing Probes (P1.5)
# ==============================================================================

@pytest.mark.asyncio
class TestWorkerIdleStateClearingP15:
    """Empirically verifies worker records in repository return to idle (current_job_id is None)."""

    async def test_probe_worker_cleared_on_job_completion(self):
        """Leasing a job sets worker current_job_id; completing job must reset current_job_id to None."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            worker_id = "worker-idle-comp"
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Completion idle probe"},
                ),
                request_id="req-idle-comp",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=worker_id,
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None

            # Verify active job ID is assigned in repo
            w_active = await repo.get_worker(worker_id)
            assert w_active["current_job_id"] == job.job_id

            # Complete the job
            await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    worker_id=worker_id,
                    lease_token=lease.lease_token,
                    result={"text": "completed successfully"},
                ),
                caller_worker_id=worker_id,
                caller_role="worker",
            )

            # Direct inspection of repository worker table
            w_idle = await repo.get_worker(worker_id)
            assert w_idle["current_job_id"] is None
            async with repo._lock:
                assert repo._workers[worker_id]["current_job_id"] is None
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_worker_cleared_on_job_failure(self):
        """Leasing a job sets worker current_job_id; failing job must reset current_job_id to None."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            worker_id = "worker-idle-fail"
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Failure idle probe"},
                ),
                request_id="req-idle-fail",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=worker_id,
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None

            # Assert current_job_id set
            w_active = await repo.get_worker(worker_id)
            assert w_active["current_job_id"] == job.job_id

            # Fail the job (retryable=False)
            await manager.fail_job(
                job.job_id,
                JobFailureRequest(
                    error=ErrorDetail(code="INTERNAL_ERROR", message="out of memory", request_id=job.request_id),
                    worker_id=worker_id,
                    lease_token=lease.lease_token,
                    retryable=False,
                ),
                caller_worker_id=worker_id,
                caller_role="worker",
            )

            # Direct inspection of repository worker table
            w_idle = await repo.get_worker(worker_id)
            assert w_idle["current_job_id"] is None
            async with repo._lock:
                assert repo._workers[worker_id]["current_job_id"] is None
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_worker_cleared_on_job_cancellation(self):
        """Leasing a job sets worker current_job_id; cancelling job must reset current_job_id to None."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            worker_id = "worker-idle-canc"
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Cancel idle probe"},
                ),
                request_id="req-idle-canc",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=worker_id,
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease.job is not None

            w_active = await repo.get_worker(worker_id)
            assert w_active["current_job_id"] == job.job_id

            # Cancel job
            await manager.cancel_job(job.job_id)

            # Direct inspection of repository worker table
            w_idle = await repo.get_worker(worker_id)
            assert w_idle["current_job_id"] is None
            async with repo._lock:
                assert repo._workers[worker_id]["current_job_id"] is None
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_worker_cleared_on_watchdog_timeout(self):
        """Watchdog sweep on expired worker heartbeat resets current_job_id to None and frees GPU partition."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            worker_id = "worker-idle-watchdog"
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Watchdog idle probe"},
                ),
                request_id="req-idle-wd",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=worker_id,
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease.job is not None
            assert scheduler.is_gpu_busy(0) is True

            w_active = await repo.get_worker(worker_id)
            assert w_active["current_job_id"] == job.job_id

            # Simulate heartbeat expired 75 seconds ago
            stale_iso = (datetime.now(timezone.utc) - timedelta(seconds=75)).isoformat()
            async with repo._lock:
                repo._workers[worker_id]["last_heartbeat"] = stale_iso

            # Run watchdog sweep
            failed_count = await check_active_jobs_watchdog(
                repository=repo,
                scheduler=scheduler,
                job_manager=manager,
                heartbeat_timeout_seconds=60.0,
            )
            assert failed_count == 1

            # Assert worker record in repo is cleared to idle
            w_idle = await repo.get_worker(worker_id)
            assert w_idle["current_job_id"] is None
            async with repo._lock:
                assert repo._workers[worker_id]["current_job_id"] is None

            # Assert GPU partition is freed
            assert scheduler.is_gpu_busy(0) is False
        finally:
            await repo.close()
            await scheduler.close()


# ==============================================================================
# Group 3: Dual T4 Multi-GPU Concurrency & Mutual Exclusion (P1.5)
# ==============================================================================

@pytest.mark.asyncio
class TestDualGPUMultiGPUConcurrencyP15:
    """Stress-tests atomic dual-GPU split allocation and mutual exclusion with single-GPU tasks."""

    async def test_probe_dual_gpu_fails_when_gpu0_busy_gpu1_remains_free(self):
        """Attempting to lease dual_gpu_split job when GPU 0 is busy must fail and leave GPU 1 free."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            # 1. Occupy GPU 0 with an image job
            job_gpu0, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Image on GPU 0"},
                ),
                request_id="req-occupy-gpu0",
            )
            lease0 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu0",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease0.job is not None
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is False

            # 2. Submit dual_gpu_split job (qwen-text)
            job_dual, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Qwen-Text requires dual_gpu_split"},
                ),
                request_id="req-dual-fail-0",
            )

            # 3. Attempt to lease dual_gpu_split job
            lease_dual = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-dual",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            # Must fail to lease
            assert lease_dual.job is None

            # GPU 0 remains busy by worker 0; GPU 1 MUST REMAIN COMPLETELY FREE
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is False

            # 4. A single-GPU job on GPU 1 (video) must still be able to lease without interference
            job_gpu1, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="video",
                    model="wan-2.2",
                    task="text_to_video",
                    inputs={"prompt": "Video on GPU 1"},
                ),
                request_id="req-occupy-gpu1",
            )
            lease1 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu1",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                    gpu_device=1,
                )
            )
            assert lease1.job is not None
            assert lease1.job.job_id == job_gpu1.job_id
            assert scheduler.is_gpu_busy(1) is True
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_dual_gpu_fails_when_gpu1_busy_gpu0_remains_free(self):
        """Attempting to lease dual_gpu_split job when GPU 1 is busy must fail and leave GPU 0 free."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            # 1. Occupy GPU 1 with a video job
            job_gpu1, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="video",
                    model="wan-2.2",
                    task="text_to_video",
                    inputs={"prompt": "Video on GPU 1"},
                ),
                request_id="req-occupy-gpu1-only",
            )
            lease1 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu1",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                    gpu_device=1,
                )
            )
            assert lease1.job is not None
            assert scheduler.is_gpu_busy(1) is True
            assert scheduler.is_gpu_busy(0) is False

            # 2. Submit dual_gpu_split job (qwen-text)
            job_dual, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Qwen-Text requires dual_gpu_split"},
                ),
                request_id="req-dual-fail-1",
            )

            # 3. Attempt to lease dual_gpu_split job
            lease_dual = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-dual",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            # Must fail to lease
            assert lease_dual.job is None

            # GPU 1 remains busy; GPU 0 MUST REMAIN COMPLETELY FREE
            assert scheduler.is_gpu_busy(1) is True
            assert scheduler.is_gpu_busy(0) is False

            # 4. A single-GPU job on GPU 0 (image) can still lease
            job_gpu0, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Image on GPU 0"},
                ),
                request_id="req-occupy-gpu0-test",
            )
            lease0 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-gpu0",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease0.job is not None
            assert lease0.job.job_id == job_gpu0.job_id
            assert scheduler.is_gpu_busy(0) is True
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_dual_gpu_atomic_acquisition_and_mutual_exclusion(self):
        """Successfully leasing dual_gpu_split locks both GPUs, blocking any single-GPU jobs until released."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            # 1. Both GPUs free
            assert scheduler.is_gpu_busy(0) is False
            assert scheduler.is_gpu_busy(1) is False

            # 2. Submit dual_gpu_split job (qwen-text)
            job_dual, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Dual GPU execution"},
                ),
                request_id="req-dual-exclusive",
            )

            # 3. Lease dual_gpu_split
            lease_dual = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-dual-main",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            assert lease_dual.job is not None
            assert lease_dual.job.job_id == job_dual.job_id

            # BOTH discrete GPUs must be busy
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is True

            # 4. Submit single-GPU job for GPU 0 (image) and GPU 1 (video)
            job_img, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Blocked image"},
                ),
                request_id="req-blocked-img",
            )
            job_vid, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="video",
                    model="wan-2.2",
                    task="text_to_video",
                    inputs={"prompt": "Blocked video"},
                ),
                request_id="req-blocked-vid",
            )

            # 5. Assert neither single-GPU worker can lease
            lease_img = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-img",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease_img.job is None

            lease_vid = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-vid",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                    gpu_device=1,
                )
            )
            assert lease_vid.job is None

            # 6. Complete dual-GPU job -> both GPUs must be freed simultaneously
            await manager.complete_job(
                job_dual.job_id,
                JobCompletionRequest(
                    worker_id="worker-dual-main",
                    lease_token=lease_dual.lease_token,
                    result={"content": "Dual GPU done"},
                ),
                caller_worker_id="worker-dual-main",
                caller_role="worker",
            )

            # Both GPUs must now be released
            assert scheduler.is_gpu_busy(0) is False
            assert scheduler.is_gpu_busy(1) is False

            # 7. Single-GPU jobs can now lease simultaneously!
            lease_img_retry = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-img",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease_img_retry.job is not None
            assert lease_img_retry.job.job_id == job_img.job_id

            lease_vid_retry = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-vid",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                    gpu_device=1,
                )
            )
            assert lease_vid_retry.job is not None
            assert lease_vid_retry.job.job_id == job_vid.job_id

            # Both single GPUs now busy independently
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is True
        finally:
            await repo.close()
            await scheduler.close()

    async def test_probe_rapid_multi_cycle_concurrency_stress(self):
        """Stress-tests 20 rapid sequential lease/complete cycles alternating between dual-GPU and single-GPU jobs."""
        repo, scheduler, stream_mgr, manager, coordinator = await build_test_environment()
        try:
            for cycle in range(10):
                # Phase A: Dual-GPU job
                dual_job, _ = await manager.submit_job(
                    JobSubmissionRequest(
                        service="text",
                        model="qwen-text",
                        task="chat_completion",
                        inputs={"cycle": cycle},
                    ),
                    request_id=f"req-stress-dual-{cycle}",
                )
                lease_dual = await coordinator.lease_job(
                    WorkerLeaseRequest(
                        worker_id=f"worker-dual-{cycle}",
                        subsystem="text",
                        supported_models=["qwen-text"],
                        supported_tasks=["chat_completion"],
                    )
                )
                assert lease_dual.job is not None
                assert scheduler.is_gpu_busy(0) is True
                assert scheduler.is_gpu_busy(1) is True

                # Complete Phase A
                await manager.complete_job(
                    dual_job.job_id,
                    JobCompletionRequest(
                        worker_id=f"worker-dual-{cycle}",
                        lease_token=lease_dual.lease_token,
                        result={"cycle": cycle},
                    ),
                    caller_worker_id=f"worker-dual-{cycle}",
                    caller_role="worker",
                )
                assert scheduler.is_gpu_busy(0) is False
                assert scheduler.is_gpu_busy(1) is False

                # Phase B: Two parallel single-GPU jobs on GPU 0 and GPU 1
                job0, _ = await manager.submit_job(
                    JobSubmissionRequest(
                        service="image",
                        model="qwen-image-2.1",
                        task="text_to_image",
                        inputs={"cycle": cycle},
                    ),
                    request_id=f"req-stress-img-{cycle}",
                )
                job1, _ = await manager.submit_job(
                    JobSubmissionRequest(
                        service="video",
                        model="wan-2.2",
                        task="text_to_video",
                        inputs={"cycle": cycle},
                    ),
                    request_id=f"req-stress-vid-{cycle}",
                )

                l0 = await coordinator.lease_job(
                    WorkerLeaseRequest(
                        worker_id=f"worker-img-{cycle}",
                        subsystem="image",
                        supported_models=["qwen-image-2.1"],
                        supported_tasks=["text_to_image"],
                        gpu_device=0,
                    )
                )
                l1 = await coordinator.lease_job(
                    WorkerLeaseRequest(
                        worker_id=f"worker-vid-{cycle}",
                        subsystem="video",
                        supported_models=["wan-2.2"],
                        supported_tasks=["text_to_video"],
                        gpu_device=1,
                    )
                )
                assert l0.job is not None
                assert l1.job is not None
                assert scheduler.is_gpu_busy(0) is True
                assert scheduler.is_gpu_busy(1) is True

                # Fail one and complete one
                await manager.fail_job(
                    job0.job_id,
                    JobFailureRequest(
                        error=ErrorDetail(code="INTERNAL_ERROR", message="test", request_id=job0.request_id),
                        worker_id=f"worker-img-{cycle}",
                        lease_token=l0.lease_token,
                        retryable=False,
                    ),
                    caller_worker_id=f"worker-img-{cycle}",
                    caller_role="worker",
                )
                assert scheduler.is_gpu_busy(0) is False
                assert scheduler.is_gpu_busy(1) is True

                await manager.complete_job(
                    job1.job_id,
                    JobCompletionRequest(
                        worker_id=f"worker-vid-{cycle}",
                        lease_token=l1.lease_token,
                        result={"status": "done"},
                    ),
                    caller_worker_id=f"worker-vid-{cycle}",
                    caller_role="worker",
                )
                assert scheduler.is_gpu_busy(0) is False
                assert scheduler.is_gpu_busy(1) is False

            # Final check: Zero leaked worker jobs and zero leaked GPU states
            async with repo._lock:
                for w in repo._workers.values():
                    assert w["current_job_id"] is None
        finally:
            await repo.close()
            await scheduler.close()


# ==============================================================================
# Group 4: HTTP Router End-to-End Adversarial Probes
# ==============================================================================

class TestHTTPWorkerRouterAdversarialProbes:
    """Probes FastAPI HTTP worker router directly via Starlette TestClient."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "test")
        monkeypatch.setenv("FACTORY_API_KEY", "admin-secret-key-probe")
        monkeypatch.setenv("JWT_SECRET_KEY", JWT_SECRET)
        auth_cfg = AuthConfig(
            api_key="admin-secret-key-probe",
            jwt_secret=JWT_SECRET,
            jwt_algorithm="HS256",
            environment="test",
        )
        control_api_app.state.auth_config = auth_cfg

    def test_http_probe_spoofed_worker_identity_rejected(self):
        """Worker JWT claiming sub='worker-alice' cannot pass req.worker_id='worker-bob' (HTTP 403 Forbidden)."""
        with TestClient(control_api_app) as client:
            alice_jwt = make_worker_jwt("worker-alice")

            # Attempt to report progress claiming identity of worker-bob
            res = client.post(
                "/v1/jobs/job-fake-123/progress",
                headers={"Authorization": f"Bearer {alice_jwt}"},
                json={
                    "stage": "sampling",
                    "step": 1,
                    "total": 10,
                    "percent": 10.0,
                    "worker_id": "worker-bob",  # Spoofed!
                    "lease_token": "token-123",
                },
            )
            assert res.status_code == 403
            data = res.json()
            err_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
            assert "cannot claim worker identity" in err_msg

    def test_http_probe_non_worker_role_rejected(self):
        """User JWT with role='user' cannot access worker endpoints (HTTP 403 Forbidden)."""
        with TestClient(control_api_app) as client:
            user_jwt = make_worker_jwt("normal-user", role="user")

            res = client.post(
                "/v1/jobs/job-fake-123/progress",
                headers={"Authorization": f"Bearer {user_jwt}"},
                json={
                    "stage": "sampling",
                    "step": 1,
                    "total": 10,
                    "percent": 10.0,
                    "worker_id": "normal-user",
                    "lease_token": "token-123",
                },
            )
            assert res.status_code == 403
            data = res.json()
            err_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
            assert "require role 'worker' or 'admin'" in err_msg

    def test_http_probe_missing_lease_token_returns_403(self):
        """HTTP POST to /jobs/{job_id}/progress without lease_token returns HTTP 403."""
        with TestClient(control_api_app) as client:
            worker_jwt = make_worker_jwt("worker-http-01")

            # First submit a job as admin
            admin_headers = {"Authorization": "Bearer admin-secret-key-probe"}
            sub_res = client.post(
                "/v1/jobs",
                headers=admin_headers,
                json={
                    "service": "text",
                    "model": "qwen-text",
                    "task": "chat_completion",
                    "inputs": {"prompt": "HTTP probe"},
                },
            )
            assert sub_res.status_code in (200, 201)
            job_id = sub_res.json().get("job_id") or sub_res.json().get("data", {}).get("job_id")
            assert job_id is not None

            # Lease the job
            lease_res = client.post(
                "/v1/jobs/lease",
                headers={"Authorization": f"Bearer {worker_jwt}"},
                json={
                    "worker_id": "worker-http-01",
                    "subsystem": "text",
                    "supported_models": ["qwen-text"],
                    "supported_tasks": ["chat_completion"],
                },
            )
            assert lease_res.status_code == 200
            lease_data = lease_res.json()
            assert lease_data["job"] is not None
            assert lease_data["lease_token"] is not None

            # Now worker calls progress WITHOUT lease_token
            prog_res = client.post(
                f"/v1/jobs/{job_id}/progress",
                headers={"Authorization": f"Bearer {worker_jwt}"},
                json={
                    "stage": "generating",
                    "step": 1,
                    "total": 10,
                    "percent": 10.0,
                    "worker_id": "worker-http-01",
                    # lease_token omitted!
                },
            )
            assert prog_res.status_code == 403
            data = prog_res.json()
            err_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
            assert "Invalid or missing lease token" in err_msg
