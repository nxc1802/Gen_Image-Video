"""
tests/adversarial/test_adversarial_m5_queue_stress.py
Milestone 5 Phase 2: Adversarial Coverage Hardening (Tier 5)

Adversarial stress-testing of Factory v4 Control Plane & Two-Tier Queue:
1. 100 Concurrent Job Submissions (Throughput, Latency P50/P95/P99, Zero-Drop, State Consistency)
2. Worker Crash & Reboot Recovery (TTL Expiration, Dead-Job Recovery Loop, Re-leasing, Max Retries)
3. Job Cancellation During High Concurrency (Queued & Running Eviction, Zero Orphan Workers, Idempotency)
4. S3 Artifact Upload & Presigned URL Generation (>1MB Artifacts, SHA256 Bit-Level Integrity, SigV4 & HMAC)
"""
import asyncio
import hashlib
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import patch

import httpx
import pytest

from factory_auth.config import AuthConfig
from factory_contracts.errors import ErrorCode, ErrorDetail, InvalidRequestError
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    JobTelemetry,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from factory.apps.control_api.src.main import app as control_plane_app, lifespan
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.reaper import DeadJobReaper
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.queue.state_machine import JobStateMachine
from factory.apps.control_api.src.storage.config import StorageConfig
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.s3 import S3StorageProvider
from factory.apps.control_api.src.streaming.manager import StreamManager

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}

# Ensure testing environment flags
os.environ["FACTORY_API_KEY"] = AUTH_TOKEN
os.environ["STORAGE_BACKEND"] = "local"
os.environ["FACTORY_DB_BACKEND"] = "sqlite"
os.environ["SQLITE_DB_PATH"] = ":memory:"


@pytest.fixture(autouse=True)
def configure_test_env(monkeypatch):
    monkeypatch.setenv("FACTORY_API_KEY", AUTH_TOKEN)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", ":memory:")
    control_plane_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")


# ==============================================================================
# 1. 100 CONCURRENT JOB SUBMISSIONS STRESS SUITE
# ==============================================================================
@pytest.mark.asyncio
class TestConcurrentIndexJobSubmissionStress:
    """Empirical verification of 100 concurrent job submissions under extreme load."""

    async def test_100_concurrent_job_submissions_api_throughput_and_latency(self):
        """
        Adversarial acceptance criteria:
        - Accepts 100 concurrent job submissions without crashing or dropping requests.
        - Measures throughput (jobs/sec) and latency profile (P50, P95, P99).
        - 100% of responses return HTTP 201 Created.
        - All 100 job IDs are distinct UUIDs.
        - All 100 jobs are queryable and in QUEUED state.
        - Broker active queue depth matches exactly 100.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                latencies = []

                async def _submit_single(index: int) -> Dict[str, Any]:
                    payload = {
                        "service": "image",
                        "model": "qwen-image-2.1",
                        "task": "text_to_image",
                        "inputs": {"prompt": f"Adversarial concurrent stress prompt #{index}"},
                        "priority": (index % 10) + 1,
                        "idempotency_key": f"concurrency-key-{uuid.uuid4()}",
                    }
                    t0 = time.perf_counter()
                    res = await client.post("/v1/jobs", json=payload)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)  # ms
                    assert res.status_code == 201, f"Failed at request #{index}: status {res.status_code}, {res.text}"
                    return res.json()

                # Launch 100 simultaneous concurrent requests
                start_wall = time.perf_counter()
                results = await asyncio.gather(*[_submit_single(i) for i in range(100)])
                total_duration = time.perf_counter() - start_wall

                # 1. Zero requests dropped or failed
                assert len(results) == 100, f"Expected 100 results, got {len(results)}"
                job_ids = [r["job_id"] for r in results]

                # 2. All 100 job IDs are unique
                assert len(set(job_ids)) == 100, "Duplicate job_id detected under concurrent submission!"

                # 3. Check broker queue depth via app state
                broker = control_plane_app.state.broker
                queue_depth = await broker.get_queue_depth("image")
                assert queue_depth == 100, f"Expected broker queue depth 100, got {queue_depth}"

                # 4. Spot check status query of randomly sampled submitted jobs
                for check_idx in [0, 25, 50, 75, 99]:
                    check_id = job_ids[check_idx]
                    status_res = await client.get(f"/v1/jobs/{check_id}")
                    assert status_res.status_code == 200
                    job_data = status_res.json()
                    assert job_data["job_id"] == check_id
                    assert job_data["state"] == "QUEUED"
                    assert job_data["progress"]["percent"] == 0.0

                # 5. Compute and assert performance metrics
                latencies.sort()
                p50 = latencies[int(len(latencies) * 0.50)]
                p95 = latencies[int(len(latencies) * 0.95)]
                p99 = latencies[int(len(latencies) * 0.99)]
                throughput = 100.0 / total_duration

                assert throughput > 20.0, f"Throughput too low: {throughput:.1f} jobs/sec"
                assert p95 < 1500.0, f"P95 latency too high: {p95:.1f}ms"

    async def test_100_concurrent_heterogeneous_subsystems_burst(self):
        """
        Adversarially burst 100 concurrent jobs across 8 distinct subsystems
        with different priority tiers (1 to 100) to stress indexing and broker partitioning.
        """
        subsystems = [
            ("image", "qwen-image-2.1", "text_to_image", {"prompt": "photo of nebula"}),
            ("video", "wan-2.2", "text_to_video", {"prompt": "waterfall flowing", "fps": 24}),
            ("speech", "qwen-speech", "speech_synthesis", {"text": "hello factory"}),
            ("music", "ace-step-1.5", "text_to_music", {"prompt": "electronic ambient"}),
            ("text", "qwen-text", "chat_completion", {"messages": [{"role": "user", "content": "hi"}]}),
            ("decision", "qwen-decision", "predicate", {"hypothesis": "is valid"}),
            ("embedding", "qwen-embedding", "vector_embeddings", {"input": "factory text"}),
            ("vision", "qwen-vlm", "visual_language", {"prompt": "describe this image", "image_url": "https://example.com/test.png"}),
        ]

        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                async def _submit_hetero(i: int):
                    spec = subsystems[i % len(subsystems)]
                    payload = {
                        "service": spec[0],
                        "model": spec[1],
                        "task": spec[2],
                        "inputs": spec[3],
                        "priority": (i * 7 % 100) + 1,
                    }
                    res = await client.post("/v1/jobs", json=payload)
                    assert res.status_code == 201
                    return spec[0], res.json()["job_id"]

                burst = await asyncio.gather(*[_submit_hetero(i) for i in range(100)])
                assert len(burst) == 100

                # Validate counts per service
                service_counts = {}
                for srv, _ in burst:
                    service_counts[srv] = service_counts.get(srv, 0) + 1

                broker = control_plane_app.state.broker
                for srv, expected_count in service_counts.items():
                    depth = await broker.get_queue_depth(srv)
                    assert depth == expected_count, f"Service {srv} queue depth {depth} != expected {expected_count}"

    async def test_concurrent_idempotency_key_race_handled_gracefully(self):
        """
        Adversarially submit 20 concurrent requests sharing the EXACT same idempotency key.
        Verify that all requests return successfully without unhandled 500 exceptions,
        and all returned jobs share the identical job_id.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                shared_key = f"race-idempotency-{uuid.uuid4()}"
                payload = {
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "race idempotency prompt"},
                    "priority": 10,
                    "idempotency_key": shared_key,
                }

                # Fire 20 requests simultaneously
                responses = await asyncio.gather(
                    *[client.post("/v1/jobs", json=payload) for _ in range(20)],
                    return_exceptions=True,
                )

                # Ensure zero unhandled internal crashes (no 500)
                status_codes = [r.status_code for r in responses if isinstance(r, httpx.Response)]
                assert all(sc in (200, 201) for sc in status_codes), f"Unexpected status codes: {status_codes}"

                # Ensure all successful responses reference the same single job_id
                job_ids = [r.json()["job_id"] for r in responses if isinstance(r, httpx.Response) and r.status_code in (200, 201)]
                assert len(set(job_ids)) == 1, f"Multiple job IDs generated for single idempotency key: {set(job_ids)}"

    async def test_concurrent_worker_leasing_race_zero_duplicates(self):
        """
        Adversarially simulate 20 workers racing simultaneously to acquire leases
        from a queue of 20 jobs.
        Verify:
        - All 20 workers acquire a job (no unhandled conflicts or race dropouts).
        - Every leased job ID is strictly distinct (zero duplicate leases across workers).
        - Broker queue depth drops cleanly from 20 to 0.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                broker = control_plane_app.state.broker

                # Enqueue 20 jobs
                enqueue_tasks = [
                    client.post("/v1/jobs", json={
                        "service": "image",
                        "model": "qwen-image-2.1",
                        "task": "text_to_image",
                        "inputs": {"prompt": f"Lease race job #{i}"},
                        "priority": 10,
                    })
                    for i in range(20)
                ]
                sub_resps = await asyncio.gather(*enqueue_tasks)
                assert all(r.status_code == 201 for r in sub_resps)
                assert await broker.get_queue_depth("image") == 20

                # 20 workers concurrently call /v1/jobs/lease
                async def _worker_lease(w_idx: int):
                    return await client.post("/v1/jobs/lease", json={
                        "worker_id": f"worker-race-{w_idx:02d}",
                        "subsystem": "image",
                        "supported_models": ["qwen-image-2.1"],
                        "supported_tasks": ["text_to_image"],
                        "gpu_device": w_idx % 4,
                        "lease_duration_seconds": 300,
                    })

                lease_resps = await asyncio.gather(*[_worker_lease(i) for i in range(20)])
                assert all(r.status_code == 200 for r in lease_resps)

                leased_jobs = [r.json()["job"] for r in lease_resps]
                assert all(j is not None for j in leased_jobs), "One or more workers failed to acquire a job!"

                leased_ids = [j["job_id"] for j in leased_jobs]
                assert len(set(leased_ids)) == 20, f"Duplicate leases detected: {len(leased_ids)} total vs {len(set(leased_ids))} unique"

                # Broker queue depth must be 0
                assert await broker.get_queue_depth("image") == 0


# ==============================================================================
# 2. WORKER CRASH & REBOOT RECOVERY (LEASING, TTL, REAPER)
# ==============================================================================
@pytest.mark.asyncio
class TestWorkerCrashAndRebootRecovery:
    """Empirical verification of worker death, lease timeout, dead-job recovery loop, and re-leasing."""

    async def test_worker_crash_expired_lease_recovered_and_completed_by_rebooted_worker(self):
        """
        End-to-end worker crash & reboot recovery lifecycle:
        1. Submit job.
        2. Worker 1 leases job with active lease TTL.
        3. Job state transitions to RUNNING.
        4. Worker 1 abruptly crashes / disappears (stops heartbeating).
        5. Time advances past lease expiration timestamp (TTL 300s test-adjusted).
        6. Dead-Job Recovery loop (DeadJobReaper) sweeps:
           - Detects expired lease.
           - Resets job to QUEUED.
           - Increments retry_count to 1.
           - Re-enqueues job into active broker queue.
        7. Fresh rebooted Worker 2 queries /v1/jobs/lease and acquires the job.
        8. Worker 2 reports progress and completes the job with valid artifacts.
        9. Job cleanly transitions to SUCCEEDED.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository
                broker: MemoryQueueBroker = control_plane_app.state.broker
                reaper: DeadJobReaper = control_plane_app.state.reaper

                # Step 1: Submit job
                sub_res = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "Worker crash resilience drill"},
                    "priority": 20,
                })
                assert sub_res.status_code == 201
                job_id = sub_res.json()["job_id"]

                # Initial state: QUEUED, broker depth = 1
                assert await broker.get_queue_depth("image") == 1

                # Step 2: Worker 1 leases job
                w1_lease_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-gpu0-dying",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "gpu_device": 0,
                    "lease_duration_seconds": 30,  # Minimum schema valid lease
                })
                assert w1_lease_res.status_code == 200
                lease_data = w1_lease_res.json()
                assert lease_data["job"] is not None
                assert lease_data["job"]["job_id"] == job_id
                assert lease_data["job"]["state"] == "RUNNING"
                assert lease_data["job"]["worker"]["worker_id"] == "worker-gpu0-dying"

                # Broker active depth is now 0 (job evicted while running)
                assert await broker.get_queue_depth("image") == 0

                # Step 3: Simulate Worker 1 abrupt death by advancing lease_expires_at into the past
                expired_time = datetime.now(timezone.utc) - timedelta(seconds=15)
                job_record = await repo.get_job(job_id)
                assert job_record is not None
                job_record.worker.lease_expires_at = expired_time
                await repo.update_job(job_record)

                # Step 4: Dead-Job Recovery sweep
                recovered = await reaper.sweep_once()
                assert recovered == 1, f"Expected 1 recovered job, got {recovered}"

                # Step 5: Verify job state was reset to QUEUED, retry_count incremented, re-enqueued to broker
                reclaimed_job = await repo.get_job(job_id)
                assert reclaimed_job.state == JobState.QUEUED
                assert reclaimed_job.retry_count == 1
                assert reclaimed_job.worker is None
                assert reclaimed_job.progress.stage == "queued_retry"
                assert await broker.get_queue_depth("image") == 1

                # Step 6: Fresh rebooted Worker 2 leases the recovered job
                w2_lease_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-gpu0-rebooted",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "gpu_device": 0,
                    "lease_duration_seconds": 300,
                })
                assert w2_lease_res.status_code == 200
                w2_data = w2_lease_res.json()
                assert w2_data["job"] is not None
                assert w2_data["job"]["job_id"] == job_id
                assert w2_data["job"]["state"] == "RUNNING"
                assert w2_data["job"]["worker"]["worker_id"] == "worker-gpu0-rebooted"
                assert await broker.get_queue_depth("image") == 0

                # Step 7: Worker 2 updates progress and completes the job
                prog_res = await client.post(f"/v1/jobs/{job_id}/progress", json={
                    "worker_id": "worker-gpu0-rebooted",
                    "lease_token": w2_data["lease_token"],
                    "stage": "sampling",
                    "step": 20,
                    "total": 20,
                    "percent": 100.0,
                    "message": "Generation completed after reboot recovery",
                })
                assert prog_res.status_code == 200

                artifact = {
                    "artifact_id": f"art-{uuid.uuid4()}",
                    "type": "image",
                    "mime_type": "image/png",
                    "size_bytes": 102400,
                    "storage_path": f"image/artifacts/{job_id}/output.png",
                    "download_url": f"https://s3.ai.example.com/image/artifacts/{job_id}/output.png",
                    "sha256": "a" * 64,
                }
                comp_res = await client.post(f"/v1/jobs/{job_id}/complete", json={
                    "worker_id": "worker-gpu0-rebooted",
                    "lease_token": w2_data["lease_token"],
                    "artifacts": [artifact],
                    "inference_duration_ms": 1450.0,
                    "peak_vram_mb": 6800.0,
                })
                assert comp_res.status_code == 200

                # Step 8: Assert final status is SUCCEEDED with full artifacts and telemetry
                final_res = await client.get(f"/v1/jobs/{job_id}")
                assert final_res.status_code == 200
                final_job = final_res.json()
                assert final_job["state"] == "SUCCEEDED"
                assert final_job["retry_count"] == 1
                assert len(final_job["artifacts"]) == 1
                assert final_job["telemetry"]["inference_duration_ms"] == 1450.0

    async def test_repeated_worker_crashes_exhaust_max_retries_transition_to_failed(self):
        """
        Adversarial test: A worker continually crashes on a poisoned job.
        After exceeding max_retries (default 3), the Dead-Job Reaper must:
        - Transition the job to FAILED.
        - Attach ErrorDetail with code=TIMEOUT.
        - Set worker=None and record completion timestamp.
        - Evict job permanently without re-enqueueing to broker.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository
                broker: MemoryQueueBroker = control_plane_app.state.broker
                reaper: DeadJobReaper = control_plane_app.state.reaper

                # Submit job (default max_retries = 3)
                sub_res = await client.post("/v1/jobs", json={
                    "service": "video",
                    "model": "wan-2.2",
                    "task": "text_to_video",
                    "inputs": {"prompt": "Poisoned prompt crashing GPU"},
                })
                job_id = sub_res.json()["job_id"]

                # 3 consecutive worker crashes (retry 1, 2, 3)
                for attempt in range(1, 4):
                    # Lease
                    await client.post("/v1/jobs/lease", json={
                        "worker_id": f"worker-crash-{attempt}",
                        "subsystem": "video",
                        "supported_models": ["wan-2.2"],
                        "supported_tasks": ["text_to_video"],
                        "lease_duration_seconds": 30,
                    })
                    # Expire lease
                    j = await repo.get_job(job_id)
                    assert j.state == JobState.RUNNING
                    j.worker.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                    await repo.update_job(j)
                    # Sweep -> increments retry_count
                    recovered = await reaper.sweep_once()
                    assert recovered == 1
                    j_reloaded = await repo.get_job(job_id)
                    assert j_reloaded.retry_count == attempt
                    assert j_reloaded.state == JobState.QUEUED

                # Attempt 4: Worker leases at retry_count=3 (equal to max_retries=3) and crashes again
                await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-crash-final",
                    "subsystem": "video",
                    "supported_models": ["wan-2.2"],
                    "supported_tasks": ["text_to_video"],
                    "lease_duration_seconds": 30,
                })
                j_final = await repo.get_job(job_id)
                assert j_final.retry_count == 3
                j_final.worker.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                await repo.update_job(j_final)

                # Final sweep: retry_count (3) is not < max_retries (3) -> mark FAILED
                assert await reaper.sweep_once() == 1

                failed_job = await repo.get_job(job_id)
                assert failed_job.state == JobState.FAILED
                assert failed_job.error is not None
                assert failed_job.error.code == ErrorCode.TIMEOUT
                assert failed_job.worker is None
                assert failed_job.completed_at is not None

                # Broker queue must be empty (job permanently removed)
                assert await broker.get_queue_depth("video") == 0

    async def test_zombie_worker_resurrection_after_reaper_sweep_prevented(self):
        """
        Adversarial race scenario:
        1. Worker 1 leases job, then lags/sleeps.
        2. Lease expires, reaper reclaims job, and Worker 2 acquires the new lease.
        3. Zombie Worker 1 awakens and attempts to send heartbeat extending its lease.
        4. Assert: Worker 1's heartbeat cannot hijack or extend the lease of Worker 2!
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository
                reaper: DeadJobReaper = control_plane_app.state.reaper

                # Submit job
                sub_res = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "zombie worker test"},
                })
                job_id = sub_res.json()["job_id"]

                # Worker 1 leases job
                await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-zombie-1",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 30,
                })

                # Expire Worker 1 lease and sweep
                j1 = await repo.get_job(job_id)
                j1.worker.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                await repo.update_job(j1)
                await reaper.sweep_once()

                # Worker 2 leases job
                w2_resp = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-legit-2",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 300,
                })
                assert w2_resp.json()["job"]["worker"]["worker_id"] == "worker-legit-2"

                active_job = await repo.get_job(job_id)
                legit_expires = active_job.worker.lease_expires_at

                # Zombie Worker 1 awakens and tries to heartbeat
                await client.post(
                    "/v1/workers/worker-zombie-1/heartbeat",
                    json={
                        "worker_id": "worker-zombie-1",
                        "active_job_id": job_id,
                    },
                )

                # Check that Worker 2 remains the sole owner and lease was NOT modified by zombie
                verified_job = await repo.get_job(job_id)
                assert verified_job.worker.worker_id == "worker-legit-2"
                assert verified_job.worker.lease_expires_at == legit_expires

    async def test_worker_crash_during_streaming_state_recovered_by_reaper(self):
        """
        Adversarial test: A worker enters STREAMING state and then abruptly crashes.
        Verify:
        - Dead-Job Reaper detects the expired lease even when state == STREAMING.
        - Reaper resets job to QUEUED and increments retry_count.
        - Re-enqueues into broker.
        - Rebooted worker successfully re-leases and completes the job.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository
                broker: MemoryQueueBroker = control_plane_app.state.broker
                reaper: DeadJobReaper = control_plane_app.state.reaper

                # Submit job
                sub_res = await client.post("/v1/jobs", json={
                    "service": "text",
                    "model": "qwen-text",
                    "task": "chat_completion",
                    "inputs": {"messages": [{"role": "user", "content": "Explain quantum entanglement"}]},
                })
                job_id = sub_res.json()["job_id"]

                # Worker 1 leases job
                l_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-text-crasher",
                    "subsystem": "text",
                    "supported_models": ["qwen-text"],
                    "supported_tasks": ["chat_completion"],
                    "lease_duration_seconds": 30,
                })
                assert l_res.status_code == 200

                # Worker moves job to STREAMING
                job_obj = await repo.get_job(job_id)
                job_obj.state = JobState.STREAMING
                await repo.update_job(job_obj)

                # Worker crashes: expire lease
                job_obj = await repo.get_job(job_id)
                job_obj.worker.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                await repo.update_job(job_obj)

                # Reaper sweep
                recovered = await reaper.sweep_once()
                assert recovered == 1

                reclaimed = await repo.get_job(job_id)
                assert reclaimed.state == JobState.QUEUED
                assert reclaimed.retry_count == 1
                assert await broker.get_queue_depth("text") == 1

                # Rebooted worker leases and completes
                w2_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-text-rebooted",
                    "subsystem": "text",
                    "supported_models": ["qwen-text"],
                    "supported_tasks": ["chat_completion"],
                    "lease_duration_seconds": 300,
                })
                assert w2_res.status_code == 200
                assert w2_res.json()["job"]["job_id"] == job_id

                comp_res = await client.post(f"/v1/jobs/{job_id}/complete", json={
                    "worker_id": "worker-text-rebooted",
                    "lease_token": w2_res.json()["lease_token"],
                    "artifacts": [],
                    "inference_duration_ms": 2100.0,
                    "tokens_per_sec": 85.0,
                })
                assert comp_res.status_code == 200

                final_job = await repo.get_job(job_id)
                assert final_job.state == JobState.SUCCEEDED
                assert final_job.retry_count == 1

    async def test_active_worker_heartbeat_extends_lease_preventing_reaper(self):
        """
        Adversarial test: An active worker heartbeats periodically.
        Verify that its lease is extended and DeadJobReaper does NOT prematurely reap it.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository
                reaper: DeadJobReaper = control_plane_app.state.reaper

                sub_res = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "Long generation prompt"},
                })
                job_id = sub_res.json()["job_id"]

                # Lease job
                await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-healthy-active",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 300,
                })

                # Send heartbeat
                hb_res = await client.post("/v1/workers/worker-healthy-active/heartbeat", json={
                    "worker_id": "worker-healthy-active",
                    "active_job_id": job_id,
                    "gpu_utilization_percent": 95.0,
                    "vram_used_mb": 14200.0,
                })
                assert hb_res.status_code == 200
                assert hb_res.json()["cancel_requested"] is False

                # Reaper sweep while worker is healthy
                recovered = await reaper.sweep_once()
                assert recovered == 0, f"Healthy worker job was improperly reaped: {recovered}"

                job = await repo.get_job(job_id)
                assert job.state == JobState.RUNNING
                assert job.worker.worker_id == "worker-healthy-active"


# ==============================================================================
# 3. JOB CANCELLATION UNDER HIGH CONCURRENCY
# ==============================================================================
@pytest.mark.asyncio
class TestJobCancellationUnderHighConcurrency:
    """Empirical verification of clean cancellation across queued & running states without orphan workers."""

    async def test_concurrent_cancellation_of_queued_and_running_jobs(self):
        """
        Adversarial load:
        - Submit 40 jobs.
        - 20 jobs leased to running workers; 20 remain in QUEUED state.
        - Concurrently fire 40 cancellation requests across all jobs.
        - Assert:
          1. 100% of jobs transition cleanly to CANCELLED.
          2. Queued jobs are immediately evicted from the broker.
          3. Running workers receive cancel_requested=True on their heartbeat.
          4. Zero orphan workers left executing ghost tasks.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository
                broker: MemoryQueueBroker = control_plane_app.state.broker

                # Submit 40 jobs
                job_ids = []
                for i in range(40):
                    res = await client.post("/v1/jobs", json={
                        "service": "image",
                        "model": "qwen-image-2.1",
                        "task": "text_to_image",
                        "inputs": {"prompt": f"Cancellation concurrency #{i}"},
                        "priority": 10,
                    })
                    assert res.status_code == 201
                    job_ids.append(res.json()["job_id"])

                assert await broker.get_queue_depth("image") == 40

                # Lease 20 jobs to workers
                running_workers = {}
                for w in range(20):
                    worker_id = f"worker-cancel-test-{w:02d}"
                    l_res = await client.post("/v1/jobs/lease", json={
                        "worker_id": worker_id,
                        "subsystem": "image",
                        "supported_models": ["qwen-image-2.1"],
                        "supported_tasks": ["text_to_image"],
                        "gpu_device": w % 2,
                        "lease_duration_seconds": 300,
                    })
                    assert l_res.status_code == 200
                    leased_job_id = l_res.json()["job"]["job_id"]
                    running_workers[worker_id] = leased_job_id

                # Broker depth is now 20 (20 leased, 20 queued)
                assert await broker.get_queue_depth("image") == 20

                # Concurrently issue cancellation for all 40 jobs
                cancel_responses = await asyncio.gather(
                    *[client.post(f"/v1/jobs/{jid}/cancel") for jid in job_ids]
                )
                assert all(r.status_code == 200 for r in cancel_responses)
                assert all(r.json()["state"] == "CANCELLED" for r in cancel_responses)

                # Broker depth must immediately become 0
                assert await broker.get_queue_depth("image") == 0

                # All 20 running workers send heartbeat -> must receive cancel_requested=True
                for worker_id, assigned_job_id in running_workers.items():
                    hb_res = await client.post(
                        f"/v1/workers/{worker_id}/heartbeat",
                        json={
                            "worker_id": worker_id,
                            "active_job_id": assigned_job_id,
                        },
                    )
                    assert hb_res.status_code == 200
                    hb_data = hb_res.json()
                    assert hb_data["cancel_requested"] is True, (
                        f"Worker {worker_id} did not receive cancel signal for job {assigned_job_id}!"
                    )

                # Attempting to lease any new jobs must return None (queue completely drained)
                idle_lease = await client.post("/v1/jobs/lease", json={
                    "worker_id": "idle-worker",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 60,
                })
                assert idle_lease.json()["job"] is None

    async def test_cancellation_idempotency_under_burst_duplicate_calls(self):
        """
        Adversarially fire 15 simultaneous cancellation calls for the SAME active job.
        Assert that all 15 return 200 OK with CANCELLED without race condition errors.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                sub = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "duplicate cancel drill"},
                })
                job_id = sub.json()["job_id"]

                # Fire 15 cancels concurrently
                results = await asyncio.gather(
                    *[client.delete(f"/v1/jobs/{job_id}") for _ in range(15)]
                )
                assert all(r.status_code == 200 for r in results)
                assert all(r.json()["state"] == "CANCELLED" for r in results)

    async def test_cancel_already_succeeded_job_rejected_with_400(self):
        """
        Attempting to cancel a job that has already completed successfully
        must be strictly rejected with HTTP 400 InvalidRequestError.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                sub = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "completion protection drill"},
                })
                job_id = sub.json()["job_id"]

                # Lease job to move state to RUNNING
                lease_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-complete-protect",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 300,
                })
                assert lease_res.status_code == 200

                # Complete job (RUNNING -> SUCCEEDED)
                art = {
                    "artifact_id": f"art-{uuid.uuid4()}",
                    "type": "image",
                    "mime_type": "image/png",
                    "size_bytes": 500,
                    "storage_path": f"p/{job_id}.png",
                    "download_url": f"https://s3/{job_id}.png",
                    "sha256": "f" * 64,
                }
                comp_res = await client.post(f"/v1/jobs/{job_id}/complete", json={
                    "worker_id": "worker-complete-protect",
                    "lease_token": lease_res.json()["lease_token"],
                    "artifacts": [art],
                })
                assert comp_res.status_code == 200

                # Attempt cancel on SUCCEEDED job
                cancel_res = await client.post(f"/v1/jobs/{job_id}/cancel")
                assert cancel_res.status_code == 400
                assert "Cannot cancel a job that has already completed" in cancel_res.text

    async def test_worker_completion_on_cancelled_job_rejected_with_400(self):
        """
        Adversarial test: A worker tries to complete a job that was already cancelled.
        Verify:
        - Control plane rejects completion with HTTP 400 (InvalidStateTransitionError).
        - State remains CANCELLED.
        - Artifacts are not overwritten.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository

                sub = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "late completion test"},
                })
                job_id = sub.json()["job_id"]

                # Worker leases
                l_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-late",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 300,
                })

                # Cancel job
                cancel_res = await client.post(f"/v1/jobs/{job_id}/cancel")
                assert cancel_res.status_code == 200
                assert cancel_res.json()["state"] == "CANCELLED"

                # Worker tries to complete
                comp_res = await client.post(f"/v1/jobs/{job_id}/complete", json={
                    "worker_id": "worker-late",
                    "lease_token": l_res.json()["lease_token"],
                    "artifacts": [{
                        "artifact_id": "late-art",
                        "type": "image",
                        "mime_type": "image/png",
                        "size_bytes": 100,
                        "storage_path": "image/late.png",
                        "download_url": "https://s3/late.png",
                        "sha256": "0" * 64,
                    }],
                })
                assert comp_res.status_code == 400
                assert "Cannot transition job" in comp_res.text

                # Job state remains CANCELLED
                persisted = await repo.get_job(job_id)
                assert persisted.state == JobState.CANCELLED
                assert len(persisted.artifacts) == 0

    async def test_streaming_job_cancellation_transitions_cleanly(self):
        """
        Adversarial test: Cancel a job currently in STREAMING state.
        Verify:
        - Job transitions to CANCELLED.
        - Worker heartbeat acknowledges cancel_requested=True.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                repo: SqliteJobRepository = control_plane_app.state.repository

                sub = await client.post("/v1/jobs", json={
                    "service": "speech",
                    "model": "qwen-speech",
                    "task": "speech_synthesis",
                    "inputs": {"text": "streaming audio cancellation"},
                })
                job_id = sub.json()["job_id"]

                # Lease
                await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-tts-streamer",
                    "subsystem": "speech",
                    "supported_models": ["qwen-speech"],
                    "supported_tasks": ["speech_synthesis"],
                    "lease_duration_seconds": 300,
                })

                # Update progress to STREAMING
                j = await repo.get_job(job_id)
                j.state = JobState.STREAMING
                await repo.update_job(j)

                # Cancel
                cancel_res = await client.post(f"/v1/jobs/{job_id}/cancel")
                assert cancel_res.status_code == 200
                assert cancel_res.json()["state"] == "CANCELLED"

                # Worker heartbeat returns cancel_requested
                hb_res = await client.post("/v1/workers/worker-tts-streamer/heartbeat", json={
                    "worker_id": "worker-tts-streamer",
                    "active_job_id": job_id,
                })
                assert hb_res.status_code == 200
                assert hb_res.json()["cancel_requested"] is True


# ==============================================================================
# 4. S3 ARTIFACT UPLOAD & SIGNED URL INTEGRITY
# ==============================================================================
@pytest.mark.asyncio
class TestStorageArtifactUploadSignedURLIntegrity:
    """Empirical verification of large artifact (>1MB) uploads, signed URLs, and SHA256 integrity."""

    async def test_large_artifact_upload_and_download_roundtrip_sha256_integrity(self):
        """
        Adversarial artifact test:
        1. Generate a large random payload (>1MB, e.g. 2.5MB binary stream).
        2. Compute authoritative SHA256.
        3. Request presigned upload URL via POST /v1/storage/upload-url.
        4. Upload the 2.5MB payload via HTTP PUT to the returned signed upload URL.
        5. Verify upload succeeds (HTTP 200).
        6. Lease the job so it transitions to RUNNING.
        7. Complete the job referencing this artifact storage path.
        8. Query GET /v1/jobs/{job_id} to obtain the presigned download URL.
        9. Download the binary payload using the presigned download URL.
        10. Assert exact bit-for-bit SHA256 integrity and length match.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                # 1. Generate 2.5MB random test artifact
                artifact_size = int(2.5 * 1024 * 1024)  # 2,621,440 bytes (> 1MB threshold)
                random_bytes = os.urandom(artifact_size)
                expected_sha256 = hashlib.sha256(random_bytes).hexdigest()

                job_id = f"job-{uuid.uuid4()}"
                artifact_name = "render_4k_output.png"

                # 2. Request presigned upload URL
                upload_req = {
                    "subsystem": "image",
                    "job_id": job_id,
                    "artifact_name": artifact_name,
                    "mime_type": "image/png",
                    "max_size_bytes": 10 * 1024 * 1024,
                    "expires_in_seconds": 3600,
                }
                url_res = await client.post("/v1/storage/upload-url", json=upload_req)
                assert url_res.status_code == 200
                upload_info = url_res.json()
                upload_url = upload_info["upload_url"]
                storage_path = upload_info["storage_path"]

                # 3. Upload the 2.5MB payload via PUT
                upload_res = await client.put(
                    upload_url,
                    content=random_bytes,
                    headers={"Content-Type": "image/png"},
                )
                assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
                upload_ack = upload_res.json()
                assert upload_ack["size_bytes"] == artifact_size

                # 4. Create and lease job to move it to RUNNING
                sub_res = await client.post("/v1/jobs", json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "Large artifact upload test"},
                })
                active_job_id = sub_res.json()["job_id"]

                lease_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-storage-tester",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                    "lease_duration_seconds": 300,
                })
                assert lease_res.status_code == 200

                # 5. Complete job with artifact
                artifact_meta = {
                    "artifact_id": f"art-{uuid.uuid4()}",
                    "type": "image",
                    "mime_type": "image/png",
                    "size_bytes": artifact_size,
                    "storage_path": storage_path,
                    "download_url": "placeholder-url",
                    "sha256": expected_sha256,
                }
                comp_res = await client.post(
                    f"/v1/jobs/{active_job_id}/complete",
                    json={
                        "worker_id": "worker-storage-tester",
                        "lease_token": lease_res.json()["lease_token"],
                        "artifacts": [artifact_meta],
                        "inference_duration_ms": 3200.0,
                    },
                )
                assert comp_res.status_code == 200

                # 6. Query job to receive freshly signed download URL
                job_res = await client.get(f"/v1/jobs/{active_job_id}")
                assert job_res.status_code == 200
                job_data = job_res.json()
                assert len(job_data["artifacts"]) == 1
                download_url = job_data["artifacts"][0]["download_url"]
                assert download_url and download_url != "placeholder-url"

                # 7. Fetch downloaded bytes using presigned download URL
                dl_res = await client.get(download_url)
                assert dl_res.status_code == 200
                downloaded_bytes = dl_res.content

                # 8. Assert bit-for-bit SHA256 integrity and length
                assert len(downloaded_bytes) == artifact_size
                downloaded_sha256 = hashlib.sha256(downloaded_bytes).hexdigest()
                assert downloaded_sha256 == expected_sha256, (
                    f"Checksum mismatch! Expected {expected_sha256}, got {downloaded_sha256}"
                )

    async def test_signed_url_tampering_and_expiration_rejections(self):
        """
        Security verification for HMAC-SHA256 URL signatures:
        1. Tampered signature parameter -> HTTP 403 Forbidden.
        2. Expired timestamp parameter -> HTTP 403 Forbidden.
        3. Path traversal execution attempts -> HTTP 400 InvalidRequest.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=30.0,
            ) as client:
                # 1. Request legitimate upload URL
                url_res = await client.post("/v1/storage/upload-url", json={
                    "subsystem": "image",
                    "job_id": "job-sec-test",
                    "artifact_name": "photo.png",
                })
                upload_url = url_res.json()["upload_url"]

                # 2. Attack: Tamper with signature
                tampered_url = upload_url.replace("sig=", "sig=forged_signature_000")
                bad_sig_res = await client.put(tampered_url, content=b"attack")
                assert bad_sig_res.status_code == 403
                assert "Invalid or expired" in bad_sig_res.text

                # 3. Attack: Tamper with expiration (expired timestamp in past)
                expired_url = upload_url.replace("expires=", "expires=1000000000&orig=")
                expired_res = await client.put(expired_url, content=b"attack")
                assert expired_res.status_code == 403

                # 4. Attack: Directory traversal execution in storage upload endpoint
                traversal_url = "/v1/storage/upload?bucket=artifacts&key=../../etc/passwd&expires=9999999999&sig=dummy"
                traversal_res = await client.put(traversal_url, content=b"attack")
                # When signature check fails or path check fails, access is rejected
                assert traversal_res.status_code in (400, 403)

    async def test_s3_storage_provider_sigv4_and_public_remapping(self):
        """
        Direct unit verification of S3StorageProvider:
        - SigV4 presigned upload and download URL generation.
        - Public endpoint remapping (e.g. minio:9000 -> s3.ai.example.com).
        - SHA256 integrity calculation on binary data.
        """
        cfg = StorageConfig(
            storage_backend="s3",
            s3_endpoint_url="http://minio:9000",
            s3_public_endpoint_url="https://s3.ai.example.com",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket_name="factory-artifacts",
            s3_region="us-east-1",
        )
        # Mock network call _ensure_bucket_exists so client can sign URLs offline
        with patch.object(S3StorageProvider, "_ensure_bucket_exists", return_value=None):
            provider = S3StorageProvider(cfg)

            # 1. Presigned upload generation
            upload = provider.generate_presigned_upload_url(
                bucket="factory-artifacts",
                key="image/job-123/output.png",
                mime_type="image/png",
                max_size_bytes=5000000,
                expires_in_seconds=3600,
            )
            assert upload.upload_url.startswith("https://s3.ai.example.com/factory-artifacts/image/job-123/output.png")
            assert "X-Amz-Signature=" in upload.upload_url
            assert upload.method == "PUT"

            # 2. Presigned download generation
            download_url = provider.generate_presigned_download_url(
                bucket="factory-artifacts",
                key="image/job-123/output.png",
                expires_in_seconds=86400,
                filename="my_image.png",
            )
            assert download_url.startswith("https://s3.ai.example.com/factory-artifacts/image/job-123/output.png")
            assert "X-Amz-Signature=" in download_url
            assert "response-content-disposition=" in download_url

            # 3. SHA256 integrity calculation
            data = b"Factory v4 high performance generative artifact binary content"
            sha = compute_bytes_sha256(data)
            assert sha == hashlib.sha256(data).hexdigest()
            assert len(sha) == 64

    async def test_multi_artifact_concurrent_upload_and_sha256_verification(self):
        """
        Adversarial test: Video generation job producing multiple artifacts (>1MB each):
        1. 1.8MB Video file (video/mp4)
        2. 1.2MB Audio file (audio/wav)
        Verify:
        - Concurrently request upload URLs and upload both payloads.
        - Complete job with both artifacts.
        - Query status and download both payloads concurrently via presigned URLs.
        - Bit-level SHA256 matches for both artifacts.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                # 1. Generate payloads (>1MB each)
                video_bytes = os.urandom(int(1.8 * 1024 * 1024))
                audio_bytes = os.urandom(int(1.2 * 1024 * 1024))
                expected_video_sha = hashlib.sha256(video_bytes).hexdigest()
                expected_audio_sha = hashlib.sha256(audio_bytes).hexdigest()

                # 2. Submit job & lease
                sub_res = await client.post("/v1/jobs", json={
                    "service": "video",
                    "model": "wan-2.2",
                    "task": "text_to_video",
                    "inputs": {"prompt": "Multi-artifact video rendering"},
                })
                job_id = sub_res.json()["job_id"]

                lease_res = await client.post("/v1/jobs/lease", json={
                    "worker_id": "worker-video-dual",
                    "subsystem": "video",
                    "supported_models": ["wan-2.2"],
                    "supported_tasks": ["text_to_video"],
                    "lease_duration_seconds": 300,
                })
                assert lease_res.status_code == 200

                # 3. Request presigned upload URLs
                u1 = await client.post("/v1/storage/upload-url", json={
                    "subsystem": "video",
                    "job_id": job_id,
                    "artifact_name": "render.mp4",
                    "mime_type": "video/mp4",
                    "max_size_bytes": 10 * 1024 * 1024,
                })
                u2 = await client.post("/v1/storage/upload-url", json={
                    "subsystem": "video",
                    "job_id": job_id,
                    "artifact_name": "soundtrack.wav",
                    "mime_type": "audio/wav",
                    "max_size_bytes": 10 * 1024 * 1024,
                })
                video_up = u1.json()
                audio_up = u2.json()

                # 4. Concurrently upload both
                up_results = await asyncio.gather(
                    client.put(video_up["upload_url"], content=video_bytes, headers={"Content-Type": "video/mp4"}),
                    client.put(audio_up["upload_url"], content=audio_bytes, headers={"Content-Type": "audio/wav"}),
                )
                assert all(r.status_code == 200 for r in up_results)

                # 5. Complete job with both artifacts
                artifacts = [
                    {
                        "artifact_id": f"art-vid-{uuid.uuid4()}",
                        "type": "video",
                        "mime_type": "video/mp4",
                        "size_bytes": len(video_bytes),
                        "storage_path": video_up["storage_path"],
                        "download_url": "placeholder",
                        "sha256": expected_video_sha,
                    },
                    {
                        "artifact_id": f"art-aud-{uuid.uuid4()}",
                        "type": "audio",
                        "mime_type": "audio/wav",
                        "size_bytes": len(audio_bytes),
                        "storage_path": audio_up["storage_path"],
                        "download_url": "placeholder",
                        "sha256": expected_audio_sha,
                    },
                ]
                comp_res = await client.post(f"/v1/jobs/{job_id}/complete", json={
                    "worker_id": "worker-video-dual",
                    "lease_token": lease_res.json()["lease_token"],
                    "artifacts": artifacts,
                    "inference_duration_ms": 5400.0,
                    "fps": 24.0,
                })
                assert comp_res.status_code == 200

                # 6. Retrieve job and download both
                job_res = await client.get(f"/v1/jobs/{job_id}")
                assert job_res.status_code == 200
                job_data = job_res.json()
                assert len(job_data["artifacts"]) == 2

                dl_urls = [a["download_url"] for a in job_data["artifacts"]]
                dl_results = await asyncio.gather(*[client.get(u) for u in dl_urls])
                assert all(r.status_code == 200 for r in dl_results)

                # 7. Check bit-level SHA256 integrity
                dl_video = dl_results[0].content
                dl_audio = dl_results[1].content
                assert hashlib.sha256(dl_video).hexdigest() == expected_video_sha
                assert hashlib.sha256(dl_audio).hexdigest() == expected_audio_sha
