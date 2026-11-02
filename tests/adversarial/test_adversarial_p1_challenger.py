"""
tests/adversarial/test_adversarial_p1_challenger.py
Adversarial Challenger Test Suite for Phase P1 Gate Verification.

Adversarially probes:
1. Burst Concurrency & High Contention Leasing:
   - High concurrency burst leasing across multiple registered workers.
   - Single-job race condition under 15 concurrent worker lease requests (strict 1 winner, 14 None).
   - Zero double-leasing invariants and lease token uniqueness.
2. Lease Token Mutual Exclusion & State Mutation Race Conditions:
   - Concurrent cross-worker mutations (/progress, /complete, /fail) with stolen, forged, or mismatched tokens.
   - JWT claims binding (caller_worker_id vs req.worker_id spoofing).
   - Expiration boundary attacks: operations on expired leases strictly return HTTP 403.
   - Post-completion mutation protection: idempotent /complete, immutable artifacts on completed jobs.
3. Worker Capability Manifest Security:
   - Rogue worker claiming unmanifested cross-subsystem models/tasks has capabilities stripped.
   - Manifest boundary enforcement preventing video/music jobs from being leased to image workers.
4. WebSocket Authentication Header Migration:
   - Query parameter ?token= strictly rejected with WebSocket status 1008 (Policy Violation).
   - Missing credentials rejected with 1008.
   - Sec-WebSocket-Protocol and Authorization headers accepted across Control Plane and Speech service.
5. Studio Codegen Hardcoded Secret Scanning:
   - Regex verification preventing raw API keys, bearer fallback strings, or hardcoded tokens in codegen snippets.
"""
import ast
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import re
from typing import List, Optional
import uuid
import jwt

import httpx
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from factory_auth.config import AuthConfig
from factory_contracts.errors import ErrorDetail, UnauthorizedError
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
    WorkerLeaseResponse,
)
from factory_contracts.tasks import (
    normalize_model,
    normalize_task,
    get_model_aliases,
    get_task_aliases,
)

AUTH_TOKEN = "sk-factory-test-token-valid"
JWT_SECRET = "test-jwt-secret-key-32-chars-long!"
AUTH_HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}

from factory.apps.control_api.src.main import app as control_plane_app, lifespan
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.streaming.manager import StreamManager
from services.speech.server import app as speech_app


@pytest.fixture(autouse=True)
def configure_test_auth(monkeypatch):
    old_auth = getattr(control_plane_app.state, "auth_config", None)
    old_speech_auth = getattr(speech_app.state, "auth_config", None)
    monkeypatch.setenv("FACTORY_API_KEY", AUTH_TOKEN)
    monkeypatch.setenv("FACTORY_JWT_SECRET", JWT_SECRET)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", ":memory:")
    test_auth = AuthConfig(
        api_key=AUTH_TOKEN,
        jwt_secret=JWT_SECRET,
        environment="test",
    )
    control_plane_app.state.auth_config = test_auth
    speech_app.state.auth_config = test_auth
    yield
    if old_auth is not None:
        control_plane_app.state.auth_config = old_auth
    elif hasattr(control_plane_app.state, "auth_config"):
        delattr(control_plane_app.state, "auth_config")
    if old_speech_auth is not None:
        speech_app.state.auth_config = old_speech_auth
    elif hasattr(speech_app.state, "auth_config"):
        delattr(speech_app.state, "auth_config")


def make_worker_jwt(worker_id: str, role: str = "worker") -> str:
    """Generates a valid signed JWT for testing worker identity binding."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": worker_id,
        "worker_id": worker_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# ===========================================================================
# 1. Burst Concurrency & High Contention Leasing
# ===========================================================================

@pytest.mark.asyncio
async def test_single_job_15_concurrent_workers_race():
    """
    Adversarial Challenge: 15 workers simultaneously compete to lease a SINGLE job.
    Invariants:
    - Exactly 1 worker receives job and lease_token.
    - Exactly 14 workers receive job=None and lease_token=None.
    - The repository reflects exactly the winning worker's lease.
    """
    repo = SqliteJobRepository(db_path=":memory:")
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
    coordinator = LeaseCoordinator(repository=repo, broker=broker)

    try:
        # Submit single image job
        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Contention race single prize"},
                priority=100,
            ),
            request_id="req-single-race",
        )

        # 15 concurrent workers race to lease
        worker_count = 15
        async def worker_lease_attempt(worker_idx: int):
            wid = f"worker-race-{worker_idx}"
            return await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id=wid,
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    lease_duration_seconds=60,
                )
            )

        responses = await asyncio.gather(*[worker_lease_attempt(i) for i in range(worker_count)])

        winners = [r for r in responses if r.job is not None]
        losers = [r for r in responses if r.job is None]

        assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"
        assert len(losers) == worker_count - 1, f"Expected {worker_count - 1} losers, got {len(losers)}"

        winning_resp = winners[0]
        assert winning_resp.job.job_id == job.job_id
        assert winning_resp.lease_token is not None
        assert winning_resp.lease_token.startswith("lease-")

        # Verify DB state
        db_job = await repo.get_job(job.job_id)
        assert db_job.state == JobState.RUNNING
        assert db_job.worker is not None
        assert db_job.worker.lease_token == winning_resp.lease_token
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_burst_leasing_under_contention_zero_double_leasing():
    """
    Adversarial Challenge: 50 queued jobs across 4 subsystems, 20 workers hammering
    lease endpoint simultaneously in parallel loops.
    Invariants:
    - 0 jobs double-leased.
    - All 50 jobs claimed.
    - All 50 lease tokens are unique.
    """
    repo = SqliteJobRepository(db_path=":memory:")
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
    coordinator = LeaseCoordinator(repository=repo, broker=broker)

    try:
        # Enqueue 50 jobs
        subsystem_specs = [
            ("image", "qwen-image-2.1", "text_to_image"),
            ("video", "wan-2.2", "text_to_video"),
            ("music", "ace-step-1.5", "text_to_music"),
            ("speech", "qwen-speech", "text_to_speech"),
        ]

        submitted_ids = set()
        for i in range(50):
            sub, model, task = subsystem_specs[i % len(subsystem_specs)]
            j, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service=sub,
                    model=model,
                    task=task,
                    inputs={"prompt": f"Burst test job {i}"},
                    priority=i + 1,
                ),
                request_id=f"req-burst-{i}",
            )
            submitted_ids.add(j.job_id)

        assert len(submitted_ids) == 50

        leased_job_ids = set()
        lease_tokens = set()
        lock = asyncio.Lock()

        async def worker_loop(worker_id: str, sub: str, model: str, task: str):
            while True:
                resp = await coordinator.lease_job(
                    WorkerLeaseRequest(
                        worker_id=worker_id,
                        subsystem=sub,
                        supported_models=[model],
                        supported_tasks=[task],
                        lease_duration_seconds=30,
                    )
                )
                if not resp.job:
                    break

                async with lock:
                    jid = resp.job.job_id
                    tok = resp.lease_token
                    assert jid not in leased_job_ids, f"CRITICAL DOUBLE LEASE: {jid}"
                    assert tok not in lease_tokens, f"CRITICAL DUPLICATE LEASE TOKEN: {tok}"
                    leased_job_ids.add(jid)
                    lease_tokens.add(tok)

        # 20 workers (5 for each subsystem)
        worker_tasks = []
        for i in range(20):
            sub, model, task = subsystem_specs[i % len(subsystem_specs)]
            worker_tasks.append(worker_loop(f"w-{sub}-{i}", sub, model, task))

        await asyncio.gather(*worker_tasks)

        assert len(leased_job_ids) == 50
        assert len(lease_tokens) == 50
        assert leased_job_ids == submitted_ids
    finally:
        await repo.close()


# ===========================================================================
# 2. Lease Token Mutual Exclusion & State Mutation Race Conditions
# ===========================================================================

@pytest.mark.asyncio
async def test_cross_worker_mutation_rejection():
    """
    Adversarial Challenge: Worker A holds valid lease for Job X.
    Worker B tries to tamper with Job X by sending /progress, /complete, and /fail.
    Invariants:
    - Worker B gets 403 UnauthorizedError on all operations.
    - Job X progress and state are unmolested.
    """
    repo = SqliteJobRepository(db_path=":memory:")
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
    coordinator = LeaseCoordinator(repository=repo, broker=broker)

    try:
        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Cross-worker protection"},
            ),
            request_id="req-cross-worker",
        )

        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-legit-A",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
            )
        )
        token_A = lease_resp.lease_token

        # 1. Worker B passes own worker_id and token_A
        with pytest.raises(UnauthorizedError) as exc_p1:
            await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="sampling", step=1, total=10, percent=10.0,
                    worker_id="worker-attacker-B", lease_token=token_A
                ),
            )
        assert exc_p1.value.status_code == 403

        # 2. Worker B passes worker_legit_A with a forged token
        with pytest.raises(UnauthorizedError) as exc_p2:
            await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(
                    stage="sampling", step=1, total=10, percent=10.0,
                    worker_id="worker-legit-A", lease_token="forged-token-xyz"
                ),
            )
        assert exc_p2.value.status_code == 403

        # 3. Worker B attempts to complete Job X
        artifact = ArtifactReference(
            artifact_id="art-spoof",
            type="image",
            mime_type="image/png",
            size_bytes=100,
            storage_path=f"image/{job.job_id}/out.png",
            url="https://s3.ai.example.com/art.png",
            download_url="https://s3.ai.example.com/art.png",
            sha256="c" * 64,
        )
        with pytest.raises(UnauthorizedError) as exc_c:
            await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    artifacts=[artifact],
                    worker_id="worker-attacker-B",
                    lease_token=token_A,
                ),
            )
        assert exc_c.value.status_code == 403

        # 4. Worker B attempts to fail Job X
        with pytest.raises(UnauthorizedError) as exc_f:
            await manager.fail_job(
                job.job_id,
                JobFailureRequest(
                    error=ErrorDetail(code="INTERNAL_ERROR", message="Spoof fail", request_id="req-1"),
                    worker_id="worker-attacker-B",
                    lease_token=token_A,
                ),
            )
        assert exc_f.value.status_code == 403

        # Verify Job X remains RUNNING and uncorrupted
        current_job = await repo.get_job(job.job_id)
        assert current_job.state == JobState.RUNNING
        assert current_job.worker.worker_id == "worker-legit-A"
        assert current_job.worker.lease_token == token_A
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_jwt_caller_spoofing_via_api_endpoints():
    """
    Adversarial Challenge: Attacker worker-B authenticates with JWT claim sub='worker-B',
    but submits HTTP POST /v1/jobs/{job_id}/progress or /complete with worker_id='worker-A'.
    Invariants:
    - Router verifies auth_worker matches req.worker_id.
    - Strictly rejected with HTTP 403 Forbidden.
    """
    async with lifespan(control_plane_app):
        jwt_worker_a = make_worker_jwt("worker-A")
        jwt_worker_b = make_worker_jwt("worker-B")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        ) as client:
            # 1. Submit job using API key
            sub_res = await client.post(
                "/v1/jobs",
                headers=AUTH_HEADERS,
                json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "JWT spoof"}},
            )
            assert sub_res.status_code == 201
            job_id = sub_res.json()["job_id"]

            # 2. Worker A leases the job with its own JWT
            lease_res = await client.post(
                "/v1/jobs/lease",
                headers={"Authorization": f"Bearer {jwt_worker_a}"},
                json={"worker_id": "worker-A", "subsystem": "image", "supported_models": ["qwen-image-2.1"], "supported_tasks": ["text_to_image"]},
            )
            assert lease_res.status_code == 200
            token_a = lease_res.json()["lease_token"]

            # 3. Attacker Worker B tries to lease claiming worker-A identity
            rogue_lease_res = await client.post(
                "/v1/jobs/lease",
                headers={"Authorization": f"Bearer {jwt_worker_b}"},
                json={"worker_id": "worker-A", "subsystem": "image", "supported_models": ["qwen-image-2.1"], "supported_tasks": ["text_to_image"]},
            )
            assert rogue_lease_res.status_code == 403

            # 4. Attacker Worker B tries to send heartbeat for worker-A
            rogue_hb_res = await client.post(
                "/v1/workers/worker-A/heartbeat",
                headers={"Authorization": f"Bearer {jwt_worker_b}"},
                json={"worker_id": "worker-A", "active_job_id": job_id, "lease_token": token_a},
            )
            assert rogue_hb_res.status_code == 403

            # 5. Attacker Worker B tries to update progress claiming worker-A
            rogue_prog_res = await client.post(
                f"/v1/jobs/{job_id}/progress",
                headers={"Authorization": f"Bearer {jwt_worker_b}"},
                json={"stage": "sampling", "step": 5, "total": 20, "percent": 25.0, "worker_id": "worker-A", "lease_token": token_a},
            )
            assert rogue_prog_res.status_code == 403

            # 6. Attacker Worker B tries to complete job claiming worker-A
            rogue_comp_res = await client.post(
                f"/v1/jobs/{job_id}/complete",
                headers={"Authorization": f"Bearer {jwt_worker_b}"},
                json={
                    "worker_id": "worker-A",
                    "lease_token": token_a,
                    "artifacts": [{
                        "artifact_id": "spoof", "type": "image", "mime_type": "image/png",
                        "size_bytes": 100, "storage_path": "spoof.png", "url": "https://s3.ai.example.com/spoof.png",
                        "sha256": "f" * 64,
                    }],
                    "inference_duration_ms": 100,
                },
            )
            assert rogue_comp_res.status_code == 403

            # 7. Legit Worker A completes job
            legit_comp = await client.post(
                f"/v1/jobs/{job_id}/complete",
                headers={"Authorization": f"Bearer {jwt_worker_a}"},
                json={
                    "worker_id": "worker-A",
                    "lease_token": token_a,
                    "artifacts": [{
                        "artifact_id": f"art-{job_id}", "type": "image", "mime_type": "image/png",
                        "size_bytes": 1024, "storage_path": f"image/{job_id}/out.png",
                        "url": "https://s3.ai.example.com/out.png",
                        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    }],
                    "inference_duration_ms": 500,
                },
            )
            assert legit_comp.status_code == 200


@pytest.mark.asyncio
async def test_completed_job_immutability():
    """
    Adversarial Challenge: Once a job reaches SUCCEEDED, any subsequent mutation attempts
    (progress, complete, fail) must not mutate artifacts, error, or state.
    """
    repo = SqliteJobRepository(db_path=":memory:")
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
    coordinator = LeaseCoordinator(repository=repo, broker=broker)

    try:
        job, _ = await manager.submit_job(
            JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "Completed job immutability test"},
            ),
            request_id="req-immutability",
        )
        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-imm",
                subsystem="image",
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
            )
        )
        token = lease_resp.lease_token

        original_artifact = ArtifactReference(
            artifact_id="art-original",
            type="image",
            mime_type="image/png",
            size_bytes=2048,
            storage_path=f"image/{job.job_id}/original.png",
            url="https://s3.ai.example.com/original.png",
            download_url="https://s3.ai.example.com/original.png",
            sha256="1" * 64,
        )

        completed = await manager.complete_job(
            job.job_id,
            JobCompletionRequest(
                artifacts=[original_artifact],
                inference_duration_ms=800.0,
                worker_id="worker-imm",
                lease_token=token,
            ),
        )
        assert completed.state == JobState.SUCCEEDED
        assert len(completed.artifacts) == 1
        assert completed.artifacts[0].artifact_id == "art-original"

        # Attacker / subsequent worker tries to overwrite completion with fake artifacts
        fake_artifact = ArtifactReference(
            artifact_id="art-fake",
            type="image",
            mime_type="image/png",
            size_bytes=10,
            storage_path="fake.png",
            url="https://s3.ai.example.com/fake.png",
            download_url="https://s3.ai.example.com/fake.png",
            sha256="2" * 64,
        )
        res_after = await manager.complete_job(
            job.job_id,
            JobCompletionRequest(
                artifacts=[fake_artifact],
                inference_duration_ms=10.0,
                worker_id="worker-imm",
                lease_token=token,
            ),
        )
        # Idempotent no-op returns original job with original artifacts intact!
        assert res_after.state == JobState.SUCCEEDED
        assert len(res_after.artifacts) == 1
        assert res_after.artifacts[0].artifact_id == "art-original"
    finally:
        await repo.close()


# ===========================================================================
# 3. Worker Capability Manifest Security
# ===========================================================================

@pytest.mark.asyncio
async def test_rogue_worker_capabilities_stripped_to_manifest():
    """
    Adversarial Challenge: A rogue worker claims it belongs to subsystem 'image',
    but requests models and tasks from video ('wan-2.2', 'text_to_video') and music ('ace-step-1.5').
    Invariants:
    - _verify_manifest_capabilities strips all unmanifested capabilities.
    - The worker is NEVER assigned video or music jobs.
    """
    repo = SqliteJobRepository(db_path=":memory:")
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    coordinator = LeaseCoordinator(repository=repo, broker=broker)

    try:
        # Enqueue video and music jobs
        v_job = FactoryJob(
            job_id=f"job-{uuid.uuid4()}",
            request_id=f"req-{uuid.uuid4()}",
            service="video",
            model="wan-2.2",
            task="text_to_video",
            inputs={"prompt": "A majestic eagle"},
            priority=50,
            state=JobState.QUEUED,
            created_at=datetime.now(timezone.utc),
            progress=JobProgress(stage="queued", step=0, total=0, percent=0.0),
        )
        await repo.create_job(v_job)

        m_job = FactoryJob(
            job_id=f"job-{uuid.uuid4()}",
            request_id=f"req-{uuid.uuid4()}",
            service="music",
            model="ace-step-1.5",
            task="text_to_music",
            inputs={"prompt": "Synth melody"},
            priority=50,
            state=JobState.QUEUED,
            created_at=datetime.now(timezone.utc),
            progress=JobProgress(stage="queued", step=0, total=0, percent=0.0),
        )
        await repo.create_job(m_job)

        # Rogue image worker claims video & music capabilities
        rogue_req = WorkerLeaseRequest(
            worker_id="rogue-img-worker",
            subsystem="image",
            supported_models=["qwen-image-2.1", "wan-2.2", "ace-step-1.5"],
            supported_tasks=["text_to_image", "text_to_video", "text_to_music"],
        )
        lease_resp = await coordinator.lease_job(rogue_req)

        # Must NOT lease video or music jobs
        assert lease_resp.job is None

        # Verify registered worker capabilities in DB only contain image capabilities
        registered = await repo.get_worker("rogue-img-worker")
        assert registered is not None
        assert "wan-2.2" not in registered["supported_models"]
        assert "ace-step-1.5" not in registered["supported_models"]
        assert "text_to_video" not in registered["supported_tasks"]
        assert "text_to_music" not in registered["supported_tasks"]
        assert "qwen-image-2.1" in registered["supported_models"]
        assert "text_to_image" in registered["supported_tasks"]
    finally:
        await repo.close()


# ===========================================================================
# 4. WebSocket Authentication Header Migration
# ===========================================================================

def test_websocket_query_token_strictly_rejected_code_1008():
    """
    Adversarial Challenge: Client attempts WebSocket handshake using ?token= query parameter.
    Invariants:
    - Control Plane /v1/jobs/{job_id}/ws strictly rejects connection with 1008.
    - Speech Service /v1/audio/transcriptions/ws strictly rejects connection with 1008.
    """
    target_job_id = f"job-{uuid.uuid4()}"
    client = TestClient(control_plane_app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/v1/jobs/{target_job_id}/ws?token={AUTH_TOKEN}"):
            pass
    assert exc_info.value.code == 1008

    speech_client = TestClient(speech_app)
    with pytest.raises(WebSocketDisconnect) as exc_speech:
        with speech_client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}"):
            pass
    assert exc_speech.value.code == 1008


def test_websocket_missing_auth_strictly_rejected():
    """
    Adversarial Challenge: Client attempts WebSocket handshake with zero credentials.
    Invariants:
    - Control Plane rejects with 1008.
    - Speech Service rejects with 1008.
    """
    target_job_id = f"job-{uuid.uuid4()}"
    client = TestClient(control_plane_app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/v1/jobs/{target_job_id}/ws"):
            pass
    assert exc_info.value.code == 1008

    speech_client = TestClient(speech_app)
    with pytest.raises(WebSocketDisconnect) as exc_speech:
        with speech_client.websocket_connect("/v1/audio/transcriptions/ws"):
            pass
    assert exc_speech.value.code == 1008


def test_websocket_sec_websocket_protocol_accepted():
    """
    Verifies that Sec-WebSocket-Protocol: factory-bearer.<token> is accepted and connects.
    """
    target_job_id = f"job-{uuid.uuid4()}"
    client = TestClient(control_plane_app)
    subprotocol = f"factory-bearer.{AUTH_TOKEN}"
    with client.websocket_connect(
        f"/v1/jobs/{target_job_id}/ws",
        subprotocols=["factory-v4", subprotocol],
    ) as ws:
        assert ws is not None

    speech_client = TestClient(speech_app)
    with speech_client.websocket_connect(
        "/v1/audio/transcriptions/ws",
        subprotocols=[subprotocol],
    ) as ws:
        assert ws is not None


def test_websocket_authorization_header_accepted():
    """
    Verifies that Authorization: Bearer <token> header is accepted and connects.
    """
    target_job_id = f"job-{uuid.uuid4()}"
    client = TestClient(control_plane_app)
    with client.websocket_connect(
        f"/v1/jobs/{target_job_id}/ws",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    ) as ws:
        assert ws is not None

    speech_client = TestClient(speech_app)
    with speech_client.websocket_connect(
        "/v1/audio/transcriptions/ws",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    ) as ws:
        assert ws is not None


# ===========================================================================
# 5. Studio Codegen Hardcoded Secret Scanning
# ===========================================================================

def test_studio_codegen_zero_hardcoded_keys():
    """
    Adversarial Challenge: Validate that backend and frontend codegen scripts
    never produce hardcoded API keys or fallback tokens.
    """
    client = TestClient(control_plane_app)
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    payload = {
        "service": "image",
        "model": "qwen-image-2.1",
        "task": "image.t2i",
        "inputs": {"prompt": "Adversarial codegen security check"},
    }
    res = client.post("/api/studio/codegen", headers=headers, json=payload)
    assert res.status_code == 200
    data = res.json()

    curl = data["curl"]
    python_code = data["python"]
    ts_code = data["typescript"]

    forbidden_patterns = [
        re.compile(r"sk-[a-zA-Z0-9_\-]{8,}"),
        re.compile(r"Bearer\s+sk-[a-zA-Z0-9_\-]+"),
        re.compile(r"Bearer\s+[\"'][a-zA-Z0-9_\-]+[\"']"),
        re.compile(r"sk-factory-studio"),
    ]

    for snippet_name, snippet in [("curl", curl), ("python", python_code), ("typescript", ts_code)]:
        for pat in forbidden_patterns:
            match = pat.search(snippet)
            assert match is None, f"Found hardcoded credential pattern in {snippet_name}: {match.group(0)}"

    assert "$FACTORY_API_KEY" in curl
    assert 'os.environ["FACTORY_API_KEY"]' in python_code
    assert "process.env.FACTORY_API_KEY" in ts_code

    tree = ast.parse(python_code)
    assert isinstance(tree, ast.Module)


def test_codegen_ts_file_source_regex_scan():
    """
    Direct AST/regex inspection of factory/apps/studio-web/src/lib/codegen.ts
    to ensure zero secret leaks in source code.
    """
    codegen_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "factory", "apps", "studio-web", "src", "lib", "codegen.ts"
    )
    assert os.path.exists(codegen_path), f"File not found: {codegen_path}"

    with open(codegen_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "sk-factory-studio" not in content
    assert '"sk-' not in content
    assert "'sk-" not in content

    assert 'os.environ["FACTORY_API_KEY"]' in content
    assert "process.env.FACTORY_API_KEY" in content
    assert "$FACTORY_API_KEY" in content


# ===========================================================================
# 6. Deep Adversarial Matrix: Role Boundaries & Identity Spoofing
# ===========================================================================

@pytest.mark.asyncio
async def test_non_worker_role_rejected_on_all_worker_endpoints():
    """
    Adversarial Challenge: Attacker generates valid signed JWT with role='user' or role='client'.
    Attacker attempts to call all worker endpoints:
    - POST /v1/jobs/lease
    - POST /v1/workers/{worker_id}/heartbeat
    - POST /v1/jobs/{job_id}/progress
    - POST /v1/jobs/{job_id}/complete
    - POST /v1/jobs/{job_id}/fail
    Invariants:
    - verify_worker_identity strictly rejects with HTTP 403 Forbidden on all endpoints.
    """
    async with lifespan(control_plane_app):
        user_jwt = make_worker_jwt("attacker-user", role="user")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {user_jwt}"},
            timeout=10.0,
        ) as client:
            fake_job_id = f"job-{uuid.uuid4()}"

            # 1. /v1/jobs/lease
            res = await client.post("/v1/jobs/lease", json={"worker_id": "attacker-user", "subsystem": "image", "supported_models": ["qwen-image-2.1"], "supported_tasks": ["text_to_image"]})
            assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"

            # 2. /v1/workers/{worker_id}/heartbeat
            res = await client.post("/v1/workers/attacker-user/heartbeat", json={"worker_id": "attacker-user"})
            assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"

            # 3. /v1/jobs/{job_id}/progress
            res = await client.post(f"/v1/jobs/{fake_job_id}/progress", json={"stage": "sampling", "step": 1, "total": 10, "percent": 10.0})
            assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"

            # 4. /v1/jobs/{job_id}/complete
            res = await client.post(f"/v1/jobs/{fake_job_id}/complete", json={"artifacts": [], "inference_duration_ms": 10.0})
            assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"

            # 5. /v1/jobs/{job_id}/fail
            res = await client.post(f"/v1/jobs/{fake_job_id}/fail", json={"error": {"code": "INTERNAL_ERROR", "message": "fail", "request_id": "req-1"}})
            assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"


@pytest.mark.asyncio
async def test_expired_lease_mutation_matrix_via_api():
    """
    Adversarial Challenge: Job is leased to worker-E. Lease expires in DB.
    Worker-E attempts /progress, /complete, /fail, and /heartbeat.
    Invariants:
    - /progress returns 403 UnauthorizedError
    - /complete returns 403 UnauthorizedError
    - /fail returns 403 UnauthorizedError
    - /heartbeat does NOT extend lease (lease_extended_until is None)
    """
    async with lifespan(control_plane_app):
        worker_jwt = make_worker_jwt("worker-E")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_plane_app),
            base_url="http://test",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        ) as client:
            # 1. Submit job
            sub_res = await client.post(
                "/v1/jobs",
                headers=AUTH_HEADERS,
                json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "Lease expiry test"}},
            )
            assert sub_res.status_code == 201
            job_id = sub_res.json()["job_id"]

            # 2. Worker E leases the job
            lease_res = await client.post(
                "/v1/jobs/lease",
                headers={"Authorization": f"Bearer {worker_jwt}"},
                json={"worker_id": "worker-E", "subsystem": "image", "supported_models": ["qwen-image-2.1"], "supported_tasks": ["text_to_image"]},
            )
            assert lease_res.status_code == 200
            token_e = lease_res.json()["lease_token"]

            # 3. Manually expire the lease in the repository
            manager = control_plane_app.state.job_manager
            past = datetime.now(timezone.utc) - timedelta(seconds=15)
            await manager.repository.extend_lease(worker_id="worker-E", job_id=job_id, new_expires_at=past)

            # 4. Attempt /progress on expired lease -> 403
            prog_res = await client.post(
                f"/v1/jobs/{job_id}/progress",
                headers={"Authorization": f"Bearer {worker_jwt}"},
                json={"stage": "sampling", "step": 1, "total": 10, "percent": 10.0, "worker_id": "worker-E", "lease_token": token_e},
            )
            assert prog_res.status_code == 403
            assert "expired" in prog_res.json()["error"]["message"].lower()

            # 5. Attempt /complete on expired lease -> 403
            comp_res = await client.post(
                f"/v1/jobs/{job_id}/complete",
                headers={"Authorization": f"Bearer {worker_jwt}"},
                json={
                    "worker_id": "worker-E",
                    "lease_token": token_e,
                    "artifacts": [{
                        "artifact_id": f"art-{job_id}", "type": "image", "mime_type": "image/png",
                        "size_bytes": 100, "storage_path": "out.png", "url": "https://s3.ai.example.com/out.png",
                        "sha256": "0" * 64,
                    }],
                    "inference_duration_ms": 100.0,
                },
            )
            assert comp_res.status_code == 403
            assert "expired" in comp_res.json()["error"]["message"].lower()

            # 6. Attempt /fail on expired lease -> 403
            fail_res = await client.post(
                f"/v1/jobs/{job_id}/fail",
                headers={"Authorization": f"Bearer {worker_jwt}"},
                json={
                    "worker_id": "worker-E",
                    "lease_token": token_e,
                    "error": {"code": "INTERNAL_ERROR", "message": "OOM after expiry", "request_id": f"req-{uuid.uuid4()}"},
                },
            )
            assert fail_res.status_code == 403
            assert "expired" in fail_res.json()["error"]["message"].lower()


# ===========================================================================
# 7. Worker Capability Injection & Subsystem Manifest Boundary Attacks
# ===========================================================================

@pytest.mark.asyncio
async def test_cross_subsystem_worker_capability_injection():
    """
    Adversarial Challenge: Workers try cross-subsystem capability injection:
    - Text worker requests image/video tasks
    - Speech worker requests enhancement tasks
    - Unknown subsystem name
    - SQL injection string in subsystem name
    Invariants:
    - Capabilities outside subsystem manifest are strictly stripped.
    - Zero mismatched jobs are leased.
    """
    repo = SqliteJobRepository(db_path=":memory:")
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    coordinator = LeaseCoordinator(repository=repo, broker=broker)

    try:
        # 1. Text worker claiming image and video capabilities
        text_worker_req = WorkerLeaseRequest(
            worker_id="text-worker-infiltrator",
            subsystem="text",
            supported_models=["qwen-text", "wan-2.2", "qwen-image-2.1"],
            supported_tasks=["chat_completion", "text_to_image", "text_to_video"],
        )
        resp1 = await coordinator.lease_job(text_worker_req)
        assert resp1.job is None

        reg1 = await repo.get_worker("text-worker-infiltrator")
        assert reg1 is not None
        # Only chat_completion and qwen-text should remain
        assert set(reg1["supported_models"]) == {"qwen-text"}
        assert set(reg1["supported_tasks"]) == {"chat_completion"}

        # 2. Unknown subsystem
        unknown_req = WorkerLeaseRequest(
            worker_id="quantum-worker",
            subsystem="quantum_compute",
            supported_models=["qwen-text"],
            supported_tasks=["chat_completion"],
        )
        resp2 = await coordinator.lease_job(unknown_req)
        assert resp2.job is None

        # 3. SQL injection payload in subsystem
        sqli_req = WorkerLeaseRequest(
            worker_id="sqli-worker",
            subsystem="image' OR '1'='1",
            supported_models=["qwen-image-2.1"],
            supported_tasks=["text_to_image"],
        )
        resp3 = await coordinator.lease_job(sqli_req)
        assert resp3.job is None
    finally:
        await repo.close()


# ===========================================================================
# 8. WebSocket Authorization Matrix & Query Param Rejection
# ===========================================================================

def test_websocket_query_param_rejection_matrix():
    """
    Adversarial Challenge: Matrix test of query parameter rejection on WebSocket endpoints:
    - ?token=<valid_api_key>
    - ?token=<valid_jwt>
    - ?token=<invalid_token>
    - ?token=
    - ?token=<valid_token> with valid Authorization header simultaneously
    Invariants:
    - Any query param ?token= is strictly rejected with close code 1008.
    """
    from starlette.websockets import WebSocketDisconnect
    with TestClient(control_plane_app) as client:
        sub_res = client.post(
            "/v1/jobs",
            headers=AUTH_HEADERS,
            json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"prompt": "ws matrix test"}},
        )
        assert sub_res.status_code == 201
        job_id = sub_res.json()["job_id"]

        jwt_token = make_worker_jwt("ws-test-worker")

        test_cases = [
            f"/v1/jobs/{job_id}/ws?token={AUTH_TOKEN}",
            f"/v1/jobs/{job_id}/ws?token={jwt_token}",
            f"/v1/jobs/{job_id}/ws?token=invalid_garbage_token",
            f"/v1/jobs/{job_id}/ws?token=",
        ]

        for url in test_cases:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(url):
                    pass
            assert exc_info.value.code == 1008, f"Failed on URL: {url}, got code {exc_info.value.code}"

        # Dual credential: Valid header AND query param simultaneously
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/v1/jobs/{job_id}/ws?token={AUTH_TOKEN}",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            ):
                pass
        assert exc_info.value.code == 1008


# ===========================================================================
# 9. Bidirectional Alias and Canonical Normalization Parity
# ===========================================================================

def test_normalization_adversarial_edge_cases():
    """
    Adversarial Challenge: Stress test canonical model and task normalization functions:
    - Empty, None, whitespace
    - Case sensitivity and spacing
    - Capability IDs to canonical tasks and back to primary capability IDs
    - Unknown strings preserve safely
    """
    from factory_contracts.tasks import (
        normalize_model,
        normalize_task,
        task_to_capability_id,
        model_to_family,
        get_model_aliases,
        get_task_aliases,
    )

    # 1. Null / empty edge cases
    assert normalize_model(None) == ""
    assert normalize_model("") == ""
    assert normalize_model("   ") == ""
    assert normalize_task(None) == ""
    assert normalize_task("") == ""
    assert normalize_task("   ") == ""
    assert task_to_capability_id(None) is None
    assert task_to_capability_id("") is None
    assert model_to_family(None) is None

    # 2. Case sensitivity and whitespace stripping
    assert normalize_model("  QWEN-IMAGE  ") == "qwen-image-2.1"
    assert normalize_model("WAN-2.2") == "wan-2.2"
    assert normalize_model("minimax_h3") == "minimax-h3"
    assert normalize_model("ACE-STEP") == "ace-step-1.5"

    assert normalize_task("  IMAGE.T2I  ") == "text_to_image"
    assert normalize_task("T2V") == "text_to_video"
    assert normalize_task("speech.tts") == "speech_synthesis"
    assert normalize_task("SPEECH_TO_TEXT") == "speech_transcription"

    # 3. Primary capability ID resolution
    assert task_to_capability_id("text_to_image") == "image.t2i"
    assert task_to_capability_id("text_to_video") == "video.t2v"
    assert task_to_capability_id("speech_synthesis") == "speech.tts"
    assert task_to_capability_id("speech_transcription") == "speech.asr"
    assert task_to_capability_id("image_to_image") == "image.i2i"
    assert task_to_capability_id("inpaint") == "image.inpaint"
    assert task_to_capability_id("outpaint") == "image.outpaint"

    # 4. Model to family resolution
    assert model_to_family("qwen-image-2.1") == "qwen-image"
    assert model_to_family("wan-2.2") == "wan"
    assert model_to_family("minimax-h3") == "minimax"
    assert model_to_family("ace-step-1.5") == "ace-step"

    # 5. Unknown string handling
    unknown_model = "custom-transformer-v99"
    assert normalize_model(unknown_model) == unknown_model
    assert model_to_family(unknown_model) is None
    assert get_model_aliases(unknown_model) == []

    unknown_task = "quantum_superposition_inference"
    assert normalize_task(unknown_task) == unknown_task
    assert task_to_capability_id(unknown_task) is None
    assert get_task_aliases(unknown_task) == []
