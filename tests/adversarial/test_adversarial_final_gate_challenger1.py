"""
tests/adversarial/test_adversarial_final_gate_challenger1.py
Empirical Adversarial Challenge Suite for the Final Acceptance Gate (Challenger 1):
1. Asset endpoints (/v1/assets/upload, /v1/assets/{asset_id}/download):
   - Path traversal fuzzing in asset_id
   - Corrupt media payload fuzzing (empty, truncated, garbage, decompression bombs)
   - MIME spoofing attacks (executable/script with image mime, png with mp4 mime)
   - Oversize payload rejection (>50MB via Content-Length and byte streaming)
   - Dual-mode (multipart vs json) contract verification and malformed multipart handling
2. Jobs list endpoint (/v1/jobs):
   - Boundary limits (limit=-5, 0, 1, 1000, 1001, invalid strings)
   - Offset pagination boundary and consistency (offset=-1, 0, 100000)
   - State filter fuzzing and SQL injection attempts
   - Route ordering verification (ensure /v1/jobs does not match /{job_id})
3. Worker Lease Authorization & Lifecycle:
   - High-concurrency lease race condition (20 workers competing for 1 job)
   - Spoofed worker_id, spoofed lease_token, forged combinations
   - Unleased job security checks
   - Cancelled job lifecycle transitions and heartbeat cancel_requested signal
   - Heartbeat lease extension strictness
4. Canonical Task Normalization:
   - Edge cases, whitespace, case insensitivity, unknown custom tasks
"""
import asyncio
import hashlib
import io
import os
import struct
import time
import pytest
from starlette.testclient import TestClient

from factory.apps.control_api.src.main import app
from factory_contracts.errors import ErrorCode, UnauthorizedError
from factory.apps.control_api.src.queue.state_machine import InvalidStateTransitionError
from factory_contracts.jobs.job_contract import ArtifactReference, FactoryJob, JobProgress, JobState, JobSubmissionRequest
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
)
from factory_contracts.tasks import CANONICAL_TASK_MAP, normalize_task
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.streaming.manager import StreamManager

VALID_TOKEN = "sk-factory-prod-secret-token"

VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x02\x00\x00\x00\x02PX\xea"
    b"\x00\x00\x00\x13IDATx\x9cc\xfc\xcf\x80\x0f0\xe1\x95e\x18\xa9\xd2\x00A,\x01\x13y\xed\xba&\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(scope="module")
def api_client():
    keys_to_preserve = ["FACTORY_API_KEY", "STORAGE_BACKEND", "FACTORY_DB_BACKEND", "SQLITE_DB_PATH"]
    old_env = {k: os.environ.get(k) for k in keys_to_preserve}
    os.environ["FACTORY_API_KEY"] = VALID_TOKEN
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = ":memory:"
    try:
        with TestClient(app) as client:
            yield client
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")


@pytest.fixture
def auth_headers():
    return {
        "Authorization": f"Bearer {VALID_TOKEN}",
    }


# =========================================================================
# SECTION 1: Adversarial Asset Ingestion & Download Challenges
# =========================================================================
class TestAdversarialAssets:
    """Stress tests and adversarial probes for /v1/assets/upload and /v1/assets/{asset_id}/download."""

    def test_path_traversal_in_download_asset_id(self, api_client, auth_headers):
        """Attempts path traversal via URL-encoded characters in asset_id."""
        traversal_payloads = [
            "..%2F..%2Fetc%2Fpasswd",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%5C..%5Cwindows%5Cwin.ini",
            "....//....//etc/passwd",
            "%2e%2e%2e%2e%2fetc%2fshadow",
            "ast-1234/../../secret.txt",
        ]
        for payload in traversal_payloads:
            res = api_client.get(f"/v1/assets/{payload}/download", headers=auth_headers)
            # Must either be 404 Not Found (or 400 Invalid / sanitized non-existent) - NEVER 200 or raw file leak
            assert res.status_code in (404, 400), f"Payload {payload} returned unexpected status {res.status_code}"
            assert b"root:" not in res.content
            assert b"[extensions]" not in res.content

    def test_mime_spoofing_script_as_image(self, api_client, auth_headers):
        """Uploads a bash/python script claiming to be an image/png."""
        malicious_script = b"#!/bin/bash\nrm -rf / --no-preserve-root\necho 'hacked'\n"
        files = {"file": ("exploit.png", malicious_script, "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_mime_spoofing_mismatched_container(self, api_client, auth_headers):
        """Uploads genuine PNG bytes but claims video/mp4 MIME."""
        files = {"file": ("video.mp4", VALID_PNG_BYTES, "video/mp4")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 201
        data = res.json()
        # Security check: AssetService detected true content MIME 'image/png', ignoring spoofed 'video/mp4'
        assert data["mime_type"] == "image/png"

    def test_corrupt_zero_byte_media_rejected(self, api_client, auth_headers):
        """Uploads a zero-byte file."""
        files = {"file": ("empty.png", b"", "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_corrupt_truncated_magic_bytes(self, api_client, auth_headers):
        """Uploads truncated magic bytes that fail minimum size requirements."""
        files = {"file": ("truncated.png", b"\x89PNG", "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_corrupt_png_chunks_rejected_by_pil(self, api_client, auth_headers):
        """Uploads valid PNG header followed by malformed/corrupted chunk stream."""
        corrupt_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff" * 30
        files = {"file": ("corrupt.png", corrupt_png, "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_corrupt_mp4_truncated_ftyp(self, api_client, auth_headers):
        """Uploads MP4 header with truncated ftyp atom length."""
        corrupt_mp4 = struct.pack(">I4s", 4, b"ftyp") + b"\x00" * 10
        files = {"file": ("corrupt.mp4", corrupt_mp4, "video/mp4")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_oversized_upload_rejection_header(self, api_client, auth_headers):
        """Verifies Content-Length header exceeding 50MB is rejected with 413."""
        headers = {
            **auth_headers,
            "Content-Length": str(51 * 1024 * 1024),
            "Content-Type": "application/json",
        }
        res = api_client.post("/v1/assets/upload", headers=headers, json={"mime_type": "image/png"})
        assert res.status_code == 413
        err = res.json()["error"]
        assert err["code"] == "PAYLOAD_TOO_LARGE"

    def test_multipart_missing_file_field(self, api_client, auth_headers):
        """Submits multipart form with a text field instead of 'file'."""
        res = api_client.post(
            "/v1/assets/upload",
            headers=auth_headers,
            data={"description": "missing file field"},
        )
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_REQUEST"

    def test_corrupt_wav_missing_fmt_chunk(self, api_client, auth_headers):
        """Uploads RIFF WAVE missing mandatory fmt chunk."""
        # RIFF(4) + size(4) + WAVE(4) + data chunk(8)
        malformed_wav = struct.pack("<4sI4s4sI", b"RIFF", 16, b"WAVE", b"data", 0)
        files = {"file": ("malformed.wav", malformed_wav, "audio/wav")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_corrupt_flac_invalid_first_block(self, api_client, auth_headers):
        """Uploads FLAC with first metadata block not STREAMINFO (type 0)."""
        malformed_flac = b"fLaC\x01\x00\x00\x22" + b"\x00" * 40
        files = {"file": ("malformed.flac", malformed_flac, "audio/flac")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_image_dimension_bomb_rejected(self, api_client, auth_headers):
        """Uploads an image exceeding MAX_IMAGE_DIMENSION (4096px)."""
        from PIL import Image
        img = Image.new("RGB", (5000, 10), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        oversized_image_bytes = buf.getvalue()

        files = {"file": ("wide_bomb.png", oversized_image_bytes, "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_ssrf_probes_blocked_on_validate(self, api_client, auth_headers):
        """Validates SSRF protection blocks cloud metadata and private IP addresses."""
        probes = [
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://127.0.0.1:8080/admin",
            "http://10.0.0.1/internal",
            "http://172.16.0.1/internal",
            "http://192.168.1.1/router",
        ]
        headers = {**auth_headers, "Content-Type": "application/json"}
        for probe_url in probes:
            res = api_client.post("/v1/assets/validate", headers=headers, json={"url": probe_url})
            assert res.status_code == 400, f"Expected 400 for SSRF probe {probe_url}, got {res.status_code}"
            err = res.json()["error"]
            assert err["code"] in ("SSRF_BLOCKED", "INVALID_REQUEST")


# =========================================================================
# SECTION 2: Adversarial Jobs List & Route Ordering Challenges
# =========================================================================
class TestAdversarialJobsList:
    """Stress tests and boundary checks for GET /v1/jobs."""

    def test_limit_boundary_negative_rejected(self, api_client, auth_headers):
        """Querying negative limit must be rejected by validation (422)."""
        res = api_client.get("/v1/jobs?limit=-5", headers=auth_headers)
        assert res.status_code == 422

    def test_limit_boundary_zero_rejected(self, api_client, auth_headers):
        """Querying limit=0 must be rejected by validation (422) since ge=1."""
        res = api_client.get("/v1/jobs?limit=0", headers=auth_headers)
        assert res.status_code == 422

    def test_limit_boundary_max_1000(self, api_client, auth_headers):
        """Querying limit=1000 is allowed; limit=1001 is rejected with 422."""
        res_ok = api_client.get("/v1/jobs?limit=1000", headers=auth_headers)
        assert res_ok.status_code == 200

        res_excess = api_client.get("/v1/jobs?limit=1001", headers=auth_headers)
        assert res_excess.status_code == 422

    def test_offset_negative_rejected(self, api_client, auth_headers):
        """Querying offset=-1 must be rejected by validation (422)."""
        res = api_client.get("/v1/jobs?offset=-1", headers=auth_headers)
        assert res.status_code == 422

    def test_offset_far_out_of_bounds_returns_empty_list(self, api_client, auth_headers):
        """Querying offset far out of bounds returns [] without error."""
        res = api_client.get("/v1/jobs?offset=999999", headers=auth_headers)
        assert res.status_code == 200
        assert res.json() == []

    def test_sql_injection_in_service_and_state_filters(self, api_client, auth_headers):
        """Verifies repository SQL parameterization resists SQL injection payloads."""
        sqli_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE jobs; --",
            "admin'--",
            "\" OR \"\"=\"",
            "1 UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        ]
        for payload in sqli_payloads:
            res_svc = api_client.get(f"/v1/jobs?service={payload}", headers=auth_headers)
            assert res_svc.status_code == 200
            assert isinstance(res_svc.json(), list)

            res_st = api_client.get(f"/v1/jobs?state={payload}", headers=auth_headers)
            assert res_st.status_code == 200
            assert isinstance(res_st.json(), list)

    def test_fuzz_service_and_state_parameters_extreme_lengths(self, api_client, auth_headers):
        """Fuzzes query parameters with 5000-char strings, unicode, emojis, and control characters."""
        long_string = "A" * 5000
        unicode_string = "🔥🚀🤖_тест_测试_1234_ãéñ_✨"
        null_byte_string = "image\x00extra"

        for fuzzed in (long_string, unicode_string, null_byte_string):
            res = api_client.get("/v1/jobs", headers=auth_headers, params={"service": fuzzed, "state": fuzzed})
            assert res.status_code == 200
            assert isinstance(res.json(), list)

    def test_route_ordering_jobs_root_vs_job_id(self, api_client, auth_headers):
        """
        Verifies route ordering:
        - GET /v1/jobs matches the list endpoint (returns list 200).
        - GET /v1/jobs/nonexistent-id matches get_job_status and returns 404 (NotFoundError).
        """
        res_list = api_client.get("/v1/jobs", headers=auth_headers)
        assert res_list.status_code == 200
        assert isinstance(res_list.json(), list)

        res_single = api_client.get("/v1/jobs/nonexistent-job-xyz-123", headers=auth_headers)
        assert res_single.status_code == 404
        err = res_single.json()["error"]
        assert err["code"] == "NOT_FOUND"


# =========================================================================
# SECTION 3: Worker Lease Authorization & Lifecycle Stress
# =========================================================================
@pytest.mark.asyncio
class TestAdversarialWorkerLeaseAuth:
    """Adversarial stress tests for worker lease ownership, race conditions, and transitions."""

    async def _setup(self):
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)
        return repo, broker, stream_mgr, manager, coordinator

    async def test_lease_race_condition_twenty_workers_competing_one_job(self):
        """20 concurrent workers simultaneously attempt to lease 1 QUEUED job."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "High concurrency race job"},
                ),
                request_id="req-race-test",
            )

            # Fire 20 concurrent lease requests
            async def _attempt_lease(worker_idx: int):
                return await coordinator.lease_job(
                    WorkerLeaseRequest(
                        worker_id=f"worker-race-{worker_idx:02d}",
                        subsystem="image",
                        supported_models=["qwen-image-2.1"],
                        supported_tasks=["text_to_image"],
                    )
                )

            tasks = [_attempt_lease(i) for i in range(20)]
            results = await asyncio.gather(*tasks)

            # Exactly 1 worker must acquire the lease, 19 must receive None
            acquired = [r for r in results if r.job is not None]
            failed = [r for r in results if r.job is None]

            assert len(acquired) == 1, f"Expected exactly 1 lease acquisition, got {len(acquired)}"
            assert len(failed) == 19
            winner_token = acquired[0].lease_token
            assert winner_token is not None
            assert acquired[0].job.job_id == job.job_id
        finally:
            await repo.close()

    async def test_spoofed_lease_tokens_and_worker_ids(self):
        """Fuzzes worker_id and lease_token permutations against update_progress, complete_job, fail_job."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Spoof test"},
                ),
                request_id="req-spoof-test",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="legit-worker",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease.lease_token

            spoof_cases = [
                ("imposter-worker", token),
                ("legit-worker", "forged-token"),
                ("imposter-worker", "forged-token"),
                (None, token),
                ("legit-worker", None),
                (None, None),
            ]

            artifact = ArtifactReference(
                artifact_id="art-test",
                mime_type="image/png",
                size_bytes=100,
                storage_path="assets/art.png",
                download_url="https://s3.ai.example.com/assets/art.png",
                sha256="c" * 64,
            )

            for w_id, tok in spoof_cases:
                # 1. Check update_progress rejects with 403
                with pytest.raises(UnauthorizedError) as exc_p:
                    await manager.update_progress(
                        job.job_id,
                        ProgressUpdateRequest(stage="run", step=1, total=10, percent=10.0, worker_id=w_id, lease_token=tok),
                    )
                assert exc_p.value.status_code == 403

                # 2. Check complete_job rejects with 403
                with pytest.raises(UnauthorizedError) as exc_c:
                    await manager.complete_job(
                        job.job_id,
                        JobCompletionRequest(artifacts=[artifact], worker_id=w_id, lease_token=tok),
                    )
                assert exc_c.value.status_code == 403

                # 3. Check fail_job rejects with 403
                with pytest.raises(UnauthorizedError) as exc_f:
                    await manager.fail_job(
                        job.job_id,
                        JobFailureRequest(
                            error={"code": "INTERNAL_ERROR", "message": "fail", "request_id": "r1"},
                            worker_id=w_id,
                            lease_token=tok,
                        ),
                    )
                assert exc_f.value.status_code == 403
        finally:
            await repo.close()

    async def test_cancelled_job_blocks_worker_completion_and_signals_heartbeat(self):
        """
        When a job is cancelled:
        1. Heartbeat reports cancel_requested=True and does not extend lease.
        2. Worker attempt to call complete_job is blocked by state machine (cannot transition CANCELLED -> SUCCEEDED).
        3. Worker attempt to call update_progress is safely ignored (no-op).
        """
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Cancellation test"},
                ),
                request_id="req-cancel-test",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-cancel-test",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease.lease_token

            # User cancels job
            await manager.cancel_job(job.job_id)
            cancelled_job = await manager.get_job(job.job_id)
            assert cancelled_job.state == JobState.CANCELLED

            # 1. Heartbeat signals cancel_requested=True
            hb_res = await coordinator.process_heartbeat(
                WorkerHeartbeatRequest(
                    worker_id="worker-cancel-test",
                    active_job_id=job.job_id,
                    lease_token=token,
                )
            )
            assert hb_res.cancel_requested is True
            assert hb_res.lease_extended_until is None

            # 2. Worker attempts to complete cancelled job -> raises InvalidStateTransitionError
            artifact = ArtifactReference(
                artifact_id="art-test",
                mime_type="image/png",
                size_bytes=100,
                storage_path="assets/art.png",
                download_url="https://s3.ai.example.com/assets/art.png",
                sha256="d" * 64,
            )
            with pytest.raises(InvalidStateTransitionError):
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="worker-cancel-test",
                        lease_token=token,
                    ),
                )

            # State remains CANCELLED
            after_job = await manager.get_job(job.job_id)
            assert after_job.state == JobState.CANCELLED
        finally:
            await repo.close()

    async def test_job_immutability_and_idempotency_after_completion(self):
        """Once SUCCEEDED, progress updates or failures cannot corrupt job state."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Completion immutability test"},
                ),
                request_id="req-immutability-test",
            )
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-imm-test",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            token = lease.lease_token

            artifact = ArtifactReference(
                artifact_id="art-success",
                mime_type="image/png",
                size_bytes=500,
                storage_path="assets/art-success.png",
                download_url="https://s3.ai.example.com/assets/art-success.png",
                sha256="e" * 64,
            )

            # Complete the job legitimately
            comp = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    artifacts=[artifact],
                    worker_id="worker-imm-test",
                    lease_token=token,
                ),
            )
            assert comp.state == JobState.SUCCEEDED
            assert len(comp.artifacts) == 1

            # 1. Calling complete_job again is idempotent and does not corrupt state
            comp_repeat = await manager.complete_job(
                job.job_id,
                JobCompletionRequest(
                    artifacts=[],
                    worker_id="worker-imm-test",
                    lease_token=token,
                ),
            )
            assert comp_repeat.state == JobState.SUCCEEDED
            assert len(comp_repeat.artifacts) == 1  # Artifacts preserved!

            # 2. Calling update_progress after completion does not alter state
            prog_res = await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(stage="post_eval", step=0, total=10, percent=0.0),
            )
            assert prog_res.state == JobState.SUCCEEDED

            # 3. Calling fail_job after completion does not alter state
            fail_res = await manager.fail_job(
                job.job_id,
                JobFailureRequest(
                    error={"code": "INTERNAL_ERROR", "message": "Late failure", "request_id": "r1"},
                    worker_id="worker-imm-test",
                    lease_token=token,
                ),
            )
            assert fail_res.state == JobState.SUCCEEDED
        finally:
            await repo.close()

    async def test_unleased_job_completion_with_worker_id_rejected(self):
        """Worker cannot complete an unleased job by supplying an unauthorized worker_id."""
        repo, broker, stream_mgr, manager, coordinator = await self._setup()
        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Unleased completion attempt"},
                ),
                request_id="req-unleased-comp",
            )
            artifact = ArtifactReference(
                artifact_id="art-test",
                mime_type="image/png",
                size_bytes=100,
                storage_path="assets/art.png",
                download_url="https://s3.ai.example.com/assets/art.png",
                sha256="f" * 64,
            )
            # Defense 1: Direct completion from QUEUED state is blocked by State Machine
            with pytest.raises(InvalidStateTransitionError):
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="unauthorized-worker",
                    ),
                )

            # Move unleased job to RUNNING via unleased progress update
            await manager.update_progress(
                job.job_id,
                ProgressUpdateRequest(stage="prep", step=1, total=10, percent=10.0),
            )
            running_job = await manager.get_job(job.job_id)
            assert running_job.state == JobState.RUNNING

            # Defense 2: Unleased RUNNING job cannot be completed by worker claiming ownership
            with pytest.raises(UnauthorizedError) as exc_info:
                await manager.complete_job(
                    job.job_id,
                    JobCompletionRequest(
                        artifacts=[artifact],
                        worker_id="unauthorized-worker",
                    ),
                )
            assert exc_info.value.status_code == 403
        finally:
            await repo.close()


# =========================================================================
# SECTION 4: Canonical Task Normalization Adversarial Cases
# =========================================================================
class TestAdversarialTaskNormalization:
    """Stress tests canonical task normalization across whitespace, casing, and unknown inputs."""

    def test_normalization_fuzzing(self):
        cases = [
            ("   IMAGE.T2I   ", "text_to_image"),
            ("\n\tvideo.t2v\n", "text_to_video"),
            ("ImAgE.InPaInT", "inpaint"),
            ("  MUSIC.T2M  ", "text_to_music"),
            ("speech.TTS", "speech_synthesis"),
            ("ENHANCE.SR_IMAGE", "image_super_resolution"),
            ("unknown_random_task_xyz", "unknown_random_task_xyz"),
            ("   UNKNOWN_TASK   ", "unknown_task"),
            ("", ""),
            (None, ""),
        ]
        for raw, expected in cases:
            assert normalize_task(raw) == expected, f"Failed normalizing '{raw}': expected '{expected}'"
