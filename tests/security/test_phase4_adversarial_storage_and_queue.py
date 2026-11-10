"""
tests/security/test_phase4_adversarial_storage_and_queue.py
Milestone 4 Empirical Challenger Adversarial Security & Reliability Attack Harness.

Scope of Adversarial Challenge:
1. P1.6 Storage Security - Streaming Uploads:
   - Stream chunks exceeding 50MB (52MB in chunks). Verify HTTP 413 rejection.
   - Verify that 0 orphaned .tmp_* files remain in storage directory after breach.
   - Verify Content-Length header exceeding 50MB rejected before stream read.
   - Mid-stream network drop / exception cleans up all temporary files immediately.
   - HMAC signature tampering on max_size query parameter rejected with HTTP 403.
   - Asset ingestion endpoint rejects multipart streams exceeding 50MB.

2. P1.6 Media Delivery - HTTP Range Requests:
   - Valid range requests (bytes=0-100, bytes=100-200) return HTTP 206, Content-Range, Accept-Ranges, and accurate slices.
   - Boundary single-byte ranges (bytes=0-0, bytes={total-1}-{total-1}) return accurate byte slices.
   - Open-ended ranges (bytes=0-, bytes=500-) return accurate byte slices.
   - Suffix ranges (bytes=-50, bytes=-1) return accurate trailing byte slices.
   - Out-of-bounds / overflow range requests (bytes=999999-, bytes=5000-6000) handle gracefully without HTTP 500.
   - Malformed range headers (bytes=invalid-format, bytes=-, bytes=100-50, characters=0-100, bytes=NaN-NaN) handle gracefully.
   - Video media delivery preserves inline Content-Disposition and Accept-Ranges.

3. P1.6 Proxy Hardening - Rate Limiter Header Spoofing:
   - Untrusted direct socket IP cycling spoofed X-Forwarded-For headers is rate limited on the socket IP (HTTP 429).
   - Spoofed localhost/127.0.0.1 in X-Forwarded-For from untrusted socket cannot bypass rate limiting.
   - Comma-separated or malformed X-Forwarded-For headers from untrusted sockets remain bound to socket IP.
   - Verified trusted proxy forwarding correctly allocates independent rate limit quotas per unique client IP.

4. P2.3 Queue Reliability - complete_job Gating:
   - Trigger complete_job with missing artifact path: transitions directly to FAILED, NEVER enters SUCCEEDED.
   - Trigger complete_job with empty (0-byte) artifact file: transitions directly to FAILED, NEVER enters SUCCEEDED.
   - Trigger complete_job with req.artifacts pointing to missing file: transitions to FAILED.
   - Trigger complete_job with req.artifacts pointing to 0-byte file: transitions to FAILED.
   - Trigger complete_job with forged / mismatched SHA-256 hash: transitions to FAILED.
   - Trigger complete_job on media job with no artifacts: transitions to FAILED.
   - Trigger complete_job on non-media jobs (text, decision, embedding) with None, empty string, empty list, or empty dict: transitions to FAILED.
   - State transition interception: verify that job.state NEVER equals JobState.SUCCEEDED during any failure scenario.
"""
import asyncio
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import uuid
import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.testclient import TestClient

from factory_contracts.errors import (
    ErrorCode,
    ErrorDetail,
    FactoryError,
    InvalidRequestError,
    PayloadTooLargeError,
)
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    AssetRecord,
    FactoryJob,
    JobProgress,
    JobState,
    JobTelemetry,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
)
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware.rate_limit import RateLimitMiddleware
from factory.apps.control_api.src.api.routers.assets import router as assets_router
from factory.apps.control_api.src.api.routers.assets import get_asset_service, get_repository, get_storage_provider as get_assets_storage_provider
from factory.apps.control_api.src.assets.service import AssetService
from factory.apps.control_api.src.queue.broker import InMemoryScheduler
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.storage.config import StorageConfig
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.router import (
    router as storage_router,
    get_storage_provider as get_storage_router_provider,
    get_scratch_disk,
)
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskConfig, ScratchDiskManager
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory_auth import verify_bearer_token


# ============================================================================
# TEST FIXTURES & HARNESS SETUP
# ============================================================================

@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    bucket_name = "factory-test-bucket"
    monkeypatch.setenv("S3_BUCKET_NAME", bucket_name)
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("SCRATCH_ARTIFACTS_DIR", str(scratch_dir))

    config = StorageConfig(
        local_storage_dir=str(storage_dir),
        local_storage_secret_key="challenger-adversarial-test-key-32b!",
        s3_bucket_name=bucket_name,
        local_base_url="http://testserver",
    )
    provider = LocalStorageProvider(config)
    scratch = ScratchDiskManager(ScratchDiskConfig(root_dir=str(scratch_dir)))
    repo = InMemoryJobStore()
    asset_service = AssetService(max_size_bytes=50 * 1024 * 1024)

    return {
        "storage_dir": storage_dir,
        "scratch_dir": scratch_dir,
        "provider": provider,
        "scratch": scratch,
        "repo": repo,
        "asset_service": asset_service,
        "config": config,
        "bucket_name": bucket_name,
    }


@pytest.fixture
def storage_test_app(storage_env):
    app = FastAPI()
    register_exception_handlers(app)

    # Dependency overrides
    app.dependency_overrides[verify_bearer_token] = lambda: {"sub": "challenger", "role": "admin"}
    app.dependency_overrides[get_storage_router_provider] = lambda: storage_env["provider"]
    app.dependency_overrides[get_assets_storage_provider] = lambda: storage_env["provider"]
    app.dependency_overrides[get_scratch_disk] = lambda: storage_env["scratch"]
    app.dependency_overrides[get_repository] = lambda: storage_env["repo"]
    app.dependency_overrides[get_asset_service] = lambda: storage_env["asset_service"]

    app.include_router(storage_router)
    app.include_router(assets_router)
    return app


# ============================================================================
# 1. STREAMING UPLOADS ADVERSARIAL STRESS TESTS (P1.6 Storage Security)
# ============================================================================

class TestStreamingUploadsAdversarialHarness:
    """Stress tests upload streaming with hard 50MB limits, temp file purging, and tampering attacks."""

    def test_streaming_upload_52mb_rejected_with_413_and_zero_tmp_leak(self, storage_test_app, storage_env):
        """
        Adversarial stream exceeding 50MB (52MB in 1MB chunks).
        Must return HTTP 413 and leave ZERO orphaned .tmp_* files on disk.
        """
        client = TestClient(storage_test_app)
        provider = storage_env["provider"]
        storage_dir = storage_env["storage_dir"]

        # Hard limit 50MB = 52,428,800 bytes
        limit_bytes = 50 * 1024 * 1024
        presigned = provider.generate_presigned_upload_url(
            bucket=storage_env["bucket_name"],
            key="artifacts/stress-job/52mb_stream.bin",
            mime_type="application/octet-stream",
            max_size_bytes=limit_bytes,
        )
        target_path = presigned.upload_url.replace("http://testserver", "")

        # Generator yielding 52 chunks of 1MB (52MB total)
        chunk_1mb = b"A" * (1024 * 1024)
        def stream_52mb_generator():
            for _ in range(52):
                yield chunk_1mb

        resp = client.put(
            target_path,
            content=stream_52mb_generator(),
            headers={"Content-Type": "application/octet-stream"},
        )

        assert resp.status_code == 413
        assert "exceeded hard limit" in resp.text.lower()

        # Empirical assertion: ZERO .tmp_* files exist anywhere in storage_dir
        tmp_files = list(storage_dir.rglob(".tmp_*"))
        assert len(tmp_files) == 0, f"Leaked temporary files found: {tmp_files}"

        # Destination file must NOT exist
        dest_file = storage_dir / storage_env["bucket_name"] / "artifacts" / "stress-job" / "52mb_stream.bin"
        assert not dest_file.exists(), "Oversized artifact was written to destination!"

    def test_content_length_exceeding_50mb_rejected_before_stream_read(self, storage_test_app, storage_env):
        """
        Request sending Content-Length: 54525952 (52MB) must be rejected with HTTP 413
        immediately before consuming stream or creating temporary files.
        """
        client = TestClient(storage_test_app)
        provider = storage_env["provider"]
        storage_dir = storage_env["storage_dir"]

        limit_bytes = 50 * 1024 * 1024
        presigned = provider.generate_presigned_upload_url(
            bucket=storage_env["bucket_name"],
            key="artifacts/stress-job/declared_52mb.bin",
            mime_type="application/octet-stream",
            max_size_bytes=limit_bytes,
        )
        target_path = presigned.upload_url.replace("http://testserver", "")

        resp = client.put(
            target_path,
            content=b"X" * 1024,  # Small actual content, but declaring 52MB in header
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": "54525952",  # 52 MB
            },
        )

        assert resp.status_code == 413
        assert "exceeds maximum permitted limit" in resp.text.lower()

        # Zero temporary files created or leaked
        tmp_files = list(storage_dir.rglob(".tmp_*"))
        assert len(tmp_files) == 0

        # Destination file must not exist
        dest_file = storage_dir / storage_env["bucket_name"] / "artifacts" / "stress-job" / "declared_52mb.bin"
        assert not dest_file.exists()

    def test_stream_network_abortion_purges_temporary_file(self, storage_test_app, storage_env):
        """
        If client disconnects or aborts the stream midway, the exception handler
        must immediately unlink the partial .tmp_* file.
        """
        client = TestClient(storage_test_app)
        provider = storage_env["provider"]
        storage_dir = storage_env["storage_dir"]

        presigned = provider.generate_presigned_upload_url(
            bucket=storage_env["bucket_name"],
            key="artifacts/stress-job/aborted.bin",
            mime_type="application/octet-stream",
            max_size_bytes=50 * 1024 * 1024,
        )
        target_path = presigned.upload_url.replace("http://testserver", "")

        # Generator that throws an exception midway through upload
        def aborted_stream():
            yield b"PARTIAL_CHUNK_1" * 1024
            yield b"PARTIAL_CHUNK_2" * 1024
            raise ConnectionResetError("Simulated client connection drop")

        with pytest.raises(ConnectionResetError):
            client.put(
                target_path,
                content=aborted_stream(),
                headers={"Content-Type": "application/octet-stream"},
            )

        # Assert no temporary files were left behind
        tmp_files = list(storage_dir.rglob(".tmp_*"))
        assert len(tmp_files) == 0, f"Aborted stream left orphaned files: {tmp_files}"

    def test_presigned_url_tampering_of_max_size_fails_signature(self, storage_test_app, storage_env):
        """
        Adversary tampering with max_size query param to artificially inflate upload ceiling
        must be rejected with HTTP 403 Forbidden due to HMAC binding.
        """
        client = TestClient(storage_test_app)
        provider = storage_env["provider"]

        # Generate URL with 1024 bytes limit
        presigned = provider.generate_presigned_upload_url(
            bucket=storage_env["bucket_name"],
            key="artifacts/tamper-job/output.bin",
            mime_type="application/octet-stream",
            max_size_bytes=1024,
        )
        url = presigned.upload_url.replace("http://testserver", "")

        # Tamper query param: change max_size=1024 to max_size=104857600 (100MB)
        tampered_url = url.replace("max_size=1024", "max_size=104857600")

        resp = client.put(
            tampered_url,
            content=b"A" * 2048,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 403
        assert "invalid or expired upload signature" in resp.text.lower()

    def test_asset_multipart_upload_rejects_content_length_over_50mb(self, storage_test_app):
        """
        /v1/assets/upload rejects requests declaring Content-Length > 50MB with HTTP 413.
        """
        client = TestClient(storage_test_app)
        files = {"file": ("video.mp4", io.BytesIO(b"DUMMY_DATA"), "video/mp4")}
        resp = client.post(
            "/v1/assets/upload",
            files=files,
            headers={
                "Authorization": "Bearer sk-challenger-token",
                "Content-Length": str(55 * 1024 * 1024),  # 55MB declared
            },
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


# ============================================================================
# 2. HTTP RANGE REQUESTS ADVERSARIAL STRESS TESTS (P1.6 Media Delivery)
# ============================================================================

class TestMediaRangeRequestsAdversarialHarness:
    """Stress tests HTTP 206 Partial Content, boundaries, byte slices, and malformed range headers."""

    @pytest.fixture(autouse=True)
    def setup_asset(self, storage_env):
        # Create a sample 2048-byte media payload with deterministic pattern
        self.payload = bytes([i % 256 for i in range(2048)])
        self.total_bytes = len(self.payload)
        self.sha256 = compute_bytes_sha256(self.payload)

        # Ingest directly into repository and storage
        storage = storage_env["provider"]
        repo = storage_env["repo"]

        self.asset_id = f"ast-{uuid.uuid4()}"
        self.storage_path = f"assets/{self.asset_id}.mp4"

        asyncio.run(storage.upload_bytes(
            bucket=storage_env["bucket_name"],
            key=self.storage_path,
            data=self.payload,
            mime_type="video/mp4",
        ))

        now_dt = datetime.now(timezone.utc)
        record = AssetRecord(
            asset_id=self.asset_id,
            sha256=self.sha256,
            mime_type="video/mp4",
            size_bytes=self.total_bytes,
            storage_path=self.storage_path,
            s3_url=f"/v1/assets/{self.asset_id}/download",
            download_url=f"/v1/assets/{self.asset_id}/download",
            filename="sample_video.mp4",
            created_at=now_dt,
            verified=True,
        )
        asyncio.run(repo.create_asset(record))

    def test_range_valid_standard_slices(self, storage_test_app):
        """Standard range requests (bytes=0-100, bytes=100-200) return 206 and exact byte slices."""
        client = TestClient(storage_test_app)
        auth = {"Authorization": "Bearer sk-test-token"}

        # 1. bytes=0-100 (101 bytes)
        r1 = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=0-100"},
        )
        assert r1.status_code == 206
        assert r1.headers["Content-Range"] == f"bytes 0-100/{self.total_bytes}"
        assert r1.headers["Accept-Ranges"] == "bytes"
        assert r1.headers["Content-Length"] == "101"
        assert r1.content == self.payload[0:101]

        # 2. bytes=100-200 (101 bytes)
        r2 = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=100-200"},
        )
        assert r2.status_code == 206
        assert r2.headers["Content-Range"] == f"bytes 100-200/{self.total_bytes}"
        assert r2.headers["Content-Length"] == "101"
        assert r2.content == self.payload[100:201]

    def test_range_boundary_single_byte(self, storage_test_app):
        """Boundary single-byte range requests (bytes=0-0 and bytes={total-1}-{total-1})."""
        client = TestClient(storage_test_app)
        auth = {"Authorization": "Bearer sk-test-token"}

        # First byte
        r_start = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=0-0"},
        )
        assert r_start.status_code == 206
        assert r_start.headers["Content-Range"] == f"bytes 0-0/{self.total_bytes}"
        assert r_start.headers["Content-Length"] == "1"
        assert r_start.content == self.payload[0:1]

        # Last byte
        last_idx = self.total_bytes - 1
        r_end = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": f"bytes={last_idx}-{last_idx}"},
        )
        assert r_end.status_code == 206
        assert r_end.headers["Content-Range"] == f"bytes {last_idx}-{last_idx}/{self.total_bytes}"
        assert r_end.headers["Content-Length"] == "1"
        assert r_end.content == self.payload[last_idx:last_idx + 1]

    def test_range_open_ended(self, storage_test_app):
        """Open-ended range requests (bytes=0- and bytes=500-)."""
        client = TestClient(storage_test_app)
        auth = {"Authorization": "Bearer sk-test-token"}

        # bytes=0- (full content via 206)
        r_all = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=0-"},
        )
        assert r_all.status_code == 206
        assert r_all.headers["Content-Range"] == f"bytes 0-{self.total_bytes - 1}/{self.total_bytes}"
        assert r_all.content == self.payload

        # bytes=500-
        r_tail = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=500-"},
        )
        assert r_tail.status_code == 206
        assert r_tail.headers["Content-Range"] == f"bytes 500-{self.total_bytes - 1}/{self.total_bytes}"
        assert r_tail.content == self.payload[500:]

    def test_range_suffix_specification(self, storage_test_app):
        """Suffix range requests (bytes=-50 and bytes=-1)."""
        client = TestClient(storage_test_app)
        auth = {"Authorization": "Bearer sk-test-token"}

        # Last 50 bytes
        r_50 = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=-50"},
        )
        assert r_50.status_code == 206
        assert r_50.headers["Content-Range"] == f"bytes {self.total_bytes - 50}-{self.total_bytes - 1}/{self.total_bytes}"
        assert r_50.headers["Content-Length"] == "50"
        assert r_50.content == self.payload[-50:]

        # Last 1 byte
        r_1 = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": "bytes=-1"},
        )
        assert r_1.status_code == 206
        assert r_1.headers["Content-Range"] == f"bytes {self.total_bytes - 1}-{self.total_bytes - 1}/{self.total_bytes}"
        assert r_1.headers["Content-Length"] == "1"
        assert r_1.content == self.payload[-1:]

    @pytest.mark.parametrize(
        "range_header,desc",
        [
            ("bytes=999999-", "Start offset far exceeds total size"),
            ("bytes=5000-6000", "Both start and end far exceed total size"),
            ("bytes=2048-", "Start offset equals total size (0-indexed out of bounds)"),
            ("bytes=invalid-format", "Malformed non-numeric format"),
            ("bytes=-", "Missing both start and end"),
            ("bytes=100-50", "Inverted start greater than end"),
            ("characters=0-100", "Invalid range unit"),
            ("bytes=NaN-NaN", "NaN values"),
            ("bytes=0-99999999999999999999999999999999999999", "Huge overflow integer"),
            ("bytes=--100", "Double negative"),
        ],
    )
    def test_range_malformed_and_out_of_bounds_gracefully_degrades(self, storage_test_app, range_header, desc):
        """
        Adversarial malformed or out-of-bounds range headers must NEVER crash with 500.
        They must gracefully return either 200 (full content fallback) or 206.
        """
        client = TestClient(storage_test_app)
        auth = {"Authorization": "Bearer sk-test-token"}

        resp = client.get(
            f"/v1/assets/{self.asset_id}/download",
            headers={**auth, "Range": range_header},
        )
        # Must never return 500 Internal Server Error
        assert resp.status_code in (200, 206), f"Failed for {desc} with status {resp.status_code}"
        if resp.status_code == 200:
            assert resp.content == self.payload
        assert resp.headers.get("Accept-Ranges") == "bytes"


# ============================================================================
# 3. RATE LIMITER HEADER SPOOFING ADVERSARIAL TESTS (P1.6 Proxy Hardening)
# ============================================================================

class TestRateLimiterHeaderSpoofingAdversarialHarness:
    """Stress tests proxy-aware client IP resolution against spoofed X-Forwarded-For headers."""

    def test_untrusted_direct_socket_cycling_spoofed_xff_is_rate_limited(self):
        """
        Adversary connects from untrusted socket IP and cycles a different spoofed
        X-Forwarded-For header on every request.
        The rate limiter MUST track requests against the direct socket IP and return HTTP 429.
        """
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=3,
            window_seconds=60,
            enabled=True,
            trusted_proxies={"127.0.0.1", "10.0.0.1"},  # Only these two are trusted
        )

        @app.get("/api/test-quota")
        async def endpoint():
            return {"status": "ok"}

        # TestClient uses request.client.host = "testclient" by default
        # Since "testclient" is NOT in {"127.0.0.1", "10.0.0.1"}, it is an untrusted direct socket!
        client = TestClient(app)

        # 3 allowed requests with spoofed IPs
        for i in range(3):
            spoofed_ip = f"198.51.100.{i+1}"
            r = client.get("/api/test-quota", headers={"X-Forwarded-For": spoofed_ip})
            assert r.status_code == 200, f"Request {i+1} failed"

        # 4th request with yet another new spoofed IP: MUST be blocked (429)
        r_blocked = client.get("/api/test-quota", headers={"X-Forwarded-For": "203.0.113.99"})
        assert r_blocked.status_code == 429
        data = r_blocked.json()
        assert data["error"]["code"] == "RATE_LIMITED"
        assert "Retry-After" in r_blocked.headers

    def test_untrusted_socket_attempting_localhost_impersonation_in_xff(self):
        """
        Adversary from untrusted socket sends X-Forwarded-For: 127.0.0.1 to attempt
        bypassing rate limits by impersonating localhost.
        Rate limiter must track against the socket IP, not the header.
        """
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=2,
            window_seconds=60,
            enabled=True,
            trusted_proxies={"10.0.0.1"},  # Only 10.0.0.1 is trusted proxy
        )

        @app.get("/api/test-quota")
        async def endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Request 1 & 2 impersonating localhost
        r1 = client.get("/api/test-quota", headers={"X-Forwarded-For": "127.0.0.1"})
        assert r1.status_code == 200
        r2 = client.get("/api/test-quota", headers={"X-Forwarded-For": "127.0.0.1"})
        assert r2.status_code == 200

        # Request 3 must be rate-limited
        r3 = client.get("/api/test-quota", headers={"X-Forwarded-For": "127.0.0.1"})
        assert r3.status_code == 429

    def test_untrusted_socket_with_comma_separated_and_garbage_xff(self):
        """
        Adversary sends multi-hop X-Forwarded-For or garbage strings.
        Must safely fall back to direct socket host and enforce rate limit.
        """
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=2,
            window_seconds=60,
            enabled=True,
            trusted_proxies={"192.168.1.1"},
        )

        @app.get("/api/test-quota")
        async def endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Multi-hop string
        r1 = client.get("/api/test-quota", headers={"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"})
        assert r1.status_code == 200
        # Garbage string
        r2 = client.get("/api/test-quota", headers={"X-Forwarded-For": "../../../etc/shadow"})
        assert r2.status_code == 200

        # 3rd request blocked
        r3 = client.get("/api/test-quota", headers={"X-Forwarded-For": "4.4.4.4"})
        assert r3.status_code == 429

    def test_verified_trusted_proxy_correctly_isolates_downstream_clients(self):
        """
        When direct socket host IS in trusted_proxies (e.g. 'testclient'),
        downstream client IPs from X-Forwarded-For receive distinct quota buckets.
        """
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=2,
            window_seconds=60,
            enabled=True,
            trusted_proxies={"testclient"},  # 'testclient' is verified trusted
        )

        @app.get("/api/test-quota")
        async def endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Client A: exhaust quota (2 allowed)
        r_a1 = client.get("/api/test-quota", headers={"X-Forwarded-For": "198.51.100.10"})
        assert r_a1.status_code == 200
        r_a2 = client.get("/api/test-quota", headers={"X-Forwarded-For": "198.51.100.10"})
        assert r_a2.status_code == 200
        r_a3 = client.get("/api/test-quota", headers={"X-Forwarded-For": "198.51.100.10"})
        assert r_a3.status_code == 429

        # Client B: distinct quota, still has 2 allowed
        r_b1 = client.get("/api/test-quota", headers={"X-Forwarded-For": "198.51.100.20"})
        assert r_b1.status_code == 200
        r_b2 = client.get("/api/test-quota", headers={"X-Forwarded-For": "198.51.100.20"})
        assert r_b2.status_code == 200
        r_b3 = client.get("/api/test-quota", headers={"X-Forwarded-For": "198.51.100.20"})
        assert r_b3.status_code == 429


# ============================================================================
# 4. COMPLETE_JOB GATING ADVERSARIAL STRESS TESTS (P2.3 Queue Reliability)
# ============================================================================

class StateAuditRepository(InMemoryJobStore):
    """
    Subclass that audits every state mutation persisted via update_job
    to prove whether a job ever entered SUCCEEDED during failure paths.
    """
    def __init__(self):
        super().__init__()
        self.recorded_states = []

    async def update_job(self, job: FactoryJob) -> FactoryJob:
        self.recorded_states.append((job.job_id, job.state))
        return await super().update_job(job)


class TestCompleteJobGatingAdversarialHarness:
    """Stress tests complete_job gating against missing artifacts, empty files, and empty non-media payloads."""

    async def _setup_harness(self, tmp_path):
        repo = StateAuditRepository()
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
    async def test_media_job_missing_artifact_path_transitions_to_failed_never_succeeded(self, tmp_path):
        """
        When complete_job specifies a non-existent artifact_path:
        - Job transitions directly to FAILED.
        - Job NEVER enters SUCCEEDED in memory or database.
        - Error code is ARTIFACT_MISSING.
        - Allocated GPU is released.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"
        worker_id = "worker-adv-01"
        lease_token = "lease-token-adv-01"

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "cyberpunk city"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="rendering", percent=90.0),
            worker=WorkerLease(worker_id=worker_id, gpu_device=0, lease_token=lease_token),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifact_path="/tmp/non_existent_ghost_file.png",
            inference_duration_ms=1500.0,
        )

        res = await manager.complete_job(job_id, req, caller_worker_id=worker_id)

        # Assert terminal FAILED state
        assert res.state == JobState.FAILED
        assert res.error is not None
        assert res.error.code == ErrorCode.ARTIFACT_MISSING
        assert not res.artifacts

        # Assert database persisted state is FAILED
        db_job = await repo.get_job(job_id)
        assert db_job.state == JobState.FAILED

        # Assert that SUCCEEDED was NEVER recorded at any point
        job_states = [s for jid, s in repo.recorded_states if jid == job_id]
        assert JobState.SUCCEEDED not in job_states, f"Job entered SUCCEEDED state! Observed: {job_states}"

        # Assert GPU 0 was freed
        assert not broker.is_gpu_busy(0)

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_media_job_empty_zero_byte_artifact_file_fails(self, tmp_path):
        """
        When complete_job points to a 0-byte empty file:
        - Job transitions directly to FAILED with ARTIFACT_MISSING.
        - Job NEVER enters SUCCEEDED.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"
        worker_id = "worker-adv-02"
        lease_token = "lease-token-adv-02"

        # Create empty 0-byte file on disk
        empty_file = tmp_path / "empty_output.png"
        empty_file.write_bytes(b"")
        assert empty_file.stat().st_size == 0

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "empty output probe"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="rendering", percent=90.0),
            worker=WorkerLease(worker_id=worker_id, gpu_device=0, lease_token=lease_token),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifact_path=str(empty_file),
            inference_duration_ms=1000.0,
        )

        res = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.ARTIFACT_MISSING
        assert "empty" in res.error.message.lower()

        job_states = [s for jid, s in repo.recorded_states if jid == job_id]
        assert JobState.SUCCEEDED not in job_states

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_media_job_artifacts_list_with_empty_file_fails(self, tmp_path):
        """
        When req.artifacts provides an ArtifactReference pointing to a 0-byte file,
        the manager must reject it with FAILED.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"
        worker_id = "worker-adv-03"
        lease_token = "lease-token-adv-03"

        empty_file = tmp_path / "empty_in_artifacts.mp4"
        empty_file.write_bytes(b"")

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service="video",
            model="minimax-h3",
            task="text_to_video",
            inputs={"prompt": "video prompt"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="rendering", percent=90.0),
            worker=WorkerLease(worker_id=worker_id, gpu_device=0, lease_token=lease_token),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifacts=[
                ArtifactReference(
                    artifact_id=f"art-{uuid.uuid4()}",
                    type="video",
                    mime_type="video/mp4",
                    size_bytes=0,
                    storage_path=str(empty_file),
                    sha256="0" * 64,
                    download_url=f"/v1/jobs/{job_id}/result",
                )
            ],
            inference_duration_ms=2000.0,
        )

        res = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.ARTIFACT_MISSING
        assert "empty" in res.error.message.lower()

        job_states = [s for jid, s in repo.recorded_states if jid == job_id]
        assert JobState.SUCCEEDED not in job_states

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_media_job_forged_sha256_mismatch_fails(self, tmp_path):
        """
        When req.artifacts specifies a sha256 that does not match the actual on-disk file,
        job must fail with ARTIFACT_MISSING checksum mismatch.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"
        worker_id = "worker-adv-04"
        lease_token = "lease-token-adv-04"

        real_file = tmp_path / "actual_video.mp4"
        real_file.write_bytes(b"REAL_VIDEO_STREAM_BYTES_ABCDEF")

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service="video",
            model="minimax-h3",
            task="text_to_video",
            inputs={"prompt": "video prompt"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="rendering", percent=90.0),
            worker=WorkerLease(worker_id=worker_id, gpu_device=0, lease_token=lease_token),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        # Claim a fabricated SHA-256 hash
        fake_sha256 = "0123456789abcdef" * 4
        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifacts=[
                ArtifactReference(
                    artifact_id=f"art-{uuid.uuid4()}",
                    type="video",
                    mime_type="video/mp4",
                    size_bytes=real_file.stat().st_size,
                    storage_path=str(real_file),
                    sha256=fake_sha256,
                    download_url=f"/v1/jobs/{job_id}/result",
                )
            ],
            inference_duration_ms=2500.0,
        )

        res = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.ARTIFACT_MISSING
        assert "mismatch" in res.error.message.lower()

        job_states = [s for jid, s in repo.recorded_states if jid == job_id]
        assert JobState.SUCCEEDED not in job_states

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.parametrize(
        "service,task,empty_result,desc",
        [
            ("text", "text_generation", None, "None result"),
            ("text", "text_generation", "", "Empty string result"),
            ("text", "text_generation", [], "Empty list result"),
            ("text", "text_generation", {}, "Empty dict result"),
            ("decision", "classification", None, "Decision None result"),
            ("decision", "classification", "", "Decision empty string result"),
            ("embedding", "text_embedding", None, "Embedding None result"),
            ("embedding", "text_embedding", [], "Embedding empty list result"),
        ],
    )
    @pytest.mark.asyncio
    async def test_non_media_job_empty_result_fails_directly(self, tmp_path, service, task, empty_result, desc):
        """
        Non-media jobs (text, decision, embedding) completed with missing or empty result
        payloads must transition directly to FAILED and NEVER enter SUCCEEDED.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"
        worker_id = f"worker-{service}-01"
        lease_token = f"lease-{service}-01"

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service=service,
            model="test-model",
            task=task,
            inputs={"prompt": "test non-media input"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="inference", percent=90.0),
            worker=WorkerLease(worker_id=worker_id, gpu_device=0, lease_token=lease_token),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            result=empty_result,
            inference_duration_ms=450.0,
        )

        res = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.INVALID_REQUEST
        assert "without result payload" in res.error.message.lower()

        job_states = [s for jid, s in repo.recorded_states if jid == job_id]
        assert JobState.SUCCEEDED not in job_states, f"Job entered SUCCEEDED for {desc}!"

        # Assert GPU was cleanly released
        assert not broker.is_gpu_busy(0)

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_media_job_completely_missing_artifacts_fails_directly(self, tmp_path):
        """
        When complete_job is called on a media job with no artifacts and no result payload,
        it must transition directly to FAILED and NEVER enter SUCCEEDED.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"
        worker_id = "worker-adv-05"
        lease_token = "lease-token-adv-05"

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "image without artifacts"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="rendering", percent=90.0),
            worker=WorkerLease(worker_id=worker_id, gpu_device=0, lease_token=lease_token),
        )
        await repo.create_job(job)
        broker.acquire_gpu(0, job_id)

        req = JobCompletionRequest(
            worker_id=worker_id,
            lease_token=lease_token,
            artifacts=[],
            artifact_path=None,
            result=None,
            inference_duration_ms=1000.0,
        )

        res = await manager.complete_job(job_id, req, caller_worker_id=worker_id)
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.ARTIFACT_MISSING
        assert "completed without required output artifact" in res.error.message.lower()

        job_states = [s for jid, s in repo.recorded_states if jid == job_id]
        assert JobState.SUCCEEDED not in job_states

        await scratch.shutdown()
        await broker.close()
        await repo.close()

    @pytest.mark.asyncio
    async def test_complete_job_unleased_worker_raises_forbidden_and_preserves_state(self, tmp_path):
        """
        Attempting to complete a job with mismatched or missing lease credentials
        must be rejected with ForbiddenError/403 and never alter the job state.
        """
        repo, broker, manager, scratch = await self._setup_harness(tmp_path)
        job_id = f"job-{uuid.uuid4()}"

        job = FactoryJob(
            job_id=job_id,
            request_id=f"req-{uuid.uuid4()}",
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test"},
            state=JobState.RUNNING,
            progress=JobProgress(stage="rendering", percent=50.0),
            worker=WorkerLease(worker_id="legit-worker", gpu_device=0, lease_token="valid-token-1234"),
        )
        await repo.create_job(job)

        req = JobCompletionRequest(
            worker_id="attacker-worker",
            lease_token="forged-token-5678",
            artifact_path="/tmp/output.png",
        )

        with pytest.raises(Exception) as exc_info:
            await manager.complete_job(job_id, req, caller_worker_id="attacker-worker")

        # State must remain untouched (RUNNING)
        persisted = await repo.get_job(job_id)
        assert persisted.state == JobState.RUNNING

        await scratch.shutdown()
        await broker.close()
        await repo.close()

