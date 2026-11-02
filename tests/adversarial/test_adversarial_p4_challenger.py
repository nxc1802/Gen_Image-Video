"""
tests/adversarial/test_adversarial_p4_challenger.py
Adversarial Stress Test Suite for Milestone 5 (Phase P4: Production Hardening & Full Release Gate).

Adversarially probes:
1. Strict JWT Verification & Algorithm Allowlists:
   - Exhaustive algorithm matrix: rejection of none, NONE, None, HS384, HS512, RS384, RS512, ES256, etc.
   - Header vs decode algorithm confusion & forgery attacks.
   - Expiration claims, expired tokens, missing exp, null/string exp, future-dated nbf/iat.
   - Strict issuer matching & audience allowlists.
   - Tampered signatures, bit flips, malformed base64, SQL/XSS injection in claims.
   - Bearer token dependency matrix & error sanitization.
2. Worker Identity Binding & Authorization Boundaries:
   - Role matrix: worker vs admin vs user/operator/guest.
   - Token identity binding: sub vs worker_id vs missing claims.
   - Worker lease endpoint impersonation & parameter tampering (/jobs/lease).
   - Worker heartbeat URL path vs payload mismatch attacks (/workers/{id}/heartbeat).
   - Cross-worker lease progress/complete/fail state mutation attacks.
   - Stolen lease tokens, expired lease state, unleased jobs protection.
3. In-App Rate Limiting Middleware:
   - Burst exhaustion and strict 429 RATE_LIMITED envelope verification.
   - Retry-After header and sliding-window quota reset after window expiration.
   - Client key isolation (token hash vs IP address) and token priority over exhausted IP.
   - Path exemptions (/health, /docs, /openapi.json) and custom per-path limits.
   - High concurrency race conditions (exact limit winners under concurrent load).
"""
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import time
from typing import Dict, List, Optional
import uuid

import httpx
import jwt
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from factory_auth import (
    ALLOWED_ALGORITHMS,
    AuthConfig,
    decode_jwt_token,
    load_auth_config,
    verify_bearer_token,
)
from factory_contracts.errors import ForbiddenError, UnauthorizedError
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    JobTelemetry,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
)
from factory_contracts.errors import ErrorDetail
from factory.apps.control_api.src.api.middleware.rate_limit import RateLimitMiddleware
from factory.apps.control_api.src.api.routers.workers import (
    router as workers_router,
    verify_worker_identity,
)
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.streaming.manager import StreamManager

JWT_SECRET = "test-jwt-secret-key-32-chars-long!"
DEFAULT_ISSUER = "factory-control-api"
DEFAULT_AUDIENCE = "factory-v4"


def make_jwt(
    payload: Dict,
    secret: str = JWT_SECRET,
    algorithm: str = "HS256",
    headers: Optional[Dict] = None,
) -> str:
    """Helper to generate JWT tokens with arbitrary headers/payloads for adversarial testing."""
    return jwt.encode(payload, secret, algorithm=algorithm, headers=headers)


# ===========================================================================
# 1. Strict JWT Verification & Cryptographic Attack Probes
# ===========================================================================


class TestJwtStrictVerificationAdversarial:
    """Adversarial challenge suite for packages/auth/src/factory_auth/jwt_auth.py."""

    @pytest.mark.parametrize(
        "disallowed_alg",
        [
            "none",
            "NONE",
            "None",
            "nOnE",
            "HS384",
            "HS512",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
            "PS256",
            "PS384",
            "PS512",
            "",
            "UNKNOWN_ALG",
            "hs256",  # Case-sensitive allowlist check
        ],
    )
    def test_disallowed_algorithms_strictly_rejected(self, disallowed_alg: str):
        """
        Adversarial: Test that any algorithm outside {'HS256', 'RS256'} is rejected with UnauthorizedError.
        """
        now = int(time.time())
        token = make_jwt({"sub": "victim", "exp": now + 3600}, algorithm="HS256")
        with pytest.raises(UnauthorizedError, match="Disallowed or unsupported JWT algorithm"):
            decode_jwt_token(token, JWT_SECRET, algorithm=disallowed_alg)

    def test_algorithm_confusion_header_none_against_hs256_verifier(self):
        """
        Adversarial: Attack where token header declares alg='none' without signature,
        while verifier is configured with algorithm='HS256'.
        Must fail signature/algorithm verification with UnauthorizedError.
        """
        # Create unsigned token with alg="none"
        token = jwt.encode(
            {"sub": "attacker", "exp": int(time.time()) + 3600, "role": "admin"},
            key="",
            algorithm="none",
        )
        with pytest.raises(UnauthorizedError) as exc_info:
            decode_jwt_token(token, JWT_SECRET, algorithm="HS256")
        assert "Invalid JWT" in str(exc_info.value) or "algorithm" in str(exc_info.value).lower()

    def test_algorithm_confusion_header_hs512_against_hs256_verifier(self):
        """
        Adversarial: Token signed with HS512 passed to HS256 verifier.
        """
        now = int(time.time())
        token = jwt.encode(
            {"sub": "attacker", "exp": now + 3600, "role": "admin"},
            JWT_SECRET,
            algorithm="HS512",
        )
        with pytest.raises(UnauthorizedError):
            decode_jwt_token(token, JWT_SECRET, algorithm="HS256")

    @pytest.mark.parametrize(
        "exp_delta,should_succeed",
        [
            (-3600, False),  # 1 hour expired
            (-60, False),    # 1 minute expired
            (-1, False),     # 1 second expired
            (60, True),      # 1 minute valid
            (3600, True),    # 1 hour valid
        ],
    )
    def test_token_expiration_delta_boundaries(self, exp_delta: int, should_succeed: bool):
        now = int(time.time())
        payload = {"sub": "user-boundary", "exp": now + exp_delta}
        token = make_jwt(payload)

        if should_succeed:
            decoded = decode_jwt_token(token, JWT_SECRET, algorithm="HS256")
            assert decoded["sub"] == "user-boundary"
        else:
            with pytest.raises(UnauthorizedError, match="expired"):
                decode_jwt_token(token, JWT_SECRET, algorithm="HS256")

    def test_missing_mandatory_exp_claim_rejected(self):
        payload = {"sub": "user-no-exp", "role": "worker"}
        token = make_jwt(payload)
        with pytest.raises(UnauthorizedError, match="missing required claim 'exp'"):
            decode_jwt_token(token, JWT_SECRET, algorithm="HS256")

    def test_future_dated_not_before_claim_rejected(self):
        """
        Adversarial: Token with 'nbf' (not before) in future must be rejected.
        """
        now = int(time.time())
        payload = {
            "sub": "user-future",
            "exp": now + 3600,
            "nbf": now + 1800,  # Valid only 30m in future
        }
        token = make_jwt(payload)
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(token, JWT_SECRET, algorithm="HS256")

    def test_issuer_validation_boundaries(self):
        now = int(time.time())
        # Valid issuer
        valid_tok = make_jwt({"sub": "user-iss", "exp": now + 3600, "iss": DEFAULT_ISSUER})
        decoded = decode_jwt_token(valid_tok, JWT_SECRET, algorithm="HS256", issuer=DEFAULT_ISSUER)
        assert decoded["iss"] == DEFAULT_ISSUER

        # Mismatched issuer
        rogue_tok = make_jwt({"sub": "user-iss", "exp": now + 3600, "iss": "rogue-authority"})
        with pytest.raises(UnauthorizedError, match="Invalid JWT issuer"):
            decode_jwt_token(rogue_tok, JWT_SECRET, algorithm="HS256", issuer=DEFAULT_ISSUER)

        # Missing issuer when expected
        missing_iss_tok = make_jwt({"sub": "user-iss", "exp": now + 3600})
        with pytest.raises(UnauthorizedError):
            decode_jwt_token(missing_iss_tok, JWT_SECRET, algorithm="HS256", issuer=DEFAULT_ISSUER)

    def test_audience_validation_boundaries(self):
        now = int(time.time())
        # Valid audience (string)
        tok = make_jwt({"sub": "user-aud", "exp": now + 3600, "aud": DEFAULT_AUDIENCE})
        decoded = decode_jwt_token(tok, JWT_SECRET, algorithm="HS256", audience=DEFAULT_AUDIENCE)
        assert decoded["aud"] == DEFAULT_AUDIENCE

        # Mismatched audience
        wrong_aud_tok = make_jwt({"sub": "user-aud", "exp": now + 3600, "aud": "wrong-service"})
        with pytest.raises(UnauthorizedError, match="Invalid JWT audience"):
            decode_jwt_token(wrong_aud_tok, JWT_SECRET, algorithm="HS256", audience=DEFAULT_AUDIENCE)

        # Missing audience when required
        no_aud_tok = make_jwt({"sub": "user-aud", "exp": now + 3600})
        with pytest.raises(UnauthorizedError):
            decode_jwt_token(no_aud_tok, JWT_SECRET, algorithm="HS256", audience=DEFAULT_AUDIENCE)

        # Audience list containing valid target
        multi_aud_tok = make_jwt({"sub": "user-aud", "exp": now + 3600, "aud": [DEFAULT_AUDIENCE, "billing"]})
        decoded_multi = decode_jwt_token(multi_aud_tok, JWT_SECRET, algorithm="HS256", audience=DEFAULT_AUDIENCE)
        assert DEFAULT_AUDIENCE in decoded_multi["aud"]

    def test_tampered_payload_and_corrupted_signature(self):
        now = int(time.time())
        valid_tok = make_jwt({"sub": "user-victim", "role": "worker", "exp": now + 3600})
        parts = valid_tok.split(".")
        assert len(parts) == 3

        # Tamper payload (replace payload part with modified base64)
        tampered_tok = f"{parts[0]}.eyAic3ViIjogImF0dGFja2VyIiwgInJvbGUiOiAiYWRtaW4iIH0.{parts[2]}"
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(tampered_tok, JWT_SECRET, algorithm="HS256")

        # Tamper signature (flip last char)
        corrupted_sig = parts[2][:-1] + ("X" if parts[2][-1] != "X" else "Y")
        corrupted_tok = f"{parts[0]}.{parts[1]}.{corrupted_sig}"
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(corrupted_tok, JWT_SECRET, algorithm="HS256")

    def test_wrong_secret_key_rejection(self):
        now = int(time.time())
        token = make_jwt({"sub": "user", "exp": now + 3600}, secret="wrong-secret-key-that-does-not-match!")
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(token, JWT_SECRET, algorithm="HS256")

    def test_injection_strings_in_claims_decoded_safely(self):
        now = int(time.time())
        payload = {
            "sub": "'; DROP TABLE jobs; --",
            "role": "<script>alert('xss')</script>",
            "worker_id": "../../etc/passwd",
            "exp": now + 3600,
        }
        token = make_jwt(payload)
        decoded = decode_jwt_token(token, JWT_SECRET, algorithm="HS256")
        assert decoded["sub"] == "'; DROP TABLE jobs; --"
        assert decoded["role"] == "<script>alert('xss')</script>"
        assert decoded["worker_id"] == "../../etc/passwd"


# ===========================================================================
# 2. Worker Identity Binding & Authorization Boundaries
# ===========================================================================


class TestWorkerIdentityBindingAdversarial:
    """Adversarial challenge suite for worker identity binding and router controls."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user,expected_error",
        [
            ({}, "require role 'worker' or 'admin'"),
            ({"role": "user"}, "require role 'worker' or 'admin'"),
            ({"role": "operator"}, "require role 'worker' or 'admin'"),
            ({"role": "guest"}, "require role 'worker' or 'admin'"),
            ({"role": "worker"}, "must contain 'sub' or 'worker_id'"),
            ({"role": "worker", "sub": ""}, "must contain 'sub' or 'worker_id'"),
            ({"role": "worker", "sub": None, "worker_id": ""}, "must contain 'sub' or 'worker_id'"),
        ],
    )
    async def test_verify_worker_identity_unauthorized_roles_and_claims(
        self, user: Dict, expected_error: str
    ):
        with pytest.raises(ForbiddenError, match=expected_error):
            await verify_worker_identity(user)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user,expected_sub",
        [
            ({"role": "worker", "sub": "gpu-worker-1"}, "gpu-worker-1"),
            ({"role": "worker", "worker_id": "gpu-worker-2"}, None),
            ({"role": "admin", "sub": "admin-control"}, "admin-control"),
            ({"role": "ADMIN", "sub": "admin-caps"}, "admin-caps"),
            ({"role": "WORKER", "sub": "worker-caps"}, "worker-caps"),
        ],
    )
    async def test_verify_worker_identity_valid_roles_and_claims(
        self, user: Dict, expected_sub: Optional[str]
    ):
        result = await verify_worker_identity(user)
        assert result["role"].lower() in {"worker", "admin"}
        if expected_sub:
            assert result.get("sub") == expected_sub

    @pytest.mark.asyncio
    async def test_worker_lease_and_cross_worker_mutation_matrix(self):
        """
        Adversarial Challenge: Comprehensive lifecycle test for cross-worker state mutations:
        1. Worker A leases Job 1.
        2. Worker B attempts to hijack Job 1:
           - Update progress with Worker B identity
           - Update progress with spoofed Worker A identity in body
           - Complete Job 1 with stolen lease token
           - Fail Job 1
        3. Expired lease operations blocked.
        4. Valid Worker A operations succeed.
        """
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)

        try:
            # 1. Submit a test job
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Adversarial cross-worker test"},
                ),
                request_id="req-adv-p4",
            )
            job_id = job.job_id

            # 2. Worker A leases Job 1
            lease_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-A",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            assert lease_resp.job is not None
            assert lease_resp.job.job_id == job_id
            token_A = lease_resp.lease_token

            # 3. Cross-worker attacks by Worker B

            # Attack 3a: Worker B tries update_progress with caller_worker_id="worker-B"
            with pytest.raises(UnauthorizedError, match="does not hold active lease") as exc:
                await manager.update_progress(
                    job_id,
                    ProgressUpdateRequest(
                        worker_id="worker-B",
                        lease_token=token_A,
                        stage="hijacked",
                        step=5,
                        total=20,
                        percent=25.0,
                    ),
                    caller_worker_id="worker-B",
                    caller_role="worker",
                )
            assert exc.value.status_code == 403

            # Attack 3b: Worker B spoofs caller_worker_id vs req.worker_id
            with pytest.raises(UnauthorizedError, match="not authorized to act as worker"):
                await manager.update_progress(
                    job_id,
                    ProgressUpdateRequest(
                        worker_id="worker-A",
                        lease_token=token_A,
                        stage="spoofed",
                        step=5,
                        total=20,
                        percent=25.0,
                    ),
                    caller_worker_id="worker-B",
                    caller_role="worker",
                )

            def make_test_artifact(art_id: str) -> ArtifactReference:
                return ArtifactReference(
                    artifact_id=art_id,
                    type="image",
                    mime_type="image/png",
                    size_bytes=1024,
                    storage_path=f"artifacts/{art_id}.png",
                    url=f"https://storage.factory.internal/artifacts/{art_id}.png",
                    sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                )

            # Attack 3c: Worker B tries complete_job with stolen token_A
            with pytest.raises(UnauthorizedError, match="does not hold active lease"):
                await manager.complete_job(
                    job_id,
                    JobCompletionRequest(
                        worker_id="worker-B",
                        lease_token=token_A,
                        artifacts=[make_test_artifact("art-forged")],
                    ),
                    caller_worker_id="worker-B",
                    caller_role="worker",
                )

            # Attack 3d: Worker B tries fail_job
            with pytest.raises(UnauthorizedError):
                await manager.fail_job(
                    job_id,
                    JobFailureRequest(
                        worker_id="worker-B",
                        lease_token=token_A,
                        error=ErrorDetail(code="INTERNAL_ERROR", message="Tampered", request_id="req-adv-p4"),
                    ),
                    caller_worker_id="worker-B",
                    caller_role="worker",
                )

            # 4. Legitimate Worker A updates progress
            updated_job = await manager.update_progress(
                job_id,
                ProgressUpdateRequest(
                    worker_id="worker-A",
                    lease_token=token_A,
                    stage="denoising",
                    step=10,
                    total=20,
                    percent=50.0,
                    message="Halfway complete",
                ),
                caller_worker_id="worker-A",
                caller_role="worker",
            )
            assert updated_job.progress.percent == 50.0

            # 5. Legitimate Worker A completes job
            completed_job = await manager.complete_job(
                job_id,
                JobCompletionRequest(
                    worker_id="worker-A",
                    lease_token=token_A,
                    artifacts=[make_test_artifact("art-valid")],
                    inference_duration_ms=450.0,
                ),
                caller_worker_id="worker-A",
                caller_role="worker",
            )
            assert completed_job.state == JobState.SUCCEEDED
            assert len(completed_job.artifacts) == 1
            assert completed_job.artifacts[0].artifact_id == "art-valid"

            # 6. Expired lease test: create a new job and simulate lease expiration
            exp_job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Expired lease test"},
                ),
                request_id="req-exp-lease",
            )
            exp_resp = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-exp",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            exp_token = exp_resp.lease_token

            # Manually expire the lease in repository
            fetched_job = await repo.get_job(exp_job.job_id)
            fetched_job.worker.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
            await repo.update_job(fetched_job)

            # Attempt progress on expired lease
            with pytest.raises(UnauthorizedError, match="expired"):
                await manager.update_progress(
                    exp_job.job_id,
                    ProgressUpdateRequest(
                        worker_id="worker-exp",
                        lease_token=exp_token,
                        stage="too_late",
                        step=1,
                        total=10,
                        percent=10.0,
                    ),
                    caller_worker_id="worker-exp",
                    caller_role="worker",
                )

        finally:
            await repo.close()


# ===========================================================================
# 3. In-App Rate Limiting Middleware Adversarial Probes
# ===========================================================================


class TestRateLimitMiddlewareAdversarial:
    """Adversarial challenge suite for sliding-window RateLimitMiddleware."""

    def _create_app(
        self,
        default_limit: int = 5,
        window_seconds: int = 60,
        exempt_paths: Optional[set] = None,
        path_limits: Optional[Dict[str, int]] = None,
    ):
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=default_limit,
            window_seconds=window_seconds,
            exempt_paths=exempt_paths,
            path_limits=path_limits,
            enabled=True,
        )

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/api/standard")
        async def standard(request: Request):
            return {"status": "standard"}

        @app.get("/api/strict")
        async def strict(request: Request):
            return {"status": "strict"}

        return app

    def test_burst_exhaustion_and_429_envelope_verification(self):
        """
        Adversarial: Fire burst exceeding limit=4.
        Verify 4 requests return 200, 5th request returns 429 with sanitized error envelope and Retry-After.
        """
        app = self._create_app(default_limit=4, window_seconds=30)
        client = TestClient(app)
        headers = {"Authorization": "Bearer token-burst-test", "X-Request-ID": "req-p4-rate-test"}

        # 4 requests pass
        for i in range(4):
            res = client.get("/api/standard", headers=headers)
            assert res.status_code == 200
            assert res.headers["X-RateLimit-Limit"] == "4"
            assert int(res.headers["X-RateLimit-Remaining"]) == 3 - i

        # 5th request triggers 429
        res = client.get("/api/standard", headers=headers)
        assert res.status_code == 429
        assert res.headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in res.headers
        retry_after = int(res.headers["Retry-After"])
        assert 1 <= retry_after <= 30

        body = res.json()
        assert body["error"]["code"] == "RATE_LIMITED"
        assert body["error"]["request_id"] == "req-p4-rate-test"
        assert body["error"]["details"]["limit"] == 4
        assert body["error"]["details"]["window_seconds"] == 30

    def test_window_reset_and_quota_recovery(self):
        """
        Adversarial: After window_seconds expires, quota must fully recover.
        """
        # 1-second window
        app = self._create_app(default_limit=2, window_seconds=1)
        client = TestClient(app)
        headers = {"Authorization": "Bearer token-window-test"}

        # Consume 2 requests
        assert client.get("/api/standard", headers=headers).status_code == 200
        assert client.get("/api/standard", headers=headers).status_code == 200

        # Blocked
        assert client.get("/api/standard", headers=headers).status_code == 429

        # Sleep past the 1-second sliding window
        time.sleep(1.1)

        # Quota recovered!
        rec_res = client.get("/api/standard", headers=headers)
        assert rec_res.status_code == 200
        assert int(rec_res.headers["X-RateLimit-Remaining"]) == 1

    def test_client_key_isolation_and_token_priority_over_ip(self):
        """
        Adversarial: Verify client isolation:
        - Exhausting IP quota does not block authenticated Bearer token from same IP.
        - Exhausting Token A does not block Token B.
        """
        app = self._create_app(default_limit=2, window_seconds=30)
        client = TestClient(app)

        shared_ip_header = {"X-Forwarded-For": "203.0.113.42"}

        # Exhaust IP quota anonymously
        assert client.get("/api/standard", headers=shared_ip_header).status_code == 200
        assert client.get("/api/standard", headers=shared_ip_header).status_code == 200
        assert client.get("/api/standard", headers=shared_ip_header).status_code == 429

        # An authenticated token from the SAME IP is evaluated by token key and must succeed
        auth_from_same_ip = {
            "Authorization": "Bearer auth-token-vip",
            "X-Forwarded-For": "203.0.113.42",
        }
        vip_res = client.get("/api/standard", headers=auth_from_same_ip)
        assert vip_res.status_code == 200

    def test_custom_per_path_limits(self):
        """
        Adversarial: Path /api/strict has limit=1, while default is 5.
        """
        app = self._create_app(
            default_limit=5,
            window_seconds=30,
            path_limits={"/api/strict": 1},
        )
        client = TestClient(app)
        headers = {"Authorization": "Bearer token-path-limits"}

        # 1 call to strict passes, 2nd call fails
        assert client.get("/api/strict", headers=headers).status_code == 200
        assert client.get("/api/strict", headers=headers).status_code == 429

        # Default path is still healthy (has separate limit of 5)
        assert client.get("/api/standard", headers=headers).status_code == 200

    def test_exempt_paths_never_throttled(self):
        """
        Adversarial: /health endpoint must withstand 20 consecutive requests without throttling.
        """
        app = self._create_app(default_limit=2, window_seconds=60)
        client = TestClient(app)

        for _ in range(20):
            res = client.get("/health")
            assert res.status_code == 200

    @pytest.mark.asyncio
    async def test_high_concurrency_race_condition(self):
        """
        Adversarial: 30 concurrent requests competing for a limit of 5.
        Invariants:
        - Exactly 5 requests succeed (200 OK).
        - Exactly 25 requests are throttled (429 RATE_LIMITED).
        - No race conditions in internal sliding window deque.
        """
        app = self._create_app(default_limit=5, window_seconds=60)
        transport = httpx.ASGITransport(app=app)
        headers = {"Authorization": "Bearer token-concurrent-race"}

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            tasks = [async_client.get("/api/standard", headers=headers) for _ in range(30)]
            responses = await asyncio.gather(*tasks)

        status_codes = [r.status_code for r in responses]
        successes = status_codes.count(200)
        throttled = status_codes.count(429)

        assert successes == 5, f"Expected exactly 5 successes, got {successes}"
        assert throttled == 25, f"Expected exactly 25 throttled, got {throttled}"


# ===========================================================================
# 4. Worker Router Endpoints HTTP Adversarial Probes
# ===========================================================================


class TestWorkerRouterEndpointsAdversarial:
    """Full HTTP-level adversarial testing of factory/apps/control-api/src/api/routers/workers.py."""

    @pytest.fixture
    def test_setup(self):
        from fastapi import FastAPI
        from factory.apps.control_api.src.api.errors import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        repo = SqliteJobRepository(db_path=":memory:")
        asyncio.run(repo.initialize())
        broker = MemoryQueueBroker()
        asyncio.run(broker.initialize())
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=broker)

        app.state.lease_coordinator = coordinator
        app.state.job_manager = manager
        app.state.auth_config = AuthConfig(
            api_key="system-api-key",
            jwt_secret=JWT_SECRET,
            environment="test",
            jwt_algorithm="HS256",
            jwt_audience=DEFAULT_AUDIENCE,
            jwt_issuer=DEFAULT_ISSUER,
        )
        app.include_router(workers_router)
        client = TestClient(app)

        yield {
            "app": app,
            "client": client,
            "repo": repo,
            "broker": broker,
            "manager": manager,
            "coordinator": coordinator,
        }

        asyncio.run(repo.close())

    def _make_auth_header(self, role: str = "worker", sub: str = "worker-1") -> Dict[str, str]:
        now = int(time.time())
        token = make_jwt(
            {
                "sub": sub,
                "role": role,
                "worker_id": sub,
                "exp": now + 3600,
                "iss": DEFAULT_ISSUER,
                "aud": DEFAULT_AUDIENCE,
            }
        )
        return {"Authorization": f"Bearer {token}"}

    def test_http_lease_unauthenticated_rejected(self, test_setup):
        client = test_setup["client"]
        res = client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-1",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
            },
        )
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

    def test_http_lease_non_worker_role_rejected(self, test_setup):
        client = test_setup["client"]
        headers = self._make_auth_header(role="user", sub="user-123")
        res = client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "user-123",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
            },
            headers=headers,
        )
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "FORBIDDEN"
        assert "require role 'worker' or 'admin'" in res.json()["error"]["message"]

    def test_http_lease_identity_spoofing_rejected(self, test_setup):
        """Worker 1 tries to lease as Worker 2."""
        client = test_setup["client"]
        headers = self._make_auth_header(role="worker", sub="worker-1")
        res = client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-2",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
            },
            headers=headers,
        )
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "FORBIDDEN"
        assert "not authorized to lease jobs as worker 'worker-2'" in res.json()["error"]["message"]

    def test_http_heartbeat_path_mismatch_rejected(self, test_setup):
        """Worker 1 calls heartbeat for worker-2."""
        client = test_setup["client"]
        headers = self._make_auth_header(role="worker", sub="worker-1")
        res = client.post(
            "/v1/workers/worker-2/heartbeat",
            json={"worker_id": "worker-2"},
            headers=headers,
        )
        assert res.status_code == 403
        assert "not authorized to act as worker 'worker-2'" in res.json()["error"]["message"]

    def test_http_heartbeat_body_mismatch_rejected(self, test_setup):
        """Worker 1 calls heartbeat with worker-2 in payload."""
        client = test_setup["client"]
        headers = self._make_auth_header(role="worker", sub="worker-1")
        res = client.post(
            "/v1/workers/worker-1/heartbeat",
            json={"worker_id": "worker-2"},
            headers=headers,
        )
        assert res.status_code == 403
        assert "cannot spoof worker_id 'worker-2'" in res.json()["error"]["message"]

    def test_http_cross_worker_mutation_rejection(self, test_setup):
        """Submit a job, lease to worker-1, then worker-2 tries to progress, complete, fail."""
        client = test_setup["client"]
        manager = test_setup["manager"]

        # Submit job
        job = asyncio.run(
            manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "HTTP cross-worker test"},
                ),
                request_id="req-http-adv",
            )
        )[0]
        job_id = job.job_id

        # Worker 1 leases job
        lease_res = client.post(
            "/v1/jobs/lease",
            json={
                "worker_id": "worker-1",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
            },
            headers=self._make_auth_header(role="worker", sub="worker-1"),
        )
        assert lease_res.status_code == 200
        token_1 = lease_res.json()["lease_token"]

        # Worker 2 attempts progress (spoofing worker-1 in body)
        res = client.post(
            f"/v1/jobs/{job_id}/progress",
            json={
                "worker_id": "worker-1",
                "lease_token": token_1,
                "stage": "hack",
                "step": 1,
                "total": 10,
                "percent": 10.0,
            },
            headers=self._make_auth_header(role="worker", sub="worker-2"),
        )
        assert res.status_code == 403
        assert "cannot claim worker identity 'worker-1'" in res.json()["error"]["message"]

        # Worker 2 attempts progress (with own worker_id and stolen token)
        res = client.post(
            f"/v1/jobs/{job_id}/progress",
            json={
                "worker_id": "worker-2",
                "lease_token": token_1,
                "stage": "hack",
                "step": 1,
                "total": 10,
                "percent": 10.0,
            },
            headers=self._make_auth_header(role="worker", sub="worker-2"),
        )
        assert res.status_code == 403
        assert "does not hold active lease" in res.json()["error"]["message"]

        # Worker 2 attempts complete with stolen token
        art_payload = {
            "artifact_id": "art-hacked",
            "type": "image",
            "mime_type": "image/png",
            "size_bytes": 100,
            "storage_path": "artifacts/hacked.png",
            "url": "https://storage/hacked.png",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
        res = client.post(
            f"/v1/jobs/{job_id}/complete",
            json={"worker_id": "worker-2", "lease_token": token_1, "artifacts": [art_payload]},
            headers=self._make_auth_header(role="worker", sub="worker-2"),
        )
        assert res.status_code == 403
        assert "does not hold active lease" in res.json()["error"]["message"]

        # Worker 2 attempts fail with stolen token
        res = client.post(
            f"/v1/jobs/{job_id}/fail",
            json={
                "worker_id": "worker-2",
                "lease_token": token_1,
                "error": {"code": "INTERNAL_ERROR", "message": "fail", "request_id": "req-1"},
            },
            headers=self._make_auth_header(role="worker", sub="worker-2"),
        )
        assert res.status_code == 403

        # Legitimate worker-1 updates progress successfully
        res = client.post(
            f"/v1/jobs/{job_id}/progress",
            json={
                "worker_id": "worker-1",
                "lease_token": token_1,
                "stage": "rendering",
                "step": 5,
                "total": 10,
                "percent": 50.0,
            },
            headers=self._make_auth_header(role="worker", sub="worker-1"),
        )
        assert res.status_code == 200
        assert res.json()["status"] == "acknowledged"

        # Legitimate worker-1 completes job successfully
        res = client.post(
            f"/v1/jobs/{job_id}/complete",
            json={"worker_id": "worker-1", "lease_token": token_1, "artifacts": [art_payload]},
            headers=self._make_auth_header(role="worker", sub="worker-1"),
        )
        assert res.status_code == 200
        assert res.json()["status"] == "completed"

