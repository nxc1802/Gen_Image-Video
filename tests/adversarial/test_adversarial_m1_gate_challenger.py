"""
tests/adversarial/test_adversarial_m1_gate_challenger.py
Adversarial challenge test harness for Factory v4 Lite Milestone 1 Gate.

Probes:
1. Single API key auth: empty headers, malformed prefixes, invalid tokens, constant-time comparison.
2. Zero external dependencies: verifies 0 socket connection attempts to ports 5432, 6379, 9000 and 0 legacy imports.
3. Worker status reporting without lease tokens: probes whether status updates work cleanly without cryptographic lease tokens.
"""
import os
import secrets
import socket
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import pytest
from fastapi.testclient import TestClient

from factory_auth.config import AuthConfig, load_auth_config
from factory_auth.bearer import verify_bearer_token
from factory_contracts.errors import UnauthorizedError
from factory_contracts.jobs.job_contract import JobSubmissionRequest, JobState
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
)


TEST_API_KEY = "factory-test-key-sec-9876543210-adversarial"


@pytest.fixture(autouse=True)
def configure_test_env(monkeypatch):
    monkeypatch.setenv("FACTORY_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@127.0.0.1:5432/fake_db")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")


class TestSingleApiKeyAuth:
    """Probes Single API Key authentication under adversarial inputs and timing constraints."""

    def test_missing_and_empty_authorization_headers(self):
        from factory_control_api import app

        with TestClient(app) as client:
            # 1. Missing header
            r1 = client.get("/v1/jobs")
            assert r1.status_code == 401, f"Expected 401 for missing header, got {r1.status_code}"
            assert r1.json()["error"]["code"] == "UNAUTHORIZED"
            assert "Missing Authorization header" in r1.json()["error"]["message"]

            # 2. Empty header string
            r2 = client.get("/v1/jobs", headers={"Authorization": ""})
            assert r2.status_code == 401, f"Expected 401 for empty header, got {r2.status_code}"
            assert r2.json()["error"]["code"] == "UNAUTHORIZED"

            # 3. None/whitespace header
            r3 = client.get("/v1/jobs", headers={"Authorization": "   "})
            assert r3.status_code == 401, f"Expected 401 for whitespace header, got {r3.status_code}"
            assert r3.json()["error"]["code"] == "UNAUTHORIZED"

    def test_bearer_prefix_and_malformed_formats(self):
        from factory_control_api import app

        with TestClient(app) as client:
            # 1. 'Bearer ' without token
            r1 = client.get("/v1/jobs", headers={"Authorization": "Bearer "})
            assert r1.status_code == 401
            assert "empty" in r1.json()["error"]["message"].lower()

            # 2. 'Bearer    ' (multiple spaces)
            r2 = client.get("/v1/jobs", headers={"Authorization": "Bearer     "})
            assert r2.status_code == 401
            assert "empty" in r2.json()["error"]["message"].lower()

            # 3. Malformed prefix: 'Basic <key>'
            r3 = client.get("/v1/jobs", headers={"Authorization": f"Basic {TEST_API_KEY}"})
            assert r3.status_code == 401
            assert "Invalid Authorization header format" in r3.json()["error"]["message"]

            # 4. Malformed prefix: 'Token <key>'
            r4 = client.get("/v1/jobs", headers={"Authorization": f"Token {TEST_API_KEY}"})
            assert r4.status_code == 401
            assert "Invalid Authorization header format" in r4.json()["error"]["message"]

            # 5. Malformed prefix: 'BearerSecret' (no space delimiter)
            r5 = client.get("/v1/jobs", headers={"Authorization": f"Bearer{TEST_API_KEY}"})
            assert r5.status_code == 401
            assert "Invalid Authorization header format" in r5.json()["error"]["message"]

            # 6. Lowercase 'bearer <valid_key>' is valid per RFC 6750
            r6 = client.get("/v1/jobs", headers={"Authorization": f"bearer {TEST_API_KEY}"})
            assert r6.status_code == 200

    def test_invalid_tokens_rejected_with_401(self):
        from factory_control_api import app

        with TestClient(app) as client:
            adversarial_tokens = [
                "wrong-key-completely",
                TEST_API_KEY[:-1],           # 1 character shorter (prefix match)
                TEST_API_KEY + "extra",       # 1 character longer
                "x" * len(TEST_API_KEY),      # Same length, completely different bytes
                "Bearer " + TEST_API_KEY,     # Nested Bearer prefix
                "' OR '1'='1",               # SQL injection string
                "../../../../etc/passwd",     # Path traversal string
                "null\\x00token",             # Escaped null byte
                "token_with_special_chars!@#$%^&*()",
            ]
            for token in adversarial_tokens:
                res = client.get("/v1/jobs", headers={"Authorization": f"Bearer {token}"})
                assert res.status_code == 401, f"Expected 401 for token '{token[:15]}...', got {res.status_code}"
                assert res.json()["error"]["code"] == "UNAUTHORIZED"
                assert "Invalid API key" in res.json()["error"]["message"]

    def test_public_health_accessible_without_auth(self):
        from factory_control_api import app

        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200
            assert res.json()["status"] == "healthy"

    def test_production_fail_fast_on_missing_api_key(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("FACTORY_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FATAL: FACTORY_API_KEY environment variable is missing or empty"):
            load_auth_config()

    def test_constant_time_comparison_empirical(self):
        """
        Empirically verifies constant-time string comparison by measuring time distribution
        between an almost matching key (differs at final byte) vs completely different key (differs at byte 0).
        """
        target_key = "a" * 64
        almost_matching = "a" * 63 + "b"
        completely_different = "b" + "a" * 63

        cfg = AuthConfig(api_key=target_key, environment="test")

        # Warmup
        for _ in range(500):
            secrets.compare_digest(almost_matching, cfg.api_key)
            secrets.compare_digest(completely_different, cfg.api_key)

        times_almost = []
        times_diff = []

        iterations = 2000
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            secrets.compare_digest(almost_matching, cfg.api_key)
            t1 = time.perf_counter_ns()
            times_almost.append(t1 - t0)

            t2 = time.perf_counter_ns()
            secrets.compare_digest(completely_different, cfg.api_key)
            t3 = time.perf_counter_ns()
            times_diff.append(t3 - t2)

        mean_almost = statistics.mean(times_almost)
        mean_diff = statistics.mean(times_diff)
        ratio = mean_almost / mean_diff if mean_diff > 0 else 1.0

        # In constant time comparison, ratio should be approximately 1.0 (within noise margin on modern CPU)
        assert 0.70 <= ratio <= 1.30, f"Potential timing leak detected: ratio={ratio:.4f} (almost={mean_almost}ns, diff={mean_diff}ns)"


class TestZeroExternalDependencies:
    """Verifies that 0 socket connections are made to ports 5432, 6379, 9000 and 0 legacy modules loaded."""

    def test_zero_sockets_to_postgres_redis_s3_ports(self, monkeypatch):
        connected_addresses = []
        real_connect = socket.socket.connect

        def spy_connect(self, address):
            connected_addresses.append(address)
            return real_connect(self, address)

        monkeypatch.setattr(socket.socket, "connect", spy_connect)

        from factory_control_api import app

        with TestClient(app) as client:
            # 1. Job submission
            r1 = client.post(
                "/v1/jobs",
                json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "zero external dependency verification"},
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r1.status_code == 201
            job_id = r1.json()["job_id"]

            # 2. Job query
            r2 = client.get(f"/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
            assert r2.status_code == 200

            # 3. Job listing
            r3 = client.get("/v1/jobs", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
            assert r3.status_code == 200

            # 4. Models listing
            r4 = client.get("/v1/models", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
            assert r4.status_code == 200

        forbidden_ports = {5432, 6379, 9000}
        contacted_ports = [addr[1] for addr in connected_addresses if isinstance(addr, tuple) and len(addr) >= 2]
        violations = [p for p in contacted_ports if p in forbidden_ports]

        assert violations == [], f"Violations detected! Attempted socket connection to forbidden ports: {violations}"

    def test_zero_forbidden_modules_imported(self):
        import sys
        forbidden_modules = [
            "asyncpg",
            "redis",
            "boto3",
            "botocore",
            "aiosqlite",
            "alembic",
            "psycopg2",
        ]
        loaded_forbidden = [m for m in forbidden_modules if m in sys.modules]
        assert loaded_forbidden == [], f"Forbidden legacy enterprise modules found in sys.modules: {loaded_forbidden}"


class TestWorkerStatusWithoutLeaseTokens:
    """
    Probes whether worker status updates (progress, complete, fail) work cleanly
    WITHOUT cryptographic lease tokens, as mandated by Factory v4 Lite specifications.

    SPECIFICATION (ORIGINAL_REQUEST.md & PROJECT.md):
    - 'Worker kết nối và báo trạng thái qua HTTP (RUNNING, SUCCEEDED, FAILED) mà không dùng lease token.'
    - 'Không còn logic lease token hay distributed worker ownership trong codebase.'
    - Feature 4: 'Remove Distributed Lease Tokens: Eliminate lease_token, WorkerLease, 300s TTL leasing, and DeadJobReaper'
    """

    def test_worker_progress_without_lease_token_under_lease(self):
        """
        When a worker executes a job, reporting progress WITHOUT a lease_token MUST succeed (HTTP 200).
        Empirical observation: If it returns HTTP 403 Forbidden UNAUTHORIZED, lease tokens are still enforced.
        """
        from factory_control_api import app

        with TestClient(app) as client:
            r_sub = client.post(
                "/v1/jobs",
                json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "probe lease token progress"},
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r_sub.status_code == 201
            job_id = r_sub.json()["job_id"]

            r_lease = client.post(
                "/v1/jobs/lease",
                json={
                    "worker_id": "worker-t4-test-01",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r_lease.status_code == 200

            # Worker reports progress WITHOUT lease_token
            r_prog = client.post(
                f"/v1/jobs/{job_id}/progress",
                json={
                    "stage": "sampling",
                    "step": 5,
                    "total": 20,
                    "percent": 25.0,
                    "worker_id": "worker-t4-test-01",
                    # lease_token is omitted
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )

            # Per Factory v4 Lite spec: must succeed cleanly without lease token!
            assert r_prog.status_code == 200, (
                f"SPEC VIOLATION: Worker progress without lease token was rejected with HTTP {r_prog.status_code}: {r_prog.text}"
            )

    def test_worker_complete_without_lease_token_under_lease(self):
        """
        When a worker finishes a job, reporting completion WITHOUT a lease_token MUST succeed (HTTP 200).
        """
        from factory_control_api import app

        with TestClient(app) as client:
            r_sub = client.post(
                "/v1/jobs",
                json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "probe complete lease token"},
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            job_id = r_sub.json()["job_id"]

            r_lease = client.post(
                "/v1/jobs/lease",
                json={
                    "worker_id": "worker-t4-test-02",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r_lease.status_code == 200

            r_comp = client.post(
                f"/v1/jobs/{job_id}/complete",
                json={
                    "artifacts": [
                        {
                            "artifact_id": "art-01",
                            "type": "image",
                            "mime_type": "image/png",
                            "size_bytes": 1024,
                            "storage_path": "image/out.png",
                            "url": "http://localhost/out.png",
                            "download_url": "http://localhost/out.png",
                            "sha256": "0" * 64,
                        }
                    ],
                    "worker_id": "worker-t4-test-02",
                    # lease_token omitted!
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )

            # Per Factory v4 Lite spec: must succeed cleanly without lease token!
            assert r_comp.status_code == 200, (
                f"SPEC VIOLATION: Worker completion without lease token was rejected with HTTP {r_comp.status_code}: {r_comp.text}"
            )

    def test_worker_fail_without_lease_token_under_lease(self):
        """
        When a worker encounters an error, reporting failure WITHOUT a lease_token MUST succeed (HTTP 200).
        """
        from factory_control_api import app

        with TestClient(app) as client:
            r_sub = client.post(
                "/v1/jobs",
                json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "probe fail lease token"},
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            job_id = r_sub.json()["job_id"]

            r_lease = client.post(
                "/v1/jobs/lease",
                json={
                    "worker_id": "worker-t4-test-03",
                    "subsystem": "image",
                    "supported_models": ["qwen-image-2.1"],
                    "supported_tasks": ["text_to_image"],
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r_lease.status_code == 200

            r_fail = client.post(
                f"/v1/jobs/{job_id}/fail",
                json={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Out of memory",
                        "request_id": "req-probe",
                    },
                    "retryable": False,
                    "worker_id": "worker-t4-test-03",
                    # lease_token omitted!
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )

            # Per Factory v4 Lite spec: must succeed cleanly without lease token!
            assert r_fail.status_code == 200, (
                f"SPEC VIOLATION: Worker failure without lease token was rejected with HTTP {r_fail.status_code}: {r_fail.text}"
            )

    def test_worker_progress_unleased_job(self):
        """
        When a job is created in Factory v4 Lite (where distributed leasing is eliminated),
        a worker reporting progress with its worker_id should be accepted rather than rejected with 403.
        """
        from factory_control_api import app

        with TestClient(app) as client:
            r_sub = client.post(
                "/v1/jobs",
                json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "text_to_image",
                    "inputs": {"prompt": "probe unleased progress"},
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r_sub.status_code == 201
            job_id = r_sub.json()["job_id"]

            r_prog = client.post(
                f"/v1/jobs/{job_id}/progress",
                json={
                    "stage": "sampling",
                    "step": 5,
                    "total": 20,
                    "percent": 25.0,
                    "worker_id": "worker-direct-t4-0",
                },
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert r_prog.status_code == 200, (
                f"SPEC VIOLATION: Worker progress on job was rejected with HTTP {r_prog.status_code}: {r_prog.text}"
            )

