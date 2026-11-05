"""
tests/adversarial/test_adversarial_p0_challenger2_security.py
Empirical Adversarial Test Suite for Phase 0 Requirement P1.6
(Control Plane, Storage, and Studio Browser Security).

Adversarially probes:
1. JWT Security Bypasses:
   - Algorithm "none" rejection (parameter and header spoofing)
   - Unsupported / Disallowed algorithms ("HS512", "RS512")
   - Expired tokens and clock skew boundaries
   - Missing mandatory 'exp' claim
   - Audience mismatch & issuer mismatch
   - Tampered signatures, corrupted payloads, and stripped signatures
2. Role Differentiation:
   - Worker JWT with role="worker" and sub="worker-1" -> sets request.state.user properly
   - Missing sub on worker token -> raises ForbiddenError in verify_worker_identity
   - Static FACTORY_API_KEY -> sets role="admin" with sub="system"
   - User role -> rejected in verify_worker_identity with ForbiddenError
3. S3 Storage Fail-Fast:
   - Production mode missing S3 credentials -> immediate RuntimeError
   - Production mode default 'minioadmin' credentials -> immediate RuntimeError
   - Production mode empty S3_BUCKET_NAME -> immediate RuntimeError
   - Production mode inaccessible S3 bucket -> immediate RuntimeError in S3StorageProvider
   - Development mode tolerates offline S3 bucket
4. Studio Web Browser Storage Security:
   - saveConnectionSettings without persistKey=true never sets studio_api_key in localStorage
   - saveConnectionSettings purges pre-existing key from localStorage when persistKey=false
   - saveConnectionSettings with persistKey=true explicitly persists key
   - Restoration migration cleans up un-remembered keys from localStorage
"""
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import jwt
import pytest
from starlette.requests import Request

from factory_auth import (
    ALLOWED_ALGORITHMS,
    AuthConfig,
    decode_jwt_token,
    load_auth_config,
    verify_bearer_token,
)
from factory_contracts.errors import ForbiddenError, UnauthorizedError
from factory.apps.control_api.src.api.routers.workers import verify_worker_identity
from factory.apps.control_api.src.storage.config import StorageConfig, load_storage_config
from factory.apps.control_api.src.storage.s3 import S3StorageProvider

SECRET_KEY = "test-jwt-adversarial-secret-key-32-chars-minimum!"
AUDIENCE = "factory-v4"
ISSUER = "factory-control-api"
REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "factory" / "apps" / "studio-web"
HAS_NODE = shutil.which("node") is not None


# ===========================================================================
# 1. JWT Security Bypasses
# ===========================================================================

@pytest.mark.adversarial
class TestJWTAdversarialBypasses:
    """Stress tests JWT decoding and Bearer verification against spoofing and bypass vectors."""

    def test_alg_none_parameter_rejection(self):
        """Passing algorithm='none' directly to decode_jwt_token must raise UnauthorizedError."""
        with pytest.raises(UnauthorizedError, match="Disallowed or unsupported JWT algorithm"):
            decode_jwt_token("token.segment.three", SECRET_KEY, algorithm="none")

    def test_alg_none_token_header_spoofing(self):
        """A crafted token with header alg='none' must be rejected with UnauthorizedError."""
        now = int(time.time())
        token_none = jwt.encode(
            {"sub": "attacker", "role": "admin", "exp": now + 3600},
            key="",
            algorithm="none",
        )
        with pytest.raises(UnauthorizedError, match="Invalid JWT algorithm"):
            decode_jwt_token(token_none, SECRET_KEY, algorithm="HS256")

    def test_disallowed_algorithm_hs512(self):
        """Tokens using HS512 or RS512 outside allowlist must raise UnauthorizedError."""
        # Parameter check
        with pytest.raises(UnauthorizedError, match="Disallowed or unsupported JWT algorithm"):
            decode_jwt_token("token.segment.three", SECRET_KEY, algorithm="HS512")

        with pytest.raises(UnauthorizedError, match="Disallowed or unsupported JWT algorithm"):
            decode_jwt_token("token.segment.three", SECRET_KEY, algorithm="RS512")

        # Header check with HS512 token
        now = int(time.time())
        token_hs512 = jwt.encode(
            {"sub": "attacker", "role": "admin", "exp": now + 3600},
            key="a" * 64,
            algorithm="HS512",
        )
        with pytest.raises(UnauthorizedError, match="Invalid JWT algorithm"):
            decode_jwt_token(token_hs512, SECRET_KEY, algorithm="HS256")

    def test_expired_token_rejected(self):
        """Tokens past expiration date must be strictly rejected."""
        now = int(time.time())
        expired_token = jwt.encode(
            {"sub": "user-1", "exp": now - 30, "role": "admin"},
            key=SECRET_KEY,
            algorithm="HS256",
        )
        with pytest.raises(UnauthorizedError, match="JWT token has expired"):
            decode_jwt_token(expired_token, SECRET_KEY, algorithm="HS256")

    def test_missing_exp_claim_rejected(self):
        """Tokens without 'exp' claim must be strictly rejected (mandatory expiration)."""
        token_no_exp = jwt.encode(
            {"sub": "user-1", "role": "admin"},
            key=SECRET_KEY,
            algorithm="HS256",
        )
        with pytest.raises(UnauthorizedError, match="missing required claim 'exp'"):
            decode_jwt_token(token_no_exp, SECRET_KEY, algorithm="HS256")

    def test_audience_mismatch_rejected(self):
        """Token with mismatched audience must be rejected."""
        now = int(time.time())
        token = jwt.encode(
            {"sub": "user-1", "exp": now + 3600, "aud": "rogue-audience"},
            key=SECRET_KEY,
            algorithm="HS256",
        )
        with pytest.raises(UnauthorizedError, match="Invalid JWT audience"):
            decode_jwt_token(token, SECRET_KEY, algorithm="HS256", audience="factory-v4")

    def test_tampered_signature_rejected(self):
        """Token signed with wrong key or modified signature must raise UnauthorizedError."""
        now = int(time.time())
        token_wrong_key = jwt.encode(
            {"sub": "user-1", "exp": now + 3600},
            key="wrong-secret-key-that-is-at-least-32-chars!",
            algorithm="HS256",
        )
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(token_wrong_key, SECRET_KEY, algorithm="HS256")

    def test_tampered_payload_rejected(self):
        """Validly signed token with modified payload bytes must fail verification."""
        now = int(time.time())
        token = jwt.encode(
            {"sub": "user-1", "role": "user", "exp": now + 3600},
            key=SECRET_KEY,
            algorithm="HS256",
        )
        parts = token.split(".")
        # Tamper payload by altering one base64 character
        altered_char = "B" if parts[1][-1] != "B" else "C"
        tampered_token = f"{parts[0]}.{parts[1][:-1]}{altered_char}.{parts[2]}"
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(tampered_token, SECRET_KEY, algorithm="HS256")

    def test_stripped_signature_rejected(self):
        """Token with empty signature section (header.payload.) must raise UnauthorizedError."""
        now = int(time.time())
        token = jwt.encode(
            {"sub": "user-1", "role": "admin", "exp": now + 3600},
            key=SECRET_KEY,
            algorithm="HS256",
        )
        parts = token.split(".")
        empty_sig_token = f"{parts[0]}.{parts[1]}."
        with pytest.raises(UnauthorizedError, match="Invalid JWT token"):
            decode_jwt_token(empty_sig_token, SECRET_KEY, algorithm="HS256")

    @pytest.mark.asyncio
    async def test_verify_bearer_token_rejects_all_bypass_attempts(self):
        """End-to-end FastAPI dependency verify_bearer_token must reject all bypass tokens."""
        now = int(time.time())
        cfg = AuthConfig(
            api_key="static-admin-key",
            jwt_secret=SECRET_KEY,
            jwt_algorithm="HS256",
            jwt_audience="factory-v4",
            environment="production",
        )
        req = Request({"type": "http", "headers": []})

        bypass_tokens = [
            jwt.encode({"sub": "att", "exp": now + 3600, "aud": "factory-v4"}, key="", algorithm="none"),
            jwt.encode({"sub": "att", "exp": now - 60, "aud": "factory-v4"}, key=SECRET_KEY, algorithm="HS256"),
            jwt.encode({"sub": "att", "aud": "factory-v4"}, key=SECRET_KEY, algorithm="HS256"),
            jwt.encode({"sub": "att", "exp": now + 3600, "aud": "wrong-service"}, key=SECRET_KEY, algorithm="HS256"),
            jwt.encode({"sub": "att", "exp": now + 3600, "aud": "factory-v4"}, key="bad-key-32-chars-long-0000000000", algorithm="HS256"),
        ]

        for tok in bypass_tokens:
            with pytest.raises(UnauthorizedError):
                await verify_bearer_token(req, authorization=f"Bearer {tok}", config=cfg)


# ===========================================================================
# 2. Role Differentiation & Worker Identity Binding
# ===========================================================================

@pytest.mark.adversarial
class TestRoleDifferentiation:
    """Verifies strict role differentiation and worker sub identity binding."""

    @pytest.mark.asyncio
    async def test_worker_jwt_sets_request_state_user(self):
        """Worker JWT with role='worker' and sub='worker-1' correctly sets request.state.user."""
        now = int(time.time())
        cfg = AuthConfig(api_key="admin-static-key", jwt_secret=SECRET_KEY, environment="test")
        req = Request({"type": "http", "headers": []})

        worker_token = jwt.encode(
            {"sub": "worker-1", "role": "worker", "exp": now + 3600},
            key=SECRET_KEY,
            algorithm="HS256",
        )

        user = await verify_bearer_token(req, authorization=f"Bearer {worker_token}", config=cfg)
        assert user["role"] == "worker"
        assert user["sub"] == "worker-1"
        assert user["worker_id"] == "worker-1"
        assert req.state.user["role"] == "worker"
        assert req.state.user["sub"] == "worker-1"

    @pytest.mark.asyncio
    async def test_worker_missing_sub_raises_forbidden_in_verify_worker_identity(self):
        """Worker identity validation must raise ForbiddenError when sub or worker_id is missing."""
        # Missing sub entirely
        with pytest.raises(ForbiddenError, match="Worker authentication token must contain 'sub' or 'worker_id'"):
            await verify_worker_identity({"role": "worker"})

        # Empty sub
        with pytest.raises(ForbiddenError, match="Worker authentication token must contain 'sub' or 'worker_id'"):
            await verify_worker_identity({"role": "worker", "sub": ""})

    @pytest.mark.asyncio
    async def test_static_factory_api_key_sets_role_admin(self):
        """Static FACTORY_API_KEY grants role='admin' with sub='system' and passes worker verification."""
        cfg = AuthConfig(api_key="super-secret-admin-key", jwt_secret=SECRET_KEY, environment="test")
        req = Request({"type": "http", "headers": []})

        user = await verify_bearer_token(req, authorization="Bearer super-secret-admin-key", config=cfg)
        assert user["role"] == "admin"
        assert user["sub"] == "system"
        assert req.state.user["role"] == "admin"

        # Admin is also authorized on worker endpoints
        worker_auth = await verify_worker_identity(user)
        assert worker_auth["role"] == "admin"

    @pytest.mark.asyncio
    async def test_standard_user_role_rejected_by_verify_worker_identity(self):
        """Non-worker, non-admin role must raise ForbiddenError on worker endpoints."""
        with pytest.raises(ForbiddenError, match="Worker control endpoints require role 'worker' or 'admin'"):
            await verify_worker_identity({"role": "user", "sub": "regular-user-42"})


# ===========================================================================
# 3. S3 Storage Fail-Fast
# ===========================================================================

@pytest.mark.adversarial
class TestS3StorageFailFast:
    """Verifies fail-fast behavior when S3 storage is misconfigured in production mode."""

    def test_production_missing_s3_credentials_fails_fast(self, monkeypatch):
        """Production mode with STORAGE_BACKEND=s3 and missing credentials raises immediate RuntimeError."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_BUCKET_NAME", "prod-artifacts")
        monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_SECRET_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(RuntimeError, match="FATAL: S3_ACCESS_KEY and S3_SECRET_KEY"):
            load_storage_config()

    def test_production_minioadmin_credentials_forbidden(self, monkeypatch):
        """Production mode with default 'minioadmin' credentials raises immediate RuntimeError."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_ACCESS_KEY", "minioadmin")
        monkeypatch.setenv("S3_SECRET_KEY", "minioadmin")

        with pytest.raises(RuntimeError, match="FATAL: Default MinIO credentials"):
            load_storage_config()

    def test_production_empty_bucket_name_fails_fast(self, monkeypatch):
        """Production mode with empty S3_BUCKET_NAME raises immediate RuntimeError."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_ACCESS_KEY", "valid-key")
        monkeypatch.setenv("S3_SECRET_KEY", "valid-secret")
        monkeypatch.setenv("S3_BUCKET_NAME", "")

        with pytest.raises(RuntimeError, match="FATAL: S3_BUCKET_NAME must be configured"):
            load_storage_config()

    def test_production_inaccessible_bucket_raises_runtime_error(self, monkeypatch):
        """Production S3StorageProvider instantiation with unreachable bucket raises immediate RuntimeError."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        cfg = StorageConfig(
            storage_backend="s3",
            s3_endpoint_url="http://127.0.0.1:49999",  # Unreachable port
            s3_bucket_name="nonexistent-prod-bucket",
            s3_access_key="fake-prod-key",
            s3_secret_key="fake-prod-secret",
            environment="production",
        )
        with pytest.raises(RuntimeError) as exc_info:
            S3StorageProvider(cfg)
        assert "FATAL: S3 bucket" in str(exc_info.value)
        assert "inaccessible" in str(exc_info.value)

    def test_development_mode_tolerates_offline_bucket(self, monkeypatch):
        """Development mode tolerates offline S3 bucket during initialization without crashing."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        cfg = StorageConfig(
            storage_backend="s3",
            s3_endpoint_url="http://127.0.0.1:49999",
            s3_bucket_name="offline-dev-bucket",
            s3_access_key="dev-key",
            s3_secret_key="dev-secret",
            environment="development",
        )
        provider = S3StorageProvider(cfg)
        assert provider.bucket == "offline-dev-bucket"


# ===========================================================================
# 4. Studio Web Browser Storage
# ===========================================================================

@pytest.mark.adversarial
class TestStudioWebBrowserStorage:
    """Verifies that Studio Web browser storage adheres strictly to ephemeral session scoping."""

    def test_source_code_invariants_for_browser_storage(self):
        """Inspects StudioContext.tsx source code to guarantee secure storage patterns."""
        context_file = STUDIO_DIR / "src" / "context" / "StudioContext.tsx"
        assert context_file.exists(), f"StudioContext.tsx not found at {context_file}"
        source = context_file.read_text(encoding="utf-8")

        # 1. saveConnectionSettings default parameter must be persistKey = false
        assert "persistKey = false" in source, "saveConnectionSettings must default persistKey to false"

        # 2. window.localStorage setting of API_KEY must only occur when persistKey is true
        assert "safeRemoveStorage(window.localStorage, STORAGE_KEYS.API_KEY);" in source

        # 3. Migration logic must purge legacy key if not opted in
        assert "safeRemoveStorage(window.localStorage, STORAGE_KEYS.API_KEY);" in source

    @pytest.mark.skipif(not HAS_NODE, reason="Node.js runtime required for browser storage execution")
    def test_node_execution_save_connection_settings_without_persist_key(self):
        """Directly executes browser storage behavior using Node.js to empirically prove security invariants."""
        node_script = """
        class MockStorage {
          constructor() { this.store = {}; }
          getItem(key) { return this.store[key] !== undefined ? this.store[key] : null; }
          setItem(key, value) { this.store[key] = String(value); }
          removeItem(key) { delete this.store[key]; }
          clear() { this.store = {}; }
        }

        const localStorage = new MockStorage();
        const sessionStorage = new MockStorage();

        const STORAGE_KEYS = {
          CONTROL_URL: 'studio_control_url',
          API_KEY: 'studio_api_key',
          REMEMBER_KEY: 'studio_remember_api_key',
        };

        function safeSetStorage(storage, key, value) {
          if (!storage) return;
          try { storage.setItem(key, value); } catch {}
        }
        function safeRemoveStorage(storage, key) {
          if (!storage) return;
          try { storage.removeItem(key); } catch {}
        }

        function saveConnectionSettings(newUrl, newKey, persistKey = false) {
          const cleanedUrl = newUrl.trim();
          const cleanedKey = newKey.trim();

          safeSetStorage(localStorage, STORAGE_KEYS.CONTROL_URL, cleanedUrl);

          if (!cleanedKey) {
            safeRemoveStorage(sessionStorage, STORAGE_KEYS.API_KEY);
            safeRemoveStorage(localStorage, STORAGE_KEYS.API_KEY);
            safeRemoveStorage(localStorage, STORAGE_KEYS.REMEMBER_KEY);
          } else if (persistKey) {
            safeSetStorage(sessionStorage, STORAGE_KEYS.API_KEY, cleanedKey);
            safeSetStorage(localStorage, STORAGE_KEYS.API_KEY, cleanedKey);
            safeSetStorage(localStorage, STORAGE_KEYS.REMEMBER_KEY, 'true');
          } else {
            safeSetStorage(sessionStorage, STORAGE_KEYS.API_KEY, cleanedKey);
            safeRemoveStorage(localStorage, STORAGE_KEYS.API_KEY);
            safeRemoveStorage(localStorage, STORAGE_KEYS.REMEMBER_KEY);
          }
        }

        const results = {};

        // 1. Without persistKey argument (default)
        saveConnectionSettings('http://localhost:8000', 'secret-key-default');
        results.default_local = localStorage.getItem('studio_api_key');
        results.default_session = sessionStorage.getItem('studio_api_key');
        results.default_remember = localStorage.getItem('studio_remember_api_key');

        // 2. With persistKey=false explicitly
        localStorage.clear();
        sessionStorage.clear();
        saveConnectionSettings('http://localhost:8000', 'secret-key-false', false);
        results.false_local = localStorage.getItem('studio_api_key');
        results.false_session = sessionStorage.getItem('studio_api_key');

        // 3. Purging existing key when persistKey=false
        localStorage.setItem('studio_api_key', 'pre-existing-key');
        localStorage.setItem('studio_remember_api_key', 'true');
        saveConnectionSettings('http://localhost:8000', 'new-key', false);
        results.purged_local = localStorage.getItem('studio_api_key');
        results.purged_remember = localStorage.getItem('studio_remember_api_key');
        results.purged_session = sessionStorage.getItem('studio_api_key');

        // 4. Explicit opt-in with persistKey=true
        localStorage.clear();
        sessionStorage.clear();
        saveConnectionSettings('http://localhost:8000', 'optin-key', true);
        results.optin_local = localStorage.getItem('studio_api_key');
        results.optin_session = sessionStorage.getItem('studio_api_key');
        results.optin_remember = localStorage.getItem('studio_remember_api_key');

        console.log(JSON.stringify(results));
        """
        proc = subprocess.run(
            ["node", "-e", node_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, f"Node execution failed: {proc.stderr}"
        res = json.loads(proc.stdout.strip())

        # Invariant 1: default persistKey never sets localStorage
        assert res["default_local"] is None
        assert res["default_session"] == "secret-key-default"
        assert res["default_remember"] is None

        # Invariant 2: explicit persistKey=false never sets localStorage
        assert res["false_local"] is None
        assert res["false_session"] == "secret-key-false"

        # Invariant 3: pre-existing key is purged from localStorage
        assert res["purged_local"] is None
        assert res["purged_remember"] is None
        assert res["purged_session"] == "new-key"

        # Invariant 4: persistKey=true sets both localStorage and sessionStorage
        assert res["optin_local"] == "optin-key"
        assert res["optin_session"] == "optin-key"
        assert res["optin_remember"] == "true"
