"""
tests/unit/security/test_jwt_auth.py
Strict Security Contracts & JWT Claim Verification Test Suite.
Verifies algorithm allowlists, exp requirements, issuer/audience validation, and role differentiation.
"""
import time
import jwt
import pytest
from starlette.requests import Request

from factory_auth import (
    AuthConfig,
    ALLOWED_ALGORITHMS,
    decode_jwt_token,
    load_auth_config,
    verify_bearer_token,
)
from factory_contracts.errors import UnauthorizedError, ForbiddenError
from factory.apps.control_api.src.api.routers.workers import verify_worker_identity

SECRET_KEY = "test-jwt-secret-key-32-chars-long!"


def test_allowed_algorithms_allowlist():
    assert "HS256" in ALLOWED_ALGORITHMS
    assert "RS256" in ALLOWED_ALGORITHMS
    assert "none" not in ALLOWED_ALGORITHMS
    assert "HS512" not in ALLOWED_ALGORITHMS


def test_disallowed_algorithm_rejected():
    with pytest.raises(UnauthorizedError, match="Disallowed or unsupported JWT algorithm"):
        decode_jwt_token("some.fake.token", SECRET_KEY, algorithm="HS512")

    with pytest.raises(UnauthorizedError, match="Disallowed or unsupported JWT algorithm"):
        decode_jwt_token("some.fake.token", SECRET_KEY, algorithm="none")


def test_valid_hs256_token_succeeds():
    now = int(time.time())
    payload = {
        "sub": "user-123",
        "role": "admin",
        "exp": now + 3600,
        "iss": "factory-control-api",
        "aud": "factory-v4",
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    decoded = decode_jwt_token(
        token,
        SECRET_KEY,
        algorithm="HS256",
        audience="factory-v4",
        issuer="factory-control-api",
    )
    assert decoded["sub"] == "user-123"
    assert decoded["role"] == "admin"


def test_missing_exp_claim_rejected():
    payload = {
        "sub": "user-123",
        "role": "admin",
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    with pytest.raises(UnauthorizedError, match="missing required claim 'exp'"):
        decode_jwt_token(token, SECRET_KEY, algorithm="HS256")


def test_expired_token_rejected():
    now = int(time.time())
    payload = {
        "sub": "user-123",
        "role": "admin",
        "exp": now - 60,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    with pytest.raises(UnauthorizedError, match="expired"):
        decode_jwt_token(token, SECRET_KEY, algorithm="HS256")


def test_issuer_verification_mismatch_rejected():
    now = int(time.time())
    payload = {
        "sub": "user-123",
        "exp": now + 3600,
        "iss": "rogue-issuer",
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    with pytest.raises(UnauthorizedError, match="Invalid JWT issuer"):
        decode_jwt_token(token, SECRET_KEY, algorithm="HS256", issuer="factory-control-api")


def test_audience_verification_mismatch_rejected():
    now = int(time.time())
    payload = {
        "sub": "user-123",
        "exp": now + 3600,
        "aud": "other-service",
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    with pytest.raises(UnauthorizedError, match="Invalid JWT audience"):
        decode_jwt_token(token, SECRET_KEY, algorithm="HS256", audience="factory-v4")


def test_rs256_algorithm_is_allowed_in_allowlist():
    assert "RS256" in ALLOWED_ALGORITHMS


def test_production_mode_requires_jwt_audience(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FACTORY_API_KEY", "prod-api-key-test")
    monkeypatch.setenv("JWT_SECRET_KEY", "prod-jwt-secret-key-32-chars-long!")
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)

    with pytest.raises(RuntimeError, match="FATAL: JWT_AUDIENCE must be explicitly configured"):
        load_auth_config()


@pytest.mark.asyncio
async def test_bearer_role_differentiation_admin_via_api_key():
    scope = {"type": "http", "headers": [], "state": {}}
    req = Request(scope)
    cfg = AuthConfig(api_key="admin-static-secret-key", jwt_secret=SECRET_KEY, environment="test")

    user = await verify_bearer_token(req, authorization="Bearer admin-static-secret-key", config=cfg)
    assert user["role"] == "admin"
    assert user["sub"] in ("system", "system:api_key")


@pytest.mark.asyncio
async def test_bearer_role_differentiation_worker_via_jwt():
    scope = {"type": "http", "headers": [], "state": {}}
    req = Request(scope)
    cfg = AuthConfig(api_key="admin-static-secret-key", jwt_secret=SECRET_KEY, environment="test")

    now = int(time.time())
    worker_jwt = jwt.encode(
        {"sub": "worker-gpu-image-0", "role": "worker", "exp": now + 3600},
        SECRET_KEY,
        algorithm="HS256",
    )

    user = await verify_bearer_token(req, authorization=f"Bearer {worker_jwt}", config=cfg)
    assert user["role"] == "worker"
    assert user["sub"] == "worker-gpu-image-0"
    assert user["worker_id"] == "worker-gpu-image-0"


@pytest.mark.asyncio
async def test_bearer_role_differentiation_user_via_jwt():
    scope = {"type": "http", "headers": [], "state": {}}
    req = Request(scope)
    cfg = AuthConfig(api_key="admin-static-secret-key", jwt_secret=SECRET_KEY, environment="test")

    now = int(time.time())
    user_jwt = jwt.encode(
        {"sub": "usr-regular-42", "role": "user", "exp": now + 3600},
        SECRET_KEY,
        algorithm="HS256",
    )

    user = await verify_bearer_token(req, authorization=f"Bearer {user_jwt}", config=cfg)
    assert user["role"] == "user"
    assert user["sub"] == "usr-regular-42"


@pytest.mark.asyncio
async def test_worker_identity_binding_rejects_user_role():
    user = {"role": "user", "sub": "usr-123"}
    with pytest.raises(ForbiddenError, match="require role 'worker' or 'admin'"):
        await verify_worker_identity(user)


@pytest.mark.asyncio
async def test_worker_identity_binding_rejects_missing_sub():
    user = {"role": "worker"}
    with pytest.raises(ForbiddenError, match="must contain 'sub' or 'worker_id'"):
        await verify_worker_identity(user)


@pytest.mark.asyncio
async def test_worker_identity_binding_accepts_valid_sub():
    user = {"role": "worker", "sub": "worker-gpu-0"}
    verified = await verify_worker_identity(user)
    assert verified["sub"] == "worker-gpu-0"
