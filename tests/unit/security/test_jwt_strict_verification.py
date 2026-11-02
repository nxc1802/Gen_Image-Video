"""
tests/unit/security/test_jwt_strict_verification.py
Strict Security Contracts & JWT Claim Verification Test Suite.
Verifies algorithm allowlists, exp requirements, issuer/audience validation, and worker identity binding.
"""
import time
from datetime import datetime, timezone, timedelta
import jwt
import pytest
from fastapi import FastAPI, Depends, Request

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
        # exp omitted intentionally
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    with pytest.raises(UnauthorizedError, match="missing required claim"):
        decode_jwt_token(token, SECRET_KEY, algorithm="HS256")


def test_expired_token_rejected():
    now = int(time.time())
    payload = {
        "sub": "user-123",
        "role": "admin",
        "exp": now - 60,  # Expired 60s ago
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
async def test_worker_identity_binding_rejects_missing_sub():
    user = {"role": "worker"}  # missing sub and worker_id
    with pytest.raises(ForbiddenError, match="Worker authentication token must contain 'sub' or 'worker_id'"):
        await verify_worker_identity(user)


@pytest.mark.asyncio
async def test_worker_identity_binding_accepts_valid_sub():
    user = {"role": "worker", "sub": "worker-gpu-0"}
    verified = await verify_worker_identity(user)
    assert verified["sub"] == "worker-gpu-0"


@pytest.mark.parametrize(
    "invalid_alg",
    [
        "HS512",
        "none",
        "RS512",
        "ES256",
        "hs256",  # Case-sensitive allowlist check
        "INVALID_ALG",
    ],
)
def test_load_auth_config_disallowed_jwt_algorithm_fails_fast(monkeypatch, invalid_alg: str):
    """
    Verifies that load_auth_config() fails fast with RuntimeError/ValueError
    whenever JWT_ALGORITHM is configured with an unsupported or disallowed algorithm.
    """
    monkeypatch.setenv("JWT_ALGORITHM", invalid_alg)
    with pytest.raises((RuntimeError, ValueError), match="FATAL: Unsupported or disallowed JWT_ALGORITHM"):
        load_auth_config()


@pytest.mark.parametrize("valid_alg", ["HS256", "RS256"])
def test_load_auth_config_valid_jwt_algorithm_accepted(monkeypatch, valid_alg: str):
    """
    Verifies that load_auth_config() successfully loads and preserves valid algorithms in ALLOWED_ALGORITHMS.
    """
    monkeypatch.setenv("JWT_ALGORITHM", valid_alg)
    cfg = load_auth_config()
    assert cfg.jwt_algorithm == valid_alg


def test_auth_config_direct_instantiation_rejects_disallowed_algorithm():
    """
    Verifies defense-in-depth: direct AuthConfig dataclass construction fails fast
    when passed an algorithm outside ALLOWED_ALGORITHMS.
    """
    with pytest.raises((RuntimeError, ValueError), match="Disallowed or unsupported JWT algorithm"):
        AuthConfig(
            api_key="sk-test-key",
            jwt_secret="secret",
            environment="test",
            jwt_algorithm="HS512",
        )

    # Valid direct instantiation succeeds
    cfg = AuthConfig(
        api_key="sk-test-key",
        jwt_secret="secret",
        environment="test",
        jwt_algorithm="RS256",
    )
    assert cfg.jwt_algorithm == "RS256"

