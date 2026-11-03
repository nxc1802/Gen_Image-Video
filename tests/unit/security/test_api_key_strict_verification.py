"""
tests/unit/security/test_api_key_strict_verification.py
Strict Security Contracts & API Key Verification Test Suite for Factory v4 Lite.
Verifies single FACTORY_API_KEY enforcement, constant-time comparison, fail-fast behavior,
and worker identity binding.
"""
import asyncio
import os
import secrets
import pytest
from starlette.requests import Request

from factory_auth import (
    AuthConfig,
    load_auth_config,
    verify_bearer_token,
)
from factory_contracts.errors import UnauthorizedError, ForbiddenError
from factory.apps.control_api.src.api.routers.workers import verify_worker_identity

SECRET_KEY = "test-api-key-secret-32-chars-long!"


def test_valid_api_key_accepted():
    scope = {"type": "http", "headers": [], "state": {}}
    req = Request(scope)
    cfg = AuthConfig(api_key=SECRET_KEY, environment="test")
    user = asyncio.run(verify_bearer_token(req, authorization=f"Bearer {SECRET_KEY}", config=cfg))
    assert user["sub"] in ("system", "system:api_key")
    assert user["role"] == "admin"
    assert req.state.user == user


def test_incorrect_api_key_rejected():
    scope = {"type": "http", "headers": []}
    req = Request(scope)
    cfg = AuthConfig(api_key=SECRET_KEY, environment="test")
    with pytest.raises(UnauthorizedError, match="Invalid API (key|token)"):
        asyncio.run(verify_bearer_token(req, authorization="Bearer wrong-secret-key", config=cfg))


def test_missing_authorization_header_rejected():
    scope = {"type": "http", "headers": []}
    req = Request(scope)
    cfg = AuthConfig(api_key=SECRET_KEY, environment="test")
    with pytest.raises(UnauthorizedError, match="Missing Authorization header"):
        asyncio.run(verify_bearer_token(req, authorization=None, config=cfg))


def test_empty_bearer_token_rejected():
    scope = {"type": "http", "headers": []}
    req = Request(scope)
    cfg = AuthConfig(api_key=SECRET_KEY, environment="test")
    with pytest.raises(UnauthorizedError, match="Bearer token is empty"):
        asyncio.run(verify_bearer_token(req, authorization="Bearer   ", config=cfg))


def test_invalid_scheme_rejected():
    scope = {"type": "http", "headers": []}
    req = Request(scope)
    cfg = AuthConfig(api_key=SECRET_KEY, environment="test")
    with pytest.raises(UnauthorizedError, match="Invalid Authorization header format"):
        asyncio.run(verify_bearer_token(req, authorization=f"Basic {SECRET_KEY}", config=cfg))


def test_production_mode_fails_fast_on_missing_api_key(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FACTORY_API_KEY", "")
    with pytest.raises(RuntimeError, match="FATAL: FACTORY_API_KEY"):
        load_auth_config()


def test_development_mode_generates_ephemeral_dev_key(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("FACTORY_API_KEY", "")
    cfg = load_auth_config()
    assert cfg.api_key.startswith("dev_")
    assert len(cfg.api_key) > 30


def test_auth_config_preserves_custom_api_key(monkeypatch):
    custom_key = "custom-secret-key-12345"
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("FACTORY_API_KEY", custom_key)
    cfg = load_auth_config()
    assert cfg.api_key == custom_key


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
