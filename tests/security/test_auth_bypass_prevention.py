"""
tests/security/test_auth_bypass_prevention.py
Security tests verifying mandatory Bearer authentication and prevention of public bypass.
"""
import asyncio
import pytest
from starlette.requests import Request
from fastapi.testclient import TestClient

from factory_auth.config import load_auth_config, AuthConfig
from factory_auth.bearer import verify_bearer_token
from factory_contracts.errors import UnauthorizedError
from factory.apps.control_api.src.main import app


@pytest.mark.security
class TestAuthBypassPrevention:
    @pytest.fixture(autouse=True)
    def clean_auth_config(self):
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")
        yield
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")

    def test_production_fails_fast_on_empty_api_key(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("FACTORY_API_KEY", "")
        with pytest.raises(RuntimeError, match="FATAL: FACTORY_API_KEY"):
            load_auth_config()

    def test_development_generates_ephemeral_dev_key(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("FACTORY_API_KEY", "")
        cfg = load_auth_config()
        assert cfg.api_key.startswith("dev_")
        assert len(cfg.api_key) > 30

    def test_missing_authorization_header_rejected(self):
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        cfg = AuthConfig(api_key="secret123", jwt_secret=None, environment="test")
        with pytest.raises(UnauthorizedError, match="Missing Authorization header"):
            asyncio.run(verify_bearer_token(req, authorization=None, config=cfg))

    def test_invalid_scheme_rejected(self):
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        cfg = AuthConfig(api_key="secret123", jwt_secret=None, environment="test")
        with pytest.raises(UnauthorizedError, match="Invalid Authorization header format"):
            asyncio.run(verify_bearer_token(req, authorization="Basic dXNlcjpwYXNz", config=cfg))

    def test_empty_bearer_token_rejected(self):
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        cfg = AuthConfig(api_key="secret123", jwt_secret=None, environment="test")
        with pytest.raises(UnauthorizedError, match="Bearer token is empty"):
            asyncio.run(verify_bearer_token(req, authorization="Bearer   ", config=cfg))

    def test_incorrect_bearer_token_rejected(self):
        scope = {"type": "http", "headers": []}
        req = Request(scope)
        cfg = AuthConfig(api_key="secret123", jwt_secret=None, environment="test")
        with pytest.raises(UnauthorizedError, match="Invalid API token"):
            asyncio.run(verify_bearer_token(req, authorization="Bearer wrong_token", config=cfg))

    def test_valid_bearer_token_accepted(self):
        scope = {"type": "http", "headers": [], "state": {}}
        req = Request(scope)
        cfg = AuthConfig(api_key="secret123", jwt_secret=None, environment="test")
        user = asyncio.run(verify_bearer_token(req, authorization="Bearer secret123", config=cfg))
        assert user["sub"] == "system:api_key"
        assert req.state.user == user

    def test_fastapi_endpoints_enforce_auth(self, monkeypatch):
        monkeypatch.setenv("FACTORY_API_KEY", "prod-secret-999")
        monkeypatch.setenv("ENVIRONMENT", "test")

        client = TestClient(app)

        # 1. Health is public
        res = client.get("/health")
        assert res.status_code == 200

        # 2. Models endpoint requires auth
        res_no_auth = client.get("/v1/models")
        assert res_no_auth.status_code == 401
        assert res_no_auth.json()["error"]["code"] == "UNAUTHORIZED"

        # 3. Models endpoint with invalid token fails
        res_bad_auth = client.get("/v1/models", headers={"Authorization": "Bearer bad-token"})
        assert res_bad_auth.status_code == 401
        assert res_bad_auth.json()["error"]["code"] == "UNAUTHORIZED"

        # 4. Models endpoint with valid token succeeds
        res_valid = client.get("/v1/models", headers={"Authorization": "Bearer prod-secret-999"})
        assert res_valid.status_code == 200
        assert "data" in res_valid.json()
