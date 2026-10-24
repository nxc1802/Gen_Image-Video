"""
tests/security/test_error_sanitization.py
Security tests verifying zero stack traces, absolute paths, or internal class names in error responses.
"""
import re
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from factory_contracts.errors import (
    UnauthorizedError,
    SSRFBlockedError,
    ModelUnavailableError,
)
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from factory.apps.control_api.src.api.errors import register_exception_handlers

FORBIDDEN_LEAK_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"/Volumes/[A-Za-z0-9_/:-]+",
    r"/Users/[A-Za-z0-9_/:-]+",
    r"/home/[A-Za-z0-9_/:-]+",
    r"C:\\",
    r"Exception:",
    r"KeyError:",
    r"ZeroDivisionError:",
    r"pydantic_core\._pydantic_core\.ValidationError",
    r"\.py:\d+",
]


@pytest.fixture
def error_test_app():
    test_app = FastAPI()
    test_app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(test_app)

    @test_app.get("/trigger-unhandled")
    def trigger_unhandled():
        # Raise unexpected internal error with sensitive paths
        raise KeyError("Secret database key missing in /Volumes/WorkSpace/Secret/db.sqlite at line 42")

    @test_app.get("/trigger-unauthorized")
    def trigger_unauthorized():
        raise UnauthorizedError(message="Authentication required to access resource.")

    @test_app.get("/trigger-ssrf")
    def trigger_ssrf():
        raise SSRFBlockedError(message="Target IP 169.254.169.254 is blocked.")

    @test_app.get("/trigger-model-unavailable")
    def trigger_model_unavailable():
        raise ModelUnavailableError(message="Model weights checkpoint not found on host.")

    return test_app


@pytest.mark.security
class TestErrorSanitization:
    def test_unhandled_exception_sanitizes_public_response(self, error_test_app):
        client = TestClient(error_test_app, raise_server_exceptions=False)
        res = client.get("/trigger-unhandled")

        assert res.status_code == 500
        body = res.json()
        assert "error" in body
        err = body["error"]
        assert err["code"] == "INTERNAL_SERVER_ERROR"
        assert err["request_id"].startswith("req-")
        assert res.headers.get("X-Request-ID") == err["request_id"]

        # Ensure message does NOT leak internal details
        assert "db.sqlite" not in err["message"]
        assert "/Volumes" not in err["message"]

        # Check raw text against all leak patterns
        for pattern in FORBIDDEN_LEAK_PATTERNS:
            match = re.search(pattern, res.text)
            assert not match, f"Security Violation! Sanitized error leaked '{pattern}': {res.text}"

    def test_domain_exceptions_adhere_to_unified_contract(self, error_test_app):
        client = TestClient(error_test_app)

        # 1. Unauthorized
        res_auth = client.get("/trigger-unauthorized")
        assert res_auth.status_code == 401
        err_auth = res_auth.json()["error"]
        assert err_auth["code"] == "UNAUTHORIZED"
        assert err_auth["request_id"].startswith("req-")
        assert res_auth.headers.get("X-Request-ID") == err_auth["request_id"]

        # 2. SSRF Blocked
        res_ssrf = client.get("/trigger-ssrf")
        assert res_ssrf.status_code == 400
        err_ssrf = res_ssrf.json()["error"]
        assert err_ssrf["code"] == "SSRF_BLOCKED"

        # 3. Model Unavailable
        res_model = client.get("/trigger-model-unavailable")
        assert res_model.status_code == 503
        err_model = res_model.json()["error"]
        assert err_model["code"] == "MODEL_UNAVAILABLE"
