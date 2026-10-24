"""
Pytest configuration and global fixtures for Factory v4 E2E Test Suite.
Supports both in-process contract simulation and live cluster testing.
"""

import os
import re
import pytest
import httpx
from typing import Generator, Dict, Any

from tests.e2e.mock_contract_server import FactoryContractServer, VALID_API_KEYS

VALID_TEST_TOKEN = "sk-factory-test-token-valid"
LIVE_MODE = os.environ.get("FACTORY_E2E_LIVE", "").strip() in ("1", "true", "TRUE")
BASE_URL = os.environ.get("FACTORY_BASE_URL", "https://jobs.ai.example.com")


@pytest.fixture(scope="session")
def contract_server() -> FactoryContractServer:
    """Provides a singleton instance of the contract simulation server."""
    return FactoryContractServer()


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Provides valid Bearer authentication headers."""
    api_key = os.environ.get("FACTORY_API_KEY", VALID_TEST_TOKEN)
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def empty_auth_headers() -> Dict[str, str]:
    """Provides empty Bearer token headers for auth bypass testing."""
    return {
        "Authorization": "Bearer ",
        "Content-Type": "application/json"
    }


@pytest.fixture
def invalid_auth_headers() -> Dict[str, str]:
    """Provides malformed/invalid Bearer token headers."""
    return {
        "Authorization": "Bearer bad_key_unauthorized",
        "Content-Type": "application/json"
    }


@pytest.fixture
def client(contract_server: FactoryContractServer) -> Generator[httpx.Client, None, None]:
    """
    HTTP client configured for E2E tests.
    Uses MockTransport in simulation mode; uses real networking in live mode.
    """
    if LIVE_MODE:
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as cl:
            yield cl
    else:
        transport = httpx.MockTransport(contract_server.handle_request)
        with httpx.Client(transport=transport, base_url=BASE_URL, timeout=30.0) as cl:
            yield cl


def assert_sanitized_error(response: httpx.Response, expected_status: int, expected_code: str):
    """
    Verifies that an error response strictly satisfies the Unified Error Contract
    and exposes ZERO internal stack traces, Python classes, or filesystem paths.
    """
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"
    body = response.json()
    assert "error" in body, f"Response missing 'error' key: {body}"
    err = body["error"]
    assert "code" in err, f"Missing 'code' in error: {err}"
    assert err["code"] == expected_code, f"Expected code '{expected_code}', got '{err['code']}'"
    assert "message" in err and len(err["message"]) > 0, "Missing or empty error message"
    assert "request_id" in err and err["request_id"].startswith("req-"), f"Invalid request_id: {err.get('request_id')}"

    # Stringent error sanitization assertion: 0 stack traces or file paths
    text = response.text
    forbidden_patterns = [
        r"Traceback \(most recent call last\)",
        r"/Volumes/[A-Za-z0-9_/:-]+",
        r"/Users/[A-Za-z0-9_/:-]+",
        r"/home/[A-Za-z0-9_/:-]+",
        r"C:\\",
        r"Exception:",
        r"KeyError:",
        r"ZeroDivisionError:",
        r"pydantic_core\._pydantic_core\.ValidationError",
    ]
    for pattern in forbidden_patterns:
        match = re.search(pattern, text)
        assert not match, f"Security Violation! Sanitized error leaked internal details matching '{pattern}': {text}"


def assert_valid_job(job: Dict[str, Any], expected_service: str, expected_task: str):
    """Verifies that a job dictionary adheres to the Factory Job Contract."""
    assert "job_id" in job and job["job_id"].startswith("job-")
    assert "request_id" in job and job["request_id"].startswith("req-")
    assert job["service"] == expected_service
    assert job["task"] == expected_task
    assert job["state"] in ["CREATED", "QUEUED", "ADMITTED", "RUNNING", "STREAMING", "SUCCEEDED", "FAILED", "CANCELLED"]
    assert "created_at" in job
    assert "progress" in job
    assert "artifacts" in job
    assert isinstance(job["artifacts"], list)
