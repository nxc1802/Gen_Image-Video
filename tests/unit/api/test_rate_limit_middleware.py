"""
tests/unit/api/test_rate_limit_middleware.py
Unit tests for sliding-window RateLimitMiddleware in control-api.
Verifies rate limit enforcement, 429 RATE_LIMITED envelope, client key isolation, and exemptions.
"""
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from factory.apps.control_api.src.api.middleware.rate_limit import RateLimitMiddleware


def create_test_app(default_limit: int = 5, window_seconds: int = 60) -> FastAPI:
    app = FastAPI()
    # Explicitly enable with low limit for testing
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=default_limit,
        window_seconds=window_seconds,
        enabled=True,
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/data")
    async def get_data(request: Request):
        return {"data": "ok"}

    return app


def test_requests_within_limit_succeed():
    app = create_test_app(default_limit=5)
    client = TestClient(app)

    for i in range(5):
        resp = client.get("/api/data", headers={"Authorization": "Bearer token-1"})
        assert resp.status_code == 200
        assert resp.headers["X-RateLimit-Limit"] == "5"
        assert int(resp.headers["X-RateLimit-Remaining"]) == 4 - i


def test_exceeding_rate_limit_returns_429_rate_limited():
    app = create_test_app(default_limit=3, window_seconds=30)
    client = TestClient(app)

    headers = {"Authorization": "Bearer token-burst", "X-Request-ID": "req-rate-limit-test"}
    # 3 requests allowed
    for _ in range(3):
        res = client.get("/api/data", headers=headers)
        assert res.status_code == 200

    # 4th request must be blocked with 429
    res = client.get("/api/data", headers=headers)
    assert res.status_code == 429
    assert res.headers["X-RateLimit-Remaining"] == "0"
    assert "Retry-After" in res.headers
    assert int(res.headers["Retry-After"]) >= 1

    body = res.json()
    assert "error" in body
    err = body["error"]
    assert err["code"] == "RATE_LIMITED"
    assert "Rate limit" in err["message"]
    assert err["request_id"] == "req-rate-limit-test"
    assert err["details"]["limit"] == 3
    assert err["details"]["window_seconds"] == 30


def test_client_key_isolation_by_token():
    app = create_test_app(default_limit=2)
    client = TestClient(app)

    # Exhaust token A
    client.get("/api/data", headers={"Authorization": "Bearer client-A"})
    client.get("/api/data", headers={"Authorization": "Bearer client-A"})
    res_a = client.get("/api/data", headers={"Authorization": "Bearer client-A"})
    assert res_a.status_code == 429

    # Token B has its own fresh quota
    res_b = client.get("/api/data", headers={"Authorization": "Bearer client-B"})
    assert res_b.status_code == 200


def test_client_key_isolation_by_ip():
    app = create_test_app(default_limit=2)
    client = TestClient(app)

    # Exhaust IP 10.0.0.1
    client.get("/api/data", headers={"X-Forwarded-For": "10.0.0.1"})
    client.get("/api/data", headers={"X-Forwarded-For": "10.0.0.1"})
    res_ip1 = client.get("/api/data", headers={"X-Forwarded-For": "10.0.0.1"})
    assert res_ip1.status_code == 429

    # IP 10.0.0.2 is separate
    res_ip2 = client.get("/api/data", headers={"X-Forwarded-For": "10.0.0.2"})
    assert res_ip2.status_code == 200


def test_exempt_path_health_is_not_limited():
    app = create_test_app(default_limit=2)
    client = TestClient(app)

    # 10 calls to /health should all succeed
    for _ in range(10):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_periodic_key_eviction_removes_expired_clients():
    """Verify that idle client keys are evicted after window expiration."""
    import time
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=5,
        window_seconds=1,
        prune_interval=0.2,
        enabled=True,
    )

    @app.get("/api/test")
    async def get_test():
        return {"ok": True}

    client = TestClient(app)
    # Fire 5 distinct tokens
    for i in range(5):
        resp = client.get("/api/test", headers={"Authorization": f"Bearer token-{i}"})
        assert resp.status_code == 200

    # Locate middleware instance in stack
    curr = app.middleware_stack
    mw = None
    while curr is not None:
        if isinstance(curr, RateLimitMiddleware):
            mw = curr
            break
        curr = getattr(curr, "app", None)

    assert mw is not None
    assert len(mw._history) == 5

    # Advance time beyond window
    time.sleep(1.1)

    # Next request triggers eviction sweep
    resp = client.get("/api/test", headers={"Authorization": "Bearer token-active"})
    assert resp.status_code == 200

    # The 5 expired tokens should have been evicted, leaving only token-active
    assert len(mw._history) == 1


def test_capacity_limit_caps_unique_clients_under_burst():
    """Verify that history length is strictly capped at max_keys during scanner flood."""
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=5,
        window_seconds=60,
        max_keys=15,
        enabled=True,
    )

    @app.get("/api/test")
    async def get_test():
        return {"ok": True}

    client = TestClient(app)
    # 50 unique client IPs
    for i in range(50):
        resp = client.get("/api/test", headers={"X-Forwarded-For": f"198.51.100.{i}"})
        assert resp.status_code == 200

    # Locate middleware instance in stack
    curr = app.middleware_stack
    mw = None
    while curr is not None:
        if isinstance(curr, RateLimitMiddleware):
            mw = curr
            break
        curr = getattr(curr, "app", None)

    assert mw is not None
    # Must never exceed max_keys
    assert len(mw._history) <= 15

