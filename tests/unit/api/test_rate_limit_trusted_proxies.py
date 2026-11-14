"""
tests/unit/api/test_rate_limit_trusted_proxies.py
Unit tests verifying proxy-aware rate limiting (Requirement R3, P1.6):
- X-Forwarded-For is ignored when the direct client socket host is NOT in trusted_proxies
- Spoofed X-Forwarded-For cannot bypass rate limits
- X-Forwarded-For is respected when direct client host IS a verified trusted proxy
- Custom TRUSTED_PROXIES env var parsing and defaults
"""
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from factory.apps.control_api.src.api.middleware.rate_limit import RateLimitMiddleware


def create_rate_limited_app(trusted_proxies=None, limit=3, window=60):
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=limit,
        window_seconds=window,
        enabled=True,
        trusted_proxies=trusted_proxies,
    )

    @app.get("/api/ping")
    async def ping(request: Request):
        return {"pong": True}

    return app


def test_rate_limiter_ignores_spoofed_xff_from_untrusted_direct_client():
    """
    When an untrusted direct client (e.g. 198.51.100.1) connects and sends
    arbitrary/random X-Forwarded-For headers, the rate limiter MUST track the quota
    against the socket IP (198.51.100.1) and NOT the spoofed header.
    """
    # Trusted proxies only contain 127.0.0.1 and localhost
    app = create_rate_limited_app(trusted_proxies={"127.0.0.1", "localhost"}, limit=3)

    # TestClient by default sends request.client.host = "testclient"
    # Since "testclient" is NOT in trusted_proxies {"127.0.0.1", "localhost"}, it is untrusted!
    client = TestClient(app)

    # Send 3 requests with different spoofed X-Forwarded-For headers
    for i in range(3):
        resp = client.get("/api/ping", headers={"X-Forwarded-For": f"10.0.0.{i+1}"})
        assert resp.status_code == 200

    # 4th request with yet another new spoofed IP must be BLOCKED (429)
    resp4 = client.get("/api/ping", headers={"X-Forwarded-For": "10.0.0.99"})
    assert resp4.status_code == 429
    assert resp4.json()["error"]["code"] == "RATE_LIMITED"


def test_rate_limiter_respects_xff_from_trusted_proxy():
    """
    When a verified trusted proxy (e.g. "testclient") forwards requests with
    different X-Forwarded-For headers, each downstream client IP gets its own quota.
    """
    app = create_rate_limited_app(trusted_proxies={"testclient", "127.0.0.1"}, limit=2)
    client = TestClient(app)

    # Downstream IP 1: exhaust quota (2 allowed)
    r1 = client.get("/api/ping", headers={"X-Forwarded-For": "203.0.113.10"})
    assert r1.status_code == 200
    r2 = client.get("/api/ping", headers={"X-Forwarded-For": "203.0.113.10"})
    assert r2.status_code == 200
    r3 = client.get("/api/ping", headers={"X-Forwarded-For": "203.0.113.10"})
    assert r3.status_code == 429

    # Downstream IP 2: distinct fresh quota
    r_ip2 = client.get("/api/ping", headers={"X-Forwarded-For": "203.0.113.20"})
    assert r_ip2.status_code == 200


def test_trusted_proxies_environment_configuration(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXIES", "10.0.0.1, 10.0.0.2, 192.168.1.1")
    mw = RateLimitMiddleware(app=FastAPI(), enabled=False)
    assert mw.trusted_proxies == {"10.0.0.1", "10.0.0.2", "192.168.1.1"}
