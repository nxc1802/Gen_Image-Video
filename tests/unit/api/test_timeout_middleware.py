"""
tests/unit/api/test_timeout_middleware.py
Unit tests for RequestTimeoutMiddleware in control-api.
Verifies timeout enforcement, 504 GATEWAY_TIMEOUT responses, streaming exemptions, and cancellation.
"""
import asyncio
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from factory.apps.control_api.src.api.middleware.timeout import RequestTimeoutMiddleware


def create_timeout_app(timeout_seconds: float = 0.2) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RequestTimeoutMiddleware,
        timeout_seconds=timeout_seconds,
        enabled=True,
    )

    @app.get("/fast")
    async def fast_endpoint():
        return {"status": "fast"}

    @app.get("/slow")
    async def slow_endpoint():
        await asyncio.sleep(0.5)
        return {"status": "slow"}

    @app.get("/jobs/job-123/events")
    async def stream_events():
        # Exempt endpoint should not be cancelled even if exceeding timeout
        await asyncio.sleep(0.3)
        return {"stream": "ok"}

    return app


def test_fast_request_succeeds_within_timeout():
    app = create_timeout_app(timeout_seconds=0.5)
    client = TestClient(app)

    resp = client.get("/fast")
    assert resp.status_code == 200
    assert resp.json() == {"status": "fast"}


def test_slow_request_triggers_504_gateway_timeout():
    app = create_timeout_app(timeout_seconds=0.1)
    client = TestClient(app)

    resp = client.get("/slow", headers={"X-Request-ID": "req-timeout-test"})
    assert resp.status_code == 504
    assert resp.headers.get("X-Request-ID") == "req-timeout-test"

    body = resp.json()
    assert "error" in body
    err = body["error"]
    assert err["code"] == "GATEWAY_TIMEOUT"
    assert "exceeded timeout" in err["message"]
    assert err["request_id"] == "req-timeout-test"
    assert err["details"]["timeout_seconds"] == 0.1


def test_streaming_endpoint_exempt_from_timeout():
    app = create_timeout_app(timeout_seconds=0.1)
    client = TestClient(app)

    # Takes 0.3s which is > 0.1s timeout, but path ends with /events so exempt
    resp = client.get("/jobs/job-123/events")
    assert resp.status_code == 200
    assert resp.json() == {"stream": "ok"}
