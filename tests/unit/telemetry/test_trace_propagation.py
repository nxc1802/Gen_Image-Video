"""
tests/unit/telemetry/test_trace_propagation.py
Unit tests for W3C trace context propagation and FactoryJob.trace_id persistence.
"""
import json
import logging
import pytest
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from factory_contracts.jobs.job_contract import FactoryJob, JobSubmissionRequest, JobProgress, JobState
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.telemetry.context import set_correlation_context, clear_correlation_context
from factory.apps.control_api.src.telemetry.logging import TelemetryLogFormatter
from factory.apps.control_api.src.telemetry.propagation import build_traceparent, extract_traceparent
from factory.apps.control_api.src.telemetry.middleware import TelemetryTracingMiddleware


@pytest.mark.asyncio
async def test_job_lifecycle_manager_persists_trace_id():
    repo = InMemoryJobStore()
    await repo.initialize()
    broker = MemoryQueueBroker()
    await broker.initialize()
    stream_mgr = StreamManager(repository=repo)
    manager = JobLifecycleManager(repository=repo, broker=broker, stream_manager=stream_mgr)

    req = JobSubmissionRequest(
        service="text",
        model="qwen-text",
        task="chat_completion",
        inputs={"messages": [{"role": "user", "content": "hello"}]},
    )
    test_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

    job, created = await manager.submit_job(
        req,
        request_id="req-test-trace-1234",
        trace_id=test_trace_id,
    )

    assert created is True
    assert job.trace_id == test_trace_id

    # Verify retrieval from repository preserves trace_id
    retrieved = await repo.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.trace_id == test_trace_id

    await repo.close()
    await broker.close()


def test_telemetry_tracing_middleware_generates_and_attaches_trace_headers():
    app = FastAPI()
    app.add_middleware(TelemetryTracingMiddleware)

    @app.get("/test-endpoint")
    async def handler():
        return {"ok": True}

    client = TestClient(app)
    custom_trace_id = "0af7651916cd43dd8448eb211c80319c"
    custom_span_id = "b7ad6b7169203331"
    incoming_traceparent = f"00-{custom_trace_id}-{custom_span_id}-01"

    resp = client.get("/test-endpoint", headers={"traceparent": incoming_traceparent})
    assert resp.status_code == 200
    assert "traceparent" in resp.headers
    assert "X-Trace-ID" in resp.headers
    assert resp.headers["X-Trace-ID"] == custom_trace_id
    assert custom_trace_id in resp.headers["traceparent"]


def test_telemetry_log_formatter_includes_trace_id():
    formatter = TelemetryLogFormatter()
    clear_correlation_context()

    set_correlation_context(
        request_id="req-log-test",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        job_id="job-112233",
    )

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Executing model inference step",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    log_data = json.loads(formatted)

    assert log_data["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert log_data["span_id"] == "00f067aa0ba902b7"
    assert log_data["request_id"] == "req-log-test"
    assert log_data["job_id"] == "job-112233"
    assert log_data["message"] == "Executing model inference step"

    clear_correlation_context()
