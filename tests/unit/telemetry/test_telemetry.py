"""
tests/unit/telemetry/test_telemetry.py
Comprehensive unit tests for OpenTelemetry 5-hop correlation context, W3C traceparent,
span decorators, structured JSON telemetry logging, and metrics collectors.
"""
import json
import logging
import pytest

from factory.apps.control_api.src.telemetry.context import (
    clear_correlation_context,
    get_correlation_dict,
    set_correlation_context,
)
from factory.apps.control_api.src.telemetry.decorators import (
    trace_scope,
    traced_span,
)
from factory.apps.control_api.src.telemetry.logging import (
    TelemetryLogFormatter,
)
from factory.apps.control_api.src.telemetry.metrics import (
    TelemetryMetrics,
    metrics,
)
from factory.apps.control_api.src.telemetry.propagation import (
    build_traceparent,
    extract_traceparent,
    inject_traceparent,
)
from factory.apps.control_api.src.telemetry.tracer import (
    generate_span_id,
    generate_trace_id,
)


class TestCorrelationContext:
    def test_set_and_get_5_hop_correlation(self):
        clear_correlation_context()
        set_correlation_context(
            request_id="req-123",
            job_id="job-456",
            worker_id="worker-789",
            model_id="qwen-image-2.1",
            artifact_id="art-001",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
        )

        corr = get_correlation_dict()
        assert corr["request_id"] == "req-123"
        assert corr["job_id"] == "job-456"
        assert corr["worker_id"] == "worker-789"
        assert corr["model_id"] == "qwen-image-2.1"
        assert corr["artifact_id"] == "art-001"
        assert corr["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert corr["span_id"] == "00f067aa0ba902b7"

        clear_correlation_context()
        empty_corr = get_correlation_dict()
        assert empty_corr["request_id"] is None
        assert empty_corr["job_id"] is None


class TestTraceparentPropagation:
    def test_build_and_extract_traceparent(self):
        t_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        s_id = "00f067aa0ba902b7"
        header_val = build_traceparent(t_id, s_id, sampled=True)
        assert header_val == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

        extracted_t, extracted_s = extract_traceparent({"traceparent": header_val})
        assert extracted_t == t_id
        assert extracted_s == s_id

    def test_extract_traceparent_case_insensitive(self):
        headers = {"TraceParent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        extracted_t, extracted_s = extract_traceparent(headers)
        assert extracted_t == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert extracted_s == "00f067aa0ba902b7"

    def test_extract_fallback_x_trace_id(self):
        t_id = "11223344556677889900aabbccddeeff"
        headers = {"X-Trace-ID": t_id}
        extracted_t, extracted_s = extract_traceparent(headers)
        assert extracted_t == t_id
        assert len(extracted_s) == 16

    def test_inject_traceparent(self):
        carrier = {}
        t_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        s_id = "00f067aa0ba902b7"
        tp = inject_traceparent(carrier, trace_id=t_id, span_id=s_id)
        assert carrier["traceparent"] == tp
        assert carrier["x-trace-id"] == t_id


@pytest.mark.asyncio
class TestSpanDecorators:
    async def test_trace_scope_attaches_correlation(self):
        set_correlation_context(request_id="req-span-test", job_id="job-span-test")
        async with trace_scope("test_span_operation", custom_attr="val"):
            corr = get_correlation_dict()
            assert corr["span_id"] is not None

    async def test_traced_span_decorator(self):
        set_correlation_context(request_id="req-dec-test")

        @traced_span("decorated_func", service="text")
        async def my_func(x: int) -> int:
            return x * 2

        res = await my_func(21)
        assert res == 42


class TestTelemetryLogFormatter:
    def test_json_log_formatting_with_correlation(self):
        clear_correlation_context()
        set_correlation_context(
            request_id="req-log-123",
            job_id="job-log-456",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
        )

        formatter = TelemetryLogFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Worker completed generation successfully",
            args=(),
            exc_info=None,
        )

        formatted_str = formatter.format(record)
        data = json.loads(formatted_str)

        assert data["level"] == "INFO"
        assert data["message"] == "Worker completed generation successfully"
        assert data["request_id"] == "req-log-123"
        assert data["job_id"] == "job-log-456"
        assert data["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert data["span_id"] == "00f067aa0ba902b7"


class TestTelemetryMetrics:
    def test_metrics_collection_and_snapshot(self):
        tm = TelemetryMetrics()
        tm.set_active_jobs(3)
        tm.set_queue_depth(7)
        tm.set_gpu_vram(10.5, 14.2)
        tm.increment_requests()
        tm.increment_requests()
        tm.increment_failures()

        snap = tm.get_snapshot()
        assert snap["active_jobs"] == 3
        assert snap["queue_depth"] == 7
        assert snap["gpu_0_vram_used_gb"] == 10.5
        assert snap["gpu_1_vram_used_gb"] == 14.2
        assert snap["requests_total"] == 2
        assert snap["failures_total"] == 1
