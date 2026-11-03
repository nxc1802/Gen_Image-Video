"""
factory/apps/control-api/src/telemetry/__init__.py
Public exports for structured logging and correlation in Factory v4 Lite.
"""
from .context import (
    set_correlation_context,
    get_correlation_dict,
    clear_correlation_context,
    ctx_request_id,
    ctx_job_id,
    ctx_worker_id,
    ctx_model_id,
    ctx_artifact_id,
    ctx_trace_id,
    ctx_span_id,
)
from .middleware import (
    generate_trace_id,
    generate_span_id,
    TelemetryTracingMiddleware,
)
from .propagation import (
    build_traceparent,
    extract_traceparent,
    inject_traceparent,
)
from .decorators import (
    trace_scope,
    traced_span,
)
from .logging import (
    LiteJsonLogFormatter,
    TelemetryLogFormatter,
    configure_lite_logging,
    configure_telemetry_logging,
)
from .metrics import (
    metrics,
    TelemetryMetrics,
)


def init_tracer(service_name: str = "factory.control-api"):
    """Deprecated: OpenTelemetry tracer replaced by LiteJsonLogFormatter."""
    return None


def get_tracer():
    """Deprecated: OpenTelemetry tracer replaced by LiteJsonLogFormatter."""
    return None


__all__ = [
    "set_correlation_context",
    "get_correlation_dict",
    "clear_correlation_context",
    "ctx_request_id",
    "ctx_job_id",
    "ctx_worker_id",
    "ctx_model_id",
    "ctx_artifact_id",
    "ctx_trace_id",
    "ctx_span_id",
    "generate_trace_id",
    "generate_span_id",
    "init_tracer",
    "get_tracer",
    "build_traceparent",
    "extract_traceparent",
    "inject_traceparent",
    "trace_scope",
    "traced_span",
    "LiteJsonLogFormatter",
    "TelemetryLogFormatter",
    "configure_lite_logging",
    "configure_telemetry_logging",
    "metrics",
    "TelemetryMetrics",
    "TelemetryTracingMiddleware",
]
