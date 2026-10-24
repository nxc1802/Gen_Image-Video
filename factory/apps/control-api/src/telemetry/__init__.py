"""
factory/apps/control-api/src/telemetry/__init__.py
Public exports for OpenTelemetry distributed tracing and observability.
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
from .tracer import (
    generate_trace_id,
    generate_span_id,
    init_tracer,
    get_tracer,
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
    TelemetryLogFormatter,
    configure_telemetry_logging,
)
from .metrics import (
    metrics,
    TelemetryMetrics,
)
from .middleware import TelemetryTracingMiddleware

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
    "TelemetryLogFormatter",
    "configure_telemetry_logging",
    "metrics",
    "TelemetryMetrics",
    "TelemetryTracingMiddleware",
]
