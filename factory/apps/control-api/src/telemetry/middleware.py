"""
factory/apps/control-api/src/telemetry/middleware.py
Request correlation and trace header middleware for FastAPI in Factory v4 Lite.
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from factory.apps.control_api.src.telemetry.context import set_correlation_context
from factory.apps.control_api.src.telemetry.propagation import (
    build_traceparent,
    extract_traceparent,
)


def generate_trace_id() -> str:
    """Generates a valid 128-bit trace ID (32 hex characters)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generates a valid 64-bit span ID (16 hex characters)."""
    return uuid.uuid4().hex[:16]


class TelemetryTracingMiddleware(BaseHTTPMiddleware):
    """
    Extracts or mints trace IDs and request_id, propagates them
    to ContextVars, and attaches correlation headers to every response.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id, parent_span_id = extract_traceparent(dict(request.headers))
        if not trace_id:
            trace_id = generate_trace_id()
        span_id = generate_span_id()

        request_id = getattr(request.state, "request_id", None)
        if not request_id:
            request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = f"req-{uuid.uuid4()}"
            request.state.request_id = request_id

        request.state.trace_id = trace_id
        request.state.span_id = span_id

        set_correlation_context(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
        )

        response = await call_next(request)

        # Attach standard observability headers
        tp = build_traceparent(trace_id, span_id)
        response.headers["traceparent"] = tp
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Request-ID"] = request_id

        return response


__all__ = ["TelemetryTracingMiddleware", "generate_trace_id", "generate_span_id"]
