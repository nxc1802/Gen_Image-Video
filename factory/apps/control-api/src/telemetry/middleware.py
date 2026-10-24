"""
factory/apps/control-api/src/telemetry/middleware.py
OpenTelemetry W3C traceparent and correlation middleware for FastAPI.
"""
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from factory.apps.control_api.src.telemetry.context import (
    clear_correlation_context,
    set_correlation_context,
)
from factory.apps.control_api.src.telemetry.propagation import (
    build_traceparent,
    extract_traceparent,
)
from factory.apps.control_api.src.telemetry.tracer import (
    generate_span_id,
    generate_trace_id,
)


class TelemetryTracingMiddleware(BaseHTTPMiddleware):
    """
    Extracts or mints W3C traceparent and request_id, propagates them
    to ContextVars, and attaches trace headers to every response.
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
            import uuid
            request_id = f"req-{uuid.uuid4()}"
            request.state.request_id = request_id

        set_correlation_context(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
        )

        try:
            response = await call_next(request)
        finally:
            # We preserve correlation during response header attachment
            pass

        # Attach standard observability headers
        tp = build_traceparent(trace_id, span_id)
        response.headers["traceparent"] = tp
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Request-ID"] = request_id

        return response
