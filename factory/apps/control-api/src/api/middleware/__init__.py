"""
factory/apps/control-api/src/api/middleware/__init__.py
Control API Middleware Suite: Request ID, Telemetry, Rate Limiting, and Timeouts.
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .rate_limit import RateLimitMiddleware
from .timeout import RequestTimeoutMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a correlated request_id in request.state and response headers."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID")
        if not req_id or not req_id.startswith("req-"):
            req_id = f"req-{uuid.uuid4()}"
        request.state.request_id = req_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


__all__ = [
    "RequestIDMiddleware",
    "RateLimitMiddleware",
    "RequestTimeoutMiddleware",
]
