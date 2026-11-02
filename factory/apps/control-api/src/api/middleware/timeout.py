"""
factory/apps/control-api/src/api/middleware/timeout.py
Global Request Timeout Middleware for FastAPI / Starlette.
Enforces execution time bounds and returns 504 GATEWAY_TIMEOUT on expiration.
"""
import asyncio
import os
from typing import Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Enforces a maximum execution duration per HTTP request.
    Long-lived connections (SSE streams, WebSockets) are automatically exempt.
    """

    def __init__(
        self,
        app,
        timeout_seconds: float = 60.0,
        exempt_path_prefixes: Optional[Set[str]] = None,
        exempt_path_suffixes: Optional[Set[str]] = None,
        enabled: Optional[bool] = None,
    ):
        super().__init__(app)
        env_timeout = os.environ.get("REQUEST_TIMEOUT_SECONDS")
        if env_timeout:
            try:
                self.timeout_seconds = float(env_timeout)
            except ValueError:
                self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = timeout_seconds

        if enabled is not None:
            self.enabled = enabled
        else:
            env_enabled = os.environ.get("REQUEST_TIMEOUT_ENABLED", "true").lower()
            self.enabled = env_enabled in ("true", "1", "yes")

        self.exempt_path_prefixes = exempt_path_prefixes or {"/docs", "/openapi.json", "/redoc"}
        self.exempt_path_suffixes = exempt_path_suffixes or {"/events", "/tokens", "/ws", "/stream"}

    def _is_exempt(self, request: Request) -> bool:
        path = request.url.path
        if any(path.startswith(prefix) for prefix in self.exempt_path_prefixes):
            return True
        if any(path.endswith(suffix) for suffix in self.exempt_path_suffixes):
            return True
        # Check for WebSocket upgrade headers
        if request.headers.get("upgrade", "").lower() == "websocket":
            return True
        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled or self._is_exempt(request):
            return await call_next(request)

        req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or "req-unknown"

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "code": "GATEWAY_TIMEOUT",
                        "message": f"Request execution exceeded timeout limit of {self.timeout_seconds}s.",
                        "request_id": req_id,
                        "details": {"timeout_seconds": self.timeout_seconds},
                    }
                },
                headers={"X-Request-ID": req_id},
            )
