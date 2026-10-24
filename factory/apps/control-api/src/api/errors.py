"""
factory/apps/control-api/src/api/errors.py
Sanitized Global Exception Handlers enforcing UnifiedErrorResponse
"""
import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from factory_contracts.errors import FactoryError

logger = logging.getLogger("factory.control_api.errors")


def register_exception_handlers(app: FastAPI) -> None:
    """Registers sanitized exception handlers that prevent traceback or internal path leaks."""

    @app.exception_handler(FactoryError)
    async def factory_error_handler(request: Request, exc: FactoryError):
        req_id = getattr(request.state, "request_id", "req-unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": req_id,
                    "details": exc.details,
                }
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", "req-unknown")
        sanitized_details = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            sanitized_details.append({
                "field": loc,
                "issue": str(err.get("msg", "Validation error")),
                "type": str(err.get("type", "value_error")),
            })
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed. Inspect details for field errors.",
                    "request_id": req_id,
                    "details": {"validation_errors": sanitized_details},
                }
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = getattr(request.state, "request_id", "req-unknown")
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            409: "CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMITED",
            502: "BAD_GATEWAY",
            503: "SERVICE_UNAVAILABLE",
            504: "GATEWAY_TIMEOUT",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "request_id": req_id,
                    "details": {},
                }
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "req-unknown")
        tb = traceback.format_exc()
        logger.error(
            f"💥 UNHANDLED EXCEPTION [request_id={req_id}] on {request.method} {request.url.path}:\n{tb}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred. Please contact system support with the request ID.",
                    "request_id": req_id,
                    "details": {},
                }
            },
            headers={"X-Request-ID": req_id},
        )
