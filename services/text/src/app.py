"""
services/text/src/app.py
FastAPI application for the Factory v4 Text & VLM Subsystem (Port 8080).
"""
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from factory_auth import verify_bearer_token
from factory_contracts.errors import FactoryError
from services.text.src.config import load_text_config
from services.text.src.engine import TextEngine
from services.text.src.models import ChatCompletionRequest, ResponseApiRequest
from services.text.src.responses_handler import ResponsesHandler

logger = logging.getLogger("services.text")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID")
        if not req_id or not req_id.startswith("req-"):
            req_id = f"req-{uuid.uuid4()}"
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(FactoryError)
    async def handle_factory_error(request: Request, exc: FactoryError):
        req_id = getattr(request.state, "request_id", f"req-{uuid.uuid4()}")
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
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", f"req-{uuid.uuid4()}")
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
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        req_id = getattr(request.state, "request_id", f"req-{uuid.uuid4()}")
        code_map = {
            400: "INVALID_REQUEST",
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
    async def handle_generic_exception(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", f"req-{uuid.uuid4()}")
        logger.error(f"Unhandled error in text service [req_id={req_id}]: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "request_id": req_id,
                    "details": {},
                }
            },
            headers={"X-Request-ID": req_id},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    from factory_auth import load_auth_config
    app.state.auth_config = load_auth_config()
    app.state.config = load_text_config()
    app.state.engine = TextEngine()
    app.state.responses_handler = ResponsesHandler()
    yield


app = FastAPI(
    title="Factory v4 Text Generation Daemon",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)


@app.get("/health")
async def health_check():
    """Health endpoint for container probes and load balancers."""
    return {
        "status": "healthy",
        "service": "text-service",
        "version": "1.0.0",
        "port": 8080,
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_bearer_token)])
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible Chat Completions endpoint with streaming and structured output."""
    engine: TextEngine = getattr(app.state, "engine", None) or TextEngine()
    engine.validate_request(request)

    if request.stream:
        return StreamingResponse(
            engine.stream_generate(request),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return engine.generate(request)


@app.post("/v1/responses", dependencies=[Depends(verify_bearer_token)])
async def responses_api(request: ResponseApiRequest):
    """OpenAI Responses API endpoint for structured multi-step reasoning."""
    handler: ResponsesHandler = getattr(app.state, "responses_handler", None) or ResponsesHandler()
    return handler.execute_reasoning(request)
