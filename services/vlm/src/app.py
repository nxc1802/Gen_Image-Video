"""
services/vlm/src/app.py
FastAPI application for the Factory v4 Vision-Language (VLM) Subsystem (Port 8081).
"""
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from factory_auth import verify_bearer_token
from services.text.src.app import RequestIDMiddleware, register_error_handlers
from services.text.src.models import ChatCompletionRequest, ResponseApiRequest
from services.text.src.responses_handler import ResponsesHandler
from services.vlm.src.config import load_vlm_config
from services.vlm.src.engine import VLMEngine

logger = logging.getLogger("services.vlm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from factory_auth import load_auth_config
    app.state.auth_config = load_auth_config()
    app.state.config = load_vlm_config()
    app.state.engine = VLMEngine()
    app.state.responses_handler = ResponsesHandler()
    yield


app = FastAPI(
    title="Factory v4 Vision-Language Daemon (Qwen2-VL)",
    version="1.0.0",
    lifespan=lifespan,
)

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
    return {
        "status": "healthy",
        "service": "vision-service",
        "version": "1.0.0",
        "port": 8081,
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_bearer_token)])
async def chat_completions(request: ChatCompletionRequest):
    engine: VLMEngine = getattr(app.state, "engine", None) or VLMEngine()
    engine.validate_request(request)
    if request.stream:
        return StreamingResponse(
            engine.stream_generate(request),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return engine.generate(request)


@app.post("/v1/responses", dependencies=[Depends(verify_bearer_token)])
async def responses_api(request: ResponseApiRequest):
    handler: ResponsesHandler = getattr(app.state, "responses_handler", None) or ResponsesHandler()
    return handler.execute_reasoning(request)
