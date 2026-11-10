"""
OpenAI-compatible chat completions and responses router for Factory Control API.
"""
import os
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from factory_auth import verify_bearer_token
from services.text.src.models import ChatCompletionRequest, ResponseApiRequest
from services.text.src.engine import TextEngine
from services.text.src.responses_handler import ResponsesHandler
from services.vlm.src.engine import VLMEngine

router = APIRouter(prefix="/v1", tags=["Chat"])


def _get_text_engine() -> TextEngine:
    upstream = os.environ.get("VLLM_TEXT_URL") or os.environ.get("TEXT_SERVICE_URL")
    return TextEngine(upstream_url=upstream)


def _get_vlm_engine() -> VLMEngine:
    upstream = os.environ.get("VLLM_VLM_URL") or os.environ.get("VLM_SERVICE_URL")
    return VLMEngine(upstream_url=upstream)


@router.post("/chat/completions", dependencies=[Depends(verify_bearer_token)])
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible Chat Completions endpoint with streaming and structured output."""
    model_lower = str(request.model or "").lower()
    has_multimodal = any(isinstance(m.content, list) for m in request.messages) if request.messages else False
    is_vlm = "vl" in model_lower or "vlm" in model_lower or has_multimodal

    engine = _get_vlm_engine() if is_vlm else _get_text_engine()
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


@router.post("/responses", dependencies=[Depends(verify_bearer_token)])
async def responses_api(request: ResponseApiRequest):
    """OpenAI Responses API endpoint for structured multi-step reasoning."""
    upstream = os.environ.get("VLLM_TEXT_URL") or os.environ.get("TEXT_SERVICE_URL")
    handler = ResponsesHandler(upstream_url=upstream)
    return handler.execute_reasoning(request)
