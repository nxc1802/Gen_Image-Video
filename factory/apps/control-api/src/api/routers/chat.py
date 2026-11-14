"""OpenAI Text/Vision contracts over one canonical ResponsesEngine."""

import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from factory_auth import verify_bearer_token
from factory_contracts.errors import FactoryError, ModelNotLoadedError
from factory.engines.responses import ResponsesEngine
from factory.engines.responses.contracts import ChatCompletionRequest, ResponseApiRequest
from factory.apps.control_api.src.api.public import normalize_messages, normalize_response_input

router = APIRouter(prefix="/v1", tags=["Responses"])


def _require_authorized_session(request, model):
    registry = getattr(request.app.state, "deployment_registry", None)
    if registry is None:
        raise ModelNotLoadedError("No authorized deployment in this session.")
    registry.require_active(model)


def _get_text_engine(request):
    return ResponsesEngine(
        registry=request.app.state.deployment_registry,
        session=getattr(request.app.state, "backend_session", None),
    )


def _get_vlm_engine(request):
    return _get_text_engine(request)


async def stream_result(events):
    # Preflight and the first upstream record must succeed before HTTP 200.
    first = await anext(events)

    async def chunks():
        try:
            yield first
            async for event in events:
                yield event
        except Exception as exc:
            code = exc.code if isinstance(exc, FactoryError) else "BACKEND_UNAVAILABLE"
            yield (
                "event: error\ndata: "
                + json.dumps(
                    {
                        "error": {
                            "type": "server_error",
                            "code": code,
                            "param": None,
                            "message": "Selected backend stream did not complete.",
                        }
                    }
                )
                + "\n\n"
            )
        finally:
            await events.aclose()

    return StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/completions", dependencies=[Depends(verify_bearer_token)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    _require_authorized_session(http_request, request.model)
    await normalize_messages(request.messages, http_request)
    instance = _get_text_engine(http_request)
    instance.validate_request(request)
    if request.stream:
        return await stream_result(instance.stream_generate(request))
    return await asyncio.to_thread(instance.generate, request)


@router.post("/responses", dependencies=[Depends(verify_bearer_token)])
async def responses_api(request: ResponseApiRequest, http_request: Request):
    _require_authorized_session(http_request, request.model)
    request.input = await normalize_response_input(request.input, http_request)
    instance = _get_text_engine(http_request)
    if request.stream:
        return await stream_result(instance.stream_response(request))
    return await asyncio.to_thread(instance.execute_reasoning, request)
