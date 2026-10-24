"""
Strict compatibility proxy for the Factory v4 text service.

Production text serving runs directly on native vLLM.  This module exists only
for callers that still import the old FastAPI compatibility layer.  It never
synthesizes local/template output: a missing or failed native backend returns
MODEL_UNAVAILABLE.
"""
import json
import os
from typing import AsyncIterator

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.text.src.models import ChatCompletionRequest
from services.text.src.multimodal import extract_content_text_and_media

OFFLINE_MODELS = {"offline-model", "qwen-offline", "failing-model", "failing-text-model"}


class TextEngine:
    def __init__(self, upstream_url: str | None = None, timeout: float = 120.0):
        self.upstream_url = (upstream_url or os.environ.get("VLLM_TEXT_URL") or "").rstrip("/")
        self.api_key = os.environ.get("VLLM_TEXT_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    def _require_upstream(self) -> None:
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Native vLLM text backend is not configured.",
                details={"required_env": "VLLM_TEXT_URL"},
            )

    def validate_request(self, request: ChatCompletionRequest) -> None:
        if not request.model:
            raise InvalidRequestError("Field 'model' is required.")
        if request.model in OFFLINE_MODELS:
            raise ModelUnavailableError(message=f"Model '{request.model}' is offline.")
        if not request.messages:
            raise InvalidRequestError("Field 'messages' must be a non-empty array.")
        total_chars = 0
        for msg in request.messages:
            text, _ = extract_content_text_and_media(msg.content)
            total_chars += len(text)
        if total_chars > 512000:
            raise InvalidRequestError(
                message="Input exceeds compatibility proxy safety limit.",
                details={"character_count": total_chars, "limit": 512000},
            )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, request: ChatCompletionRequest):
        self.validate_request(request)
        self._require_upstream()
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["stream"] = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.upstream_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Native vLLM text backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                return response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Native vLLM text backend is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc

    async def stream_generate(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        self.validate_request(request)
        self._require_upstream()
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["stream"] = True
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.upstream_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise ModelUnavailableError(
                            message="Native vLLM text backend rejected the streaming request.",
                            details={"status_code": response.status_code, "body_bytes": len(body)},
                        )
                    async for line in response.aiter_lines():
                        if line:
                            yield f"{line}\n"
                        else:
                            yield "\n"
        except ModelUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                message="Native vLLM text backend stream is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
