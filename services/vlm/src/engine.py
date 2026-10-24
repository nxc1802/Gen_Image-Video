"""
Strict compatibility proxy for the native vLLM VLM service.
"""
import os
from typing import AsyncIterator

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.text.src.models import ChatCompletionRequest

OFFLINE_MODELS = {"offline-model", "qwen-offline", "failing-model", "failing-vlm-model"}


class VLMEngine:
    def __init__(self, upstream_url: str | None = None, timeout: float = 180.0):
        self.upstream_url = (upstream_url or os.environ.get("VLLM_VLM_URL") or "").rstrip("/")
        self.api_key = os.environ.get("VLLM_VLM_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    def _require_upstream(self) -> None:
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Native vLLM VLM backend is not configured.",
                details={"required_env": "VLLM_VLM_URL"},
            )

    def validate_request(self, request: ChatCompletionRequest) -> None:
        if not request.model:
            raise InvalidRequestError("Field 'model' is required.")
        if request.model in OFFLINE_MODELS:
            raise ModelUnavailableError(message=f"Model '{request.model}' is offline.")
        if not request.messages:
            raise InvalidRequestError("Field 'messages' must be a non-empty array.")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def generate(self, request: ChatCompletionRequest):
        self.validate_request(request)
        self._require_upstream()
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["stream"] = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.upstream_url}/v1/chat/completions", headers=self._headers(), json=payload)
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Native vLLM VLM backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                return response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Native vLLM VLM backend is unavailable.",
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
                        await response.aread()
                        raise ModelUnavailableError(
                            message="Native vLLM VLM backend rejected the streaming request.",
                            details={"status_code": response.status_code},
                        )
                    async for line in response.aiter_lines():
                        yield f"{line}\n" if line else "\n"
        except ModelUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                message="Native vLLM VLM backend stream is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
