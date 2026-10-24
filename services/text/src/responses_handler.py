"""
Strict compatibility proxy for vLLM's OpenAI-compatible Responses API.
"""
import os
import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.text.src.models import ResponseApiRequest


class ResponsesHandler:
    def __init__(self, upstream_url: str | None = None, timeout: float = 120.0):
        self.upstream_url = (upstream_url or os.environ.get("VLLM_TEXT_URL") or "").rstrip("/")
        self.api_key = os.environ.get("VLLM_TEXT_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    def execute_reasoning(self, request: ResponseApiRequest):
        if not request.model:
            raise InvalidRequestError("Field 'model' is required.")
        if not request.input or not request.input.strip():
            raise InvalidRequestError("Field 'input' cannot be empty.")
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Native vLLM Responses backend is not configured.",
                details={"required_env": "VLLM_TEXT_URL"},
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.upstream_url}/v1/responses",
                    headers=headers,
                    json=request.model_dump(mode="json", exclude_none=True),
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Native vLLM Responses backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                return response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Native vLLM Responses backend is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
