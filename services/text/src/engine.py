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


def decouple_think_tokens(raw_text: str | None) -> tuple[str | None, str | None]:
    """
    Decouples <think>...</think> reasoning tokens from final generated content.
    Returns (reasoning_content, clean_content).
    """
    if not raw_text:
        return None, raw_text

    if "<think>" in raw_text and "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        think_part = parts[0]
        clean = parts[1].strip()
        if "<think>" in think_part:
            pre, r = think_part.split("<think>", 1)
            reasoning = r.strip()
            if pre.strip():
                clean = f"{pre.strip()} {clean}".strip()
        else:
            reasoning = think_part.strip()
        return (reasoning if reasoning else None), clean
    elif "</think>" in raw_text:
        parts = raw_text.rsplit("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        clean = parts[1].strip()
        return (reasoning if reasoning else None), clean
    elif "<think>" in raw_text:
        parts = raw_text.split("<think>", 1)
        clean = parts[0].strip()
        reasoning = parts[1].strip()
        return (reasoning if reasoning else None), clean
    else:
        return None, raw_text


class StreamingReasoningFilter:
    """
    Stateful SSE line filter that detects <think> tags across streaming chunks.
    Isolates tokens inside <think> into delta.reasoning_content and tokens outside into delta.content.
    """
    def __init__(self):
        self.in_think = False

    def process_chunk_delta(self, delta: dict) -> dict:
        content = delta.get("content")
        if content is None:
            return delta

        res = dict(delta)
        if "reasoning_content" in delta and delta["reasoning_content"]:
            return delta

        text = content
        if "<think>" in text:
            self.in_think = True
            text = text.replace("<think>", "")

        if "</think>" in text:
            parts = text.split("</think>", 1)
            reasoning_part = parts[0]
            answer_part = parts[1]
            self.in_think = False
            res.pop("content", None)
            if reasoning_part:
                res["reasoning_content"] = reasoning_part
            if answer_part:
                res["content"] = answer_part
            return res

        res.pop("content", None)
        if self.in_think:
            if text:
                res["reasoning_content"] = text
        else:
            if text:
                res["content"] = text
        return res


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
        if request.reasoning_effort:
            norm = request.reasoning_effort.lower()
            if norm not in {"none", "low", "medium", "high"}:
                raise InvalidRequestError(
                    message="Field 'reasoning_effort' must be one of: 'none', 'low', 'medium', 'high'.",
                    details={"reasoning_effort": request.reasoning_effort},
                )
            request.reasoning_effort = norm
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
                data = response.json()
                if isinstance(data, dict) and "choices" in data:
                    for choice in data.get("choices", []):
                        msg = choice.get("message")
                        if isinstance(msg, dict):
                            raw_content = msg.get("content")
                            if raw_content and ("<think>" in raw_content or "</think>" in raw_content):
                                r_content, c_content = decouple_think_tokens(raw_content)
                                if r_content is not None:
                                    msg["reasoning_content"] = r_content
                                msg["content"] = c_content
                return data
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
        filter_obj = StreamingReasoningFilter()
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
                        if not line:
                            yield "\n"
                            continue
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            payload_str = line[6:].strip()
                            try:
                                chunk_data = json.loads(payload_str)
                                choices = chunk_data.get("choices", [])
                                for choice in choices:
                                    delta = choice.get("delta")
                                    if isinstance(delta, dict):
                                        choice["delta"] = filter_obj.process_chunk_delta(delta)
                                yield f"data: {json.dumps(chunk_data)}\n"
                            except Exception:
                                yield f"{line}\n"
                        else:
                            yield f"{line}\n"
        except ModelUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                message="Native vLLM text backend stream is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
