"""
Strict Qwen3-ASR client backed by native vLLM Speech-to-Text serving.

No fabricated transcript fallback is permitted.
"""
import os
from typing import Any, Dict, Optional

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class ASREngine:
    def __init__(self, model_id: str = "qwen3-asr", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.upstream_url = os.environ.get("QWEN_ASR_URL", "").rstrip("/")
        self.api_key = os.environ.get("QWEN_ASR_API_KEY") or os.environ.get("FACTORY_API_KEY", "")

    def transcribe(
        self,
        audio_bytes: Optional[bytes] = None,
        audio_url: Optional[str] = None,
        language: Optional[str] = None,
        model: str = "qwen3-asr",
        temperature: float = 0.0,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Native Qwen3-ASR/vLLM backend is not configured.",
                details={"required_env": "QWEN_ASR_URL"},
            )
        if audio_url:
            raise InvalidRequestError(
                "Remote audio_url ingestion is disabled at the ASR worker boundary. "
                "Upload/ingest the asset through the Factory Asset Service and submit audio bytes."
            )
        if not audio_bytes:
            raise InvalidRequestError("Audio bytes are required for transcription.")

        data: Dict[str, str] = {
            "model": model or self.model_id,
            "response_format": "verbose_json",
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = str(temperature)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.upstream_url}/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Native ASR backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                payload = response.json()
                if not isinstance(payload, dict) or "text" not in payload:
                    raise ModelUnavailableError(message="Native ASR backend returned an invalid response.")
                return payload
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Native ASR backend is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
