"""
Strict Qwen3-TTS client backed by native vLLM-Omni.

No harmonic/sine-wave speech synthesis fallback is permitted.
"""
import os
from typing import Optional

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class TTSEngine:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.upstream_url = os.environ.get("QWEN_TTS_URL", "").rstrip("/")
        self.api_key = os.environ.get("QWEN_TTS_API_KEY") or os.environ.get("FACTORY_API_KEY", "")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = "default",
        reference_audio_url: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "wav",
        model: str = "qwen3-tts",
        reference_text: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            raise InvalidRequestError("Field 'input' cannot be empty for TTS.")
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Native Qwen3-TTS/vLLM-Omni backend is not configured.",
                details={"required_env": "QWEN_TTS_URL"},
            )
        if reference_audio_url and not reference_text:
            raise InvalidRequestError(
                "Qwen3-TTS Base voice cloning requires reference_text together with reference_audio_url."
            )

        payload = {
            "input": text,
            "response_format": response_format or "wav",
            "stream": False,
        }
        if voice and voice != "default":
            payload["voice"] = voice
        if reference_audio_url:
            payload["ref_audio"] = reference_audio_url
            payload["ref_text"] = reference_text
            payload["task_type"] = "Base"
        if speed and speed != 1.0:
            payload["speed"] = speed

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.upstream_url}/v1/audio/speech",
                    headers=headers,
                    json=payload,
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Native TTS backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                if not response.content:
                    raise ModelUnavailableError(message="Native TTS backend returned empty audio.")
                return response.content
        except ModelUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                message="Native TTS backend is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
