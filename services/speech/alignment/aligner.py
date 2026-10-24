"""
Strict forced-alignment adapter.

Word timestamps must come from a real aligner backend. Character-duration
estimation is intentionally forbidden.
"""
import os
from typing import Any, Dict

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class ForcedAligner:
    def __init__(self, upstream_url: str | None = None):
        self.upstream_url = (upstream_url or os.environ.get("QWEN_ALIGNER_URL") or "").rstrip("/")
        self.api_key = os.environ.get("QWEN_ALIGNER_API_KEY") or os.environ.get("FACTORY_API_KEY", "")

    def align(self, audio_url: str, transcript_text: str) -> Dict[str, Any]:
        if not audio_url or not transcript_text or not transcript_text.strip():
            raise InvalidRequestError("Forced alignment requires 'audio_url' and non-empty 'transcript_text'.")
        if not self.upstream_url:
            raise ModelUnavailableError(
                message="Real forced-alignment backend is not configured.",
                details={"required_env": "QWEN_ALIGNER_URL"},
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.upstream_url}/v1/audio/alignments",
                    headers=headers,
                    json={"audio_url": audio_url, "transcript_text": transcript_text},
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Forced-alignment backend rejected the request.",
                        details={"status_code": response.status_code},
                    )
                payload = response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Forced-alignment backend is unavailable.",
                details={"error_type": type(exc).__name__},
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("aligned_words"), list):
            raise ModelUnavailableError(message="Forced-alignment backend returned an invalid response.")
        return payload
