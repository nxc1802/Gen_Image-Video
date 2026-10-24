"""
Real audio enhancement adapter.
"""
from typing import Any, Dict
from factory_contracts.errors import InvalidRequestError
from services.enhancement.backend import call_backend


class AudioEnhancementProcessor:
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not inputs.get("audio_url"):
            raise InvalidRequestError("Audio enhancement requires 'audio_url'.")
        mode = str(inputs.get("mode", "denoise")).lower()
        if mode not in ("denoise", "voice_clarity"):
            raise InvalidRequestError("Unsupported audio enhancement mode.")
        return call_backend("AUDIO_ENHANCE_BACKEND_URL", "/v1/audio_enhancement", inputs)
