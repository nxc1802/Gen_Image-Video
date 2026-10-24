"""
Real frame interpolation adapter.
"""
from typing import Any, Dict
from factory_contracts.errors import InvalidRequestError
from services.enhancement.backend import call_backend


class FrameInterpolationProcessor:
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not inputs.get("video_url"):
            raise InvalidRequestError("Frame interpolation requires 'video_url'.")
        multiplier = int(inputs.get("multiplier", 2))
        if multiplier not in (2, 4):
            raise InvalidRequestError("Frame interpolation supports multiplier 2 or 4.")
        return call_backend("RIFE_BACKEND_URL", "/v1/frame_interpolation", inputs)
