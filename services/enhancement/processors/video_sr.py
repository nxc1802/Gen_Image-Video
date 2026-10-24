"""
Real video super-resolution adapter.
"""
from typing import Any, Dict
from factory_contracts.errors import InvalidRequestError
from services.enhancement.backend import call_backend


class VideoSuperResolutionProcessor:
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not inputs.get("video_url"):
            raise InvalidRequestError("Video super-resolution requires 'video_url'.")
        scale_factor = int(inputs.get("scale_factor", 2))
        if scale_factor not in (2, 4):
            raise InvalidRequestError("Video super-resolution supports scale_factor 2 or 4.")
        return call_backend("VIDEO_SR_BACKEND_URL", "/v1/video_super_resolution", inputs)
