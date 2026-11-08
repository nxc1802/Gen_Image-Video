"""
Real image super-resolution adapter.
"""
from typing import Any, Dict
from factory_contracts.errors import InvalidRequestError
from services.enhancement.backend import call_backend


class ImageSuperResolutionProcessor:
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not inputs.get("image_url"):
            raise InvalidRequestError("Image super-resolution requires 'image_url'.")
        scale_factor = int(inputs.get("scale_factor", 2))
        if scale_factor not in (2, 4):
            raise InvalidRequestError("Image super-resolution supports scale_factor 2 or 4.")
        model_name = inputs.get("model_name") or inputs.get("model")
        if model_name:
            norm_model = str(model_name).lower().strip().replace("_", "-")
            if norm_model not in ("real-esrgan", "realesrgan", "bsrgan", "enhancement-suite-v1"):
                raise InvalidRequestError(f"Unsupported image SR model '{model_name}'. Supported models: Real-ESRGAN, BSRGAN.")
        return call_backend("IMAGE_SR_BACKEND_URL", "/v1/image_super_resolution", inputs)
