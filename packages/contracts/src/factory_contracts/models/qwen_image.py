"""
packages/contracts/src/factory_contracts/models/qwen_image.py
Authoritative constants and helper specifications for Qwen-Image 2.1 Family.
"""
from typing import Dict
from factory_contracts.manifests.model import (
    ModelVariantSpec,
    ModelWeightFormat,
    QuantizationType,
    PrecisionType,
)

QWEN_IMAGE_2_1_TURBO_GGUF = ModelVariantSpec(
    id="turbo",
    display_name="Qwen-Image 2.1 Turbo GGUF (8 Euler steps)",
    checkpoint="qwen-image-2.1-turbo-Q4_K_M.gguf",
    format=ModelWeightFormat.GGUF,
    quantization=QuantizationType.Q4_K_M,
    default_precision=PrecisionType.BF16,
    recommended_steps=8,
    kaggle_dataset="cuongnguyen1802/qwen-image-2-1-turbo-gguf",
    huggingface_repo="unsloth/Qwen-Image-2.1-Turbo-GGUF",
    file_name="qwen-image-2.1-turbo-Q4_K_M.gguf",
    size_bytes=4199565120,
    min_vram_gb=14.0,
)

QWEN_IMAGE_VARIANTS: Dict[str, ModelVariantSpec] = {
    "turbo": QWEN_IMAGE_2_1_TURBO_GGUF,
}

__all__ = [
    "QWEN_IMAGE_2_1_TURBO_GGUF",
    "QWEN_IMAGE_VARIANTS",
]
