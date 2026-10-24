"""
services/image/config.py
Configuration for Image Service Daemon and Worker (GPU 0, Port 8188).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageServiceConfig:
    port: int = 8188
    internal_comfy_port: int = 18188
    comfy_host: str = "127.0.0.1"
    gpu_device: int = 0
    worker_id: str = "worker-image-gpu-0"
    control_api_url: str = field(
        default_factory=lambda: os.environ.get("CONTROL_PLANE_URL") or os.environ.get("FACTORY_BASE_URL", "http://127.0.0.1:8000")
    )
    api_key: str = field(default_factory=lambda: os.environ.get("FACTORY_API_KEY", "sk-factory-test-token-valid"))
    comfy_dir: Path = field(default_factory=lambda: Path(os.environ.get("COMFYUI_DIR", "ComfyUI")))
    input_dir: Path = field(default_factory=lambda: Path(os.environ.get("COMFYUI_INPUT_DIR", "ComfyUI/input")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("COMFYUI_OUTPUT_DIR", "ComfyUI/output")))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET_NAME", "factory-artifacts"))
    s3_public_url: str = field(default_factory=lambda: os.environ.get("S3_PUBLIC_ENDPOINT_URL", "https://s3.ai.example.com"))
    lease_duration_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    poll_interval_seconds: float = 2.0


def load_image_config() -> ImageServiceConfig:
    port = int(os.environ.get("IMAGE_SERVICE_PORT", "8188"))
    internal_port = int(os.environ.get("IMAGE_COMFY_INTERNAL_PORT", "18188"))
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    worker_id = os.environ.get("WORKER_ID", "worker-image-gpu-0")
    return ImageServiceConfig(
        port=port,
        internal_comfy_port=internal_port,
        gpu_device=gpu,
        worker_id=worker_id,
    )
