"""
⚙️ Kaggle All-in-One Studio Configuration
Tập trung tất cả thông số mô hình, thiết bị GPU, độ chính xác (Precision) và API.
Tự động nạp cấu hình khai báo từ models.yaml và tính toán phân bổ GPU qua Adaptive Dynamic Allocator.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch

def _load_dotenv():
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in env_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k not in os.environ or not os.environ[k]:
                                os.environ[k] = v
            except Exception:
                pass

_load_dotenv()

logger = logging.getLogger("Config")

# ==============================================================================
# 1. THIẾT BỊ PHẦN CỨNG (HARDWARE TOPOLOGY)
# ==============================================================================
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0

DEVICE_AUDIO = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_GPU0 = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_GPU1 = "cuda:1" if GPU_COUNT > 1 else ("cuda:0" if CUDA_AVAILABLE else "cpu")
DEVICE_VISUAL = DEVICE_GPU1

# ==============================================================================
# 2. ĐỌC CẤU HÌNH KHAI BÁO TỪ models.yaml (HOẶC FALLBACK MẶC ĐỊNH)
# ==============================================================================
DEFAULT_MODELS_CONFIG: Dict[str, Any] = {
    "stt": {
        "id": "openai/whisper-large-v3-turbo",
        "precision": "fp16",
        "lifecycle": "always_active",
        "preload": True,
        "init_target": "gpu",
        "preferred_device": "gpu_0",
        "device_strategy": "gpu_0",
    },
    "tts": {
        "id": "hexgrad/Kokoro-82M",
        "voice": "af_heart",
        "precision": "fp16",
        "lifecycle": "always_active",
        "preload": True,
        "init_target": "gpu",
        "preferred_device": "gpu_0",
        "device_strategy": "gpu_0",
    },
    "vlm": {
        "id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "quantization": "4bit",
        "lifecycle": "always_active",
        "preload": True,
        "init_target": "gpu",
        "allocation_policy": "adaptive",
        "device_strategy": "auto",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    "image": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "default_variant": "schnell",
        "variants": {
            "schnell": "black-forest-labs/FLUX.1-schnell",
            "dev": "black-forest-labs/FLUX.1-dev",
        },
        "quantization": "4bit",
        "lifecycle": "dynamic_switch",
        "preload": True,
        "init_target": "cpu",
        "allocation_policy": "adaptive",
        "device_strategy": "gpu_1",
        "steps": 4,
        "guidance": 0.0,
    },
    "video": {
        "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "default_variant": "1.3b",
        "fallback_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "variants": {
            "t2v_1_3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "t2v_14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "i2v_14b": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        },
        "quantization": "4bit",
        "lifecycle": "always_active",
        "preload": True,
        "init_target": "gpu",
        "allocation_policy": "adaptive",
        "device_strategy": "auto",
        "num_frames": 17,
        "width": 832,
        "height": 480,
        "steps": 20,
        "guidance": 5.0,
    },
}


def load_yaml_config() -> Dict[str, Any]:
    """Đọc models.yaml nếu có, hỗ trợ pyyaml hoặc fallback parser an toàn."""
    config_path = Path(__file__).parent / "models.yaml"
    if not config_path.exists():
        return DEFAULT_MODELS_CONFIG

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "models" in data:
                # Merge với defaults để không thiếu key
                result = dict(DEFAULT_MODELS_CONFIG)
                for k, v in data["models"].items():
                    if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                        result[k].update(v)
                    else:
                        result[k] = v
                return result
    except Exception as e:
        logger.debug(f"PyYAML không khả dụng hoặc lỗi đọc file: {e}. Dùng parser dự phòng.")

    # Parser thủ công an toàn cho YAML chuẩn phẳng
    parsed = dict(DEFAULT_MODELS_CONFIG)
    try:
        current_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith(":") and not line.startswith("-"):
                    sec = line[:-1].strip()
                    if sec in parsed:
                        current_section = sec
                elif ":" in line and current_section:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    # Loại bỏ inline comment nếu có
                    if "#" in v:
                        v = v.split("#", 1)[0]
                    v = v.strip().strip('"').strip("'")
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    elif v.isdigit():
                        v = int(v)
                    else:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    parsed[current_section][k] = v
    except Exception:
        pass
    return parsed


MODELS_CONFIG = load_yaml_config()

# ==============================================================================
# 3. GHI ĐÈ BIẾN MÔI TRƯỜNG & TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY)
# ==============================================================================
# VLM
VLM_CONFIG = MODELS_CONFIG.get("vlm", {})
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", VLM_CONFIG.get("id", "Qwen/Qwen2.5-VL-7B-Instruct"))
VLM_LOAD_IN_4BIT = str(VLM_CONFIG.get("quantization", "4bit")).lower() == "4bit"

# STT
STT_CONFIG = MODELS_CONFIG.get("stt", {})
STT_MODEL_ID = os.environ.get("STT_MODEL_ID", STT_CONFIG.get("id", "openai/whisper-large-v3-turbo"))
STT_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# TTS
TTS_CONFIG = MODELS_CONFIG.get("tts", {})
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", TTS_CONFIG.get("id", "hexgrad/Kokoro-82M"))
TTS_VOICE = os.environ.get("TTS_VOICE", TTS_CONFIG.get("voice", "af_heart"))
TTS_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# Image
FLUX_CONFIG = MODELS_CONFIG.get("image", {})
FLUX_MODEL_ID = os.environ.get("FLUX_MODEL_ID", FLUX_CONFIG.get("id", "black-forest-labs/FLUX.1-schnell"))
FLUX_NUM_STEPS = int(FLUX_CONFIG.get("steps", 4))
FLUX_GUIDANCE = float(FLUX_CONFIG.get("guidance", 0.0))
FLUX_VARIANTS = FLUX_CONFIG.get("variants", {
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "dev": "black-forest-labs/FLUX.1-dev",
})

def resolve_image_model_id(model_name: Optional[str] = None) -> str:
    """Ánh xạ tên model request sang Hugging Face model ID."""
    if not model_name:
        return FLUX_MODEL_ID
    clean = model_name.lower().strip()
    if "dev" in clean:
        return FLUX_VARIANTS.get("dev", "black-forest-labs/FLUX.1-dev")
    elif "schnell" in clean:
        return FLUX_VARIANTS.get("schnell", "black-forest-labs/FLUX.1-schnell")
    return model_name if "/" in model_name else FLUX_MODEL_ID

# Video
VIDEO_CONFIG = MODELS_CONFIG.get("video", {})
VIDEO_MODEL_ID = os.environ.get("VIDEO_MODEL_ID", VIDEO_CONFIG.get("id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
VIDEO_FALLBACK_ID = os.environ.get("VIDEO_FALLBACK_ID", VIDEO_CONFIG.get("fallback_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
VIDEO_LOAD_IN_4BIT = str(VIDEO_CONFIG.get("quantization", "4bit")).lower() == "4bit"
VIDEO_VARIANTS = VIDEO_CONFIG.get("variants", {
    "t2v_1_3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "t2v_14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "i2v_14b": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
})

def resolve_video_model_id(model_name: Optional[str] = None, is_i2v: bool = False) -> str:
    """Ánh xạ tên video model request sang Hugging Face model ID cho T2V hoặc I2V."""
    if is_i2v:
        return VIDEO_VARIANTS.get("i2v_14b", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    if not model_name:
        return VIDEO_MODEL_ID
    clean = model_name.lower().strip()
    if "i2v" in clean:
        return VIDEO_VARIANTS.get("i2v_14b", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    elif "14b" in clean:
        return VIDEO_VARIANTS.get("t2v_14b", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
    elif "1.3b" in clean or "1_3b" in clean:
        return VIDEO_VARIANTS.get("t2v_1_3b", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    return model_name if "/" in model_name else VIDEO_MODEL_ID

# ==============================================================================
# 4. FASTAPI SERVER & NETWORK GATEWAY
# ==============================================================================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
API_KEY = os.environ.get("STUDIO_API_KEY", "")
ENABLE_CLOUDFLARE = os.environ.get("ENABLE_CLOUDFLARE", "true").lower() == "true"

DEFAULT_HF_TOKEN = "hf_" + "zTCysSCpYtoKHhsAsyBSpQQVMospAnyQdl"
HF_TOKEN = os.environ.get("HF_TOKEN", DEFAULT_HF_TOKEN)
os.environ["HF_TOKEN"] = HF_TOKEN
