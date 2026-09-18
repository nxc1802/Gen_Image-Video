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
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "preferred_device": "gpu_0",
        "device_strategy": "gpu_0",
    },
    "tts": {
        "id": "hexgrad/Kokoro-82M",
        "voice": "af_heart",
        "precision": "fp16",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "preferred_device": "gpu_0",
        "device_strategy": "gpu_0",
    },
    "vlm": {
        "id": "nguynxuncngde180528/qwen38-27b-vlm-gguf",
        "fallback_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "format": "gguf",
        "model_file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "quantization": "q4_k_m",
        "precision": "fp16",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "allocation_policy": "dual_gpu",
        "device_strategy": "dual_gpu",
        "gpu_count": 2,
        "gpu_split": [0.5, 0.5],
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "max_tokens": 1024,
        "temperature": 0.7,
    },
    "image": {
        "id": "unsloth/FLUX.2-klein-4B-GGUF",
        "default_variant": "klein_4b",
        "model_file": "flux-2-klein-4b-Q4_K_M.gguf",
        "text_encoder_repo": "unsloth/Qwen3-4B-GGUF",
        "text_encoder_file": "Qwen3-4B-Q4_K_M.gguf",
        "variants": {
            "klein_4b": "unsloth/FLUX.2-klein-4B-GGUF",
            "4b": "unsloth/FLUX.2-klein-4B-GGUF",
            "klein_9b": "unsloth/FLUX.2-klein-9B-GGUF",
            "9b": "unsloth/FLUX.2-klein-9B-GGUF",
            "dev": "black-forest-labs/FLUX.2-dev",
        },
        "quantization": "q4_k_m",
        "precision": "fp16",
        "lifecycle": "dynamic_switch",
        "preload": True,
        "init_target": "gpu",
        "allocation_policy": "single_gpu",
        "device_strategy": "single_gpu",
        "steps": 4,
        "guidance": 1.0,
    },
    "video": {
        "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "default_variant": "ti2v_5b",
        "fallback_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "variants": {
            "ti2v_5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "t2v_a14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "a14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "i2v_a14b": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "ti2v_5b_22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "t2v_a14b_22": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "i2v_a14b_22": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "t2v_1_3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "t2v_14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "i2v_14b": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        },
        "quantization": "4bit",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
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
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", VLM_CONFIG.get("id", "nguynxuncngde180528/qwen38-27b-vlm-gguf"))
VLM_FALLBACK_ID = os.environ.get("VLM_FALLBACK_ID", VLM_CONFIG.get("fallback_id", "Qwen/Qwen2.5-VL-3B-Instruct"))
VLM_FORMAT = str(VLM_CONFIG.get("format", "gguf")).lower()
VLM_LOAD_IN_4BIT = str(VLM_CONFIG.get("quantization", "4bit")).lower() in ("4bit", "q4_k_m")
VLM_N_CTX = int(VLM_CONFIG.get("n_ctx", 4096))
VLM_GPU_SPLIT = VLM_CONFIG.get("gpu_split", [0.5, 0.5])

def resolve_vlm_model_paths(model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Tự động quét và xác định đường dẫn file GGUF chính và mmproj cho VLM Qwen3.8-27B.
    Ưu tiên:
    1. Biến môi trường: VLM_GGUF_PATH, VLM_MMPROJ_PATH, VLM_MODEL_DIR, QWEN38_VLM_PATH.
    2. Quét /kaggle/input/ tìm thư mục dataset chứa qwen38 hoặc gguf.
    3. Tự động tải từ Hugging Face qua huggingface_hub nếu chưa có sẵn.
    """
    cfg = VLM_CONFIG
    target_id = model_id or VLM_MODEL_ID
    expected_model_file = cfg.get("model_file", "Qwen3.8-27B-UD-Q4_K_M.gguf")
    expected_mmproj_file = cfg.get("mmproj_file", "mmproj-F16.gguf")
    fallback_id = VLM_FALLBACK_ID

    # 1. Kiểm tra biến môi trường
    env_gguf = os.environ.get("VLM_GGUF_PATH") or os.environ.get("QWEN38_VLM_PATH")
    env_mmproj = os.environ.get("VLM_MMPROJ_PATH") or os.environ.get("QWEN38_MMPROJ_PATH")
    if env_gguf and os.path.exists(env_gguf):
        logger.info(f"📂 [Env Config] Tìm thấy VLM GGUF qua biến môi trường: {env_gguf}")
        return {
            "model_file": env_gguf,
            "mmproj_file": env_mmproj if env_mmproj and os.path.exists(env_mmproj) else None,
            "is_gguf": True,
            "hf_repo_id": target_id,
            "fallback_id": fallback_id,
        }

    # 2. Quét thông minh trong /kaggle/input/
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        match_keywords = ["qwen38-27b-vlm-gguf", "qwen38", "qwen3.8", "qwen-27b", "qwen27b"]
        for item in kaggle_input.iterdir():
            if not item.is_dir():
                continue
            item_lower = item.name.lower()
            if any(kw in item_lower for kw in match_keywords):
                gguf_files = list(item.glob("*.gguf")) + list(item.glob("**/*.gguf"))
                main_model = None
                mmproj_model = None
                for f in gguf_files:
                    f_name = f.name.lower()
                    if "mmproj" in f_name:
                        mmproj_model = str(f.resolve())
                    elif main_model is None or "q4_k_m" in f_name or "27b" in f_name:
                        main_model = str(f.resolve())

                if main_model and os.path.exists(main_model):
                    logger.info(f"⚡ [Kaggle Dataset Direct] Tìm thấy VLM GGUF: {main_model} | mmproj: {mmproj_model}")
                    return {
                        "model_file": main_model,
                        "mmproj_file": mmproj_model,
                        "is_gguf": True,
                        "hf_repo_id": target_id,
                        "fallback_id": fallback_id,
                    }

    # 3. Kiểm tra nếu target_id là đường dẫn file hoặc thư mục cục bộ
    if os.path.exists(target_id):
        target_path = Path(target_id)
        if target_path.is_file() and target_path.name.endswith(".gguf"):
            mmproj_candidate = target_path.parent / expected_mmproj_file
            return {
                "model_file": str(target_path.resolve()),
                "mmproj_file": str(mmproj_candidate.resolve()) if mmproj_candidate.exists() else None,
                "is_gguf": True,
                "hf_repo_id": target_id,
                "fallback_id": fallback_id,
            }
        elif target_path.is_dir():
            main_f = target_path / expected_model_file
            mm_f = target_path / expected_mmproj_file
            if main_f.exists():
                return {
                    "model_file": str(main_f.resolve()),
                    "mmproj_file": str(mm_f.resolve()) if mm_f.exists() else None,
                    "is_gguf": True,
                    "hf_repo_id": target_id,
                    "fallback_id": fallback_id,
                }

    # 4. Canonical / Fallback về Hugging Face repo
    is_gguf_repo = any(k in target_id.lower() for k in ["gguf", "qwen38"])
    return {
        "model_file": None,
        "mmproj_file": None,
        "is_gguf": is_gguf_repo,
        "hf_repo_id": target_id,
        "expected_model_file": expected_model_file,
        "expected_mmproj_file": expected_mmproj_file,
        "fallback_id": fallback_id,
    }

# STT
STT_CONFIG = MODELS_CONFIG.get("stt", {})
STT_MODEL_ID = os.environ.get("STT_MODEL_ID", STT_CONFIG.get("id", "openai/whisper-large-v3-turbo"))
STT_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# TTS
TTS_CONFIG = MODELS_CONFIG.get("tts", {})
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", TTS_CONFIG.get("id", "hexgrad/Kokoro-82M"))
TTS_VOICE = os.environ.get("TTS_VOICE", TTS_CONFIG.get("voice", "af_heart"))
TTS_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# Image (FLUX.2 Klein SOTA GGUF + Qwen3-4B)
FLUX2_CONFIG = MODELS_CONFIG.get("image", {})
FLUX2_MODEL_ID = os.environ.get("FLUX2_MODEL_ID", FLUX2_CONFIG.get("id", "unsloth/FLUX.2-klein-4B-GGUF"))
FLUX2_NUM_STEPS = int(FLUX2_CONFIG.get("steps", 4))
FLUX2_GUIDANCE = float(FLUX2_CONFIG.get("guidance", 1.0))
FLUX2_VARIANTS = FLUX2_CONFIG.get("variants", {
    "klein_4b": "unsloth/FLUX.2-klein-4B-GGUF",
    "4b": "unsloth/FLUX.2-klein-4B-GGUF",
    "klein_9b": "unsloth/FLUX.2-klein-9B-GGUF",
    "9b": "unsloth/FLUX.2-klein-9B-GGUF",
    "dev": "black-forest-labs/FLUX.2-dev",
})

# Backward compatibility alias
FLUX_CONFIG = FLUX2_CONFIG
FLUX_MODEL_ID = FLUX2_MODEL_ID
FLUX_NUM_STEPS = FLUX2_NUM_STEPS
FLUX_GUIDANCE = FLUX2_GUIDANCE
FLUX_VARIANTS = FLUX2_VARIANTS

def resolve_image_model_id(model_name: Optional[str] = None) -> str:
    """Ánh xạ tên model request sang Hugging Face model ID cho FLUX.2."""
    if not model_name:
        return FLUX2_MODEL_ID
    clean = model_name.lower().strip()
    for k, v in FLUX2_VARIANTS.items():
        if k in clean:
            return v
    if "dev" in clean:
        return FLUX2_VARIANTS.get("dev", "black-forest-labs/FLUX.2-dev")
    elif "9b" in clean:
        return FLUX2_VARIANTS.get("klein_9b", "unsloth/FLUX.2-klein-9B-GGUF")
    elif "4b" in clean or "klein" in clean or "flux" in clean:
        return FLUX2_VARIANTS.get("klein_4b", "unsloth/FLUX.2-klein-4B-GGUF")
    return model_name if "/" in model_name else FLUX2_MODEL_ID

def resolve_flux2_paths(model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Tự động quét và xác định đường dẫn file GGUF của FLUX.2 Klein và Qwen3-4B Text Encoder.
    Ưu tiên:
    1. Biến môi trường: FLUX2_GGUF_PATH, FLUX2_TEXT_ENCODER_PATH, FLUX2_VAE_PATH.
    2. Quét /kaggle/input/ tìm dataset flux2-klein-4b-gguf và qwen3-4b-gguf.
    3. Tải từ Hugging Face qua huggingface_hub nếu chưa có sẵn.
    """
    cfg = FLUX2_CONFIG
    target_id = model_id or FLUX2_MODEL_ID
    expected_model_file = cfg.get("model_file", "flux-2-klein-4b-Q4_K_M.gguf")
    expected_te_file = cfg.get("text_encoder_file", "Qwen3-4B-Q4_K_M.gguf")
    te_repo = cfg.get("text_encoder_repo", "unsloth/Qwen3-4B-GGUF")
    base_repo = "black-forest-labs/FLUX.2-klein-4B"

    env_flux_gguf = os.environ.get("FLUX2_GGUF_PATH")
    env_te_gguf = os.environ.get("FLUX2_TEXT_ENCODER_PATH") or os.environ.get("QWEN3_GGUF_PATH")
    env_vae = os.environ.get("FLUX2_VAE_PATH")

    flux_gguf_path = env_flux_gguf if env_flux_gguf and os.path.exists(env_flux_gguf) else None
    te_gguf_path = env_te_gguf if env_te_gguf and os.path.exists(env_te_gguf) else None
    vae_path = env_vae if env_vae and os.path.exists(env_vae) else None
    config_dir = None

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for item in kaggle_input.iterdir():
            if not item.is_dir():
                continue
            item_lower = item.name.lower()
            if "flux2" in item_lower or "flux-2" in item_lower or "klein" in item_lower:
                gguf_files = list(item.glob("*.gguf")) + list(item.glob("**/*.gguf"))
                for f in gguf_files:
                    if "q4_k_m" in f.name.lower() or "4b" in f.name.lower():
                        flux_gguf_path = str(f.resolve())
                        break
                safetensors = list(item.glob("*.safetensors")) + list(item.glob("**/*.safetensors"))
                for s in safetensors:
                    if "vae" in s.name.lower():
                        vae_path = str(s.resolve())
                if (item / "model_index.json").exists():
                    config_dir = str(item.resolve())

            if "qwen3" in item_lower or "qwen-3" in item_lower:
                te_files = list(item.glob("*.gguf")) + list(item.glob("**/*.gguf"))
                for f in te_files:
                    if "q4_k_m" in f.name.lower() or "4b" in f.name.lower():
                        te_gguf_path = str(f.resolve())
                        break

    return {
        "transformer_gguf": flux_gguf_path,
        "text_encoder_gguf": te_gguf_path,
        "vae_path": vae_path,
        "config_dir": config_dir or base_repo,
        "hf_repo": target_id,
        "text_encoder_repo": te_repo,
        "base_repo": base_repo,
        "expected_model_file": expected_model_file,
        "expected_te_file": expected_te_file,
    }

# Video
VIDEO_CONFIG = MODELS_CONFIG.get("video", {})
VIDEO_MODEL_ID = os.environ.get("VIDEO_MODEL_ID", VIDEO_CONFIG.get("id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
VIDEO_FALLBACK_ID = os.environ.get("VIDEO_FALLBACK_ID", VIDEO_CONFIG.get("fallback_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
VIDEO_LOAD_IN_4BIT = str(VIDEO_CONFIG.get("quantization", "4bit")).lower() == "4bit"
VIDEO_VARIANTS = VIDEO_CONFIG.get("variants", {
    "ti2v_5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "t2v_a14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "a14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "i2v_a14b": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "ti2v_5b_22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "t2v_a14b_22": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "i2v_a14b_22": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "t2v_1_3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "t2v_14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "i2v_14b": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
})

def resolve_video_model_id(model_name: Optional[str] = None, is_i2v: bool = False) -> str:
    """Ánh xạ tên video model request sang Hugging Face model ID cho T2V hoặc I2V."""
    if not model_name:
        if is_i2v:
            return VIDEO_VARIANTS.get("ti2v_5b", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        return VIDEO_MODEL_ID
    clean = model_name.lower().strip()
    # Check exact match
    if clean in VIDEO_VARIANTS:
        return VIDEO_VARIANTS[clean]
    # Check Wan2.1 explicitly
    if "2.1" in clean:
        if is_i2v or "i2v" in clean:
            return VIDEO_VARIANTS.get("i2v_14b", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
        elif "14b" in clean:
            return VIDEO_VARIANTS.get("t2v_14b", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
        return VIDEO_VARIANTS.get("t2v_1_3b", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    # Check Wan2.2 / 5B / A14B
    if "5b" in clean or "ti2v" in clean:
        return VIDEO_VARIANTS.get("ti2v_5b", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    elif "a14b" in clean or ("14b" in clean and "2.2" in clean):
        if is_i2v or "i2v" in clean:
            return VIDEO_VARIANTS.get("i2v_a14b", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        return VIDEO_VARIANTS.get("t2v_a14b", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
