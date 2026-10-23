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

# Đường dẫn chuẩn cho Google Colab & Google Drive Cache (hỗ trợ cả Shared Drive Shortcut giữa nhiều tài khoản)
def _resolve_colab_drive_root() -> str:
    raw_root = os.environ.get("COLAB_DRIVE_ROOT", "/content/drive/MyDrive/AI_Studio_Colab")
    if os.path.exists(raw_root):
        return os.path.realpath(raw_root)
    import glob as _glob
    for cand in (
        ["/content/drive/MyDrive/AI_Studio_Colab", "/content/drive/Shareddrives/AI_Studio_Colab"]
        + _glob.glob("/content/drive/.shortcut-targets-by-id/*/AI_Studio_Colab")
    ):
        if os.path.exists(cand):
            return os.path.realpath(cand)
    return raw_root

COLAB_DRIVE_ROOT = _resolve_colab_drive_root()
COLAB_MODELS_DIR = os.path.realpath(os.environ.get("COLAB_MODELS_DIR", f"{COLAB_DRIVE_ROOT}/models"))
COLAB_WHEELS_DIR = os.path.realpath(os.environ.get("COLAB_WHEELS_DIR", f"{COLAB_DRIVE_ROOT}/wheels"))
IS_COLAB = os.path.exists("/content") and not os.path.exists("/kaggle/input")

# ==============================================================================
# 2. ĐỌC CẤU HÌNH KHAI BÁO TỪ models.yaml (HOẶC FALLBACK MẶC ĐỊNH)
# ==============================================================================
DEFAULT_MODELS_CONFIG: Dict[str, Any] = {
    "stt": {
        "id": "Qwen/Qwen3-ASR-1.7B",
        "fallback_id": "openai/whisper-large-v3-turbo",
        "precision": "fp16",
        "backend": "audio_service",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "preferred_device": "gpu_0",
        "device_strategy": "gpu_0",
    },
    "tts": {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "xtts_id": "thivux/XTTS-v2-vietnamse",
        "fallback_id": "hexgrad/Kokoro-82M",
        "voice": "default",
        "precision": "fp16",
        "backend": "audio_service",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "preferred_device": "gpu_0",
        "device_strategy": "gpu_0",
    },
    "vlm": {
        "id": "cuongnguyen1802/qwen38-27b-gguf",
        "fallback_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "format": "gguf",
        "model_file": "Qwen3.8-27B-Q4_K_M.gguf",
        "mmproj_file": "mmproj-Qwen3.8-27B-Q8_0.gguf",
        "quantization": "q4_k_m",
        "precision": "fp16",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "allocation_policy": "dual_gpu",
        "device_strategy": "dual_gpu",
        "gpu_count": 2,
        "gpu_split": [0.5, 0.5],
        "n_ctx": 131072,
        "n_gpu_layers": -1,
        "max_tokens": 1024,
        "temperature": 0.7,
    },
    "image": {
        "id": "abenzerps/Qwen-Image-2.1-GGUF",
        "default_variant": "qwen_image_21",
        "model_file": "qwen-image-2.1-Q4_K_M.gguf",
        "vae_file": "qwen_image_2.1_vae_bf16.safetensors",
        "text_encoder_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "text_encoder_file": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
        "variants": {
            "qwen_image_21": "abenzerps/Qwen-Image-2.1-GGUF",
            "2.1": "abenzerps/Qwen-Image-2.1-GGUF",
            "qwen": "abenzerps/Qwen-Image-2.1-GGUF",
        },
        "quantization": "q4_k_m",
        "precision": "fp16",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "gpu",
        "allocation_policy": "single",
        "device_strategy": "single",
        "device": os.environ.get("IMAGE_DEVICE", "cuda:0"),
        "gpu_device": int(os.environ.get("IMAGE_GPU_DEVICE", "0")),
        "preferred_device": "gpu_0",
        "steps": 20,
        "guidance": 1.0,
    },
    "video": {
        "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "default_variant": "ti2v_5b",
        "fallback_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "backend": "comfyui",
        "device": os.environ.get("VIDEO_DEVICE", "cuda:1" if GPU_COUNT > 1 else "cuda:0"),
        "gpu_device": int(os.environ.get("VIDEO_GPU_DEVICE", "1" if GPU_COUNT > 1 else "0")),
        "preferred_device": "gpu_1" if GPU_COUNT > 1 else "gpu_0",
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
        "device_strategy": "single",
        "num_frames": 17,
        "width": 832,
        "height": 480,
        "steps": 20,
        "guidance": 5.0,
    },
    "minimax": {
        "id": "Comfy-Org/MiniMax-H3",
        "default_variant": "minimax_h3",
        "backend": "comfyui",
        "device": os.environ.get("MINIMAX_DEVICE", "cuda:1" if GPU_COUNT > 1 else "cuda:0"),
        "gpu_device": int(os.environ.get("MINIMAX_GPU_DEVICE", "1" if GPU_COUNT > 1 else "0")),
        "preferred_device": "gpu_1" if GPU_COUNT > 1 else "gpu_0",
        "variants": {
            "minimax_h3": "Comfy-Org/MiniMax-H3",
            "h3": "Comfy-Org/MiniMax-H3",
            "fl2va": "Comfy-Org/MiniMax-H3",
        },
        "quantization": "q4_k_m",
        "precision": "fp16",
        "lifecycle": "dynamic_switch",
        "preload": False,
        "init_target": "cpu",
        "allocation_policy": "adaptive",
        "device_strategy": "single",
        "num_frames": 17,
        "width": 416,
        "height": 240,
        "steps": 4,
        "guidance": 1.0,
        "shift": 8.0,
        "fps": 24,
    },
    "embedding": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "fallback_id": "Qwen/Qwen3-Embedding-4B",
        "format": "gguf",
        "model_file": "Qwen3-Embedding-0.6B.Q4_K_M.gguf",
        "quantization": "q4_k_m",
        "backend": "llama_cpp",
        "lifecycle": "always_active",
        "preload": True,
        "init_target": "gpu",
        "device": "cuda:0",
        "device_strategy": "single",
        "dimensions": 1024,
    },
    "rerank": {
        "id": "Qwen/Qwen3-Reranker-0.6B",
        "fallback_id": "Qwen/Qwen3-Reranker-4B",
        "format": "gguf",
        "model_file": "Qwen3-Reranker-0.6B.Q4_K_M.gguf",
        "quantization": "q4_k_m",
        "backend": "llama_cpp",
        "lifecycle": "always_active",
        "preload": True,
        "init_target": "gpu",
        "device": "cuda:0",
        "device_strategy": "single",
    },
}


_raw_mode = os.environ.get("STUDIO_MODE", "speech").strip().lower()
if _raw_mode in ("visual",):
    _raw_mode = "image_video"
if _raw_mode not in ("speech", "vlm", "image_video", "embedding"):
    logger.warning(f"⚠️ Mode '{_raw_mode}' không hợp lệ hoặc đã bị loại bỏ (luồng 'all' không còn tồn tại). Mặc định dùng 'speech'.")
    _raw_mode = "speech"
STUDIO_MODE = _raw_mode
os.environ["STUDIO_MODE"] = STUDIO_MODE


def load_yaml_config() -> Dict[str, Any]:
    """Đọc cấu hình theo STUDIO_MODE (speech | vlm | image_video | embedding)."""
    mode_map = {
        "speech": Path(__file__).parent / "configs" / "models_speech.yaml",
        "vlm": Path(__file__).parent / "configs" / "models_vlm.yaml",
        "image_video": Path(__file__).parent / "configs" / "models_image_video.yaml",
        "embedding": Path(__file__).parent / "configs" / "models_embedding.yaml",
    }
    config_path = mode_map.get(STUDIO_MODE, Path(__file__).parent / "configs" / "models_speech.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent / "models.yaml"

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "models" in data:
                result = {}
                for k, v in data["models"].items():
                    base = dict(DEFAULT_MODELS_CONFIG.get(k, {}))
                    if isinstance(v, dict):
                        base.update(v)
                    result[k] = base
                return result
    except Exception as e:
        logger.debug(f"PyYAML không khả dụng hoặc lỗi đọc file: {e}. Dùng parser dự phòng.")

    # Fallback chỉ giữ đúng các section thuộc STUDIO_MODE
    mode_sections = {
        "speech": {"stt", "tts"},
        "vlm": {"vlm"},
        "image_video": {"image", "video", "minimax"},
        "embedding": {"embedding", "rerank"},
    }
    allowed = mode_sections.get(STUDIO_MODE, {"stt", "tts"})
    return {k: v for k, v in DEFAULT_MODELS_CONFIG.items() if k in allowed}


MODELS_CONFIG = load_yaml_config()


def resolve_model_gpu_device(model_key: str) -> tuple[str, int]:
    """
    Xác định thiết bị GPU (device string & GPU index) cho model được chỉ định.
    Ưu tiên:
    1. Biến môi trường: <MODEL_KEY>_DEVICE hoặc <MODEL_KEY>_GPU_DEVICE (ví dụ IMAGE_DEVICE, MINIMAX_DEVICE).
    2. Cấu hình trong models.yaml / MODELS_CONFIG.
    3. Mặc định: image -> GPU 0, minimax / video -> GPU 1 (nếu có >= 2 GPU, ngược lại fallback GPU 0).
    """
    k = model_key.lower().strip()
    env_dev = os.environ.get(f"{k.upper()}_DEVICE")
    env_gpu_idx = os.environ.get(f"{k.upper()}_GPU_DEVICE")

    cfg = MODELS_CONFIG.get(k, {})
    dev_str = env_dev or cfg.get("device")

    if env_gpu_idx is not None:
        try:
            gpu_idx = int(env_gpu_idx)
            dev_str = f"cuda:{gpu_idx}"
        except ValueError:
            gpu_idx = 0
    elif dev_str and "cuda:" in dev_str:
        try:
            gpu_idx = int(dev_str.split(":")[-1])
        except ValueError:
            gpu_idx = 0
    else:
        # Default mapping: image -> 0, minimax/video -> 1
        if k in ("image", "qwen_image", "stt", "tts"):
            gpu_idx = 0
        else:
            gpu_idx = 1 if GPU_COUNT > 1 else 0
        dev_str = f"cuda:{gpu_idx}"

    if CUDA_AVAILABLE and GPU_COUNT > 0:
        gpu_idx = min(gpu_idx, GPU_COUNT - 1)
        dev_str = f"cuda:{gpu_idx}"
    else:
        dev_str = "cpu"
        gpu_idx = 0

    return dev_str, gpu_idx


# ==============================================================================
# 3. GHI ĐÈ BIẾN MÔI TRƯỜNG & TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY)
# ==============================================================================
# VLM
VLM_CONFIG = MODELS_CONFIG.get("vlm", {})
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", VLM_CONFIG.get("id", "nguynxuncngde180528/qwen38-27b-vlm-gguf"))
VLM_FALLBACK_ID = os.environ.get("VLM_FALLBACK_ID", VLM_CONFIG.get("fallback_id", "Qwen/Qwen2.5-VL-3B-Instruct"))
VLM_FORMAT = str(VLM_CONFIG.get("format", "gguf")).lower()
VLM_LOAD_IN_4BIT = str(VLM_CONFIG.get("quantization", "4bit")).lower() in ("4bit", "q4_k_m")
VLM_N_CTX = int(os.environ.get("VLM_N_CTX", VLM_CONFIG.get("n_ctx", 131072)))
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
    expected_model_file = cfg.get("model_file", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
    expected_mmproj_file = cfg.get("mmproj_file", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")
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

    # 2. Quét đệ quy toàn diện trong Google Drive Cache (Colab) và /kaggle/input/ (Kaggle)
    hf_canonical_repo = "ggml-org/Qwen3.8-27B-GGUF" if any(k in target_id.lower() for k in ["cuongnguyen1802", "nguynxuncngde180528", "qwen38"]) else target_id
    search_roots = [
        Path(COLAB_MODELS_DIR),
        Path("/content/models"),
        Path("/kaggle/input"),
        Path("/tmp/hf_cache"),
        Path("/kaggle/working"),
    ]
    for root_dir in search_roots:
        if not root_dir.exists():
            continue
        all_ggufs = list(root_dir.rglob("*.gguf"))
        if not all_ggufs:
            continue

        # Bước 2a: Tìm chính xác expected_model_file
        for f in all_ggufs:
            if f.name.lower() == expected_model_file.lower():
                cand_mm = f.parent / expected_mmproj_file
                if not cand_mm.exists():
                    mm_cands = [m for m in f.parent.glob("*.gguf") if "mmproj" in m.name.lower()]
                    cand_mm = mm_cands[0] if mm_cands else cand_mm
                logger.info(f"⚡ [Dataset Exact rglob] Tìm thấy VLM GGUF: {f} | mmproj: {cand_mm if cand_mm.exists() else None}")
                return {
                    "model_file": str(f.resolve()),
                    "mmproj_file": str(cand_mm.resolve()) if cand_mm.exists() else None,
                    "is_gguf": True,
                    "hf_repo_id": hf_canonical_repo,
                    "fallback_id": fallback_id,
                }

        # Bước 2b: Quét mờ theo từ khóa nếu chưa tìm thấy file chính xác
        match_keywords = ["qwen3.8", "qwen38", "27b", "qwen3vl"]
        main_model = None
        mmproj_model = None
        for f in all_ggufs:
            p_str = str(f).lower()
            f_name = f.name.lower()
            if "image" in p_str or "wan" in p_str or "flux" in p_str:
                continue
            if any(kw in p_str for kw in match_keywords):
                if "mmproj" in f_name:
                    mmproj_model = str(f.resolve())
                elif "mtp" not in f_name and "dflash" not in f_name:
                    if main_model is None or "q4_k_m" in f_name or "27b" in f_name:
                        main_model = str(f.resolve())

        if main_model and os.path.exists(main_model):
            logger.info(f"⚡ [Dataset Direct rglob] Tìm thấy VLM GGUF: {main_model} | mmproj: {mmproj_model}")
            return {
                "model_file": main_model,
                "mmproj_file": mmproj_model,
                "is_gguf": True,
                "hf_repo_id": hf_canonical_repo,
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
                "hf_repo_id": hf_canonical_repo,
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
                    "hf_repo_id": hf_canonical_repo,
                    "fallback_id": fallback_id,
                }

    # 4. Canonical / Fallback về Hugging Face repo
    is_gguf_repo = any(k in target_id.lower() for k in ["gguf", "qwen38"])
    return {
        "model_file": None,
        "mmproj_file": None,
        "is_gguf": is_gguf_repo,
        "hf_repo_id": hf_canonical_repo,
        "expected_model_file": expected_model_file,
        "expected_mmproj_file": expected_mmproj_file,
        "fallback_id": fallback_id,
    }

# STT
STT_CONFIG = MODELS_CONFIG.get("stt", {})
STT_MODEL_ID = os.environ.get("STT_MODEL_ID", STT_CONFIG.get("id", "Qwen/Qwen3-ASR-1.7B"))
STT_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# TTS
TTS_CONFIG = MODELS_CONFIG.get("tts", {})
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", TTS_CONFIG.get("id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
XTTS_MODEL_ID = os.environ.get("XTTS_MODEL_ID", TTS_CONFIG.get("xtts_id", "thivux/XTTS-v2-vietnamse"))
TTS_VOICE = os.environ.get("TTS_VOICE", TTS_CONFIG.get("voice", "default"))
TTS_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# Image (Qwen-Image-2.1 GGUF + Qwen3-VL 8B Dual-GPU)
QWEN_IMAGE_CONFIG = MODELS_CONFIG.get("image", {})
QWEN_IMAGE_MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL_ID", QWEN_IMAGE_CONFIG.get("id", "abenzerps/Qwen-Image-2.1-GGUF"))
QWEN_IMAGE_NUM_STEPS = int(QWEN_IMAGE_CONFIG.get("steps", 20))
QWEN_IMAGE_GUIDANCE = float(QWEN_IMAGE_CONFIG.get("guidance", 5.0))
QWEN_IMAGE_VARIANTS = QWEN_IMAGE_CONFIG.get("variants", {
    "qwen_image_21": "abenzerps/Qwen-Image-2.1-GGUF",
    "2.1": "abenzerps/Qwen-Image-2.1-GGUF",
    "qwen": "abenzerps/Qwen-Image-2.1-GGUF",
})

# Backward compatibility aliases
FLUX2_CONFIG = QWEN_IMAGE_CONFIG
FLUX2_MODEL_ID = QWEN_IMAGE_MODEL_ID
FLUX2_NUM_STEPS = QWEN_IMAGE_NUM_STEPS
FLUX2_GUIDANCE = QWEN_IMAGE_GUIDANCE
FLUX2_VARIANTS = QWEN_IMAGE_VARIANTS
FLUX_CONFIG = QWEN_IMAGE_CONFIG
FLUX_MODEL_ID = QWEN_IMAGE_MODEL_ID
FLUX_NUM_STEPS = QWEN_IMAGE_NUM_STEPS
FLUX_GUIDANCE = QWEN_IMAGE_GUIDANCE
FLUX_VARIANTS = QWEN_IMAGE_VARIANTS

def resolve_image_model_id(model_name: Optional[str] = None) -> str:
    """Ánh xạ tên model request sang Hugging Face model ID cho Qwen-Image."""
    if not model_name:
        return QWEN_IMAGE_MODEL_ID
    clean = model_name.lower().strip()
    for k, v in QWEN_IMAGE_VARIANTS.items():
        if k in clean:
            return v
    return model_name if "/" in model_name else QWEN_IMAGE_MODEL_ID

def resolve_qwen_image_paths(model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Tự động quét và xác định đường dẫn file GGUF của Qwen-Image-2.1 và Qwen3-VL 8B.
    Ưu tiên:
    1. Biến môi trường: QWEN_IMAGE_GGUF_PATH, QWEN3_VL_GGUF_PATH, QWEN_IMAGE_VAE_PATH, QWEN3_VL_MMPROJ_PATH.
    2. Quét Google Drive Cache (Colab) hoặc /kaggle/input/ tìm dataset qwen-image-2-1-gguf và qwen3vl-8b-instruct-gguf.
    """
    cfg = QWEN_IMAGE_CONFIG
    target_id = model_id or QWEN_IMAGE_MODEL_ID
    expected_model_file = cfg.get("model_file", "qwen-image-2.1-Q4_K_M.gguf")
    expected_vae_file = cfg.get("vae_file", "qwen_image_2.1_vae_bf16.safetensors")
    expected_llm_file = cfg.get("text_encoder_file", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
    expected_mmproj_file = cfg.get("mmproj_file", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")

    env_dit_gguf = os.environ.get("QWEN_IMAGE_GGUF_PATH") or os.environ.get("FLUX2_GGUF_PATH")
    env_llm_gguf = os.environ.get("QWEN3_VL_GGUF_PATH") or os.environ.get("FLUX2_TEXT_ENCODER_PATH")
    env_vae = os.environ.get("QWEN_IMAGE_VAE_PATH") or os.environ.get("FLUX2_VAE_PATH")
    env_mmproj = os.environ.get("QWEN3_VL_MMPROJ_PATH")

    dit_gguf_path = env_dit_gguf if env_dit_gguf and os.path.exists(env_dit_gguf) else None
    llm_gguf_path = env_llm_gguf if env_llm_gguf and os.path.exists(env_llm_gguf) else None
    vae_path = env_vae if env_vae and os.path.exists(env_vae) else None
    mmproj_path = env_mmproj if env_mmproj and os.path.exists(env_mmproj) else None
    config_dir = None

    for base_input in [Path(COLAB_MODELS_DIR), Path("/content/models"), Path("/kaggle/input")]:
        if not base_input.exists():
            continue
        for item in base_input.iterdir():
            if not item.is_dir():
                continue
            item_lower = item.name.lower()
            if "qwen-image" in item_lower or "qwen_image" in item_lower:
                gguf_files = list(item.glob("*.gguf")) + list(item.glob("**/*.gguf"))
                for f in gguf_files:
                    if not dit_gguf_path and ("q4_k_m" in f.name.lower() or "2.1" in f.name.lower() or "image" in f.name.lower()):
                        dit_gguf_path = str(f.resolve())
                        break
                safetensors = list(item.glob("*.safetensors")) + list(item.glob("**/*.safetensors"))
                for s in safetensors:
                    if not vae_path and "vae" in s.name.lower():
                        vae_path = str(s.resolve())
                        break
                if (item / "model_index.json").exists():
                    config_dir = str(item.resolve())

            if "qwen3vl" in item_lower or "qwen3-vl" in item_lower:
                gguf_files = list(item.glob("*.gguf")) + list(item.glob("**/*.gguf"))
                for f in gguf_files:
                    fname = f.name.lower()
                    if not mmproj_path and "mmproj" in fname:
                        mmproj_path = str(f.resolve())
                    elif not llm_gguf_path and ("q4_k_m" in fname or "8b" in fname):
                        llm_gguf_path = str(f.resolve())

    return {
        "transformer_gguf": dit_gguf_path,
        "llm_gguf": llm_gguf_path,
        "mmproj_gguf": mmproj_path,
        "vae_path": vae_path,
        "config_dir": config_dir or "Qwen/Qwen-Image-2.1",
        "hf_repo": target_id,
        "expected_model_file": expected_model_file,
        "expected_vae_file": expected_vae_file,
        "expected_llm_file": expected_llm_file,
        "expected_mmproj_file": expected_mmproj_file,
    }

# Backward compatibility alias
resolve_flux2_paths = resolve_qwen_image_paths

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
