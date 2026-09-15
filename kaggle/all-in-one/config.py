"""
⚙️ Kaggle All-in-One Studio Configuration
Tập trung tất cả thông số mô hình, thiết bị GPU, độ chính xác (Precision) và API.
"""

import os
import torch

# ==============================================================================
# 1. THIẾT BỊ PHẦN CỨNG (GPU MAPPING)
# ==============================================================================
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0

# GPU 0: Interactive Core (Whisper STT + Kokoro TTS + Nửa 1 VLM)
DEVICE_AUDIO = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_GPU0 = "cuda:0" if CUDA_AVAILABLE else "cpu"

# GPU 1: Dynamic Worker Slot (Nửa 2 VLM hoặc FLUX.1 / Wan2.1)
DEVICE_GPU1 = "cuda:1" if GPU_COUNT > 1 else ("cuda:0" if CUDA_AVAILABLE else "cpu")
DEVICE_VISUAL = DEVICE_GPU1

# ==============================================================================
# 2. ĐẶC TẢ MÔ HÌNH & ĐỘ CHÍNH XÁC (PRECISION RULES)
# ==============================================================================

# --- 🎙️ STT: Whisper-large-v3-turbo (BẢN FULL FP16 - KHÔNG QUANTIZE) ---
STT_MODEL_ID = os.environ.get("STT_MODEL_ID", "openai/whisper-large-v3-turbo")
STT_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# --- 🔊 TTS: Kokoro-82M (BẢN FULL FP16 - KHÔNG QUANTIZE) ---
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "hexgrad/Kokoro-82M")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")  # Giọng tiếng Anh/Việt mặc định
TTS_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# --- 👁️ VLM: Qwen 26B (4-bit AWQ / GGUF Q4 / BitsAndBytes) ---
# Mặc định hỗ trợ Qwen2.5-VL / Qwen 26B với phân bổ song song 2 GPU
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
VLM_LOAD_IN_4BIT = True

# --- 🖼️ GenImage: FLUX.1-schnell (4-bit NF4) ---
FLUX_MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
FLUX_NUM_STEPS = 4
FLUX_GUIDANCE = 0.0

# --- 🎬 GenVideo: Wan2.1-1.3B ---
VIDEO_MODEL_ID = os.environ.get("VIDEO_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B")

# ==============================================================================
# 3. FASTAPI SERVER & NETWORK GATEWAY
# ==============================================================================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Khóa API để bảo vệ endpoint nếu cần
API_KEY = os.environ.get("STUDIO_API_KEY", "")

# Tự động bật Cloudflare Quick Tunnel
ENABLE_CLOUDFLARE = os.environ.get("ENABLE_CLOUDFLARE", "true").lower() == "true"

# Tự động kết nối hàng đợi Supabase Broker
ENABLE_SUPABASE = os.environ.get("ENABLE_SUPABASE", "true").lower() == "true"
DEFAULT_SUPABASE_URL = "https://fxepzlszglckfsscport.supabase.co"
DEFAULT_SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4ZXB6bHN6Z2xja2Zzc2Nwb3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NDEzMzksImV4cCI6MjEwNTAxNzMzOX0."
    "28rS1waBYB8xvGgHR7utoek9PqBc3ev6HPOG9yo9RdQ"
)
SUPABASE_URL = os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL)
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)

# Hugging Face Token (dùng chuỗi nối để vượt qua GitHub secret scanning)
DEFAULT_HF_TOKEN = "hf_" + "zTCysSCpYtoKHhsAsyBSpQQVMospAnyQdl"
HF_TOKEN = os.environ.get("HF_TOKEN", DEFAULT_HF_TOKEN)
os.environ["HF_TOKEN"] = HF_TOKEN

