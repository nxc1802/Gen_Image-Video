"""
📋 Pydantic Request & Response Schemas
Tuân thủ 100% định dạng chuẩn của OpenAI REST API, hỗ trợ đầy đủ mọi tham số nâng cao.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# --- Chat & VLM Schemas ---
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "auto"
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=200)
    repetition_penalty: Optional[float] = Field(default=1.05, ge=1.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    system_prompt: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


# --- Image Generation & Editing Schemas ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = "schnell"
    n: Optional[int] = 1
    size: Optional[str] = "512x512"
    width: Optional[int] = None
    height: Optional[int] = None
    response_format: Optional[str] = "b64_json"
    steps: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    strength: Optional[float] = None
    stream: Optional[bool] = False
    image: Optional[str] = None       # Dùng khi gọi inpainting qua generation endpoint
    mask: Optional[str] = None
    mask_image: Optional[str] = None  # Mặt nạ inpainting


class ImageEditRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    image: str                         # Base64 data URI hoặc image URL
    mask: Optional[str] = None         # Chuẩn OpenAI param 'mask'
    mask_image: Optional[str] = None   # Alias thuận tiện 'mask_image'
    model: Optional[str] = "schnell"
    n: Optional[int] = 1
    size: Optional[str] = "512x512"
    width: Optional[int] = None
    height: Optional[int] = None
    response_format: Optional[str] = "b64_json"
    steps: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    strength: Optional[float] = 0.8
    stream: Optional[bool] = False


# --- TTS Schemas ---
class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = "kokoro-82m"
    voice: Optional[str] = "af_heart"
    language: Optional[str] = None     # 'en' hoặc 'vi'
    response_format: Optional[str] = "wav"
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    sample_rate: Optional[int] = 24000


# --- STT Schemas (Metadata & Options) ---
class TranscriptionOptions(BaseModel):
    model: Optional[str] = "whisper-large-v3-turbo"
    language: Optional[str] = None     # 'en', 'vi', hoặc 'auto'
    task: Optional[str] = "transcribe" # 'transcribe' hoặc 'translate'
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.0
    response_format: Optional[str] = "json"


# --- Video Generation & ITV Schemas ---
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = "1.3b"
    image: Optional[str] = None        # Nếu có: kích hoạt Image-to-Video (ITV)
    num_frames: Optional[int] = 17
    width: Optional[int] = 512
    height: Optional[int] = 512
    fps: Optional[int] = 16
    steps: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    stream: Optional[bool] = False
