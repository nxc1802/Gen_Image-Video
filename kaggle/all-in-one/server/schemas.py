"""
📋 Pydantic Request & Response Schemas
Tuân thủ 100% định dạng chuẩn của OpenAI REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Chat & VLM Schemas ---
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen-26b"
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    stream: Optional[bool] = False


# --- Image Generation & Editing Schemas ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux-1-schnell"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    stream: Optional[bool] = False
    image: Optional[str] = None       # Dùng khi gọi inpainting qua generation endpoint
    mask_image: Optional[str] = None  # Mặt nạ inpainting


class ImageEditRequest(BaseModel):
    prompt: str
    image: str                         # Base64 data URI hoặc image URL
    mask: Optional[str] = None         # Chuẩn OpenAI param 'mask'
    mask_image: Optional[str] = None   # Alias thuận tiện 'mask_image'
    model: Optional[str] = "flux-1-schnell"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    stream: Optional[bool] = False


# --- TTS Schemas ---
class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = "kokoro-82m"
    voice: Optional[str] = "af_heart"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


# --- Video Generation & ITV Schemas ---
class VideoGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "wan-2.1-1.3b"
    image: Optional[str] = None        # Nếu có: kích hoạt Image-to-Video (ITV)
    num_frames: Optional[int] = 25
    width: Optional[int] = 768
    height: Optional[int] = 512
    seed: Optional[int] = None
    stream: Optional[bool] = False
