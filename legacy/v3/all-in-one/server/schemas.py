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
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32768)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=200)
    repetition_penalty: Optional[float] = Field(default=1.05, ge=1.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    system_prompt: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = None
    stream: Optional[bool] = False
    # OpenAI Tool Calling & Structured Outputs
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None


# --- Image Generation & Editing Schemas ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = "qwen_image_21"
    n: Optional[int] = 1
    size: Optional[str] = "1280x720"
    width: Optional[int] = 1280
    height: Optional[int] = 720
    response_format: Optional[str] = "b64_json"
    steps: Optional[int] = 20
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = 1.0
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    strength: Optional[float] = None
    stream: Optional[bool] = False
    image: Optional[str] = None       # Dùng khi gọi inpainting qua generation endpoint
    mask: Optional[str] = None
    mask_image: Optional[str] = None  # Alias cho inpainting mask
    reference_image: Optional[str] = None  # Dùng cho Multimodal Instruction Edit
    instruction_image: Optional[str] = None
    sampler_name: Optional[str] = "euler"
    scheduler: Optional[str] = "simple"


class ImageEditRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    image: str                         # Base64 data URI hoặc image URL
    mask: Optional[str] = None         # Chuẩn OpenAI param 'mask'
    mask_image: Optional[str] = None   # Alias thuận tiện 'mask_image'
    reference_image: Optional[str] = None
    model: Optional[str] = "qwen-image-2.1"
    n: Optional[int] = 1
    size: Optional[str] = "1280x720"
    width: Optional[int] = 1280
    height: Optional[int] = 720
    response_format: Optional[str] = "b64_json"
    steps: Optional[int] = 20
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = 1.0
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    strength: Optional[float] = 0.8
    stream: Optional[bool] = False
    sampler_name: Optional[str] = "euler"
    scheduler: Optional[str] = "simple"


# --- TTS Schemas ---
class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = "qwen3-tts-1.7b"  # 'qwen3-tts-1.7b', 'xtts-v2-vietnamese', 'kokoro-82m'
    voice: Optional[str] = "default"
    language: Optional[str] = "vi"           # 'vi', 'en', 'zh', etc.
    response_format: Optional[str] = "wav"
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    sample_rate: Optional[int] = 24000
    speaker_wav: Optional[str] = None        # Dùng cho XTTS-v2 / Qwen3-TTS Voice Cloning (Base64 hoặc URL/path)
    voice_description: Optional[str] = None  # Dùng cho Qwen3-TTS VoiceDesign (mô tả ngữ âm/chất giọng bằng text)
    prompt_audio: Optional[str] = None       # Dùng cho Qwen3-TTS Reference prompt audio
    ref_text: Optional[str] = None           # Văn bản của file giọng mẫu (Full ICL Voice Cloning trên Qwen3-TTS)
    x_vector_only_mode: Optional[bool] = None  # Chế độ chỉ dùng x-vector (True) hay kèm ref_text (False)
    engine: Optional[str] = None             # 'qwen_tts', 'xtts', 'kokoro'
    # Hyperparameters nâng cao cho Qwen3-TTS & XTTS-v2
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=500)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.1, le=20.0)
    length_penalty: Optional[float] = Field(default=None, ge=0.1, le=10.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    gpt_cond_len: Optional[int] = Field(default=None, ge=1, le=120)
    max_ref_len: Optional[int] = Field(default=None, ge=1, le=120)
    sound_norm_refs: Optional[bool] = None
    enable_text_splitting: Optional[bool] = None


# --- STT Schemas (Metadata & Options) ---
class TranscriptionOptions(BaseModel):
    model: Optional[str] = "qwen3-asr-1.7b"  # 'qwen3-asr-1.7b' hoặc 'whisper-large-v3-turbo'
    language: Optional[str] = "vi"           # 'vi', 'en', 'zh', hoặc 'auto'
    task: Optional[str] = "transcribe"       # 'transcribe' hoặc 'translate'
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.0
    max_new_tokens: Optional[int] = None
    return_time_stamps: Optional[bool] = False
    response_format: Optional[str] = "json"


# --- Runtime Config Update Schema ---
class RuntimeConfigUpdateRequest(BaseModel):
    stt: Optional[Dict[str, Any]] = None
    tts: Optional[Dict[str, Any]] = None
    vlm: Optional[Dict[str, Any]] = None
    image: Optional[Dict[str, Any]] = None
    video: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Dict[str, Any]]] = None
    extra: Optional[Dict[str, Any]] = None


# --- Video Generation & ITV Schemas ---
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = "Comfy-Org/MiniMax-H3"
    image: Optional[str] = None                  # Nếu có: kích hoạt Image-to-Video (ITV / FL2V)
    first_frame: Optional[str] = None            # Alias cho start image
    last_frame: Optional[str] = None             # Frame kết thúc cho FL2V / FL2VA
    reference_image: Optional[str] = None        # Reference image đơn cho Ref2V / Ref2VA
    reference_images: Optional[List[str]] = None # Danh sách đa ảnh tham chiếu (<Picture 1>, <Picture 2>, ...) cho Ref2VA
    guide_frames: Optional[List[str]] = None     # Chuỗi đa khung hình liên tiếp từ video trước để neo vector vận tốc (5 frames)
    ref_image_size: Optional[str] = "match"      # 'match' hoặc 'max' cho Ref2VA
    mode: Optional[str] = "auto"                 # 'auto', 'fl2va', 'ref2va'
    turbo: Optional[bool] = None                 # Bật/tắt MiniMax H3 Turbo LoRA (4-8 steps)
    use_turbo: Optional[bool] = None             # Alias cho turbo
    lora_name: Optional[str] = None              # Tên file LoRA tùy chọn
    lora_strength: Optional[float] = 1.0         # Cường độ LoRA
    num_frames: Optional[int] = 33
    width: Optional[int] = 768
    height: Optional[int] = 448
    fps: Optional[int] = 24
    steps: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance: Optional[float] = 1.0
    guidance_scale: Optional[float] = None
    shift: Optional[float] = 8.0
    sampler_name: Optional[str] = "euler"
    scheduler: Optional[str] = "simple"
    enable_audio: Optional[bool] = False         # Mặc định False (no audio)
    audio: Optional[bool] = False                # Alias
    high_steps: Optional[int] = None
    low_steps: Optional[int] = None
    seed: Optional[int] = None
    stream: Optional[bool] = False
    gpu_device: Optional[int] = None             # 0 hoặc 1 (Hỗ trợ Dual-GPU Parallel Testing)
    device: Optional[str] = None                 # 'cuda:0' hoặc 'cuda:1'


# --- Embedding & Rerank Schemas (OpenAI & Cohere Compatible) ---
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "Qwen/Qwen3-Embedding-0.6B"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class RerankRequest(BaseModel):
    query: str
    documents: List[Union[str, Dict[str, Any]]]
    model: Optional[str] = "Qwen/Qwen3-Reranker-0.6B"
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True
    max_chunks_per_doc: Optional[int] = None


class MultimodalEmbeddingRequest(BaseModel):
    input: List[Dict[str, Any]]   # Danh sách text hoặc base64 image dicts
    model: Optional[str] = "Qwen/Qwen3-VL-Embedding-2B"
    dimensions: Optional[int] = None
