"""
🌐 FastAPI Server: OpenAI-Compatible Multi-Service Gateway
Cung cấp toàn bộ các endpoint chuẩn của OpenAI cho:
- Chat / VLM: POST /v1/chat/completions
- Audio STT: POST /v1/audio/transcriptions
- Audio TTS: POST /v1/audio/speech
- Image Gen: POST /v1/images/generations
- Video Gen: POST /v1/videos/generations
- Memory Monitor: GET /v1/memory
"""

import base64
import time
import uuid
from typing import Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import API_KEY, STT_MODEL_ID, TTS_MODEL_ID, VLM_MODEL_ID, FLUX_MODEL_ID, VIDEO_MODEL_ID
from core.memory_manager import get_memory_manager
from audio.stt import get_stt_engine
from audio.tts import get_tts_engine
from vlm.qwen import get_vlm_engine
from visual.flux_image import get_flux_engine
from visual.wan_video import get_wan_engine
from server.schemas import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    SpeechRequest,
    VideoGenerationRequest,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="🎨 Kaggle All-in-One AI Studio API",
        version="3.0.0",
        description="Unified OpenAI-Compatible Gateway for VLM, STT, TTS, GenImage, and GenVideo.",
    )

    # Cấu hình CORS để frontend web hoặc ứng dụng bên ngoài gọi tự do
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_auth(authorization: Optional[str] = Header(None)):
        if API_KEY:
            expected = f"Bearer {API_KEY}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ==========================================================================
    # SYSTEM & HEALTH
    # ==========================================================================
    @app.get("/health")
    def health():
        mem = get_memory_manager()
        return {
            "status": "online",
            "service": "Kaggle All-in-One Studio",
            "vram_status": mem.report_vram(),
        }

    @app.get("/v1/models")
    def list_models(authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        return {
            "object": "list",
            "data": [
                {"id": "qwen-26b", "object": "model", "owned_by": "qwen", "type": "vlm"},
                {"id": "whisper-large-v3-turbo", "object": "model", "owned_by": "openai", "type": "stt"},
                {"id": "kokoro-82m", "object": "model", "owned_by": "hexgrad", "type": "tts"},
                {"id": "flux-1-schnell", "object": "model", "owned_by": "black-forest-labs", "type": "image"},
                {"id": "dall-e-3", "object": "model", "owned_by": "alias-flux", "type": "image"},
                {"id": "wan-2.1", "object": "model", "owned_by": "alibaba", "type": "video"},
            ],
        }

    @app.get("/v1/memory")
    def get_memory_info():
        mem = get_memory_manager()
        return mem.report_vram()

    # ==========================================================================
    # 1. 👁️ CHAT & VLM (QWEN 26B)
    # ==========================================================================
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        vlm = get_vlm_engine()

        res = vlm.chat_completion(
            messages=req.messages,
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )

        req_id = f"chatcmpl-{uuid.uuid4().hex}"
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "qwen-26b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": res["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": res["tokens"],
                "total_tokens": res["tokens"],
            },
            "x_inference_time_seconds": res["elapsed"],
        }

    # ==========================================================================
    # 2. 🎙️ SPEECH-TO-TEXT (WHISPER TURBO FULL FP16)
    # ==========================================================================
    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: Optional[str] = Form("whisper-large-v3-turbo"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        authorization: Optional[str] = Header(None),
    ):
        verify_auth(authorization)
        stt = get_stt_engine()
        audio_content = await file.read()
        res = stt.transcribe(audio_content, language=language, prompt=prompt)
        return {
            "text": res["text"],
            "language": res.get("language"),
            "x_duration_seconds": res.get("duration_seconds"),
        }

    # ==========================================================================
    # 3. 🔊 TEXT-TO-SPEECH (KOKORO-82M FULL FP16)
    # ==========================================================================
    @app.post("/v1/audio/speech")
    def audio_speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        tts = get_tts_engine()
        audio_bytes = tts.synthesize(
            text=req.input,
            voice=req.voice,
            response_format=req.response_format or "wav",
        )
        media_type = "audio/wav" if (req.response_format or "wav").lower() == "wav" else "audio/ogg"
        return Response(content=audio_bytes, media_type=media_type)

    # ==========================================================================
    # 4. 🖼️ IMAGE GENERATION (FLUX.1-SCHNELL NF4)
    # ==========================================================================
    @app.post("/v1/images/generations")
    def images_generations(req: ImageGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        flux = get_flux_engine()

        b64_str, elapsed = flux.generate(
            prompt=req.prompt,
            size=req.size or "1024x1024",
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
        )

        fmt = req.response_format or "b64_json"
        if fmt == "url":
            data_item = {"url": f"data:image/png;base64,{b64_str}", "revised_prompt": req.prompt}
        else:
            data_item = {"b64_json": b64_str, "revised_prompt": req.prompt}

        return {
            "created": int(time.time()),
            "data": [data_item],
            "x_inference_time_seconds": elapsed,
        }

    # ==========================================================================
    # 5. 🎬 VIDEO GENERATION (WAN2.1-1.3B)
    # ==========================================================================
    @app.post("/v1/videos/generations")
    def videos_generations(req: VideoGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        wan = get_wan_engine()

        video_bytes, elapsed = wan.generate(
            prompt=req.prompt,
            num_frames=req.num_frames or 25,
            width=req.width or 768,
            height=req.height or 512,
            seed=req.seed,
        )

        b64_video = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "created": int(time.time()),
            "data": [
                {
                    "b64_json": b64_video,
                    "revised_prompt": req.prompt,
                    "mime_type": "video/mp4",
                }
            ],
            "x_inference_time_seconds": elapsed,
        }

    return app
