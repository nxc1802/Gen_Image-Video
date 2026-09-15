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

from config import API_KEY
from core.memory_manager import get_memory_manager
from core.model_registry import get_model_registry
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

    registry = get_model_registry()

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
            "catalog": registry.get_catalog(),
            "vram_status": mem.report_vram(),
        }

    @app.get("/v1/models")
    def list_models(authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        catalog = registry.get_catalog()
        model_items = []
        for task, info in catalog.items():
            model_items.append({
                "id": info.get("model_id"),
                "object": "model",
                "owned_by": "studio",
                "type": task,
                "device_strategy": info.get("device_strategy"),
                "quantization": info.get("quantization"),
            })
        return {"object": "list", "data": model_items}

    @app.get("/v1/memory")
    def get_memory_info():
        mem = get_memory_manager()
        return mem.report_vram()

    # ==========================================================================
    # 1. 👁️ CHAT & VLM (Qwen / Llama-Vision...)
    # ==========================================================================
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        vlm = registry.get_vlm()

        res = vlm.chat(
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
            "model": req.model or vlm.model_id,
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
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        authorization: Optional[str] = Header(None),
    ):
        verify_auth(authorization)
        stt = registry.get_stt()
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
        tts = registry.get_tts()
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
        flux = registry.get_image()

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
    # 5. 🎬 VIDEO GENERATION (WAN2.1-14B / 1.3B)
    # ==========================================================================
    @app.post("/v1/videos/generations")
    def videos_generations(req: VideoGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        wan = registry.get_video()

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
