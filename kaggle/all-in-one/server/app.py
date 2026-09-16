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
import json
import time
import uuid
from typing import Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import API_KEY, FLUX_VARIANTS, VIDEO_VARIANTS
from core.memory_manager import get_memory_manager
from core.model_registry import get_model_registry
from server.schemas import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    ImageEditRequest,
    SpeechRequest,
    VideoGenerationRequest,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="🎨 Kaggle All-in-One AI Studio API",
        version="3.5.0",
        description="Unified OpenAI-Compatible Gateway for VLM, STT, TTS, GenImage, Inpainting, and GenVideo.",
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
        # Bổ sung các model aliases / variants
        model_items.extend([
            {"id": "flux-1-schnell", "object": "model", "owned_by": "black-forest-labs", "type": "image"},
            {"id": "flux-1-dev", "object": "model", "owned_by": "black-forest-labs", "type": "image"},
            {"id": "wan-2.1-1.3b", "object": "model", "owned_by": "wan-ai", "type": "video"},
            {"id": "wan-2.1-14b", "object": "model", "owned_by": "wan-ai", "type": "video"},
            {"id": "wan-2.1-i2v", "object": "model", "owned_by": "wan-ai", "type": "video_i2v"},
            {"id": "whisper-large-v3-turbo", "object": "model", "owned_by": "openai", "type": "stt"},
            {"id": "kokoro-82m", "object": "model", "owned_by": "hexgrad", "type": "tts"},
        ])
        return {"object": "list", "data": model_items}

    @app.get("/v1/memory")
    def get_memory_info():
        mem = get_memory_manager()
        return mem.report_vram()

    # ==========================================================================
    # 1. 👁️ CHAT & VLM (Qwen / Llama-Vision...)
    # ==========================================================================
    # ==========================================================================
    # 1. 👁️ CHAT & VLM (Qwen / Llama-Vision...)
    # ==========================================================================
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        vlm = registry.get_vlm()

        if req.stream:
            def sse_generator():
                yield ": keepalive\n\n"
                created_ts = int(time.time())
                req_id = f"chatcmpl-{uuid.uuid4().hex}"
                for token in vlm.chat_stream(
                    messages=req.messages,
                    max_tokens=req.max_tokens or 512,
                    temperature=req.temperature or 0.7,
                    top_p=req.top_p or 0.9,
                ):
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model or vlm.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model or vlm.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

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
    # 4. 🖼️ IMAGE GENERATION & INPAINTING (FLUX.1 SCHNELL / DEV)
    # ==========================================================================
    @app.post("/v1/images/generations")
    def images_generations(req: ImageGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        flux = registry.get_image()

        # Nếu có image + mask_image: kích hoạt Inpainting
        if req.image and req.mask_image:
            if req.stream:
                def sse_inpaint_stream():
                    yield ": keepalive\n\n"
                    yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': req.steps or 4, 'progress': 0, 'status': 'preparing'})}\n\n"
                    try:
                        for ev in flux.inpaint_stream(
                            prompt=req.prompt,
                            image=req.image,
                            mask_image=req.mask_image,
                            size=req.size or "1024x1024",
                            steps=req.steps,
                            guidance=req.guidance,
                            seed=req.seed,
                            model_variant=req.model,
                        ):
                            if ev.get("type") == "heartbeat":
                                yield ": keepalive\n\n"
                            elif ev["type"] == "progress":
                                yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
                            elif ev["type"] == "complete":
                                res_payload = {
                                    "created": int(time.time()),
                                    "data": [{"b64_json": ev["b64_json"], "revised_prompt": req.prompt}],
                                    "x_inference_time_seconds": ev["elapsed"],
                                }
                                yield f"event: complete\ndata: {json.dumps(res_payload)}\n\n"
                                yield "data: [DONE]\n\n"
                            elif ev["type"] == "error":
                                yield f"event: error\ndata: {json.dumps(ev)}\n\n"
                                yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(sse_inpaint_stream(), media_type="text/event-stream")

            b64_str, elapsed = flux.inpaint(
                prompt=req.prompt,
                image=req.image,
                mask_image=req.mask_image,
                size=req.size or "1024x1024",
                steps=req.steps,
                guidance=req.guidance,
                seed=req.seed,
                model_variant=req.model,
            )
            return {
                "created": int(time.time()),
                "data": [{"b64_json": b64_str, "revised_prompt": req.prompt}],
                "x_inference_time_seconds": elapsed,
            }

        # Text-to-Image tiêu chuẩn
        if req.stream:
            def sse_image_stream():
                yield ": keepalive\n\n"
                yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': req.steps or 4, 'progress': 0, 'status': 'preparing'})}\n\n"
                try:
                    for ev in flux.generate_stream(
                        prompt=req.prompt,
                        size=req.size or "1024x1024",
                        steps=req.steps,
                        guidance=req.guidance,
                        seed=req.seed,
                        model_variant=req.model,
                    ):
                        if ev.get("type") == "heartbeat":
                            yield ": keepalive\n\n"
                        elif ev["type"] == "progress":
                            yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
                        elif ev["type"] == "complete":
                            res_payload = {
                                "created": int(time.time()),
                                "data": [{"b64_json": ev["b64_json"], "revised_prompt": req.prompt}],
                                "x_inference_time_seconds": ev["elapsed"],
                            }
                            yield f"event: complete\ndata: {json.dumps(res_payload)}\n\n"
                            yield "data: [DONE]\n\n"
                        elif ev["type"] == "error":
                            yield f"event: error\ndata: {json.dumps(ev)}\n\n"
                            yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_image_stream(), media_type="text/event-stream")

        b64_str, elapsed = flux.generate(
            prompt=req.prompt,
            size=req.size or "1024x1024",
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            model_variant=req.model,
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

    @app.post("/v1/images/edits")
    def images_edits(req: ImageEditRequest, authorization: Optional[str] = Header(None)):
        """Chỉnh sửa ảnh / Inpainting chuẩn OpenAI."""
        verify_auth(authorization)
        flux = registry.get_image()
        mask_val = req.mask_image or req.mask

        if req.stream:
            def sse_edit_stream():
                yield ": keepalive\n\n"
                yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': req.steps or 4, 'progress': 0, 'status': 'preparing'})}\n\n"
                try:
                    for ev in flux.inpaint_stream(
                        prompt=req.prompt,
                        image=req.image,
                        mask_image=mask_val,
                        size=req.size or "1024x1024",
                        steps=req.steps,
                        guidance=req.guidance,
                        seed=req.seed,
                        model_variant=req.model,
                    ):
                        if ev.get("type") == "heartbeat":
                            yield ": keepalive\n\n"
                        elif ev["type"] == "progress":
                            yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
                        elif ev["type"] == "complete":
                            res_payload = {
                                "created": int(time.time()),
                                "data": [{"b64_json": ev["b64_json"], "revised_prompt": req.prompt}],
                                "x_inference_time_seconds": ev["elapsed"],
                            }
                            yield f"event: complete\ndata: {json.dumps(res_payload)}\n\n"
                            yield "data: [DONE]\n\n"
                        elif ev["type"] == "error":
                            yield f"event: error\ndata: {json.dumps(ev)}\n\n"
                            yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_edit_stream(), media_type="text/event-stream")

        b64_str, elapsed = flux.inpaint(
            prompt=req.prompt,
            image=req.image,
            mask_image=mask_val,
            size=req.size or "1024x1024",
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            model_variant=req.model,
        )

        return {
            "created": int(time.time()),
            "data": [{"b64_json": b64_str, "revised_prompt": req.prompt}],
            "x_inference_time_seconds": elapsed,
        }

    # ==========================================================================
    # 5. 🎬 VIDEO GENERATION & ITV (WAN2.1 1.3B / 14B / I2V)
    # ==========================================================================
    @app.post("/v1/videos/generations")
    def videos_generations(req: VideoGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        wan = registry.get_video()

        # Image-to-Video (ITV)
        if req.image:
            if req.stream:
                def sse_i2v_stream():
                    yield ": keepalive\n\n"
                    yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': req.num_frames or 25, 'progress': 0, 'status': 'preparing'})}\n\n"
                    try:
                        for ev in wan.generate_i2v_stream(
                            prompt=req.prompt,
                            image=req.image,
                            num_frames=req.num_frames or 25,
                            width=req.width or 768,
                            height=req.height or 512,
                            seed=req.seed,
                            model_variant=req.model,
                        ):
                            if ev.get("type") == "heartbeat":
                                yield ": keepalive\n\n"
                            elif ev["type"] == "progress":
                                yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
                            elif ev["type"] == "complete":
                                b64_vid = base64.b64encode(ev["video_bytes"]).decode("utf-8")
                                res_payload = {
                                    "created": int(time.time()),
                                    "data": [{"b64_json": b64_vid, "mime_type": "video/mp4", "revised_prompt": req.prompt}],
                                    "x_inference_time_seconds": ev["elapsed"],
                                }
                                yield f"event: complete\ndata: {json.dumps(res_payload)}\n\n"
                                yield "data: [DONE]\n\n"
                            elif ev["type"] == "error":
                                yield f"event: error\ndata: {json.dumps(ev)}\n\n"
                                yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(sse_i2v_stream(), media_type="text/event-stream")

            video_bytes, elapsed = wan.generate_i2v(
                prompt=req.prompt,
                image=req.image,
                num_frames=req.num_frames or 25,
                width=req.width or 768,
                height=req.height or 512,
                seed=req.seed,
                model_variant=req.model,
            )
            b64_video = base64.b64encode(video_bytes).decode("utf-8")
            return {
                "created": int(time.time()),
                "data": [{"b64_json": b64_video, "revised_prompt": req.prompt, "mime_type": "video/mp4"}],
                "x_inference_time_seconds": elapsed,
            }

        # Text-to-Video (T2V)
        if req.stream:
            def sse_t2v_stream():
                yield ": keepalive\n\n"
                yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': req.num_frames or 25, 'progress': 0, 'status': 'preparing'})}\n\n"
                try:
                    for ev in wan.generate_stream(
                        prompt=req.prompt,
                        num_frames=req.num_frames or 25,
                        width=req.width or 768,
                        height=req.height or 512,
                        seed=req.seed,
                        model_variant=req.model,
                    ):
                        if ev.get("type") == "heartbeat":
                            yield ": keepalive\n\n"
                        elif ev["type"] == "progress":
                            yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
                        elif ev["type"] == "complete":
                            b64_vid = base64.b64encode(ev["video_bytes"]).decode("utf-8")
                            res_payload = {
                                "created": int(time.time()),
                                "data": [{"b64_json": b64_vid, "mime_type": "video/mp4", "revised_prompt": req.prompt}],
                                "x_inference_time_seconds": ev["elapsed"],
                            }
                            yield f"event: complete\ndata: {json.dumps(res_payload)}\n\n"
                            yield "data: [DONE]\n\n"
                        elif ev["type"] == "error":
                            yield f"event: error\ndata: {json.dumps(ev)}\n\n"
                            yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_t2v_stream(), media_type="text/event-stream")

        video_bytes, elapsed = wan.generate(
            prompt=req.prompt,
            num_frames=req.num_frames or 25,
            width=req.width or 768,
            height=req.height or 512,
            seed=req.seed,
            model_variant=req.model,
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

    @app.post("/v1/admin/shutdown")
    def admin_shutdown():
        """Dừng kernel server an toàn."""
        def kill_soon():
            time.sleep(0.5)
            import os
            os._exit(0)
        threading.Thread(target=kill_soon).start()
        return {"status": "shutting_down", "message": "Kernel process is terminating"}

    return app
