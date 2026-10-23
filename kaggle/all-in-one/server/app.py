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
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

logger = logging.getLogger("StudioServer")

from config import API_KEY, FLUX_VARIANTS, VIDEO_VARIANTS, MODELS_CONFIG, PORT
from core.memory_manager import get_memory_manager
from core.model_registry import get_model_registry
from server.schemas import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    ImageEditRequest,
    MultimodalEmbeddingRequest,
    RerankRequest,
    RuntimeConfigUpdateRequest,
    SpeechRequest,
    VideoGenerationRequest,
)


def create_app() -> FastAPI:
    curr_studio_mode = os.environ.get("STUDIO_MODE", "speech").strip().lower()
    app = FastAPI(
        title=f"🚀 Kaggle {curr_studio_mode.title()} AI Studio API",
        version="3.5.0",
        description=f"Dedicated OpenAI-Compatible Gateway for {curr_studio_mode.upper()} Mode.",
    )

    # Cấu hình CORS để frontend web hoặc ứng dụng bên ngoài gọi tự do
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Unhandled error on {request.url.path}: {tb}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "detail": str(exc), "path": str(request.url.path)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    registry = get_model_registry()
    last_activity = [time.time()]
    idle_minutes = int(os.environ.get("IDLE_SHUTDOWN_MINUTES", "120") or 120)

    @app.middleware("http")
    async def track_activity_middleware(request, call_next):
        if request.url.path.startswith("/v1/"):
            last_activity[0] = time.time()
        return await call_next(request)

    if idle_minutes > 0:
        def _idle_watchdog():
            while True:
                time.sleep(30)
                idle_sec = time.time() - last_activity[0]
                if idle_sec >= idle_minutes * 60:
                    logger.warning(f"⏰ Không có request tới /v1/* trong {idle_minutes} phút. Tự động Shutdown (exit 99) để bảo toàn GPU Kaggle!")
                    os._exit(99)
        threading.Thread(target=_idle_watchdog, daemon=True).start()

    def verify_auth(authorization: Optional[str] = Header(None)):
        if API_KEY:
            expected = f"Bearer {API_KEY}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def require_studio_mode(*allowed_modes: str):
        curr_mode = os.environ.get("STUDIO_MODE", "speech").strip().lower()
        if curr_mode in ("visual",):
            curr_mode = "image_video"
        if curr_mode not in allowed_modes:
            modes_str = " | ".join(allowed_modes)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Endpoint not supported in current STUDIO_MODE='{curr_mode}'. This kernel is dedicated to mode: '{curr_mode}'. Required mode for this endpoint is: {modes_str}.",
                    "current_mode": curr_mode,
                    "required_modes": list(allowed_modes),
                },
            )

    # ==========================================================================
    # PLAYGROUND UI & SYSTEM HEALTH
    # ==========================================================================
    @app.get("/", response_class=HTMLResponse)
    @app.get("/ui", response_class=HTMLResponse)
    def serve_playground_ui():
        ui_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ui", "index.html"),
        ]
        for p in ui_candidates:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read(), status_code=200)
        return HTMLResponse(content="<h1>Kaggle AI Studio Gateway is Online</h1><p>Visit <a href='/docs'>/docs</a> or <a href='/health'>/health</a>.</p>", status_code=200)

    @app.get("/health")
    def health():
        mem = get_memory_manager()
        curr_mode = os.environ.get("STUDIO_MODE", "speech").strip().lower()
        if curr_mode in ("visual",):
            curr_mode = "image_video"
        resp = {
            "status": "online",
            "service": f"Kaggle {curr_mode.upper()} Studio",
            "studio_mode": curr_mode,
            "idle_shutdown_minutes": idle_minutes,
            "seconds_since_last_request": round(time.time() - last_activity[0], 1),
            "catalog": registry.get_catalog(),
            "vram_status": mem.report_vram(),
        }
        if curr_mode == "vlm" and "vlm" in MODELS_CONFIG:
            try:
                vlm = registry.get_vlm()
                if hasattr(vlm, "get_diagnostics"):
                    resp["vlm_diagnostics"] = vlm.get_diagnostics()
            except Exception as e:
                resp["vlm_diagnostics"] = {"error": str(e)}
        elif curr_mode == "image_video":
            try:
                from core.comfyui_manager import get_all_comfyui_diagnostics
                resp["comfyui_diagnostics"] = get_all_comfyui_diagnostics()
            except Exception as e:
                resp["comfyui_diagnostics"] = {"error": str(e)}
        elif curr_mode == "speech":
            try:
                tts = registry.get_tts()
                if hasattr(tts, "audio_manager"):
                    resp["audio_diagnostics"] = tts.audio_manager.get_diagnostics(try_load=False)
            except Exception as e:
                resp["audio_diagnostics"] = {"error": str(e)}
        elif curr_mode == "embedding":
            try:
                from embedding.embedding_service import get_embedding_service_manager
                resp["embedding_catalog"] = get_embedding_service_manager().get_catalog()
            except Exception as e:
                resp["embedding_catalog"] = {"error": str(e)}
        return resp

    @app.get("/v1/models")
    def list_models(authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        catalog = registry.get_catalog()
        model_items = []
        for task, info in catalog.items():
            mid = info.get("model_id") or task
            model_items.append({
                "id": mid,
                "object": "model",
                "owned_by": "studio",
                "type": task,
                "device_strategy": info.get("device_strategy"),
                "quantization": info.get("quantization"),
            })

        curr_mode = os.environ.get("STUDIO_MODE", "speech").strip().lower()
        if curr_mode in ("visual",):
            curr_mode = "image_video"

        # Bổ sung các aliases chuẩn THEO ĐÚNG STUDIO_MODE
        if curr_mode == "speech":
            model_items.extend([
                {"id": "qwen3-asr-1.7b", "object": "model", "owned_by": "qwen", "type": "stt"},
                {"id": "qwen3-tts-1.7b", "object": "model", "owned_by": "qwen", "type": "tts"},
                {"id": "xtts-v2-vietnamese", "object": "model", "owned_by": "coqui-community", "type": "tts"},
                {"id": "whisper-large-v3-turbo", "object": "model", "owned_by": "openai", "type": "stt"},
                {"id": "kokoro-82m", "object": "model", "owned_by": "hexgrad", "type": "tts"},
            ])
        elif curr_mode == "vlm":
            model_items.extend([
                {"id": "qwen3.8-27b-vlm", "object": "model", "owned_by": "qwen", "type": "vlm"},
                {"id": "qwen2.5-vl-3b-instruct", "object": "model", "owned_by": "qwen", "type": "vlm"},
            ])
        elif curr_mode == "image_video":
            model_items.extend([
                {"id": "qwen-image-2.1", "object": "model", "owned_by": "qwen", "type": "image"},
                {"id": "wan2.2-ti2v-5b", "object": "model", "owned_by": "wan-ai", "type": "video"},
                {"id": "wan2.2-t2v-14b", "object": "model", "owned_by": "wan-ai", "type": "video"},
                {"id": "wan2.2-i2v-14b", "object": "model", "owned_by": "wan-ai", "type": "video_i2v"},
                {"id": "minimax-h3", "object": "model", "owned_by": "minimax", "type": "video_audio"},
                {"id": "Comfy-Org/MiniMax-H3", "object": "model", "owned_by": "comfy-org", "type": "video_audio"},
            ])
        elif curr_mode == "embedding":
            model_items.extend([
                {"id": "qwen3-embedding-0.6b", "object": "model", "owned_by": "qwen", "type": "embedding"},
                {"id": "qwen3-embedding-4b", "object": "model", "owned_by": "qwen", "type": "embedding"},
                {"id": "qwen3-embedding-8b", "object": "model", "owned_by": "qwen", "type": "embedding"},
                {"id": "qwen3-vl-embedding-2b", "object": "model", "owned_by": "qwen", "type": "multimodal_embedding"},
                {"id": "qwen3-vl-embedding-8b", "object": "model", "owned_by": "qwen", "type": "multimodal_embedding"},
                {"id": "qwen3-reranker-0.6b", "object": "model", "owned_by": "qwen", "type": "reranker"},
                {"id": "qwen3-reranker-4b", "object": "model", "owned_by": "qwen", "type": "reranker"},
                {"id": "qwen3-reranker-8b", "object": "model", "owned_by": "qwen", "type": "reranker"},
                {"id": "qwen3-vl-reranker-2b", "object": "model", "owned_by": "qwen", "type": "multimodal_reranker"},
                {"id": "qwen3-vl-reranker-8b", "object": "model", "owned_by": "qwen", "type": "multimodal_reranker"},
            ])

        seen = set()
        deduped = []
        for it in model_items:
            mid = it.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                deduped.append(it)
        return {"object": "list", "data": deduped}

    @app.get("/v1/memory")
    def get_memory_info():
        mem = get_memory_manager()
        return mem.report_vram()

    @app.post("/v1/memory/free")
    def free_memory_info(authorization: Optional[str] = Header(None)):
        """Giải phóng chủ động toàn bộ VRAM (VLM, ComfyUI, Audio) mà không tắt Server."""
        verify_auth(authorization)
        mem = get_memory_manager()
        mem.clean_all_vram()
        return {
            "status": "freed",
            "vram_status": mem.report_vram(),
        }

    # ==========================================================================
    # 1. 👁️ CHAT & VLM (Qwen3.8-27B GGUF Dual-GPU)
    # ==========================================================================
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        require_studio_mode("vlm")
        vlm = registry.get_vlm()

        if req.stream:
            def sse_generator():
                yield ": keepalive\n\n"
                created_ts = int(time.time())
                req_id = f"chatcmpl-{uuid.uuid4().hex}"
                stream_meta = {}
                for item in vlm.chat_stream(
                    messages=req.messages,
                    max_tokens=req.max_tokens or 4096,
                    temperature=req.temperature if req.temperature is not None else 0.7,
                    top_p=req.top_p if req.top_p is not None else 0.9,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    system_prompt=req.system_prompt,
                    stop=req.stop,
                    enable_thinking=req.enable_thinking,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    response_format=req.response_format,
                ):
                    if isinstance(item, dict) and "_meta" in item:
                        stream_meta.update(item["_meta"] or {})
                        continue
                    delta_obj = item if isinstance(item, dict) else {"content": str(item)}
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model or vlm.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_obj,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                finish_reason = stream_meta.get("finish_reason") or "stop"
                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model or vlm.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                if stream_meta.get("usage"):
                    final_chunk["usage"] = stream_meta["usage"]
                if stream_meta.get("timings"):
                    final_chunk["timings"] = stream_meta["timings"]
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        res = vlm.chat(
            messages=req.messages,
            max_tokens=req.max_tokens or 4096,
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 0.9,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            system_prompt=req.system_prompt,
            stop=req.stop,
            enable_thinking=req.enable_thinking,
            tools=req.tools,
            tool_choice=req.tool_choice,
            response_format=req.response_format,
        )

        req_id = f"chatcmpl-{uuid.uuid4().hex}"
        msg_payload = {"role": "assistant", "content": res["text"]}
        if res.get("reasoning_content"):
            msg_payload["reasoning_content"] = res["reasoning_content"]
        if res.get("tool_calls"):
            msg_payload["tool_calls"] = res["tool_calls"]
        finish_reason = res.get("finish_reason") or ("tool_calls" if res.get("tool_calls") else "stop")
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or vlm.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": msg_payload,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": res.get("prompt_tokens", 0),
                "completion_tokens": res["tokens"],
                "total_tokens": res.get("prompt_tokens", 0) + res["tokens"],
            },
            "x_inference_time_seconds": res["elapsed"],
        }

    # ==========================================================================
    # 1.5 ⚙️ RUNTIME CONFIGURATION MANAGEMENT (GET / POST / PATCH /v1/config)
    # ==========================================================================
    @app.get("/v1/config")
    def get_runtime_config(authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        return {
            "status": "ok",
            "studio_mode": os.environ.get("STUDIO_MODE", "speech"),
            "models_config": MODELS_CONFIG,
        }

    @app.post("/v1/config")
    @app.patch("/v1/config")
    def update_runtime_config(req: RuntimeConfigUpdateRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        updated_keys = []
        payload = req.model_dump(exclude_none=True) if hasattr(req, "model_dump") else req.dict(exclude_none=True)
        for sec in ["stt", "tts", "vlm", "image", "video"]:
            if sec in payload and isinstance(payload[sec], dict):
                MODELS_CONFIG.setdefault(sec, {}).update(payload[sec])
                updated_keys.append(sec)
        if "models" in payload and isinstance(payload["models"], dict):
            for sec, cfg_dict in payload["models"].items():
                if isinstance(cfg_dict, dict):
                    MODELS_CONFIG.setdefault(sec, {}).update(cfg_dict)
                    if sec not in updated_keys:
                        updated_keys.append(sec)
        if "extra" in payload and isinstance(payload["extra"], dict):
            for sec, cfg_dict in payload["extra"].items():
                if isinstance(cfg_dict, dict):
                    MODELS_CONFIG.setdefault(sec, {}).update(cfg_dict)
                    if sec not in updated_keys:
                        updated_keys.append(sec)
        return {
            "status": "updated",
            "updated_sections": updated_keys,
            "models_config": MODELS_CONFIG,
        }

    # ==========================================================================
    # 2. 🎙️ SPEECH-TO-TEXT (QWEN3-ASR 1.7B MULTILINGUAL / VIETNAMESE)
    # ==========================================================================
    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        task: Optional[str] = Form("transcribe"),
        temperature: Optional[float] = Form(0.0),
        max_new_tokens: Optional[int] = Form(None),
        return_time_stamps: Optional[bool] = Form(None),
        response_format: Optional[str] = Form("json"),
        authorization: Optional[str] = Header(None),
    ):
        verify_auth(authorization)
        require_studio_mode("speech")
        stt = registry.get_stt()
        audio_content = await file.read()
        res = stt.transcribe(
            audio_content,
            language=language or MODELS_CONFIG.get("stt", {}).get("language", "vi"),
            prompt=prompt,
            task=task or "transcribe",
            temperature=temperature if temperature is not None else 0.0,
            model=model or "qwen3-asr-1.7b",
            max_new_tokens=max_new_tokens,
            return_time_stamps=return_time_stamps,
        )
        fmt = (response_format or "json").lower().strip()
        if fmt == "text":
            return Response(content=res["text"], media_type="text/plain; charset=utf-8")
        elif fmt == "verbose_json":
            out_resp = {
                "task": task or "transcribe",
                "language": res.get("language", language or "vi"),
                "duration": res.get("duration_seconds"),
                "text": res["text"],
                "model": res.get("model", model or "qwen3-asr-1.7b"),
            }
            if "time_stamps" in res:
                out_resp["segments"] = res["time_stamps"]
            return out_resp
        else:
            out_resp = {
                "text": res["text"],
                "language": res.get("language", language or "vi"),
                "model": res.get("model", model or "qwen3-asr-1.7b"),
                "task": task or "transcribe",
                "x_duration_seconds": res.get("duration_seconds"),
            }
            if "time_stamps" in res:
                out_resp["time_stamps"] = res["time_stamps"]
            return out_resp

    # ==========================================================================
    # 3. 🔊 TEXT-TO-SPEECH (QWEN3-TTS 1.7B & XTTS-V2 VIETNAMESE)
    # ==========================================================================
    @app.post("/v1/audio/speech")
    def audio_speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        require_studio_mode("speech")
        tts = registry.get_tts()
        audio_bytes = tts.synthesize(
            text=req.input,
            voice=req.voice,
            language=req.language or MODELS_CONFIG.get("tts", {}).get("language", "vi"),
            speed=req.speed or MODELS_CONFIG.get("tts", {}).get("speed", 1.0),
            sample_rate=req.sample_rate or MODELS_CONFIG.get("tts", {}).get("sample_rate", 24000),
            response_format=req.response_format or "wav",
            speaker_wav=req.speaker_wav,
            voice_description=req.voice_description,
            prompt_audio=req.prompt_audio,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            engine=req.engine,
            model=req.model,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            length_penalty=req.length_penalty,
            max_new_tokens=req.max_new_tokens,
            gpt_cond_len=req.gpt_cond_len,
            max_ref_len=req.max_ref_len,
            sound_norm_refs=req.sound_norm_refs,
            enable_text_splitting=req.enable_text_splitting,
        )
        fmt = (req.response_format or "wav").lower().strip()
        if fmt in ("mp3", "audio/mp3", "audio/mpeg"):
            media_type = "audio/mpeg"
        elif fmt in ("ogg", "opus", "audio/ogg", "audio/opus"):
            media_type = "audio/ogg"
        else:
            media_type = "audio/wav"
        return Response(content=audio_bytes, media_type=media_type)

    @app.get("/v1/audio/diagnostics")
    def audio_diagnostics(try_load: bool = False):
        require_studio_mode("speech")
        if "tts" not in MODELS_CONFIG and "stt" not in MODELS_CONFIG:
            return {
                "status": "disabled",
                "studio_mode": os.environ.get("STUDIO_MODE", "speech"),
                "models": {"asr": "standby", "tts": "standby", "xtts": "standby"},
            }
        try:
            tts = registry.get_tts()
            if hasattr(tts, "audio_manager"):
                return tts.audio_manager.get_diagnostics(try_load=try_load)
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}
        return {"error": "audio_manager not available"}

    def _json_keepalive_response(work_fn):
        """Giữ kết nối Cloudflare Tunnel sống (chống lỗi HTTP 524 sau 100s) bằng whitespace keepalive."""
        res_box = {}
        err_box = {}

        def _target():
            try:
                res_box["data"] = work_fn()
            except Exception as e:
                import traceback
                err_box["error"] = str(e)
                err_box["traceback"] = traceback.format_exc()

        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout=20.0)
        if not th.is_alive():
            if "error" in err_box:
                logger.error(f"Error in endpoint: {err_box['traceback']}")
                raise HTTPException(status_code=500, detail=err_box)
            return res_box["data"]

        def _gen():
            yield b"\n"
            while th.is_alive():
                th.join(timeout=15.0)
                if th.is_alive():
                    yield b"\n"
            if "error" in err_box:
                logger.error(f"Error in keepalive endpoint: {err_box['traceback']}")
                yield json.dumps({"error": err_box["error"], "detail": err_box}).encode("utf-8")
            else:
                yield json.dumps(res_box["data"]).encode("utf-8")

        return StreamingResponse(_gen(), media_type="application/json")

    # ==========================================================================
    # 🧬 EMBEDDING & RERANKING APIS (QWEN3-EMBEDDING & QWEN3-RERANKER)
    # ==========================================================================
    @app.post("/v1/embeddings")
    def create_embeddings(req: EmbeddingRequest, authorization: Optional[str] = Header(None)):
        """Tạo vector embeddings chuẩn OpenAI API."""
        verify_auth(authorization)
        require_studio_mode("embedding")
        try:
            from embedding.embedding_service import get_embedding_service_manager
            mgr = get_embedding_service_manager()
            return mgr.embed(inputs=req.input, model=req.model, dimensions=req.dimensions)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Lỗi tạo Embeddings: {tb}")
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

    @app.post("/v1/rerank")
    @app.post("/rerank")
    def create_rerank(req: RerankRequest, authorization: Optional[str] = Header(None)):
        """Sắp xếp lại văn bản chuẩn Cohere/Jina Rerank API."""
        verify_auth(authorization)
        require_studio_mode("embedding")
        try:
            from embedding.embedding_service import get_embedding_service_manager
            mgr = get_embedding_service_manager()
            return mgr.rerank(
                query=req.query,
                documents=req.documents,
                model=req.model,
                top_n=req.top_n,
                return_documents=req.return_documents,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Lỗi Reranking: {tb}")
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

    @app.post("/v1/multimodal/embeddings")
    def create_multimodal_embeddings(req: MultimodalEmbeddingRequest, authorization: Optional[str] = Header(None)):
        """Tạo vector embeddings đa phương thức (văn bản và hình ảnh)."""
        verify_auth(authorization)
        require_studio_mode("embedding")
        try:
            from embedding.embedding_service import get_embedding_service_manager
            mgr = get_embedding_service_manager()
            # Trích xuất content từ input
            inputs = []
            for item in req.input:
                if isinstance(item, dict):
                    inputs.append(item.get("text") or str(item))
                else:
                    inputs.append(str(item))
            return mgr.embed(inputs=inputs, model=req.model, dimensions=req.dimensions)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Lỗi Multimodal Embeddings: {tb}")
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

    # ==========================================================================
    # 4. 🖼️ IMAGE GENERATION & INPAINTING (QWEN-IMAGE-2.1 GGUF + QWEN3-VL 8B)
    # ==========================================================================
    @app.post("/v1/images/generations")
    def images_generations(req: ImageGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        require_studio_mode("image_video")
        try:
            img_engine = registry.get_image()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error initializing image engine: {tb}")
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

        steps_val = req.steps or req.num_inference_steps
        guidance_val = req.guidance if req.guidance is not None else req.guidance_scale
        mask_val = getattr(req, "mask_image", None) or getattr(req, "mask", None)

        # Nếu có image + mask: kích hoạt Inpainting
        if req.image and mask_val:
            if req.stream:
                def sse_inpaint_stream():
                    yield ": keepalive\n\n"
                    yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': steps_val or 20, 'progress': 0, 'status': 'preparing'})}\n\n"
                    try:
                        for ev in img_engine.inpaint_stream(
                            prompt=req.prompt,
                            image=req.image,
                            mask_image=mask_val,
                            negative_prompt=req.negative_prompt,
                            size=req.size or "1024x1024",
                            width=req.width,
                            height=req.height,
                            steps=steps_val,
                            guidance=guidance_val,
                            seed=req.seed,
                            strength=req.strength,
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

            def _do_inpaint():
                b64_str, elapsed = img_engine.inpaint(
                    prompt=req.prompt,
                    image=req.image,
                    mask_image=mask_val,
                    negative_prompt=req.negative_prompt,
                    size=req.size or "1024x1024",
                    width=req.width,
                    height=req.height,
                    steps=steps_val,
                    guidance=guidance_val,
                    seed=req.seed,
                    strength=req.strength,
                    model_variant=req.model,
                )
                return {
                    "created": int(time.time()),
                    "data": [{"b64_json": b64_str, "revised_prompt": req.prompt}],
                    "x_inference_time_seconds": elapsed,
                }

            return _json_keepalive_response(_do_inpaint)

        # Text-to-Image tiêu chuẩn
        if req.stream:
            def sse_image_stream():
                yield ": keepalive\n\n"
                yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': steps_val or 20, 'progress': 0, 'status': 'preparing'})}\n\n"
                try:
                    for ev in img_engine.generate_stream(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        size=req.size or "1024x1024",
                        width=req.width,
                        height=req.height,
                        steps=steps_val,
                        guidance=guidance_val,
                        seed=req.seed,
                        strength=req.strength,
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

        def _do_t2i():
            b64_str, elapsed = img_engine.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                size=req.size or "1280x720",
                width=req.width,
                height=req.height,
                steps=steps_val,
                guidance=guidance_val,
                seed=req.seed,
                strength=req.strength,
                model_variant=req.model,
                sampler_name=req.sampler_name,
                scheduler=req.scheduler,
                reference_image=getattr(req, "reference_image", None) or getattr(req, "instruction_image", None),
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

        return _json_keepalive_response(_do_t2i)

    @app.post("/v1/images/edits")
    def images_edits(req: ImageEditRequest, authorization: Optional[str] = Header(None)):
        """Chỉnh sửa ảnh / Inpainting chuẩn OpenAI."""
        verify_auth(authorization)
        require_studio_mode("image_video")
        flux = registry.get_image()
        mask_val = req.mask_image or req.mask
        steps_val = req.steps or req.num_inference_steps
        guidance_val = req.guidance if req.guidance is not None else req.guidance_scale

        if req.stream:
            def sse_edit_stream():
                yield ": keepalive\n\n"
                yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': steps_val or 4, 'progress': 0, 'status': 'preparing'})}\n\n"
                try:
                    for ev in flux.inpaint_stream(
                        prompt=req.prompt,
                        image=req.image,
                        mask_image=mask_val,
                        negative_prompt=req.negative_prompt,
                        size=req.size or "1024x1024",
                        width=req.width,
                        height=req.height,
                        steps=steps_val,
                        guidance=guidance_val,
                        seed=req.seed,
                        strength=req.strength,
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

        def _do_edit():
            b64_str, elapsed = flux.inpaint(
                prompt=req.prompt,
                image=req.image,
                mask_image=mask_val,
                negative_prompt=req.negative_prompt,
                size=req.size or "1280x720",
                width=req.width,
                height=req.height,
                steps=steps_val,
                guidance=guidance_val,
                seed=req.seed,
                strength=req.strength,
                model_variant=req.model,
                sampler_name=req.sampler_name,
                scheduler=req.scheduler,
                reference_image=getattr(req, "reference_image", None),
            )
            return {
                "created": int(time.time()),
                "data": [{"b64_json": b64_str, "revised_prompt": req.prompt}],
                "x_inference_time_seconds": elapsed,
            }

        return _json_keepalive_response(_do_edit)

    # ==========================================================================
    # 5. 🎬 VIDEO GENERATION & ITV (WAN2.1 1.3B / 14B / I2V & MINIMAX H3)
    # ==========================================================================
    @app.post("/v1/videos/generations")
    def videos_generations(req: VideoGenerationRequest, authorization: Optional[str] = Header(None)):
        verify_auth(authorization)
        require_studio_mode("image_video")
        req_model_str = (req.model or "").lower()
        target_gpu = req.gpu_device
        if target_gpu is None and req.device:
            if "cuda:" in str(req.device):
                try:
                    target_gpu = int(str(req.device).split(":")[-1])
                except Exception:
                    pass
        if "minimax" in req_model_str or "h3" in req_model_str:
            vid_engine = registry.get_minimax(gpu_device=target_gpu)
        else:
            vid_engine = registry.get_video()

        steps_val = req.steps or req.num_inference_steps
        guidance_val = req.guidance if req.guidance is not None else req.guidance_scale
        start_img = req.image or getattr(req, "first_frame", None)
        has_audio = getattr(req, "enable_audio", False) or getattr(req, "audio", False)
        ref_imgs = getattr(req, "reference_images", None)
        guide_frames_val = getattr(req, "guide_frames", None)
        extra_h3_opts = {
            "last_frame": getattr(req, "last_frame", None),
            "reference_image": getattr(req, "reference_image", None),
            "reference_images": ref_imgs,
            "guide_frames": guide_frames_val,
            "ref_image_size": getattr(req, "ref_image_size", "match"),
            "mode": getattr(req, "mode", "auto"),
            "turbo": getattr(req, "turbo", None),
            "use_turbo": getattr(req, "use_turbo", None),
            "lora_name": getattr(req, "lora_name", None),
            "lora_strength": getattr(req, "lora_strength", 1.0),
            "enable_audio": has_audio,
        }

        # Image-to-Video (ITV / FL2V / Ref2V / Multi-Reference Ref2VA / Multi-Frame Guide)
        if start_img or getattr(req, "last_frame", None) or getattr(req, "reference_image", None) or ref_imgs or guide_frames_val:
            if req.stream:
                def sse_i2v_stream():
                    yield ": keepalive\n\n"
                    yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': steps_val or 30, 'progress': 0, 'status': 'preparing'})}\n\n"
                    try:
                        for ev in vid_engine.generate_i2v_stream(
                            prompt=req.prompt,
                            image=start_img,
                            negative_prompt=req.negative_prompt,
                            num_frames=req.num_frames or vid_engine.num_frames,
                            width=req.width or vid_engine.width,
                            height=req.height or vid_engine.height,
                            fps=req.fps or 24,
                            steps=steps_val,
                            guidance=guidance_val,
                            seed=req.seed,
                            model_variant=req.model,
                            **extra_h3_opts,
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
                        import traceback
                        tb = traceback.format_exc()
                        logger.error(f"Error in sse_i2v_stream: {tb}")
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e), 'traceback': tb})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(sse_i2v_stream(), media_type="text/event-stream")

            def _do_i2v():
                video_bytes, elapsed = vid_engine.generate_i2v(
                    prompt=req.prompt,
                    image=start_img,
                    negative_prompt=req.negative_prompt,
                    num_frames=req.num_frames or vid_engine.num_frames,
                    width=req.width or vid_engine.width,
                    height=req.height or vid_engine.height,
                    fps=req.fps or 24,
                    steps=steps_val,
                    guidance=guidance_val,
                    shift=req.shift,
                    sampler_name=req.sampler_name,
                    scheduler=req.scheduler,
                    high_steps=req.high_steps,
                    low_steps=req.low_steps,
                    seed=req.seed,
                    model_variant=req.model,
                    **extra_h3_opts,
                )
                b64_video = base64.b64encode(video_bytes).decode("utf-8")
                return {
                    "created": int(time.time()),
                    "data": [{"b64_json": b64_video, "revised_prompt": req.prompt, "mime_type": "video/mp4"}],
                    "x_inference_time_seconds": elapsed,
                }

            return _json_keepalive_response(_do_i2v)

        # Text-to-Video (T2V / T2VA)
        if req.stream:
            def sse_t2v_stream():
                yield ": keepalive\n\n"
                yield f"event: progress\ndata: {json.dumps({'step': 0, 'total_steps': steps_val or 30, 'progress': 0, 'status': 'preparing'})}\n\n"
                try:
                    for ev in vid_engine.generate_stream(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        num_frames=req.num_frames or vid_engine.num_frames,
                        width=req.width or vid_engine.width,
                        height=req.height or vid_engine.height,
                        fps=req.fps or 24,
                        steps=steps_val,
                        guidance=guidance_val,
                        seed=req.seed,
                        model_variant=req.model,
                        **extra_h3_opts,
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
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"Error in sse_t2v_stream: {tb}")
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e), 'traceback': tb})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_t2v_stream(), media_type="text/event-stream")

        def _do_t2v():
            video_bytes, elapsed = vid_engine.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_frames=req.num_frames or vid_engine.num_frames,
                width=req.width or vid_engine.width,
                height=req.height or vid_engine.height,
                fps=req.fps or 24,
                steps=steps_val,
                guidance=guidance_val,
                shift=req.shift,
                sampler_name=req.sampler_name,
                scheduler=req.scheduler,
                high_steps=req.high_steps,
                low_steps=req.low_steps,
                seed=req.seed,
                model_variant=req.model,
                **extra_h3_opts,
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

        return _json_keepalive_response(_do_t2v)

    @app.post("/v1/admin/reconfigure-vlm")
    async def admin_reconfigure_vlm(request: Request, authorization: Optional[str] = Header(None)):
        """Restart llama-server with a different speculative decoding mode (baseline/mtp/dflash) and/or context length (n_ctx)."""
        verify_auth(authorization)
        require_studio_mode("vlm")

        try:
            body = await request.json()
        except Exception:
            body = {}

        spec_mode = body.get("spec_mode") or request.query_params.get("spec_mode")
        n_ctx = body.get("n_ctx") or request.query_params.get("n_ctx")
        tensor_split = body.get("tensor_split") or request.query_params.get("tensor_split")

        registry = get_model_registry()
        vlm = registry.get_vlm()
        import asyncio
        result = await asyncio.to_thread(vlm.reconfigure_spec_mode, spec_mode=spec_mode, n_ctx=n_ctx, tensor_split=tensor_split)
        return JSONResponse(content=result)

    @app.post("/v1/admin/run-ruler-benchmark")
    def admin_run_ruler_benchmark(authorization: Optional[str] = Header(None)):
        """Trigger in-kernel NVIDIA RULER Context Scaling Benchmark (4K → 128K)."""
        verify_auth(authorization)
        require_studio_mode("vlm")
        try:
            from vlm.ruler_runner import start_ruler_scaling_thread, RULER_STATE
            if RULER_STATE.get("status") == "running":
                return JSONResponse(content={"status": "already_running", "state": RULER_STATE})
            start_ruler_scaling_thread(server_base_url=f"http://127.0.0.1:{PORT}")
            return JSONResponse(content={"status": "started", "message": "RULER Scaling benchmark running in background"})
        except Exception as e:
            raise HTTPException(500, f"Failed to start RULER benchmark: {e}")

    @app.get("/v1/admin/ruler-status")
    def admin_ruler_status(authorization: Optional[str] = Header(None)):
        """Check live progress of the in-kernel RULER benchmark."""
        verify_auth(authorization)
        require_studio_mode("vlm")
        try:
            from vlm.ruler_runner import RULER_STATE
            return JSONResponse(content=RULER_STATE)
        except Exception as e:
            return JSONResponse(content={"status": "unknown", "error": str(e)})

    @app.get("/v1/admin/ruler-report")
    def admin_ruler_report(authorization: Optional[str] = Header(None)):
        """Retrieve full RULER scaling report and task markdown files."""
        verify_auth(authorization)
        require_studio_mode("vlm")
        report_path = "/kaggle/working/test_outputs/ruler_scale_report.json"
        if os.path.exists(report_path):
            try:
                import json as _json
                with open(report_path, "r", encoding="utf-8") as f:
                    return JSONResponse(content=_json.load(f))
            except Exception as e:
                return JSONResponse(content={"error": f"Failed to read report: {e}"})
        try:
            from vlm.ruler_runner import RULER_STATE
            return JSONResponse(content=RULER_STATE)
        except Exception:
            return JSONResponse(content={"error": "Report not found"})

    @app.api_route("/v1/admin/shutdown", methods=["GET", "POST"])
    def admin_shutdown(confirm: int = 0, authorization: Optional[str] = Header(None)):
        """Tắt toàn bộ AI Studio (trả mã thoát 99 để dừng hẳn Kaggle Commit Kernel)."""
        verify_auth(authorization)
        if confirm != 1:
            return {"status": "ignored", "message": "Shutdown ignored unless ?confirm=1"}

        def _exit_later():
            try:
                from core.comfyui_manager import stop_all_comfyui_managers
                stop_all_comfyui_managers()
            except Exception:
                pass
            try:
                get_memory_manager().clean_all_vram()
            except Exception:
                pass
            time.sleep(1.0)
            os._exit(99)

        threading.Thread(target=_exit_later, daemon=True).start()
        return {
            "status": "shutting_down_permanently",
            "message": "Server will clean VRAM and exit with code 99 in 1s.",
        }

    return app
