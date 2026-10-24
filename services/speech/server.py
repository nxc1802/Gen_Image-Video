"""
services/speech/server.py
FastAPI standalone service daemon for Speech Subsystem (port 8288, domain speech.ai.example.com).
Implements Features 15-18:
- POST /v1/audio/transcriptions
- POST /v1/audio/transcriptions/stream
- WS   /v1/audio/transcriptions/ws
- POST /v1/audio/alignments
- POST /v1/audio/speech
"""
import os
import argparse
import json
import logging
import secrets
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Optional

# Bootstrap workspace package search paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _subpath in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "packages" / "contracts" / "src",
    _PROJECT_ROOT / "packages" / "auth" / "src",
    _PROJECT_ROOT / "packages" / "model-registry" / "src",
]:
    if str(_subpath) not in sys.path:
        sys.path.insert(0, str(_subpath))

from fastapi import Depends, FastAPI, Header, Query, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from factory_auth import verify_bearer_token, load_auth_config, decode_jwt_token
from factory_contracts.errors import InvalidRequestError, UnauthorizedError
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from services.speech.alignment.aligner import ForcedAligner
from services.speech.asr.engine import ASREngine
from services.speech.asr.streaming import StreamingASRBuffer
from services.speech.tts.engine import TTSEngine
from services.text.src.multimodal import validate_url_security

logger = logging.getLogger(__name__)


def validate_factory_asset_url(url: str) -> None:
    """Require remote speech references to be pre-ingested Factory-owned assets."""
    validate_url_security(url)
    hostname = (urlparse(url).hostname or "").lower()
    allowed = {
        host.strip().lower()
        for host in os.environ.get("FACTORY_ASSET_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }
    s3_host = (urlparse(os.environ.get("S3_PUBLIC_ENDPOINT_URL", "")).hostname or "").lower()
    if s3_host:
        allowed.add(s3_host)
    if not hostname or hostname not in allowed:
        raise InvalidRequestError(
            "Remote speech media must be ingested through Factory Asset Service before use."
        )

app = FastAPI(
    title="Factory v4 Speech Service",
    version="1.0.0",
    description="Dedicated Speech Subsystem for ASR, Streaming ASR, Alignment, and TTS",
)

# Middlewares & Sanitized Handlers
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("FACTORY_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)

# Service Engines
asr_engine = ASREngine()
streaming_buffer = StreamingASRBuffer()
forced_aligner = ForcedAligner()
tts_engine = TTSEngine()


# Request Schemas
class TranscriptionRequest(BaseModel):
    model: str = "qwen3-asr"
    audio_url: Optional[str] = None
    language: Optional[str] = None
    temperature: Optional[float] = 0.0
    prompt: Optional[str] = None


class StreamChunkRequest(BaseModel):
    chunk_id: int
    audio_base64: Optional[str] = None
    model: Optional[str] = "qwen3-asr"


class AlignmentRequest(BaseModel):
    audio_url: str
    transcript_text: str


class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: Optional[str] = "alloy"
    reference_audio_url: Optional[str] = None
    reference_text: Optional[str] = None
    speed: Optional[float] = 1.0
    response_format: Optional[str] = "wav"


# Public Health Endpoints
@app.get("/health")
@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy",
        "service": "speech-service",
        "version": "1.0.0",
        "subsystem": "speech",
    }


# Feature 15: Audio Transcription
@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_bearer_token)])
async def create_transcription(request: Request):
    content_type = request.headers.get("content-type", "")
    body_bytes = await request.body()

    if "application/json" in content_type:
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            raise InvalidRequestError("Malformed JSON payload.")

        audio_url = payload.get("audio_url")
        if not audio_url:
            raise InvalidRequestError("Audio input 'audio_url' is required when submitting JSON.")

        validate_factory_asset_url(audio_url)

        res = asr_engine.transcribe(
            audio_url=audio_url,
            language=payload.get("language"),
            model=payload.get("model", "qwen3-asr"),
            temperature=payload.get("temperature", 0.0),
            prompt=payload.get("prompt"),
        )
        return res

    # Binary or raw audio upload
    if not body_bytes:
        raise InvalidRequestError("Audio input (binary body or audio_url) is required.")

    res = asr_engine.transcribe(
        audio_bytes=body_bytes,
        model=request.headers.get("x-model", "qwen3-asr"),
    )
    return res


# Feature 16: Streaming ASR (Chunked REST)
@app.post("/v1/audio/transcriptions/stream", dependencies=[Depends(verify_bearer_token)])
async def stream_transcription_chunk(payload: Dict[str, Any]):
    chunk_id = payload.get("chunk_id", 1)
    audio_base64 = payload.get("audio_base64")
    model = payload.get("model", "qwen3-asr")

    res = streaming_buffer.process_chunk(chunk_id=chunk_id, audio_base64=audio_base64, model=model)
    return res


# Feature 16: Streaming ASR (WebSocket Bidirectional)
@app.websocket("/v1/audio/transcriptions/ws")
async def websocket_transcription_stream(
    websocket: WebSocket,
    model: str = Query("qwen3-asr"),
):
    auth_config = getattr(websocket.app.state, "auth_config", None) or load_auth_config()
    expected_key = getattr(auth_config, "api_key", None)
    jwt_secret = getattr(auth_config, "jwt_secret", None)

    token: Optional[str] = None
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    if not token:
        protocols = [
            item.strip()
            for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        ]
        bearer_protocol = next(
            (item for item in protocols if item.startswith("factory-bearer.")),
            None,
        )
        if bearer_protocol:
            token = bearer_protocol[len("factory-bearer."):]

    authenticated = False
    if expected_key and token and secrets.compare_digest(token, expected_key):
        authenticated = True
    elif jwt_secret and token and token.count(".") == 2:
        try:
            decode_jwt_token(token, jwt_secret)
            authenticated = True
        except Exception:
            authenticated = False

    if not authenticated:
        await websocket.close(code=1008)  # Policy Violation / Unauthorized
        return

    await websocket.accept()
    session_id = f"wss-{uuid.uuid4()}"
    chunk_counter = 0

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    data = {}

                if data.get("action") == "end":
                    final_event = streaming_buffer.finalize_ws_session(session_id)
                    await websocket.send_json(final_event)
                    await websocket.close()
                    break

                chunk_counter += 1
                chunk_id = data.get("chunk_id", chunk_counter)
                event = streaming_buffer.process_ws_chunk(session_id, chunk_id)
                await websocket.send_json(event)

            elif "bytes" in message:
                chunk_counter += 1
                event = streaming_buffer.process_ws_chunk(session_id, chunk_counter, message["bytes"])
                await websocket.send_json(event)

    except (WebSocketDisconnect, WebSocketDisconnected, Exception) as exc:
        logger.debug(f"Speech WebSocket stream disconnected: {exc}")
    finally:
        streaming_buffer.finalize_ws_session(session_id)
        try:
            await websocket.close()
        except Exception:
            pass


# Feature 17: Forced Alignment
@app.post("/v1/audio/alignments", dependencies=[Depends(verify_bearer_token)])
async def forced_alignment(req: AlignmentRequest):
    validate_factory_asset_url(req.audio_url)
    return forced_aligner.align(audio_url=req.audio_url, transcript_text=req.transcript_text)


# Feature 18: Speech Synthesis (TTS)
@app.post("/v1/audio/speech", dependencies=[Depends(verify_bearer_token)])
async def synthesize_speech(req: SpeechRequest):
    if req.reference_audio_url:
        validate_factory_asset_url(req.reference_audio_url)
    wav_bytes = tts_engine.synthesize(
        text=req.input,
        voice=req.voice,
        reference_audio_url=req.reference_audio_url,
        speed=req.speed or 1.0,
        response_format=req.response_format or "wav",
        model=req.model,
        reference_text=req.reference_text,
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


def main():
    parser = argparse.ArgumentParser(description="Speech Subsystem Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8288, help="Bind port")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("services.speech.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
