"""
services/music/server.py
FastAPI standalone service daemon for Music Subsystem (port 8001, static domain music.ai.example.com).
Integrates with ACE-Step 1.5 async REST server.
"""
import os
import argparse
import sys
from pathlib import Path
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

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from factory_auth import verify_bearer_token
from factory_contracts.errors import InvalidRequestError
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from services.music.acestep.adapter import ACEStepAdapter
from services.music.acestep.client import ACEStepClient
from services.text.src.multimodal import validate_url_security

app = FastAPI(
    title="Factory v4 Music Service",
    version="1.5.0",
    description="Dedicated Music Generation Subsystem (ACE-Step 1.5)",
)

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

client = ACEStepClient()


# Schemas
class TextToMusicRequest(BaseModel):
    prompt: str = Field(..., description="Musical description prompt")
    duration_seconds: Optional[int] = Field(default=60, ge=5, le=300)
    genre: Optional[str] = "pop"
    tempo: Optional[int] = 120


class LyricsToMusicRequest(BaseModel):
    prompt: str
    lyrics: str
    vocal_style: Optional[str] = "default"
    duration_seconds: Optional[int] = Field(default=60, ge=5, le=300)


class AudioToAudioRequest(BaseModel):
    audio_url: str
    prompt: str
    audio_strength: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    seed: Optional[int] = 42


@app.get("/health")
@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy",
        "service": "music-service",
        "version": "1.5.0",
        "subsystem": "music",
        "gpu_device": 1,
    }


@app.post("/v1/text_to_music", dependencies=[Depends(verify_bearer_token)])
async def generate_text_to_music(req: TextToMusicRequest):
    params = req.model_dump()
    result = await client.generate_music("text_to_music", params)
    artifact = ACEStepAdapter.create_artifact_reference(result)
    return {
        "status": "completed",
        "task": "text_to_music",
        "artifact": artifact.model_dump(mode="json"),
    }


@app.post("/v1/lyrics_to_music", dependencies=[Depends(verify_bearer_token)])
async def generate_lyrics_to_music(req: LyricsToMusicRequest):
    params = req.model_dump()
    result = await client.generate_music("lyrics_to_music", params)
    artifact = ACEStepAdapter.create_artifact_reference(result)
    return {
        "status": "completed",
        "task": "lyrics_to_music",
        "artifact": artifact.model_dump(mode="json"),
    }


@app.post("/v1/audio_to_audio", dependencies=[Depends(verify_bearer_token)])
async def generate_audio_to_audio(req: AudioToAudioRequest):
    validate_url_security(req.audio_url)
    params = req.model_dump()
    result = await client.generate_music("audio_to_audio", params)
    artifact = ACEStepAdapter.create_artifact_reference(result)
    return {
        "status": "completed",
        "task": "audio_to_audio",
        "artifact": artifact.model_dump(mode="json"),
    }


def main():
    parser = argparse.ArgumentParser(description="Music Subsystem Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("services.music.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
