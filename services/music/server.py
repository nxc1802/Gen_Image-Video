"""
Deprecated Factory Music compatibility gateway.

Production Factory v4 exposes ACE-Step 1.5's native REST API directly on the
Music service domain. This module intentionally does not reimplement or mimic
ACE-Step generation endpoints.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Factory v4 Music Compatibility Gateway",
    version="2.0.0",
    description="Deprecated. Use the native ACE-Step 1.5 API.",
)

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


@app.get("/health")
async def health():
    return {
        "status": "deprecated",
        "service": "music-compatibility-gateway",
        "native_api": "ACE-Step 1.5",
    }


def _native_only() -> None:
    raise HTTPException(
        status_code=410,
        detail=(
            "This custom Music endpoint was removed. Use ACE-Step native "
            "/release_task, /query_result, /v1/audio or /v1/chat/completions."
        ),
    )


@app.post("/v1/text_to_music")
async def text_to_music():
    _native_only()


@app.post("/v1/lyrics_to_music")
async def lyrics_to_music():
    _native_only()


@app.post("/v1/audio_to_audio")
async def audio_to_audio():
    _native_only()
