"""
services/enhancement/server.py
FastAPI standalone service daemon for Media Enhancement Subsystem (port 8002, domain enhance.ai.example.com).
Implements Features 36-39:
- POST /v1/image_super_resolution (F36)
- POST /v1/video_super_resolution (F37)
- POST /v1/frame_interpolation (F38)
- POST /v1/audio_enhancement (F39)
"""
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

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from factory_auth import verify_bearer_token
from factory_contracts.errors import InvalidRequestError
from factory_contracts.jobs.job_contract import ArtifactReference
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from services.enhancement.processors.audio_enhance import AudioEnhancementProcessor
from services.enhancement.processors.image_sr import ImageSuperResolutionProcessor
from services.enhancement.processors.interpolation import FrameInterpolationProcessor
from services.enhancement.processors.video_sr import VideoSuperResolutionProcessor
from services.text.src.multimodal import validate_url_security

app = FastAPI(
    title="Factory v4 Media Enhancement Service",
    version="1.0.0",
    description="Dedicated Media Post-Processing and Enhancement Subsystem",
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)

image_sr = ImageSuperResolutionProcessor()
video_sr = VideoSuperResolutionProcessor()
interpolation = FrameInterpolationProcessor()
audio_enhance = AudioEnhancementProcessor()


class ImageSRRequest(BaseModel):
    image_url: str
    scale_factor: Optional[int] = Field(default=2, ge=2, le=4)
    pre_denoise: Optional[bool] = False


class VideoSRRequest(BaseModel):
    video_url: str
    scale_factor: Optional[int] = Field(default=2, ge=2, le=2)


class InterpolationRequest(BaseModel):
    video_url: str
    multiplier: Optional[int] = Field(default=2, ge=2, le=4)
    target_fps: Optional[int] = None


class AudioEnhanceRequest(BaseModel):
    audio_url: str
    mode: Optional[str] = "denoise"
    auto_gain_control: Optional[bool] = False


@app.get("/health")
@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy",
        "service": "enhancement-service",
        "version": "1.0.0",
        "subsystem": "enhancement",
        "gpu_device": 1,
    }


@app.post("/v1/image_super_resolution", dependencies=[Depends(verify_bearer_token)])
async def super_resolve_image(req: ImageSRRequest):
    validate_url_security(req.image_url)
    res = image_sr.process(req.model_dump())
    art_id = res["artifact_id"]
    s3_url = f"https://s3.ai.example.com/assets/{art_id}.png"
    art = ArtifactReference(
        artifact_id=art_id,
        type="image",
        mime_type="image/png",
        size_bytes=res["size_bytes"],
        storage_path=f"artifacts/enhancement/{art_id}.png",
        url=s3_url,
        download_url=s3_url,
        sha256=res["sha256"],
    )
    return {"status": "completed", "artifact": art.model_dump(mode="json")}


@app.post("/v1/video_super_resolution", dependencies=[Depends(verify_bearer_token)])
async def super_resolve_video(req: VideoSRRequest):
    validate_url_security(req.video_url)
    res = video_sr.process(req.model_dump())
    art_id = res["artifact_id"]
    s3_url = f"https://s3.ai.example.com/assets/{art_id}.mp4"
    art = ArtifactReference(
        artifact_id=art_id,
        type="video",
        mime_type="video/mp4",
        size_bytes=res["size_bytes"],
        storage_path=f"artifacts/enhancement/{art_id}.mp4",
        url=s3_url,
        download_url=s3_url,
        sha256=res["sha256"],
    )
    return {"status": "completed", "artifact": art.model_dump(mode="json")}


@app.post("/v1/frame_interpolation", dependencies=[Depends(verify_bearer_token)])
async def interpolate_frames(req: InterpolationRequest):
    validate_url_security(req.video_url)
    res = interpolation.process(req.model_dump())
    art_id = res["artifact_id"]
    s3_url = f"https://s3.ai.example.com/assets/{art_id}.mp4"
    art = ArtifactReference(
        artifact_id=art_id,
        type="video",
        mime_type="video/mp4",
        size_bytes=res["size_bytes"],
        storage_path=f"artifacts/enhancement/{art_id}.mp4",
        url=s3_url,
        download_url=s3_url,
        sha256=res["sha256"],
    )
    return {"status": "completed", "artifact": art.model_dump(mode="json")}


@app.post("/v1/audio_enhancement", dependencies=[Depends(verify_bearer_token)])
async def enhance_audio(req: AudioEnhanceRequest):
    validate_url_security(req.audio_url)
    res = audio_enhance.process(req.model_dump())
    art_id = res["artifact_id"]
    s3_url = f"https://s3.ai.example.com/assets/{art_id}.wav"
    art = ArtifactReference(
        artifact_id=art_id,
        type="audio",
        mime_type="audio/wav",
        size_bytes=res["size_bytes"],
        storage_path=f"artifacts/enhancement/{art_id}.wav",
        url=s3_url,
        download_url=s3_url,
        sha256=res["sha256"],
    )
    return {"status": "completed", "artifact": art.model_dump(mode="json")}


def main():
    parser = argparse.ArgumentParser(description="Media Enhancement Subsystem Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8002, help="Bind port")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("services.enhancement.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
