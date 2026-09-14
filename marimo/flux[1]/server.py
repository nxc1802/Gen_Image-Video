#!/usr/bin/env python3
"""
OpenAI-Compatible DALL-E Image Generation API Server for FLUX.1
----------------------------------------------------------------
Provides a drop-in replacement for OpenAI's Images API (POST /v1/images/generations).
Compatible with official OpenAI SDKs, LangChain, LlamaIndex, OpenWebUI, Cursor, etc.
"""

import argparse
import base64
import gc
import io
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import uvicorn


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
HOST = "0.0.0.0"
DEFAULT_PORT = 8000

IMAGES_DIR = Path("/tmp/flux_api_images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("FLUX_API_KEY")
if not API_KEY:
    API_KEY = "flux-" + base64.urlsafe_b64encode(os.urandom(24)).decode().rstrip("=")
    os.environ["FLUX_API_KEY"] = API_KEY


# Global pipeline reference and concurrency lock
pipeline = None
generation_lock = threading.Lock()
public_tunnel_url = None
cloudflared_process = None


# ============================================================
# 2. FASTAPI APP
# ============================================================

app = FastAPI(
    title="FLUX.1 OpenAI-Compatible DALL-E API",
    description="Drop-in replacement for OpenAI Images API (POST /v1/images/generations)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount directory to serve images via URL
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ============================================================
# 3. REQUEST & RESPONSE SCHEMAS (OPENAI DALL-E SPEC)
# ============================================================

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="A text description of the desired image(s).")
    model: Optional[str] = Field(default="flux-1-dev", description="The model to use (flux-1-dev, flux-1-schnell, dall-e-3).")
    n: Optional[int] = Field(default=1, ge=1, le=4, description="The number of images to generate.")
    size: Optional[str] = Field(default="1024x1024", description="Resolution: 1024x1024, 768x1360, 1360x768, 1280x720, etc.")
    response_format: Optional[str] = Field(default="b64_json", description="'b64_json' or 'url'")
    quality: Optional[str] = Field(default="standard", description="'standard' or 'hd'")
    style: Optional[str] = Field(default=None, description="Image style (optional)")
    user: Optional[str] = Field(default=None, description="Unique user identifier")

    # Custom FLUX extensions
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    steps: Optional[int] = Field(default=None, ge=1, le=50, description="Custom inference steps")
    guidance: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Custom guidance scale")


def check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Incorrect API key provided. Include 'Authorization: Bearer <key>'.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


def parse_resolution(size_str: str) -> tuple[int, int]:
    if not size_str:
        return (1024, 1024)
    size_clean = size_str.lower().replace(" ", "")
    parts = size_clean.split("x")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        w = int(parts[0])
        h = int(parts[1])
        # Ensure dimensions are multiples of 16 for DiT
        w = (w // 16) * 16
        h = (h // 16) * 16
        return (w, h)
    return (1024, 1024)


# ============================================================
# 4. ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0.0,
    }


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "flux-1-dev",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "black-forest-labs",
            },
            {
                "id": "flux-1-schnell",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "black-forest-labs",
            },
            {
                "id": "dall-e-3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
            },
        ],
    }


@app.post("/v1/images/generations")
def generate_images(
    request: ImageGenerationRequest,
    req_http: Request,
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Model pipeline is not yet loaded on GPU.",
                    "type": "server_error",
                    "code": "model_not_ready",
                }
            },
        )

    width, height = parse_resolution(request.size or "1024x1024")
    num_images = request.n or 1

    # Infer steps & guidance based on model / quality
    model_name = (request.model or "flux-1-dev").lower()
    is_schnell = "schnell" in model_name

    if request.steps is not None:
        steps = request.steps
    elif is_schnell:
        steps = 4
    elif request.quality == "hd":
        steps = 35
    else:
        steps = 28

    if request.guidance is not None:
        guidance = request.guidance
    elif is_schnell:
        guidance = 0.0
    else:
        guidance = 3.5

    # Determine base url for image links
    base_url = public_tunnel_url or str(req_http.base_url).rstrip("/")

    data_items = []
    start_time = time.time()

    print(f"\n[API INCOMING] Generating {num_images} image(s) | Size: {width}x{height} | Steps: {steps} | Guidance: {guidance}")
    print(f"Prompt: {request.prompt[:100]}...")

    with generation_lock:
        for i in range(num_images):
            if request.seed is not None:
                current_seed = request.seed + i
            else:
                current_seed = int(time.time() * 1000 + i * 997) % 2147483647

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=dev).manual_seed(current_seed)
            max_seq_len = 256 if steps <= 4 else 512

            try:
                result = pipeline(
                    prompt=request.prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    max_sequence_length=max_seq_len,
                )
                img = result.images[0]
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HTTPException(
                    status_code=507,
                    detail={"error": {"message": "GPU out of memory during generation.", "type": "insufficient_storage"}},
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"message": f"Generation failed: {e}", "type": "server_error"}},
                )

            item = {"revised_prompt": request.prompt}

            if request.response_format == "url":
                filename = f"flux_{uuid.uuid4().hex}.png"
                filepath = IMAGES_DIR / filename
                img.save(filepath)
                item["url"] = f"{base_url}/images/{filename}"
            else:
                # Default: b64_json
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                b64_str = base64.b64encode(buffer.getvalue()).decode("ascii")
                item["b64_json"] = b64_str

            data_items.append(item)

    elapsed = time.time() - start_time
    print(f"[API SUCCESS] Completed in {elapsed:.2f}s ({elapsed/num_images:.2f}s/image)")

    return {
        "created": int(time.time()),
        "data": data_items,
    }


# ============================================================
# 5. PIPELINE & TUNNEL INITIALIZER
# ============================================================

def load_pipeline_instance(model_id: str = DEFAULT_MODEL, vram_strategy: str = "auto"):
    global pipeline
    from diffusers import FluxPipeline

    token = os.environ.get("HF_TOKEN") or None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {model_id} ({dtype})...")

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    )

    if vram_strategy == "auto":
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_strategy = "full_gpu" if vram_gb >= 28.0 else ("cpu_offload" if vram_gb >= 14.0 else "sequential_offload")
        else:
            vram_strategy = "cpu"

    if vram_strategy == "full_gpu" and torch.cuda.is_available():
        pipe.to("cuda")
        print("Pipeline running 100% on GPU (Max speed).")
    elif vram_strategy == "cpu_offload":
        pipe.enable_model_cpu_offload()
        print("Pipeline running with Model CPU Offload.")
    elif vram_strategy == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
        print("Pipeline running with Sequential CPU Offload.")
    else:
        pipe.to("cpu")
        print("Pipeline running on CPU.")

    pipeline = pipe
    return pipeline


def start_cloudflare_tunnel(port: int) -> Optional[str]:
    global cloudflared_process, public_tunnel_url

    cloudflared_bin = "/tmp/cloudflared" if os.path.exists("/tmp/cloudflared") else "cloudflared"

    try:
        cloudflared_process = subprocess.Popen(
            [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        deadline = time.time() + 25
        while time.time() < deadline:
            line = cloudflared_process.stdout.readline()
            if not line:
                break
            m = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
            if m:
                public_tunnel_url = m.group(0)
                return public_tunnel_url
            time.sleep(0.1)
    except Exception as e:
        print(f"Could not start Cloudflare tunnel: {e}")

    return None


def run_server(port: int = DEFAULT_PORT):
    uvicorn.run(app, host=HOST, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible DALL-E Server for FLUX.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="FLUX model identifier")
    parser.add_argument("--vram", type=str, default="auto", choices=["auto", "full_gpu", "cpu_offload", "sequential_offload"])
    parser.add_argument("--tunnel", action="store_true", help="Start Cloudflare public tunnel")
    args = parser.parse_args()

    print("=" * 70)
    print("🎨 FLUX.1 OPENAI DALL-E COMPATIBLE SERVER")
    print("=" * 70)
    print(f"Model          : {args.model}")
    print(f"API Key        : {API_KEY}")
    print(f"Local Endpoint : http://127.0.0.1:{args.port}/v1")
    print(f"Images Endpoint: http://127.0.0.1:{args.port}/v1/images/generations")

    load_pipeline_instance(args.model, args.vram)

    if args.tunnel:
        print("\nStarting Cloudflare Tunnel...")
        t_url = start_cloudflare_tunnel(args.port)
        if t_url:
            print(f"\n🌐 PUBLIC OPENAI API BASE URL:")
            print(f"{t_url}/v1")
            print(f"\n🌐 Open in OpenAI SDK:")
            print(f"client = OpenAI(base_url='{t_url}/v1', api_key='{API_KEY}')")

    print("\nStarting Uvicorn Server...")
    print("=" * 70)
    run_server(args.port)


if __name__ == "__main__":
    main()
