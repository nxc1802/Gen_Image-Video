#!/usr/bin/env python3
"""
Supabase <-> OpenAI DALL-E Local Bridge Server
----------------------------------------------------------------
Chạy trên máy tính cá nhân (hoặc server của bạn) tại http://localhost:8000.
Đóng vai trò máy chủ OpenAI DALL-E chuẩn cho mọi client (OpenAI SDK, Cursor, OpenWebUI).
Điều phối công việc qua Supabase database tới GPU worker trên Molab/Cloud mà KHÔNG CẦN TUNNEL.
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import Client, create_client

app = FastAPI(
    title="Supabase FLUX.1 OpenAI DALL-E Bridge",
    description="Drop-in replacement for OpenAI Images API backed by Supabase Queue",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_client: Optional[Client] = None
API_KEY = os.environ.get("FLUX_API_KEY", "flux-sk-test-dalle-2026")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to generate")
    model: Optional[str] = Field(default="flux-1-dev", description="flux-1-dev, flux-1-schnell, dall-e-3")
    n: Optional[int] = Field(default=1, ge=1, le=4)
    size: Optional[str] = Field(default="1024x1024")
    response_format: Optional[str] = Field(default="b64_json")
    quality: Optional[str] = Field(default="standard")
    seed: Optional[int] = Field(default=None)
    steps: Optional[int] = Field(default=None)
    guidance: Optional[float] = Field(default=None)


def check_auth(authorization: Optional[str]):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API Key. Include 'Authorization: Bearer <key>'"}}
        )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "bridge": "supabase-job-queue",
        "supabase_connected": supabase_client is not None,
    }


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {"id": "flux-1-dev", "object": "model", "created": int(time.time()), "owned_by": "black-forest-labs"},
            {"id": "flux-1-schnell", "object": "model", "created": int(time.time()), "owned_by": "black-forest-labs"},
            {"id": "dall-e-3", "object": "model", "created": int(time.time()), "owned_by": "openai"},
        ],
    }


@app.post("/v1/images/generations")
async def generate_images(
    request: ImageGenerationRequest,
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    num_images = request.n or 1
    data_items = []
    start_time = time.time()

    print(f"\n[BRIDGE] Nhận request tạo {num_images} ảnh: '{request.prompt[:80]}...' ({request.size})")

    for i in range(num_images):
        seed_val = (request.seed + i) if request.seed is not None else None

        # 1. Tạo Job trong Supabase
        job_data = {
            "prompt": request.prompt,
            "model": request.model or "flux-1-dev",
            "size": request.size or "1024x1024",
            "seed": seed_val,
            "steps": request.steps,
            "guidance": request.guidance,
            "response_format": request.response_format or "b64_json",
            "status": "pending",
        }

        try:
            insert_res = supabase_client.table("image_jobs").insert(job_data).execute()
            if not insert_res.data:
                raise HTTPException(status_code=500, detail="Không thể tạo job trên Supabase")
            job_id = insert_res.data[0]["id"]
            print(f"[BRIDGE] Đã đẩy Job #{job_id} vào Supabase queue. Đang chờ GPU worker xử lý...")
        except Exception as e:
            print(f"[BRIDGE ERROR] Lỗi khi insert Supabase: {e}")
            raise HTTPException(status_code=500, detail=f"Supabase Error: {e}")

        # 2. Polling chờ kết quả từ GPU worker
        timeout = 180  # 3 phút tối đa
        poll_interval = 0.5
        elapsed_poll = 0
        completed_job = None

        while elapsed_poll < timeout:
            await asyncio.sleep(poll_interval)
            elapsed_poll += poll_interval

            check_res = supabase_client.table("image_jobs").select("*").eq("id", job_id).single().execute()
            row = check_res.data
            if not row:
                continue

            if row.get("status") == "completed":
                completed_job = row
                break
            elif row.get("status") == "failed":
                err = row.get("error_message", "Worker báo lỗi không xác định")
                raise HTTPException(status_code=500, detail=f"GPU Worker failed: {err}")

        if not completed_job:
            raise HTTPException(status_code=504, detail="Hết thời gian chờ GPU worker (Timeout 180s)")

        total_t = time.time() - start_time
        print(f"[BRIDGE SUCCESS] Job #{job_id} hoàn thành trong {total_t:.2f}s (Inference: {completed_job.get('inference_time_sec', 'N/A')}s)")

        item = {"revised_prompt": request.prompt}
        if request.response_format == "url" and completed_job.get("image_url"):
            item["url"] = completed_job["image_url"]
        else:
            item["b64_json"] = completed_job.get("result_b64", "")

        data_items.append(item)

    return {
        "created": int(time.time()),
        "data": data_items,
    }


def main():
    global supabase_client

    parser = argparse.ArgumentParser(description="Supabase <-> OpenAI DALL-E Local Bridge Server")
    parser.add_argument("--port", type=int, default=8000, help="Local port (default: 8000)")
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL", "https://fxepzlszglckfsscport.supabase.co"))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_KEY"))
    args = parser.parse_args()

    if not args.supabase_key:
        print("❌ Thiếu Supabase API Key! Truyền qua --supabase-key hoặc export SUPABASE_KEY='...'")
        print("   (Lấy 'anon' hoặc 'service_role' key từ Supabase Dashboard -> Project Settings -> API)")
        sys.exit(1)

    print("=" * 70)
    print("🌉 SUPABASE <-> OPENAI DALL-E BRIDGE SERVER")
    print("=" * 70)
    print(f"Supabase Project : {args.supabase_url}")
    print(f"Local Endpoint   : http://127.0.0.1:{args.port}/v1")
    print(f"API Key          : {API_KEY}")
    print("=" * 70)

    supabase_client = create_client(args.supabase_url, args.supabase_key)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
