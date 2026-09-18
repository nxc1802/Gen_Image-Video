#!/usr/bin/env python3
"""
⚡ FLUX.1 Supabase Local Bridge Server (OpenAI DALL-E 3 Compatible)
------------------------------------------------------------------
Chạy cục bộ trên máy dev/server của bạn.
Cung cấp endpoint OpenAI chuẩn: POST /v1/images/generations
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
from typing import Optional

DEFAULT_SUPABASE_URL = "https://fxepzlszglckfsscport.supabase.co"
DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4ZXB6bHN6Z2xja2Zzc2Nwb3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NDEzMzksImV4cCI6MjEwNTAxNzMzOX0.28rS1waBYB8xvGgHR7utoek9PqBc3ev6HPOG9yo9RdQ"


def run_bridge_server(supabase_url: str, supabase_key: str, host: str = "0.0.0.0", port: int = 8000, timeout_sec: int = 180):
    """Chạy Local FastAPI Server chuẩn OpenAI DALL-E 3, đồng bộ qua Supabase."""
    try:
        from fastapi import FastAPI, HTTPException, Header
        from pydantic import BaseModel
        from supabase import create_client
        import uvicorn
    except ImportError as e:
        print(f"❌ Thiếu thư viện: {e}")
        print("Cài đặt nhanh: pip install fastapi uvicorn supabase pydantic")
        sys.exit(1)

    print("=" * 70)
    print("🚀 KHỞI ĐỘNG LOCAL SUPABASE BRIDGE SERVER (OPENAI DALL-E COMPATIBLE)")
    print(f"   Supabase URL : {supabase_url}")
    print(f"   Local Endpoint: http://{host}:{port}/v1")
    print("=" * 70)

    sb = create_client(supabase_url, supabase_key)
    api_app = FastAPI(title="FLUX.1 Supabase Bridge API", version="2.0.0")

    class ImageGenRequest(BaseModel):
        prompt: str
        model: Optional[str] = "flux-2-klein-4b"
        n: Optional[int] = 1
        size: Optional[str] = "1024x1024"
        response_format: Optional[str] = "b64_json"
        seed: Optional[int] = None
        steps: Optional[int] = None
        guidance: Optional[float] = None

    @api_app.get("/health")
    def health():
        return {"status": "ok", "service": "FLUX.2 Supabase Bridge", "supabase": supabase_url}

    @api_app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "flux-2-klein-4b", "object": "model", "owned_by": "unsloth"},
                {"id": "flux-2-klein-9b", "object": "model", "owned_by": "unsloth"},
                {"id": "flux-2-dev", "object": "model", "owned_by": "black-forest-labs"},
                {"id": "dall-e-3", "object": "model", "owned_by": "openai-alias"},
            ],
        }

    @api_app.post("/v1/images/generations")
    async def generate_images(req: ImageGenRequest, authorization: Optional[str] = Header(None)):
        job_id = str(uuid.uuid4())
        insert_payload = {
            "id": job_id,
            "prompt": req.prompt,
            "model": req.model,
            "size": req.size or "1024x1024",
            "seed": req.seed,
            "steps": req.steps or (4 if "schnell" in (req.model or "").lower() else 28),
            "guidance": req.guidance or (0.0 if "schnell" in (req.model or "").lower() else 3.5),
            "response_format": req.response_format or "b64_json",
            "status": "pending",
        }

        print(f"[BRIDGE] 📤 Gửi Job #{job_id}: '{req.prompt[:50]}...' ({insert_payload['size']})")
        res = sb.table("image_jobs").insert(insert_payload).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Không thể tạo job trong Supabase")

        start_wait = time.time()
        poll_count = 0
        while time.time() - start_wait < timeout_sec:
            time.sleep(1.0)
            poll_count += 1
            check = sb.table("image_jobs").select("*").eq("id", job_id).execute()
            if check.data and len(check.data) > 0:
                cur = check.data[0]
                status = cur.get("status")
                if status == "completed":
                    elapsed = cur.get("inference_time_sec") or (time.time() - start_wait)
                    b64 = cur.get("result_b64")
                    print(f"[BRIDGE] ✨ Job #{job_id} hoàn tất sau {elapsed:.1f}s!")
                    if req.response_format == "url":
                        return {
                            "created": int(time.time()),
                            "data": [{"url": f"data:image/png;base64,{b64}", "revised_prompt": req.prompt}],
                        }
                    return {
                        "created": int(time.time()),
                        "data": [{"b64_json": b64, "revised_prompt": req.prompt}],
                    }
                elif status == "failed":
                    err = cur.get("error_message") or "Worker GPU báo lỗi sinh ảnh"
                    print(f"[BRIDGE] ❌ Job #{job_id} thất bại: {err}")
                    raise HTTPException(status_code=500, detail=f"GPU Worker Error: {err}")
            if poll_count % 5 == 0:
                print(f"[BRIDGE] ⏳ Đang đợi GPU xử lý Job #{job_id}... ({int(time.time() - start_wait)}s)")

        raise HTTPException(status_code=504, detail=f"Timeout sau {timeout_sec}s chờ GPU Worker")

    uvicorn.run(api_app, host=host, port=port)


def run_test_client(base_url: str = "http://localhost:8000/v1", prompt: str = "A cute baby astronaut floating in deep space nebula, 8k, Unreal Engine 5", output: str = "flux_result.png"):
    """Gửi test request tới Local Bridge."""
    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": "flux-1-dev",
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    print("=" * 70)
    print(f"🧪 GỬI TEST REQUEST TỚI: {url}")
    print(f"📝 Prompt: {prompt}")
    print("=" * 70)

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - t0
            b64 = res_json["data"][0].get("b64_json")
            if b64:
                with open(output, "wb") as f:
                    f.write(base64.b64decode(b64))
                print(f"✅ SINH ẢNH THÀNH CÔNG trong {elapsed:.2f}s!")
                print(f"💾 File ảnh lưu tại: {os.path.abspath(output)}")
            else:
                print("⚠️ Kết quả không chứa b64_json:", res_json)
    except urllib.error.URLError as e:
        print(f"❌ Lỗi kết nối: {e}")
        sys.exit(1)


def run_heartbeat(url: str, token: str, interval: int = 15):
    """Duy trì kết nối Marimo Molab liên tục để không bị timeout."""
    import urllib.request
    clean_url = url.rstrip("/")
    ping_url = f"{clean_url}/api/kernel/ping"

    print(f"🚀 Bắt đầu Heartbeat tới {clean_url} (Mỗi {interval}s)...")
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "MarimoHeartbeat/2.0"}
    count = 0
    while True:
        count += 1
        try:
            req = urllib.request.Request(ping_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[{time.strftime('%H:%M:%S')}] [Nhịp #{count}] Heartbeat OK (HTTP {resp.status})")
        except Exception as e:
            try:
                req = urllib.request.Request(clean_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    print(f"[{time.strftime('%H:%M:%S')}] [Nhịp #{count}] Fallback Ping OK (HTTP {resp.status})")
            except Exception as fe:
                print(f"[{time.strftime('%H:%M:%S')}] [Nhịp #{count}] Ping lỗi: {fe}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="⚡ FLUX.1 Supabase Bridge & Test Client")
    parser.add_argument("--test", action="store_true", help="Gửi test request sinh ảnh")
    parser.add_argument("--heartbeat", action="store_true", help="Duy trì heartbeat tới Marimo session")
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--prompt", default="A cute baby astronaut floating in deep space nebula, 8k, Unreal Engine 5")
    parser.add_argument("--output", default="flux_result.png")
    parser.add_argument("--url", help="URL session Marimo cho heartbeat")
    parser.add_argument("--token", help="Auth token cho heartbeat")
    parser.add_argument("--interval", type=int, default=15)

    args = parser.parse_args()

    if args.test:
        run_test_client(base_url=args.base_url, prompt=args.prompt, output=args.output)
    elif args.heartbeat:
        if not args.url or not args.token:
            print("❌ Lỗi: Cần cung cấp --url và --token cho heartbeat!")
            sys.exit(1)
        run_heartbeat(args.url, args.token, interval=args.interval)
    else:
        # Mặc định: Chạy Local Bridge Server
        run_bridge_server(args.supabase_url, args.supabase_key, host=args.host, port=args.port)
