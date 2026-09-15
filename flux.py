import argparse
import base64
import gc
import io
import json
import os
import sys
import threading
import time
import uuid
from typing import Optional

# ==============================================================================
# 1. SQL SCHEMA CONSTANT (Dùng cho --sql hoặc auto-init)
# ==============================================================================
SUPABASE_SCHEMA_SQL = """-- ============================================================
-- FLUX.1 Image Generation Queue Schema for Supabase
-- ============================================================

CREATE TABLE IF NOT EXISTS public.image_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    prompt TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'flux-1-dev',
    size TEXT NOT NULL DEFAULT '1024x1024',
    seed BIGINT,
    steps INT,
    guidance FLOAT,
    response_format TEXT DEFAULT 'b64_json',
    status TEXT NOT NULL DEFAULT 'pending',
    result_b64 TEXT,
    image_url TEXT,
    error_message TEXT,
    inference_time_sec FLOAT,
    device_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_image_jobs_queue ON public.image_jobs (status, created_at);
ALTER TABLE public.image_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public select on image_jobs" ON public.image_jobs FOR SELECT USING (true);
CREATE POLICY "Allow public insert on image_jobs" ON public.image_jobs FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update on image_jobs" ON public.image_jobs FOR UPDATE USING (true);
ALTER PUBLICATION supabase_realtime ADD TABLE public.image_jobs;
"""

try:
    import marimo
    app = marimo.App(
        width="full",
        app_title="⚡ FLUX.1 Studio + Supabase AI Bridge",
    )
except ImportError:
    marimo = None
    class _DummyApp:
        def cell(self, *args, **kwargs):
            return lambda fn: fn
        def run(self):
            print("❌ Marimo chưa được cài đặt trên môi trường này. Cài đặt bằng: pip install marimo")
    app = _DummyApp()


@app.cell
def _():
    import base64
    import gc
    import io
    import os
    import sys
    import threading
    import time
    from pathlib import Path
    from typing import Optional

    import marimo as mo
    import torch
    from PIL import Image

    try:
        from diffusers import FluxPipeline
    except ImportError:
        FluxPipeline = None

    try:
        from supabase import create_client
    except ImportError:
        create_client = None

    return (
        FluxPipeline,
        Image,
        Optional,
        Path,
        base64,
        create_client,
        gc,
        io,
        mo,
        os,
        sys,
        threading,
        time,
        torch,
    )


@app.cell
def _(mo, torch):
    def detect_gpu():
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            tot = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            return {
                "type": "cuda",
                "name": dev,
                "vram": f"{tot:.1f} GB",
                "free_vram": f"{tot - alloc:.1f} GB",
                "bf16": torch.cuda.is_bf16_supported(),
                "status": "🟢 GPU Online",
            }
        return {
            "type": "cpu",
            "name": "CPU Fallback",
            "vram": "System RAM",
            "free_vram": "N/A",
            "bf16": False,
            "status": "⚠️ CPU Fallback",
        }

    gpu_info = detect_gpu()

    header = mo.md(
        f"""
        # ⚡ FLUX.1 Studio + Supabase AI Bridge
        > **Kiến trúc All-In-One:** 12B Flow Matching Transformer + Supabase Database Broker (Miễn nhiễm ngắt kết nối 100%)

        | Phần cứng | Thiết bị | VRAM | bfloat16 | Trạng thái |
        | :--- | :--- | :--- | :--- | :--- |
        | `{gpu_info["type"].upper()}` | **{gpu_info["name"]}** | **{gpu_info["vram"]}** | `{"Hỗ trợ" if gpu_info["bf16"] else "Không"}` | {gpu_info["status"]} |
        """
    )
    header
    return detect_gpu, gpu_info, header


@app.cell
def _(mo, os):
    model_choice = mo.ui.dropdown(
        options={
            "FLUX.1-dev (28 steps, Cực chi tiết, Cần HF Token)": "black-forest-labs/FLUX.1-dev",
            "FLUX.1-schnell (4 steps, Siêu nhanh, Không cần token)": "black-forest-labs/FLUX.1-schnell",
        },
        value="FLUX.1-dev (28 steps, Cực chi tiết, Cần HF Token)",
        label="Biến thể FLUX.1:",
    )

    vram_mode = mo.ui.dropdown(
        options={
            "Full GPU (Tốc độ tối đa, VRAM >= 24GB)": "full_gpu",
            "Model CPU Offload (Tiết kiệm VRAM, 12GB - 24GB)": "cpu_offload",
        },
        value="Full GPU (Tốc độ tối đa, VRAM >= 24GB)",
        label="Tối ưu VRAM:",
    )

    hf_token = mo.ui.text(
        value=os.environ.get("HF_TOKEN", ""),
        placeholder="hf_xxxxxxxx (Gated access token)",
        label="Hugging Face Token:",
        kind="password",
    )

    load_btn = mo.ui.run_button(
        label="📥 Tải / Nạp Mô Hình Vào VRAM",
        kind="success",
    )

    model_settings_ui = mo.vstack(
        [
            mo.md("### ⚙️ 1. Cấu Hình & Nạp Mô Hình Vào VRAM"),
            mo.hstack([model_choice, vram_mode], gap=2, justify="start"),
            mo.hstack([hf_token, load_btn], gap=2, justify="start", align="end"),
        ],
        gap=1,
    )
    model_settings_ui
    return hf_token, load_btn, model_choice, model_settings_ui, vram_mode


@app.cell
def _(
    FluxPipeline,
    gc,
    gpu_info,
    hf_token,
    load_btn,
    mo,
    model_choice,
    os,
    torch,
    vram_mode,
):
    try:
        _current_pipe = pipeline
    except NameError:
        _current_pipe = None

    should_load = load_btn.value or (_current_pipe is None)

    if should_load:
        if FluxPipeline is None:
            pipeline = None
            status_banner = mo.callout("❌ Thiếu thư viện diffusers. Hãy cài đặt: `pip install diffusers transformers accelerate`", kind="danger")
        else:
            sel_model = model_choice.value
            tok = hf_token.value.strip() or os.environ.get("HF_TOKEN") or None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            dtype = torch.bfloat16 if gpu_info.get("bf16", False) else torch.float32

            try:
                with mo.status.spinner(title=f"Đang nạp {sel_model} vào VRAM ({dtype})..."):
                    pipe = FluxPipeline.from_pretrained(sel_model, torch_dtype=dtype, token=tok)
                    if vram_mode.value == "full_gpu" and gpu_info["type"] == "cuda":
                        pipe.to("cuda")
                        desc = f"100% trên {gpu_info['name']} (Tối đa tốc độ)"
                    elif vram_mode.value == "cpu_offload":
                        pipe.enable_model_cpu_offload()
                        desc = "Model CPU Offload kích hoạt"
                    else:
                        pipe.to("cpu")
                        desc = "Chạy CPU"

                    pipeline = pipe
                    status_banner = mo.callout(f"✅ Đã nạp `{sel_model}` ({dtype}). {desc}", kind="success")
            except Exception as e:
                pipeline = _current_pipe
                status_banner = mo.callout(f"❌ Lỗi nạp mô hình: {e}", kind="danger")
    else:
        pipeline = _current_pipe
        status_banner = mo.callout("✅ Mô hình đã sẵn sàng trong VRAM.", kind="success")

    status_banner
    return (
        desc,
        dtype,
        pipe,
        pipeline,
        sel_model,
        should_load,
        status_banner,
        tok,
    )


@app.cell
def _(mo):
    prompt_input = mo.ui.text_area(
        value="A futuristic cybernetic neon lion standing atop a misty skyscraper at night, cyberpunk aesthetic, photorealistic, 8k resolution, cinematic lighting",
        label="Prompt mô tả ảnh:",
        rows=3,
    )

    resolution_choice = mo.ui.dropdown(
        options={
            "1024x1024 (Vuông 1:1)": "1024x1024",
            "768x1360 (Dọc 9:16)": "768x1360",
            "1360x768 (Ngang 16:9)": "1360x768",
            "1280x720 (HD 16:9)": "1280x720",
        },
        value="1024x1024 (Vuông 1:1)",
        label="Kích thước:",
    )

    seed_input = mo.ui.number(value=-1, start=-1, stop=2147483647, step=1, label="Seed (-1 để ngẫu nhiên):")
    steps_slider = mo.ui.slider(start=1, stop=50, value=28, step=1, label="Steps:")
    guidance_slider = mo.ui.slider(start=0.0, stop=10.0, value=3.5, step=0.1, label="Guidance Scale:")
    generate_btn = mo.ui.run_button(label="✨ Sinh Ảnh Trực Tiếp", kind="warn")

    interactive_panel = mo.vstack(
        [
            mo.md("### 🎨 2. Sinh Ảnh Trực Tiếp Trên Web"),
            prompt_input,
            mo.hstack([resolution_choice, seed_input], gap=2),
            mo.hstack([steps_slider, guidance_slider], gap=2),
            generate_btn,
        ],
        gap=1,
    )
    interactive_panel
    return (
        generate_btn,
        guidance_slider,
        interactive_panel,
        prompt_input,
        resolution_choice,
        seed_input,
        steps_slider,
    )


@app.cell
def _(
    generate_btn,
    gpu_info,
    guidance_slider,
    mo,
    pipeline,
    prompt_input,
    resolution_choice,
    seed_input,
    steps_slider,
    time,
    torch,
):
    gen_result_view = mo.md("*Bấm nút sinh ảnh ở trên để bắt đầu tạo ảnh...*")

    if generate_btn.value:
        if pipeline is None:
            gen_result_view = mo.callout("⚠️ Vui lòng nạp mô hình vào VRAM trước!", kind="warn")
        else:
            p_text = prompt_input.value.strip()
            w, h = [int(x) for x in resolution_choice.value.split("x")]
            s_val = int(seed_input.value)
            cur_seed = s_val if s_val >= 0 else int(time.time() * 1000) % 2147483647
            st = int(steps_slider.value)
            gd = float(guidance_slider.value)

            dev = gpu_info["type"]
            gen = torch.Generator(device=dev).manual_seed(cur_seed)

            t0 = time.time()
            with mo.status.spinner(title=f"Đang sinh ảnh FLUX.1 ({st} steps, {w}x{h})..."):
                try:
                    res = pipeline(
                        prompt=p_text,
                        width=w,
                        height=h,
                        num_inference_steps=st,
                        guidance_scale=gd,
                        generator=gen,
                        max_sequence_length=256 if st <= 4 else 512,
                    )
                    gen_img = res.images[0]
                    elapsed = time.time() - t0
                    gen_img.save(f"flux_{cur_seed}.png")

                    gen_result_view = mo.vstack(
                        [
                            mo.md(f"### ✨ Hoàn tất trong `{elapsed:.2f}s` | Kích thước: `{w}x{h}` | Seed: `{cur_seed}`"),
                            mo.image(gen_img),
                        ],
                        gap=1,
                    )
                except Exception as err:
                    gen_result_view = mo.callout(f"❌ Lỗi sinh ảnh: {err}", kind="danger")

    gen_result_view
    return (
        cur_seed,
        dev,
        elapsed,
        gen,
        gen_img,
        gen_result_view,
        h,
        p_text,
        res,
        s_val,
        st,
        t0,
        w,
    )


@app.cell
def _(mo, os):
    sb_url_ui = mo.ui.text(
        value=os.environ.get("SUPABASE_URL", "https://fxepzlszglckfsscport.supabase.co"),
        label="Supabase Project URL:",
    )
    sb_key_ui = mo.ui.text(
        value=os.environ.get("SUPABASE_KEY", ""),
        placeholder="Anon hoặc Service Role Key...",
        label="Supabase API Key:",
        kind="password",
    )
    worker_toggle = mo.ui.switch(
        value=False,
        label="Kích hoạt Background Queue Worker (Lắng nghe Supabase)",
    )

    worker_ui_panel = mo.vstack(
        [
            mo.md("### 🌉 3. Supabase GPU Queue Worker (Bridge Cho API OpenAI)"),
            mo.md(
                "> **Cơ chế Outbound 100%:** Chỉ gửi request HTTPS (Port 443) tới Supabase. "
                "Không mở port, không dùng Cloudflare/SSH tunnel, tuyệt đối không bị bot Molab kill!"
            ),
            mo.hstack([sb_url_ui, sb_key_ui], gap=2),
            worker_toggle,
        ],
        gap=1,
    )
    worker_ui_panel
    return sb_key_ui, sb_url_ui, worker_toggle, worker_ui_panel


@app.cell
def _():
    _worker_state = {"running": False, "thread": None}
    return (_worker_state,)


@app.cell
def _(
    _worker_state,
    base64,
    create_client,
    gpu_info,
    io,
    mo,
    pipeline,
    sb_key_ui,
    sb_url_ui,
    threading,
    time,
    torch,
    worker_toggle,
):
    worker_status_ui = mo.md("*Gạt công tắc ở trên để bắt đầu lắng nghe hàng đợi Supabase.*")

    if worker_toggle.value:
        url_val = sb_url_ui.value.strip()
        key_val = sb_key_ui.value.strip()

        if not create_client:
            worker_status_ui = mo.callout("❌ Thiếu `supabase` package (`pip install supabase`).", kind="danger")
        elif not key_val:
            worker_status_ui = mo.callout("⚠️ Vui lòng nhập Supabase API Key!", kind="warn")
        elif pipeline is None:
            worker_status_ui = mo.callout("⚠️ Vui lòng nạp mô hình ở Mục 1 trước khi bật Worker!", kind="warn")
        else:
            if not _worker_state["running"]:
                _worker_state["running"] = True

                def _run_worker_loop():
                    try:
                        sb = create_client(url_val, key_val)
                        print("[WORKER] Đã kết nối Supabase thành công!")
                    except Exception as ce:
                        print(f"[WORKER] Lỗi kết nối Supabase: {ce}")
                        return

                    while _worker_state["running"]:
                        try:
                            res = sb.table("image_jobs").select("*").eq("status", "pending").order("created_at").limit(1).execute()
                            if res.data and len(res.data) > 0:
                                job = res.data[0]
                                jid = job["id"]
                                lock = sb.table("image_jobs").update({"status": "processing"}).eq("id", jid).eq("status", "pending").execute()
                                if lock.data:
                                    prompt_t = job.get("prompt", "")
                                    size_t = job.get("size", "1024x1024")
                                    try:
                                        pw, ph = [int(x) for x in size_t.split("x")]
                                    except Exception:
                                        pw, ph = 1024, 1024
                                    steps_t = job.get("steps") or 28
                                    guidance_t = job.get("guidance") or 3.5
                                    seed_t = job.get("seed") or (int(time.time() * 1000) % 2147483647)

                                    dev_t = gpu_info.get("type", "cuda")
                                    g_t = torch.Generator(device=dev_t).manual_seed(seed_t)
                                    start_inf = time.time()
                                    try:
                                        img_res = pipeline(
                                            prompt=prompt_t,
                                            width=pw,
                                            height=ph,
                                            num_inference_steps=steps_t,
                                            guidance_scale=guidance_t,
                                            generator=g_t,
                                            max_sequence_length=256 if steps_t <= 4 else 512,
                                        ).images[0]
                                        inf_dt = time.time() - start_inf
                                        buf = io.BytesIO()
                                        img_res.save(buf, format="PNG")
                                        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

                                        sb.table("image_jobs").update({
                                            "status": "completed",
                                            "result_b64": b64_str,
                                            "inference_time_sec": round(inf_dt, 2),
                                            "device_name": gpu_info.get("name", "GPU"),
                                        }).eq("id", jid).execute()
                                        print(f"[WORKER] Xong Job #{jid} trong {inf_dt:.2f}s")
                                    except Exception as ge:
                                        sb.table("image_jobs").update({
                                            "status": "failed",
                                            "error_message": str(ge),
                                        }).eq("id", jid).execute()
                                        print(f"[WORKER] Lỗi Job #{jid}: {ge}")
                        except Exception as loop_e:
                            print(f"[WORKER ERROR] {loop_e}")
                        time.sleep(1.0)

                th = threading.Thread(target=_run_worker_loop, daemon=True)
                _worker_state["thread"] = th
                th.start()

            worker_status_ui = mo.vstack([
                mo.callout("🟢 Worker ĐANG LẮNG NGHE HÀNG ĐỢI SUPABASE!", kind="success"),
                mo.md(f"- Endpoint: `{url_val}` | Bảng: `image_jobs` | Thiết bị: `{gpu_info.get('name')}`"),
            ])
    else:
        _worker_state["running"] = False
        worker_status_ui = mo.callout("⚪ Worker đang dừng.", kind="neutral")

    worker_status_ui
    return (
        key_val,
        url_val,
        worker_status_ui,
    )


# ==============================================================================
# 3. CLI MODES (Bridge API, Worker CLI, Test Client, Heartbeat, SQL)
# ==============================================================================

def run_sql_schema():
    """In ra mã SQL Supabase để người dùng copy-paste vào Supabase Dashboard."""
    print("=" * 70)
    print("📋 SUPABASE DATABASE QUEUE SCHEMA (Chạy tại Supabase SQL Editor)")
    print("=" * 70)
    print(SUPABASE_SCHEMA_SQL)
    print("=" * 70)


def run_bridge_server(supabase_url: str, supabase_key: str, host: str = "0.0.0.0", port: int = 8000, timeout_sec: int = 180):
    """Chạy Local FastAPI Server chuẩn OpenAI DALL-E 3, đồng bộ qua Supabase."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Header
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
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
        model: Optional[str] = "flux-1-dev"
        n: Optional[int] = 1
        size: Optional[str] = "1024x1024"
        response_format: Optional[str] = "b64_json"
        seed: Optional[int] = None
        steps: Optional[int] = None
        guidance: Optional[float] = None

    @api_app.get("/health")
    def health():
        return {"status": "ok", "service": "FLUX.1 Supabase Bridge", "supabase": supabase_url}

    @api_app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "flux-1-dev", "object": "model", "owned_by": "black-forest-labs"},
                {"id": "flux-1-schnell", "object": "model", "owned_by": "black-forest-labs"},
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


def run_test_client(base_url: str = "http://localhost:8000/v1", prompt: str = "A cute little baby astronaut floating in a colorful space nebula, photorealistic, 8k", output: str = "flux_result.png"):
    """Gửi test request tới Local Bridge hoặc OpenAI API."""
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
    if not clean_url.endswith("/api/kernel/ping"):
        ping_url = f"{clean_url}/api/kernel/ping"
    else:
        ping_url = clean_url

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


def run_worker_cli(supabase_url: str, supabase_key: str, model_id: str = "black-forest-labs/FLUX.1-dev"):
    """Chạy standalone worker trong terminal (không cần mở web Marimo)."""
    import torch
    from diffusers import FluxPipeline
    from supabase import create_client

    print("=" * 70)
    print("🚀 FLUX.1 STANDALONE GPU WORKER")
    print(f"   Model: {model_id} | Supabase: {supabase_url}")
    print("=" * 70)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, token=os.environ.get("HF_TOKEN"))
    if dev == "cuda":
        pipe.to("cuda")

    sb = create_client(supabase_url, supabase_key)
    print("✅ Đã nạp xong mô hình vào GPU! Đang lắng nghe Supabase...")

    while True:
        try:
            res = sb.table("image_jobs").select("*").eq("status", "pending").order("created_at").limit(1).execute()
            if res.data:
                j = res.data[0]
                jid = j["id"]
                lock = sb.table("image_jobs").update({"status": "processing"}).eq("id", jid).eq("status", "pending").execute()
                if lock.data:
                    p = j.get("prompt", "")
                    sz = j.get("size", "1024x1024")
                    try:
                        pw, ph = [int(x) for x in sz.split("x")]
                    except Exception:
                        pw, ph = 1024, 1024
                    st = j.get("steps") or 28
                    gd = j.get("guidance") or 3.5
                    sd = j.get("seed") or int(time.time() * 1000) % 2147483647
                    g = torch.Generator(device=dev).manual_seed(sd)

                    t0 = time.time()
                    try:
                        img = pipe(prompt=p, width=pw, height=ph, num_inference_steps=st, guidance_scale=gd, generator=g).images[0]
                        dt = time.time() - t0
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        sb.table("image_jobs").update({"status": "completed", "result_b64": b64, "inference_time_sec": round(dt, 2)}).eq("id", jid).execute()
                        print(f"✨ Xong Job #{jid} trong {dt:.2f}s")
                    except Exception as ge:
                        sb.table("image_jobs").update({"status": "failed", "error_message": str(ge)}).eq("id", jid).execute()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1.0)


# ==============================================================================
# 4. ENTRYPOINT DISPATCHER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="⚡ FLUX.1 All-In-One: Marimo Studio, Supabase Bridge & Worker", add_help=False)
    parser.add_argument("--mode", choices=["app", "bridge", "worker", "test", "heartbeat", "sql"], default=None, help="Chế độ chạy")
    parser.add_argument("--bridge", action="store_true", help="Chạy Local Bridge API Server")
    parser.add_argument("--worker", action="store_true", help="Chạy Standalone GPU Worker")
    parser.add_argument("--test", action="store_true", help="Gửi test request sinh ảnh")
    parser.add_argument("--heartbeat", action="store_true", help="Duy trì heartbeat tới Marimo session")
    parser.add_argument("--sql", action="store_true", help="In ra mã SQL khởi tạo Supabase")

    # Arguments bổ trợ
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL", "https://fxepzlszglckfsscport.supabase.co"))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_KEY", ""))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--prompt", default="A cute baby astronaut floating in deep space, hyper-detailed 8k, Unreal Engine 5")
    parser.add_argument("--output", default="flux_result.png")
    parser.add_argument("--url", help="URL session Marimo cho heartbeat")
    parser.add_argument("--token", help="Auth token cho heartbeat")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("-h", "--help", action="help", help="Hiển thị hướng dẫn này")

    args, unknown = parser.parse_known_args()

    # Xác định mode
    active_mode = args.mode
    if args.bridge:
        active_mode = "bridge"
    elif args.worker:
        active_mode = "worker"
    elif args.test:
        active_mode = "test"
    elif args.heartbeat:
        active_mode = "heartbeat"
    elif args.sql:
        active_mode = "sql"

    if active_mode == "sql":
        run_sql_schema()
    elif active_mode == "bridge":
        if not args.supabase_key:
            print("❌ Lỗi: Cần cung cấp --supabase-key hoặc biến môi trường SUPABASE_KEY!")
            sys.exit(1)
        run_bridge_server(args.supabase_url, args.supabase_key, host=args.host, port=args.port)
    elif active_mode == "test":
        run_test_client(base_url=args.base_url, prompt=args.prompt, output=args.output)
    elif active_mode == "worker":
        if not args.supabase_key:
            print("❌ Lỗi: Cần cung cấp --supabase-key hoặc biến môi trường SUPABASE_KEY!")
            sys.exit(1)
        run_worker_cli(args.supabase_url, args.supabase_key)
    elif active_mode == "heartbeat":
        if not args.url or not args.token:
            print("❌ Lỗi: Cần cung cấp --url và --token cho chế độ heartbeat!")
            sys.exit(1)
        run_heartbeat(args.url, args.token, interval=args.interval)
    else:
        # Mặc định: Chạy Marimo App
        app.run()
