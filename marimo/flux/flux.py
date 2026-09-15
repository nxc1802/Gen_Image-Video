import marimo

__generated_with = "0.11.0"
app = marimo.App(
    width="full",
    app_title="⚡ FLUX.1 Developer Studio + Supabase Queue Worker",
)


@app.cell
def _():
    import base64
    import gc
    import io
    import os
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path
    from typing import Optional

    import marimo as mo
    import torch
    from PIL import Image

    # Tự động cài đặt supabase nếu môi trường mới chưa có
    try:
        from supabase import create_client
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase"])
        try:
            from supabase import create_client
        except ImportError:
            create_client = None

    # Tự động cài đặt diffusers nếu môi trường mới chưa có
    try:
        from diffusers import FluxPipeline
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "sentencepiece", "protobuf"])
        try:
            from diffusers import FluxPipeline
        except ImportError:
            FluxPipeline = None

    worker_state = {"running": False, "thread": None}

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
        worker_state,
    )


@app.cell
def _(mo, torch):
    def detect_gpu():
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            alloc_vram = torch.cuda.memory_allocated(0) / (1024**3)
            return {
                "type": "cuda",
                "name": dev_name,
                "vram": f"{total_vram:.1f} GB",
                "free_vram": f"{total_vram - alloc_vram:.1f} GB",
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
        # ⚡ FLUX.1 Developer Studio + Supabase Queue Worker
        > **Kiến trúc:** 12B Flow Matching Transformer + Supabase Database Broker (Outbound HTTPS 443)

        | Phần cứng | Thiết bị | VRAM | bfloat16 | Trạng thái |
        | :--- | :--- | :--- | :--- | :--- |
        | `{gpu_info["type"].upper()}` | **{gpu_info["name"]}** | **{gpu_info["vram"]}** | `{"Hỗ trợ" if gpu_info["bf16"] else "Không"}` | {gpu_info["status"]} |
        """
    )
    header
    return detect_gpu, gpu_info, header


@app.cell
def _():
    # =========================================================================
    # ⚙️ 1. CẤU HÌNH MÔ HÌNH FLUX.1 (CHỈNH SỬA TRỰC TIẾP BIẾN TẠI ĐÂY)
    # =========================================================================
    # Biến thể: "black-forest-labs/FLUX.1-schnell" (4 steps, Apache 2.0, không cần token)
    #           "black-forest-labs/FLUX.1-dev"     (28 steps, chi tiết cao, cần token)
    MODEL_ID = "black-forest-labs/FLUX.1-schnell"

    # Chiến lược VRAM: "full_gpu" (nhanh nhất) hoặc "cpu_offload" (tiết kiệm VRAM)
    VRAM_MODE = "full_gpu"

    # Hugging Face Token (Chỉ bắt buộc nếu dùng bản gated FLUX.1-dev)
    HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Hoặc gõ trực tiếp token của bạn vào đây: "hf_..."


    return HF_TOKEN, MODEL_ID, VRAM_MODE


@app.cell
def _(
    FluxPipeline,
    HF_TOKEN,
    MODEL_ID,
    VRAM_MODE,
    gc,
    gpu_info,
    mo,
    torch,
):
    print(f"⏳ Đang nạp mô hình {MODEL_ID} ({VRAM_MODE})...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _dtype = torch.bfloat16 if gpu_info.get("bf16", False) else torch.float32

    with mo.status.spinner(title=f"Đang nạp {MODEL_ID} vào VRAM ({_dtype})..."):
        try:
            _pipe = FluxPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=_dtype,
                token=HF_TOKEN or None,
            )
            if VRAM_MODE == "full_gpu" and gpu_info["type"] == "cuda":
                _pipe.to("cuda")
                _desc = f"100% trên {gpu_info['name']} (Tối đa tốc độ)"
            elif VRAM_MODE == "cpu_offload":
                _pipe.enable_model_cpu_offload()
                _desc = "Model CPU Offload kích hoạt"
            else:
                _pipe.to("cpu")
                _desc = "Chạy CPU"

            pipeline = _pipe
            print(f"✅ Nạp mô hình {MODEL_ID} thành công!")
            status_banner = mo.callout(f"✅ Đã nạp `{MODEL_ID}` ({_dtype}). {_desc}", kind="success")
        except Exception as e:
            pipeline = None
            print(f"❌ Lỗi nạp mô hình: {e}")
            status_banner = mo.callout(f"❌ Lỗi nạp mô hình: {e}", kind="danger")

    status_banner
    return (pipeline, status_banner)


@app.cell
def _(gpu_info, mo, pipeline, time, torch):
    # =========================================================================
    # 🎨 2. SINH ẢNH TRỰC TIẾP TRONG NOTEBOOK (CHỈNH SỬA BIẾN TẠI ĐÂY)
    # =========================================================================
    PROMPT = "A futuristic cybernetic neon lion standing atop a misty skyscraper at night, cyberpunk aesthetic, photorealistic, 8k resolution, cinematic lighting"
    WIDTH = 1024
    HEIGHT = 1024
    STEPS = 4       # 4 cho schnell, 28 cho dev
    GUIDANCE = 0.0  # 0.0 cho schnell, 3.5 cho dev
    SEED = -1       # -1 để ngẫu nhiên mỗi lần chạy

    if pipeline is None:
        result_view = mo.callout("⚠️ Hãy nạp mô hình ở cell trên trước!", kind="warn")
    else:
        _active_seed = SEED if SEED >= 0 else int(time.time() * 1000) % 2147483647
        _dev = gpu_info["type"]
        _gen = torch.Generator(device=_dev).manual_seed(_active_seed)

        _t0 = time.time()
        with mo.status.spinner(title=f"Đang sinh ảnh FLUX ({STEPS} steps, {WIDTH}x{HEIGHT})..."):
            try:
                _pipe_res = pipeline(
                    prompt=PROMPT,
                    width=WIDTH,
                    height=HEIGHT,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=_gen,
                    max_sequence_length=256 if STEPS <= 4 else 512,
                )
                _img = _pipe_res.images[0]
                _elapsed = time.time() - _t0
                _img.save(f"flux_{_active_seed}.png")

                result_view = mo.vstack(
                    [
                        mo.md(f"### ✨ Hoàn tất trong `{_elapsed:.2f}s` | Kích thước: `{WIDTH}x{HEIGHT}` | Seed: `{_active_seed}`"),
                        mo.image(_img),
                    ],
                    gap=1,
                )
            except Exception as err:
                result_view = mo.callout(f"❌ Lỗi sinh ảnh: {err}", kind="danger")

    result_view
    return (
        GUIDANCE,
        HEIGHT,
        PROMPT,
        SEED,
        STEPS,
        WIDTH,
        result_view,
    )


@app.cell
def _():
    # =========================================================================
    # 🌉 3. CẤU HÌNH SUPABASE BROKER (CHỈNH SỬA BIẾN TẠI ĐÂY)
    # =========================================================================
    SUPABASE_URL = "https://fxepzlszglckfsscport.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4ZXB6bHN6Z2xja2Zzc2Nwb3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NDEzMzksImV4cCI6MjEwNTAxNzMzOX0.28rS1waBYB8xvGgHR7utoek9PqBc3ev6HPOG9yo9RdQ"
    ENABLE_WORKER = True

    return ENABLE_WORKER, SUPABASE_KEY, SUPABASE_URL


@app.cell
def _(
    ENABLE_WORKER,
    SUPABASE_KEY,
    SUPABASE_URL,
    base64,
    create_client,
    gpu_info,
    io,
    mo,
    pipeline,
    sys,
    threading,
    time,
    torch,
    worker_state,
):
    if ENABLE_WORKER:
        if not create_client:
            worker_status_ui = mo.callout("❌ Thiếu `supabase` package (`pip install supabase`).", kind="danger")
        elif not SUPABASE_KEY:
            worker_status_ui = mo.callout("⚠️ Vui lòng điền SUPABASE_KEY!", kind="warn")
        elif pipeline is None:
            worker_status_ui = mo.callout("⚠️ Vui lòng nạp mô hình ở cell trên trước khi bật Worker!", kind="warn")
        else:
            if not worker_state["running"]:
                worker_state["running"] = True

                def _run_worker_loop():
                    try:
                        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
                        print("[WORKER] Đã kết nối Supabase thành công! Lắng nghe hàng đợi...")
                    except Exception as ce:
                        print(f"[WORKER] Lỗi kết nối Supabase: {ce}")
                        return

                    while worker_state["running"]:
                        try:
                            _query_res = _sb.table("image_jobs").select("*").eq("status", "pending").order("created_at").limit(1).execute()
                            if _query_res.data and len(_query_res.data) > 0:
                                _job = _query_res.data[0]
                                _jid = _job["id"]
                                _lock_res = _sb.table("image_jobs").update({"status": "processing"}).eq("id", _jid).eq("status", "pending").execute()
                                if _lock_res.data:
                                    _prompt_t = _job.get("prompt", "")
                                    _size_t = _job.get("size", "1024x1024")
                                    try:
                                        _pw, _ph = [int(x) for x in _size_t.split("x")]
                                    except Exception:
                                        _pw, _ph = 1024, 1024
                                    _steps_t = _job.get("steps") or (4 if "schnell" in str(pipeline).lower() else 28)
                                    _guidance_t = _job.get("guidance") or (0.0 if "schnell" in str(pipeline).lower() else 3.5)
                                    _seed_t = _job.get("seed") or (int(time.time() * 1000) % 2147483647)

                                    _dev_t = gpu_info.get("type", "cuda")
                                    _g_t = torch.Generator(device=_dev_t).manual_seed(_seed_t)
                                    _start_inf = time.time()
                                    try:
                                        _img_res = pipeline(
                                            prompt=_prompt_t,
                                            width=_pw,
                                            height=_ph,
                                            num_inference_steps=_steps_t,
                                            guidance_scale=_guidance_t,
                                            generator=_g_t,
                                            max_sequence_length=256 if _steps_t <= 4 else 512,
                                        ).images[0]
                                        _inf_dt = time.time() - _start_inf
                                        _buf = io.BytesIO()
                                        _img_res.save(_buf, format="PNG")
                                        _b64_str = base64.b64encode(_buf.getvalue()).decode("ascii")

                                        _sb.table("image_jobs").update({
                                            "status": "completed",
                                            "result_b64": _b64_str,
                                            "inference_time_sec": round(_inf_dt, 2),
                                            "device_name": gpu_info.get("name", "GPU"),
                                        }).eq("id", _jid).execute()
                                        print(f"[WORKER] Xong Job #{_jid} trong {_inf_dt:.2f}s")
                                    except Exception as ge:
                                        _sb.table("image_jobs").update({
                                            "status": "failed",
                                            "error_message": str(ge),
                                        }).eq("id", _jid).execute()
                                        print(f"[WORKER] Lỗi Job #{_jid}: {ge}")
                        except Exception as loop_e:
                            print(f"[WORKER ERROR] {loop_e}")
                        time.sleep(1.0)

                _th = threading.Thread(target=_run_worker_loop, daemon=True)
                worker_state["thread"] = _th
                _th.start()

            worker_status_ui = mo.vstack([
                mo.callout("🟢 Worker ĐANG LẮNG NGHE HÀNG ĐỢI SUPABASE!", kind="success"),
                mo.md(f"- Endpoint: `{SUPABASE_URL}` | Bảng: `image_jobs` | Thiết bị: `{gpu_info.get('name')}`"),
            ])
    else:
        worker_state["running"] = False
        worker_status_ui = mo.callout("⚪ Worker đang dừng (Đặt ENABLE_WORKER = True để bật).", kind="neutral")

    worker_status_ui
    return (worker_status_ui,)


if __name__ == "__main__":
    app.run()
