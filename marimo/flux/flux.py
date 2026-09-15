import marimo

__generated_with = "0.11.0"
app = marimo.App(
    width="full",
    app_title="⚡ FLUX.1 Studio + Supabase AI Bridge",
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

    # Biến trạng thái worker toàn cục dùng chung trong notebook
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
        # ⚡ FLUX.1 Studio + Supabase AI Bridge
        > **Kiến trúc:** 12B Flow Matching Transformer + Supabase Database Broker (Miễn nhiễm ngắt kết nối 100%)

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
            _sel_model = model_choice.value
            _tok = hf_token.value.strip() or os.environ.get("HF_TOKEN") or None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _dtype = torch.bfloat16 if gpu_info.get("bf16", False) else torch.float32

            try:
                with mo.status.spinner(title=f"Đang nạp {_sel_model} vào VRAM ({_dtype})..."):
                    _pipe = FluxPipeline.from_pretrained(_sel_model, torch_dtype=_dtype, token=_tok)
                    if vram_mode.value == "full_gpu" and gpu_info["type"] == "cuda":
                        _pipe.to("cuda")
                        _desc = f"100% trên {gpu_info['name']} (Tối đa tốc độ)"
                    elif vram_mode.value == "cpu_offload":
                        _pipe.enable_model_cpu_offload()
                        _desc = "Model CPU Offload kích hoạt"
                    else:
                        _pipe.to("cpu")
                        _desc = "Chạy CPU"

                    pipeline = _pipe
                    status_banner = mo.callout(f"✅ Đã nạp `{_sel_model}` ({_dtype}). {_desc}", kind="success")
            except Exception as e:
                pipeline = _current_pipe
                status_banner = mo.callout(f"❌ Lỗi nạp mô hình: {e}", kind="danger")
    else:
        pipeline = _current_pipe
        status_banner = mo.callout("✅ Mô hình đã sẵn sàng trong VRAM.", kind="success")

    status_banner
    return (pipeline, status_banner)


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
            _p_text = prompt_input.value.strip()
            _w, _h = [int(x) for x in resolution_choice.value.split("x")]
            _s_val = int(seed_input.value)
            _cur_seed = _s_val if _s_val >= 0 else int(time.time() * 1000) % 2147483647
            _st = int(steps_slider.value)
            _gd = float(guidance_slider.value)

            _dev = gpu_info["type"]
            _gen = torch.Generator(device=_dev).manual_seed(_cur_seed)

            _t0 = time.time()
            with mo.status.spinner(title=f"Đang sinh ảnh FLUX.1 ({_st} steps, {_w}x{_h})..."):
                try:
                    _pipe_res = pipeline(
                        prompt=_p_text,
                        width=_w,
                        height=_h,
                        num_inference_steps=_st,
                        guidance_scale=_gd,
                        generator=_gen,
                        max_sequence_length=256 if _st <= 4 else 512,
                    )
                    _gen_img = _pipe_res.images[0]
                    _elapsed = time.time() - _t0
                    _gen_img.save(f"flux_{_cur_seed}.png")

                    gen_result_view = mo.vstack(
                        [
                            mo.md(f"### ✨ Hoàn tất trong `{_elapsed:.2f}s` | Kích thước: `{_w}x{_h}` | Seed: `{_cur_seed}`"),
                            mo.image(_gen_img),
                        ],
                        gap=1,
                    )
                except Exception as err:
                    gen_result_view = mo.callout(f"❌ Lỗi sinh ảnh: {err}", kind="danger")

    gen_result_view
    return (gen_result_view,)


@app.cell
def _(mo, os):
    DEFAULT_SUPABASE_URL = "https://fxepzlszglckfsscport.supabase.co"
    DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4ZXB6bHN6Z2xja2Zzc2Nwb3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NDEzMzksImV4cCI6MjEwNTAxNzMzOX0.28rS1waBYB8xvGgHR7utoek9PqBc3ev6HPOG9yo9RdQ"

    sb_url_ui = mo.ui.text(
        value=os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL),
        label="Supabase Project URL:",
    )
    sb_key_ui = mo.ui.text(
        value=os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY),
        label="Supabase API Key:",
        kind="password",
    )
    worker_toggle = mo.ui.switch(
        value=True,
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
def _(
    base64,
    create_client,
    gpu_info,
    io,
    mo,
    pipeline,
    sb_key_ui,
    sb_url_ui,
    sys,
    threading,
    time,
    torch,
    worker_state,
    worker_toggle,
):
    worker_status_ui = mo.md("*Gạt công tắc ở trên để bắt đầu lắng nghe hàng đợi Supabase.*")

    if worker_toggle.value:
        _url_val = sb_url_ui.value.strip()
        _key_val = sb_key_ui.value.strip()

        if not create_client:
            worker_status_ui = mo.callout("❌ Thiếu `supabase` package (`pip install supabase`).", kind="danger")
        elif not _key_val:
            worker_status_ui = mo.callout("⚠️ Vui lòng nhập Supabase API Key!", kind="warn")
        elif pipeline is None:
            worker_status_ui = mo.callout("⚠️ Vui lòng nạp mô hình ở Mục 1 trước khi bật Worker!", kind="warn")
        else:
            if not worker_state["running"]:
                worker_state["running"] = True

                def _run_worker_loop():
                    try:
                        _sb = create_client(_url_val, _key_val)
                        print("[WORKER] Đã kết nối Supabase thành công!")
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
                                    _steps_t = _job.get("steps") or 28
                                    _guidance_t = _job.get("guidance") or 3.5
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
                mo.md(f"- Endpoint: `{_url_val}` | Bảng: `image_jobs` | Thiết bị: `{gpu_info.get('name')}`"),
            ])
    else:
        worker_state["running"] = False
        worker_status_ui = mo.callout("⚪ Worker đang dừng.", kind="neutral")

    worker_status_ui
    return (worker_status_ui,)


if __name__ == "__main__":
    app.run()
