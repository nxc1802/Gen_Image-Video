import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="FLUX.1 Studio + Supabase AI Bridge — Marimo Lab",
)


@app.cell
def _():
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
    from typing import Optional

    import marimo as mo
    import torch
    from PIL import Image
    from diffusers import FluxPipeline

    return (
        FluxPipeline,
        Image,
        Optional,
        Path,
        base64,
        gc,
        io,
        mo,
        os,
        re,
        subprocess,
        threading,
        time,
        torch,
        uuid,
    )


@app.cell
def _(mo, torch):
    def detect_device_info():
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            free_vram_gb = total_vram_gb - allocated_gb
            bf16_ok = torch.cuda.is_bf16_supported()

            return {
                "device_type": "cuda",
                "device_name": device_name,
                "total_vram": f"{total_vram_gb:.1f} GB",
                "free_vram": f"{free_vram_gb:.1f} GB",
                "total_vram_num": total_vram_gb,
                "cuda_version": torch.version.cuda or "N/A",
                "bf16_supported": bf16_ok,
                "status_badge": "🟢 GPU Online",
            }
        else:
            return {
                "device_type": "cpu",
                "device_name": "CPU Fallback",
                "total_vram": "System RAM",
                "free_vram": "N/A",
                "total_vram_num": 0.0,
                "cuda_version": "None",
                "bf16_supported": False,
                "status_badge": "⚠️ CPU Fallback",
            }

    system_info = detect_device_info()

    header = mo.md(
        f"""
        # ⚡ FLUX.1 Studio + Supabase AI Bridge on Marimo Lab
        > **Kiến trúc:** 12B Flow Matching Transformer + Supabase Database-as-a-Broker (Chống Ngắt Kết Nối 100%)

        | Phần cứng | Thiết bị | Tổng VRAM khả dụng | bfloat16 | Trạng thái |
        | :--- | :--- | :--- | :--- | :--- |
        | `{system_info["device_type"].upper()}` | **{system_info["device_name"]}** | **{system_info["total_vram"]}** | `{"Có hỗ trợ (Tối ưu)" if system_info["bf16_supported"] else "Không"}` | {system_info["status_badge"]} |
        """
    )
    header
    return detect_device_info, header, system_info


@app.cell
def _(mo, os):
    model_choice = mo.ui.dropdown(
        options={
            "FLUX.1-dev (28-Step Guidance, Chi tiết cực cao, Cần HF Token)": "black-forest-labs/FLUX.1-dev",
            "FLUX.1-schnell (4-Step Distilled, Apache 2.0, Nhanh, Không cần HF Token)": "black-forest-labs/FLUX.1-schnell",
        },
        value="FLUX.1-dev (28-Step Guidance, Chi tiết cực cao, Cần HF Token)",
        label="Chọn biến thể mô hình FLUX.1:",
    )

    vram_mode = mo.ui.dropdown(
        options={
            "Full GPU (Tốc độ tối đa, yêu cầu VRAM >= 24GB)": "full_gpu",
            "Model CPU Offload (Tiết kiệm VRAM, 12GB - 24GB)": "cpu_offload",
        },
        value="Full GPU (Tốc độ tối đa, yêu cầu VRAM >= 24GB)",
        label="Chiến lược tối ưu VRAM:",
    )

    default_hf_token = os.environ.get("HF_TOKEN", "")

    hf_token = mo.ui.text(
        value=default_hf_token,
        placeholder="hf_xxxxxxxx (Gated access token)",
        label="Hugging Face Token (FLUX.1-dev):",
        kind="password",
    )

    load_btn = mo.ui.run_button(
        label="📥 Tải / Nạp Mô Hình Vào VRAM",
        kind="success",
    )

    model_settings_ui = mo.vstack(
        [
            mo.md("### ⚙️ 1. Cấu Hình & Nạp Mô Hình"),
            mo.hstack([model_choice, vram_mode], gap=2, justify="start"),
            mo.hstack([hf_token, load_btn], gap=2, justify="start", align="end"),
        ],
        gap=1,
    )
    model_settings_ui
    return default_hf_token, hf_token, load_btn, model_choice, model_settings_ui, vram_mode


@app.cell
def _(
    FluxPipeline,
    gc,
    hf_token,
    load_btn,
    mo,
    model_choice,
    os,
    system_info,
    torch,
    vram_mode,
):
    try:
        _current_pipe = pipeline
    except NameError:
        _current_pipe = None

    should_load = load_btn.value or (_current_pipe is None)

    if should_load:
        selected_model = model_choice.value
        token_val = hf_token.value.strip() or os.environ.get("HF_TOKEN") or None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dtype = torch.bfloat16 if system_info.get("bf16_supported", False) else torch.float32

        try:
            with mo.status.spinner(title=f"Đang nạp {selected_model} vào VRAM ({dtype})..."):
                pipe = FluxPipeline.from_pretrained(
                    selected_model,
                    torch_dtype=dtype,
                    token=token_val,
                )

                strategy = vram_mode.value
                if strategy == "full_gpu" and system_info["device_type"] == "cuda":
                    pipe.to("cuda")
                    mode_desc = f"Chạy trực tiếp 100% trên GPU {system_info['device_name']} (Tốc độ tối đa)."
                elif strategy == "cpu_offload":
                    pipe.enable_model_cpu_offload()
                    mode_desc = "Đã kích hoạt Model CPU Offload."
                else:
                    pipe.to("cpu")
                    mode_desc = "Chạy trên CPU."

                pipeline = pipe
                load_status = {
                    "ok": True,
                    "msg": f"✅ Tải thành công `{selected_model}` ({dtype}). {mode_desc}",
                }
        except Exception as e:
            pipeline = _current_pipe
            load_status = {
                "ok": False,
                "msg": f"❌ Lỗi khi nạp mô hình: {type(e).__name__} - {e}",
            }
    else:
        pipeline = _current_pipe
        load_status = {
            "ok": True,
            "msg": "✅ Mô hình đã nạp sẵn trong VRAM.",
        }

    if load_status["ok"]:
        status_banner = mo.callout(load_status["msg"], kind="success")
    elif "Lỗi" in load_status["msg"]:
        status_banner = mo.callout(load_status["msg"], kind="danger")
    else:
        status_banner = mo.callout(load_status["msg"], kind="info")
    status_banner
    return dtype, load_status, mode_desc, pipe, pipeline, selected_model, should_load, status_banner, strategy, token_val


@app.cell
def _(mo):
    prompt_input = mo.ui.text_area(
        value="A futuristic golden cyber dragon soaring through storm clouds, neon electric arcs, hyper-detailed 8k resolution, photorealistic, Unreal Engine 5 render",
        label="Nội dung Prompt mô tả ảnh:",
        rows=3,
    )

    resolution_choice = mo.ui.dropdown(
        options={
            "1024x1024 (Vuông 1:1)": "1024x1024",
            "768x1360 (Dọc 9:16)": "768x1360",
            "1360x768 (Ngang 16:9)": "1360x768",
            "1280x720 (Cinematic HD)": "1280x720",
        },
        value="1024x1024 (Vuông 1:1)",
        label="Tỷ lệ kích thước:",
    )

    seed_input = mo.ui.number(
        value=-1,
        start=-1,
        stop=2147483647,
        step=1,
        label="Seed (-1 để ngẫu nhiên):",
    )

    steps_slider = mo.ui.slider(
        start=1,
        stop=50,
        value=28,
        step=1,
        label="Số bước suy luận (Steps):",
    )

    guidance_slider = mo.ui.slider(
        start=0.0,
        stop=10.0,
        value=3.5,
        step=0.1,
        label="Guidance Scale:",
    )

    generate_btn = mo.ui.run_button(
        label="✨ Sinh Ảnh Ngay Trực Tiếp Trên Web",
        kind="warn",
    )

    generation_panel = mo.vstack(
        [
            mo.md("### 🎨 2. Sinh Ảnh Trực Tiếp Trên Giao Diện Marimo"),
            prompt_input,
            mo.hstack([resolution_choice, seed_input], gap=2),
            mo.hstack([steps_slider, guidance_slider], gap=2),
            generate_btn,
        ],
        gap=1,
    )
    generation_panel
    return (
        generate_btn,
        generation_panel,
        guidance_slider,
        prompt_input,
        resolution_choice,
        seed_input,
        steps_slider,
    )


@app.cell
def _(
    generate_btn,
    guidance_slider,
    mo,
    pipeline,
    prompt_input,
    resolution_choice,
    seed_input,
    steps_slider,
    system_info,
    time,
    torch,
):
    output_view = mo.md("*Bấm nút sinh ảnh ở trên để bắt đầu tạo ảnh...*")

    if generate_btn.value:
        if pipeline is None:
            output_view = mo.callout(
                "⚠️ Vui lòng nạp mô hình vào VRAM trước khi sinh ảnh!",
                kind="warn",
            )
        else:
            prompt_text = prompt_input.value.strip()
            res_str = resolution_choice.value
            w, h = [int(x) for x in res_str.split("x")]

            s_val = int(seed_input.value)
            active_seed = (
                s_val
                if s_val >= 0
                else int(time.time() * 1000) % 2147483647
            )

            num_steps = int(steps_slider.value)
            guidance_scale = float(guidance_slider.value)

            dev = system_info["device_type"]
            generator = torch.Generator(device=dev).manual_seed(active_seed)
            max_seq_len = 256 if num_steps <= 4 else 512

            start_time = time.time()
            with mo.status.spinner(title=f"Đang sinh ảnh FLUX.1 ({num_steps} steps, {w}x{h})..."):
                try:
                    result = pipeline(
                        prompt=prompt_text,
                        width=w,
                        height=h,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        max_sequence_length=max_seq_len,
                    )
                    img = result.images[0]
                    inference_time = time.time() - start_time

                    fname = f"flux_{active_seed}.png"
                    img.save(fname)

                    output_view = mo.vstack(
                        [
                            mo.md(
                                f"""
                                ### ✨ Sinh Ảnh Thành Công!
                                - **Thời gian suy luận:** `{inference_time:.2f}s` (`{inference_time / num_steps:.2f}s` / step)
                                - **Kích thước:** `{w} x {h}` px | **Seed:** `{active_seed}`
                                """
                            ),
                            mo.image(img),
                        ],
                        gap=1,
                    )
                except Exception as ex:
                    output_view = mo.callout(f"❌ Lỗi: {ex}", kind="danger")

    output_view
    return active_seed, dev, fname, generator, img, inference_time, max_seq_len, num_steps, output_view, prompt_text, res_str, s_val, start_time, w


@app.cell
def _(mo, os):
    supabase_url_ui = mo.ui.text(
        value=os.environ.get("SUPABASE_URL", "https://fxepzlszglckfsscport.supabase.co"),
        label="Supabase Project URL:",
    )

    supabase_key_ui = mo.ui.text(
        value=os.environ.get("SUPABASE_KEY", ""),
        placeholder="sbp_... hoặc eyJhbGciOi...",
        label="Supabase API Key (Anon hoặc Service Role):",
        kind="password",
    )

    worker_toggle = mo.ui.switch(
        value=False,
        label="Kích hoạt Background Queue Worker (Lắng nghe Supabase)",
    )

    supabase_panel = mo.vstack(
        [
            mo.md("### 🌉 3. Supabase GPU Queue Worker (Bridge Cho API OpenAI)"),
            mo.md(
                """
                > **Cơ chế an toàn 100%:** Worker chỉ gửi request **Outbound HTTPS (Port 443)** đến Supabase.
                > Không mở port, không dùng Cloudflare/SSH tunnel, không bao giờ bị bot Molab quét hay ngắt kết nối!
                """
            ),
            mo.hstack([supabase_url_ui, supabase_key_ui], gap=2),
            worker_toggle,
        ],
        gap=1,
    )
    supabase_panel
    return supabase_key_ui, supabase_panel, supabase_url_ui, worker_toggle


@app.cell
def _():
    _worker_state = {"running": False, "thread": None}
    return (_worker_state,)


@app.cell
def _(
    _worker_state,
    base64,
    io,
    mo,
    pipeline,
    supabase_key_ui,
    supabase_url_ui,
    system_info,
    threading,
    time,
    torch,
    worker_toggle,
):
    try:
        from supabase import create_client
    except ImportError:
        create_client = None

    worker_banner = mo.md("*Gạt công tắc ở trên để bắt đầu lắng nghe hàng đợi Supabase.*")

    if worker_toggle.value:
        sb_url = supabase_url_ui.value.strip()
        sb_key = supabase_key_ui.value.strip()

        if not create_client:
            worker_banner = mo.callout("❌ Thiếu thư viện supabase-py. Cài đặt bằng: `pip install supabase`", kind="danger")
        elif not sb_key:
            worker_banner = mo.callout("⚠️ Vui lòng nhập Supabase API Key để kích hoạt worker!", kind="warn")
        elif pipeline is None:
            worker_banner = mo.callout("⚠️ Vui lòng nạp mô hình FLUX.1 vào VRAM ở Mục 1 trước khi bật Worker!", kind="warn")
        else:
            if not _worker_state["running"]:
                _worker_state["running"] = True

                def _background_queue_worker():
                    try:
                        sb = create_client(sb_url, sb_key)
                        print("[MARIMO WORKER] Worker Supabase đã khởi động ngầm thành công!")
                    except Exception as conn_err:
                        print(f"[MARIMO WORKER] Không thể kết nối Supabase: {conn_err}")
                        return

                    while _worker_state["running"]:
                        try:
                            # Lấy 1 pending job
                            res = sb.table("image_jobs").select("*").eq("status", "pending").order("created_at").limit(1).execute()
                            if res.data and len(res.data) > 0:
                                j = res.data[0]
                                j_id = j["id"]

                                # Khóa job
                                lock = sb.table("image_jobs").update({"status": "processing"}).eq("id", j_id).eq("status", "pending").execute()
                                if lock.data:
                                    p = j.get("prompt", "")
                                    sz = j.get("size", "1024x1024")
                                    try:
                                        pw, ph = [int(x) for x in sz.split("x")]
                                    except Exception:
                                        pw, ph = 1024, 1024
                                    st = j.get("steps") or 28
                                    gd = j.get("guidance") or 3.5
                                    sd = j.get("seed") or (int(time.time() * 1000) % 2147483647)

                                    dev_w = system_info.get("device_type", "cuda")
                                    g_gen = torch.Generator(device=dev_w).manual_seed(sd)

                                    t0 = time.time()
                                    try:
                                        r_img = pipeline(
                                            prompt=p,
                                            width=pw,
                                            height=ph,
                                            num_inference_steps=st,
                                            guidance_scale=gd,
                                            generator=g_gen,
                                            max_sequence_length=256 if st <= 4 else 512,
                                        ).images[0]
                                        dt = time.time() - t0

                                        buf = io.BytesIO()
                                        r_img.save(buf, format="PNG")
                                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                                        sb.table("image_jobs").update({
                                            "status": "completed",
                                            "result_b64": b64,
                                            "inference_time_sec": round(dt, 2),
                                            "device_name": system_info.get("device_name", "GPU"),
                                        }).eq("id", j_id).execute()
                                        print(f"[MARIMO WORKER] Hoàn tất Job #{j_id} trong {dt:.2f}s!")
                                    except Exception as gen_err:
                                        sb.table("image_jobs").update({
                                            "status": "failed",
                                            "error_message": str(gen_err),
                                        }).eq("id", j_id).execute()
                                        print(f"[MARIMO WORKER] Lỗi sinh ảnh Job #{j_id}: {gen_err}")
                        except Exception as w_err:
                            print(f"[MARIMO WORKER ERROR] {w_err}")
                        time.sleep(1.0)

                worker_thread = threading.Thread(target=_background_queue_worker, daemon=True)
                _worker_state["thread"] = worker_thread
                worker_thread.start()

            worker_banner = mo.vstack([
                mo.callout("🟢 Worker Supabase ĐANG CHẠY NGẦM & LẮNG NGHE HÀNG ĐỢI!", kind="success"),
                mo.md(f"""
                - **Supabase Endpoint:** `{sb_url}`
                - **Hàng đợi bảng:** `image_jobs`
                - **Thiết bị xử lý:** `{system_info.get('device_name', 'GPU')}`
                - **Sẵn sàng nhận lệnh từ Local Bridge Server!**
                """)
            ])
    else:
        _worker_state["running"] = False
        worker_banner = mo.callout("⚪ Worker đang dừng. Gạt công tắc ở trên để bắt đầu lắng nghe Supabase.", kind="neutral")

    worker_banner
    return (
        create_client,
        sb_key,
        sb_url,
        worker_banner,
    )


if __name__ == "__main__":
    app.run()
