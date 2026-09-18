import marimo

__generated_with = "0.11.0"
app = marimo.App(
    width="full",
    app_title="🎬 Wan2.1 Video Studio & VRAM Profiler",
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
    from typing import Optional, Tuple

    import marimo as mo
    import torch
    from PIL import Image

    # Tối ưu CUDA Memory Allocator: Triệt tiêu phân mảnh bộ nhớ (Memory Fragmentation)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Kích hoạt PyTorch SDPA & FlashAttention
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass

    # Tự động đảm bảo các thư viện cần thiết đã cài đặt
    try:
        from diffusers import WanPipeline
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "accelerate", "transformers", "sentencepiece"])
        try:
            from diffusers import WanPipeline
        except ImportError:
            WanPipeline = None

    try:
        from diffusers.utils import export_to_video
    except ImportError:
        export_to_video = None

    try:
        import imageio_ffmpeg
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio[ffmpeg]"])
        try:
            import imageio_ffmpeg
        except ImportError:
            imageio_ffmpeg = None

    return (
        Image,
        Optional,
        Path,
        Tuple,
        WanPipeline,
        base64,
        export_to_video,
        gc,
        imageio_ffmpeg,
        io,
        mo,
        os,
        subprocess,
        sys,
        threading,
        time,
        torch,
    )


@app.cell
def _(mo, torch):
    def detect_hardware():
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            alloc_vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_vram_gb = torch.cuda.memory_reserved(0) / (1024**3)
            bf16_support = torch.cuda.is_bf16_supported()
            return {
                "type": "cuda",
                "name": dev_name,
                "total_vram": f"{total_vram_gb:.2f} GB",
                "free_vram": f"{total_vram_gb - alloc_vram_gb:.2f} GB",
                "allocated": f"{alloc_vram_gb:.2f} GB",
                "reserved": f"{reserved_vram_gb:.2f} GB",
                "bf16": bf16_support,
                "status": "🟢 GPU Online (Blackwell Ready)" if "Blackwell" in dev_name or "PRO 6000" in dev_name else "🟢 GPU Online",
            }
        return {
            "type": "cpu",
            "name": "CPU Fallback",
            "total_vram": "System RAM",
            "free_vram": "N/A",
            "allocated": "0 GB",
            "reserved": "0 GB",
            "bf16": False,
            "status": "⚠️ CPU Fallback",
        }

    hw_info = detect_hardware()

    header = mo.md(
        f"""
        # 🎬 Wan2.1 Video Studio & VRAM Profiler
        > **Kiến trúc:** Wan2.1 Diffusion Transformer 3D (UMT5-XXL + 30 DiT Blocks + 3D Causal VAE)

        | Phần cứng | Tên Thiết Bị | Tổng VRAM | Đã Cấp Phát | Khả Dụng | bfloat16 | Trạng Thái |
        | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
        | `{hw_info["type"].upper()}` | **{hw_info["name"]}** | **{hw_info["total_vram"]}** | `{hw_info["allocated"]}` | **{hw_info["free_vram"]}** | `{"Hỗ trợ" if hw_info["bf16"] else "Không"}` | {hw_info["status"]} |
        """
    )
    header
    return detect_hardware, header, hw_info


@app.cell
def _(mo, os):
    # =========================================================================
    # ⚙️ 1. CẤU HÌNH MÔ HÌNH WAN2.1
    # =========================================================================
    model_selector = mo.ui.dropdown(
        options={
            "Wan2.1-T2V-1.3B (Nhẹ, Siêu nhanh, ~3.6GB VRAM)": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "Wan2.1-T2V-14B (Flagship SOTA, Chi tiết khuôn mặt & điện ảnh cao, ~27.8GB VRAM)": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        value="Wan2.1-T2V-1.3B (Nhẹ, Siêu nhanh, ~3.6GB VRAM)",
        label="Phiên bản Wan2.1:",
    )

    vram_mode_selector = mo.ui.dropdown(
        options={
            "⚡ Host RAM Swap (Khuyến nghị: Text Encoder lưu trên CPU RAM, chỉ nạp GPU khi cần)": "ram_swap",
            "🚀 Full GPU (Nạp 100% VRAM, tốc độ cực đại nếu GPU >= 32GB)": "full_gpu",
            "💾 Model CPU Offload (Dành cho GPU dưới 16GB)": "cpu_offload",
        },
        value="⚡ Host RAM Swap (Khuyến nghị: Text Encoder lưu trên CPU RAM, chỉ nạp GPU khi cần)",
        label="Chiến lược tối ưu VRAM:",
    )

    # Hugging Face Token (Hỗ trợ tải tốc độ cao)
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    mo.md(
        f"""
        ### 🛠️ Cấu hình Phiên Bản Mô Hình & Chiến Lược VRAM
        {mo.as_html(model_selector)}
        {mo.as_html(vram_mode_selector)}
        """
    )

    return HF_TOKEN, model_selector, vram_mode_selector


@app.cell
def _(
    HF_TOKEN,
    WanPipeline,
    gc,
    hw_info,
    mo,
    model_selector,
    torch,
    vram_mode_selector,
):
    MODEL_ID = model_selector.value
    VRAM_MODE = vram_mode_selector.value
    print(f"⏳ Đang khởi tạo & nạp Wan2.1 ({MODEL_ID}) - Chế độ: {VRAM_MODE}...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _dtype = torch.bfloat16 if hw_info.get("bf16", False) else torch.float16

    with mo.status.spinner(title=f"Đang nạp {MODEL_ID} vào GPU ({_dtype})..."):
        try:
            _pipe = WanPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=_dtype,
                token=HF_TOKEN or None,
            )

            # Bật tiling và slicing cho VAE để tối ưu bộ nhớ decoder
            if hasattr(_pipe, "vae"):
                if hasattr(_pipe.vae, "enable_slicing"):
                    try:
                        _pipe.vae.enable_slicing()
                    except Exception:
                        pass
                if hasattr(_pipe.vae, "enable_tiling"):
                    try:
                        _pipe.vae.enable_tiling()
                    except Exception:
                        pass

            if VRAM_MODE == "ram_swap" and hw_info["type"] == "cuda":
                # Transformer và VAE lên CUDA, Text Encoder lưu trên CPU Host RAM
                _pipe.transformer.to("cuda")
                _pipe.vae.to("cuda")
                _pipe.text_encoder.to("cpu")
                _desc = f"⚡ Host RAM Swap trên {hw_info['name']} (Text Encoder trên CPU RAM, Transformer+VAE trên GPU: chỉ chiếm ~2.9GB VRAM tĩnh!)"
            elif VRAM_MODE == "full_gpu" and hw_info["type"] == "cuda":
                _pipe.to("cuda")
                _desc = f"100% trên {hw_info['name']} (Không offload, tốc độ cực đại)"
            elif VRAM_MODE == "cpu_offload" and hw_info["type"] == "cuda":
                _pipe.enable_model_cpu_offload()
                _desc = "Diffusers Sequential/Model CPU Offload kích hoạt"
            else:
                _pipe.to("cpu")
                _desc = "Chạy CPU"

            pipeline = _pipe
            print(f"✅ Nạp mô hình {MODEL_ID} thành công! ({_desc})")
            status_callout = mo.callout(
                f"✅ **Đã nạp thành công `{MODEL_ID}`** ({_dtype}). {_desc}",
                kind="success",
            )
        except Exception as e:
            pipeline = None
            print(f"❌ Lỗi nạp Wan2.1: {e}")
            status_callout = mo.callout(f"❌ **Lỗi nạp mô hình:** {e}", kind="danger")

    status_callout
    return pipeline, status_callout


@app.cell
def _(mo):
    # =========================================================================
    # 🎨 2. GIAO DIỆN ĐIỀU KHIỂN THÔNG SỐ SINH VIDEO (INTERACTIVE UI CONTROLS)
    # =========================================================================
    prompt_preset = mo.ui.dropdown(
        options={
            "☕ Đời Thường (Khuôn mặt, Cà phê & Cửa sổ mưa)": "A close-up cinematic shot of a young woman in a soft knitted cream sweater sitting by a sunlit wooden cafe table, gently holding a steaming ceramic mug with both hands, taking a slow delicate sip and looking out the rainy window with a calm natural smile, soft warm morning lighting, raindrops on window glass, shallow depth of field, photorealistic 8k, fluid realistic subtle motion",
            "🐉 Kỳ Ảo / Điện Ảnh (Rồng Hoàng Kim Tokyo Cyberpunk)": "A magnificent golden dragon soaring gracefully through misty neon clouds above a cyberpunk metropolis at dusk, cinematic camera panning, 4k ultra-detailed, photorealistic fluid motion, vivid reflections",
            "✍️ Tùy Chỉnh (Nhập thủ công bên dưới)": "",
        },
        value="☕ Đời Thường (Khuôn mặt, Cà phê & Cửa sổ mưa)",
        label="Chọn Prompt mẫu gợi ý:",
    )

    prompt_input = mo.ui.text_area(
        value="A close-up cinematic shot of a young woman in a soft knitted cream sweater sitting by a sunlit wooden cafe table, gently holding a steaming ceramic mug with both hands, taking a slow delicate sip and looking out the rainy window with a calm natural smile, soft warm morning lighting, raindrops on window glass, shallow depth of field, photorealistic 8k, fluid realistic subtle motion",
        label="Prompt (Mô tả video):",
        rows=3,
        full_width=True,
    )

    negative_prompt_input = mo.ui.text(
        value="",
        label="Negative Prompt (Để trống theo khuyến nghị chuẩn Wan2.1):",
        full_width=True,
    )

    resolution_select = mo.ui.dropdown(
        options={
            "832x480 (16:9 Ngang chuẩn 480p)": (832, 480),
            "1280x720 (16:9 HD)": (1280, 720),
            "1920x1080 (16:9 Full HD 1080p)": (1920, 1080),
            "3840x2160 (16:9 4K Ultra HD - Yêu cầu >=48GB VRAM)": (3840, 2160),
            "480x832 (9:16 Dọc TikTok/Reels)": (480, 832),
            "512x512 (1:1 Vuông Cân Bằng)": (512, 512),
            "640x360 (16:9 Nhẹ Tiết Kiệm)": (640, 360),
        },
        value="832x480 (16:9 Ngang chuẩn 480p)",
        label="Độ phân giải (Width x Height):",
    )

    frames_slider = mo.ui.slider(
        start=9,
        stop=81,
        step=8,
        value=17,
        label="Số khung hình (Frames, công thức 4n+1: 17=~1s, 33=~2s, 49=~3s, 65=~4s, 81=~5s):",
    )

    steps_slider = mo.ui.slider(
        start=10,
        stop=40,
        step=5,
        value=25,
        label="Số bước khử nhiễu (Denoising Steps):",
    )

    guidance_slider = mo.ui.slider(
        start=1.0,
        stop=9.0,
        step=0.5,
        value=5.0,
        label="Guidance Scale (CFG):",
    )

    seed_input = mo.ui.number(
        value=-1,
        label="Seed (-1 để chọn ngẫu nhiên):",
    )

    fps_select = mo.ui.dropdown(
        options=[8, 12, 16, 24, 30, 32, 60],
        value=16,
        label="FPS (Khung hình/giây):",
    )

    quality_select = mo.ui.dropdown(
        options={
            "H.264 Chuẩn (CRF 23, Tiết kiệm)": "standard",
            "H.264 Cao Nhất (CRF 17, Slow preset, High Profile)": "high_lossless",
        },
        value="H.264 Cao Nhất (CRF 17, Slow preset, High Profile)",
        label="Chất lượng H.264 Encoder:",
    )

    generate_btn = mo.ui.run_button(
        label="🚀 Bắt Đầu Sinh Video (Generate Wan2.1)",
        kind="success",
    )

    controls_panel = mo.vstack(
        [
            mo.md("### ⚙️ Bảng Điều Khiển Sinh Video"),
            prompt_preset,
            prompt_input,
            negative_prompt_input,
            mo.hstack([resolution_select, frames_slider, fps_select], justify="start", gap=2),
            mo.hstack([steps_slider, guidance_slider, seed_input, quality_select], justify="start", gap=2),
            generate_btn,
        ],
        gap=1,
    )
    controls_panel
    return (
        controls_panel,
        fps_select,
        frames_slider,
        generate_btn,
        guidance_slider,
        negative_prompt_input,
        prompt_input,
        prompt_preset,
        quality_select,
        resolution_select,
        seed_input,
        steps_slider,
    )


@app.cell
def _(
    VRAM_MODE,
    base64,
    export_to_video,
    fps_select,
    frames_slider,
    gc,
    generate_btn,
    guidance_slider,
    hw_info,
    mo,
    negative_prompt_input,
    pipeline,
    prompt_input,
    quality_select,
    resolution_select,
    seed_input,
    steps_slider,
    time,
    torch,
):
    # =========================================================================
    # 🎥 3. SINH VIDEO VÀ HIỂN THỊ TRỰC TIẾP
    # =========================================================================
    if not generate_btn.value:
        video_view = mo.callout("ℹ️ Hãy tùy chỉnh thông số ở trên và nhấn nút **Bắt Đầu Sinh Video** để thực thi.", kind="info")
    elif pipeline is None:
        video_view = mo.callout("⚠️ Pipeline chưa được nạp. Vui lòng kiểm tra lại Cell nạp mô hình!", kind="warn")
    else:
        width, height = resolution_select.value
        num_frames = frames_slider.value
        steps = steps_slider.value
        guidance = guidance_slider.value
        fps = fps_select.value
        raw_seed = seed_input.value
        quality_mode = quality_select.value

        # Dọn dẹp cache trước khi sinh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            vram_start_alloc = torch.cuda.memory_allocated() / (1024**3)
        else:
            vram_start_alloc = 0

        # Xác định seed ngẫu nhiên hoặc cố định
        active_seed = int(raw_seed) if raw_seed >= 0 else int(time.time() * 1000) % (2**31)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(active_seed)

        output_mp4 = f"wan21_{width}x{height}_{num_frames}f_seed{active_seed}.mp4"
        t_start = time.time()

        with mo.status.progress_bar(title=f"Đang sinh video Wan2.1 ({width}x{height}, {num_frames}f, {steps} steps)...") as bar:
            try:
                if VRAM_MODE == "ram_swap" and torch.cuda.is_available():
                    # 1. Fast Swap Text Encoder lên CUDA
                    pipeline.text_encoder.to("cuda")
                    with torch.inference_mode():
                        p_embeds, neg_embeds = pipeline.encode_prompt(
                            prompt=prompt_input.value,
                            negative_prompt=negative_prompt_input.value or None,
                            do_classifier_free_guidance=(guidance > 1.0),
                            device=torch.device("cuda"),
                            dtype=pipeline.transformer.dtype,
                        )
                    # 2. Swap ngay Text Encoder về Host CPU RAM (Giải phóng 10.58 GB VRAM!)
                    pipeline.text_encoder.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()

                    # 3. Denoising với precomputed embeddings
                    with torch.inference_mode():
                        pipe_output = pipeline(
                            prompt_embeds=p_embeds,
                            negative_prompt_embeds=neg_embeds,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            generator=generator,
                        )
                else:
                    with torch.inference_mode():
                        pipe_output = pipeline(
                            prompt=prompt_input.value,
                            negative_prompt=negative_prompt_input.value or None,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            generator=generator,
                        )
                video_frames = pipe_output.frames[0]
                t_infer = time.time() - t_start

                # Export sang MP4 chất lượng cao
                import imageio
                if quality_mode == "high_lossless":
                    writer = imageio.get_writer(
                        output_mp4,
                        fps=fps,
                        codec="libx264",
                        ffmpeg_params=[
                            "-crf", "17",
                            "-preset", "slow",
                            "-pix_fmt", "yuv420p",
                            "-profile:v", "high",
                            "-level", "5.2"
                        ]
                    )
                    for frame in video_frames:
                        writer.append_data((frame * 255).astype("uint8") if frame.dtype != "uint8" else frame)
                    writer.close()
                elif export_to_video is not None:
                    export_to_video(video_frames, output_mp4, fps=fps)
                else:
                    imageio.mimsave(output_mp4, video_frames, fps=fps)

                t_total = time.time() - t_start

                # Đọc video thành base64 để nhúng mượt mà
                with open(output_mp4, "rb") as vf:
                    video_bytes = vf.read()
                b64_vid = base64.b64encode(video_bytes).decode("ascii")

                # Đo đạc VRAM
                if torch.cuda.is_available():
                    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
                    vram_current = torch.cuda.memory_allocated() / (1024**3)
                    vram_res = torch.cuda.memory_reserved() / (1024**3)
                    vram_stat_str = f"VRAM Ban Đầu: `{vram_start_alloc:.2f} GB` | VRAM Đỉnh Điểm: **`{vram_peak:.2f} GB`** | VRAM Hiện Tại: `{vram_current:.2f} GB` (Reserved: `{vram_res:.2f} GB`)"
                else:
                    vram_stat_str = "Chạy trên CPU (Không đo VRAM)"

                # Gửi Marimo toast notification
                try:
                    mo.status.toast(
                        title="🎬 Wan2.1 Video Hoàn Tất!",
                        description=f"Thời gian: {t_total:.1f}s | {num_frames} frames @ {fps}fps | Kích thước: {width}x{height}",
                        kind="success",
                    )
                except Exception:
                    pass

                video_view = mo.vstack(
                    [
                        mo.md(
                            f"""
                            ### ✨ Kết Quả Video Sinh Bằng Wan2.1
                            - **File:** `{output_mp4}` ({len(video_bytes) / 1024:.1f} KB)
                            - **Thời gian sinh:** `{t_infer:.2f}s` (Inference) | `{t_total:.2f}s` (Toàn bộ)
                            - **Tốc độ:** `{num_frames / t_infer:.2f} frames/s` (`{steps / t_infer:.2f} steps/s`)
                            - **Thông số:** `{width}x{height}` | `{num_frames} frames` | `{fps} FPS` | Seed: `{active_seed}`
                            - **Bộ nhớ:** {vram_stat_str}
                            """
                        ),
                        mo.Html(
                            f"""
                            <div style="display: flex; justify-content: center; align-items: center; background: #0f172a; padding: 16px; border-radius: 12px; border: 1px solid #334155; margin-top: 8px;">
                                <video controls autoplay loop muted style="max-width: 100%; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
                                    <source src="data:video/mp4;base64,{b64_vid}" type="video/mp4">
                                    Trình duyệt không hỗ trợ phát video MP4.
                                </video>
                            </div>
                            """
                        ),
                    ],
                    gap=1,
                )
            except Exception as render_err:
                video_view = mo.callout(f"❌ **Lỗi trong quá trình sinh video:** {render_err}", kind="danger")

    video_view
    return (video_view,)


@app.cell
def _(mo, torch):
    # =========================================================================
    # 📊 4. CÔNG CỤ TRỰC QUAN HÓA BỘ NHỚ VRAM TỨC THỜI
    # =========================================================================
    refresh_vram_btn = mo.ui.run_button(label="🔄 Cập Nhật Trạng Thái VRAM GPU", kind="neutral")

    def get_vram_table():
        if not torch.cuda.is_available():
            return mo.md("_Hệ thống hiện không sử dụng CUDA GPU._")

        dev = 0
        total = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        alloc = torch.cuda.memory_allocated(dev) / (1024**3)
        reserved = torch.cuda.memory_reserved(dev) / (1024**3)
        peak = torch.cuda.max_memory_allocated(dev) / (1024**3)
        free = total - alloc

        return mo.md(
            f"""
            ### 📈 Trạng Thái VRAM Thời Điểm Hiện Tại
            | Chỉ số | Dung lượng | Tỷ lệ / Tổng VRAM |
            | :--- | :--- | :--- |
            | **Tổng VRAM Phần Cứng** | **{total:.2f} GB** | 100.0% |
            | **Đang Cấp Phát (Allocated)** | `{alloc:.2f} GB` | {alloc / total * 100:.1f}% |
            | **Bộ Nhớ Đệm PyTorch (Reserved)** | `{reserved:.2f} GB` | {reserved / total * 100:.1f}% |
            | **Đỉnh Điểm Lịch Sử (Peak Allocated)** | **`{peak:.2f} GB`** | {peak / total * 100:.1f}% |
            | **Dung Lượng Trống (Free)** | **`{free:.2f} GB`** | {free / total * 100:.1f}% |
            """
        )

    vram_display = mo.vstack([refresh_vram_btn, get_vram_table()], gap=1)
    vram_display
    return get_vram_table, refresh_vram_btn, vram_display


if __name__ == "__main__":
    app.run()
