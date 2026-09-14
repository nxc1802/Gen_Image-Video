import marimo

__generated_with = "0.11.0"
app = marimo.App(
    width="full",
    app_title="FLUX.1 Studio — Marimo Lab",
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
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    return (
        BaseModel,
        CORSMiddleware,
        FastAPI,
        Field,
        FluxPipeline,
        Header,
        HTTPException,
        Image,
        Optional,
        Path,
        Request,
        StaticFiles,
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
        uvicorn,
    )


@app.cell
def _(torch):
    def detect_device_info():
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
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
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "device_type": "mps",
                "device_name": "Apple Silicon (MPS)",
                "total_vram": "Unified Memory",
                "free_vram": "Dynamic",
                "total_vram_num": 16.0,
                "cuda_version": "MPS Metal",
                "bf16_supported": True,
                "status_badge": "🍏 Apple Silicon",
            }
        else:
            return {
                "device_type": "cpu",
                "device_name": "CPU Only (Not recommended for Flux)",
                "total_vram": "System RAM",
                "free_vram": "N/A",
                "total_vram_num": 0.0,
                "cuda_version": "None",
                "bf16_supported": False,
                "status_badge": "⚠️ CPU Fallback",
            }

    system_info = detect_device_info()
    return detect_device_info, system_info


@app.cell
def _(mo, system_info):
    header = mo.md(
        f"""
        # ⚡ FLUX.1 Generation Studio on Marimo Lab
        > **Kiến trúc:** Flow Matching + 12B Multimodal Diffusion Transformer (MMDiT) + T5-XXL / CLIP-L + VAE
        
        | Phần cứng | Thiết bị | Tổng VRAM khả dụng | Độ chuẩn bfloat16 | Trạng thái |
        | :--- | :--- | :--- | :--- | :--- |
        | `{system_info["device_type"].upper()}` | **{system_info["device_name"]}** | **{system_info["total_vram"]}** | `{"Có hỗ trợ (Khuyên dùng)" if system_info["bf16_supported"] else "Không"}` | {system_info["status_badge"]} |
        """
    )
    return (header,)


@app.cell
def _(mo, os):
    # UI controls for model loading and VRAM offloading
    model_choice = mo.ui.dropdown(
        options={
            "FLUX.1-dev (28-Step Guidance, Chi tiết cao, Cần HF Token)": "black-forest-labs/FLUX.1-dev",
            "FLUX.1-schnell (4-Step Distilled, Apache 2.0, Nhanh, Không cần HF Token)": "black-forest-labs/FLUX.1-schnell",
        },
        value="black-forest-labs/FLUX.1-dev",
        label="Chọn biến thể mô hình FLUX.1:",
    )

    vram_mode = mo.ui.dropdown(
        options={
            "Auto (Tự động theo VRAM)": "auto",
            "Full GPU (Tốc độ tối đa, yêu cầu VRAM >= 28GB)": "full_gpu",
            "Model CPU Offload (Khuyên dùng cho VRAM 12GB - 24GB)": "cpu_offload",
            "Sequential CPU Offload (Tiết kiệm VRAM tối đa, ~8GB - 10GB)": "sequential_offload",
        },
        value="auto",
        label="Chiến lược tối ưu VRAM:",
    )

    hf_token = mo.ui.text(
        value=os.environ.get("HF_TOKEN", ""),
        placeholder="hf_xxxxxxxx (Tự động nhận diện từ biến môi trường)",
        label="Hugging Face Token (Gated Access):",
        kind="password",
    )

    load_btn = mo.ui.run_button(
        label="📥 Tải / Nạp Mô Hình Vào Bộ Nhớ",
        kind="success",
    )

    return hf_token, load_btn, model_choice, vram_mode


@app.cell
def _(hf_token, load_btn, mo, model_choice, vram_mode):
    model_settings_ui = mo.vstack(
        [
            mo.md("### ⚙️ 1. Cấu Hình & Tải Mô Hình"),
            mo.hstack([model_choice, vram_mode], gap=2, justify="start"),
            mo.hstack([hf_token, load_btn], gap=2, justify="start", align="end"),
        ],
        gap=1,
    )
    return (model_settings_ui,)


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
    # Model Loading Logic with auto-detection and persistence
    pipeline = None
    load_status = {
        "ok": False,
        "msg": "Mô hình chưa được nạp. Vui lòng bấm nút 'Tải / Nạp Mô Hình Vào Bộ Nhớ' để bắt đầu.",
    }

    if load_btn.value:
        selected_model = model_choice.value
        token_val = hf_token.value.strip() or os.environ.get("HF_TOKEN") or None

        # Dọn dẹp cache VRAM trước khi nạp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Chọn kiểu dữ liệu tối ưu: bfloat16 là chuẩn tốt nhất cho FLUX
        if system_info["device_type"] == "cuda":
            dtype = torch.bfloat16 if system_info["bf16_supported"] else torch.float16
        elif system_info["device_type"] == "mps":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        try:
            with mo.status.spinner(title=f"Đang nạp {selected_model} vào VRAM ({dtype})..."):
                pipe = FluxPipeline.from_pretrained(
                    selected_model,
                    torch_dtype=dtype,
                    token=token_val,
                )

                strategy = vram_mode.value
                total_vram_num = system_info.get("total_vram_num", 0.0)

                if strategy == "auto":
                    if system_info["device_type"] == "cuda":
                        if total_vram_num >= 28.0:
                            strategy = "full_gpu"
                        elif total_vram_num >= 14.0:
                            strategy = "cpu_offload"
                        else:
                            strategy = "sequential_offload"
                    elif system_info["device_type"] == "mps":
                        strategy = "full_gpu"
                    else:
                        strategy = "cpu_offload"

                # Áp dụng chiến lược phân bổ bộ nhớ
                if strategy == "full_gpu" and system_info["device_type"] == "cuda":
                    pipe.to("cuda")
                    mode_desc = "Toàn bộ mô hình chạy trực tiếp trên GPU (Max Throughput)."
                elif strategy == "full_gpu" and system_info["device_type"] == "mps":
                    pipe.to("mps")
                    mode_desc = "Chạy trên Apple Silicon MPS."
                elif strategy == "cpu_offload":
                    pipe.enable_model_cpu_offload()
                    mode_desc = "Đã kích hoạt Model CPU Offload (Tiết kiệm VRAM, phù hợp GPU 12GB - 24GB)."
                elif strategy == "sequential_offload":
                    pipe.enable_sequential_cpu_offload()
                    mode_desc = "Đã kích hoạt Sequential CPU Offload (Tiết kiệm VRAM tối đa, phù hợp GPU 8GB - 12GB)."
                else:
                    pipe.to("cpu")
                    mode_desc = "Chạy trên CPU."

                pipeline = pipe
                load_status = {
                    "ok": True,
                    "msg": f"✅ Tải thành công `{selected_model}` ({dtype}). {mode_desc}",
                }
        except Exception as e:
            load_status = {
                "ok": False,
                "msg": f"❌ Lỗi khi tải mô hình: {type(e).__name__} - {e}",
            }

    return load_status, pipeline


@app.cell
def _(load_status, mo):
    if load_status["ok"]:
        status_banner = mo.callout(load_status["msg"], kind="success")
    elif "Lỗi" in load_status["msg"] or "Thiếu" in load_status["msg"]:
        status_banner = mo.callout(load_status["msg"], kind="danger")
    else:
        status_banner = mo.callout(load_status["msg"], kind="info")
    return (status_banner,)


@app.cell
def _(mo, model_choice):
    # Dynamic defaults depending on model
    is_schnell_selected = "schnell" in model_choice.value

    prompt = mo.ui.text_area(
        value="A cinematic wide shot of a futuristic cyberpunk laboratory in Neo-Tokyo, neon reflections in rainwater, an advanced humanoid robot holding a glowing glass sphere with a miniature galaxy inside, 8k resolution, photorealistic, intricate mechanical details, text 'FLUX MARIMO' laser-engraved on the robotic chest plate",
        label="Nhập Prompt sinh ảnh (English mang lại chất lượng tốt nhất):",
        rows=3,
        full_width=True,
    )

    resolution = mo.ui.dropdown(
        options={
            "Vuông chuẩn (1024 x 1024) [Khuyên dùng]": (1024, 1024),
            "Khổ dọc điện thoại (768 x 1360)": (768, 1360),
            "Khổ ngang màn hình (1360 x 768)": (1360, 768),
            "Chân dung nghệ thuật (896 x 1152)": (896, 1152),
            "Phong cảnh rộng Cinematic (1280 x 720)": (1280, 720),
        },
        value="Vuông chuẩn (1024 x 1024) [Khuyên dùng]",
        label="Kích thước ảnh (Width x Height):",
    )

    steps = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=4 if is_schnell_selected else 28,
        label=f"Số bước khử nhiễu (Steps): {'[Schnell khuyên dùng: 4 bước]' if is_schnell_selected else '[Dev khuyên dùng: 28 bước]'}",
    )

    guidance = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.2,
        value=0.0 if is_schnell_selected else 3.5,
        label=f"Guidance Scale: {'[Schnell: 0.0 (guidance distilled)]' if is_schnell_selected else '[Dev: 3.5 (CFG scale)]'}",
    )

    seed = mo.ui.number(
        start=-1,
        stop=2147483647,
        value=-1,
        step=1,
        label="Seed (-1 để ngẫu nhiên mỗi lần):",
    )

    generate_btn = mo.ui.run_button(
        label="🎨 BẮT ĐẦU TẠO ẢNH",
        kind="warn",
    )

    return generate_btn, guidance, is_schnell_selected, prompt, resolution, seed, steps


@app.cell
def _(generate_btn, guidance, mo, prompt, resolution, seed, steps):
    prompt_settings_ui = mo.vstack(
        [
            mo.md("### 🎨 2. Prompt & Tham Số Sinh Ảnh"),
            prompt,
            mo.hstack([resolution, steps, guidance, seed], gap=2, justify="start"),
            mo.hstack([generate_btn], justify="start"),
        ],
        gap=1,
    )
    return (prompt_settings_ui,)


@app.cell
def _(
    generate_btn,
    guidance,
    mo,
    pipeline,
    prompt,
    resolution,
    seed,
    steps,
    system_info,
    time,
    torch,
):
    output_view = mo.md("*Chờ cấu hình. Sau khi nạp mô hình, bấm 'BẮT ĐẦU TẠO ẢNH' để render.*")

    if generate_btn.value:
        if pipeline is None:
            output_view = mo.callout(
                "⚠️ Vui lòng nạp mô hình trước bằng nút 'Tải / Nạp Mô Hình Vào Bộ Nhớ' ở mục 1.",
                kind="warn",
            )
        else:
            width, height = resolution.value
            prompt_text = prompt.value.strip()
            num_steps = steps.value
            guidance_scale = guidance.value

            # Xử lý Seed
            if seed.value != -1:
                active_seed = int(seed.value)
            else:
                active_seed = int(time.time() * 1000) % 2147483647

            # Thiết lập generator seed theo device
            dev_type = system_info["device_type"]
            gen_device = "cuda" if dev_type == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(active_seed)

            # FLUX.1 max sequence length (256 cho schnell, 512 cho dev)
            max_seq_len = 256 if num_steps <= 4 else 512

            start_time = time.time()
            with mo.status.spinner(title=f"Đang sinh ảnh FLUX.1 ({num_steps} steps, {width}x{height})..."):
                try:
                    result = pipeline(
                        prompt=prompt_text,
                        width=width,
                        height=height,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        max_sequence_length=max_seq_len,
                    )
                    generated_img = result.images[0]
                    inference_time = time.time() - start_time

                    # Lưu ảnh ra đĩa
                    filename = f"flux_{active_seed}_{width}x{height}.png"
                    generated_img.save(filename)

                    output_view = mo.vstack(
                        [
                            mo.md(
                                f"""
                                ### ✨ Kết Quả Sinh Ảnh Hoàn Thành!
                                - **Thời gian suy luận:** `{inference_time:.2f}s` (`{inference_time / num_steps:.2f}s` / step)
                                - **Kích thước:** `{width} x {height}` px
                                - **Seed:** `{active_seed}` | **Số bước:** `{num_steps}` | **Guidance Scale:** `{guidance_scale}`
                                - **File đã lưu tại server:** `{filename}`
                                """
                            ),
                            mo.image(generated_img),
                        ],
                        gap=1,
                    )
                except Exception as ex:
                    output_view = mo.callout(
                        f"❌ Lỗi trong quá trình suy luận: {type(ex).__name__} - {ex}",
                        kind="danger",
                    )

    return (output_view,)


@app.cell
def _(mo):
    # UI controls for OpenAI DALL-E Compatible Server
    api_port = mo.ui.number(
        value=8000,
        start=1000,
        stop=65535,
        step=1,
        label="Cổng Server API (Port):",
    )

    enable_tunnel = mo.ui.checkbox(
        value=True,
        label="Mở kết nối ra Internet qua Cloudflare Tunnel (HTTPS miễn phí)",
    )

    start_api_btn = mo.ui.run_button(
        label="🚀 Khởi Chạy / Restart OpenAI DALL-E Server",
        kind="success",
    )

    api_controls_card = mo.vstack(
        [
            mo.md(
                """
                ### 🌐 3. OpenAI DALL-E Compatible API Server
                > Cho phép bất kỳ ứng dụng nào hỗ trợ OpenAI API (OpenWebUI, LangChain, Cursor, Python SDK) gọi trực tiếp endpoint `POST /v1/images/generations` để sinh ảnh qua FLUX.1.
                """
            ),
            mo.hstack([api_port, enable_tunnel, start_api_btn], gap=2, align="end"),
        ],
        gap=1,
    )
    return api_controls_card, api_port, enable_tunnel, start_api_btn


@app.cell
def _(
    BaseModel,
    CORSMiddleware,
    FastAPI,
    Field,
    Header,
    HTTPException,
    Optional,
    Path,
    Request,
    StaticFiles,
    api_port,
    base64,
    enable_tunnel,
    io,
    mo,
    os,
    pipeline,
    re,
    start_api_btn,
    subprocess,
    threading,
    time,
    torch,
    uuid,
    uvicorn,
):
    api_view = mo.md("*Bấm 'Khởi Chạy OpenAI DALL-E Server' để kích hoạt API.*")

    if start_api_btn.value:
        port = int(api_port.value)
        images_dir = Path("/tmp/flux_api_images")
        images_dir.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("FLUX_API_KEY")
        if not api_key:
            api_key = "flux-" + base64.urlsafe_b64encode(os.urandom(24)).decode().rstrip("=")
            os.environ["FLUX_API_KEY"] = api_key

        api_fastapi = FastAPI(title="FLUX.1 OpenAI DALL-E API", version="1.0.0")
        api_fastapi.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        api_fastapi.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

        class ImgGenReq(BaseModel):
            prompt: str
            model: Optional[str] = "flux-1-dev"
            n: Optional[int] = 1
            size: Optional[str] = "1024x1024"
            response_format: Optional[str] = "b64_json"
            quality: Optional[str] = "standard"
            style: Optional[str] = None
            seed: Optional[int] = None
            steps: Optional[int] = None
            guidance: Optional[float] = None

        def auth_check(auth: Optional[str]):
            if auth != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API key.", "type": "invalid_request_error"}})

        @api_fastapi.get("/health")
        def health_ep():
            return {
                "status": "healthy",
                "model_loaded": pipeline is not None,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "vram_gb": round(torch.cuda.memory_allocated(0)/(1024**3), 2) if torch.cuda.is_available() else 0.0,
            }

        @api_fastapi.get("/v1/models")
        def models_ep(authorization: Optional[str] = Header(default=None)):
            auth_check(authorization)
            return {
                "object": "list",
                "data": [
                    {"id": "flux-1-dev", "object": "model", "created": int(time.time()), "owned_by": "black-forest-labs"},
                    {"id": "flux-1-schnell", "object": "model", "created": int(time.time()), "owned_by": "black-forest-labs"},
                    {"id": "dall-e-3", "object": "model", "created": int(time.time()), "owned_by": "openai"},
                ],
            }

        @api_fastapi.post("/v1/images/generations")
        def gen_ep(req: ImgGenReq, req_http: Request, authorization: Optional[str] = Header(default=None)):
            auth_check(authorization)
            if pipeline is None:
                raise HTTPException(status_code=503, detail={"error": {"message": "Model pipeline is not loaded on GPU yet."}})

            parts = (req.size or "1024x1024").lower().replace(" ", "").split("x")
            try:
                w, h = (int(parts[0]) // 16) * 16, (int(parts[1]) // 16) * 16
            except:
                w, h = 1024, 1024

            num_imgs = req.n or 1
            is_sch = "schnell" in (req.model or "").lower()
            stps = req.steps if req.steps is not None else (4 if is_sch else (35 if req.quality == "hd" else 28))
            gd = req.guidance if req.guidance is not None else (0.0 if is_sch else 3.5)

            data_list = []
            for i in range(num_imgs):
                s = (req.seed + i) if req.seed is not None else int(time.time()*1000 + i*997) % 2147483647
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                gen = torch.Generator(device=dev).manual_seed(s)

                res = pipeline(
                    prompt=req.prompt,
                    width=w,
                    height=h,
                    num_inference_steps=stps,
                    guidance_scale=gd,
                    generator=gen,
                    max_sequence_length=256 if stps <= 4 else 512,
                )
                img = res.images[0]
                item = {"revised_prompt": req.prompt}

                if req.response_format == "url":
                    fname = f"flux_{uuid.uuid4().hex}.png"
                    img.save(images_dir / fname)
                    base_u = str(req_http.base_url).rstrip("/")
                    item["url"] = f"{base_u}/images/{fname}"
                else:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    item["b64_json"] = base64.b64encode(buf.getvalue()).decode("ascii")

                data_list.append(item)

            return {"created": int(time.time()), "data": data_list}

        def run_uvicorn():
            uvicorn.run(api_fastapi, host="0.0.0.0", port=port, log_level="warning")

        th = threading.Thread(target=run_uvicorn, daemon=True)
        th.start()
        time.sleep(2)

        # Tunnel handling
        public_url = f"http://127.0.0.1:{port}"
        if enable_tunnel.value:
            c_bin = "/tmp/cloudflared" if os.path.exists("/tmp/cloudflared") else "cloudflared"
            try:
                proc = subprocess.Popen(
                    [c_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                deadline = time.time() + 20
                while time.time() < deadline:
                    l = proc.stdout.readline()
                    if not l: break
                    m = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", l)
                    if m:
                        public_url = m.group(0)
                        break
            except Exception as e:
                pass

        api_view = mo.vstack(
            [
                mo.callout("✅ OpenAI DALL-E API Server Đang Chạy!", kind="success"),
                mo.md(
                    f"""
                    #### 📡 Thông Tin Kết Nối API:
                    - **OpenAI Base URL:** `{public_url}/v1`
                    - **DALL-E Endpoint:** `{public_url}/v1/images/generations`
                    - **API Key:** `{api_key}`
                    
                    ```python
                    from openai import OpenAI

                    client = OpenAI(
                        base_url="{public_url}/v1",
                        api_key="{api_key}",
                    )

                    response = client.images.generate(
                        model="flux-1-dev",  # hoặc "dall-e-3", "flux-1-schnell"
                        prompt="A majestic lion made of golden clockwork gears, photorealistic 8k",
                        size="1024x1024",
                        response_format="b64_json",  # hoặc "url"
                    )
                    image_b64 = response.data[0].b64_json
                    ```
                    """
                ),
            ],
            gap=1,
        )

    return (api_view,)


@app.cell
def _(
    api_controls_card,
    api_view,
    header,
    model_settings_ui,
    output_view,
    prompt_settings_ui,
    status_banner,
):
    # Ghép layout hoàn chỉnh hiển thị trên Marimo Lab
    [
        header,
        model_settings_ui,
        status_banner,
        prompt_settings_ui,
        output_view,
        api_controls_card,
        api_view,
    ]
    return


if __name__ == "__main__":
    app.run()
