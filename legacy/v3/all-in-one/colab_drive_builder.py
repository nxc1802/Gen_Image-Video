#!/usr/bin/env python3
"""
🏗️ Google Colab CPU-Only Drive Builder (Models + Offline Wheels & Runtime Bundles)
================================================================================
Chạy trên một phiên Google Colab **CPU-only** (không tốn Compute Units của GPU A100/G4)
để tải sẵn toàn bộ trọng số mô hình (Models) và đóng gói sẵn thư viện/binary (Wheels + Runtime Bundles)
lưu trực tiếp vào Google Drive (`/content/drive/MyDrive/AI_Studio_Colab/`).

Tương tự cơ chế tạo Dataset trên Kaggle (`build_wheels_dataset.py` + `download_models_to_dataset.py`),
nhưng tận dụng Google Drive làm ổ đĩa bền vững giữa các phiên Colab:
- `/content/drive/MyDrive/AI_Studio_Colab/wheels/`
  ├── vlm_runtime_bundle.tar.bin         (llama-server b8763 CUDA + Noble GLIBC 2.39 + cloudflared + VLM wheels)
  ├── image_video_runtime_bundle.tar.bin (ComfyUI + ComfyUI-GGUF + VHS + cloudflared + Visual wheels)
  ├── speech_runtime_bundle.tar.bin      (coqui-tts + qwen-asr + qwen-tts + cloudflared + Audio wheels)
  └── embedding_runtime_bundle.tar.bin   (llama-server b8763 CUDA + Noble GLIBC 2.39 + Embedding wheels)
- `/content/drive/MyDrive/AI_Studio_Colab/models/`
  ├── vlm/         (Qwen3.8-27B-Q4_K_M.gguf, mmproj-Qwen3.8-27B-Q8_0.gguf, mtp-*, dflash-*)
  ├── image_video/ (Qwen-Image 2.1 GGUF + MiniMax H3 FL2VA/Ref2VA/Turbo/VAE/TextEncoder + Wan 2.2)
  ├── speech/      (Qwen3-ASR-1.7B, Qwen3-TTS-12Hz-1.7B-Base, XTTS-v2-vietnamse)
  └── embedding/   (10 GGUF models: Qwen3-Embedding & Reranker 0.6B/4B/8B/VL-2B/VL-8B)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_DRIVE_ROOT = os.environ.get("COLAB_DRIVE_ROOT", "/content/drive/MyDrive/AI_Studio_Colab")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd: str, check: bool = False) -> int:
    log(f"  $ {cmd}")
    res = subprocess.run(cmd, shell=True, check=check)
    return res.returncode


def ensure_build_tools() -> None:
    """Đảm bảo huggingface_hub và kaggle CLI có sẵn trên phiên CPU."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "kaggle", "pyyaml"],
        check=False,
    )
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq aria2 zstd pigz unzip >/dev/null 2>&1 || true",
        shell=True,
        check=False,
    )


def create_torch_constraints() -> str:
    """Ghi file constraint giữ nguyên bản torch/torchaudio/torchvision mặc định của Colab VM."""
    c_file = "/tmp/colab_torch_constraints.txt"
    lines = []
    for pkg in ("torch", "torchaudio", "torchvision"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "").split("+")[0]
            if ver:
                lines.append(f"{pkg}=={ver}")
        except Exception:
            pass
    Path(c_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"📌 Đã khóa ABI PyTorch của Colab: {', '.join(lines)}")
    return c_file


def _prepare_llama_cpp_and_noble_libs(stage_dir: Path) -> None:
    """Tải sẵn llama-server b8763 CUDA 12.4 + Ubuntu 24.04 Noble libs (GLIBC 2.39) + cloudflared."""
    bin_dir = stage_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    cf_path = bin_dir / "cloudflared"
    if not cf_path.exists():
        run_cmd(
            f"curl -sL --retry 3 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            f"-o {cf_path} && chmod +x {cf_path}"
        )

    llama_dir = stage_dir / "llama-cpp"
    llama_dir.mkdir(parents=True, exist_ok=True)
    if not (llama_dir / "llama-b8763" / "llama-server").exists():
        log("📥 Đang tải llama.cpp b8763 official Ubuntu x64 + CUDA 12.4 bundle...")
        llama_tar = "/tmp/llama-b8763-bin-ubuntu-x64.tar.gz"
        cudart_tar = "/tmp/cudart-llama-bin-linux-cuda-12.4-x64.tar.gz"
        run_cmd(
            f"aria2c -x 16 -s 16 -k 1M -q -d /tmp -o llama-b8763-bin-ubuntu-x64.tar.gz "
            f"'https://github.com/ggml-org/llama.cpp/releases/download/b8763/llama-b8763-bin-ubuntu-x64.tar.gz' || "
            f"curl -sL --retry 3 'https://github.com/ggml-org/llama.cpp/releases/download/b8763/llama-b8763-bin-ubuntu-x64.tar.gz' -o {llama_tar}"
        )
        run_cmd(
            f"aria2c -x 16 -s 16 -k 1M -q -d /tmp -o cudart-llama-bin-linux-cuda-12.4-x64.tar.gz "
            f"'https://github.com/ggml-org/llama.cpp/releases/download/b8763/cudart-llama-bin-linux-cuda-12.4-x64.tar.gz' || "
            f"curl -sL --retry 3 'https://github.com/ggml-org/llama.cpp/releases/download/b8763/cudart-llama-bin-linux-cuda-12.4-x64.tar.gz' -o {cudart_tar}"
        )
        if os.path.exists(llama_tar):
            run_cmd(f"tar -xzf {llama_tar} -C {llama_dir}")
        if os.path.exists(cudart_tar):
            run_cmd(f"tar -xzf {cudart_tar} -C {llama_dir}")
        run_cmd(f"chmod -R +x {llama_dir}")
        for so_f in glob.glob(str(llama_dir / "**" / "*.so*"), recursive=True):
            dest_so = llama_dir / "llama-b8763" / os.path.basename(so_f)
            if not dest_so.exists():
                try:
                    shutil.copy2(so_f, dest_so, follow_symlinks=False)
                except Exception:
                    pass

    noble_dir = stage_dir / "noble-libs"
    noble_dir.mkdir(parents=True, exist_ok=True)
    if not list(noble_dir.glob("**/ld-linux-x86-64.so.2")):
        log("📥 Đang tải thư viện GLIBC 2.39 & GLIBCXX 3.4.32 (Ubuntu 24.04 Noble) cho llama-server...")
        debs = [
            "http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8.4_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libstdc++6_14.2.0-4ubuntu2~24.04_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libgcc-s1_14.2.0-4ubuntu2~24.04_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libgomp1_14.2.0-4ubuntu2~24.04_amd64.deb",
        ]
        for idx, url in enumerate(debs):
            deb_file = f"/tmp/noble_{idx}.deb"
            run_cmd(f"curl -sL --retry 3 '{url}' -o {deb_file} && dpkg-deb -x {deb_file} {noble_dir} && rm -f {deb_file}")


# ==============================================================================
# 1. BUILD OFFLINE WHEELS & RUNTIME BUNDLES ON COLAB CPU -> GOOGLE DRIVE
# ==============================================================================
def build_wheels_to_drive(drive_wheels_dir: Path, targets: List[str], force: bool = False) -> None:
    """
    Xây dựng các gói `*_runtime_bundle.tar.bin` tương thích 100% với Python & PyTorch của Google Colab,
    lưu trực tiếp vào `/content/drive/MyDrive/AI_Studio_Colab/wheels/`.
    Khi mở phiên GPU (A100 / G4), chỉ cần giải nén trong ~5-10s là chạy ngay không cần `pip install` từ mạng.
    """
    drive_wheels_dir.mkdir(parents=True, exist_ok=True)
    c_file = create_torch_constraints()

    # 1. VLM Bundle
    if "all" in targets or "vlm" in targets:
        out_bundle = drive_wheels_dir / "vlm_runtime_bundle.tar.bin"
        if out_bundle.exists() and out_bundle.stat().st_size > 10_000_000 and not force:
            log(f"⏩ Đã có sẵn {out_bundle.name} ({out_bundle.stat().st_size / (1024**2):.1f} MB) trên Drive, bỏ qua.")
        else:
            log("📦 [1/4] Đang đóng gói VLM Runtime Bundle (llama-server CUDA + wheels) lên Google Drive...")
            stage = Path("/tmp/colab_stage_vlm")
            shutil.rmtree(stage, ignore_errors=True)
            (stage / "wheels").mkdir(parents=True, exist_ok=True)
            _prepare_llama_cpp_and_noble_libs(stage)
            run_cmd(
                f"{sys.executable} -m pip download -q -d {stage / 'wheels'} -c {c_file} "
                f"fastapi uvicorn python-multipart requests httpx sse-starlette pydantic pyyaml "
                f"accelerate bitsandbytes huggingface_hub pillow sentencepiece protobuf psutil"
            )
            tmp_tar = "/tmp/vlm_runtime_bundle.tar.bin"
            run_cmd(f"tar -cf {tmp_tar} -C {stage} .")
            shutil.copy2(tmp_tar, out_bundle)
            os.remove(tmp_tar)
            shutil.rmtree(stage, ignore_errors=True)
            log(f"✅ Đã lưu {out_bundle} ({out_bundle.stat().st_size / (1024**2):.1f} MB)")

    # 2. Image & Video Bundle (ComfyUI + Custom Nodes + Wheels)
    if "all" in targets or "image_video" in targets:
        out_bundle = drive_wheels_dir / "image_video_runtime_bundle.tar.bin"
        if out_bundle.exists() and out_bundle.stat().st_size > 10_000_000 and not force:
            log(f"⏩ Đã có sẵn {out_bundle.name} ({out_bundle.stat().st_size / (1024**2):.1f} MB) trên Drive, bỏ qua.")
        else:
            log("📦 [2/4] Đang đóng gói Image & Video Runtime Bundle (ComfyUI + GGUF + VHS + wheels) lên Google Drive...")
            stage = Path("/tmp/colab_stage_visual")
            shutil.rmtree(stage, ignore_errors=True)
            (stage / "wheels").mkdir(parents=True, exist_ok=True)
            (stage / "bin").mkdir(parents=True, exist_ok=True)

            cf_path = stage / "bin" / "cloudflared"
            run_cmd(
                f"curl -sL --retry 3 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
                f"-o {cf_path} && chmod +x {cf_path}"
            )

            comfy_dir = stage / "ComfyUI"
            run_cmd(f"git clone --depth 1 -q https://github.com/Comfy-Org/ComfyUI.git {comfy_dir}")
            run_cmd(
                f"git clone --depth 1 -q https://github.com/leejet/ComfyUI-GGUF.git "
                f"{comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF'}"
            )
            run_cmd(
                f"git clone --depth 1 -q https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "
                f"{comfy_dir / 'custom_nodes' / 'ComfyUI-VideoHelperSuite'}"
            )

            run_cmd(
                f"{sys.executable} -m pip download -q -d {stage / 'wheels'} -c {c_file} "
                f"-r {comfy_dir / 'requirements.txt'} "
                f"-r {comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF' / 'requirements.txt'} "
                f"-r {comfy_dir / 'custom_nodes' / 'ComfyUI-VideoHelperSuite' / 'requirements.txt'} "
                f"fastapi uvicorn python-multipart requests httpx sse-starlette pydantic pyyaml "
                f"gguf comfy-aimdo comfy-kitchen comfyui-frontend-package comfyui-workflow-templates "
                f"comfyui-embedded-docs av imageio-ffmpeg opencv-python-headless torchsde einops spandrel kornia scipy psutil"
            )
            tmp_tar = "/tmp/image_video_runtime_bundle.tar.bin"
            run_cmd(f"tar -cf {tmp_tar} -C {stage} .")
            shutil.copy2(tmp_tar, out_bundle)
            os.remove(tmp_tar)
            shutil.rmtree(stage, ignore_errors=True)
            log(f"✅ Đã lưu {out_bundle} ({out_bundle.stat().st_size / (1024**2):.1f} MB)")

    # 3. Speech Bundle
    if "all" in targets or "speech" in targets:
        out_bundle = drive_wheels_dir / "speech_runtime_bundle.tar.bin"
        if out_bundle.exists() and out_bundle.stat().st_size > 10_000_000 and not force:
            log(f"⏩ Đã có sẵn {out_bundle.name} ({out_bundle.stat().st_size / (1024**2):.1f} MB) trên Drive, bỏ qua.")
        else:
            log("📦 [3/4] Đang đóng gói Speech Runtime Bundle (Qwen3-ASR/TTS + XTTS-v2 wheels) lên Google Drive...")
            stage = Path("/tmp/colab_stage_speech")
            shutil.rmtree(stage, ignore_errors=True)
            (stage / "wheels").mkdir(parents=True, exist_ok=True)
            (stage / "bin").mkdir(parents=True, exist_ok=True)

            cf_path = stage / "bin" / "cloudflared"
            run_cmd(
                f"curl -sL --retry 3 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
                f"-o {cf_path} && chmod +x {cf_path}"
            )
            run_cmd(
                f"{sys.executable} -m pip download -q -d {stage / 'wheels'} -c {c_file} "
                f"fastapi uvicorn python-multipart requests httpx sse-starlette pydantic pyyaml "
                f"soundfile librosa scipy huggingface_hub accelerate transformers sentencepiece tiktoken "
                f"sox nagisa soynlp jieba einops onnxruntime psutil"
            )
            run_cmd(
                f"{sys.executable} -m pip download -q --no-deps -d {stage / 'wheels'} "
                f"qwen-asr qwen-tts qwen-omni-utils coqpit-config coqui-tts-trainer "
                f"ko-speech-tools monotonic-alignment-search pysbd num2words unidecode inflect anyascii pypinyin coqui-tts"
            )
            tmp_tar = "/tmp/speech_runtime_bundle.tar.bin"
            run_cmd(f"tar -cf {tmp_tar} -C {stage} .")
            shutil.copy2(tmp_tar, out_bundle)
            os.remove(tmp_tar)
            shutil.rmtree(stage, ignore_errors=True)
            log(f"✅ Đã lưu {out_bundle} ({out_bundle.stat().st_size / (1024**2):.1f} MB)")

    # 4. Embedding Bundle
    if "all" in targets or "embedding" in targets:
        out_bundle = drive_wheels_dir / "embedding_runtime_bundle.tar.bin"
        if out_bundle.exists() and out_bundle.stat().st_size > 10_000_000 and not force:
            log(f"⏩ Đã có sẵn {out_bundle.name} ({out_bundle.stat().st_size / (1024**2):.1f} MB) trên Drive, bỏ qua.")
        else:
            log("📦 [4/4] Đang đóng gói Embedding Runtime Bundle lên Google Drive...")
            stage = Path("/tmp/colab_stage_embedding")
            shutil.rmtree(stage, ignore_errors=True)
            (stage / "wheels").mkdir(parents=True, exist_ok=True)
            _prepare_llama_cpp_and_noble_libs(stage)
            run_cmd(
                f"{sys.executable} -m pip download -q -d {stage / 'wheels'} -c {c_file} "
                f"fastapi uvicorn python-multipart requests httpx sse-starlette pydantic pyyaml "
                f"huggingface_hub numpy pillow psutil"
            )
            tmp_tar = "/tmp/embedding_runtime_bundle.tar.bin"
            run_cmd(f"tar -cf {tmp_tar} -C {stage} .")
            shutil.copy2(tmp_tar, out_bundle)
            os.remove(tmp_tar)
            shutil.rmtree(stage, ignore_errors=True)
            log(f"✅ Đã lưu {out_bundle} ({out_bundle.stat().st_size / (1024**2):.1f} MB)")


def _drive_free_gb(drive_path: Path) -> float:
    """Kiểm tra dung lượng trống khả dụng trên Google Drive (GB)."""
    try:
        check_p = drive_path if drive_path.exists() else Path("/content/drive/MyDrive")
        return shutil.disk_usage(str(check_p)).free / (1024**3)
    except Exception:
        return 999.0


# ==============================================================================
# 2. DOWNLOAD MODELS TO GOOGLE DRIVE (/content/drive/MyDrive/AI_Studio_Colab/models)
# ==============================================================================
def _hf_download_single(repo_id: str, filename: str, dest_dir: Path, hf_token: Optional[str] = None) -> bool:
    """Tải 1 file từ Hugging Face qua ổ SSD tạm `/tmp/dl_stage` rồi chuyển vào `dest_dir` trên Google Drive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_file = dest_dir / Path(filename).name
    if target_file.exists() and target_file.stat().st_size > 1_000_000:
        log(f"  ⏩ Đã có sẵn trên Drive: {target_file.name} ({target_file.stat().st_size / (1024**3):.2f} GB)")
        return True

    free_gb = _drive_free_gb(dest_dir)
    if free_gb < 2.5:
        log(f"  ⚠️ Dung lượng Google Drive còn trống thấp ({free_gb:.2f} GB < 2.5 GB), tạm dừng tải thêm {filename} để bảo vệ Drive.")
        return False

    stage_dir = Path("/tmp/dl_stage_hf")
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    log(f"  ⬇️ Đang tải {filename} từ HF ({repo_id}) [Drive trống: {free_gb:.1f} GB] -> {target_file}...")
    try:
        from huggingface_hub import hf_hub_download
        dl_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(stage_dir),
            token=hf_token or os.environ.get("HF_TOKEN") or None,
        )
        dl_p = Path(dl_path)
        if dl_p.exists():
            shutil.copy2(str(dl_p), str(target_file))
        shutil.rmtree(stage_dir, ignore_errors=True)
        log(f"  ✅ Hoàn tất lưu lên Drive: {target_file.name} ({target_file.stat().st_size / (1024**3):.2f} GB)")
        return True
    except Exception as e:
        shutil.rmtree(stage_dir, ignore_errors=True)
        log(f"  ⚠️ Không thể tải {filename} từ {repo_id}: {e}")
        return False


def _kaggle_dataset_download(dataset_slug: str, dest_dir: Path, expected_files: Optional[List[str]] = None) -> bool:
    """Tải dataset từ Kaggle về ổ SSD `/tmp/dl_stage_kg`, giải nén và chuyển các file sang Google Drive."""
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    if expected_files and all((dest_dir / f).exists() and (dest_dir / f).stat().st_size > 1_000_000 for f in expected_files):
        log(f"  ⏩ Toàn bộ file từ Kaggle dataset '{dataset_slug}' đã có sẵn trong {dest_dir}.")
        return True

    free_gb = _drive_free_gb(dest_dir)
    if free_gb < 3.0:
        log(f"  ⚠️ Dung lượng Google Drive còn trống thấp ({free_gb:.2f} GB < 3.0 GB), bỏ qua tải '{dataset_slug}'.")
        return True

    stage_dir = Path("/tmp/dl_stage_kg")
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    log(f"  ⬇️ Đang tải Kaggle Dataset '{dataset_slug}' qua SSD tạm [Drive trống: {free_gb:.1f} GB] -> {dest_dir}...")
    rc = run_cmd(f"kaggle datasets download -d {dataset_slug} -p {stage_dir} --unzip -q")
    if rc == 0:
        for f in stage_dir.rglob("*"):
            if f.is_file() and not f.name.endswith(".zip"):
                target_f = dest_dir / f.name
                f_gb = f.stat().st_size / (1024**3)
                if _drive_free_gb(dest_dir) < f_gb + 1.5:
                    log(f"  ⚠️ Không đủ chỗ trống trên Drive cho {f.name} ({f_gb:.2f} GB), bỏ qua file này.")
                    continue
                log(f"  📂 Đang chép {f.name} ({f_gb:.2f} GB) sang Google Drive...")
                shutil.copy2(str(f), str(target_f))
        shutil.rmtree(stage_dir, ignore_errors=True)
        return True

    shutil.rmtree(stage_dir, ignore_errors=True)
    return False


def download_models_to_drive(
    drive_models_dir: Path,
    targets: List[str],
    include_wan22: bool = False,
) -> None:
    """Tải các mô hình được chỉ định vào `/content/drive/MyDrive/AI_Studio_Colab/models/`."""
    drive_models_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or None

    # 1. VLM Models (Qwen 3.8 27B GGUF + mmproj + MTP + DFlash)
    if "all" in targets or "vlm" in targets:
        log("👁️ [Models: VLM] Kiểm tra & tải Qwen 3.8 27B GGUF + Speculative Decoding (MTP / DFlash)...")
        vlm_dir = drive_models_dir / "vlm"
        vlm_files = [
            ("ggml-org/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf"),
            ("ggml-org/Qwen3.8-27B-GGUF", "mmproj-Qwen3.8-27B-Q8_0.gguf"),
            ("ggml-org/Qwen3.8-27B-GGUF", "mtp-Qwen3.8-27B-Q4_0.gguf"),
            ("ggml-org/Qwen3.8-27B-GGUF", "dflash-Qwen3.8-27B-Q4_0.gguf"),
        ]
        for repo, fname in vlm_files:
            _hf_download_single(repo, fname, vlm_dir, hf_token=hf_token)

    # 2. Image & Video Models (Qwen-Image 2.1 + MiniMax H3 FL2VA/Ref2VA/Turbo + Optional Wan 2.2)
    if "all" in targets or "image_video" in targets:
        log("🎬 [Models: Image & Video] Kiểm tra & tải Qwen-Image 2.1 + MiniMax H3...")
        vis_dir = drive_models_dir / "image_video"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 2a. Qwen-Image 2.1
        qwen_img_expected = [
            "qwen-image-2.1-Q4_K_M.gguf",
            "qwen3vl_8b_fp8_scaled.safetensors",
            "qwen_image_2.1_vae_bf16.safetensors",
        ]
        if not _kaggle_dataset_download("nguynxuncngde180528/qwen-image-2-1-gguf", vis_dir, qwen_img_expected):
            _hf_download_single("unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q4_K_M.gguf", vis_dir, hf_token)
            _hf_download_single(
                "Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
                vis_dir,
                hf_token,
            )
            _hf_download_single(
                "Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/vae/qwen_image_2.1_vae_bf16.safetensors",
                vis_dir,
                hf_token,
            )

        # 2b. MiniMax H3 (FL2VA + Ref2VA + Text Encoder + Video/Audio VAE + Turbo LoRA)
        minimax_kaggle_sources = [
            ("asdwddjksa/minimax-h3-diffusion", ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"]),
            ("asdwddjksa/minimax-h3-text-encoder", ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]),
            ("asdwddjksa/minimax-h3-assets", ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"]),
            ("wxujie/minimax-h3-ref2va", ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"]),
            ("fenchuan/h3-turbo-lora", ["minimax_h3_turbo_4step_pruned_comfyui.safetensors"]),
        ]
        for slug, exp_list in minimax_kaggle_sources:
            ok = _kaggle_dataset_download(slug, vis_dir, exp_list)
            if not ok:
                # Fallback qua Hugging Face nếu không cấu hình Kaggle API
                for exp_f in exp_list:
                    if not (vis_dir / exp_f).exists():
                        for hf_repo in [
                            "Comfy-Org/MiniMax-H3_ComfyUI",
                            "Kijai/MiniMax-H3-ComfyUI",
                            "city96/MiniMax-H3-gguf",
                        ]:
                            if _hf_download_single(hf_repo, exp_f, vis_dir, hf_token):
                                break

        # 2c. Tuỳ chọn Wan 2.2
        if include_wan22:
            log("🎞️ [Models: Wan 2.2] Đang tải thêm bộ trọng số Wan 2.2 GGUF...")
            wan_files = [
                ("QuantStack/Wan2.2-TI2V-5B-GGUF", "Q4_K_M/Wan2.2-TI2V-5B-Q4_K_M.gguf"),
                ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/Wan2.2_VAE.safetensors"),
                ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/Wan2.1_VAE.safetensors"),
                ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf"),
            ]
            for repo, fname in wan_files:
                _hf_download_single(repo, fname, vis_dir, hf_token)

    # 3. Speech Models (Qwen3-ASR 1.7B + Qwen3-TTS 1.7B + XTTS-v2 Vietnamese)
    if "all" in targets or "speech" in targets:
        log("🎙️ [Models: Speech] Kiểm tra & tải Qwen3-ASR 1.7B, Qwen3-TTS 1.7B và XTTS-v2 Vietnamese...")
        speech_dir = drive_models_dir / "speech"
        speech_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        speech_repos = [
            ("Qwen/Qwen3-ASR-1.7B", speech_dir / "qwen3-asr-1.7b"),
            ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", speech_dir / "qwen3-tts-1.7b"),
            ("thivux/XTTS-v2-vietnamse", speech_dir / "xtts-v2-vietnamese"),
        ]
        for repo_id, local_subdir in speech_repos:
            if (local_subdir / "config.json").exists():
                log(f"  ⏩ Đã có sẵn {repo_id} tại {local_subdir}")
                continue
            free_gb = _drive_free_gb(speech_dir)
            if free_gb < 5.0:
                log(f"  ⚠️ Drive còn trống {free_gb:.2f} GB (< 5.0 GB), bỏ qua snapshot {repo_id}.")
                continue
            stage_sp = Path("/tmp/dl_stage_speech")
            shutil.rmtree(stage_sp, ignore_errors=True)
            log(f"  ⬇️ Đang tải snapshot {repo_id} [Drive trống: {free_gb:.1f} GB] -> {local_subdir}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(stage_sp),
                    token=hf_token,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", ".git*"],
                )
                shutil.rmtree(stage_sp / ".cache", ignore_errors=True)
                shutil.copytree(str(stage_sp), str(local_subdir), dirs_exist_ok=True)
                shutil.rmtree(stage_sp, ignore_errors=True)
                log(f"  ✅ Hoàn tất snapshot {repo_id}")
            except Exception as e:
                shutil.rmtree(stage_sp, ignore_errors=True)
                log(f"  ⚠️ Lỗi tải {repo_id}: {e}")

    # 4. Embedding & Reranker Models (10 GGUF models)
    if "all" in targets or "embedding" in targets:
        log("🧬 [Models: Embedding & Rerank] Kiểm tra & tải 10 model GGUF Qwen3 Embedding/Reranker...")
        emb_dir = drive_models_dir / "embedding"
        emb_models = [
            ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q4_K_M.gguf", "Qwen3-Embedding-0.6B.Q4_K_M.gguf"),
            ("Mungert/Qwen3-Reranker-0.6B-GGUF", "Qwen3-Reranker-0.6B-q4_k_m.gguf", "Qwen3-Reranker-0.6B.Q4_K_M.gguf"),
            ("Qwen/Qwen3-Embedding-4B-GGUF", "Qwen3-Embedding-4B-Q4_K_M.gguf", "Qwen3-Embedding-4B.Q4_K_M.gguf"),
            ("Mungert/Qwen3-Reranker-4B-GGUF", "Qwen3-Reranker-4B-q4_k_m.gguf", "Qwen3-Reranker-4B.Q4_K_M.gguf"),
            ("Qwen/Qwen3-Embedding-8B-GGUF", "Qwen3-Embedding-8B-Q4_K_M.gguf", "Qwen3-Embedding-8B.Q4_K_M.gguf"),
            ("Mungert/Qwen3-Reranker-8B-GGUF", "Qwen3-Reranker-8B-q4_k_m.gguf", "Qwen3-Reranker-8B.Q4_K_M.gguf"),
            ("FlippyDora/Qwen3-VL-Embedding-2B-GGUF", "Qwen3-VL-Embedding-2B-Q4_K_M.gguf", "Qwen3-VL-Embedding-2B.Q4_K_M.gguf"),
            ("QuantStack/Qwen3-VL-Reranker-2B-GGUF", "Qwen3-VL-Reranker-2B-Q4_K_M.gguf", "Qwen3-VL-Reranker-2B.Q4_K_M.gguf"),
            ("FlippyDora/Qwen3-VL-Embedding-8B-GGUF", "Qwen3-VL-Embedding-8B-Q4_K_M.gguf", "Qwen3-VL-Embedding-8B.Q4_K_M.gguf"),
            ("QuantStack/Qwen3-VL-Reranker-8B-GGUF", "Qwen3-VL-Reranker-8B-Q4_K_M.gguf", "Qwen3-VL-Reranker-8B.Q4_K_M.gguf"),
        ]
        emb_dir.mkdir(parents=True, exist_ok=True)
        for repo_id, hf_fname, canonical_name in emb_models:
            canonical_path = emb_dir / canonical_name
            if canonical_path.exists() and canonical_path.stat().st_size > 1_000_000:
                log(f"  ⏩ Đã có sẵn: {canonical_name}")
                continue
            if _hf_download_single(repo_id, hf_fname, emb_dir, hf_token):
                dl_file = emb_dir / Path(hf_fname).name
                if dl_file.exists() and dl_file != canonical_path:
                    shutil.move(str(dl_file), str(canonical_path))


def print_drive_summary(drive_root: Path) -> None:
    """In bảng tổng kết dung lượng các file đã lưu trên Google Drive."""
    log("=" * 72)
    log(f"📊 TỔNG KẾT DỮ LIỆU TRÊN GOOGLE DRIVE: {drive_root}")
    log("=" * 72)
    if not drive_root.exists():
        log("⚠️ Thư mục chưa tồn tại.")
        return

    for sub in ["wheels", "models"]:
        sub_p = drive_root / sub
        if not sub_p.exists():
            continue
        files = [f for f in sub_p.rglob("*") if f.is_file()]
        total_gb = sum(f.stat().st_size for f in files) / (1024**3)
        log(f"📁 {sub.upper()} ({len(files)} files - Tổng: {total_gb:.2f} GB):")
        for f in sorted(files):
            rel = f.relative_to(sub_p)
            sz_mb = f.stat().st_size / (1024**2)
            unit = f"{sz_mb / 1024:.2f} GB" if sz_mb >= 1024 else f"{sz_mb:.1f} MB"
            log(f"   • {rel}: {unit}")
    log("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Google Colab CPU-Only Drive Builder (Tải Models & Đóng gói Wheels lưu vào Google Drive)"
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default=DEFAULT_DRIVE_ROOT,
        help=f"Đường dẫn thư mục gốc trên Google Drive (mặc định: {DEFAULT_DRIVE_ROOT})",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="vlm,image_video",
        help="Các phân hệ cần chuẩn bị, cách nhau bởi dấu phẩy: vlm,image_video,speech,embedding,all (mặc định: vlm,image_video)",
    )
    parser.add_argument(
        "--skip-wheels",
        action="store_true",
        help="Chỉ tải models, bỏ qua bước đóng gói wheels/runtime bundle",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Chỉ đóng gói wheels/runtime bundle, bỏ qua bước tải models",
    )
    parser.add_argument(
        "--include-wan22",
        action="store_true",
        help="Tải thêm trọng số Wan 2.2 bên cạnh Qwen-Image 2.1 & MiniMax H3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc đóng gói lại wheels bundle kể cả khi đã tồn tại trên Drive",
    )
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    if not Path("/content/drive").exists():
        log("⚠️ Cảnh báo: '/content/drive' chưa được mount! Hãy chạy `colab drivemount` trước khi chạy builder.")

    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    log(f"🚀 Bắt đầu chuẩn bị dữ liệu trên phiên CPU-Only -> {drive_root} (targets={targets})")
    ensure_build_tools()

    if not args.skip_wheels:
        build_wheels_to_drive(drive_root / "wheels", targets, force=args.force)

    if not args.skip_models:
        download_models_to_drive(drive_root / "models", targets, include_wan22=args.include_wan22)

    print_drive_summary(drive_root)
    log("🎉 Hoàn tất chuẩn bị Google Drive! Giờ bạn có thể tắt phiên CPU này và bật phiên GPU (A100 / G4).")


if __name__ == "__main__":
    main()
