#!/usr/bin/env python3
"""
🛠️ Kaggle 3-in-1 Specialized Runtime & Wheels Dataset Builder
================================================================================
Xây dựng và xuất bản 3 bộ Dataset Runtime/Wheels độc lập cho 3 Session:
1. `kaggle-speech-wheels`      -> Dùng cho `run_speech.ipynb` (Speech AI Studio)
2. `kaggle-vlm-wheels`         -> Dùng cho `run_vlm.ipynb` (VLM AI Studio)
3. `kaggle-image-video-wheels` -> Dùng cho `run_image_video.ipynb` (Image & Video AI Studio)

Mỗi dataset chứa:
- 1 file bundle `.tar.bin` (giữ nguyên symlink, quyền thực thi +x và không bị Kaggle tự giải nén)
- 1 file `manifest.json` ghi rõ danh sách các gói `.whl`, `.deb` và binary đi kèm.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from email.parser import Parser
from pathlib import Path


def setup_kaggle_credentials() -> str:
    """Thiết lập ~/.kaggle/kaggle.json từ biến môi trường hoặc Kaggle Secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        for k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN"]:
            try:
                val = secrets.get_secret(k)
                if val:
                    os.environ[k] = val
            except Exception:
                pass
    except Exception:
        pass

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    user = os.environ.get("KAGGLE_USERNAME", "cuongnguyen1802")
    key = os.environ.get("KAGGLE_KEY", "")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    with open(kaggle_json, "w", encoding="utf-8") as f:
        json.dump({"username": user, "key": key}, f)
    kaggle_json.chmod(0o600)
    print(f"🔑 Đã cấu hình Kaggle API cho user: {user}")
    return user


def check_dataset_exists(api, user: str, slug: str) -> bool:
    dataset_ref = f"{user}/{slug}"
    try:
        api.dataset_status(dataset_ref)
        return True
    except Exception:
        return False


def publish_dataset(api, user: str, slug: str, title: str, description: str, staging_dir: Path):
    """Tạo mới hoặc cập nhật Kaggle Dataset từ thư mục staging_dir."""
    dataset_ref = f"{user}/{slug}"
    meta_path = staging_dir / "dataset-metadata.json"
    meta_content = {
        "title": title,
        "id": dataset_ref,
        "licenses": [{"name": "apache-2.0"}],
        "description": description,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)

    print(f"\n⬆️ Đang xuất bản Kaggle Dataset: {dataset_ref} ...")
    t0 = time.time()
    exists = check_dataset_exists(api, user, slug)
    if not exists:
        resp = api.dataset_create_new(
            folder=str(staging_dir),
            public=True,
            quiet=False,
            dir_mode="skip",
        )
        print(f"🎉 Đã tạo mới dataset {dataset_ref}: {resp}")
    else:
        resp = api.dataset_create_version(
            folder=str(staging_dir),
            version_notes=f"Updated offline runtime bundle at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            quiet=False,
            dir_mode="skip",
        )
        print(f"🎉 Đã cập nhật version mới cho {dataset_ref}: {resp}")
    print(f"✅ Xuất bản {dataset_ref} hoàn tất trong {time.time() - t0:.1f}s!")


def collect_venv_delta_wheels(venv_dir: Path, pip_cache_dir: Path, wheels_out_dir: Path) -> list:
    """
    Đọc toàn bộ *.dist-info trong site-packages của venv để lấy danh sách chính xác
    các package (name==version) được cài thêm/nâng cấp so với base image của Kaggle,
    sau đó đóng gói thành file .whl vào wheels_out_dir.
    """
    wheels_out_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"pip", "setuptools", "wheel", "virtualenv", "torch", "torchaudio", "torchvision"}
    pkg_specs = []

    for dist_info in glob.glob(str(venv_dir / "lib" / "python*" / "site-packages" / "*.dist-info")):
        meta_file = Path(dist_info) / "METADATA"
        if not meta_file.exists():
            continue
        headers = Parser().parsestr(meta_file.read_text(encoding="utf-8", errors="ignore"))
        name = (headers.get("Name") or "").strip()
        version = (headers.get("Version") or "").strip()
        if not name or not version:
            continue
        if name.lower().replace("_", "-") in skip_names:
            continue
        pkg_specs.append(f"{name}=={version}")

    pkg_specs = sorted(set(pkg_specs))
    print(f"   🔍 Phát hiện {len(pkg_specs)} gói Python cần đóng gói wheel: {', '.join(pkg_specs[:12])}...")

    if pkg_specs:
        venv_pip = str(venv_dir / "bin" / "pip")
        cmd = [
            venv_pip,
            "wheel",
            "-q",
            "--no-deps",
            "--cache-dir",
            str(pip_cache_dir),
            "-w",
            str(wheels_out_dir),
            *pkg_specs,
        ]
        subprocess.run(cmd, check=True)

    wheel_files = sorted(f.name for f in wheels_out_dir.glob("*.whl"))
    total_mb = sum(f.stat().st_size for f in wheels_out_dir.glob("*.whl")) / (1024 ** 2)
    print(f"   ✅ Đã đóng gói {len(wheel_files)} file .whl ({total_mb:.1f} MB)")
    return wheel_files


def ensure_cloudflared_deb(dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    deb_path = dest_dir / "cloudflared-linux-amd64.deb"
    if not deb_path.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(deb_path),
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
            ],
            check=True,
        )
    return deb_path.name


def write_torch_constraints() -> Path:
    import torch
    import torchvision
    import torchaudio

    c_path = Path("/tmp/torch_constraints.txt")
    c_path.write_text(
        f"torch=={torch.__version__}\n"
        f"torchvision=={torchvision.__version__}\n"
        f"torchaudio=={torchaudio.__version__}\n",
        encoding="utf-8",
    )
    print(f"📌 Đã khóa constraint gốc của Kaggle: torch=={torch.__version__}, torchaudio=={torchaudio.__version__}")
    return c_path


# ==============================================================================
# 1. BUILD SPEECH WHEELS DATASET (`kaggle-speech-wheels`)
# ==============================================================================
def build_speech_dataset(api, user: str, constraints_path: Path):
    print("\n" + "=" * 76)
    print("🎙️ [1/3] XÂY DỰNG DATASET: kaggle-speech-wheels")
    print("=" * 76)
    t0 = time.time()

    work_dir = Path("/tmp/build_speech_rt")
    staging_dir = Path("/tmp/staging_speech_wheels")
    venv_dir = Path("/tmp/venv_speech")
    pip_cache = Path("/tmp/pip_cache_speech")
    for p in (work_dir, staging_dir, venv_dir):
        shutil.rmtree(p, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải các gói .deb của hệ thống âm thanh (sox, libsox-fmt-all, espeak-ng) + cloudflared
    print("📥 [Speech 1/4] Tải các gói .deb hệ thống (sox, libsox-fmt-all, espeak-ng, cloudflared)...")
    debs_dir = work_dir / "debs"
    debs_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run("apt-get update -qq && apt-get clean", shell=True, check=True)
    subprocess.run(
        "apt-get install -y -qq --download-only sox libsox-dev libsox-fmt-all espeak-ng",
        shell=True,
        check=True,
    )
    for deb in glob.glob("/var/cache/apt/archives/*.deb"):
        shutil.copy2(deb, debs_dir / os.path.basename(deb))
    ensure_cloudflared_deb(debs_dir)
    deb_files = sorted(f.name for f in debs_dir.glob("*.deb"))
    print(f"   ✅ Đã thu thập {len(deb_files)} gói .deb: {', '.join(deb_files[:8])}...")

    # 2. Tạo venv kế thừa base Kaggle và cài đặt bộ thư viện Speech chuẩn
    print("📦 [Speech 2/4] Cài đặt bộ thư viện Speech vào isolated venv để trích xuất đúng delta wheels...")
    subprocess.run([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(venv_dir)], check=True)
    venv_pip = str(venv_dir / "bin" / "pip")

    cmds = [
        f"{venv_pip} install -q --cache-dir {pip_cache} -c {constraints_path} fastapi uvicorn pyyaml soundfile librosa einops onnxruntime sox sentencepiece tiktoken nagisa soynlp jieba qwen-omni-utils",
        f"{venv_pip} install -q --cache-dir {pip_cache} --upgrade --no-deps qwen-asr qwen-tts",
        f"{venv_pip} install -q --cache-dir {pip_cache} --upgrade --no-deps coqpit-config coqui-tts-trainer ko-speech-tools monotonic-alignment-search pysbd num2words unidecode inflect anyascii pypinyin coqui-tts",
        f"{venv_pip} install -q --cache-dir {pip_cache} -c {constraints_path} 'transformers==4.57.6' 'huggingface-hub>=0.34.0,<1.0' 'accelerate>=1.12.0'",
    ]
    for c in cmds:
        subprocess.run(c, shell=True, check=True)

    # 3. Trích xuất toàn bộ delta wheels
    print("⚙️ [Speech 3/4] Đóng gói các file .whl vào bundle...")
    wheels_dir = work_dir / "wheels"
    wheel_files = collect_venv_delta_wheels(venv_dir, pip_cache, wheels_dir)

    # 4. Đóng gói thành speech_runtime_bundle.tar.bin và xuất bản
    bundle_path = staging_dir / "speech_runtime_bundle.tar.bin"
    print(f"🗜️ [Speech 4/4] Nén {bundle_path.name}...")
    subprocess.run(["tar", "-czf", str(bundle_path), "-C", str(work_dir), "debs", "wheels"], check=True)
    bundle_mb = bundle_path.stat().st_size / (1024 ** 2)
    print(f"   ✅ Kích thước {bundle_path.name}: {bundle_mb:.1f} MB (build trong {time.time() - t0:.1f}s)")

    manifest = {
        "studio_mode": "speech",
        "bundle_file": bundle_path.name,
        "bundle_size_mb": round(bundle_mb, 2),
        "deb_packages": deb_files,
        "wheel_packages": wheel_files,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    publish_dataset(
        api=api,
        user=user,
        slug="kaggle-speech-wheels",
        title="Kaggle Speech AI Studio Offline Runtime & Wheels",
        description="Offline runtime bundle (.whl + .deb) for Kaggle Speech AI Studio (Qwen3-ASR 1.7B, Qwen3-TTS 1.7B, XTTS-v2 Vietnamese).",
        staging_dir=staging_dir,
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(venv_dir, ignore_errors=True)
    shutil.rmtree(pip_cache, ignore_errors=True)


# ==============================================================================
# 2. BUILD VLM WHEELS DATASET (`kaggle-vlm-wheels`)
# ==============================================================================
def build_vlm_dataset(api, user: str, constraints_path: Path):
    print("\n" + "=" * 76)
    print("🧠 [2/3] XÂY DỰNG DATASET: kaggle-vlm-wheels")
    print("=" * 76)
    t0 = time.time()

    work_dir = Path("/tmp/build_vlm_rt")
    staging_dir = Path("/tmp/staging_vlm_wheels")
    venv_dir = Path("/tmp/venv_vlm")
    pip_cache = Path("/tmp/pip_cache_vlm")
    for p in (work_dir, staging_dir, venv_dir):
        shutil.rmtree(p, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải llama-server b11184 (CUDA 12.8) + cudart + Ubuntu 24.04 Noble runtime libs
    print("📥 [VLM 1/4] Tải và hợp nhất llama-server b11184 (CUDA 12.8) + cudart + noble-libs...")
    llama_dir = work_dir / "llama-cpp"
    noble_dir = work_dir / "noble-libs"
    llama_dir.mkdir(parents=True, exist_ok=True)
    noble_dir.mkdir(parents=True, exist_ok=True)

    dl_cmd = (
        f"(curl -sL --retry 3 https://github.com/ggml-org/llama.cpp/releases/download/b11184/llama-b11184-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz -C {llama_dir}) & "
        f"(curl -sL --retry 3 https://github.com/ggml-org/llama.cpp/releases/download/b11184/cudart-llama-b11184-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz -C {llama_dir}) & "
        f"(curl -sL --retry 3 -o /tmp/libc6.deb http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8_amd64.deb && dpkg-deb -x /tmp/libc6.deb {noble_dir}) & "
        f"(curl -sL --retry 3 -o /tmp/libstdcpp6.deb http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libstdc++6_14-20240412-0ubuntu1_amd64.deb && dpkg-deb -x /tmp/libstdcpp6.deb {noble_dir}) & "
        f"(curl -sL --retry 3 -o /tmp/libgcc.deb http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libgcc-s1_14-20240412-0ubuntu1_amd64.deb && dpkg-deb -x /tmp/libgcc.deb {noble_dir}) & "
        f"wait && chmod -R +x {llama_dir}"
    )
    subprocess.run(dl_cmd, shell=True, check=True)

    found_bins = glob.glob(f"{llama_dir}/**/llama-server", recursive=True)
    llama_bin = found_bins[0]
    bin_dir = os.path.dirname(os.path.abspath(llama_bin))

    # Gom .so* trong llama_dir về bin_dir (dùng move để tránh nhân đôi dung lượng)
    for so_f in glob.glob(f"{llama_dir}/**/*.so*", recursive=True):
        if os.path.dirname(os.path.abspath(so_f)) != bin_dir:
            dst = os.path.join(bin_dir, os.path.basename(so_f))
            if not os.path.exists(dst):
                try:
                    shutil.move(so_f, dst)
                except Exception:
                    pass

    noble_lds = glob.glob(f"{noble_dir}/**/ld-linux-x86-64.so.2", recursive=True)
    noble_ld = os.path.realpath(noble_lds[0])
    local_ld = os.path.join(bin_dir, "ld-linux-x86-64.so.2")
    if not os.path.exists(local_ld):
        shutil.copy2(noble_ld, local_ld)
        os.chmod(local_ld, 0o755)

    lib_path = f"{bin_dir}:{noble_dir}/lib/x86_64-linux-gnu:{noble_dir}/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
    ver_check = subprocess.run(
        [local_ld, "--library-path", lib_path, llama_bin, "--version"],
        cwd=bin_dir,
        capture_output=True,
        text=True,
    )
    print(f"   ✅ Kiểm tra llama-server binary (code={ver_check.returncode}): {(ver_check.stdout or ver_check.stderr).strip()}")

    # 2. Tải cloudflared.deb
    debs_dir = work_dir / "debs"
    ensure_cloudflared_deb(debs_dir)

    # 3. Đóng gói các wheel API nhẹ cho VLM
    print("📦 [VLM 2/4] Đóng gói các wheel phụ trợ API cho VLM...")
    subprocess.run([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(venv_dir)], check=True)
    venv_pip = str(venv_dir / "bin" / "pip")
    subprocess.run(
        f"{venv_pip} install -q --cache-dir {pip_cache} -c {constraints_path} fastapi uvicorn pyyaml python-multipart requests pillow psutil",
        shell=True,
        check=True,
    )
    wheels_dir = work_dir / "wheels"
    wheel_files = collect_venv_delta_wheels(venv_dir, pip_cache, wheels_dir)
    subprocess.run(
        [venv_pip, "wheel", "-q", "--no-deps", "-w", str(wheels_dir), "fastapi", "uvicorn", "python-multipart", "pyyaml"],
        check=True,
    )
    wheel_files = sorted(f.name for f in wheels_dir.glob("*.whl"))

    # 4. Nén thành vlm_runtime_bundle.tar.bin (dùng gzip -1 để nén nhanh) và xuất bản
    bundle_path = staging_dir / "vlm_runtime_bundle.tar.bin"
    print(f"🗜️ [VLM 3/4] Nén {bundle_path.name}...")
    subprocess.run(
        f"tar -cf - -C {work_dir} llama-cpp noble-libs debs wheels | gzip -1 > {bundle_path}",
        shell=True,
        check=True,
    )
    bundle_mb = bundle_path.stat().st_size / (1024 ** 2)
    print(f"   ✅ Kích thước {bundle_path.name}: {bundle_mb:.1f} MB (build trong {time.time() - t0:.1f}s)")

    manifest = {
        "studio_mode": "vlm",
        "bundle_file": bundle_path.name,
        "bundle_size_mb": round(bundle_mb, 2),
        "llama_server_build": "b11184-ubuntu-cuda-12.8-x64",
        "wheel_packages": wheel_files,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    publish_dataset(
        api=api,
        user=user,
        slug="kaggle-vlm-wheels",
        title="Kaggle VLM Studio Offline Runtime Wheels",
        description="Offline runtime bundle for Kaggle VLM AI Studio containing llama-server b11184 CUDA 12.8, cudart, Ubuntu 24.04 Noble glibc/libstdc++ runtime, cloudflared, and FastAPI wheels.",
        staging_dir=staging_dir,
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(venv_dir, ignore_errors=True)
    shutil.rmtree(pip_cache, ignore_errors=True)


# ==============================================================================
# 3. BUILD IMAGE & VIDEO WHEELS DATASET (`kaggle-image-video-wheels`)
# ==============================================================================
def build_image_video_dataset(api, user: str, constraints_path: Path):
    print("\n" + "=" * 76)
    print("🖼️🎬 [3/3] XÂY DỰNG DATASET: kaggle-image-video-wheels")
    print("=" * 76)
    t0 = time.time()

    work_dir = Path("/tmp/build_imgvid_rt")
    staging_dir = Path("/tmp/staging_imgvid_wheels")
    venv_dir = Path("/tmp/venv_imgvid")
    pip_cache = Path("/tmp/pip_cache_imgvid")
    for p in (work_dir, staging_dir, venv_dir):
        shutil.rmtree(p, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clone ComfyUI + leejet/ComfyUI-GGUF + ComfyUI-VideoHelperSuite
    print("📥 [Image/Video 1/4] Clone ComfyUI + leejet/ComfyUI-GGUF + VideoHelperSuite...")
    comfy_dir = work_dir / "ComfyUI"
    custom_nodes = comfy_dir / "custom_nodes"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(comfy_dir)],
        check=True,
    )
    custom_nodes.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            "https://github.com/leejet/ComfyUI-GGUF.git",
            str(custom_nodes / "ComfyUI-GGUF"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
            str(custom_nodes / "ComfyUI-VideoHelperSuite"),
        ],
        check=True,
    )
    # Xóa thư mục .git để giảm dung lượng archive và tăng tốc giải nén
    for git_dir in comfy_dir.rglob(".git"):
        if git_dir.is_dir():
            shutil.rmtree(git_dir, ignore_errors=True)

    # 2. Tải cloudflared.deb
    debs_dir = work_dir / "debs"
    ensure_cloudflared_deb(debs_dir)

    # 3. Cài đặt dependencies của ComfyUI + GGUF + VHS vào isolated venv và trích xuất delta wheels
    print("📦 [Image/Video 2/4] Cài đặt dependencies ComfyUI vào isolated venv...")
    subprocess.run([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(venv_dir)], check=True)
    venv_pip = str(venv_dir / "bin" / "pip")

    pip_cmd = (
        f"{venv_pip} install -q --cache-dir {pip_cache} -c {constraints_path} "
        f"-r {comfy_dir}/requirements.txt "
        f"-r {custom_nodes}/ComfyUI-GGUF/requirements.txt "
        f"-r {custom_nodes}/ComfyUI-VideoHelperSuite/requirements.txt "
        "comfy-aimdo comfy-kitchen av fastapi uvicorn pydantic pyyaml"
    )
    subprocess.run(pip_cmd, shell=True, check=True)

    print("⚙️ [Image/Video 3/4] Đóng gói các file .whl của ComfyUI...")
    wheels_dir = work_dir / "wheels"
    wheel_files = collect_venv_delta_wheels(venv_dir, pip_cache, wheels_dir)

    # 4. Nén thành image_video_runtime_bundle.tar.bin và xuất bản
    bundle_path = staging_dir / "image_video_runtime_bundle.tar.bin"
    print(f"🗜️ [Image/Video 4/4] Nén {bundle_path.name}...")
    subprocess.run(
        f"tar -cf - -C {work_dir} ComfyUI debs wheels | gzip -1 > {bundle_path}",
        shell=True,
        check=True,
    )
    bundle_mb = bundle_path.stat().st_size / (1024 ** 2)
    print(f"   ✅ Kích thước {bundle_path.name}: {bundle_mb:.1f} MB (build trong {time.time() - t0:.1f}s)")

    manifest = {
        "studio_mode": "image_video",
        "bundle_file": bundle_path.name,
        "bundle_size_mb": round(bundle_mb, 2),
        "comfyui_repos": [
            "Comfy-Org/ComfyUI",
            "leejet/ComfyUI-GGUF (qwen_image21 + wan)",
            "Kosinkadink/ComfyUI-VideoHelperSuite",
        ],
        "wheel_packages": wheel_files,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    publish_dataset(
        api=api,
        user=user,
        slug="kaggle-image-video-wheels",
        title="Kaggle Image Video Studio Offline Wheels",
        description="Offline runtime bundle for Kaggle Image & Video AI Studio containing ComfyUI, leejet/ComfyUI-GGUF (Qwen-Image 2.1 + Wan2.2), VideoHelperSuite, cloudflared, and all required Python wheels.",
        staging_dir=staging_dir,
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(venv_dir, ignore_errors=True)
    shutil.rmtree(pip_cache, ignore_errors=True)


def build_embedding_dataset(api, user: str, constraints_path: Path):
    """
    Xây dựng Kaggle Dataset: kaggle-embedding-wheels
    Chứa llama-server chính thức + thư viện bổ trợ cho Embedding & Rerank Studio.
    """
    t0 = time.time()
    print("\n" + "=" * 76)
    print("🧬 [Embedding 1/4] XÂY DỰNG DATASET: kaggle-embedding-wheels")
    print("=" * 76)

    work_dir = Path("/tmp/build_embedding_runtime")
    staging_dir = Path("/tmp/staging_embedding_wheels")
    venv_dir = Path("/tmp/venv_embedding_builder")
    pip_cache = Path("/tmp/pip_cache_embedding")

    for p in [work_dir, staging_dir, venv_dir, pip_cache]:
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)

    llama_dir = work_dir / "llama-cpp"
    noble_dir = work_dir / "noble-libs"
    llama_dir.mkdir(parents=True, exist_ok=True)
    noble_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải llama-server binary chính thức từ ggml-org/llama.cpp
    print("⬇️ [Embedding 1/4] Tải llama-server binary (b11184+)...")
    dl_cmd = (
        f"curl -sL --retry 3 https://github.com/ggml-org/llama.cpp/releases/download/b11184/llama-b11184-bin-ubuntu-x64.zip -o /tmp/llama_bin.zip && "
        f"unzip -q -o /tmp/llama_bin.zip -d {llama_dir} && "
        f"(curl -sL --retry 3 -o /tmp/libc6.deb http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8_amd64.deb && dpkg-deb -x /tmp/libc6.deb {noble_dir}) & "
        f"(curl -sL --retry 3 -o /tmp/libstdcpp6.deb http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libstdc++6_14-20240412-0ubuntu1_amd64.deb && dpkg-deb -x /tmp/libstdcpp6.deb {noble_dir}) & "
        f"(curl -sL --retry 3 -o /tmp/libgcc.deb http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libgcc-s1_14-20240412-0ubuntu1_amd64.deb && dpkg-deb -x /tmp/libgcc.deb {noble_dir}) & "
        f"wait && chmod -R +x {llama_dir}"
    )
    subprocess.run(dl_cmd, shell=True, check=True)

    # 2. Cloudflared deb
    debs_dir = work_dir / "debs"
    ensure_cloudflared_deb(debs_dir)

    # 3. Đóng gói wheel phụ trợ
    print("📦 [Embedding 2/4] Đóng gói các wheel Python cho Embedding/Rerank...")
    subprocess.run([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(venv_dir)], check=True)
    venv_pip = str(venv_dir / "bin" / "pip")
    subprocess.run(
        f"{venv_pip} install -q --cache-dir {pip_cache} -c {constraints_path} fastapi uvicorn pyyaml python-multipart requests psutil huggingface-hub diskcache",
        shell=True,
        check=True,
    )
    wheels_dir = work_dir / "wheels"
    wheel_files = collect_venv_delta_wheels(venv_dir, pip_cache, wheels_dir)

    # 4. Nén bundle
    bundle_path = staging_dir / "embedding_runtime_bundle.tar.bin"
    print(f"🗜️ [Embedding 3/4] Nén {bundle_path.name}...")
    subprocess.run(
        f"tar -cf - -C {work_dir} llama-cpp noble-libs debs wheels | gzip -1 > {bundle_path}",
        shell=True,
        check=True,
    )
    bundle_mb = bundle_path.stat().st_size / (1024 ** 2)

    manifest = {
        "studio_mode": "embedding",
        "bundle_file": bundle_path.name,
        "bundle_size_mb": round(bundle_mb, 2),
        "llama_server_build": "b11184-ubuntu-cuda-12.8-x64",
        "wheel_packages": wheel_files,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    publish_dataset(
        api=api,
        user=user,
        slug="kaggle-embedding-wheels",
        title="Kaggle Embedding Studio Offline Wheels",
        description="Offline runtime bundle for Kaggle Embedding & Rerank AI Studio containing official llama-server with CUDA acceleration, cloudflared, and Python wheels.",
        staging_dir=staging_dir,
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(venv_dir, ignore_errors=True)
    shutil.rmtree(pip_cache, ignore_errors=True)


def main():
    t_start = time.time()
    print("=" * 76)
    print("🚀 BẮT ĐẦU XÂY DỰNG CÁC BỘ KAGGLE WHEELS & RUNTIME DATASETS")
    print("=" * 76)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True)

    user = setup_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(f"✅ Đã xác thực Kaggle API thành công ({user})!")

    constraints_path = write_torch_constraints()

    if check_dataset_exists(api, user, "kaggle-speech-wheels") and os.environ.get("FORCE_REBUILD_SPEECH") != "1":
        print(f"\n⏭️ Dataset {user}/kaggle-speech-wheels đã tồn tại trên Kaggle, bỏ qua bước build lại Speech.")
    else:
        build_speech_dataset(api, user, constraints_path)

    build_vlm_dataset(api, user, constraints_path)
    build_image_video_dataset(api, user, constraints_path)
    build_embedding_dataset(api, user, constraints_path)

    print("\n" + "=" * 76)
    print(f"🎉 HOÀN TẤT XUẤT BẢN CẢ 4 DATASET WHEELS TRONG {time.time() - t_start:.1f}s!")
    print(f"   1. https://www.kaggle.com/datasets/{user}/kaggle-speech-wheels")
    print(f"   2. https://www.kaggle.com/datasets/{user}/kaggle-vlm-wheels")
    print(f"   3. https://www.kaggle.com/datasets/{user}/kaggle-image-video-wheels")
    print(f"   4. https://www.kaggle.com/datasets/{user}/kaggle-embedding-wheels")
    print("=" * 76)


if __name__ == "__main__":
    main()
