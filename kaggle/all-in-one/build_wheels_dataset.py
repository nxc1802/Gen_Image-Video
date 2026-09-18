#!/usr/bin/env python3
"""
⚙️ Kaggle Wheel Builder & Dataset Creator
================================================================================
Tự động biên dịch (wheel) mã nguồn Diffusers mới nhất (hỗ trợ Wan2.2) cùng các gói
thư viện trọng tâm của Studio AI trên môi trường máy ảo Linux của Kaggle,
sau đó đóng gói và xuất bản thành Kaggle Dataset:
`nguynxuncngde180528/kaggle-studio-wheels`
"""

import os
import sys
import json
import time
import shutil
import subprocess
import urllib.request
from pathlib import Path


def notify_ntfy(message: str):
    """Gửi thông báo trạng thái qua ntfy.sh."""
    topic = os.environ.get("NTFY_TOPIC", "studio-ai-url-nguynxuncngde180528")
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": "Kaggle Wheels Builder", "Priority": "3"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def setup_kaggle_credentials():
    """Thiết lập ~/.kaggle/kaggle.json từ biến môi trường."""
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    cred = {"username": user, "key": key}
    with open(kaggle_json, "w", encoding="utf-8") as f:
        json.dump(cred, f)
    kaggle_json.chmod(0o600)
    print(f"🔑 Đã cấu hình xác thực Kaggle API: {kaggle_json} (User: {user})")


def check_dataset_exists(api, user: str, slug: str) -> bool:
    dataset_ref = f"{user}/{slug}"
    try:
        api.dataset_status(dataset_ref)
        return True
    except Exception:
        return False


def main():
    print("=" * 76)
    print("🛠️ KAGGLE STUDIO WHEELS BUILDER & DATASET PUBLISHER")
    print("=" * 76)

    # 1. Nạp biến môi trường
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

    user = os.environ.get("KAGGLE_USERNAME", "nguynxuncngde180528")
    slug = "kaggle-studio-wheels"
    title = "Kaggle Studio Wheels (Wan2.2 & Diffusers Prebuilt)"
    dataset_ref = f"{user}/{slug}"

    setup_kaggle_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"✅ Đã xác thực Kaggle API thành công với tài khoản: {user}")
    except Exception as e:
        print(f"❌ Không thể xác thực Kaggle API: {e}")
        sys.exit(1)

    # 2. Tạo thư mục staging để gom file .whl
    staging_dir = Path("/tmp/staging_wheels")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    notify_ntfy("⏳ Bắt đầu biên dịch và tải các gói .whl cho Kaggle...")

    # 3. Biên dịch diffusers từ GitHub main (hỗ trợ Wan2.2 mới nhất)
    print("\n📦 [1/4] Đang build wheel cho diffusers (hỗ trợ Wan2.2)...")
    t0 = time.time()
    cmd_diffusers = [
        sys.executable, "-m", "pip", "wheel",
        "git+https://github.com/huggingface/diffusers.git",
        "--no-deps",
        "-w", str(staging_dir)
    ]
    res = subprocess.run(cmd_diffusers, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Lỗi khi build diffusers: {res.stderr}")
        sys.exit(1)
    print(f"✅ Build diffusers wheel thành công trong {time.time() - t0:.1f}s!")

    # 4. Tải prebuilt CUDA 12.1 wheel cho llama-cpp-python (hỗ trợ Qwen3.8-27B VLM GGUF)
    print("\n📦 [2/4] Đang tải prebuilt wheel cho llama-cpp-python (CUDA 12.1)...")
    t_llama = time.time()
    cmd_llama = [
        sys.executable, "-m", "pip", "wheel",
        "llama-cpp-python",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121",
        "--no-deps",
        "-w", str(staging_dir)
    ]
    res_llama = subprocess.run(cmd_llama, capture_output=True, text=True)
    if res_llama.returncode != 0:
        print(f"⚠️ Cảnh báo tải llama-cpp-python qua pip wheel: {res_llama.stderr}")
        try:
            direct_whl_url = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19-cu121/llama_cpp_python-0.3.19-cp310-cp310-linux_x86_64.whl"
            whl_name = "llama_cpp_python-0.3.19-cp310-cp310-linux_x86_64.whl"
            print(f"🔄 Đang thử tải trực tiếp wheel fallback: {whl_name}...")
            urllib.request.urlretrieve(direct_whl_url, str(staging_dir / whl_name))
            print(f"✅ Đã tải trực tiếp thành công {whl_name}!")
        except Exception as ex:
            print(f"❌ Không thể tải fallback llama-cpp-python wheel: {ex}")
    else:
        print(f"✅ Tải llama-cpp-python wheel thành công trong {time.time() - t_llama:.1f}s!")

    # 5. Tải các gói thư viện phụ thuộc trọng tâm dưới dạng .whl
    packages = [
        "kokoro>=0.8.4",
        "openai-whisper>=20240930",
        "torchao>=0.8.0",
        "bitsandbytes>=0.45.0",
        "accelerate>=1.2.0",
        "transformers>=4.49.0",
        "sentencepiece>=0.2.0",
        "soundfile>=0.13.0",
        "imageio[ffmpeg]>=2.35.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.32.0",
        "huggingface-hub>=0.28.0",
        "diskcache>=5.6.3",
        "pydantic>=2.10.0",
        "gguf>=0.10.0",
    ]

    print(f"\n📦 [3/4] Đang tải các gói .whl phụ thuộc ({len(packages)} packages)...")
    t1 = time.time()
    cmd_deps = [
        sys.executable, "-m", "pip", "wheel",
        *packages,
        "-w", str(staging_dir)
    ]
    res_deps = subprocess.run(cmd_deps, capture_output=True, text=True)
    if res_deps.returncode != 0:
        print(f"⚠️ Cảnh báo khi tải một số gói phụ thuộc: {res_deps.stderr}")
    else:
        print(f"✅ Tải các gói .whl hoàn tất trong {time.time() - t1:.1f}s!")

    # Liệt kê danh sách các file .whl đã tạo
    wheel_files = list(staging_dir.glob("*.whl"))
    total_mb = sum(f.stat().st_size for f in wheel_files) / (1024 ** 2)
    print(f"\n📋 Đã đóng gói thành công {len(wheel_files)} file .whl (Tổng dung lượng: {total_mb:.1f} MB):")
    for f in wheel_files[:10]:
        print(f"  - {f.name} ({f.stat().st_size / (1024**2):.2f} MB)")
    if len(wheel_files) > 10:
        print(f"  ... và {len(wheel_files) - 10} file khác.")

    # 5. Tạo metadata cho Kaggle Dataset
    meta_path = staging_dir / "dataset-metadata.json"
    meta_content = {
        "title": title,
        "id": dataset_ref,
        "licenses": [{"name": "apache-2.0"}],
        "description": "Prebuilt python wheel packages for Kaggle All-in-One AI Studio, including Wan2.2-enabled Diffusers.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)

    # 6. Xuất bản lên Kaggle Datasets
    print(f"\n⬆️ [4/4] Đang tải lên Kaggle Datasets ({dataset_ref})...")
    notify_ntfy("⬆️ Đang tải dataset wheels lên Kaggle...")
    exists = check_dataset_exists(api, user, slug)
    t_up = time.time()

    if not exists:
        resp = api.dataset_create_new(
            folder=str(staging_dir),
            public=False,
            quiet=False,
            dir_mode="skip"
        )
        print(f"🎉 Kết quả tạo mới dataset: {resp}")
    else:
        resp = api.dataset_create_version(
            folder=str(staging_dir),
            version_notes=f"Updated prebuilt wheels at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            quiet=False,
            dir_mode="skip"
        )
        print(f"🎉 Kết quả cập nhật dataset: {resp}")

    print(f"✅ Hoàn tất xuất bản {dataset_ref} trong {time.time() - t_up:.1f}s!")
    notify_ntfy("🎉 Kaggle Studio Wheels dataset đã xuất bản thành công!")

    # 7. Dọn dẹp
    shutil.rmtree(staging_dir, ignore_errors=True)
    print("🧹 Đã dọn dẹp thư mục staging.")


if __name__ == "__main__":
    main()
