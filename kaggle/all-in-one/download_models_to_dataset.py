#!/usr/bin/env python3
"""
📥 Kaggle Model Downloader & Dataset Creator (V2 - Direct Local Download)
================================================================================
Tự động tải các model GGUF (Wan2.2 TI2V-5B, Wan2.2 T2V-A14B, Wan2.2 I2V-A14B, Qwen3.8-27B)
từ Hugging Face về môi trường Kaggle và xuất bản thành các Kaggle Dataset riêng biệt.

Cải tiến V2:
1. Sử dụng trực tiếp `local_dir` trong `hf_hub_download` -> Tải file thật 100%, KHÔNG dùng symlink/blobs.
2. Tự động phát hiện phân vùng có dung lượng lớn nhất (/tmp ~73GB hoặc /kaggle/working) để chứa weights.
3. Làm sạch triệt để bộ nhớ đệm trước và sau mỗi model để đảm bảo 0% nguy cơ đầy đĩa.
"""

import os
import sys
import json
import time
import shutil
import urllib.request
from pathlib import Path

# Cấu hình danh sách model cần tải và xuất bản
MODEL_DOWNLOAD_JOBS = [
    {
        "id": "wan22_5b",
        "title": "Wan2.2 TI2V 5B GGUF",
        "slug": "wan22-ti2v-5b-gguf",
        "repo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "files": [
            {"src": "Wan2.2-TI2V-5B-Q4_K_M.gguf", "dest": "Wan2.2-TI2V-5B-Q4_K_M.gguf"},
            {"src": "VAE/Wan2.2_VAE.safetensors", "dest": "Wan2.2_VAE.safetensors"},
        ],
        "estimated_gb": 4.51,
    },
    {
        "id": "wan22_14b_t2v",
        "title": "Wan2.2 T2V A14B GGUF",
        "slug": "wan22-t2v-a14b-gguf",
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "files": [
            {"src": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", "dest": "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"},
            {"src": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf", "dest": "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"},
            {"src": "VAE/Wan2.1_VAE.safetensors", "dest": "Wan2.1_VAE.safetensors"},
        ],
        "estimated_gb": 18.22,
    },
    {
        "id": "wan22_14b_i2v",
        "title": "Wan2.2 I2V A14B GGUF",
        "slug": "wan22-i2v-a14b-gguf",
        "repo": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "files": [
            {"src": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", "dest": "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"},
            {"src": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", "dest": "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"},
            {"src": "VAE/Wan2.1_VAE.safetensors", "dest": "Wan2.1_VAE.safetensors"},
        ],
        "estimated_gb": 18.22,
    },
    {
        "id": "qwen38_27b",
        "title": "Qwen3.8 27B VLM GGUF",
        "slug": "qwen38-27b-vlm-gguf",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "files": [
            {"src": "Qwen3.8-27B-UD-Q4_K_M.gguf", "dest": "Qwen3.8-27B-UD-Q4_K_M.gguf"},
            {"src": "mmproj-F16.gguf", "dest": "mmproj-F16.gguf"},
        ],
        "estimated_gb": 16.19,
    },
    {
        "id": "umt5_xxl_encoder",
        "title": "UMT5 XXL Encoder GGUF",
        "slug": "umt5-xxl-encoder-gguf",
        "repo": "city96/umt5-xxl-encoder-gguf",
        "files": [
            {"src": "umt5-xxl-encoder-Q4_K_M.gguf", "dest": "umt5-xxl-encoder-Q4_K_M.gguf"},
        ],
        "extra_downloads": [
            {"repo": "google/umt5-xxl", "src": "tokenizer_config.json", "dest": "tokenizer_config.json"},
            {"repo": "google/umt5-xxl", "src": "tokenizer.json", "dest": "tokenizer.json"},
            {"repo": "google/umt5-xxl", "src": "spiece.model", "dest": "spiece.model"},
            {"repo": "google/umt5-xxl", "src": "special_tokens_map.json", "dest": "special_tokens_map.json"},
        ],
        "estimated_gb": 3.42,
    },
    {
        "id": "flux2_klein_4b",
        "title": "FLUX.2 Klein 4B GGUF",
        "slug": "flux2-klein-4b-gguf",
        "repo": "unsloth/FLUX.2-klein-4B-GGUF",
        "files": [
            {"src": "flux-2-klein-4b-Q4_K_M.gguf", "dest": "flux-2-klein-4b-Q4_K_M.gguf"},
        ],
        "extra_downloads": [
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "model_index.json", "dest": "model_index.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "scheduler/scheduler_config.json", "dest": "scheduler_config.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "transformer/config.json", "dest": "transformer_config.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "vae/config.json", "dest": "vae_config.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "vae/diffusion_pytorch_model.safetensors", "dest": "vae.safetensors"},
        ],
        "estimated_gb": 2.65,
    },
    {
        "id": "qwen3_4b",
        "title": "Qwen3 4B GGUF",
        "slug": "qwen3-4b-gguf",
        "repo": "unsloth/Qwen3-4B-GGUF",
        "files": [
            {"src": "Qwen3-4B-Q4_K_M.gguf", "dest": "Qwen3-4B-Q4_K_M.gguf"},
        ],
        "extra_downloads": [
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "text_encoder/config.json", "dest": "config.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/tokenizer.json", "dest": "tokenizer.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/tokenizer_config.json", "dest": "tokenizer_config.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/vocab.json", "dest": "vocab.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/merges.txt", "dest": "merges.txt"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/special_tokens_map.json", "dest": "special_tokens_map.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/added_tokens.json", "dest": "added_tokens.json"},
            {"repo": "black-forest-labs/FLUX.2-klein-4B", "src": "tokenizer/chat_template.jinja", "dest": "chat_template.jinja"},
        ],
        "estimated_gb": 2.45,
    },
]


def notify_ntfy(message: str):
    """Gửi thông báo trạng thái qua ntfy.sh."""
    topic = os.environ.get("NTFY_TOPIC", "studio-ai-url-nguynxuncngde180528")
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": "Kaggle Dataset Creator", "Priority": "3"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def setup_kaggle_credentials():
    """Thiết lập tệp xác thực ~/.kaggle/kaggle.json từ biến môi trường."""
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


def get_best_workspace() -> Path:
    """Tự động kiểm tra và chọn phân vùng có dung lượng trống lớn nhất."""
    candidates = [Path("/tmp"), Path("/kaggle/working"), Path("/kaggle/temp")]
    best_dir = Path("/tmp")
    max_free = 0

    print("🔍 Kiểm tra dung lượng các phân vùng ổ đĩa:")
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(c)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            print(f"  - {c}: Trống {free_gb:.2f} GB / Tổng {total_gb:.2f} GB")
            if usage.free > max_free:
                max_free = usage.free
                best_dir = c
        except Exception:
            pass

    selected = best_dir / "kaggle_staging"
    selected.mkdir(parents=True, exist_ok=True)
    print(f"🎯 Đã chọn thư mục làm việc tối ưu: {selected} (Trống: {max_free / (1024**3):.2f} GB)\n")
    return selected


def check_dataset_exists(api, user: str, slug: str) -> bool:
    """Kiểm tra xem dataset đã tồn tại trên Kaggle chưa."""
    dataset_ref = f"{user}/{slug}"
    try:
        api.dataset_status(dataset_ref)
        return True
    except Exception:
        return False


def clean_all_staging(base_dir: Path):
    """Xóa sạch mọi file tạm và cache để giải phóng đĩa tối đa."""
    print("🧹 Đang dọn dẹp các thư mục tạm và cache...")
    if base_dir.exists():
        for item in base_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass
    # Dọn dẹp cả thư mục cache mặc định của HF nếu có
    hf_default_cache = Path.home() / ".cache" / "huggingface"
    if hf_default_cache.exists():
        shutil.rmtree(hf_default_cache, ignore_errors=True)
    
    # Kiểm tra dung lượng sau khi dọn
    usage = shutil.disk_usage(base_dir)
    print(f"✨ Dung lượng trống hiện tại: {usage.free / (1024**3):.2f} GB")


def run_single_job(job: dict, api, user: str, hf_token: str, base_dir: Path):
    """Thực thi tải và tạo dataset cho 1 job."""
    slug = job["slug"]
    title = job["title"]
    repo = job["repo"]
    files = job["files"]
    dataset_ref = f"{user}/{slug}"

    print("\n" + "=" * 76)
    print(f"🚀 BẮT ĐẦU XỬ LÝ MODEL: {title} ({job['estimated_gb']} GB)")
    print(f"📦 Dataset đích: https://www.kaggle.com/datasets/{dataset_ref}")
    print(f"🌐 Hugging Face: https://huggingface.co/{repo}")
    print("=" * 76)

    notify_ntfy(f"⏳ Bắt đầu tải {title} ({job['estimated_gb']} GB)...")

    # Dọn dẹp trước khi bắt đầu
    clean_all_staging(base_dir)

    staging_dir = base_dir / slug
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra dataset đã tồn tại hay chưa
    exists = check_dataset_exists(api, user, slug)
    if exists:
        print(f"ℹ️ Dataset {dataset_ref} đã tồn tại trên Kaggle. Sẽ tạo version mới nếu cần.")

    # 2. Tải từng file từ Hugging Face trực tiếp vào staging_dir
    from huggingface_hub import hf_hub_download

    total_files = len(files)
    for idx, f_info in enumerate(files, 1):
        src_file = f_info["src"]
        dest_filename = f_info["dest"]
        dest_path = staging_dir / dest_filename

        print(f"\n[{idx}/{total_files}] 📥 Đang tải: {src_file} -> {dest_filename}...")
        t_start = time.time()
        
        try:
            # Tải trực tiếp vào staging_dir bằng local_dir (file thật, không symlink)
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=src_file,
                token=hf_token if hf_token else None,
                local_dir=str(staging_dir),
            )
            downloaded_path = Path(downloaded).resolve()
            dest_path_resolved = dest_path.resolve()

            # Nếu file nằm trong subfolder (như HighNoise/...), di chuyển lên thư mục gốc của dataset
            if downloaded_path != dest_path_resolved:
                if dest_path_resolved.exists():
                    dest_path_resolved.unlink()
                shutil.move(str(downloaded_path), str(dest_path_resolved))

                # Xóa subfolder rỗng nếu có
                subfolder = downloaded_path.parent
                if subfolder != staging_dir and subfolder.exists():
                    shutil.rmtree(subfolder, ignore_errors=True)

            if not dest_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file đích: {dest_path}")

            file_size_gb = dest_path.stat().st_size / (1024 ** 3)
            duration = time.time() - t_start
            speed_mb = (file_size_gb * 1024) / max(duration, 0.1)
            print(f"✅ Hoàn tất tải {dest_filename} ({file_size_gb:.2f} GB) trong {duration:.1f}s (~{speed_mb:.1f} MB/s)")

        except Exception as e:
            print(f"❌ Lỗi tải file {src_file} từ {repo}: {e}")
            raise e

    # Tải các file bổ sung nếu có (ví dụ Tokenizer từ repo khác)
    extra_list = job.get("extra_downloads", [])
    if extra_list:
        print(f"\n📦 Đang tải {len(extra_list)} file bổ sung (Tokenizer, config)...")
        for e_idx, e_info in enumerate(extra_list, 1):
            e_repo = e_info["repo"]
            e_src = e_info["src"]
            e_dest = e_info["dest"]
            e_dest_path = staging_dir / e_dest

            print(f"[{e_idx}/{len(extra_list)}] 📥 Tải: {e_repo} -> {e_dest}...")
            try:
                e_dl = hf_hub_download(
                    repo_id=e_repo,
                    filename=e_src,
                    token=hf_token if hf_token else None,
                    local_dir=str(staging_dir),
                )
                e_dl_path = Path(e_dl).resolve()
                e_dest_resolved = e_dest_path.resolve()
                if e_dl_path != e_dest_resolved:
                    if e_dest_resolved.exists():
                        e_dest_resolved.unlink()
                    shutil.move(str(e_dl_path), str(e_dest_resolved))
                print(f"✅ Đã tải: {e_dest}")
            except Exception as err:
                print(f"⚠️ Cảnh báo tải {e_src}: {err}")

    # Xóa thư mục ẩn .huggingface nếu có được tạo trong staging_dir
    hf_hidden = staging_dir / ".huggingface"
    if hf_hidden.exists():
        shutil.rmtree(hf_hidden, ignore_errors=True)

    # 3. Tạo metadata cho Kaggle Dataset
    meta_path = staging_dir / "dataset-metadata.json"
    metadata_content = {
        "title": title,
        "id": dataset_ref,
        "licenses": [{"name": "apache-2.0"}],
        "description": f"Quantized GGUF models for {title}, sourced from https://huggingface.co/{repo}.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_content, f, indent=2)

    print(f"\n📋 Đã tạo metadata dataset: {meta_path}")

    # 4. Xuất bản lên Kaggle Datasets
    print(f"⬆️ Đang tải lên Kaggle Datasets ({dataset_ref})...")
    notify_ntfy(f"⬆️ Đang upload {title} lên Kaggle Datasets...")
    t_upload_start = time.time()

    if not exists:
        # Tạo dataset mới
        resp = api.dataset_create_new(
            folder=str(staging_dir),
            public=False,
            quiet=False,
            dir_mode="skip"
        )
        print(f"🎉 Kết quả tạo mới dataset: {resp}")
    else:
        # Tạo version mới
        resp = api.dataset_create_version(
            folder=str(staging_dir),
            version_notes=f"Updated weights at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            quiet=False,
            dir_mode="skip"
        )
        print(f"🎉 Kết quả tạo version dataset: {resp}")

    upload_duration = time.time() - t_upload_start
    print(f"✅ Đã xuất bản thành công {dataset_ref} trong {upload_duration:.1f}s!")
    notify_ntfy(f"✅ Hoàn thành xuất bản {title} trên Kaggle!")

    # 5. Dọn dẹp sạch sau khi hoàn tất
    clean_all_staging(base_dir)


def main():
    print("=" * 76)
    print("🌟 KAGGLE MULTI-MODEL GGUF DOWNLOADER & DATASET PUBLISHER (V2)")
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
    hf_token = os.environ.get("HF_TOKEN", "")

    # 2. Cấu hình Kaggle API
    setup_kaggle_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"✅ Đã xác thực thành công Kaggle API với tài khoản: {user}\n")
    except Exception as e:
        print(f"❌ Không thể xác thực Kaggle API: {e}")
        sys.exit(1)

    # 3. Chọn thư mục làm việc có dung lượng đĩa lớn nhất
    base_dir = get_best_workspace()

    # Dọn dẹp sạch trước khi chạy bất kỳ job nào
    clean_all_staging(base_dir)

    # Lựa chọn model từ tham số dòng lệnh
    target_model_id = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            target_model_id = arg.split("=", 1)[1]
        elif arg == "--all":
            target_model_id = "all"

    jobs_to_run = MODEL_DOWNLOAD_JOBS
    if target_model_id and target_model_id != "all":
        jobs_to_run = [j for j in MODEL_DOWNLOAD_JOBS if j["id"] == target_model_id]
        if not jobs_to_run:
            print(f"❌ Không tìm thấy model có id: {target_model_id}")
            print(f"Các ID khả dụng: {[j['id'] for j in MODEL_DOWNLOAD_JOBS]}")
            sys.exit(1)

    print(f"📋 Danh sách model sẽ thực thi ({len(jobs_to_run)} models):")
    for j in jobs_to_run:
        print(f"  - [{j['id']}] {j['title']} (~{j['estimated_gb']} GB)")

    total_start = time.time()
    success_count = 0

    for idx, job in enumerate(jobs_to_run, 1):
        print(f"\n▶️ TIẾN TRÌNH TỔNG THỂ: [{idx}/{len(jobs_to_run)}]")
        try:
            run_single_job(job, api, user, hf_token, base_dir)
            success_count += 1
        except Exception as e:
            print(f"❌ THẤT BẠI KHI XỬ LÝ {job['title']}: {e}")
            notify_ntfy(f"❌ Thất bại khi xử lý {job['title']}: {e}")

    total_time = (time.time() - total_start) / 60
    print("\n" + "=" * 76)
    print(f"🏁 ĐÃ HOÀN TẤT QUÁ TRÌNH TẢI & TẠO DATASET!")
    print(f"   Thành công: {success_count}/{len(jobs_to_run)} models")
    print(f"   Tổng thời gian thực thi: {total_time:.1f} phút")
    print("=" * 76)
    notify_ntfy(f"🏁 Hoàn tất: {success_count}/{len(jobs_to_run)} dataset đã được tạo thành công!")


if __name__ == "__main__":
    main()
