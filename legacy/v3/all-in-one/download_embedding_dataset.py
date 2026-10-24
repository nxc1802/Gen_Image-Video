#!/usr/bin/env python3
"""
📥 Kaggle Embedding & Rerank Model Downloader & Dataset Creator
================================================================================
Tự động tải toàn bộ 10 model GGUF (Q4_K_M) của nhóm Qwen3 từ Hugging Face
và xuất bản thành Kaggle Dataset chuyên biệt: `cuongnguyen1802/kaggle-embedding-models`.

Phân bổ phần cứng thông minh trên Dual Tesla T4 (32GB VRAM):
- GPU 0 (cuda:0) - 3 Model Dòng Text:
  1. Qwen3-Embedding-0.6B (Text Embed)  -> mradermacher/Qwen3-Embedding-0.6B-GGUF (~378 MB)
  2. Qwen3-Reranker-0.6B  (Text Rerank) -> mradermacher/Qwen3-Reranker-0.6B-GGUF  (~378 MB)
  3. Qwen3-Embedding-4B   (Text Embed)  -> Qwen/Qwen3-Embedding-4B-GGUF          (~2.38 GB)
  4. Qwen3-Reranker-4B    (Text Rerank) -> mradermacher/Qwen3-Reranker-4B-GGUF   (~2.38 GB)
  5. Qwen3-Embedding-8B   (Text Embed)  -> Qwen/Qwen3-Embedding-8B-GGUF          (~4.36 GB)
  6. Qwen3-Reranker-8B    (Text Rerank) -> mradermacher/Qwen3-Reranker-8B-GGUF   (~4.68 GB)
- GPU 1 (cuda:1) - 2 Model Dòng Vision-Language (VL):
  7. Qwen3-VL-Embedding-2B (VL Embed)   -> mradermacher/Qwen3-VL-Embedding-2B-GGUF (~1.48 GB inc mmproj)
  8. Qwen3-VL-Reranker-2B  (VL Rerank)  -> mradermacher/Qwen3-VL-Reranker-2B-GGUF  (~1.48 GB inc mmproj)
  9. Qwen3-VL-Embedding-8B (VL Embed)   -> mradermacher/Qwen3-VL-Embedding-8B-GGUF (~5.18 GB inc mmproj)
  10. Qwen3-VL-Reranker-8B (VL Rerank)  -> mradermacher/Qwen3-VL-Reranker-8B-GGUF  (~5.51 GB inc mmproj)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

EMBEDDING_DOWNLOAD_JOBS = [
    {
        "id": "qwen3_emb_0_6b",
        "title": "Qwen3 Embedding 0.6B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-Embedding-0.6B-GGUF",
        "files": [
            {"src": "Qwen3-Embedding-0.6B.Q4_K_M.gguf", "dest": "Qwen3-Embedding-0.6B.Q4_K_M.gguf"},
        ],
        "target_gpu": "GPU 0",
        "estimated_gb": 0.38,
    },
    {
        "id": "qwen3_rerank_0_6b",
        "title": "Qwen3 Reranker 0.6B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-Reranker-0.6B-GGUF",
        "files": [
            {"src": "Qwen3-Reranker-0.6B.Q4_K_M.gguf", "dest": "Qwen3-Reranker-0.6B.Q4_K_M.gguf"},
        ],
        "target_gpu": "GPU 0",
        "estimated_gb": 0.38,
    },
    {
        "id": "qwen3_emb_4b",
        "title": "Qwen3 Embedding 4B GGUF Q4_K_M",
        "repo": "Qwen/Qwen3-Embedding-4B-GGUF",
        "files": [
            {"src": "Qwen3-Embedding-4B-Q4_K_M.gguf", "dest": "Qwen3-Embedding-4B-Q4_K_M.gguf"},
        ],
        "target_gpu": "GPU 0",
        "estimated_gb": 2.38,
    },
    {
        "id": "qwen3_rerank_4b",
        "title": "Qwen3 Reranker 4B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-Reranker-4B-GGUF",
        "files": [
            {"src": "Qwen3-Reranker-4B.Q4_K_M.gguf", "dest": "Qwen3-Reranker-4B.Q4_K_M.gguf"},
        ],
        "target_gpu": "GPU 0",
        "estimated_gb": 2.38,
    },
    {
        "id": "qwen3_emb_8b",
        "title": "Qwen3 Embedding 8B GGUF Q4_K_M",
        "repo": "Qwen/Qwen3-Embedding-8B-GGUF",
        "files": [
            {"src": "Qwen3-Embedding-8B-Q4_K_M.gguf", "dest": "Qwen3-Embedding-8B-Q4_K_M.gguf"},
        ],
        "target_gpu": "GPU 0",
        "estimated_gb": 4.36,
    },
    {
        "id": "qwen3_rerank_8b",
        "title": "Qwen3 Reranker 8B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-Reranker-8B-GGUF",
        "files": [
            {"src": "Qwen3-Reranker-8B.Q4_K_M.gguf", "dest": "Qwen3-Reranker-8B.Q4_K_M.gguf"},
        ],
        "target_gpu": "GPU 0",
        "estimated_gb": 4.68,
    },
    {
        "id": "qwen3_vl_emb_2b",
        "title": "Qwen3-VL Embedding 2B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-VL-Embedding-2B-GGUF",
        "files": [
            {"src": "Qwen3-VL-Embedding-2B.Q4_K_M.gguf", "dest": "Qwen3-VL-Embedding-2B.Q4_K_M.gguf"},
            {"src": "Qwen3-VL-Embedding-2B.mmproj-Q8_0.gguf", "dest": "Qwen3-VL-Embedding-2B.mmproj-Q8_0.gguf"},
        ],
        "target_gpu": "GPU 1",
        "estimated_gb": 1.48,
    },
    {
        "id": "qwen3_vl_rerank_2b",
        "title": "Qwen3-VL Reranker 2B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-VL-Reranker-2B-GGUF",
        "files": [
            {"src": "Qwen3-VL-Reranker-2B.Q4_K_M.gguf", "dest": "Qwen3-VL-Reranker-2B.Q4_K_M.gguf"},
            {"src": "Qwen3-VL-Reranker-2B.mmproj-Q8_0.gguf", "dest": "Qwen3-VL-Reranker-2B.mmproj-Q8_0.gguf"},
        ],
        "target_gpu": "GPU 1",
        "estimated_gb": 1.48,
    },
    {
        "id": "qwen3_vl_emb_8b",
        "title": "Qwen3-VL Embedding 8B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-VL-Embedding-8B-GGUF",
        "files": [
            {"src": "Qwen3-VL-Embedding-8B.Q4_K_M.gguf", "dest": "Qwen3-VL-Embedding-8B.Q4_K_M.gguf"},
            {"src": "Qwen3-VL-Embedding-8B.mmproj-Q8_0.gguf", "dest": "Qwen3-VL-Embedding-8B.mmproj-Q8_0.gguf"},
        ],
        "target_gpu": "GPU 1 (Isolated to prevent OOM)",
        "estimated_gb": 5.18,
    },
    {
        "id": "qwen3_vl_rerank_8b",
        "title": "Qwen3-VL Reranker 8B GGUF Q4_K_M",
        "repo": "mradermacher/Qwen3-VL-Reranker-8B-GGUF",
        "files": [
            {"src": "Qwen3-VL-Reranker-8B.Q4_K_M.gguf", "dest": "Qwen3-VL-Reranker-8B.Q4_K_M.gguf"},
            {"src": "Qwen3-VL-Reranker-8B.mmproj-Q8_0.gguf", "dest": "Qwen3-VL-Reranker-8B.mmproj-Q8_0.gguf"},
        ],
        "target_gpu": "GPU 1 (Isolated to prevent OOM)",
        "estimated_gb": 5.51,
    },
]


def setup_kaggle_credentials() -> str:
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
    with open(kaggle_dir / "kaggle.json", "w", encoding="utf-8") as f:
        json.dump({"username": user, "key": key}, f)
    (kaggle_dir / "kaggle.json").chmod(0o600)
    return user


def download_model_job(job: Dict, target_dir: Path, token: Optional[str] = None):
    from huggingface_hub import hf_hub_download
    print(f"\n📥 [{job['id']}] Bắt đầu tải: {job['title']} (Ước tính: {job['estimated_gb']} GB)")
    print(f"   • HF Repo   : {job['repo']}")
    print(f"   • Phân bổ GPU: {job['target_gpu']}")

    target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for item in job["files"]:
        src_file = item["src"]
        dest_name = item["dest"]
        dest_path = target_dir / dest_name

        if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
            print(f"   ⏩ Đã tồn tại: {dest_name} ({dest_path.stat().st_size / (1024**2):.1f} MB), bỏ qua.")
            continue

        print(f"   ⬇️ Đang tải {src_file} -> {dest_name}...")
        downloaded = hf_hub_download(
            repo_id=job["repo"],
            filename=src_file,
            token=token,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        if Path(downloaded).name != dest_name:
            shutil.move(downloaded, dest_path)

    elapsed = time.time() - t0
    print(f"✅ Hoàn tất tải {job['id']} trong {elapsed:.1f}s!")


def publish_dataset(staging_dir: Path, user: str, slug: str = "kaggle-embedding-models"):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    dataset_ref = f"{user}/{slug}"
    meta_path = staging_dir / "dataset-metadata.json"
    meta = {
        "title": "Kaggle Embedding & Rerank Models (Qwen3 Q4_K_M GGUF)",
        "id": dataset_ref,
        "licenses": [{"name": "apache-2.0"}],
        "description": "Bộ mô hình nhúng và sắp xếp lại văn bản/hình ảnh Qwen3 (0.6B, 4B, VL-2B, VL-8B) lượng tử hóa Q4_K_M.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n⬆️ Đang xuất bản Dataset {dataset_ref}...")
    try:
        api.dataset_status(dataset_ref)
        exists = True
    except Exception:
        exists = False

    if not exists:
        api.dataset_create_new(folder=str(staging_dir), public=True, dir_mode="skip")
        print(f"🎉 Đã tạo mới Kaggle Dataset: {dataset_ref}")
    else:
        api.dataset_create_version(
            folder=str(staging_dir),
            version_notes=f"Updated embedding models at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            dir_mode="skip",
        )
        print(f"🎉 Đã cập nhật version mới cho {dataset_ref}")


def main():
    parser = argparse.ArgumentParser(description="Tải model Embedding & Rerank thành Kaggle Dataset")
    parser.add_argument("--job", help="Chỉ tải 1 job cụ thể (qwen3_emb_0_6b, qwen3_rerank_0_6b, ...)")
    parser.add_argument("--out-dir", default="./staging_embedding_dataset", help="Thư mục chứa weights")
    parser.add_argument("--publish", action="store_true", help="Xuất bản lên Kaggle Dataset sau khi tải")
    parser.add_argument("--slug", default="kaggle-embedding-models", help="Slug Kaggle Dataset")
    args = parser.parse_args()

    user = setup_kaggle_credentials()
    hf_token = os.environ.get("HF_TOKEN")
    target_dir = Path(args.out_dir).resolve()

    jobs_to_run = EMBEDDING_DOWNLOAD_JOBS
    if args.job:
        jobs_to_run = [j for j in EMBEDDING_DOWNLOAD_JOBS if j["id"] == args.job]
        if not jobs_to_run:
            print(f"❌ Không tìm thấy job: {args.job}")
            sys.exit(1)

    print("=" * 78)
    print("🧬 KAGGLE EMBEDDING & RERANK MODEL DATASET BUILDER")
    print(f"   • Số lượng Jobs  : {len(jobs_to_run)}")
    print(f"   • Thư mục Staging: {target_dir}")
    print("=" * 78)

    for job in jobs_to_run:
        download_model_job(job, target_dir, token=hf_token)

    if args.publish:
        publish_dataset(target_dir, user, slug=args.slug)

    print("\n✨ Tất cả tác vụ hoàn tất!")


if __name__ == "__main__":
    main()
