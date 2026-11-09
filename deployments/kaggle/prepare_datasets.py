"""
deployments.kaggle.prepare_datasets
Manifest-driven Kaggle Dataset preparation & deployment orchestrator for Factory v4 Lite.

Key capabilities:
1. Aligns with all 11 Model Manifests in manifests/models/*.yaml.
2. Direct integration with Kaggle API via Bearer token (KGAT_*) authentication.
3. Queries existing dataset readiness (e.g. cuongnguyen1802/qwen25-7b-instruct-awq, minimax-h3-gguf).
4. Generates standalone remote downloader kernels deployed directly to Kaggle Cloud
   (utilizing Kaggle's high-speed network and 73GB /tmp storage without consuming local disk).
5. Monitors download, packaging, and dataset publication progress in real-time.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from deployments.kaggle.runner import KaggleAPIClient, resolve_kaggle_username

logger = logging.getLogger("prepare_datasets")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Authoritative dataset specifications for Factory v4 Lite 11 models
MODEL_DATASETS: List[Dict[str, Any]] = [
    {
        "id": "wan-2.2",
        "slug": "wan22-ti2v-5b-gguf",
        "title": "Wan 2.2 TI2V 5B GGUF",
        "repo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "service_id": "video-service",
        "target_gpu": 1,
        "estimated_gb": 4.5,
        "files": [
            {"src": "Wan2.2-TI2V-5B-Q4_K_M.gguf", "dest": "Wan2.2-TI2V-5B-Q4_K_M.gguf"},
            {"src": "VAE/Wan2.2_VAE.safetensors", "dest": "Wan2.2_VAE.safetensors"},
        ],
    },
    {
        "id": "minimax-h3",
        "slug": "minimax-h3-gguf",
        "title": "MiniMax Hailuo H3 GGUF Video & Audio",
        "repo": "unsloth/MiniMax-H3-GGUF",
        "service_id": "video-service",
        "target_gpu": 1,
        "estimated_gb": 33.1,
        "files": [
            {"src": "minimax_h3_fl2va_pruned-Q4_K.gguf", "dest": "MiniMax-H3-FL2VA-Q4_K_M.gguf"},
            {"src": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "dest": "qwen3_vl_32b_q4_k_m.gguf"},
            {"src": "vae/minimax_h3_video_vae_fp16.safetensors", "dest": "minimax_h3_video_vae_fp16.safetensors"},
            {"src": "vae/minimax_h3_audio_vae_fp32.safetensors", "dest": "minimax_h3_audio_vae_fp32.safetensors"},
        ],
    },
    {
        "id": "qwen-image-2.1",
        "slug": "qwen-image-2-1-gguf",
        "title": "Qwen Image 2.1 GGUF",
        "repo": "abenzerps/Qwen-Image-2.1-GGUF",
        "service_id": "image-service",
        "target_gpu": 0,
        "estimated_gb": 4.2,
        "files": [
            {"src": "qwen-image-2.1-Q4_K_M.gguf", "dest": "qwen-image-2.1-Q4_K_M.gguf"},
            {"src": "vae/qwen_image_2.1_vae_bf16.safetensors", "dest": "qwen_image_2.1_vae_bf16.safetensors"},
        ],
    },
    {
        "id": "qwen-image-2.1-turbo",
        "slug": "qwen-image-2-1-turbo-gguf",
        "title": "Qwen Image 2.1 Turbo GGUF",
        "repo": "unsloth/Qwen-Image-2.1-Turbo-GGUF",
        "service_id": "image-service",
        "target_gpu": 0,
        "estimated_gb": 3.91,
        "files": [
            {"src": "qwen-image-2.1-turbo-Q4_K_M.gguf", "dest": "qwen-image-2.1-turbo-Q4_K_M.gguf"},
        ],
    },
    {
        "id": "qwen-text",
        "slug": "qwen25-7b-instruct-awq",
        "title": "Qwen 2.5 7B Instruct AWQ Official",
        "repo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "service_id": "text-service",
        "target_gpu": 0,
        "estimated_gb": 4.5,
        "snapshot": True,
        "files": [],
    },
    {
        "id": "qwen-vlm",
        "slug": "qwen2-vl-7b-instruct-awq",
        "title": "Qwen2-VL 7B Instruct AWQ Official",
        "repo": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "service_id": "vision-service",
        "target_gpu": 0,
        "estimated_gb": 5.5,
        "snapshot": True,
        "files": [],
    },
    {
        "id": "qwen-speech",
        "slug": "qwen3-asr-tts",
        "title": "Qwen3 ASR & TTS Multilingual Suite",
        "repo": "Qwen/Qwen3-ASR-1.7B",
        "service_id": "speech-service",
        "target_gpu": 0,
        "estimated_gb": 7.4,
        "snapshot": True,
        "files": [],
        "extra_repos": [
            {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "dest_dir": "tts_base"},
        ],
    },
    {
        "id": "qwen-embedding",
        "slug": "kaggle-embedding-models",
        "title": "Qwen3 Embedding & Reranker GGUF",
        "repo": "mradermacher/Qwen3-Embedding-0.6B-GGUF",
        "service_id": "embed-service",
        "target_gpu": 1,
        "estimated_gb": 5.2,
        "files": [
            {"src": "Qwen3-Embedding-0.6B.Q4_K_M.gguf", "dest": "Qwen3-Embedding-0.6B.Q4_K_M.gguf"},
        ],
        "extra_downloads": [
            {"repo": "mradermacher/Qwen3-Reranker-0.6B-GGUF", "src": "Qwen3-Reranker-0.6B.Q4_K_M.gguf", "dest": "Qwen3-Reranker-0.6B.Q4_K_M.gguf"},
            {"repo": "Qwen/Qwen3-Embedding-4B-GGUF", "src": "Qwen3-Embedding-4B-Q4_K_M.gguf", "dest": "Qwen3-Embedding-4B-Q4_K_M.gguf"},
        ],
    },
    {
        "id": "qwen-decision",
        "slug": "qwen-decision-router",
        "title": "Qwen 2.5 Decision Router",
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "service_id": "decision-service",
        "target_gpu": 0,
        "estimated_gb": 3.2,
        "snapshot": True,
        "files": [],
    },
    {
        "id": "ace-step-1.5",
        "slug": "acestep-music-1-5",
        "title": "ACE-Step 1.5 Music Generation",
        "repo": "ACE-Studio/ACE-Step-1.5",
        "service_id": "music-service",
        "target_gpu": 1,
        "estimated_gb": 3.5,
        "snapshot": True,
        "files": [],
    },
    {
        "id": "enhancement-suite-v1",
        "slug": "enhancement-suite",
        "title": "Media Post-Processing & Enhancement Suite",
        "repo": "xinntao/Real-ESRGAN",
        "service_id": "enhancement-service",
        "target_gpu": 1,
        "estimated_gb": 2.5,
        "snapshot": True,
        "files": [],
    },
    {
        "id": "yue2",
        "slug": "yue-s1-7b",
        "title": "YuE s1 7B Annealed En CoT",
        "repo": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "service_id": "music-service",
        "target_gpu": 1,
        "estimated_gb": 13.5,
        "snapshot": True,
        "files": [],
    },
]


class DatasetPreparationManager:
    """Manages discovery, verification, and automated deployment of Kaggle model datasets."""

    def __init__(self, username: Optional[str] = None, api_key: Optional[str] = None):
        self.client = KaggleAPIClient(username=username, api_key=api_key)
        self.username = self.client.username
        self.api_key = self.client.api_key

    def check_dataset_status(self, slug: str) -> Dict[str, Any]:
        """Checks readiness of a specific Kaggle dataset."""
        try:
            status = self.client._request("GET", f"/datasets/status/{self.username}/{slug}")
            return {"exists": True, "status": str(status).strip(), "ref": f"{self.username}/{slug}"}
        except Exception as e:
            code = getattr(e, "status_code", None)
            return {"exists": False, "status": "not_found", "error": str(e), "code": code}

    def inspect_all_datasets(self) -> List[Dict[str, Any]]:
        """Inspects status of all 11 Factory v4 model datasets on Kaggle."""
        results = []
        for item in MODEL_DATASETS:
            st = self.check_dataset_status(item["slug"])
            results.append({
                "model_id": item["id"],
                "service_id": item["service_id"],
                "slug": item["slug"],
                "title": item["title"],
                "target_gpu": item["target_gpu"],
                "estimated_gb": item["estimated_gb"],
                "status": st.get("status"),
                "exists": st.get("exists", False),
            })
        return results

    def build_downloader_script(self, target_slugs: Optional[List[str]] = None) -> str:
        """Generates self-contained Python script to run inside Kaggle Kernel for dataset download & publish."""
        selected_jobs = [
            item for item in MODEL_DATASETS
            if target_slugs is None or item["slug"] in target_slugs or item["id"] in target_slugs
        ]

        jobs_json = json.dumps(selected_jobs, indent=2)

        script_code = f"""# Auto-generated by Factory v4 Lite DatasetPreparationManager
import json
import os
from pathlib import Path
import shutil
import sys
import time

print("=" * 80)
print("🚀 FACTORY V4 LITE REMOTE KAGGLE MODEL DOWNLOADER & PUBLISHER")
print("=" * 80)

# Dynamically resolve Kaggle credentials inside kernel without embedding raw keys
kaggle_user = os.environ.get("KAGGLE_USERNAME", "{self.username}")
kaggle_key = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    if not kaggle_key:
        try:
            kaggle_key = secrets.get_secret("KAGGLE_API_TOKEN")
        except Exception:
            pass
    if not kaggle_key:
        try:
            kaggle_key = secrets.get_secret("KAGGLE_KEY")
        except Exception:
            pass
    try:
        sec_user = secrets.get_secret("KAGGLE_USERNAME")
        if sec_user:
            kaggle_user = sec_user
    except Exception:
        pass
except Exception:
    pass

if not kaggle_key:
    kjson = Path.home() / ".kaggle" / "kaggle.json"
    if kjson.exists():
        try:
            with open(kjson, "r") as f:
                kd = json.load(f)
                kaggle_key = kd.get("key")
                if kd.get("username"):
                    kaggle_user = kd.get("username")
        except Exception:
            pass

if not kaggle_key:
    print("❌ Fatal: Kaggle credentials missing. Set KAGGLE_API_TOKEN or KAGGLE_KEY via Kaggle Secrets or environment.")
    sys.exit(1)

os.environ["KAGGLE_USERNAME"] = kaggle_user
os.environ["KAGGLE_KEY"] = kaggle_key

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)
with open(kaggle_dir / "kaggle.json", "w") as f:
    json.dump({{"username": kaggle_user, "key": kaggle_key}}, f)
(kaggle_dir / "kaggle.json").chmod(0o600)

try:
    kaggle_ext = __import__("kaggle.api.kaggle_api_extended", fromlist=["KaggleApi"])
    KaggleApi = getattr(kaggle_ext, "KaggleApi")
    k_api = KaggleApi()
    k_api.authenticate()
    print("✅ Kaggle API authenticated successfully as", kaggle_user)
except Exception as e:
    print("❌ Failed to authenticate Kaggle API:", e)
    sys.exit(1)

from huggingface_hub import hf_hub_download, snapshot_download

# Select optimal scratch storage partition (/tmp has ~73GB free on Kaggle)
candidates = [Path("/tmp"), Path("/kaggle/working"), Path("/kaggle/temp")]
staging_root = Path("/tmp/factory_staging")
staging_root.mkdir(parents=True, exist_ok=True)
usage = shutil.disk_usage(staging_root)
print(f"📁 Staging workspace: {{staging_root}} (Free space: {{usage.free / (1024**3):.2f}} GB)\\n")

JOBS = {jobs_json}

def clean_staging():
    for item in staging_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)

for job in JOBS:
    slug = job["slug"]
    title = job["title"]
    repo = job["repo"]
    files = job.get("files", [])
    snapshot = job.get("snapshot", False)
    extra_downloads = job.get("extra_downloads", [])
    dataset_ref = f"{{kaggle_user}}/{{slug}}"

    print("=" * 80)
    print(f"📦 Processing: {{title}} ({{dataset_ref}})")
    print(f"   HuggingFace Repo: {{repo}} (Estimated: {{job.get('estimated_gb', 0)}} GB)")
    print("=" * 80)

    # Check if already exists
    already_ready = False
    try:
        cur_st = str(k_api.dataset_status(dataset_ref)).strip().lower()
        if cur_st == "ready":
            print(f"ℹ️ Dataset {{dataset_ref}} is already published and READY. Skipping.")
            continue
    except Exception:
        pass

    clean_staging()
    model_dir = staging_root / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        if snapshot:
            print(f"📥 Synchronizing repository snapshot from {{repo}}...")
            snapshot_download(repo_id=repo, local_dir=str(model_dir), ignore_patterns=["*.git*", "*.md"])
        else:
            for f_info in files:
                src = f_info["src"]
                dest = f_info["dest"]
                dest_path = model_dir / dest
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"📥 Downloading {{src}} -> {{dest}}...")
                dl_path = hf_hub_download(repo_id=repo, filename=src, local_dir=str(model_dir))
                if Path(dl_path).resolve() != dest_path.resolve():
                    shutil.move(str(dl_path), str(dest_path))

        for extra in extra_downloads:
            e_repo = extra["repo"]
            e_src = extra["src"]
            e_dest = extra["dest"]
            e_dest_path = model_dir / e_dest
            e_dest_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"📥 Extra download: {{e_repo}}/{{e_src}} -> {{e_dest}}...")
            dl_path = hf_hub_download(repo_id=e_repo, filename=e_src, local_dir=str(model_dir))
            if Path(dl_path).resolve() != e_dest_path.resolve():
                shutil.move(str(dl_path), str(e_dest_path))

        dl_time = time.time() - t0
        print(f"✅ Download completed in {{dl_time:.1f}}s.")

        # Create dataset-metadata.json
        meta = {{
            "title": title,
            "id": dataset_ref,
            "licenses": [{{"name": "apache-2.0"}}],
            "description": f"Model weights for Factory v4 Lite: {{title}}.",
        }}
        with open(model_dir / "dataset-metadata.json", "w") as mf:
            json.dump(meta, mf, indent=2)

        # Upload and publish to Kaggle Datasets
        print(f"⬆️ Uploading and publishing dataset {{dataset_ref}} to Kaggle...")
        try:
            k_api.dataset_create_new(folder=str(model_dir), public=False, quiet=False, dir_mode="skip")
            print(f"🎉 Dataset {{dataset_ref}} created successfully!")
        except Exception as up_err:
            print(f"Creating new dataset failed ({{up_err}}), attempting create_version...")
            k_api.dataset_create_version(folder=str(model_dir), version_notes="Automated Factory v4 update", quiet=False, dir_mode="skip")
            print(f"🎉 Dataset version for {{dataset_ref}} uploaded successfully!")

    except Exception as exc:
        print(f"❌ Error processing {{title}}: {{exc}}")

    clean_staging()

print("\\n🏁 All dataset download & publish jobs processed.")
"""
        return script_code

    def deploy_downloader(
        self,
        target_slugs: Optional[List[str]] = None,
        kernel_slug: str = "factory-v4-lite-model-downloader",
        poll_interval_sec: float = 10.0,
        timeout_sec: float = 7200.0,
    ) -> Dict[str, Any]:
        """Deploys remote downloader kernel to Kaggle and monitors its execution."""
        code_content = self.build_downloader_script(target_slugs=target_slugs)

        logger.info(f"Pushing remote downloader kernel '{kernel_slug}' to Kaggle...")
        push_resp = self.client.push_kernel(
            slug=kernel_slug,
            title="Factory v4 Lite Model Downloader",
            code_content=code_content,
            is_private=True,
            enable_gpu=False,  # CPU is sufficient and saves GPU quota
            enable_internet=True,
            kernel_type="script",
            language="python",
        )
        logger.info(f"Kernel pushed successfully: {push_resp.get('url')}")

        actual_slug = kernel_slug
        if push_resp.get("ref"):
            actual_slug = push_resp["ref"].strip("/").split("/")[-1]
        elif push_resp.get("url"):
            actual_slug = push_resp["url"].strip("/").split("/")[-1]

        def _log_streamer(line: str):
            logger.info(f"[Kaggle Downloader] {line}")

        output = self.client.poll_until_complete(
            slug=actual_slug,
            poll_interval_sec=poll_interval_sec,
            timeout_sec=timeout_sec,
            output_callback=_log_streamer,
        )
        return output


def main():
    parser = argparse.ArgumentParser(description="Factory v4 Lite Kaggle Dataset Preparation Manager")
    parser.add_argument("--status", action="store_true", help="Inspect status of all 10 model datasets on Kaggle")
    parser.add_argument("--deploy", action="store_true", help="Deploy remote downloader kernel to Kaggle")
    parser.add_argument("--models", nargs="*", default=None, help="Specific model IDs or slugs to prepare")
    parser.add_argument("--slug", default="factory-v4-lite-model-downloader", help="Kaggle kernel slug for downloader")
    parser.add_argument("--user", default=None, help="Kaggle username")
    parser.add_argument("--token", default=None, help="Kaggle Bearer token")
    args = parser.parse_args()

    mgr = DatasetPreparationManager(username=args.user, api_key=args.token)

    if args.status or not args.deploy:
        print("\n" + "=" * 90)
        print("📊 FACTORY V4 LITE MODEL DATASETS STATUS (KAGGLE)")
        print("=" * 90)
        items = mgr.inspect_all_datasets()
        print(f"{'Model ID':<20} | {'Service ID':<20} | {'Kaggle Dataset Slug':<28} | {'Status':<12} | {'Est GB'}")
        print("-" * 90)
        ready_count = 0
        for it in items:
            st = it["status"]
            ready_str = "READY ✅" if st == "ready" else ("EXISTS ⏳" if it["exists"] else "NOT FOUND ❌")
            if st == "ready":
                ready_count += 1
            print(f"{it['model_id']:<20} | {it['service_id']:<20} | {it['slug']:<28} | {ready_str:<12} | {it['estimated_gb']} GB")
        print("-" * 90)
        print(f"Total: {len(items)} models | Ready on Kaggle: {ready_count}/{len(items)}\n")

    if args.deploy:
        print(f"🚀 Deploying remote model downloader kernel ({args.slug}) to Kaggle...")
        mgr.deploy_downloader(target_slugs=args.models, kernel_slug=args.slug)


if __name__ == "__main__":
    main()
