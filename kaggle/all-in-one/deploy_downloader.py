#!/usr/bin/env python3
"""
🚀 Kaggle Model Downloader Deployer & Job Monitor
================================================================================
Tự động đẩy notebook tải model lên Kaggle qua Kaggle REST API, theo dõi tiến trình
tải và xuất bản dataset thời gian thực, đồng thời xác thực kết quả sau khi hoàn tất.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

# Nạp biến môi trường từ .env
def load_env_file():
    env_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ or not os.environ[k]:
                                os.environ[k] = v
            except Exception:
                pass

load_env_file()

DEFAULT_KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "")
DEFAULT_KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")
DOWNLOADER_SLUG = "kaggle-model-downloader"


def get_auth_header(user: str, key: str) -> str:
    if key.startswith("KGAT_"):
        return f"Bearer {key}"
    cred = f"{user}:{key}".encode("utf-8")
    return f"Basic {base64.b64encode(cred).decode('ascii')}"


def push_downloader_kernel(user: str, key: str, slug: str = DOWNLOADER_SLUG) -> bool:
    """Đóng gói và đẩy notebook tải model lên Kaggle qua REST API."""
    print("=" * 76)
    print(f"🚀 [1/3] ĐẨY NOTEBOOK DOWNLOADER LÊN KAGGLE ({user}/{slug})...")
    print("=" * 76)

    # 1. Sinh lại notebook và metadata mới nhất
    try:
        from build_downloader_notebook import build_notebook
        build_notebook()
    except Exception as e:
        print(f"⚠️ Không thể chạy build_downloader_notebook: {e}")

    notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_downloader.ipynb")
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloader-metadata.json")

    if not os.path.exists(notebook_path):
        print(f"❌ Không tìm thấy file notebook: {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_content = f.read()

    meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    title = meta.get("title", "Kaggle Model Downloader")
    is_private = meta.get("is_private", "true")
    if isinstance(is_private, str):
        is_private = is_private.lower() == "true"

    payload = {
        "slug": f"{user}/{slug}",
        "newTitle": title,
        "text": nb_content,
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "notebook"),
        "isPrivate": is_private,
        "enableGpu": False,
        "enableInternet": True,
        "datasetDataSources": meta.get("dataset_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
        "modelDataSources": meta.get("model_sources", []),
    }

    url = "https://www.kaggle.com/api/v1/kernels/push"
    headers = {
        "Authorization": get_auth_header(user, key),
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("hasError"):
                print(f"❌ Lỗi từ Kaggle API: {data.get('error') or data.get('errorNullable')}")
                return False
            v_num = data.get("versionNumberNullable") or data.get("versionNumber")
            print(f"✅ Đẩy Downloader Kernel thành công! Version: {v_num}")
            print(f"🔗 URL Kernel: {data.get('url', f'https://www.kaggle.com/code/{user}/{slug}')}")
            return True

    except urllib.error.HTTPError as he:
        body = he.read().decode("utf-8", errors="replace")
        print(f"❌ Lỗi HTTP khi đẩy Kernel ({he.code}): {body}")
        return False
    except Exception as e:
        print(f"❌ Lỗi kết nối Kaggle API: {e}")
        return False


def get_kernel_status(user: str, key: str, slug: str = DOWNLOADER_SLUG) -> dict:
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName={user}&kernelSlug={slug}"
    headers = {"Authorization": get_auth_header(user, key)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


def get_kernel_output(user: str, key: str, slug: str = DOWNLOADER_SLUG) -> dict:
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName={user}&kernelSlug={slug}"
    headers = {"Authorization": get_auth_header(user, key)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def list_user_datasets(user: str, key: str):
    """Liệt kê danh sách các dataset hiện có của user trên Kaggle."""
    url = f"https://www.kaggle.com/api/v1/datasets/list?user={user}"
    headers = {"Authorization": get_auth_header(user, key)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ Không thể lấy danh sách dataset: {e}")
        return []


def monitor_downloader_kernel(user: str, key: str, slug: str = DOWNLOADER_SLUG, timeout_seconds: int = 7200) -> bool:
    """Theo dõi tiến trình thực thi kernel tải model và in logs thời gian thực."""
    print("\n" + "=" * 76)
    print(f"⏳ [2/3] THEO DÕI TIẾN TRÌNH TẢI & TẠO DATASET ({user}/{slug})...")
    print(f"   Thời gian chờ tối đa: {timeout_seconds // 60} phút")
    print("=" * 76)

    deadline = time.time() + timeout_seconds
    last_status = None
    seen_lines = set()

    while time.time() < deadline:
        status_data = get_kernel_status(user, key, slug=slug)
        cur_status = status_data.get("status")

        if cur_status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] 🔄 Trạng thái Kernel: {cur_status.upper()} (Failure msg: {status_data.get('failureMessage') or 'None'})")
            last_status = cur_status

        # Đọc output logs
        output_data = get_kernel_output(user, key, slug=slug)
        raw_log = output_data.get("log", "")
        extracted_lines = []
        if raw_log:
            try:
                items = json.loads(raw_log)
                if isinstance(items, list):
                    for it in items:
                        d = it.get("data", "")
                        if d:
                            extracted_lines.extend(d.splitlines())
                else:
                    extracted_lines.extend(raw_log.splitlines())
            except Exception:
                extracted_lines.extend(raw_log.splitlines())

        for line in extracted_lines:
            if line not in seen_lines and line.strip():
                seen_lines.add(line)
                print(f"   📝 [Kaggle Output] {line.strip()}")

        if cur_status == "complete":
            print("\n" + "=" * 76)
            print("🎉 [3/3] KERNEL ĐÃ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH TẢI & TẠO DATASET!")
            print("=" * 76)
            return True

        if cur_status == "error":
            print("\n" + "=" * 76)
            print(f"❌ Kernel kết thúc với lỗi: {status_data.get('failureMessage')}")
            print("=" * 76)
            return False

        time.sleep(15.0)

    print("⚠️ Hết thời gian chờ giám sát (timeout).")
    return False


def main():
    parser = argparse.ArgumentParser(description="🚀 Kaggle Model Downloader Deployer")
    parser.add_argument("--user", default=DEFAULT_KAGGLE_USER, help="Kaggle username")
    parser.add_argument("--key", default=DEFAULT_KAGGLE_KEY, help="Kaggle API Key")
    parser.add_argument("--slug", default=DOWNLOADER_SLUG, help="Kernel slug")
    parser.add_argument("--push-only", action="store_true", help="Chỉ đẩy kernel, không đợi")
    parser.add_argument("--monitor-only", action="store_true", help="Chỉ theo dõi kernel")
    parser.add_argument("--list-datasets", action="store_true", help="Liệt kê danh sách dataset hiện có")
    parser.add_argument("--timeout", type=int, default=7200, help="Thời gian chờ tối đa (giây)")
    args = parser.parse_args()

    if args.list_datasets:
        datasets = list_user_datasets(args.user, args.key)
        print(f"\n📦 Danh sách Kaggle Datasets của {args.user} ({len(datasets)} items):")
        for d in datasets:
            size_mb = (d.get("totalBytes") or 0) / (1024 ** 2)
            print(f"  - {d.get('ref')}: {d.get('title')} ({size_mb:.1f} MB)")
            print(f"    URL: https://www.kaggle.com/datasets/{d.get('ref')}")
        return

    if args.monitor_only:
        ok = monitor_downloader_kernel(args.user, args.key, slug=args.slug, timeout_seconds=args.timeout)
        if not ok:
            sys.exit(1)
        return

    ok = push_downloader_kernel(args.user, args.key, slug=args.slug)
    if not ok:
        sys.exit(1)

    if not args.push_only:
        ok = monitor_downloader_kernel(args.user, args.key, slug=args.slug, timeout_seconds=args.timeout)
        
        # Sau khi hoàn tất, hiển thị danh sách dataset mới
        print("\n🔍 Kiểm tra danh sách Dataset sau khi chạy:")
        datasets = list_user_datasets(args.user, args.key)
        for d in datasets:
            print(f"  ✨ https://www.kaggle.com/datasets/{d.get('ref')} ({d.get('title')})")


if __name__ == "__main__":
    main()
