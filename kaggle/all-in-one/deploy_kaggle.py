#!/usr/bin/env python3
"""
🚀 Kaggle Automated Deployment & Session Manager
Tự động đóng gói, đẩy notebook lên Kaggle thông qua Kaggle REST API,
theo dõi tiến trình thực thi của Kernel và trích xuất URL Cloudflare Public.
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

DEFAULT_KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "nguynxuncngde180528")
DEFAULT_KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "KGAT_8cf30e03c2129179e5e0870f50b86773")
KERNEL_SLUG = "kaggle-all-in-one-studio"


def get_auth_header(user: str, key: str) -> str:
    if key.startswith("KGAT_"):
        return f"Bearer {key}"
    cred = f"{user}:{key}".encode("utf-8")
    return f"Basic {base64.b64encode(cred).decode('ascii')}"


def pack_and_update_run_notebook(notebook_path: str = "run_kaggle.ipynb") -> bool:
    """
    Tự động đóng gói toàn bộ mã nguồn python, yaml, config hiện hành thành tar.gz
    và nhúng vào cell payload_data của run_kaggle.ipynb.
    Đảm bảo 100% code mới nhất luôn được nạp trên Kaggle mà không bị dính cache cũ.
    """
    import io
    import tarfile

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    buf = io.BytesIO()

    packed_count = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(curr_dir):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "test_outputs", "checkpoints", ".ipynb_checkpoints")]
            for f in files:
                if (f.endswith((".py", ".yaml", ".txt")) or f == ".env") and not f.startswith("deploy_") and not f.startswith("build_") and f != "public_url.txt":
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, curr_dir)
                    tar.add(full_p, arcname=rel_p)
                    packed_count += 1

    tar_bytes = buf.getvalue()
    b64_payload = base64.b64encode(tar_bytes).decode("ascii")
    print(f"📦 Đã đóng gói {packed_count} files thành Live Payload tar.gz ({len(tar_bytes):,} bytes | base64: {len(b64_payload):,} chars)")

    if not os.path.exists(notebook_path):
        print(f"⚠️ Không tìm thấy file {notebook_path} để nhúng payload.")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src_str = "".join(src)
        else:
            src_str = str(src)

        if "payload_data =" in src_str:
            new_src_str = re.sub(r'payload_data\s*=\s*"[^"]*"', f'payload_data = "{b64_payload}"', src_str)
            cell["source"] = new_src_str
            updated = True
            break

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✅ Đã cập nhật live payload mới nhất vào {notebook_path}!")
    else:
        print(f"⚠️ Không tìm thấy cell chứa 'payload_data =' trong {notebook_path}.")
    return updated


def push_kernel(user: str, key: str, notebook_path: str = "run_kaggle.ipynb", metadata_path: str = "kernel-metadata.json", slug: str = KERNEL_SLUG) -> bool:
    """Đẩy notebook lên Kaggle qua REST API."""
    print("=" * 72)
    print(f"🚀 [1/3] ĐẨY NOTEBOOK LÊN KAGGLE API ({user}/{slug})...")
    print("=" * 72)

    # 1. Đóng gói mã nguồn mới nhất vào notebook trước khi đẩy
    pack_and_update_run_notebook(notebook_path)

    if not os.path.exists(notebook_path):
        print(f"❌ Không tìm thấy file notebook: {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_content = f.read()

    meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    title = meta.get("title", "Kaggle All in One Studio")
    if slug != KERNEL_SLUG:
        title = f"{title} V2"
    is_private = meta.get("is_private", "true")
    if isinstance(is_private, str):
        is_private = is_private.lower() == "true"

    kernel_type = meta.get("kernel_type", "notebook")
    if kernel_type == "notebook":
        try:
            nb_json = json.loads(nb_content)
            if "cells" in nb_json:
                for cell in nb_json["cells"]:
                    if "outputs" in cell and cell.get("cell_type") == "code":
                        cell["outputs"] = []
                    if "source" in cell and isinstance(cell["source"], list):
                        cell["source"] = "".join(cell["source"])
            nb_content = json.dumps(nb_json)
        except Exception as e:
            print(f"⚠️ Cảnh báo khi format notebook: {e}")

    payload = {
        "slug": f"{user}/{slug}",
        "newTitle": title,
        "text": nb_content,
        "language": meta.get("language", "python"),
        "kernelType": kernel_type,
        "isPrivate": is_private,
        "enableGpu": True,
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
            print(f"✅ Đẩy Kernel thành công! Version: {v_num}")
            print(f"🔗 URL Kernel: {data.get('url', f'https://www.kaggle.com/code/{user}/{KERNEL_SLUG}')}")
            return True

    except urllib.error.HTTPError as he:
        body = he.read().decode("utf-8", errors="replace")
        print(f"❌ Lỗi HTTP khi đẩy Kernel ({he.code}): {body}")
        return False
    except Exception as e:
        print(f"❌ Lỗi kết nối Kaggle API: {e}")
        return False


def get_kernel_status(user: str, key: str, slug: str = KERNEL_SLUG) -> dict:
    """Lấy trạng thái thực thi hiện tại của kernel."""
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName={user}&kernelSlug={slug}"
    headers = {"Authorization": get_auth_header(user, key)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


def get_kernel_output(user: str, key: str, slug: str = KERNEL_SLUG) -> dict:
    """Lấy stdout / logs của kernel đang chạy."""
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName={user}&kernelSlug={slug}"
    headers = {"Authorization": get_auth_header(user, key)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def monitor_and_extract_url(user: str, key: str, slug: str = KERNEL_SLUG, timeout_seconds: int = 900) -> str:
    """Theo dõi kernel chạy trên Kaggle và trích xuất URL Cloudflare."""
    print("\n" + "=" * 72)
    print(f"⏳ [2/3] THEO DÕI PHIÊN KAGGLE ({user}/{slug}) & BẮT URL CLOUDFLARE PUBLIC...")
    print(f"   Thời gian chờ tối đa: {timeout_seconds}s (15 phút)")
    print("=" * 72)

    deadline = time.time() + timeout_seconds
    last_status = None
    seen_lines = set()
    public_url = None

    while time.time() < deadline:
        status_data = get_kernel_status(user, key, slug=slug)
        cur_status = status_data.get("status")

        if cur_status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] 🔄 Trạng thái Kernel: {cur_status.upper()} (Failure message: {status_data.get('failureMessage') or 'None'})")
            last_status = cur_status

        # 📡 Kiểm tra kênh phát sóng ntfy.sh để bắt URL tức thì (bỏ qua trễ log Kaggle)
        try:
            ntfy_topic = os.environ.get("NTFY_TOPIC", f"studio-ai-url-{user}")
            ntfy_check_req = urllib.request.Request(f"https://ntfy.sh/{ntfy_topic}/raw?poll=1")
            with urllib.request.urlopen(ntfy_check_req, timeout=5) as ntfy_resp:
                content = ntfy_resp.read().decode("utf-8").strip()
                if content:
                    # Duyệt từ dòng mới nhất (cuối cùng) ngược lên
                    for cline in reversed(content.splitlines()):
                        match_n = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", cline)
                        if match_n:
                            cand_url = match_n.group(0)
                            # Kiểm tra xem URL này có đang online thật không
                            try:
                                with urllib.request.urlopen(f"{cand_url}/health", timeout=3) as h_resp:
                                    if h_resp.status == 200:
                                        public_url = cand_url
                                        print(f"[{time.strftime('%H:%M:%S')}] 📡 Đã xác nhận Cloudflare URL online: {public_url}")
                                        break
                            except Exception:
                                pass
        except Exception:
            pass

        if public_url:
            break

        if cur_status in ("running", "complete", "queued"):
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
                    if any(kw in line for kw in ["CLOUDFLARE", "trycloudflare.com", "Uvicorn running", "KHỞI ĐỘNG", "GPU", "VRAM", "Warmup", "nạp thành công", "Fast-Swap", "Đã nạp", "Public Base URL"]):
                        print(f"   📝 [Kaggle Log] {line.strip()}")

                match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
                if match:
                    public_url = match.group(0)

            if public_url:
                break

        if cur_status == "error":
            print(f"❌ Kernel báo lỗi dừng thực thi: {status_data.get('failureMessage')}")
            break

        time.sleep(10.0)

    if public_url:
        full_api_url = f"{public_url}/v1"
        print("\n" + "=" * 72)
        print("🎉 [3/3] CLOUDFLARE QUICK TUNNEL ĐÃ SẴN SÀNG!")
        print(f"👉 API Base URL : {full_api_url}")
        print("=" * 72 + "\n")

        with open("public_url.txt", "w") as f:
            f.write(f"{full_api_url}\n")
        return full_api_url
    else:
        print("⚠️ Chưa bắt được URL trong thời gian chờ hoặc Kernel đang khởi động.")
        return ""


def main():
    parser = argparse.ArgumentParser(description="🚀 Kaggle Automated Deployer")
    parser.add_argument("--user", default=DEFAULT_KAGGLE_USER, help="Kaggle username")
    parser.add_argument("--key", default=DEFAULT_KAGGLE_KEY, help="Kaggle API Key")
    parser.add_argument("--slug", default=KERNEL_SLUG, help="Kaggle kernel slug")
    parser.add_argument("--push-only", action="store_true", help="Chỉ đẩy kernel, không đợi")
    parser.add_argument("--monitor-only", action="store_true", help="Chỉ theo dõi kernel đang chạy và bắt URL")
    parser.add_argument("--status-only", action="store_true", help="Chỉ kiểm tra trạng thái")
    parser.add_argument("--timeout", type=int, default=900, help="Thời gian chờ tối đa (giây)")
    args = parser.parse_args()

    if args.status_only:
        st = get_kernel_status(args.user, args.key, slug=args.slug)
        print(f"Kernel Status ({args.slug}):", json.dumps(st, indent=2))
        return

    if args.monitor_only:
        url = monitor_and_extract_url(args.user, args.key, slug=args.slug, timeout_seconds=args.timeout)
        if not url:
            sys.exit(2)
        return

    ok = push_kernel(args.user, args.key, slug=args.slug)
    if not ok:
        sys.exit(1)

    if not args.push_only:
        url = monitor_and_extract_url(args.user, args.key, slug=args.slug, timeout_seconds=args.timeout)
        if not url:
            sys.exit(2)


if __name__ == "__main__":
    main()
