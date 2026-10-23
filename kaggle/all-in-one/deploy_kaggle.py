#!/usr/bin/env python3
"""
🚀 Kaggle Automated Deployment & Session Manager
Tự động đóng gói chọn lọc theo Mode (Selective Packing), đẩy notebook lên Kaggle
thông qua Kaggle REST API, theo dõi tiến trình thực thi của Kernel và trích xuất URL Cloudflare Public.
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from typing import Optional


def load_env_file():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    env_paths = [
        os.path.join(curr_dir, ".env"),
        os.path.join(os.path.dirname(os.path.dirname(curr_dir)), ".env"),
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
                            os.environ[k] = v
            except Exception:
                pass


load_env_file()

DEFAULT_KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "")
DEFAULT_KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")
KERNEL_SLUG = "kaggle-all-in-one-studio"

ROOT_RUNTIME_FILES = {"main.py", "config.py", "models.yaml", "requirements.txt"}
BASE_RUNTIME_DIRS = {"core", "server", "tunnel", "configs"}
MODE_RUNTIME_DIRS = {
    "speech": {"audio"},
    "vlm": {"vlm"},
    "image_video": {"visual"},
    "visual": {"visual"},
    "embedding": {"embedding"},
}


def get_auth_header(user: str, key: str) -> str:
    if key.startswith("KGAT_"):
        return f"Bearer {key}"
    cred = f"{user}:{key}".encode("utf-8")
    return f"Basic {base64.b64encode(cred).decode('ascii')}"


def infer_mode_from_notebook(notebook_path: str, explicit_mode: Optional[str] = None) -> str:
    if explicit_mode:
        m = explicit_mode.strip().lower()
        if m in ("visual",):
            return "image_video"
        if m in ("speech", "vlm", "image_video", "embedding"):
            return m
    nb_base = os.path.basename(notebook_path).lower()
    if "speech" in nb_base:
        return "speech"
    if "vlm" in nb_base:
        return "vlm"
    if "image_video" in nb_base or "visual" in nb_base:
        return "image_video"
    if "embedding" in nb_base or "embed" in nb_base:
        return "embedding"
    return "speech"


def pack_and_update_run_notebook(notebook_path: str = "run_speech.ipynb", mode: Optional[str] = None) -> bool:
    """
    Đóng gói chọn lọc (Selective Mode-Aware Packing) các module runtime cần thiết thành tar.gz
    và nhúng vào cell payload_data của notebook đích.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(notebook_path) and not os.path.exists(notebook_path):
        cand = os.path.join(curr_dir, notebook_path)
        if os.path.exists(cand):
            notebook_path = cand

    eff_mode = infer_mode_from_notebook(notebook_path, mode)
    allowed_dirs = set(BASE_RUNTIME_DIRS) | MODE_RUNTIME_DIRS.get(eff_mode, set())

    buf = io.BytesIO()
    packed_count = 0

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # 1. Đóng gói các file runtime gốc
        for rf in sorted(ROOT_RUNTIME_FILES):
            full_p = os.path.join(curr_dir, rf)
            if os.path.isfile(full_p):
                tar.add(full_p, arcname=rf)
                packed_count += 1

        # 2. Đóng gói các thư mục module theo đúng mode
        for d_name in sorted(allowed_dirs):
            dir_path = os.path.join(curr_dir, d_name)
            if not os.path.isdir(dir_path):
                continue
            for root, dirs, files in os.walk(dir_path):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", ".ipynb_checkpoints")]
                for f in sorted(files):
                    if f.endswith((".py", ".yaml", ".html")) or (d_name == "audio" and f == "default_speaker.wav"):
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, curr_dir)
                        tar.add(full_p, arcname=rel_p)
                        packed_count += 1

    tar_bytes = buf.getvalue()
    b64_payload = base64.b64encode(tar_bytes).decode("ascii")
    print(
        f"📦 [{eff_mode.upper()}] Đã đóng gói chọn lọc {packed_count} files "
        f"({len(tar_bytes):,} bytes | base64: {len(b64_payload):,} chars)"
    )

    if not os.path.exists(notebook_path):
        print(f"⚠️ Không tìm thấy file {notebook_path} để nhúng payload.")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src_str = "".join(src) if isinstance(src, list) else str(src)

        if "payload_data =" in src_str:
            new_src_str = re.sub(r'payload_data\s*=\s*"[^"]*"', f'payload_data = "{b64_payload}"', src_str)
            cell["source"] = new_src_str
            updated = True

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✅ Đã cập nhật live payload mới nhất vào {os.path.basename(notebook_path)}!")
    else:
        print(f"⚠️ Không tìm thấy cell chứa 'payload_data =' trong {notebook_path}.")
    return updated


def push_kernel(
    user: str,
    key: str,
    notebook_path: str = "run_speech.ipynb",
    metadata_path: str = "kernel-speech-metadata.json",
    slug: str = "kaggle-speech-ai-studio",
    mode: Optional[str] = None,
) -> bool:
    """Đẩy notebook lên Kaggle qua REST API."""
    print("=" * 72)
    print(f"🚀 [1/3] ĐẨY NOTEBOOK LÊN KAGGLE API ({user}/{slug})...")
    print("=" * 72)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(notebook_path) and not os.path.exists(notebook_path):
        candidate_nb = os.path.join(curr_dir, notebook_path)
        if os.path.exists(candidate_nb):
            notebook_path = candidate_nb

    if not os.path.isabs(metadata_path) and not os.path.exists(metadata_path):
        candidate_meta = os.path.join(curr_dir, metadata_path)
        if os.path.exists(candidate_meta):
            metadata_path = candidate_meta

    nb_basename = os.path.basename(notebook_path)
    if nb_basename in ("run_speech.ipynb", "run_vlm.ipynb", "run_image_video.ipynb", "run_embedding.ipynb"):
        pack_and_update_run_notebook(notebook_path, mode=mode)

    if not os.path.exists(notebook_path):
        print(f"❌ Không tìm thấy file notebook: {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_content = f.read()

    meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    title = meta.get("title", "")
    if slug == KERNEL_SLUG:
        title = "Kaggle All in One Studio"
    elif not title:
        if slug == "kaggle-wheels-builder":
            title = "Kaggle Wheels Builder"
        else:
            title = "Kaggle AI Studio"
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
        "sessionTimeoutSeconds": max(7200, int(os.environ.get("IDLE_SHUTDOWN_MINUTES", "120")) * 60),
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
            print(f"🔗 URL Kernel: {data.get('url', f'https://www.kaggle.com/code/{user}/{slug}')}")
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


def monitor_and_extract_url(
    user: str,
    key: str,
    slug: str = KERNEL_SLUG,
    timeout_seconds: int = 900,
    ignore_url: Optional[str] = None,
) -> str:
    """Theo dõi kernel chạy trên Kaggle và trích xuất URL Cloudflare mới."""
    print("\n" + "=" * 72)
    print(f"⏳ [2/3] THEO DÕI PHIÊN KAGGLE ({user}/{slug}) & BẮT URL CLOUDFLARE PUBLIC...")
    if ignore_url:
        print(f"   (Bỏ qua URL cũ: {ignore_url})")
    print(f"   Thời gian chờ tối đa: {timeout_seconds}s ({timeout_seconds // 60} phút)")
    print("=" * 72)

    deadline = time.time() + timeout_seconds
    last_status = None
    seen_lines = set()
    public_url = None

    while time.time() < deadline:
        status_data = get_kernel_status(user, key, slug=slug)
        cur_status = status_data.get("status")

        if cur_status != last_status:
            print(
                f"[{time.strftime('%H:%M:%S')}] 🔄 Trạng thái Kernel: {str(cur_status).upper()} "
                f"(Failure message: {status_data.get('failureMessage') or 'None'})",
                flush=True,
            )
            last_status = cur_status

        # 📡 Kiểm tra kênh phát sóng ntfy.sh để bắt URL tức thì
        for ntfy_topic in sorted(set([
            os.environ.get("NTFY_TOPIC", f"studio-ai-url-{user}"),
            f"studio-ai-url-{user}",
            "studio-ai-url-cuongnguyen1802",
            "studio-ai-url-nguynxuncngde180528",
        ])):
            if public_url:
                break
            try:
                ntfy_check_req = urllib.request.Request(f"https://ntfy.sh/{ntfy_topic}/raw?poll=1&since=15m")
                with urllib.request.urlopen(ntfy_check_req, timeout=5) as ntfy_resp:
                    content = ntfy_resp.read().decode("utf-8").strip()
                    if content:
                        for cline in reversed(content.splitlines()):
                            match_n = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", cline)
                            if match_n:
                                cand_url = match_n.group(0)
                                if ignore_url and cand_url == ignore_url:
                                    continue
                                try:
                                    with urllib.request.urlopen(f"{cand_url}/health", timeout=3) as h_resp:
                                        if h_resp.status == 200:
                                            public_url = cand_url
                                            print(f"[{time.strftime('%H:%M:%S')}] 📡 Đã xác nhận Cloudflare URL online: {public_url}", flush=True)
                                            break
                                except Exception:
                                    pass
            except Exception:
                pass

        # 📡 In các sự kiện debug từ kernel qua ntfy.sh
        try:
            dbg_topic = f"studio-ai-debug-{user}"
            with urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{dbg_topic}/raw?poll=1&since=10m"), timeout=3) as dbg_resp:
                dbg_content = dbg_resp.read().decode("utf-8").strip()
                if dbg_content:
                    for dline in dbg_content.splitlines():
                        if dline.strip() and dline not in seen_lines:
                            seen_lines.add(dline)
                            print(f"   📡 [Kaggle Debug] {dline.strip()}", flush=True)
        except Exception:
            pass

        if public_url:
            break

        if cur_status in ("running", "complete", "queued"):
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
                    if any(
                        kw in line
                        for kw in [
                            "CLOUDFLARE",
                            "trycloudflare.com",
                            "Uvicorn running",
                            "KHỞI ĐỘNG",
                            "GPU",
                            "VRAM",
                            "Warmup",
                            "nạp thành công",
                            "Fast-Swap",
                            "Đã nạp",
                            "Public Base URL",
                        ]
                    ):
                        print(f"   📝 [Kaggle Log] {line.strip()}")

                match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
                if match:
                    found_cand = match.group(0)
                    if not (ignore_url and found_cand == ignore_url):
                        try:
                            with urllib.request.urlopen(f"{found_cand}/health", timeout=3) as h_resp:
                                if h_resp.status == 200:
                                    public_url = found_cand
                        except Exception:
                            pass

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
        try:
            url_target = os.path.join(os.path.dirname(os.path.dirname(curr_dir)), "current_url.txt")
            with open(url_target, "w", encoding="utf-8") as f:
                f.write(public_url + "\n")
            print(f"💾 Đã tự động cập nhật current_url.txt -> {public_url}")
        except Exception as e:
            print(f"⚠️ Không thể ghi current_url.txt: {e}")
        return full_api_url
    else:
        print("⚠️ Chưa bắt được URL trong thời gian chờ hoặc Kernel đang khởi động.")
        return ""


def main():
    parser = argparse.ArgumentParser(description="🚀 Kaggle Automated Deployer")
    parser.add_argument(
        "action",
        nargs="?",
        default="",
        help="Phân hệ triển khai nhanh: speech | vlm | image_video | embedding | wheels | downloader",
    )
    parser.add_argument("--user", default=DEFAULT_KAGGLE_USER, help="Kaggle username")
    parser.add_argument("--key", default=DEFAULT_KAGGLE_KEY, help="Kaggle API Key")
    parser.add_argument("--slug", default="", help="Kaggle kernel slug")
    parser.add_argument(
        "--target",
        "--mode",
        default="",
        choices=["", "speech", "vlm", "image_video", "visual", "embedding", "embed", "wheels", "downloader"],
        help="Phân hệ triển khai: speech | vlm | image_video | embedding",
    )
    parser.add_argument("--push-only", action="store_true", help="Chỉ đẩy kernel, không đợi")
    parser.add_argument("--monitor-only", action="store_true", help="Chỉ theo dõi kernel đang chạy và bắt URL")
    parser.add_argument("--status-only", action="store_true", help="Chỉ kiểm tra trạng thái")
    parser.add_argument("--build-wheels", action="store_true", help="Build và đẩy Kaggle Wheels Builder kernel")
    parser.add_argument("--pack-all", action="store_true", help="Chỉ đóng gói lại payload cho toàn bộ 4 file run_*.ipynb")
    parser.add_argument("--notebook", default="", help="Notebook file path")
    parser.add_argument("--meta", default="", help="Metadata file path")
    parser.add_argument("--timeout", type=int, default=900, help="Thời gian chờ tối đa (giây)")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    if args.pack_all:
        for nb_file, m_name in [
            ("run_speech.ipynb", "speech"),
            ("run_vlm.ipynb", "vlm"),
            ("run_image_video.ipynb", "image_video"),
            ("run_embedding.ipynb", "embedding"),
        ]:
            pack_and_update_run_notebook(os.path.join(curr_dir, nb_file), mode=m_name)
        return

    target = (args.target or args.action or os.environ.get("STUDIO_MODE", "speech")).strip().lower()
    if target == "wheels":
        args.build_wheels = True
    elif target == "downloader":
        dl_script = os.path.join(curr_dir, "deploy_downloader.py")
        cmd = [sys.executable, dl_script]
        if args.push_only:
            cmd.append("--push-only")
        if args.status_only:
            cmd.append("--status-only")
        subprocess.run(cmd, check=True)
        return

    if target == "speech":
        if not args.notebook:
            args.notebook = "run_speech.ipynb"
        if not args.meta:
            args.meta = "kernel-speech-metadata.json"
        if not args.slug:
            args.slug = KERNEL_SLUG
    elif target == "vlm":
        if not args.notebook:
            args.notebook = "run_vlm.ipynb"
        if not args.meta:
            args.meta = "kernel-vlm-metadata.json"
        if not args.slug:
            args.slug = KERNEL_SLUG
    elif target in ("image_video", "visual"):
        target = "image_video"
        if not args.notebook:
            args.notebook = "run_image_video.ipynb"
        if not args.meta:
            args.meta = "kernel-image-video-metadata.json"
        if not args.slug:
            args.slug = KERNEL_SLUG
    elif target in ("embedding", "embed"):
        target = "embedding"
        if not args.notebook:
            args.notebook = "run_embedding.ipynb"
        if not args.meta:
            args.meta = "kernel-embedding-metadata.json"
        if not args.slug:
            args.slug = KERNEL_SLUG
    else:
        print(f"❌ Chế độ '{target}' không hợp lệ hoặc luồng All đã bị loại bỏ.")
        print("💡 Vui lòng chọn một trong các phân hệ độc lập: speech | vlm | image_video | embedding")
        sys.exit(1)

    if args.status_only:
        slug = "kaggle-wheels-builder" if args.build_wheels else args.slug
        st = get_kernel_status(args.user, args.key, slug=slug)
        print(f"Kernel Status ({slug}):", json.dumps(st, indent=2))
        return

    if args.build_wheels:
        print("🔨 [Wheels Builder] Đang tái tạo run_wheels_builder.ipynb...")
        subprocess.run([sys.executable, os.path.join(curr_dir, "build_wheels_notebook.py")], cwd=curr_dir, check=True)
        ok = push_kernel(
            args.user,
            args.key,
            notebook_path=os.path.join(curr_dir, "run_wheels_builder.ipynb"),
            metadata_path=os.path.join(curr_dir, "wheels-metadata.json"),
            slug="kaggle-wheels-builder",
        )
        if not ok:
            sys.exit(1)
        if not args.push_only:
            print("\n⏳ Đang theo dõi tiến trình đóng gói Wheels trên Kaggle...")
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                st = get_kernel_status(args.user, args.key, slug="kaggle-wheels-builder")
                status = st.get("status")
                print(f"[{time.strftime('%H:%M:%S')}] 🔄 Wheels Builder Status: {status}")
                if status == "complete":
                    print("🎉 Kaggle Studio Wheels dataset đã được build và xuất bản thành công!")
                    break
                elif status == "error":
                    print(f"❌ Lỗi khi build wheels: {st.get('failureMessage')}")
                    sys.exit(1)
                time.sleep(15)
        return

    if args.monitor_only:
        url = monitor_and_extract_url(args.user, args.key, slug=args.slug, timeout_seconds=args.timeout)
        if not url:
            sys.exit(2)
        return

    ok = push_kernel(
        args.user,
        args.key,
        notebook_path=args.notebook,
        metadata_path=args.meta,
        slug=args.slug,
        mode=target,
    )
    if not ok:
        sys.exit(1)

    if not args.push_only:
        url = monitor_and_extract_url(args.user, args.key, slug=args.slug, timeout_seconds=args.timeout)
        if not url:
            sys.exit(2)


if __name__ == "__main__":
    main()
