#!/usr/bin/env python3
"""
🎬 Auto Render Pipeline & Master Film Assembler
Tự động theo dõi Kaggle GPU / Cloudflare Tunnel và lập tức kích hoạt:
1. Render Hồi 2 - Cảnh 5: "Cầu Rễ Cây & Thử Thách Vực Pha Lê"
2. Render Hồi 1 - Cảnh 3: "Khởi Hành Trong Ánh Hoàng Hôn"
3. Tự động kiểm tra chất lượng và ghép thành Master Chapter Video.
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/Volumes/WorkSpace/Project/Gen_Image:Video")
URL_FILE = BASE_DIR / "current_url.txt"
SCRIPTS_DIR = BASE_DIR / "scripts"

sys.path.insert(0, str(BASE_DIR / "kaggle" / "all-in-one"))
try:
    import deploy_kaggle
except Exception:
    deploy_kaggle = None


def check_url_online(url: str) -> bool:
    if not url:
        return False
    url = url.strip().rstrip("/")
    for endpoint in ["/health", "/system_stats"]:
        try:
            with urllib.request.urlopen(f"{url}{endpoint}", timeout=5) as r:
                if r.status in (200, 404):
                    return True
        except Exception:
            pass
    return False


def wait_for_active_server(max_wait_seconds: int = 3600) -> str:
    print("⏳ [Pipeline Listener] Đang theo dõi máy chủ Kaggle & ntfy.sh...")
    deadline = time.time() + max_wait_seconds

    # 1. Kiểm tra URL sẵn có trong file
    if URL_FILE.exists():
        existing_url = URL_FILE.read_text().strip()
        if check_url_online(existing_url):
            print(f"🎉 Máy chủ hiện tại đang ONLINE: {existing_url}")
            return existing_url

    seen_status = {}
    seen_dbg = set()
    while time.time() < deadline:
        # A. Quét kênh ntfy.sh của cả 2 tài khoản (cách nhau 2 giây)
        for topic in ["studio-ai-url-cuongnguyen1802", "studio-ai-url-nguynxuncngde180528"]:
            try:
                req = urllib.request.Request(f"https://ntfy.sh/{topic}/raw?poll=1")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    content = resp.read().decode("utf-8").strip()
                    if content:
                        for line in reversed(content.splitlines()):
                            match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
                            if match:
                                cand = match.group(0)
                                if check_url_online(cand):
                                    print(f"🎉 Bắt được Cloudflare URL từ {topic}: {cand}")
                                    URL_FILE.write_text(cand + "\n", encoding="utf-8")
                                    return cand
            except urllib.error.HTTPError as he:
                if he.code == 429:
                    print("⚠️ Gặp HTTP 429 rate limit trên ntfy.sh, tạm nghỉ 20s...")
                    time.sleep(20.0)
            except Exception:
                pass
            time.sleep(1.0)

        # B. Kiểm tra trạng thái Kernel trên Kaggle API
        if deploy_kaggle:
            deploy_kaggle.load_env_file()
            u = os.environ.get("KAGGLE_USERNAME", "")
            k = os.environ.get("KAGGLE_KEY", "")
            auth_pairs = [(u, k)] if (u and k) else []
            for u, k in auth_pairs:
                try:
                    st = deploy_kaggle.get_kernel_status(u, k, slug="kaggle-all-in-one-studio")
                    cur_st = st.get("status")
                    if seen_status.get(u) != cur_st:
                        print(f"   📊 [{u}] Trạng thái Kernel: {cur_st}")
                        seen_status[u] = cur_st

                    # Nếu kernel đang chạy, thử kéo log để lấy URL
                    if cur_st == "running":
                        out_data = deploy_kaggle.get_kernel_output(u, k, slug="kaggle-all-in-one-studio")
                        raw_log = out_data.get("log", "")
                        for match in re.finditer(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", raw_log):
                            cand = match.group(0)
                            if check_url_online(cand):
                                print(f"🎉 Bắt được Cloudflare URL từ Kaggle Log của {u}: {cand}")
                                URL_FILE.write_text(cand + "\n", encoding="utf-8")
                                return cand
                except Exception:
                    pass

        # C. Kiểm tra lại current_url.txt
        if URL_FILE.exists():
            existing_url = URL_FILE.read_text().strip()
            if check_url_online(existing_url):
                print(f"🎉 Máy chủ hiện tại đang ONLINE: {existing_url}")
                return existing_url

        time.sleep(10.0)

    raise TimeoutError("Hết thời gian chờ máy chủ GPU trực tuyến.")


def run_render_step(script_name: str, desc: str):
    script_path = SCRIPTS_DIR / script_name
    print(f"\n=======================================================")
    print(f"🚀 [Auto Pipeline] Bắt đầu render: {desc}")
    print(f"   Script: {script_path}")
    print(f"=======================================================\n")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(BASE_DIR))


def main():
    print("🎬 [Pipeline Manager] Bắt đầu pipeline tự động sản xuất phim hoạt hình...")
    active_url = wait_for_active_server(max_wait_seconds=7200)
    print(f"✅ Đã xác nhận Server URL: {active_url}")

    # Bước 1: Render Hồi 2 - Cảnh 5 (Cầu Rễ Cây & Thử Thách Vực Pha Lê)
    run_render_step("generate_scene5_crystal_chasm.py", "Hồi 2 - Cảnh 5: Cầu Rễ Cây & Thử Thách Vực Pha Lê")

    # Bước 2: Render Hồi 1 - Cảnh 3 (Khởi Hành Trong Ánh Hoàng Hôn)
    run_render_step("generate_scene3_sunset_departure.py", "Hồi 1 - Cảnh 3: Khởi Hành Trong Ánh Hoàng Hôn")

    # Bước 3: Ghép toàn bộ các cảnh thành Master Film
    run_render_step("master_film_assembler.py", "Ghép Master Film Hoàn Chỉnh")

    print("\n🎉 Toàn bộ các cảnh và Master Film đã được xuất bản thành công!")


if __name__ == "__main__":
    main()
