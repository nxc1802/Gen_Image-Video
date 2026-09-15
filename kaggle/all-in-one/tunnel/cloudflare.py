"""
🌐 Cloudflare Quick Tunnel Module
Tự động thiết lập đường hầm HTTPS công khai từ Kaggle ra ngoài Internet mà không cần tài khoản.
Xuất URL public chuẩn OpenAI cho client gọi trực tiếp.
"""

import logging
import os
import re
import shutil
import subprocess
import time
from typing import Optional

logger = logging.getLogger("CloudflareTunnel")


def ensure_cloudflared_installed() -> str:
    """Kiểm tra và cài đặt cloudflared nếu chưa có trên môi trường Linux (Kaggle)."""
    binary_path = shutil.which("cloudflared")
    if binary_path:
        return binary_path

    custom_path = "/tmp/cloudflared"
    if os.path.exists(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    logger.info("📥 Đang tải Cloudflare Tunnel binary cho Linux x86_64...")
    try:
        download_cmd = (
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            f"-O {custom_path} && chmod +x {custom_path}"
        )
        subprocess.run(download_cmd, shell=True, check=True)
        logger.info(f"✅ Đã tải cloudflared về {custom_path}")
        return custom_path
    except Exception as e:
        logger.error(f"❌ Không thể tải cloudflared: {e}")
        return ""


def start_cloudflare_tunnel(port: int = 8000, timeout_sec: int = 35) -> Optional[str]:
    """
    Khởi động Cloudflare Quick Tunnel trỏ vào cổng local `port`.
    Trả về public URL (ví dụ: https://xxx.trycloudflare.com).
    """
    bin_path = ensure_cloudflared_installed()
    if not bin_path:
        logger.error("Không tìm thấy binary cloudflared, bỏ qua tunnel.")
        return None

    logger.info(f"🚀 Đang tạo Cloudflare Quick Tunnel chuyển tiếp tới http://127.0.0.1:{port}...")

    cmd = [
        bin_path,
        "tunnel",
        "--url",
        f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    public_url = None
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue

        match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        logger.info("=" * 70)
        logger.info("🎉 CLOUDFLARE QUICK TUNNEL ĐÃ KÍCH HOẠT THÀNH CÔNG!")
        logger.info(f"👉 Public Base URL : {public_url}/v1")
        logger.info(f"👉 OpenAI SDK     : client = OpenAI(base_url='{public_url}/v1', api_key='...')")
        logger.info("=" * 70)
    else:
        logger.warning("⚠️ Không thể tự động phát hiện URL Cloudflare trong thời gian chờ.")

    return public_url
