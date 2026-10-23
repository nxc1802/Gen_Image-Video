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


def notify_ntfy_debug(msg: str) -> None:
    """Gửi log chẩn đoán thời gian thực lên kênh debug ntfy.sh."""
    try:
        import urllib.request
        for u in ["cuongnguyen1802", "nguynxuncngde180528"]:
            topic = f"studio-ai-debug-{u}"
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=msg.encode("utf-8"),
                headers={"Title": "Studio AI Debug"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=4)
            except Exception:
                pass
    except Exception:
        pass


def ensure_cloudflared_installed() -> str:
    """Kiểm tra và cài đặt cloudflared nếu chưa có trên môi trường Linux (Kaggle)."""
    for candidate in [shutil.which("cloudflared"), "/usr/local/bin/cloudflared", "/tmp/cloudflared"]:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            try:
                chk = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=5)
                if chk.returncode == 0:
                    return candidate
            except Exception:
                pass

    custom_path = "/tmp/cloudflared"
    logger.info("📥 Đang tải Cloudflare Tunnel binary cho Linux x86_64...")
    notify_ntfy_debug("Downloading /tmp/cloudflared binary...")
    try:
        download_cmd = (
            "curl -sL --retry 3 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            f"-o {custom_path} && chmod +x {custom_path}"
        )
        subprocess.run(download_cmd, shell=True, check=True)
        logger.info(f"✅ Đã tải cloudflared về {custom_path}")
        return custom_path
    except Exception as e:
        logger.error(f"❌ Không thể tải cloudflared: {e}")
        notify_ntfy_debug(f"ERROR downloading cloudflared: {e}")
        return ""


_ACTIVE_TUNNEL_PROCESS: Optional[subprocess.Popen] = None


def start_cloudflare_tunnel(port: int = 8000, timeout_sec: int = 40) -> Optional[str]:
    """
    Khởi động Cloudflare Quick Tunnel trỏ vào cổng local `port` với cơ chế tự động thử lại (tối đa 5 lần).
    Trả về public URL (ví dụ: https://xxx.trycloudflare.com).
    """
    global _ACTIVE_TUNNEL_PROCESS
    bin_path = ensure_cloudflared_installed()
    if not bin_path:
        logger.error("Không tìm thấy binary cloudflared, bỏ qua tunnel.")
        notify_ntfy_debug("FATAL: cloudflared binary not found")
        return None

    public_url = None
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        logger.info(f"🚀 [Lần {attempt}/{max_attempts}] Đang tạo Cloudflare Quick Tunnel chuyển tiếp tới http://127.0.0.1:{port}...")
        notify_ntfy_debug(f"Starting cloudflared attempt {attempt}/{max_attempts} on port {port}...")

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
        _ACTIVE_TUNNEL_PROCESS = process

        deadline = time.time() + timeout_sec
        recent_lines = []

        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            recent_lines.append(line.strip())
            if len(recent_lines) > 15:
                recent_lines.pop(0)

            match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", line)
            if match:
                cand = match.group(0)
                if "api.trycloudflare.com" not in cand:
                    public_url = cand
                    break

        if public_url:
            def _drain_tunnel_logs(proc: subprocess.Popen):
                try:
                    if proc.stdout:
                        for _ in proc.stdout:
                            pass
                except Exception:
                    pass

            import threading
            threading.Thread(target=_drain_tunnel_logs, args=(process,), daemon=True).start()
            break
        else:
            tail_str = " | ".join(recent_lines[-5:])
            logger.warning(f"⚠️ Lần thử {attempt}/{max_attempts} chưa lấy được URL Cloudflare. Log: {tail_str}")
            notify_ntfy_debug(f"cloudflared attempt {attempt} failed: {tail_str[:300]}")
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            time.sleep(3.0)

    if public_url:
        print("\n" + "=" * 70, flush=True)
        print("🎉 CLOUDFLARE QUICK TUNNEL ĐÃ KÍCH HOẠT THÀNH CÔNG!", flush=True)
        print(f"👉 Public Base URL : {public_url}/v1", flush=True)
        print(f"👉 OpenAI SDK     : client = OpenAI(base_url='{public_url}/v1', api_key='dummy')", flush=True)
        print("=" * 70 + "\n", flush=True)

        # Lưu URL vào file để các script khác đọc nếu cần
        try:
            with open("public_url.txt", "w") as f:
                f.write(f"{public_url}/v1\n")
            for extra_dir in ["/kaggle/working", "/content", os.environ.get("COLAB_DRIVE_ROOT", "/content/drive/MyDrive/AI_Studio_Colab")]:
                if extra_dir and os.path.isdir(extra_dir):
                    try:
                        with open(os.path.join(extra_dir, "public_url.txt"), "w") as f:
                            f.write(f"{public_url}/v1\n")
                    except Exception:
                        pass
        except Exception:
            pass

        # 📡 Phát sóng URL qua ntfy.sh để local client nhận tức thì (thử lại tối đa 3 lần)
        try:
            import urllib.request
            topics = [
                os.environ.get("NTFY_TOPIC", ""),
                f"studio-ai-url-{os.environ.get('KAGGLE_USERNAME', '')}",
                "studio-ai-url-cuongnguyen1802",
                "studio-ai-url-nguynxuncngde180528",
            ]
            valid_topics = sorted(set([t for t in topics if t and t != "studio-ai-url-"]))
            for ntfy_topic in valid_topics:
                for _retry in range(3):
                    try:
                        ntfy_url = f"https://ntfy.sh/{ntfy_topic}"
                        ntfy_req = urllib.request.Request(
                            ntfy_url,
                            data=f"{public_url}/v1".encode("utf-8"),
                            headers={"Title": "Cloudflare URL Ready"},
                            method="POST",
                        )
                        with urllib.request.urlopen(ntfy_req, timeout=8) as resp:
                            logger.info(f"📡 Đã phát sóng URL qua ntfy.sh/{ntfy_topic} ({resp.status}) thành công!")
                            break
                    except Exception:
                        time.sleep(1.0)
            notify_ntfy_debug(f"SUCCESS Tunnel Ready: {public_url}/v1")
        except Exception as ne:
            logger.debug(f"Không thể phát sóng qua ntfy: {ne}")
    else:
        logger.error("❌ Không thể khởi tạo Cloudflare Quick Tunnel sau 5 lần thử. Dừng server để tránh treo phiên Kaggle.")
        notify_ntfy_debug("FATAL: All 5 cloudflared attempts failed. Exiting process.")
        os._exit(1)

    return public_url




