#!/usr/bin/env python3
import sys
import json
import re
import time
import kaggle

api = kaggle.KaggleApi()
api.authenticate()
client = api.build_kaggle_client()
mod = sys.modules[client.kernels.kernels_api_client.__module__]
req = mod.ApiGetKernelSessionLogsStreamRequest()
req.user_name = "cuongnguyen1802"
req.kernel_slug = "kaggle-all-in-one-studio"

print("🔍 Đang kết nối tới live log stream của Kaggle...", flush=True)
try:
    res = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
except Exception as e:
    print(f"Lỗi kết nối stream: {e}", flush=True)
    sys.exit(1)

cloudflare_url = None
url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

for line in res.iter_lines():
    if not line:
        continue
    line_str = line.decode("utf-8", errors="ignore").strip()
    if line_str.startswith("data: "):
        payload_str = line_str[6:]
        try:
            payload = json.loads(payload_str)
            text = payload.get("data", "")
            stream_name = payload.get("stream_name", "")
            # In ra log
            print(f"[{stream_name}] {text}", end="", flush=True)

            # Tìm kiếm Cloudflare URL
            match = url_regex.search(text)
            if match:
                cloudflare_url = match.group(0)
                print(f"\n\n🎯 TÌM THẤY CLOUDFLARE PUBLIC URL: {cloudflare_url}", flush=True)
                with open("/Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/public_url.txt", "w") as f:
                    f.write(f"{cloudflare_url}/v1\n")
                break
        except Exception:
            pass

if cloudflare_url:
    print(f"✅ Đã lưu URL vào kaggle/all-in-one/public_url.txt: {cloudflare_url}/v1", flush=True)
