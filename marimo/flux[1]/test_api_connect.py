#!/usr/bin/env python3
"""
Test Local Connect Script for Marimo Lab API
---------------------------------------------
Script kiểm tra kết nối từ máy local tới Marimo Lab Server thông qua REST API & SSE.
Hỗ trợ kiểm tra:
1. Endpoint /health (Liveness check)
2. Endpoint /api/sessions (Discover active notebook sessions)
3. Endpoint /api/kernel/execute (Execute arbitrary Python code in the remote kernel via SSE)
4. Kiểm tra tài nguyên GPU NVIDIA (CUDA, VRAM, Device Name)
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


DEFAULT_URL = "https://sb-e39a3b2399eeaa23.sb.molab.run"
DEFAULT_TOKEN = "b69c0120249e60491caac777da824ec995a991d0a8ab3684c4db1c89059ead34"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) MarimoClient/1.0"


def make_request(url: str, token: str = "", data: bytes | None = None, session_id: str = "", method: str | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", USER_AGENT)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if session_id:
        req.add_header("Marimo-Session-Id", session_id)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    return req


def check_health(base_url: str, token: str) -> bool:
    print(f"\n[1/3] Đang kiểm tra Health Check tại: {base_url}/health ...")
    req = make_request(f"{base_url}/health", token=token)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            print(f"  --> HTTP {status}: {body.strip() if body else 'OK'}")
            return True
    except urllib.error.HTTPError as e:
        # Some servers return 404 or empty on /health if it's protected or not mounted, but check status code
        print(f"  --> HTTP {e.code} ({e.reason})")
        return e.code in (200, 204)
    except Exception as ex:
        print(f"  --> Lỗi kết nối: {ex}")
        return False


def get_active_sessions(base_url: str, token: str) -> list[str]:
    print(f"\n[2/3] Đang truy vấn Active Sessions tại: {base_url}/api/sessions ...")
    req = make_request(f"{base_url}/api/sessions", token=token)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            session_ids = list(data.keys())
            print(f"  --> Tìm thấy {len(session_ids)} active session(s):")
            for sid, info in data.items():
                fname = info.get("filename", "untitled")
                print(f"      • ID: {sid} (File: {fname})")
            return session_ids
    except Exception as ex:
        print(f"  --> Lỗi khi lấy danh sách sessions: {ex}")
        return []


def execute_remote_code(base_url: str, token: str, session_id: str, python_code: str) -> bool:
    print(f"\n[3/3] Thực thi mã Python từ xa trên Session '{session_id}' qua /api/kernel/execute ...")
    url = f"{base_url}/api/kernel/execute"
    payload = json.dumps({"code": python_code}).encode("utf-8")

    req = make_request(url, token=token, data=payload, session_id=session_id, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            current_event = None
            stdout_acc = []
            stderr_acc = []

            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    current_event = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data_str = line.replace("data:", "").strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "stdout":
                        chunk = data.get("data", "")
                        stdout_acc.append(chunk)
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    elif current_event == "stderr":
                        chunk = data.get("data", "")
                        stderr_acc.append(chunk)
                        sys.stderr.write(chunk)
                        sys.stderr.flush()
                    elif current_event == "done":
                        if not data.get("success", False):
                            err_msg = data.get("error", {}).get("msg", "Unknown error")
                            print(f"\n  ❌ Kernel báo lỗi: {err_msg}", file=sys.stderr)
                            return False
                        output_data = data.get("output", {}).get("data", "")
                        if output_data:
                            print(f"\n  [Result Output]: {output_data}")
                        return True
            return True
    except Exception as ex:
        print(f"  --> Lỗi thực thi qua execute API: {ex}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test connection to Marimo Lab API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Marimo Server URL")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Marimo Auth Bearer Token")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    token = args.token

    print("=" * 70)
    print("🚀 MARIMO LAB API CONNECTION TEST")
    print(f"Target URL : {base_url}")
    print(f"Auth Token : {token[:8]}...{token[-8:] if len(token) > 16 else ''}")
    print("=" * 70)

    # 1. Health check
    check_health(base_url, token)

    # 2. Get sessions
    sessions = get_active_sessions(base_url, token)
    if not sessions:
        print("\n❌ Không tìm thấy active session nào trên server. Vui lòng mở notebook trên trình duyệt trước!")
        sys.exit(1)

    target_session = sessions[0]

    # 3. Test Code Execution: Diagnostics & GPU Info
    diagnostic_code = """
import sys
import torch
import marimo as mo

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'
total_vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else 'N/A'

print("-" * 60)
print(f"✅ Remote Python Kernel : OK (Python {sys.version.split()[0]})")
print(f"✅ PyTorch Version      : {torch.__version__}")
print(f"✅ CUDA Hardware        : {gpu_name}")
print(f"✅ Total VRAM           : {total_vram}")
print(f"✅ bfloat16 Acceleration: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
print(f"✅ Active Cells in DAG  : Ready for FLUX.1 generation")
print("-" * 60)
"""

    ok = execute_remote_code(base_url, token, target_session, diagnostic_code)

    print("\n" + "=" * 70)
    if ok:
        print("🎉 KẾT NỐI API THÀNH CÔNG VÀ ĐIỀU KHIỂN TỪ LOCAL HOÀN TẤT!")
    else:
        print("⚠️ Có lỗi xảy ra trong quá trình gọi API.")
    print("=" * 70)


if __name__ == "__main__":
    main()
