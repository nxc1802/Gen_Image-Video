#!/usr/bin/env python3
"""
Marimo Lab Anti-Disconnect Heartbeat Daemon
--------------------------------------------
Gửi ping định kỳ 20 giây một lần để ngăn chặn Molab Sandbox Idle Timeout (410).
Thực hiện truy vấn /api/sessions và thực thi ping nhẹ trong kernel.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) MarimoClient/1.0"


def make_request(url: str, token: str = ""):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def run_heartbeat(url: str, token: str, interval: int = 20):
    base_url = url.rstrip("/")
    print(f"🚀 Bắt đầu Anti-Disconnect Heartbeat tới: {base_url}")
    print(f"⏱️ Chu kỳ: {interval} giây/nhịp")

    count = 0
    while True:
        count += 1
        now = time.strftime("%H:%M:%S")
        try:
            # 1. Ping /api/sessions
            req = make_request(f"{base_url}/api/sessions", token=token)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                sids = list(data.keys())

            # 2. If sessions exist, send a tiny keep-alive eval
            if sids:
                target_sid = sids[0]
                exec_url = f"{base_url}/api/kernel/execute"
                body = json.dumps({"code": f"# heartbeat {count}\npass"}).encode("utf-8")
                req_exec = urllib.request.Request(exec_url, data=body, method="POST")
                req_exec.add_header("User-Agent", USER_AGENT)
                req_exec.add_header("Content-Type", "application/json")
                req_exec.add_header("Marimo-Session-Id", target_sid)
                if token:
                    req_exec.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(req_exec, timeout=10) as _:
                    pass

            print(f"[{now}] [Nhịp #{count}] Heartbeat OK (Active sessions: {len(sids)})", flush=True)

        except urllib.error.HTTPError as he:
            print(f"[{now}] [Nhịp #{count}] HTTP Error: {he.code} - {he.reason}", file=sys.stderr, flush=True)
            if he.code == 410:
                print(f"[{now}] ❌ Sandbox đã bị đóng (410). Dừng heartbeat.", file=sys.stderr, flush=True)
                break
        except Exception as e:
            print(f"[{now}] [Nhịp #{count}] Cảnh báo kết nối: {e}", file=sys.stderr, flush=True)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Marimo Lab Anti-Disconnect Heartbeat")
    parser.add_argument("--url", required=True, help="Marimo Server URL")
    parser.add_argument("--token", default="", help="Marimo Auth Token")
    parser.add_argument("--interval", type=int, default=20, help="Chu kỳ giây giữa các nhịp")
    args = parser.parse_args()

    run_heartbeat(args.url, args.token, args.interval)


if __name__ == "__main__":
    main()
