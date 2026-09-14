#!/usr/bin/env python3
"""
Marimo & Molab Anti-Disconnect WebSocket + HTTP Heartbeat Daemon
----------------------------------------------------------------
Maintains active communication with both:
1. HTTP API endpoints (/api/sessions)
2. WebSocket connection (/ws or session websocket)
This prevents the Molab cloud container orchestrator from flagging the
GPU sandbox as idle and terminating it with HTTP 410 Gone.
"""

import argparse
import asyncio
import json
import ssl
import sys
import time
import urllib.request

try:
    import websockets
except ImportError:
    websockets = None


def http_ping(base_url: str, token: str) -> bool:
    try:
        url = f"{base_url.rstrip('/')}/api/sessions"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
            return True, len(data)
    except urllib.error.HTTPError as he:
        if he.code == 410:
            return False, "410_GONE"
        return False, f"HTTP {he.code}"
    except Exception as e:
        return False, str(e)


async def ws_keepalive(ws_url: str, token: str, session_id: str = None):
    headers = {"Authorization": f"Bearer {token}"}
    if session_id:
        headers["Marimo-Session-Id"] = session_id

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    while True:
        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ssl=ssl_ctx,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                print(f"[{time.strftime('%H:%M:%S')}] 🔌 WebSocket kết nối thành công tới Molab!")
                while True:
                    # Send a harmless ping / message
                    await ws.ping()
                    await asyncio.sleep(15)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ WebSocket reconnecting ({e})...")
            await asyncio.sleep(5)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

    print(f"🚀 Bắt đầu Anti-Disconnect Daemon:")
    print(f"   HTTP  : {base_url}")
    print(f"   WS    : {ws_url}")
    print(f"   Chu kỳ: {args.interval}s")

    # Start WS task if websockets available
    if websockets:
        asyncio.create_task(ws_keepalive(ws_url, args.token))

    count = 0
    while True:
        count += 1
        ok, detail = http_ping(base_url, args.token)
        now_str = time.strftime("%H:%M:%S")

        if ok:
            print(f"[{now_str}] [Nhịp #{count}] Heartbeat OK (Sessions: {detail})")
        else:
            if detail == "410_GONE":
                print(f"[{now_str}] ❌ Sandbox đã bị đóng (410 Gone). Dừng heartbeat.")
                break
            else:
                print(f"[{now_str}] [Nhịp #{count}] Cảnh báo HTTP: {detail}")

        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
