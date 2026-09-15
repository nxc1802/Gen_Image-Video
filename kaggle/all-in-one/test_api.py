#!/usr/bin/env python3
"""
🧪 Test Client cho Kaggle All-in-One AI Studio
Kiểm thử toàn diện end-to-end các dịch vụ qua endpoint API chuẩn OpenAI:
1. Chat & VLM (Qwen 26B)
2. TTS (Kokoro-82M Full FP16)
3. STT (Whisper-large-v3-turbo Full FP16)
4. GenImage (FLUX.1-schnell NF4)
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error


def test_chat(base_url: str, prompt: str = "Xin chào! Bạn là ai và có thể làm được gì?"):
    print("\n" + "=" * 65)
    print("🧪 1. TEST CHAT / VLM (Qwen 26B)")
    print("=" * 65)
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "qwen-26b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }
    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            reply = data["choices"][0]["message"]["content"]
            print(f"✅ VLM Phản hồi trong {elapsed:.2f}s:")
            print(f"👉 {reply.strip()}")
            return True
    except Exception as e:
        print(f"❌ Lỗi gọi Chat: {e}")
        return False


def test_tts(base_url: str, text: str = "Xin chào, tôi là trợ lý âm thanh thế hệ mới."):
    print("\n" + "=" * 65)
    print("🧪 2. TEST TEXT-TO-SPEECH (Kokoro-82M Full FP16)")
    print("=" * 65)
    url = f"{base_url.rstrip('/')}/audio/speech"
    payload = {
        "model": "kokoro-82m",
        "input": text,
        "voice": "af_heart",
        "response_format": "wav",
    }
    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            audio_bytes = resp.read()
            elapsed = time.time() - t0
            out_file = "test_tts_result.wav"
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
            print(f"✅ Sinh giọng nói thành công trong {elapsed:.2f}s ({len(audio_bytes)} bytes)")
            print(f"💾 File audio lưu tại: {os.path.abspath(out_file)}")
            return True
    except Exception as e:
        print(f"❌ Lỗi gọi TTS: {e}")
        return False


def test_image(base_url: str, prompt: str = "A majestic mechanical dragon with glowing neon wings, cyberpunk Tokyo rooftop, 8k render"):
    print("\n" + "=" * 65)
    print("🧪 3. TEST GEN IMAGE (FLUX.1-schnell NF4)")
    print("=" * 65)
    url = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": "flux-1-schnell",
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": "b64_json",
        "steps": 4,
    }
    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            b64 = data["data"][0].get("b64_json")
            if b64:
                out_file = "test_kaggle_image.png"
                with open(out_file, "wb") as f:
                    f.write(base64.b64decode(b64))
                print(f"✅ Sinh ảnh FLUX.1 thành công trong {elapsed:.2f}s!")
                print(f"💾 File ảnh lưu tại: {os.path.abspath(out_file)}")
                return True
            else:
                print(f"⚠️ Không nhận được base64: {data}")
                return False
    except Exception as e:
        print(f"❌ Lỗi gọi Image: {e}")
        return False


def resolve_target_url(specified_url: str) -> str:
    """Tự động tìm endpoint Cloudflare mới nhất từ Supabase nếu không truyền URL."""
    if specified_url and specified_url != "auto" and "localhost" not in specified_url:
        return specified_url

    # Thử kết nối localhost trước
    try:
        req = urllib.request.Request(f"{specified_url.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                print(f"📡 Đang kết nối tới Local Endpoint: {specified_url}")
                return specified_url
    except Exception:
        pass

    # Tự động truy vấn Supabase tìm URL Cloudflare của Kaggle
    SUPABASE_URL = "https://fxepzlszglckfsscport.supabase.co"
    SUPABASE_KEY = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4ZXB6bHN6Z2xja2Zzc2Nwb3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NDEzMzksImV4cCI6MjEwNTAxNzMzOX0."
        "28rS1waBYB8xvGgHR7utoek9PqBc3ev6HPOG9yo9RdQ"
    )
    try:
        q_url = f"{SUPABASE_URL}/rest/v1/image_jobs?prompt=eq.__system_announcement__&order=created_at.desc&limit=1"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        req = urllib.request.Request(q_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
            if rows and rows[0].get("image_url"):
                found_url = rows[0]["image_url"]
                print(f"📡 Tự động phát hiện Kaggle Endpoint từ Supabase: {found_url}")
                return found_url
    except Exception as e:
        print(f"⚠️ Không thể lấy URL từ Supabase: {e}")

    return specified_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Client cho Kaggle Studio API")
    parser.add_argument("--url", default="auto", help="Base URL của API (/v1) hoặc 'auto' để tự phát hiện")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ các test")
    parser.add_argument("--chat", action="store_true", help="Chỉ test Chat VLM")
    parser.add_argument("--tts", action="store_true", help="Chỉ test TTS")
    parser.add_argument("--image", action="store_true", help="Chỉ test Image")

    args = parser.parse_args()

    target_url = resolve_target_url(args.url if args.url != "auto" else "http://localhost:8000/v1")
    print(f"🎯 Target Base URL: {target_url}")

    if args.all or (not args.chat and not args.tts and not args.image):
        test_chat(target_url)
        test_tts(target_url)
        test_image(target_url)
    else:
        if args.chat:
            test_chat(target_url)
        if args.tts:
            test_tts(target_url)
        if args.image:
            test_image(target_url)

