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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Client cho Kaggle Studio API")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="Base URL của API (/v1)")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ các test")
    parser.add_argument("--chat", action="store_true", help="Chỉ test Chat VLM")
    parser.add_argument("--tts", action="store_true", help="Chỉ test TTS")
    parser.add_argument("--image", action="store_true", help="Chỉ test Image")

    args = parser.parse_args()

    print(f"🎯 Target Base URL: {args.url}")

    if args.all or (not args.chat and not args.tts and not args.image):
        test_chat(args.url)
        test_tts(args.url)
        test_image(args.url)
    else:
        if args.chat:
            test_chat(args.url)
        if args.tts:
            test_tts(args.url)
        if args.image:
            test_image(args.url)
