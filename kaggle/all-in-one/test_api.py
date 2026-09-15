#!/usr/bin/env python3
"""
🧪 Test Client chuẩn OpenAI REST API cho Kaggle All-in-One AI Studio
Không phụ thuộc bất kỳ database nào (Zero Supabase).
Gọi trực tiếp tới Endpoint Cloudflare Public URL xuất ra từ Kaggle:
- POST /v1/chat/completions      (Qwen 26B VLM)
- POST /v1/audio/speech          (Kokoro-82M Full FP16 TTS)
- POST /v1/audio/transcriptions  (Whisper-large-v3-turbo Full FP16 STT)
- POST /v1/images/generations    (FLUX.1-schnell NF4)
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
    print(f"   URL: {base_url.rstrip('/')}/chat/completions")
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
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            reply = data["choices"][0]["message"]["content"]
            print(f"✅ VLM Phản hồi trong {elapsed:.2f}s:", flush=True)
            print(f"👉 {reply.strip()}", flush=True)
            return True
    except Exception as e:
        print(f"❌ Lỗi gọi Chat: {e}", flush=True)
        return False


def test_tts(base_url: str, text: str = "Xin chào, tôi là trợ lý âm thanh thế hệ mới."):
    print("\n" + "=" * 65)
    print("🧪 2. TEST TEXT-TO-SPEECH (Kokoro-82M Full FP16)")
    print(f"   URL: {base_url.rstrip('/')}/audio/speech")
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


def test_stt(base_url: str, audio_file: str = "test_tts_result.wav"):
    print("\n" + "=" * 65)
    print("🧪 2b. TEST SPEECH-TO-TEXT (Whisper-large-v3-turbo Full FP16)")
    print(f"   URL: {base_url.rstrip('/')}/audio/transcriptions")
    print("=" * 65)
    if not os.path.exists(audio_file):
        print(f"⚠️ Không tìm thấy file audio: {audio_file}")
        return False
    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    t0 = time.time()
    try:
        import requests
        with open(audio_file, "rb") as f:
            files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
            data = {"model": "whisper-large-v3-turbo"}
            resp = requests.post(url, files=files, data=data, timeout=120)
            elapsed = time.time() - t0
            if resp.status_code == 200:
                res_data = resp.json()
                print(f"✅ STT Nhận diện thành công trong {elapsed:.2f}s:")
                print(f"👉 Text: {res_data.get('text')}")
                return True
            else:
                print(f"❌ Lỗi STT ({resp.status_code}): {resp.text}")
                return False
    except Exception as e:
        print(f"❌ Lỗi gọi STT: {e}")
        return False


def test_image(base_url: str, prompt: str = "A majestic mechanical tiger with glowing neon circuitry, cyberpunk Tokyo rooftop, 8k render"):
    print("\n" + "=" * 65)
    print("🧪 3. TEST GEN IMAGE (FLUX.1-schnell NF4)")
    print(f"   URL: {base_url.rstrip('/')}/images/generations")
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
        with urllib.request.urlopen(req, timeout=360) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            b64 = data["data"][0].get("b64_json")
            if b64:
                out_file = "test_kaggle_image.png"
                with open(out_file, "wb") as f:
                    f.write(base64.b64decode(b64))
                print(f"✅ Sinh ảnh FLUX.1 thành công trong {elapsed:.2f}s!", flush=True)
                print(f"💾 File ảnh lưu tại: {os.path.abspath(out_file)}", flush=True)
                return True
            else:
                print(f"⚠️ Không nhận được base64: {data}", flush=True)
                return False
    except Exception as e:
        print(f"❌ Lỗi gọi Image: {e}", flush=True)
        return False


def detect_base_url(arg_url: str) -> str:
    if arg_url and arg_url != "auto":
        return arg_url.rstrip("/")

    # Kiểm tra file public_url.txt
    for path in ["public_url.txt", "/kaggle/working/public_url.txt", "kaggle/all-in-one/public_url.txt"]:
        if os.path.exists(path):
            with open(path) as f:
                line = f.read().strip()
                if line.startswith("http"):
                    print(f"📂 Đọc được URL từ {path}: {line}")
                    return line.rstrip("/")

    return "http://localhost:8000/v1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Client chuẩn OpenAI REST API")
    parser.add_argument("--url", default="auto", help="Base URL của API (Ví dụ: https://xxx.trycloudflare.com/v1)")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ các test")
    parser.add_argument("--chat", action="store_true", help="Chỉ test Chat VLM")
    parser.add_argument("--tts", action="store_true", help="Chỉ test TTS")
    parser.add_argument("--stt", action="store_true", help="Chỉ test STT")
    parser.add_argument("--image", action="store_true", help="Chỉ test Image")

    args = parser.parse_args()

    target_url = detect_base_url(args.url)
    print(f"🎯 Target Endpoint: {target_url}")

    if args.all or (not args.chat and not args.tts and not args.stt and not args.image):
        test_chat(target_url)
        test_tts(target_url)
        test_stt(target_url)
        test_image(target_url)
    else:
        if args.chat:
            test_chat(target_url)
        if args.tts:
            test_tts(target_url)
        if args.stt:
            test_stt(target_url)
        if args.image:
            test_image(target_url)
