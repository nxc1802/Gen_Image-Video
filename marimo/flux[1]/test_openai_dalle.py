#!/usr/bin/env python3
"""
Test Script: Call FLUX.1 via Official OpenAI DALL-E SDK
-------------------------------------------------------
Kiểm tra khả năng tương thích 100% với OpenAI API chuẩn (Images API):
`client = OpenAI(base_url=..., api_key=...)`
`client.images.generate(model="flux-1-dev", prompt="...", ...)`
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path


DEFAULT_BASE_URL = "https://cassette-insulation-grill-claire.trycloudflare.com/v1"
DEFAULT_API_KEY = "flux-NQJSQYlbXgwXBo9gb-yt_8EhP5MMttN5"


def main():
    parser = argparse.ArgumentParser(description="Test FLUX.1 via official OpenAI DALL-E API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI Base URL (e.g. https://...trycloudflare.com/v1)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--model", default="flux-1-dev", help="Model name (flux-1-dev, flux-1-schnell, dall-e-3)")
    parser.add_argument("--prompt", default="A stunning cinematic portrait of a cyberpunk cat astronaut wearing a high-tech glowing helmet, reflections of distant nebulas in the visor, hyper-detailed 8k, photorealistic, Unreal Engine 5 render, text 'OPENAI FLUX' on the spacesuit collar", help="Prompt text")
    parser.add_argument("--size", default="1024x1024", help="Resolution (1024x1024, 768x1360, 1360x768)")
    parser.add_argument("--format", default="b64_json", choices=["b64_json", "url"], help="Response format")
    parser.add_argument("--out", default="dalle_api_result.png", help="Output filename to save")
    args = parser.parse_args()

    print("=" * 70)
    print("🎨 TEST FLUX.1 QUA OPENAI DALL-E SDK")
    print(f"Base URL : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Size     : {args.size}")
    print(f"Format   : {args.format}")
    print(f"Prompt   : {args.prompt[:80]}...")
    print("=" * 70)

    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Thiếu thư viện `openai`. Cài đặt bằng: pip install openai")
        sys.exit(1)

    # 1. Khởi tạo client OpenAI tiêu chuẩn
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # 2. Kiểm tra danh sách models (/v1/models)
    print("\n[1/2] Kiểm tra danh sách Models (/v1/models)...")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"  --> Models khả dụng: {model_ids}")
    except Exception as e:
        print(f"  --> Cảnh báo khi lấy models: {e}")

    # 3. Gửi request sinh ảnh qua client.images.generate
    print(f"\n[2/2] Gửi request sinh ảnh tới `{args.model}` qua client.images.generate()...")
    start_t = time.time()

    try:
        response = client.images.generate(
            model=args.model,
            prompt=args.prompt,
            size=args.size,
            response_format=args.format,
            n=1,
        )
        elapsed = time.time() - start_t
        print(f"  --> Hoàn thành sau: {elapsed:.2f}s!")

        data = response.data[0]
        out_path = Path(__file__).parent / args.out

        if args.format == "b64_json" and data.b64_json:
            img_bytes = base64.b64decode(data.b64_json)
            out_path.write_bytes(img_bytes)
            print(f"  --> Ảnh đã giải mã từ base64 và lưu tại: {out_path.resolve()}")
            print(f"  --> Kích thước file: {len(img_bytes) / 1024:.1f} KB")
        elif args.format == "url" and data.url:
            print(f"  --> URL ảnh thành phẩm: {data.url}")

        print("\n" + "=" * 70)
        print("🎉 GỌI API CHUẨN OPENAI DALL-E THÀNH CÔNG RỰC RỠ!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Lỗi khi gọi OpenAI API: {type(e).__name__} - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
