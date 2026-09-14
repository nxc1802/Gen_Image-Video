#!/usr/bin/env bash
# ==============================================================================
# FLUX.1 OpenAI DALL-E 3 End-to-End Automation Script (e2e_run.sh)
# ------------------------------------------------------------------------------
# Chạy toàn bộ quy trình: Cài đặt -> Tải Cloudflared -> Nạp FLUX.1 vào GPU ->
# Mở Cloudflare Tunnel -> Chờ Server Sẵn Sàng -> Gọi Test Sinh Ảnh OpenAI SDK.
# ==============================================================================
set -e

echo "======================================================================"
echo "🚀 BẮT ĐẦU QUY TRÌNH E2E: FLUX.1 OPENAI DALL-E SERVER & TEST"
echo "======================================================================"

# 1. Đảm bảo ở đúng thư mục dự án
cd /marimo || cd /root || cd ~
if [ ! -d "Gen_Image-Video" ]; then
    echo "📥 Đang clone repository..."
    git clone https://github.com/nxc1802/Gen_Image-Video.git
fi
cd Gen_Image-Video/marimo/flux[1]
git pull origin main 2>/dev/null || true

# 2. Cài đặt toàn bộ dependencies cần thiết (không cài đè torch CUDA)
echo "📦 [1/6] Đang cài đặt thư viện cần thiết..."
pip install -q "fastapi>=0.115.0" "uvicorn>=0.30.0" "pydantic>=2.8.0" \
            "diffusers>=0.30.0" "transformers>=4.43.3" "accelerate>=0.33.0" \
            sentencepiece protobuf openai

# 3. Tải Cloudflared binary chuẩn theo kiến trúc máy
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    CF_BIN="cloudflared-linux-arm64"
else
    CF_BIN="cloudflared-linux-amd64"
fi

if [ ! -s /tmp/cloudflared ] || ! /tmp/cloudflared --version >/dev/null 2>&1; then
    echo "🌐 [2/6] Đang tải Cloudflared ($CF_BIN)..."
    rm -f /tmp/cloudflared
    curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/${CF_BIN}" -o /tmp/cloudflared
    chmod +x /tmp/cloudflared
fi

# 4. Dọn dẹp tiến trình cũ (nếu có)
pkill -9 -f "server.py" 2>/dev/null || true
pkill -9 -f "cloudflared" 2>/dev/null || true
sleep 1

# 5. Cấu hình biến môi trường
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ Biến HF_TOKEN chưa được đặt."
    echo "Vui lòng chạy: export HF_TOKEN=\"hf_xxxx\" trước khi chạy script."
    exit 1
fi
export FLUX_API_KEY="${FLUX_API_KEY:-flux-sk-test-dalle-2026}"

# 6. Khởi chạy Server FLUX.1 trong nền
echo "⚡ [3/6] Đang khởi chạy FLUX.1 Server trên GPU VRAM..."
rm -f server.log
nohup python3 -u server.py \
  --port 8000 \
  --model black-forest-labs/FLUX.1-dev \
  --vram full_gpu > server.log 2>&1 &

# 7. Khởi chạy Cloudflare Tunnel trong nền (giao thức HTTP/2 không crash terminal)
echo "🔗 [4/6] Đang khởi chạy Cloudflare Tunnel..."
rm -f /tmp/tunnel.log
nohup /tmp/cloudflared tunnel \
  --url http://127.0.0.1:8000 \
  --protocol http2 \
  --no-autoupdate > /tmp/tunnel.log 2>&1 &

# 8. Lấy Public URL Cloudflare
TUNNEL_URL=""
for i in {1..20}; do
    TUNNEL_URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' /tmp/tunnel.log 2>/dev/null | head -n 1 || true)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 1
done

echo ""
echo "----------------------------------------------------------------------"
echo "🌐 Public Base URL : ${TUNNEL_URL}/v1"
echo "🔑 API Key         : ${FLUX_API_KEY}"
echo "----------------------------------------------------------------------"

# 9. Chờ Server hoàn tất tải weights và nạp vào GPU (polling /health)
echo "⏳ [5/6] Đang đợi mô hình FLUX.1 nạp xong vào VRAM (khoảng 30-60s)..."
for i in {1..120}; do
    if curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"status":"healthy"'; then
        echo "✅ Model đã nạp vào GPU VRAM thành công!"
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# 10. Chạy Test E2E tự động qua OpenAI SDK!
echo "🎨 [6/6] Đang gửi request test E2E qua OpenAI SDK (client.images.generate)..."
python3 test_openai_dalle.py \
  --base-url "http://127.0.0.1:8000/v1" \
  --api-key "${FLUX_API_KEY}" \
  --prompt "A stunning cinematic portrait of a cybernetic dragon with glowing neon scales, hyper-detailed 8k, photorealistic" \
  --out "e2e_test_result.png"

echo ""
echo "======================================================================"
echo "🎉 TOÀN BỘ QUY TRÌNH E2E ĐÃ HOÀN TẤT THÀNH CÔNG!"
echo "📸 File ảnh test thành phẩm: e2e_test_result.png"
if [ -n "$TUNNEL_URL" ]; then
    echo "🔗 Public OpenAI Endpoint: ${TUNNEL_URL}/v1/images/generations"
    echo "💡 Bạn có thể dùng URL này ở bất kỳ đâu (OpenAI SDK, LangChain, Cursor)!"
fi
echo "======================================================================"
