# 🎨 Gen_Image-Video: FLUX.1 & AI Generation Server

Hệ thống triển khai mô hình AI tạo ảnh **FLUX.1** (Black Forest Labs) với giao diện tương tác phản ứng (Marimo Studio) và máy chủ API tương thích 100% chuẩn **OpenAI DALL-E 3** (`POST /v1/images/generations`).

---

## 📁 Cấu Trúc Dự Án

```tree
Gen_Image-Video/
├── .gitignore
├── README.md
├── kaggle/
│   └── gemma/
│       └── gemma-4-e2b-api.ipynb      # Notebook triển khai Gemma-4 qua API
└── marimo/
    └── flux[1]/
        ├── app.py                     # Ứng dụng Marimo Reactive Studio + API
        ├── server.py                  # Standalone FastAPI Server chuẩn OpenAI DALL-E
        ├── test_openai_dalle.py       # Script kiểm thử bằng thư viện OpenAI SDK
        ├── test_api_connect.py        # Script kiểm tra kết nối & health check
        ├── marimo_ws_heartbeat.py     # Daemon chống timeout/ngắt kết nối Molab
        ├── requirements.txt           # Danh sách thư viện cần thiết
        ├── flux_generated_sample.png  # Ảnh kết quả benchmark thực tế (1024x1024)
        └── README.md                  # Hướng dẫn chi tiết cho FLUX.1 Studio
```

---

## 🚀 Chạy Nhanh Bằng Terminal / CLI (Không Lo Disconnect)

### Bước 1: Cài đặt môi trường & thư viện
```bash
# Clone repository
git clone https://github.com/nxc1802/Gen_Image-Video.git
cd Gen_Image-Video/marimo/flux[1]

# Cài đặt uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Tạo Virtual Environment & Cài dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Bước 2: Tải Cloudflared (Mở Public URL HTTPS tự động)
```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
chmod +x /tmp/cloudflared
```

### Bước 3: Cấu hình Token & Khởi chạy Server trong Background (Tmux / Nohup)
```bash
# Thiết lập biến môi trường
export HF_TOKEN="your_huggingface_token_here"
export FLUX_API_KEY="flux-sk-test-dalle-2026"

# Khởi chạy độc lập bằng nohup hoặc tmux
nohup python -u server.py \
  --port 8000 \
  --model black-forest-labs/FLUX.1-dev \
  --vram full_gpu \
  --tunnel > server.log 2>&1 &

# Xem tiến độ nạp mô hình & lấy Public URL Cloudflare
tail -f server.log
```

---

## 🧪 Kiểm Thử Bằng OpenAI SDK Chính Thức

Sau khi `server.log` hiển thị URL dạng `https://xxxx.trycloudflare.com/v1`:

```bash
# Chạy script test tự động
python test_openai_dalle.py \
  --base-url "https://xxxx.trycloudflare.com/v1" \
  --api-key "flux-sk-test-dalle-2026" \
  --prompt "A futuristic golden cyber dragon soaring through storm clouds, neon electric arcs, 8k resolution, photorealistic"
```

Hoặc trong Python code:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://xxxx.trycloudflare.com/v1",
    api_key="flux-sk-test-dalle-2026",
)

response = client.images.generate(
    model="flux-1-dev",  # hoặc "flux-1-schnell", "dall-e-3"
    prompt="A futuristic cyberpunk street at night, neon reflections, 8k",
    size="1024x1024",
    response_format="b64_json",
)
```

---

## ⚡ Kết Quả Benchmark Thực Tế (NVIDIA RTX PRO 6000 - 95GB VRAM)

- **Mô hình**: `black-forest-labs/FLUX.1-dev` (Precision `bfloat16`, nạp 100% trên VRAM GPU).
- **Thời gian sinh ảnh 1024x1024**: **8.42 giây** (28 steps, tốc độ **3.31 it/s**).
- **VRAM chiếm dụng**: ~24GB.
