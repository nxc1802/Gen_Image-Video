# ⚡ FLUX.1 Studio on Marimo Lab

Ứng dụng tạo ảnh tương tác phản ứng (Reactive Studio) chạy trên nền tảng **Marimo** dành cho mô hình **FLUX.1** (Black Forest Labs).

---

## 🌟 Tính Năng Nổi Bật

1. **Kiến trúc Reactive tách biệt**:
   - Quá trình **Nạp mô hình** (~24GB) được tách độc lập với **Quá trình sinh ảnh**. Khi thay đổi prompt, seed hoặc thanh trượt kích thước, Marimo **không bị reload lại mô hình**, giúp tiết kiệm thời gian và tài nguyên.
2. **Tự động nhận diện phần cứng & VRAM**:
   - Nhận diện GPU NVIDIA CUDA (VRAM, compute capability, hỗ trợ `bfloat16`), Apple Silicon (MPS), hoặc fallback CPU.
3. **Chiến lược tối ưu VRAM linh hoạt**:
   - **Full GPU**: Dành cho GPU khủng (RTX PRO 6000, A100, H100 $\ge$ 28GB VRAM) – tốc độ nhanh nhất.
   - **Model CPU Offload**: Dành cho GPU tầm trung (12GB - 24GB VRAM như RTX 3090/4090, A10G, T4 16GB).
   - **Sequential CPU Offload**: Dành cho GPU bộ nhớ thấp (~8GB - 12GB VRAM).
4. **Hỗ trợ đầy đủ các biến thể FLUX.1**:
   - `black-forest-labs/FLUX.1-schnell`: 4 bước (Distilled), giấy phép Apache 2.0 mở, sinh ảnh siêu tốc không cần HF Token.
   - `black-forest-labs/FLUX.1-dev`: 28 bước, chất lượng chi tiết cực cao, hỗ trợ nhập Hugging Face Token.
5. **Giao diện trực quan**:
   - Lựa chọn tỷ lệ ảnh chuẩn: $1024\times 1024$ (Square), $768\times 1360$ (Portrait 9:16), $1360\times 768$ (Landscape 16:9), $1280\times 720$ (Cinematic).
   - Tự động đo thời gian suy luận (Latency per step) và hiển thị ảnh thành phẩm ngay trong notebook.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

*(Hoặc dùng `uv` để cài siêu tốc)*:
```bash
uv pip install -r requirements.txt
```

### 2. Khởi chạy trên Marimo Lab / Remote GPU Server

#### Chế độ Chỉnh sửa & Tương tác (Edit Mode):
```bash
marimo edit app.py --host 0.0.0.0 --port 2718
```

#### Chế độ Web App thành phẩm (Run / App Mode):
```bash
marimo run app.py --host 0.0.0.0 --port 2718
```

> **Mẹo truy cập Remote**: Nếu Marimo chạy trên server từ xa không mở port công khai, bạn có thể forward port về máy local qua SSH:
> ```bash
> ssh -L 2718:localhost:2718 user@remote-ip
> ```
> Sau đó mở trình duyệt tại máy cá nhân: `http://localhost:2718`.

---

---

## 🌐 Khởi Chạy OpenAI DALL-E Compatible API Server

Mô hình FLUX.1 có thể đóng vai trò như một máy chủ API tương thích 100% với chuẩn OpenAI Images API (DALL-E 3). Bạn có thể kết nối từ LangChain, LlamaIndex, OpenWebUI, Cursor hoặc bất kỳ ứng dụng nào dùng thư viện `openai`.

### Cách 1: Khởi chạy API Server độc lập (Khuyên dùng cho Production)
```bash
python server.py --port 8000 --tunnel
```
- `--port 8000`: Cổng API cục bộ.
- `--tunnel`: Tự động kích hoạt Cloudflare Tunnel (cấp phát URL HTTPS công khai `https://xxxx.trycloudflare.com/v1`).
- `--model black-forest-labs/FLUX.1-dev`: Chỉ định model (hoặc `black-forest-labs/FLUX.1-schnell`).

### Cách 2: Bật trực tiếp bên trong giao diện Marimo
Ngay trong giao diện notebook `app.py`, cuộn xuống mục **`3. OpenAI DALL-E Compatible API Server`** và bấm **`🚀 Khởi Chạy / Restart OpenAI DALL-E Server`**. Server sẽ chia sẻ trực tiếp trọng số đã nạp trong VRAM mà không tốn thêm bộ nhớ.

### 🧪 Gọi Thử Bằng OpenAI SDK Chính Thức

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-tunnel-url.trycloudflare.com/v1",  # Hoặc http://localhost:8000/v1
    api_key="your-api-key",
)

# Gọi sinh ảnh chuẩn DALL-E
response = client.images.generate(
    model="flux-1-dev",  # Hỗ trợ "flux-1-dev", "flux-1-schnell", "dall-e-3"
    prompt="A futuristic golden cyber dragon soaring through storm clouds, neon electric arcs, 8k resolution, photorealistic",
    size="1024x1024",
    response_format="b64_json",  # Hỗ trợ cả "b64_json" và "url"
    quality="standard",  # "standard" (28 steps) hoặc "hd" (35 steps)
)

# Lấy ảnh
import base64

image_bytes = base64.b64decode(response.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

### Chạy Script Test Tự Động:
```bash
python test_openai_dalle.py --base-url "https://your-url/v1" --api-key "your-key"
```

---

## 🛡️ Cơ Chế Chống Ngắt Kết Nối Molab (Anti-Disconnect Daemon)

Molab GPU Sandboxes sử dụng cơ chế giám sát kết nối **WebSocket** và sự kiện tương tác giao diện người dùng. Nếu tab đóng hoặc không có WebSocket ping trong 10-12 phút, Molab sẽ tự động giải phóng GPU pod (`HTTP 410 Gone`).

Để giữ sandbox hoạt động liên tục:
```bash
# Khởi chạy Heartbeat kép (HTTP + WebSocket ping)
uv run --with websockets python marimo_ws_heartbeat.py --url "https://<subdomain>.sb.molab.run/" --token "<auth_token>"
```

---

## ⚡ Kết Quả Benchmark & Kiểm Thử Thực Tế

| Mô hình | Phần cứng | Độ phân giải | Số bước (Steps) | Thời gian suy luận | Tốc độ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FLUX.1-dev** | NVIDIA RTX PRO 6000 (95GB VRAM) | $1024 \times 1024$ | 28 steps | **8.42s** | **3.31 it/s** |
| **FLUX.1-schnell** | NVIDIA RTX PRO 6000 (95GB VRAM) | $1024 \times 1024$ | 4 steps | **~1.5s** | **~3.4 it/s** |

---

## 💡 Lưu Ý Khi Sử Dụng FLUX.1

- **Kiểu dữ liệu (Precision)**: FLUX.1 sử dụng kiến trúc DiT + RoPE, chỉ hoạt động ổn định và chính xác trên `bfloat16` hoặc `float32`. Tuyệt đối tránh ép về `float16` vì dễ bị lỗi NaN/đen ảnh.
- **FLUX.1-schnell vs FLUX.1-dev**:
  - `FLUX.1-schnell` dùng guidance distillation nên `guidance_scale` đặt ở **0.0** và chỉ cần **4 bước**.
  - `FLUX.1-dev` cần `guidance_scale` khoảng **3.5** và số bước từ **28 - 50 bước**.
- **Chữ viết (Typography)**: Nhờ có Text Encoder T5-XXL, bạn có thể yêu cầu FLUX vẽ chữ chính xác bằng cách đặt nội dung cần viết trong dấu nháy đơn `'...'` (ví dụ: `text 'HELLO WORLD' written on the chalkboard`).


