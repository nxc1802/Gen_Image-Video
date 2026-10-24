# Hướng Dẫn Khởi Chạy Và Vận Hành Kaggle VLM Studio (Qwen 3.8 27B + MTP)

Tài liệu này cung cấp hướng dẫn toàn diện về cách khởi chạy, cấu hình Speculative Decoding (MTP / DFlash), cấu hình Context Window (từ 4K đến 262K / 1M với YaRN), kiểm tra trạng thái và tương tác với API của phiên bản **VLM Dedicated Server** trên Kaggle Dual Tesla T4 (2x 16GB VRAM).

---

## 1. Tổng Quan Kiến Trúc VLM Server

- **Model chính**: `Qwen3.8-27B-Instruct` lượng tử hóa `Q4_K_M` (~17.1 GB weights).
- **Vision Projector (mmproj)**: `mmproj-Qwen3.8-27B-Q8_0.gguf` (~1.4 GB weights).
- **Speculative Decoding**:
  - **MTP (Multi-Token Prediction)**: Sử dụng checkpoint `mtp-Qwen3.8-27B-Q4_0.gguf` (~18.8 GB tích hợp cả thân model + 3 MTP draft heads), cờ khởi chạy `--spec-type draft-mtp --spec-draft-n-max 3`.
  - **DFlash**: Model draft riêng biệt `dflash-Qwen3.8-27B-Q4_0.gguf` (~1.2 GB), cờ `--spec-type draft-dflash --spec-draft-n-max 15`.
- **Phân bổ bộ nhớ Dual-GPU**:
  - Sử dụng `--split-mode layer`.
  - Tỉ lệ `--tensor-split`:
    - Context tiêu chuẩn ($n_{\text{ctx}} < 64\text{K}$): `0.5,0.5` chia đều ~32 layers mỗi GPU.
    - Context mở rộng ($n_{\text{ctx}} \ge 64\text{K}$): `0.54,0.46` dồn thêm 2-3 layers sang GPU 0 để giải phóng headroom KV cache cho GPU 1.
- **Context Scaling (YaRN RoPE)**:
  - Ngưỡng kích hoạt YaRN: `$n_{\text{ctx}} > 262,144$` (256K).
  - Tần số RoPE gốc: `--yarn-orig-ctx 262144`.
  - Hệ số YaRN scale: tại 1M tokens là $\text{scale} = 4.0$.

---

## 2. Cách Khởi Chạy VLM Lên Kaggle

### 2.1. Kiểm tra tài khoản & cấu hình

Hệ thống hỗ trợ xác thực bằng thông tin trong file `.env` hoặc truyền trực tiếp qua tham số CLI.

Kiểm tra trạng thái session hiện tại:
```bash
python3 kaggle/all-in-one/deploy_kaggle.py --status
```
*(Nếu muốn chỉ định rõ tài khoản: thêm `--user <username> --key <token>`)*.

### 2.2. Đóng gói mã nguồn & Đẩy Kernel VLM

Chạy lệnh deploy với target `vlm`:

```bash
# Sử dụng tài khoản mặc định trong .env
python3 kaggle/all-in-one/deploy_kaggle.py --target vlm --push-only

# Hoặc truyền trực tiếp user & Kaggle API token
python3 kaggle/all-in-one/deploy_kaggle.py \
  --user your_kaggle_username \
  --key KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --target vlm \
  --push-only
```

> **Lưu ý**: Lệnh trên sẽ tự động:
> 1. Đóng gói mã nguồn mới nhất của VLM (`kaggle/all-in-one/vlm/`, `server/`, `core/`,...) thành chuỗi Base64 `payload_data`.
> 2. Ghi đè vào Cell 2 của `run_vlm.ipynb`.
> 3. Cập nhật `kernel-vlm-metadata.json` với dataset `qwen38-27b-gguf` chứa đầy đủ model, mmproj, và các model draft MTP/DFlash.
> 4. Đẩy notebook lên Kaggle Kernel `cuongnguyen1802/kaggle-all-in-one-studio` và khởi động máy ảo Dual T4.

---

## 3. Theo Dõi & Nhận Thông Báo Cloudflare URL

Khi Kernel khởi chạy, hệ thống sẽ:
1. Giải nén offline wheels và binary `llama-server`.
2. Khởi tạo FastAPI gateway tại port 8000.
3. Kích hoạt Cloudflared Quick Tunnel và tự động publish URL về kênh `ntfy.sh`.
4. Nạp song song model Qwen 3.8 27B lên Dual T4 qua `llama-server` (thời gian nạp weights ~3-4 phút).

### 3.1. Lắng nghe URL từ ntfy.sh

```bash
# Polling lấy thông điệp URL mới nhất
curl -s "https://ntfy.sh/studio-ai-url-cuongnguyen1802/raw?poll=1&since=10m"

# Xem stream log debug
curl -s "https://ntfy.sh/studio-ai-debug-cuongnguyen1802/raw?poll=1&since=10m"
```

### 3.2. Kiểm tra tiến trình nạp Model qua `/health`

Thực hiện gọi API `/health` tới Cloudflare URL vừa nhận được:

```bash
curl -s "https://<TUNNEL_URL>/health" | jq .
```

Phản hồi khi hệ thống đã sẵn sàng:
```json
{
  "status": "healthy",
  "studio_mode": "vlm",
  "vlm_diagnostics": {
    "is_loaded": true,
    "spec_mode": "mtp",
    "backend": "llama-server",
    "model_id": "Qwen/Qwen3.5-27B-GGUF",
    "port": 8081
  }
}
```
Khi `is_loaded: true`, server đã hoàn toàn sẵn sàng nhận request sinh text / phân tích hình ảnh và chạy benchmark RULER.

---

## 4. Danh Sách Endpoint Cho VLM

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái gateway và VLM backend (is_loaded, spec_mode, memory). |
| `GET` | `/v1/models` | Trả về danh sách model VLM OpenAI-compatible khả dụng. |
| `POST` | `/v1/chat/completions` | Endpoint chính để chat và hỏi đáp ảnh/multimodal (OpenAI-compatible). Hỗ trợ cả `stream: true` và `stream: false`. |
| `POST` | `/v1/admin/reconfigure-vlm` | Hot-swap chế độ Speculative Decoding (`baseline`, `mtp`, `dflash`) mà không cần restart Kaggle Kernel. |
| `POST` | `/v1/admin/shutdown` | Tắt hoàn toàn Kaggle Kernel từ xa để bảo toàn quota GPU. |

---

## 5. Ví Dụ Gọi API

### 5.1. Chat Completion Thông Thường & Reasoning

```bash
curl -X POST "https://<TUNNEL_URL>/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B-GGUF",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful and rigorous AI assistant. Think carefully before answering."
      },
      {
        "role": "user",
        "content": "Chứng minh rằng căn bậc hai của 2 là một số vô tỉ."
      }
    ],
    "temperature": 0.6,
    "max_tokens": 1024,
    "stream": false
  }'
```

### 5.2. Nhận Diện Hình Ảnh (Multimodal Vision)

Gửi ảnh dạng base64 thông qua cấu trúc OpenAI Vision:

```bash
curl -X POST "https://<TUNNEL_URL>/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B-GGUF",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Mô tả chi tiết bức ảnh này và giải thích các đối tượng trong đó:"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,<BASE64_IMAGE_DATA>"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```

### 5.3. Hot-Swap Đổi Chế Độ Speculative Decoding

Không cần phải kill hay deploy lại Kaggle Kernel, chỉ cần gửi POST request tới `/v1/admin/reconfigure-vlm`:

```bash
# Chuyển sang MTP (mặc định cho độ chính xác & tốc độ cao nhất)
curl -X POST "https://<TUNNEL_URL>/v1/admin/reconfigure-vlm" \
  -H "Content-Type: application/json" \
  -d '{"spec_mode": "mtp"}'

# Chuyển sang DFlash
curl -X POST "https://<TUNNEL_URL>/v1/admin/reconfigure-vlm" \
  -H "Content-Type: application/json" \
  -d '{"spec_mode": "dflash"}'

# Chuyển về Baseline (chạy trực tiếp Q4_K_M không có draft model)
curl -X POST "https://<TUNNEL_URL>/v1/admin/reconfigure-vlm" \
  -H "Content-Type: application/json" \
  -d '{"spec_mode": "baseline"}'
```

---

## 6. Chạy Benchmark Đánh Giá Ngữ Cảnh Dài (RULER Test)

Bộ test RULER (NIAH, Multi-keys, Variable Tracking, Aggregation) được tích hợp sẵn qua script runner:

```bash
# Chạy benchmark RULER tự động từ 4K đến 128K context
python3 test_outputs/auto_ruler_scale_runner.py \
  --base-url "https://<TUNNEL_URL>" \
  --output-dir "test_outputs/ruler_results"
```

Khi chạy xong các tác vụ, hãy chủ động tắt Kernel để tiết kiệm quota 30h GPU/tuần:
```bash
curl -X POST "https://<TUNNEL_URL>/v1/admin/shutdown"
```
