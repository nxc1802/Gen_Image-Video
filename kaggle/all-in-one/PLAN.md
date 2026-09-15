# 🎨 Kaggle All-in-One Studio: Kiến Trúc & Kế Hoạch Triển Khai
> **Mục tiêu:** Xây dựng một Studio AI toàn diện trên **Kaggle (2x Tesla T4 16GB)** gom trọn 5 trụ cột AI đỉnh cao vào **1 Session / 1 Notebook duy nhất**:
> - 👁️ **VLM:** Qwen3.8 26B (4-bit AWQ / GGUF Q4)
> - 🎙️ **STT:** faster-whisper-large-v3-turbo
> - 🔊 **TTS:** Kokoro-82M
> - 🖼️ **GenImage:** FLUX.1-schnell (NF4 qua `diffusers` + `bitsandbytes`)
> - 🎬 **GenVideo:** Wan2.1-1.3B
> - ⚡ **Gateway:** Tương thích 100% chuẩn OpenAI API (Chat, Audio, Images, Videos)

---

## 1. Phân Tích Tài Nguyên & Thách Thức Phần Cứng

| Tài Nguyên Kaggle | Thông số | Đánh giá & Ràng buộc |
| :--- | :--- | :--- |
| **GPU VRAM** | 2x Tesla T4 (16GB mỗi card = 32GB) | Không gộp tự động; cần phân bổ tải đều 2 GPU (Tensor/Pipeline Parallelism). |
| **Kiến trúc GPU** | Turing (Compute Capability 7.5) | Không có FP8 phần cứng $\rightarrow$ Tối ưu vượt trội với **INT8 và 4-bit (NF4, AWQ, GGUF)**. |
| **System RAM** | 30 GB CPU RAM | Cực kỳ dồi dào, đóng vai trò **Bộ nhớ trung gian (Fast Cache)** để tráo model trong 1.5 giây. |
| **Ổ cứng (Disk)** | 73 GB SSD | Dư sức chứa trọng số của cả 5 mô hình nén (~22 GB tổng cộng). |
| **Thời gian phiên** | 9h - 12h liên tục | Thiết kế theo triết lý: Bật khi cần dùng, tắt khi xong việc. |

---

## 2. Kiến Trúc Phân Bổ VRAM (Dual-GPU Partitioning)

Mô hình tương tác thực tế của người dùng:
$$\text{Voice/Text Input (STT)} \longrightarrow \text{Tư duy & Thị giác (VLM)} \longrightarrow \text{Sáng tạo Visual (FLUX/Video)} \longrightarrow \text{Đọc âm thanh (TTS)}$$

### Bản Đồ VRAM trên 2 Card Tesla T4:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ GPU 0 (16GB VRAM) — [INTERACTIVE CORE: LUÔN THƯỜNG TRỰC]                │
├─────────────────────────────────────────────────────────────────────────┤
│  • faster-whisper-large-v3-turbo (BẢN FULL FP16): ~1.60 GB              │
│    (Giữ 100% độ chính xác âm thanh gốc, không nén, không suy hao)     │
│  • Kokoro-82M TTS (BẢN FULL FP16/FP32)          : ~0.35 GB             │
│    (Giữ 100% chất lượng giọng người thật chuẩn phòng thu)               │
│  • Qwen 26B 4-bit (Phần 1: Layer 0 -> N/2)      : ~7.50 GB             │
│  • Headroom đệm (KV Cache, Context, Vision latents): ~6.55 GB (CỰC RỘNG)│
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ GPU 1 (16GB VRAM) — [DYNAMIC WORKER SLOT: HOÁN ĐỔI LINH HOẠT]           │
├─────────────────────────────────────────────────────────────────────────┤
│  [TRẠNG THÁI 1: CHAT / SUY LUẬN / VOICE INTERACTION]                     │
│  • Qwen 26B 4-bit (Phần 2: Layer N/2 -> N)      : ~7.50 GB             │
│    ==> Phối hợp cùng GPU 0 để suy luận VLM 26B tức thì không độ trễ.    │
│                                                                         │
│  [TRẠNG THÁI 2: KHI CÓ LỆNH SINH ẢNH HOẶC SINH VIDEO]                    │
│  • Offload tạm Nửa 2 VLM sang 30GB RAM hệ thống (~1.5s qua PCIe)        │
│  • Nạp FLUX.1-schnell NF4 (~8.5GB) HOẶC Wan2.1-1.3B (~9.5GB)           │
│  • Thực thi render ảnh (10-15s) hoặc video (40-60s)                     │
│  • Đưa Nửa 2 VLM trở lại GPU 1 sau khi trả kết quả                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Danh Mục Mô Hình Đỉnh Cao Được Lựa Chọn

### 1. 🎙️ STT: `Whisper-large-v3-turbo` (BẢN FULL FP16 — Không Quantize)
* Chạy ở **Full Precision FP16** gốc (~1.6 GB VRAM) thay vì nén INT8.
* Giữ trọn vẹn 100% khả năng lọc nhiễu, nhận diện ngữ cảnh và độ chính xác dấu câu tiếng Việt chuẩn quốc tế.
* Tốc độ xử lý trên T4 cực nhanh: ~0.5s cho 30s âm thanh.

### 2. 🔊 TTS: `Kokoro-82M` (BẢN FULL FP16 — Không Quantize)
* Chạy ở **Full Precision FP16** (~350 MB VRAM).
* Giữ trọn vẹn 100% âm sắc tự nhiên của giọng người thật, độ luyến láy và cảm xúc chuẩn phòng thu (không bị méo tiếng do lượng tử hóa). Tốc độ 80x realtime.

### 3. 👁️ VLM: `Qwen 26B` (4-bit AWQ / GGUF Q4)
* Mô hình thị giác & suy luận ngôn ngữ mạnh mẽ dòng Qwen.
* Dùng bản lượng tử hoá 4-bit để vừa vặn ngân sách, phân bổ chia đôi qua 2 card T4 (~7.5GB mỗi GPU).

### 4. 🖼️ GenImage: `FLUX.1-schnell` (4-bit NF4)
* Mô hình tạo ảnh SOTA của Black Forest Labs.
* Ở bản NF4 (`bitsandbytes`), toàn bộ Transformer + T5-XXL + VAE nằm trọn trong **~8.5 GB VRAM**, render ảnh 1024x1024 chỉ trong 4 bước (~12-15s trên T4).

### 5. 🎬 GenVideo: `Wan2.1-1.3B`
* Mô hình sinh video điện ảnh mới nhất từ Alibaba.
* Cho phép tạo video chuyển động mượt mà trong tầm giới hạn 16GB VRAM của GPU 1.

---

## 4. Đặc Tả Giao Tiếp API Chuẩn OpenAI

Tất cả dịch vụ được thống nhất dưới một FastAPI Server duy nhất:

| Endpoint | Chức năng | Phương thức | Dịch vụ AI | Định dạng Model |
| :--- | :--- | :--- | :--- | :--- |
| `/v1/chat/completions` | Trò chuyện, thị giác (VLM) | POST | Qwen 26B | 4-bit Dual-GPU |
| `/v1/audio/transcriptions` | Nhận diện giọng nói (STT) | POST | Whisper-large-v3-turbo | **Full FP16** |
| `/v1/audio/speech` | Đọc văn bản thành tiếng (TTS) | POST | Kokoro-82M | **Full FP16** |
| `/v1/images/generations` | Tạo ảnh chất lượng cao | POST | FLUX.1-schnell | NF4 |
| `/v1/videos/generations` | Tạo video chuyển động | POST | Wan2.1-1.3B | FP8 / NF4 |
| `/v1/models` | Liệt kê các model khả dụng | GET | System Registry | JSON |

---

## 5. Kết Nối & Xuất API (Hai Tuỳ Chọn)

1. **Tuỳ chọn A — Direct Cloudflare Tunnel (Đơn giản nhất):**
   * Kaggle tự động bật Quick Tunnel `cloudflared`.
   * Xuất ra URL: `https://xxx.trycloudflare.com/v1`.
   * Client (App / Cursor / Web) gọi thẳng URL này, **không cần chạy bất kỳ file nào ở máy local**.

2. **Tuỳ chọn B — Supabase Broker Queue (Bền bỉ nhất):**
   * Dùng Supabase làm trung gian qua HTTPS (Port 443).
   * Phối hợp với [`marimo/flux/bridge.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/marimo/flux/bridge.py) ở máy local để có URL cố định vĩnh viễn `http://localhost:8000/v1`.

---

## 6. Lộ Trình Triển Khai (Roadmap)

- [x] **Giai đoạn 1:** Khảo sát kiến trúc, tính toán ngân sách VRAM và viết tài liệu thiết kế.
- [ ] **Giai đoạn 2:** Xây dựng notebook mẫu `studio_notebook.ipynb` hoàn chỉnh trong `kaggle/all-in-one/`.
- [ ] **Giai đoạn 3:** Hiện thực hoá module `MemoryLifecycleManager` (Fast PCIe Swapping giữa VLM và FLUX/Video).
- [ ] **Giai đoạn 4:** Viết FastAPI Server tích hợp đầy đủ 5 endpoint chuẩn OpenAI.
- [ ] **Giai đoạn 5:** Tạo tài liệu README hướng dẫn import và chạy trên Kaggle chỉ với 1 click "Run All".
