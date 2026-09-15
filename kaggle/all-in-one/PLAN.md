# 🎨 Kaggle All-in-One Studio: Kiến Trúc & Kế Hoạch Triển Khai
> **Mục tiêu:** Xây dựng một Studio AI toàn diện trên **Kaggle (2x Tesla T4 16GB)** gom trọn 5 trụ cột AI đỉnh cao vào **1 Session / 1 Notebook duy nhất**:
> - 👁️ **VLM:** Qwen3.8 26B (4-bit AWQ / GGUF Q4)
> - 🎙️ **STT:** faster-whisper-large-v3-turbo
> - 🔊 **TTS:** Kokoro-82M (Giọng người thật, 80x realtime)
> - 🖼️ **GenImage:** FLUX.1-schnell (NF4 qua `diffusers` + `bitsandbytes`)
> - 🎬 **GenVideo:** LTX-Video / Wan2.1-1.3B
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
│  • faster-whisper-large-v3-turbo (INT8)     : ~1.0 GB                  │
│  • Kokoro-82M TTS (FP16)                    : ~0.35 GB                 │
│  • VLM 26B 4-bit (Phần 1: Layer 0 -> N/2)   : ~7.5 GB                  │
│  • Headroom đệm (KV Cache, Context, Audio)  : ~7.1 GB (RẤT AN TOÀN)   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ GPU 1 (16GB VRAM) — [DYNAMIC WORKER SLOT: HOÁN ĐỔI LINH HOẠT]           │
├─────────────────────────────────────────────────────────────────────────┤
│  [TRẠNG THÁI 1: CHAT / SUY LUẬN / VOICE INTERACTION]                     │
│  • VLM 26B 4-bit (Phần 2: Layer N/2 -> N)   : ~7.5 GB                  │
│    ==> Phối hợp cùng GPU 0 để suy luận VLM 26B tức thì không độ trễ.    │
│                                                                         │
│  [TRẠNG THÁI 2: KHI CÓ LỆNH SINH ẢNH HOẶC SINH VIDEO]                    │
│  • Offload tạm Nửa 2 VLM sang 30GB RAM hệ thống (~1.5s qua PCIe)        │
│  • Nạp FLUX.1-schnell NF4 (~8.5GB) HOẶC LTX-Video (~9.5GB)              │
│  • Thực thi render ảnh (10-15s) hoặc video (40-60s)                     │
│  • Đưa Nửa 2 VLM trở lại GPU 1 sau khi trả kết quả                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Danh Mục Mô Hình Đỉnh Cao Được Lựa Chọn

### 1. 🎙️ STT: `faster-whisper-large-v3-turbo`
* Đứng đầu toàn cầu về nhận diện giọng nói, hỗ trợ tiếng Việt xuất sắc (tự động phân tích ngữ cảnh, chấm câu chuẩn xác).
* Chạy qua CTranslate2 (INT8) chỉ mất **~1.0 GB VRAM**, tốc độ xử lý nhanh gấp **30x - 50x thời gian thực**.

### 2. 🔊 TTS: `Kokoro-82M`
* Mô hình TTS mã nguồn mở xuất sắc nhất hiện tại, phá vỡ định kiến "nhẹ là dở".
* Trọng số chỉ **82M tham số** (~350MB VRAM), giọng đọc mượt mà như người thật, biểu cảm tự nhiên, tốc độ sinh **80x realtime**.

### 3. 👁️ VLM: `Qwen2.5-VL` / `InternVL2-26B` (4-bit)
* Khả năng đọc hiểu ảnh, sơ đồ, bảng biểu và video cực kỳ sâu sắc.
* Ở định dạng 4-bit (AWQ / GPTQ / GGUF), chiếm ~15GB VRAM. Khi chia đôi sang 2 card T4, mỗi bên chỉ gánh ~7.5GB, vận hành trơn tru.

### 4. 🖼️ GenImage: `FLUX.1-schnell` (4-bit NF4)
* Mô hình tạo ảnh SOTA của Black Forest Labs.
* Ở bản NF4 (`bitsandbytes`), toàn bộ Transformer + T5-XXL + VAE nằm trọn trong **~8.5 GB VRAM**, render ảnh 1024x1024 chỉ trong 4 bước (~12-15s trên T4).

### 5. 🎬 GenVideo: `LTX-Video` / `Wan2.1-1.3B`
* Cho phép tạo video 3-5 giây chất lượng cao từ text hoặc image-to-video trong tầm giới hạn 16GB VRAM.

---

## 4. Đặc Tả Giao Tiếp API Chuẩn OpenAI

Tất cả dịch vụ được thống nhất dưới một FastAPI Server duy nhất:

| Endpoint | Chức năng | Phương thức | Dịch vụ AI |
| :--- | :--- | :--- | :--- |
| `/v1/chat/completions` | Trò chuyện, thị giác (VLM) | POST | Qwen2.5-VL / InternVL2 26B |
| `/v1/audio/transcriptions` | Nhận diện giọng nói (STT) | POST | Whisper-large-v3-turbo |
| `/v1/audio/speech` | Đọc văn bản thành giọng nói (TTS) | POST | Kokoro-82M |
| `/v1/images/generations` | Tạo ảnh chất lượng cao | POST | FLUX.1-schnell (NF4) |
| `/v1/videos/generations` | Tạo video chuyển động | POST | LTX-Video / Wan2.1 |
| `/v1/models` | Liệt kê các model khả dụng | GET | System Registry |

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
