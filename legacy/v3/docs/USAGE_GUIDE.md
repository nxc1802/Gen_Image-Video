# Hướng Dẫn Sử Dụng Hệ Thống Kaggle AI Studio Factory

Tài liệu này hướng dẫn chi tiết cách cấu hình biến môi trường (`.env`), khởi chạy các Kernel chuyên biệt trên Kaggle, tra cứu toàn bộ các API Endpoint theo từng phân hệ (`speech`, `vlm`, `image_video`, `all`), và cách **Shutdown** an toàn để bảo toàn hạn mức 30 giờ GPU/tuần trên Kaggle.

---

## 1. Cách Khởi Chạy Các File Kaggle Thông Qua `.env`

Hệ thống sử dụng cơ chế **Zero-Install Offline Bundles** kết hợp script [`deploy_kaggle.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/deploy_kaggle.py) để tự động:
1. Đọc cấu hình từ file [`.env`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/.env).
2. Đóng gói toàn bộ mã nguồn (`payload_data` dạng Base64) và tiêm cấu hình `.env` vào Notebook đích (`run_*.ipynb`).
3. Đẩy (Push) trực tiếp Notebook lên Kaggle dưới dạng **Background Commit Session (2x Tesla T4 16GB)**.
4. Tự động mở **Cloudflare Quick Tunnel** (`https://*.trycloudflare.com`) và bắn thông báo URL về kênh **ntfy.sh** cá nhân của bạn.

### 1.1. Chuẩn bị file `.env`

Tạo hoặc chỉnh sửa file [`.env`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/.env) ở thư mục gốc của dự án (hoặc `kaggle/all-in-one/.env`):

```dotenv
# ==============================================================================
# 1. THÔNG TIN XÁC THỰC KAGGLE & HUGGINGFACE (BẮT BUỘC KHI DEPLOY)
# ==============================================================================
KAGGLE_USERNAME=your_kaggle_username
# Hỗ trợ cả API Key truyền thống lẫn Token mới (KGAT_...)
KAGGLE_KEY=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ==============================================================================
# 2. CẤU HÌNH THÔNG BÁO & BẢO MẬT GATEWAY
# ==============================================================================
# Topic nhận thông báo URL Cloudflare Tunnel tự động qua https://ntfy.sh/<NTFY_TOPIC>
NTFY_TOPIC=studio-ai-url-yourname

# Chế độ khởi chạy mặc định: speech | vlm | image_video | all
STUDIO_MODE=all

# (Tùy chọn) Đặt mật khẩu Bearer Token bảo vệ API. Để trống nếu không khóa auth.
STUDIO_API_KEY=

# (Tùy chọn) Tự động tắt Kernel sau N phút không có request tới /v1/* để tiết kiệm 30h GPU/tuần (0 = tắt)
IDLE_SHUTDOWN_MINUTES=120

# Cấu hình Server & Cloudflare Tunnel
HOST=0.0.0.0
PORT=8000
ENABLE_CLOUDFLARE=true

# ==============================================================================
# 3. CẤU HÌNH PHÂN HỆ VLM (Qwen3.8-27B / Qwen3.5-27B GGUF - llama-server)
# ==============================================================================
VLM_MODEL_ID=Qwen/Qwen3.5-27B-GGUF
VLM_GGUF_PATH=/kaggle/input/datasets/hungnq1802/qwen3-8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf
VLM_MMPROJ_PATH=/kaggle/input/datasets/hungnq1802/qwen3-8-27b-gguf/mmproj-Qwen3.8-27B-Q8_0.gguf
# (Tùy chọn) Model nháp (Draft model) cho Speculative Decoding
VLM_DRAFT_PATH=
LLAMA_SERVER_PORT=8081

# ==============================================================================
# 4. CẤU HÌNH PHÂN HỆ SPEECH (Qwen3-ASR, Qwen3-TTS, XTTS-v2 Vietnamese)
# ==============================================================================
STT_MODEL_ID=Qwen/Qwen3-ASR-1.7B
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
XTTS_MODEL_ID=thivux/XTTS-v2-vietnamse
TTS_VOICE=serena
AUDIO_SERVICE_PORT=8288
# 0 = nạp trực tiếp trong tiến trình chính (khuyên dùng trên Kaggle), 1 = chạy microservice riêng
AUDIO_SERVICE_SPAWN=0

# ==============================================================================
# 5. CẤU HÌNH PHÂN HỆ IMAGE & VIDEO (Qwen-Image-2.1 & Wan2.2 - ComfyUI)
# ==============================================================================
QWEN_IMAGE_MODEL_ID=QuantStack/Qwen-Image-2.1-GGUF
VIDEO_MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-GGUF
COMFYUI_DIR=/kaggle/working/ComfyUI
```

---

### 1.2. Các Lệnh Khởi Chạy Lên Kaggle

Di chuyển vào thư mục `kaggle/all-in-one` hoặc gọi trực tiếp từ thư mục gốc:

| Chế độ (Target) | Lệnh khởi chạy | Notebook & Metadata tương ứng | Mô tả nhiệm vụ |
| :--- | :--- | :--- | :--- |
| **Speech Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py speech` | [`run_speech.ipynb`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/run_speech.ipynb)<br>[`kernel-speech-metadata.json`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/kernel-speech-metadata.json) | Chuyên biệt cho **STT** (`Qwen3-ASR-1.7B`) & **TTS** (`Qwen3-TTS-1.7B` + `XTTS-v2 Vietnamese` + Voice Cloning). Khởi động siêu tốc (~2 phút). |
| **VLM Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py vlm` | [`run_vlm.ipynb`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/run_vlm.ipynb)<br>[`kernel-vlm-metadata.json`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/kernel-vlm-metadata.json) | Chuyên biệt cho **LLM / Vision-Language** (`Qwen3.8-27B-Q4_K_M.gguf`) chạy song song trên 2x T4 GPU qua `llama-server` (`--tensor-split 0.5,0.5`). |
| **Image & Video Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py image_video` | [`run_image_video.ipynb`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/run_image_video.ipynb)<br>[`kernel-image-video-metadata.json`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/kernel-image-video-metadata.json) | Chuyên biệt cho tạo ảnh **Qwen-Image-2.1** (T2I, Inpaint/Edit) và tạo video **Wan2.2** (`TI2V-5B`, `A14B-T2V`, `A14B-I2V`) qua ComfyUI. |
| **Embedding Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py embedding` | [`run_embedding.ipynb`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/run_embedding.ipynb)<br>[`kernel-embedding-metadata.json`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/kernel-embedding-metadata.json) | Chuyên biệt cho **Text & VL Embedding / Reranking** (Qwen3-0.6B, 4B, 8B trên GPU 0 và Qwen3-VL-2B, 8B trên GPU 1). Chuẩn OpenAI & Cohere API. |
| **All-in-One Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py all` | [`run_kaggle.ipynb`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/run_kaggle.ipynb)<br>[`kernel-metadata.json`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/kernel-metadata.json) | Chạy hợp nhất tất cả các dịch vụ trên cùng một Kernel với cơ chế `MemoryManager` tự động hoán đổi VRAM. |
| **Build Offline Wheels** | `python3 kaggle/all-in-one/deploy_kaggle.py wheels` | [`build_wheels_notebook.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/build_wheels_notebook.py) | Đẩy Kernel lên Kaggle để đóng gói các gói thư viện `.tar.bin` (PyTorch, ComfyUI, llama.cpp, XTTS, Qwen-ASR) lưu thành Kaggle Dataset. |
| **Download Models** | `python3 kaggle/all-in-one/deploy_kaggle.py downloader` | [`deploy_downloader.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/deploy_downloader.py) | Đẩy Kernel tải trọng số mô hình từ HuggingFace thẳng về Kaggle Dataset (không tốn băng thông máy cá nhân). |

> **Cách lấy Public URL sau khi chạy lệnh:**
> Ngay khi Kernel khởi chạy xong, URL công khai (`https://xxxx.trycloudflare.com`) sẽ được đẩy về kênh `ntfy.sh` của bạn. Bạn có thể lắng nghe trực tiếp trên terminal bằng lệnh:
> ```bash
> curl -s https://ntfy.sh/<NTFY_TOPIC>/json
> ```
> Hoặc mở trình duyệt truy cập `https://ntfy.sh/<NTFY_TOPIC>`.

### 1.3. Khởi Chạy Trực Tiếp (Local / Máy chủ GPU riêng)

Nếu chạy trực tiếp thông qua [`main.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/main.py) mà không đẩy qua Kaggle:

```bash
# Chạy theo STUDIO_MODE trong .env
python3 kaggle/all-in-one/main.py

# Hoặc ghi đè chế độ và cổng qua CLI flags
python3 kaggle/all-in-one/main.py --mode speech --port 8000 --no-tunnel
```

---

## 2. Danh Sách Các Endpoint Trong Từng API

Hệ thống bao gồm **1 Gateway API chính** (tiếp nhận toàn bộ request từ bên ngoài qua Cloudflare Tunnel hoặc cổng `8000`) và **3 Internal Daemons** (chạy ngầm nội bộ).

> **Xác thực (Authentication):**
> Nếu bạn đặt `STUDIO_API_KEY` trong `.env`, mọi request tới `/v1/*` cần truyền kèm Header:
> `Authorization: Bearer <STUDIO_API_KEY>`

---

### 2.1. Gateway API Chính (`PORT 8000` — [`server/app.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/server/app.py))

#### A. Nhóm Quản Trị & Hệ Thống (System & Config API) — *Hoạt động ở mọi chế độ*

| Method | Endpoint | Mô tả chức năng |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái sống của server, chế độ `STUDIO_MODE`, thời gian không hoạt động (`seconds_since_last_request`) và trạng thái VRAM. |
| `GET` | `/v1/models` | Liệt kê danh sách toàn bộ model khả dụng theo chuẩn OpenAI (`/v1/models`), kèm trạng thái `loaded` trên VRAM. |
| `GET` | `/v1/memory` | Xem chi tiết dung lượng RAM hệ thống và VRAM trên từng GPU (`cuda:0`, `cuda:1`), danh sách model đang nạp, cùng tiến trình `llama-server` / `ComfyUI`. |
| `POST` | `/v1/memory/free` | Giải phóng chủ động 100% VRAM (xả VLM, ComfyUI và các mô hình Audio về CPU/RAM) mà không cần tắt Server. |
| `GET` | `/v1/config` | Xem cấu hình runtime hiện tại (Model IDs, Voices, đường dẫn GGUF, trạng thái khoá API). |
| `POST` / `PATCH` | `/v1/config` | Cập nhật nóng (Hot-reload) cấu hình runtime mà không cần khởi động lại Kernel (ví dụ: đổi `tts_voice`, `stt_model_id`, `vlm_gguf_path`). |
| `POST` | `/v1/admin/shutdown?confirm=1` | Dọn sạch VRAM và tắt nóng toàn bộ tiến trình AI Studio (trả mã thoát `99` để dừng hẳn vòng lặp Notebook trên Kaggle). |

**Ví dụ cập nhật cấu hình nóng (`PATCH /v1/config`):**
```bash
curl -X PATCH "$BASE_URL/v1/config" \
  -H "Content-Type: application/json" \
  -d '{"tts_voice": "vivian"}'
```

---

#### B. Nhóm VLM / Chat API (`STUDIO_MODE=vlm` hoặc `all`)

Tương thích hoàn toàn với chuẩn **OpenAI Chat Completions API**, hỗ trợ cả văn bản thuần và phân tích đa phương thức (Multimodal Image + Text) qua `Qwen3.8-27B-GGUF`.

| Method | Endpoint | Tham số chính (JSON Body) | Mô tả |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | `model`, `messages`, `max_tokens` (mặc định `512`), `temperature` (`0.7`), `top_p` (`0.9`), `stream` (`true`/`false`), `enable_thinking` (`true`/`false`) | Chat văn bản hoặc gửi kèm ảnh (`image_url` dạng URL http/https hoặc `data:image/jpeg;base64,...`). Khi `stream: true`, trả về luồng **Server-Sent Events (SSE)** chuẩn OpenAI. |

**Ví dụ gọi VLM phân tích hình ảnh:**
```bash
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vlm",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Mô tả chi tiết bức ảnh này bằng tiếng Việt."},
          {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.6,
    "enable_thinking": false,
    "stream": false
  }'
```

---

#### C. Nhóm Speech API — STT & TTS (`STUDIO_MODE=speech` hoặc `all`)

| Method | Endpoint | Định dạng Request | Tham số chính | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/v1/audio/transcriptions` | `multipart/form-data` | `file` (file âm thanh `.wav`, `.mp3`, ...), `model` (`qwen3-asr`), `language` (`vi`, `en`, ...), `prompt`, `response_format` (`json` hoặc `text`) | Chuyển giọng nói thành văn bản (**STT**) sử dụng `Qwen3-ASR-1.7B` (hoặc fallback `Whisper-large-v3-turbo`). |
| `POST` | `/v1/audio/speech` | `application/json` | `input` (văn bản cần đọc), `model` (`qwen3-tts`, `xtts-v2`, `kokoro`), `voice` (`serena`, `vivian`, `uncle_fu`, ...), `language` (`vi`, `en`, `zh`, ...), `speed` (`1.0`), `instruct` (mô tả giọng cho VoiceDesign), `speaker_wav` (Base64 hoặc đường dẫn file `.wav` mẫu để **Clone giọng**) | Chuyển văn bản thành giọng nói (**TTS**), trả về trực tiếp luồng binary `audio/wav`. Tự động định tuyến tiếng Việt (`language="vi"`) sang `XTTS-v2 Vietnamese` hoặc `Qwen3-TTS` tùy theo `model`. |
| `GET` | `/v1/audio/diagnostics` | — | — | Kiểm tra chi tiết trạng thái các engine âm thanh (`qwen3_asr`, `qwen3_tts`, `xtts_v2`), các gói thư viện cài đặt và dung lượng VRAM từng card GPU. |

**Ví dụ 1 — Nhận diện giọng nói tiếng Việt (`POST /v1/audio/transcriptions`):**
```bash
curl -X POST "$BASE_URL/v1/audio/transcriptions" \
  -F "file=@sample_vi.wav" \
  -F "model=qwen3-asr" \
  -F "language=vi"
```

**Ví dụ 2 — Tổng hợp tiếng Việt & Clone giọng nói (`POST /v1/audio/speech`):**
```bash
curl -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xtts-v2",
    "input": "Xin chào, đây là hệ thống tổng hợp tiếng nói chạy trên Kaggle.",
    "language": "vi",
    "speed": 1.0
  }' --output output_vi.wav
```

---

#### D. Nhóm Image & Video API (`STUDIO_MODE=image_video` hoặc `all`)

Sử dụng backend **ComfyUI** kết hợp mô hình lượng tử hóa GGUF (`Qwen-Image-2.1` cho ảnh và họ `Wan2.2` cho video). Tất cả các endpoint đều hỗ trợ `stream: true` để nhận tiến độ từng bước (`progress` / `step`) qua **SSE** và chống lỗi timeout `524` của Cloudflare.

| Method | Endpoint | Định dạng Request | Tham số chính | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/v1/images/generations` | `application/json` | `prompt`, `negative_prompt`, `model` (`qwen-image`), `size` (`"1024x1024"`), `num_inference_steps` (`20`), `guidance_scale` (`3.5`), `seed`, `image_b64` (ảnh gốc nếu Inpaint), `mask_b64` (mặt nạ nếu Inpaint), `strength` (`0.75`), `response_format` (`b64_json`), `stream` (`true`/`false`) | Tạo ảnh từ văn bản (**Text-to-Image**) hoặc chỉnh sửa vùng chọn (**Inpainting**) bằng `Qwen-Image-2.1-GGUF`. |
| `POST` | `/v1/images/edits` | `multipart/form-data` | `prompt`, `image` (file ảnh gốc), `mask` (file ảnh mask tùy chọn), `size` (`"1024x1024"`), `num_inference_steps` (`20`), `guidance_scale` (`3.5`), `strength` (`0.85`), `seed`, `stream` (`false`/`true`) | Chỉnh sửa / Inpaint ảnh chuẩn Multipart form upload của OpenAI. |
| `POST` | `/v1/videos/generations` | `application/json` | `prompt`, `negative_prompt`, `model` (`wan2.2-ti2v-5b`, `wan2.2-t2v-14b`, `wan2.2-i2v-14b`), `width` (`832`), `height` (`480`), `num_frames` (`81`, tuân thủ công thức $4k+1$), `num_inference_steps` (`20`), `guidance_scale` (`5.0`), `fps` (`16`), `seed`, `image_b64` (ảnh đầu vào nếu tạo **Image-to-Video**), `lora_name`, `lora_strength` (`1.0`), `stream` (`true`/`false`) | Tạo video từ văn bản (**T2V**) hoặc từ ảnh tĩnh (**I2V**). Trả về video `.mp4` mã hóa Base64 (`b64_mp4`). |

**Ví dụ 1 — Tạo ảnh Text-to-Image (`POST /v1/images/generations`):**
```bash
curl -X POST "$BASE_URL/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-image",
    "prompt": "A cinematic portrait of a futuristic astronaut in Hanoi, neon lights, rain, 8k masterpiece",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "guidance_scale": 3.5,
    "stream": false
  }'
```

**Ví dụ 2 — Tạo video với SSE Streaming chống Timeout Cloudflare (`POST /v1/videos/generations`):**
```bash
curl -N -X POST "$BASE_URL/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-ti2v-5b",
    "prompt": "A golden retriever running happily on a sunny beach, slow motion, cinematic lighting",
    "width": 832,
    "height": 480,
    "num_frames": 33,
    "num_inference_steps": 20,
    "fps": 16,
    "stream": true
  }'
```

---

#### E. Nhóm Embedding & Rerank API (`STUDIO_MODE=embedding` hoặc `all`)

Cung cấp dịch vụ trích xuất vector đặc trưng (**Text & Multimodal Embedding**) và xếp hạng lại tài liệu ngữ nghĩa (**Reranking**). Vận hành trên nền tảng **llama.cpp** chính thức với 100% trọng số lượng tử hóa **Q4_K_M**.

| Method | Endpoint | Định dạng Request | Tham số chính | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/v1/embeddings` | `application/json` | `input` (chuỗi hoặc mảng chuỗi), `model` (`qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-vl-2b`, `qwen3-vl-8b`), `dimensions` (tùy chọn cắt giảm chiều) | Tạo vector nhúng chuẩn OpenAI REST API. Hỗ trợ Matryoshka dimensions. |
| `POST` | `/v1/rerank` | `application/json` | `query` (chuỗi câu hỏi), `documents` (danh sách đoạn văn), `model` (`qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-vl-2b`, `qwen3-vl-8b`), `top_n` (số lượng kết quả), `return_documents` (`true`/`false`) | Tính toán Cross-Attention Logits, sắp xếp lại tài liệu theo mức độ liên quan chuẩn Cohere/Jina API. |
| `POST` | `/v1/multimodal/embeddings` | `application/json` | `input` (danh sách text/ảnh), `model` (`qwen3-vl-2b`, `qwen3-vl-8b`), `dimensions` | Tạo vector nhúng đa phương thức cho cả văn bản và hình ảnh. |

**Ví dụ 1 — Tạo Vector Embeddings (`POST /v1/embeddings`):**
```bash
curl -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "input": [
      "Hệ thống AI Studio tự động triển khai trên Kaggle 2x NVIDIA Tesla T4.",
      "Cơ chế MemoryManager hoán đổi VRAM qua PCIe bus tốc độ cao."
    ]
  }'
```

**Ví dụ 2 — Tái sắp xếp tài liệu Reranking (`POST /v1/rerank`):**
```bash
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "query": "Làm sao để tiết kiệm hạn mức 30 giờ GPU Kaggle mỗi tuần?",
    "documents": [
      "Tính năng Idle Watchdog tự động shutdown kernel sau 45 phút không có request.",
      "Cách làm món phở bò truyền thống của Hà Nội rất thơm ngon.",
      "Quy tắc thiết kế giao diện Minimalist và bảng màu Warm Monochrome."
    ],
    "top_n": 2
  }'
```

---

### 2.2. Các API Nội Bộ (Internal Microservices — Chỉ lắng nghe trên `127.0.0.1`)

Các dịch vụ này được Gateway (`PORT 8000`) tự động khởi chạy và điều phối ngầm bên trong máy ảo Kaggle:

1. **Audio Microservice (`127.0.0.1:8288` — [`audio/audio_service.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/audio/audio_service.py))** *(Khi bật `AUDIO_SERVICE_SPAWN=1`)*:
   - `GET /health` — Kiểm tra trạng thái Audio Daemon.
   - `GET /audio/diagnostics` — Thông số chi tiết các mô hình âm thanh và GPU.
   - `POST /asr/transcribe` — Chạy nhận diện giọng nói `Qwen3-ASR-1.7B` trên `cuda:0`.
   - `POST /tts/qwen` — Chạy tổng hợp giọng nói `Qwen3-TTS-1.7B` (CustomVoice / VoiceDesign / ICL Clone) trên `cuda:0`.
   - `POST /tts/xtts` — Chạy tổng hợp tiếng Việt `XTTS-v2` trên `cuda:1`.
   - `POST /vram/free` — Giải phóng mô hình âm thanh khỏi VRAM (`{"target": "all" | "asr" | "qwen_tts" | "xtts"}`).

2. **Llama.cpp Server (`127.0.0.1:8081` — Quản lý bởi [`vlm/qwen.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/vlm/qwen.py))**:
   - `GET /health` — Kiểm tra trạng thái nạp mô hình GGUF.
   - `POST /v1/chat/completions` — OpenAI-compatible multimodal inference server.

3. **ComfyUI Headless Server (`127.0.0.1:8188` — Quản lý bởi [`core/comfyui_manager.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/core/comfyui_manager.py))**:
   - `GET /system_stats` — Kiểm tra trạng thái ComfyUI daemon.
   - `POST /prompt` — Đẩy workflow JSON tạo ảnh/video vào hàng đợi thực thi.
   - `GET /history/{prompt_id}` & `GET /view` — Truy xuất kết quả ảnh/video đầu ra.
   - `POST /free` — Yêu cầu ComfyUI (`unload_models: true, free_memory: true`) xả toàn bộ VRAM.

---

## 3. Cách Shutdown Khi Không Dùng Nữa (Tiết Kiệm Quota GPU Kaggle)

Khi đẩy Kernel qua [`deploy_kaggle.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/deploy_kaggle.py), Kaggle chạy Notebook ở chế độ **Background Commit Session**. Phiên chạy này sẽ giữ GPU cho đến khi **ô code cuối cùng của Notebook kết thúc** (hoặc chạm ngưỡng timeout 12 tiếng). Do đó, việc tắt đúng cách khi không sử dụng sẽ giúp bạn **bảo toàn tối đa 30 giờ GPU/tuần**.

### Cách 1: Shutdown từ xa qua API (Khuyên dùng — Nhanh & Tự động giải phóng GPU)

Gửi một request `POST` tới endpoint `/v1/admin/shutdown?confirm=1` thông qua URL Cloudflare Tunnel của bạn:

```bash
curl -X POST "https://<your-tunnel-url>.trycloudflare.com/v1/admin/shutdown?confirm=1"
```

*(Nếu có bật `STUDIO_API_KEY`, thêm `-H "Authorization: Bearer <STUDIO_API_KEY>"`)*

**Cơ chế hoạt động:**
1. Server trả về JSON `{"status": "shutting_down_permanently"}`.
2. Sau `1.0` giây, tiến trình [`main.py`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/main.py) gọi `os._exit(99)` (mã thoát `99`).
3. Vòng lặp giám sát trong Notebook (`run_kaggle.ipynb`, `run_image_video.ipynb`, `run_speech.ipynb`, `run_vlm.ipynb`) nhận diện mã thoát `99` là lệnh **Intentional Shutdown**, lập tức thoát khỏi vòng lặp `while True` và kết thúc ô Notebook.
4. Kaggle nhận thấy toàn bộ các ô trong Commit Session đã hoàn tất nên **tự động tắt máy ảo (VM) và ngừng tính giờ GPU ngay lập tức**.

---

### Cách 2: Tắt trực tiếp trên giao diện Web của Kaggle

Nếu bạn mất URL Cloudflare Tunnel hoặc muốn hủy phiên chạy ngay trong lúc Kernel đang khởi tạo:
1. Truy cập trang chủ [https://www.kaggle.com](https://www.kaggle.com).
2. Nhìn vào góc dưới bên trái màn hình, nhấn vào mục **Active Events** (biểu tượng tia sét / số phiên đang chạy).
3. Tìm Kernel đang chạy (`Kaggle Speech AI Studio`, `Kaggle VLM Studio`, `Kaggle Image & Video Studio`, hoặc `Kaggle All-in-One AI Studio`).
4. Nhấn vào dấu **`...`** (More Options) bên cạnh tên Kernel và chọn **Cancel Session** / **Stop Session**.

---

### Cách 3: Giải phóng VRAM tạm thời (Giữ nguyên Server nhưng xả GPU)

Nếu bạn chưa muốn tắt hẳn Kernel mà chỉ muốn giải phóng VRAM của các tiến trình con:
- **Khi chạy Audio Microservice riêng (`PORT 8288`):**
  ```bash
  curl -X POST "http://127.0.0.1:8288/vram/free" \
    -H "Content-Type: application/json" \
    -d '{"target": "all"}'
  ```
- **Khi chạy ở chế độ `STUDIO_MODE=all`:** Hệ thống [`MemoryManager`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/core/memory_manager.py) sẽ tự động gọi `unload()` để tắt `llama-server` hoặc yêu cầu `ComfyUI` xả VRAM (`POST /free`) mỗi khi bạn chuyển đổi qua lại giữa tác vụ VLM, Speech và Image/Video.

---

### Cách 4: Tắt khi chạy Local / SSH trực tiếp trên máy chủ

1. Nhấn tổ hợp phím `Ctrl + C` tại terminal đang chạy `main.py`.
2. Hoặc chạy lệnh dọn sạch toàn bộ tiến trình con (`llama-server`, `ComfyUI`, `cloudflared`, `audio_service`):
   ```bash
   pkill -f "kaggle/all-in-one/main.py"
   pkill -f "llama-server"
   pkill -f "cloudflared"
   pkill -f "audio_service.py"
   ```
