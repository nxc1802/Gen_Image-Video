# 🎨 Thiết Kế Giao Diện Kaggle AI Studio Factory (Full API Playground)

Mục tiêu là xây dựng **UI để trực tiếp trải nghiệm, kiểm thử và quản trị toàn bộ các model/API của Factory trên Kaggle**. Giao diện tuyệt đối không mang cảm giác nặng nề của Postman hay Swagger (không phơi bày HTTP method, endpoint, headers hay JSON thô ra màn hình chính), mà hoạt động như một **AI Studio Playground chuyên nghiệp**:

> **Chọn Model → Nhập Input → Run (Streaming) → Xem Output → Đo Hiệu Năng (Performance)**

---

## 1. Bố Cục Tổng Thể (Master Layout)

Giao diện gồm **3 khu vực chính**:
1. **Top Status & Connection Bar**: Hiển thị kết nối Cloudflare Tunnel (tự động đồng bộ qua `ntfy.sh`), chế độ `STUDIO_MODE` hiện tại, mức chiếm dụng 2x GPU T4, bộ đếm tự tắt khi nhàn rỗi (`Idle Timer`), và nút tắt nóng Kernel.
2. **Left Sidebar**: Chia thành 2 nhóm rõ rệt là **WORKLOAD** (5 phân hệ AI) và **SYSTEM** (3 phân hệ quản trị).
3. **Main Workspace**: Khu vực nhập liệu (Input), nút điều khiển (`Parameters`, `Run`, `Compare`), và khu vực hiển thị kết quả thời gian thực (Output & Telemetry).

```text
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 KAGGLE AI STUDIO    [Mode: ALL]  ● Online   GPU0: 8.1/16G │ GPU1: 7.8/16G   [🔗] [⏻] │
├──────────────────┬────────────────────────────────────────────────────────────────────┤
│                  │                                                                    │
│  WORKLOAD        │  💬 Chat Playground                                                │
│                  │  ────────────────────────────────────────────────────────────────  │
│  💬 Chat         │                                                                    │
│  👁 Vision       │  Model                                                             │
│  🎨 Image        │  [ Qwen3.8-27B-GGUF (Dual-T4 Q4_K_M)                   ▼ ]         │
│  🎬 Video        │                                                                    │
│  🔊 Audio        │  ┌──────────────────────────────────────────────────────────────┐  │
│                  │  │                                                              │  │
│  ──────────────  │  │  Nhập tin nhắn hoặc câu hỏi của bạn...                       │  │
│                  │  │                                                              │  │
│  SYSTEM          │  └──────────────────────────────────────────────────────────────┘  │
│  ◉ Models        │                                                                    │
│  ◉ GPU & VRAM    │  [+ Image]   [🧠 Thinking: OFF]   [⚙ Parameters]   [⚖ Compare] [Run ▶] │
│  ◉ Config/Admin  │                                                                    │
├──────────────────┴────────────────────────────────────────────────────────────────────┤
│  RESPONSE                                                                             │
│  ───────────────────────────────────────────────────────────────────────────────────  │
│  Qwen3.8-27B-GGUF                                                                     │
│                                                                                       │
│  Máy tính lượng tử (Quantum Computing) sử dụng các bit lượng tử (qubit) có khả năng   │
│  tồn tại ở trạng thái chồng chập...                                                   │
│                                                                                       │
│  ───────────────────────────────────────────────────────────────────────────────────  │
│  ⏱ 1.42s    •    ⚡ 48.5 tok/s    •    🔢 97 tokens    •    🖥 Dual-GPU (50/50)       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi Tiết Giao Diện Từng Phân Hệ WORKLOAD (Bao Phủ 100% API)

### 2.1. 💬 Chat Playground (`POST /v1/chat/completions`)

Hỗ trợ hội thoại nhiều lượt (Multi-turn conversation), System Prompt, bật/tắt chế độ suy luận sâu (`enable_thinking`) và nhận phản hồi từng từ qua **SSE Streaming**.

```text
Model                                              Chế độ suy luận
[ Qwen3.8-27B-GGUF                         ▼ ]     [ 🧠 Enable Thinking: ✓ ]

System Prompt (Tùy chọn)
[ Bạn là trợ lý AI chuyên gia bằng tiếng Việt...                             ]

Conversation History
┌────────────────────────────────────────────────────────────────────────────┐
│ 👤 User                                                                    │
│ Giải thích kiến trúc Mixture-of-Experts (MoE) ngắn gọn.                    │
│                                                                            │
│ 🤖 Assistant                                                               │
│ Mixture-of-Experts (MoE) chia mô hình thành nhiều mạng chuyên gia nhỏ...   │
└────────────────────────────────────────────────────────────────────────────┘

Tin nhắn mới
┌────────────────────────────────────────────────────────────────────────────┐
│ So sánh MoE với Dense Model trên 2x GPU Tesla T4...                        │
└────────────────────────────────────────────────────────────────────────────┘

[ + Image ]   [ 🗑 Clear Chat ]   [ ⚙ Parameters ]         [ ⚖ Compare ] [ Run ▶ ]
```

**Khi bấm vào `[ ⚙ Parameters ]` (ẩn mặc định để giữ UI gọn):**
```text
┌─ Chat & VLM Parameters ────────────────────────────────────────────────────┐
│ Temperature        ──────●───── 0.70     Top P              ──────●─── 0.90 │
│ Max Tokens         [ 1024     ]          Top K              [ 50          ] │
│ Repetition Penalty [ 1.05     ]          Presence Penalty   [ 0.00        ] │
│ Frequency Penalty  [ 0.00     ]          Stop Sequences     [ <|im_end|>  ] │
│ Streaming (SSE)    [ ✓        ]                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.2. 👁 Vision / VLM Playground (`POST /v1/chat/completions` — Multimodal)

Tối ưu riêng cho tác vụ phân tích hình ảnh, trích xuất văn bản (OCR), đọc biểu đồ với `Qwen3.8-27B-GGUF` + `mmproj-Qwen3.8-27B-Q8_0.gguf`.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                        🖼 Kéo & thả hình ảnh vào đây                        │
│                   [ Tải ảnh lên ]   hoặc   [ Dán URL ảnh ]                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Model
[ Qwen3.8-27B-GGUF (Vision Multimodal)     ▼ ]

Prompt yêu cầu phân tích
┌────────────────────────────────────────────────────────────────────────────┐
│ Mô tả chi tiết nội dung bức ảnh này và trích xuất toàn bộ chữ trong ảnh.   │
└────────────────────────────────────────────────────────────────────────────┘

[ 🧠 Thinking: OFF ]   [ ⚙ Parameters ]                              [ Run ▶ ]
```

---

### 2.3. 🎨 Image Playground (`POST /v1/images/generations` & `POST /v1/images/edits`)

Bao phủ cả 2 chế độ **Text-to-Image (T2I)** và **Inpaint / Image Editing** trên nền `Qwen-Image-2.1-GGUF` (ComfyUI backend), tích hợp thanh tiến trình **SSE Real-time Progress** chống lỗi Cloudflare 524 Timeout.

#### Chế độ A: Tab `[ ✨ Text-to-Image ]` (`POST /v1/images/generations`)
```text
Chế độ:  [● ✨ Text-to-Image ]   [○ 🖌 Inpaint / Edit Image ]

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                            [ Ảnh Kết Quả 1024×1024 ]                       │
│                                                                            │
│          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  Step 14/20 (70%) • 18.2s        │
│                                                                            │
│                     [ 🔍 Phóng to ]   [ ⬇ Tải PNG ]   [ 🖌 Chuyển qua Edit ]│
└────────────────────────────────────────────────────────────────────────────┘

Model
[ Qwen-Image-2.1-GGUF (Q4_K_M)             ▼ ]

Prompt
┌────────────────────────────────────────────────────────────────────────────┐
│ A cinematic cyberpunk street in Hanoi at night, neon signs, rain, 8k       │
└────────────────────────────────────────────────────────────────────────────┘

Negative Prompt (Tùy chọn)
[ blurry, low quality, distorted, watermark                                  ]

Size [ 1024×1024 ▼ ]     Steps [ 20 ]     CFG [ 3.5 ]     [ ⚙ Parameters ]   [ Generate ✨ ]
```

#### Chế độ B: Tab `[ 🖌 Inpaint / Edit Image ]` (`POST /v1/images/edits`)
```text
Chế độ:  [○ ✨ Text-to-Image ]   [● 🖌 Inpaint / Edit Image ]

┌─────────────────────────────────────┬──────────────────────────────────────┐
│ 🖼 Ảnh gốc & Cọ vẽ vùng chọn (Mask)  │ ✨ Ảnh sau khi chỉnh sửa (Output)    │
│                                     │                                      │
│   [ Canvas tô vùng Mask trực tiếp ] │        [ Kết quả Inpainting ]        │
│                                     │                                      │
│ [Tải ảnh gốc] [Tải file Mask] [Xóa] │            [ ⬇ Tải PNG ]             │
└─────────────────────────────────────┴──────────────────────────────────────┘
Cỡ cọ (Brush Size): ────●──── 32px      Độ mạnh (Strength): ──────●── 0.80

Prompt mô tả vùng cần thay thế / chỉnh sửa
┌────────────────────────────────────────────────────────────────────────────┐
│ Replace the background with a sunny tropical beach                         │
└────────────────────────────────────────────────────────────────────────────┘

Size [ 1024×1024 ▼ ]     Steps [ 20 ]     CFG [ 3.5 ]     [ ⚙ Parameters ]   [ Apply Edit 🖌 ]
```

**Drawer `[ ⚙ Parameters ]` của Image Playground:**
```text
┌─ Image Advanced Parameters ────────────────────────────────────────────────┐
│ Seed            [ Random (-1) ]        Sampler Name    [ euler         ▼ ] │
│ Scheduler       [ simple    ▼ ]        Response Format [ b64_json      ▼ ] │
│ SSE Streaming   [ ✓           ]                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.4. 🎬 Video Playground (`POST /v1/videos/generations`)

Bao phủ toàn diện họ mô hình tạo video **Wan2.2** (`Wan2.2-TI2V-5B`, `Wan2.2-T2V-A14B` Dual-Stage MoE, `Wan2.2-I2V-A14B`) cho cả **Text-to-Video (T2V)** và **Image-to-Video (I2V)**.

```text
Chế độ:  [● 🎥 Text-to-Video (T2V) ]   [○ 🖼 Image-to-Video (I2V) ]

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                    ▶ [ Trình phát Video MP4 (832×480 @ 16fps) ]            │
│                                                                            │
│          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  Step 11/20 (55%) • 44.5s        │
│                                                                            │
│                        [ 🔁 Loop: ON ]     [ ⬇ Tải Video MP4 ]             │
└────────────────────────────────────────────────────────────────────────────┘

(Chỉ hiện khi chọn Tab Image-to-Video)
┌────────────────────────────────────────────────────────────────────────────┐
│ 🖼 Ảnh khởi đầu (First Frame): [ Tải ảnh lên ] hoặc [ Kéo thả ảnh vào đây ] │
└────────────────────────────────────────────────────────────────────────────┘

Model
[ Wan2.2-TI2V-5B-GGUF (Fast Single/Dual GPU)                             ▼ ]
  ├─ Wan2.2-TI2V-5B-GGUF (Khuyên dùng - Nhanh, hỗ trợ cả T2V & I2V)
  ├─ Wan2.2-T2V-A14B-GGUF (Chất lượng cao - MoE HighNoise + LowNoise)
  └─ Wan2.2-I2V-A14B-GGUF (Chuyên biệt chuyển ảnh tĩnh thành Video 14B)

Prompt chuyển động
┌────────────────────────────────────────────────────────────────────────────┐
│ A golden retriever running happily on a sunny beach, slow motion, 4k       │
└────────────────────────────────────────────────────────────────────────────┘

Negative Prompt (Tùy chọn)
[ static, blurry, deformed, jerky motion, low resolution                     ]

Resolution [ 832×480 (16:9) ▼ ]   Frames (4k+1) [ 33 frames ▼ ]   FPS [ 16 ▼ ]
Steps      [ 20 ]                 CFG           [ 5.0 ]           [ ⚙ Parameters ] [ Generate Video 🎬 ]
```

**Drawer `[ ⚙ Parameters ]` của Video Playground:**
```text
┌─ Video Advanced Parameters (Wan2.2 ComfyUI) ───────────────────────────────┐
│ Shift           [ 8.0         ]        Seed            [ Random (-1)     ] │
│ Sampler Name    [ euler     ▼ ]        Scheduler       [ simple        ▼ ] │
│ High Noise Steps (A14B) [ Auto ]       Low Noise Steps (A14B) [ Auto     ] │
│ SSE Streaming   [ ✓           ]                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.5. 🔊 Audio Playground (`POST /v1/audio/transcriptions` & `POST /v1/audio/speech`)

Chia thành **2 Tab chuyên biệt** cho Nhận diện giọng nói (**STT**) và Tổng hợp / Nhân bản giọng nói (**TTS & Voice Cloning**).

#### Tab 1: `[ 🎙 Speech-to-Text (STT) ]` (`POST /v1/audio/transcriptions`)
```text
Chế độ Audio:  [● 🎙 Speech-to-Text (STT) ]   [○ 🗣 Text-to-Speech & Clone (TTS) ]

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│       [ 🔴 Thu âm trực tiếp từ Micro ]   hoặc   [ 📂 Tải file Audio ]      │
│                     Hỗ trợ: .wav, .mp3, .ogg, .m4a, .flac                  │
│                                                                            │
│       ▶ ──●────────────────────────── 00:04 / 00:12  (sample_vi.wav)       │
└────────────────────────────────────────────────────────────────────────────┘

Model                                  Ngôn ngữ              Tác vụ
[ Qwen3-ASR-1.7B (Multilingual)  ▼ ]   [ Tiếng Việt (vi) ▼ ] [ Transcribe ▼ ]
  ├─ Qwen3-ASR-1.7B (SOTA 50+ ngôn ngữ)
  └─ Whisper-Large-v3-Turbo (Fallback)

Context Prompt / Từ khóa chuyên ngành (Tùy chọn)
[ Tên riêng, thuật ngữ kỹ thuật...                                           ]

[ ✓ Trả kèm mốc thời gian (return_time_stamps) ]    [ ⚙ Parameters ]   [ Transcribe 🎙 ]

─── KẾT QUẢ NHẬN DIỆN (TRANSCRIPTION) ────────────────────────────────────────
┌────────────────────────────────────────────────────────────────────────────┐
│ Xin chào, đây là hệ thống nhận diện giọng nói tiếng Việt chạy trên Kaggle. │
│                                                                            │
│ ⏱ Timestamps:                                                              │
│ [00:00.00 → 00:02.10] Xin chào, đây là hệ thống nhận diện giọng nói        │
│ [00:02.10 → 00:04.50] tiếng Việt chạy trên Kaggle.                         │
└────────────────────────────────────────────────────────────────────────────┘
⏱ Thời lượng audio: 4.5s   •   ⚡ Thời gian xử lý: 0.38s   •   [ 📋 Copy Text ]
```

#### Tab 2: `[ 🗣 Text-to-Speech & Clone (TTS) ]` (`POST /v1/audio/speech`)
Hỗ trợ đầy đủ **3 phương thức tạo giọng** của `Qwen3-TTS-1.7B` và `XTTS-v2 Vietnamese`:
1. **Preset Voice** (Giọng chuẩn có sẵn)
2. **Voice Cloning** (Nhân bản giọng nói từ file mẫu 3–15s + ICL Reference Text)
3. **Voice Design** (Thiết kế giọng mới bằng câu lệnh mô tả chất giọng)

```text
Chế độ Audio:  [○ 🎙 Speech-to-Text (STT) ]   [● 🗣 Text-to-Speech & Clone (TTS) ]

Model / Engine                         Ngôn ngữ              Tốc độ đọc (Speed)
[ XTTS-v2 Vietnamese (Clone/Vi)  ▼ ]   [ Tiếng Việt (vi) ▼ ] ─────●──── 1.0x
  ├─ XTTS-v2 Vietnamese (Tối ưu Tiếng Việt & Clone giọng)
  ├─ Qwen3-TTS-1.7B (Đa ngôn ngữ, VoiceDesign, ICL Clone)
  └─ Kokoro-82M (Siêu nhẹ, Tiếng Anh)

Phương thức Giọng đọc (Voice Mode):
  (●) 🎧 Giọng có sẵn (Preset Voice):   [ serena ▼ ] (serena, vivian, uncle_fu, default...)
  (○) 🧬 Nhân bản giọng (Voice Clone):  [ 📂 Upload/Thu âm giọng mẫu (.wav 3-15s) ]
      └─ Văn bản của file giọng mẫu (ref_text cho Qwen3-TTS ICL):
         [ Nhập nội dung câu nói trong file mẫu...                           ]
         [ ☐ Chỉ dùng X-Vector (x_vector_only_mode - không cần ref_text)     ]
  (○) ✨ Thiết kế giọng (Voice Design):
      └─ Mô tả chất giọng (voice_description):
         [ Giọng nữ miền Bắc ấm áp, truyền cảm, nhịp điệu kể chuyện nhẹ nhàng ]

Văn bản cần đọc (Input Text)
┌────────────────────────────────────────────────────────────────────────────┐
│ Chào mừng bạn đến với hệ thống tổng hợp và nhân bản giọng nói AI Studio.   │
└────────────────────────────────────────────────────────────────────────────┘

[ ⚙ Parameters ]                                            [ Synthesize 🔊 ]

─── KẾT QUẢ ÂM THANH (AUDIO OUTPUT) ──────────────────────────────────────────
┌────────────────────────────────────────────────────────────────────────────┐
│  ▶ ────●──────────────────────────── 00:01 / 00:05   ▂▄▆█▃▂▅▇█▆▄▂▃▅▆▄▂     │
│                                                                            │
│  Định dạng: WAV (24000 Hz)                  [ ⬇ Tải file .wav ]            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Drawer `[ ⚙ Parameters ]` của TTS (`/v1/audio/speech`):**
```text
┌─ TTS & Voice Cloning Hyperparameters ──────────────────────────────────────┐
│ Sample Rate        [ 24000 Hz  ▼ ]     Response Format    [ wav        ▼ ] │
│ Temperature        [ 0.75        ]     Top P              [ 0.85         ] │
│ Top K              [ 50          ]     Repetition Penalty [ 5.0          ] │
│ Length Penalty     [ 1.0         ]     Max New Tokens     [ 2048         ] │
│ GPT Cond Len (XTTS)[ 30s         ]     Max Ref Len (XTTS) [ 60s          ] │
│ Sound Norm Refs    [ ☐           ]     Text Splitting     [ ✓            ] │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi Tiết Giao Diện Nhóm SYSTEM (Quản Trị Tài Nguyên & Kaggle Kernel)

### 3.1. ◉ Models & Diagnostics (`GET /v1/models` & `GET /v1/audio/diagnostics`)

Hiển thị trạng thái nạp thực tế của từng mô hình và các daemon nội bộ (`llama-server`, `ComfyUI`, `AudioEngines`).

```text
MODELS & BACKEND ENGINES                              [ 🔄 Làm mới ] [ 🩺 Chẩn đoán Audio ]

┌────────────────────────────────────────────────────────────────────────────┐
│ 👁 Qwen3.8-27B-GGUF                                         ● LOADED (GPU) │
│ VLM • llama-server (:8081) • Dual-GPU Split (50/50) • Quantization: Q4_K_M │
│                                                        [ Mở trong VLM ▶ ]  │
├────────────────────────────────────────────────────────────────────────────┤
│ 🎨 Qwen-Image-2.1-GGUF                                      ○ STANDBY      │
│ Image T2I & Inpaint • ComfyUI (:8188) • Dual-GPU • Quantization: Q4_K_M    │
│                                                        [ Mở trong Image ▶ ]│
├────────────────────────────────────────────────────────────────────────────┤
│ 🎬 Wan2.2 (TI2V-5B / A14B)                                  ○ STANDBY      │
│ Video T2V & I2V • ComfyUI (:8188) • Quantization: Q4_K_M                   │
│                                                        [ Mở trong Video ▶ ]│
├────────────────────────────────────────────────────────────────────────────┤
│ 🔊 Speech Engines (Qwen3-ASR 1.7B / Qwen3-TTS 1.7B / XTTS-v2 Vi)   ● READY │
│ Audio Service • GPU 0 (ASR/Qwen-TTS) + GPU 1 (XTTS-v2) • Precision: FP16   │
│                                                        [ Mở trong Audio ▶ ]│
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.2. ◉ GPU & VRAM Manager (`GET /v1/memory` & `POST /v1/memory/free`)

Giám sát trực quan 2 card **NVIDIA Tesla T4 (2×16GB VRAM)** và **System RAM (30GB)** trên Kaggle, kèm nút giải phóng nóng toàn bộ VRAM mà không cần tắt Server.

```text
HARDWARE & VRAM TELEMETRY                               [ 🧹 Giải phóng 100% VRAM ]
                                                        (POST /v1/memory/free)
┌────────────────────────────────────────────────────────────────────────────┐
│ 🖥 GPU 0: NVIDIA Tesla T4 (16.0 GB)                                        │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  8.12 GB / 15.84 GB (Còn trống: 7.72 GB)    │
│                                                                            │
│ 🖥 GPU 1: NVIDIA Tesla T4 (16.0 GB)                                        │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  7.85 GB / 15.84 GB (Còn trống: 7.99 GB)    │
│                                                                            │
│ 🧠 System CPU RAM (30.0 GB)                                                │
│ ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░  8.40 GB / 30.00 GB (28.0%)                 │
├────────────────────────────────────────────────────────────────────────────┤
│ 📌 Active Dynamic Slot : vlm (Qwen3.8-27B-GGUF)                            │
│ 🔄 Cached Dynamic Slots: [vlm]                                             │
│ ⏱ Idle Auto-Shutdown   : Tự tắt sau 45 phút không hoạt động (Đã nghỉ: 12s) │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.3. ◉ Config, Connection & Kernel Shutdown (`GET/PATCH /v1/config` & `POST /v1/admin/shutdown`)

Giải quyết bài toán đặc thù của Kaggle:
1. **Tự động bắt URL Cloudflare Tunnel mới nhất từ `ntfy.sh/<NTFY_TOPIC>`** mà không cần mở log Kaggle.
2. **Hot-reload cấu hình Runtime (`PATCH /v1/config`)** ngay trên giao diện.
3. **Tắt nóng Kaggle Commit Kernel (`POST /v1/admin/shutdown?confirm=1`)** để dừng tính giờ GPU ngay lập tức khi không dùng nữa.

```text
GATEWAY CONNECTION & KAGGLE KERNEL CONTROL

┌─ 1. Kết nối Cloudflare Tunnel & Xác thực ──────────────────────────────────┐
│ Ntfy Topic tự động : [ studio-ai-url-yourname ]  [ 📡 Tự lấy URL mới nhất ]│
│ Gateway Base URL   : [ https://xxxx-xxxx.trycloudflare.com               ] │
│ Studio API Key     : [ •••••••••••••••••••••• ]  [ 💾 Lưu & Kiểm tra Ping] │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 2. Cập nhật nóng Cấu hình Runtime (Hot-Reload /v1/config) ────────────────┐
│ Default STT Language : [ vi ▼ ]          Default TTS Voice : [ serena  ▼ ] │
│ Default Image Steps  : [ 20   ]          Default Video FPS : [ 16      ▼ ] │
│                                                  [ ⚡ Áp dụng Cấu hình ]   │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 3. Quản lý Phiên Kaggle (Bảo vệ hạn mức 30h GPU/tuần) ────────────────────┐
│ Trạng thái Kernel : ● Đang chạy (Mode: all)                                │
│ Khi ngừng sử dụng, hãy tắt Kernel để Kaggle thu hồi máy ảo và ngừng trừ    │
│ giờ GPU (Server sẽ xả sạch VRAM và thoát với mã 99).                       │
│                                                                            │
│                                  [ ⏻ TẮT KAGGLE KERNEL NGAY (SHUTDOWN) ]   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Kiến Trúc UI 3 Tầng & Cơ Chế Compare Thông Minh

```text
                              FACTORY PLAYGROUND UI
                                        │
       ┌─────────────┬─────────────┬────┴────────┬─────────────┬─────────────┐
       ▼             ▼             ▼             ▼             ▼             ▼
   💬 Chat       👁 Vision     🎨 Image      🎬 Video      🔊 Audio     ⚙ System/GPU
       │             │         (T2I/Edit)    (T2V/I2V)     (STT/TTS)   (Memory/Admin)
       └─────────────┴─────────────┴────┬────────┴─────────────┴─────────────┘
                                        ▼
                        Cloudflare Tunnel / FastAPI Gateway (:8000)
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
       llama-server (:8081)     ComfyUI Daemon (:8188)   AudioEngines (:8288)
        (Qwen3.8-27B GGUF)      (Qwen-Image & Wan2.2)   (Qwen3-ASR/TTS & XTTS)
                 │                      │                      │
                 └──────────────────────┼──────────────────────┘
                                        ▼
                         Kaggle 2x NVIDIA Tesla T4 (32GB VRAM)
```

### Lưu ý kỹ thuật cho nút `[ ⚖ Compare ]` trên Kaggle 2x T4:
- Khi người dùng muốn so sánh 2–3 model hoặc 2 cấu hình tham số (ví dụ so sánh giọng đọc `Qwen3-TTS` vs `XTTS-v2`, hoặc so sánh video `Wan2.2-5B` vs `Wan2.2-A14B`) bằng nút **`[ ⚖ Compare ]`**:
- Thay vì bắn request song song (dễ gây tranh chấp VRAM trên 2 card T4 16GB), UI sẽ thực thi theo **Hàng đợi Tuần tự (Sequential Compare Queue)**: chạy xong Model A → thu thập kết quả + `Latency` + `tok/s` → tự động chạy tiếp Model B → hiển thị kết quả **Song song cạnh nhau (Side-by-Side)** để đối chiếu trực quan.