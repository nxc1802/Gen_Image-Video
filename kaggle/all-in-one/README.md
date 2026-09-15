# 🎨 Kaggle All-in-One AI Studio (2x Tesla T4 16GB)

Hệ thống AI Studio hợp nhất toàn diện, gom trọn 5 trụ cột AI đỉnh cao hiện nay vào **1 Session Kaggle duy nhất**:
1. 👁️ **VLM (Vision-Language)**: **Qwen 26B (4-bit AWQ/GGUF)** — Phân bổ song song qua 2 card GPU.
2. 🎙️ **STT (Speech-to-Text)**: **Whisper-large-v3-turbo (Bản FULL FP16)** — Giữ 100% độ chính xác âm thanh gốc trên GPU 0 (~1.6GB VRAM).
3. 🔊 **TTS (Text-to-Speech)**: **Kokoro-82M (Bản FULL FP16)** — Giữ 100% nhạc tính, giọng người thật trên GPU 0 (~350MB VRAM).
4. 🖼️ **GenImage (Tạo ảnh)**: **FLUX.1-schnell (4-bit NF4)** — Tạo ảnh 1024x1024 trong 4 bước (~8.5GB VRAM trên GPU 1).
5. 🎬 **GenVideo (Tạo video)**: **Wan2.1-1.3B** — Tạo video điện ảnh mượt mà trên GPU 1.
6. 🌐 **OpenAI Gateway**: Tương thích 100% chuẩn OpenAI API (`/v1/chat/completions`, `/v1/audio/*`, `/v1/images/*`, `/v1/videos/*`).

---

## 📁 Cấu Trúc Thư Mục Modular

Toàn bộ dự án được thiết kế theo tư duy kỹ thuật phần mềm chuẩn (Clean Code Architecture):

```tree
kaggle/all-in-one/
├── PLAN.md                     # Tài liệu thiết kế kiến trúc chi tiết
├── README.md                   # Hướng dẫn sử dụng & gọi API
├── requirements.txt            # Danh sách thư viện Python
├── run_kaggle.ipynb            # Notebook bootstrap 1-click trên Kaggle
├── main.py                     # Entrypoint điều phối toàn bộ Studio
├── config.py                   # Cấu hình tập trung (Model IDs, VRAM, Ports, Precision)
├── core/
│   ├── __init__.py
│   └── memory_manager.py       # Điều phối hoán đổi VRAM GPU 1 <-> 30GB CPU RAM
├── audio/
│   ├── __init__.py
│   ├── stt.py                  # Whisper Turbo (Bản FULL FP16 trên GPU 0)
│   └── tts.py                  # Kokoro-82M (Bản FULL FP16 trên GPU 0)
├── vlm/
│   ├── __init__.py
│   └── qwen.py                 # Qwen 26B 4-bit (Song song GPU 0 và GPU 1)
├── visual/
│   ├── __init__.py
│   ├── flux_image.py           # FLUX.1-schnell (NF4 trên GPU 1)
│   └── wan_video.py            # Wan2.1-1.3B (trên GPU 1)
├── server/
│   ├── __init__.py
│   ├── app.py                  # FastAPI Server chuẩn OpenAI
│   └── schemas.py              # Pydantic Schemas
└── tunnel/
    ├── __init__.py
    ├── cloudflare.py           # Cloudflare Quick Tunnel tự động mở URL Public
    └── supabase_queue.py       # Supabase Message Broker (kết nối bền bỉ vĩnh viễn)
```

---

## 🚀 Hướng Dẫn Chạy Trên Kaggle (Chỉ 2 Bước)

### Bước 1: Tạo Notebook trên Kaggle
1. Vào [Kaggle Notebooks](https://www.kaggle.com/code) $\rightarrow$ bấm **New Notebook**.
2. Tại bảng bên phải (**Notebook options**):
   * **Accelerator**: Chọn **GPU T4 x2**.
   * **Internet**: Chuyển sang **Internet on**.

### Bước 2: Import & Chạy
* Tải file [`run_kaggle.ipynb`](file:///Volumes/WorkSpace/Project/Gen_Image:Video/kaggle/all-in-one/run_kaggle.ipynb) lên và bấm **Run All**.
* Hoặc chỉ cần copy cell này vào notebook trống trên Kaggle:
```python
!git clone https://github.com/nxc1802/Gen_Image-Video.git
%cd Gen_Image-Video/kaggle/all-in-one
!pip install -q -r requirements.txt
!python main.py
```

Khi chạy xong, terminal sẽ in ra URL Public của Cloudflare:
```
======================================================================
🎉 CLOUDFLARE QUICK TUNNEL ĐÃ KÍCH HOẠT THÀNH CÔNG!
👉 Public Base URL : https://choice-vhs-alive-turner.trycloudflare.com/v1
👉 OpenAI SDK     : client = OpenAI(base_url='https://.../v1', api_key='dummy')
======================================================================
```

---

## 💻 Hướng Dẫn Sử Dụng API Từ Máy Local / Ứng Dụng

Bạn có thể dùng thư viện chính thức `openai-python` để gọi trực tiếp tới Kaggle:

### 1. Trò chuyện & Thị giác VLM (Qwen 26B)
```python
from openai import OpenAI

client = OpenAI(base_url="https://your-tunnel.trycloudflare.com/v1", api_key="dummy")

res = client.chat.completions.create(
    model="qwen-26b",
    messages=[
        {"role": "user", "content": "Phân tích kiến trúc của một hệ thống AI đa phương thức."},
    ],
)
print(res.choices[0].message.content)
```

### 2. Tạo ảnh FLUX.1 (1024x1024)
```python
img_res = client.images.generate(
    model="flux-1-schnell",
    prompt="A futuristic mechanical tiger with neon circuitry on a rainy Tokyo rooftop, 8k render",
    size="1024x1024",
)
# Lấy base64 ảnh:
b64 = img_res.data[0].b64_json
```

### 3. Nhận diện giọng nói STT (Whisper Turbo Full FP16)
```python
with open("recording.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=audio_file,
    )
print(transcript.text)
```

### 4. Đọc văn bản thành giọng nói TTS (Kokoro-82M Full FP16)
```python
response = client.audio.speech.create(
    model="kokoro-82m",
    voice="af_heart",
    input="Xin chào! Tôi là trợ lý AI thế hệ mới vận hành trên cụm GPU Kaggle của bạn.",
)
response.stream_to_file("output.wav")
```
