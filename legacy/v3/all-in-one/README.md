# 🎨 Kaggle All-in-One AI Studio (2x Tesla T4 16GB)

Hệ thống AI Studio hợp nhất toàn diện, gom trọn 5 trụ cột AI đỉnh cao SOTA hiện nay vào **1 Session Kaggle duy nhất** (32GB VRAM gộp + 30GB CPU RAM):
1. 👁️ **VLM (Vision-Language)**: **Qwen 26B (4-bit AWQ/GGUF)** — Phân bổ song song qua 2 card GPU Tesla T4.
2. 🎙️ **STT (Speech-to-Text)**: **Whisper-large-v3-turbo (Bản FULL FP16)** — Giữ 100% độ chính xác âm thanh gốc trên GPU 0 (~1.6GB VRAM).
3. 🔊 **TTS (Text-to-Speech)**: **Kokoro-82M (Bản FULL FP16)** — Giữ 100% nhạc tính, giọng người thật trên GPU 0 (~350MB VRAM).
4. 🖼️ **GenImage (Tạo ảnh)**: **FLUX.2-klein-4B (4-bit GGUF)** — Tạo ảnh 1024x1024 siêu tốc 4 steps, chạy trọn gói trên 1 GPU (~5.2GB VRAM).
5. 🎬 **GenVideo (Tạo video)**: **Wan2.1-14B (Flagship 14 Tỷ params SOTA, 4-bit)** — Tạo video điện ảnh hàng đầu thế giới qua cơ chế PCIe Swapping.
6. ⚡ **Unified Memory Orchestrator**: Hoán đổi tức thì giữa CPU RAM $\leftrightarrow$ GPU VRAM qua bus PCIe tốc độ 14GB/s (Zero Disk I/O).
7. 🌐 **OpenAI Gateway**: Tương thích 100% chuẩn OpenAI API (`/v1/chat/completions`, `/v1/audio/*`, `/v1/images/*`, `/v1/videos/*`).

---

## 📁 Cấu Trúc Thư Mục Modular

Toàn bộ dự án được thiết kế theo tư duy kỹ thuật phần mềm chuẩn (Clean Code Architecture):

```tree
kaggle/all-in-one/
├── PLAN.md                     # Tài liệu thiết kế kiến trúc chi tiết
├── README.md                   # Hướng dẫn sử dụng & gọi API
├── models.yaml                 # Khai báo cấu hình mô hình 1-Click
├── requirements.txt            # Danh sách thư viện Python
├── run_kaggle.ipynb            # Notebook bootstrap 1-click trên Kaggle
├── test_studio.ipynb           # Notebook test tương tác (Hiển thị Audio, Image, Video ngay dưới cell)
├── main.py                     # Entrypoint điều phối toàn bộ Studio
├── config.py                   # Cấu hình nạp models.yaml & fallback
├── core/
│   ├── __init__.py
│   ├── base_engine.py          # Abstract Base Classes (Template cho mọi model)
│   ├── device_resolver.py      # Trí thông minh tự động phân bổ GPU & Topology
│   ├── model_registry.py       # Factory & Registry điều phối vòng đời mô hình
│   └── memory_manager.py       # Điều phối RAM ↔ VRAM PCIe Fast Swapping
├── audio/
│   ├── __init__.py
│   ├── stt.py                  # Whisper Turbo Adapter (Bản FULL FP16 trên GPU 0)
│   └── tts.py                  # Kokoro-82M Adapter (Bản FULL FP16 trên GPU 0)
├── vlm/
│   ├── __init__.py
│   └── qwen.py                 # Qwen VLM Adapter (Tự động 1 GPU nếu <= 8B, 2 GPU nếu >= 14B)
├── visual/
│   ├── __init__.py
│   ├── flux_image.py           # FLUX.2 Adapter (4B GGUF DiT + Qwen3-4B GGUF Text Encoder)
│   └── wan_video.py            # Wan2.1 Adapter (14B params SOTA 4-bit)
├── server/
│   ├── __init__.py
│   ├── app.py                  # FastAPI Server chuẩn OpenAI
│   └── schemas.py              # Pydantic Schemas
└── tunnel/
    ├── __init__.py
    └── cloudflare.py           # Cloudflare Quick Tunnel tự động mở Public URL
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
