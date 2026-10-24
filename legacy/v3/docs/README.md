# 🚀 Kaggle AI Studio Factory (v3.5)

Hệ thống **Zero-Install Offline AI Factory** triển khai tự động trên **Kaggle (2x NVIDIA Tesla T4 — 32GB VRAM)** thông qua 1 lệnh duy nhất, cung cấp chuẩn **OpenAI-Compatible REST API** qua **Cloudflare Quick Tunnel** và thông báo URL tức thì qua **ntfy.sh**.

---

## ✨ Kiến Trúc 5 Phân Hệ (`STUDIO_MODE`)

| Phân hệ | Lệnh Deploy nhanh | Mô hình AI chủ lực | Hạ tầng Thực thi |
| :--- | :--- | :--- | :--- |
| **🎙️ Speech Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py speech` | • **STT**: `Qwen/Qwen3-ASR-1.7B`<br>• **TTS**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`<br>• **Vi-TTS & Clone**: `XTTS-v2 Vietnamese` | PyTorch Native (`cuda:0` + `cuda:1`) |
| **👁️ VLM Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py vlm` | • **VLM**: `Qwen3.8-27B-Q4_K_M.gguf`<br>• **Vision**: `mmproj-Qwen3.8-27B-Q8_0.gguf` | `llama-server` Dual-GPU (`--tensor-split 0.5,0.5`, KV Cache `q4_0`) |
| **🖼️🎬 Image & Video** | `python3 kaggle/all-in-one/deploy_kaggle.py image_video` | • **Image**: `Qwen-Image-2.1-GGUF` (T2I & Inpaint)<br>• **Video**: `Wan2.2-TI2V-5B`, `Wan2.2-A14B` (T2V & I2V) | Headless `ComfyUI` + `ComfyUI-GGUF` + `VideoHelperSuite` |
| **🧬 Embedding Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py embedding` | • **Embed**: `Qwen3-Embedding-0.6B & 4B`, `Qwen3-VL-Embedding-2B & 8B`<br>• **Rerank**: `Qwen3-Reranker-0.6B & 4B`, `Qwen3-VL-Reranker-2B & 8B` | Official `llama-server` / `llama_cpp` (GPU 0: 0.6B/4B/2B, GPU 1: 8B) |
| **🌐 All-in-One Studio** | `python3 kaggle/all-in-one/deploy_kaggle.py all` | Hợp nhất toàn bộ mô hình STT, TTS, VLM, Image, Video, Embedding & Rerank | `MemoryManager` tự động hoán đổi sạch VRAM |

---

## 📂 Cấu Trúc Thư Mục Tinh Gọn

```text
Gen_Image:Video/
├── .env.example                            # Mẫu cấu hình biến môi trường
├── USAGE_GUIDE.md                          # Hướng dẫn sử dụng, API Endpoints & Shutdown chi tiết
├── test_embedding.py                       # Benchmark cục bộ phân hệ Embedding & Rerank trên tài liệu md
└── kaggle/all-in-one/
    ├── deploy_kaggle.py                    # Trình đóng gói chọn lọc (Selective Packer) & Kaggle Push CLI
    ├── download_embedding_dataset.py       # Tải 8 model GGUF Q4_K_M về Kaggle Dataset
    ├── main.py                             # Entrypoint khởi chạy FastAPI Gateway & Cloudflare Tunnel
    ├── config.py                           # Quản lý cấu hình & tự động dò tìm đường dẫn Dataset
    ├── models.yaml                         # Cấu hình tổng hợp
    ├── configs/                            # Cấu hình YAML chuyên biệt (speech, vlm, image_video, embedding)
    ├── core/                               # Quản lý VRAM (MemoryManager), ModelRegistry & ComfyUIManager
    ├── audio/                              # Engine STT/TTS (Qwen3-ASR, Qwen3-TTS, XTTS-v2 Vietnamese)
    ├── vlm/                                # Engine VLM (Qwen3.8-27B GGUF Dual-GPU llama-server)
    ├── visual/                             # Engine Image & Video (ComfyUI Qwen-Image-2.1 & Wan2.2)
    ├── embedding/                          # Engine Embedding & Reranker (Qwen3 Embed & Rerank Q4_K_M)
    ├── server/                             # FastAPI OpenAI-Compatible Gateway & Pydantic Schemas
    ├── tunnel/                             # Cloudflare Quick Tunnel & ntfy.sh URL Broadcaster
    ├── run_speech.ipynb                    # Notebook chuyên biệt Speech Studio
    ├── run_vlm.ipynb                       # Notebook chuyên biệt VLM Studio
    ├── run_image_video.ipynb               # Notebook chuyên biệt Image & Video Studio
    ├── run_embedding.ipynb                 # Notebook chuyên biệt Embedding & Rerank Studio
    ├── run_kaggle.ipynb                    # Notebook tổng hợp All-in-One Studio
    ├── build_wheels_dataset.py             # Công cụ đóng gói Offline Wheels (.tar.bin) cho 4 phân hệ
    └── deploy_downloader.py                # Công cụ tải Model từ HuggingFace sang Kaggle Dataset
```

---

## 📖 Tài Liệu Hướng Dẫn Chi Tiết

Xem toàn bộ hướng dẫn cấu hình `.env`, danh sách các API Endpoint kèm ví dụ `curl`, và cách Shutdown bảo toàn 30 giờ GPU/tuần tại:
👉 **[USAGE_GUIDE.md](USAGE_GUIDE.md)**
