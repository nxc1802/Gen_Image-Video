# 🎬 Wan2.1 Video Studio & VRAM Profiler

Hệ thống triển khai mô hình AI tạo video tiên tiến **Wan2.1 (Wan-AI/Wan2.1-T2V-1.3B-Diffusers & 14B)** với giao diện phản ứng thời gian thực (Marimo Reactive Studio) và bộ đo lường chuyên sâu mức tiêu thụ VRAM (Granular VRAM Profiler) qua từng giai đoạn và từng cụm block trong Pipeline.

---

## 📁 Cấu Trúc Thư Mục

```tree
marimo/wan/
├── wan.py                               # Ứng dụng Marimo Reactive Studio tương tác sinh video & Host RAM Swap
├── benchmark_vram.py                    # Script đo đạc & phân tích VRAM chi tiết từng giai đoạn & cụm block
├── wan21_vram_benchmark_report.md       # Báo cáo kết quả VRAM chuyên sâu & đối soát 3 Test Cases
├── wan21_3cases_benchmark.json          # Dữ liệu VRAM chi tiết 3 Test Cases dạng JSON
├── wan21_testcase1_ultralow_832x480.mp4 # Video Test Case 1: Ultra-Low VRAM (Đỉnh 3.66 GB VRAM)
├── wan21_testcase2_balanced_720p_33f.mp4# Video Test Case 2: Balanced HD 720p 33f (Đỉnh 4.44 GB VRAM)
├── wan21_testcase3_extreme_4k_32fps.mp4 # Video Test Case 3: Extreme 4K @ 32 FPS (Đỉnh 10.57 GB VRAM)
├── wan21_4k_sample_32fps.mp4            # Video mẫu 4K UHD thực tế
└── README.md                            # Tài liệu hướng dẫn sử dụng và thông số kỹ thuật
```

---

## ⚡ 6 Trụ Cột Tối Ưu Hóa VRAM Đã Triển Khai

1. **WanTransformer3D Precision**: Chạy hoàn toàn trên `torch.bfloat16` giúp giảm 50% dung lượng DiT và dynamic activations.
2. **Cơ Chế "Host RAM Swap" Cho Text Encoder (Tiết Kiệm 10.58 GB VRAM)**:
   - Nạp `UMT5-XXL` trên **CPU System RAM**.
   - Khi có Prompt mới: Swap lên GPU (`text_encoder.to("cuda")`) trong ~1.4s, mã hóa Prompt trong **0.037s**.
   - Lập tức đẩy về CPU RAM (`text_encoder.to("cpu")`), thu hồi ngay lập tức **10.58 GB VRAM** trên GPU.
   - Giúp đỉnh VRAM khi khử nhiễu rơi xuống mức kỷ lục **`3.66 GB`** (giảm **79.5%** so với 17.86 GB ban đầu)!
3. **Quantization 4-bit (NF4 & GGUF Q4_K_M)**: Hỗ trợ BitsAndBytes 4-bit NF4 và GGUF Q4_K_M.
4. **PyTorch SDPA & FlashAttention**: Ép kích hoạt FlashAttention & Mem-Efficient Attention, chuyển độ phức tạp bộ nhớ attention về $O(1)$.
5. **3D VAE Tiling & Slicing**: Kích hoạt `enable_tiling()` và `enable_slicing()` cho Spatiotemporal VAE.
6. **CUDA Allocator & Runtime**: Cấu hình `expandable_segments:True`, `torch.inference_mode()`, và Garbage Collection chủ động.

---

## 📊 Kết Quả Thực Nghiệm 3 Mức Cấu Hình

| Tiêu Chí | Test Case 1: Ultra-Low VRAM | Test Case 2: Balanced Production | Test Case 3: Extreme High-Res 4K |
| :--- | :--- | :--- | :--- |
| **Độ Phân Giải** | **`832 x 480`** | **`1280 x 720 (720p HD)`** | **`3840 x 2160 (4K UHD)`** |
| **Số Khung Hình**| `17 frames @ 16 FPS` | **`33 frames @ 24 FPS`** | `17 frames @ 32 FPS` |
| **Đỉnh VRAM** | **`3.66 GB`** (Giảm 79.5%!) | **`4.44 GB`** | **`10.57 GB`** (Giảm 78.1%!) |
| **Reserved VRAM**| `4.21 GB` | `5.35 GB` | `14.02 GB` |
| **Thời Gian Denoise**| `4.14s` (`2.42 steps/s`) | `26.34s` (`0.38 steps/s`) | `338.09s` (`0.03 steps/s`) |
| **Tệp Video** | `wan21_testcase1_ultralow_832x480.mp4` | `wan21_testcase2_balanced_720p_33f.mp4` | `wan21_testcase3_extreme_4k_32fps.mp4` |
| **Phần Cứng Tối Thiểu**| **GPU 6GB - 8GB** (RTX 3060/4060) | **GPU 8GB - 16GB** | **GPU 12GB - 16GB** (RTX 4070/4080) |

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Khởi chạy Marimo Studio
```bash
marimo edit marimo/wan/wan.py
```
Hoặc chạy dạng Web App tương tác:
```bash
marimo run marimo/wan/wan.py --port 8080 --host 0.0.0.0
```

### 2. Chạy Benchmark Đo VRAM Từng Giai Đoạn
```bash
python marimo/wan/benchmark_vram.py
```

