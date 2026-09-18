#!/usr/bin/env python3
"""
📊 Wan2.1 Granular VRAM Profiler & Benchmark Engine
---------------------------------------------------
Đo lường và phân tích chi tiết mức tiêu thụ VRAM (Allocated, Reserved, Peak)
qua từng giai đoạn (Stages) và từng cụm Block chính của toàn bộ Pipeline Wan2.1:
1. Baseline & Từng thành phần mô hình (Text Encoder, DiT Backbone, VAE)
2. Stage 1: Text Encoding (UMT5-XXL Tokenization & Forward Pass)
3. Stage 2: Latent Preparation (3D Spatiotemporal Noise & Timestep Schedule)
4. Stage 3: Diffusion Denoising Backbone (WanTransformer3D - 30 Blocks):
            - Patch Embedding & 3D RoPE
            - Cluster 1: Blocks 0-9 (Early Spatiotemporal Representation)
            - Cluster 2: Blocks 10-19 (Deep Text-Video Cross-Attention)
            - Cluster 3: Blocks 20-29 (High-Frequency Video Synthesis)
            - Final Head & Projection (norm_out + proj_out)
5. Stage 4: 3D VAE Decoding (AutoencoderKLWan Latents -> Video RGB)
6. Stage 5: Post-Processing & MP4 Video Export
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import torch

def get_vram_mb(device=0) -> Dict[str, float]:
    """Lấy số liệu VRAM hiện tại tính bằng MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "reserved": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "max_allocated": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
    }

def format_mb(val_mb: float) -> str:
    if val_mb >= 1024:
        return f"{val_mb / 1024:.2f} GB ({val_mb:.1f} MB)"
    return f"{val_mb:.1f} MB"

def calc_module_vram(module: torch.nn.Module) -> float:
    """Tính toán dung lượng tham số và buffer của một module (MB)."""
    mem_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    mem_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    return mem_bytes / (1024 * 1024)

class WanVRAMProfiler:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        width: int = 832,
        height: int = 480,
        num_frames: int = 17,
        num_steps: int = 25,
        guidance_scale: float = 5.0,
        fps: int = 16,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.fps = fps

        self.reports: List[Dict[str, Any]] = []
        self.hardware_info: Dict[str, Any] = {}
        self.components_vram: Dict[str, Any] = {}
        self.block_cluster_stats: Dict[str, Any] = {}

    def log_stage(self, stage_name: str, category: str, details: str = ""):
        torch.cuda.synchronize(self.device)
        mem = get_vram_mb(self.device)
        entry = {
            "timestamp": time.time(),
            "stage": stage_name,
            "category": category,
            "details": details,
            "allocated_mb": round(mem["allocated"], 2),
            "reserved_mb": round(mem["reserved"], 2),
            "peak_allocated_mb": round(mem["max_allocated"], 2),
        }
        self.reports.append(entry)
        print(f"[{category:12}] {stage_name:<42} | Alloc: {format_mb(mem['allocated']):<18} | Peak: {format_mb(mem['max_allocated']):<18} | Res: {format_mb(mem['reserved'])}")
        return entry

    def collect_hardware_info(self):
        if not torch.cuda.is_available():
            self.hardware_info = {"device": "CPU", "total_vram_mb": 0}
            return
        props = torch.cuda.get_device_properties(self.device)
        self.hardware_info = {
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_mb": round(props.total_memory / (1024 * 1024), 2),
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "pytorch_version": torch.__version__,
        }
        print("=" * 105)
        print(f"🚀 PHẦN CỨNG: {self.hardware_info['device_name']} ({self.hardware_info['total_vram_gb']} GB VRAM)")
        print(f"   PyTorch: {self.hardware_info['pytorch_version']} | BF16 Hỗ Trợ: {self.hardware_info['bf16_supported']}")
        print("=" * 105)

    def run_benchmark(self, prompt: str = "A magnificent golden dragon soaring gracefully through misty neon clouds above a cyberpunk metropolis at dusk, cinematic camera panning, 4k ultra-detailed, photorealistic fluid motion, vivid reflections") -> Dict[str, Any]:
        self.collect_hardware_info()

        # 0. Baseline Initial VRAM
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.log_stage("0. Baseline (GPU Idle)", "Init", "Trước khi nạp bất kỳ model nào vào GPU")

        from diffusers import WanPipeline
        from diffusers.utils import export_to_video

        # 1. Loading Pipeline
        print("\n--- [GIAI ĐOẠN 0: NẠP TRỌNG SỐ MÔ HÌNH VÀO VRAM] ---")
        t_load_start = time.time()

        pipe = WanPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.log_stage("0.1 Pipeline Khởi Tạo (Host RAM)", "Model Load", "Nạp cấu trúc mô hình trên CPU")

        # Nạp vào GPU
        pipe.to(self.device)
        self.log_stage("0.2 Toàn Bộ Pipeline Đã Lên GPU VRAM", "Model Load", f"Text Encoder + Transformer3D + VAE trên {self.device}")

        # Tối ưu hóa VAE
        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
        self.log_stage("0.3 Kích Hoạt VAE Tiling & Slicing", "Model Load", "Bật Spatiotemporal Tiling & Slicing cho VAE")

        # Tính toán dung lượng VRAM thực tế từng cụm module
        te_mem = calc_module_vram(pipe.text_encoder) if hasattr(pipe, "text_encoder") else 0
        tr_mem = calc_module_vram(pipe.transformer) if hasattr(pipe, "transformer") else 0
        vae_mem = calc_module_vram(pipe.vae) if hasattr(pipe, "vae") else 0

        self.components_vram = {
            "text_encoder_umt5": {
                "name": "UMT5-XXL Text Encoder",
                "param_vram_mb": round(te_mem, 2),
                "param_vram_gb": round(te_mem / 1024, 2),
                "dtype": str(next(pipe.text_encoder.parameters()).dtype) if hasattr(pipe, "text_encoder") else "N/A",
            },
            "transformer_3d": {
                "name": "WanTransformer3D (30 DiT Blocks)",
                "param_vram_mb": round(tr_mem, 2),
                "param_vram_gb": round(tr_mem / 1024, 2),
                "dtype": str(next(pipe.transformer.parameters()).dtype) if hasattr(pipe, "transformer") else "N/A",
            },
            "vae_3d": {
                "name": "AutoencoderKLWan (Spatiotemporal 3D VAE)",
                "param_vram_mb": round(vae_mem, 2),
                "param_vram_gb": round(vae_mem / 1024, 2),
                "dtype": str(next(pipe.vae.parameters()).dtype) if hasattr(pipe, "vae") else "N/A",
            },
            "total_weights_mb": round(te_mem + tr_mem + vae_mem, 2),
            "total_weights_gb": round((te_mem + tr_mem + vae_mem) / 1024, 2),
        }

        print("\n🔍 CHI TIẾT TRỌNG SỐ TỪNG MODULE TRÊN VRAM:")
        print(f"   • Text Encoder (UMT5-XXL)  : {format_mb(te_mem)}")
        print(f"   • Transformer (Wan 30 DiT) : {format_mb(tr_mem)}")
        print(f"   • VAE Decoder (3D Causal)  : {format_mb(vae_mem)}")
        print(f"   • Tổng Trọng Số VRAM       : {format_mb(te_mem + tr_mem + vae_mem)}")

        # -------------------------------------------------------------
        # STAGE 1: TEXT ENCODING (UMT5-XXL)
        # -------------------------------------------------------------
        print("\n--- [GIAI ĐOẠN 1: ENCODER VĂN BẢN (UMT5-XXL)] ---")
        torch.cuda.reset_peak_memory_stats(self.device)
        self.log_stage("1.0 VRAM Trước Khi Mã Hóa Prompt", "Text Encoder", "Trạng thái sẵn sàng cho text encoding")

        t_text_start = time.time()
        # Chạy encode prompt thực tế
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            device=self.device,
            dtype=self.dtype,
        )
        t_text_elapsed = time.time() - t_text_start

        self.log_stage(
            "1.1 Hoàn Tất UMT5-XXL Forward Pass",
            "Text Encoder",
            f"Embeddings Shape: {list(prompt_embeds.shape)} trong {t_text_elapsed:.2f}s"
        )

        # -------------------------------------------------------------
        # STAGE 2: LATENT PREPARATION
        # -------------------------------------------------------------
        print("\n--- [GIAI ĐOẠN 2: KHỞI TẠO LATENTS KHÔNG-THỜI GIAN 3D] ---")
        torch.cuda.reset_peak_memory_stats(self.device)
        t_lat_start = time.time()

        generator = torch.Generator(device=self.device).manual_seed(42)
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        t_lat_elapsed = time.time() - t_lat_start

        self.log_stage(
            "2.0 Khởi Tạo 3D Gaussian Noise Latents",
            "Latent Prep",
            f"Latents Shape: {list(latents.shape)} ({latents.numel() * latents.element_size() / (1024*1024):.2f} MB)"
        )

        # -------------------------------------------------------------
        # STAGE 3: DIFFUSION DENOISING (BACKBONE TRANSFORMER 30 BLOCKS)
        # -------------------------------------------------------------
        print("\n--- [GIAI ĐOẠN 3: KHỬ NHIỄU DIT BACKBONE (30 TRANSFORMER BLOCKS)] ---")
        torch.cuda.reset_peak_memory_stats(self.device)
        self.log_stage("3.0 Bắt Đầu Denoising Loop", "Denoising", f"25 Steps Euler Flow Match, Guidance: {self.guidance_scale}")

        # Cài đặt forward hooks cho các cụm Block trong WanTransformer3D
        num_blocks = len(pipe.transformer.blocks)
        cluster_records = {
            "patch_embed": {"allocated_mb": 0.0, "peak_mb": 0.0},
            "cluster_1_early": {"allocated_mb": 0.0, "peak_mb": 0.0},   # Blocks 0 - 9
            "cluster_2_mid": {"allocated_mb": 0.0, "peak_mb": 0.0},     # Blocks 10 - 19
            "cluster_3_late": {"allocated_mb": 0.0, "peak_mb": 0.0},    # Blocks 20 - 29
            "proj_out_head": {"allocated_mb": 0.0, "peak_mb": 0.0},
        }

        def make_hook(key):
            def hook_fn(module, inp, out):
                torch.cuda.synchronize(self.device)
                m = get_vram_mb(self.device)
                cluster_records[key]["allocated_mb"] = max(cluster_records[key]["allocated_mb"], m["allocated"])
                cluster_records[key]["peak_mb"] = max(cluster_records[key]["peak_mb"], m["max_allocated"])
            return hook_fn

        # Đăng ký hooks
        h_patch = pipe.transformer.patch_embedding.register_forward_hook(make_hook("patch_embed"))
        h_c1 = pipe.transformer.blocks[9].register_forward_hook(make_hook("cluster_1_early"))
        h_c2 = pipe.transformer.blocks[19].register_forward_hook(make_hook("cluster_2_mid"))
        h_c3 = pipe.transformer.blocks[29].register_forward_hook(make_hook("cluster_3_late"))
        h_proj = pipe.transformer.proj_out.register_forward_hook(make_hook("proj_out_head"))

        step_records = {}

        def step_callback(pipeline, step_idx, timestep, callback_kwargs):
            torch.cuda.synchronize(self.device)
            m = get_vram_mb(self.device)
            if step_idx in (0, 1, 4, 9, 14, 19, self.num_steps - 1):
                step_records[f"step_{step_idx + 1}"] = {
                    "step": step_idx + 1,
                    "allocated_mb": round(m["allocated"], 2),
                    "peak_mb": round(m["max_allocated"], 2),
                    "reserved_mb": round(m["reserved"], 2),
                }
                print(f"   ↳ [Denoise Step {step_idx + 1:02d}/{self.num_steps:02d}] Alloc: {format_mb(m['allocated']):<18} | Peak: {format_mb(m['max_allocated']):<18}")
            return callback_kwargs

        t_denoise_start = time.time()
        with torch.inference_mode():
            pipe_out = pipe(
                prompt=prompt,
                negative_prompt="",
                width=self.width,
                height=self.height,
                num_frames=self.num_frames,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                callback_on_step_end=step_callback,
            )
        t_denoise_elapsed = time.time() - t_denoise_start

        # Gỡ bỏ hooks
        h_patch.remove()
        h_c1.remove()
        h_c2.remove()
        h_c3.remove()
        h_proj.remove()

        self.log_stage(
            "3.1 Hoàn Tất Denoising Loop",
            "Denoising",
            f"25 steps trong {t_denoise_elapsed:.2f}s ({self.num_steps / t_denoise_elapsed:.2f} it/s)"
        )

        self.block_cluster_stats = {
            "0. Patch Embedding & 3D RoPE": cluster_records["patch_embed"],
            "1. Cụm 1 (Blocks 0-9 Sơ Cấp)": cluster_records["cluster_1_early"],
            "2. Cụm 2 (Blocks 10-19 Cross-Attention)": cluster_records["cluster_2_mid"],
            "3. Cụm 3 (Blocks 20-29 Tinh Chỉnh Sâu)": cluster_records["cluster_3_late"],
            "4. Projection Out Head": cluster_records["proj_out_head"],
        }

        print("\n🔍 ĐỈNH VRAM THEO TỪNG CỤM BLOCK TRONG TRANSFORMER:")
        for c_name, c_stat in self.block_cluster_stats.items():
            print(f"   • {c_name:<38} : Peak {format_mb(c_stat['peak_mb']):<18} | Alloc {format_mb(c_stat['allocated_mb'])}")

        # -------------------------------------------------------------
        # STAGE 4: VAE DECODING
        # -------------------------------------------------------------
        print("\n--- [GIAI ĐOẠN 4: GIẢI MÃ VAE 3D (LATENTS -> VIDEO RGB)] ---")
        self.log_stage("4.0 Sau Khi Giải Mã VAE", "VAE Decode", f"Khung hình RGB: {self.num_frames} frames ({self.width}x{self.height})")

        # -------------------------------------------------------------
        # STAGE 5: POST-PROCESSING & MP4 EXPORT
        # -------------------------------------------------------------
        print("\n--- [GIAI ĐOẠN 5: XUẤT VIDEO FILE MP4 H.264] ---")
        torch.cuda.reset_peak_memory_stats(self.device)
        t_export_start = time.time()
        video_frames = pipe_out.frames[0]
        out_mp4_path = f"wan21_benchmark_{self.width}x{self.height}_{self.num_frames}f.mp4"

        export_to_video(video_frames, out_mp4_path, fps=self.fps)
        t_export_elapsed = time.time() - t_export_start
        mp4_size_kb = os.path.getsize(out_mp4_path) / 1024

        self.log_stage("5.0 Xuất File Video MP4 Hoàn Tất", "Export", f"File: {out_mp4_path} ({mp4_size_kb:.1f} KB) trong {t_export_elapsed:.2f}s")

        # 6. Post-generation Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        self.log_stage("6.0 Sau Khi empty_cache()", "Cleanup", "Giải phóng intermediate activation cache")

        total_time = time.time() - t_load_start

        benchmark_summary = {
            "hardware": self.hardware_info,
            "components_vram": self.components_vram,
            "config": {
                "model_id": self.model_id,
                "resolution": f"{self.width}x{self.height}",
                "num_frames": self.num_frames,
                "num_steps": self.num_steps,
                "guidance_scale": self.guidance_scale,
                "fps": self.fps,
                "dtype": str(self.dtype),
            },
            "performance": {
                "text_encode_sec": round(t_text_elapsed, 2),
                "denoise_sec": round(t_denoise_elapsed, 2),
                "steps_per_sec": round(self.num_steps / t_denoise_elapsed, 2),
                "frames_per_sec": round(self.num_frames / t_denoise_elapsed, 2),
                "export_sec": round(t_export_elapsed, 2),
                "total_sec": round(total_time, 2),
                "output_video_file": out_mp4_path,
                "output_video_size_kb": round(mp4_size_kb, 1),
            },
            "stages_vram": self.reports,
            "block_clusters_vram": self.block_cluster_stats,
            "denoising_steps_vram": step_records,
        }

        # Lưu JSON
        report_json_path = "wan21_vram_benchmark_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Đã lưu file báo cáo JSON chi tiết tại: {report_json_path}")

        self.print_markdown_report(benchmark_summary)
        return benchmark_summary

    def print_markdown_report(self, summary: Dict[str, Any]):
        hw = summary["hardware"]
        cfg = summary["config"]
        perf = summary["performance"]
        comp = summary["components_vram"]

        md = []
        md.append(f"# 📊 Báo Cáo Phân Tích VRAM Wan2.1 Video Pipeline")
        md.append(f"- **Thiết Bị:** `{hw.get('device_name', 'N/A')}` ({hw.get('total_vram_gb', 0)} GB VRAM)")
        md.append(f"- **Mô hình:** `{cfg['model_id']}` | Dtype: `{cfg['dtype']}`")
        md.append(f"- **Thông số Video:** `{cfg['resolution']}` | `{cfg['num_frames']} frames` @ `{cfg['fps']} fps` | `{cfg['num_steps']} steps` (Euler Flow Match)")
        md.append(f"- **Tốc độ:** `{perf['steps_per_sec']} it/s` (`{perf['frames_per_sec']} frames/s`) | Denoise: `{perf['denoise_sec']}s` | Tổng: `{perf['total_sec']}s`")
        md.append(f"- **File Xuất:** `{perf['output_video_file']}` ({perf['output_video_size_kb']} KB)")
        md.append("")
        md.append("## 1. Dung Lượng Trọng Số Từng Thành Phần (Model Weights)")
        md.append("| Thành Phần (Component) | Module Class | VRAM Trọng Số (MB) | VRAM (GB) | Kiểu Dữ Liệu |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        md.append(f"| **Text Encoder** | `{comp['text_encoder_umt5']['name']}` | `{comp['text_encoder_umt5']['param_vram_mb']:.1f} MB` | **`{comp['text_encoder_umt5']['param_vram_gb']:.2f} GB`** | `{comp['text_encoder_umt5']['dtype']}` |")
        md.append(f"| **Diffusion Backbone** | `{comp['transformer_3d']['name']}` | `{comp['transformer_3d']['param_vram_mb']:.1f} MB` | **`{comp['transformer_3d']['param_vram_gb']:.2f} GB`** | `{comp['transformer_3d']['dtype']}` |")
        md.append(f"| **VAE 3D Decoder** | `{comp['vae_3d']['name']}` | `{comp['vae_3d']['param_vram_mb']:.1f} MB` | **`{comp['vae_3d']['param_vram_gb']:.2f} GB`** | `{comp['vae_3d']['dtype']}` |")
        md.append(f"| **TỔNG CỘNG WEIGHTS** | _Toàn bộ tham số nạp GPU_ | `{comp['total_weights_mb']:.1f} MB` | **`{comp['total_weights_gb']:.2f} GB`** | - |")

        md.append("")
        md.append("## 2. Mức Tiêu Thụ VRAM Từng Giai Đoạn (Pipeline Stages)")
        md.append("| Giai Đoạn (Stage) | Phân Loại | Allocated VRAM | Peak Allocated VRAM | Reserved VRAM | Chi Tiết |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

        for r in summary["stages_vram"]:
            md.append(f"| **{r['stage']}** | `{r['category']}` | `{format_mb(r['allocated_mb'])}` | **`{format_mb(r['peak_allocated_mb'])}`** | `{format_mb(r['reserved_mb'])}` | {r['details']} |")

        md.append("")
        md.append("## 3. Tiêu Thụ VRAM Từng Cụm Block Chính (WanTransformer3D - 30 Blocks)")
        md.append("| Cụm Block DiT | Chức Năng Chính | Allocated VRAM | Peak VRAM | Ghi Chú |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for c_name, c_data in summary["block_clusters_vram"].items():
            alloc_str = format_mb(c_data["allocated_mb"]) if c_data["allocated_mb"] > 0 else "N/A"
            peak_str = format_mb(c_data["peak_mb"]) if c_data["peak_mb"] > 0 else "N/A"
            md.append(f"| **{c_name}** | Spatiotemporal DiT Processing | `{alloc_str}` | **`{peak_str}`** | Forward Pass Attention + FFN |")

        md.append("")
        md.append("## 4. Độ Ổn Định VRAM Qua Các Bước Khử Nhiễu (Denoising Steps)")
        md.append("| Bước (Step) | Allocated VRAM | Peak VRAM | Reserved VRAM |")
        md.append("| :--- | :--- | :--- | :--- |")
        for s_key, s_data in summary["denoising_steps_vram"].items():
            md.append(f"| Step {s_data['step']:02d} | `{format_mb(s_data['allocated_mb'])}` | **`{format_mb(s_data['peak_mb'])}`** | `{format_mb(s_data['reserved_mb'])}` |")

        md_content = "\n".join(md)
        print("\n" + "=" * 105)
        print(md_content)
        print("=" * 105)

        report_md_path = "wan21_vram_benchmark_report.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n✅ Đã xuất báo cáo Markdown tại: {report_md_path}")

if __name__ == "__main__":
    profiler = WanVRAMProfiler(
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        width=832,
        height=480,
        num_frames=17,
        num_steps=25,
        guidance_scale=5.0,
        fps=16,
    )
    profiler.run_benchmark()
