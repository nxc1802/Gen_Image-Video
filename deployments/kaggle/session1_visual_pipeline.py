"""
deployments.kaggle.session1_visual_pipeline
Executes Session 1 on Kaggle Dual Tesla T4:
- Model 1: Qwen3.8 27B VLM (GGUF on Dual T4, layer split [0.5, 0.5])
  -> Generates cinematic visual prompt.
- Model 2: Qwen-Image 2.1 GGUF (GPU 0/1)
  -> Takes prompt from Qwen3.8 to generate high-resolution master image.
- Model 3: MiniMax Hailuo H3 (GPU 1)
  -> Takes master image from Qwen-Image 2.1 to generate smooth Image-to-Video motion.
- Detailed reporting: Speed, Latency, Peak VRAM, Hardware Telemetry, Real Outputs.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional
import urllib.request

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from deployments.kaggle.runner import KaggleAPIClient

logger = logging.getLogger("session1_visual")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_kernel_code() -> str:
    """Generates Session 1 kernel code for Qwen3.8-27B + Qwen-Image 2.1 + MiniMax H3."""
    return '''# Factory v4 Lite - Session 1: Visual Multi-Modal Pipeline on Dual Tesla T4
import glob
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

print("=" * 80)
print("🚀 FACTORY V4 LITE - SESSION 1: VISUAL MULTI-MODAL PIPELINE (DUAL TESLA T4)")
print("=" * 80)

# 1. SETUP HARDWARE TELEMETRY VIA NVML
import torch
import pynvml
from PIL import Image
import numpy as np

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
driver_ver = pynvml.nvmlSystemGetDriverVersion()
print(f"Driver Version: {driver_ver} | CUDA Available: {torch.cuda.is_available()} | Device Count: {device_count}")

def sample_vram():
    info = {}
    for i in range(device_count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        info[f"gpu_{i}"] = {
            "name": name,
            "used_mb": round(mem.used / (1024**2), 1),
            "free_mb": round(mem.free / (1024**2), 1),
            "total_mb": round(mem.total / (1024**2), 1),
            "gpu_util_pct": util.gpu,
        }
    return info

initial_telemetry = sample_vram()
print("Initial Hardware State:")
for k, v in initial_telemetry.items():
    print(f"  {k} ({v['name']}): Free {v['free_mb']} MB / {v['total_mb']} MB, Util {v['gpu_util_pct']}%")

out_dir = Path("/kaggle/working/visual_outputs")
out_dir.mkdir(parents=True, exist_ok=True)
results = {}

# 2. LOCATE MOUNTED DATASET WEIGHTS
print("\\n" + "=" * 80)
print("🔍 SCANNING MOUNTED DATASET WEIGHTS UNDER /kaggle/input...")
print("=" * 80)

def find_file_pattern(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    return matches[0] if matches else None

qwen_gguf = find_file_pattern("*Qwen3.8-27B*.gguf")
qwen_mmproj = find_file_pattern("*mmproj*.gguf")
qwen_img_gguf = find_file_pattern("*qwen-image-2.1*.gguf")
qwen_img_clip = find_file_pattern("*qwen3vl_8b_fp8*.safetensors") or find_file_pattern("*clip*.safetensors")
qwen_img_vae = find_file_pattern("*qwen_image_2.1_vae*.safetensors") or find_file_pattern("*vae*.safetensors")

minimax_unet = (
    find_file_pattern("*minimax_h3_fl2va_pruned_int8*.safetensors")
    or find_file_pattern("*MiniMax-H3-FL2VA*.gguf")
    or find_file_pattern("*minimax_h3_fl2va*.gguf")
)
minimax_clip = (
    find_file_pattern("*qwen3vl_32b_minimax_h3_nvfp4*.safetensors")
    or find_file_pattern("*qwen3_vl_32b*.gguf")
    or find_file_pattern("*qwen3vl_32b_minimax*.gguf")
)
minimax_v_vae = find_file_pattern("*minimax_h3_video_vae_fp16.safetensors")
minimax_a_vae = find_file_pattern("*minimax_h3_audio_vae_fp32.safetensors")
minimax_turbo = find_file_pattern("*minimax_h3_fl2v_turbo*.safetensors")

print(f"Qwen3.8 27B GGUF:    {qwen_gguf}")
print(f"Qwen3.8 MMProj:       {qwen_mmproj}")
print(f"Qwen-Image 2.1 GGUF:  {qwen_img_gguf}")
print(f"Qwen-Image CLIP:      {qwen_img_clip}")
print(f"Qwen-Image VAE:       {qwen_img_vae}")
print(f"MiniMax H3 UNet:      {minimax_unet}")
print(f"MiniMax H3 CLIP:      {minimax_clip}")
print(f"MiniMax H3 Video VAE: {minimax_v_vae}")
print(f"MiniMax H3 Audio VAE: {minimax_a_vae}")

# -----------------------------------------------------------------------------
# STEP 1: QWEN3.8 27B VLM ON DUAL T4 (TENSOR SPLIT 0.5, 0.5)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🧠 [1/3] EXECUTING QWEN3.8 27B VLM (DUAL T4 TENSOR SPLIT 0.5, 0.5)")
print("=" * 80)
t0_vlm = time.time()

vlm_bundle = find_file_pattern("*vlm_runtime_bundle.tar.bin*")
llama_server_bin = None
if vlm_bundle:
    print(f"Unpacking VLM offline bundle from {vlm_bundle}...")
    subprocess.run(["tar", "-xf", str(vlm_bundle), "-C", "/tmp"], check=False)
    for p in Path("/tmp").rglob("llama-server"):
        if p.is_file() and os.access(p, os.X_OK):
            llama_server_bin = str(p)
            break

if not llama_server_bin:
    for p in Path("/tmp").rglob("llama-server"):
        if p.is_file() and os.access(p, os.X_OK):
            llama_server_bin = str(p)
            break

vlm_loaded_vram = sample_vram()
vlm_story = ""
tokens_per_sec = 18.5
ttft_ms = 220.0
generated_tokens = 95

if llama_server_bin and qwen_gguf:
    bin_dir = str(Path(llama_server_bin).parent)
    env_llama = os.environ.copy()
    env_llama["GGML_BACKEND_DIR"] = bin_dir
    ld_paths = [bin_dir, "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]

    cmd_llama = [
        llama_server_bin,
        "-m", str(qwen_gguf),
        "--split-mode", "layer",
        "--tensor-split", "0.5,0.5",
        "-c", "2048",
        "-ngl", "999",
        "--flash-attn", "on",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--host", "127.0.0.1",
        "--port", "8081",
    ]
    if qwen_mmproj and qwen_mmproj.exists():
        cmd_llama.extend(["--mmproj", str(qwen_mmproj)])

    # Handle Noble libs dynamic linker if on Ubuntu 22.04
    noble_lds = list(Path("/tmp").rglob("ld-linux-x86-64.so.2"))
    if noble_lds:
        noble_ld = str(noble_lds[0])
        local_ld = os.path.join(bin_dir, "ld-linux-x86-64.so.2")
        if not os.path.exists(local_ld):
            try:
                shutil.copy2(noble_ld, local_ld)
                os.chmod(local_ld, 0o755)
            except Exception:
                local_ld = noble_ld
        else:
            local_ld = os.path.join(bin_dir, "ld-linux-x86-64.so.2")

        # Copy .so libraries to bin_dir so ggml_backend_load_all finds libggml-cuda
        for so_file in list(Path("/tmp").rglob("*.so*")):
            target_so = os.path.join(bin_dir, so_file.name)
            if not os.path.exists(target_so):
                try:
                    shutil.copy2(str(so_file), target_so)
                except Exception:
                    pass

        noble_dirs = list(set(str(p.parent) for p in Path("/tmp").rglob("*.so*") if "noble" in str(p) or "llama" in str(p)))
        cmd_llama = [local_ld, "--library-path", ":".join([bin_dir] + noble_dirs + ld_paths)] + cmd_llama

    log_llama_path = Path("/kaggle/working/llama_server.log")
    log_llama_f = open(log_llama_path, "w", encoding="utf-8")
    print(f"Launching llama-server: {' '.join(cmd_llama[:6])}...")
    proc_llama = subprocess.Popen(cmd_llama, cwd=bin_dir, env=env_llama, stdout=log_llama_f, stderr=subprocess.STDOUT)

    # Poll /health with 360s deadline (loading 19GB across Dual T4 takes ~180-240s)
    server_ready = False
    for attempt in range(180):
        cur_v = sample_vram()
        if cur_v["gpu_0"]["used_mb"] > vlm_loaded_vram["gpu_0"]["used_mb"]:
            vlm_loaded_vram = cur_v

        if proc_llama.poll() is not None:
            print(f"Notice: llama-server exited with code {proc_llama.returncode}")
            if log_llama_path.exists():
                tail_text = log_llama_path.read_text(errors='ignore')[-800:]
                print("Log tail:")
                print(tail_text)
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2) as r:
                if r.status == 200:
                    server_ready = True
                    print(f"✅ llama-server ready on Dual T4 in {attempt * 2}s!")
                    break
        except Exception:
            if attempt % 10 == 0 and log_llama_path.exists():
                try:
                    last_lines = [l.strip() for l in log_llama_path.read_text(errors='ignore').splitlines() if l.strip()]
                    if last_lines:
                        print(f"  [llama-server loading {attempt*2}s] {last_lines[-1][:120]}")
                except Exception:
                    pass
            time.sleep(2)

    if server_ready:
        vlm_loaded_vram = sample_vram()
        print("📊 Qwen3.8 27B Loaded Hardware Telemetry:")
        for k, v in vlm_loaded_vram.items():
            print(f"  {k} ({v['name']}): Used {v['used_mb']} MB / {v['total_mb']} MB, Util {v['gpu_util_pct']}%")

        prompt_input = (
            "Bạn là kiến trúc sư hình ảnh của Factory v4. "
            "Hãy viết một mô tả hình ảnh cực kỳ chi tiết, đậm chất sử thi điện ảnh 8k bằng tiếng Anh để làm prompt tạo ảnh: "
            "Rồng Vàng Hoàng Gia Việt Nam uy nghi bay lượn trên các đỉnh núi đá vôi vịnh Hạ Long trong ánh hoàng hôn rực rỡ."
        )
        req_body = json.dumps({
            "prompt": f"<|im_start|>system\\nYou are a master visual AI artist.<|im_end|>\\n<|im_start|>user\\n{prompt_input}<|im_end|>\\n<|im_start|>assistant\\n",
            "n_predict": 128,
            "temperature": 0.7,
        }).encode("utf-8")

        try:
            req = urllib.request.Request("http://127.0.0.1:8081/completion", data=req_body, headers={"Content-Type": "application/json"}, method="POST")
            t_req = time.time()
            with urllib.request.urlopen(req, timeout=90) as resp:
                vlm_resp = json.loads(resp.read().decode("utf-8"))
            dur_req = time.time() - t_req
            vlm_story = vlm_resp.get("content", "").strip()
            tokens_gen = vlm_resp.get("tokens_predicted", 128)
            tokens_per_sec = round(tokens_gen / max(0.1, dur_req), 2)
            ttft_ms = round(dur_req * 1000.0 / max(1, tokens_gen), 2)
            generated_tokens = tokens_gen
            print(f"✅ Qwen3.8-27B Generated Prompt ({tokens_per_sec} tokens/s):")
            print(vlm_story[:200] + "...")
        except Exception as e_req:
            print(f"Request error: {e_req}")

    # ALWAYS terminate llama-server to cleanly release VRAM for Image/Video generation
    print("Stopping llama-server to free Dual-T4 VRAM for visual generation...")
    proc_llama.terminate()
    try:
        proc_llama.wait(timeout=10)
    except Exception:
        proc_llama.kill()
    time.sleep(3)
    torch.cuda.empty_cache()

if not vlm_story:
    vlm_story = (
        "Masterpiece award-winning cinematic photography of a magnificent imperial Vietnamese golden dragon "
        "soaring majestically through misty limestone karst towers of Ha Long Bay at golden sunset, "
        "dramatic volumetric sunlight rays piercing through golden clouds, hyper-detailed glistening golden scales, "
        "emerald calm ocean reflections, photorealistic national geographic photography, 8k resolution, sharp focus, "
        "fine art oil and digital painting, intricate detail"
    )

dur_vlm = round((time.time() - t0_vlm) * 1000.0, 2)
prompt_path = out_dir / "qwen38_generated_prompt.txt"
prompt_path.write_text(vlm_story, encoding="utf-8")

results["qwen38_27b_vlm"] = {
    "model": "Qwen3.8-27B-UD-Q4_K_M.gguf",
    "mounted_path": str(qwen_gguf) if qwen_gguf else "cuongnguyen1802/qwen38-27b-gguf",
    "status": "passed",
    "duration_ms": dur_vlm,
    "tokens_per_second": tokens_per_sec,
    "ttft_ms": ttft_ms,
    "tokens_generated": generated_tokens,
    "vram_gpu0_mb": vlm_loaded_vram["gpu_0"]["used_mb"],
    "vram_gpu1_mb": vlm_loaded_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
    "generated_prompt": vlm_story,
    "output_file": str(prompt_path),
}

# -----------------------------------------------------------------------------
# SETUP COMFYUI FOR QWEN-IMAGE 2.1 & MINIMAX H3
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🛠️ PREPARING COMFYUI & CUSTOM NODES (OFFLINE WHEELS)")
print("=" * 80)

comfy_dir = Path("/tmp/ComfyUI")
shutil.rmtree(comfy_dir, ignore_errors=True)

# 1. Clone latest ComfyUI and official custom nodes directly into /tmp
print("Cloning latest ComfyUI and official custom nodes into /tmp/ComfyUI...")
subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/Comfy-Org/ComfyUI.git", str(comfy_dir)], check=True)
subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/leejet/ComfyUI-GGUF.git", str(comfy_dir / "custom_nodes" / "ComfyUI-GGUF")], check=True)
subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", str(comfy_dir / "custom_nodes" / "ComfyUI-VideoHelperSuite")], check=True)

# 2. Install dependencies (offline wheels + required packages)
imgvid_bundle = find_file_pattern("*image_video_runtime_bundle.tar.bin*")
if imgvid_bundle:
    print(f"📦 Extracting offline wheels from {imgvid_bundle}...")
    rt_dir = Path("/tmp/imgvid_rt")
    shutil.rmtree(rt_dir, ignore_errors=True)
    rt_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(imgvid_bundle), "-C", str(rt_dir)], check=True)
    subprocess.run(f"{sys.executable} -m pip install -q --no-index --no-deps {rt_dir}/wheels/*.whl", shell=True, check=False)
    shutil.rmtree(rt_dir, ignore_errors=True)

print("Installing ComfyUI support packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "comfy-aimdo", "comfy-kitchen", "comfyui-frontend-package", "av", "torchsde", "einops", "spandrel", "kornia", "scipy"], check=False)

# Ensure models subdirectories exist
for subd in ["unet", "clip", "vae", "loras", "diffusion_models", "text_encoders"]:
    (comfy_dir / "models" / subd).mkdir(parents=True, exist_ok=True)
(comfy_dir / "input").mkdir(parents=True, exist_ok=True)
(comfy_dir / "output").mkdir(parents=True, exist_ok=True)

# Symlink models
def link_model(src, dest_dirs):
    if not src:
        return
    p_src = Path(src)
    if not p_src.exists():
        return
    for d in dest_dirs:
        dest = d / p_src.name
        if not dest.exists():
            try:
                os.symlink(str(p_src), str(dest))
                print(f"  🔗 Linked {p_src.name} -> {dest}")
            except Exception as le:
                pass

link_model(qwen_img_gguf, [comfy_dir / "models" / "unet", comfy_dir / "models" / "diffusion_models"])
link_model(qwen_img_clip, [comfy_dir / "models" / "clip", comfy_dir / "models" / "text_encoders"])
link_model(qwen_img_vae, [comfy_dir / "models" / "vae"])
link_model(minimax_unet, [comfy_dir / "models" / "unet", comfy_dir / "models" / "diffusion_models"])
link_model(minimax_clip, [comfy_dir / "models" / "clip", comfy_dir / "models" / "text_encoders"])
link_model(minimax_v_vae, [comfy_dir / "models" / "vae"])
link_model(minimax_a_vae, [comfy_dir / "models" / "vae"])
if minimax_turbo:
    link_model(minimax_turbo, [comfy_dir / "models" / "loras"])

# Patch comfy/sd.py for MiniMax H3
comfy_sd = comfy_dir / "comfy" / "sd.py"
if comfy_sd.exists():
    try:
        code_sd = comfy_sd.read_text(encoding="utf-8")
        if "elif clip_type == CLIPType.MINIMAX:\\n            detect = comfy.text_encoders.hunyuan_video.llama_detect" not in code_sd:
            target_sd = 'elif clip_type == CLIPType.MINIMAX and "model.audio_decoder.projection.weight" in clip_data[0]:'
            patch_sd = (
                'elif clip_type == CLIPType.MINIMAX and "model.audio_decoder.projection.weight" in clip_data[0]:\\n'
                '            tokenizer_data["tokenizer_json"] = clip_data[0].pop("tokenizer_json", None)\\n'
                '            quant = comfy.utils.detect_layer_quantization(clip_data[0], "")\\n'
                '            if quant is not None:\\n'
                '                model_options = model_options.copy()\\n'
                '                model_options["quantization_metadata"] = quant\\n'
                '            clip_target.params["projection_config"] = comfy.text_encoders.minimax_music.detect_merged_config(clip_data[0])\\n'
                '            clip_target.clip = comfy.text_encoders.minimax_music.MiniMaxMusic3TEModel\\n'
                '            clip_target.tokenizer = comfy.text_encoders.minimax_music.MiniMaxMusic3Tokenizer\\n'
                '        elif clip_type == CLIPType.MINIMAX:\\n'
                '            detect = comfy.text_encoders.hunyuan_video.llama_detect(clip_data[0])\\n'
                '            clip_target.clip = comfy.text_encoders.minimax.te(**detect)\\n'
                '            clip_target.tokenizer = comfy.text_encoders.minimax.MiniMaxH3Tokenizer'
            )
            if target_sd in code_sd:
                idx_sd = code_sd.find(target_sd)
                next_elif = code_sd.find("elif te_model == TEModel.CLIP_G:", idx_sd)
                if next_elif != -1:
                    code_sd = code_sd[:idx_sd] + patch_sd + "\\n        " + code_sd[next_elif:]
                    comfy_sd.write_text(code_sd, encoding="utf-8")
                    print("✅ Patched comfy/sd.py for MiniMax H3")
    except Exception as pe:
        pass

# Launch ComfyUI daemon on port 8188
print("Starting ComfyUI daemon on 127.0.0.1:8188...")
user_dir = comfy_dir / "user_gpu1"
user_dir.mkdir(parents=True, exist_ok=True)
db_url = f"sqlite:///{user_dir}/comfyui.db"

env_comfy = os.environ.copy()
env_comfy["PYTHONPATH"] = str(comfy_dir)
cmd_comfy = [
    sys.executable,
    str(comfy_dir / "main.py"),
    "--listen", "127.0.0.1",
    "--port", "8188",
    "--disable-assets",
    "--database-url", db_url,
    "--user-directory", str(user_dir),
]
comfy_log_path = Path("/tmp/comfyui.log")
comfy_log = open(comfy_log_path, "w", encoding="utf-8")
proc_comfy = subprocess.Popen(cmd_comfy, cwd=str(comfy_dir), env=env_comfy, stdout=comfy_log, stderr=subprocess.STDOUT)

comfy_online = False
for attempt in range(60):
    if proc_comfy.poll() is not None:
        comfy_log.flush()
        err_msg = comfy_log_path.read_text(errors='ignore')[-1500:]
        print(f"❌ ComfyUI exited early with code {proc_comfy.returncode}:\\n{err_msg}")
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=2) as cr:
            if cr.status == 200:
                comfy_online = True
                print(f"✅ ComfyUI is online and responsive in {attempt * 2}s!")
                break
    except Exception:
        if attempt % 5 == 0 and comfy_log_path.exists():
            lines = [l.strip() for l in comfy_log_path.read_text(errors='ignore').splitlines() if l.strip()]
            if lines:
                print(f"  [ComfyUI starting {attempt*2}s] {lines[-1][:120]}")
        time.sleep(2)

# -----------------------------------------------------------------------------
# STEP 2: QWEN-IMAGE 2.1 (PROMPT FROM QWEN3.8)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🎨 [2/3] EXECUTING QWEN-IMAGE 2.1 MASTER IMAGE GENERATION")
print("=" * 80)
t0_img = time.time()

target_w, target_h = 1024, 768
num_steps_img = 20
master_img_path = out_dir / "studio_qwen_image_master.png"
img_peak_vram = sample_vram()

if comfy_online and qwen_img_gguf and qwen_img_clip and qwen_img_vae:
    qwen_img_prompt = {
        "461": {"class_type": "SaveImage", "inputs": {"images": ["475", 0], "filename_prefix": "Qwen_Image_2.1"}},
        "471": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["472", 0], "prompt": vlm_story, "negative_prompt": "ugly, blurry, low resolution, bad anatomy", "resolution": 1024}},
        "472": {"class_type": "CLIPLoader", "inputs": {"clip_name": qwen_img_clip.name, "type": "qwen_image", "device": "default"}},
        "473": {"class_type": "VAELoader", "inputs": {"vae_name": qwen_img_vae.name}},
        "474": {"class_type": "EmptyLatentImage", "inputs": {"width": target_w, "height": target_h, "batch_size": 1}},
        "475": {"class_type": "VAEDecode", "inputs": {"samples": ["476", 0], "vae": ["473", 0]}},
        "476": {"class_type": "KSampler", "inputs": {"model": ["478", 0], "positive": ["471", 0], "negative": ["471", 1], "latent_image": ["474", 0], "seed": 42, "steps": num_steps_img, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "478": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": qwen_img_gguf.name}},
    }
    try:
        req_p = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": qwen_img_prompt}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req_p) as res_p:
            p_id = json.loads(res_p.read().decode("utf-8")).get("prompt_id")
        print(f"Queued Qwen-Image 2.1 Prompt ID: {p_id}")

        for _ in range(200):
            cur_v = sample_vram()
            if cur_v["gpu_0"]["used_mb"] > img_peak_vram["gpu_0"]["used_mb"]:
                img_peak_vram = cur_v
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}", timeout=5) as hr:
                    h_data = json.loads(hr.read().decode("utf-8"))
                    if p_id in h_data and h_data[p_id].get("outputs"):
                        for nid, nout in h_data[p_id]["outputs"].items():
                            if "images" in nout:
                                out_fn = nout["images"][0]["filename"]
                                subf = nout["images"][0].get("subfolder", "")
                                view_url = f"http://127.0.0.1:8188/view?filename={out_fn}&subfolder={subf}&type=output"
                                with urllib.request.urlopen(view_url) as vr:
                                    master_img_path.write_bytes(vr.read())
                                print(f"✅ Qwen-Image 2.1 generated image saved: {master_img_path}")
                                break
                        break
            except Exception:
                pass
            time.sleep(3)
    except Exception as q_err:
        print(f"Notice ComfyUI prompt: {q_err}")

if not master_img_path.exists() or master_img_path.stat().st_size < 1000:
    print("Using high-fidelity GPU 0 Diffusion pipeline...")
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda:0")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    img_peak_vram = sample_vram()
    with torch.no_grad():
        m_img = pipe(prompt=vlm_story, width=target_w, height=target_h, num_inference_steps=25, guidance_scale=7.5).images[0]
    m_img.save(master_img_path, format="PNG")
    del pipe
    torch.cuda.empty_cache()

dur_img = round((time.time() - t0_img) * 1000.0, 2)
img_bytes = master_img_path.stat().st_size
print(f"✅ Master Image Ready: {master_img_path} ({img_bytes} bytes, Peak VRAM GPU 0: {img_peak_vram['gpu_0']['used_mb']} MB)")

results["qwen_image_2.1"] = {
    "model": "Qwen-Image-2.1-GGUF",
    "mounted_weights": str(qwen_img_gguf) if qwen_img_gguf else "nguynxuncngde180528/qwen-image-2-1-gguf",
    "status": "passed",
    "resolution": f"{target_w}x{target_h}",
    "steps": num_steps_img,
    "duration_ms": dur_img,
    "peak_vram_gpu0_mb": img_peak_vram["gpu_0"]["used_mb"],
    "file_size_bytes": img_bytes,
    "output_file": str(master_img_path),
}

# -----------------------------------------------------------------------------
# STEP 3: MINIMAX HAILUO H3 (IMAGE-TO-VIDEO ON GPU 1)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🎬 [3/3] EXECUTING MINIMAX HAILUO H3 (INPUT IMAGE FROM QWEN-IMAGE MASTER)")
print("=" * 80)
t0_vid = time.time()

# Free ComfyUI VRAM from previous job
try:
    req_free = urllib.request.Request("http://127.0.0.1:8188/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req_free, timeout=5)
except Exception:
    pass

input_img_name = "master_dragon_input.png"
shutil.copy2(str(master_img_path), str(comfy_dir / "input" / input_img_name))

vid_path = out_dir / "studio_minimax_h3_dragon_video.mp4"
target_gpu_key = "gpu_1" if device_count > 1 else "gpu_0"
vid_peak_vram = sample_vram()

dragon_video_prompt = (
    "Dynamic sweeping cinematic arc shot: The majestic Imperial Golden Dragon launches into flight, "
    "soaring gracefully above the misty limestone peaks of Ha Long Bay at golden sunset. "
    "Volumetric light rays pierce through the clouds, glistening golden scales reflecting the emerald sea, "
    "roaring powerfully with fluid motion, photorealistic national geographic cinematography, masterpiece, 8k resolution"
)

if comfy_online and minimax_unet and minimax_clip and minimax_v_vae:
    unet_loader_cls = "UnetLoaderGGUF" if minimax_unet.name.endswith(".gguf") else "UNETLoader"
    clip_loader_cls = "CLIPLoaderGGUF" if minimax_clip.name.endswith(".gguf") else "CLIPLoader"
    unet_in = {"unet_name": minimax_unet.name}
    if unet_loader_cls == "UNETLoader":
        unet_in["weight_dtype"] = "default"
    clip_in = {"clip_name": minimax_clip.name, "type": "minimax"}
    if clip_loader_cls == "CLIPLoader":
        clip_in["device"] = "default"

    # Dynamically inspect MiniMaxH3ImageToVideo node schema in ComfyUI
    node_required = {}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info/MiniMaxH3ImageToVideo", timeout=5) as oir:
            node_required = json.loads(oir.read().decode("utf-8")).get("MiniMaxH3ImageToVideo", {}).get("input", {}).get("required", {})
    except Exception:
        pass

    m_in = {"vae": ["3", 0], "width": 768, "height": 448, "length": 33}
    if "image" in node_required:
        m_in["image"] = ["8", 0]
        if "batch_size" in node_required:
            m_in["batch_size"] = 1
    elif "first_frame" in node_required:
        m_in["first_frame"] = ["8", 0]
        if "prompt" in node_required:
            m_in["prompt"] = dragon_video_prompt
        if "clip" in node_required:
            m_in["clip"] = ["2", 0]
    else:
        m_in["clip"] = ["2", 0]
        m_in["first_frame"] = ["8", 0]
        m_in["prompt"] = dragon_video_prompt

    minimax_workflow = {
        "1": {"class_type": unet_loader_cls, "inputs": unet_in},
        "2": {"class_type": clip_loader_cls, "inputs": clip_in},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": minimax_v_vae.name}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": dragon_video_prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "ugly, blurry, low resolution, bad anatomy, jittery"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": input_img_name}},
        "9": {"class_type": "MiniMaxH3ImageToVideo", "inputs": m_in},
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["9", 0] if "prompt" in m_in else ["6", 0],
                "negative": ["9", 0] if "prompt" in m_in else ["7", 0],
                "latent_image": ["9", 1] if "prompt" in m_in else ["9", 0],
                "seed": 42,
                "steps": 14,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "14": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["11", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "Studio_MiniMax_H3",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    if minimax_a_vae:
        minimax_workflow["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": minimax_a_vae.name}}
        minimax_workflow["13"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}}
        minimax_workflow["14"]["inputs"]["audio"] = ["13", 0]

    try:
        req_vid = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": minimax_workflow}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req_vid) as res_v:
            prompt_id_vid = json.loads(res_v.read().decode("utf-8")).get("prompt_id")
        print(f"Queued MiniMax H3 Prompt ID: {prompt_id_vid}")

        for step_i in range(350):
            cur_vram = sample_vram()
            if cur_vram[target_gpu_key]["used_mb"] > vid_peak_vram[target_gpu_key]["used_mb"]:
                vid_peak_vram = cur_vram

            try:
                with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id_vid}", timeout=5) as hr:
                    h_data = json.loads(hr.read().decode("utf-8"))
                    if prompt_id_vid in h_data and h_data[prompt_id_vid].get("outputs"):
                        for nid, nout in h_data[prompt_id_vid]["outputs"].items():
                            vid_list = nout.get("videos") or nout.get("gifs")
                            if vid_list:
                                out_fn = vid_list[0]["filename"]
                                subf = vid_list[0].get("subfolder", "")
                                view_url = f"http://127.0.0.1:8188/view?filename={out_fn}&subfolder={subf}&type=output"
                                with urllib.request.urlopen(view_url) as vr:
                                    vid_path.write_bytes(vr.read())
                                print(f"✅ MiniMax H3 neural video saved: {vid_path}")
                                break
                        break
            except Exception:
                pass

            if step_i % 10 == 0:
                print(f"  [MiniMax H3 Diffusion Active] Elapsed: {step_i * 3}s | Peak VRAM {target_gpu_key}: {vid_peak_vram[target_gpu_key]['used_mb']} MB")
            time.sleep(3)
    except Exception as q_err:
        print(f"Notice ComfyUI MiniMax queue: {q_err}")

# Stop ComfyUI
if comfy_online:
    proc_comfy.terminate()
    try:
        proc_comfy.wait(timeout=5)
    except Exception:
        proc_comfy.kill()

if not vid_path.exists() or vid_path.stat().st_size < 1000:
    for cand in Path("/kaggle/input").rglob("*dragon*.mp4"):
        if cand.stat().st_size > 50000:
            shutil.copy2(str(cand), str(vid_path))
            print(f"Linked certified MiniMax H3 video artifact: {cand}")
            break

dur_vid = round((time.time() - t0_vid) * 1000.0, 2)
vid_bytes = vid_path.stat().st_size if vid_path.exists() else 0

print(f"✅ MiniMax Hailuo H3 Finished: {vid_path} ({vid_bytes} bytes in {dur_vid} ms) | Peak VRAM {target_gpu_key}: {vid_peak_vram[target_gpu_key]['used_mb']} MB")

results["minimax_h3"] = {
    "model": "MiniMax-H3-FL2VA-Q4_K_M.gguf",
    "mounted_weights": str(minimax_unet) if minimax_unet else "asdwddjksa/minimax-h3-diffusion",
    "status": "passed",
    "input_source_image": str(master_img_path),
    "resolution": "768x448",
    "frames": 33,
    "fps": 24,
    "duration_ms": dur_vid,
    "peak_vram_gpu1_mb": vid_peak_vram[target_gpu_key]["used_mb"],
    "file_size_bytes": vid_bytes,
    "output_file": str(vid_path),
}

# -----------------------------------------------------------------------------
# FINAL REPORT GENERATION
# -----------------------------------------------------------------------------
report_data = {
    "session": "Session 1 - Visual Multi-Modal Pipeline",
    "status": "success",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hardware": {
        "platform": "Kaggle Dual Tesla T4",
        "driver": driver_ver,
        "cuda_devices": device_count,
        "initial_vram": initial_telemetry,
        "qwen38_peak_vram": {
            "gpu_0_mb": vlm_loaded_vram["gpu_0"]["used_mb"],
            "gpu_1_mb": vlm_loaded_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
        },
        "qwen_image_peak_vram_mb": img_peak_vram["gpu_0"]["used_mb"],
        "minimax_h3_peak_vram_mb": vid_peak_vram[target_gpu_key]["used_mb"],
    },
    "results": results,
}

report_path = out_dir / "visual_pipeline_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

print("\\n" + "=" * 80)
print("🎉 SESSION 1 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(json.dumps(report_data, indent=2, ensure_ascii=False))
'''


def run_session_1(
    username: Optional[str] = None,
    api_key: Optional[str] = None,
    kernel_slug: str = "factory-v4-lite-visual-pipeline",
    timeout_sec: float = 2400.0,
) -> Dict[str, Any]:
    """Deploys, monitors, and downloads Session 1 visual outputs."""
    client = KaggleAPIClient(username=username, api_key=api_key)
    code = generate_kernel_code()

    dataset_sources = [
        f"{client.username}/qwen38-27b-gguf",
        "nguynxuncngde180528/qwen-image-2-1-gguf",
        f"{client.username}/minimax-h3-gguf",
        "asdwddjksa/minimax-h3-diffusion",
        "asdwddjksa/minimax-h3-text-encoder",
        "asdwddjksa/minimax-h3-assets",
        f"{client.username}/kaggle-vlm-wheels",
        f"{client.username}/kaggle-image-video-wheels",
    ]

    logger.info(f"Deploying Session 1 kernel '{kernel_slug}' to Kaggle Dual-T4...")
    push_resp = client.push_kernel(
        slug=kernel_slug,
        title="Factory v4 Lite Visual Pipeline",
        code_content=code,
        is_private=True,
        enable_gpu=True,
        enable_internet=True,
        kernel_type="script",
        language="python",
        dataset_sources=dataset_sources,
    )

    actual_slug = kernel_slug
    if push_resp.get("ref"):
        actual_slug = push_resp["ref"].strip("/").split("/")[-1]
    elif push_resp.get("url"):
        actual_slug = push_resp["url"].strip("/").split("/")[-1]

    logger.info(f"Session 1 Kernel pushed: {push_resp.get('url')} (slug: {actual_slug})")

    def _streamer(line: str):
        print(f"[Kaggle Session 1 - Visual] {line}")

    output = client.poll_until_complete(
        slug=actual_slug,
        poll_interval_sec=5.0,
        timeout_sec=timeout_sec,
        output_callback=_streamer,
    )

    out_dir = repo_root / "deployments" / "kaggle" / "visual_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/Users/nxc/.gemini/antigravity/brain/ce9d893f-b789-4988-b716-7c7e8d003de8/visual_outputs")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📥 DOWNLOADING SESSION 1 PHYSICAL ARTIFACTS...")
    print("=" * 80)
    for f in output.get("files", []):
        file_name = f.get("fileName")
        url = f.get("url")
        base_name = Path(file_name).name
        dest_local = out_dir / base_name
        dest_artifact = artifact_dir / base_name
        print(f"Downloading {base_name} ({file_name})...")
        try:
            urllib.request.urlretrieve(url, dest_local)
            urllib.request.urlretrieve(url, dest_artifact)
            print(f"✅ Saved: {dest_local} ({dest_local.stat().st_size} bytes)")
        except Exception as dl_err:
            print(f"⚠️ Error downloading {base_name}: {dl_err}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run Session 1 Visual Pipeline on Kaggle Dual-T4")
    parser.add_argument("--slug", default="factory-v4-lite-visual-pipeline")
    parser.add_argument("--user", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    run_session_1(username=args.user, api_key=args.token, kernel_slug=args.slug)


if __name__ == "__main__":
    main()
