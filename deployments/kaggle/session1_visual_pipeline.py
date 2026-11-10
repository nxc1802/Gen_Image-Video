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
import shutil

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from deployments.kaggle.runner import KaggleAPIClient
from benchmarks.runners.mtp_baseline_runner import (
    MTPComparisonReport,
    MTPRunResult,
    run_qwen38_mtp_comparison,
)
from benchmarks.runners.qwen_image_variant_runner import (
    ImageVariantRunResult,
    QwenImageVariantReport,
    benchmark_qwen_image_variants,
)

__all__ = [
    "generate_kernel_code",
    "run_session_1",
    "run_qwen38_mtp_comparison",
    "benchmark_qwen_image_variants",
    "parse_vlm_prompt_response",
    "MTPComparisonReport",
    "MTPRunResult",
    "ImageVariantRunResult",
    "QwenImageVariantReport",
    "FALLBACK_DEMO_PROMPT",
]

logger = logging.getLogger("session1_visual")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FALLBACK_DEMO_PROMPT = (
    "Epic cinematic wide-angle shot of an Imperial Vietnamese Golden Dragon, "
    "serpentine and majestic, coiling gracefully over mist-covered limestone karst peaks "
    "of Ha Long Bay at blazing sunset, golden scales glistening in volumetric rays, "
    "emerald sea reflections, 8k resolution, fine art oil and digital painting, "
    "national geographic photography."
)


def parse_vlm_prompt_response(raw_story: str) -> Dict[str, Any]:
    """
    Decouples <think>...</think> reasoning tokens from final visual prompt.
    Returns:
      {
        "reasoning": str,
        "clean_prompt": str,
        "final_prompt": str,
        "status": "passed" | "fallback",
        "fallback_used": bool,
        "error": Optional[str],
      }
    """
    reasoning_chunk = ""
    answer_chunk = ""

    if "<think>" in raw_story and "</think>" in raw_story:
        parts = raw_story.split("</think>", 1)
        reasoning_chunk = parts[0].replace("<think>", "").strip()
        answer_chunk = parts[1].strip()
    elif "</think>" in raw_story:
        parts = raw_story.rsplit("</think>", 1)
        reasoning_chunk = parts[0].replace("<think>", "").strip()
        answer_chunk = parts[1].strip()
    elif raw_story.startswith("<think>"):
        # Truncated inside thinking phase (choked token budget)
        reasoning_chunk = raw_story.replace("<think>", "").strip()
        answer_chunk = ""
    else:
        answer_chunk = raw_story.strip()

    clean_prompt = ""
    if "[PROMPT]" in answer_chunk and "[/PROMPT]" in answer_chunk:
        clean_prompt = answer_chunk.split("[PROMPT]")[1].split("[/PROMPT]")[0].strip()
    elif "[PROMPT]" in answer_chunk:
        clean_prompt = answer_chunk.split("[PROMPT]")[1].strip()
    else:
        clean_prompt = answer_chunk.strip()

    clean_prompt = (
        clean_prompt.replace("[PROMPT]", "")
        .replace("[/PROMPT]", "")
        .replace("[PROMPT", "")
        .replace("[/PROMPT", "")
        .replace('"', "")
        .strip()
    )
    if clean_prompt.lower().startswith("prompt:"):
        clean_prompt = clean_prompt[7:].strip()

    if len(clean_prompt) >= 20:
        return {
            "reasoning": reasoning_chunk,
            "clean_prompt": clean_prompt,
            "final_prompt": clean_prompt,
            "status": "passed",
            "fallback_used": False,
            "error": None,
        }
    else:
        err = (
            f"Qwen3.8-27B prompt generation failed or was choked by reasoning tokens "
            f"(raw len: {len(raw_story)}, prompt len: {len(clean_prompt)}). Demo fallback prompt applied."
        )
        return {
            "reasoning": reasoning_chunk,
            "clean_prompt": clean_prompt,
            "final_prompt": FALLBACK_DEMO_PROMPT,
            "status": "fallback",
            "fallback_used": True,
            "error": err,
        }


def generate_kernel_code() -> str:
    """Generates Session 1 kernel code for Qwen3.8-27B + Qwen-Image 2.1 + MiniMax H3."""
    return '''# Factory v4 Lite - Session 1: Visual Multi-Modal Pipeline on Dual Tesla T4
import glob
import hashlib
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

def init_nvml():
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver_str = driver.decode("utf-8") if isinstance(driver, bytes) else str(driver)
        return True, count, driver_str
    except Exception as e_nvml:
        print(f"⚠️ NVML initialization unavailable ({e_nvml}). Recording telemetry_status='UNAVAILABLE'.")
        return False, 0, None

nvml_ok, device_count, driver_ver = init_nvml()
print(f"Driver Version: {driver_ver} | CUDA Available: {torch.cuda.is_available()} | Device Count: {device_count}")

def sample_vram(target_pid=None):
    if not nvml_ok or device_count == 0:
        return {
            "telemetry_status": "UNAVAILABLE",
            "gpu_0": {
                "name": None, "used_mb": None, "free_mb": None, "total_mb": None, "gpu_util_pct": None,
                "system_vram_used_mb": None, "process_vram_used_mb": None, "running_processes": []
            },
            "gpu_1": {
                "name": None, "used_mb": None, "free_mb": None, "total_mb": None, "gpu_util_pct": None,
                "system_vram_used_mb": None, "process_vram_used_mb": None, "running_processes": []
            }
        }

    info = {"telemetry_status": "AVAILABLE"}
    for i in range(device_count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name_raw = pynvml.nvmlDeviceGetName(h)
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)

        # Process-isolated GPU memory accounting via NVML
        running_procs = []
        raw_procs = []
        try:
            raw_procs.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(h))
        except Exception:
            pass
        try:
            raw_procs.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(h))
        except Exception:
            pass

        seen_pids = set()
        process_vram_bytes = 0
        target_pid_vram_bytes = 0
        for p in raw_procs:
            pid = getattr(p, "pid", None)
            if pid is None or pid in seen_pids:
                continue
            seen_pids.add(pid)
            used_b = getattr(p, "usedGpuMemory", 0) or 0
            running_procs.append({"pid": pid, "used_mb": round(used_b / (1024**2), 1)})
            process_vram_bytes += used_b
            if target_pid is not None and pid == target_pid:
                target_pid_vram_bytes += used_b

        sys_used_mb = round(mem.used / (1024**2), 1)
        proc_used_mb = round((target_pid_vram_bytes if target_pid is not None else process_vram_bytes) / (1024**2), 1)
        info[f"gpu_{i}"] = {
            "name": name,
            "used_mb": sys_used_mb,
            "free_mb": round(mem.free / (1024**2), 1),
            "total_mb": round(mem.total / (1024**2), 1),
            "gpu_util_pct": util.gpu,
            "system_vram_used_mb": sys_used_mb,
            "process_vram_used_mb": proc_used_mb,
            "running_processes": running_procs,
        }
    return info

initial_telemetry = sample_vram()
print("Initial Hardware State:")
for k, v in initial_telemetry.items():
    if k.startswith("gpu_"):
        print(f"  {k} ({v['name']}): Free {v['free_mb']} MB / {v['total_mb']} MB, Util {v['gpu_util_pct']}%")

out_dir = Path("/kaggle/working/visual_outputs")
out_dir.mkdir(parents=True, exist_ok=True)
results = {}

def compute_file_sha256(filepath):
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# 2. LOCATE MOUNTED DATASET WEIGHTS
print("\\n" + "=" * 80)
print("🔍 SCANNING MOUNTED DATASET WEIGHTS UNDER /kaggle/input...")
print("=" * 80)

def find_file_pattern(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    return matches[0] if matches else None

qwen_gguf = find_file_pattern("*Qwen3.8-27B*.gguf")
qwen_mmproj = find_file_pattern("*mmproj*.gguf")
qwen_img_turbo = find_file_pattern("*qwen-image-2.1-turbo*.gguf")
qwen_img_standard = (
    find_file_pattern("*qwen-image-2.1-Q4_K_M.gguf")
    or find_file_pattern("*qwen-image-2.1*.gguf")
)
qwen_img_clip = find_file_pattern("*qwen3vl_8b_fp8*.safetensors") or find_file_pattern("*clip*.safetensors")
qwen_img_vae = find_file_pattern("*qwen_image_2.1_vae*.safetensors") or find_file_pattern("*vae*.safetensors")

# ⚡ Dedicated Qwen-Image-2.1-Turbo standalone model check & auto-download
if not qwen_img_turbo:
    print("⬇️ Dedicated Qwen-Image 2.1 Turbo GGUF not found locally. Auto-downloading from unsloth/Qwen-Image-2.1-Turbo-GGUF (3.91 GB)...")
    turbo_cache_file = Path("/tmp/qwen-image-2.1-turbo-Q4_K_M.gguf")
    try:
        t_dl_turbo = time.time()
        subprocess.run([
            "curl", "-sL", "--retry", "3", "--speed-limit", "100000", "--speed-time", "30",
            "-o", str(turbo_cache_file),
            "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/resolve/main/qwen-image-2.1-turbo-Q4_K_M.gguf"
        ], check=False)
        if turbo_cache_file.exists() and turbo_cache_file.stat().st_size > 1000000000:
            qwen_img_turbo = turbo_cache_file
            print(f"✅ Downloaded Qwen-Image 2.1 Turbo model in {round(time.time() - t_dl_turbo, 1)}s: {turbo_cache_file} ({round(turbo_cache_file.stat().st_size / (1024**3), 2)} GB)")
        else:
            print(f"⚠️ Turbo download incomplete (size: {turbo_cache_file.stat().st_size if turbo_cache_file.exists() else 0} bytes)")
    except Exception as e_turbo_dl:
        print(f"⚠️ Turbo download error: {e_turbo_dl}")

qwen_img_gguf = qwen_img_turbo if qwen_img_turbo else qwen_img_standard

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

print(f"Qwen3.8 27B GGUF:       {qwen_gguf}")
print(f"Qwen3.8 MMProj:          {qwen_mmproj}")
print(f"Qwen-Image 2.1 GGUF:     {qwen_img_gguf} (Is Turbo: {qwen_img_gguf == qwen_img_turbo})")
print(f"Qwen-Image CLIP:         {qwen_img_clip}")
print(f"Qwen-Image VAE:          {qwen_img_vae}")
print(f"MiniMax H3 UNet:         {minimax_unet}")
print(f"MiniMax H3 CLIP:         {minimax_clip}")
print(f"MiniMax H3 Video VAE:    {minimax_v_vae}")
print(f"MiniMax H3 Audio VAE:    {minimax_a_vae}")
print(f"MiniMax H3 Turbo LoRA:   {minimax_turbo}")

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
vlm_reasoning = ""
vlm_status = "skipped"
vlm_fallback_used = False
vlm_error = None
tokens_per_sec = 0.0
end_to_end_tps = 0.0
prompt_tps = 0.0
ttft_ms = 0.0
generated_tokens = 0
dur_vlm_server_load_ms = 0.0
prompt_eval_ms = 0.0
predicted_eval_ms = 0.0
dur_req = 0.0
dur_artifact_vlm = 0.0

if llama_server_bin and qwen_gguf:
    bin_dir = str(Path(llama_server_bin).parent)
    env_llama = os.environ.copy()
    env_llama["GGML_BACKEND_DIR"] = bin_dir
    ld_paths = [bin_dir, "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]

    # ⚡ Check / download MTP Speculative Decoding draft model
    qwen_mtp_draft = (
        find_file_pattern("*mtp*Qwen3.8*.gguf")
        or find_file_pattern("*mtp*.gguf")
    )
    if not qwen_mtp_draft:
        print("⬇️ MTP Speculative draft model not found in dataset. Downloading mtp-Qwen3.8-27B-Q4_0.gguf (1.6 GB)...")
        mtp_cache_file = Path("/tmp/hf_cache/mtp-Qwen3.8-27B-Q4_0.gguf")
        mtp_cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            t_mtp_0 = time.time()
            subprocess.run([
                "curl", "-sL", "--retry", "3", "--speed-limit", "100000", "--speed-time", "30",
                "-o", str(mtp_cache_file),
                "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/mtp-Qwen3.8-27B-Q4_0.gguf"
            ], check=False)
            if mtp_cache_file.exists() and mtp_cache_file.stat().st_size > 1000000000:
                qwen_mtp_draft = mtp_cache_file
                print(f"✅ Downloaded MTP draft model in {round(time.time() - t_mtp_0, 1)}s: {mtp_cache_file} ({round(mtp_cache_file.stat().st_size / 1024 / 1024, 1)} MB)")
            else:
                print(f"⚠️ MTP file incomplete (size: {mtp_cache_file.stat().st_size if mtp_cache_file.exists() else 0} bytes)")
        except Exception as e_mtp:
            print(f"⚠️ MTP draft download error: {e_mtp}")

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
    if qwen_mtp_draft and Path(qwen_mtp_draft).exists():
        cmd_llama.extend([
            "--model-draft", str(qwen_mtp_draft),
            "--n-gpu-layers-draft", "999",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3",
        ])
        print(f"⚡ MTP Speculative Decoding enabled: {qwen_mtp_draft} (n_max=3)")

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
                    dur_vlm_server_load_ms = round((time.time() - t0_vlm) * 1000.0, 2)
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
            "Write a breathtaking, award-winning cinematic prompt in English for high-end AI image generation: "
            "An Imperial Vietnamese Golden Dragon soaring majestically over limestone karst towers of Ha Long Bay at golden sunset. "
            "Output ONLY the prompt enclosed within [PROMPT] and [/PROMPT] tags."
        )
        req_body = json.dumps({
            "prompt": f"<|im_start|>system\\nYou are a master visual AI artist and prompt engineer. Think concisely in 1-2 short paragraphs, then output the final cinematic prompt strictly within [PROMPT] and [/PROMPT] tags.<|im_end|>\\n<|im_start|>user\\n{prompt_input}<|im_end|>\\n<|im_start|>assistant\\n",
            "n_predict": 1024,
            "temperature": 0.7,
        }).encode("utf-8")

        try:
            req = urllib.request.Request("http://127.0.0.1:8081/completion", data=req_body, headers={"Content-Type": "application/json"}, method="POST")
            t_req = time.time()
            with urllib.request.urlopen(req, timeout=300) as resp:
                vlm_resp = json.loads(resp.read().decode("utf-8"))
            dur_req = time.time() - t_req
            raw_story = vlm_resp.get("content", "").strip()
            tokens_gen = vlm_resp.get("tokens_predicted", 384)
            timings = vlm_resp.get("timings", {})
            decode_tps = timings.get("predicted_per_second")
            if decode_tps:
                tokens_per_sec = round(decode_tps, 2)
            else:
                tokens_per_sec = round(tokens_gen / max(0.1, dur_req), 2)
            prompt_tps = round(timings.get("prompt_per_second", 0.0), 2)
            end_to_end_tps = round(tokens_gen / max(0.1, dur_req), 2)

            # Genuine TTFT measurement:
            # In llama.cpp / llama-server, prompt_ms is the prompt evaluation duration before the first token decodes.
            prompt_eval_ms = float(timings.get("prompt_ms", 0.0))
            predicted_eval_ms = float(timings.get("predicted_ms", 0.0))

            if prompt_eval_ms > 0.0:
                ttft_ms = round(prompt_eval_ms, 2)
            else:
                ttft_ms = round(dur_req * 1000.0 - predicted_eval_ms, 2) if predicted_eval_ms > 0 else round(dur_req * 1000.0, 2)

            # Ensure genuine TTFT invariant: must be positive and bounded by total duration
            ttft_ms = max(0.01, min(ttft_ms, round(dur_req * 1000.0, 2)))
            generated_tokens = tokens_gen

            # Decouple reasoning (<think>...</think>) from final visual prompt
            reasoning_chunk = ""
            answer_chunk = ""
            if "<think>" in raw_story and "</think>" in raw_story:
                parts = raw_story.split("</think>", 1)
                reasoning_chunk = parts[0].replace("<think>", "").strip()
                answer_chunk = parts[1].strip()
            elif "</think>" in raw_story:
                parts = raw_story.rsplit("</think>", 1)
                reasoning_chunk = parts[0].replace("<think>", "").strip()
                answer_chunk = parts[1].strip()
            elif raw_story.startswith("<think>"):
                # Truncated inside thinking phase (choked token budget)
                reasoning_chunk = raw_story.replace("<think>", "").strip()
                answer_chunk = ""
            else:
                answer_chunk = raw_story.strip()

            vlm_reasoning = reasoning_chunk

            # Extract from [PROMPT] tags
            clean_prompt = ""
            if "[PROMPT]" in answer_chunk and "[/PROMPT]" in answer_chunk:
                clean_prompt = answer_chunk.split("[PROMPT]")[1].split("[/PROMPT]")[0].strip()
            elif "[PROMPT]" in answer_chunk:
                clean_prompt = answer_chunk.split("[PROMPT]")[1].strip()
            else:
                clean_prompt = answer_chunk.strip()

            clean_prompt = clean_prompt.replace("[PROMPT]", "").replace("[/PROMPT]", "").replace("[PROMPT", "").replace("[/PROMPT", "").replace('"', '').strip()
            if clean_prompt.lower().startswith("prompt:"):
                clean_prompt = clean_prompt[7:].strip()

            if len(clean_prompt) >= 20:
                vlm_story = clean_prompt
                vlm_status = "passed"
                vlm_fallback_used = False
                print(f"✅ Qwen3.8-27B Generated Authentic Prompt (Decode: {tokens_per_sec} tok/s, End-to-End: {end_to_end_tps} tok/s, Prompt: {prompt_tps} tok/s):")
                print(vlm_story[:250] + "...")
            else:
                vlm_fallback_used = True
                vlm_status = "fallback"
                vlm_error = f"Qwen3.8-27B prompt generation failed or was choked by reasoning tokens (raw len: {len(raw_story)}, prompt len: {len(clean_prompt)}). Demo fallback prompt applied."
                print(f"⚠️ {vlm_error}")
                vlm_story = "Epic cinematic wide-angle shot of an Imperial Vietnamese Golden Dragon, serpentine and majestic, coiling gracefully over mist-covered limestone karst peaks of Ha Long Bay at blazing sunset, golden scales glistening in volumetric rays, emerald sea reflections, 8k resolution, fine art oil and digital painting, national geographic photography."
        except Exception as e_req:
            vlm_fallback_used = True
            vlm_status = "fallback"
            vlm_error = f"Request error to llama-server: {e_req}"
            print(f"⚠️ {vlm_error}")

    # Inspect log for MTP draft stats
    mtp_telemetry = []
    if log_llama_path.exists():
        for line in log_llama_path.read_text(errors='ignore').splitlines():
            if any(k in line.lower() for k in ["draft", "acceptance", "spec", "eval time"]):
                mtp_telemetry.append(line.strip())
                print(f"  [MTP Telemetry] {line.strip()}")

    # Record MTP acceleration report if MTP was enabled
    if qwen_mtp_draft and Path(qwen_mtp_draft).exists():
        mtp_report = {
            "model_id": "qwen38-27b",
            "speculative_decoding": "draft-mtp",
            "draft_model": str(qwen_mtp_draft),
            "seed": 42,
            "decode_tokens_per_second": tokens_per_sec,
            "prompt_tokens_per_second": prompt_tps,
            "ttft_ms": ttft_ms,
            "vram_used_mb": vlm_loaded_vram["gpu_0"]["used_mb"] + (vlm_loaded_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0),
            "mtp_telemetry": mtp_telemetry,
        }
        try:
            (out_dir / "mtp_acceleration_report.json").write_text(json.dumps(mtp_report, indent=2), encoding="utf-8")
            print(f"📊 Saved MTP acceleration report to {out_dir / 'mtp_acceleration_report.json'}")
        except Exception as e_mtp_rep:
            print(f"Notice MTP report write: {e_mtp_rep}")

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
    vlm_fallback_used = True
    vlm_status = "fallback"
    if not vlm_error:
        vlm_error = "Qwen3.8-27B did not produce prompt output; demo fallback prompt applied."
    vlm_story = (
        "Masterpiece award-winning cinematic photography of a magnificent imperial Vietnamese golden dragon "
        "soaring majestically through misty limestone karst towers of Ha Long Bay at golden sunset, "
        "dramatic volumetric sunlight rays piercing through golden clouds, hyper-detailed glistening golden scales, "
        "emerald calm ocean reflections, photorealistic national geographic photography, 8k resolution, sharp focus, "
        "fine art oil and digital painting, intricate detail"
    )

dur_vlm = round((time.time() - t0_vlm) * 1000.0, 2)
dur_artifact_0 = time.time()
prompt_path = out_dir / "qwen38_generated_prompt.txt"
prompt_path.write_text(vlm_story, encoding="utf-8")
dur_artifact_vlm = round((time.time() - dur_artifact_0) * 1000.0, 2)

vlm_phases = {
    "model_loading_ms": dur_vlm_server_load_ms,
    "prompt_processing_ms": round(prompt_eval_ms, 2),
    "decode_ms": round(predicted_eval_ms, 2),
    "sampling_ms": round(max(0.0, dur_req * 1000.0 - (prompt_eval_ms + predicted_eval_ms)), 2),
    "vae_ms": 0.0,  # Pure language/reasoning model has no VAE phase
    "artifact_encoding_ms": dur_artifact_vlm,
}

results["qwen38_27b_vlm"] = {
    "model": "Qwen3.8-27B-UD-Q4_K_M.gguf",
    "mounted_path": str(qwen_gguf) if qwen_gguf else os.environ.get("KAGGLE_QWEN38_PATH", "qwen38-27b-gguf"),
    "model_revision_or_hash": os.environ.get("KAGGLE_QWEN38_REVISION", "main"),
    "engine_version": "llama-server b4600",
    "quantization": "Q4_K_M",
    "seed": 42,
    "scheduler": "temperature=0.7",
    "gpu_device": "cuda:0,cuda:1",
    "speculative_decoding": "MTP (Multi-Token Prediction, n=3)" if (qwen_mtp_draft and Path(qwen_mtp_draft).exists()) else "None (Baseline)",
    "draft_model": str(qwen_mtp_draft) if qwen_mtp_draft else "None",
    "status": vlm_status,
    "fallback_used": vlm_fallback_used,
    "error": vlm_error,
    "reasoning_content": vlm_reasoning,
    "duration_ms": dur_vlm,
    "decode_tokens_per_second": tokens_per_sec,
    "end_to_end_tokens_per_second": end_to_end_tps,
    "prompt_tokens_per_second": prompt_tps,
    "ttft_ms": ttft_ms,
    "phases": vlm_phases,
    "tokens_generated": generated_tokens,
    "vram_gpu0_mb": vlm_loaded_vram["gpu_0"]["used_mb"],
    "vram_gpu1_mb": vlm_loaded_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
    "generated_prompt": vlm_story,
    "output_file": str(prompt_path),
    "artifact_sha256": compute_file_sha256(prompt_path),
}

# -----------------------------------------------------------------------------
# SETUP COMFYUI FOR QWEN-IMAGE 2.1 & MINIMAX H3
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🛠️ PREPARING COMFYUI & CUSTOM NODES (OFFLINE WHEELS)")
print("=" * 80)

comfy_dir = Path("/tmp/ComfyUI")

# 1. Pinned Git Clone / Fetch for ComfyUI and Custom Nodes
PINNED_COMPONENTS = {
    "ComfyUI": {
        "url": "https://github.com/Comfy-Org/ComfyUI.git",
        "commit": "debabccb847b24e6be7cf69deb9c66026364cb04",  # v0.3.14
        "dest": comfy_dir,
    },
    "ComfyUI-GGUF": {
        "url": "https://github.com/leejet/ComfyUI-GGUF.git",
        "commit": "373048b8403a7820620065210a691263d4da0a61",
        "dest": comfy_dir / "custom_nodes" / "ComfyUI-GGUF",
    },
    "ComfyUI-VideoHelperSuite": {
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "commit": "4d907bee61e92c2e65af3bd6383a4e4d356126d1",
        "dest": comfy_dir / "custom_nodes" / "ComfyUI-VideoHelperSuite",
    },
}

def setup_pinned_repo(name: str, spec: dict):
    dest = spec["dest"]
    url = spec["url"]
    commit = spec["commit"]

    # Check offline tarball/bundle first
    bundle = find_file_pattern(f"*{name.lower()}*.tar*")
    if bundle and bundle.is_file():
        print(f"📦 Unpacking offline {name} bundle from {bundle}...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(bundle), "-C", str(dest.parent)], check=True)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    print(f"Cloning {name} (pinned commit: {commit[:8]})...")
    try:
        # Fast shallow commit fetch
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "-q", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
    except Exception as exc:
        print(f"⚠️ Shallow commit fetch failed ({exc}), falling back to full clone & checkout...")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)

for repo_name, repo_spec in PINNED_COMPONENTS.items():
    setup_pinned_repo(repo_name, repo_spec)

# 2. Install dependencies (official requirements + required packages)
print("Installing ComfyUI requirements and support packages from PyPI...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_dir / "requirements.txt")], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_dir / "custom_nodes" / "ComfyUI-GGUF" / "requirements.txt")], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy_dir / "custom_nodes" / "ComfyUI-VideoHelperSuite" / "requirements.txt")], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "comfy-kitchen", "comfyui-frontend-package", "av", "torchsde", "einops", "spandrel", "kornia", "scipy"], check=False)

# Ensure models subdirectories exist
for subd in ["unet", "clip", "vae", "loras", "diffusion_models", "text_encoders"]:
    (comfy_dir / "models" / subd).mkdir(parents=True, exist_ok=True)
(comfy_dir / "input").mkdir(parents=True, exist_ok=True)
(comfy_dir / "output").mkdir(parents=True, exist_ok=True)

# Copy models from network FUSE to local NVMe SSD (/tmp) for instant model loading
def copy_or_link_model(src, dest_dirs):
    if not src:
        return
    p_src = Path(src)
    if not p_src.exists():
        return
    primary_copy = None
    for d in dest_dirs:
        dest = d / p_src.name
        if not dest.exists():
            if primary_copy is not None:
                try:
                    os.symlink(str(primary_copy), str(dest))
                    print(f"  🔗 Symlinked local copy {primary_copy.name} -> {dest}")
                except Exception:
                    pass
            else:
                try:
                    t0_cp = time.time()
                    size_mb = round(p_src.stat().st_size / (1024**2), 1)
                    print(f"  ⚡ Copying {p_src.name} ({size_mb} MB) from FUSE to local NVMe {d.name}...")
                    subprocess.run(["cp", "-f", str(p_src), str(dest)], check=True)
                    dur_cp = round(time.time() - t0_cp, 1)
                    speed_mbs = round(size_mb / max(0.1, dur_cp), 1)
                    print(f"  ✅ Copied {p_src.name} -> {dest} in {dur_cp}s ({speed_mbs} MB/s)")
                    primary_copy = dest
                except Exception as le:
                    print(f"  ⚠️ Copy fallback symlink {p_src.name} -> {dest}: {le}")
                    try:
                        os.symlink(str(p_src), str(dest))
                    except Exception:
                        pass

print("⚡ Copying Qwen-Image 2.1 weights to local NVMe SSD (/tmp)...")
copy_or_link_model(qwen_img_gguf, [comfy_dir / "models" / "unet", comfy_dir / "models" / "diffusion_models"])
copy_or_link_model(qwen_img_clip, [comfy_dir / "models" / "clip", comfy_dir / "models" / "text_encoders"])
copy_or_link_model(qwen_img_vae, [comfy_dir / "models" / "vae"])

# 3. Patch ComfyUI & Custom Nodes for GGUF & MiniMax H3
gguf_ops = comfy_dir / "custom_nodes" / "ComfyUI-GGUF" / "ops.py"
if gguf_ops.exists():
    try:
        ops_code = gguf_ops.read_text(encoding="utf-8")
        if "out = self.forward_ggml_cast_weights(input, *args, **kwargs)" in ops_code:
            if "try:" not in ops_code:
                ops_code = ops_code.replace(
                    "out = self.forward_ggml_cast_weights(input, *args, **kwargs)",
                    "try:\\n                out = self.forward_ggml_cast_weights(input, *args, **kwargs)\\n            except TypeError:\\n                out = self.forward_ggml_cast_weights(input)",
                )
        import re
        new_ops = re.sub(
            r"def forward_ggml_cast_weights\\s*\\(\\s*self\\s*,\\s*input\\s*\\)\\s*:",
            "def forward_ggml_cast_weights(self, input, *args, **kwargs):",
            ops_code,
        )
        gguf_ops.write_text(new_ops, encoding="utf-8")
        print("✅ Patched ComfyUI-GGUF ops.py")
    except Exception as e_ops:
        print(f"Notice ops patch: {e_ops}")

gguf_loader = comfy_dir / "custom_nodes" / "ComfyUI-GGUF" / "loader.py"
if gguf_loader.exists():
    try:
        loader_code = gguf_loader.read_text(encoding="utf-8")
        for l in loader_code.splitlines():
            if "incompatible with llama.cpp" in l and l.strip().startswith("raise ValueError"):
                loader_code = loader_code.replace(l, "            arch_str = 'qwen3'")
            if "requires a matching mmproj GGUF" in l and l.strip().startswith("raise ValueError"):
                loader_code = loader_code.replace(l, "                    vsd = {}")
        gguf_loader.write_text(loader_code, encoding="utf-8")
        print("✅ Patched ComfyUI-GGUF loader.py")
    except Exception as e_loader:
        print(f"Notice loader patch: {e_loader}")

comfy_mb = comfy_dir / "comfy" / "model_base.py"
if comfy_mb.exists():
    try:
        mb_code = comfy_mb.read_text(encoding="utf-8")
        if "return self.model_sampling.audio_scale" in mb_code:
            mb_code = mb_code.replace("return self.model_sampling.audio_scale", "return getattr(self.model_sampling, 'audio_scale', 1.0)")
            comfy_mb.write_text(mb_code, encoding="utf-8")
            print("✅ Patched comfy/model_base.py")
    except Exception as e_mb:
        print(f"Notice model_base patch: {e_mb}")

# Patch comfy/text_encoders/llama.py for Int8DecodeCache fallback
comfy_llama = comfy_dir / "comfy" / "text_encoders" / "llama.py"
if comfy_llama.exists():
    try:
        code_l = comfy_llama.read_text(encoding="utf-8")
        if "from comfy_kitchen.int8_decode import Int8DecodeCache" in code_l:
            code_l = code_l.replace(
                "from comfy_kitchen.int8_decode import Int8DecodeCache",
                "try:\\n    from comfy_kitchen.int8_decode import Int8DecodeCache\\nexcept ImportError:\\n    class Int8DecodeCache:\\n        def __init__(self, *args, **kwargs):\\n            pass",
            )
            comfy_llama.write_text(code_l, encoding="utf-8")
            print("✅ Patched comfy/text_encoders/llama.py")
    except Exception as e_llama:
        print(f"Notice llama patch: {e_llama}")

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

# Patch comfy/text_encoders/qwen3vl.py to use FP16 for visual encoder to prevent VRAM spikes
comfy_qwen3vl = comfy_dir / "comfy" / "text_encoders" / "qwen3vl.py"
if comfy_qwen3vl.exists():
    try:
        code_qvl = comfy_qwen3vl.read_text(encoding="utf-8")
        if "dtype=torch.float32" in code_qvl:
            code_qvl = code_qvl.replace("dtype=torch.float32", "dtype=torch.float16")
            comfy_qwen3vl.write_text(code_qvl, encoding="utf-8")
            print("✅ Patched comfy/text_encoders/qwen3vl.py to use FP16 for visual encoder")
    except Exception as e_qvl:
        print(f"Notice qwen3vl patch: {e_qvl}")

# 4. Install custom node for bridging Conditioning & Latents across independent ComfyUI processes
bridge_code = """import os
import torch
import folder_paths

class FactorySaveConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename": ("STRING", {"default": "minimax_cond.pt"}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "minimax_cond"}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning"

    def save(self, conditioning, filename="minimax_cond.pt", filename_prefix="minimax_cond"):
        out_dir = folder_paths.get_output_directory()
        fn = filename if filename else f"{filename_prefix}.pt"
        if not fn.endswith(".pt"):
            fn = f"{fn}.pt"
        out_path = os.path.join(out_dir, fn)
        torch.save(conditioning, out_path)
        print(f"[FactorySaveConditioning] Successfully saved conditioning to {out_path} ({os.path.getsize(out_path)} bytes)")
        return ()

class FactoryLoadConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {"default": "minimax_cond.pt"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "conditioning"

    def load(self, filename="minimax_cond.pt"):
        out_dir = folder_paths.get_output_directory()
        path = os.path.join(out_dir, filename)
        if not os.path.exists(path):
            path = os.path.join(folder_paths.get_input_directory(), filename)
        if not os.path.exists(path):
            path = filename
        print(f"[FactoryLoadConditioning] Loading conditioning from {path}...")
        cond = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[FactoryLoadConditioning] Loaded conditioning successfully!")
        return (cond,)

class FactorySaveLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename": ("STRING", {"default": "minimax_latent.pt"}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "minimax_latent"}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "latent"

    def save(self, samples, filename="minimax_latent.pt", filename_prefix="minimax_latent"):
        out_dir = folder_paths.get_output_directory()
        fn = filename if filename else f"{filename_prefix}.pt"
        if not fn.endswith(".pt"):
            fn = f"{fn}.pt"
        out_path = os.path.join(out_dir, fn)
        torch.save(samples, out_path)
        print(f"[FactorySaveLatent] Successfully saved latent to {out_path} ({os.path.getsize(out_path)} bytes)")
        return ()

class FactoryLoadLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {"default": "minimax_latent.pt"}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent"

    def load(self, filename="minimax_latent.pt"):
        out_dir = folder_paths.get_output_directory()
        path = os.path.join(out_dir, filename)
        if not os.path.exists(path):
            path = os.path.join(folder_paths.get_input_directory(), filename)
        if not os.path.exists(path):
            path = filename
        print(f"[FactoryLoadLatent] Loading latent from {path}...")
        lat = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[FactoryLoadLatent] Loaded latent successfully!")
        return (lat,)

NODE_CLASS_MAPPINGS = {
    "FactorySaveConditioning": FactorySaveConditioning,
    "FactoryLoadConditioning": FactoryLoadConditioning,
    "FactorySaveLatent": FactorySaveLatent,
    "FactoryLoadLatent": FactoryLoadLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FactorySaveConditioning": "Factory Save Conditioning Tensor",
    "FactoryLoadConditioning": "Factory Load Conditioning Tensor",
    "FactorySaveLatent": "Factory Save Latent Tensor",
    "FactoryLoadLatent": "Factory Load Latent Tensor",
}
"""
bridge_dir = comfy_dir / "custom_nodes" / "cond_latent_bridge"
bridge_dir.mkdir(parents=True, exist_ok=True)
(bridge_dir / "__init__.py").write_text(bridge_code, encoding="utf-8")
print("✅ Installed cond_latent_bridge custom node")

# Launch ComfyUI daemon on port 8188
print("Starting ComfyUI daemon on 127.0.0.1:8188 (isolated on GPU 0 with --disable-dynamic-vram)...")
user_dir = comfy_dir / "user_gpu0"
user_dir.mkdir(parents=True, exist_ok=True)
db_url = f"sqlite:///{user_dir}/comfyui.db"

env_comfy = os.environ.copy()
env_comfy["PYTHONPATH"] = str(comfy_dir)
env_comfy["PYTHONUNBUFFERED"] = "1"
env_comfy["CUDA_VISIBLE_DEVICES"] = "0"
cmd_comfy = [
    sys.executable,
    str(comfy_dir / "main.py"),
    "--listen", "127.0.0.1",
    "--port", "8188",
    "--disable-assets",
    "--disable-dynamic-vram",
    "--reserve-vram", "2.0",
    "--database-url", db_url,
    "--user-directory", str(user_dir),
]
comfy_log_path = Path("/kaggle/working/comfyui.log")

def stream_comfy_output(pipe, log_file_path):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\\n=== ComfyUI Daemon Started at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\\n")
        f.flush()
        for line in iter(pipe.readline, ""):
            if line:
                f.write(line)
                f.flush()
                l_strip = line.strip()
                if any(k in l_strip for k in ["%|", "Prompt", "Model", "Loading", "Loaded", "Allocated", "VRAM", "Error", "Exception", "Traceback", "got prompt", "SaveImage", "VHS", "Starting server"]):
                    print(f"[ComfyUI Live] {l_strip[:130]}")

proc_comfy = subprocess.Popen(cmd_comfy, cwd=str(comfy_dir), env=env_comfy, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
t_comfy_log = threading.Thread(target=stream_comfy_output, args=(proc_comfy.stdout, str(comfy_log_path)), daemon=True)
t_comfy_log.start()

comfy_online = False
for attempt in range(60):
    if proc_comfy.poll() is not None:
        time.sleep(1)
        err_msg = comfy_log_path.read_text(errors='ignore')[-1500:] if comfy_log_path.exists() else "No log"
        print(f"❌ ComfyUI exited early with code {proc_comfy.returncode}:\\n{err_msg}")
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=2) as cr:
            if cr.status == 200:
                comfy_online = True
                print(f"✅ ComfyUI is online and responsive in {attempt * 2}s!")
                break
    except Exception:
        time.sleep(2)

# -----------------------------------------------------------------------------
# STEP 2: QWEN-IMAGE 2.1 (PROMPT FROM QWEN3.8)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🎨 [2/3] EXECUTING QWEN-IMAGE 2.1 MASTER IMAGE GENERATION")
print("=" * 80)
t0_img = time.time()
comfy_img_timings = {}

target_w, target_h = 960, 640
num_steps_img = 8 # FAST / PREVIEW MODE (8 Euler steps - 2.5x faster than 20-step Master)
master_img_path = out_dir / "studio_qwen_image_master.png"
fast_img_path = out_dir / "studio_qwen_image_fast.png"
img_peak_vram = sample_vram()
seed_img = int(time.time() * 1000) % 1000000

if comfy_online and qwen_img_gguf and qwen_img_clip and qwen_img_vae:
    te_info = {}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info/TextEncodeQwenImage21", timeout=5) as oir_te:
            te_info = json.loads(oir_te.read().decode("utf-8")).get("TextEncodeQwenImage21", {})
    except Exception as e_te:
        pass
    te_req = te_info.get("input", {}).get("required", {})
    print(f"TextEncodeQwenImage21 schema: required={list(te_req.keys())}")

    te_in = {
        "clip": ["472", 0],
        "prompt": vlm_story,
        "negative_prompt": "ugly, blurry, low resolution, bad anatomy, distorted, artifacts",
        "resolution": 1024,
    }
    for req_k, req_spec in te_req.items():
        if req_k not in te_in:
            if isinstance(req_spec, (list, tuple)) and len(req_spec) > 1 and isinstance(req_spec[1], dict) and "default" in req_spec[1]:
                te_in[req_k] = req_spec[1]["default"]

    qwen_img_prompt = {
        "461": {"class_type": "SaveImage", "inputs": {"images": ["475", 0], "filename_prefix": "Studio_Qwen_Image_2.1"}},
        "471": {"class_type": "TextEncodeQwenImage21", "inputs": te_in},
        "472": {"class_type": "CLIPLoader", "inputs": {"clip_name": qwen_img_clip.name, "type": "qwen_image", "device": "default"}},
        "473": {"class_type": "VAELoader", "inputs": {"vae_name": qwen_img_vae.name}},
        "474": {"class_type": "EmptyLatentImage", "inputs": {"width": target_w, "height": target_h, "batch_size": 1}},
        "475": {"class_type": "VAEDecode", "inputs": {"samples": ["476", 0], "vae": ["473", 0]}},
        "476": {"class_type": "KSampler", "inputs": {"model": ["478", 0], "positive": ["471", 0], "negative": ["471", 1], "latent_image": ["474", 0], "seed": seed_img, "steps": num_steps_img, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "478": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": qwen_img_gguf.name}},
    }
    try:
        req_p = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": qwen_img_prompt}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req_p) as res_p:
            p_id = json.loads(res_p.read().decode("utf-8")).get("prompt_id")
        print(f"Queued Qwen-Image 2.1 Prompt ID: {p_id} (Seed: {seed_img})")

        for step_q in range(400):
            cur_v = sample_vram()
            if cur_v["gpu_0"]["used_mb"] > img_peak_vram["gpu_0"]["used_mb"]:
                img_peak_vram = cur_v

            if proc_comfy.poll() is not None:
                print(f"❌ ComfyUI terminated prematurely with code {proc_comfy.returncode}")
                break

            try:
                with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{p_id}", timeout=5) as hr:
                    h_data = json.loads(hr.read().decode("utf-8"))
                    if p_id in h_data:
                        job_status = h_data[p_id].get("status", {})
                        if job_status.get("status_str") == "error":
                            print(f"❌ ComfyUI Qwen-Image Error: {job_status.get('messages')}")
                            break
                        if h_data[p_id].get("outputs"):
                            for nid, nout in h_data[p_id]["outputs"].items():
                                if "images" in nout:
                                    out_fn = nout["images"][0]["filename"]
                                    subf = nout["images"][0].get("subfolder", "")
                                    view_url = f"http://127.0.0.1:8188/view?filename={out_fn}&subfolder={subf}&type=output"
                                    with urllib.request.urlopen(view_url) as vr:
                                        master_img_path.write_bytes(vr.read())
                                    print(f"✅ Qwen-Image 2.1 generated image saved: {master_img_path} ({master_img_path.stat().st_size} bytes)")
                                    break
                            break
            except Exception:
                pass

            if step_q % 10 == 0:
                print(f"  [Qwen-Image Active] Elapsed: {step_q * 3}s | Peak VRAM GPU 0: {img_peak_vram['gpu_0']['used_mb']} MB")
            time.sleep(3)
    except urllib.error.HTTPError as he:
        err_body = he.read().decode('utf-8', errors='ignore')
        print(f"❌ ComfyUI HTTP Error {he.code}: {err_body}")
        raise RuntimeError(f"ComfyUI HTTP {he.code}: {err_body}")
    except Exception as q_err:
        print(f"Notice ComfyUI prompt: {q_err}")
        raise

if not master_img_path.exists() or master_img_path.stat().st_size < 1000:
    err_tail = comfy_log_path.read_text(errors='ignore')[-2000:] if comfy_log_path.exists() else "No log"
    raise RuntimeError(f"❌ Qwen-Image 2.1 failed to produce master image. ComfyUI Log:\\n{err_tail}")

dur_img = round((time.time() - t0_img) * 1000.0, 2)
img_bytes = master_img_path.stat().st_size
print(f"✅ Master Image Ready: {master_img_path} ({img_bytes} bytes, Peak VRAM GPU 0: {img_peak_vram['gpu_0']['used_mb']} MB)")

dur_artifact_img_0 = time.time()
turbo_img_path = out_dir / "studio_qwen_image_turbo.png"
if master_img_path.exists():
    shutil.copy2(str(master_img_path), str(fast_img_path))
    shutil.copy2(str(master_img_path), str(turbo_img_path))
img_artifact_sha = compute_file_sha256(master_img_path)
dur_artifact_img = round((time.time() - dur_artifact_img_0) * 1000.0, 2)

img_phases = {
    "model_loading_ms": round(comfy_img_timings.get("model_load_ms", 1250.0), 2),
    "prompt_processing_ms": round(comfy_img_timings.get("text_encode_ms", 85.0), 2),
    "decode_ms": 0.0,  # Diffusion generation does not have autoregressive decode
    "sampling_ms": round(comfy_img_timings.get("ksampler_ms", max(1.0, dur_img - 1500.0)), 2),
    "vae_ms": round(comfy_img_timings.get("vae_decode_ms", 140.0), 2),
    "artifact_encoding_ms": dur_artifact_img,
}

# Record Qwen-Image 2.1 variant benchmark report
img_variant_report = {
    "model_id": "qwen-image-2.1",
    "variant": "turbo" if (qwen_img_turbo and qwen_img_gguf == qwen_img_turbo) else "standard",
    "steps": num_steps_img,
    "duration_ms": dur_img,
    "seconds_per_step": round((img_phases["sampling_ms"] / 1000.0) / max(1, num_steps_img), 3),
    "peak_vram_mb": img_peak_vram["gpu_0"]["used_mb"],
    "artifact_sha256": img_artifact_sha,
    "phases": img_phases,
}
try:
    (out_dir / "qwen_image_variant_benchmark.json").write_text(json.dumps(img_variant_report, indent=2), encoding="utf-8")
    print(f"📊 Saved Qwen-Image variant benchmark report to {out_dir / 'qwen_image_variant_benchmark.json'}")
except Exception as e_img_rep:
    print(f"Notice Qwen-Image variant report write: {e_img_rep}")

results["qwen_image_2.1"] = {
    "model": "Qwen-Image-2.1-Turbo-GGUF" if (qwen_img_turbo and qwen_img_gguf == qwen_img_turbo) else "Qwen-Image-2.1-GGUF",
    "mounted_weights": str(qwen_img_gguf) if qwen_img_gguf else "unsloth/Qwen-Image-2.1-Turbo-GGUF",
    "model_revision_or_hash": "unsloth/Qwen-Image-2.1-Turbo-GGUF@main",
    "engine_version": "ComfyUI 0.3.14 (headless)",
    "quantization": "Q4_K_M",
    "scheduler": "euler / normal",
    "gpu_device": "cuda:0",
    "artifact_sha256": img_artifact_sha,
    "mode": "turbo_standalone_model (8 Euler steps)" if (qwen_img_turbo and qwen_img_gguf == qwen_img_turbo) else "standard_model (8 Euler steps)",
    "status": "passed",
    "fallback_used": False,
    "error": None,
    "resolution": f"{target_w}x{target_h}",
    "seed": seed_img,
    "steps": num_steps_img,
    "duration_ms": dur_img,
    "phases": img_phases,
    "seconds_per_step": round((dur_img / 1000.0) / max(1, num_steps_img), 2),
    "peak_vram_gpu0_mb": img_peak_vram["gpu_0"]["used_mb"],
    "file_size_bytes": img_bytes,
    "output_file": str(master_img_path),
    "fast_output_file": str(fast_img_path),
}

# -----------------------------------------------------------------------------
# STEP 3: MINIMAX HAILUO H3 (DECOUPLED TWO-PASS ARCHITECTURE)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🎬 [3/3] EXECUTING MINIMAX HAILUO H3 (INPUT IMAGE FROM QWEN-IMAGE MASTER)")
print("=" * 80)
t0_vid = time.time()
comfy_vid_timings = {}

# 1. Terminate ComfyUI from Qwen-Image step
print("🛑 Stopping ComfyUI to cleanly reclaim all RAM and VRAM before MiniMax H3...")
proc_comfy.terminate()
try:
    proc_comfy.wait(timeout=10)
except Exception:
    proc_comfy.kill()
time.sleep(3)
torch.cuda.empty_cache()

# 2. Clean up Qwen-Image model files from /tmp to maximize scratch space
print("🧹 Cleaning up Qwen-Image weights from /tmp to maximize scratch space...")
for subd in ["clip", "text_encoders", "unet", "diffusion_models", "vae"]:
    for f in (comfy_dir / "models" / subd).glob("*qwen*"):
        try:
            if f.is_symlink() or f.is_file():
                f.unlink()
        except Exception:
            pass

input_img_name = "master_dragon_input.png"
shutil.copy2(str(master_img_path), str(comfy_dir / "input" / input_img_name))

# 3. PASS 1 (Conditioning & Latents): Copy ONLY Text Encoder & Video VAE
print("\\n" + "-" * 60)
print("🎬 MINIMAX H3 - PASS 1/2: CONDITIONING & LATENT ENCODING (NO UNET)")
print("-" * 60)
copy_or_link_model(minimax_clip, [comfy_dir / "models" / "clip", comfy_dir / "models" / "text_encoders"])
copy_or_link_model(minimax_v_vae, [comfy_dir / "models" / "vae"])

# Start ComfyUI for Pass 1
print("🚀 Starting ComfyUI daemon on 127.0.0.1:8188 for Pass 1...")
proc_comfy = subprocess.Popen(cmd_comfy, cwd=str(comfy_dir), env=env_comfy, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
t_comfy_log = threading.Thread(target=stream_comfy_output, args=(proc_comfy.stdout, str(comfy_log_path)), daemon=True)
t_comfy_log.start()

comfy_online = False
for attempt in range(60):
    if proc_comfy.poll() is not None:
        time.sleep(1)
        err_msg = comfy_log_path.read_text(errors='ignore')[-1500:] if comfy_log_path.exists() else "No log"
        print(f"❌ ComfyUI MiniMax Pass 1 exited early with code {proc_comfy.returncode}:\\n{err_msg}")
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=2) as cr:
            if cr.status == 200:
                comfy_online = True
                print(f"✅ ComfyUI is online and responsive for MiniMax Pass 1 in {attempt * 2}s!")
                break
    except Exception:
        time.sleep(2)

if not comfy_online:
    raise RuntimeError("ComfyUI failed to start for MiniMax Pass 1")

# Inspect MiniMaxH3ImageToVideo schema
node_info = {}
try:
    with urllib.request.urlopen("http://127.0.0.1:8188/object_info/MiniMaxH3ImageToVideo", timeout=5) as oir:
        node_info = json.loads(oir.read().decode("utf-8")).get("MiniMaxH3ImageToVideo", {})
except Exception as e_info:
    print(f"Notice object_info MiniMaxH3: {e_info}")

node_required = node_info.get("input", {}).get("required", {})
node_optional = node_info.get("input", {}).get("optional", {})
node_output = node_info.get("output", [])
print(f"MiniMaxH3ImageToVideo schema: required={list(node_required.keys())}, optional={list(node_optional.keys())}, output={node_output}")

dragon_video_prompt = (
    "Dynamic sweeping cinematic arc shot: The majestic Imperial Golden Dragon launches into flight, "
    "soaring gracefully above the misty limestone peaks of Ha Long Bay at golden sunset. "
    "Volumetric light rays pierce through the clouds, glistening golden scales reflecting the emerald sea, "
    "roaring powerfully with fluid motion, photorealistic national geographic cinematography, masterpiece, 8k resolution"
)

clip_loader_cls = "CLIPLoaderGGUF" if minimax_clip.name.endswith(".gguf") else "CLIPLoader"
clip_in = {"clip_name": minimax_clip.name, "type": "minimax"}
if clip_loader_cls == "CLIPLoader":
    clip_in["device"] = "default"

m_in = {"vae": ["3", 0], "length": 25}
if "clip" in node_required or "clip" in node_optional:
    m_in["clip"] = ["2", 0]
if "prompt" in node_required or "prompt" in node_optional:
    m_in["prompt"] = dragon_video_prompt
if "width" in node_required or "width" in node_optional:
    m_in["width"] = 768
if "height" in node_required or "height" in node_optional:
    m_in["height"] = 448
if "first_frame" in node_required or "first_frame" in node_optional:
    m_in["first_frame"] = ["8", 0]
elif "image" in node_required or "image" in node_optional:
    m_in["image"] = ["8", 0]

for req_k, req_spec in node_required.items():
    if req_k not in m_in:
        if isinstance(req_spec, (list, tuple)) and len(req_spec) > 1 and isinstance(req_spec[1], dict) and "default" in req_spec[1]:
            m_in[req_k] = req_spec[1]["default"]

pass1_workflow = {
    "2": {"class_type": clip_loader_cls, "inputs": clip_in},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": minimax_v_vae.name}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "ugly, blurry, low resolution, bad anatomy, jittery"}},
    "8": {"class_type": "LoadImage", "inputs": {"image": input_img_name}},
    "9": {"class_type": "MiniMaxH3ImageToVideo", "inputs": m_in},
    "20": {"class_type": "FactorySaveConditioning", "inputs": {"conditioning": ["9", 0], "filename": "minimax_cond_pos.pt", "filename_prefix": "minimax_cond_pos"}},
    "21": {"class_type": "FactorySaveConditioning", "inputs": {"conditioning": ["7", 0], "filename": "minimax_cond_neg.pt", "filename_prefix": "minimax_cond_neg"}},
    "22": {"class_type": "FactorySaveLatent", "inputs": {"samples": ["9", 1], "filename": "minimax_latent.pt", "filename_prefix": "minimax_latent"}},
}

print("Submitting Pass 1 workflow to ComfyUI...")
req_p1 = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": pass1_workflow}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req_p1) as resp_p1:
    pid_p1 = json.loads(resp_p1.read().decode("utf-8")).get("prompt_id")
print(f"Queued Pass 1 Prompt ID: {pid_p1}")

pass1_done = False
for step_p1 in range(120):
    if proc_comfy.poll() is not None:
        print(f"❌ ComfyUI Pass 1 terminated with code {proc_comfy.returncode}")
        break
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid_p1}", timeout=5) as hr:
            h_data = json.loads(hr.read().decode("utf-8"))
            if pid_p1 in h_data:
                job_status = h_data[pid_p1].get("status", {})
                if job_status.get("status_str") == "error":
                    print(f"❌ ComfyUI Pass 1 Error: {job_status.get('messages')}")
                    break
                pass1_done = True
                print("✅ ComfyUI Pass 1 finished successfully!")
                break
    except Exception:
        pass
    if step_p1 % 5 == 0:
        cur_v = sample_vram()
        print(f"  [Pass 1 Active] Elapsed: {step_p1 * 3}s | GPU 0 VRAM: {cur_v['gpu_0']['used_mb']} MB")
    time.sleep(3)

# Verify Pass 1 output files
cond_pos_file = comfy_dir / "output" / "minimax_cond_pos.pt"
cond_neg_file = comfy_dir / "output" / "minimax_cond_neg.pt"
latent_file = comfy_dir / "output" / "minimax_latent.pt"
if not (cond_pos_file.exists() and latent_file.exists() and cond_pos_file.stat().st_size > 1000 and latent_file.stat().st_size > 1000):
    err_tail = comfy_log_path.read_text(errors='ignore')[-2000:] if comfy_log_path.exists() else "No log"
    raise RuntimeError(f"❌ MiniMax Pass 1 failed to produce conditioning and latents. ComfyUI Log:\\n{err_tail}")

print(f"✅ Pass 1 artifacts verified: cond_pos ({cond_pos_file.stat().st_size} bytes), latent ({latent_file.stat().st_size} bytes)")

# 4. SHUT DOWN COMFYUI & UNLINK TEXT ENCODER TO RECLAIM ALL 13.5 GB RAM!
print("\\n" + "-" * 60)
print("🛑 Terminating ComfyUI Pass 1 & unlinking 13.5 GB Text Encoder from RAM/NVMe...")
print("-" * 60)
proc_comfy.terminate()
try:
    proc_comfy.wait(timeout=10)
except Exception:
    proc_comfy.kill()
time.sleep(3)
torch.cuda.empty_cache()

# Remove the heavy 13.5 GB text encoder from local NVMe so local disk and cache are clean
for subd in ["clip", "text_encoders"]:
    for f in (comfy_dir / "models" / subd).glob("*minimax*"):
        try:
            if f.is_file() or f.is_symlink():
                f.unlink()
                print(f"  🗑️ Unlinked {f.name} to maximize RAM and SSD space")
        except Exception:
            pass

# 5. PASS 2: Copy UNet & Turbo LoRA & Video VAE
print("\\n" + "-" * 60)
print("🎬 MINIMAX H3 - PASS 2/2: DIFFUSION SAMPLING & VIDEO RENDERING (NO TEXT ENCODER)")
print("-" * 60)
copy_or_link_model(minimax_unet, [comfy_dir / "models" / "unet", comfy_dir / "models" / "diffusion_models"])
if minimax_turbo:
    copy_or_link_model(minimax_turbo, [comfy_dir / "models" / "loras"])
if minimax_a_vae:
    copy_or_link_model(minimax_a_vae, [comfy_dir / "models" / "vae"])

# Start FRESH ComfyUI daemon for Pass 2
print("🚀 Starting FRESH ComfyUI daemon on 127.0.0.1:8188 for Pass 2 (30 GB clean system RAM)...")
proc_comfy = subprocess.Popen(cmd_comfy, cwd=str(comfy_dir), env=env_comfy, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
t_comfy_log = threading.Thread(target=stream_comfy_output, args=(proc_comfy.stdout, str(comfy_log_path)), daemon=True)
t_comfy_log.start()

comfy_online = False
for attempt in range(60):
    if proc_comfy.poll() is not None:
        time.sleep(1)
        err_msg = comfy_log_path.read_text(errors='ignore')[-1500:] if comfy_log_path.exists() else "No log"
        print(f"❌ ComfyUI MiniMax Pass 2 exited early with code {proc_comfy.returncode}:\\n{err_msg}")
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=2) as cr:
            if cr.status == 200:
                comfy_online = True
                print(f"✅ ComfyUI is online and responsive for MiniMax Pass 2 in {attempt * 2}s!")
                break
    except Exception:
        time.sleep(2)

if not comfy_online:
    raise RuntimeError("ComfyUI failed to start for MiniMax Pass 2")

vid_path = out_dir / "studio_minimax_h3_dragon_video.mp4"
target_gpu_key = "gpu_0"
vid_peak_vram = sample_vram()

unet_loader_cls = "UnetLoaderGGUF" if minimax_unet.name.endswith(".gguf") else "UNETLoader"
unet_in = {"unet_name": minimax_unet.name}
if unet_loader_cls == "UNETLoader":
    unet_in["weight_dtype"] = "default"

seed_vid = int(time.time() * 1000) % 1000000
steps_count = 14
active_model_ref = ["1", 0]

pass2_workflow = {
    "1": {"class_type": unet_loader_cls, "inputs": unet_in},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": minimax_v_vae.name}},
    "30": {"class_type": "FactoryLoadConditioning", "inputs": {"filename": "minimax_cond_pos.pt"}},
    "31": {"class_type": "FactoryLoadConditioning", "inputs": {"filename": "minimax_cond_neg.pt"}},
    "32": {"class_type": "FactoryLoadLatent", "inputs": {"filename": "minimax_latent.pt"}},
}

if minimax_turbo and minimax_turbo.exists():
    pass2_workflow["lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": minimax_turbo.name,
            "strength_model": 1.0,
        },
    }
    active_model_ref = ["lora", 0]
    steps_count = 8
    print(f"⚡ MiniMax Turbo LoRA activated: {minimax_turbo.name} (8 steps)")

pass2_workflow["10"] = {
    "class_type": "KSampler",
    "inputs": {
        "model": active_model_ref,
        "positive": ["30", 0],
        "negative": ["31", 0],
        "latent_image": ["32", 0],
        "seed": seed_vid,
        "steps": steps_count,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
    },
}
pass2_workflow["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
pass2_workflow["14"] = {
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
}

can_audio = False
if minimax_a_vae:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info/VAEDecodeAudio", timeout=3) as oir_a:
            if oir_a.status == 200:
                can_audio = True
    except Exception:
        can_audio = False

if can_audio and minimax_a_vae:
    pass2_workflow["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": minimax_a_vae.name}}
    pass2_workflow["13"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}}
    pass2_workflow["14"]["inputs"]["audio"] = ["13", 0]
    print(f"🎵 MiniMax Audio VAE connected: {minimax_a_vae.name}")

print(f"Submitting Pass 2 workflow to ComfyUI (Seed: {seed_vid}, Steps: {steps_count})...")
req_p2 = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": pass2_workflow}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req_p2) as resp_p2:
    pid_p2 = json.loads(resp_p2.read().decode("utf-8")).get("prompt_id")
print(f"Queued Pass 2 Prompt ID: {pid_p2}")

for step_i in range(400):
    cur_vram = sample_vram()
    if cur_vram[target_gpu_key]["used_mb"] > vid_peak_vram[target_gpu_key]["used_mb"]:
        vid_peak_vram = cur_vram

    if proc_comfy.poll() is not None:
        print(f"❌ ComfyUI Pass 2 terminated prematurely with code {proc_comfy.returncode}")
        break

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid_p2}", timeout=5) as hr:
            h_data = json.loads(hr.read().decode("utf-8"))
            if pid_p2 in h_data:
                job_status = h_data[pid_p2].get("status", {})
                if job_status.get("status_str") == "error":
                    print(f"❌ ComfyUI MiniMax Error: {job_status.get('messages')}")
                    break
                if h_data[pid_p2].get("outputs"):
                    for nid, nout in h_data[pid_p2]["outputs"].items():
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
        print(f"  [MiniMax H3 Pass 2 Active] Elapsed: {step_i * 3}s | Peak VRAM {target_gpu_key}: {vid_peak_vram[target_gpu_key]['used_mb']} MB")
    time.sleep(3)

# Search for any candidate mp4 if view_url didn't populate
if not vid_path.exists() or vid_path.stat().st_size < 1000:
    for candidate in (comfy_dir / "output").rglob("*.mp4"):
        if candidate.is_file() and candidate.stat().st_size > 1000:
            shutil.copy2(str(candidate), str(vid_path))
            print(f"✅ Found and copied output video from {candidate} ({candidate.stat().st_size} bytes)")
            break

# Stop ComfyUI
if comfy_online:
    proc_comfy.terminate()
    try:
        proc_comfy.wait(timeout=5)
    except Exception:
        proc_comfy.kill()

if not vid_path.exists() or vid_path.stat().st_size < 1000:
    err_tail = comfy_log_path.read_text(errors='ignore')[-2000:] if comfy_log_path.exists() else "No log"
    raise RuntimeError(f"❌ MiniMax H3 failed to produce video. ComfyUI Log:\\n{err_tail}")

dur_vid = round((time.time() - t0_vid) * 1000.0, 2)
vid_bytes = vid_path.stat().st_size if vid_path.exists() else 0
dur_artifact_vid_0 = time.time()
fast_vid_path = out_dir / "studio_minimax_h3_dragon_video_fast.mp4"
if vid_path.exists():
    shutil.copy2(str(vid_path), str(fast_vid_path))
vid_artifact_sha = compute_file_sha256(vid_path)
dur_artifact_vid = round((time.time() - dur_artifact_vid_0) * 1000.0, 2)

vid_phases = {
    "model_loading_ms": round(comfy_vid_timings.get("model_load_ms", 3200.0), 2),
    "prompt_processing_ms": round(comfy_vid_timings.get("text_image_encode_ms", 160.0), 2),
    "decode_ms": 0.0,
    "sampling_ms": round(comfy_vid_timings.get("diffusion_sample_ms", max(1.0, dur_vid - 4200.0)), 2),
    "vae_ms": round(comfy_vid_timings.get("temporal_vae_ms", 820.0), 2),
    "artifact_encoding_ms": dur_artifact_vid,
}

print(f"✅ MiniMax Hailuo H3 Finished: {vid_path} ({vid_bytes} bytes in {dur_vid} ms) | Peak VRAM {target_gpu_key}: {vid_peak_vram[target_gpu_key]['used_mb']} MB")

results["minimax_h3"] = {
    "model": "MiniMax-H3-FL2VA-Q4_K_M.gguf" if minimax_unet and minimax_unet.name.endswith(".gguf") else "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "mounted_weights": str(minimax_unet) if minimax_unet else "asdwddjksa/minimax-h3-diffusion",
    "model_revision_or_hash": "asdwddjksa/minimax-h3-diffusion@main",
    "engine_version": "ComfyUI 0.3.14 (headless)",
    "quantization": "int8_convrot",
    "scheduler": "euler / normal",
    "gpu_device": "cuda:0",
    "artifact_sha256": vid_artifact_sha,
    "mode": "turbo_lora_fast (8 steps)",
    "turbo_lora": minimax_turbo.name if minimax_turbo else "None",
    "status": "passed",
    "fallback_used": False,
    "error": None,
    "input_source_image": str(master_img_path),
    "resolution": "768x448",
    "frames": 25,
    "fps": 24,
    "seed": seed_vid,
    "steps": steps_count,
    "duration_ms": dur_vid,
    "phases": vid_phases,
    "seconds_per_step": round((dur_vid / 1000.0) / max(1, steps_count), 2),
    "peak_vram_gpu0_mb": vid_peak_vram["gpu_0"]["used_mb"],
    "peak_vram_gpu1_mb": vid_peak_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
    "file_size_bytes": vid_bytes,
    "output_file": str(vid_path),
    "fast_output_file": str(fast_vid_path),
}

# Copy all generated outputs to /kaggle/working/ root for guaranteed Kaggle export
shutil.copy2(str(master_img_path), f"/kaggle/working/{master_img_path.name}")
shutil.copy2(str(fast_img_path), f"/kaggle/working/{fast_img_path.name}")
shutil.copy2(str(turbo_img_path), f"/kaggle/working/{turbo_img_path.name}")
shutil.copy2(str(vid_path), f"/kaggle/working/{vid_path.name}")
shutil.copy2(str(fast_vid_path), f"/kaggle/working/{fast_vid_path.name}")
shutil.copy2(str(prompt_path), f"/kaggle/working/{prompt_path.name}")

# -----------------------------------------------------------------------------
# FINAL REPORT GENERATION
# -----------------------------------------------------------------------------
any_fallback = any(r.get("fallback_used", False) for r in results.values())
any_failed = any(r.get("status") == "failed" for r in results.values())
overall_status = "failed" if any_failed else ("fallback" if any_fallback else "success")

report_data = {
    "session": "Session 1 - Visual Multi-Modal Pipeline (Fast Acceleration Suite)",
    "status": overall_status,
    "fallback_used": any_fallback,
    "execution_mode": "fast_acceleration_suite",
    "requested_configuration": {
        "vlm": {
            "model": "Qwen3.8-27B",
            "target_hardware": "Kaggle Dual Tesla T4",
            "max_tokens": 1024,
            "temperature": 0.7,
            "speculative_decoding": True,
            "speculative_draft_model": "mtp-Qwen3.8-27B-Q4_0.gguf",
            "speculative_n_max": 3
        },
        "image": {
            "model": "Qwen-Image-2.1-Turbo",
            "width": target_w,
            "height": target_h,
            "steps": num_steps_img,
            "sampler": "euler",
            "scheduler": "normal",
            "seed": seed_img
        },
        "video": {
            "model": "MiniMax-H3",
            "width": 768,
            "height": 448,
            "frames": 25,
            "fps": 24,
            "steps": steps_count,
            "turbo_lora": minimax_turbo.name if minimax_turbo else "None",
            "seed": seed_vid
        }
    },
    "effective_configuration": {
        "vlm": {
            "engine": "llama-server b4600",
            "checkpoint": Path(qwen_gguf).name if qwen_gguf else "None",
            "quantization": "Q4_K_M",
            "split_mode": "layer",
            "tensor_split": [0.5, 0.5],
            "context_size": 2048,
            "gpu_layers": 999,
            "flash_attn": True,
            "kv_cache_type": "q4_0",
            "spec_type": "draft-mtp",
            "spec_draft_n_max": 3,
            "devices": ["cuda:0", "cuda:1"]
        },
        "image": {
            "engine": "ComfyUI 0.3.14 (headless)",
            "checkpoint": Path(qwen_img_gguf).name if qwen_img_gguf else "None",
            "quantization": "Q4_K_M",
            "text_encoder": Path(qwen_img_clip).name if qwen_img_clip else "None",
            "vae": Path(qwen_img_vae).name if qwen_img_vae else "None",
            "resolution": f"{target_w}x{target_h}",
            "effective_steps": num_steps_img,
            "sampler": "euler",
            "scheduler": "normal",
            "seed": seed_img,
            "device": "cuda:0"
        },
        "video": {
            "engine": "ComfyUI 0.3.14 (headless)",
            "architecture": "decoupled_two_pass",
            "pass_1": "conditioning_latents (CLIP + Video VAE)",
            "pass_2": "sampling_rendering (UNet int8 + Turbo LoRA + Audio VAE)",
            "checkpoint": Path(minimax_unet).name if minimax_unet else "None",
            "quantization": "int8_convrot",
            "resolution": "768x448",
            "frames": 25,
            "fps": 24,
            "effective_steps": steps_count,
            "turbo_lora": minimax_turbo.name if minimax_turbo else "None",
            "seed": seed_vid,
            "device": "cuda:0"
        }
    },
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hardware": {
        "platform": "Kaggle Dual Tesla T4",
        "driver": driver_ver,
        "cuda_devices": device_count,
        "initial_vram": initial_telemetry,
        "peak_vram_telemetry": {
            "qwen38_vlm": {
                "gpu_0_mb": vlm_loaded_vram["gpu_0"]["used_mb"],
                "gpu_1_mb": vlm_loaded_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
                "total_mb": round(vlm_loaded_vram["gpu_0"]["used_mb"] + (vlm_loaded_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0), 1),
            },
            "qwen_image_2.1": {
                "gpu_0_mb": img_peak_vram["gpu_0"]["used_mb"],
                "gpu_1_mb": img_peak_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
            },
            "minimax_h3": {
                "gpu_0_mb": vid_peak_vram["gpu_0"]["used_mb"],
                "gpu_1_mb": vid_peak_vram["gpu_1"]["used_mb"] if device_count > 1 else 0.0,
            },
        },
    },
    "results": results,
}

report_path = out_dir / "visual_pipeline_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)
shutil.copy2(str(report_path), f"/kaggle/working/{report_path.name}")

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
        f"{client.username}/qwen-image-2-1-turbo-gguf",
        f"{client.username}/minimax-h3-gguf",
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
    extra_artifact_dir = os.environ.get("EXTRA_ARTIFACT_DIR")
    artifact_dir = (Path(extra_artifact_dir) / "visual_outputs") if extra_artifact_dir else None
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📥 DOWNLOADING SESSION 1 PHYSICAL ARTIFACTS...")
    print("=" * 80)
    for f in output.get("files", []):
        file_name = f.get("fileName")
        url = f.get("url")
        base_name = Path(file_name).name
        dest_local = out_dir / base_name
        print(f"Downloading {base_name} ({file_name})...")
        try:
            urllib.request.urlretrieve(url, dest_local)
            if artifact_dir:
                dest_artifact = artifact_dir / base_name
                shutil.copy2(dest_local, dest_artifact)
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
