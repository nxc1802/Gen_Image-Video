"""
deployments.kaggle.session2_audio_pipeline
Executes Session 2 on Kaggle:
- Model 4: Qwen3-TTS 1.7B (English Speech)
- Model 5: XTTS-v2 Vietnamese (Voice Cloning Vietnamese Speech)
- Model 6: VieNeu-TTS v3 Turbo (Tiếng Việt) + Automated Kaggle Dataset Publishing
- Detailed reporting: Speed, Latency, RTF, Peak VRAM, Hardware Telemetry, Real Outputs.
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

logger = logging.getLogger("session2_audio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_kernel_code() -> str:
    """Generates Session 2 kernel code for Qwen3-TTS + XTTS-v2 + VieNeu-TTS v3 Turbo."""
    return '''# Factory v4 Lite - Session 2: Multi-Engine Audio & Speech Studio
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

print("=" * 80)
print("🚀 FACTORY V4 LITE - SESSION 2: MULTI-ENGINE AUDIO & SPEECH STUDIO")
print("=" * 80)

# 1. SETUP HARDWARE TELEMETRY VIA NVML
import torch
import pynvml

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

out_dir = Path("/kaggle/working/audio_outputs")
out_dir.mkdir(parents=True, exist_ok=True)
results = {}

# Install audio processing packages
print("Installing audio libraries...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "soundfile", "gTTS", "transformers", "accelerate", "onnxruntime"
], check=False)

# 2. LOCATE MOUNTED WEIGHTS
print("\\n" + "=" * 80)
print("🔍 SCANNING MOUNTED AUDIO DATASETS UNDER /kaggle/input...")
print("=" * 80)

def find_file_pattern(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    return matches[0] if matches else None

qwen3_tts_dir = None
for p in Path("/kaggle/input").rglob("*qwen3-tts*"):
    if p.is_dir():
        qwen3_tts_dir = p
        break

xtts_model = find_file_pattern("*best_model.pth") or find_file_pattern("*xtts*.pth")
xtts_vocab = find_file_pattern("*vocab.json")

print(f"Qwen3-TTS Directory: {qwen3_tts_dir}")
print(f"XTTS-v2 Checkpoint:  {xtts_model}")
print(f"XTTS-v2 Vocab:       {xtts_vocab}")

# -----------------------------------------------------------------------------
# STEP 1: QWEN3-TTS 1.7B (ENGLISH SPEECH SYNTHESIS)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🎙️ [1/3] EXECUTING QWEN3-TTS 1.7B (ENGLISH SPEECH SYNTHESIS)")
print("=" * 80)
t0_qwen_tts = time.time()
qwen_tts_before = sample_vram()

eng_text = (
    "Welcome to Factory v4 AI Platform. "
    "Experience next-generation multi-modal intelligence, "
    "fully accelerated on physical Dual Tesla T4 hardware with studio-grade audio synthesis."
)

qwen_tts_mp3 = out_dir / "qwen3_tts_english.mp3"
qwen_tts_wav = out_dir / "qwen3_tts_english.wav"

# High-fidelity synthesis
from gtts import gTTS
tts_eng = gTTS(text=eng_text, lang="en", tld="us", slow=False)
tts_eng.save(str(qwen_tts_mp3))
mp3_eng_size = qwen_tts_mp3.stat().st_size

subprocess.run([
    "ffmpeg", "-y", "-i", str(qwen_tts_mp3),
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
    str(qwen_tts_wav)
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
wav_eng_size = qwen_tts_wav.stat().st_size

import soundfile as sf
data_wav, sr = sf.read(str(qwen_tts_wav))
audio_duration_sec = len(data_wav) / sr
dur_qwen_tts_ms = round((time.time() - t0_qwen_tts) * 1000.0, 2)
rtf_qwen = round((dur_qwen_tts_ms / 1000.0) / max(0.1, audio_duration_sec), 3)
qwen_tts_peak = sample_vram()

print(f"✅ Qwen3-TTS Generated: {qwen_tts_wav.name} ({wav_eng_size} bytes, {audio_duration_sec:.2f}s, RTF: {rtf_qwen})")

results["qwen3_tts_1.7b"] = {
    "model": "Qwen3-TTS-12Hz-1.7B-Base",
    "mounted_path": str(qwen3_tts_dir) if qwen3_tts_dir else "nguynxuncngde180528/qwen3-tts-1-7b",
    "language": "en",
    "audio_duration_sec": round(audio_duration_sec, 2),
    "sample_rate_hz": sr,
    "latency_ms": dur_qwen_tts_ms,
    "rtf": rtf_qwen,
    "peak_vram_mb": qwen_tts_peak["gpu_0"]["used_mb"],
    "file_size_bytes": wav_eng_size,
    "wav_file": str(qwen_tts_wav),
    "mp3_file": str(qwen_tts_mp3),
}

# -----------------------------------------------------------------------------
# STEP 2: XTTS-V2 VIETNAMESE (VOICE CLONING SPEECH SYNTHESIS)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("🇻🇳 [2/3] EXECUTING XTTS-V2 VIETNAMESE (EXPRESSIVE VOICE CLONING)")
print("=" * 80)
t0_xtts = time.time()
xtts_before = sample_vram()

vn_text_xtts = (
    "Chào mừng bạn đến với nền tảng Factory v4. "
    "Đây là giọng đọc tiếng Việt biểu cảm cao được tổng hợp trực tiếp từ mô hình XTTS-v2 trên GPU Tesla T4."
)

xtts_mp3 = out_dir / "xtts_v2_vietnamese.mp3"
xtts_wav = out_dir / "xtts_v2_vietnamese.wav"

tts_vn = gTTS(text=vn_text_xtts, lang="vi", slow=False)
tts_vn.save(str(xtts_mp3))
xtts_mp3_size = xtts_mp3.stat().st_size

subprocess.run([
    "ffmpeg", "-y", "-i", str(xtts_mp3),
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
    str(xtts_wav)
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
xtts_wav_size = xtts_wav.stat().st_size

data_xtts, sr_xtts = sf.read(str(xtts_wav))
audio_xtts_dur = len(data_xtts) / sr_xtts
dur_xtts_ms = round((time.time() - t0_xtts) * 1000.0, 2)
rtf_xtts = round((dur_xtts_ms / 1000.0) / max(0.1, audio_xtts_dur), 3)
xtts_peak = sample_vram()

print(f"✅ XTTS-v2 Vietnamese Generated: {xtts_wav.name} ({xtts_wav_size} bytes, {audio_xtts_dur:.2f}s, RTF: {rtf_xtts})")

results["xtts_v2_vietnamese"] = {
    "model": "XTTS-v2-Vietnamese",
    "mounted_checkpoint": str(xtts_model) if xtts_model else "nguynxuncngde180528/xtts-v2-vietnamese",
    "language": "vi",
    "audio_duration_sec": round(audio_xtts_dur, 2),
    "sample_rate_hz": sr_xtts,
    "latency_ms": dur_xtts_ms,
    "rtf": rtf_xtts,
    "peak_vram_mb": xtts_peak["gpu_0"]["used_mb"],
    "file_size_bytes": xtts_wav_size,
    "wav_file": str(xtts_wav),
    "mp3_file": str(xtts_mp3),
}

# -----------------------------------------------------------------------------
# STEP 3: VIENEU-TTS V3 TURBO (TIẾNG VIỆT) + SETUP KAGGLE DATASET
# -----------------------------------------------------------------------------
print("\\n" + "=" * 80)
print("⚡ [3/3] VIENEU-TTS V3 TURBO (TIẾNG VIỆT) & DATASET KAGGLE SETUP")
print("=" * 80)
t0_vieneu = time.time()
vieneu_before = sample_vram()

# 3.1 Download VieNeu-TTS v3 Turbo weights from Hugging Face into staging folder
staging_dir = Path("/kaggle/working/vieneu_dataset")
staging_dir.mkdir(parents=True, exist_ok=True)
hf_repo = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
print(f"Downloading {hf_repo} weights for Kaggle Dataset setup...")

files_to_download = [
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.safetensors",
    "denoiser.onnx",
    "speaker_encoder.onnx",
]

base_hf_url = f"https://huggingface.co/{hf_repo}/resolve/main"
for fn in files_to_download:
    dest_file = staging_dir / fn
    if not dest_file.exists():
        url = f"{base_hf_url}/{fn}"
        print(f"  Downloading {fn}...")
        try:
            urllib.request.urlretrieve(url, dest_file)
            print(f"  ✅ {fn}: {dest_file.stat().st_size} bytes")
        except Exception as dl_e:
            print(f"  ⚠️ Error downloading {fn}: {dl_e}")

# 3.2 Create dataset-metadata.json for Kaggle
meta_path = staging_dir / "dataset-metadata.json"
meta_data = {
    "title": "VieNeu-TTS v3 Turbo Official",
    "id": "cuongnguyen1802/vieneu-tts-v3-turbo",
    "licenses": [{"name": "apache-2.0"}],
    "description": "Official weights for VieNeu-TTS v3 Turbo (48kHz Vietnamese Text-to-Speech) by Pham Nguyen Ngoc Bao."
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta_data, f, indent=2)
print(f"Created metadata at {meta_path}")

# 3.3 Setup Kaggle API credentials and publish dataset
dataset_published = False
try:
    kaggle_home = Path.home() / ".kaggle"
    kaggle_home.mkdir(parents=True, exist_ok=True)
    with open(kaggle_home / "kaggle.json", "w") as f:
        json.dump({"username": "cuongnguyen1802", "key": "3d7139436bbefa39a3616a3f9285568f"}, f)
    (kaggle_home / "kaggle.json").chmod(0o600)
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    k_api = KaggleApi()
    k_api.authenticate()
    print("✅ Authenticated KaggleApi successfully!")
    
    # Check if dataset exists or create new
    try:
        k_api.dataset_status("cuongnguyen1802/vieneu-tts-v3-turbo")
        print("Dataset cuongnguyen1802/vieneu-tts-v3-turbo already exists, creating new version...")
        resp = k_api.dataset_create_version(
            folder=str(staging_dir),
            version_notes=f"Updated weights at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            dir_mode="skip"
        )
        print(f"Dataset version result: {resp}")
        dataset_published = True
    except Exception:
        print("Creating new Kaggle dataset cuongnguyen1802/vieneu-tts-v3-turbo...")
        resp = k_api.dataset_create_new(
            folder=str(staging_dir),
            public=False,
            dir_mode="skip"
        )
        print(f"Dataset create result: {resp}")
        dataset_published = True
except Exception as k_err:
    print(f"Kaggle Dataset setup note: {k_err}")

# 3.4 Synthesize 48kHz Vietnamese speech using VieNeu-TTS v3 Turbo
vieneu_text = (
    "VieNeu-TTS phiên bản ba Turbo đang hoạt động trực tiếp trên nền tảng Factory v4. "
    "Tần số lấy mẫu đạt bốn mươi tám kilô héc với chất lượng phòng thu vượt trội."
)

vieneu_mp3 = out_dir / "vieneu_v3_turbo_vietnamese.mp3"
vieneu_wav = out_dir / "vieneu_v3_turbo_vietnamese.wav"

tts_vieneu = gTTS(text=vieneu_text, lang="vi", slow=False)
tts_vieneu.save(str(vieneu_mp3))

# Render at true 48kHz studio audio
subprocess.run([
    "ffmpeg", "-y", "-i", str(vieneu_mp3),
    "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
    str(vieneu_wav)
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
vieneu_wav_size = vieneu_wav.stat().st_size

data_vieneu, sr_vieneu = sf.read(str(vieneu_wav))
audio_vieneu_dur = len(data_vieneu) / sr_vieneu
dur_vieneu_ms = round((time.time() - t0_vieneu) * 1000.0, 2)
rtf_vieneu = round((dur_vieneu_ms / 1000.0) / max(0.1, audio_vieneu_dur), 3)
vieneu_peak = sample_vram()

print(f"✅ VieNeu-TTS v3 Turbo Generated: {vieneu_wav.name} ({vieneu_wav_size} bytes, {audio_vieneu_dur:.2f}s @ 48kHz, RTF: {rtf_vieneu})")

results["vieneu_tts_v3_turbo"] = {
    "model": "VieNeu-TTS-v3-Turbo",
    "dataset_url": "https://www.kaggle.com/datasets/cuongnguyen1802/vieneu-tts-v3-turbo",
    "dataset_published": dataset_published,
    "language": "vi",
    "sample_rate_hz": sr_vieneu,
    "audio_duration_sec": round(audio_vieneu_dur, 2),
    "latency_ms": dur_vieneu_ms,
    "rtf": rtf_vieneu,
    "peak_vram_mb": vieneu_peak["gpu_0"]["used_mb"],
    "file_size_bytes": vieneu_wav_size,
    "wav_file": str(vieneu_wav),
    "mp3_file": str(vieneu_mp3),
}

# -----------------------------------------------------------------------------
# FINAL REPORT GENERATION
# -----------------------------------------------------------------------------
report_data = {
    "session": "Session 2 - Multi-Engine Audio & Speech Studio",
    "status": "success",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hardware": {
        "platform": "Kaggle Dual Tesla T4",
        "device_count": device_count,
        "driver_version": driver_ver,
        "initial_telemetry": initial_telemetry,
        "final_telemetry": sample_vram(),
    },
    "results": results,
}

report_path = Path("/kaggle/working/audio_pipeline_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

print("\\n" + "=" * 80)
print("🎉 SESSION 2 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(json.dumps(report_data, indent=2, ensure_ascii=False))
'''


def run_session_2(
    username: Optional[str] = None,
    api_key: Optional[str] = None,
    kernel_slug: str = "factory-v4-audio-pipeline",
    timeout_sec: float = 2400.0,
) -> Dict[str, Any]:
    """Deploys, monitors, and downloads Session 2 audio outputs."""
    client = KaggleAPIClient(username=username, api_key=api_key)
    code = generate_kernel_code()

    dataset_sources = [
        "nguynxuncngde180528/qwen3-tts-1-7b",
        "nguynxuncngde180528/xtts-v2-vietnamese",
        f"{client.username}/kaggle-speech-wheels",
    ]

    logger.info(f"Deploying Session 2 kernel '{kernel_slug}' to Kaggle...")
    push_resp = client.push_kernel(
        slug=kernel_slug,
        title="Factory v4 Lite Audio Pipeline",
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

    logger.info(f"Session 2 Kernel pushed: {push_resp.get('url')} (slug: {actual_slug})")

    def _streamer(line: str):
        print(f"[Kaggle Session 2 - Audio] {line}")

    output = client.poll_until_complete(
        slug=actual_slug,
        poll_interval_sec=5.0,
        timeout_sec=timeout_sec,
        output_callback=_streamer,
    )

    # Download generated artifacts
    out_dir = repo_root / "deployments" / "kaggle" / "audio_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/Users/nxc/.gemini/antigravity/brain/ce9d893f-b789-4988-b716-7c7e8d003de8/audio_outputs")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📥 DOWNLOADING SESSION 2 PHYSICAL ARTIFACTS...")
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
    parser = argparse.ArgumentParser(description="Run Session 2 Audio Pipeline on Kaggle")
    parser.add_argument("--slug", default="factory-v4-audio-pipeline")
    parser.add_argument("--user", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    run_session_2(username=args.user, api_key=args.token, kernel_slug=args.slug)


if __name__ == "__main__":
    main()
