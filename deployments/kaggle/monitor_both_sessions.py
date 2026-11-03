"""
deployments.kaggle.monitor_both_sessions
Pushes and monitors both Session 1 (Visual Pipeline) and Session 2 (Audio Pipeline) concurrently on Kaggle.
Streams logs in real-time, downloads all physical artifacts upon completion, and validates full output reports.
"""

import json
from pathlib import Path
import sys
import time
import urllib.request

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from deployments.kaggle.runner import KaggleAPIClient
from deployments.kaggle.session1_visual_pipeline import generate_kernel_code as gen_visual_code
from deployments.kaggle.session2_audio_pipeline import generate_kernel_code as gen_audio_code

client = KaggleAPIClient(username="cuongnguyen1802", api_key="KGAT_3d7139436bbefa39a3616a3f9285568f")

sessions = {
    "visual": {
        "slug": "factory-v4-lite-visual-pipeline",
        "title": "Factory v4 Lite Visual Pipeline",
        "name": "Session 1 [Visual: Qwen3.8 + Qwen-Image + MiniMax]",
        "code_fn": gen_visual_code,
        "dataset_sources": [
            f"{client.username}/qwen38-27b-gguf",
            "nguynxuncngde180528/qwen-image-2-1-gguf",
            f"{client.username}/minimax-h3-gguf",
            "asdwddjksa/minimax-h3-diffusion",
            "asdwddjksa/minimax-h3-text-encoder",
            "asdwddjksa/minimax-h3-assets",
            f"{client.username}/kaggle-vlm-wheels",
            f"{client.username}/kaggle-image-video-wheels",
        ],
        "out_dir": repo_root / "deployments" / "kaggle" / "visual_outputs",
        "artifact_dir": Path("/Users/nxc/.gemini/antigravity/brain/ce9d893f-b789-4988-b716-7c7e8d003de8/visual_outputs"),
        "done": False,
        "last_log_line": 0,
        "result": None,
    },
    "audio": {
        "slug": "factory-v4-lite-audio-pipeline",
        "title": "Factory v4 Lite Audio Pipeline",
        "name": "Session 2 [Audio: Qwen3-TTS + XTTS-v2 + VieNeu-TTS]",
        "code_fn": gen_audio_code,
        "dataset_sources": [
            "nguynxuncngde180528/qwen3-tts-1-7b",
            "nguynxuncngde180528/xtts-v2-vietnamese",
            f"{client.username}/vieneu-tts-v3-turbo",
            f"{client.username}/kaggle-speech-wheels",
        ],
        "out_dir": repo_root / "deployments" / "kaggle" / "audio_outputs",
        "artifact_dir": Path("/Users/nxc/.gemini/antigravity/brain/ce9d893f-b789-4988-b716-7c7e8d003de8/audio_outputs"),
        "done": False,
        "last_log_line": 0,
        "result": None,
    },
}

for s_info in sessions.values():
    s_info["out_dir"].mkdir(parents=True, exist_ok=True)
    s_info["artifact_dir"].mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🚀 DEPLOYING CONCURRENT KAGGLE SESSIONS (VISUAL & AUDIO PIPELINES)")
print("=" * 80)

# 1. PUSH BOTH KERNELS
for s_key, s_info in sessions.items():
    slug = s_info["slug"]
    print(f"Deploying {s_info['name']} (slug: {slug})...")
    code_content = s_info["code_fn"]()
    push_resp = client.push_kernel(
        slug=slug,
        title=s_info["title"],
        code_content=code_content,
        is_private=True,
        enable_gpu=True,
        enable_internet=True,
        kernel_type="script",
        language="python",
        dataset_sources=s_info["dataset_sources"],
    )
    print(f"✅ {s_info['name']} pushed: {push_resp.get('url')} (ref: {push_resp.get('ref')})")

print("\n" + "=" * 80)
print("📡 MONITORING BOTH SESSIONS RUNNING CONCURRENTLY ON KAGGLE")
print("=" * 80)

start_time = time.time()
poll_count = 0

while not all(s["done"] for s in sessions.values()):
    poll_count += 1
    for s_key, s_info in sessions.items():
        if s_info["done"]:
            continue

        slug = s_info["slug"]
        try:
            status_resp = client.get_kernel_status(slug)
            status = (status_resp.get("status") or "unknown").lower()
            failure_msg = status_resp.get("failureMessage")

            out_resp = client.get_kernel_output(slug)
            log_text = out_resp.get("log", "")
            if log_text:
                lines = log_text.splitlines()
                if len(lines) > s_info["last_log_line"]:
                    for l in lines[s_info["last_log_line"]:]:
                        print(f"[{s_info['name']}] {l}")
                    s_info["last_log_line"] = len(lines)

            if status == "complete":
                print(f"\n🎉 {s_info['name']} COMPLETED SUCCESSFULLY!")
                s_info["done"] = True
                s_info["result"] = out_resp

                # Download all physical artifacts
                for f in out_resp.get("files", []):
                    fn = f.get("fileName")
                    url = f.get("url")
                    base = Path(fn).name
                    dest1 = s_info["out_dir"] / base
                    dest2 = s_info["artifact_dir"] / base
                    print(f"[{s_info['name']}] Downloading {base}...")
                    try:
                        urllib.request.urlretrieve(url, dest1)
                        urllib.request.urlretrieve(url, dest2)
                        print(f"[{s_info['name']}] ✅ Saved: {dest1} ({dest1.stat().st_size} bytes)")
                    except Exception as err:
                        print(f"[{s_info['name']}] ⚠️ Error downloading {base}: {err}")

            elif status in ("error", "failed"):
                print(f"\n❌ {s_info['name']} FAILED with message: {failure_msg}")
                s_info["done"] = True
                s_info["result"] = {"error": failure_msg}

        except Exception as e:
            if poll_count % 6 == 0:
                print(f"[{s_info['name']}] Polling note: {e}")

    time.sleep(6)

print("\n" + "=" * 80)
print("🏁 ALL KAGGLE SESSIONS HAVE CONCLUDED!")
print(f"Total time elapsed: {round(time.time() - start_time, 1)} seconds.")
print("=" * 80)
