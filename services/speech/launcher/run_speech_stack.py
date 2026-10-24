"""
Launch the real Factory v4 speech stack:
- Qwen3-ASR served by native vLLM
- Qwen3-TTS served by native vLLM-Omni
- thin Factory speech gateway on :8288

The gateway only adapts contracts/auth; it never performs fake inference.
"""
import os
import shutil
import signal
import subprocess
import sys
import time


def _require(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' is not installed or not on PATH.")


def main() -> None:
    _require("vllm")
    api_key = os.environ.get("FACTORY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FACTORY_API_KEY is required.")

    asr_port = os.environ.get("QWEN_ASR_PORT", "8290")
    tts_port = os.environ.get("QWEN_TTS_PORT", "8291")
    gateway_port = os.environ.get("SPEECH_SERVICE_PORT", "8288")
    asr_model = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
    tts_model = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    common_env = os.environ.copy()
    common_env["VLLM_MEDIA_URL_ALLOW_REDIRECTS"] = "0"

    asr_env = common_env.copy()
    asr_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("ASR_CUDA_VISIBLE_DEVICES", "0")
    tts_env = common_env.copy()
    tts_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("TTS_CUDA_VISIBLE_DEVICES", "1")

    asr = subprocess.Popen([
        "vllm", "serve", asr_model,
        "--host", "127.0.0.1",
        "--port", asr_port,
        "--served-model-name", "qwen3-asr",
        "--api-key", api_key,
    ], env=asr_env)

    tts = subprocess.Popen([
        "vllm", "serve", tts_model,
        "--omni",
        "--host", "127.0.0.1",
        "--port", tts_port,
        "--api-key", api_key,
    ], env=tts_env)

    gateway_env = common_env.copy()
    gateway_env["QWEN_ASR_URL"] = f"http://127.0.0.1:{asr_port}"
    gateway_env["QWEN_TTS_URL"] = f"http://127.0.0.1:{tts_port}"

    gateway = subprocess.Popen([
        sys.executable, "-m", "services.speech.server",
        "--host", "0.0.0.0",
        "--port", gateway_port,
    ], env=gateway_env)

    children = [gateway, asr, tts]

    def shutdown(*_args):
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 10
        for proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    proc.kill()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        code = gateway.wait()
        if code != 0:
            raise SystemExit(code)
        if asr.poll() not in (None, 0) or tts.poll() not in (None, 0):
            raise RuntimeError("A native speech backend exited unexpectedly.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
