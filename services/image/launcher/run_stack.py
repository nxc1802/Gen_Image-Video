"""Run native ComfyUI plus the Factory image queue-worker sidecar."""
import signal
import subprocess
import sys
import time
import urllib.request

PORT = int(__import__("os").environ.get("IMAGE_COMFYUI_PORT", "8188"))

def _wait_ready(proc: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{PORT}/system_stats"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("ComfyUI exited before becoming ready.")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("ComfyUI did not become ready before startup timeout.")

def main() -> None:
    comfy = subprocess.Popen([sys.executable, "-m", "services.image.launcher.run_comfyui"])
    worker = None

    def stop(*_args) -> None:
        for proc in (worker, comfy):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for proc in (worker, comfy):
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        _wait_ready(comfy)
        worker = subprocess.Popen([sys.executable, "-m", "services.image.launcher.run_worker"])
        while True:
            if comfy.poll() is not None:
                raise RuntimeError("ComfyUI exited unexpectedly.")
            if worker.poll() is not None:
                raise RuntimeError("Factory image worker exited unexpectedly.")
            time.sleep(1.0)
    finally:
        stop()

if __name__ == "__main__":
    main()
