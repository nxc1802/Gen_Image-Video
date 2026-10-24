"""
Run ACE-Step 1.5 native REST API plus the Factory queue-worker sidecar.

Public traffic still reaches ACE-Step directly on port 8008; the sidecar only
bridges durable Factory jobs into ACE-Step's native async queue.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


def _wait_ready(proc: subprocess.Popen, port: int, timeout: float = 240.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("ACE-Step exited before becoming ready.")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("ACE-Step did not become ready before startup timeout.")


def main() -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to launch ACE-Step.")
    ace_dir = os.environ.get("ACESTEP_DIR", "").strip()
    if not ace_dir:
        raise RuntimeError("ACESTEP_DIR must point to an installed ACE-Step 1.5 checkout.")
    api_key = os.environ.get("FACTORY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FACTORY_API_KEY is required.")

    port = int(os.environ.get("MUSIC_SERVICE_PORT", "8008"))
    ace = subprocess.Popen(
        [
            "uv", "run", "acestep-api",
            "--port", str(port),
            "--server-name", "0.0.0.0",
            "--api-key", api_key,
        ],
        cwd=ace_dir,
        env=os.environ.copy(),
    )
    worker = None

    def stop(*_args) -> None:
        for proc in (worker, ace):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for proc in (worker, ace):
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        _wait_ready(ace, port)
        worker_env = os.environ.copy()
        worker_env["ACESTEP_API_URL"] = f"http://127.0.0.1:{port}"
        worker = subprocess.Popen([sys.executable, "-m", "services.music.worker"], env=worker_env)
        while True:
            if ace.poll() is not None:
                raise RuntimeError("ACE-Step exited unexpectedly.")
            if worker.poll() is not None:
                raise RuntimeError("Factory music worker exited unexpectedly.")
            time.sleep(1.0)
    finally:
        stop()


if __name__ == "__main__":
    main()
