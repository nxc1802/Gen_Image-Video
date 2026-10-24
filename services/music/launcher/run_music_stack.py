"""
Launch ACE-Step 1.5 native REST API and the thin Factory music adapter.
"""
import os
import shutil
import signal
import subprocess
import sys
import time


def main() -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to launch the ACE-Step native REST service.")
    ace_dir = os.environ.get("ACESTEP_DIR", "").strip()
    if not ace_dir:
        raise RuntimeError("ACESTEP_DIR must point to an installed ACE-Step 1.5 checkout.")

    api_key = os.environ.get("FACTORY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FACTORY_API_KEY is required.")

    ace_port = os.environ.get("ACESTEP_INTERNAL_PORT", "8008")
    gateway_port = os.environ.get("MUSIC_SERVICE_PORT", "8001")
    ace_env = os.environ.copy()
    ace_env["PORT"] = ace_port
    ace_env["SERVER_NAME"] = "127.0.0.1"
    ace_env["ACESTEP_API_KEY"] = api_key

    ace = subprocess.Popen(
        ["uv", "run", "acestep-api", "--port", ace_port, "--server-name", "127.0.0.1", "--api-key", api_key],
        cwd=ace_dir,
        env=ace_env,
    )

    gateway_env = os.environ.copy()
    gateway_env["ACESTEP_API_URL"] = f"http://127.0.0.1:{ace_port}"
    gateway = subprocess.Popen(
        [sys.executable, "-m", "services.music.server", "--host", "0.0.0.0", "--port", gateway_port],
        env=gateway_env,
    )

    children = [gateway, ace]

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
    finally:
        shutdown()


if __name__ == "__main__":
    main()
