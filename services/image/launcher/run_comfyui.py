"""
Launch the native ComfyUI image worker on GPU 0.

ComfyUI remains the public/native API surface for image generation. Factory
orchestration submits versioned API-format workflows to /prompt and consumes
progress over /ws.
"""
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    comfy_dir = Path(os.environ.get("COMFYUI_DIR", "ComfyUI")).expanduser().resolve()
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        raise RuntimeError(f"ComfyUI main.py not found at {main_py}. Set COMFYUI_DIR.")
    if shutil.which(sys.executable) is None:
        raise RuntimeError("Python executable is unavailable.")

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("IMAGE_CUDA_VISIBLE_DEVICES", "0")
    port = os.environ.get("IMAGE_SERVICE_PORT", "8188")
    cmd = [
        sys.executable,
        str(main_py),
        "--listen", "0.0.0.0",
        "--port", port,
        "--cuda-device", "0",
    ]
    os.chdir(comfy_dir)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
