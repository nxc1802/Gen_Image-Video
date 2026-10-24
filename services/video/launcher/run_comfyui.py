"""
Launch the native ComfyUI video worker on GPU 1.
"""
import os
import sys
from pathlib import Path


def main() -> None:
    comfy_dir = Path(os.environ.get("COMFYUI_DIR", "ComfyUI")).expanduser().resolve()
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        raise RuntimeError(f"ComfyUI main.py not found at {main_py}. Set COMFYUI_DIR.")

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("VIDEO_CUDA_VISIBLE_DEVICES", "1")
    port = os.environ.get("VIDEO_SERVICE_PORT", "8189")
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
