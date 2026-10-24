"""
Launch ACE-Step 1.5 as the native public Music API.

Factory does not wrap ACE-Step's generation endpoints in a second custom API.
The native API exposes /release_task, /query_result, /v1/audio and
/v1/chat/completions.
"""
import os
import shutil


def main() -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to launch ACE-Step.")
    ace_dir = os.environ.get("ACESTEP_DIR", "").strip()
    if not ace_dir:
        raise RuntimeError("ACESTEP_DIR must point to an installed ACE-Step 1.5 checkout.")
    api_key = os.environ.get("FACTORY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FACTORY_API_KEY is required.")

    port = os.environ.get("MUSIC_SERVICE_PORT", "8008")
    os.chdir(ace_dir)
    cmd = [
        "uv", "run", "acestep-api",
        "--port", port,
        "--server-name", "0.0.0.0",
        "--api-key", api_key,
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
