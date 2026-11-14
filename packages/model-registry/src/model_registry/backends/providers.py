"""Provider availability; missing SDKs never select a different provider."""
import importlib.util
import shutil
import os
from pathlib import Path
from .process import ProcessBackend


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class VLLMBackend(ProcessBackend):
    executable_name = "vllm"
    @classmethod
    def available(cls):
        return has_module("vllm") and shutil.which("vllm") is not None


class LlamaCppBackend(ProcessBackend):
    executable_name = "llama-server"
    @classmethod
    def available(cls):
        return shutil.which(cls.executable_name) is not None


class ComfyUIBackend(ProcessBackend):
    @classmethod
    def available(cls):
        value = os.environ.get("COMFYUI_DIR")
        return has_module("torch") and bool(value and (Path(value) / "main.py").is_file())

    @classmethod
    def available_for(cls, binding, config_path):
        selected = binding.backend_options.source_dir
        if selected is None:
            return cls.available()
        path = selected if selected.is_absolute() else Path(config_path).parent / selected
        return has_module("torch") and (path / "main.py").is_file()


class OmniBackend(VLLMBackend):
    @classmethod
    def available(cls):
        return super().available() and has_module("vllm_omni")


class ASRBackend(ProcessBackend):
    @classmethod
    def available(cls):
        return has_module("qwen_asr") and has_module("torch") and has_module("uvicorn")


class DiffusersBackend(ProcessBackend):
    @classmethod
    def available(cls):
        return has_module("diffusers") and has_module("torch") and has_module("uvicorn")


class ACEStepBackend(ProcessBackend):
    @classmethod
    def available(cls):
        return has_module("acestep") and has_module("torch") and has_module("uvicorn")


class YuEBackend(ProcessBackend):
    @classmethod
    def available(cls):
        return has_module("yue2") and has_module("torch") and has_module("uvicorn")
