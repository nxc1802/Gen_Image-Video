"""Activate only the hash-checked Qwen TTS FP16 native source overlays."""

import importlib.abc
import importlib.util
from pathlib import Path
import sys

MODULES = {
    "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker": "qwen3_tts_talker.py",
    "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder": "prompt_embeds_builder.py",
}


class QwenTTSOverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES:
            return None
        selected = Path(__file__).parent / "overlays" / MODULES[fullname]
        return importlib.util.spec_from_file_location(fullname, selected)


def register():
    for module_name, filename in MODULES.items():
        loaded = sys.modules.get(module_name)
        if (
            loaded is not None
            and Path(loaded.__file__).resolve()
            != (Path(__file__).parent / "overlays" / filename).resolve()
        ):
            raise RuntimeError("Qwen TTS imported before its explicit FP16 adapter extension")
    if not any(isinstance(finder, QwenTTSOverlayFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, QwenTTSOverlayFinder())
