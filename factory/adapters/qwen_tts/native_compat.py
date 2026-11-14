"""Stage precise FP16 fixes for the pinned native Omni 0.32.0 Qwen TTS code."""

import hashlib
from importlib.util import find_spec
import os
from pathlib import Path

SOURCES = {
    "prompt_embeds_builder.py": {
        "sha256": "b462c400bd9a2244e73faed2f756e891008e431ccc2a9790fc4dc8ba51310eb6",
        "replacements": [
            (
                "self._embedding_dtype = torch.bfloat16",
                "self._embedding_dtype = tts_pad_embed.dtype",
                1,
            )
        ],
    },
    "qwen3_tts_talker.py": {
        "sha256": "b6b0dbdd790643657af022843d37266b437068a18667239468640dde9072657c",
        "replacements": [
            (
                'model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)',
                "model_dtype = vllm_config.model_config.dtype",
                1,
            ),
            ("self._embedding_dtype = torch.bfloat16", "self._embedding_dtype = model_dtype", 1),
            (
                "self.encoder.to(dtype=torch.bfloat16)",
                "self.encoder.to(dtype=self._embedding_dtype)",
                1,
            ),
            (
                ".to(device=device, dtype=torch.bfloat16)",
                ".to(device=device, dtype=self._embedding_dtype)",
                2,
            ),
            (
                "StreamingCodecDecoder(holder.decoder, num_slots=slots, dtype=torch.bfloat16)",
                "StreamingCodecDecoder(holder.decoder, num_slots=slots, dtype=self._embedding_dtype)",
                1,
            ),
        ],
    },
}


def stage_environment(backend):
    package = find_spec("vllm_omni")
    if package is None or package.origin is None:
        raise RuntimeError("Explicit Qwen TTS native package is unavailable")
    source_root = Path(package.origin).parent / "model_executor/models/qwen3_tts"
    root = backend.directory / "native_plugins"
    (root / "overlays").mkdir(parents=True, exist_ok=True)
    originals = root / "upstream"
    originals.mkdir(exist_ok=True)
    for name, spec in SOURCES.items():
        raw = (source_root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise RuntimeError(
                "Qwen TTS native source differs from the explicitly supported backend revision"
            )
        (originals / name).write_bytes(raw)
        patched = raw.decode()
        for before, after, count in spec["replacements"]:
            if patched.count(before) != count:
                raise RuntimeError("Qwen TTS FP16 native fix no longer matches the checked source")
            patched = patched.replace(before, after)
        compile(patched, name, "exec")
        (root / "overlays" / name).write_text(patched)
    (root / "factory_qwen_tts_fp16.py").write_bytes(
        Path(__file__).with_name("vllm_plugin.py").read_bytes()
    )
    metadata = root / "factory_qwen_tts_fp16-1.0.dist-info"
    metadata.mkdir(exist_ok=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: factory-qwen-tts-fp16\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[vllm.general_plugins]\nfactory_qwen_tts_fp16 = factory_qwen_tts_fp16:register\n"
    )
    for path in root.rglob("*"):
        if path.is_file():
            backend.adapter_assets[str(path.relative_to(backend.directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            stat = path.stat()
            backend._files[str(path)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return {
        "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "VLLM_PLUGINS": "factory_qwen_tts_fp16,vllm_omni_register_models",
    }
