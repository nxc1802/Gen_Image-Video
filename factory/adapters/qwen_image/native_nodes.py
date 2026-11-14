"""Private, pinned Qwen Image Comfy nodes; this code generates no image output."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

GGUF_REVISION = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
GGUF_LOADER_SHA256 = "ffff39d209ce22654e67378b80c6a3ca3d349cf8f00ec267085bd16e194730c2"
GGUF_NODES_SHA256 = "16be3b08b13de6279fc432addc628320019fcb24963cbc6b52b248de8f06316e"
DTYPE_ANCHOR = 'sd, model_options={"custom_operations": ops}, **kwargs,'
ATTEST_ANCHOR = "        model = GGUFModelPatcher.clone(model)"
ATTESTATION = """        actual_dtype = model.model.get_dtype()
        if actual_dtype != torch.float32 or model.load_device.type != "cuda":
            raise RuntimeError("Selected Qwen Image Manifest requires native FP32 diffusion on CUDA")
        logging.info("FACTORY_QWEN_IMAGE_RUNTIME " + __import__("json").dumps({
            "precision": "fp32", "compute_dtype": str(actual_dtype),
            "manual_cast_dtype": str(model.model.manual_cast_dtype),
            "load_device": str(model.load_device), "offload_device": str(model.offload_device),
            "checkpoint": unet_name,
        }, sort_keys=True))
"""
ARCH_ANCHOR = '    type_str = get_field(reader, "general.type", str)'
ARCH_HINT = """
    # Explicit Factory Qwen Image 2.1 adapter hint for its metadata-free GGUF.
    # Native Comfy still selects the actual diffusion model from these weights.
    if arch_str is None and not is_text_model and {
        "img_in.weight", "modulation.1.weight", "txt_in.text_norm.weight",
        "transformer_blocks.0.img_mlp.gate_up.weight",
    }.issubset({name for name, _ in tensors}):
        arch_str = "qwen_image"
"""


def native_node_overlay(raw, precision):
    if precision != "fp32":
        raise RuntimeError(
            "Pinned native Qwen Image 2.1 integration explicitly supports FP32 diffusion; no automatic dtype fallback"
        )
    if hashlib.sha256(raw).hexdigest() != GGUF_NODES_SHA256:
        raise RuntimeError("Qwen Image native node source differs from its pinned dependency")
    text = raw.decode()
    if text.count(DTYPE_ANCHOR) != 1 or text.count(ATTEST_ANCHOR) != 1:
        raise RuntimeError("Native Qwen Image loader layout changed")
    patched = text.replace(
        DTYPE_ANCHOR,
        'sd, model_options={"custom_operations": ops, "dtype": torch.float32}, **kwargs,',
    )
    patched = patched.replace(ATTEST_ANCHOR, ATTESTATION + ATTEST_ANCHOR)
    compile(patched, "qwen_image_native_nodes", "exec")
    return patched.encode()


def install(base, backend, source_dir):
    extension = source_dir / "custom_nodes/ComfyUI-GGUF"
    observed = subprocess.check_output(
        ["git", "-C", str(extension), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != GGUF_REVISION:
        raise RuntimeError(
            "Selected Qwen Image adapter requires its explicit pinned ComfyUI-GGUF checkout"
        )
    original = (extension / "loader.py").read_bytes()
    if hashlib.sha256(original).hexdigest() != GGUF_LOADER_SHA256:
        raise RuntimeError(
            "Qwen Image GGUF loader source differs from its pinned native dependency"
        )
    text = original.decode()
    if text.count(ARCH_ANCHOR) != 1:
        raise RuntimeError("Native GGUF loader has an unsupported architecture-detection layout")
    patched = text.replace(ARCH_ANCHOR, ARCH_ANCHOR + ARCH_HINT)
    compile(patched, "qwen_image_gguf_loader", "exec")
    nodes_original = (extension / "nodes.py").read_bytes()
    nodes_patched = native_node_overlay(nodes_original, backend.model.weights.default_precision)
    target_extension = base / "custom_nodes/ComfyUI-GGUF"
    shutil.copytree(
        extension, target_extension, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
    )
    (target_extension / "loader.py").write_text(patched)
    (target_extension / "nodes.py").write_bytes(nodes_patched)
    upstream = base / "qwen_image_gguf_loader_upstream.py"
    upstream.write_bytes(original)
    nodes_upstream = base / "qwen_image_gguf_nodes_upstream.py"
    nodes_upstream.write_bytes(nodes_original)
    target = base / "custom_nodes/factory_qwen_latent"
    target.mkdir(parents=True)
    (target / "__init__.py").write_bytes(Path(__file__).with_name("latent_node.py").read_bytes())
    metadata = base / "qwen_image_native_dependencies.json"
    metadata.write_text(
        json.dumps(
            {
                "repository": "city96/ComfyUI-GGUF",
                "revision": GGUF_REVISION,
                "loader_upstream_sha256": GGUF_LOADER_SHA256,
                "loader_patched_sha256": hashlib.sha256(
                    (target_extension / "loader.py").read_bytes()
                ).hexdigest(),
                "nodes_upstream_sha256": GGUF_NODES_SHA256,
                "nodes_patched_sha256": hashlib.sha256(nodes_patched).hexdigest(),
                "diffusion_compute_precision": backend.model.weights.default_precision,
            },
            sort_keys=True,
        )
        + "\n"
    )
    for path in [
        metadata,
        upstream,
        nodes_upstream,
        *[p for p in (base / "custom_nodes").rglob("*") if p.is_file()],
    ]:
        backend.adapter_assets[str(path.relative_to(backend.directory))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        stat = path.stat()
        backend._files[str(path)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
