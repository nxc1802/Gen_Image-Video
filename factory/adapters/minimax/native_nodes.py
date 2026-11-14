"""Install the private MiniMax loader and pin its real upstream API dependencies."""

import hashlib
from pathlib import Path

SOURCE_REVISION = "c5751c81abe131202eea9e4c66d000589013dd0b"
SOURCE_FILES = {
    "comfy/supported_models.py": "4fdcdc20c50c26c280a4aefad9611c712ab441e577d05864ae67215b45a744d0",
    "nodes.py": "bc13c5b30448ab9af58491a07c343b21c5300bb952bb22d2552b1c7c8194ac35",
    "comfy_extras/nodes_minimax_h3.py": "217f3b50ede649831c34be347196926211606904610bcb267c8725c18eb62498",
}


def install(base, backend, source_dir):
    if backend.model.weights.default_precision != "fp32":
        raise RuntimeError(
            "MiniMax native loader requires explicitly selected Manifest FP32; no dtype fallback"
        )
    files = []
    for name, sha in SOURCE_FILES.items():
        raw = (source_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise RuntimeError(
                "Selected MiniMax native API source differs from pinned dependency: " + name
            )
        target = base / ("minimax_upstream_" + Path(name).name)
        target.write_bytes(raw)
        files.append(target)
    target = base / "custom_nodes/factory_minimax_loader/__init__.py"
    target.parent.mkdir(parents=True)
    raw = Path(__file__).with_name("loader_node.py").read_bytes()
    compile(raw, "factory_minimax_loader", "exec")
    target.write_bytes(raw)
    files.append(target)
    for path in files:
        backend.adapter_assets[str(path.relative_to(backend.directory))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        stat = path.stat()
        backend._files[str(path)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
