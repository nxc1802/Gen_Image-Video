"""Recompute native image semantics from real collected bytes; never grants READY."""

import hashlib
import json
from pathlib import Path

from benchmarks.native.image_inputs import derive
from benchmarks.native.image_semantic import semantic
from benchmarks.native.wave_a import digest
from benchmarks.native.wave_a_gate import evaluate
from factory.adapters.qwen_image.native_nodes import (
    GGUF_LOADER_SHA256,
    GGUF_NODES_SHA256,
    GGUF_REVISION,
)

ROOT = Path(__file__).resolve().parents[2]


def evaluate_images(root=ROOT):
    inputs = {}

    def validate_report(kind, setup, observed, verified_raw):
        pin = json.loads((root / "deployments/kaggle/runtime_wave_c.json").read_text())["comfyui"]
        assert setup["runtime_pins"]["comfyui"] == pin, "Native Comfy pin drift"
        assert all(
            observed["runtime_versions"][name] == pin[name]
            for name in [
                "torch",
                "torchvision",
                "torchaudio",
                "transformers",
                "gguf",
                "comfy-aimdo",
                "comfy-kitchen",
            ]
        ), "Native image runtime drift"
        run = json.loads((root / f"validation/native_{kind}_run.json").read_text())
        folder = root / f"validation/native_{kind}_v{run['response']['versionNumber']}"
        packages = {
            name.casefold().replace("_", "-"): version
            for name, version in json.loads(verified_raw(folder / "runtime_packages.json")).items()
        }
        assert all(
            packages[name] == version for name, version in observed["runtime_versions"].items()
        ), "Collected package inventory disagrees with native runtime"
        for name, spec in observed["input_assets"].items():
            raw = verified_raw(folder / "inputs" / name / "source.bin")
            assert (
                len(raw) == spec["size_bytes"] and hashlib.sha256(raw).hexdigest() == spec["sha256"]
            )
            if "recipe" in spec:
                assert raw == derive(spec["recipe"], inputs), "Technical image input recipe drift"
            inputs[name] = raw
        verification = observed.get("semantic_verifiers")
        if verification:
            pins = json.loads((root / "benchmarks/native/image_semantic_pins.json").read_text())
            assert verification["scope"] == pins["scope"] and verification["pins_sha256"] == digest(
                pins
            )
            assert verification["verified_files"] == {
                name: {**spec, "matched": True} for name, spec in pins["files"].items()
            }
            loaded = verification["loaded"]
            assert (
                loaded["devices"] == ["cpu"]
                and loaded["dtypes"] == ["torch.float32"]
                and loaded["parameter_count"] == 151277313
            )
        for model in observed["models"]:
            backend = model.get("runtime_after_inference")
            if not backend:
                continue
            assert backend["source_revision"] == pin["source_revision"] and backend["devices"] == [
                0
            ]
            assets = backend["adapter_assets"]
            for name, sha in assets.items():
                raw = verified_raw(folder / model["model"] / "adapter_assets" / name)
                assert hashlib.sha256(raw).hexdigest() == sha, (
                    "Private image Adapter source integrity failed"
                )
            name = next(n for n in assets if n.endswith("/qwen_image_native_dependencies.json"))
            metadata = json.loads(verified_raw(folder / model["model"] / "adapter_assets" / name))
            assert (
                metadata["revision"] == GGUF_REVISION
                and metadata["loader_upstream_sha256"] == GGUF_LOADER_SHA256
            )
            upstream_name = next(
                n for n in assets if n.endswith("/qwen_image_gguf_loader_upstream.py")
            )
            assert assets[upstream_name] == GGUF_LOADER_SHA256
            loader_name = next(n for n in assets if n.endswith("/ComfyUI-GGUF/loader.py"))
            assert assets[loader_name] == metadata["loader_patched_sha256"]
            assert metadata["nodes_upstream_sha256"] == GGUF_NODES_SHA256
            assert metadata["diffusion_compute_precision"] == "fp32"
            nodes_original = next(
                n for n in assets if n.endswith("/qwen_image_gguf_nodes_upstream.py")
            )
            assert assets[nodes_original] == GGUF_NODES_SHA256
            nodes_patched = next(n for n in assets if n.endswith("/ComfyUI-GGUF/nodes.py"))
            assert assets[nodes_patched] == metadata["nodes_patched_sha256"]
            marker = "FACTORY_QWEN_IMAGE_RUNTIME "
            attestations = [
                json.loads(line.split(marker, 1)[1])
                for line in verified_raw(folder / model["model"] / "backend.log")
                .decode(errors="replace")
                .splitlines()
                if marker in line
            ]
            assert attestations and model["precision_attestations"] == attestations, (
                "Missing native diffusion precision attestation"
            )
            from model_registry.deployment import ModelCatalog

            selected_model = ModelCatalog(root / "manifests").get(model["model"])
            assert selected_model.weights.default_precision == "fp32"
            assert all(
                a["precision"] == "fp32"
                and a["compute_dtype"] == "torch.float32"
                and a["manual_cast_dtype"] in {"None", "torch.float32"}
                and a["load_device"] == "cuda:0"
                and a["offload_device"] == "cpu"
                and a["checkpoint"]
                == Path(
                    selected_model.weights.files["primary"].destination
                    or selected_model.weights.files["primary"].path
                ).name
                for a in attestations
            ), "Native precision or selected checkpoint mismatch"

    report = evaluate(
        root,
        wave="c",
        models=("qwen-image-2.1", "qwen-image-2.1-turbo"),
        kinds=("image_t2i", "image_i2i", "image_masked", "image_edit", "image_multi"),
        validator=lambda case, output: semantic(case, output, assets=inputs),
        report_validator=validate_report,
    )
    for dependency in ("a", "b"):
        gate = json.loads((root / f"validation/native_wave_{dependency}_gate.json").read_text())
        if gate["exit_gate"] != "PASS":
            report["exit_gate"] = "NOT_MET"
            report["findings"].append("Wave " + dependency.upper() + " dependency gate is not PASS")
    report["excluded"] = json.loads((root / "benchmarks/native/golden_wave_c.json").read_text())[
        "excluded"
    ]
    return report


def main():
    report = evaluate_images()
    (ROOT / "validation/native_wave_c_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "exit_gate": report["exit_gate"],
                "native_pass": sum(c["status"] == "PASS" for c in report["cases"]),
                "findings": report["findings"],
            }
        )
    )
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
