"""Exact native image probes; no public readiness or replacement checkpoint."""

import argparse
import asyncio
import base64
import hashlib
import json
import traceback
import urllib.request
from importlib.metadata import version
from pathlib import Path

from model_registry.deployment_validator import inspect_environment

from benchmarks.native.image_inputs import derive
from benchmarks.native.image_semantic import ImageVerifier, semantic
from benchmarks.native.wave_a import digest, encoded, model_run
from factory.core.media import MediaRequest

ROOT = Path(__file__).resolve().parents[2]


def load_assets(specs, directory):
    loaded = {}
    for name, spec in specs.items():
        if "recipe" in spec:
            raw = derive(spec["recipe"], loaded)
        else:
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                raw = response.read(spec["size_bytes"] + 1)
        if len(raw) != spec["size_bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Exact image input integrity failed: " + name)
        loaded[name] = raw
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "source.bin").write_bytes(raw)
        (target / "input_identity.json").write_text(json.dumps(spec, indent=2) + "\n")
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_c.json").read_text())
    pins = json.loads((ROOT / "deployments/kaggle/runtime_wave_c.json").read_text())["comfyui"]
    cases = [
        c
        for c in suite["cases"]
        if c["model"] in args.models and (not args.cases or c["id"] in args.cases)
    ]
    if not cases or (args.cases and set(args.cases) != {c["id"] for c in cases}):
        raise ValueError("Exact nonempty image case subset required")
    if set(args.models) != {c["model"] for c in cases}:
        raise ValueError("Explicit image models must match the selected case subset")
    names = {n for c in cases for n in c["assets"]}
    assets = {n: s for n, s in suite["assets"].items() if n in names}
    loaded = load_assets(assets, args.output / "inputs")

    async def execute(context, adapter, case):
        def resolve(value):
            if isinstance(value, str) and value.startswith("@"):
                raw = loaded[value[1:]]
                return "data:image/png;base64," + base64.b64encode(raw).decode()
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        request = MediaRequest(
            model=case["model"],
            capability=case["capability"],
            inputs={k: resolve(v) for k, v in case["request"].items()},
        )
        capability = adapter.resolve_capability(request, case["capability"])
        required = adapter.required_capabilities(request, capability) | {capability}
        if any(
            not context.model.capabilities.get(c) or c not in adapter.capabilities for c in required
        ):
            raise ValueError("Undeclared image capability")
        adapter.validate_request(context.model, request, capability)
        prepared = await asyncio.to_thread(adapter.prepare, context, request, capability)
        raw = await asyncio.to_thread(adapter.infer, context, prepared)
        result = await asyncio.to_thread(adapter.normalize_output, context, prepared, raw)
        for artifact in result["artifacts"]:
            artifact["data_base64"] = base64.b64encode(artifact.pop("data")).decode()
        result.update(
            graph=prepared.payload["graph"],
            native_history=raw["native_history"],
            prompt_id=raw["prompt_id"],
            seed=case["request"]["seed"],
        )
        return result

    report = {
        "classification": "REAL_NATIVE_VALIDATION_ATTEMPT",
        "hardware": inspect_environment().gpus,
        "cases_sha256": digest({"cases": cases, "assets": assets}),
        "case_definitions": cases,
        "input_assets": assets,
        "runtime_versions": {
            n: version(n)
            for n in [
                "torch",
                "torchvision",
                "torchaudio",
                "transformers",
                "gguf",
                "comfy-aimdo",
                "comfy-kitchen",
                "huggingface-hub",
                "psutil",
            ]
        },
        "models": [],
        "status": "NOT_MET",
        "fallback": False,
        "production_ready": False,
        "full_factory_e2e_executed": False,
    }
    assert set(report["hardware"]) == {0, 1} and all(
        "t4" in v["name"].lower() for v in report["hardware"].values()
    ), "Exact Dual-T4 required"
    for model in args.models:
        directory = args.output / model
        directory.mkdir(parents=True, exist_ok=True)
        result = asyncio.run(
            model_run(
                model,
                args.root,
                directory,
                [c for c in cases if c["model"] == model],
                executor=execute,
                validator=semantic,
                backend_options={
                    "source_dir": args.root / "ComfyUI",
                    "source_revision": pins["source_revision"],
                },
            )
        )
        log = directory / "backend.log"
        marker = "FACTORY_QWEN_IMAGE_RUNTIME "
        result["precision_attestations"] = (
            [
                json.loads(line.split(marker, 1)[1])
                for line in log.read_text(errors="replace").splitlines()
                if marker in line
            ]
            if log.exists()
            else []
        )
        (directory / "observation.json").write_text(json.dumps(result, indent=2) + "\n")
        report["models"].append(result)
        (args.output / "wave_c.json").write_text(json.dumps(report, indent=2) + "\n")
    # Every selected target backend has stopped before this separate CPU evaluator loads.
    if any(c["runs"] for m in report["models"] for c in m["cases"]):
        try:
            verifier = ImageVerifier(args.root / "semantic_weights/clip", loaded)
            report["semantic_verifiers"] = verifier.evidence
            for model in report["models"]:
                directory = args.output / model["model"]
                for native in model["cases"]:
                    case = next(c for c in cases if c["id"] == native["case_id"])
                    for run in native["runs"]:
                        path = directory / run["output_file"]
                        actual = json.loads(path.read_text())
                        raw = base64.b64decode(actual["artifacts"][0]["data_base64"], validate=True)
                        actual["verification"] = verifier.evaluate(raw, case["expected"])
                        path.write_bytes(encoded(actual))
                        run["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        run["semantic_validation"] = semantic(case, actual, assets=loaded)
                    native["status"] = (
                        "PASS"
                        if len(native["runs"]) == 4
                        and all(
                            r["semantic_validation"]["status"] == "PASS" for r in native["runs"]
                        )
                        else "NOT_MET"
                    )
                model["status"] = (
                    "PASS"
                    if model.get("stop_verified")
                    and len(model["cases"])
                    == len([c for c in cases if c["model"] == model["model"]])
                    and all(c["status"] == "PASS" for c in model["cases"])
                    else "NOT_MET"
                )
                (directory / "observation.json").write_text(json.dumps(model, indent=2) + "\n")
        except Exception as exc:
            report["semantic_verifier_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            for model in report["models"]:
                model["status"] = "NOT_MET"
    report["status"] = (
        "PASS"
        if report["models"] and all(m["status"] == "PASS" for m in report["models"])
        else "NOT_MET"
    )
    (args.output / "wave_c.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "models": {m["model"]: m["status"] for m in report["models"]},
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
