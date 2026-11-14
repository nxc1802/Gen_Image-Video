"""Exact native video validation; initial Wan baseline, never public READY."""

import argparse
import asyncio
import base64
import hashlib
import json
import traceback
from importlib.metadata import version
from pathlib import Path

from model_registry.deployment_validator import inspect_environment

from benchmarks.native.video_semantic import VideoVerifier, semantic
from benchmarks.native.wave_a import digest, encoded, model_run
from factory.core.media import MediaRequest

ROOT = Path(__file__).resolve().parents[2]


def inline_fixture_inputs(value, assets, loaded):
    if isinstance(value, str) and value.startswith("@"):
        name = value[1:]
        return (
            "data:"
            + assets[name]["mime_type"]
            + ";base64,"
            + base64.b64encode(loaded[name]).decode()
        )
    if isinstance(value, list):
        return [inline_fixture_inputs(item, assets, loaded) for item in value]
    if isinstance(value, dict):
        return {key: inline_fixture_inputs(item, assets, loaded) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    cases = [c for c in suite["cases"] if c["id"] in args.cases and c["model"] in args.models]
    if set(args.cases) != {c["id"] for c in cases} or set(args.models) != {
        c["model"] for c in cases
    }:
        raise ValueError("Exact selected video cases/models required")
    assets = {name: suite["assets"][name] for c in cases for name in c["assets"]}
    loaded = {}
    for name, spec in assets.items():
        path = (ROOT / spec["fixture_path"]).resolve()
        if not path.is_relative_to((ROOT / "tests/fixtures/native").resolve()):
            raise ValueError("Video input fixture escaped trusted source bundle")
        raw = path.read_bytes()
        if len(raw) != spec["size_bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Exact video input fixture mismatch: " + name)
        loaded[name] = raw
        folder = args.output / "inputs" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source.bin").write_bytes(raw)
        (folder / "input_identity.json").write_text(json.dumps(spec, indent=2) + "\n")

    async def execute(context, adapter, case):
        inputs = inline_fixture_inputs(case["request"], assets, loaded)
        request = MediaRequest(model=case["model"], capability=case["capability"], inputs=inputs)
        capability = adapter.resolve_capability(request, case["capability"])
        required = adapter.required_capabilities(request, capability) | {capability}
        if any(
            not context.model.capabilities.get(c) or c not in adapter.capabilities for c in required
        ):
            raise ValueError("Undeclared video capability")
        adapter.validate_request(context.model, request, capability)
        prepared = await asyncio.to_thread(adapter.prepare, context, request, capability)
        if "graph" in prepared.payload:
            with adapter.client(context) as client:
                response = client.get("/object_info")
                response.raise_for_status()
                installed = response.json()
            needed = {n["class_type"] for n in prepared.payload["graph"].values()}
            (args.output / case["model"] / "native_node_schemas.json").write_text(
                json.dumps(
                    {name: installed[name] for name in sorted(needed) if name in installed},
                    indent=2,
                )
                + "\n"
            )
        raw = await asyncio.to_thread(adapter.infer, context, prepared)
        result = await asyncio.to_thread(adapter.normalize_output, context, prepared, raw)
        for item in result["artifacts"]:
            item["data_base64"] = base64.b64encode(item.pop("data")).decode()
        result["seed"] = case["request"]["seed"]
        if "graph" in prepared.payload:
            result.update(
                graph=prepared.payload["graph"],
                native_history=raw["native_history"],
                prompt_id=raw["prompt_id"],
            )
        return result

    comfy = all(m.startswith("minimax") for m in args.models)
    pins = json.loads((ROOT / "deployments/kaggle/runtime_wave_d.json").read_text())
    if len({m.startswith("minimax") for m in args.models}) != 1:
        raise ValueError("Video native backends require separate explicit runtime batches")
    report = {
        "classification": "REAL_NATIVE_VALIDATION_ATTEMPT",
        "hardware": inspect_environment().gpus,
        "case_definitions": cases,
        "input_assets": assets,
        "cases_sha256": digest({"cases": cases, "assets": assets}),
        "runtime_versions": {
            n: version(n)
            for n in [
                "torch",
                "torchvision",
                "torchaudio",
                "transformers",
                "huggingface-hub",
                "psutil",
                *(
                    ["gguf", "comfy-aimdo", "comfy-kitchen"]
                    if comfy
                    else ["diffusers", "accelerate", "imageio", "imageio-ffmpeg"]
                ),
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
                validator=lambda case, output: semantic(case, output, assets=loaded),
                backend_options={
                    "source_dir": args.root / "ComfyUI",
                    "source_revision": pins["comfyui"]["source_revision"],
                }
                if comfy
                else None,
            )
        )
        if comfy:
            marker = "FACTORY_MINIMAX_RUNTIME "
            log = directory / "backend.log"
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
    if any(c["runs"] for m in report["models"] for c in m["cases"]):
        try:
            verifier = VideoVerifier(args.root / "semantic_weights/clip", loaded)
            report["semantic_verifiers"] = verifier.evidence
            speech = None
            if any("spoken_audio" in c["expected"] for c in cases):
                from benchmarks.native.video_audio import VideoSpeechVerifier

                speech = VideoSpeechVerifier(args.root / "semantic_weights/video_speech")
                report["audio_semantic_verifiers"] = speech.evidence
            for model in report["models"]:
                directory = args.output / model["model"]
                for native in model["cases"]:
                    case = next(c for c in cases if c["id"] == native["case_id"])
                    for run in native["runs"]:
                        path = directory / run["output_file"]
                        output = json.loads(path.read_text())
                        output["verification"] = verifier.evaluate_video(
                            base64.b64decode(output["artifacts"][0]["data_base64"], validate=True),
                            case["expected"],
                        )
                        if "spoken_audio" in case["expected"]:
                            output["audio_verification"] = speech.evaluate(
                                base64.b64decode(
                                    output["artifacts"][0]["data_base64"], validate=True
                                )
                            )
                        path.write_bytes(encoded(output))
                        run["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        run["semantic_validation"] = semantic(case, output, assets=loaded)
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
    report["status"] = (
        "PASS"
        if report.get("semantic_verifiers")
        and not report.get("semantic_verifier_error")
        and all(m["status"] == "PASS" for m in report["models"])
        else "NOT_MET"
    )
    (args.output / "wave_d.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "scope": "INITIAL_T2V_BASELINES_ONLY_NOT_FULL_WAVE_D"}
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
