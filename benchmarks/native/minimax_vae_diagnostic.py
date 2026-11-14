"""Real one-call/same-latent GPU diagnosis, explicitly excluded from V3 gates."""

import argparse
import asyncio
import base64
import copy
import hashlib
import json
import shutil
import time
import traceback
from dataclasses import replace
from pathlib import Path

import psutil
import yaml
from factory_contracts.deployment import BackendOptions, DeploymentConfig, DeploymentEntry
from model_registry.adapters.launch import ComfyLaunchAdapter
from model_registry.backends.session import BackendSession
from model_registry.deployment import DeploymentRegistry, ModelCatalog
from model_registry.deployment_validator import inspect_environment

from benchmarks.native.video_semantic import VideoVerifier, decode
from benchmarks.native.wave_a import memory_observation
from benchmarks.native.wave_d import inline_fixture_inputs
from factory.adapters.comfy import checkpoint
from factory.adapters.minimax import Adapter
from factory.core.media import MediaRequest
from factory.core.model_adapter import ExecutionContext, PreparedCall

ROOT = Path(__file__).resolve().parents[2]


def native_canvas_case(case):
    case = copy.deepcopy(case)
    case["id"] = "minimax-i2v-native-768p-one-call-diagnosis"
    case["request"].update(width=1152, height=768)
    return case


def sampler_control_case(case):
    case = copy.deepcopy(case)
    case["id"] = "minimax-i2v-res-multistep-one-call-diagnosis"
    return case


def diagnostic_manifest(candidate, precision, *, native_canvas=False):
    candidate = copy.deepcopy(candidate)
    candidate["weights"]["component_precisions"] = {"vae": precision}
    if native_canvas:
        candidate["parameters"]["max_height"]["default"] = 768
        candidate["runtime_profiles"]["kaggle-dual-t4"].update(
            inference_timeout_seconds=5400, extra_vram_headroom_gb=2.0,
        )
    return candidate


async def run(root, output, case, *, native_canvas=False, sampler_control=False):
    assert not (native_canvas and sampler_control)
    single_call = native_canvas or sampler_control
    report = {
        "classification": "REAL_ONE_CALL_AND_SAME_LATENT_VAE_GPU_DIAGNOSIS_NOT_V3",
        "status": "NOT_MET", "production_ready": False, "v3_certified": False,
        "full_factory_e2e_executed": False, "fallback": False, "stages": [],
        "canonical_manifest_changed": False, "case_definition": case,
        "source_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if native_canvas:
        report["classification"] = "REAL_ONE_CALL_NATIVE_768P_CANVAS_GPU_DIAGNOSIS_NOT_V3"
        report["diagnostic_runtime_policy"] = {
            "inference_timeout_seconds": 5400, "extra_vram_headroom_gb": 2.0,
            "fixed_gpu": [0], "unchanged_gpu_budget_gib": 14.2,
        }
    if sampler_control:
        report["classification"] = "REAL_ONE_CALL_RES_MULTISTEP_GPU_DIAGNOSIS_NOT_V3"
        report["diagnostic_runtime_policy"] = {
            "sampler_name": "res_multistep", "fixed_gpu": [0],
            "unchanged_canvas_steps_seed_deadline_headroom_oracles": True,
        }
    report["hardware"] = inspect_environment().gpus
    assert set(report["hardware"]) == {0, 1} and all("t4" in g["name"].lower() for g in report["hardware"].values())
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    assets = {name: suite["assets"][name] for name in case["assets"]}
    loaded = {}
    for name, spec in assets.items():
        raw = (ROOT / spec["fixture_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
        loaded[name] = raw
    report["input_assets"] = assets
    original_install, original_plan = Adapter.install_nodes, ComfyLaunchAdapter.plan
    plugin = Path(__file__).with_name("minimax_vae_diagnostic_nodes.py").read_bytes()

    def install(self, base, *, backend, source_dir):
        original_install(self, base, backend=backend, source_dir=source_dir)
        path = base / "custom_nodes/factory_vae_diagnostic/__init__.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(plugin)
        backend.adapter_assets[str(path.relative_to(backend.directory))] = hashlib.sha256(plugin).hexdigest()
        stat = path.stat()
        backend._files[str(path)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def plan(self, backend, view):
        selected = backend.model.weights.component_precisions
        if set(selected) != {"vae"} or selected["vae"].value not in {"fp16", "fp32"}:
            raise RuntimeError("Explicit diagnostic Manifest VAE precision required")
        launch = original_plan(self, backend, view)
        return replace(launch, argv=(*launch.argv, "--" + selected["vae"].value + "-vae"))

    Adapter.install_nodes, ComfyLaunchAdapter.plan = install, plan
    captured = output / "actual_sampled_video.safetensors"
    try:
        for precision in (("fp16",) if single_call else ("fp16", "fp32")):
            stage = {"vae_precision": precision, "gpu_samples": [], "status": "NOT_MET",
                     "full_transformer_inference_executed": False, "production_ready": False}
            report["stages"].append(stage)
            manifest_root = root / ("diagnostic_manifests_" + precision)
            shutil.copytree(ROOT / "manifests", manifest_root)
            path = manifest_root / "models/minimax-h3.yaml"
            candidate = diagnostic_manifest(yaml.safe_load(path.read_text()), precision, native_canvas=native_canvas)
            path.write_text(yaml.safe_dump(candidate, sort_keys=False))
            stage["diagnostic_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (output / ("manifest_" + precision + ".yaml")).write_bytes(path.read_bytes())
            catalog = ModelCatalog(manifest_root)
            model = catalog.get("minimax-h3")
            from huggingface_hub import hf_hub_download

            stage["checkpoint_files"] = []
            for key, spec in model.weights.files.items():
                source = Path(hf_hub_download(spec.repository, filename=spec.path, revision=spec.revision))
                h = hashlib.sha256()
                with source.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        h.update(chunk)
                assert h.hexdigest() == spec.sha256 and source.stat().st_size == spec.size_bytes
                stage["checkpoint_files"].append({"key": key, "sha256": h.hexdigest(), "matched": True})
                target = root / "models/minimax-h3" / (spec.destination or spec.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.symlink_to(source.resolve())
            config = DeploymentConfig(models_dir=root / "models", runtime_dir=root / ("runtime_" + precision),
                deployments=(DeploymentEntry(model="minimax-h3", devices=(0,), port=18990,
                    backend_options=BackendOptions(source_dir=root / "ComfyUI",
                    source_revision="c5751c81abe131202eea9e4c66d000589013dd0b", startup_timeout_seconds=600)),))
            registry = DeploymentRegistry(config, catalog, config_path=root / "diagnostic.yaml")
            session = BackendSession(registry)
            backend = None
            try:
                await session.start()
                assert session.healthy_models() == ("minimax-h3",)
                backend = session.backends["minimax-h3"]
                stage["health_check_failures"] = []

                def observe_check(original, name, stage=stage):
                    def observed(*args, **kwargs):
                        try:
                            result = original(*args, **kwargs)
                        except Exception as exc:
                            if len(stage["health_check_failures"]) < 32:
                                stage["health_check_failures"].append({
                                    "check": name, "monotonic_ns": time.monotonic_ns(),
                                    "exception_type": type(exc).__name__, "message": str(exc),
                                })
                            raise
                        if result is False and len(stage["health_check_failures"]) < 32:
                            stage["health_check_failures"].append({
                                "check": name, "monotonic_ns": time.monotonic_ns(),
                                "returned_false": True,
                            })
                        return result
                    return observed

                for name in ("_json", "_listener_owned", "_gpu_ownership"):
                    setattr(backend, name, observe_check(getattr(backend, name), name))
                stage["lifecycle_before"] = session.status()
                stage["actual_owned_argv"] = psutil.Process(backend.process.pid).cmdline()
                assert "--" + precision + "-vae" in stage["actual_owned_argv"]
                context = ExecutionContext(registry, session, registry.assignment("minimax-h3"), model)
                adapter = Adapter()
                if precision == "fp16":
                    request_inputs = inline_fixture_inputs(copy.deepcopy(case["request"]), assets, loaded)
                    request_inputs["seed"] = case["seed_sequence"][0]
                    request = MediaRequest(model="minimax-h3", capability=case["capability"], inputs=request_inputs)
                    adapter.validate_request(model, request, request.capability)
                    prepared = await asyncio.to_thread(adapter.prepare, context, request, request.capability)
                    graph = prepared.payload["graph"]
                    if sampler_control:
                        graph["sample"]["inputs"]["sampler_name"] = "res_multistep"
                    graph["capture"] = {"class_type": "FactoryDiagnosticCaptureVideoLatent", "inputs": {"samples": ["sample", 0], "path": str(captured)}}
                    graph["decode"]["inputs"]["samples"] = ["capture", 0]
                else:
                    latent_sha = hashlib.sha256(captured.read_bytes()).hexdigest()
                    graph = {
                        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": checkpoint(model, "video_vae")}},
                        "load": {"class_type": "FactoryDiagnosticLoadCapturedVideoLatent", "inputs": {"path": str(captured), "expected_sha256": latent_sha}},
                        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["load", 0], "vae": ["vae", 0]}},
                        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 24.0}},
                        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": "diagnostic", "format": "mp4", "codec": "h264"}},
                    }
                    prepared = PreparedCall(case["capability"], {"graph": graph})
                    stage["input_latent_file_sha256"] = latent_sha
                stage["actual_submitted_graph"] = graph
                active = {"value": True}

                async def sample(stage=stage, backend=backend, active=active):
                    while active["value"]:
                        stage["gpu_samples"].append({"monotonic_ns": time.monotonic_ns(),
                            "gpu_processes": backend.runtime_info().get("gpu_processes", []), "memory": memory_observation()})
                        await asyncio.sleep(.5)

                sampler = asyncio.create_task(sample())
                start = time.monotonic()
                try:
                    actual = await asyncio.to_thread(adapter.infer, context, prepared)
                    normalized = await asyncio.to_thread(adapter.normalize_output, context, prepared, actual)
                except Exception:
                    # Observe a possible exit race without replaying the submitted graph.
                    stage["failure_observations"] = []
                    for _ in range(20):
                        code = backend.process.poll()
                        stage["failure_observations"].append({
                            "monotonic_ns": time.monotonic_ns(),
                            "backend_exit_code": code,
                            "memory": memory_observation(),
                        })
                        if code is not None:
                            break
                        await asyncio.sleep(.5)
                    raise
                finally:
                    active["value"] = False
                    await sampler
                stage["latency_seconds"] = time.monotonic() - start
                assert len(normalized["artifacts"]) == 1
                video = normalized["artifacts"][0].pop("data")
                target = output / ("actual_" + precision + ".mp4")
                target.write_bytes(video)
                stage["output_mp4"] = {"file": target.name, "sha256": hashlib.sha256(video).hexdigest(), "size_bytes": len(video)}
                stage["native_history"] = actual["native_history"]
                if single_call:
                    # This is solely the real returned artifact plus its actual submitted history.
                    stage["actual_output"] = {
                        **normalized, "seed": case["seed_sequence"][0], "graph": graph,
                        "native_history": actual["native_history"], "prompt_id": actual["prompt_id"],
                    }
                    stage["actual_output"]["artifacts"][0]["data_base64"] = base64.b64encode(video).decode()
                if precision == "fp16":
                    assert captured.is_file(), "Actual sampled latent was not captured"
                    stage["full_transformer_inference_executed"] = True
                    stage["captured_latent_file_sha256"] = hashlib.sha256(captured.read_bytes()).hexdigest()
                assert not session.failures
                assert max((p["used_memory_mb"] for s in stage["gpu_samples"] for p in s["gpu_processes"]), default=0) > 1024, "Actual owned GPU residency missing"
                stage["lifecycle_after"] = session.status()
                stage["status"] = "REAL_STAGE_COMPLETE_NOT_CERTIFICATION"
            finally:
                # Preserve failure evidence before our own stop can change the exit code.
                stage["memory_before_shutdown"] = memory_observation()
                if backend is not None and backend.process is not None:
                    stage["backend_exit_code_before_shutdown"] = backend.process.poll()
                await session.stop()
                stage["shutdown"] = json.loads((session.directory / "shutdown.json").read_text())
                assert stage["shutdown"]["stop_completed"] and not stage["shutdown"]["errors"]
                if backend is not None:
                    raw = (backend.directory / "backend.log").read_bytes()
                    (output / ("backend_" + precision + ".log")).write_bytes(raw)
                    dtype_lines = [line for line in raw.decode(errors="replace").splitlines() if "VAE load device:" in line]
                    assert dtype_lines and all("dtype: torch.float" + ("16" if precision == "fp16" else "32") in line for line in dtype_lines)
        if not single_call:
            assert report["stages"][0]["captured_latent_file_sha256"] == report["stages"][1]["input_latent_file_sha256"]
        verifier = VideoVerifier(root / "semantic_weights/clip", loaded)
        report["semantic_verifiers"] = verifier.evidence
        decoded_stages = []
        for stage in report["stages"]:
            data = (output / stage["output_mp4"]["file"]).read_bytes()
            frames, info = decode(data)
            assert info["frames"] == case["request"]["num_frames"] and info["fps"] == 24.0
            decoded_stages.append(frames)
            stage["decoded"] = info
            stage["visual_verification"] = verifier.evaluate_video(data, case["expected"])
            if single_call:
                from benchmarks.native.video_semantic import semantic

                actual_record = stage.pop("actual_output")
                actual_record["verification"] = stage["visual_verification"]
                stage["semantic_validation"] = semantic(case, actual_record, assets=loaded)
                assert stage["semantic_validation"]["status"] in {"PASS", "FAIL"}, "Actual numeric semantics were not evaluated"
                raw = (json.dumps(actual_record, sort_keys=True) + "\n").encode()
                target = output / "actual_inference_output.json"
                target.write_bytes(raw)
                stage["actual_output_json_sha256"] = hashlib.sha256(raw).hexdigest()
            from PIL import Image, ImageDraw

            sheet = Image.new("RGB", (8 * 192, ((len(frames) + 7) // 8) * 146), "white")
            draw = ImageDraw.Draw(sheet)
            for index, frame in enumerate(frames):
                thumbnail = frame.copy()
                thumbnail.thumbnail((192, 128))
                x, y = index % 8 * 192, index // 8 * 146
                sheet.paste(thumbnail, (x, y))
                draw.text((x + 2, y + 128), str(index), fill="black")
            target = output / (stage["vae_precision"] + "_all_frames.png")
            sheet.save(target)
            stage["contact_sheet_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        import numpy as np

        if not single_call:
            report["same_latent_decoder_pixel_mae_by_frame"] = [
                float(np.abs(np.asarray(a).astype(np.int16) - np.asarray(b).astype(np.int16)).mean() / 255)
                for a, b in zip(*decoded_stages, strict=True)
            ]
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        Adapter.install_nodes, ComfyLaunchAdapter.plan = original_install, original_plan
        # The setup wrapper needs this minimal shape, but no complete native case exists.
        report["models"] = [{"model": "minimax-h3", "cases": [{"native_inference_executed": bool(report["stages"] and report["stages"][0].get("full_transformer_inference_executed")), "status": "DIAGNOSTIC_ONLY", "runs": []}]}]
        (output / "wave_d.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "classification": report["classification"], "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--cases", nargs="+")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--native-canvas", action="store_true")
    modes.add_argument("--sampler-control", action="store_true")
    args = parser.parse_args()
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    case = next(c for c in suite["cases"] if c["id"] == "minimax-i2v-astronaut-native-temporal-v2")
    assert args.models == ["minimax-h3"] and args.cases == [case["id"]]
    if args.native_canvas:
        case = native_canvas_case(case)
    if args.sampler_control:
        case = sampler_control_case(case)
    return asyncio.run(run(args.root, args.output, case, native_canvas=args.native_canvas, sampler_control=args.sampler_control))


if __name__ == "__main__":
    raise SystemExit(main())
