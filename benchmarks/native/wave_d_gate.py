"""Native video evidence gate. Missing Wan/MiniMax coverage remains NOT_MET."""

import hashlib
import json
from pathlib import Path

from benchmarks.native.video_semantic import semantic
from benchmarks.native.wave_a import digest
from benchmarks.native.wave_a_gate import evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    inputs = {}

    def verify_report(kind, setup, observed, verified_raw):
        runtime = json.loads((ROOT / "deployments/kaggle/runtime_wave_d.json").read_text())
        is_comfy = kind.startswith("minimax")
        pin = runtime["comfyui" if is_comfy else "diffusers"]
        assert setup["runtime_pins"]["comfyui" if is_comfy else "diffusers"] == pin
        versions = observed["runtime_versions"]
        names = ["torch", "torchvision", "torchaudio", "transformers"]
        if is_comfy:
            names += ["gguf", "comfy-aimdo", "comfy-kitchen"]
        else:
            assert (
                setup.get("backend_wheel_verified") is True
                and versions["diffusers"] == pin["version"]
            )
            names += ["accelerate", "imageio", "imageio-ffmpeg"]
        assert all(versions[name] == pin[name] for name in names)
        run = json.loads((ROOT / f"validation/native_{kind}_run.json").read_text())
        folder = ROOT / f"validation/native_{kind}_v{run['response']['versionNumber']}"
        packages = {
            k.lower().replace("_", "-"): v
            for k, v in json.loads(verified_raw(folder / "runtime_packages.json")).items()
        }
        assert all(packages[k] == v for k, v in versions.items())
        if not is_comfy:
            for model in observed["models"]:
                backend = model.get("runtime_after_inference")
                if not backend:
                    continue
                identity = backend["sdk_runtime"]
                assert (
                    identity["precision"] == "fp16"
                    and identity["component_precisions"] == {"vae": "fp32"}
                    and identity["cpu_offload"] is True
                    and identity["offload_strategy"] == "sequential"
                    and identity["cuda_device"] == "cuda:0"
                )
                assert set(identity["components"]) == {"transformer", "text_encoder", "vae"}
                from model_registry.deployment import ModelCatalog

                selected = ModelCatalog(ROOT / "manifests").get(model["model"])
                expected_tiling = {
                    name: policy.model_dump()
                    for name, policy in selected.runtime_profiles[
                        "kaggle-dual-t4"
                    ].component_tiling.items()
                }
                assert identity.get("component_tiling") == expected_tiling, (
                    "Wan attempt does not attest the current explicit component tiling policy"
                )
                overrides = selected.weights.model_dump(mode="json")[
                    "component_parameter_precisions"
                ]
                assert identity["component_parameter_precisions"] == overrides
                for component, parameters in overrides.items():
                    assert identity["components"][component][
                        "parameter_precision_overrides"
                    ] == dict.fromkeys(parameters, "torch.float32")
                assert all(
                    v["dtypes"]
                    == (["torch.float32"] if name == "vae" else ["torch.float16", "torch.float32"])
                    and v["shared_between_wrappers"] is True
                    and v["parameter_count"] > 1_000_000
                    for name, v in identity["components"].items()
                ), "Wan components violated the exact mixed precision Manifest contract"
                endpoint_cases = [
                    c for c in model["cases"]
                    if c["capability"] == "video.first_last_frame_to_video" and c["runs"]
                ]
                if endpoint_cases:
                    from benchmarks.native.wan_endpoint_evidence import verify
                    from factory.adapters.wan.endpoint_conditioning import runtime_identity

                    assert identity.get("endpoint_conditioning") == runtime_identity()
                    marker = "FACTORY_WAN_ENDPOINT_CONDITIONING "
                    records = [
                        json.loads(line.split(marker, 1)[1])
                        for line in verified_raw(folder / model["model"] / "backend.log")
                        .decode(errors="replace").splitlines()
                        if marker in line
                    ]
                    for native_case in endpoint_cases:
                        definition = next(
                            c for c in observed["case_definitions"] if c["id"] == native_case["case_id"]
                        )
                        for sample in native_case["runs"]:
                            output = json.loads(verified_raw(folder / model["model"] / sample["output_file"]))
                            verify(definition, sample, output, records, observed["input_assets"])
        if is_comfy:
            from model_registry.deployment import ModelCatalog

            from factory.adapters.minimax.native_nodes import SOURCE_FILES, SOURCE_REVISION

            for model in observed["models"]:
                backend = model.get("runtime_after_inference")
                if not backend:
                    continue
                assert backend["source_revision"] == SOURCE_REVISION == pin[
                    "source_revision"
                ] and backend["devices"] == [0]
                files = backend["adapter_assets"]
                for name, sha in files.items():
                    assert (
                        hashlib.sha256(
                            verified_raw(folder / model["model"] / "adapter_assets" / name)
                        ).hexdigest()
                        == sha
                    )
                for name, sha in SOURCE_FILES.items():
                    key = next(
                        k for k in files if k.endswith("/minimax_upstream_" + Path(name).name)
                    )
                    assert files[key] == sha
                key = next(k for k in files if k.endswith("/factory_minimax_loader/__init__.py"))
                assert (
                    files[key]
                    == hashlib.sha256(
                        (ROOT / "factory/adapters/minimax/loader_node.py").read_bytes()
                    ).hexdigest()
                )
                marker = "FACTORY_MINIMAX_RUNTIME "
                attestations = [
                    json.loads(line.split(marker, 1)[1])
                    for line in verified_raw(folder / model["model"] / "backend.log")
                    .decode(errors="replace")
                    .splitlines()
                    if marker in line
                ]
                assert attestations and model.get("precision_attestations") == attestations, (
                    "Missing exact MiniMax FP32 runtime attestation; diagnostic attempt is not V3"
                )
                selected = ModelCatalog(ROOT / "manifests").get(model["model"])
                assert selected.weights.default_precision == "fp32"
                profile = selected.runtime_profiles["kaggle-dual-t4"]
                if profile.extra_vram_headroom_gb:
                    argv = model.get("owned_comfy_launch_argv", [])
                    assert (
                        argv.count("--vram-headroom") == 1
                        and argv[argv.index("--vram-headroom") + 1]
                        == f"{profile.extra_vram_headroom_gb:.3f}"
                    ), "Missing actual DynamicVRAM per-device headroom policy"
                    topology = ModelCatalog(ROOT / "manifests").loader.load_hardware_topologies()[
                        "kaggle-dual-t4"
                    ]
                    reserve = (
                        max(
                            g.total_vram_gb - g.max_allocatable_vram_gb
                            for g in topology.gpus
                            if g.index in backend["devices"]
                        )
                        + profile.extra_vram_headroom_gb
                    )
                    assert (
                        argv.count("--reserve-vram") == 1
                        and argv[argv.index("--reserve-vram") + 1] == f"{reserve:.3f}"
                    ), (
                        "Missing actual owned Comfy launch argv for the explicit Manifest VRAM headroom policy"
                    )
                checkpoint = Path(
                    selected.weights.files["primary"].destination
                    or selected.weights.files["primary"].path
                ).name
                assert all(
                    a["precision"] == "fp32"
                    and a["compute_dtype"] == "torch.float32"
                    and a["manual_cast_dtype"] in {"None", "torch.float32"}
                    and a["load_device"] == "cuda:0"
                    and a["offload_device"] == "cpu"
                    and a["checkpoint"] == checkpoint
                    for a in attestations
                )
        for name, spec in observed["input_assets"].items():
            raw = verified_raw(folder / "inputs" / name / "source.bin")
            assert (
                len(raw) == spec["size_bytes"] and hashlib.sha256(raw).hexdigest() == spec["sha256"]
            )
            inputs[name] = raw
        needs_audio = any("spoken_audio" in c["expected"] for c in observed["case_definitions"])
        if needs_audio:
            evidence = observed.get("audio_semantic_verifiers")
            audio_pins = json.loads(
                (ROOT / "benchmarks/native/video_audio_semantic_pins.json").read_text()
            )
            assert (
                evidence
                and evidence["pins_sha256"] == digest(audio_pins)
                and evidence["scope"] == audio_pins["scope"]
            )
            pin = audio_pins["whisper"]
            assert setup["runtime_pins"]["video_whisper"] == pin
            assert evidence["verified_files"] == {
                "base.en.pt": {
                    "sha256": pin["sha256"],
                    "size_bytes": pin["size_bytes"],
                    "matched": True,
                }
            }
            assert evidence["loaded"] == {
                "parameter_count": 71825408,
                "devices": ["cpu"],
                "dtypes": ["torch.float32"],
            }
            assert (
                evidence["runtime_versions"]["openai-whisper"]
                == packages["openai-whisper"]
                == pin["version"]
            )
            assert evidence["runtime_versions"]["torch"] == versions["torch"]
        verifier = observed.get("semantic_verifiers")
        if verifier:
            pins = json.loads((ROOT / "benchmarks/native/image_semantic_pins.json").read_text())
            assert verifier["pins_sha256"] == digest(pins) and verifier["scope"] == pins["scope"]
            assert verifier["verified_files"] == {
                name: {**spec, "matched": True} for name, spec in pins["files"].items()
            }
            assert verifier["loaded"] == {
                "parameter_count": 151277313,
                "devices": ["cpu"],
                "dtypes": ["torch.float32"],
            }

    result = evaluate(
        wave="d",
        models=("wan-2.2", "minimax-h3", "minimax-h3-turbo", "minimax-h3-reference"),
        kinds=(
            "wan_t2v",
            "wan_i2v",
            "wan_fl2v",
            "wan_extension",
            "minimax_t2v",
            "minimax_conditioned",
            "minimax_turbo",
            "minimax_reference",
            "minimax_audio",
            "minimax_reference_audio",
        ),
        validator=lambda case, output: semantic(case, output, assets=inputs),
        report_validator=verify_report,
    )
    from benchmarks.native.video_content_review import apply

    result = apply(result, ROOT)
    previous = json.loads((ROOT / "validation/native_wave_c_gate.json").read_text())
    if previous["exit_gate"] != "PASS":
        result["exit_gate"] = "NOT_MET"
        result["findings"].append("Wave C dependency gate is not PASS")
    (ROOT / "validation/native_wave_d_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "exit_gate": result["exit_gate"],
                "native_pass": sum(c["status"] == "PASS" for c in result["cases"]),
                "findings": result["findings"],
            }
        )
    )
    return 0 if result["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
