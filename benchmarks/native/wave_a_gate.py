"""Recheck native Wave A artifacts and semantics. Never authorizes public READY."""

import argparse
import copy
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from model_registry.deployment import ModelCatalog

from benchmarks.native.wave_a import digest, semantic

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen-decision", "qwen-embedding", "qwen-reranker", "qwen-text")


def evaluate(
    root=ROOT,
    *,
    wave="a",
    models=MODELS,
    kinds=("vllm", "text"),
    validator=semantic,
    report_validator=None,
):
    suite = json.loads((root / f"benchmarks/native/golden_wave_{wave}.json").read_text())
    catalog = ModelCatalog(root / "manifests")
    expected = {case["id"]: case for case in suite["cases"]}
    rows = []
    findings = []
    attempts = []
    declared = {
        (model_id, capability)
        for model_id in models
        for capability, enabled in catalog.get(model_id).capabilities.items()
        if enabled
    }
    covered = {(case["model"], case["capability"]) for case in suite["cases"]}
    if len(expected) != len(suite["cases"]):
        findings.append("Duplicate golden case identifiers")
    if covered != declared:
        findings.append(
            f"Golden coverage differs from declared Wave {wave.upper()} capabilities: "
            + repr(
                {"missing": sorted(declared - covered), "undeclared": sorted(covered - declared)}
            )
        )
    for kind in kinds:
        run_path = root / f"validation/native_{kind}_run.json"
        if not run_path.exists():
            findings.append(f"{kind}: native run not started")
            continue
        run = json.loads(run_path.read_text())
        version = run["response"]["versionNumber"]
        folder = root / f"validation/native_{kind}_v{version}"
        try:
            receipt = json.loads(
                (root / f"validation/native_{kind}_receipt_v{version}.json").read_text()
            )
            assert (
                receipt["version"] == version
                and receipt["kind"] == kind
                and receipt["kernel_url"] == run["response"]["url"]
                and receipt["run_record_sha256"]
                == hashlib.sha256(run_path.read_bytes()).hexdigest()
            )
            assert receipt["kernel_status"]["status"] in {"complete", "error", "failed"}
            assets = {item["file"]: item for item in receipt["artifacts"]}

            def verified_raw(path, assets=assets):
                row = assets[str(path.relative_to(root))]
                assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), (
                    "Collected artifact hash mismatch"
                )
                return path.read_bytes()

            def verified(path):
                return json.loads(verified_raw(path))

            setup = verified(folder / "setup.json")
            assert (
                setup["source_bundle_sha256"] == run["source_bundle_sha256"]
                and setup["validation_bundle_sha256"] == run["validation_bundle_sha256"]
            )
            if not (folder / f"wave_{wave}.json").is_file():
                raise ValueError("Native runtime never reached model validation")
            observed = verified(folder / f"wave_{wave}.json")
            assert (
                setup["classification"]
                == observed["classification"]
                == "REAL_NATIVE_VALIDATION_ATTEMPT"
            ), "Not native evidence"
            assert observed["fallback"] is False and setup["full_factory_e2e_executed"] is False
            assert setup["production_ready"] is False
            assert (
                observed["production_ready"] is False
                and observed["full_factory_e2e_executed"] is False
            ), "Native evidence cannot authorize production or Full Factory E2E"
            versions = observed["runtime_versions"]
            pins = setup["runtime_pins"]
            if report_validator is not None:
                report_validator(kind, setup, observed, verified_raw)
            if kind in {"vllm", "vlm"}:
                assert versions["vllm"] == pins["vllm"]["version"], "Native vLLM version drift"
            elif kind == "asr":
                expected_versions = {
                    "qwen-asr": pins["qwen_asr"]["version"],
                    "torch": pins["qwen_asr"]["torch"],
                    "transformers": pins["qwen_asr"]["transformers"],
                    "huggingface-hub": pins["qwen_asr"]["huggingface_hub"],
                }
                assert all(
                    versions[name] == pinned for name, pinned in expected_versions.items()
                ), "Native ASR runtime version drift"
            semantic_verifiers_verified = False
            if kind == "tts":
                selected = pins["vllm_omni"]
                expected_versions = {
                    "vllm": selected["vllm_version"],
                    "vllm-omni": selected["version"],
                    "torch": selected["torch"],
                    "transformers": selected["transformers"],
                }
                assert all(versions[name] == pin for name, pin in expected_versions.items()), (
                    "Native TTS runtime version drift"
                )
                verifiers = observed.get("semantic_verifiers")
                if verifiers is not None:
                    semantic_pins = json.loads(
                        (root / "benchmarks/native/semantic_pins.json").read_text()
                    )
                    assert verifiers["pins_sha256"] == digest(semantic_pins)
                    verifier_files = {
                        "whisper/base.en.pt": semantic_pins["whisper"],
                        **{
                            "speaker/" + name: spec
                            for name, spec in semantic_pins["speaker"]["files"].items()
                        },
                    }
                    assert set(verifiers["verified_files"]) == set(verifier_files)
                    assert all(
                        verifiers["verified_files"][name]
                        == {
                            "sha256": spec["sha256"],
                            "size_bytes": spec["size_bytes"],
                            "matched": True,
                        }
                        for name, spec in verifier_files.items()
                    )
                    assert set(verifiers["loaded"]) == {"whisper", "speaker"}
                    assert all(
                        m["devices"] == ["cpu"]
                        and m["dtypes"] == ["torch.float32"]
                        and m["parameter_count"] > 1_000_000
                        for m in verifiers["loaded"].values()
                    )
                    semantic_verifiers_verified = True
            if wave == "a":
                assert observed["cases_sha256"] == digest(suite), "Golden case drift"
            else:
                definitions = observed["case_definitions"]
                input_assets = observed["input_assets"]
                assert definitions and len({c["id"] for c in definitions}) == len(definitions)
                assert all(c == expected[c["id"]] for c in definitions), "Golden case drift"
                assert input_assets == {
                    name: suite["assets"][name]
                    for c in definitions
                    for name in ([c["asset"]] if c.get("asset") else c.get("assets", []))
                }
                assert observed["cases_sha256"] == digest(
                    {"cases": definitions, "assets": input_assets}
                )
                for asset_name, spec in input_assets.items():
                    base = folder / "inputs" / asset_name
                    if kind == "asr" and not base.exists():
                        base = folder / "inputs"
                    source = base / (
                        "source.flac" if "decoded_wav_sha256" in spec else "source.bin"
                    )
                    raw = verified_raw(source)
                    assert (
                        len(raw) == spec["size_bytes"]
                        and hashlib.sha256(raw).hexdigest() == spec["sha256"]
                    )
                    assert verified(base / "input_identity.json") == spec
            assert set(observed["hardware"]) == {"0", "1"} and all(
                "t4" in row["name"].lower() and row["total_vram_gb"] >= 14.2
                for row in observed["hardware"].values()
            )
            attempts.append(
                {
                    "kind": kind,
                    "version": version,
                    "observation": str((folder / f"wave_{wave}.json").relative_to(root)),
                    "source_bundle_sha256": run["source_bundle_sha256"],
                    "validation_bundle_sha256": run["validation_bundle_sha256"],
                }
            )
            for model_observed in observed["models"]:
                assert (
                    model_observed["production_ready"] is False
                    and model_observed["full_factory_e2e_executed"] is False
                    and model_observed["public_models"] == []
                ), "Native model evidence cannot authorize public models"
                model = catalog.get(model_observed["model"])
                assert model_observed == verified(
                    folder / model.metadata.id / "observation.json"
                ), "Aggregate and per-model native evidence disagree"
                wanted = {
                    key: (s.repository, s.revision, s.path, s.sha256, s.size_bytes)
                    for key, s in model.weights.files.items()
                }
                files = model_observed["checkpoint_files"]
                actual = {
                    s["key"]: (
                        s["repository"],
                        s["revision"],
                        s["file"],
                        s["sha256"],
                        s["size_bytes"],
                    )
                    for s in files
                }
                matched = (
                    actual == wanted
                    and len(files) == len(wanted)
                    and all(s["matched"] is True for s in files)
                )
                backend = model_observed.get("runtime_after_inference", {})
                if kind == "tts":
                    import yaml

                    path = folder / model.metadata.id / "adapter_assets/qwen3_tts_deploy.yaml"
                    raw = verified_raw(path)
                    assert (
                        hashlib.sha256(raw).hexdigest()
                        == backend["adapter_assets"]["qwen3_tts_deploy.yaml"]
                    )
                    config = yaml.safe_load(raw)
                    stages = config["stages"]
                    assert config["model_runner"] == "v1" and [s["stage_id"] for s in stages] == [
                        0,
                        1,
                    ]
                    assert all(
                        s["dtype"] == "float16"
                        and s["devices"] == "0"
                        and s["max_num_seqs"] == 1
                        and s["enforce_eager"] is True
                        and s["async_scheduling"] is False
                        for s in stages
                    )
                    topology = catalog.loader.load_hardware_topologies()["kaggle-dual-t4"]
                    total = topology.gpus[0].total_vram_gb
                    assert (
                        sum(s["gpu_memory_utilization"] for s in stages) * total
                        <= model.runtime_profiles["kaggle-dual-t4"].min_vram_gb + 0.005
                    )
                for case in (
                    c
                    for c in expected.values()
                    if c["model"] == model.metadata.id
                    and (wave == "a" or c["id"] in {d["id"] for d in definitions})
                ):
                    native = next(
                        (c for c in model_observed["cases"] if c["case_id"] == case["id"]), None
                    )
                    result = {
                        "run_id": f"kaggle:{run['slug']}:v{version}:{case['id']}",
                        "model": model.metadata.id,
                        "capability": case["capability"],
                        "backend": model.backend.supported[0],
                        "checkpoint": model.weights.files["primary"].repository,
                        "revision": model.weights.files["primary"].revision,
                        "hardware": "kaggle-dual-t4",
                        "input_sha256": digest(case),
                        "weights_verified": matched,
                        "commit": json.loads((root / "validation/phase_status.json").read_text())[
                            "source_commit"
                        ],
                        "commit_scope": "baseline anchor; actual uncommitted code identified by source_bundle_sha256",
                        "source_bundle_sha256": run["source_bundle_sha256"],
                        "runtime_versions": versions,
                        "fallback": False,
                        "status": "NOT_MET",
                        "checks": [],
                        "native_inference_verified": False,
                        "production_ready": False,
                        "full_factory_e2e_executed": False,
                        "case_evidence": str(
                            (folder / model.metadata.id / "observation.json").relative_to(root)
                        ),
                    }

                    def check(name, passed, result=result):
                        result["checks"].append({"check": name, "passed": bool(passed)})

                    check("complete_exact_checkpoint_graph", matched)
                    if kind == "tts":
                        check("exact_native_semantic_verifiers", semantic_verifiers_verified)
                    check(
                        "manifest_declares_capability",
                        model.capabilities.get(case["capability"]) is True,
                    )
                    check(
                        "owned_backend_runtime",
                        backend.get("model") == model.metadata.id
                        and backend.get("backend") == model.backend.supported[0]
                        and backend.get("weights_verified") is True
                        and backend.get("adapter") == model.backend.adapter
                        and backend.get("state") == "HEALTHY"
                        and not model_observed.get("lifecycle_after_inference", {}).get("failures")
                        and model_observed.get("fallback") is False,
                    )
                    check(
                        "clean_owned_process_shutdown", model_observed.get("stop_verified") is True
                    )
                    if native is None:
                        result["error"] = model_observed.get(
                            "error", {"message": "No actual native output"}
                        )
                        rows.append(result)
                        continue
                    check(
                        "exact_golden_input_no_fallback",
                        native["input_sha256"] == digest(case) and native["fallback"] is False,
                    )
                    runs = native["runs"]
                    check(
                        "one_warmup_three_measured",
                        len(runs) == 4
                        and [r["role"] for r in runs]
                        == ["warmup", "measured", "measured", "measured"],
                    )
                    semantics = []
                    latencies = []
                    hashes = []
                    for item in runs:
                        output_path = folder / model.metadata.id / item["output_file"]
                        assert output_path.resolve().is_relative_to(
                            (folder / model.metadata.id).resolve()
                        )
                        actual_output = verified(output_path)
                        assert (
                            item["output_sha256"]
                            == hashlib.sha256(output_path.read_bytes()).hexdigest()
                        )
                        semantics.append(validator(case, actual_output))
                        if "seed_sequence" in case:
                            iteration = item["iteration"]
                            executed = copy.deepcopy(case)
                            executed["request"]["seed"] = case["seed_sequence"][iteration]
                            check(
                                "predeclared_distinct_seed_" + str(iteration),
                                len(case["seed_sequence"]) == len(set(case["seed_sequence"])) == 4
                                and item["seed"]
                                == executed["request"]["seed"]
                                == actual_output["seed"]
                                and item["executed_input_sha256"] == digest(executed),
                            )
                        hashes.append(item["output_sha256"])
                        devices = (
                            {0, 1}
                            if model.runtime_profiles["kaggle-dual-t4"].gpu_count == 2
                            else {0}
                        )
                        check(
                            "assigned_gpu_telemetry_" + str(item["iteration"]),
                            bool(item["gpu_samples"])
                            and all(
                                {
                                    p["device"]
                                    for p in s["gpu_processes"]
                                    if p["pid"] > 0 and p["used_memory_mb"] > 0
                                }
                                == devices
                                for s in item["gpu_samples"]
                            ),
                        )
                        check(
                            "positive_measured_latency_" + str(item["iteration"]),
                            isinstance(item["latency_ms"], (float, int)) and item["latency_ms"] > 0,
                        )
                        if item["role"] == "measured":
                            latencies.append(item["latency_ms"])
                    check(
                        "semantic_validation_all_runs",
                        len(semantics) == 4 and all(s["status"] == "PASS" for s in semantics),
                    )
                    if len(latencies) == 3:
                        statistics_expected = {
                            "median": statistics.median(latencies),
                            "min": min(latencies),
                            "max": max(latencies),
                        }
                        check(
                            "latency_statistics_from_actual_runs",
                            native["latency_ms"] == statistics_expected,
                        )
                        result["latency_ms"] = statistics_expected
                    result["output_sha256"] = hashes
                    all_samples = [sample for run in runs for sample in run["gpu_samples"]]
                    device_ids = {
                        p["device"] for sample in all_samples for p in sample["gpu_processes"]
                    }
                    result["peak_vram_mb_by_device"] = {
                        str(device): max(
                            sum(
                                p["used_memory_mb"]
                                for p in sample["gpu_processes"]
                                if p["device"] == device
                            )
                            for sample in all_samples
                        )
                        for device in sorted(device_ids)
                    }
                    result["peak_total_owned_vram_mb"] = max(
                        (
                            sum(p["used_memory_mb"] for p in sample["gpu_processes"])
                            for sample in all_samples
                        ),
                        default=0,
                    )
                    topology = catalog.loader.load_hardware_topologies()["kaggle-dual-t4"]
                    limits = {str(g.index): g.max_allocatable_vram_gb * 1024 for g in topology.gpus}
                    check(
                        "resident_memory_within_selected_hardware_budget",
                        all(
                            memory <= limits[device]
                            for device, memory in result["peak_vram_mb_by_device"].items()
                        ),
                    )
                    result["semantic_validation"] = semantics
                    result["error"] = native.get("error")
                    if result["checks"] and all(c["passed"] for c in result["checks"]):
                        result.update(status="PASS", native_inference_verified=True)
                    rows.append(result)
        except Exception as exc:
            findings.append(f"{kind}: {type(exc).__name__}: {exc}")
    represented = {r["model"] for r in rows}
    for model in models:
        if model not in represented:
            findings.append(f"{model}: no collected native case evidence")
    failures = [r for r in rows if r["status"] != "PASS"]
    if failures:
        findings.append(f"{len(failures)} native capabilities have not met the gate")
    return {
        "wave": wave.upper(),
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "exit_gate": "PASS" if not findings and len(rows) == len(expected) else "NOT_MET",
        "classification": "NATIVE_V3_SEMANTIC_CANDIDATES_ONLY",
        "golden_sha256": digest(suite),
        "attempts": attempts,
        "cases": rows,
        "findings": findings,
        "production_ready": False,
        "full_factory_e2e_executed": False,
        "phase_8_exit_gate": "NOT_MET (remaining native waves require validation)",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/native_wave_a_gate.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
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
