"""Phase 4 lifecycle evidence gate. Never authorizes model inference/READY."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.deployment_gate import evaluate as deployment_gate
from deployments.kaggle.backend_lifecycle_probe import render_kernel, source_bundle
from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
from model_registry.deployment import ModelCatalog, DeploymentRegistry


def evaluate():
    findings, checks = [], []
    def check(name, passed):
        checks.append({"check":name, "passed":bool(passed)})
        if not passed:
            findings.append(name)
    prior = deployment_gate()
    check("previous_phase_gates", prior["exit_gate"] == "PASS")
    observation_path = None
    try:
        run_path = ROOT / "validation/backend_lifecycle_run.json"
        run = json.loads(run_path.read_text())
        version = run["response"]["versionNumber"]
        receipt_path = ROOT / f"validation/backend_lifecycle_receipt_v{version}.json"
        observation_path = ROOT / f"validation/backend_lifecycle_observation_v{version}.json"
        receipt = json.loads(receipt_path.read_text())
        observation = json.loads(observation_path.read_text())
        check("recorded_exact_kernel", run["kernel_code_sha256"] == hashlib.sha256(render_kernel().encode()).hexdigest())
        check("current_source_bundle", observation["source_bundle_sha256"] == hashlib.sha256(source_bundle()).hexdigest())
        check("completed_kaggle_dual_t4_run", run["accelerator"] == "NvidiaTeslaT4" and receipt["kernel_status"]["status"] == "complete"
            and receipt["version"] == version and receipt["kernel_url"] == run["response"]["url"]
            and receipt["run_record_sha256"] == hashlib.sha256(run_path.read_bytes()).hexdigest())
        artifact = [item for item in receipt["artifacts"] if item["remote_name"] == "phase4_lifecycle.json"]
        check("collected_observation_integrity", len(artifact) == 1 and artifact[0]["sha256"] == hashlib.sha256(observation_path.read_bytes()).hexdigest()
            and artifact[0]["file"] == str(observation_path.relative_to(ROOT)))
        check("real_lifecycle_classification", observation["classification"] == "REAL_BACKEND_LIFECYCLE_ONLY")
        hardware = observation["hardware"]
        check("two_real_t4_devices", set(hardware) == {"0", "1"} and all("t4" in row["name"].lower() and row["total_vram_gb"] >= 14.2 for row in hardware.values()))
        catalog = ModelCatalog(ROOT / "manifests")
        model = catalog.get("qwen-asr")
        binding = DeploymentRegistry(DeploymentConfig(deployments=(DeploymentEntry(model="qwen-asr",devices=(0,)),)),catalog).assignment("qwen-asr")
        files = observation["checkpoint_files"]
        expected = {key:(spec.repository,spec.revision,spec.path,spec.sha256,spec.size_bytes) for key,spec in model.weights.files.items()}
        actual = {row["key"]:(row["repository"],row["revision"],row["file"],row["sha256"],row["size_bytes"]) for row in files}
        check("complete_exact_checkpoint_bundle", actual == expected and len(files) == len(expected) and all(row["matched"] is True for row in files))
        session = observation["session"]
        native = session["backend_lifecycle"]["backends"]["qwen-asr"]
        check("configured_provider_loaded_on_assigned_gpu", session["healthy_backends"] == ["qwen-asr"] and native["state"] == "HEALTHY"
            and native["model"] == "qwen-asr" and native["backend"] == "qwen_asr" and native["adapter"] == model.backend.adapter
            and native["devices"] == [0] and native["weights_verified"] is True and native["model_loaded"] is True
            and native["checkpoint_graph_sha256"] == binding.checkpoint_graph_sha256
            and native["session_id"] == session["session_id"] and native["pid"] > 0 and native["pid_started_at"] > 0
            and native["gpu_processes"] and {row["device"] for row in native["gpu_processes"]} == {0}
            and all(row["pid"] > 0 and row["used_memory_mb"] > 0 for row in native["gpu_processes"]))
        check("runtime_identity_recorded", bool(native["executable_sha256"] and native["backend_package_version"] and observation["runtime_versions"]))
        sdk = native["sdk_runtime"]
        check("real_asr_parameters_on_cuda", sdk["implementation"] == "transformers" and sdk["parameter_count"] > 0
            and sdk["parameter_devices"] == ["cuda:0"] and sdk["parameter_dtypes"] == ["torch.float16"])
        check("unvalidated_public_inference_closed", session["active_models"] == [] and observation["public_discovery"] == {"object":"list","data":[]}
            and observation["unvalidated_inference_rejected"] is True)
        shutdown = observation["shutdown_receipt"]
        check("graceful_owned_process_cleanup", observation["stop_verified"] is True and observation["api_exit_code"] in (0,-15) and observation["backend_processes_leaked"] == []
            and shutdown["stop_completed"] is True and shutdown["errors"] == [] and shutdown["session_id"] == session["session_id"]
            and shutdown["config_sha256"] == session["config_sha256"] and shutdown["backends"]["qwen-asr"]["state"] == "STOPPED"
            and shutdown["backends"]["qwen-asr"]["pid"] == native["pid"])
        check("live_checks_passed_without_error", observation["phase4_exit_gate"] == "PASS" and observation["launch_health_identity_gpu_checks"] is True and "error" not in observation)
        check("no_inference_or_production_claim", all(observation[key] is False for key in ["native_inference_executed","semantic_validation_executed","full_factory_e2e_executed","production_ready"])
            and native["inference_verified"] is False and native["production_ready"] is False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        findings.append(f"Missing or invalid current lifecycle evidence: {type(exc).__name__}: {exc}")
    return {"phase":4, "observed_at_utc":datetime.now(timezone.utc).isoformat(), "exit_gate":"PASS" if not findings else "NOT_MET",
        "classification":"REAL_BACKEND_LIFECYCLE_ONLY", "checks":checks, "findings":findings,
        "observation":str(observation_path.relative_to(ROOT)) if observation_path else None,
        "scope":"Selected ASR checkpoint load, current-session process/health/GPU identity and shutdown through factory serve; other providers need their own runtime validation",
        "production_ready":False, "native_inference_executed":False, "semantic_validation_executed":False, "full_factory_e2e_executed":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/backend_lifecycle_gate.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key:report[key] for key in ["exit_gate","classification","findings","observation"]}))
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
