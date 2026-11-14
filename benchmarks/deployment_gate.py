"""Phase 3 configuration exit gate; does not run backends or GPU inference."""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.capability_gate import evaluate as capability_gate
from model_registry.cli import factory_main
from model_registry.deployment import ModelCatalog, DeploymentRegistry, read_config
from model_registry.deployment_validator import DeploymentValidator, inspect_environment


def evaluate():
    failures, examples, sources = [], [], []
    catalog = ModelCatalog(ROOT / "manifests")
    for name in ("dual-t4-shared", "dual-t4-exclusive"):
        path = ROOT / f"deployments/examples/{name}.yaml"
        registry = DeploymentRegistry(read_config(path), catalog, config_path=path)
        report = DeploymentValidator(registry).validate(config_only=True)
        examples.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "validation": report})
        if not report["configuration_valid"] or registry.active_models():
            failures.append(f"{name}: configuration failed or falsely activated")
    for name in ("policy", "manager", "leasing", "scheduler", "watchdog", "model_runtime"):
        path = ROOT / f"factory/apps/control-api/src/queue/{name}.py"
        sources.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.id] if isinstance(node, ast.Name) else [item.name for item in node.names] if isinstance(node, ast.ImportFrom) else []
            if any(name in {"SUBSYSTEM_GPU_MAP", "get_subsystem_gpu"} for name in names):
                failures.append(f"{name}: subsystem placement remains in scheduling")
    commands = []
    with tempfile.TemporaryDirectory(prefix="factory-deployment-gate-") as temp:
        path = Path(temp) / "session.yaml"
        cases = [(["deployment", "init"], 0),
            (["deployment", "add", "qwen-decision", "--gpu", "0"], 0),
            (["deployment", "add", "qwen-embedding", "--gpu", "0"], 0),
            (["deployment", "add", "qwen-reranker", "--gpu", "1"], 0),
            (["deployment", "validate", "--config-only"], 0),
            (["deployment", "add", "qwen-image-2.1", "--gpu", "0"], 2),
            (["deployment", "show"], 0), (["status"], 0),
            *[(["deployment", "remove", mid], 0) for mid in ("qwen-decision", "qwen-embedding", "qwen-reranker")],
            (["deployment", "add", "qwen-text", "--gpu", "0,1", "--exclusive"], 0),
            (["deployment", "validate", "--config-only"], 0)]
        for args, expected in cases:
            stdout, stderr = io.StringIO(), io.StringIO()
            before = path.read_bytes() if path.is_file() else None
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = factory_main(["--manifests-dir", str(ROOT / "manifests"), "--deployment", str(path), *args])
            commands.append({"arguments": args, "expected_exit": expected, "actual_exit": code})
            if code != expected:
                failures.append(f"CLI command failed: {args}")
            if expected != 0 and path.read_bytes() != before:
                failures.append("Rejected CLI mutation changed configuration")
    prior = capability_gate()
    if prior["exit_gate"] != "PASS":
        failures.append("Phase 1/2 regression gate failed")
    environment = inspect_environment()
    return {"phase": 3, "exit_gate": "PASS" if not failures else "NOT_MET",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "CONFIGURATION_AND_SOFTWARE_ONLY", "examples": examples,
        "cli_checks": commands, "scheduling_sources": sources, "findings": failures,
        "prior_model_truth_and_capability_gates": prior["exit_gate"],
        "local_host_observation": {"gpus": environment.gpus, "error": environment.error,
            "installed_backend_providers": sorted(environment.backends), "installed_adapter_providers": sorted(environment.adapters)},
        "backend_lifecycle": "NOT_EXECUTED_BY_THIS_CONFIGURATION_GATE", "active_models": [], "production_ready": False,
        "native_inference_executed": False, "full_factory_e2e_executed": False,
        "reproduce": ".venv/bin/python benchmarks/deployment_gate.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/deployment_gate.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("exit_gate", "classification", "findings", "local_host_observation")}))
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
