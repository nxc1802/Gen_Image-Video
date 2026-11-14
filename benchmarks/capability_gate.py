"""Phase 2 static vocabulary audit. No backend/GPU/model inference is executed."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from factory_contracts.capabilities import CAPABILITIES, normalize_capability, require_canonical
from factory_contracts.manifests.catalog import load_model_manifests
from benchmarks.model_truth_gate import evaluate_catalog

OWNER = "packages/contracts/src/factory_contracts/capabilities.py"
RUNTIME_ROOTS = ("factory/apps/control-api/src", "factory/core", "factory/engines", "factory/adapters", "packages/contracts/src", "packages/model-registry/src", "services", "workflows")
# Request/config/catalog/storage deserialization boundaries only. Dispatch
# modules may assert canonical IDs; they must not resolve or expand aliases.
NORMALIZATION_BOUNDARIES = frozenset({OWNER,
    "packages/contracts/src/factory_contracts/tasks.py",
    "packages/contracts/src/factory_contracts/jobs/job_contract.py",
    "packages/contracts/src/factory_contracts/jobs/worker_contract.py",
    "packages/contracts/src/factory_contracts/manifests/capability.py",
    "packages/model-registry/src/model_registry/validator.py",
    "factory/apps/control-api/src/queue/repository/memory.py",
    "services/decision/src/schemas.py",
    "factory/engines/decision/contracts.py",  # Official public request DTO ingress.
})


def audit_source(source: str, path: str) -> list[str]:
    tree = ast.parse(source)
    if path == OWNER:
        return []
    findings = []
    imports = {item.asname or item.name: item.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for item in node.names}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                name = target.id if isinstance(target, ast.Name) else ""
                if isinstance(value, ast.Dict) and re.search(r"TASK.*ALIASES|CAPABILITY.*ALIASES|SUBSYSTEM_MANIFEST_CAPABILITIES|CANONICAL_(TASK|TO_CAPABILITY)_MAP", name):
                    findings.append(f"{path}:{node.lineno}: independent alias table {name}")
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in CAPABILITIES:
                    continue
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                for engine in {item.engine for item in CAPABILITIES.values()}:
                    cid = normalize_capability(key.value, engine=engine)
                    if cid in CAPABILITIES and normalize_capability(value.value, engine=engine) == cid:
                        findings.append(f"{path}:{node.lineno}: alias pair {key.value!r} is maintained outside registry")
                        break
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            name = imports.get(name, name)
            if name in {"normalize_task", "normalize_capability", "get_task_aliases", "capability_aliases"} and path not in NORMALIZATION_BOUNDARIES:
                findings.append(f"{path}:{node.lineno}: alias resolution/expansion inside runtime dispatch")
    return sorted(set(findings))


def evaluate(root: Path = ROOT) -> dict:
    findings, sources = [], []
    for directory in RUNTIME_ROOTS:
        for path in sorted((root / directory).rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            relative = path.relative_to(root).as_posix()
            sources.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            findings.extend(audit_source(path.read_text(), relative))
    models = load_model_manifests(root / "manifests")
    for mid, model in models.items():
        for cid in model.capabilities:
            require_canonical(cid, engine=model.engine)
    for path in (root / "services").glob("*/configs/*.yaml"):
        data = yaml.safe_load(path.read_text()) or {}
        for cid in data.get("supported_tasks", []):
            require_canonical(cid, engine=data.get("subsystem"))
    for path in (root / "manifests/capabilities").glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        require_canonical(data["id"], engine=data["subsystem"])
    phase1 = evaluate_catalog(root / "manifests", root / "validation/checkpoint_metadata_current.json")
    if phase1["exit_gate"] != "PASS":
        findings.extend("Phase 1 regression: " + error for error in phase1["errors"])
    return {"phase": 2, "exit_gate": "PASS" if not findings else "NOT_MET",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "STATIC_VOCABULARY_AND_SOFTWARE_CONTRACTS_ONLY",
        "registry_owner": OWNER, "canonical_capabilities": list(CAPABILITIES),
        "capability_count": len(CAPABILITIES), "model_count": len(models),
        "phase_1_checkpoint_graphs": phase1["exit_gate"],
        "normalization_boundaries": sorted(NORMALIZATION_BOUNDARIES),
        "scanned_sources": sources, "findings": findings,
        "scope_note": "Audits production Python/config/catalog vocabulary. Historical benchmark runners, test fixtures and Studio are retained for their ordered migration phases. Native adapter operation bindings are not request aliases.",
        "production_ready": False, "native_inference_executed": False, "full_factory_e2e_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/capability_gate.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("exit_gate", "capability_count", "model_count", "findings")}))
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
