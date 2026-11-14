"""Reproduce the Phase 1 static model-truth exit gate; never certify inference."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

from factory_contracts.manifests.catalog import load_model_manifests
from model_registry.cli import factory_main
from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator

ROOT = Path(__file__).resolve().parents[1]


def evaluate_catalog(catalog: Path, probe_path: Path) -> dict:
    models = load_model_manifests(catalog)
    errors = []
    if not models:
        errors.append("Model catalog is empty")
    valid, validation_errors = ManifestValidator(ManifestLoader(catalog)).validate_all()
    if not valid:
        errors.extend(validation_errors)
    probe = json.loads(probe_path.read_text())
    observations = {item["model"]: item for item in probe["models"]}
    if set(observations) != set(models) or len(observations) != len(probe["models"]):
        errors.append("Probe inventory does not exactly match current catalog")
    results = []
    for mid, model in models.items():
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = factory_main(["--manifests-dir", str(catalog), "models", "inspect", mid])
        inspected = json.loads(stdout.getvalue()) if code == 0 else {}
        truth = inspected.get("canonical_truth", {})
        checks = {
            "inspector_complete": code == 0 and truth.get("complete") is True,
            "canonical_v2": model.apiVersion == "factory.ai/v2",
            "no_checkpoint_variants": not model.weights.variants and not model.task_variant_matrix,
            "no_static_readiness": not any((model.installed, model.checkpoint_valid, model.backend_ready,
                model.evidence_run_id, model.validated_hardware_profiles)) and
                not any(t.live_verified for t in model.capabilities_truth.values()) and
                inspected.get("current_session") == {"attested": False, "production_ready": False},
        }
        observed = observations.get(mid, {})
        source = ROOT / observed.get("manifest", "missing")
        # Later phases may change capability names/parameters without changing
        # checkpoint identity. Reuse the original upstream observation only when
        # the complete pinned file graph below still matches exactly.
        checks["probe_inventory_matches_model"] = source.is_file() and source.parent.resolve() == (catalog / "models").resolve() and (
            observed.get("model") == mid and observed.get("expected_sha256") == model.weights.sha256)
        files = {item["key"]: item for item in observed.get("file_identities", [])}
        checks["complete_bundle_identity"] = bool(files) and set(files) == set(model.weights.files) and all(
            files[key].get("status") == "IDENTITY_MATCH" and all(files[key].get(field) == expected for field, expected in {
                "provider": spec.provider, "repository": spec.repository, "revision": spec.revision,
                "file": spec.path, "destination": spec.destination or spec.path,
                "expected_sha256": spec.sha256, "expected_size_bytes": spec.size_bytes,
            }.items()) for key, spec in model.weights.files.items())
        for check, passed in checks.items():
            if not passed:
                errors.append(f"{mid}: {check}")
        results.append({"model": mid, "engine": model.engine.value if model.engine else None,
            "current_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
            "upstream_observation_manifest_sha256": observed.get("manifest_sha256"),
            "backend": model.backend.model_dump(mode="json") if model.backend else None,
            "validation_state": model.validation.state.value if model.validation else None,
            "checkpoint_files": len(model.weights.files), "checks": checks})
    return {"phase": 1, "exit_gate": "PASS" if not errors else "NOT_MET",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "STATIC_MODEL_TRUTH_ONLY",
        "authority": "docs/MASTER_PLAN.md#28-phase-1--canonical-model-truth",
        "reproduce": [".venv/bin/python benchmarks/checkpoint_probe.py --output validation/checkpoint_metadata_current.json",
                      ".venv/bin/python benchmarks/model_truth_gate.py"],
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "models": results, "model_count": len(results),
        "model_states": dict(Counter(item["validation_state"] for item in results)),
        "production_ready": False, "native_inference_executed": False,
        "full_factory_e2e_executed": False, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=ROOT / "validation/checkpoint_metadata_current.json")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/model_truth_gate.json")
    args = parser.parse_args()
    report = evaluate_catalog(ROOT / "manifests", args.probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("exit_gate", "model_count", "model_states", "errors")}))
    return 0 if report["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
