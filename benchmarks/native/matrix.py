"""Generate truthful native progress from canonical Manifest and evidence JSON."""

import json
from pathlib import Path
from model_registry.deployment import ModelCatalog

ROOT = Path(__file__).resolve().parents[2]


def main():
    evidence_path = ROOT / "validation/native_wave_a_gate.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {"cases": []}
    observed = {(row["model"], row["capability"]): row for row in evidence["cases"]}
    for wave in ("b", "c", "d", "e", "f"):
        path = ROOT / f"validation/native_wave_{wave}_gate.json"
        if path.exists():
            gate = json.loads(path.read_text())
            observed.update({(row["model"], row["capability"]): row for row in gate["cases"]})
    rows = []
    models = ModelCatalog(ROOT / "manifests").models()
    for model in models.values():
        for capability, declared in model.capabilities.items():
            native = observed.get((model.metadata.id, capability))
            rows.append(
                {
                    "model": model.metadata.id,
                    "capability": capability,
                    "declared": declared,
                    "model_state": model.validation.state.value,
                    "v3": "PASS"
                    if native and native["status"] == "PASS"
                    else model.validation.state.value
                    if model.validation.state.value in {"DISABLED", "BLOCKED"}
                    else "NOT_MET"
                    if native
                    else "NOT_RUN"
                    if declared
                    else "UNSUPPORTED_OR_DISABLED",
                    "semantic": "PASS" if native and native["status"] == "PASS" else "NOT_VERIFIED",
                    "v4": "NOT_RUN",
                    "production_ready": False,
                    "native_run_id": native["run_id"] if native else None,
                    "evidence": native.get("case_evidence") if native else None,
                }
            )
    report = {
        "classification": "NATIVE_PROGRESS_ONLY_NOT_PRODUCTION_AUTHORIZATION",
        "models": len(models),
        "matrix": rows,
        "native_pass_pairs": sum(r["v3"] == "PASS" for r in rows),
        "production_ready_pairs": 0,
    }
    (ROOT / "validation/native_matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Native validation progress",
        "",
        "Generated from `validation/native_matrix.json`, canonical Manifest and collected native evidence. Software/lifecycle observations are not promoted to inference. V3 PASS is scoped to the recorded exact checkpoint/backend/golden input; V4 and production readiness remain unverified.",
        "",
        "| Model | Capability | Declared | V3 | Semantic | V4 | Ready |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["capability"],
                    str(row["declared"]),
                    row["v3"],
                    row["semantic"],
                    row["v4"],
                    "false",
                ]
            )
            + " |"
        )
    lines.extend(
        ["", f"Native PASS pairs: {report['native_pass_pairs']}. Production-ready pairs: 0.", ""]
    )
    (ROOT / "validation/native_matrix.md").write_text("\n".join(lines))
    print(
        json.dumps({"native_pass_pairs": report["native_pass_pairs"], "production_ready_pairs": 0})
    )


if __name__ == "__main__":
    main()
