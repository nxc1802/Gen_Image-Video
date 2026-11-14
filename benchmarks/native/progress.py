"""Derive Phase 8 progress from collected gates; never grants production readiness."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / "validation/phase_status.json"
    status = json.loads(path.read_text())
    phase = status["phase_8"]
    matrix = json.loads((ROOT / "validation/native_matrix.json").read_text())
    phase["native_pass_pairs"] = matrix["native_pass_pairs"]
    phase["native_verified_pairs"] = [
        {"model": row["model"], "capability": row["capability"], "run_id": row["native_run_id"]}
        for row in matrix["matrix"]
        if row["v3"] == "PASS"
    ]
    pending, records = [], []
    for run_path in sorted((ROOT / "validation").glob("native_*_run.json")):
        run = json.loads(run_path.read_text())
        version = run["response"]["versionNumber"]
        kind = run_path.stem.removeprefix("native_").removesuffix("_run")
        if kind in {"gate_audit", "minimax_vae_diag", "minimax_native_canvas_diag", "minimax_sampler_diag"}:
            continue
        receipt = ROOT / f"validation/native_{kind}_receipt_v{version}.json"
        relative = str(run_path.relative_to(ROOT))
        records.append(relative)
        collected = json.loads(receipt.read_text()) if receipt.exists() else None
        if (
            collected is None
            or collected["run_record_sha256"] != hashlib.sha256(run_path.read_bytes()).hexdigest()
        ):
            pending.append(relative)
    phase["native_run_records"] = records
    phase["pending_native_runs"] = pending
    for wave in ("A", "B", "C", "D", "E", "F"):
        gate_path = ROOT / f"validation/native_wave_{wave.lower()}_gate.json"
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            phase["waves"][wave] = "PASS" if gate["exit_gate"] == "PASS" else "IN_PROGRESS"
    # C–F and the Phase 8 exit still require their own explicit gates.
    phase["status"] = "IN_PROGRESS"
    phase["exit_gate"] = "NOT_MET"
    phase["production_ready"] = False
    status["production_ready_model_capability_pairs"] = 0
    path.write_text(json.dumps(status, indent=2) + "\n")
    print(
        json.dumps(
            {
                "native_pass_pairs": phase["native_pass_pairs"],
                "pending_native_runs": pending,
                "phase_8_exit_gate": phase["exit_gate"],
                "production_ready_pairs": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
