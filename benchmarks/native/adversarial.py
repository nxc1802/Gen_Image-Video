"""Replay immutable v3 evidence mutations; never executes or certifies inference."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from benchmarks.native.wave_a_gate import evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix="factory-native-evidence-") as td:
        root = Path(td)
        shutil.copytree(ROOT / "manifests", root / "manifests")
        (root / "benchmarks/native").mkdir(parents=True)
        shutil.copyfile(
            ROOT / "benchmarks/native/golden_wave_a.json",
            root / "benchmarks/native/golden_wave_a.json",
        )
        (root / "validation").mkdir()
        for name in ["phase_status.json", "native_vllm_receipt_v3.json"]:
            shutil.copyfile(ROOT / "validation" / name, root / "validation" / name)
        run = ROOT / "validation/native_vllm_run_v3.json"
        if not run.exists():
            run = ROOT / "validation/native_vllm_run.json"
        assert json.loads(run.read_text())["response"]["versionNumber"] == 3
        shutil.copyfile(run, root / "validation/native_vllm_run.json")
        shutil.copytree(ROOT / "validation/native_vllm_v3", root / "validation/native_vllm_v3")
        baseline = evaluate(root)
        assert sum(c["status"] == "PASS" for c in baseline["cases"]) == 5
        checks.append(
            {"case": "unmodified_collected_native_vllm", "passed": True, "expected_pass_pairs": 5}
        )
        folder = root / "validation/native_vllm_v3"
        paths = list(folder.rglob("*.json")) + [root / "validation/native_vllm_receipt_v3.json"]
        original = {p: p.read_bytes() for p in paths}

        def reset():
            for p, b in original.items():
                p.write_bytes(b)

        def save(p, data):
            p.write_text(json.dumps(data, indent=2) + "\n")

        def receipt():
            p = root / "validation/native_vllm_receipt_v3.json"
            r = json.loads(p.read_text())
            for a in r["artifacts"]:
                artifact = root / a["file"]
                if artifact.is_file():
                    a["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            save(p, r)

        def change_native(fn):
            for p in [folder / "wave_a.json", folder / "qwen-embedding/observation.json"]:
                r = json.loads(p.read_text())
                m = (
                    next(x for x in r["models"] if x["model"] == "qwen-embedding")
                    if "models" in r
                    else r
                )
                fn(m)
                save(p, r)

        def verify(name):
            result = evaluate(root)
            embedding = [c for c in result["cases"] if c["model"] == "qwen-embedding"]
            rejected = not any(c["status"] == "PASS" for c in embedding)
            assert rejected, name
            checks.append(
                {
                    "case": name,
                    "passed": rejected,
                    "classification": "REPLAY_MUTATION_ONLY_NOT_NEW_INFERENCE",
                }
            )

        # Corrupt output bytes without changing its receipt.
        p = folder / "qwen-embedding/embedding-france-retrieval-0.json"
        p.write_bytes(p.read_bytes() + b" ")
        verify("output_digest_mismatch_rejected")
        reset()
        # Match all digests, while changing actual semantic answer.
        hashes = {}
        for p in (folder / "qwen-embedding").glob("embedding-france-retrieval-*.json"):
            r = json.loads(p.read_text())
            r["data"][1]["embedding"], r["data"][2]["embedding"] = (
                r["data"][2]["embedding"],
                r["data"][1]["embedding"],
            )
            save(p, r)
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

        def alter_output(m):
            for case in m["cases"]:
                for run in case["runs"]:
                    run["output_sha256"] = hashes[run["output_file"]]

        change_native(alter_output)
        receipt()
        verify("semantic_wrong_relevance_with_consistent_hashes_rejected")
        reset()

        def alter_checkpoint(m):
            m["checkpoint_files"][0]["revision"] = "0" * 40

        change_native(alter_checkpoint)
        receipt()
        verify("wrong_exact_checkpoint_with_consistent_hashes_rejected")
        reset()

        def alter_gpu(m):
            for c in m["cases"]:
                for r in c["runs"]:
                    for sample in r["gpu_samples"]:
                        for proc in sample["gpu_processes"]:
                            proc["device"] = 1

        change_native(alter_gpu)
        receipt()
        verify("wrong_assigned_gpu_with_consistent_hashes_rejected")
        reset()

        def alter_memory(m):
            for c in m["cases"]:
                for r in c["runs"]:
                    for sample in r["gpu_samples"]:
                        for proc in sample["gpu_processes"]:
                            proc["used_memory_mb"] = 20000.0

        change_native(alter_memory)
        receipt()
        verify("resident_gpu_budget_violation_with_consistent_hashes_rejected")
        reset()

        def alter_rounds(m):
            m["cases"][0]["runs"].pop()

        change_native(alter_rounds)
        receipt()
        verify("missing_measured_native_round_rejected")
        reset()
        p = folder / "setup.json"
        r = json.loads(p.read_text())
        r["production_ready"] = True
        save(p, r)
        receipt()
        verify("unauthorized_production_promotion_rejected")
        reset()
        change_native(lambda m: m.update(production_ready=True))
        receipt()
        verify("model_level_production_promotion_rejected")
        reset()
        p = folder / "wave_a.json"
        r = json.loads(p.read_text())
        r["full_factory_e2e_executed"] = True
        save(p, r)
        receipt()
        verify("native_evidence_cannot_claim_full_factory_e2e")
        reset()
    report = {
        "classification": "SOFTWARE_EVIDENCE_VALIDATOR_REPLAY_ONLY",
        "native_inference_executed": False,
        "source_observation": "validation/native_vllm_v3/wave_a.json",
        "source_receipt": "validation/native_vllm_receipt_v3.json",
        "validator_sha256": hashlib.sha256(
            (ROOT / "benchmarks/native/wave_a_gate.py").read_bytes()
        ).hexdigest(),
        "mutation_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": checks,
        "tamper_cases_rejected": len(checks) - 1,
        "status": "PASS",
        "production_ready": False,
    }
    (ROOT / "validation/native_gate_adversarial.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
