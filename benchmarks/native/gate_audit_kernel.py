"""CPU revalidation of receipt-bound native artifacts; never new inference."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

DECODER_SHA256 = "ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41"
CLASSIFICATION = "CPU_NATIVE_EVIDENCE_REVALIDATION_NOT_NEW_MODEL_INFERENCE"


def unpack_native(archive, spec, root):
    """Verify the complete original receipt before making any gate input visible."""
    prefix = spec["folder"] + "/"
    wanted = {item["file"].removeprefix(prefix): item["sha256"] for item in spec["artifacts"]}
    if any(not item["file"].startswith(prefix) for item in spec["artifacts"]):
        raise ValueError("Receipt artifact is outside the exact native attempt")
    entries = archive.infolist()
    if (
        len(entries) != len(wanted)
        or {e.filename for e in entries} != set(wanted)
        or sum(e.file_size for e in entries) > 128 * 1024 * 1024
    ):
        raise ValueError("Mounted archive differs from the complete native receipt")
    verified = []
    for entry in entries:
        path = (root / spec["folder"] / entry.filename).resolve()
        if entry.is_dir() or not path.is_relative_to((root / spec["folder"]).resolve()):
            raise ValueError("Invalid native evidence path")
        raw = archive.read(entry)
        if hashlib.sha256(raw).hexdigest() != wanted[entry.filename]:
            raise ValueError("Actual native artifact digest mismatch")
        verified.append((path, raw))
    for path, raw in verified:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def main():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "validation/gate_audit_inputs.json").read_text())
    output = Path("/kaggle/working/native_gate_audit")
    output.mkdir()
    report = {
        "classification": CLASSIFICATION,
        "status": "NOT_MET",
        "native_inference_executed": False,
        "production_ready": False,
        "full_factory_e2e_executed": False,
        "inputs": config,
        "verified_native_attempts": [],
        "unavailable_native_attempts": [],
    }
    try:
        decoder = Path(shutil.which("ffmpeg") or "").resolve()
        assert decoder.is_file()
        report["decoder"] = {
            "sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
            "version_output": subprocess.check_output([str(decoder), "-version"], text=True),
        }
        assert report["decoder"]["sha256"] == DECODER_SHA256, "Compatible decoder executable drift"
        assert report["decoder"]["version_output"].startswith("ffmpeg version 6.1.1-3ubuntu5 ")
        candidates = []
        for path in Path("/kaggle/input").rglob("native_wave_d.zip"):
            with zipfile.ZipFile(path) as archive:
                candidates.append((path, hashlib.sha256(archive.read("setup.json")).hexdigest()))
        for transfer in config.get("evidence_transfer", {}).get("attempts", []):
            matches = list(Path("/kaggle/input").rglob(transfer["zip"]))
            assert len(matches) == 1, "Missing or ambiguous exact evidence transfer"
            path = matches[0]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == transfer["sha256"]
            with zipfile.ZipFile(path) as archive:
                candidates.append((path, hashlib.sha256(archive.read("setup.json")).hexdigest()))
        report["mounted_archive_setup_digests"] = [{"path": str(path), "sha256": digest} for path, digest in candidates]
        for spec in config["native_attempts"]:
            setup = next(r["sha256"] for r in spec["artifacts"] if r["file"] == spec["folder"] + "/setup.json")
            matched = [path for path, digest in candidates if digest == setup]
            if not matched:
                # A failed notebook's mounted output may differ from the REST output.
                # Do not replace its receipt or certify it from another version.
                report["unavailable_native_attempts"].append({"kind": spec["kind"], "version": spec["version"], "expected_setup_sha256": setup})
                continue
            assert len(matched) == 1, "Multiple mounted archives match one native attempt"
            with zipfile.ZipFile(matched[0]) as archive:
                unpack_native(archive, spec, root)
            report["verified_native_attempts"].append({"kind": spec["kind"], "version": spec["version"], "setup_sha256": setup})
        # The original gate uses the original oracles, actual outputs and negative reviews.
        paths = [str(root), str(root / "packages/contracts/src"), str(root / "packages/model-registry/src")]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
        with (output / "gate.log").open("wb") as sink:
            result = subprocess.run(
                [sys.executable, "-m", "benchmarks.native.wave_d_gate"],
                cwd=root, env=env, stdout=sink, stderr=subprocess.STDOUT, timeout=2400,
            )
        gate = root / "validation/native_wave_d_gate.json"
        assert gate.is_file(), "Unchanged native gate did not produce a report"
        raw = gate.read_bytes()
        report["gate_exit_code"] = result.returncode
        report["native_exit_gate"] = json.loads(raw)["exit_gate"]
        assert not report["unavailable_native_attempts"] or report["native_exit_gate"] == "NOT_MET"
        assert result.returncode == (0 if report["native_exit_gate"] == "PASS" else 1)
        (output / gate.name).write_bytes(raw)
        report["gate_sha256"] = hashlib.sha256(raw).hexdigest()
        # Completed revalidation can report an unmet native gate; it never fabricates PASS.
        report["status"] = "REVALIDATION_COMPLETE"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        with zipfile.ZipFile("/kaggle/working/native_gate_audit.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                archive.write(path, path.name)
        print(json.dumps({"status": report["status"], "native_exit_gate": report.get("native_exit_gate"), "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "REVALIDATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
