"""Explicit compatible Linux CPU execution of the unchanged Wave D native gate."""

import argparse
import base64
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from deployments.kaggle.backend_lifecycle_probe import source_bundle
from deployments.kaggle.runner import KaggleAPIClient

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("wan_t2v", "wan_i2v", "wan_fl2v", "wan_extension", "minimax_t2v", "minimax_conditioned", "minimax_turbo", "minimax_reference", "minimax_audio", "minimax_reference_audio")
SLUG = "factory-phase-8-native-gate-audit"


def prepare(root=ROOT):
    core = source_bundle()
    files = {}
    with zipfile.ZipFile(io.BytesIO(core)) as archive:
        files.update({e.filename: archive.read(e) for e in archive.infolist()})
    for path in (root / "benchmarks/native").iterdir():
        if path.suffix in {".py", ".json"} and not path.name.startswith("minimax_vae_diagnostic"):
            files[str(path.relative_to(root))] = path.read_bytes()
    files["benchmarks/__init__.py"] = b""
    for relative in ("deployments/kaggle/runtime_wave_d.json", "validation/native_wave_c_gate.json", "validation/native_video_content_review.json"):
        files[relative] = (root / relative).read_bytes()
    # The unchanged gate reads only the real baseline commit from this status file.
    # Do not bind the audit to unrelated mutable monitoring timestamps.
    anchor = json.loads((root / "validation/phase_status.json").read_text())["source_commit"]
    files["validation/phase_status.json"] = (json.dumps({"source_commit": anchor}) + "\n").encode()
    inputs = {"source_bundle_sha256": hashlib.sha256(core).hexdigest(), "input_files_sha256": {}, "native_attempts": []}
    transfer = root / "validation/diagnostics/native_evidence_transfer/identity.json"
    if transfer.is_file():
        inputs["evidence_transfer"] = json.loads(transfer.read_text())
    sources = []
    for kind in KINDS:
        path = root / f"validation/native_{kind}_run.json"
        if not path.exists():
            continue
        run_raw = path.read_bytes()
        run = json.loads(run_raw)
        version = run["response"]["versionNumber"]
        files[str(path.relative_to(root))] = run_raw
        receipt_path = root / f"validation/native_{kind}_receipt_v{version}.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        assert receipt["run_record_sha256"] == hashlib.sha256(run_raw).hexdigest()
        assert receipt["version"] == version and receipt["kind"] == kind
        assert receipt["kernel_status"]["status"] in {"complete", "error", "failed"}
        assert receipt["kernel_url"] == run["response"]["url"]
        assert receipt["artifacts"], "Cannot mount an attempt with no actual native archive"
        files[str(receipt_path.relative_to(root))] = receipt_path.read_bytes()
        inputs["native_attempts"].append({"kind": kind, "version": version, "folder": f"validation/native_{kind}_v{version}", "artifacts": receipt["artifacts"]})
        sources.append(run["response"]["ref"].strip("/").removeprefix("code/"))
    inputs["input_files_sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
    files["validation/gate_audit_inputs.json"] = (json.dumps(inputs, indent=2) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    bundle = stream.getvalue()
    prefix = "BUNDLE=" + repr(base64.b64encode(bundle).decode()) + "\nEXPECTED=" + repr(hashlib.sha256(bundle).hexdigest()) + "\n"
    code = prefix + r'''
import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
raw=base64.b64decode(BUNDLE);assert hashlib.sha256(raw).hexdigest()==EXPECTED
root=Path('/tmp/factory_native_gate_audit');root.mkdir()
zipfile.ZipFile(io.BytesIO(raw)).extractall(root)
subprocess.run([sys.executable,'-m','pip','install','--quiet','pydantic>=2.7','pyyaml','httpx','psutil','pillow'],check=True,timeout=300)
env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root),str(root/'packages/contracts/src'),str(root/'packages/model-registry/src')])}
result=subprocess.run([sys.executable,'-m','benchmarks.native.gate_audit_kernel'],cwd=root,env=env,timeout=2700)
raise SystemExit(result.returncode)
'''
    compile(code, SLUG, "exec")
    assert len(code.encode()) < 1_000_000, "CPU audit must fit the actual kernel code bound"
    return code, inputs, sorted(set(sources))


def collect():
    record = ROOT / "validation/native_gate_audit_run.json"
    run = json.loads(record.read_text())
    client = KaggleAPIClient()
    status = client.get_kernel_status(SLUG)
    if status["status"] not in {"complete", "error", "failed"}:
        print(json.dumps({"status": status["status"]}))
        return
    version = run["response"]["versionNumber"]
    folder = ROOT / f"validation/native_gate_audit_v{version}"
    folder.mkdir(exist_ok=True)
    artifacts = []
    for item in client.get_kernel_output(SLUG).get("files", []):
        if Path(item["fileName"]).name != "native_gate_audit.zip":
            continue
        with urllib.request.urlopen(item["url"], timeout=120) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        assert len(raw) <= 16 * 1024 * 1024
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            assert sum(e.file_size for e in archive.infolist()) < 32 * 1024 * 1024
            for entry in archive.infolist():
                assert Path(entry.filename).name == entry.filename and not entry.is_dir()
                path = folder / entry.filename
                data = archive.read(entry)
                assert not path.exists() or path.read_bytes() == data
                path.write_bytes(data)
                artifacts.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()})
    receipt = {"kernel_status": status, "run_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(), "artifacts": artifacts}
    (ROOT / f"validation/native_gate_audit_receipt_v{version}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Preserve the audit independently. Adoption requires an explicit current-input check.
    print(json.dumps({"status": status["status"], "version": version, "artifacts": len(artifacts)}))


def adopt():
    """Adopt an actual audit only while every submitted input still matches."""
    record = ROOT / "validation/native_gate_audit_run.json"
    run = json.loads(record.read_text())
    version = run["response"]["versionNumber"]
    receipt_path = ROOT / f"validation/native_gate_audit_receipt_v{version}.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["run_record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert receipt["kernel_status"]["status"] == "complete"
    folder = ROOT / f"validation/native_gate_audit_v{version}"
    for item in receipt["artifacts"]:
        path = (ROOT / item["file"]).resolve()
        assert path.is_relative_to(folder.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert {Path(item["file"]).name for item in receipt["artifacts"]} == {"audit.json", "gate.log", "native_wave_d_gate.json"}
    audit = json.loads((folder / "audit.json").read_text())
    assert audit["inputs"] == run["inputs"]
    assert audit["status"] == "REVALIDATION_COMPLETE"
    assert audit["classification"] == "CPU_NATIVE_EVIDENCE_REVALIDATION_NOT_NEW_MODEL_INFERENCE"
    assert audit["native_inference_executed"] is False and audit["production_ready"] is False
    from benchmarks.native.gate_audit_kernel import DECODER_SHA256

    assert audit["decoder"]["sha256"] == DECODER_SHA256
    assert len(audit["verified_native_attempts"]) == len(run["inputs"]["native_attempts"])
    # Rebuild deterministically; a new collection, review, golden or source invalidates adoption.
    code, inputs, _ = prepare()
    assert inputs == run["inputs"], "Current native audit inputs changed; rerun revalidation"
    assert hashlib.sha256(code.encode()).hexdigest() == run["kernel_code_sha256"]
    raw = (folder / "native_wave_d_gate.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == audit["gate_sha256"]
    gate = json.loads(raw)
    assert gate["exit_gate"] == audit["native_exit_gate"]
    assert gate["production_ready"] is False and gate["full_factory_e2e_executed"] is False
    (ROOT / "validation/native_wave_d_gate.json").write_bytes(raw)
    provenance = {
        "classification": audit["classification"], "gate_sha256": audit["gate_sha256"],
        "run": str(record.relative_to(ROOT)), "run_record_sha256": receipt["run_record_sha256"],
        "receipt": str(receipt_path.relative_to(ROOT)), "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "decoder": audit["decoder"], "native_inference_executed": False, "production_ready": False,
    }
    (ROOT / "validation/native_wave_d_gate_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"exit_gate": gate["exit_gate"], "native_pass": sum(c["status"] == "PASS" for c in gate["cases"]), "production_ready": False}))


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--collect", action="store_true")
    actions.add_argument("--adopt", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
        return
    if args.adopt:
        adopt()
        return
    code, inputs, sources = prepare()
    if not args.run:
        print(json.dumps({"kernel_bytes": len(code.encode()), "mounted_attempts": len(sources), "classification": "PREPARED_CPU_REVALIDATION_NOT_EXECUTED"}))
        return
    client = KaggleAPIClient()
    response = client.push_kernel(slug=SLUG, title="Factory Phase 8 Native Gate Audit", code_content=code,
                                 is_private=True, enable_gpu=False, enable_internet=True, kernel_sources=sources,
                                 dataset_sources=([inputs["evidence_transfer"]["dataset_ref"]] if "evidence_transfer" in inputs else []))
    path = ROOT / "validation/native_gate_audit_run.json"
    if path.exists():
        previous = json.loads(path.read_text())
        path.with_name(f"native_gate_audit_run_v{previous['response']['versionNumber']}.json").write_bytes(path.read_bytes())
    run = {"observed_at_utc": datetime.now(UTC).isoformat(), "response": response, "slug": SLUG,
           "kernel_code_sha256": hashlib.sha256(code.encode()).hexdigest(), "inputs": inputs,
           "classification": "CPU_NATIVE_EVIDENCE_REVALIDATION_NOT_NEW_MODEL_INFERENCE", "production_ready": False}
    path.write_text(json.dumps(run, indent=2) + "\n")
    (ROOT / f"validation/native_gate_audit_submitted_v{response['versionNumber']}.py").write_text(code)
    print(json.dumps({"url": response["url"], "version": response["versionNumber"]}))


if __name__ == "__main__":
    main()
