"""Retrieve immutable submitted native source, without running or certifying inference."""

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from deployments.kaggle.runner import KaggleAPIClient

ROOT = Path(__file__).resolve().parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_immutable(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Historical submitted source changed: " + str(path))
        return
    path.write_bytes(raw)


def archive(kind):
    run_path = ROOT / f"validation/native_{kind}_run.json"
    run_raw = run_path.read_bytes()
    run = json.loads(run_raw)
    client = KaggleAPIClient()
    response = client._request("GET", f"/kernels/pull/{client.username}/{run['slug']}")
    version = run["response"]["versionNumber"]
    if response["metadata"]["currentVersionNumber"] != version:
        raise ValueError(
            "Remote latest version differs from the selected run; do not archive another version"
        )
    code = response["blob"]["source"]
    if len(code) > 8 * 1024 * 1024 or sha(code.encode()) != run["kernel_code_sha256"]:
        raise ValueError("Submitted kernel source digest differs from the exact selected run")
    values = {}
    for node in ast.parse(code).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"BUNDLE", "BUNDLE_SHA", "CORE_SHA", "RUNTIME"}
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    raw = base64.b64decode(values["BUNDLE"], validate=True)
    if (
        sha(raw) != run["validation_bundle_sha256"]
        or sha(raw) != values["BUNDLE_SHA"]
        or values["CORE_SHA"] != run["source_bundle_sha256"]
    ):
        raise ValueError("Submitted validation/core bundle identity mismatch")
    stream = io.BytesIO()
    entries = []
    with (
        zipfile.ZipFile(io.BytesIO(raw)) as full,
        zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as core,
    ):
        if sum(info.file_size for info in full.infolist()) > 16 * 1024 * 1024:
            raise ValueError("Unexpected oversized source archive")
        for name in full.namelist():
            data = full.read(name)
            entries.append({"path": name, "sha256": sha(data), "size_bytes": len(data)})
            if not name.startswith(("benchmarks/native/", "tests/fixtures/native/")):
                core.writestr(full.getinfo(name), data)
    core_raw = stream.getvalue()
    if sha(core_raw) != run["source_bundle_sha256"]:
        raise ValueError(
            "Reconstructed original production source bundle differs from its recorded SHA-256"
        )
    folder = ROOT / f"validation/native_sources/{kind}_v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    write_immutable(folder / "core.zip", core_raw)
    write_immutable(folder / "validation.zip", raw)
    write_immutable(folder / "kernel.py", code.encode())
    write_immutable(
        folder / "runtime_pins.json",
        (json.dumps(values["RUNTIME"], sort_keys=True, indent=2) + "\n").encode(),
    )
    path = folder / "source_identity.json"
    observed = (
        json.loads(path.read_text())["retrieved_at_utc"]
        if path.exists()
        else datetime.now(UTC).isoformat()
    )
    identity = {
        "classification": "SOURCE_ARCHIVE_ONLY_NOT_NEW_INFERENCE_EVIDENCE",
        "kind": kind,
        "version": version,
        "kernel_url": run["response"]["url"],
        "retrieved_at_utc": observed,
        "run_record_sha256": sha(run_raw),
        "kernel_code_sha256": sha(code.encode()),
        "core_zip_sha256": sha(core_raw),
        "validation_zip_sha256": sha(raw),
        "files": entries,
        "native_inference_executed_by_this_retrieval": False,
        "production_ready": False,
    }
    write_immutable(path, (json.dumps(identity, sort_keys=True, indent=2) + "\n").encode())
    return {
        "kind": kind,
        "version": version,
        "identity": str(path.relative_to(ROOT)),
        "core_zip_sha256": sha(core_raw),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(archive, args.kinds))
    print(json.dumps({"classification": "SOURCE_ARCHIVE_RETRIEVAL_ONLY", "archives": results}))


if __name__ == "__main__":
    main()
