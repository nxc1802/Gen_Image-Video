"""CPU-only weight-byte identity inspection; no model loading or inference.

The specification is supplied explicitly, never discovered as a model fallback.
This migration utility is not a deployment registry or a certifier.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def inspect_weights(specs, input_root, output):
    if not isinstance(specs, list) or not specs:
        raise ValueError("A nonempty explicit file specification is required")
    for spec in specs:
        filename = spec["filename"]
        if not filename or Path(filename).name != filename or any(c in filename for c in "*?[]"):
            raise ValueError("Only exact file basenames are permitted")
        if len(spec["dataset"].split("/")) != 2:
            raise ValueError("Specify the exact dataset owner/slug")
    results = []
    for spec in specs:
        dataset_slug = spec["dataset"].split("/")[-1]
        matches = [p for p in input_root.rglob(spec["filename"]) if p.is_file() and dataset_slug in p.parts]
        result = dict(spec, status="BLOCKED", weights_verified=False)
        if len(matches) != 1:
            result["reason"] = f"Expected exactly one file in specified dataset, found {len(matches)}"
        else:
            path = matches[0]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            result.update(path=str(path), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)
            if spec.get("expected_size_bytes") and result["size_bytes"] != spec["expected_size_bytes"]:
                result["reason"] = "Dataset bytes differ from expected size"
            elif not spec.get("expected_sha256"):
                result.update(status="OBSERVED_UNBOUND", reason="Hash observed; no upstream expected digest supplied")
            elif result["sha256"] != spec["expected_sha256"]:
                result["reason"] = "Dataset digest differs from pinned upstream digest"
            else:
                result.update(status="MATCH", weights_verified=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "WEIGHT_IDENTITY_ONLY", "model_inference_executed": False,
        "note": "Hash equality verifies file identity only, never backend load, GPU feasibility, semantics or V3/V4.",
        "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/weight_identity_probe.json"))
    args = parser.parse_args()
    if args.spec:
        specs = json.loads(args.spec.read_text())
    elif os.environ.get("FACTORY_WEIGHT_PROBE_SPEC"):
        specs = json.loads(os.environ["FACTORY_WEIGHT_PROBE_SPEC"])
    else:
        parser.error("Supply an explicit --spec or FACTORY_WEIGHT_PROBE_SPEC")
    report = inspect_weights(specs, args.input_root, args.output)
    return 0 if all(r["status"] == "MATCH" for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
