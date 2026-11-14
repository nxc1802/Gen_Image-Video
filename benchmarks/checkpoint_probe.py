"""Read-only Phase 1 metadata probe. Never downloads weights or changes a model.

Pinned revision failures remain failures even when a repository's head exists.
Head metadata is a proposed pin candidate, never a fallback or inference proof.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import time
from functools import lru_cache
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import yaml
from factory_contracts.manifests.model import ModelManifest

ROOT = Path(__file__).resolve().parents[1]


def open_public_url(url, timeout):
    """Retry transient failures at the identical pinned source; never substitute."""
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            exc.close()
        except OSError:
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


@lru_cache(maxsize=128)
def _cached_metadata(repository: str, revision: str | None) -> dict:
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repository, safe="/")
    if revision:
        url += "/revision/" + urllib.parse.quote(revision, safe="")
    url += "?blobs=true"
    result = {"url": url, "authentication": "anonymous"}
    try:
        with open_public_url(url, timeout=25) as response:
            data = json.load(response)
        result.update(status="METADATA_AVAILABLE", resolved_revision=data.get("sha"), files=data.get("siblings", []))
    except urllib.error.HTTPError as exc:
        # Only retain the public API's error field, never headers/credentials.
        raw = exc.read(1024).decode(errors="replace")
        try:
            detail = json.loads(raw).get("error", "HTTP request failed")
        except ValueError:
            detail = "Non-JSON HTTP error"
        result.update(status="HTTP_ERROR", http_status=exc.code, detail=detail)
    except (OSError, ValueError) as exc:
        result.update(status="UNAVAILABLE", detail=type(exc).__name__)
    return result


def request_metadata(repository, revision):
    return copy.deepcopy(_cached_metadata(repository, revision))


@lru_cache(maxsize=1024)
def small_file_hash(repository, revision, path):
    url = f"https://huggingface.co/{repository}/resolve/{revision}/" + urllib.parse.quote(path, safe="/")
    with open_public_url(url, timeout=45) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Refusing large non-LFS content download")
    return hashlib.sha256(data).hexdigest(), len(data)


def probe_file(key, spec):
    result = {"key": key, "provider": spec.provider, "repository": spec.repository, "revision": spec.revision,
              "file": spec.path, "destination": spec.destination or spec.path,
              "expected_sha256": spec.sha256, "expected_size_bytes": spec.size_bytes, "status": "BLOCKED"}
    if spec.provider != "huggingface":
        return {**result, "reason": "Provider metadata probe is not implemented"}
    metadata = _cached_metadata(spec.repository, spec.revision)
    entry = next((f for f in metadata.get("files", []) if f["rfilename"] == spec.path), None)
    if not entry or entry.get("size") != spec.size_bytes:
        return {**result, "reason": "Pinned file missing or size differs"}
    if entry.get("lfs"):
        matches = entry["lfs"].get("sha256") == spec.sha256
        kind = "UPSTREAM_LFS_METADATA_ONLY"
    elif spec.size_bytes <= 32 * 1024 * 1024:
        try:
            digest, size = small_file_hash(spec.repository, spec.revision, spec.path)
            matches = digest == spec.sha256 and size == spec.size_bytes
        except (OSError, ValueError) as exc:
            return {**result, "reason": type(exc).__name__}
        kind = "SMALL_SUPPORT_FILE_CONTENT_HASH"
    else:
        return {**result, "reason": "Large non-LFS file cannot be checked through metadata"}
    return {**result, "status": "IDENTITY_MATCH" if matches else "BLOCKED", "check_kind": kind}


def probe_model(path: Path) -> dict:
    manifest = ModelManifest.model_validate(yaml.safe_load(path.read_text()))
    weights = manifest.weights
    repository = weights.repository
    revision = weights.checkpoint_revision
    filename = weights.weights_file
    pinned = request_metadata(repository, revision)
    candidate = request_metadata(repository, None)
    for metadata in (pinned, candidate):
        files = metadata.pop("files", [])
        metadata["exact_filename_matches"] = [f for f in files if f["rfilename"] == filename]
        metadata["file_count"] = len(files)
        if not metadata["exact_filename_matches"]:
            metadata["weight_filename_examples"] = [f["rfilename"] for f in files if f["rfilename"].endswith((".safetensors", ".gguf", ".bin", ".pth"))][:12]
    expected = weights.files.get("primary")
    pinned["identity_matches"] = bool(expected) and any(
        f.get("lfs", {}).get("sha256") == expected.sha256 and f.get("size") == expected.size_bytes
        for f in pinned["exact_filename_matches"])
    bundle = [probe_file(key, spec) for key, spec in weights.files.items()]
    return {"model": manifest.metadata.id, "manifest": str(path.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "repository": repository, "revision": revision, "file": filename,
        "expected_sha256": expected.sha256 if expected else None,
        "expected_size_bytes": expected.size_bytes if expected else None,
        "pinned": pinned, "current_head_candidate": candidate,
        "file_identities": bundle,
        "all_file_identities_match": bool(bundle) and all(f["status"] == "IDENTITY_MATCH" for f in bundle),
        "weights_downloaded": False, "weights_verified": False,
        "inference_executed": False, "model_changed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/checkpoint_metadata_probe.json")
    args = parser.parse_args()
    paths = sorted((ROOT / "manifests/models").glob("*.yaml"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(probe_model, paths))
    report = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Anonymous upstream identity checks: LFS hashes/sizes are metadata declarations, small support files are content-hashed. Neural weights are not downloaded or locally verified; no model inference. HTTP 401 does not distinguish private/gated/nonexistent repositories. Current-head candidates are unselected.",
        "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    blocked = sum(not r["pinned"]["identity_matches"] or not r["all_file_identities_match"] for r in results)
    print(json.dumps({"models": len(results), "pinned_identity_blocked": blocked}))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
