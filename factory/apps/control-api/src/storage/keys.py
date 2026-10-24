"""
factory/apps/control-api/src/storage/keys.py
Canonical object key formatting for artifacts and assets.
"""
import re
from typing import Optional

VALID_SUBSYSTEMS = {
    "image",
    "video",
    "music",
    "speech",
    "text",
    "vlm",
    "decision",
    "embedding",
    "enhancement",
    "assets",
}


def sanitize_filename(name: str) -> str:
    """Sanitizes filename keeping only alphanumeric, hyphen, underscore, and dot."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return sanitized.strip("._") or "artifact.bin"


def sanitize_id(identifier: str, default: str = "unknown") -> str:
    """
    Sanitizes an identifier (e.g. job_id, asset_id) removing path separators,
    directory traversal sequences, and special characters.
    """
    # Replace backslashes with forward slashes and strip
    clean = identifier.replace("\\", "/").strip()
    # Filter out empty parts and '..' segments
    parts = [p for p in clean.split("/") if p and p != ".."]
    combined = "_".join(parts)
    # Retain only safe alphanumeric characters, underscores, and hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", combined)
    return sanitized.strip("._") or default


def format_artifact_key(subsystem: str, job_id: str, artifact_name: str) -> str:
    """
    Returns canonical artifact path: {subsystem}/{job_id}/{artifact_name}
    e.g. image/job-11223344/output_0.png
    """
    clean_subsystem = subsystem.lower().strip()
    if clean_subsystem not in VALID_SUBSYSTEMS:
        clean_subsystem = "artifacts"
    clean_job = sanitize_id(job_id, default="job-unknown")
    clean_name = sanitize_filename(artifact_name)
    return f"{clean_subsystem}/{clean_job}/{clean_name}"


def format_asset_key(asset_id: str, filename: Optional[str] = None) -> str:
    """
    Returns canonical ingested asset path: assets/{asset_id}.bin or assets/{asset_id}/{filename}
    """
    clean_id = sanitize_id(asset_id, default="ast-unknown")
    if filename:
        clean_name = sanitize_filename(filename)
        return f"assets/{clean_id}/{clean_name}"
    return f"assets/{clean_id}.bin"

