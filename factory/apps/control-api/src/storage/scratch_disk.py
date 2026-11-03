"""
factory/apps/control-api/src/storage/scratch_disk.py
Scratch Disk Storage and Dual-Mode Automatic Cleanup for Factory v4 Lite.
Manages ephemeral artifacts under /tmp/factory_lite/artifacts/{job_id}/{filename}
with synchronous post-download deletion and periodic background TTL/LRU sweeping.
"""
import asyncio
from datetime import datetime, timezone
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from factory_contracts.errors import InvalidRequestError, NotFoundError
from factory_contracts.jobs.job_contract import ArtifactReference
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.storage.keys import sanitize_filename, sanitize_id

logger = logging.getLogger("factory.storage.scratch_disk")


class ScratchDiskConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    root_dir: str = Field(
        default_factory=lambda: (
            os.environ.get("SCRATCH_ARTIFACTS_DIR")
            or os.environ.get("FACTORY_ARTIFACT_DIR")
            or "/tmp/factory_lite/artifacts"
        )
    )
    ttl_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("ARTIFACT_TTL_SECONDS", "3600")),
        ge=1,
        description="Time-to-live for unretrieved artifact files (default: 3600s / 1h)",
    )
    sweep_interval_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("ARTIFACT_SWEEP_INTERVAL_SECONDS", "60")),
        ge=1,
        description="Frequency of periodic background sweeper (default: 60s)",
    )
    max_storage_bytes: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_ARTIFACT_STORAGE_MB", "5120")) * 1024 * 1024,
        ge=1024 * 1024,
        description="Maximum disk quota for scratch artifacts (default: 5GB)",
    )
    cleanup_on_download: bool = Field(
        default_factory=lambda: os.environ.get("ARTIFACT_CLEANUP_ON_DOWNLOAD", "true").lower() in ("true", "1", "yes"),
        description="Whether to immediately unlink file upon completed client download",
    )


class ScratchDiskManager:
    """
    High-performance, zero-external-dependency scratch disk manager.
    Coordinates local artifact persistence, security containment boundaries,
    synchronous post-download cleanup, and periodic background TTL/LRU sweepers.
    """

    def __init__(self, config: Optional[ScratchDiskConfig] = None):
        self.config = config or ScratchDiskConfig()
        self.root_dir = Path(self.config.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._sweeper_task: Optional[asyncio.Task] = None
        self._running = False

    # =========================================================================
    # Path & Security Resolution
    # =========================================================================

    def get_job_dir(self, job_id: str) -> Path:
        """Returns verified job directory path preventing path traversal."""
        if not job_id or "\x00" in job_id or ".." in job_id or "/" in job_id or "\\" in job_id:
            if not job_id or "\x00" in job_id:
                raise InvalidRequestError("Directory traversal detected in job_id.")
            try:
                raw_target = (self.root_dir / job_id).resolve()
                if not raw_target.is_relative_to(self.root_dir) or raw_target == self.root_dir or ".." in job_id:
                    raise InvalidRequestError("Directory traversal detected in job_id.")
            except (ValueError, OSError) as exc:
                raise InvalidRequestError(f"Directory traversal detected in job_id: {exc}") from exc

        clean_job = sanitize_id(job_id, default="unknown")
        try:
            target_dir = (self.root_dir / clean_job).resolve()
            is_rel = target_dir.is_relative_to(self.root_dir)
        except (ValueError, OSError, AttributeError) as exc:
            raise InvalidRequestError(f"Directory traversal detected in job_id: {exc}") from exc

        if not is_rel or target_dir == self.root_dir:
            raise InvalidRequestError("Directory traversal detected in job_id.")
        return target_dir

    get_job_scratch_dir = get_job_dir

    def get_artifact_path(self, job_id: str, filename: str) -> Path:
        """Returns verified file path under job directory preventing path traversal."""
        job_dir = self.get_job_dir(job_id)
        if not filename or "\x00" in filename or ".." in filename or "/" in filename or "\\" in filename:
            if not filename or "\x00" in filename:
                raise InvalidRequestError("Directory traversal detected in filename.")
            try:
                raw_target = (job_dir / filename).resolve()
                if not raw_target.is_relative_to(job_dir) or raw_target == job_dir or ".." in filename:
                    raise InvalidRequestError("Directory traversal detected in filename.")
            except (ValueError, OSError) as exc:
                raise InvalidRequestError(f"Directory traversal detected in filename: {exc}") from exc

        clean_file = sanitize_filename(filename)
        try:
            target_path = (job_dir / clean_file).resolve()
            is_rel = target_path.is_relative_to(job_dir)
        except (ValueError, OSError, AttributeError) as exc:
            raise InvalidRequestError(f"Directory traversal detected in filename: {exc}") from exc

        if not is_rel or target_path == job_dir:
            raise InvalidRequestError("Directory traversal detected in filename.")
        return target_path

    # =========================================================================
    # Artifact Operations
    # =========================================================================

    async def save_artifact(
        self,
        job_id: str,
        filename: str,
        data: bytes,
        mime_type: Optional[str] = None,
    ) -> ArtifactReference:
        """Writes binary data to scratch disk and returns standardized ArtifactReference."""
        target_path = self.get_artifact_path(job_id, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        def _write():
            target_path.write_bytes(data)

        await asyncio.to_thread(_write)

        detected_mime = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        sha256 = compute_bytes_sha256(data)
        size_bytes = len(data)

        if detected_mime.startswith("image/"):
            art_type = "image"
        elif detected_mime.startswith("video/"):
            art_type = "video"
        elif detected_mime.startswith("audio/"):
            art_type = "audio"
        elif detected_mime.startswith("text/"):
            art_type = "text"
        else:
            art_type = "media"

        clean_job = sanitize_id(job_id)
        clean_name = sanitize_filename(filename)
        storage_rel_path = f"{clean_job}/{clean_name}"
        direct_url = f"/v1/jobs/{job_id}/result"

        return ArtifactReference(
            artifact_id=f"art-{uuid.uuid4()}",
            type=art_type,
            mime_type=detected_mime,
            size_bytes=size_bytes,
            storage_path=storage_rel_path,
            download_url=direct_url,
            url=direct_url,
            sha256=sha256,
            created_at=datetime.now(timezone.utc),
        )

    def get_artifact(self, job_id: str, filename: Optional[str] = None) -> Optional[Path]:
        """Locates artifact file for a job on scratch disk. Returns None if absent."""
        try:
            job_dir = self.get_job_dir(job_id)
        except Exception:
            return None

        if not job_dir.exists() or not job_dir.is_dir():
            clean_job = sanitize_id(job_id, default="unknown")
            try:
                candidates = list(self.root_dir.glob(f"*/{clean_job}")) + list(self.root_dir.glob(f"*/*/{clean_job}"))
                for c in candidates:
                    if c.is_dir():
                        res_c = c.resolve()
                        if res_c.is_relative_to(self.root_dir.resolve()):
                            job_dir = c
                            break
            except Exception:
                pass

        if not job_dir.exists() or not job_dir.is_dir():
            return None

        try:
            resolved_job_dir = job_dir.resolve()
        except Exception:
            return None

        if filename:
            try:
                path = self.get_artifact_path(job_id, filename)
                if path.is_symlink():
                    return None
                resolved = path.resolve()
                if not resolved.is_relative_to(resolved_job_dir):
                    return None
                return path if path.is_file() else None
            except Exception:
                return None

        # Return first non-hidden, non-symlink file in job directory verified to be contained
        try:
            for item in sorted(job_dir.iterdir()):
                if item.is_symlink():
                    continue
                try:
                    resolved = item.resolve()
                    if not resolved.is_relative_to(resolved_job_dir):
                        continue
                except (ValueError, OSError):
                    continue
                if item.is_file() and not item.name.startswith("."):
                    return item
        except Exception:
            return None
        return None

    # =========================================================================
    # Synchronous Mode: Post-Download Cleanup Hook
    # =========================================================================

    def cleanup_job_artifact(self, job_id: str, artifact_path: Optional[Any] = None) -> int:
        """
        Synchronously unlinks the downloaded artifact and deletes the job directory.
        Designed to be called by FastAPI BackgroundTasks immediately after transmission.
        """
        if not self.config.cleanup_on_download:
            return 0

        unlinked_count = 0
        try:
            job_dir = self.get_job_dir(job_id)
        except Exception:
            return 0

        try:
            if artifact_path:
                p = Path(artifact_path)
                if p.is_file():
                    p.unlink(missing_ok=True)
                    unlinked_count += 1
                    logger.debug(f"Unlinked artifact: {p} for job {job_id}")

            if job_dir.exists() and job_dir.is_dir():
                # Unlink any remaining files if artifact_path was not specified
                if not artifact_path:
                    for item in list(job_dir.iterdir()):
                        if item.is_symlink():
                            try:
                                item.unlink(missing_ok=True)
                            except OSError:
                                pass
                        elif item.is_file():
                            item.unlink(missing_ok=True)
                            unlinked_count += 1

                # Clean up empty directory
                try:
                    remaining = list(job_dir.iterdir())
                    if not remaining:
                        job_dir.rmdir()
                        logger.debug(f"Removed empty job directory: {job_dir}")
                    elif all(f.is_file() for f in remaining) and unlinked_count > 0:
                        # Clean up completely if all files were transient artifacts
                        shutil.rmtree(job_dir, ignore_errors=True)
                except OSError:
                    pass
        except Exception as exc:
            logger.warning(f"Error during synchronous cleanup for job {job_id}: {exc}")

        return unlinked_count

    # =========================================================================
    # Asynchronous Mode: Background Periodic Sweeper
    # =========================================================================

    def sweep_once(self) -> Dict[str, Any]:
        """
        Executes a single sweep pass:
        1. TTL Eviction: Deletes files older than ttl_seconds.
        2. LRU Quota Eviction: Deletes oldest files if total size exceeds max_storage_bytes.
        3. Directory Pruning: Removes empty job directories.
        """
        now = time.time()
        ttl_threshold = self.config.ttl_seconds
        quota_bytes = self.config.max_storage_bytes
        target_bytes = int(quota_bytes * 0.8)

        evicted_ttl = 0
        evicted_lru = 0
        freed_bytes = 0
        active_files: List[Tuple[float, int, Path]] = []

        if not self.root_dir.exists():
            return {"status": "dir_missing", "evicted_ttl": 0, "evicted_lru": 0, "freed_bytes": 0, "remaining_bytes": 0}

        # 1. Scan and apply TTL
        for job_dir in list(self.root_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            for item in list(job_dir.iterdir()):
                if item.is_symlink():
                    try:
                        item.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                if not item.is_file():
                    continue
                try:
                    stat = item.stat()
                    age = now - stat.st_mtime
                    if age > ttl_threshold:
                        size = stat.st_size
                        item.unlink(missing_ok=True)
                        evicted_ttl += 1
                        freed_bytes += size
                        logger.info(f"TTL sweeper evicted expired artifact: {item} (age={age:.1f}s > {ttl_threshold}s)")
                    else:
                        active_files.append((stat.st_mtime, stat.st_size, item))
                except OSError as exc:
                    logger.debug(f"File stat/unlink error during sweep: {exc}")

        # 2. Check storage quota and apply LRU eviction
        total_size = sum(size for _, size, _ in active_files)
        if total_size > quota_bytes:
            logger.warning(f"Scratch disk quota exceeded: {total_size} bytes > {quota_bytes} bytes. Initiating LRU eviction.")
            # Sort by mtime ascending (oldest first)
            active_files.sort(key=lambda x: x[0])
            for mtime, size, file_path in active_files:
                if total_size <= target_bytes:
                    break
                try:
                    file_path.unlink(missing_ok=True)
                    total_size -= size
                    freed_bytes += size
                    evicted_lru += 1
                    logger.info(f"LRU sweeper evicted artifact: {file_path} (size={size} bytes)")
                except OSError as exc:
                    logger.debug(f"File unlink error during LRU sweep: {exc}")

        # 3. Prune empty job directories
        for job_dir in list(self.root_dir.iterdir()):
            if job_dir.is_dir():
                try:
                    for item in list(job_dir.iterdir()):
                        if item.is_symlink():
                            try:
                                item.unlink(missing_ok=True)
                            except OSError:
                                pass
                    if not any(job_dir.iterdir()):
                        job_dir.rmdir()
                except OSError:
                    pass

        return {
            "evicted_ttl": evicted_ttl,
            "evicted_lru": evicted_lru,
            "freed_bytes": freed_bytes,
            "remaining_bytes": total_size,
        }

    async def _sweeper_loop(self):
        """Infinite loop executing periodic sweep every sweep_interval_seconds."""
        logger.info(f"Started scratch disk sweeper (interval={self.config.sweep_interval_seconds}s, TTL={self.config.ttl_seconds}s)")
        try:
            while self._running:
                await asyncio.sleep(self.config.sweep_interval_seconds)
                try:
                    await asyncio.to_thread(self.sweep_once)
                except Exception as exc:
                    logger.error(f"Unexpected error in scratch disk sweeper pass: {exc}")
        except asyncio.CancelledError:
            logger.info("Scratch disk sweeper task cancelled.")

    async def initialize(self):
        """Initializes directory and starts the background sweeper task."""
        self._running = True
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self._sweeper_task is None or self._sweeper_task.done():
            self._sweeper_task = asyncio.create_task(self._sweeper_loop())

    async def shutdown(self):
        """Cancels background sweeper task and awaits its exit."""
        self._running = False
        if self._sweeper_task and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
        self._sweeper_task = None


# Global singleton and FastAPI dependency
_global_scratch_disk: Optional[ScratchDiskManager] = None


def get_scratch_disk(request: Request) -> ScratchDiskManager:
    if hasattr(request.app.state, "scratch_disk") and request.app.state.scratch_disk:
        return request.app.state.scratch_disk
    global _global_scratch_disk
    if _global_scratch_disk is None:
        _global_scratch_disk = ScratchDiskManager()
    return _global_scratch_disk
