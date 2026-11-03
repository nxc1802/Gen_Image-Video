"""
tests/unit/storage/test_scratch_disk.py
Unit tests for ScratchDiskManager:
- Ephemeral artifact persistence and metadata hashing
- Path traversal rejection on malicious job_id and filename
- Synchronous cleanup hook for post-download unlinking and empty dir removal
- Periodic sweeper TTL eviction of expired artifacts
- Periodic sweeper LRU eviction on quota breach
- Empty directory pruning
- Lifecycle initialize and graceful shutdown
"""
import asyncio
import os
from pathlib import Path
import shutil
import tempfile
import time
import pytest

from factory_contracts.errors import InvalidRequestError
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskConfig, ScratchDiskManager


@pytest.fixture
def temp_scratch_dir():
    tmp = tempfile.mkdtemp(prefix="factory_scratch_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def scratch_manager(temp_scratch_dir):
    cfg = ScratchDiskConfig(
        root_dir=temp_scratch_dir,
        ttl_seconds=60,
        sweep_interval_seconds=1,
        max_storage_bytes=10 * 1024 * 1024,  # 10 MB
        cleanup_on_download=True,
    )
    return ScratchDiskManager(config=cfg)


class TestScratchDiskManager:
    @pytest.mark.asyncio
    async def test_save_and_get_artifact_roundtrip(self, scratch_manager):
        job_id = "job-test-uuid-001"
        data = b"\x89PNG\r\n\x1a\nFakePngBytesForTesting"

        ref = await scratch_manager.save_artifact(
            job_id=job_id,
            filename="image.png",
            data=data,
            mime_type="image/png",
        )

        assert ref.type == "image"
        assert ref.mime_type == "image/png"
        assert ref.size_bytes == len(data)
        assert ref.storage_path == f"{job_id}/image.png"
        assert ref.download_url == f"/v1/jobs/{job_id}/result"
        assert len(ref.sha256) == 64

        # Locate artifact
        found = scratch_manager.get_artifact(job_id, "image.png")
        assert found is not None
        assert found.is_file()
        assert found.read_bytes() == data

        # Locate default artifact without filename
        found_default = scratch_manager.get_artifact(job_id)
        assert found_default is not None
        assert found_default.name == "image.png"

    def test_path_traversal_rejection(self, scratch_manager):
        # 1. Job ID traversal
        with pytest.raises(InvalidRequestError, match="traversal"):
            scratch_manager.get_job_dir("../../etc")

        with pytest.raises(InvalidRequestError, match="traversal"):
            scratch_manager.get_job_dir("..")

        # 2. Filename traversal
        with pytest.raises(InvalidRequestError, match="traversal"):
            scratch_manager.get_artifact_path("job-123", "../../etc/passwd")

        with pytest.raises(InvalidRequestError, match="traversal"):
            scratch_manager.get_artifact_path("job-123", "../secret.key")

    @pytest.mark.asyncio
    async def test_synchronous_cleanup_unlinks_file_and_directory(self, scratch_manager):
        job_id = "job-clean-001"
        data = b"transient result payload"

        await scratch_manager.save_artifact(job_id, "output.bin", data)
        artifact_path = scratch_manager.get_artifact(job_id, "output.bin")
        job_dir = scratch_manager.get_job_dir(job_id)

        assert artifact_path.exists()
        assert job_dir.exists()

        # Execute synchronous post-download cleanup
        unlinked = scratch_manager.cleanup_job_artifact(job_id, artifact_path)
        assert unlinked == 1
        assert not artifact_path.exists()
        assert not job_dir.exists()

    @pytest.mark.asyncio
    async def test_sweeper_ttl_eviction(self, temp_scratch_dir):
        # Configure short TTL of 5 seconds
        cfg = ScratchDiskConfig(
            root_dir=temp_scratch_dir,
            ttl_seconds=5,
            sweep_interval_seconds=1,
            max_storage_bytes=100 * 1024 * 1024,
        )
        mgr = ScratchDiskManager(config=cfg)

        job_id_old = "job-old-ttl"
        job_id_new = "job-new-ttl"

        await mgr.save_artifact(job_id_old, "old.txt", b"old content")
        await mgr.save_artifact(job_id_new, "new.txt", b"new content")

        old_file = mgr.get_artifact(job_id_old, "old.txt")
        new_file = mgr.get_artifact(job_id_new, "new.txt")

        # Age the old file artificially (10 seconds ago)
        old_mtime = time.time() - 10
        os.utime(old_file, (old_mtime, old_mtime))

        stats = mgr.sweep_once()
        assert stats["evicted_ttl"] >= 1
        assert not old_file.exists()
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_sweeper_lru_quota_eviction(self, temp_scratch_dir):
        # Quota: 2 MB
        cfg = ScratchDiskConfig(
            root_dir=temp_scratch_dir,
            ttl_seconds=3600,
            sweep_interval_seconds=1,
            max_storage_bytes=2 * 1024 * 1024,
        )
        mgr = ScratchDiskManager(config=cfg)

        payload_1mb = b"X" * (1024 * 1024)

        # Create 3 jobs of 1MB each (total 3MB > 2MB quota)
        await mgr.save_artifact("job-lru-1", "file1.bin", payload_1mb)
        await mgr.save_artifact("job-lru-2", "file2.bin", payload_1mb)
        await mgr.save_artifact("job-lru-3", "file3.bin", payload_1mb)

        f1 = mgr.get_artifact("job-lru-1", "file1.bin")
        f2 = mgr.get_artifact("job-lru-2", "file2.bin")
        f3 = mgr.get_artifact("job-lru-3", "file3.bin")

        # Set staggered past mtimes so f1 is oldest, f2 middle, f3 newest
        now = time.time()
        os.utime(f1, (now - 300, now - 300))
        os.utime(f2, (now - 200, now - 200))
        os.utime(f3, (now - 100, now - 100))

        stats = mgr.sweep_once()
        assert stats["evicted_lru"] >= 1
        # Oldest file f1 should be evicted first
        assert not f1.exists()

    @pytest.mark.asyncio
    async def test_sweeper_lifecycle_start_and_shutdown(self, scratch_manager):
        await scratch_manager.initialize()
        assert scratch_manager._running is True
        assert scratch_manager._sweeper_task is not None
        assert not scratch_manager._sweeper_task.done()

        await scratch_manager.shutdown()
        assert scratch_manager._running is False
        assert scratch_manager._sweeper_task is None
