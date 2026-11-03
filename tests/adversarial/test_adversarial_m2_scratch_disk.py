"""
tests/adversarial/test_adversarial_m2_scratch_disk.py
Adversarial Challenge & Stress Verification Suite for Milestone M2:
ScratchDiskManager, Path Traversal Boundaries, Sweeper TTL, and LRU Quota Guard.

Targeting:
1. Path Traversal & Security Probes:
   - Malicious paths in get_job_dir / get_job_scratch_dir
   - Parent directory traversal (../../etc/passwd, job-123/../../secret, etc.)
   - Embedded null bytes
   - Symlink directories pointing outside root_dir
   - Symlink files pointing to sensitive host paths (/etc/passwd)
2. Sweeper & Quota Pressure:
   - 50+ fake artifact directories with varying mtimes and sizes
   - Precise TTL eviction (expired deleted, fresh untouched, empty dirs pruned)
   - Mock disk overflow beyond max_storage_bytes (LRU Quota Guard eviction down to 80%)
3. Concurrent Churn Stress:
   - Simultaneous reads, writes, and sweep passes without crashes or deadlocks
"""
import asyncio
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List, Tuple
import pytest

from factory_contracts.errors import InvalidRequestError
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskConfig, ScratchDiskManager


@pytest.fixture
def scratch_fixture():
    temp_dir = tempfile.mkdtemp(prefix="adv_scratch_test_")
    cfg = ScratchDiskConfig(
        root_dir=temp_dir,
        ttl_seconds=60,
        sweep_interval_seconds=1,
        max_storage_bytes=5 * 1024 * 1024,  # 5 MB
        cleanup_on_download=True,
    )
    mgr = ScratchDiskManager(config=cfg)
    yield mgr, Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestAdversarialPathTraversal:
    """Vector 1: Path Traversal & Security Probes."""

    def test_job_id_traversal_rejection(self, scratch_fixture):
        mgr, root_dir = scratch_fixture

        malicious_job_ids = [
            "../../etc/passwd",
            "job-123/../../secret",
            "..",
            "../",
            "/etc/passwd",
            "job-123/../../../etc/shadow",
            "job-123/../job-123",
            "....//....//etc/passwd",
            "..\\..\\windows\\win.ini",
        ]

        for probe in malicious_job_ids:
            with pytest.raises(InvalidRequestError) as exc_info:
                mgr.get_job_dir(probe)
            assert "traversal" in str(exc_info.value).lower(), f"Failed to reject job_id: {probe}"

    def test_filename_traversal_rejection(self, scratch_fixture):
        mgr, root_dir = scratch_fixture
        job_id = "job-secure-001"

        malicious_filenames = [
            "../../etc/passwd",
            "../secret.png",
            "/etc/passwd",
            "subdir/../../etc/passwd",
            "..\\..\\boot.ini",
            "sub/../../secret.bin",
        ]

        for probe in malicious_filenames:
            with pytest.raises(InvalidRequestError) as exc_info:
                mgr.get_artifact_path(job_id, probe)
            assert "traversal" in str(exc_info.value).lower(), f"Failed to reject filename: {probe}"

    def test_null_byte_probe_handling(self, scratch_fixture):
        mgr, root_dir = scratch_fixture
        job_id = "job-secure-002"

        # Embedded null byte in traversal path: must NOT crash unhandled with ValueError
        null_probes = [
            "job-123/..\x00../secret",
            "../\x00/secret",
        ]

        for probe in null_probes:
            try:
                mgr.get_job_dir(probe)
                pytest.fail(f"Null probe {probe!r} should have been rejected!")
            except InvalidRequestError:
                pass  # Strictly rejected
            except ValueError as ve:
                # Documenting lstat null byte unhandled crash if it occurs
                pytest.fail(f"Unhandled ValueError on null byte probe: {ve}")

    def test_symlink_directory_escape_rejection(self, scratch_fixture):
        mgr, root_dir = scratch_fixture

        # Create symlink directory pointing to /etc
        symlink_dir = root_dir / "job-symlink-dir"
        try:
            symlink_dir.symlink_to(Path("/etc"))
        except OSError:
            pytest.skip("Symlink creation not permitted")

        # get_job_dir must strictly reject the symlink directory pointing outside root_dir
        with pytest.raises(InvalidRequestError) as exc_info:
            mgr.get_job_dir("job-symlink-dir")
        assert "traversal" in str(exc_info.value).lower()

    def test_symlink_file_leak_containment(self, scratch_fixture):
        mgr, root_dir = scratch_fixture
        job_id = "job-symlink-leak-test"
        job_dir = root_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        target_file = Path("/etc/passwd")
        if not target_file.exists():
            pytest.skip("/etc/passwd does not exist on host")

        symlink_file = job_dir / "output.png"
        try:
            symlink_file.symlink_to(target_file)
        except OSError:
            pytest.skip("Symlink creation not permitted")

        # 1. Lookup with explicit filename
        res_explicit = mgr.get_artifact(job_id, "output.png")
        assert res_explicit is None, "get_artifact with explicit filename must reject symlink outside job_dir"

        # 2. Lookup without filename (default artifact discovery)
        res_default = mgr.get_artifact(job_id)
        if res_default is not None:
            # If an artifact is returned, its resolved path MUST be strictly contained inside job_dir!
            resolved = res_default.resolve()
            assert resolved.is_relative_to(job_dir.resolve()), (
                f"VULNERABILITY: get_artifact default returned symlink pointing outside job_dir: {resolved}"
            )

    def test_e2e_symlink_arbitrary_file_disclosure_prevention(self):
        """
        Validates that GET /v1/jobs/{job_id}/result never serves host files
        outside the scratch disk boundary if a symlink exists in the job directory.
        """
        from fastapi.testclient import TestClient
        from factory.apps.control_api.src.main import app

        os.environ["FACTORY_API_KEY"] = "test-challenger-key"
        headers = {"Authorization": "Bearer test-challenger-key"}

        with TestClient(app) as client:
            submit_resp = client.post("/v1/jobs", json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Adversarial symlink test"}
            }, headers=headers)
            assert submit_resp.status_code == 201
            job_id = submit_resp.json()["job_id"]

            scratch_dir = Path("/tmp/factory_lite/artifacts") / job_id
            scratch_dir.mkdir(parents=True, exist_ok=True)
            symlink = scratch_dir / "output.png"
            try:
                symlink.symlink_to(Path("/etc/hosts"))
            except OSError:
                pytest.skip("Symlink creation not permitted")

            try:
                res_resp = client.get(f"/v1/jobs/{job_id}/result", headers=headers)
                # If symlink leak occurs, localhost / host db content is served with 200 OK
                assert b"localhost" not in res_resp.content, (
                    "CRITICAL EXPLOIT: /etc/hosts content was served via GET /v1/jobs/{id}/result!"
                )
            finally:
                shutil.rmtree(scratch_dir, ignore_errors=True)



class TestAdversarialSweeperAndQuota:
    """Vector 2: Sweeper & Quota Pressure."""

    def test_50_plus_directories_ttl_eviction(self, scratch_fixture):
        mgr, root_dir = scratch_fixture
        mgr.config.ttl_seconds = 60  # 60s TTL
        now = time.time()

        # Create 60 artifact directories:
        # 35 expired (mtime between now - 120s and now - 600s)
        # 25 fresh (mtime between now - 1s and now - 20s)
        num_expired = 35
        num_fresh = 25
        total_dirs = num_expired + num_fresh

        expired_files: List[Path] = []
        fresh_files: List[Path] = []

        for i in range(total_dirs):
            jid = f"job-batch-{i:03d}"
            jdir = root_dir / jid
            jdir.mkdir(parents=True, exist_ok=True)
            fname = f"artifact_{i}.bin"
            fpath = jdir / fname

            # Varying payload size: (10KB to 100KB)
            size = (10 + (i % 10) * 10) * 1024
            fpath.write_bytes(b"A" * size)

            if i < num_expired:
                # Expired
                age = 120 + (i * 10)
                mtime = now - age
                os.utime(fpath, (mtime, mtime))
                expired_files.append(fpath)
            else:
                # Fresh
                age = 5 + (i - num_expired)
                mtime = now - age
                os.utime(fpath, (mtime, mtime))
                fresh_files.append(fpath)

        assert len(list(root_dir.iterdir())) == total_dirs

        # Execute sweep pass
        stats = mgr.sweep_once()

        # Verification 1: Exactly 35 expired files were evicted
        assert stats["evicted_ttl"] == num_expired, f"Expected {num_expired} TTL evictions, got {stats['evicted_ttl']}"
        assert stats["evicted_lru"] == 0, "No LRU evictions should have occurred in TTL-only pass"

        # Verification 2: All expired files and their parent directories are gone
        for exp in expired_files:
            assert not exp.exists(), f"Expired file still exists: {exp}"
            assert not exp.parent.exists(), f"Empty job directory was not pruned: {exp.parent}"

        # Verification 3: All fresh files and their parent directories remain intact
        for fr in fresh_files:
            assert fr.exists(), f"Fresh file was erroneously evicted: {fr}"
            assert fr.parent.exists(), f"Fresh job directory missing: {fr.parent}"

        # Verification 4: Remaining directories count exactly equals num_fresh
        remaining_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
        assert len(remaining_dirs) == num_fresh

    def test_lru_quota_guard_pressure_down_to_target(self, scratch_fixture):
        mgr, root_dir = scratch_fixture
        # Quota: 5 MB (5,242,880 bytes). Target: 80% = 4 MB (4,194,304 bytes).
        quota_bytes = 5 * 1024 * 1024
        target_bytes = int(quota_bytes * 0.8)
        mgr.config.max_storage_bytes = quota_bytes
        mgr.config.ttl_seconds = 3600  # High TTL so no TTL eviction occurs

        now = time.time()
        # Create 50 fresh files of 200 KB each = 10,000 KB ≈ 9.76 MB (overflowing 5 MB quota)
        file_size = 200 * 1024
        total_files = 50
        files_created: List[Tuple[float, Path, str]] = []

        for i in range(total_files):
            jid = f"job-quota-{i:03d}"
            jdir = root_dir / jid
            jdir.mkdir(parents=True, exist_ok=True)
            fpath = jdir / "output.bin"
            fpath.write_bytes(b"Q" * file_size)

            # Staggered mtime: job-000 is oldest (now - 600s), job-049 is newest (now - 10s)
            mtime = now - (600 - (i * 10))
            os.utime(fpath, (mtime, mtime))
            files_created.append((mtime, fpath, jid))

        total_bytes_before = sum(f[1].stat().st_size for f in files_created)
        assert total_bytes_before > quota_bytes, f"Test setup error: {total_bytes_before} not > {quota_bytes}"

        # Execute sweep pass
        stats = mgr.sweep_once()

        # Verification 1: LRU eviction triggered without crash
        assert stats["evicted_ttl"] == 0
        assert stats["evicted_lru"] > 0
        assert stats["freed_bytes"] > 0

        # Verification 2: Total storage remaining is <= 80% target (4 MB)
        remaining_files = [f for f in root_dir.glob("*/*") if f.is_file()]
        remaining_bytes = sum(f.stat().st_size for f in remaining_files)
        assert remaining_bytes <= target_bytes, f"Remaining bytes {remaining_bytes} exceeds target {target_bytes}"
        assert stats["remaining_bytes"] == remaining_bytes

        # Verification 3: LRU order strictly enforced — oldest files evicted first
        # Sorted files_created: oldest first
        files_created.sort(key=lambda x: x[0])
        evicted_count = stats["evicted_lru"]

        for i in range(evicted_count):
            _, path, _ = files_created[i]
            assert not path.exists(), f"Oldest file {path} should have been evicted by LRU"

        for i in range(evicted_count, total_files):
            _, path, _ = files_created[i]
            assert path.exists(), f"Newer file {path} should have been preserved by LRU"

        # Verification 4: Pruned directories
        for i in range(evicted_count):
            _, path, jid = files_created[i]
            assert not (root_dir / jid).exists(), f"Evicted job dir {jid} should have been pruned"

    @pytest.mark.asyncio
    async def test_concurrent_churn_stress(self, scratch_fixture):
        mgr, root_dir = scratch_fixture
        mgr.config.max_storage_bytes = 2 * 1024 * 1024  # 2MB quota
        mgr.config.ttl_seconds = 2  # 2s fast TTL

        num_producers = 15
        writes_per_producer = 10

        async def producer(p_id: int):
            for step in range(writes_per_producer):
                jid = f"job-conc-{p_id}-{step}"
                data = b"C" * (30 * 1024)  # 30KB
                await mgr.save_artifact(jid, "data.bin", data)
                await asyncio.sleep(0.01)

        async def sweeper():
            for _ in range(20):
                await asyncio.sleep(0.02)
                mgr.sweep_once()

        # Run concurrent producers and periodic sweeper simultaneously
        await asyncio.gather(
            *(producer(p) for p in range(num_producers)),
            sweeper(),
        )

        # Final sweep should not fail and prune cleanly
        final_stats = mgr.sweep_once()
        assert isinstance(final_stats, dict)
        assert final_stats["remaining_bytes"] <= mgr.config.max_storage_bytes
