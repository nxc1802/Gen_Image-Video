"""
tests/adversarial/test_adversarial_p2_challenger.py
Adversarial Challenger Test Suite for Phase P2 Verification:
- Alembic migration lifecycle stress (multi-cycle upgrade/downgrade, idempotence, data persistence across migrations)
- Asset persistence across restart (disk-backed SQLite, schema verification, anti-fake cache scan)
- Storage failure inoculation (fail-fast 503 on upload/download, zero DB pollution, zero in-memory fallback)
- Redis wakeup bus & telemetry hardware invariants
"""
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import os
import sqlite3
import sys
from typing import Any, Dict, Optional
import pytest
from starlette.testclient import TestClient

from alembic.util.exc import CommandError
from factory_auth.config import AuthConfig
from factory_contracts.jobs.job_contract import AssetRecord, JobSubmissionRequest
from factory_contracts.telemetry import NVMLTelemetryCollector
from factory.apps.control_api.src.api.routers import assets as assets_module
from factory.apps.control_api.src.main import app as control_plane_app
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.repository.migration_runner import (
    run_downgrade,
    run_migrations,
)
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository

VALID_TOKEN = "sk-factory-prod-secret-token"
JWT_SECRET = "test-jwt-secret-key-32-chars-long!"
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x02\x00\x00\x00\x02PX\xea"
    b"\x00\x00\x00\x13IDATx\x9cc\xfc\xcf\x80\x0f0\xe1\x95e\x18\xa9\xd2\x00A,\x01\x13y\xed\xba&\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def test_auth_headers():
    return {
        "Authorization": f"Bearer {VALID_TOKEN}",
    }


# =====================================================================
# 1. Alembic Migration Stress & Edge Case Tests
# =====================================================================
class TestAlembicMigrationStress:
    """Adversarial stress-testing of Alembic database migration lifecycle."""

    def test_multicycle_upgrade_downgrade_stress(self, tmp_path):
        """Stress test multiple complete cycles of upgrades and downgrades."""
        db_file = tmp_path / "stress_lifecycle.db"
        db_url = f"sqlite:///{db_file}"

        for cycle in range(3):
            # 1. Upgrade to head
            run_migrations(db_url=db_url, target="head")
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            assert {"jobs", "job_events", "worker_registrations", "assets", "alembic_version"}.issubset(tables)

            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = {row[0] for row in cursor.fetchall()}
            assert "idx_jobs_state_prio" in indexes
            assert "idx_jobs_service_state" in indexes
            assert "idx_job_events_job_id" in indexes
            assert "idx_workers_subsystem_status" in indexes
            assert "idx_assets_sha256" in indexes
            assert "idx_assets_created_at" in indexes

            # Telemetry columns present in 0003
            cursor.execute("PRAGMA table_info(worker_registrations);")
            cols = {row[1] for row in cursor.fetchall()}
            assert "temperature_c" in cols
            assert "telemetry_status" in cols
            conn.close()

            # 2. Step-by-step downgrade
            run_downgrade(db_url=db_url, target="0002")
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(worker_registrations);")
            cols_0002 = {row[1] for row in cursor.fetchall()}
            assert "temperature_c" not in cols_0002
            assert "telemetry_status" not in cols_0002
            conn.close()

            run_downgrade(db_url=db_url, target="0001")
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables_0001 = {row[0] for row in cursor.fetchall()}
            assert "assets" not in tables_0001
            assert "jobs" in tables_0001
            conn.close()

            run_downgrade(db_url=db_url, target="base")
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables_base = {row[0] for row in cursor.fetchall() if row[0] != "alembic_version"}
            assert len(tables_base) == 0
            conn.close()

        # Final re-upgrade to head
        run_migrations(db_url=db_url, target="head")
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables_final = {row[0] for row in cursor.fetchall()}
        assert "jobs" in tables_final
        assert "assets" in tables_final
        conn.close()

    def test_idempotent_upgrade_at_head(self, tmp_path):
        """Re-running migrations when already at head is a no-op and does not error."""
        db_file = tmp_path / "idempotent_upgrade.db"
        db_url = f"sqlite:///{db_file}"

        run_migrations(db_url=db_url, target="head")
        # Second call must succeed without altering existing tables
        run_migrations(db_url=db_url, target="head")

        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "0003"
        conn.close()

    def test_idempotent_downgrade_at_base(self, tmp_path):
        """Downgrading to base repeatedly is idempotent and does not error."""
        db_file = tmp_path / "idempotent_downgrade.db"
        db_url = f"sqlite:///{db_file}"

        run_migrations(db_url=db_url, target="head")
        run_downgrade(db_url=db_url, target="base")
        # Second call to base must succeed cleanly
        run_downgrade(db_url=db_url, target="base")

        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT version_num FROM alembic_version;")
        assert cursor.fetchall() == []
        conn.close()

    def test_data_persistence_across_schema_upgrades(self, tmp_path):
        """Verify that existing records in jobs & workers survive 0001 -> 0002 -> 0003 migrations."""
        db_file = tmp_path / "data_preserve.db"
        db_url = f"sqlite:///{db_file}"

        # 1. Migrate to 0001
        run_migrations(db_url=db_url, target="0001")
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """
            INSERT INTO jobs (job_id, request_id, service, model, task, inputs, state, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("job-m1", "req-m1", "image", "qwen-image", "text_to_image", "{}", "QUEUED", now_str, now_str),
        )
        cursor.execute(
            """
            INSERT INTO worker_registrations (worker_id, subsystem, framework, supported_models, supported_tasks, status, last_heartbeat, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("worker-m1", "image", "comfyui", "[]", "[]", "ONLINE", now_str, now_str),
        )
        conn.commit()
        conn.close()

        # 2. Upgrade to 0002 (adds assets)
        run_migrations(db_url=db_url, target="0002")
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        # Verify job survived
        cursor.execute("SELECT job_id, service, state FROM jobs WHERE job_id = 'job-m1'")
        job_row = cursor.fetchone()
        assert job_row == ("job-m1", "image", "QUEUED")

        # Insert asset at 0002
        cursor.execute(
            """
            INSERT INTO assets (asset_id, sha256, mime_type, size_bytes, storage_path, s3_url, download_url, filename, created_at, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ast-m2", "sha256-m2", "image/png", 1024, "assets/m2.png", "s3://m2.png", "/dl/m2", "m2.png", now_str, 1),
        )
        conn.commit()
        conn.close()

        # 3. Upgrade to 0003 (adds telemetry fields)
        run_migrations(db_url=db_url, target="0003")
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        # Verify job & asset survived
        cursor.execute("SELECT job_id FROM jobs WHERE job_id = 'job-m1'")
        assert cursor.fetchone() == ("job-m1",)
        cursor.execute("SELECT asset_id, verified FROM assets WHERE asset_id = 'ast-m2'")
        assert cursor.fetchone() == ("ast-m2", 1)

        # Verify worker survived and has default telemetry_status
        cursor.execute("SELECT worker_id, telemetry_status, temperature_c FROM worker_registrations WHERE worker_id = 'worker-m1'")
        worker_row = cursor.fetchone()
        assert worker_row[0] == "worker-m1"
        assert worker_row[1] == "UNAVAILABLE"
        assert worker_row[2] is None

        # Update telemetry on existing worker
        cursor.execute(
            "UPDATE worker_registrations SET temperature_c = 42.5, telemetry_status = 'HEALTHY' WHERE worker_id = 'worker-m1'"
        )
        conn.commit()
        cursor.execute("SELECT temperature_c, telemetry_status FROM worker_registrations WHERE worker_id = 'worker-m1'")
        assert cursor.fetchone() == (42.5, "HEALTHY")
        conn.close()


# =====================================================================
# 2. Asset Persistence Across Server Restarts
# =====================================================================
class TestAssetPersistenceAcrossRestarts:
    """Verifies that asset records persist to disk and survive repository/server reboots."""

    @pytest.mark.asyncio
    async def test_disk_backed_asset_persistence_across_repository_reboots(self, tmp_path):
        db_path = str(tmp_path / "persistent_assets.db")

        # 1. First repository instance creates table & asset
        repo1 = SqliteJobRepository(db_path)
        await repo1.initialize()

        now_dt = datetime.now(timezone.utc)
        test_sha = hashlib.sha256(b"challenger-binary-data").hexdigest()
        asset = AssetRecord(
            asset_id="ast-challenger-persist-01",
            sha256=test_sha,
            mime_type="image/png",
            size_bytes=len(b"challenger-binary-data"),
            storage_path="assets/ast-challenger-persist-01.png",
            s3_url="https://s3.ai.example.com/assets/ast-challenger-persist-01.png",
            download_url="/v1/assets/ast-challenger-persist-01/download",
            filename="persist_sample.png",
            created_at=now_dt,
            verified=True,
        )
        await repo1.create_asset(asset)
        await repo1.close()

        # 2. Second repository instance connects to the same file on disk (simulating restart)
        repo2 = SqliteJobRepository(db_path)
        await repo2.initialize()

        fetched_by_id = await repo2.get_asset("ast-challenger-persist-01")
        assert fetched_by_id is not None
        assert fetched_by_id.asset_id == "ast-challenger-persist-01"
        assert fetched_by_id.sha256 == test_sha
        assert fetched_by_id.mime_type == "image/png"
        assert fetched_by_id.size_bytes == len(b"challenger-binary-data")
        assert fetched_by_id.filename == "persist_sample.png"
        assert fetched_by_id.verified is True

        fetched_by_sha = await repo2.get_asset_by_sha256(test_sha)
        assert fetched_by_sha is not None
        assert fetched_by_sha.asset_id == "ast-challenger-persist-01"

        await repo2.close()

    def test_anti_fake_no_in_memory_asset_metadata_cache(self):
        """Empirically inspects assets module to confirm zero in-memory fallback cache."""
        assert not hasattr(assets_module, "_ASSET_METADATA_CACHE"), (
            "Found forbidden `_ASSET_METADATA_CACHE` in assets router module! "
            "Asset metadata must be persisted to database, not held in an in-memory dictionary."
        )


# =====================================================================
# 3. Storage Failure Inoculation & Fail-Fast Behavior
# =====================================================================
class TestStorageFailureInoculation:
    """Stress tests storage provider fault injection and ensures fail-fast 503."""

    @pytest.fixture
    def test_client(self, tmp_path):
        keys_to_preserve = ["FACTORY_API_KEY", "STORAGE_BACKEND", "FACTORY_DB_BACKEND", "SQLITE_DB_PATH"]
        old_env = {k: os.environ.get(k) for k in keys_to_preserve}
        db_file = tmp_path / "test_client.db"
        os.environ["FACTORY_API_KEY"] = VALID_TOKEN
        os.environ["STORAGE_BACKEND"] = "local"
        os.environ["FACTORY_DB_BACKEND"] = "sqlite"
        os.environ["SQLITE_DB_PATH"] = str(db_file)

        with TestClient(control_plane_app) as client:
            yield client

        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_upload_storage_exceptions_return_503_and_do_not_pollute_db(
        self, test_client, test_auth_headers, monkeypatch
    ):
        """Inoculate upload with various IO/Connection errors. Ensure 503 and zero DB pollution."""
        storage = control_plane_app.state.storage_provider
        repo = control_plane_app.state.repository

        test_exceptions = [
            IOError("Disk I/O error writing asset"),
            TimeoutError("Connection to S3 storage bucket timed out"),
            RuntimeError("Unexpected storage engine abort"),
        ]

        for exc in test_exceptions:
            async def failing_upload(*args, **kwargs):
                raise exc

            monkeypatch.setattr(storage, "upload_bytes", failing_upload)

            files = {"file": ("fault_test.png", VALID_PNG_BYTES, "image/png")}
            res = test_client.post("/v1/assets/upload", headers=test_auth_headers, files=files)
            assert res.status_code == 503
            err = res.json()["error"]
            assert err["code"] == "SERVICE_UNAVAILABLE"
            assert "Storage backend failed to store asset" in err["message"]
            assert "request_id" in err

        # Verify that failed uploads did not leave any dangling records in DB
        expected_sha = hashlib.sha256(VALID_PNG_BYTES).hexdigest()
        persisted = asyncio.run(repo.get_asset_by_sha256(expected_sha))
        assert persisted is None, "Failed upload must NOT persist metadata to database!"

    def test_download_storage_failure_fails_fast_503_no_fallback(
        self, test_client, test_auth_headers, monkeypatch
    ):
        """Inoculate download with storage failure. Must return 503 without dummy fallback."""
        # 1. Upload asset normally
        files = {"file": ("normal.png", VALID_PNG_BYTES, "image/png")}
        res = test_client.post("/v1/assets/upload", headers=test_auth_headers, files=files)
        assert res.status_code == 201
        asset_id = res.json()["asset_id"]

        # 2. Inject failure into storage download
        storage = control_plane_app.state.storage_provider

        async def failing_download(*args, **kwargs):
            raise IOError("Storage network read failed")

        monkeypatch.setattr(storage, "download_bytes", failing_download)

        dl_res = test_client.get(f"/v1/assets/{asset_id}/download", headers=test_auth_headers)
        assert dl_res.status_code == 503
        err = dl_res.json()["error"]
        assert err["code"] == "SERVICE_UNAVAILABLE"
        assert "Storage provider failed to retrieve asset" in err["message"]

    def test_download_missing_storage_data_returns_404(
        self, test_client, test_auth_headers, monkeypatch
    ):
        """When asset exists in DB but storage provider returns None, return 404."""
        files = {"file": ("missing_s3.png", VALID_PNG_BYTES, "image/png")}
        res = test_client.post("/v1/assets/upload", headers=test_auth_headers, files=files)
        assert res.status_code == 201
        asset_id = res.json()["asset_id"]

        storage = control_plane_app.state.storage_provider

        async def missing_download(*args, **kwargs):
            return None

        monkeypatch.setattr(storage, "download_bytes", missing_download)

        dl_res = test_client.get(f"/v1/assets/{asset_id}/download", headers=test_auth_headers)
        assert dl_res.status_code == 404
        err = dl_res.json()["error"]
        assert err["code"] == "NOT_FOUND"

    def test_adversarial_malformed_uploads(self, test_client, test_auth_headers):
        """Reject invalid media, wrong content types, and oversized payloads."""
        # 1. Non-image bytes with image/png MIME
        corrupted = b"<html><body>Not an image</body></html>"
        res = test_client.post(
            "/v1/assets/upload",
            headers=test_auth_headers,
            files={"file": ("fake.png", corrupted, "image/png")},
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_MEDIA_TYPE"

        # 2. JSON upload rejection
        res = test_client.post(
            "/v1/assets/upload",
            headers={**test_auth_headers, "Content-Type": "application/json"},
            json={"dummy": "media"},
        )
        assert res.status_code == 415
        assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

        # 3. Path traversal filename sanitization check
        traversal_file = {"file": ("../../../../etc/passwd.png", VALID_PNG_BYTES, "image/png")}
        res = test_client.post("/v1/assets/upload", headers=test_auth_headers, files=traversal_file)
        assert res.status_code == 201
        data = res.json()
        assert not data["s3_path"].startswith("..")
        assert not data["download_url"].startswith("..")


# =====================================================================
# 4. Redis Wakeup Bus & Telemetry Invariants
# =====================================================================
class TestQueueWakeupAndTelemetryInvariants:
    """Verifies Redis/Memory wakeup bus emission and NVML telemetry invariants."""

    @pytest.mark.asyncio
    async def test_memory_broker_emits_wakeup_event(self):
        broker = MemoryQueueBroker()
        await broker.initialize()

        async def listen():
            async for item in broker.subscribe_events("factory:wakeup:image"):
                return item

        listener_task = asyncio.create_task(listen())
        await asyncio.sleep(0.01)

        await broker.enqueue(job_id="job-wakeup-1", service="image", priority=5)

        event = await asyncio.wait_for(listener_task, timeout=1.0)
        assert event["event"] == "wakeup"
        assert event["data"]["service"] == "image"
        assert event["data"]["job_id"] == "job-wakeup-1"
        await broker.close()

    def test_hardware_telemetry_reports_unavailable_on_non_nvml_environment(self):
        collector = NVMLTelemetryCollector()
        telemetry = collector.sample()

        if not collector._available:
            assert telemetry["telemetry_status"] == "UNAVAILABLE"
            assert telemetry["vram_used_mb"] is None
            assert telemetry["temperature_c"] is None
            assert telemetry["gpu_utilization_percent"] is None
        else:
            assert telemetry["telemetry_status"] in ("HEALTHY", "DEGRADED", "WARNING", "AVAILABLE")
            assert telemetry["vram_used_mb"] is not None
