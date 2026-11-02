"""
tests/unit/repository/test_alembic_migrations.py
Verifies Alembic migrations for Factory v4:
- Clean upgrade from base to head
- Table and column creation (jobs, job_events, worker_registrations, assets)
- Worker telemetry columns in 0003
- Clean downgrade to base
- Re-upgrade to head
"""
import os
import sqlite3
import pytest
from factory.apps.control_api.src.queue.repository.migration_runner import (
    run_migrations,
    run_downgrade,
)


def test_alembic_migrations_lifecycle(tmp_path):
    db_file = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{db_file}"

    # 1. Upgrade to head (0001 -> 0002 -> 0003)
    run_migrations(db_url=db_url, target="head")

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Verify tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "jobs" in tables
    assert "job_events" in tables
    assert "worker_registrations" in tables
    assert "assets" in tables
    assert "alembic_version" in tables

    # Verify jobs has trace_id column added in 0004
    cursor.execute("PRAGMA table_info(jobs);")
    job_cols = {row[1] for row in cursor.fetchall()}
    assert "trace_id" in job_cols

    # Verify worker_registrations has telemetry columns added in 0003
    cursor.execute("PRAGMA table_info(worker_registrations);")
    worker_cols = {row[1] for row in cursor.fetchall()}
    assert "temperature_c" in worker_cols
    assert "telemetry_status" in worker_cols
    assert "gpu_utilization" in worker_cols
    assert "vram_used_mb" in worker_cols

    # Verify assets has columns added in 0002
    cursor.execute("PRAGMA table_info(assets);")
    asset_cols = {row[1] for row in cursor.fetchall()}
    assert "asset_id" in asset_cols
    assert "sha256" in asset_cols
    assert "mime_type" in asset_cols
    assert "size_bytes" in asset_cols
    assert "storage_path" in asset_cols
    assert "s3_url" in asset_cols
    assert "download_url" in asset_cols
    assert "filename" in asset_cols
    assert "verified" in asset_cols

    conn.close()

    # 2. Downgrade to 0002 (removes 0003 telemetry columns)
    run_downgrade(db_url=db_url, target="0002")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(worker_registrations);")
    worker_cols_0002 = {row[1] for row in cursor.fetchall()}
    assert "temperature_c" not in worker_cols_0002
    assert "telemetry_status" not in worker_cols_0002
    conn.close()

    # 3. Downgrade to 0001 (removes 0002 assets table)
    run_downgrade(db_url=db_url, target="0001")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_0001 = {row[0] for row in cursor.fetchall()}
    assert "assets" not in tables_0001
    assert "jobs" in tables_0001
    conn.close()

    # 4. Downgrade to base (removes all tables)
    run_downgrade(db_url=db_url, target="base")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_base = {row[0] for row in cursor.fetchall()}
    assert "jobs" not in tables_base
    assert "job_events" not in tables_base
    assert "worker_registrations" not in tables_base
    conn.close()

    # 5. Re-upgrade to head
    run_migrations(db_url=db_url, target="head")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_reupgrade = {row[0] for row in cursor.fetchall()}
    assert "jobs" in tables_reupgrade
    assert "assets" in tables_reupgrade
    assert "worker_registrations" in tables_reupgrade
    conn.close()
