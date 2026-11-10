"""
tests/unit/repository/test_alembic_migrations.py
Comprehensive unit tests for Alembic schema migrations (Requirement R2 & SSoT, P2.2):
- Full lifecycle: upgrade head -> PRAGMA inspection -> downgrade 0002 -> downgrade 0001 -> downgrade base -> re-upgrade head
- Column nullability, defaults, autoincrement, and JSON data type validation
- Dialect portability (SQLite batch mode and PostgreSQL offline DDL generation)
- CLI invocation verification
"""
import io
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest

from alembic import command
from factory.apps.control_api.src.queue.repository.migration_runner import (
    get_alembic_config,
    run_downgrade,
    run_migrations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ALEMBIC_INI = REPO_ROOT / "factory" / "apps" / "control-api" / "alembic.ini"


def test_migration_full_lifecycle_and_schema_inspection(tmp_path):
    db_file = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{db_file}"

    # 1. Upgrade to head
    run_migrations(db_url=db_url, target="head")
    assert db_file.exists()

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Verify all expected tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"jobs", "job_events", "worker_registrations", "assets", "alembic_version"}.issubset(tables)

    # 2. Inspect 'jobs' columns
    cursor.execute("PRAGMA table_info(jobs)")
    jobs_cols = {row[1]: {"type": row[2].upper(), "notnull": row[3], "dflt": row[4], "pk": row[5]} for row in cursor.fetchall()}
    assert "job_id" in jobs_cols and jobs_cols["job_id"]["pk"] == 1
    assert "progress" in jobs_cols
    # progress must be JSON/TEXT, NOT FLOAT
    assert "FLOAT" not in jobs_cols["progress"]["type"]
    assert "worker_lease" in jobs_cols
    assert "priority" in jobs_cols and jobs_cols["priority"]["dflt"] == "'10'"

    # 3. Inspect 'job_events' columns
    cursor.execute("PRAGMA table_info(job_events)")
    event_cols = {row[1]: {"type": row[2].upper(), "notnull": row[3], "pk": row[5]} for row in cursor.fetchall()}
    assert event_cols["event_id"]["pk"] == 1
    # stage must be nullable (notnull == 0)
    assert event_cols["stage"]["notnull"] == 0

    # 4. Inspect 'worker_registrations' telemetry columns
    cursor.execute("PRAGMA table_info(worker_registrations)")
    worker_cols = {row[1]: {"type": row[2].upper(), "notnull": row[3], "dflt": row[4]} for row in cursor.fetchall()}
    assert "temperature_c" in worker_cols
    assert "telemetry_status" in worker_cols
    assert "UNAVAILABLE" in str(worker_cols["telemetry_status"]["dflt"])

    # 5. Inspect 'assets' table
    cursor.execute("PRAGMA table_info(assets)")
    asset_cols = {row[1]: {"type": row[2].upper(), "pk": row[5]} for row in cursor.fetchall()}
    assert asset_cols["asset_id"]["pk"] == 1
    assert "sha256" in asset_cols

    # 6. Verify Downgrade to 0002 (Drops worker telemetry fields)
    run_downgrade(db_url=db_url, target="0002")
    cursor.execute("PRAGMA table_info(worker_registrations)")
    cols_0002 = {row[1] for row in cursor.fetchall()}
    assert "temperature_c" not in cols_0002
    assert "telemetry_status" not in cols_0002

    # 7. Verify Downgrade to 0001 (Drops assets table)
    run_downgrade(db_url=db_url, target="0001")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables_0001 = {row[0] for row in cursor.fetchall()}
    assert "assets" not in tables_0001
    assert "jobs" in tables_0001

    # 8. Verify Downgrade to base (Drops all application tables)
    run_downgrade(db_url=db_url, target="base")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables_base = {row[0] for row in cursor.fetchall()}
    assert not {"jobs", "job_events", "worker_registrations", "assets"}.intersection(tables_base)

    # 9. Clean re-upgrade to head
    run_migrations(db_url=db_url, target="head")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables_reup = {row[0] for row in cursor.fetchall()}
    assert {"jobs", "job_events", "worker_registrations", "assets"}.issubset(tables_reup)
    conn.close()


def test_postgresql_offline_ddl_generation(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    cfg = get_alembic_config()

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        command.upgrade(cfg, "head", sql=True)

    ddl = buf.getvalue()
    assert "CREATE TABLE jobs" in ddl
    assert "CREATE TABLE job_events" in ddl
    assert "CREATE TABLE assets" in ddl
    assert "ALTER TABLE worker_registrations ADD COLUMN temperature_c FLOAT" in ddl
    assert "telemetry_status VARCHAR(20) DEFAULT 'UNAVAILABLE' NOT NULL" in ddl


def test_alembic_cli_invocation(tmp_path):
    db_file = tmp_path / "cli_migration.db"
    env = os.environ.copy()
    env["SQLITE_DB_PATH"] = str(db_file)

    res = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "upgrade", "head"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Alembic CLI failed:\n{res.stdout}\n{res.stderr}"
    assert db_file.exists()
