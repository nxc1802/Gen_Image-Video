"""
tests/unit/infrastructure/test_backup_restore_scripts.py
Unit tests for disaster recovery and backup/restore bash scripts in infrastructure/.
Verifies bash syntax validation, help flags, dry-run safety, and fail-safe exit codes.
"""
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INFRA_DIR = REPO_ROOT / "infrastructure"

SCRIPTS = [
    INFRA_DIR / "postgres" / "backup.sh",
    INFRA_DIR / "postgres" / "restore.sh",
    INFRA_DIR / "redis" / "backup.sh",
    INFRA_DIR / "redis" / "restore.sh",
    INFRA_DIR / "s3" / "backup.sh",
    INFRA_DIR / "s3" / "restore.sh",
]


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_script_exists_and_executable(script_path: Path):
    assert script_path.exists(), f"Script {script_path} does not exist"
    assert (script_path.stat().st_mode & 0o111) != 0, f"Script {script_path} is not executable"


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_bash_syntax_validation(script_path: Path):
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Bash syntax error in {script_path}:\n{result.stderr}"


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_help_flag_returns_zero(script_path: Path):
    result = subprocess.run(
        [str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script {script_path} failed on --help:\n{result.stderr}"
    assert "Usage:" in result.stdout


def test_postgres_backup_and_restore_dry_run(tmp_path: Path):
    backup_script = INFRA_DIR / "postgres" / "backup.sh"
    restore_script = INFRA_DIR / "postgres" / "restore.sh"

    # Dry run backup
    res_b = subprocess.run([str(backup_script), "--dry-run"], capture_output=True, text=True)
    assert res_b.returncode == 0
    assert "[DRY-RUN]" in res_b.stdout

    # Dry run restore requires a valid file path
    dummy_dump = tmp_path / "dummy.dump"
    dummy_dump.write_text("dummy pg dump content")

    res_r = subprocess.run([str(restore_script), "--dry-run", str(dummy_dump)], capture_output=True, text=True)
    assert res_r.returncode == 0
    assert "[DRY-RUN]" in res_r.stdout


def test_redis_backup_and_restore_dry_run(tmp_path: Path):
    backup_script = INFRA_DIR / "redis" / "backup.sh"
    restore_script = INFRA_DIR / "redis" / "restore.sh"

    res_b = subprocess.run([str(backup_script), "--dry-run"], capture_output=True, text=True)
    assert res_b.returncode == 0
    assert "[DRY-RUN]" in res_b.stdout

    dummy_rdb = tmp_path / "dump.rdb"
    dummy_rdb.write_text("dummy redis rdb")

    res_r = subprocess.run([str(restore_script), "--dry-run", str(dummy_rdb)], capture_output=True, text=True)
    assert res_r.returncode == 0
    assert "[DRY-RUN]" in res_r.stdout


def test_s3_backup_and_restore_dry_run(tmp_path: Path):
    backup_script = INFRA_DIR / "s3" / "backup.sh"
    restore_script = INFRA_DIR / "s3" / "restore.sh"

    res_b = subprocess.run([str(backup_script), "--dry-run"], capture_output=True, text=True)
    assert res_b.returncode == 0
    assert "[DRY-RUN]" in res_b.stdout

    dummy_dir = tmp_path / "s3_backup"
    dummy_dir.mkdir()

    res_r = subprocess.run([str(restore_script), "--dry-run", str(dummy_dir)], capture_output=True, text=True)
    assert res_r.returncode == 0
    assert "[DRY-RUN]" in res_r.stdout


def test_restore_scripts_reject_missing_arguments():
    for script_name in ["postgres/restore.sh", "redis/restore.sh", "s3/restore.sh"]:
        script = INFRA_DIR / script_name
        res = subprocess.run([str(script)], capture_output=True, text=True)
        assert res.returncode != 0, f"Expected non-zero exit code when argument missing for {script}"
        assert "ERROR:" in res.stderr
