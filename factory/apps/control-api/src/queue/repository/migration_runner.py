"""
factory/apps/control-api/src/queue/repository/migration_runner.py
Programmatic Alembic migration runner for Factory v4 Control Plane.
"""
import os
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config


def get_alembic_config(db_url: Optional[str] = None) -> Config:
    """Constructs and returns Alembic Config pointing to control-api alembic directory."""
    current_dir = Path(__file__).resolve().parent
    ini_path = None
    for p in [current_dir.parents[2], current_dir.parents[1], current_dir.parents[3], current_dir]:
        candidate = p / "alembic.ini"
        if candidate.exists():
            ini_path = candidate
            break

    if not ini_path:
        ini_path = Path("factory/apps/control-api/alembic.ini").resolve()

    cfg = Config(str(ini_path))
    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def run_migrations(db_url: Optional[str] = None, target: str = "head") -> None:
    """Programmatically runs Alembic migrations up to specified target (default 'head')."""
    cfg = get_alembic_config(db_url)
    command.upgrade(cfg, target)


def run_downgrade(db_url: Optional[str] = None, target: str = "base") -> None:
    """Programmatically runs Alembic downgrade down to specified target (default 'base')."""
    cfg = get_alembic_config(db_url)
    command.downgrade(cfg, target)


if __name__ == "__main__":
    import sys

    target_rev = sys.argv[1] if len(sys.argv) > 1 else "head"
    url = os.environ.get("POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    run_migrations(db_url=url, target=target_rev)
