"""Initial queue and workers schema

Revision ID: 0001
Revises: None
Create Date: 2026-11-02 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. jobs table
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(64), primary_key=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("service", sa.String(32), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("state", sa.String(20), nullable=False, server_default="CREATED"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("worker_id", sa.String(64), nullable=True),
        sa.Column("worker_lease", sa.String(128), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("artifacts", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("telemetry", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_jobs_state_prio", "jobs", ["state", "priority", "created_at"])
    op.create_index("idx_jobs_service_state", "jobs", ["service", "state"])
    op.create_index("idx_jobs_idempotency", "jobs", ["idempotency_key"])
    op.create_index("idx_jobs_created_at", "jobs", ["created_at"])

    # 2. job_events table
    op.create_table(
        "job_events",
        sa.Column("event_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("percent", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("data", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_job_events_job_id", "job_events", ["job_id", "event_id"])

    # 3. worker_registrations table (initial schema before telemetry expansion)
    op.create_table(
        "worker_registrations",
        sa.Column("worker_id", sa.String(64), primary_key=True),
        sa.Column("subsystem", sa.String(32), nullable=False),
        sa.Column("framework", sa.String(32), nullable=False),
        sa.Column("supported_models", sa.JSON(), nullable=False),
        sa.Column("supported_tasks", sa.JSON(), nullable=False),
        sa.Column("gpu_device", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="ONLINE"),
        sa.Column("current_job_id", sa.String(64), sa.ForeignKey("jobs.job_id", ondelete="SET NULL"), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("gpu_utilization", sa.Float(), nullable=True),
        sa.Column("vram_used_mb", sa.Float(), nullable=True),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_workers_subsystem_status", "worker_registrations", ["subsystem", "status", "last_heartbeat"])


def downgrade() -> None:
    op.drop_index("idx_workers_subsystem_status", table_name="worker_registrations")
    op.drop_table("worker_registrations")
    op.drop_index("idx_job_events_job_id", table_name="job_events")
    op.drop_table("job_events")
    op.drop_index("idx_jobs_created_at", table_name="jobs")
    op.drop_index("idx_jobs_idempotency", table_name="jobs")
    op.drop_index("idx_jobs_service_state", table_name="jobs")
    op.drop_index("idx_jobs_state_prio", table_name="jobs")
    op.drop_table("jobs")
