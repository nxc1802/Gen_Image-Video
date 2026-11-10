"""Add worker telemetry fields

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-02 17:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("worker_registrations") as batch_op:
        batch_op.add_column(sa.Column("temperature_c", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "telemetry_status",
                sa.String(20),
                nullable=False,
                server_default="UNAVAILABLE",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("worker_registrations") as batch_op:
        batch_op.drop_column("telemetry_status")
        batch_op.drop_column("temperature_c")
