"""Add assets table and indexes

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-02 17:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "assets",
        sa.Column("asset_id", sa.String(64), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("storage_path", sa.String(512), nullable=False),
        sa.Column("s3_url", sa.Text(), nullable=False),
        sa.Column("download_url", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index("idx_assets_sha256", "assets", ["sha256"])
    op.create_index("idx_assets_created_at", "assets", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_assets_created_at", table_name="assets")
    op.drop_index("idx_assets_sha256", table_name="assets")
    op.drop_table("assets")
