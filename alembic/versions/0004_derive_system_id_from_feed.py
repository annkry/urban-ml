"""Derive system_id from the GBFS system_information feed instead of a
hand-configured label. Adds raw_gbfs_payloads.system_information_url and
makes ingestion_runs.system_id nullable (unknown until system_information
is successfully fetched). Existing rows were labeled with the old
hand-typed "toronto" value, which won't match the GBFS-derived
"bike_share_toronto" going forward.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "TRUNCATE TABLE station_snapshots, stations, raw_gbfs_payloads, "
        "ingestion_runs RESTART IDENTITY"
    )

    op.add_column(
        "raw_gbfs_payloads",
        sa.Column("system_information_url", sa.String(), nullable=False),
    )
    op.alter_column(
        "ingestion_runs", "system_id", existing_type=sa.String(), nullable=True
    )


def downgrade() -> None:
    op.alter_column(
        "ingestion_runs", "system_id", existing_type=sa.String(), nullable=False
    )
    op.drop_column("raw_gbfs_payloads", "system_information_url")
