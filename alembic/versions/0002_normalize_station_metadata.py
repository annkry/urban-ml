"""Normalize station metadata: add stations dimension table, narrow
station_snapshots and raw_gbfs_payloads to drop duplicated data.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-18

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "stations",
        sa.Column("system_id", sa.String(), primary_key=True),
        sa.Column("station_id", sa.String(), primary_key=True),
        sa.Column("station_name", sa.String(), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("capacity", sa.Integer(), nullable=True),
    )

    op.drop_column("station_snapshots", "station_name")
    op.drop_column("station_snapshots", "lat")
    op.drop_column("station_snapshots", "lon")
    op.drop_column("station_snapshots", "capacity")

    op.drop_column("raw_gbfs_payloads", "station_information_url")
    op.drop_column("raw_gbfs_payloads", "station_information_payload")


def downgrade() -> None:
    op.add_column(
        "raw_gbfs_payloads",
        sa.Column("station_information_payload", sa.JSON(), nullable=True),
    )
    op.add_column(
        "raw_gbfs_payloads",
        sa.Column("station_information_url", sa.String(), nullable=True),
    )

    op.add_column(
        "station_snapshots", sa.Column("capacity", sa.Integer(), nullable=True)
    )
    op.add_column("station_snapshots", sa.Column("lon", sa.Float(), nullable=True))
    op.add_column("station_snapshots", sa.Column("lat", sa.Float(), nullable=True))
    op.add_column(
        "station_snapshots", sa.Column("station_name", sa.String(), nullable=True)
    )

    op.drop_table("stations")
