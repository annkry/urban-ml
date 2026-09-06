"""Initial schema: raw_gbfs_payloads, station_snapshots, ingestion_runs

Revision ID: 0001
Revises:
Create Date: 2026-07-18

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op

PortableJSON = sa.JSON().with_variant(JSONB(), "postgresql")

revision: str = "0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_gbfs_payloads",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("discovery_url", sa.String(), nullable=False),
        sa.Column("station_information_url", sa.String(), nullable=False),
        sa.Column("station_status_url", sa.String(), nullable=False),
        sa.Column("discovery_payload", PortableJSON, nullable=False),
        sa.Column("station_information_payload", PortableJSON, nullable=False),
        sa.Column("station_status_payload", PortableJSON, nullable=False),
    )
    op.create_index(
        "ix_raw_gbfs_payloads_system_id", "raw_gbfs_payloads", ["system_id"]
    )
    op.create_index(
        "ix_raw_gbfs_payloads_observed_at", "raw_gbfs_payloads", ["observed_at"]
    )

    op.create_table(
        "station_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("station_id", sa.String(), nullable=False),
        sa.Column("station_name", sa.String(), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("capacity", sa.Integer(), nullable=True),
        sa.Column("num_vehicles_available", sa.Integer(), nullable=False),
        sa.Column("num_docks_available", sa.Integer(), nullable=True),
        sa.Column("is_installed", sa.Boolean(), nullable=False),
        sa.Column("is_renting", sa.Boolean(), nullable=False),
        sa.Column("is_returning", sa.Boolean(), nullable=False),
        sa.Column("last_reported", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "system_id",
            "station_id",
            "observed_at",
            name="uq_station_snapshot_identity",
        ),
    )
    op.create_index(
        "ix_station_snapshots_observed_at", "station_snapshots", ["observed_at"]
    )
    op.create_index(
        "ix_station_snapshots_system_id", "station_snapshots", ["system_id"]
    )
    op.create_index(
        "ix_station_snapshots_station_id", "station_snapshots", ["station_id"]
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.String(), nullable=True),
    )
    op.create_index("ix_ingestion_runs_system_id", "ingestion_runs", ["system_id"])


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("station_snapshots")
    op.drop_table("raw_gbfs_payloads")
