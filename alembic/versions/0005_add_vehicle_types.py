"""Add vehicle_types table and station_vehicle_availability
table, capturing GBFS's per-vehicle-type breakdown (station_status's
vehicle_types_available) that was previously fetched, validated, and then
discarded during transform. vehicle_docks_available is deliberately not
modeled.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vehicle_types",
        sa.Column("system_id", sa.String(), primary_key=True),
        sa.Column("vehicle_type_id", sa.String(), primary_key=True),
        sa.Column("form_factor", sa.String(), nullable=False),
        sa.Column("propulsion_type", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
    )

    op.create_table(
        "station_vehicle_availability",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("station_id", sa.String(), nullable=False),
        sa.Column("vehicle_type_id", sa.String(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "system_id",
            "station_id",
            "vehicle_type_id",
            "observed_at",
            name="uq_station_vehicle_availability_identity",
        ),
    )
    op.create_index(
        "ix_station_vehicle_availability_observed_at",
        "station_vehicle_availability",
        ["observed_at"],
    )
    op.create_index(
        "ix_station_vehicle_availability_system_id",
        "station_vehicle_availability",
        ["system_id"],
    )
    op.create_index(
        "ix_station_vehicle_availability_station_id",
        "station_vehicle_availability",
        ["station_id"],
    )
    op.create_index(
        "ix_station_vehicle_availability_vehicle_type_id",
        "station_vehicle_availability",
        ["vehicle_type_id"],
    )


def downgrade() -> None:
    op.drop_table("station_vehicle_availability")
    op.drop_table("vehicle_types")
