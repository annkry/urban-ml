"""Rename station_snapshots to station_status: it's a flat, per-run parse of
the station_status.json feed, and the old name read as inconsistent next to
station_vehicle_availability (which is derived from the same feed).

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-19

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.rename_table("station_snapshots", "station_status")

    op.execute(
        "ALTER TABLE station_status "
        "RENAME CONSTRAINT station_snapshots_pkey TO station_status_pkey"
    )
    op.execute(
        "ALTER TABLE station_status "
        "RENAME CONSTRAINT uq_station_snapshot_identity "
        "TO uq_station_status_identity"
    )
    op.execute(
        "ALTER SEQUENCE station_snapshots_id_seq RENAME TO station_status_id_seq"
    )
    op.execute(
        "ALTER INDEX ix_station_snapshots_observed_at "
        "RENAME TO ix_station_status_observed_at"
    )
    op.execute(
        "ALTER INDEX ix_station_snapshots_system_id "
        "RENAME TO ix_station_status_system_id"
    )
    op.execute(
        "ALTER INDEX ix_station_snapshots_station_id "
        "RENAME TO ix_station_status_station_id"
    )


def downgrade() -> None:
    op.execute(
        "ALTER INDEX ix_station_status_station_id "
        "RENAME TO ix_station_snapshots_station_id"
    )
    op.execute(
        "ALTER INDEX ix_station_status_system_id "
        "RENAME TO ix_station_snapshots_system_id"
    )
    op.execute(
        "ALTER INDEX ix_station_status_observed_at "
        "RENAME TO ix_station_snapshots_observed_at"
    )
    op.execute(
        "ALTER SEQUENCE station_status_id_seq RENAME TO station_snapshots_id_seq"
    )
    op.execute(
        "ALTER TABLE station_status "
        "RENAME CONSTRAINT uq_station_status_identity "
        "TO uq_station_snapshot_identity"
    )
    op.execute(
        "ALTER TABLE station_status "
        "RENAME CONSTRAINT station_status_pkey TO station_snapshots_pkey"
    )

    op.rename_table("station_status", "station_snapshots")
