"""Capture the fields the feed already publishes, and stop overwriting history.

Three changes, all driven by what Toronto's GBFS feeds actually contain:

1. `station_status` gains `num_vehicles_disabled` and `num_docks_disabled`.
   16% of Toronto's fleet is disabled at any moment and 56% of stations hold
   at least one. Broken bikes occupy docks, so a station's usable capacity is
   lower than `capacity` suggests -- the model could not see any of this.

2. `station_status` gains `num_vehicles_electric` and `num_vehicles_human`,
   replacing the `station_vehicle_availability` table. That table wrote four
   rows per station per run (1.2M rows/day, 217 MB/day) to record counts
   whose payload is four bytes; as columns the same information costs about
   4.9 MB/day. Splitting by propulsion rather than by vehicle_type_id means a
   new bike model needs no migration -- Toronto defines ten types and only
   four ever appear.

3. `stations` becomes a change log. It was upserted in place, so a station
   that was renamed, moved, or gained docks looked as though it always had
   today's details, and feature computation applied today's capacity to every
   historical row. Rows are now appended only when something differs.

Also dropped: `last_reported` (nothing read it, and it was 80% of the
archive's size) and `raw_gbfs_payloads` (raw JSON already parsed into
`station_status`).

The existing `stations` rows are preserved as each station's first
observation, timestamped from the newest status row so the baseline sits at a
real moment rather than at migration time.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-06

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "station_status",
        sa.Column("num_vehicles_disabled", sa.Integer(), nullable=True),
    )
    op.add_column(
        "station_status", sa.Column("num_docks_disabled", sa.Integer(), nullable=True)
    )
    op.add_column(
        "station_status",
        sa.Column("num_vehicles_electric", sa.Integer(), nullable=True),
    )
    op.add_column(
        "station_status", sa.Column("num_vehicles_human", sa.Integer(), nullable=True)
    )
    op.drop_column("station_status", "last_reported")

    op.drop_table("station_vehicle_availability")
    op.drop_table("raw_gbfs_payloads")

    op.add_column("stations", sa.Column("address", sa.String(), nullable=True))
    op.add_column(
        "stations", sa.Column("is_charging_station", sa.Boolean(), nullable=True)
    )
    op.add_column(
        "stations", sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.execute(
        "UPDATE stations SET observed_at = COALESCE("
        "(SELECT max(observed_at) FROM station_status"
        "  WHERE station_status.system_id = stations.system_id"
        "    AND station_status.station_id = stations.station_id),"
        " now())"
    )
    op.alter_column("stations", "observed_at", nullable=False)

    op.add_column(
        "stations", sa.Column("id", sa.Integer(), autoincrement=True, nullable=True)
    )
    op.execute("CREATE SEQUENCE stations_id_seq OWNED BY stations.id")
    op.execute("UPDATE stations SET id = nextval('stations_id_seq')")
    op.execute("ALTER TABLE stations ALTER COLUMN id SET NOT NULL")
    op.execute(
        "ALTER TABLE stations ALTER COLUMN id SET DEFAULT nextval('stations_id_seq')"
    )

    op.drop_constraint("stations_pkey", "stations", type_="primary")
    op.create_primary_key("stations_pkey", "stations", ["id"])
    op.create_unique_constraint(
        "uq_stations_identity", "stations", ["system_id", "station_id", "observed_at"]
    )
    op.create_index("ix_stations_observed_at", "stations", ["observed_at"])


def downgrade() -> None:
    op.drop_index("ix_stations_observed_at", table_name="stations")
    op.drop_constraint("uq_stations_identity", "stations", type_="unique")
    op.drop_constraint("stations_pkey", "stations", type_="primary")
    op.execute(
        "DELETE FROM stations a USING stations b"
        " WHERE a.system_id = b.system_id AND a.station_id = b.station_id"
        "   AND a.observed_at < b.observed_at"
    )
    op.drop_column("stations", "id")
    op.execute("DROP SEQUENCE IF EXISTS stations_id_seq")
    op.create_primary_key("stations_pkey", "stations", ["system_id", "station_id"])
    op.drop_column("stations", "observed_at")
    op.drop_column("stations", "is_charging_station")
    op.drop_column("stations", "address")

    op.create_table(
        "raw_gbfs_payloads",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("discovery_url", sa.String(), nullable=False),
        sa.Column("system_information_url", sa.String(), nullable=False),
        sa.Column("station_status_url", sa.String(), nullable=False),
        sa.Column("station_status_payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "station_vehicle_availability",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("station_id", sa.String(), nullable=False),
        sa.Column("vehicle_type_id", sa.String(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
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

    op.add_column(
        "station_status",
        sa.Column("last_reported", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_column("station_status", "num_vehicles_human")
    op.drop_column("station_status", "num_vehicles_electric")
    op.drop_column("station_status", "num_docks_disabled")
    op.drop_column("station_status", "num_vehicles_disabled")
