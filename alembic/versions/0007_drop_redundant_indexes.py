"""Drop single-column indexes that no query uses.

Measured on 48 days of Toronto data (12 GB), these nine indexes cost ~1.2 GB
between them and serve nothing:

* Every ``ix_*_system_id``. The column holds one distinct value across all
  50M rows, so the index cannot discriminate anything — it is a book index
  whose only entry points at every page.
* The four single-column indexes on ``station_vehicle_availability``. Nothing
  reads that table at all; it is written by ingestion and never queried back.
* ``ix_station_status_station_id``. Redundant with the leading columns of
  ``uq_station_status_identity`` (system_id, station_id, observed_at), which
  EXPLAIN confirms is what the serving read actually resolves through.
* The two on ``raw_gbfs_payloads`` and one on ``ingestion_runs``. Tiny, but
  the same reasoning applies and leaving them invites the same confusion.

Deliberately kept:

* Both composite unique constraints. They enforce duplicate-run protection,
  which matters more once a scheduler with retries drives ingestion. Note
  that ``uq_station_vehicle_availability_identity`` reports zero index scans
  — that statistic does not count uniqueness checks during INSERT, so it is
  working every run despite appearing idle.
* ``ix_station_status_observed_at``, which is genuinely used.

No rows are deleted and no ingestion behaviour changes. Index files are
released to the operating system on drop, so no VACUUM FULL is required.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-05

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


# (index name, table name, indexed column) - ordered largest first, which is
# also the order the space comes back in.
_REDUNDANT_INDEXES: tuple[tuple[str, str, str], ...] = (
    (
        "ix_station_vehicle_availability_observed_at",
        "station_vehicle_availability",
        "observed_at",
    ),
    (
        "ix_station_vehicle_availability_station_id",
        "station_vehicle_availability",
        "station_id",
    ),
    (
        "ix_station_vehicle_availability_system_id",
        "station_vehicle_availability",
        "system_id",
    ),
    (
        "ix_station_vehicle_availability_vehicle_type_id",
        "station_vehicle_availability",
        "vehicle_type_id",
    ),
    ("ix_station_status_station_id", "station_status", "station_id"),
    ("ix_station_status_system_id", "station_status", "system_id"),
    ("ix_raw_gbfs_payloads_observed_at", "raw_gbfs_payloads", "observed_at"),
    ("ix_raw_gbfs_payloads_system_id", "raw_gbfs_payloads", "system_id"),
    ("ix_ingestion_runs_system_id", "ingestion_runs", "system_id"),
)


def upgrade() -> None:
    for index_name, table_name, _ in _REDUNDANT_INDEXES:
        op.drop_index(index_name, table_name=table_name)


def downgrade() -> None:
    for index_name, table_name, column_name in reversed(_REDUNDANT_INDEXES):
        op.create_index(index_name, table_name, [column_name])
