from __future__ import annotations

from datetime import date

STATION_STATUS = "station_status"
STATION_VEHICLE_AVAILABILITY = "station_vehicle_availability"
STATIONS = "stations"

ARCHIVED_TABLES = (STATION_STATUS, STATION_VEHICLE_AVAILABILITY)

PARQUET_FILENAME = "data.parquet"


def partition_path(table: str, day: date) -> str:
    """Path within the dataset repo. Re-exporting a day overwrites in place,
    so a re-run repairs a bad day rather than duplicating it."""

    return f"{table}/date={day.isoformat()}/{PARQUET_FILENAME}"


def table_glob(table: str) -> str:
    return f"{table}/**/*.parquet"


def catalog_path(day: date) -> str:
    """One station-catalog snapshot per day, as observed that day.

    `stations` is upserted in place in Postgres, so the database only ever
    holds current capacities — a station that gained docks in August looks as
    though it always had them. Snapshotting daily here is the only way that
    history ever comes to exist, and it cannot be reconstructed later, so the
    archive starts capturing it now even though nothing consumes it yet.

    Days before the first snapshot have no file. Feature computation still
    uses the most recent catalog for every row; joining capacity as-of each
    observation is a later change, once there is enough history to make the
    difference measurable.
    """

    return f"{STATIONS}/date={day.isoformat()}/{PARQUET_FILENAME}"
