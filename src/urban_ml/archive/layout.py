from __future__ import annotations

from datetime import date

STATION_STATUS = "station_status"
STATIONS = "stations"

ARCHIVED_TABLES = (STATION_STATUS, STATIONS)

PARQUET_FILENAME = "data.parquet"


def partition_path(table: str, day: date) -> str:
    """Path within the dataset repo. Re-exporting a day overwrites in place,
    so a re-run repairs a bad day rather than duplicating it."""

    return f"{table}/date={day.isoformat()}/{PARQUET_FILENAME}"


def table_glob(table: str) -> str:
    return f"{table}/**/*.parquet"
