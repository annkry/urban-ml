from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
from sqlalchemy import Select, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from urban_ml.archive.layout import STATION_STATUS, STATIONS, partition_path
from urban_ml.core.logging import get_logger
from urban_ml.storage.models import Station, StationStatusRecord

logger = get_logger(__name__)

_COMPRESSION: Literal["zstd"] = "zstd"
_COMPRESSION_LEVEL = 9

_EXPORT_COLUMNS = {
    STATION_STATUS: (
        StationStatusRecord.system_id,
        StationStatusRecord.station_id,
        StationStatusRecord.observed_at,
        StationStatusRecord.num_vehicles_available,
        StationStatusRecord.num_vehicles_disabled,
        StationStatusRecord.num_docks_available,
        StationStatusRecord.num_docks_disabled,
        StationStatusRecord.is_installed,
        StationStatusRecord.is_renting,
        StationStatusRecord.is_returning,
        StationStatusRecord.num_vehicles_electric,
        StationStatusRecord.num_vehicles_human,
    ),
    STATIONS: (
        Station.system_id,
        Station.station_id,
        Station.station_name,
        Station.address,
        Station.lat,
        Station.lon,
        Station.capacity,
        Station.is_charging_station,
        Station.observed_at,
    ),
}

_PARQUET_SCHEMA: dict[str, dict[str, pl.DataType]] = {
    STATION_STATUS: {
        "system_id": pl.String(),
        "station_id": pl.String(),
        "observed_at": pl.Datetime("us", "UTC"),
        "num_vehicles_available": pl.Int64(),
        "num_vehicles_disabled": pl.Int64(),
        "num_docks_available": pl.Int64(),
        "num_docks_disabled": pl.Int64(),
        "is_installed": pl.Boolean(),
        "is_renting": pl.Boolean(),
        "is_returning": pl.Boolean(),
        "num_vehicles_electric": pl.Int64(),
        "num_vehicles_human": pl.Int64(),
    },
    STATIONS: {
        "system_id": pl.String(),
        "station_id": pl.String(),
        "station_name": pl.String(),
        "address": pl.String(),
        "lat": pl.Float64(),
        "lon": pl.Float64(),
        "capacity": pl.Int64(),
        "is_charging_station": pl.Boolean(),
        "observed_at": pl.Datetime("us", "UTC"),
    },
}

_OBSERVED_AT = {
    STATION_STATUS: StationStatusRecord.observed_at,
    STATIONS: Station.observed_at,
}


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """UTC day boundaries. observed_at is timezone-aware and written in UTC by
    ingestion, so partitioning on UTC days keeps a row's partition independent
    of whoever runs the export and in which timezone."""

    start = datetime.combine(day, time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def day_query(table: str, day: date) -> Select[tuple[object, ...]]:
    start, end = day_bounds(day)
    observed_at = _OBSERVED_AT[table]
    return (
        select(*_EXPORT_COLUMNS[table])
        .where(observed_at >= start, observed_at < end)
        .order_by(observed_at)
    )


def read_day(engine: Engine, *, table: str, day: date) -> pl.DataFrame:
    with engine.connect() as connection:
        return pl.read_database(
            day_query(table, day), connection, infer_schema_length=None
        )


def conform(frame: pl.DataFrame, *, table: str) -> pl.DataFrame:
    """Cast a frame to the archive's declared dtypes and column order."""

    schema = _PARQUET_SCHEMA[table]
    return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


def write_parquet(frame: pl.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(
        destination, compression=_COMPRESSION, compression_level=_COMPRESSION_LEVEL
    )
    return destination


def export_day(
    engine: Engine, *, table: str, day: date, staging_dir: Path
) -> tuple[Path, int] | None:
    """Write one table-day to Parquet under staging_dir.

    Returns None for a day with no rows, so an ingestion outage produces a
    missing partition rather than an empty file that later reads as real.
    """

    frame = read_day(engine, table=table, day=day)
    if frame.height == 0:
        logger.info("No %s rows for %s; skipping", table, day)
        return None

    destination = staging_dir / partition_path(table, day)
    write_parquet(conform(frame, table=table), destination)

    written = pl.read_parquet(destination).height
    if written != frame.height:
        raise RuntimeError(
            f"{table} {day}: wrote {written} rows but read {frame.height} from Postgres"
        )

    logger.info(
        "%s %s: %d rows -> %s (%.1f KB, %.2f bytes/row)",
        table,
        day,
        written,
        destination.name,
        destination.stat().st_size / 1024,
        destination.stat().st_size / written,
    )
    return destination, written


def utc_day(
    column: InstrumentedAttribute[datetime], *, dialect: str
) -> ColumnElement[date]:
    """The UTC calendar day of a timestamp, independent of session timezone."""

    if dialect == "postgresql":
        return func.date(func.timezone("UTC", column))
    return func.date(column)


def observed_day_range(engine: Engine, *, table: str) -> tuple[date, date] | None:
    """Oldest and newest UTC day present for a table, for driving a backfill."""

    observed_at = _OBSERVED_AT[table]
    with engine.connect() as connection:
        row = connection.execute(
            select(observed_at.label("lo")).order_by(observed_at).limit(1)
        ).one_or_none()
        if row is None:
            return None
        newest = connection.execute(
            select(observed_at.label("hi")).order_by(observed_at.desc()).limit(1)
        ).one()
    return _utc_date(row.lo), _utc_date(newest.hi)


def _utc_date(moment: datetime) -> date:
    """The UTC day of a moment the driver may have rendered in another zone."""

    if moment.tzinfo is None:
        return moment.date()
    return moment.astimezone(UTC).date()


def days_present(engine: Engine, *, table: str) -> set[date]:
    """Every UTC day for which the table holds rows."""

    observed_at = _OBSERVED_AT[table]
    with engine.connect() as connection:
        rows = connection.execute(
            select(
                utc_day(observed_at, dialect=engine.dialect.name).label("day")
            ).distinct()
        ).all()
    return {
        row.day if isinstance(row.day, date) else date.fromisoformat(row.day)
        for row in rows
    }


def days_needing_export(
    present: set[date],
    already_archived: set[date],
    *,
    today: date,
    start: date | None = None,
) -> list[date]:
    """Complete days held in the database but not yet in the archive.

    Pure set arithmetic, kept out of the CLI so the rules that actually matter
    are testable: a day that failed to publish must be retried on the next run,
    a day already published must not be re-uploaded, and nothing before `start`
    is ever published at all.

    That last rule is a floor, not a convenience. The archive was reset to
    begin at cloud-ingestion cutover, and without it an empty archive plus a
    database full of older rows reads as "38 days missing" and re-uploads
    precisely what was removed.
    """

    complete = {day for day in present if day < today}
    if start is not None:
        complete = {day for day in complete if day >= start}
    return sorted(complete - already_archived)
