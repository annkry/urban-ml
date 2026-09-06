from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
from sqlalchemy import Select, select
from sqlalchemy.engine import Engine

from urban_ml.archive.layout import (
    STATION_STATUS,
    STATION_VEHICLE_AVAILABILITY,
    catalog_path,
    partition_path,
)
from urban_ml.core.logging import get_logger
from urban_ml.storage.models import (
    Station,
    StationStatusRecord,
    StationVehicleAvailabilityRecord,
)

logger = get_logger(__name__)

_COMPRESSION: Literal["zstd"] = "zstd"
_COMPRESSION_LEVEL = 9

_EXPORT_COLUMNS = {
    STATION_STATUS: (
        StationStatusRecord.system_id,
        StationStatusRecord.station_id,
        StationStatusRecord.observed_at,
        StationStatusRecord.num_vehicles_available,
        StationStatusRecord.num_docks_available,
        StationStatusRecord.is_installed,
        StationStatusRecord.is_renting,
        StationStatusRecord.is_returning,
        StationStatusRecord.last_reported,
    ),
    STATION_VEHICLE_AVAILABILITY: (
        StationVehicleAvailabilityRecord.system_id,
        StationVehicleAvailabilityRecord.station_id,
        StationVehicleAvailabilityRecord.vehicle_type_id,
        StationVehicleAvailabilityRecord.observed_at,
        StationVehicleAvailabilityRecord.count,
    ),
}

_OBSERVED_AT = {
    STATION_STATUS: StationStatusRecord.observed_at,
    STATION_VEHICLE_AVAILABILITY: StationVehicleAvailabilityRecord.observed_at,
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
        return pl.read_database(day_query(table, day), connection)


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
    write_parquet(frame, destination)

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
    return row.lo.date(), newest.hi.date()


def export_station_catalog(
    engine: Engine, *, day: date, staging_dir: Path
) -> tuple[Path, int]:
    """Snapshot the station catalog as it stands on `day`.

    compute_features needs capacity, so without this the archive is not a
    complete training input and training still has to reach into Postgres.
    Snapshotting per day also builds the capacity history that Postgres
    destroys by upserting in place.
    """

    with engine.connect() as connection:
        frame = pl.read_database(
            select(
                Station.system_id,
                Station.station_id,
                Station.station_name,
                Station.lat,
                Station.lon,
                Station.capacity,
            ).order_by(Station.system_id, Station.station_id),
            connection,
        )

    destination = write_parquet(frame, staging_dir / catalog_path(day))
    logger.info("stations %s: %d rows -> %s", day, frame.height, destination)
    return destination, frame.height
