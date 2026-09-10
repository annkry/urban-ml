from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import polars as pl

from urban_ml.archive.export import conform, parquet_schema, write_parquet
from urban_ml.archive.layout import STATION_STATUS, STATIONS, partition_path
from urban_ml.core.logging import get_logger
from urban_ml.staging.objects import ObjectStore
from urban_ml.staging.snapshots import SNAPSHOT_PREFIX
from urban_ml.domain.station import TRACKED_STATION_FIELDS

logger = get_logger(__name__)


def _table_prefix(table: str) -> str:
    return f"{SNAPSHOT_PREFIX}/{table}/"


def _day_from_key(key: str, *, table: str) -> date | None:
    """The partition day a staged key belongs to, or None if it is not one."""

    prefix = f"{_table_prefix(table)}date="
    if not key.startswith(prefix) or not key.endswith(".parquet"):
        return None
    try:
        return date.fromisoformat(key[len(prefix) :].split("/", 1)[0])
    except ValueError:
        logger.warning("Ignoring staged key with an unparseable day: %s", key)
        return None


def staged_days(store: ObjectStore, *, table: str) -> set[date]:
    """Every day the bucket holds staged snapshots for."""

    return {
        day
        for key in store.list_keys(_table_prefix(table))
        if (day := _day_from_key(key, table=table)) is not None
    }


def staged_keys_for_day(store: ObjectStore, *, table: str, day: date) -> list[str]:
    """Every staged object for one table-day, in key order."""

    return sorted(
        key
        for key in store.list_keys(f"{_table_prefix(table)}date={day.isoformat()}/")
        if key.endswith(".parquet")
    )


def read_staged_day(store: ObjectStore, *, table: str, day: date) -> pl.DataFrame:
    """Every staged row for one table-day, oldest first."""

    keys = staged_keys_for_day(store, table=table, day=day)
    if not keys:
        return pl.DataFrame(schema=parquet_schema(table))

    frames = [pl.read_parquet(io.BytesIO(store.get(key))) for key in keys]
    return pl.concat(frames, how="vertical").sort("observed_at")


def collapse_station_changes(frame: pl.DataFrame) -> pl.DataFrame:
    """Reduce a day of full station snapshots to that day's distinct states."""

    if frame.is_empty():
        return frame

    ordered = frame.sort(["station_id", "observed_at"])
    tracked = list(TRACKED_STATION_FIELDS)
    changed = pl.any_horizontal(
        [
            pl.col(field).ne_missing(pl.col(field).shift(1).over("station_id"))
            for field in tracked
        ]
    )
    is_first = pl.col("observed_at") == pl.col("observed_at").min().over("station_id")

    return ordered.filter(is_first | changed).sort(["observed_at", "station_id"])


_COLLAPSE = {STATIONS: collapse_station_changes}


def export_staged_day(
    store: ObjectStore, *, table: str, day: date, staging_dir: Path
) -> tuple[Path, int] | None:
    """Write one table-day of staged snapshots to Parquet under staging_dir."""

    frame = read_staged_day(store, table=table, day=day)
    if frame.height == 0:
        logger.info("No staged %s rows for %s; skipping", table, day)
        return None

    staged_rows = frame.height
    collapse = _COLLAPSE.get(table)
    if collapse is not None:
        frame = collapse(frame)
        logger.info(
            "%s %s: collapsed %d staged rows to %d distinct states",
            table,
            day,
            staged_rows,
            frame.height,
        )

    destination = staging_dir / partition_path(table, day)
    write_parquet(conform(frame, table=table), destination)

    written = pl.read_parquet(destination).height
    if written != frame.height:
        raise RuntimeError(
            f"{table} {day}: wrote {written} rows but held {frame.height} in memory"
        )

    logger.info(
        "%s %s: %d rows -> %s (%.1f KB)",
        table,
        day,
        written,
        destination.name,
        destination.stat().st_size / 1024,
    )
    return destination, written


def delete_staged_day(store: ObjectStore, *, table: str, day: date) -> int:
    """Remove every staged object for one table-day, returning the count."""

    keys = staged_keys_for_day(store, table=table, day=day)
    for key in keys:
        store.delete(key)
    return len(keys)


def days_present_in_bucket(store: ObjectStore) -> set[date]:
    """Days staged for station_status, which drives what is exported."""

    return staged_days(store, table=STATION_STATUS)
