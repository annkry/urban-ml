from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl
from pydantic import BaseModel

from urban_ml.archive.export import COMPRESSION, COMPRESSION_LEVEL, parquet_schema
from urban_ml.archive.layout import STATION_STATUS, STATIONS
from urban_ml.core.logging import get_logger
from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.staging.objects import ObjectNotFoundError, ObjectStore, ObjectStoreError

logger = get_logger(__name__)

SNAPSHOT_PREFIX = "snapshots"
STATION_STATUS_KEY = "serving/station_status.parquet"
STATIONS_KEY = "serving/stations.parquet"

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

RECENT_WINDOW_MINUTES = 120


class SnapshotError(Exception):
    """A staged object exists but could not be read as a snapshot."""


def _require_utc(moment: datetime) -> datetime:
    """Reject anything that is not an unambiguous UTC instant."""

    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"observed_at must be timezone-aware, got {moment!r}")
    if moment.utcoffset() != timedelta(0):
        raise ValueError(f"observed_at must be UTC, got offset {moment.utcoffset()}")
    return moment.astimezone(UTC)


def snapshot_key(*, table: str, observed_at: datetime) -> str:
    """Object key for one ingestion cycle's rows."""

    moment = _require_utc(observed_at)
    return (
        f"{SNAPSHOT_PREFIX}/{table}"
        f"/date={moment.date().isoformat()}"
        f"/{moment.strftime(_TIMESTAMP_FORMAT)}.parquet"
    )


def records_frame(records: Sequence[BaseModel], *, table: str) -> pl.DataFrame:
    """Domain records as a frame matching the archive's declared schema."""

    schema = parquet_schema(table)
    return pl.DataFrame(
        {name: [getattr(record, name) for record in records] for name in schema},
        schema=schema,
    )


def status_frame(records: Sequence[StationStatus]) -> pl.DataFrame:
    """Station status records in archive schema."""

    return records_frame(records, table=STATION_STATUS)


def stations_frame(records: Sequence[Station]) -> pl.DataFrame:
    """Station detail records in archive schema."""

    return records_frame(records, table=STATIONS)


def _to_parquet_bytes(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(
        buffer, compression=COMPRESSION, compression_level=COMPRESSION_LEVEL
    )
    return buffer.getvalue()


def _from_parquet_bytes(data: bytes) -> pl.DataFrame:
    try:
        return pl.read_parquet(io.BytesIO(data))
    except Exception as exc:  # polars raises a range of backend-specific errors
        raise SnapshotError("Staged object is not readable Parquet") from exc


def write_snapshot(
    store: ObjectStore, *, frame: pl.DataFrame, observed_at: datetime, table: str
) -> str:
    """Write one cycle's rows and return the key they landed at."""

    key = snapshot_key(table=table, observed_at=observed_at)
    store.put(key, _to_parquet_bytes(frame))
    return key


def _read_serving(store: ObjectStore, *, key: str, table: str) -> pl.DataFrame:
    """A serving file, or an empty typed frame when it is missing or corrupt."""

    schema = parquet_schema(table)
    try:
        return _from_parquet_bytes(store.get(key))
    except ObjectNotFoundError:
        return pl.DataFrame(schema=schema)
    except SnapshotError:
        logger.exception("Serving file %s is unreadable; treating it as empty", key)
        return pl.DataFrame(schema=schema)


def read_station_status(store: ObjectStore) -> pl.DataFrame:
    """The rolling serving window, or an empty frame when there is none yet."""

    return _read_serving(store, key=STATION_STATUS_KEY, table=STATION_STATUS)


def read_stations(store: ObjectStore) -> pl.DataFrame:
    """Current station details, or an empty frame when there are none yet."""

    return _read_serving(store, key=STATIONS_KEY, table=STATIONS)


def _observed_bounds(frame: pl.DataFrame) -> tuple[datetime, datetime] | None:
    """Oldest and newest observed_at, or None when the frame carries neither."""

    if frame.is_empty():
        return None

    oldest = frame["observed_at"].min()
    newest = frame["observed_at"].max()
    if oldest is None or newest is None:
        return None
    return cast(datetime, oldest), cast(datetime, newest)


def trim_to_window(
    frame: pl.DataFrame, *, window_minutes: int = RECENT_WINDOW_MINUTES
) -> pl.DataFrame:
    """Drop rows older than `window_minutes` before the newest row."""

    bounds = _observed_bounds(frame)
    if bounds is None:
        return frame

    _, newest = bounds
    cutoff = newest - timedelta(minutes=window_minutes)
    return frame.filter(pl.col("observed_at") >= cutoff)


def update_recent_window(
    store: ObjectStore,
    *,
    new_rows: pl.DataFrame,
    window_minutes: int = RECENT_WINDOW_MINUTES,
) -> pl.DataFrame:
    """Merge one cycle into the serving window and write it back."""

    combined = pl.concat([read_station_status(store), new_rows], how="vertical")
    window = (
        combined.unique(subset=["system_id", "station_id", "observed_at"], keep="last")
        .pipe(trim_to_window, window_minutes=window_minutes)
        .sort(["observed_at", "station_id"])
    )
    store.put(STATION_STATUS_KEY, _to_parquet_bytes(window))
    return window


def history_span(frame: pl.DataFrame) -> timedelta | None:
    """How much history the window holds, or None when it is empty."""

    bounds = _observed_bounds(frame)
    if bounds is None:
        return None
    oldest, newest = bounds
    return newest - oldest


def newest_observed_at(frame: pl.DataFrame) -> datetime | None:
    """The most recent observation in a frame, or None when it is empty."""

    bounds = _observed_bounds(frame)
    return None if bounds is None else bounds[1]


def covers_lookback(frame: pl.DataFrame, *, lookback_minutes: int) -> bool:
    """Whether the window holds enough history to build a full feature row."""

    span = history_span(frame)
    return span is not None and span >= timedelta(minutes=lookback_minutes)


def _publish_stations(
    store: ObjectStore, *, frame: pl.DataFrame, observed_at: datetime
) -> str:
    """Write the stations snapshot, then refresh the serving copy from it."""

    data = _to_parquet_bytes(frame)
    key = snapshot_key(table=STATIONS, observed_at=observed_at)
    store.put(key, data)
    store.put(STATIONS_KEY, data)
    return key


@dataclass(frozen=True)
class StagedCycle:
    """What one cycle wrote, for the caller to log and report on."""

    station_status_key: str
    stations_key: str
    station_status_rows: int
    stations_rows: int
    window_rows: int
    window_span: timedelta | None


def stage_cycle(
    store: ObjectStore,
    *,
    status_records: Sequence[StationStatus],
    station_records: Sequence[Station],
    observed_at: datetime,
) -> StagedCycle | None:
    """Stage one ingestion cycle: durable snapshots, then the serving window."""

    if not status_records:
        logger.warning("No station status records to stage")
        return None

    status = status_frame(status_records)
    stations = stations_frame(station_records)

    status_key = write_snapshot(
        store, frame=status, observed_at=observed_at, table=STATION_STATUS
    )
    stations_key = _publish_stations(store, frame=stations, observed_at=observed_at)
    window = update_recent_window(store, new_rows=status)

    staged = StagedCycle(
        station_status_key=status_key,
        stations_key=stations_key,
        station_status_rows=status.height,
        stations_rows=stations.height,
        window_rows=window.height,
        window_span=history_span(window),
    )
    logger.info(
        "Staged %d status rows and %d stations to %s; window holds %d rows spanning %s",
        staged.station_status_rows,
        staged.stations_rows,
        observed_at.strftime(_TIMESTAMP_FORMAT),
        staged.window_rows,
        staged.window_span,
    )
    return staged


def stage_cycle_or_log(
    store: ObjectStore,
    *,
    status_records: Sequence[StationStatus],
    station_records: Sequence[Station],
    observed_at: datetime,
) -> bool:
    """stage_cycle, but a store outage does not fail the ingestion run."""

    try:
        return (
            stage_cycle(
                store,
                status_records=status_records,
                station_records=station_records,
                observed_at=observed_at,
            )
            is not None
        )
    except (ObjectStoreError, SnapshotError):
        logger.exception("Staging to object storage failed; database write stands")
        return False
