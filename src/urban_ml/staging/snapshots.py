from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl

from urban_ml.archive.export import COMPRESSION, COMPRESSION_LEVEL, parquet_schema
from urban_ml.archive.layout import STATION_STATUS
from urban_ml.core.logging import get_logger
from urban_ml.domain.station_status import StationStatus
from urban_ml.staging.objects import ObjectNotFoundError, ObjectStore, ObjectStoreError

logger = get_logger(__name__)

SNAPSHOT_PREFIX = "snapshots"
RECENT_KEY = "serving/recent.parquet"

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


def status_frame(records: Sequence[StationStatus]) -> pl.DataFrame:
    """Domain records as a frame matching the archive's declared schema."""

    schema = parquet_schema(STATION_STATUS)
    return pl.DataFrame(
        {name: [getattr(record, name) for record in records] for name in schema},
        schema=schema,
    )


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


def read_recent(store: ObjectStore) -> pl.DataFrame:
    """The rolling serving window, or an empty frame when there is none yet."""

    schema = parquet_schema(STATION_STATUS)
    try:
        return _from_parquet_bytes(store.get(RECENT_KEY))
    except ObjectNotFoundError:
        return pl.DataFrame(schema=schema)
    except SnapshotError:
        logger.exception("Recent window is unreadable; rebuilding it from scratch")
        return pl.DataFrame(schema=schema)


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

    combined = pl.concat([read_recent(store), new_rows], how="vertical")
    window = (
        combined.unique(subset=["system_id", "station_id", "observed_at"], keep="last")
        .pipe(trim_to_window, window_minutes=window_minutes)
        .sort(["observed_at", "station_id"])
    )
    store.put(RECENT_KEY, _to_parquet_bytes(window))
    return window


def history_span(frame: pl.DataFrame) -> timedelta | None:
    """How much history the window holds, or None when it is empty."""

    bounds = _observed_bounds(frame)
    if bounds is None:
        return None
    oldest, newest = bounds
    return newest - oldest


def covers_lookback(frame: pl.DataFrame, *, lookback_minutes: int) -> bool:
    """Whether the window holds enough history to build a full feature row."""

    span = history_span(frame)
    return span is not None and span >= timedelta(minutes=lookback_minutes)


def stage_cycle(
    store: ObjectStore,
    *,
    records: Sequence[StationStatus],
    observed_at: datetime,
) -> str | None:
    """Stage one ingestion cycle: durable snapshot, then serving window."""

    if not records:
        logger.warning("No station status records to stage")
        return None

    frame = status_frame(records)
    key = write_snapshot(
        store, frame=frame, observed_at=observed_at, table=STATION_STATUS
    )
    window = update_recent_window(store, new_rows=frame)

    span = history_span(window)
    logger.info(
        "Staged %d rows to %s; serving window holds %d rows spanning %s",
        frame.height,
        key,
        window.height,
        span,
    )
    return key


def stage_cycle_or_log(
    store: ObjectStore,
    *,
    records: Sequence[StationStatus],
    observed_at: datetime,
) -> bool:
    """stage_cycle, but a store outage does not fail the ingestion run."""

    try:
        return stage_cycle(store, records=records, observed_at=observed_at) is not None
    except (ObjectStoreError, SnapshotError):
        logger.exception("Staging to object storage failed; database write stands")
        return False
