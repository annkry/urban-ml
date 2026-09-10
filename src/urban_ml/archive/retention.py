from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from sqlalchemy import delete
from sqlalchemy.engine import Engine

from urban_ml.archive.export import day_bounds, days_present
from urban_ml.archive.layout import ARCHIVED_TABLES, STATION_STATUS
from urban_ml.archive.publish import archived_days
from urban_ml.archive.staged import days_present_in_bucket, delete_staged_day
from urban_ml.core.logging import get_logger
from urban_ml.staging.objects import ObjectStore
from urban_ml.storage.models import StationStatusRecord

logger = get_logger(__name__)


def days_safe_to_trim(
    present: set[date],
    archived: set[date],
    *,
    today: date,
    keep_days: int,
) -> list[date]:
    """Days the database can drop: old enough, and already in the archive.

    Three conditions, all required. The intersection with `archived` is the
    safety property: a day the archive does not hold is never deleted, however
    old it is, so a broken publish stops retention instead of destroying the
    only copy of the data.
    """

    if keep_days < 0:
        raise ValueError("keep_days must not be negative")

    cutoff = today - timedelta(days=keep_days)
    return sorted(day for day in present & archived if day < cutoff)


def trim_days(engine: Engine, *, days: Sequence[date]) -> int:
    """Delete station_status rows for the given UTC days, returning the count.

    Only station_status is trimmed. `stations` is the change log the feature
    layer joins capacity from, and it is three orders of magnitude smaller --
    trimming it would cost real accuracy to save nothing.

    One transaction per day, so an interruption leaves whole days behind
    rather than a half-deleted one. Day boundaries come from the export, so a
    trimmed day is exactly the day that was published.
    """

    removed = 0
    for day in days:
        start, end = day_bounds(day)
        with engine.begin() as connection:
            result = connection.execute(
                delete(StationStatusRecord).where(
                    StationStatusRecord.observed_at >= start,
                    StationStatusRecord.observed_at < end,
                )
            )
            removed += result.rowcount
    return removed


def apply_retention(
    engine: Engine,
    *,
    repo_id: str,
    token: str | None,
    keep_days: int | None,
    today: date,
) -> int:
    """Trim published days older than the retention window. No-op when unset.

    Retention is opt-in per environment because the same code runs against
    both databases: the cloud database holds a rolling window, while a local
    one holds the full history and must never be trimmed.

    The published day list is re-read from the Hub rather than assumed from
    whatever this run just uploaded, so deletion is driven by what the archive
    is confirmed to hold.
    """

    if keep_days is None:
        return 0

    archived = archived_days(repo_id, table=STATION_STATUS, token=token)
    present = days_present(engine, table=STATION_STATUS)
    days = days_safe_to_trim(present, archived, today=today, keep_days=keep_days)

    if not days:
        logger.info("Retention: no published day is older than %d days", keep_days)
        return 0

    removed = trim_days(engine, days=days)
    logger.info(
        "Retention: removed %d rows across %d day(s), %s .. %s",
        removed,
        len(days),
        days[0],
        days[-1],
    )
    return removed


def trim_staged_days(store: ObjectStore, *, days: Sequence[date]) -> int:
    """Delete every staged object for the given days, returning the count."""

    removed = 0
    for day in days:
        for table in ARCHIVED_TABLES:
            removed += delete_staged_day(store, table=table, day=day)
    return removed


def apply_staged_retention(
    store: ObjectStore,
    *,
    repo_id: str,
    token: str | None,
    keep_days: int | None,
    today: date,
) -> int:
    """Trim staged days the archive is confirmed to hold. No-op when unset."""

    if keep_days is None:
        return 0

    archived = archived_days(repo_id, table=STATION_STATUS, token=token)
    present = days_present_in_bucket(store)
    days = days_safe_to_trim(present, archived, today=today, keep_days=keep_days)

    if not days:
        logger.info("Retention: no staged day is older than %d days", keep_days)
        return 0

    removed = trim_staged_days(store, days=days)
    logger.info(
        "Retention: removed %d staged objects across %d day(s), %s .. %s",
        removed,
        len(days),
        days[0],
        days[-1],
    )
    return removed
