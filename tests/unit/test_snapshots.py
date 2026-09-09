from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from urban_ml.archive.layout import STATION_STATUS
from urban_ml.domain.station_status import StationStatus
from urban_ml.modeling.predict import LOOKBACK_MINUTES
from urban_ml.staging.objects import (
    LocalObjectStore,
    ObjectNotFoundError,
    ObjectStoreError,
)
from urban_ml.staging.snapshots import (
    RECENT_KEY,
    RECENT_WINDOW_MINUTES,
    covers_lookback,
    history_span,
    read_recent,
    snapshot_key,
    stage_cycle,
    stage_cycle_or_log,
    status_frame,
    trim_to_window,
    update_recent_window,
)

SYSTEM_ID = "bike_share_toronto"


def _record(station_id: str, observed_at: datetime, vehicles: int) -> StationStatus:
    return StationStatus(
        observed_at=observed_at,
        system_id=SYSTEM_ID,
        station_id=station_id,
        num_vehicles_available=vehicles,
        num_docks_available=10,
        is_installed=True,
        is_renting=True,
        is_returning=True,
    )


def _cycle(observed_at: datetime, *, stations: int = 2) -> list[StationStatus]:
    return [
        _record(f"station-{index}", observed_at, vehicles=index)
        for index in range(stations)
    ]


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(root=tmp_path)


# --- key layout ------------------------------------------------------------


def test_snapshot_key_encodes_utc_day_and_time_with_zone_marker() -> None:
    key = snapshot_key(
        table=STATION_STATUS,
        observed_at=datetime(2026, 9, 9, 12, 18, 2, tzinfo=UTC),
    )

    assert key == "snapshots/station_status/date=2026-09-09/20260909T121802Z.parquet"


def test_snapshot_key_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        snapshot_key(table=STATION_STATUS, observed_at=datetime(2026, 9, 9, 12, 18, 2))


def test_snapshot_key_rejects_non_utc_timestamps() -> None:
    toronto = datetime(2026, 9, 9, 12, 18, 2, tzinfo=ZoneInfo("America/Toronto"))

    with pytest.raises(ValueError, match="must be UTC"):
        snapshot_key(table=STATION_STATUS, observed_at=toronto)


def test_snapshot_key_partitions_on_utc_day_not_local_day() -> None:
    """20:30 in Toronto is already the next UTC day."""

    evening_in_toronto = datetime(2026, 9, 10, 0, 30, tzinfo=UTC)

    key = snapshot_key(table=STATION_STATUS, observed_at=evening_in_toronto)

    assert "date=2026-09-10" in key


def test_snapshot_partition_matches_the_rows_it_contains(
    store: LocalObjectStore,
) -> None:
    """The invariant that makes a day exportable."""

    observed_at = datetime(2026, 9, 9, 23, 59, 55, tzinfo=UTC)

    key = stage_cycle(store, records=_cycle(observed_at), observed_at=observed_at)

    assert key is not None
    assert "date=2026-09-09" in key
    staged = pl.read_parquet(store.get(key))
    assert staged["observed_at"].dt.date().unique().to_list() == [observed_at.date()]


# --- the 90-minute serving requirement -------------------------------------


def test_recent_window_covers_the_lookback_serving_requires() -> None:
    """Pins the reason RECENT_WINDOW_MINUTES is what it is."""

    assert RECENT_WINDOW_MINUTES >= LOOKBACK_MINUTES


def test_window_still_holds_the_full_lookback_after_missed_cycles(
    store: LocalObjectStore,
) -> None:
    start = datetime(2026, 9, 9, 6, 0, tzinfo=UTC)
    for index in range(RECENT_WINDOW_MINUTES // 5):
        moment = start + timedelta(minutes=5 * index)
        update_recent_window(store, new_rows=status_frame(_cycle(moment)))

    resumed = start + timedelta(minutes=RECENT_WINDOW_MINUTES + 30)
    window = update_recent_window(store, new_rows=status_frame(_cycle(resumed)))

    assert covers_lookback(window, lookback_minutes=LOOKBACK_MINUTES)


def test_covers_lookback_is_false_for_a_freshly_seeded_window(
    store: LocalObjectStore,
) -> None:
    """Freshness is not depth: a first deploy is current but too shallow."""

    now = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    window = update_recent_window(store, new_rows=status_frame(_cycle(now)))

    assert covers_lookback(window, lookback_minutes=LOOKBACK_MINUTES) is False


def test_covers_lookback_is_false_for_an_empty_window() -> None:
    assert covers_lookback(pl.DataFrame(), lookback_minutes=LOOKBACK_MINUTES) is False


def test_history_span_of_an_empty_frame_is_none() -> None:
    assert history_span(pl.DataFrame()) is None


# --- rolling window --------------------------------------------------------


def test_trim_anchors_on_the_newest_row_not_wall_clock_now() -> None:
    """After an outage the freshest data is itself old."""

    long_ago = datetime(2020, 1, 1, 0, 0, tzinfo=UTC)
    frame = pl.concat(
        [
            status_frame(_cycle(long_ago)),
            status_frame(_cycle(long_ago + timedelta(minutes=60))),
        ]
    )

    trimmed = trim_to_window(frame, window_minutes=RECENT_WINDOW_MINUTES)

    assert trimmed.height == frame.height


def test_update_drops_rows_older_than_the_window(store: LocalObjectStore) -> None:
    start = datetime(2026, 9, 9, 6, 0, tzinfo=UTC)
    update_recent_window(store, new_rows=status_frame(_cycle(start)))

    beyond = start + timedelta(minutes=RECENT_WINDOW_MINUTES + 5)
    window = update_recent_window(store, new_rows=status_frame(_cycle(beyond)))

    assert window["observed_at"].min() == beyond


def test_update_is_idempotent_for_a_retried_delivery(
    store: LocalObjectStore,
) -> None:
    """Cloud Scheduler may retry a tick; a replay must not double-count."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    rows = status_frame(_cycle(moment))

    update_recent_window(store, new_rows=rows)
    window = update_recent_window(store, new_rows=rows)

    assert window.height == rows.height


def test_read_recent_returns_an_empty_typed_frame_before_the_first_write(
    store: LocalObjectStore,
) -> None:
    window = read_recent(store)

    assert window.is_empty()
    assert "observed_at" in window.columns


def test_read_recent_recovers_from_an_unreadable_window(
    store: LocalObjectStore,
) -> None:
    store.put(RECENT_KEY, b"not parquet at all")

    assert read_recent(store).is_empty()


def test_staged_rows_round_trip_through_the_archive_schema(
    store: LocalObjectStore,
) -> None:
    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    records = _cycle(moment, stations=3)

    stage_cycle(store, records=records, observed_at=moment)
    window = read_recent(store)

    assert window.height == 3
    assert window.schema == status_frame(records).schema


def test_stage_cycle_writes_the_durable_snapshot_before_the_window(
    store: LocalObjectStore,
) -> None:
    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    key = stage_cycle(store, records=_cycle(moment), observed_at=moment)

    assert key is not None
    assert store.exists(key)
    assert store.exists(RECENT_KEY)


def test_stage_cycle_with_no_records_writes_nothing(store: LocalObjectStore) -> None:
    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    assert stage_cycle(store, records=[], observed_at=moment) is None
    assert store.list_keys("") == []


# --- failure containment ---------------------------------------------------


class _BrokenStore:
    """A store whose every write fails, as an unreachable bucket would."""

    def put(self, key: str, data: bytes) -> None:
        raise ObjectStoreError("bucket unreachable")

    def get(self, key: str) -> bytes:
        raise ObjectNotFoundError(key)

    def exists(self, key: str) -> bool:
        return False

    def list_keys(self, prefix: str) -> list[str]:
        return []

    def delete(self, key: str) -> None:
        return None


def test_store_outage_does_not_fail_the_ingestion_run() -> None:
    """The database is authoritative; staging is a shadow write."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    assert (
        stage_cycle_or_log(_BrokenStore(), records=_cycle(moment), observed_at=moment)
        is False
    )


def test_a_bug_in_staging_is_not_swallowed() -> None:
    """Only store failures are contained; programming errors must surface."""

    with pytest.raises(ValueError, match="timezone-aware"):
        stage_cycle_or_log(
            _BrokenStore(),
            records=_cycle(datetime(2026, 9, 9, 12, 0, tzinfo=UTC)),
            observed_at=datetime(2026, 9, 9, 12, 0),
        )
