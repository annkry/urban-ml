from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from urban_ml.archive.export import parquet_schema
from urban_ml.archive.layout import STATION_STATUS, STATIONS
from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.modeling.predict import LOOKBACK_MINUTES
from urban_ml.staging.objects import (
    LocalObjectStore,
    ObjectNotFoundError,
    ObjectStoreError,
)
from urban_ml.staging.snapshots import (
    RECENT_WINDOW_MINUTES,
    STATION_STATUS_KEY,
    STATIONS_KEY,
    covers_lookback,
    history_span,
    read_station_status,
    read_stations,
    snapshot_key,
    stage_cycle,
    stage_cycle_or_log,
    stations_frame,
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


def _station(station_id: str, observed_at: datetime, *, capacity: int = 20) -> Station:
    return Station(
        system_id=SYSTEM_ID,
        station_id=station_id,
        station_name=f"Name {station_id}",
        address="1 Example St",
        lat=43.65,
        lon=-79.38,
        capacity=capacity,
        is_charging_station=False,
        observed_at=observed_at,
    )


def _stations(observed_at: datetime, *, count: int = 2) -> list[Station]:
    return [_station(f"station-{index}", observed_at) for index in range(count)]


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

    staged = stage_cycle(
        store,
        status_records=_cycle(observed_at),
        station_records=_stations(observed_at),
        observed_at=observed_at,
    )

    assert staged is not None
    for key in (staged.station_status_key, staged.stations_key):
        assert "date=2026-09-09" in key
        rows = pl.read_parquet(store.get(key))
        assert rows["observed_at"].dt.date().unique().to_list() == [observed_at.date()]


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


def test_read_station_status_returns_an_empty_typed_frame_before_the_first_write(
    store: LocalObjectStore,
) -> None:
    window = read_station_status(store)

    assert window.is_empty()
    assert "observed_at" in window.columns


def test_read_station_status_recovers_from_an_unreadable_window(
    store: LocalObjectStore,
) -> None:
    store.put(STATION_STATUS_KEY, b"not parquet at all")

    assert read_station_status(store).is_empty()


def test_staged_rows_round_trip_through_the_archive_schema(
    store: LocalObjectStore,
) -> None:
    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    records = _cycle(moment, stations=3)

    stage_cycle(
        store,
        status_records=records,
        station_records=_stations(moment, count=3),
        observed_at=moment,
    )
    window = read_station_status(store)

    assert window.height == 3
    assert window.schema == status_frame(records).schema


def test_both_tables_are_staged_whole_under_snapshots(
    store: LocalObjectStore,
) -> None:
    """serving/ holds the rolling window; snapshots/ holds full rows."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    staged = stage_cycle(
        store,
        status_records=_cycle(moment, stations=3),
        station_records=_stations(moment, count=3),
        observed_at=moment,
    )

    assert staged is not None
    assert staged.station_status_key.startswith("snapshots/station_status/")
    assert staged.stations_key.startswith("snapshots/stations/")
    assert staged.station_status_rows == 3
    assert staged.stations_rows == 3
    assert pl.read_parquet(store.get(staged.stations_key)).height == 3


def test_staged_stations_match_the_archive_schema(store: LocalObjectStore) -> None:
    """The daily export concatenates these; a drifting schema fails there."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    records = _stations(moment, count=2)

    staged = stage_cycle(
        store,
        status_records=_cycle(moment),
        station_records=records,
        observed_at=moment,
    )

    assert staged is not None
    written = pl.read_parquet(store.get(staged.stations_key))
    assert written.schema == stations_frame(records).schema
    assert written.columns == list(parquet_schema(STATIONS))


def test_the_serving_window_holds_only_station_status(
    store: LocalObjectStore,
) -> None:
    """Capacity lives in its own serving file, not folded into the window."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    stage_cycle(
        store,
        status_records=_cycle(moment),
        station_records=_stations(moment),
        observed_at=moment,
    )

    assert read_station_status(store).columns == list(parquet_schema(STATION_STATUS))
    assert read_stations(store).columns == list(parquet_schema(STATIONS))


def test_serving_stations_are_published_for_predict(
    store: LocalObjectStore,
) -> None:
    """A fixed key, so serving never lists the bucket to find the newest."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    stage_cycle(
        store,
        status_records=_cycle(moment),
        station_records=_stations(moment, count=3),
        observed_at=moment,
    )

    stations = read_stations(store)
    assert store.exists(STATIONS_KEY)
    assert stations.height == 3
    assert stations.select(["station_id", "capacity"]).to_dicts() == [
        {"station_id": "station-0", "capacity": 20},
        {"station_id": "station-1", "capacity": 20},
        {"station_id": "station-2", "capacity": 20},
    ]


def test_serving_stations_never_disagree_with_the_snapshot(
    store: LocalObjectStore,
) -> None:
    """Both come from one serialisation, so they cannot drift apart."""

    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    staged = stage_cycle(
        store,
        status_records=_cycle(moment),
        station_records=_stations(moment, count=3),
        observed_at=moment,
    )

    assert staged is not None
    assert store.get(STATIONS_KEY) == store.get(staged.stations_key)


def test_serving_stations_track_a_capacity_change(store: LocalObjectStore) -> None:
    """Overwritten every cycle, so a re-signed station is current next tick."""

    first = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    stage_cycle(
        store,
        status_records=_cycle(first),
        station_records=[_station("station-0", first, capacity=20)],
        observed_at=first,
    )

    later = first + timedelta(minutes=5)
    stage_cycle(
        store,
        status_records=_cycle(later),
        station_records=[_station("station-0", later, capacity=31)],
        observed_at=later,
    )

    assert read_stations(store)["capacity"].to_list() == [31]


def test_read_stations_returns_an_empty_typed_frame_before_the_first_write(
    store: LocalObjectStore,
) -> None:
    stations = read_stations(store)

    assert stations.is_empty()
    assert "capacity" in stations.columns


def test_read_stations_recovers_from_an_unreadable_file(
    store: LocalObjectStore,
) -> None:
    store.put(STATIONS_KEY, b"not parquet at all")

    assert read_stations(store).is_empty()


def test_stage_cycle_writes_the_durable_snapshot_before_the_window(
    store: LocalObjectStore,
) -> None:
    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    staged = stage_cycle(
        store,
        status_records=_cycle(moment),
        station_records=_stations(moment),
        observed_at=moment,
    )

    assert staged is not None
    assert store.exists(staged.station_status_key)
    assert store.exists(staged.stations_key)
    assert store.exists(STATION_STATUS_KEY)


def test_stage_cycle_with_no_records_writes_nothing(store: LocalObjectStore) -> None:
    moment = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)

    staged = stage_cycle(
        store,
        status_records=[],
        station_records=_stations(moment),
        observed_at=moment,
    )

    assert staged is None
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
        stage_cycle_or_log(
            _BrokenStore(),
            status_records=_cycle(moment),
            station_records=_stations(moment),
            observed_at=moment,
        )
        is False
    )


def test_a_bug_in_staging_is_not_swallowed() -> None:
    """Only store failures are contained; programming errors must surface."""

    with pytest.raises(ValueError, match="timezone-aware"):
        stage_cycle_or_log(
            _BrokenStore(),
            status_records=_cycle(datetime(2026, 9, 9, 12, 0, tzinfo=UTC)),
            station_records=_stations(datetime(2026, 9, 9, 12, 0, tzinfo=UTC)),
            observed_at=datetime(2026, 9, 9, 12, 0),
        )
