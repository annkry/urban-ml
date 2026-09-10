from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from urban_ml.archive.export import parquet_schema
from urban_ml.archive.layout import STATION_STATUS, STATIONS, partition_path
from urban_ml.archive.retention import trim_staged_days
from urban_ml.archive.staged import (
    collapse_station_changes,
    days_present_in_bucket,
    delete_staged_day,
    export_staged_day,
    read_staged_day,
    staged_days,
    staged_keys_for_day,
)
from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.staging.objects import LocalObjectStore
from urban_ml.staging.snapshots import stage_cycle

SYSTEM_ID = "toronto"
DAY = date(2026, 9, 10)


def _status(station_id: str, observed_at: datetime, *, vehicles: int) -> StationStatus:
    return StationStatus(
        observed_at=observed_at,
        system_id=SYSTEM_ID,
        station_id=station_id,
        num_vehicles_available=vehicles,
        num_docks_available=12,
        is_installed=True,
        is_renting=True,
        is_returning=True,
    )


def _station(
    station_id: str,
    observed_at: datetime,
    *,
    capacity: int = 20,
    name: str | None = None,
) -> Station:
    return Station(
        system_id=SYSTEM_ID,
        station_id=station_id,
        station_name=name or f"Station {station_id}",
        address="1 Example St",
        lat=43.65,
        lon=-79.38,
        capacity=capacity,
        is_charging_station=False,
        observed_at=observed_at,
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(root=tmp_path)


def _stage_day(
    store: LocalObjectStore,
    *,
    day: date = DAY,
    cycles: int = 6,
    stations: tuple[str, ...] = ("a", "b"),
    capacity_change_at: int | None = None,
) -> None:
    """Stage a day of full snapshots, optionally changing a capacity midway."""

    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    for index in range(cycles):
        moment = start + timedelta(minutes=5 * index)
        changed = capacity_change_at is not None and index >= capacity_change_at
        stage_cycle(
            store,
            status_records=[_status(sid, moment, vehicles=index) for sid in stations],
            station_records=[
                _station(sid, moment, capacity=33 if changed and sid == "a" else 20)
                for sid in stations
            ],
            observed_at=moment,
        )


# --- discovery -------------------------------------------------------------


def test_staged_days_reads_the_key_prefixes(store: LocalObjectStore) -> None:
    """One listing, not a download of every object."""

    _stage_day(store, day=DAY, cycles=2)
    _stage_day(store, day=DAY - timedelta(days=1), cycles=2)

    assert staged_days(store, table=STATION_STATUS) == {DAY, DAY - timedelta(days=1)}


def test_days_present_follows_station_status(store: LocalObjectStore) -> None:
    """stations began being staged later, so a union would claim days with no
    availability history to publish."""

    _stage_day(store, cycles=2)
    for key in staged_keys_for_day(store, table=STATIONS, day=DAY):
        store.delete(key)

    assert days_present_in_bucket(store) == {DAY}


def test_a_day_with_nothing_staged_reads_as_empty(store: LocalObjectStore) -> None:
    frame = read_staged_day(store, table=STATION_STATUS, day=DAY)

    assert frame.is_empty()
    assert frame.columns == list(parquet_schema(STATION_STATUS))


# --- concatenation ---------------------------------------------------------


def test_a_day_of_snapshots_concatenates_into_one_frame(
    store: LocalObjectStore,
) -> None:
    _stage_day(store, cycles=6, stations=("a", "b"))

    frame = read_staged_day(store, table=STATION_STATUS, day=DAY)

    assert frame.height == 12  # 6 cycles x 2 stations
    assert frame["observed_at"].n_unique() == 6
    assert frame["observed_at"].is_sorted()


def test_only_the_requested_day_is_read(store: LocalObjectStore) -> None:
    _stage_day(store, day=DAY, cycles=3)
    _stage_day(store, day=DAY - timedelta(days=1), cycles=3)

    frame = read_staged_day(store, table=STATION_STATUS, day=DAY)

    assert frame["observed_at"].dt.date().unique().to_list() == [DAY]


# --- the stations collapse -------------------------------------------------


def test_collapse_keeps_one_baseline_row_per_station(store: LocalObjectStore) -> None:
    """A day with no changes still anchors every station once."""

    _stage_day(store, cycles=6, stations=("a", "b"))

    collapsed = collapse_station_changes(
        read_staged_day(store, table=STATIONS, day=DAY)
    )

    assert collapsed.height == 2
    assert sorted(collapsed["station_id"].to_list()) == ["a", "b"]


def test_collapse_records_a_change_when_one_happens(store: LocalObjectStore) -> None:
    _stage_day(store, cycles=6, stations=("a", "b"), capacity_change_at=3)

    collapsed = collapse_station_changes(
        read_staged_day(store, table=STATIONS, day=DAY)
    )

    assert collapsed.height == 3  # two baselines plus one change
    changed = collapsed.filter(pl.col("station_id") == "a").sort("observed_at")
    assert changed["capacity"].to_list() == [20, 33]


def test_collapse_is_a_pure_function_of_the_day(store: LocalObjectStore) -> None:
    """The property that lets a day be rebuilt at any time, in any order."""

    _stage_day(store, cycles=6, stations=("a", "b"), capacity_change_at=3)
    frame = read_staged_day(store, table=STATIONS, day=DAY)

    once = collapse_station_changes(frame)
    shuffled = collapse_station_changes(
        frame.sample(fraction=1.0, shuffle=True, seed=7)
    )

    assert once.equals(shuffled)
    assert collapse_station_changes(once).equals(once)


def test_collapse_of_an_empty_day_is_empty() -> None:
    empty = pl.DataFrame(schema=parquet_schema(STATIONS))

    assert collapse_station_changes(empty).is_empty()


# --- export ----------------------------------------------------------------


def test_export_writes_one_partition_per_table_day(
    store: LocalObjectStore, tmp_path: Path
) -> None:
    _stage_day(store, cycles=6, stations=("a", "b"))
    out = tmp_path / "out"

    status = export_staged_day(store, table=STATION_STATUS, day=DAY, staging_dir=out)
    stations = export_staged_day(store, table=STATIONS, day=DAY, staging_dir=out)

    assert status is not None and stations is not None
    assert status[0] == out / partition_path(STATION_STATUS, DAY)
    assert status[1] == 12  # station_status is published whole
    assert stations[1] == 2  # stations is collapsed


def test_export_of_a_day_with_nothing_staged_returns_none(
    store: LocalObjectStore, tmp_path: Path
) -> None:
    """An ingestion outage must leave a missing partition, not an empty file
    that later reads as a real day with no rows."""

    result = export_staged_day(
        store, table=STATION_STATUS, day=DAY, staging_dir=tmp_path
    )

    assert result is None
    assert not (tmp_path / partition_path(STATION_STATUS, DAY)).exists()


def test_exported_partitions_match_the_archive_schema(
    store: LocalObjectStore, tmp_path: Path
) -> None:
    _stage_day(store, cycles=3)

    for table in (STATION_STATUS, STATIONS):
        result = export_staged_day(store, table=table, day=DAY, staging_dir=tmp_path)
        assert result is not None
        written = pl.read_parquet(result[0])
        assert written.columns == list(parquet_schema(table))


def test_re_exporting_a_day_produces_the_same_file(
    store: LocalObjectStore, tmp_path: Path
) -> None:
    """Re-running the job repairs a day rather than duplicating it."""

    _stage_day(store, cycles=6, capacity_change_at=3)

    first = export_staged_day(
        store, table=STATIONS, day=DAY, staging_dir=tmp_path / "a"
    )
    second = export_staged_day(
        store, table=STATIONS, day=DAY, staging_dir=tmp_path / "b"
    )

    assert first is not None and second is not None
    assert pl.read_parquet(first[0]).equals(pl.read_parquet(second[0]))


# --- deletion --------------------------------------------------------------


def test_delete_removes_only_the_named_table_day(store: LocalObjectStore) -> None:
    _stage_day(store, day=DAY, cycles=3)
    _stage_day(store, day=DAY - timedelta(days=1), cycles=3)

    removed = delete_staged_day(store, table=STATION_STATUS, day=DAY)

    assert removed == 3
    assert staged_keys_for_day(store, table=STATION_STATUS, day=DAY) == []
    assert staged_keys_for_day(store, table=STATIONS, day=DAY) != []
    assert (
        staged_keys_for_day(store, table=STATION_STATUS, day=DAY - timedelta(days=1))
        != []
    )


def test_trim_removes_both_tables_for_a_published_day(
    store: LocalObjectStore,
) -> None:
    """A published day is published in full, so nothing staged for it is
    still needed."""

    _stage_day(store, day=DAY, cycles=3)

    removed = trim_staged_days(store, days=[DAY])

    assert removed == 6  # 3 cycles x 2 tables
    assert days_present_in_bucket(store) == set()


def test_deleting_a_day_that_is_already_gone_is_a_no_op(
    store: LocalObjectStore,
) -> None:
    assert delete_staged_day(store, table=STATION_STATUS, day=DAY) == 0
