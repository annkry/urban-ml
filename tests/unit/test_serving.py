from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.modeling.predict import (
    LOOKBACK_MINUTES,
    InsufficientHistoryError,
    build_feature_row,
)
from urban_ml.staging.objects import LocalObjectStore, ObjectStore, ObjectStoreError
from urban_ml.staging.serving import ServingData, station_details, system_ids
from urban_ml.staging.snapshots import RECENT_WINDOW_MINUTES, stage_cycle

SYSTEM_ID = "toronto"
STATION_ID = "station-1"
CYCLE_MINUTES = 5


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


def _station(station_id: str, observed_at: datetime, *, capacity: int = 20) -> Station:
    return Station(
        system_id=SYSTEM_ID,
        station_id=station_id,
        station_name=f"Station {station_id}",
        address="1 Example St",
        lat=43.6532,
        lon=-79.3832,
        capacity=capacity,
        is_charging_station=False,
        observed_at=observed_at,
    )


def _seed(store: ObjectStore, *, cycles: int, station_ids: tuple[str, ...]) -> datetime:
    """Stage `cycles` five-minute cycles, returning the newest instant."""

    newest = datetime.now(UTC).replace(microsecond=0)
    for index in reversed(range(cycles)):
        moment = newest - timedelta(minutes=CYCLE_MINUTES * index)
        stage_cycle(
            store,
            status_records=[
                _status(station_id, moment, vehicles=(index * 3) % 17)
                for station_id in station_ids
            ],
            station_records=[_station(sid, moment) for sid in station_ids],
            observed_at=moment,
        )
    return newest


class _CountingStore:
    """Wraps a store and counts reads, to observe the cache."""

    def __init__(self, inner: ObjectStore) -> None:
        self.inner = inner
        self.reads = 0

    def put(self, key: str, data: bytes) -> None:
        self.inner.put(key, data)

    def get(self, key: str) -> bytes:
        self.reads += 1
        return self.inner.get(key)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def list_keys(self, prefix: str) -> list[str]:
        return self.inner.list_keys(prefix)

    def delete(self, key: str) -> None:
        self.inner.delete(key)


# --- train/serve parity ----------------------------------------------------


def test_a_full_window_yields_no_null_lag_or_rolling_features(
    tmp_path: Path,
) -> None:
    """What "deep enough" buys: the 60-minute inputs are actually populated."""

    store = LocalObjectStore(root=tmp_path)
    _seed(
        store, cycles=RECENT_WINDOW_MINUTES // CYCLE_MINUTES, station_ids=(STATION_ID,)
    )
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)
    station = station_details(
        serving.stations(), system_id=SYSTEM_ID, station_id=STATION_ID
    )
    assert station is not None

    row = build_feature_row(
        serving.window(), system_id=SYSTEM_ID, station_id=STATION_ID, station=station
    )

    for column in ("lag_60m_vehicles", "roll_mean_60m", "roll_mean_30m"):
        assert row[column][0] is not None


def test_build_feature_row_ignores_other_stations(tmp_path: Path) -> None:
    """The window holds every station; a feature row must use only one."""

    store = LocalObjectStore(root=tmp_path)
    _seed(store, cycles=19, station_ids=(STATION_ID, "station-2"))
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)
    station = station_details(
        serving.stations(), system_id=SYSTEM_ID, station_id=STATION_ID
    )
    assert station is not None

    row = build_feature_row(
        serving.window(), system_id=SYSTEM_ID, station_id=STATION_ID, station=station
    )

    assert row.height == 1
    assert row["station_id"][0] == STATION_ID


def test_build_feature_row_without_history_raises(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path)
    _seed(store, cycles=3, station_ids=("station-2",))
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)

    with pytest.raises(InsufficientHistoryError):
        build_feature_row(
            serving.window(),
            system_id=SYSTEM_ID,
            station_id=STATION_ID,
            station=_station(STATION_ID, datetime.now(UTC)),
        )


def test_build_feature_row_ignores_rows_older_than_the_lookback(
    tmp_path: Path,
) -> None:
    """The staged window is deliberately wider than serving needs."""

    store = LocalObjectStore(root=tmp_path)
    newest = _seed(
        store, cycles=RECENT_WINDOW_MINUTES // CYCLE_MINUTES, station_ids=(STATION_ID,)
    )
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)
    station = station_details(
        serving.stations(), system_id=SYSTEM_ID, station_id=STATION_ID
    )
    assert station is not None

    window = serving.window()
    assert window["observed_at"].min() < newest - timedelta(minutes=LOOKBACK_MINUTES)

    row = build_feature_row(
        window, system_id=SYSTEM_ID, station_id=STATION_ID, station=station
    )

    assert row["observed_at"][0] == newest


# --- station lookup --------------------------------------------------------


def test_station_details_reconstructs_the_domain_model(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path)
    _seed(store, cycles=1, station_ids=(STATION_ID,))
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)

    station = station_details(
        serving.stations(), system_id=SYSTEM_ID, station_id=STATION_ID
    )

    assert station is not None
    assert station.capacity == 20
    assert station.station_name == f"Station {STATION_ID}"


def test_station_details_is_none_for_an_unknown_station(tmp_path: Path) -> None:
    """None is what becomes a 404."""

    store = LocalObjectStore(root=tmp_path)
    _seed(store, cycles=1, station_ids=(STATION_ID,))
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)

    assert (
        station_details(serving.stations(), system_id=SYSTEM_ID, station_id="nope")
        is None
    )


def test_station_details_is_none_for_another_system(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path)
    _seed(store, cycles=1, station_ids=(STATION_ID,))
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)

    assert (
        station_details(serving.stations(), system_id="montreal", station_id=STATION_ID)
        is None
    )


def test_system_ids_reads_the_station_list(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path)
    _seed(store, cycles=1, station_ids=(STATION_ID,))
    serving = ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)

    assert system_ids(serving.stations()) == [SYSTEM_ID]


def test_system_ids_of_an_empty_list_is_empty() -> None:
    assert system_ids(pl.DataFrame()) == []


# --- caching ---------------------------------------------------------------


def test_repeated_reads_inside_the_ttl_hit_the_store_once(tmp_path: Path) -> None:
    """Every prediction reads both files; without this each one is two round
    trips to Cloud Storage."""

    inner = LocalObjectStore(root=tmp_path)
    _seed(inner, cycles=2, station_ids=(STATION_ID,))
    counting = _CountingStore(inner)
    serving = ServingData(store=counting, window_ttl=60.0, stations_ttl=60.0)

    for _ in range(5):
        serving.window()
        serving.stations()

    assert counting.reads == 2


def test_an_expired_entry_is_read_again(tmp_path: Path) -> None:
    inner = LocalObjectStore(root=tmp_path)
    _seed(inner, cycles=2, station_ids=(STATION_ID,))
    counting = _CountingStore(inner)
    serving = ServingData(store=counting, window_ttl=0.0, stations_ttl=0.0)

    serving.window()
    serving.window()

    assert counting.reads == 2


def test_invalidate_forces_a_fresh_read(tmp_path: Path) -> None:
    inner = LocalObjectStore(root=tmp_path)
    _seed(inner, cycles=2, station_ids=(STATION_ID,))
    counting = _CountingStore(inner)
    serving = ServingData(store=counting, window_ttl=600.0, stations_ttl=600.0)

    serving.window()
    serving.invalidate()
    serving.window()

    assert counting.reads == 2


def test_the_window_and_stations_are_cached_separately(tmp_path: Path) -> None:
    inner = LocalObjectStore(root=tmp_path)
    _seed(inner, cycles=2, station_ids=(STATION_ID,))
    counting = _CountingStore(inner)
    serving = ServingData(store=counting, window_ttl=600.0, stations_ttl=0.0)

    serving.window()
    serving.window()
    serving.stations()
    serving.stations()

    assert counting.reads == 3  # window once, stations twice


def test_reading_without_a_configured_store_raises(tmp_path: Path) -> None:
    """A service started with no GCS_BUCKET fails the way an outage does, so
    /ready reports it rather than the process refusing to start."""

    serving = ServingData(store=None)

    with pytest.raises(ObjectStoreError, match="GCS_BUCKET"):
        serving.window()
