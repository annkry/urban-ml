from pathlib import Path

import polars as pl
import pytest

from urban_ml.archive.layout import STATION_STATUS, STATIONS
from urban_ml.ingestion.gbfs_client import GbfsFetchError
from urban_ml.ingestion.gbfs_ingest import (
    GbfsIngestionSummary,
    format_ingestion_summary,
    ingest_gbfs_station_feeds,
)
from urban_ml.processing.gbfs_transform import GbfsTransformError
from urban_ml.staging.objects import (
    LocalObjectStore,
    ObjectNotFoundError,
    ObjectStoreError,
)
from urban_ml.staging.snapshots import STATION_STATUS_KEY, STATIONS_KEY

DISCOVERY_URL = "https://example.com/gbfs/3/gbfs"
SYSTEM_INFORMATION_URL = "https://example.com/gbfs/3/system_information"
VEHICLE_TYPES_URL = "https://example.com/gbfs/3/vehicle_types"
STATION_INFORMATION_URL = "https://example.com/gbfs/3/station_information"
STATION_STATUS_URL = "https://example.com/gbfs/3/station_status"


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(root=tmp_path)


def test_format_ingestion_summary_includes_counts_and_keys() -> None:
    summary = GbfsIngestionSummary(
        discovery_url="https://example.com/gbfs/3/gbfs",
        system_id="bike_share_toronto",
        station_information_last_updated="2023-07-17T13:34:13+02:00",
        station_status_last_updated="2023-07-17T13:35:13+02:00",
        station_count=2,
        status_count=3,
        matched_station_status_count=2,
        processed_station_status_count=2,
        station_status_key="snapshots/station_status/date=2026-09-16/x.parquet",
        stations_key="snapshots/stations/date=2026-09-16/x.parquet",
    )

    formatted = format_ingestion_summary(summary)

    assert "GBFS ingestion completed" in formatted
    assert "System: bike_share_toronto" in formatted
    assert "Stations discovered: 2" in formatted
    assert "Station statuses discovered: 3" in formatted
    assert "Stations with matching status: 2" in formatted
    assert "Station status staged at: snapshots/station_status/" in formatted
    assert "Stations staged at: snapshots/stations/" in formatted


def _payloads() -> dict[str, dict]:
    return {
        DISCOVERY_URL: {
            "last_updated": "2026-07-18T12:00:00+00:00",
            "ttl": 0,
            "version": "3.0",
            "data": {
                "feeds": [
                    {"name": "system_information", "url": SYSTEM_INFORMATION_URL},
                    {"name": "vehicle_types", "url": VEHICLE_TYPES_URL},
                    {"name": "station_information", "url": STATION_INFORMATION_URL},
                    {"name": "station_status", "url": STATION_STATUS_URL},
                ]
            },
        },
        SYSTEM_INFORMATION_URL: {
            "last_updated": "2026-07-18T12:00:00+00:00",
            "ttl": 60,
            "version": "3.0",
            "data": {"system_id": "bike_share_toronto"},
        },
        VEHICLE_TYPES_URL: {
            "last_updated": "2026-07-18T12:00:00+00:00",
            "ttl": 60,
            "version": "3.0",
            "data": {
                "vehicle_types": [
                    {
                        "vehicle_type_id": "CLASSIC",
                        "form_factor": "bicycle",
                        "propulsion_type": "human",
                    },
                    {
                        "vehicle_type_id": "EBIKE",
                        "form_factor": "bicycle",
                        "propulsion_type": "electric_assist",
                    },
                ]
            },
        },
        STATION_INFORMATION_URL: {
            "last_updated": "2026-07-18T12:00:00+00:00",
            "ttl": 60,
            "version": "3.0",
            "data": {
                "stations": [
                    {
                        "station_id": "station-1",
                        "name": [{"text": "Main Station", "language": "en"}],
                        "lat": 43.6532,
                        "lon": -79.3832,
                        "capacity": 20,
                    }
                ]
            },
        },
        STATION_STATUS_URL: {
            "last_updated": "2026-07-18T12:00:00+00:00",
            "ttl": 60,
            "version": "3.0",
            "data": {
                "stations": [
                    {
                        "station_id": "station-1",
                        "num_vehicles_available": 7,
                        "num_docks_available": 13,
                        "is_installed": True,
                        "is_renting": True,
                        "is_returning": True,
                        "last_reported": "2026-07-18T12:00:00+00:00",
                        "vehicle_types_available": [
                            {"vehicle_type_id": "CLASSIC", "count": 5},
                            {"vehicle_type_id": "EBIKE", "count": 2},
                        ],
                    }
                ]
            },
        },
    }


def _fetcher_for(payloads: dict[str, dict]):
    def fake_fetcher(url: str, timeout_seconds: float) -> dict:
        return payloads[url]

    return fake_fetcher


def _read(store: LocalObjectStore, key: str) -> pl.DataFrame:
    return pl.read_parquet(store.root / key)


def test_ingest_stages_snapshots_and_serving_window(store: LocalObjectStore) -> None:
    summary = ingest_gbfs_station_feeds(
        DISCOVERY_URL,
        timeout_seconds=10.0,
        object_store=store,
        json_fetcher=_fetcher_for(_payloads()),
    )

    assert summary.system_id == "bike_share_toronto"
    assert summary.station_count == 1
    assert summary.matched_station_status_count == 1
    assert summary.processed_station_status_count == 1
    assert summary.station_status_key.startswith(f"snapshots/{STATION_STATUS}/date=")
    assert summary.stations_key.startswith(f"snapshots/{STATIONS}/date=")

    status = _read(store, summary.station_status_key)
    assert status["station_id"].to_list() == ["station-1"]
    assert status["num_vehicles_available"].to_list() == [7]
    assert status["num_vehicles_electric"].to_list() == [2]

    stations = _read(store, summary.stations_key)
    assert stations["station_name"].to_list() == ["Main Station"]

    window = _read(store, STATION_STATUS_KEY)
    assert window["station_id"].to_list() == ["station-1"]
    assert _read(store, STATIONS_KEY).equals(stations)


def test_an_upstream_failure_propagates_and_stages_nothing(
    store: LocalObjectStore,
) -> None:
    """Fails before system_information is even fetched. With no second sink
    there is nothing to record the failure in; the caller sees the exception
    and the bucket is untouched."""

    def failing_fetcher(url: str, timeout_seconds: float) -> dict:
        raise GbfsFetchError("upstream is down")

    with pytest.raises(GbfsFetchError, match="upstream is down"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            object_store=store,
            json_fetcher=failing_fetcher,
        )

    assert store.list_keys("") == []


def test_a_transform_failure_propagates_and_stages_nothing(
    store: LocalObjectStore,
) -> None:
    payloads = _payloads()
    payloads[STATION_STATUS_URL]["data"]["stations"][0]["station_id"] = "unknown"

    with pytest.raises(GbfsTransformError, match="missing from"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            object_store=store,
            json_fetcher=_fetcher_for(payloads),
        )

    assert store.list_keys("") == []


def test_an_empty_status_feed_is_a_failure_not_an_empty_snapshot(
    store: LocalObjectStore,
) -> None:
    """A feed that lists no stations is an upstream fault. Staging an empty
    snapshot would later read as a real observation of an empty system."""

    payloads = _payloads()
    payloads[STATION_STATUS_URL]["data"]["stations"] = []

    with pytest.raises(GbfsTransformError, match="no rows"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            object_store=store,
            json_fetcher=_fetcher_for(payloads),
        )

    assert store.list_keys("") == []


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


def test_a_store_outage_fails_the_cycle() -> None:
    """Object storage is the only sink, so an outage there is the run
    failing, not a shadow write to shrug off."""

    with pytest.raises(ObjectStoreError, match="bucket unreachable"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            object_store=_BrokenStore(),
            json_fetcher=_fetcher_for(_payloads()),
        )
