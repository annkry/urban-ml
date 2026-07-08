from datetime import UTC, datetime

import pytest

from urban_ml.ingestion.gbfs_client import GbfsRawStationFeeds
from urban_ml.processing.gbfs_transform import (
    GbfsTransformError,
    build_station_snapshots,
)
from urban_ml.schemas.gbfs import (
    GbfsDiscoveryResponse,
    StationInformationResponse,
    StationStatusResponse,
)


DISCOVERY_PAYLOAD = {
    "last_updated": "2023-07-17T13:34:13+02:00",
    "ttl": 0,
    "version": "3.0",
    "data": {
        "feeds": [
            {
                "name": "station_information",
                "url": "https://example.com/gbfs/3/station_information",
            },
            {
                "name": "station_status",
                "url": "https://example.com/gbfs/3/station_status",
            },
        ]
    },
}
STATION_INFORMATION_PAYLOAD = {
    "last_updated": "2023-07-17T13:34:13+02:00",
    "ttl": 60,
    "version": "3.0",
    "data": {
        "stations": [
            {
                "station_id": "station-1",
                "name": [{"text": "Main Station", "language": "en"}],
                "lat": 47.3769,
                "lon": 8.5417,
                "capacity": 20,
            }
        ]
    },
}
STATION_STATUS_PAYLOAD = {
    "last_updated": "2023-07-17T13:35:13+02:00",
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
                "last_reported": "2023-07-17T13:35:13+02:00",
            }
        ]
    },
}


def _raw_feeds(
    station_information_payload: dict = STATION_INFORMATION_PAYLOAD,
    station_status_payload: dict = STATION_STATUS_PAYLOAD,
) -> GbfsRawStationFeeds:
    return GbfsRawStationFeeds(
        discovery_url="https://example.com/gbfs/3/gbfs",
        station_information_url="https://example.com/gbfs/3/station_information",
        station_status_url="https://example.com/gbfs/3/station_status",
        discovery_payload=DISCOVERY_PAYLOAD,
        station_information_payload=station_information_payload,
        station_status_payload=station_status_payload,
        discovery=GbfsDiscoveryResponse.model_validate(DISCOVERY_PAYLOAD),
        station_information=StationInformationResponse.model_validate(
            station_information_payload
        ),
        station_status=StationStatusResponse.model_validate(station_status_payload),
    )


def test_build_station_snapshots_creates_flat_rows() -> None:
    observed_at = datetime(2026, 7, 5, 11, 6, 3, tzinfo=UTC)

    snapshots = build_station_snapshots(
        _raw_feeds(),
        system_id="toronto",
        observed_at=observed_at,
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.observed_at == observed_at
    assert snapshot.system_id == "toronto"
    assert snapshot.station_id == "station-1"
    assert snapshot.station_name == "Main Station"
    assert snapshot.lat == 47.3769
    assert snapshot.lon == 8.5417
    assert snapshot.capacity == 20
    assert snapshot.num_vehicles_available == 7
    assert snapshot.num_docks_available == 13
    assert snapshot.is_renting is True


def test_build_station_snapshots_rejects_missing_station_information() -> None:
    station_status_payload = {
        **STATION_STATUS_PAYLOAD,
        "data": {
            "stations": [
                {
                    **STATION_STATUS_PAYLOAD["data"]["stations"][0],
                    "station_id": "unknown-station",
                }
            ]
        },
    }

    with pytest.raises(GbfsTransformError):
        build_station_snapshots(
            _raw_feeds(station_status_payload=station_status_payload),
            system_id="toronto",
            observed_at=datetime(2026, 7, 5, 11, 6, 3, tzinfo=UTC),
        )
