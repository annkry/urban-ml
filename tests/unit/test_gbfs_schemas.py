from datetime import datetime

import pytest
from pydantic import ValidationError

from urban_ml.schemas.gbfs import (
    GbfsDiscoveryResponse,
    StationInformationResponse,
    StationStatusResponse,
)


def test_gbfs_discovery_response_accepts_official_v3_shape() -> None:
    payload = {
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

    response = GbfsDiscoveryResponse.model_validate(payload)

    assert response.version == "3.0"
    assert response.data.feeds[0].name == "station_information"


def test_station_information_response_accepts_localized_station_name() -> None:
    payload = {
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
                    "rental_methods": ["creditcard", "phone"],
                    "vehicle_docks_capacity": [
                        {"vehicle_type_ids": ["bike"], "count": 20}
                    ],
                }
            ]
        },
    }

    response = StationInformationResponse.model_validate(payload)
    station = response.data.stations[0]

    assert station.name[0].text == "Main Station"
    assert station.capacity == 20


def test_station_status_response_accepts_official_vehicle_count_fields() -> None:
    payload = {
        "last_updated": "2023-07-17T13:34:13+02:00",
        "ttl": 60,
        "version": "3.0",
        "data": {
            "stations": [
                {
                    "station_id": "station-1",
                    "num_vehicles_available": 7,
                    "vehicle_types_available": [
                        {"vehicle_type_id": "bike", "count": 5},
                        {"vehicle_type_id": "ebike", "count": 2},
                    ],
                    "num_docks_available": 13,
                    "vehicle_docks_available": [
                        {"vehicle_type_ids": ["bike", "ebike"], "count": 13}
                    ],
                    "is_installed": True,
                    "is_renting": True,
                    "is_returning": True,
                    "last_reported": "2023-07-17T13:34:13+02:00",
                }
            ]
        },
    }

    response = StationStatusResponse.model_validate(payload)
    station = response.data.stations[0]

    assert station.num_vehicles_available == 7
    assert isinstance(station.last_reported, datetime)


def test_station_status_rejects_negative_vehicle_count() -> None:
    payload = {
        "last_updated": "2023-07-17T13:34:13+02:00",
        "ttl": 60,
        "version": "3.0",
        "data": {
            "stations": [
                {
                    "station_id": "station-1",
                    "num_vehicles_available": -1,
                    "is_installed": True,
                    "is_renting": True,
                    "is_returning": True,
                    "last_reported": "2023-07-17T13:34:13+02:00",
                }
            ]
        },
    }

    with pytest.raises(ValidationError):
        StationStatusResponse.model_validate(payload)


def test_station_information_rejects_v2_plain_string_name() -> None:
    payload = {
        "last_updated": "2023-07-17T13:34:13+02:00",
        "ttl": 60,
        "version": "3.0",
        "data": {
            "stations": [
                {
                    "station_id": "station-1",
                    "name": "Main Station",
                    "lat": 47.3769,
                    "lon": 8.5417,
                }
            ]
        },
    }

    with pytest.raises(ValidationError):
        StationInformationResponse.model_validate(payload)
