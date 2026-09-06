from datetime import UTC, datetime

import pytest

from urban_ml.ingestion.gbfs_client import GbfsRawFeeds
from urban_ml.domain.station_status import StationStatus
from urban_ml.processing.gbfs_transform import (
    GbfsTransformError,
    build_station_status,
    build_stations,
    build_vehicle_types,
)
from urban_ml.schemas.gbfs import (
    GbfsDiscoveryResponse,
    StationInformationResponse,
    StationStatusResponse,
    SystemInformationResponse,
    VehicleTypesResponse,
)

OBSERVED_AT = datetime(2026, 7, 18, 12, tzinfo=UTC)


DISCOVERY_PAYLOAD = {
    "last_updated": "2023-07-17T13:34:13+02:00",
    "ttl": 0,
    "version": "3.0",
    "data": {
        "feeds": [
            {
                "name": "system_information",
                "url": "https://example.com/gbfs/3/system_information",
            },
            {
                "name": "vehicle_types",
                "url": "https://example.com/gbfs/3/vehicle_types",
            },
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
SYSTEM_INFORMATION_PAYLOAD = {
    "last_updated": "2023-07-17T13:34:13+02:00",
    "ttl": 60,
    "version": "3.0",
    "data": {"system_id": "example_system"},
}
VEHICLE_TYPES_PAYLOAD = {
    "last_updated": "2023-07-17T13:34:13+02:00",
    "ttl": 60,
    "version": "3.0",
    "data": {
        "vehicle_types": [
            {
                "vehicle_type_id": "CLASSIC",
                "form_factor": "bicycle",
                "propulsion_type": "human",
                "name": [{"text": "Classic Bike", "language": "en"}],
            },
            {
                "vehicle_type_id": "EBIKE",
                "form_factor": "bicycle",
                "propulsion_type": "electric_assist",
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
                "vehicle_types_available": [
                    {"vehicle_type_id": "CLASSIC", "count": 5},
                    {"vehicle_type_id": "EBIKE", "count": 2},
                ],
            }
        ]
    },
}


def _raw_feeds(
    station_information_payload: dict = STATION_INFORMATION_PAYLOAD,
    station_status_payload: dict = STATION_STATUS_PAYLOAD,
    vehicle_types_payload: dict = VEHICLE_TYPES_PAYLOAD,
) -> GbfsRawFeeds:
    return GbfsRawFeeds(
        discovery_url="https://example.com/gbfs/3/gbfs",
        system_information_url="https://example.com/gbfs/3/system_information",
        vehicle_types_url="https://example.com/gbfs/3/vehicle_types",
        station_information_url="https://example.com/gbfs/3/station_information",
        station_status_url="https://example.com/gbfs/3/station_status",
        discovery_payload=DISCOVERY_PAYLOAD,
        station_information_payload=station_information_payload,
        station_status_payload=station_status_payload,
        discovery=GbfsDiscoveryResponse.model_validate(DISCOVERY_PAYLOAD),
        system_information=SystemInformationResponse.model_validate(
            SYSTEM_INFORMATION_PAYLOAD
        ),
        vehicle_types=VehicleTypesResponse.model_validate(vehicle_types_payload),
        station_information=StationInformationResponse.model_validate(
            station_information_payload
        ),
        station_status=StationStatusResponse.model_validate(station_status_payload),
    )


def test_build_station_status_creates_narrow_status_rows() -> None:
    observed_at = datetime(2026, 7, 5, 11, 6, 3, tzinfo=UTC)

    records = build_station_status(
        _raw_feeds(),
        system_id="toronto",
        known_station_ids={"station-1"},
        observed_at=observed_at,
    )

    assert len(records) == 1
    record = records[0]
    assert record.observed_at == observed_at
    assert record.system_id == "toronto"
    assert record.station_id == "station-1"
    assert record.num_vehicles_available == 7
    assert record.num_docks_available == 13
    assert record.is_renting is True


def test_build_stations_extracts_metadata() -> None:
    stations = build_stations(
        _raw_feeds(), system_id="toronto", observed_at=OBSERVED_AT
    )

    assert len(stations) == 1
    station = stations[0]
    assert station.system_id == "toronto"
    assert station.station_id == "station-1"
    assert station.station_name == "Main Station"
    assert station.lat == 47.3769
    assert station.lon == 8.5417
    assert station.capacity == 20


def test_build_station_status_rejects_missing_station_information() -> None:
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
        build_station_status(
            _raw_feeds(station_status_payload=station_status_payload),
            system_id="toronto",
            known_station_ids={"station-1"},
            observed_at=datetime(2026, 7, 5, 11, 6, 3, tzinfo=UTC),
        )


def test_build_vehicle_types_extracts_metadata() -> None:
    vehicle_types = build_vehicle_types(_raw_feeds(), system_id="toronto")

    assert len(vehicle_types) == 2
    classic = next(vt for vt in vehicle_types if vt.vehicle_type_id == "CLASSIC")
    assert classic.system_id == "toronto"
    assert classic.form_factor == "bicycle"
    assert classic.propulsion_type == "human"
    assert classic.name == "Classic Bike"

    ebike = next(vt for vt in vehicle_types if vt.vehicle_type_id == "EBIKE")
    assert ebike.propulsion_type == "electric_assist"
    assert ebike.name is None


def _status_payload(entries: list[dict] | None) -> dict:
    """A station_status payload for one station with a given per-type split."""

    station: dict[str, object] = {
        "station_id": "station-1",
        "num_vehicles_available": 7,
        "num_vehicles_disabled": 3,
        "num_docks_available": 13,
        "num_docks_disabled": 1,
        "is_installed": True,
        "is_renting": True,
        "is_returning": True,
        "last_reported": "2023-07-17T13:35:13+02:00",
    }
    if entries is not None:
        station["vehicle_types_available"] = entries
    return {
        "last_updated": "2023-07-17T13:35:13+02:00",
        "ttl": 60,
        "version": "3.0",
        "data": {"stations": [station]},
    }


def _status_for(entries: list[dict] | None) -> StationStatus:
    return build_station_status(
        _raw_feeds(station_status_payload=_status_payload(entries)),
        system_id="toronto",
        known_station_ids={"station-1"},
        observed_at=OBSERVED_AT,
    )[0]


def test_disabled_counts_are_carried_through() -> None:
    """16% of Toronto's fleet is disabled at any moment and it occupies docks,
    so these are not cosmetic fields."""

    record = _status_for([{"vehicle_type_id": "CLASSIC", "count": 7}])

    assert record.num_vehicles_disabled == 3
    assert record.num_docks_disabled == 1


def test_vehicles_split_by_propulsion_not_by_model() -> None:
    """Grouping on propulsion_type is what keeps a new bike model from needing
    a schema change: the columns are electric and human, not per model id."""

    record = _status_for(
        [
            {"vehicle_type_id": "CLASSIC", "count": 5},
            {"vehicle_type_id": "EBIKE", "count": 2},
        ]
    )

    assert record.num_vehicles_electric == 2
    assert record.num_vehicles_human == 5
    assert (
        record.num_vehicles_electric + record.num_vehicles_human
        == record.num_vehicles_available
    )


def test_a_missing_breakdown_gives_null_not_zero() -> None:
    """No breakdown reported is different from a genuine zero, and silently
    recording zero would make the split stop summing to the total."""

    record = _status_for(None)

    assert record.num_vehicles_electric is None
    assert record.num_vehicles_human is None


def test_an_unknown_vehicle_type_fails_loudly() -> None:
    """An unrecognised type would otherwise vanish from both columns, so the
    split would quietly stop reconciling with num_vehicles_available."""

    with pytest.raises(GbfsTransformError, match="CARGO_TRIKE"):
        _status_for([{"vehicle_type_id": "CARGO_TRIKE", "count": 4}])
