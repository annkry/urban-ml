from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from urban_ml.domain.station import Station as StationData
from urban_ml.domain.station_status import StationStatus
from urban_ml.domain.station_vehicle_availability import StationVehicleAvailability
from urban_ml.domain.vehicle_type import VehicleType as VehicleTypeData
from urban_ml.ingestion.gbfs_client import GbfsRawFeeds
from urban_ml.schemas.gbfs import (
    GbfsDiscoveryResponse,
    StationInformationResponse,
    StationStatusResponse,
    SystemInformationResponse,
    VehicleTypesResponse,
)
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    RawGbfsPayload,
    Station,
    StationStatusRecord,
    StationVehicleAvailabilityRecord,
    VehicleType,
)
from urban_ml.storage.repository import (
    complete_ingestion_run,
    fail_ingestion_run,
    save_raw_gbfs_payload,
    save_station_status,
    save_station_vehicle_availability,
    start_ingestion_run,
    upsert_stations,
    upsert_vehicle_types,
)

OBSERVED_AT = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _raw_feeds() -> GbfsRawFeeds:
    discovery_payload = {
        "last_updated": "2026-07-18T12:00:00Z",
        "ttl": 0,
        "version": "3.0",
        "data": {"feeds": []},
    }
    system_information_payload = {
        "last_updated": "2026-07-18T12:00:00Z",
        "ttl": 60,
        "version": "3.0",
        "data": {"system_id": "toronto"},
    }
    vehicle_types_payload = {
        "last_updated": "2026-07-18T12:00:00Z",
        "ttl": 60,
        "version": "3.0",
        "data": {"vehicle_types": []},
    }
    station_information_payload = {
        "last_updated": "2026-07-18T12:00:00Z",
        "ttl": 60,
        "version": "3.0",
        "data": {"stations": []},
    }
    station_status_payload = {
        "last_updated": "2026-07-18T12:00:00Z",
        "ttl": 60,
        "version": "3.0",
        "data": {"stations": []},
    }
    return GbfsRawFeeds(
        discovery_url="https://example.com/gbfs.json",
        system_information_url="https://example.com/system_information",
        vehicle_types_url="https://example.com/vehicle_types",
        station_information_url="https://example.com/station_information",
        station_status_url="https://example.com/station_status",
        discovery_payload=discovery_payload,
        station_information_payload=station_information_payload,
        station_status_payload=station_status_payload,
        discovery=GbfsDiscoveryResponse.model_validate(discovery_payload),
        system_information=SystemInformationResponse.model_validate(
            system_information_payload
        ),
        vehicle_types=VehicleTypesResponse.model_validate(vehicle_types_payload),
        station_information=StationInformationResponse.model_validate(
            station_information_payload
        ),
        station_status=StationStatusResponse.model_validate(station_status_payload),
    )


def _station_data(**overrides: object) -> StationData:
    defaults = dict(
        system_id="toronto",
        station_id="station-1",
        station_name="Main Station",
        lat=43.6532,
        lon=-79.3832,
        capacity=20,
    )
    defaults.update(overrides)
    return StationData(**defaults)  # type: ignore[arg-type]


def _station_status() -> StationStatus:
    return StationStatus(
        observed_at=OBSERVED_AT,
        system_id="toronto",
        station_id="station-1",
        num_vehicles_available=7,
        num_docks_available=13,
        is_installed=True,
        is_renting=True,
        is_returning=True,
        last_reported=OBSERVED_AT,
    )


def _vehicle_type_data(**overrides: object) -> VehicleTypeData:
    defaults = dict(
        system_id="toronto",
        vehicle_type_id="CLASSIC",
        form_factor="bicycle",
        propulsion_type="human",
        name="Classic Bike",
    )
    defaults.update(overrides)
    return VehicleTypeData(**defaults)  # type: ignore[arg-type]


def _availability(**overrides: object) -> StationVehicleAvailability:
    defaults = dict(
        observed_at=OBSERVED_AT,
        system_id="toronto",
        station_id="station-1",
        vehicle_type_id="CLASSIC",
        count=5,
    )
    defaults.update(overrides)
    return StationVehicleAvailability(**defaults)  # type: ignore[arg-type]


def test_save_raw_gbfs_payload_persists_status_payload_only(session) -> None:
    save_raw_gbfs_payload(
        session, _raw_feeds(), system_id="toronto", observed_at=OBSERVED_AT
    )
    session.commit()

    stored = session.scalars(select(RawGbfsPayload)).one()
    assert stored.system_id == "toronto"
    assert stored.observed_at.replace(tzinfo=UTC) == OBSERVED_AT
    assert stored.discovery_url == "https://example.com/gbfs.json"
    assert stored.station_status_payload["version"] == "3.0"


def test_upsert_stations_inserts_new_station(session) -> None:
    upsert_stations(session, [_station_data()])
    session.commit()

    stored = session.scalars(select(Station)).one()
    assert stored.station_id == "station-1"
    assert stored.station_name == "Main Station"
    assert stored.capacity == 20


def test_upsert_stations_updates_existing_station_in_place(session) -> None:
    upsert_stations(session, [_station_data(capacity=20)])
    session.commit()

    upsert_stations(session, [_station_data(capacity=25, station_name="Renamed")])
    session.commit()

    stations = session.scalars(select(Station)).all()
    assert len(stations) == 1
    assert stations[0].capacity == 25
    assert stations[0].station_name == "Renamed"


def test_save_station_status_persists_narrow_rows(session) -> None:
    save_station_status(session, [_station_status()])
    session.commit()

    stored = session.scalars(select(StationStatusRecord)).one()
    assert stored.station_id == "station-1"
    assert stored.num_vehicles_available == 7
    assert stored.is_renting is True


def test_save_station_status_rejects_exact_duplicate(session) -> None:
    save_station_status(session, [_station_status()])
    session.commit()

    save_station_status(session, [_station_status()])
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    stored = session.scalars(select(StationStatusRecord)).all()
    assert len(stored) == 1


def test_ingestion_run_lifecycle_success(session) -> None:
    run = start_ingestion_run(session, started_at=OBSERVED_AT)
    assert run.status == IngestionRunStatus.RUNNING
    assert run.system_id is None

    complete_ingestion_run(
        session, run, system_id="toronto", finished_at=OBSERVED_AT, row_count=42
    )

    stored = session.scalars(select(IngestionRun)).one()
    assert stored.status == IngestionRunStatus.SUCCESS
    assert stored.system_id == "toronto"
    assert stored.row_count == 42


def test_ingestion_run_lifecycle_failure_before_system_id_known(session) -> None:
    run = start_ingestion_run(session, started_at=OBSERVED_AT)

    save_station_status(session, [_station_status()])

    fail_ingestion_run(
        session, run, system_id=None, finished_at=OBSERVED_AT, error_message="boom"
    )

    stored_run = session.scalars(select(IngestionRun)).one()
    assert stored_run.status == IngestionRunStatus.FAILURE
    assert stored_run.system_id is None
    assert stored_run.error_message == "boom"

    assert session.scalars(select(StationStatusRecord)).first() is None


def test_ingestion_run_lifecycle_failure_after_system_id_known(session) -> None:
    run = start_ingestion_run(session, started_at=OBSERVED_AT)

    fail_ingestion_run(
        session,
        run,
        system_id="toronto",
        finished_at=OBSERVED_AT,
        error_message="transform failed",
    )

    stored_run = session.scalars(select(IngestionRun)).one()
    assert stored_run.status == IngestionRunStatus.FAILURE
    assert stored_run.system_id == "toronto"


def test_upsert_vehicle_types_inserts_new_type(session) -> None:
    upsert_vehicle_types(session, [_vehicle_type_data()])
    session.commit()

    stored = session.scalars(select(VehicleType)).one()
    assert stored.vehicle_type_id == "CLASSIC"
    assert stored.form_factor == "bicycle"
    assert stored.propulsion_type == "human"
    assert stored.name == "Classic Bike"


def test_upsert_vehicle_types_updates_existing_type_in_place(session) -> None:
    upsert_vehicle_types(session, [_vehicle_type_data(propulsion_type="human")])
    session.commit()

    upsert_vehicle_types(
        session, [_vehicle_type_data(propulsion_type="electric_assist")]
    )
    session.commit()

    types = session.scalars(select(VehicleType)).all()
    assert len(types) == 1
    assert types[0].propulsion_type == "electric_assist"


def test_save_station_vehicle_availability_persists_rows(session) -> None:
    save_station_vehicle_availability(session, [_availability()])
    session.commit()

    stored = session.scalars(select(StationVehicleAvailabilityRecord)).one()
    assert stored.station_id == "station-1"
    assert stored.vehicle_type_id == "CLASSIC"
    assert stored.count == 5
