from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from urban_ml.domain.station import Station as StationData
from urban_ml.domain.station_status import StationStatus
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
    Station,
    IngestionRun,
    IngestionRunStatus,
    StationStatusRecord,
    VehicleType,
)
from urban_ml.storage.repository import (
    TRACKED_STATION_FIELDS,
    complete_ingestion_run,
    fail_ingestion_run,
    fetch_recent_station_status,
    get_station,
    latest_stations,
    record_station_changes,
    save_station_status,
    start_ingestion_run,
    station_status_history_query,
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


def test_get_station_returns_none_when_missing(session) -> None:
    assert get_station(session, system_id="toronto", station_id="missing") is None


def test_fetch_recent_station_status_filters_by_station_and_since(session) -> None:
    save_station_status(
        session,
        [
            _station_status(),
            StationStatus(
                **{**_station_status().model_dump(), "station_id": "station-2"}
            ),
        ],
    )
    session.commit()

    since = OBSERVED_AT - timedelta(minutes=1)
    rows = fetch_recent_station_status(
        session, system_id="toronto", station_id="station-1", since=since
    )
    assert len(rows) == 1
    assert rows[0].station_id == "station-1"

    too_recent = OBSERVED_AT + timedelta(minutes=1)
    assert (
        fetch_recent_station_status(
            session, system_id="toronto", station_id="station-1", since=too_recent
        )
        == []
    )


def test_station_status_history_query_selects_expected_columns(session) -> None:
    save_station_status(session, [_station_status()])
    session.commit()

    query = station_status_history_query(system_id="toronto")
    rows = session.execute(query).all()

    assert len(rows) == 1
    row = rows[0]
    assert row.station_id == "station-1"
    assert row.num_vehicles_available == 7
    assert row.is_renting is True


def _station_data(**overrides: object) -> StationData:
    defaults: dict[str, object] = dict(
        system_id="toronto",
        station_id="a",
        station_name="Alpha",
        address="1 Main St",
        lat=43.6,
        lon=-79.4,
        capacity=20,
        is_charging_station=False,
        observed_at=OBSERVED_AT,
    )
    defaults.update(overrides)
    return StationData(**defaults)  # type: ignore[arg-type]


def test_first_sighting_of_a_station_records_a_baseline(session: Session) -> None:
    appended = record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()

    assert appended == 1
    assert [s.capacity for s in latest_stations(session, system_id="toronto")] == [20]


def test_an_unchanged_station_writes_nothing(session: Session) -> None:
    """The point of the change log: ingestion runs 288 times a day and must not
    append 288 identical rows per station."""

    record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()

    appended = record_station_changes(
        session,
        [_station_data(observed_at=OBSERVED_AT + timedelta(minutes=5))],
        system_id="toronto",
    )
    session.commit()

    assert appended == 0
    assert len(session.scalars(select(Station)).all()) == 1


def test_any_changed_detail_appends_a_row(session: Session) -> None:
    record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()

    later = OBSERVED_AT + timedelta(days=1)
    appended = record_station_changes(
        session,
        [_station_data(capacity=30, station_name="Alpha North", observed_at=later)],
        system_id="toronto",
    )
    session.commit()

    assert appended == 1
    history = session.scalars(select(Station).order_by(Station.observed_at)).all()
    assert [row.capacity for row in history] == [20, 30]
    assert [row.station_name for row in history] == ["Alpha", "Alpha North"]


def test_latest_stations_returns_only_the_newest_row(session: Session) -> None:
    record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()
    record_station_changes(
        session,
        [_station_data(capacity=30, observed_at=OBSERVED_AT + timedelta(days=1))],
        system_id="toronto",
    )
    session.commit()

    current = latest_stations(session, system_id="toronto")

    assert len(current) == 1
    assert current[0].capacity == 30


def test_get_station_reads_through_the_change_log(session: Session) -> None:
    """Serving swapped a primary-key lookup for newest-row-per-station, so this
    guards the path /predict actually takes."""

    record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()
    record_station_changes(
        session,
        [_station_data(capacity=44, observed_at=OBSERVED_AT + timedelta(days=1))],
        system_id="toronto",
    )
    session.commit()

    station = get_station(session, system_id="toronto", station_id="a")

    assert station is not None
    assert station.capacity == 44


def test_only_the_changed_station_is_appended(session: Session) -> None:
    record_station_changes(
        session,
        [_station_data(), _station_data(station_id="b", capacity=15)],
        system_id="toronto",
    )
    session.commit()

    later = OBSERVED_AT + timedelta(days=1)
    appended = record_station_changes(
        session,
        [
            _station_data(observed_at=later),
            _station_data(station_id="b", capacity=25, observed_at=later),
        ],
        system_id="toronto",
    )
    session.commit()

    assert appended == 1
    assert {
        s.station_id: s.capacity for s in latest_stations(session, system_id="toronto")
    } == {
        "a": 20,
        "b": 25,
    }


@pytest.mark.parametrize(
    ("field", "new_value"),
    [
        ("station_name", "Alpha North"),
        ("address", "2 Other Rd"),
        ("lat", 43.7),
        ("lon", -79.5),
        ("capacity", 30),
        ("is_charging_station", True),
    ],
)
def test_a_change_to_any_single_field_appends_a_row(
    session: Session, field: str, new_value: object
) -> None:
    """Comparison is by station_id across every tracked field, so a change to
    any one of them on its own is enough."""

    record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()

    appended = record_station_changes(
        session,
        [
            _station_data(
                **{field: new_value, "observed_at": OBSERVED_AT + timedelta(days=1)}
            )
        ],
        system_id="toronto",
    )
    session.commit()

    assert appended == 1
    current = latest_stations(session, system_id="toronto")[0]
    assert getattr(current, field) == new_value


def test_observed_at_alone_is_not_a_change(session: Session) -> None:
    """observed_at moves every five minutes. Comparing it would make every run
    look like a change and defeat the whole point of the log."""

    record_station_changes(session, [_station_data()], system_id="toronto")
    session.commit()

    appended = record_station_changes(
        session,
        [_station_data(observed_at=OBSERVED_AT + timedelta(days=99))],
        system_id="toronto",
    )
    session.commit()

    assert appended == 0


def test_every_station_column_is_either_tracked_or_deliberately_excluded() -> None:
    """Guards against a column being added to the model without deciding
    whether a change to it should be recorded. Without this, a new field would
    silently never trigger a row."""

    columns = {c.name for c in Station.__table__.columns}
    identity_and_bookkeeping = {"id", "system_id", "station_id", "observed_at"}

    assert columns - identity_and_bookkeeping == set(TRACKED_STATION_FIELDS)
