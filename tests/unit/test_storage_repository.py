from datetime import UTC, datetime

from sqlalchemy import select

from urban_ml.domain.station_snapshot import StationSnapshot
from urban_ml.ingestion.gbfs_client import GbfsRawStationFeeds
from urban_ml.schemas.gbfs import (
    GbfsDiscoveryResponse,
    StationInformationResponse,
    StationStatusResponse,
)
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    RawGbfsPayload,
    StationSnapshotRecord,
)
from urban_ml.storage.repository import (
    complete_ingestion_run,
    fail_ingestion_run,
    save_raw_gbfs_payload,
    save_station_snapshots,
    start_ingestion_run,
)

OBSERVED_AT = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _raw_feeds() -> GbfsRawStationFeeds:
    discovery_payload = {
        "last_updated": "2026-07-18T12:00:00Z",
        "ttl": 0,
        "version": "3.0",
        "data": {"feeds": []},
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
    return GbfsRawStationFeeds(
        discovery_url="https://example.com/gbfs.json",
        station_information_url="https://example.com/station_information",
        station_status_url="https://example.com/station_status",
        discovery_payload=discovery_payload,
        station_information_payload=station_information_payload,
        station_status_payload=station_status_payload,
        discovery=GbfsDiscoveryResponse.model_validate(discovery_payload),
        station_information=StationInformationResponse.model_validate(
            station_information_payload
        ),
        station_status=StationStatusResponse.model_validate(station_status_payload),
    )


def _snapshot() -> StationSnapshot:
    return StationSnapshot(
        observed_at=OBSERVED_AT,
        system_id="toronto",
        station_id="station-1",
        station_name="Main Station",
        lat=43.6532,
        lon=-79.3832,
        capacity=20,
        num_vehicles_available=7,
        num_docks_available=13,
        is_installed=True,
        is_renting=True,
        is_returning=True,
        last_reported=OBSERVED_AT,
    )


def test_save_raw_gbfs_payload_persists_full_payload(session) -> None:
    save_raw_gbfs_payload(
        session, _raw_feeds(), system_id="toronto", observed_at=OBSERVED_AT
    )
    session.commit()

    stored = session.scalars(select(RawGbfsPayload)).one()
    assert stored.system_id == "toronto"
    assert stored.observed_at.replace(tzinfo=UTC) == OBSERVED_AT
    assert stored.discovery_url == "https://example.com/gbfs.json"
    assert stored.station_status_payload["version"] == "3.0"


def test_save_station_snapshots_persists_rows(session) -> None:
    save_station_snapshots(session, [_snapshot()])
    session.commit()

    stored = session.scalars(select(StationSnapshotRecord)).one()
    assert stored.station_id == "station-1"
    assert stored.num_vehicles_available == 7
    assert stored.is_renting is True


def test_ingestion_run_lifecycle_success(session) -> None:
    run = start_ingestion_run(session, system_id="toronto", started_at=OBSERVED_AT)
    assert run.status == IngestionRunStatus.RUNNING

    complete_ingestion_run(session, run, finished_at=OBSERVED_AT, row_count=42)

    stored = session.scalars(select(IngestionRun)).one()
    assert stored.status == IngestionRunStatus.SUCCESS
    assert stored.row_count == 42


def test_ingestion_run_lifecycle_failure_keeps_run_row(session) -> None:
    run = start_ingestion_run(session, system_id="toronto", started_at=OBSERVED_AT)

    save_station_snapshots(session, [_snapshot()])

    fail_ingestion_run(session, run, finished_at=OBSERVED_AT, error_message="boom")

    stored_run = session.scalars(select(IngestionRun)).one()
    assert stored_run.status == IngestionRunStatus.FAILURE
    assert stored_run.error_message == "boom"

    assert session.scalars(select(StationSnapshotRecord)).first() is None
