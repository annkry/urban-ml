import pytest
from sqlalchemy import select

from urban_ml.ingestion.gbfs_client import GbfsFetchError
from urban_ml.ingestion.gbfs_ingest import (
    GbfsIngestionSummary,
    format_ingestion_summary,
    ingest_gbfs_station_feeds,
)
from urban_ml.processing.gbfs_transform import GbfsTransformError
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    Station,
    StationStatusRecord,
    VehicleType,
)

DISCOVERY_URL = "https://example.com/gbfs/3/gbfs"
SYSTEM_INFORMATION_URL = "https://example.com/gbfs/3/system_information"
VEHICLE_TYPES_URL = "https://example.com/gbfs/3/vehicle_types"
STATION_INFORMATION_URL = "https://example.com/gbfs/3/station_information"
STATION_STATUS_URL = "https://example.com/gbfs/3/station_status"


def test_format_ingestion_summary_includes_counts() -> None:
    summary = GbfsIngestionSummary(
        discovery_url="https://example.com/gbfs/3/gbfs",
        system_id="bike_share_toronto",
        station_information_last_updated="2023-07-17T13:34:13+02:00",
        station_status_last_updated="2023-07-17T13:35:13+02:00",
        station_count=2,
        status_count=3,
        matched_station_status_count=2,
        processed_station_status_count=2,
        vehicle_type_count=4,
        station_change_count=6,
    )

    formatted = format_ingestion_summary(summary)

    assert "GBFS ingestion completed" in formatted
    assert "System: bike_share_toronto" in formatted
    assert "Stations discovered: 2" in formatted
    assert "Station statuses discovered: 3" in formatted
    assert "Stations with matching status: 2" in formatted
    assert "Vehicle types: 4" in formatted
    assert "Station detail changes recorded: 6" in formatted


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


def test_ingest_gbfs_station_feeds_persists_raw_payload_status_and_run(
    session,
) -> None:
    payloads = _payloads()

    def fake_fetcher(url: str, timeout_seconds: float) -> dict:
        return payloads[url]

    summary = ingest_gbfs_station_feeds(
        DISCOVERY_URL,
        timeout_seconds=10.0,
        session=session,
        json_fetcher=fake_fetcher,
    )

    assert summary.system_id == "bike_share_toronto"
    assert summary.station_count == 1
    assert summary.matched_station_status_count == 1
    assert summary.processed_station_status_count == 1
    assert summary.vehicle_type_count == 2
    assert summary.station_change_count == 1

    station = session.scalars(select(Station)).one()
    assert station.station_id == "station-1"
    assert station.station_name == "Main Station"

    status_record = session.scalars(select(StationStatusRecord)).one()
    assert status_record.station_id == "station-1"
    assert status_record.num_vehicles_available == 7

    vehicle_types = session.scalars(select(VehicleType)).all()
    assert {vt.vehicle_type_id for vt in vehicle_types} == {"CLASSIC", "EBIKE"}

    run = session.scalars(select(IngestionRun)).one()
    assert run.status == IngestionRunStatus.SUCCESS
    assert run.system_id == "bike_share_toronto"
    assert run.row_count == 1


def test_ingest_gbfs_station_feeds_records_failed_run_with_unknown_system_id(
    session,
) -> None:
    """Fails before system_information is even fetched, so system_id is
    genuinely unknown, not just unset.
    """

    def failing_fetcher(url: str, timeout_seconds: float) -> dict:
        raise GbfsFetchError("upstream is down")

    with pytest.raises(GbfsFetchError, match="upstream is down"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            session=session,
            json_fetcher=failing_fetcher,
        )

    run = session.scalars(select(IngestionRun)).one()
    assert run.status == IngestionRunStatus.FAILURE
    assert run.system_id is None
    assert run.error_message == "upstream is down"

    assert session.scalars(select(StationStatusRecord)).first() is None
    assert session.scalars(select(Station)).first() is None
    assert session.scalars(select(VehicleType)).first() is None


def test_ingest_gbfs_station_feeds_records_failed_run_with_known_system_id(
    session,
) -> None:
    """A failure after system_information succeeds should still record the
    now-known system_id, not lose it to the rollback inside fail_ingestion_run.
    """

    payloads = _payloads()
    payloads[STATION_STATUS_URL]["data"]["stations"][0]["station_id"] = "unknown"

    def fake_fetcher(url: str, timeout_seconds: float) -> dict:
        return payloads[url]

    with pytest.raises(GbfsTransformError, match="missing from"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            session=session,
            json_fetcher=fake_fetcher,
        )

    run = session.scalars(select(IngestionRun)).one()
    assert run.status == IngestionRunStatus.FAILURE
    assert run.system_id == "bike_share_toronto"
