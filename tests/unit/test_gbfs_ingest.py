import pytest
from sqlalchemy import select

from urban_ml.ingestion.gbfs_client import GbfsFetchError
from urban_ml.ingestion.gbfs_ingest import (
    GbfsIngestionSummary,
    format_ingestion_summary,
    ingest_gbfs_station_feeds,
)
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    RawGbfsPayload,
    StationSnapshotRecord,
)

DISCOVERY_URL = "https://example.com/gbfs/3/gbfs"
STATION_INFORMATION_URL = "https://example.com/gbfs/3/station_information"
STATION_STATUS_URL = "https://example.com/gbfs/3/station_status"


def test_format_ingestion_summary_includes_counts() -> None:
    summary = GbfsIngestionSummary(
        discovery_url="https://example.com/gbfs/3/gbfs",
        station_information_last_updated="2023-07-17T13:34:13+02:00",
        station_status_last_updated="2023-07-17T13:35:13+02:00",
        station_count=2,
        status_count=3,
        matched_station_status_count=2,
        processed_station_snapshot_count=2,
    )

    formatted = format_ingestion_summary(summary)

    assert "GBFS ingestion completed" in formatted
    assert "Stations discovered: 2" in formatted
    assert "Station statuses discovered: 3" in formatted
    assert "Stations with matching status: 2" in formatted


def _payloads() -> dict[str, dict]:
    return {
        DISCOVERY_URL: {
            "last_updated": "2026-07-18T12:00:00+00:00",
            "ttl": 0,
            "version": "3.0",
            "data": {
                "feeds": [
                    {"name": "station_information", "url": STATION_INFORMATION_URL},
                    {"name": "station_status", "url": STATION_STATUS_URL},
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
                    }
                ]
            },
        },
    }


def test_ingest_gbfs_station_feeds_persists_raw_payload_snapshots_and_run(
    session,
) -> None:
    payloads = _payloads()

    def fake_fetcher(url: str, timeout_seconds: float) -> dict:
        return payloads[url]

    summary = ingest_gbfs_station_feeds(
        DISCOVERY_URL,
        timeout_seconds=10.0,
        system_id="toronto",
        session=session,
        json_fetcher=fake_fetcher,
    )

    assert summary.station_count == 1
    assert summary.matched_station_status_count == 1
    assert summary.processed_station_snapshot_count == 1

    raw_payload = session.scalars(select(RawGbfsPayload)).one()
    assert raw_payload.system_id == "toronto"

    snapshot = session.scalars(select(StationSnapshotRecord)).one()
    assert snapshot.station_id == "station-1"
    assert snapshot.num_vehicles_available == 7

    run = session.scalars(select(IngestionRun)).one()
    assert run.status == IngestionRunStatus.SUCCESS
    assert run.row_count == 1


def test_ingest_gbfs_station_feeds_records_failed_run_and_reraises(session) -> None:
    def failing_fetcher(url: str, timeout_seconds: float) -> dict:
        raise GbfsFetchError("upstream is down")

    with pytest.raises(GbfsFetchError, match="upstream is down"):
        ingest_gbfs_station_feeds(
            DISCOVERY_URL,
            timeout_seconds=10.0,
            system_id="toronto",
            session=session,
            json_fetcher=failing_fetcher,
        )

    run = session.scalars(select(IngestionRun)).one()
    assert run.status == IngestionRunStatus.FAILURE
    assert run.error_message == "upstream is down"

    assert session.scalars(select(StationSnapshotRecord)).first() is None
    assert session.scalars(select(RawGbfsPayload)).first() is None
