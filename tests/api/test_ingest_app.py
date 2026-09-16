from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urban_ml.api import ingest_app as module
from urban_ml.ingestion.gbfs_client import GbfsClientError
from urban_ml.ingestion.gbfs_ingest import GbfsIngestionSummary
from urban_ml.processing.gbfs_transform import GbfsTransformError
from urban_ml.staging.objects import LocalObjectStore, ObjectStoreError

STATUS_KEY = "snapshots/station_status/date=2026-09-09/20260909T121802Z.parquet"
STATIONS_KEY = "snapshots/stations/date=2026-09-09/20260909T121802Z.parquet"


def _summary() -> GbfsIngestionSummary:
    """The real summary type, not a structural stand-in."""

    return GbfsIngestionSummary(
        discovery_url="https://example.com/gbfs/3/gbfs",
        system_id="toronto",
        station_information_last_updated="2026-09-09T12:18:02+00:00",
        station_status_last_updated="2026-09-09T12:18:02+00:00",
        station_count=3,
        status_count=3,
        matched_station_status_count=3,
        processed_station_status_count=3,
        station_status_key=STATUS_KEY,
        stations_key=STATIONS_KEY,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(module, "object_store", LocalObjectStore(root=tmp_path))
    return TestClient(module.app)


def test_health_touches_nothing_external(client: TestClient) -> None:
    """Cloud Run probes this on every cold start; a check that reached the
    bucket would turn a storage blip into a failed deploy."""

    assert client.get("/health").status_code == 200


def test_ingest_reports_what_it_staged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module, "ingest_gbfs_station_feeds", lambda *_a, **_k: _summary()
    )

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {
        "system_id": "toronto",
        "station_status_rows": 3,
        "stations_discovered": 3,
        "station_status_key": STATUS_KEY,
        "stations_key": STATIONS_KEY,
    }


@pytest.mark.parametrize(
    "error", [GbfsClientError, GbfsTransformError, ObjectStoreError]
)
def test_a_failed_run_returns_502_rather_than_a_silent_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """Cloud Scheduler decides whether to retry from the status code, so a
    swallowed failure would look like a healthy run that stored nothing. A
    storage failure is a failed run too: the bucket is the only sink."""

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise error("upstream is down")

    monkeypatch.setattr(module, "ingest_gbfs_station_feeds", boom)

    response = client.post("/ingest")

    assert response.status_code == 502
    assert "upstream is down" in response.json()["detail"]


def test_a_missing_bucket_is_a_500_not_a_quiet_no_op(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Misconfiguration must fail every tick loudly rather than answer 200
    while writing nothing anywhere."""

    monkeypatch.setattr(module, "object_store", None)

    response = client.post("/ingest")

    assert response.status_code == 500
    assert "GCS_BUCKET" in response.json()["detail"]
