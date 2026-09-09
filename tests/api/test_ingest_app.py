from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from urban_ml.api import ingest_app as module
from urban_ml.ingestion.gbfs_client import GbfsClientError
from urban_ml.ingestion.gbfs_ingest import GbfsIngestionSummary
from urban_ml.processing.gbfs_transform import GbfsTransformError


def _summary(*, snapshot_staged: bool = True) -> GbfsIngestionSummary:
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
        vehicle_type_count=2,
        station_change_count=1,
        snapshot_staged=snapshot_staged,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, session: Any) -> TestClient:
    from contextlib import contextmanager

    @contextmanager
    def fake_session() -> Any:
        yield session

    monkeypatch.setattr(module, "get_session", fake_session)
    return TestClient(module.app)


def test_health_touches_nothing_external(client: TestClient) -> None:
    """Cloud Run probes this on every cold start. A check that queried the
    database would fail while a scale-to-zero Postgres was still waking."""

    assert client.get("/health").status_code == 200


def test_ingest_reports_what_it_wrote(
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
        "station_detail_changes": 1,
        "stations_discovered": 3,
        "snapshot_staged": True,
    }


def test_a_staging_failure_is_reported_without_failing_the_tick(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The database is authoritative while staging is a shadow write."""

    monkeypatch.setattr(
        module,
        "ingest_gbfs_station_feeds",
        lambda *_a, **_k: _summary(snapshot_staged=False),
    )

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json()["snapshot_staged"] is False


@pytest.mark.parametrize("error", [GbfsClientError, GbfsTransformError])
def test_a_failed_run_returns_502_rather_than_a_silent_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """Cloud Scheduler decides whether to retry from the status code, so a
    swallowed failure would look like a healthy run that stored nothing."""

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise error("upstream is down")

    monkeypatch.setattr(module, "ingest_gbfs_station_feeds", boom)

    response = client.post("/ingest")

    assert response.status_code == 502
    assert "upstream is down" in response.json()["detail"]
