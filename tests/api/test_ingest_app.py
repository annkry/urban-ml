from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from urban_ml.api import ingest_app as module
from urban_ml.ingestion.gbfs_client import GbfsClientError
from urban_ml.processing.gbfs_transform import GbfsTransformError


class _Summary:
    system_id = "toronto"
    processed_station_status_count = 3
    station_change_count = 1
    station_count = 3


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
        module, "ingest_gbfs_station_feeds", lambda *_a, **_k: _Summary()
    )

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {
        "system_id": "toronto",
        "station_status_rows": 3,
        "station_detail_changes": 1,
        "stations_discovered": 3,
    }


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
