from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from urban_ml.api.main import app
from urban_ml.core.config import settings
from urban_ml.domain.station_status import StationStatus
from urban_ml.modeling.predict import LOOKBACK_MINUTES
from urban_ml.storage.db import get_db
from urban_ml.storage.repository import save_station_status

SYSTEM_ID = "toronto"
STATION_ID = "station-1"


class FakeModel:
    """Only its presence matters here — neither endpoint calls predict()."""


@pytest.fixture(autouse=True)
def _reset_app_state() -> Any:
    """app is module-level and shared across test modules, so state set by one
    test would otherwise leak into the next."""

    app.state.system_id = SYSTEM_ID
    app.state.model = None
    app.state.station_id_encoding = None
    app.state.model_run_id = None
    yield
    app.dependency_overrides.clear()


def _client(session: Any = None, *, model: Any = None) -> TestClient:
    if session is not None:
        app.dependency_overrides[get_db] = lambda: session
    app.state.model = model
    app.state.model_run_id = "fake-run-id" if model is not None else None
    return TestClient(app)


def _seed_status(session: Any, *, minutes_ago: float) -> None:
    save_station_status(
        session,
        [
            StationStatus(
                observed_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
                system_id=SYSTEM_ID,
                station_id=STATION_ID,
                num_vehicles_available=8,
                num_docks_available=12,
                is_installed=True,
                is_renting=True,
                is_returning=True,
            )
        ],
    )
    session.commit()


class _UnreachableSession:
    """Stands in for a session whose database has gone away."""

    def scalar(self, stmt: Any) -> Any:
        raise OperationalError("select 1", {}, Exception("connection refused"))


def test_health_reports_ok_when_a_model_is_loaded() -> None:
    response = _client(model=FakeModel()).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "model_loaded": True,
        "model_run_id": "fake-run-id",
    }


def test_health_reports_503_when_no_model_is_loaded() -> None:
    response = _client(model=None).get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "no_model"
    assert body["model_loaded"] is False


def test_health_never_touches_the_database() -> None:
    """The guarantee that makes /health safe as a Cloud Run startup probe:
    a cold start must not depend on a scale-to-zero database waking up."""

    client = _client(_UnreachableSession(), model=FakeModel())

    assert client.get("/health").status_code == 200


def test_ready_is_ready_when_model_and_fresh_data_are_present(session: Any) -> None:
    _seed_status(session, minutes_ago=3)
    response = _client(session, model=FakeModel()).get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["database_reachable"] is True
    assert body["data_fresh"] is True
    assert body["data_age_seconds"] < body["max_data_age_seconds"]


def test_ready_reports_503_when_ingestion_has_stopped(session: Any) -> None:
    """The failure the old /health hid: the process is fine, the data is not."""

    _seed_status(session, minutes_ago=LOOKBACK_MINUTES + 10)
    response = _client(session, model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["model_loaded"] is True
    assert body["database_reachable"] is True
    assert body["data_fresh"] is False


def test_ready_reports_503_when_there_is_no_data_at_all(session: Any) -> None:
    response = _client(session, model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["latest_observed_at"] is None
    assert body["data_age_seconds"] is None
    assert body["data_fresh"] is False


def test_ready_reports_503_when_the_database_is_unreachable() -> None:
    response = _client(_UnreachableSession(), model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["database_reachable"] is False
    assert body["model_loaded"] is True


def test_ready_reports_503_when_no_model_is_loaded(session: Any) -> None:
    _seed_status(session, minutes_ago=3)
    response = _client(session, model=None).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False
    assert body["data_fresh"] is True
