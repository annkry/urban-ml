from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urban_ml.api.main import app, get_serving
from urban_ml.core.config import settings
from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.modeling.predict import LOOKBACK_MINUTES
from urban_ml.staging.objects import LocalObjectStore, ObjectStoreError
from urban_ml.staging.serving import ServingData
from urban_ml.staging.snapshots import stage_cycle

SYSTEM_ID = "toronto"
STATION_ID = "station-1"

CYCLE_MINUTES = 5
FULL_HISTORY_CYCLES = LOOKBACK_MINUTES // CYCLE_MINUTES + 1


class FakeModel:
    """Only its presence matters here — neither endpoint calls predict()."""


class _UnreachableStore:
    """Stands in for a bucket that has gone away."""

    def put(self, key: str, data: bytes) -> None:
        raise ObjectStoreError("bucket unreachable")

    def get(self, key: str) -> bytes:
        raise ObjectStoreError("bucket unreachable")

    def exists(self, key: str) -> bool:
        raise ObjectStoreError("bucket unreachable")

    def list_keys(self, prefix: str) -> list[str]:
        raise ObjectStoreError("bucket unreachable")

    def delete(self, key: str) -> None:
        raise ObjectStoreError("bucket unreachable")


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


def _stage(
    tmp_path: Path,
    *,
    cycles: int = FULL_HISTORY_CYCLES,
    newest_minutes_ago: float = 3.0,
) -> ServingData:
    store = LocalObjectStore(root=tmp_path)
    newest = datetime.now(UTC) - timedelta(minutes=newest_minutes_ago)

    for index in reversed(range(cycles)):
        moment = newest - timedelta(minutes=CYCLE_MINUTES * index)
        stage_cycle(
            store,
            status_records=[
                StationStatus(
                    observed_at=moment,
                    system_id=SYSTEM_ID,
                    station_id=STATION_ID,
                    num_vehicles_available=8,
                    num_docks_available=12,
                    is_installed=True,
                    is_renting=True,
                    is_returning=True,
                )
            ],
            station_records=[
                Station(
                    system_id=SYSTEM_ID,
                    station_id=STATION_ID,
                    station_name="Main Station",
                    lat=43.6532,
                    lon=-79.3832,
                    capacity=20,
                    observed_at=moment,
                )
            ],
            observed_at=moment,
        )
    return ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)


def _client(serving: ServingData | None = None, *, model: Any = None) -> TestClient:
    if serving is not None:
        app.dependency_overrides[get_serving] = lambda: serving
    app.state.model = model
    app.state.model_run_id = "fake-run-id" if model is not None else None
    return TestClient(app)


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


def test_health_never_touches_object_storage() -> None:
    """The guarantee that makes /health safe as a Cloud Run startup probe:
    a cold start must not depend on a network read succeeding."""

    unreachable = ServingData(store=_UnreachableStore())
    client = _client(unreachable, model=FakeModel())

    assert client.get("/health").status_code == 200


def test_ready_is_ready_when_model_and_fresh_deep_data_are_present(
    tmp_path: Path,
) -> None:
    response = _client(_stage(tmp_path), model=FakeModel()).get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["storage_reachable"] is True
    assert body["data_fresh"] is True
    assert body["history_deep_enough"] is True
    assert body["data_age_seconds"] < body["max_data_age_seconds"]


def test_ready_reports_503_when_ingestion_has_stopped(tmp_path: Path) -> None:
    """The failure the old /health hid: the process is fine, the data is not."""

    serving = _stage(tmp_path, newest_minutes_ago=LOOKBACK_MINUTES + 10)
    response = _client(serving, model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["model_loaded"] is True
    assert body["storage_reachable"] is True
    assert body["data_fresh"] is False


def test_ready_reports_503_when_the_window_is_too_shallow(tmp_path: Path) -> None:
    """Fresh but shallow: the case freshness alone cannot catch."""

    serving = _stage(tmp_path, cycles=2, newest_minutes_ago=1.0)
    response = _client(serving, model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["data_fresh"] is True
    assert body["history_deep_enough"] is False
    assert body["history_span_seconds"] < body["required_history_seconds"]


def test_ready_reports_503_when_there_is_no_data_at_all(tmp_path: Path) -> None:
    serving = _stage(tmp_path, cycles=0)
    response = _client(serving, model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["latest_observed_at"] is None
    assert body["data_age_seconds"] is None
    assert body["history_span_seconds"] is None
    assert body["data_fresh"] is False


def test_ready_reports_503_when_object_storage_is_unreachable() -> None:
    unreachable = ServingData(store=_UnreachableStore())
    response = _client(unreachable, model=FakeModel()).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["storage_reachable"] is False
    assert body["model_loaded"] is True


def test_ready_reports_503_when_no_bucket_is_configured() -> None:
    """A misconfigured service reports the same way an outage does."""

    response = _client(ServingData(store=None), model=FakeModel()).get("/ready")

    assert response.status_code == 503
    assert response.json()["storage_reachable"] is False


def test_ready_reports_503_when_no_model_is_loaded(tmp_path: Path) -> None:
    response = _client(_stage(tmp_path), model=None).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False
    assert body["data_fresh"] is True
