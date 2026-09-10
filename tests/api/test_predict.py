from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urban_ml.api.main import app, get_serving
from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.modeling.predict import LOOKBACK_MINUTES
from urban_ml.staging.objects import LocalObjectStore
from urban_ml.staging.serving import ServingData
from urban_ml.staging.snapshots import stage_cycle

SYSTEM_ID = "toronto"
STATION_ID = "station-1"

CYCLE_MINUTES = 5
FULL_HISTORY_CYCLES = LOOKBACK_MINUTES // CYCLE_MINUTES + 1


class FakeModel:
    """Matches the DI-for-testing convention used elsewhere (a plain fake,
    no mock library) — predicts a fixed delta from the current value,
    mirroring what the real logged model's predict() contract returns."""

    def __init__(self, delta: float = 5.0) -> None:
        self.delta = delta

    def predict(self, X: Any) -> list[float]:
        return [self.delta] * len(X)


def _status(
    station_id: str,
    observed_at: datetime,
    *,
    vehicles: int,
    is_installed: bool,
    is_renting: bool,
) -> StationStatus:
    return StationStatus(
        observed_at=observed_at,
        system_id=SYSTEM_ID,
        station_id=station_id,
        num_vehicles_available=vehicles,
        num_docks_available=12,
        is_installed=is_installed,
        is_renting=is_renting,
        is_returning=True,
    )


def _station(station_id: str, observed_at: datetime, *, capacity: int) -> Station:
    return Station(
        system_id=SYSTEM_ID,
        station_id=station_id,
        station_name="Main Station",
        lat=43.6532,
        lon=-79.3832,
        capacity=capacity,
        observed_at=observed_at,
    )


def _stage(
    tmp_path: Path,
    *,
    capacity: int = 20,
    vehicles: int = 8,
    cycles: int = FULL_HISTORY_CYCLES,
    is_installed: bool = True,
    is_renting: bool = True,
    status_station_id: str = STATION_ID,
    newest_minutes_ago: float = 1.0,
) -> ServingData:
    """Stage a run of cycles into a local bucket and serve from it."""

    store = LocalObjectStore(root=tmp_path)
    newest = datetime.now(UTC) - timedelta(minutes=newest_minutes_ago)
    listed = sorted({STATION_ID, status_station_id})

    for index in reversed(range(cycles)):
        moment = newest - timedelta(minutes=CYCLE_MINUTES * index)
        stage_cycle(
            store,
            status_records=[
                _status(
                    status_station_id,
                    moment,
                    vehicles=vehicles,
                    is_installed=is_installed,
                    is_renting=is_renting,
                )
            ],
            station_records=[
                _station(station_id, moment, capacity=capacity) for station_id in listed
            ],
            observed_at=moment,
        )

    return ServingData(store=store, window_ttl=0.0, stations_ttl=0.0)


def _client(
    serving: ServingData, *, model: Any, encoding: dict[str, int] | None
) -> TestClient:
    app.dependency_overrides[get_serving] = lambda: serving
    app.state.system_id = SYSTEM_ID
    app.state.model = model
    app.state.station_id_encoding = encoding
    app.state.model_run_id = "fake-run-id" if model is not None else None
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> Any:
    yield
    app.dependency_overrides.clear()


def test_predict_happy_path(tmp_path: Path) -> None:
    serving = _stage(tmp_path, vehicles=8)
    client = _client(serving, model=FakeModel(delta=5.0), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["station_id"] == STATION_ID
    assert body["predicted_num_vehicles_available"] == 13  # 8 + 5
    assert body["last_observed_num_vehicles_available"] == 8
    assert body["horizon_minutes"] > 0


def test_predict_clips_prediction_to_station_capacity(tmp_path: Path) -> None:
    serving = _stage(tmp_path, capacity=10, vehicles=8)
    client = _client(serving, model=FakeModel(delta=50.0), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 200
    assert response.json()["predicted_num_vehicles_available"] == 10


def test_predict_station_not_found_returns_404(tmp_path: Path) -> None:
    """Absent from the staged station list means this system has no such id."""

    serving = _stage(tmp_path)
    client = _client(serving, model=FakeModel(), encoding={})

    response = client.get("/predict/unknown-station")

    assert response.status_code == 404


def test_predict_no_history_returns_422(tmp_path: Path) -> None:
    """Known station, no readings yet -- not the same as an unknown station."""

    serving = _stage(tmp_path, status_station_id="another-station")
    client = _client(serving, model=FakeModel(), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 422


def test_predict_model_not_loaded_returns_503(tmp_path: Path) -> None:
    serving = _stage(tmp_path)
    client = _client(serving, model=None, encoding=None)

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 503


def test_predict_out_of_service_station_returns_422(tmp_path: Path) -> None:
    serving = _stage(tmp_path, is_renting=False)
    client = _client(serving, model=FakeModel(), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 422


def test_predict_station_unseen_by_model_returns_422(tmp_path: Path) -> None:
    serving = _stage(tmp_path)
    client = _client(serving, model=FakeModel(), encoding={"some-other-station": 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 422


def test_predict_reads_only_the_serving_files(tmp_path: Path) -> None:
    """The Phase 2 guarantee: no database is involved in a prediction."""

    serving = _stage(tmp_path)
    client = _client(serving, model=FakeModel(delta=1.0), encoding={STATION_ID: 0})

    assert client.get(f"/predict/{STATION_ID}").status_code == 200
    assert set(serving.store.list_keys("")) >= {  # type: ignore[union-attr]
        "serving/station_status.parquet",
        "serving/stations.parquet",
    }
