from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urban_ml.api.main import app
from urban_ml.domain.station import Station as StationData
from urban_ml.domain.station_status import StationStatus
from urban_ml.storage.db import get_db
from urban_ml.storage.repository import record_station_changes, save_station_status

SYSTEM_ID = "toronto"
STATION_ID = "station-1"


class FakeModel:
    """Matches the DI-for-testing convention used elsewhere (a plain fake,
    no mock library) — predicts a fixed delta from the current value,
    mirroring what the real logged model's predict() contract returns."""

    def __init__(self, delta: float = 5.0) -> None:
        self.delta = delta

    def predict(self, X: Any) -> list[float]:
        return [self.delta] * len(X)


def _client(session: Any, *, model: Any, encoding: dict[str, int] | None) -> TestClient:
    app.dependency_overrides[get_db] = lambda: session
    app.state.system_id = SYSTEM_ID
    app.state.model = model
    app.state.station_id_encoding = encoding
    app.state.model_run_id = "fake-run-id" if model is not None else None
    return TestClient(app)


def _seed_station(session: Any, *, capacity: int = 20) -> None:
    record_station_changes(
        session,
        [
            StationData(
                system_id=SYSTEM_ID,
                station_id=STATION_ID,
                station_name="Main Station",
                lat=43.6532,
                lon=-79.3832,
                capacity=capacity,
                observed_at=datetime.now(UTC),
            )
        ],
        system_id=SYSTEM_ID,
    )
    session.commit()


def _seed_status(
    session: Any,
    *,
    num_vehicles_available: int = 8,
    is_installed: bool = True,
    is_renting: bool = True,
) -> None:
    observed_at = datetime.now(UTC)
    save_station_status(
        session,
        [
            StationStatus(
                observed_at=observed_at,
                system_id=SYSTEM_ID,
                station_id=STATION_ID,
                num_vehicles_available=num_vehicles_available,
                num_docks_available=12,
                is_installed=is_installed,
                is_renting=is_renting,
                is_returning=True,
            )
        ],
    )
    session.commit()


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> Any:
    yield
    app.dependency_overrides.clear()


def test_predict_happy_path(session: Any) -> None:
    _seed_station(session)
    _seed_status(session, num_vehicles_available=8)
    client = _client(session, model=FakeModel(delta=5.0), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["station_id"] == STATION_ID
    assert body["predicted_num_vehicles_available"] == 13  # 8 + 5
    assert body["last_observed_num_vehicles_available"] == 8
    assert body["horizon_minutes"] > 0


def test_predict_clips_prediction_to_station_capacity(session: Any) -> None:
    _seed_station(session, capacity=10)
    _seed_status(session, num_vehicles_available=8)
    client = _client(session, model=FakeModel(delta=50.0), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 200
    assert response.json()["predicted_num_vehicles_available"] == 10


def test_predict_station_not_found_returns_404(session: Any) -> None:
    client = _client(session, model=FakeModel(), encoding={})

    response = client.get("/predict/unknown-station")

    assert response.status_code == 404


def test_predict_no_history_returns_422(session: Any) -> None:
    _seed_station(session)
    client = _client(session, model=FakeModel(), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 422


def test_predict_model_not_loaded_returns_503(session: Any) -> None:
    _seed_station(session)
    _seed_status(session)
    client = _client(session, model=None, encoding=None)

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 503


def test_predict_out_of_service_station_returns_422(session: Any) -> None:
    _seed_station(session)
    _seed_status(session, is_renting=False)
    client = _client(session, model=FakeModel(), encoding={STATION_ID: 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 422


def test_predict_station_unseen_by_model_returns_422(session: Any) -> None:
    _seed_station(session)
    _seed_status(session)
    client = _client(session, model=FakeModel(), encoding={"some-other-station": 0})

    response = client.get(f"/predict/{STATION_ID}")

    assert response.status_code == 422
