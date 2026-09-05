from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import polars as pl
import pytest

from urban_ml.features.build_features import add_target, compute_features
from urban_ml.modeling.dataset import apply_quality_filters, time_split
from urban_ml.modeling.encoding import build_station_id_encoding, to_model_frame
from urban_ml.modeling.train import train_and_log_baseline, train_and_log_lightgbm

STATIONS = ["station-1", "station-2", "station-3"]
DAYS = 3
POINTS_PER_DAY = 288  # 5-min cadence — matches MIN_STATION_HISTORY_ROWS


def _synthetic_raw_and_stations() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = random.Random(42)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n_points = DAYS * POINTS_PER_DAY

    station_ids: list[str] = []
    observed_ats: list[datetime] = []
    values: list[int] = []
    for station_id in STATIONS:
        level = rng.randint(5, 15)
        for i in range(n_points):
            station_ids.append(station_id)
            observed_ats.append(base + timedelta(minutes=5 * i))
            wobble = round(3 * math.sin(i / 12) + rng.uniform(-1, 1))
            values.append(max(0, min(20, level + wobble)))

    n = len(station_ids)
    raw = pl.DataFrame(
        {
            "station_id": station_ids,
            "observed_at": observed_ats,
            "num_vehicles_available": values,
            "num_docks_available": [5] * n,
            "is_installed": [True] * n,
            "is_renting": [True] * n,
            "is_returning": [True] * n,
        }
    )
    stations = pl.DataFrame({"station_id": STATIONS, "capacity": [20, 20, 20]})
    return raw, stations


@pytest.fixture
def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


def test_train_and_log_pipeline_round_trips_through_mlflow(_tracking_uri: str) -> None:
    raw, stations = _synthetic_raw_and_stations()
    encoding = build_station_id_encoding(stations)

    features = compute_features(raw, stations)
    labeled = add_target(features, raw, horizon_minutes=120, tolerance_minutes=10)
    filtered = apply_quality_filters(labeled, min_station_history_rows=POINTS_PER_DAY)
    split = time_split(filtered)
    assert split.train.height > 0
    assert split.test.height > 0

    mlflow.set_tracking_uri(_tracking_uri)
    mlflow.set_experiment("smoke-test")

    common_params = {
        "horizon_minutes": 120,
        "target_tolerance_minutes": 10,
        "min_station_history_rows": POINTS_PER_DAY,
        "train_start": split.train_start.isoformat(),
        "train_end": split.train_end.isoformat(),
        "val_start": split.val_start.isoformat(),
        "val_end": split.val_end.isoformat(),
        "test_start": split.test_start.isoformat(),
        "test_end": split.test_end.isoformat(),
        "n_train_rows": split.train.height,
        "n_val_rows": split.val.height,
        "n_test_rows": split.test.height,
        "n_stations": split.train["station_id"].n_unique(),
    }

    baseline_run_id, baseline_test_mae, baseline_test_rmse = train_and_log_baseline(
        split, common_params
    )
    assert math.isfinite(baseline_test_mae)
    assert math.isfinite(baseline_test_rmse)

    lgbm_run_id, lgbm_test_mae, lgbm_test_rmse = train_and_log_lightgbm(
        split,
        common_params,
        baseline_test_mae=baseline_test_mae,
        representative_station_id=STATIONS[0],
        encoding=encoding,
    )
    assert math.isfinite(lgbm_test_mae)
    assert math.isfinite(lgbm_test_rmse)

    loaded_model = mlflow.pyfunc.load_model(f"runs:/{lgbm_run_id}/model")
    sample_input = split.test.head(3)
    model_input = to_model_frame(sample_input, encoding)
    predictions = loaded_model.predict(model_input)
    assert len(predictions) == 3
    assert all(math.isfinite(p) for p in predictions)
