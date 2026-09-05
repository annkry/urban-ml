from __future__ import annotations

from datetime import UTC, datetime, timedelta

import mlflow.artifacts
import numpy as np
import polars as pl
from mlflow.pyfunc import PyFuncModel, load_model
from sqlalchemy.orm import Session

from urban_ml.features.build_features import compute_features
from urban_ml.modeling.encoding import (
    STATION_ENCODING_ARTIFACT_PATH,
    encode_station_id,
    to_model_frame,
)
from urban_ml.storage.models import Station
from urban_ml.storage.repository import fetch_recent_station_status

LOOKBACK_MINUTES = 90


class InsufficientHistoryError(Exception):
    """Not enough recent data to compute a feature row for this station."""


def load_production_model(run_id: str) -> PyFuncModel:
    return load_model(f"runs:/{run_id}/model")


def load_station_id_encoding(run_id: str) -> dict[str, int]:
    return mlflow.artifacts.load_dict(
        f"runs:/{run_id}/{STATION_ENCODING_ARTIFACT_PATH}"
    )


def build_feature_row(
    session: Session,
    *,
    system_id: str,
    station_id: str,
    station: Station,
) -> pl.DataFrame:
    """Fetch recent history for one station and run it through the exact
    same compute_features() used in training — this IS train/serve parity,
    not a re-implementation of it."""

    since = datetime.now(UTC) - timedelta(minutes=LOOKBACK_MINUTES)
    records = fetch_recent_station_status(
        session, system_id=system_id, station_id=station_id, since=since
    )
    if not records:
        raise InsufficientHistoryError(station_id)

    raw = pl.DataFrame(
        {
            "station_id": [record.station_id for record in records],
            "observed_at": [record.observed_at for record in records],
            "num_vehicles_available": [
                record.num_vehicles_available for record in records
            ],
            "num_docks_available": [record.num_docks_available for record in records],
            "is_installed": [record.is_installed for record in records],
            "is_renting": [record.is_renting for record in records],
            "is_returning": [record.is_returning for record in records],
        }
    )
    stations = pl.DataFrame(
        {"station_id": [station.station_id], "capacity": [station.capacity]},
        schema={"station_id": pl.String, "capacity": pl.Int64},
    )
    return compute_features(raw, stations).tail(1)


def predict_one(
    model: PyFuncModel,
    feature_row: pl.DataFrame,
    *,
    encoding: dict[str, int],
) -> float:
    """The logged model predicts the CHANGE from the current value, not the
    absolute future count (see modeling.train — "prediction_target" tag /
    delta-from-current-value comment) — the current value must be added back
    here to get a real bike count. This is the serving-side half of that
    contract; if it drifted from training, predictions would silently be
    off by roughly the current station's occupancy."""

    if encode_station_id(feature_row, encoding)["station_id_code"].null_count() > 0:
        raise InsufficientHistoryError(
            "station_id not present in the model's training encoding"
        )

    model_input = to_model_frame(feature_row, encoding)
    predicted_delta: np.ndarray = np.asarray(model.predict(model_input))
    current_value = float(feature_row["num_vehicles_available"][0])
    return current_value + float(predicted_delta[0])
