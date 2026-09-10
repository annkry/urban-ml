from __future__ import annotations

from datetime import UTC, datetime, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from urban_ml.domain.station import Station
from urban_ml.features.build_features import RAW_COLUMNS, compute_features
from urban_ml.modeling.encoding import encode_station_id, to_model_frame

LOOKBACK_MINUTES = 90


class InsufficientHistoryError(Exception):
    """Not enough recent data to compute a feature row for this station."""


def build_feature_row(
    window: pl.DataFrame,
    *,
    system_id: str,
    station_id: str,
    station: Station,
) -> pl.DataFrame:
    """Slice one station's recent history out of the serving window and run it
    through the exact same compute_features() used in training."""

    since = datetime.now(UTC) - timedelta(minutes=LOOKBACK_MINUTES)
    raw = window.filter(
        (pl.col("system_id") == system_id)
        & (pl.col("station_id") == station_id)
        & (pl.col("observed_at") >= since)
    ).select(RAW_COLUMNS)
    if raw.is_empty():
        raise InsufficientHistoryError(station_id)

    stations = pl.DataFrame(
        {"station_id": [station.station_id], "capacity": [station.capacity]},
        schema={"station_id": pl.String, "capacity": pl.Int64},
    )
    return compute_features(raw, stations).tail(1)


def predict_one(
    model: lgb.Booster,
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
