from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import polars as pl
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from urban_ml.features.build_features import (
    FEATURE_COLUMNS,
    add_target,
    compute_features,
)
from urban_ml.modeling.artifacts import save_model_artifacts
from urban_ml.modeling.dataset import (
    MIN_STATION_HISTORY_ROWS,
    TimeSplit,
    apply_quality_filters,
    load_raw_status,
    load_stations,
    time_split,
)
from urban_ml.modeling.encoding import (
    STATION_ENCODING_ARTIFACT_PATH,
    build_station_id_encoding,
    to_model_frame,
)
from urban_ml.modeling.metrics import mean_absolute_error, root_mean_squared_error
from urban_ml.modeling.plotting import (
    plot_predicted_vs_actual,
    plot_timeseries_comparison,
)
from urban_ml.storage.db import engine, get_session
from urban_ml.storage.repository import list_system_ids

configure_logging()
logger = get_logger(__name__)

EXPERIMENT_NAME = "bike-availability-forecast"
TARGET_TOLERANCE_MINUTES = 10
CATEGORICAL_FEATURES = ["station_id_code"]


def _resolve_system_id(session: Session) -> str:
    if settings.system_id:
        return settings.system_id
    system_ids = list_system_ids(session)
    if len(system_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one system_id in the database, found {list(system_ids)}. "
            "Set SYSTEM_ID explicitly if multiple systems are expected."
        )
    return system_ids[0]


def _common_params(split: TimeSplit, *, horizon_minutes: int) -> dict[str, Any]:
    return {
        "horizon_minutes": horizon_minutes,
        "target_tolerance_minutes": TARGET_TOLERANCE_MINUTES,
        "min_station_history_rows": MIN_STATION_HISTORY_ROWS,
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


def train_and_log_baseline(
    split: TimeSplit, common_params: dict[str, Any]
) -> tuple[str, float, float]:
    """Predicted = last observed value. Never served (no log_model call) —
    wrapping this in a pyfunc model purely for MLflow-UI symmetry isn't worth
    the complexity for a run that's never loaded."""

    with mlflow.start_run(run_name="baseline_persistence") as run:
        mlflow.set_tag("model_type", "baseline_persistence")
        mlflow.log_params(common_params)

        train_true = split.train["target_num_vehicles_available"].to_numpy()
        train_pred = split.train["num_vehicles_available"].to_numpy()
        val_true = split.val["target_num_vehicles_available"].to_numpy()
        val_pred = split.val["num_vehicles_available"].to_numpy()
        test_true = split.test["target_num_vehicles_available"].to_numpy()
        test_pred = split.test["num_vehicles_available"].to_numpy()

        train_mae = mean_absolute_error(train_true, train_pred)
        train_rmse = root_mean_squared_error(train_true, train_pred)
        val_mae = mean_absolute_error(val_true, val_pred)
        val_rmse = root_mean_squared_error(val_true, val_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        test_rmse = root_mean_squared_error(test_true, test_pred)
        mlflow.log_metrics(
            {
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
            }
        )

        fig = plot_predicted_vs_actual(
            test_true,
            test_pred,
            mae=test_mae,
            rmse=test_rmse,
            title="Baseline (persistence)",
        )
        mlflow.log_figure(fig, "predicted_vs_actual.png")

        logger.info(
            "Baseline run %s: test_mae=%.3f test_rmse=%.3f",
            run.info.run_id,
            test_mae,
            test_rmse,
        )
        return run.info.run_id, test_mae, test_rmse


def _predict_absolute(
    model: lgb.LGBMRegressor, X: pd.DataFrame, current_values: Any
) -> Any:
    """The model predicts the CHANGE from the current value, not the
    absolute future count (see train_and_log_lightgbm for why) — every
    caller must add the current value back to get a real bike count."""

    return current_values + model.predict(X)


def _build_comparison_plot(
    split: TimeSplit,
    *,
    model: lgb.LGBMRegressor,
    station_id: str,
    encoding: dict[str, int],
) -> Any | None:
    station_test = split.test.filter(pl.col("station_id") == station_id).sort(
        "observed_at"
    )
    if station_test.height == 0:
        return None

    X_station = to_model_frame(station_test, encoding)
    current = station_test["num_vehicles_available"].to_numpy()
    lgbm_pred_abs = _predict_absolute(model, X_station, current)
    return plot_timeseries_comparison(
        station_test["observed_at"].to_numpy(),
        station_test["target_num_vehicles_available"].to_numpy(),
        current,
        lgbm_pred_abs,
        station_id=station_id,
    )


def train_and_log_lightgbm(
    split: TimeSplit,
    common_params: dict[str, Any],
    *,
    baseline_test_mae: float,
    representative_station_id: str,
    encoding: dict[str, int],
    model_dir: Path,
) -> tuple[str, float, float]:
    """Writes the serving artifacts to ``model_dir`` as a side effect.

    That destination is a required argument rather than reading
    ``settings.model_dir`` directly: this function is exercised by the
    training smoke test, and defaulting to the configured path would have the
    test suite silently overwrite the committed production model.
    """

    X_train = to_model_frame(split.train, encoding)
    X_val = to_model_frame(split.val, encoding)
    X_test = to_model_frame(split.test, encoding)

    current_train = split.train["num_vehicles_available"].to_numpy()
    current_val = split.val["num_vehicles_available"].to_numpy()
    current_test = split.test["num_vehicles_available"].to_numpy()
    y_train_abs = split.train["target_num_vehicles_available"].to_numpy()
    y_val_abs = split.val["target_num_vehicles_available"].to_numpy()
    y_test_abs = split.test["target_num_vehicles_available"].to_numpy()

    y_train = y_train_abs - current_train
    y_val = y_val_abs - current_val

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        random_state=42,
        verbosity=-1,
    )

    start = time.monotonic()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    train_duration_seconds = time.monotonic() - start

    train_pred_abs = _predict_absolute(model, X_train, current_train)
    val_pred_abs = _predict_absolute(model, X_val, current_val)
    test_pred_abs = _predict_absolute(model, X_test, current_test)

    train_mae = mean_absolute_error(y_train_abs, train_pred_abs)
    train_rmse = root_mean_squared_error(y_train_abs, train_pred_abs)
    val_mae = mean_absolute_error(y_val_abs, val_pred_abs)
    val_rmse = root_mean_squared_error(y_val_abs, val_pred_abs)
    test_mae = mean_absolute_error(y_test_abs, test_pred_abs)
    test_rmse = root_mean_squared_error(y_test_abs, test_pred_abs)
    improvement_pct = (baseline_test_mae - test_mae) / baseline_test_mae * 100

    with mlflow.start_run(run_name="lightgbm") as run:
        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("prediction_target", "delta_from_current_value")
        mlflow.log_params(common_params)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(
            {
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mae_improvement_over_baseline_pct": improvement_pct,
                "train_duration_seconds": train_duration_seconds,
            }
        )

        now_ms = int(time.time() * 1000)
        mlflow_client = MlflowClient()
        for prefix, evals_result in (
            ("train", model.evals_result_["training"]),
            ("val", model.evals_result_["valid_1"]),
        ):
            mlflow_client.log_batch(
                run.info.run_id,
                metrics=[
                    Metric(f"{prefix}_mae_by_round", value, now_ms, step)
                    for step, value in enumerate(evals_result["l1"], start=1)
                ],
            )
            mlflow_client.log_batch(
                run.info.run_id,
                metrics=[
                    Metric(f"{prefix}_rmse_by_round", value**0.5, now_ms, step)
                    for step, value in enumerate(evals_result["l2"], start=1)
                ],
            )

        mlflow.log_dict({"feature_columns": FEATURE_COLUMNS}, "feature_columns.json")
        mlflow.log_dict(encoding, STATION_ENCODING_ARTIFACT_PATH)

        signature = mlflow.models.infer_signature(X_val, model.predict(X_val))
        mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_val.head(2),
        )

        save_model_artifacts(
            model_dir,
            booster=model.booster_,
            encoding=encoding,
            metadata={
                "mlflow_run_id": run.info.run_id,
                "trained_at": datetime.now(UTC).isoformat(),
                "horizon_minutes": common_params["horizon_minutes"],
                "n_train_rows": common_params["n_train_rows"],
                "n_stations": common_params["n_stations"],
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mae_improvement_over_baseline_pct": improvement_pct,
                "feature_columns": FEATURE_COLUMNS,
            },
        )
        logger.info("Wrote serving artifacts to %s", model_dir)

        fig = plot_predicted_vs_actual(
            y_test_abs, test_pred_abs, mae=test_mae, rmse=test_rmse, title="LightGBM"
        )
        mlflow.log_figure(fig, "predicted_vs_actual.png")

        comparison_fig = _build_comparison_plot(
            split, model=model, station_id=representative_station_id, encoding=encoding
        )
        if comparison_fig is not None:
            mlflow.log_figure(comparison_fig, "comparison_timeseries.png")

        logger.info(
            "LightGBM run %s: test_mae=%.3f test_rmse=%.3f (%.1f%% better than baseline)",
            run.info.run_id,
            test_mae,
            test_rmse,
            improvement_pct,
        )
        return run.info.run_id, test_mae, test_rmse


def main() -> int:
    with get_session() as session:
        system_id = _resolve_system_id(session)
        stations = load_stations(session, system_id=system_id)

    encoding = build_station_id_encoding(stations)

    logger.info("Loading station_status history for system_id=%s", system_id)
    raw = load_raw_status(engine, system_id=system_id)
    logger.info("Loaded %d raw rows", raw.height)

    features = compute_features(raw, stations)
    labeled = add_target(
        features,
        raw,
        horizon_minutes=settings.forecast_horizon_minutes,
        tolerance_minutes=TARGET_TOLERANCE_MINUTES,
    )
    filtered = apply_quality_filters(labeled)
    split = time_split(filtered)
    logger.info(
        "Split: train=%d val=%d test=%d stations=%d",
        split.train.height,
        split.val.height,
        split.test.height,
        split.train["station_id"].n_unique(),
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    common_params = _common_params(
        split, horizon_minutes=settings.forecast_horizon_minutes
    )

    baseline_run_id, baseline_test_mae, baseline_test_rmse = train_and_log_baseline(
        split, common_params
    )

    representative_station_id = (
        split.test.group_by("station_id")
        .len(name="n")
        .sort("n", descending=True)["station_id"][0]
    )
    lgbm_run_id, lgbm_test_mae, lgbm_test_rmse = train_and_log_lightgbm(
        split,
        common_params,
        baseline_test_mae=baseline_test_mae,
        representative_station_id=representative_station_id,
        encoding=encoding,
        model_dir=settings.model_dir,
    )

    improvement_pct = (baseline_test_mae - lgbm_test_mae) / baseline_test_mae * 100
    print(
        f"Baseline (persistence) run_id: {baseline_run_id}   "
        f"test_mae={baseline_test_mae:.2f}  test_rmse={baseline_test_rmse:.2f}"
    )
    print(
        f"LightGBM run_id:                {lgbm_run_id}   "
        f"test_mae={lgbm_test_mae:.2f}  test_rmse={lgbm_test_rmse:.2f}   "
        f"({improvement_pct:.1f}% better than baseline)"
    )
    print()
    print(f"Serving artifacts written to {settings.model_dir}/ - commit them to")
    print("deploy this model; the API loads them at startup.")

    return 0
