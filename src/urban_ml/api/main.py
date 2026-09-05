from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

import mlflow
from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

from urban_ml.api.schemas import PredictionResponse
from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from urban_ml.modeling.predict import (
    InsufficientHistoryError,
    build_feature_row,
    load_production_model,
    load_station_id_encoding,
    predict_one,
)
from urban_ml.storage.db import get_db, get_session
from urban_ml.storage.repository import get_station, list_system_ids

configure_logging()
logger = get_logger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Starting %s in %s mode",
        settings.app_name,
        settings.app_env,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with get_session() as session:
        app.state.system_id = _resolve_system_id(session)

    if settings.model_run_id:
        app.state.model = load_production_model(settings.model_run_id)
        app.state.station_id_encoding = load_station_id_encoding(settings.model_run_id)
        app.state.model_run_id = settings.model_run_id
        logger.info("Loaded prediction model from run %s", settings.model_run_id)
    else:
        app.state.model = None
        app.state.station_id_encoding = None
        app.state.model_run_id = None
        logger.warning("MODEL_RUN_ID not set; /predict will return 503")

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


@app.get("/predict/{station_id}", response_model=PredictionResponse)
def predict_station(
    station_id: str, request: Request, session: Session = Depends(get_db)
) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not loaded. Set MODEL_RUN_ID and restart.",
        )
    encoding = request.app.state.station_id_encoding
    system_id = request.app.state.system_id

    station = get_station(session, system_id=system_id, station_id=station_id)
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    try:
        feature_row = build_feature_row(
            session, system_id=system_id, station_id=station_id, station=station
        )
    except InsufficientHistoryError:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient history to compute a prediction for station {station_id}",
        ) from None

    if not (feature_row["is_installed"][0] and feature_row["is_renting"][0]):
        raise HTTPException(
            status_code=422,
            detail=f"Station {station_id} is not currently in service; prediction not meaningful",
        )

    try:
        predicted = predict_one(model, feature_row, encoding=encoding)
    except InsufficientHistoryError:
        raise HTTPException(
            status_code=422,
            detail=f"Station {station_id} was not present in the model's training data",
        ) from None

    predicted = max(0.0, predicted)
    if station.capacity is not None:
        predicted = min(predicted, float(station.capacity))

    last_observed_at = feature_row["observed_at"][0]
    return PredictionResponse(
        station_id=station_id,
        predicted_num_vehicles_available=round(predicted),
        horizon_minutes=settings.forecast_horizon_minutes,
        predicted_at=last_observed_at
        + timedelta(minutes=settings.forecast_horizon_minutes),
        last_observed_at=last_observed_at,
        last_observed_num_vehicles_available=int(
            feature_row["num_vehicles_available"][0]
        ),
        model_run_id=request.app.state.model_run_id,
    )
