from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from urban_ml.api.schemas import (
    HealthResponse,
    PredictionResponse,
    ReadinessResponse,
)
from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from urban_ml.modeling.artifacts import (
    load_booster,
    load_metadata,
    load_station_id_encoding,
    model_artifacts_exist,
)
from urban_ml.modeling.predict import (
    LOOKBACK_MINUTES,
    InsufficientHistoryError,
    build_feature_row,
    predict_one,
)
from urban_ml.staging.objects import ObjectStoreError, store_from_settings
from urban_ml.staging.serving import ServingData, station_details, system_ids
from urban_ml.staging.snapshots import (
    covers_lookback,
    history_span,
    newest_observed_at,
)

configure_logging()
logger = get_logger(__name__)

_serving = ServingData(store=store_from_settings())


def get_serving() -> ServingData:
    """The serving files this process reads. Overridden in tests."""

    return _serving


def _resolve_system_id(serving: ServingData) -> str:
    """The system this process serves."""

    if settings.system_id:
        return settings.system_id

    found = system_ids(serving.stations())
    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one system_id in the station list, found {found}. "
            "Set SYSTEM_ID explicitly if multiple systems are expected."
        )
    return found[0]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting %s in %s mode", settings.app_name, settings.app_env)

    app.state.system_id = _resolve_system_id(get_serving())

    model_dir = settings.model_dir
    if model_artifacts_exist(model_dir):
        app.state.model = load_booster(model_dir)
        app.state.station_id_encoding = load_station_id_encoding(model_dir)
        app.state.model_run_id = load_metadata(model_dir)["mlflow_run_id"]
        logger.info(
            "Loaded prediction model from %s (trained in MLflow run %s)",
            model_dir,
            app.state.model_run_id,
        )
    else:
        app.state.model = None
        app.state.station_id_encoding = None
        app.state.model_run_id = None
        logger.warning(
            "No model artifacts under %s; /predict will return 503", model_dir
        )

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)


_MAX_DATA_AGE = timedelta(minutes=LOOKBACK_MINUTES)
_REQUIRED_HISTORY = timedelta(minutes=LOOKBACK_MINUTES)


@app.get("/health", response_model=HealthResponse)
def health_check(request: Request, response: Response) -> HealthResponse:
    """Liveness. Deliberately touches nothing external, for the same reason
    the ingestion service's /health doesn't: Cloud Run probes this on every
    cold start, and a check that read from object storage would fail on a
    transient blip and get the revision killed."""

    model = getattr(request.app.state, "model", None)
    if model is None:
        response.status_code = 503

    return HealthResponse(
        status="ok" if model is not None else "no_model",
        service=settings.app_name,
        environment=settings.app_env,
        model_loaded=model is not None,
        model_run_id=getattr(request.app.state, "model_run_id", None),
    )


@app.get("/ready", response_model=ReadinessResponse)
def readiness_check(
    request: Request,
    response: Response,
    serving: ServingData = Depends(get_serving),
) -> ReadinessResponse:
    """Whether a prediction could actually be served right now: model loaded,
    serving files readable, and the window both current and deep enough to
    build a full feature row from."""

    model = getattr(request.app.state, "model", None)

    storage_reachable = True
    window = pl.DataFrame()
    try:
        window = serving.window()
    except ObjectStoreError:
        logger.exception("Readiness check could not read the serving window")
        storage_reachable = False

    latest_observed_at = newest_observed_at(window)
    span = history_span(window)

    age_seconds: float | None = None
    if latest_observed_at is not None:
        age_seconds = (datetime.now(UTC) - latest_observed_at).total_seconds()

    data_fresh = (
        age_seconds is not None and age_seconds <= _MAX_DATA_AGE.total_seconds()
    )
    deep_enough = covers_lookback(window, lookback_minutes=LOOKBACK_MINUTES)

    ready = model is not None and storage_reachable and data_fresh and deep_enough
    if not ready:
        response.status_code = 503

    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        model_loaded=model is not None,
        model_run_id=getattr(request.app.state, "model_run_id", None),
        storage_reachable=storage_reachable,
        latest_observed_at=latest_observed_at,
        data_age_seconds=age_seconds,
        max_data_age_seconds=_MAX_DATA_AGE.total_seconds(),
        data_fresh=data_fresh,
        history_span_seconds=None if span is None else span.total_seconds(),
        required_history_seconds=_REQUIRED_HISTORY.total_seconds(),
        history_deep_enough=deep_enough,
    )


@app.get("/predict/{station_id}", response_model=PredictionResponse)
def predict_station(
    station_id: str,
    request: Request,
    serving: ServingData = Depends(get_serving),
) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not loaded.",
        )
    encoding = request.app.state.station_id_encoding
    system_id = request.app.state.system_id

    try:
        stations = serving.stations()
        window = serving.window()
    except ObjectStoreError as exc:
        logger.exception("Could not read the serving files")
        raise HTTPException(
            status_code=503, detail="Serving data is unavailable."
        ) from exc

    station = station_details(stations, system_id=system_id, station_id=station_id)
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    try:
        feature_row = build_feature_row(
            window, system_id=system_id, station_id=station_id, station=station
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
