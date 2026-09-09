from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    station_id: str
    predicted_num_vehicles_available: int
    horizon_minutes: int
    predicted_at: datetime
    last_observed_at: datetime
    last_observed_num_vehicles_available: int
    model_run_id: str


class HealthResponse(BaseModel):
    """Liveness. `model_loaded` is the part worth reading: a process that
    holds no model can never answer /predict."""

    status: str
    service: str
    environment: str
    model_loaded: bool
    model_run_id: str | None


class ReadinessResponse(BaseModel):
    """Whether this process can actually serve a prediction right now."""

    status: str
    model_loaded: bool
    model_run_id: str | None
    database_reachable: bool
    latest_observed_at: datetime | None
    data_age_seconds: float | None
    max_data_age_seconds: float
    data_fresh: bool
