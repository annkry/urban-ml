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
