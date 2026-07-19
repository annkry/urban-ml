from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StationVehicleAvailability(BaseModel):
    """Per-vehicle-type availability at a station, one row per
    station per vehicle type per run.
    """

    observed_at: datetime
    system_id: str
    station_id: str
    vehicle_type_id: str
    count: int = Field(ge=0)
