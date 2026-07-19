from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class StationStatus(BaseModel):
    """Station availability, one row per station per run."""

    observed_at: datetime
    system_id: str
    station_id: str
    num_vehicles_available: int = Field(ge=0)
    num_docks_available: int | None = Field(default=None, ge=0)
    is_installed: bool
    is_renting: bool
    is_returning: bool
    last_reported: datetime
