from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class StationSnapshot(BaseModel):
    """Station observation used as the processed dataset row.

    This is an internal domain model, not a direct GBFS API schema. It contains
    the fields we currently need for storage, analysis, and future ML features.
    """

    observed_at: datetime
    system_id: str
    station_id: str
    station_name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    capacity: int | None = Field(default=None, ge=0)
    num_vehicles_available: int = Field(ge=0)
    num_docks_available: int | None = Field(default=None, ge=0)
    is_installed: bool
    is_renting: bool
    is_returning: bool
    last_reported: datetime
