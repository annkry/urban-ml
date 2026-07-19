from __future__ import annotations

from pydantic import BaseModel, Field


class Station(BaseModel):
    """Station metadata, one row per station regardless of time."""

    system_id: str
    station_id: str
    station_name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    capacity: int | None = Field(default=None, ge=0)
