from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


TRACKED_STATION_FIELDS = (
    "station_name",
    "address",
    "lat",
    "lon",
    "capacity",
    "is_charging_station",
)


class Station(BaseModel):
    """A station's details as observed at one moment.

    Carries observed_at because these details are recorded as a change log
    rather than overwritten: capacity, position and name all change
    occasionally, and Postgres upserting them in place destroyed the only
    record that they ever had different values.
    """

    system_id: str
    station_id: str
    station_name: str
    address: str | None = None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    capacity: int | None = Field(default=None, ge=0)
    is_charging_station: bool | None = None
    observed_at: datetime
