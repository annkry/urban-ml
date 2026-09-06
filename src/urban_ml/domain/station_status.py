from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StationStatus(BaseModel):
    """Station availability, one row per station per run.

    The electric/human split is derived from vehicle_types_available via each
    type's propulsion_type, rather than stored per model: model names come
    and go (Toronto defines ten types and only four ever appear), while
    "does it have a motor" is the thing that actually changes rider
    behaviour and never needs a schema migration.
    """

    observed_at: datetime
    system_id: str
    station_id: str
    num_vehicles_available: int = Field(ge=0)
    num_vehicles_disabled: int | None = Field(default=None, ge=0)
    num_docks_available: int | None = Field(default=None, ge=0)
    num_docks_disabled: int | None = Field(default=None, ge=0)
    is_installed: bool
    is_renting: bool
    is_returning: bool
    num_vehicles_electric: int | None = Field(default=None, ge=0)
    num_vehicles_human: int | None = Field(default=None, ge=0)
