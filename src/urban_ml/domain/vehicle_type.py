from __future__ import annotations

from pydantic import BaseModel

from urban_ml.schemas.gbfs import FormFactor, PropulsionType


class VehicleType(BaseModel):
    """Vehicle type metadata, one row per type regardless of time."""

    system_id: str
    vehicle_type_id: str
    form_factor: FormFactor
    propulsion_type: PropulsionType
    name: str | None = None
