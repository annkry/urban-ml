from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PortableJSON = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class IngestionRunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class RawGbfsPayload(Base):
    """Untouched GBFS API responses."""

    __tablename__ = "raw_gbfs_payloads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    discovery_url: Mapped[str] = mapped_column(String, nullable=False)
    system_information_url: Mapped[str] = mapped_column(String, nullable=False)
    station_status_url: Mapped[str] = mapped_column(String, nullable=False)
    station_status_payload: Mapped[dict[str, Any]] = mapped_column(
        PortableJSON, nullable=False
    )


class Station(Base):
    """Station metadata, one row per station, upserted in place."""

    __tablename__ = "stations"

    system_id: Mapped[str] = mapped_column(String, primary_key=True)
    station_id: Mapped[str] = mapped_column(String, primary_key=True)
    station_name: Mapped[str] = mapped_column(String, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)


class VehicleType(Base):
    """Vehicle type metadata, one row per type, upserted in place."""

    __tablename__ = "vehicle_types"

    system_id: Mapped[str] = mapped_column(String, primary_key=True)
    vehicle_type_id: Mapped[str] = mapped_column(String, primary_key=True)
    form_factor: Mapped[str] = mapped_column(String, nullable=False)
    propulsion_type: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)


class StationVehicleAvailabilityRecord(Base):
    """Per-vehicle-type availability at a station, one row per station per
    vehicle type per run."""

    __tablename__ = "station_vehicle_availability"
    __table_args__ = (
        UniqueConstraint(
            "system_id",
            "station_id",
            "vehicle_type_id",
            "observed_at",
            name="uq_station_vehicle_availability_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    system_id: Mapped[str] = mapped_column(String, nullable=False)
    station_id: Mapped[str] = mapped_column(String, nullable=False)
    vehicle_type_id: Mapped[str] = mapped_column(String, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)


class StationStatusRecord(Base):
    """Station availability, one row per station per run."""

    __tablename__ = "station_status"
    __table_args__ = (
        UniqueConstraint(
            "system_id",
            "station_id",
            "observed_at",
            name="uq_station_status_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    system_id: Mapped[str] = mapped_column(String, nullable=False)
    station_id: Mapped[str] = mapped_column(String, nullable=False)
    num_vehicles_available: Mapped[int] = mapped_column(Integer, nullable=False)
    num_docks_available: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_installed: Mapped[bool] = mapped_column(nullable=False)
    is_renting: Mapped[bool] = mapped_column(nullable=False)
    is_returning: Mapped[bool] = mapped_column(nullable=False)
    last_reported: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class IngestionRun(Base):
    """One row per ingestion attempt."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
