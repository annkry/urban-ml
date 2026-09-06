from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class IngestionRunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class Station(Base):
    """A station's details as observed at one moment, appended on change.

    This was a current-values table upserted in place, which meant a station
    that was renamed, moved, or gained docks looked as though it always had
    today's details. Ingestion fetches the catalog every five minutes, but
    persisting all of it would write ~300k near-identical rows a day to
    capture a handful of real changes a year, so only transitions are stored.

    Serving reads the newest row per station instead of a primary-key lookup.
    Measured at 0.5 ms against 0.035 ms, which is immaterial beside feature
    building and inference, and it avoids keeping the same facts twice.
    """

    __tablename__ = "stations"
    __table_args__ = (
        UniqueConstraint(
            "system_id", "station_id", "observed_at", name="uq_stations_identity"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system_id: Mapped[str] = mapped_column(String, nullable=False)
    station_id: Mapped[str] = mapped_column(String, nullable=False)
    station_name: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str | None] = mapped_column(String, nullable=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_charging_station: Mapped[bool | None] = mapped_column(nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class VehicleType(Base):
    """Vehicle type metadata, one row per type, upserted in place."""

    __tablename__ = "vehicle_types"

    system_id: Mapped[str] = mapped_column(String, primary_key=True)
    vehicle_type_id: Mapped[str] = mapped_column(String, primary_key=True)
    form_factor: Mapped[str] = mapped_column(String, nullable=False)
    propulsion_type: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)


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
    num_vehicles_disabled: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_docks_available: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_docks_disabled: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_installed: Mapped[bool] = mapped_column(nullable=False)
    is_renting: Mapped[bool] = mapped_column(nullable=False)
    is_returning: Mapped[bool] = mapped_column(nullable=False)
    num_vehicles_electric: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_vehicles_human: Mapped[int | None] = mapped_column(Integer, nullable=True)


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
