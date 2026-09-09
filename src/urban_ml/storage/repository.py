from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from urban_ml.domain.station import Station as StationData
from urban_ml.domain.station_status import StationStatus
from urban_ml.domain.vehicle_type import VehicleType as VehicleTypeData
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    Station,
    StationStatusRecord,
    VehicleType,
)


def latest_stations(session: Session, *, system_id: str) -> Sequence[Station]:
    """The newest recorded row for each station.

    `stations` is a change log, so "current" means the most recent row rather
    than the only row.
    """

    newest = (
        select(
            Station.station_id,
            func.max(Station.observed_at).label("observed_at"),
        )
        .where(Station.system_id == system_id)
        .group_by(Station.station_id)
        .subquery()
    )
    stmt = select(Station).join(
        newest,
        (Station.station_id == newest.c.station_id)
        & (Station.observed_at == newest.c.observed_at)
        & (Station.system_id == system_id),
    )
    return session.scalars(stmt).all()


TRACKED_STATION_FIELDS = (
    "station_name",
    "address",
    "lat",
    "lon",
    "capacity",
    "is_charging_station",
)


def _station_details(station: Station | StationData) -> tuple[object, ...]:
    return tuple(getattr(station, field) for field in TRACKED_STATION_FIELDS)


def record_station_changes(
    session: Session,
    stations: list[StationData],
    *,
    system_id: str,
) -> int:
    """Append a row for each station whose details differ from last time.

    Compared by station_id, and any one differing field is enough. A station
    seen for the first time is recorded, so every station has a baseline;
    thereafter a run that changes nothing writes nothing. Returns the number of
    rows appended, which on a normal run is zero.
    """

    known = {
        row.station_id: _station_details(row)
        for row in latest_stations(session, system_id=system_id)
    }
    changed = [
        station
        for station in stations
        if known.get(station.station_id) != _station_details(station)
    ]
    session.add_all(
        [
            Station(
                system_id=station.system_id,
                station_id=station.station_id,
                station_name=station.station_name,
                address=station.address,
                lat=station.lat,
                lon=station.lon,
                capacity=station.capacity,
                is_charging_station=station.is_charging_station,
                observed_at=station.observed_at,
            )
            for station in changed
        ]
    )
    return len(changed)


def upsert_vehicle_types(
    session: Session,
    vehicle_types: list[VehicleTypeData],
) -> None:
    """Insert new vehicle types or update existing ones in place."""

    for vehicle_type in vehicle_types:
        session.merge(
            VehicleType(
                system_id=vehicle_type.system_id,
                vehicle_type_id=vehicle_type.vehicle_type_id,
                form_factor=vehicle_type.form_factor,
                propulsion_type=vehicle_type.propulsion_type,
                name=vehicle_type.name,
            )
        )


def save_station_status(
    session: Session,
    records: list[StationStatus],
) -> None:
    for record in records:
        session.add(
            StationStatusRecord(
                observed_at=record.observed_at,
                system_id=record.system_id,
                station_id=record.station_id,
                num_vehicles_available=record.num_vehicles_available,
                num_vehicles_disabled=record.num_vehicles_disabled,
                num_docks_available=record.num_docks_available,
                num_docks_disabled=record.num_docks_disabled,
                is_installed=record.is_installed,
                is_renting=record.is_renting,
                is_returning=record.is_returning,
                num_vehicles_electric=record.num_vehicles_electric,
                num_vehicles_human=record.num_vehicles_human,
            )
        )


def start_ingestion_run(
    session: Session,
    *,
    started_at: datetime,
) -> IngestionRun:
    """Create and commit a 'running' row immediately.

    system_id isn't known yet at this point, it's derived from the
    system_information feed, fetched after this call.
    """

    run = IngestionRun(
        started_at=started_at,
        status=IngestionRunStatus.RUNNING,
        row_count=0,
    )
    session.add(run)
    session.commit()
    return run


def complete_ingestion_run(
    session: Session,
    run: IngestionRun,
    *,
    system_id: str,
    finished_at: datetime,
    row_count: int,
) -> None:
    run.system_id = system_id
    run.status = IngestionRunStatus.SUCCESS
    run.finished_at = finished_at
    run.row_count = row_count
    session.commit()


def fail_ingestion_run(
    session: Session,
    run: IngestionRun,
    *,
    system_id: str | None,
    finished_at: datetime,
    error_message: str,
) -> None:
    session.rollback()
    run.system_id = system_id
    run.status = IngestionRunStatus.FAILURE
    run.finished_at = finished_at
    run.error_message = error_message
    session.commit()


def get_station(session: Session, *, system_id: str, station_id: str) -> Station | None:
    """Newest recorded details for one station."""

    stmt = (
        select(Station)
        .where(Station.system_id == system_id, Station.station_id == station_id)
        .order_by(Station.observed_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def list_stations(session: Session, *, system_id: str) -> Sequence[Station]:
    return latest_stations(session, system_id=system_id)


def list_system_ids(session: Session) -> Sequence[str]:
    stmt = select(Station.system_id).distinct()
    return session.scalars(stmt).all()


def fetch_recent_station_status(
    session: Session,
    *,
    system_id: str,
    station_id: str,
    since: datetime,
) -> Sequence[StationStatusRecord]:
    """Small, serving-time-only read; a plain ORM query is fine at this size."""

    stmt = (
        select(StationStatusRecord)
        .where(
            StationStatusRecord.system_id == system_id,
            StationStatusRecord.station_id == station_id,
            StationStatusRecord.observed_at >= since,
        )
        .order_by(StationStatusRecord.observed_at)
    )
    return session.scalars(stmt).all()


def latest_status_observed_at(session: Session, *, system_id: str) -> datetime | None:
    """Timestamp of the newest status reading, or None if there are none."""

    stmt = select(func.max(StationStatusRecord.observed_at)).where(
        StationStatusRecord.system_id == system_id
    )
    latest: datetime | None = session.scalar(stmt)
    return latest


def station_status_history_query(
    *,
    system_id: str,
    since: datetime | None = None,
) -> Select[Any]:
    """Unexecuted Core SELECT for bulk analytical reads (training).

    station_status has millions of rows in production; materializing that many
    ORM entities (with full SQLAlchemy instrumentation) is a real memory/latency
    problem. Callers execute this via polars.read_database(query, connection) to
    bypass the ORM identity map for this one heavy path only.
    """

    stmt = select(
        StationStatusRecord.station_id,
        StationStatusRecord.observed_at,
        StationStatusRecord.num_vehicles_available,
        StationStatusRecord.num_docks_available,
        StationStatusRecord.is_installed,
        StationStatusRecord.is_renting,
        StationStatusRecord.is_returning,
    ).where(StationStatusRecord.system_id == system_id)
    if since is not None:
        stmt = stmt.where(StationStatusRecord.observed_at >= since)
    return stmt.order_by(
        StationStatusRecord.station_id, StationStatusRecord.observed_at
    )
