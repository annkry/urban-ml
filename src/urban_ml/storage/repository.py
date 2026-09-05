from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from urban_ml.domain.station import Station as StationData
from urban_ml.domain.station_status import StationStatus
from urban_ml.domain.station_vehicle_availability import StationVehicleAvailability
from urban_ml.domain.vehicle_type import VehicleType as VehicleTypeData
from urban_ml.ingestion.gbfs_client import GbfsRawFeeds
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    RawGbfsPayload,
    Station,
    StationStatusRecord,
    StationVehicleAvailabilityRecord,
    VehicleType,
)


def save_raw_gbfs_payload(
    session: Session,
    raw_feeds: GbfsRawFeeds,
    *,
    system_id: str,
    observed_at: datetime,
) -> RawGbfsPayload:
    payload = RawGbfsPayload(
        system_id=system_id,
        observed_at=observed_at,
        discovery_url=raw_feeds.discovery_url,
        system_information_url=raw_feeds.system_information_url,
        station_status_url=raw_feeds.station_status_url,
        station_status_payload=raw_feeds.station_status_payload,
    )
    session.add(payload)
    return payload


def upsert_stations(
    session: Session,
    stations: list[StationData],
) -> None:
    """Insert new stations or update existing ones in place."""

    for station in stations:
        session.merge(
            Station(
                system_id=station.system_id,
                station_id=station.station_id,
                station_name=station.station_name,
                lat=station.lat,
                lon=station.lon,
                capacity=station.capacity,
            )
        )


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


def save_station_vehicle_availability(
    session: Session,
    records: list[StationVehicleAvailability],
) -> None:
    for record in records:
        session.add(
            StationVehicleAvailabilityRecord(
                observed_at=record.observed_at,
                system_id=record.system_id,
                station_id=record.station_id,
                vehicle_type_id=record.vehicle_type_id,
                count=record.count,
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
                num_docks_available=record.num_docks_available,
                is_installed=record.is_installed,
                is_renting=record.is_renting,
                is_returning=record.is_returning,
                last_reported=record.last_reported,
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
    return session.get(Station, (system_id, station_id))


def list_stations(session: Session, *, system_id: str) -> Sequence[Station]:
    stmt = select(Station).where(Station.system_id == system_id)
    return session.scalars(stmt).all()


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
