from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from urban_ml.storage.models import Station, StationStatusRecord


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
