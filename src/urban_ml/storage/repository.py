from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from urban_ml.domain.station_snapshot import StationSnapshot
from urban_ml.ingestion.gbfs_client import GbfsRawStationFeeds
from urban_ml.storage.models import (
    IngestionRun,
    IngestionRunStatus,
    RawGbfsPayload,
    StationSnapshotRecord,
)


def save_raw_gbfs_payload(
    session: Session,
    raw_feeds: GbfsRawStationFeeds,
    *,
    system_id: str,
    observed_at: datetime,
) -> RawGbfsPayload:
    payload = RawGbfsPayload(
        system_id=system_id,
        observed_at=observed_at,
        discovery_url=raw_feeds.discovery_url,
        station_information_url=raw_feeds.station_information_url,
        station_status_url=raw_feeds.station_status_url,
        discovery_payload=raw_feeds.discovery_payload,
        station_information_payload=raw_feeds.station_information_payload,
        station_status_payload=raw_feeds.station_status_payload,
    )
    session.add(payload)
    return payload


def save_station_snapshots(
    session: Session,
    snapshots: list[StationSnapshot],
) -> None:
    for snapshot in snapshots:
        session.add(
            StationSnapshotRecord(
                observed_at=snapshot.observed_at,
                system_id=snapshot.system_id,
                station_id=snapshot.station_id,
                station_name=snapshot.station_name,
                lat=snapshot.lat,
                lon=snapshot.lon,
                capacity=snapshot.capacity,
                num_vehicles_available=snapshot.num_vehicles_available,
                num_docks_available=snapshot.num_docks_available,
                is_installed=snapshot.is_installed,
                is_renting=snapshot.is_renting,
                is_returning=snapshot.is_returning,
                last_reported=snapshot.last_reported,
            )
        )


def start_ingestion_run(
    session: Session,
    *,
    system_id: str,
    started_at: datetime,
) -> IngestionRun:
    """Create and commit a 'running' row immediately."""

    run = IngestionRun(
        system_id=system_id,
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
    finished_at: datetime,
    row_count: int,
) -> None:
    run.status = IngestionRunStatus.SUCCESS
    run.finished_at = finished_at
    run.row_count = row_count
    session.commit()


def fail_ingestion_run(
    session: Session,
    run: IngestionRun,
    *,
    finished_at: datetime,
    error_message: str,
) -> None:
    session.rollback()
    run.status = IngestionRunStatus.FAILURE
    run.finished_at = finished_at
    run.error_message = error_message
    session.commit()
