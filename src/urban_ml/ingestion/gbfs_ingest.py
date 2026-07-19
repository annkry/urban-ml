from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from urban_ml.core.config import settings
from urban_ml.ingestion.gbfs_client import (
    GbfsClient,
    GbfsClientError,
    JsonFetcher,
    fetch_json_with_retry,
)
from urban_ml.processing.gbfs_transform import (
    GbfsTransformError,
    build_station_status,
    build_stations,
    build_station_vehicle_availability,
    build_vehicle_types,
)
from urban_ml.storage.db import get_session
from urban_ml.storage.repository import (
    complete_ingestion_run,
    fail_ingestion_run,
    save_raw_gbfs_payload,
    save_station_status,
    save_station_vehicle_availability,
    start_ingestion_run,
    upsert_stations,
    upsert_vehicle_types,
)


@dataclass(frozen=True)
class GbfsIngestionSummary:
    discovery_url: str
    system_id: str
    station_information_last_updated: str
    station_status_last_updated: str
    station_count: int
    status_count: int
    matched_station_status_count: int
    processed_station_status_count: int
    vehicle_type_count: int
    station_vehicle_availability_count: int


def ingest_gbfs_station_feeds(
    discovery_url: str,
    *,
    timeout_seconds: float,
    session: Session,
    json_fetcher: JsonFetcher = fetch_json_with_retry,
) -> GbfsIngestionSummary:
    """Fetch, validate, and persist GBFS station feeds to the database."""

    observed_at = datetime.now(UTC)
    run = start_ingestion_run(session, started_at=observed_at)
    system_id: str | None = None

    try:
        client = GbfsClient(
            discovery_url, timeout_seconds=timeout_seconds, json_fetcher=json_fetcher
        )
        raw_feeds = client.fetch_raw_feeds()
        system_id = raw_feeds.system_information.data.system_id

        station_ids = {
            station.station_id
            for station in raw_feeds.station_information.data.stations
        }
        status_station_ids = {
            station.station_id for station in raw_feeds.station_status.data.stations
        }

        save_raw_gbfs_payload(
            session, raw_feeds, system_id=system_id, observed_at=observed_at
        )

        stations = build_stations(raw_feeds, system_id=system_id)
        upsert_stations(session, stations)

        vehicle_types = build_vehicle_types(raw_feeds, system_id=system_id)
        upsert_vehicle_types(session, vehicle_types)

        station_status_records = build_station_status(
            raw_feeds,
            system_id=system_id,
            known_station_ids=station_ids,
            observed_at=observed_at,
        )
        save_station_status(session, station_status_records)

        station_vehicle_availability = build_station_vehicle_availability(
            raw_feeds, system_id=system_id, observed_at=observed_at
        )
        save_station_vehicle_availability(session, station_vehicle_availability)

        complete_ingestion_run(
            session,
            run,
            system_id=system_id,
            finished_at=datetime.now(UTC),
            row_count=len(station_status_records),
        )
    except (GbfsClientError, GbfsTransformError, SQLAlchemyError) as exc:
        fail_ingestion_run(
            session,
            run,
            system_id=system_id,
            finished_at=datetime.now(UTC),
            error_message=str(exc),
        )
        raise

    return GbfsIngestionSummary(
        discovery_url=discovery_url,
        system_id=system_id,
        station_information_last_updated=raw_feeds.station_information.last_updated.isoformat(),
        station_status_last_updated=raw_feeds.station_status.last_updated.isoformat(),
        station_count=len(station_ids),
        status_count=len(status_station_ids),
        matched_station_status_count=len(station_ids & status_station_ids),
        processed_station_status_count=len(station_status_records),
        vehicle_type_count=len(vehicle_types),
        station_vehicle_availability_count=len(station_vehicle_availability),
    )


def format_ingestion_summary(summary: GbfsIngestionSummary) -> str:
    lines = [
        "GBFS ingestion completed",
        f"Discovery URL: {summary.discovery_url}",
        f"System: {summary.system_id}",
        f"Station information updated: {summary.station_information_last_updated}",
        f"Station status updated: {summary.station_status_last_updated}",
        f"Stations discovered: {summary.station_count}",
        f"Station statuses discovered: {summary.status_count}",
        f"Stations with matching status: {summary.matched_station_status_count}",
        f"Processed station status rows: {summary.processed_station_status_count}",
        f"Vehicle types: {summary.vehicle_type_count}",
        f"Station vehicle availability rows: {summary.station_vehicle_availability_count}",
    ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch, validate, and persist GBFS station information and "
            "station status feeds."
        )
    )
    parser.add_argument(
        "--discovery-url",
        default=settings.gbfs_discovery_url,
        help="GBFS v3.0 gbfs.json discovery URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.timeout_seconds,
        help="HTTP timeout for each GBFS request.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with get_session() as session:
            summary = ingest_gbfs_station_feeds(
                args.discovery_url,
                timeout_seconds=args.timeout_seconds,
                session=session,
            )
    except (
        GbfsClientError,
        GbfsTransformError,
        SQLAlchemyError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(status=1, message=f"GBFS ingestion failed: {exc}\n")

    print(format_ingestion_summary(summary))
    return 0
