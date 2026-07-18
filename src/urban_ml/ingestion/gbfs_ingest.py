from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

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
    build_station_snapshots,
)
from urban_ml.storage.db import get_session
from urban_ml.storage.repository import (
    complete_ingestion_run,
    fail_ingestion_run,
    save_raw_gbfs_payload,
    save_station_snapshots,
    start_ingestion_run,
)


@dataclass(frozen=True)
class GbfsIngestionSummary:
    discovery_url: str
    station_information_last_updated: str
    station_status_last_updated: str
    station_count: int
    status_count: int
    matched_station_status_count: int
    processed_station_snapshot_count: int


def ingest_gbfs_station_feeds(
    discovery_url: str,
    *,
    timeout_seconds: float,
    system_id: str,
    session: Session,
    json_fetcher: JsonFetcher = fetch_json_with_retry,
) -> GbfsIngestionSummary:
    """Fetch, validate, and persist GBFS station feeds to the database."""

    observed_at = datetime.now(UTC)
    run = start_ingestion_run(session, system_id=system_id, started_at=observed_at)

    try:
        client = GbfsClient(
            discovery_url, timeout_seconds=timeout_seconds, json_fetcher=json_fetcher
        )
        raw_feeds = client.fetch_raw_station_feeds()

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

        station_snapshots = build_station_snapshots(
            raw_feeds, system_id=system_id, observed_at=observed_at
        )
        save_station_snapshots(session, station_snapshots)

        complete_ingestion_run(
            session,
            run,
            finished_at=datetime.now(UTC),
            row_count=len(station_snapshots),
        )
    except (GbfsClientError, GbfsTransformError) as exc:
        fail_ingestion_run(
            session, run, finished_at=datetime.now(UTC), error_message=str(exc)
        )
        raise

    return GbfsIngestionSummary(
        discovery_url=discovery_url,
        station_information_last_updated=raw_feeds.station_information.last_updated.isoformat(),
        station_status_last_updated=raw_feeds.station_status.last_updated.isoformat(),
        station_count=len(station_ids),
        status_count=len(status_station_ids),
        matched_station_status_count=len(station_ids & status_station_ids),
        processed_station_snapshot_count=len(station_snapshots),
    )


def format_ingestion_summary(summary: GbfsIngestionSummary) -> str:
    lines = [
        "GBFS ingestion completed",
        f"Discovery URL: {summary.discovery_url}",
        f"Station information updated: {summary.station_information_last_updated}",
        f"Station status updated: {summary.station_status_last_updated}",
        f"Stations discovered: {summary.station_count}",
        f"Station statuses discovered: {summary.status_count}",
        f"Stations with matching status: {summary.matched_station_status_count}",
        f"Processed station snapshots: {summary.processed_station_snapshot_count}",
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
    parser.add_argument(
        "--system-id",
        default=settings.system_id,
        help="System identifier used to tag stored rows.",
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
                system_id=args.system_id,
                session=session,
            )
    except (GbfsClientError, GbfsTransformError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"GBFS ingestion failed: {exc}\n")

    print(format_ingestion_summary(summary))
    return 0
