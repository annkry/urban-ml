from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

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
)
from urban_ml.staging.objects import ObjectStore, ObjectStoreError, store_from_settings
from urban_ml.staging.snapshots import stage_cycle


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
    station_status_key: str
    stations_key: str


def ingest_gbfs_station_feeds(
    discovery_url: str,
    *,
    timeout_seconds: float,
    object_store: ObjectStore,
    json_fetcher: JsonFetcher = fetch_json_with_retry,
) -> GbfsIngestionSummary:
    """Fetch, validate, and stage GBFS station feeds to object storage."""

    observed_at = datetime.now(UTC)

    client = GbfsClient(
        discovery_url, timeout_seconds=timeout_seconds, json_fetcher=json_fetcher
    )
    raw_feeds = client.fetch_raw_feeds()
    system_id = raw_feeds.system_information.data.system_id

    station_ids = {
        station.station_id for station in raw_feeds.station_information.data.stations
    }
    status_station_ids = {
        station.station_id for station in raw_feeds.station_status.data.stations
    }

    stations = build_stations(raw_feeds, system_id=system_id, observed_at=observed_at)
    station_status_records = build_station_status(
        raw_feeds,
        system_id=system_id,
        known_station_ids=station_ids,
        observed_at=observed_at,
    )

    staged = stage_cycle(
        object_store,
        status_records=station_status_records,
        station_records=stations,
        observed_at=observed_at,
    )
    if staged is None:
        raise GbfsTransformError(
            "station_status feed yielded no rows; nothing was staged"
        )

    return GbfsIngestionSummary(
        discovery_url=discovery_url,
        system_id=system_id,
        station_information_last_updated=raw_feeds.station_information.last_updated.isoformat(),
        station_status_last_updated=raw_feeds.station_status.last_updated.isoformat(),
        station_count=len(station_ids),
        status_count=len(status_station_ids),
        matched_station_status_count=len(station_ids & status_station_ids),
        processed_station_status_count=len(station_status_records),
        station_status_key=staged.station_status_key,
        stations_key=staged.stations_key,
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
        f"Station status staged at: {summary.station_status_key}",
        f"Stations staged at: {summary.stations_key}",
    ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch, validate, and stage GBFS station information and "
            "station status feeds to object storage."
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

    object_store = store_from_settings()
    if object_store is None:
        parser.exit(
            status=1,
            message="GBFS ingestion failed: GCS_BUCKET is not set; nowhere to stage.\n",
        )

    try:
        summary = ingest_gbfs_station_feeds(
            args.discovery_url,
            timeout_seconds=args.timeout_seconds,
            object_store=object_store,
        )
    except (
        GbfsClientError,
        GbfsTransformError,
        ObjectStoreError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(status=1, message=f"GBFS ingestion failed: {exc}\n")

    print(format_ingestion_summary(summary))
    return 0
