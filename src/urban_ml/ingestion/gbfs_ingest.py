from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from urban_ml.core.config import settings
from urban_ml.ingestion.gbfs_client import GbfsClient, GbfsClientError
from urban_ml.ingestion.gbfs_storage import save_raw_gbfs_snapshot


@dataclass(frozen=True)
class GbfsIngestionSummary:
    discovery_url: str
    station_information_last_updated: str
    station_status_last_updated: str
    station_count: int
    status_count: int
    matched_station_status_count: int
    raw_snapshot_dir: str


def ingest_gbfs_station_feeds(
    discovery_url: str,
    *,
    timeout_seconds: float,
    raw_output_dir: Path,
    system_id: str,
) -> GbfsIngestionSummary:
    client = GbfsClient(discovery_url, timeout_seconds=timeout_seconds)
    raw_feeds = client.fetch_raw_station_feeds()

    station_ids = {
        station.station_id for station in raw_feeds.station_information.data.stations
    }
    status_station_ids = {
        station.station_id for station in raw_feeds.station_status.data.stations
    }

    snapshot_paths = save_raw_gbfs_snapshot(
        raw_feeds,
        output_dir=raw_output_dir,
        system_id=system_id,
    )
    raw_snapshot_dir = str(snapshot_paths.snapshot_dir)

    return GbfsIngestionSummary(
        discovery_url=discovery_url,
        station_information_last_updated=raw_feeds.station_information.last_updated.isoformat(),
        station_status_last_updated=raw_feeds.station_status.last_updated.isoformat(),
        station_count=len(station_ids),
        status_count=len(status_station_ids),
        matched_station_status_count=len(station_ids & status_station_ids),
        raw_snapshot_dir=raw_snapshot_dir,
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
        f"Raw snapshot saved: {summary.raw_snapshot_dir}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and validate GBFS station information and station status feeds."
    )
    parser.add_argument(
        "--discovery-url",
        default=settings.gbfs_discovery_url,
        help="GBFS v3.0 gbfs.json discovery URL. ",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.default_timeout_seconds,
        help="HTTP timeout for each GBFS request.",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=settings.default_raw_gbfs_dir,
        help="Directory where raw GBFS snapshots will be saved.",
    )
    parser.add_argument(
        "--system-id",
        default=settings.system_id,
        help="System identifier used in the raw snapshot path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = ingest_gbfs_station_feeds(
            args.discovery_url,
            timeout_seconds=args.timeout_seconds,
            raw_output_dir=args.raw_output_dir,
            system_id=args.system_id,
        )
    except GbfsClientError as exc:
        parser.exit(status=1, message=f"GBFS ingestion failed: {exc}\n")

    print(format_ingestion_summary(summary))
    return 0
