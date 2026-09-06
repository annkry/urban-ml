from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from urban_ml.archive.export import (
    export_day,
    export_station_catalog,
    observed_day_range,
)
from urban_ml.archive.layout import ARCHIVED_TABLES, STATION_STATUS
from urban_ml.archive.publish import upload_partitions
from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from urban_ml.storage.db import engine

configure_logging()
logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--day", type=date.fromisoformat, help="single UTC day")
    group.add_argument(
        "--backfill", action="store_true", help="every day present in Postgres"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write Parquet locally and report sizes, but do not upload",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="where to stage Parquet (default: a temporary directory)",
    )
    return parser.parse_args()


def _days_to_export(args: argparse.Namespace) -> list[date]:
    if args.day:
        return [args.day]
    if not args.backfill:
        return [(datetime.now(UTC) - timedelta(days=1)).date()]

    bounds = observed_day_range(engine, table=STATION_STATUS)
    if bounds is None:
        return []
    first, last = bounds
    today = datetime.now(UTC).date()
    last = min(last, today - timedelta(days=1))
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def main() -> int:
    args = _parse_args()

    if not args.dry_run and not settings.hf_dataset_repo:
        raise SystemExit("HF_DATASET_REPO is not set; nowhere to publish to.")
    if not args.dry_run and not settings.hf_token:
        raise SystemExit("HF_TOKEN is not set; cannot authenticate to the Hub.")

    days = _days_to_export(args)
    if not days:
        logger.warning("No days to export.")
        return 0

    logger.info("Exporting %d day(s): %s .. %s", len(days), days[0], days[-1])

    with tempfile.TemporaryDirectory() as temporary:
        staging = args.staging_dir or Path(temporary)
        totals: dict[str, int] = dict.fromkeys(ARCHIVED_TABLES, 0)

        for day in days:
            for table in ARCHIVED_TABLES:
                result = export_day(engine, table=table, day=day, staging_dir=staging)
                if result is not None:
                    totals[table] += result[1]

        _, catalog_rows = export_station_catalog(
            engine, day=datetime.now(UTC).date(), staging_dir=staging
        )
        logger.info("stations: %d rows staged", catalog_rows)

        for table, rows in totals.items():
            logger.info("%s: %d rows staged", table, rows)

        if args.dry_run:
            logger.info("Dry run; staged under %s, nothing uploaded.", staging)
            return 0

        span = f"{days[0]}" if len(days) == 1 else f"{days[0]}..{days[-1]}"
        upload_partitions(
            staging,
            repo_id=settings.hf_dataset_repo,
            token=settings.hf_token or "",
            commit_message=f"Add station data for {span}",
        )

    logger.info("Published to %s", settings.hf_dataset_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
