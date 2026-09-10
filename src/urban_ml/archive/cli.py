"""Publish staged snapshots to the Hub, then trim what the Hub confirms."""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from urban_ml.archive.export import days_needing_export
from urban_ml.archive.layout import ARCHIVED_TABLES, STATION_STATUS
from urban_ml.archive.publish import archived_days, upload_partitions
from urban_ml.archive.retention import apply_staged_retention
from urban_ml.archive.staged import days_present_in_bucket, export_staged_day
from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from urban_ml.staging.objects import ObjectStore, store_from_settings

configure_logging()
logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--day", type=date.fromisoformat, help="single UTC day")
    group.add_argument(
        "--backfill", action="store_true", help="every day staged in the bucket"
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
    return parser.parse_args(argv)


def _days_to_export(store: ObjectStore, args: argparse.Namespace) -> list[date]:
    """Every complete day in the bucket that the archive does not hold."""

    today = datetime.now(UTC).date()
    if args.day:
        return [args.day]

    present = days_present_in_bucket(store)
    start = settings.archive_start_date
    if args.backfill:
        return days_needing_export(present, set(), today=today, start=start)

    already = archived_days(
        settings.hf_dataset_repo, table=STATION_STATUS, token=settings.hf_token
    )
    missing = days_needing_export(present, already, today=today, start=start)
    if already and len(missing) > 1:
        logger.warning(
            "Archive was missing %d complete days; exporting them now: %s",
            len(missing),
            ", ".join(str(day) for day in missing),
        )
    return missing


def _export_and_publish(
    store: ObjectStore, days: list[date], args: argparse.Namespace
) -> None:
    """Stage every table for the given days, then upload them as one commit."""

    logger.info("Exporting %d day(s): %s .. %s", len(days), days[0], days[-1])

    with tempfile.TemporaryDirectory() as temporary:
        staging = args.staging_dir or Path(temporary)
        totals: dict[str, int] = dict.fromkeys(ARCHIVED_TABLES, 0)

        for day in days:
            for table in ARCHIVED_TABLES:
                result = export_staged_day(
                    store, table=table, day=day, staging_dir=staging
                )
                if result is not None:
                    totals[table] += result[1]

        for table, rows in totals.items():
            logger.info("%s: %d rows staged", table, rows)

        if args.dry_run:
            logger.info("Dry run; staged under %s, nothing uploaded.", staging)
            return

        span = f"{days[0]}" if len(days) == 1 else f"{days[0]}..{days[-1]}"
        upload_partitions(
            staging,
            repo_id=settings.hf_dataset_repo,
            token=settings.hf_token or "",
            commit_message=f"Add station data for {span}",
        )

    logger.info("Published to %s", settings.hf_dataset_repo)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    store = store_from_settings()
    if store is None:
        raise SystemExit(
            "GCS_BUCKET is not set; there are no staged snapshots to read."
        )
    if not args.dry_run and not settings.hf_dataset_repo:
        raise SystemExit("HF_DATASET_REPO is not set; nowhere to publish to.")
    if not args.dry_run and not settings.hf_token:
        raise SystemExit("HF_TOKEN is not set; cannot authenticate to the Hub.")

    days = _days_to_export(store, args)
    if days:
        _export_and_publish(store, days, args)
    else:
        logger.warning("No days to export.")

    if args.dry_run:
        return 0

    apply_staged_retention(
        store,
        repo_id=settings.hf_dataset_repo,
        token=settings.hf_token,
        keep_days=settings.retention_days,
        today=datetime.now(UTC).date(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
