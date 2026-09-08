from __future__ import annotations

import fcntl
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from urban_ml.core.config import settings
from urban_ml.ingestion.gbfs_client import GbfsClientError
from urban_ml.ingestion.gbfs_ingest import ingest_gbfs_station_feeds
from urban_ml.processing.gbfs_transform import GbfsTransformError
from urban_ml.storage.db import get_session

INTERVAL_SECONDS = 300
LOCK_PATH = Path(tempfile.gettempdir()) / "urban-ml-ingest-loop.lock"


def _acquire_lock() -> object:
    """Refuse to start a second loop against the same machine."""

    lock_file = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Another ingest_loop is already running (lock: {LOCK_PATH}).")
        sys.exit(1)
    return lock_file


def run_once() -> None:
    started = datetime.now(UTC)
    try:
        with get_session() as session:
            summary = ingest_gbfs_station_feeds(
                settings.gbfs_discovery_url,
                timeout_seconds=settings.timeout_seconds,
                session=session,
            )
        print(
            f"[{started.isoformat()}] {summary.processed_station_status_count} stations ingested"
        )
    except (GbfsClientError, GbfsTransformError, SQLAlchemyError) as exc:
        print(f"[{started.isoformat()}] Ingestion failed: {exc}")


def main() -> int:
    _acquire_lock()
    print(f"Ingesting every {INTERVAL_SECONDS}s. Press Ctrl+C to stop.")
    try:
        while True:
            loop_started = time.monotonic()
            run_once()
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, INTERVAL_SECONDS - elapsed))
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
