from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from sqlalchemy.exc import SQLAlchemyError

from urban_ml.ingestion.gbfs_client import GbfsClientError
from urban_ml.ingestion.gbfs_ingest import ingest_gbfs_station_feeds
from urban_ml.processing.gbfs_transform import GbfsTransformError
from urban_ml.staging.objects import store_from_settings
from urban_ml.storage.db import get_session

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title=f"{settings.app_name} ingestion", version="0.1.0")

object_store = store_from_settings()
if object_store is None:
    logger.warning("GCS_BUCKET is not set; snapshots will not be staged")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Deliberately touches nothing external.

    Cloud Run probes this on every cold start, and a check that queried the
    database would fail while a scale-to-zero Postgres was still waking up.
    """

    return {"status": "ok", "service": f"{settings.app_name} ingestion"}


@app.post("/ingest")
def ingest() -> dict[str, Any]:
    """Run one ingestion cycle.

    Returns 502 on an upstream or transform failure so a failed run is visible
    in Cloud Scheduler's own retry accounting rather than being swallowed as a
    success. The next tick arrives in five minutes regardless, so a transient
    GBFS error costs one snapshot rather than the whole loop -- which is the
    real advantage of a scheduler over a long-lived polling process.
    """

    try:
        with get_session() as session:
            summary = ingest_gbfs_station_feeds(
                settings.gbfs_discovery_url,
                timeout_seconds=settings.timeout_seconds,
                session=session,
                object_store=object_store,
            )
    except (GbfsClientError, GbfsTransformError, SQLAlchemyError) as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "Ingested %d station status rows (%d station detail changes, staged=%s)",
        summary.processed_station_status_count,
        summary.station_change_count,
        summary.snapshot_staged,
    )
    return {
        "system_id": summary.system_id,
        "station_status_rows": summary.processed_station_status_count,
        "station_detail_changes": summary.station_change_count,
        "stations_discovered": summary.station_count,
        "snapshot_staged": summary.snapshot_staged,
    }
