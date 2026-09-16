from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from urban_ml.core.config import settings
from urban_ml.core.logging import configure_logging, get_logger
from urban_ml.ingestion.gbfs_client import GbfsClientError
from urban_ml.ingestion.gbfs_ingest import ingest_gbfs_station_feeds
from urban_ml.processing.gbfs_transform import GbfsTransformError
from urban_ml.staging.objects import ObjectStoreError, store_from_settings

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title=f"{settings.app_name} ingestion", version="0.1.0")

object_store = store_from_settings()
if object_store is None:
    logger.error("GCS_BUCKET is not set; every ingestion cycle will fail")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Deliberately touches nothing external.

    Cloud Run probes this on every cold start; a check that reached out to
    the bucket would turn a transient storage blip into a failed deploy.
    """

    return {"status": "ok", "service": f"{settings.app_name} ingestion"}


@app.post("/ingest")
def ingest() -> dict[str, Any]:
    """Run one ingestion cycle.

    Returns 502 on an upstream, transform or storage failure so a failed run
    is visible in Cloud Scheduler's own retry accounting rather than being
    swallowed as a success. The next tick arrives in five minutes regardless,
    so a transient error costs one snapshot rather than the whole loop --
    which is the real advantage of a scheduler over a long-lived polling
    process.
    """

    if object_store is None:
        raise HTTPException(
            status_code=500,
            detail="GCS_BUCKET is not set; there is nowhere to stage snapshots",
        )

    try:
        summary = ingest_gbfs_station_feeds(
            settings.gbfs_discovery_url,
            timeout_seconds=settings.timeout_seconds,
            object_store=object_store,
        )
    except (GbfsClientError, GbfsTransformError, ObjectStoreError) as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "Staged %d station status rows at %s",
        summary.processed_station_status_count,
        summary.station_status_key,
    )
    return {
        "system_id": summary.system_id,
        "station_status_rows": summary.processed_station_status_count,
        "stations_discovered": summary.station_count,
        "station_status_key": summary.station_status_key,
        "stations_key": summary.stations_key,
    }
