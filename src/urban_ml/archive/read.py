from __future__ import annotations

from pathlib import Path

import polars as pl

from urban_ml.archive.layout import STATION_STATUS, STATIONS, table_glob
from urban_ml.core.logging import get_logger

logger = get_logger(__name__)

_TRAINING_COLUMNS = [
    "station_id",
    "observed_at",
    "num_vehicles_available",
    "num_docks_available",
    "is_installed",
    "is_renting",
    "is_returning",
]


def download_archive(repo_id: str, *, token: str | None = None) -> Path:
    """Pull the dataset repo to the local Hugging Face cache and return it.

    Cached between runs, so a repeat training run re-downloads only partitions
    that changed.
    """

    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id, repo_type="dataset", token=token, allow_patterns=["*.parquet"]
    )
    return Path(local)


def scan_station_status(archive_dir: Path, *, system_id: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(archive_dir / table_glob(STATION_STATUS))
        .filter(pl.col("system_id") == system_id)
        .select(_TRAINING_COLUMNS)
        .sort(["station_id", "observed_at"])
    )


def load_raw_status_from_archive(archive_dir: Path, *, system_id: str) -> pl.DataFrame:
    """Drop-in replacement for modeling.dataset.load_raw_status.

    Returns the same columns in the same order, so compute_features cannot
    tell which source it was given.
    """

    frame = scan_station_status(archive_dir, system_id=system_id).collect()
    logger.info("Loaded %d rows from archive at %s", frame.height, archive_dir)
    return frame


def station_history(archive_dir: Path, *, system_id: str) -> pl.DataFrame:
    """Every recorded change to every station, oldest first."""

    return (
        pl.read_parquet(archive_dir / table_glob(STATIONS))
        .filter(pl.col("system_id") == system_id)
        .sort(["station_id", "observed_at"])
    )


def load_stations_from_archive(archive_dir: Path, *, system_id: str) -> pl.DataFrame:
    """Drop-in replacement for modeling.dataset.load_stations.

    Collapses the change log to each station's most recent details. Feature
    computation applies one capacity to every row regardless of when it was
    observed; joining as-of each observation is a later change, now that the
    history to join against is finally being recorded.
    """

    history = station_history(archive_dir, system_id=system_id)
    if history.height == 0:
        return pl.DataFrame(schema={"station_id": pl.String, "capacity": pl.Int64})
    return (
        history.group_by("station_id")
        .agg(pl.all().sort_by("observed_at").last())
        .select(["station_id", "capacity"])
        .sort("station_id")
    )


def system_ids_in_archive(archive_dir: Path) -> list[str]:
    return sorted(
        pl.read_parquet(archive_dir / table_glob(STATIONS))["system_id"]
        .unique()
        .to_list()
    )
