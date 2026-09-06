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


def _latest_catalog(archive_dir: Path) -> pl.DataFrame:
    """The most recent station-catalog snapshot.

    Snapshots are dated, but feature computation currently applies one
    catalog to every row regardless of when it was observed. Using the latest
    keeps behaviour identical to reading Postgres, which only ever holds
    current capacities. The dated snapshots accumulate so that an as-of join
    becomes possible later; nothing reads them that way yet.
    """

    snapshots = sorted((archive_dir / STATIONS).glob("date=*/*.parquet"))
    if not snapshots:
        raise FileNotFoundError(f"No station catalog snapshots under {archive_dir}")
    return pl.read_parquet(snapshots[-1])


def load_stations_from_archive(archive_dir: Path, *, system_id: str) -> pl.DataFrame:
    """Drop-in replacement for modeling.dataset.load_stations.

    Same two columns in the same order, so compute_features cannot tell which
    source it was given.
    """

    return (
        _latest_catalog(archive_dir)
        .filter(pl.col("system_id") == system_id)
        .select(["station_id", "capacity"])
    )


def system_ids_in_archive(archive_dir: Path) -> list[str]:
    return sorted(_latest_catalog(archive_dir)["system_id"].unique().to_list())


def capacity_history(archive_dir: Path, *, system_id: str) -> pl.DataFrame:
    """Every catalog snapshot, for seeing how capacity changed over time.

    Nothing in training uses this yet — it exists so the accumulating
    snapshots are reachable, and so a capacity change is visible the moment
    someone looks for one.
    """

    frames = [
        pl.read_parquet(path).with_columns(
            pl.lit(path.parent.name.removeprefix("date=")).str.to_date().alias("as_of")
        )
        for path in sorted((archive_dir / STATIONS).glob("date=*/*.parquet"))
    ]
    if not frames:
        return pl.DataFrame(
            schema={"station_id": pl.String, "as_of": pl.Date, "capacity": pl.Int64}
        )
    return (
        pl.concat(frames)
        .filter(pl.col("system_id") == system_id)
        .select(["station_id", "as_of", "capacity"])
        .sort(["station_id", "as_of"])
    )
