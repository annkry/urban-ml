from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from urban_ml.core.config import settings
from urban_ml.domain.station_snapshot import StationSnapshot


@dataclass(frozen=True)
class ProcessedGbfsSnapshotPaths:
    snapshot_dir: Path
    metadata_path: Path
    station_snapshots_path: Path


def save_processed_station_snapshots(
    snapshots: list[StationSnapshot],
    *,
    output_dir: Path,
    system_id: str,
    observed_at: datetime,
    source_raw_snapshot_dir: Path,
    snapshot_relative_dir: Path,
) -> ProcessedGbfsSnapshotPaths:
    """Persist processed station snapshots as Parquet.

    Processed snapshots are flattened, ML-friendly rows derived from raw GBFS
    JSON. They are intentionally stored separately from raw data.
    """

    if not snapshots:
        raise ValueError("Cannot save an empty station snapshot dataset")

    snapshot_dir = output_dir / snapshot_relative_dir
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    station_snapshots_path = snapshot_dir / settings.station_snapshots_filename
    metadata_path = snapshot_dir / settings.station_snapshots_metadata_filename

    _write_station_snapshots_parquet(station_snapshots_path, snapshots)
    _write_json(
        metadata_path,
        {
            "system_id": system_id,
            "observed_at": observed_at.isoformat(),
            "row_count": len(snapshots),
            "format": "parquet",
            "source_raw_snapshot_dir": (
                str(source_raw_snapshot_dir) if source_raw_snapshot_dir else None
            ),
        },
    )

    return ProcessedGbfsSnapshotPaths(
        snapshot_dir=snapshot_dir,
        metadata_path=metadata_path,
        station_snapshots_path=station_snapshots_path,
    )


def _write_station_snapshots_parquet(
    path: Path,
    snapshots: list[StationSnapshot],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Saving processed station snapshots requires pyarrow. "
            "Install it with: uv add pyarrow"
        ) from exc

    rows = [snapshot.model_dump(mode="python") for snapshot in snapshots]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
