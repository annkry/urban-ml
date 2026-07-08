from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from urban_ml.ingestion.gbfs_client import GbfsRawStationFeeds, JsonObject


@dataclass(frozen=True)
class RawGbfsSnapshotPaths:
    snapshot_dir: Path
    metadata_path: Path
    discovery_path: Path
    station_information_path: Path
    station_status_path: Path


def save_raw_gbfs_snapshot(
    raw_feeds: GbfsRawStationFeeds,
    *,
    output_dir: Path,
    system_id: str,
    observed_at: datetime,
) -> RawGbfsSnapshotPaths:
    snapshot_dir = _unique_snapshot_dir(output_dir, system_id, observed_at)
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    metadata_path = snapshot_dir / "metadata.json"
    discovery_path = snapshot_dir / "gbfs.json"
    station_information_path = snapshot_dir / "station_information.json"
    station_status_path = snapshot_dir / "station_status.json"

    metadata = {
        "system_id": system_id,
        "observed_at": observed_at.isoformat(),
        "discovery_url": raw_feeds.discovery_url,
        "station_information_url": raw_feeds.station_information_url,
        "station_status_url": raw_feeds.station_status_url,
        "gbfs_version": raw_feeds.discovery.version,
        "station_information_last_updated": raw_feeds.station_information.last_updated.isoformat(),
        "station_status_last_updated": raw_feeds.station_status.last_updated.isoformat(),
    }

    _write_json(metadata_path, metadata)
    _write_json(discovery_path, raw_feeds.discovery_payload)
    _write_json(station_information_path, raw_feeds.station_information_payload)
    _write_json(station_status_path, raw_feeds.station_status_payload)

    return RawGbfsSnapshotPaths(
        snapshot_dir=snapshot_dir,
        metadata_path=metadata_path,
        discovery_path=discovery_path,
        station_information_path=station_information_path,
        station_status_path=station_status_path,
    )


def _build_snapshot_dir(
    output_dir: Path,
    system_id: str,
    observed_at: datetime,
) -> Path:
    safe_system_id = _safe_path_part(system_id)
    date_part = observed_at.strftime("%Y-%m-%d")
    time_part = observed_at.strftime("%H-%M-%S")
    return output_dir / safe_system_id / date_part / time_part


def _unique_snapshot_dir(
    output_dir: Path,
    system_id: str,
    observed_at: datetime,
) -> Path:
    snapshot_dir = _build_snapshot_dir(output_dir, system_id, observed_at)
    if not snapshot_dir.exists():
        return snapshot_dir

    counter = 2
    while True:
        candidate = snapshot_dir.with_name(f"{snapshot_dir.name}-{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_path_part(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown-system"


def _write_json(path: Path, payload: JsonObject | dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
