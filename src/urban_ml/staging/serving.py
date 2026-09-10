from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl

from urban_ml.core.logging import get_logger
from urban_ml.domain.station import Station
from urban_ml.staging.objects import ObjectStore, ObjectStoreError
from urban_ml.staging.snapshots import read_station_status, read_stations

logger = get_logger(__name__)

WINDOW_TTL_SECONDS = 60.0

STATIONS_TTL_SECONDS = 600.0


@dataclass
class _Entry:
    frame: pl.DataFrame
    read_at: float


@dataclass
class ServingData:
    """Cached reads of the two files the prediction path needs."""

    store: ObjectStore | None
    window_ttl: float = WINDOW_TTL_SECONDS
    stations_ttl: float = STATIONS_TTL_SECONDS
    _cache: dict[str, _Entry] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _cached(
        self, name: str, ttl: float, read: Callable[[ObjectStore], pl.DataFrame]
    ) -> pl.DataFrame:
        if self.store is None:
            raise ObjectStoreError("No object storage configured (GCS_BUCKET is unset)")

        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(name)
            if entry is not None and now - entry.read_at < ttl:
                return entry.frame

        frame = read(self.store)
        with self._lock:
            self._cache[name] = _Entry(frame=frame, read_at=time.monotonic())
        return frame

    def window(self) -> pl.DataFrame:
        """The rolling station_status window."""

        return self._cached("window", self.window_ttl, read_station_status)

    def stations(self) -> pl.DataFrame:
        """Current station details."""

        return self._cached("stations", self.stations_ttl, read_stations)

    def invalidate(self) -> None:
        """Drop everything cached, so the next read goes to the store."""

        with self._lock:
            self._cache.clear()


def station_details(
    stations: pl.DataFrame, *, system_id: str, station_id: str
) -> Station | None:
    """One station's current details, or None when it is not in the feed."""

    match = stations.filter(
        (pl.col("system_id") == system_id) & (pl.col("station_id") == station_id)
    )
    if match.is_empty():
        return None
    return Station(**match.row(0, named=True))


def system_ids(stations: pl.DataFrame) -> list[str]:
    """Every system present in the station list."""

    if stations.is_empty():
        return []
    return sorted(stations["system_id"].unique().to_list())
