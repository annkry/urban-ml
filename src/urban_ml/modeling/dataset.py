from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from urban_ml.storage.repository import list_stations, station_status_history_query

MIN_STATION_HISTORY_ROWS = 288


def load_raw_status(engine: Engine, *, system_id: str) -> pl.DataFrame:
    query = station_status_history_query(system_id=system_id)
    with engine.connect() as connection:
        return pl.read_database(query, connection)


def load_stations(session: Session, *, system_id: str) -> pl.DataFrame:
    stations = list_stations(session, system_id=system_id)
    return pl.DataFrame(
        {
            "station_id": [station.station_id for station in stations],
            "capacity": [station.capacity for station in stations],
        },
        schema={"station_id": pl.String, "capacity": pl.Int64},
    )


def apply_quality_filters(
    labeled: pl.DataFrame, *, min_station_history_rows: int = MIN_STATION_HISTORY_ROWS
) -> pl.DataFrame:
    """Drop out-of-service anchor rows and stations with too little history.

    Filters on the anchor row's status only (not the target's) — predicting
    for an out-of-service station isn't the problem being modeled.
    """

    filtered = labeled.filter(pl.col("is_installed") & pl.col("is_renting"))
    counts = filtered.group_by("station_id").len(name="n")
    keep_stations = counts.filter(pl.col("n") >= min_station_history_rows)["station_id"]
    return filtered.filter(pl.col("station_id").is_in(keep_stations.to_list()))


@dataclass
class TimeSplit:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime


def time_split(
    df: pl.DataFrame, *, train_frac: float = 0.7, val_frac: float = 0.15
) -> TimeSplit:
    """Pooled (not per-station) time-based split — no shuffling, ever, for
    time-series data."""

    ordered = df.sort("observed_at")
    times = ordered["observed_at"]
    n = ordered.height
    train_cutoff = times[int(n * train_frac)]
    val_cutoff = times[int(n * (train_frac + val_frac))]

    train = ordered.filter(pl.col("observed_at") < train_cutoff)
    val = ordered.filter(
        (pl.col("observed_at") >= train_cutoff) & (pl.col("observed_at") < val_cutoff)
    )
    test = ordered.filter(pl.col("observed_at") >= val_cutoff)

    return TimeSplit(
        train=train,
        val=val,
        test=test,
        train_start=times[0],
        train_end=train_cutoff,
        val_start=train_cutoff,
        val_end=val_cutoff,
        test_start=val_cutoff,
        test_end=times[-1],
    )
