from __future__ import annotations

import math
from datetime import timedelta

import polars as pl

FEATURE_COLUMNS = [
    "station_id",
    "capacity",
    "num_vehicles_available",
    "num_docks_available",
    "occupancy_ratio",
    "lag_5m_vehicles",
    "lag_15m_vehicles",
    "lag_30m_vehicles",
    "lag_60m_vehicles",
    "delta_15m",
    "roll_mean_30m",
    "roll_std_30m",
    "roll_mean_60m",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

STATUS_COLUMNS = ["is_installed", "is_renting", "is_returning"]

_LAG_MINUTES = (5, 15, 30, 60)
_LAG_JOIN_TOLERANCE_MINUTES = 5

_REQUIRED_RAW_COLUMNS = {
    "station_id",
    "observed_at",
    "num_vehicles_available",
    "num_docks_available",
    "is_installed",
    "is_renting",
    "is_returning",
}
_REQUIRED_STATION_COLUMNS = {"station_id", "capacity"}


def _assert_columns(df: pl.DataFrame, required: set[str], *, label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _to_local_time(
    observed_at: pl.Expr, *, raw: pl.DataFrame, local_timezone: str
) -> pl.Expr:
    dtype = raw.schema["observed_at"]
    is_naive = isinstance(dtype, pl.Datetime) and dtype.time_zone is None
    if is_naive:
        return observed_at.dt.replace_time_zone("UTC").dt.convert_time_zone(
            local_timezone
        )
    return observed_at.dt.convert_time_zone(local_timezone)


def _lag_column(raw: pl.DataFrame, *, minutes: int) -> pl.DataFrame:
    """Backward join_asof, not .shift(n): a fixed row-shift assumes exactly
    n rows = n * 5min elapsed, which breaks under ingestion jitter or a
    skipped run."""

    lag_source = raw.select(
        ["station_id", "observed_at", "num_vehicles_available"]
    ).rename(
        {
            "observed_at": "_lag_observed_at",
            "num_vehicles_available": f"lag_{minutes}m_vehicles",
        }
    )
    searched = raw.select(["station_id", "observed_at"]).with_columns(
        (pl.col("observed_at") - timedelta(minutes=minutes)).alias("_search_time")
    )
    joined = searched.join_asof(
        lag_source,
        left_on="_search_time",
        right_on="_lag_observed_at",
        by="station_id",
        strategy="backward",
        tolerance=f"{_LAG_JOIN_TOLERANCE_MINUTES}m",
        check_sortedness=False,
    )
    return joined.select(["station_id", "observed_at", f"lag_{minutes}m_vehicles"])


def compute_features(
    raw: pl.DataFrame,
    stations: pl.DataFrame,
    *,
    local_timezone: str = "America/Toronto",
) -> pl.DataFrame:
    """Single source of truth for feature computation, used identically by
    training and serving — this IS train/serve parity, not just a
    convenience. Every derived column is backward-looking by construction
    (lag joins use strategy="backward", rolling windows are trailing), so a
    row's features never depend on data observed after that row's own
    observed_at.

    raw: station_id, observed_at, num_vehicles_available, num_docks_available,
         is_installed, is_renting, is_returning (any row order/sort).
    stations: station_id, capacity.
    """

    _assert_columns(raw, _REQUIRED_RAW_COLUMNS, label="raw")
    _assert_columns(stations, _REQUIRED_STATION_COLUMNS, label="stations")

    raw = raw.sort(["station_id", "observed_at"])

    features = raw.join(
        stations.select(["station_id", "capacity"]), on="station_id", how="left"
    )

    features = features.with_columns(
        pl.when(pl.col("capacity") > 0)
        .then(pl.col("num_vehicles_available") / pl.col("capacity"))
        .otherwise(None)
        .alias("occupancy_ratio")
    )

    for minutes in _LAG_MINUTES:
        lag_df = _lag_column(raw, minutes=minutes)
        features = features.join(lag_df, on=["station_id", "observed_at"], how="left")

    features = features.with_columns(
        (pl.col("num_vehicles_available") - pl.col("lag_15m_vehicles")).alias(
            "delta_15m"
        )
    )

    features = features.with_columns(
        pl.col("num_vehicles_available")
        .rolling_mean_by("observed_at", window_size="30m")
        .over("station_id")
        .alias("roll_mean_30m"),
        pl.col("num_vehicles_available")
        .rolling_std_by("observed_at", window_size="30m")
        .over("station_id")
        .alias("roll_std_30m"),
        pl.col("num_vehicles_available")
        .rolling_mean_by("observed_at", window_size="60m")
        .over("station_id")
        .alias("roll_mean_60m"),
    )

    local_time = _to_local_time(
        pl.col("observed_at"), raw=raw, local_timezone=local_timezone
    )
    hour = local_time.dt.hour()
    dow = local_time.dt.weekday()
    features = features.with_columns(
        (2 * math.pi * hour / 24).sin().alias("hour_sin"),
        (2 * math.pi * hour / 24).cos().alias("hour_cos"),
        (2 * math.pi * dow / 7).sin().alias("dow_sin"),
        (2 * math.pi * dow / 7).cos().alias("dow_cos"),
    )

    return features.select(["observed_at", *FEATURE_COLUMNS, *STATUS_COLUMNS])


def add_target(
    features: pl.DataFrame,
    raw: pl.DataFrame,
    *,
    horizon_minutes: int,
    tolerance_minutes: int = 10,
) -> pl.DataFrame:
    """Leakage-safe forward join_asof target.

    target_search_time = observed_at + horizon_minutes, matched via
    strategy="forward" (first row with observed_at >= target_search_time).
    This guarantees target_observed_at >= observed_at + horizon_minutes by
    construction, regardless of ingestion jitter — leakage-proof by
    construction, not by convention.

    tolerance_minutes' job is different: it prevents *mislabeling* (e.g. a
    multi-hour outage silently labeling "30-min-ahead" with a reading from
    hours later), not leakage. Rows with no match within tolerance get a
    null target and are dropped.
    """

    target_source = raw.select(
        ["station_id", "observed_at", "num_vehicles_available"]
    ).rename(
        {
            "observed_at": "target_observed_at",
            "num_vehicles_available": "target_num_vehicles_available",
        }
    )
    searched = features.with_columns(
        (pl.col("observed_at") + timedelta(minutes=horizon_minutes)).alias(
            "_search_time"
        )
    )
    joined = searched.join_asof(
        target_source,
        left_on="_search_time",
        right_on="target_observed_at",
        by="station_id",
        strategy="forward",
        tolerance=f"{tolerance_minutes}m",
        check_sortedness=False,
    )
    return joined.drop("_search_time").drop_nulls("target_num_vehicles_available")
