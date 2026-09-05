from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from urban_ml.features.build_features import add_target, compute_features


def _raw_frame(
    *, station_id: str, times: list[datetime], values: list[int]
) -> pl.DataFrame:
    n = len(times)
    return pl.DataFrame(
        {
            "station_id": [station_id] * n,
            "observed_at": times,
            "num_vehicles_available": values,
            "num_docks_available": [5] * n,
            "is_installed": [True] * n,
            "is_renting": [True] * n,
            "is_returning": [True] * n,
        }
    )


def _regular_series() -> tuple[pl.DataFrame, list[datetime]]:
    base = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    times = [base + timedelta(minutes=5 * i) for i in range(8)]
    values = list(range(10, 18))
    return _raw_frame(station_id="A", times=times, values=values), times


def test_lag_and_rolling_features_hand_computed() -> None:
    raw, times = _regular_series()
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})

    feats = compute_features(raw, stations)
    row = feats.filter(pl.col("observed_at") == times[6])

    assert row["num_vehicles_available"][0] == 16
    assert row["lag_5m_vehicles"][0] == 15
    assert row["lag_15m_vehicles"][0] == 13
    assert row["lag_30m_vehicles"][0] == 10
    assert row["lag_60m_vehicles"][0] is None  # no data 60 min back
    assert row["delta_15m"][0] == 3
    assert row["roll_mean_30m"][0] == pytest.approx(13.5)  # mean of values at t-25..t
    assert row["roll_mean_60m"][0] == pytest.approx(13.0)  # all 7 prior points + self
    assert row["occupancy_ratio"][0] == pytest.approx(16 / 20)


def test_occupancy_ratio_is_null_not_error_at_zero_capacity() -> None:
    raw, _ = _regular_series()
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [0]})

    feats = compute_features(raw, stations)

    assert feats["occupancy_ratio"].null_count() == feats.height
    assert feats["occupancy_ratio"].to_list() == [None] * feats.height


def test_lag_and_rolling_null_across_a_real_gap_not_stale_value() -> None:
    """A fixed-row .shift(n) would silently pull a stale value across the
    gap; the join_asof-based lag must instead null out once the gap exceeds
    its tolerance."""

    base = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    times = [base + timedelta(minutes=5 * i) for i in range(3)] + [
        base + timedelta(minutes=50 + 5 * i) for i in range(3)
    ]  # 0, 5, 10, then a 40-minute gap, then 50, 55, 60
    values = [1, 2, 3, 100, 101, 102]
    raw = _raw_frame(station_id="A", times=times, values=values)
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})

    feats = compute_features(raw, stations)
    first_after_gap = feats.filter(pl.col("observed_at") == times[3])

    assert first_after_gap["lag_5m_vehicles"][0] is None
    assert first_after_gap["lag_30m_vehicles"][0] is None
    # the trailing 30-min window at the first post-gap point contains only
    # that point itself, not any pre-gap value
    assert first_after_gap["roll_mean_30m"][0] == pytest.approx(100.0)


def test_features_unaffected_by_future_rows() -> None:
    """Direct leakage regression test: mutating a later row must not change
    any earlier row's features, since every derived column is backward
    looking by construction."""

    raw, times = _regular_series()
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})

    before = compute_features(raw, stations)

    mutated_raw = raw.with_columns(
        pl.when(pl.col("observed_at") == times[-1])
        .then(pl.lit(9999))
        .otherwise(pl.col("num_vehicles_available"))
        .alias("num_vehicles_available")
    )
    after = compute_features(mutated_raw, stations)

    earlier_before = before.filter(pl.col("observed_at") < times[-1])
    earlier_after = after.filter(pl.col("observed_at") < times[-1])
    assert earlier_before.equals(earlier_after)


def test_add_target_matches_the_real_future_reading() -> None:
    raw, times = _regular_series()
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})
    feats = compute_features(raw, stations)

    labeled = add_target(feats, raw, horizon_minutes=15, tolerance_minutes=10)

    row = labeled.filter(pl.col("observed_at") == times[0])
    assert (
        row["target_num_vehicles_available"][0] == 13
    )  # value at times[3], 15 min later
    assert row["target_observed_at"][0] == times[3]


def test_add_target_drops_rows_with_no_match_within_tolerance() -> None:
    base = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    times = [base + timedelta(minutes=5 * i) for i in range(3)] + [
        base + timedelta(minutes=60 + 5 * i) for i in range(3)
    ]  # 0, 5, 10, then a 50-minute gap, then 60, 65, 70
    values = [1, 2, 3, 100, 101, 102]
    raw = _raw_frame(station_id="A", times=times, values=values)
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})
    feats = compute_features(raw, stations)

    labeled = add_target(feats, raw, horizon_minutes=20, tolerance_minutes=5)

    # times[0] (t=0) wants a target near t=20; the nearest real reading is
    # t=60, 40 minutes away — outside a 5-minute tolerance, so it's dropped.
    assert labeled.filter(pl.col("observed_at") == times[0]).height == 0


def test_add_target_never_leaks_a_target_earlier_than_the_horizon() -> None:
    raw, _ = _regular_series()
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})
    feats = compute_features(raw, stations)

    labeled = add_target(feats, raw, horizon_minutes=15, tolerance_minutes=10)

    deltas = (labeled["target_observed_at"] - labeled["observed_at"]).dt.total_seconds()
    assert (deltas >= 15 * 60).all()


def test_cyclical_hour_reflects_local_time_not_utc() -> None:
    # 2026-01-01 03:00 UTC is 2025-12-31 22:00 in America/Toronto (EST, UTC-5)
    naive_utc = datetime(
        2026, 1, 1, 3, 0
    )  # naive on purpose — exercises the naive-input path
    raw = _raw_frame(station_id="A", times=[naive_utc], values=[10])
    stations = pl.DataFrame({"station_id": ["A"], "capacity": [20]})

    feats = compute_features(raw, stations)

    expected_hour = 22
    assert feats["hour_sin"][0] == pytest.approx(
        math.sin(2 * math.pi * expected_hour / 24)
    )
    assert feats["hour_cos"][0] == pytest.approx(
        math.cos(2 * math.pi * expected_hour / 24)
    )
