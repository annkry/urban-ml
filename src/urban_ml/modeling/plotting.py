from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402


def plot_predicted_vs_actual(
    y_true: ArrayLike, y_pred: ArrayLike, *, mae: float, rmse: float, title: str
) -> Figure:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_arr, pred_arr, alpha=0.15, s=8)
    lo = float(min(true_arr.min(), pred_arr.min()))
    hi = float(max(true_arr.max(), pred_arr.max()))
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nMAE={mae:.2f}  RMSE={rmse:.2f}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_timeseries_comparison(
    observed_at: ArrayLike,
    actual: ArrayLike,
    baseline_pred: ArrayLike,
    lgbm_pred: ArrayLike,
    *,
    station_id: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(observed_at, actual, label="actual", linewidth=1.5)
    ax.plot(observed_at, baseline_pred, label="baseline (persistence)", linestyle="--")
    ax.plot(observed_at, lgbm_pred, label="lightgbm", linestyle="--")
    ax.set_title(f"Predicted vs actual — station {station_id}")
    ax.set_xlabel("time")
    ax.set_ylabel("vehicles available")
    ax.legend()
    fig.tight_layout()
    return fig
