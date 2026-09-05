from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))
