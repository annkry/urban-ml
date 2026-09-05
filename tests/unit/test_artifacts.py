from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from urban_ml.modeling.artifacts import (
    load_booster,
    load_metadata,
    load_station_id_encoding,
    model_artifacts_exist,
    save_model_artifacts,
)

ENCODING = {"station-a": 0, "station-b": 1, "station-c": 2}
METADATA = {"mlflow_run_id": "abc123", "horizon_minutes": 120, "test_mae": 1.43}


def _training_frame(rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(rows, 4))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.1, size=rows)
    return X, y


def _fit(early_stopping: bool) -> lgb.LGBMRegressor:
    X, y = _training_frame()
    model = lgb.LGBMRegressor(n_estimators=60, random_state=0, verbosity=-1)
    if early_stopping:
        model.fit(
            X,
            y,
            eval_set=[(X[:100], y[:100])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=False)],
        )
    else:
        model.fit(X, y)
    return model


@pytest.mark.parametrize("early_stopping", [False, True])
def test_saved_booster_predicts_identically_to_the_fitted_model(
    tmp_path: Path, early_stopping: bool
) -> None:
    """The reason this phase needs a test at all.

    Training fits through LightGBM's scikit-learn wrapper, but serving reloads
    the raw booster from disk — a different code path. With early stopping the
    wrapper predicts using best_iteration_ while a reloaded booster uses
    whatever the file contains, so a mismatch here would not raise: it would
    quietly serve predictions from a different number of trees.
    """

    model = _fit(early_stopping)
    save_model_artifacts(
        tmp_path, booster=model.booster_, encoding=ENCODING, metadata=METADATA
    )

    X_new, _ = _training_frame(rows=50)
    expected = model.predict(X_new)
    actual = load_booster(tmp_path).predict(X_new)

    np.testing.assert_array_equal(np.asarray(actual).ravel(), np.asarray(expected))


def test_encoding_and_metadata_round_trip(tmp_path: Path) -> None:
    save_model_artifacts(
        tmp_path, booster=_fit(False).booster_, encoding=ENCODING, metadata=METADATA
    )

    assert load_station_id_encoding(tmp_path) == ENCODING
    assert load_metadata(tmp_path) == METADATA


def test_encoding_is_written_sorted_for_reviewable_diffs(tmp_path: Path) -> None:
    save_model_artifacts(
        tmp_path,
        booster=_fit(False).booster_,
        encoding={"station-c": 2, "station-a": 0},
        metadata=METADATA,
    )

    written = (tmp_path / "station_id_encoding.json").read_text()
    assert list(json.loads(written)) == ["station-a", "station-c"]
    assert written.endswith("\n")


def test_model_artifacts_exist_requires_every_file(tmp_path: Path) -> None:
    assert not model_artifacts_exist(tmp_path)

    save_model_artifacts(
        tmp_path, booster=_fit(False).booster_, encoding=ENCODING, metadata=METADATA
    )
    assert model_artifacts_exist(tmp_path)

    (tmp_path / "station_id_encoding.json").unlink()
    assert not model_artifacts_exist(tmp_path)


def test_committed_model_loads_and_matches_its_metadata() -> None:
    """Guards the actual served artifacts, not a synthetic fixture: a
    corrupted or half-committed models/current/ should fail CI rather than
    the deployment."""

    model_dir = Path("models/current")
    if not model_artifacts_exist(model_dir):
        pytest.skip("no committed model in models/current/")

    booster = load_booster(model_dir)
    encoding = load_station_id_encoding(model_dir)
    metadata = load_metadata(model_dir)

    assert booster.num_trees() > 0
    assert len(encoding) == len(set(encoding.values())), "encoding codes must be unique"
    assert sorted(encoding.values()) == list(range(len(encoding)))
    assert metadata["mlflow_run_id"]
    assert metadata["horizon_minutes"] > 0

    assert len(encoding) > 100, (
        f"models/current/ covers only {len(encoding)} stations - looks like a "
        "test fixture overwrote the real model"
    )
