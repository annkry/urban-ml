from __future__ import annotations

import pandas as pd
import polars as pl

from urban_ml.features.build_features import FEATURE_COLUMNS

STATION_ENCODING_ARTIFACT_PATH = "station_id_encoding.json"

MODEL_FEATURE_COLUMNS = ["station_id_code", *FEATURE_COLUMNS[1:]]


def build_station_id_encoding(stations: pl.DataFrame) -> dict[str, int]:
    """Stable station_id -> integer code mapping, built from the full known
    station catalog (not just whatever appears in a given training split) so
    a code means the same station across runs. Must be persisted alongside
    the model and reapplied identically at serving time."""

    station_ids = sorted(stations["station_id"].unique().to_list())
    return {station_id: code for code, station_id in enumerate(station_ids)}


def encode_station_id(df: pl.DataFrame, encoding: dict[str, int]) -> pl.DataFrame:
    """A station_id absent from `encoding` (never seen in training) encodes
    to null, not a guessed code — callers should treat that as "can't
    predict for this station" rather than silently extrapolating."""

    return df.with_columns(
        pl.col("station_id")
        .replace_strict(encoding, default=None, return_dtype=pl.Int32)
        .alias("station_id_code")
    )


_NUMERIC_MODEL_FEATURE_COLUMNS = [
    c for c in MODEL_FEATURE_COLUMNS if c != "station_id_code"
]


def to_model_frame(df: pl.DataFrame, encoding: dict[str, int]) -> pd.DataFrame:
    """The one place both training and serving build a model-ready pandas
    frame — this is what actually enforces train/serve parity at the dtype
    level, not just the column-selection level.

    Numeric feature columns are force-cast to float64 unconditionally, even
    though some (capacity, num_vehicles_available, ...) never happen to be
    null. Lag/rolling columns ARE genuinely nullable (insufficient history),
    and pandas infers int64 for a numeric column with no nulls in a given
    batch but float64 when nulls are present — since a single live serving
    row often has no nulls where a large training batch does, a naive
    .to_pandas() would silently produce different dtypes between training
    and serving and break MLflow's schema enforcement (this is exactly what
    "Integer columns in Python cannot represent missing values" warns about
    at training time). Forcing float64 everywhere sidesteps the ambiguity
    entirely instead of relying on a particular batch's null pattern.
    """

    pandas_df = (
        encode_station_id(df, encoding).select(MODEL_FEATURE_COLUMNS).to_pandas()
    )
    pandas_df[_NUMERIC_MODEL_FEATURE_COLUMNS] = pandas_df[
        _NUMERIC_MODEL_FEATURE_COLUMNS
    ].astype("float64")
    return pandas_df
