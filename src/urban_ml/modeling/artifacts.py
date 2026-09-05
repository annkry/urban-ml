from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb

MODEL_FILENAME = "model.txt"
ENCODING_FILENAME = "station_id_encoding.json"
METADATA_FILENAME = "metadata.json"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_model_artifacts(
    model_dir: Path,
    *,
    booster: lgb.Booster,
    encoding: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / MODEL_FILENAME))
    _write_json(model_dir / ENCODING_FILENAME, encoding)
    _write_json(model_dir / METADATA_FILENAME, metadata)


def load_booster(model_dir: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_dir / MODEL_FILENAME))


def load_station_id_encoding(model_dir: Path) -> dict[str, int]:
    encoding: dict[str, int] = json.loads((model_dir / ENCODING_FILENAME).read_text())
    return encoding


def load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = json.loads((model_dir / METADATA_FILENAME).read_text())
    return metadata


def model_artifacts_exist(model_dir: Path) -> bool:
    return all(
        (model_dir / name).is_file()
        for name in (MODEL_FILENAME, ENCODING_FILENAME, METADATA_FILENAME)
    )
