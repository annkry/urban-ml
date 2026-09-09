from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _blank_is_unset(value: object) -> object:
    """An unset GitHub Actions variable arrives as "", not as absent."""

    return None if value == "" else value


type BlankIsUnset[T] = Annotated[T | None, BeforeValidator(_blank_is_unset)]


class Settings(BaseSettings):
    app_name: str = "Urban ML Platform"
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # GBFS ingestion settings
    gbfs_discovery_url: str = (
        "https://toronto.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json"
    )
    timeout_seconds: float = 10.0

    # Database settings
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/urban_ml"

    # Object storage staging.
    gcs_bucket: str = ""

    # ML / MLflow settings
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_dir: Path = Path("models/current")
    forecast_horizon_minutes: int = 120
    system_id: str | None = None

    # Archive settings. The archive holds the full history for training
    hf_dataset_repo: str = ""
    hf_token: str | None = None

    # Days before this are never archived.
    archive_start_date: date | None = None

    # Days of station_status to keep in the database.
    retention_days: BlankIsUnset[int] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
