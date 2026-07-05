from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    default_raw_gbfs_dir: Path = Path("data/raw/gbfs")
    default_timeout_seconds: float = 10.0
    system_id: str = "toronto"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
