from __future__ import annotations

from pathlib import Path

from urban_ml.core.logging import get_logger

logger = get_logger(__name__)


def upload_partitions(
    staging_dir: Path, *, repo_id: str, token: str, commit_message: str
) -> None:
    """Upload every staged Parquet file as one commit.

    One commit for the whole batch rather than one per day: a 48-day backfill
    should read as a single event in the repo's history, and the Hub rejects
    rapid-fire commits anyway.
    """

    from huggingface_hub import HfApi

    files = sorted(staging_dir.rglob("*.parquet"))
    if not files:
        logger.warning("Nothing staged under %s; skipping upload", staging_dir)
        return

    total_bytes = sum(path.stat().st_size for path in files)
    logger.info(
        "Uploading %d partition(s), %.1f KB total, to %s",
        len(files),
        total_bytes / 1024,
        repo_id,
    )

    HfApi().upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=["**/*.parquet"],
        commit_message=commit_message,
    )
