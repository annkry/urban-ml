from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from urban_ml.core.config import settings
from urban_ml.core.logging import get_logger

logger = get_logger(__name__)


class ObjectStoreError(Exception):
    """The store could not serve a request."""


class ObjectNotFoundError(ObjectStoreError):
    """No object exists at the requested key."""


class ObjectStore(Protocol):
    """The subset of blob-store behaviour this project needs."""

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list_keys(self, prefix: str) -> list[str]: ...

    def delete(self, key: str) -> None: ...


@dataclass(frozen=True)
class LocalObjectStore:
    """Filesystem-backed store, for tests and for running the stack locally."""

    root: Path

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            raise ObjectStoreError(f"Key escapes the store root: {key!r}")
        return candidate

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.partial")
        temporary.write_bytes(data)
        temporary.replace(path)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(key) from exc

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str) -> list[str]:
        base = self.root.resolve()
        keys = (
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
        return sorted(key for key in keys if key.startswith(prefix))

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


@dataclass(frozen=True)
class GcsObjectStore:
    """Google Cloud Storage backend."""

    bucket: str

    def _blob(self, key: str) -> Any:
        from google.cloud import storage

        return storage.Client().bucket(self.bucket).blob(key)

    def put(self, key: str, data: bytes) -> None:
        from google.api_core import exceptions as gcs_exceptions

        try:
            self._blob(key).upload_from_string(
                data, content_type="application/octet-stream"
            )
        except gcs_exceptions.GoogleAPIError as exc:
            raise ObjectStoreError(f"Could not write gs://{self.bucket}/{key}") from exc

    def get(self, key: str) -> bytes:
        from google.api_core import exceptions as gcs_exceptions

        try:
            return bytes(self._blob(key).download_as_bytes())
        except gcs_exceptions.NotFound as exc:
            raise ObjectNotFoundError(key) from exc
        except gcs_exceptions.GoogleAPIError as exc:
            raise ObjectStoreError(f"Could not read gs://{self.bucket}/{key}") from exc

    def exists(self, key: str) -> bool:
        from google.api_core import exceptions as gcs_exceptions

        try:
            return bool(self._blob(key).exists())
        except gcs_exceptions.GoogleAPIError as exc:
            raise ObjectStoreError(f"Could not stat gs://{self.bucket}/{key}") from exc

    def list_keys(self, prefix: str) -> list[str]:
        from google.api_core import exceptions as gcs_exceptions
        from google.cloud import storage

        try:
            blobs = storage.Client().list_blobs(self.bucket, prefix=prefix)
            return sorted(blob.name for blob in blobs)
        except gcs_exceptions.GoogleAPIError as exc:
            raise ObjectStoreError(
                f"Could not list gs://{self.bucket}/{prefix}"
            ) from exc

    def delete(self, key: str) -> None:
        from google.api_core import exceptions as gcs_exceptions

        try:
            self._blob(key).delete()
        except gcs_exceptions.NotFound:
            return
        except gcs_exceptions.GoogleAPIError as exc:
            raise ObjectStoreError(
                f"Could not delete gs://{self.bucket}/{key}"
            ) from exc


def store_from_settings() -> ObjectStore | None:
    """The configured store, or None when staging is switched off."""

    if not settings.gcs_bucket:
        return None
    return GcsObjectStore(bucket=settings.gcs_bucket)
