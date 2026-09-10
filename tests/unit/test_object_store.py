from __future__ import annotations

from pathlib import Path

import pytest

from urban_ml.staging.objects import (
    LocalObjectStore,
    ObjectNotFoundError,
    ObjectStoreError,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(root=tmp_path)


def test_put_then_get_round_trips(store: LocalObjectStore) -> None:
    store.put("a/b/c.parquet", b"payload")

    assert store.get("a/b/c.parquet") == b"payload"


def test_put_creates_nested_prefixes(store: LocalObjectStore, tmp_path: Path) -> None:
    """Keys are flat on GCS; on disk they need the directories made."""

    store.put("snapshots/station_status/date=2026-09-09/x.parquet", b"payload")

    assert (tmp_path / "snapshots/station_status/date=2026-09-09/x.parquet").is_file()


def test_get_of_a_missing_key_is_distinguishable(store: LocalObjectStore) -> None:
    with pytest.raises(ObjectNotFoundError):
        store.get("nothing/here.parquet")


def test_overwrite_leaves_no_partial_file(
    store: LocalObjectStore, tmp_path: Path
) -> None:
    """The rolling window is overwritten every cycle while serving reads it."""

    store.put("recent.parquet", b"first")
    store.put("recent.parquet", b"second-and-longer")

    assert store.get("recent.parquet") == b"second-and-longer"
    assert [path.name for path in tmp_path.iterdir()] == ["recent.parquet"]


def test_exists_reflects_writes_and_deletes(store: LocalObjectStore) -> None:
    assert store.exists("k") is False

    store.put("k", b"v")
    assert store.exists("k") is True

    store.delete("k")
    assert store.exists("k") is False


def test_delete_of_a_missing_key_is_not_an_error(store: LocalObjectStore) -> None:
    """Retention deletes what the archive confirmed; a re-run must be a no-op."""

    store.delete("never/existed.parquet")


def test_list_keys_filters_by_prefix_and_sorts(store: LocalObjectStore) -> None:
    store.put("snapshots/b.parquet", b"")
    store.put("snapshots/a.parquet", b"")
    store.put("serving/recent.parquet", b"")

    assert store.list_keys("snapshots/") == [
        "snapshots/a.parquet",
        "snapshots/b.parquet",
    ]


def test_list_keys_works_when_the_root_is_a_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scripts point the store at a directory like "gcp_storage"."""

    monkeypatch.chdir(tmp_path)
    relative = LocalObjectStore(root=Path("gcp_storage"))
    relative.put("snapshots/a.parquet", b"payload")

    assert relative.list_keys("snapshots/") == ["snapshots/a.parquet"]


def test_keys_cannot_escape_the_store_root(store: LocalObjectStore) -> None:
    """A traversing key would write outside the bucket's equivalent."""

    with pytest.raises(ObjectStoreError, match="escapes the store root"):
        store.put("../outside.parquet", b"payload")
