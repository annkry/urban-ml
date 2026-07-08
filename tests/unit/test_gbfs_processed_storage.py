from datetime import UTC, datetime

import pytest

from urban_ml.domain.station_snapshot import StationSnapshot
from urban_ml.processing.gbfs_processed_storage import save_processed_station_snapshots


pytest.importorskip("pyarrow")


def test_save_processed_station_snapshots_writes_parquet_and_metadata(tmp_path) -> None:
    observed_at = datetime(2026, 7, 5, 11, 6, 3, tzinfo=UTC)
    snapshots = [
        StationSnapshot(
            observed_at=observed_at,
            system_id="toronto",
            station_id="station-1",
            station_name="Main Station",
            lat=47.3769,
            lon=8.5417,
            capacity=20,
            num_vehicles_available=7,
            num_docks_available=13,
            is_installed=True,
            is_renting=True,
            is_returning=True,
            last_reported=datetime(2026, 7, 5, 11, 6, 1, tzinfo=UTC),
        )
    ]

    paths = save_processed_station_snapshots(
        snapshots,
        output_dir=tmp_path,
        system_id="Bike Share Toronto",
        observed_at=observed_at,
        source_raw_snapshot_dir=str(
            tmp_path / "bike-share-toronto" / "2026-07-05" / "11-06-03"
        ),
        snapshot_relative_dir="bike-share-toronto/2026-07-05/11-06-03",
    )

    assert (
        paths.snapshot_dir
        == tmp_path / "bike-share-toronto" / "2026-07-05" / "11-06-03"
    )
    assert paths.station_snapshots_path.exists()
    assert paths.metadata_path.exists()
