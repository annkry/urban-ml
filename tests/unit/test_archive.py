from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from urban_ml.archive.export import (
    day_bounds,
    export_day,
    export_station_catalog,
    observed_day_range,
)
from urban_ml.archive.layout import (
    STATION_STATUS,
    STATION_VEHICLE_AVAILABILITY,
    partition_path,
)
from urban_ml.archive.read import (
    capacity_history,
    load_raw_status_from_archive,
    load_stations_from_archive,
    system_ids_in_archive,
)
from urban_ml.storage.models import (
    Base,
    Station,
    StationStatusRecord,
    StationVehicleAvailabilityRecord,
)

SYSTEM_ID = "toronto"


@pytest.fixture
def engine_with_rows():  # type: ignore[no-untyped-def]
    """A file-free SQLite database standing in for Postgres.

    The export path is plain Core SELECT plus Polars, so it does not depend on
    anything Postgres-specific, and testing it without a live database keeps
    this runnable in CI.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def status(day: int, hour: int, station: str, vehicles: int) -> StationStatusRecord:
        moment = datetime(2026, 9, day, hour, tzinfo=UTC)
        return StationStatusRecord(
            observed_at=moment,
            system_id=SYSTEM_ID,
            station_id=station,
            num_vehicles_available=vehicles,
            num_docks_available=20 - vehicles,
            is_installed=True,
            is_renting=True,
            is_returning=True,
            last_reported=moment,
        )

    session: Session = factory()
    session.add_all(
        [
            status(4, 0, "a", 5),
            status(4, 12, "a", 7),
            status(4, 12, "b", 2),
            status(6, 3, "a", 9),
        ]
    )
    session.add(
        StationVehicleAvailabilityRecord(
            observed_at=datetime(2026, 9, 4, 12, tzinfo=UTC),
            system_id=SYSTEM_ID,
            station_id="a",
            vehicle_type_id="ebike",
            count=3,
        )
    )
    session.add(
        Station(
            system_id=SYSTEM_ID,
            station_id="a",
            station_name="Alpha",
            lat=43.6,
            lon=-79.4,
            capacity=20,
        )
    )
    session.commit()
    session.close()
    yield engine
    engine.dispose()


def test_day_bounds_are_utc_midnight_to_midnight() -> None:
    start, end = day_bounds(date(2026, 9, 4))

    assert start == datetime(2026, 9, 4, 0, 0, tzinfo=UTC)
    assert end == datetime(2026, 9, 5, 0, 0, tzinfo=UTC)


def test_export_day_writes_only_that_day(engine_with_rows, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    result = export_day(
        engine_with_rows,
        table=STATION_STATUS,
        day=date(2026, 9, 4),
        staging_dir=tmp_path,
    )

    assert result is not None
    destination, rows = result
    assert rows == 3
    assert destination == tmp_path / partition_path(STATION_STATUS, date(2026, 9, 4))

    written = pl.read_parquet(destination)
    assert written.height == 3
    assert written["observed_at"].dt.date().unique().to_list() == [date(2026, 9, 4)]


def test_export_day_returns_none_for_a_day_with_no_rows(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """An ingestion outage must leave a missing partition, not an empty file
    that later reads as a real day with zero stations."""

    assert (
        export_day(
            engine_with_rows,
            table=STATION_STATUS,
            day=date(2026, 9, 5),
            staging_dir=tmp_path,
        )
        is None
    )
    assert list(tmp_path.rglob("*.parquet")) == []


def test_export_day_handles_the_vehicle_availability_table(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    result = export_day(
        engine_with_rows,
        table=STATION_VEHICLE_AVAILABILITY,
        day=date(2026, 9, 4),
        staging_dir=tmp_path,
    )

    assert result is not None
    assert pl.read_parquet(result[0])["vehicle_type_id"].to_list() == ["ebike"]


def test_re_exporting_a_day_overwrites_rather_than_appends(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    for _ in range(2):
        export_day(
            engine_with_rows,
            table=STATION_STATUS,
            day=date(2026, 9, 4),
            staging_dir=tmp_path,
        )

    written = pl.read_parquet(
        tmp_path / partition_path(STATION_STATUS, date(2026, 9, 4))
    )
    assert written.height == 3


def test_observed_day_range_spans_first_to_last(engine_with_rows) -> None:  # type: ignore[no-untyped-def]
    assert observed_day_range(engine_with_rows, table=STATION_STATUS) == (
        date(2026, 9, 4),
        date(2026, 9, 6),
    )


def test_archive_round_trip_matches_the_training_column_contract(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """compute_features must not be able to tell whether its input came from
    Postgres or from the archive."""

    for day in (date(2026, 9, 4), date(2026, 9, 6)):
        export_day(
            engine_with_rows, table=STATION_STATUS, day=day, staging_dir=tmp_path
        )

    loaded = load_raw_status_from_archive(tmp_path, system_id=SYSTEM_ID)

    assert loaded.columns == [
        "station_id",
        "observed_at",
        "num_vehicles_available",
        "num_docks_available",
        "is_installed",
        "is_renting",
        "is_returning",
    ]
    assert loaded.height == 4
    assert loaded["station_id"].to_list() == ["a", "a", "a", "b"]


def test_archive_read_filters_by_system_id(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    export_day(
        engine_with_rows,
        table=STATION_STATUS,
        day=date(2026, 9, 4),
        staging_dir=tmp_path,
    )

    assert load_raw_status_from_archive(tmp_path, system_id="montreal").height == 0


def test_station_catalog_is_exported_because_features_need_capacity(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """Without the catalog the archive is not a complete training input:
    compute_features requires {station_id, capacity}."""

    destination, rows = export_station_catalog(
        engine_with_rows, day=date(2026, 9, 6), staging_dir=tmp_path
    )

    assert rows == 1
    assert destination.exists()
    assert load_stations_from_archive(tmp_path, system_id=SYSTEM_ID).to_dicts() == [
        {"station_id": "a", "capacity": 20}
    ]


def test_catalog_columns_match_what_compute_features_requires(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    export_station_catalog(engine_with_rows, day=date(2026, 9, 6), staging_dir=tmp_path)

    assert load_stations_from_archive(tmp_path, system_id=SYSTEM_ID).columns == [
        "station_id",
        "capacity",
    ]


def test_system_ids_in_archive_reads_the_catalog(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    export_station_catalog(engine_with_rows, day=date(2026, 9, 6), staging_dir=tmp_path)

    assert system_ids_in_archive(tmp_path) == [SYSTEM_ID]


def test_catalog_snapshots_accumulate_and_capture_capacity_changes(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """The whole point of dating the catalog: Postgres upserts capacity in
    place, so a change is invisible there the moment it happens. Snapshots
    are the only record that it ever changed."""

    export_station_catalog(engine_with_rows, day=date(2026, 9, 6), staging_dir=tmp_path)

    factory = sessionmaker(bind=engine_with_rows, expire_on_commit=False)
    session: Session = factory()
    station = session.get(Station, (SYSTEM_ID, "a"))
    assert station is not None
    station.capacity = 30
    session.commit()
    session.close()

    export_station_catalog(engine_with_rows, day=date(2026, 9, 7), staging_dir=tmp_path)

    history = capacity_history(tmp_path, system_id=SYSTEM_ID)
    assert history["as_of"].to_list() == [date(2026, 9, 6), date(2026, 9, 7)]
    assert history["capacity"].to_list() == [20, 30]


def test_features_use_the_most_recent_catalog(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """Documents current behaviour rather than endorsing it: every row gets
    the latest capacity, matching what reading Postgres does. An as-of join
    is a later change, once there is history worth joining against."""

    export_station_catalog(engine_with_rows, day=date(2026, 9, 6), staging_dir=tmp_path)

    factory = sessionmaker(bind=engine_with_rows, expire_on_commit=False)
    session: Session = factory()
    station = session.get(Station, (SYSTEM_ID, "a"))
    assert station is not None
    station.capacity = 30
    session.commit()
    session.close()

    export_station_catalog(engine_with_rows, day=date(2026, 9, 7), staging_dir=tmp_path)

    assert load_stations_from_archive(tmp_path, system_id=SYSTEM_ID).to_dicts() == [
        {"station_id": "a", "capacity": 30}
    ]
