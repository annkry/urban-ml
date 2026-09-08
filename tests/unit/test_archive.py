from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from urban_ml.archive.export import (
    _utc_date,
    day_bounds,
    days_needing_export,
    days_present,
    export_day,
    observed_day_range,
    utc_day,
)
from urban_ml.archive.layout import (
    STATION_STATUS,
    STATIONS,
    partition_path,
)
from urban_ml.archive.read import (
    load_raw_status_from_archive,
)
from urban_ml.storage.models import Base, Station, StationStatusRecord

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
    session.add_all(
        [
            Station(
                system_id=SYSTEM_ID,
                station_id="a",
                station_name="Alpha",
                lat=43.6,
                lon=-79.4,
                capacity=20,
                observed_at=datetime(2026, 9, 4, tzinfo=UTC),
            ),
            Station(
                system_id=SYSTEM_ID,
                station_id="a",
                station_name="Alpha",
                lat=43.6,
                lon=-79.4,
                capacity=30,
                observed_at=datetime(2026, 9, 6, tzinfo=UTC),
            ),
        ]
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


def test_partitions_share_one_schema_even_when_a_column_is_all_null(
    engine_with_rows,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """A column that is entirely NULL on one day would otherwise be written as
    Null there and as its real type elsewhere, and a scan across the archive
    fails with a dtype mismatch. Columns added by a migration are all-NULL for
    every earlier day, so this is the normal case."""

    for day in (date(2026, 9, 4), date(2026, 9, 6)):
        export_day(engine_with_rows, table=STATIONS, day=day, staging_dir=tmp_path)
        export_day(
            engine_with_rows, table=STATION_STATUS, day=day, staging_dir=tmp_path
        )

    for table in (STATIONS, STATION_STATUS):
        schemas = {
            tuple(pl.read_parquet_schema(f).items())
            for f in sorted((tmp_path / table).glob("date=*/*.parquet"))
        }
        assert len(schemas) == 1, f"{table} partitions disagree on schema"

    assert pl.read_parquet(tmp_path / STATIONS / "**" / "*.parquet").height > 0


TODAY = date(2026, 9, 6)


def test_a_failed_day_is_retried_on_the_next_run() -> None:
    """The rule that stops one failed job becoming permanent data loss:
    exporting only 'yesterday' never retries, so a missed day is gone once
    retention trims it out of the database."""

    present = {date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 5)}
    already = {date(2026, 9, 3), date(2026, 9, 5)}

    assert days_needing_export(present, already, today=TODAY) == [date(2026, 9, 4)]


def test_today_is_never_exported_because_it_is_still_being_written() -> None:
    assert days_needing_export({date(2026, 9, 5), TODAY}, set(), today=TODAY) == [
        date(2026, 9, 5)
    ]


def test_a_fully_published_archive_exports_nothing() -> None:
    present = {date(2026, 9, 4), date(2026, 9, 5)}

    assert days_needing_export(present, present, today=TODAY) == []


def test_days_present_reports_the_days_holding_rows(engine_with_rows) -> None:  # type: ignore[no-untyped-def]
    assert days_present(engine_with_rows, table=STATION_STATUS) == {
        date(2026, 9, 4),
        date(2026, 9, 6),
    }


def test_nothing_before_the_start_date_is_ever_archived() -> None:
    """The archive was reset to begin at cloud-ingestion cutover. Without a
    floor, an empty archive plus a database full of older rows reads as
    "every day is missing" and re-uploads exactly what was removed."""

    present = {date(2026, 7, 19), date(2026, 9, 4), date(2026, 9, 5)}

    assert days_needing_export(present, set(), today=TODAY, start=date(2026, 9, 4)) == [
        date(2026, 9, 4),
        date(2026, 9, 5),
    ]


def test_no_start_date_means_no_floor() -> None:
    present = {date(2026, 9, 4), date(2026, 9, 5)}

    assert days_needing_export(present, set(), today=TODAY, start=None) == [
        date(2026, 9, 4),
        date(2026, 9, 5),
    ]


def test_day_bucketing_is_utc_regardless_of_postgres_session_timezone() -> None:
    """Postgres reads date(timestamptz) in the session timezone."""

    statement = select(utc_day(StationStatusRecord.observed_at, dialect="postgresql"))
    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )

    assert "timezone('UTC', station_status.observed_at)" in compiled


def test_day_bucketing_stays_plain_on_dialects_without_a_session_timezone() -> None:
    statement = select(utc_day(StationStatusRecord.observed_at, dialect="sqlite"))
    compiled = str(statement.compile(dialect=sqlite.dialect()))

    assert "timezone" not in compiled
    assert "date(" in compiled


def test_observed_day_range_is_utc_whatever_zone_the_driver_returns() -> None:
    """psycopg renders timestamptz in the session zone, so .date() on what it
    hands back is the local day, not the UTC one."""

    berlin = timezone(timedelta(hours=2))
    late_evening_utc = datetime(2026, 9, 6, 23, 30, tzinfo=UTC)

    assert _utc_date(late_evening_utc.astimezone(berlin)) == date(2026, 9, 6)
    assert _utc_date(late_evening_utc) == date(2026, 9, 6)
