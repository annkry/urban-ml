from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from urban_ml.archive.export import days_present
from urban_ml.archive.layout import STATION_STATUS
from urban_ml.archive.retention import days_safe_to_trim, trim_days
from urban_ml.storage.models import Base, StationStatusRecord

SYSTEM_ID = "toronto"
TODAY = date(2026, 9, 8)


@pytest.fixture
def engine_with_days():  # type: ignore[no-untyped-def]
    """One station_status row per hour across 2026-09-04 .. 2026-09-07."""

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    session: Session = factory()
    session.add_all(
        [
            StationStatusRecord(
                observed_at=datetime(2026, 9, day, hour, tzinfo=UTC),
                system_id=SYSTEM_ID,
                station_id="a",
                num_vehicles_available=5,
                num_docks_available=15,
                is_installed=True,
                is_renting=True,
                is_returning=True,
            )
            for day in (4, 5, 6, 7)
            for hour in (0, 12)
        ]
    )
    session.commit()
    session.close()
    yield engine
    engine.dispose()


def _row_count(engine) -> int:  # type: ignore[no-untyped-def]
    with engine.connect() as connection:
        return connection.execute(
            select(func.count()).select_from(StationStatusRecord)
        ).scalar_one()


def test_an_unarchived_day_is_never_trimmed_however_old() -> None:
    """The safety property: no archive copy, no deletion."""

    present = {date(2026, 1, 1), date(2026, 9, 4)}
    assert days_safe_to_trim(present, set(), today=TODAY, keep_days=1) == []


def test_only_days_the_archive_holds_are_trimmed() -> None:
    present = {date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)}
    archived = {date(2026, 9, 1), date(2026, 9, 3)}

    trimmed = days_safe_to_trim(present, archived, today=TODAY, keep_days=1)

    assert trimmed == [date(2026, 9, 1), date(2026, 9, 3)]


def test_days_inside_the_window_are_kept() -> None:
    """keep_days=5 on 2026-09-08 retains 09-03 onwards."""

    present = archived = {date(2026, 9, day) for day in (2, 3, 4, 7)}

    trimmed = days_safe_to_trim(present, archived, today=TODAY, keep_days=5)

    assert trimmed == [date(2026, 9, 2)]


def test_zero_keep_days_still_spares_today() -> None:
    present = archived = {date(2026, 9, 7), TODAY}

    assert days_safe_to_trim(present, archived, today=TODAY, keep_days=0) == [
        date(2026, 9, 7)
    ]


def test_a_negative_window_is_rejected_rather_than_deleting_the_future() -> None:
    with pytest.raises(ValueError, match="keep_days"):
        days_safe_to_trim(set(), set(), today=TODAY, keep_days=-1)


def test_trim_removes_only_the_named_days(engine_with_days) -> None:  # type: ignore[no-untyped-def]
    removed = trim_days(engine_with_days, days=[date(2026, 9, 4), date(2026, 9, 5)])

    assert removed == 4
    assert days_present(engine_with_days, table=STATION_STATUS) == {
        date(2026, 9, 6),
        date(2026, 9, 7),
    }
    assert _row_count(engine_with_days) == 4


def test_trimming_nothing_touches_nothing(engine_with_days) -> None:  # type: ignore[no-untyped-def]
    assert trim_days(engine_with_days, days=[]) == 0
    assert _row_count(engine_with_days) == 8


def test_trimming_a_day_twice_is_harmless(engine_with_days) -> None:  # type: ignore[no-untyped-def]
    """Retention reruns after a partial failure without needing a cursor."""

    assert trim_days(engine_with_days, days=[date(2026, 9, 4)]) == 2
    assert trim_days(engine_with_days, days=[date(2026, 9, 4)]) == 0
    assert _row_count(engine_with_days) == 6
