from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from urban_ml.domain.station import TRACKED_STATION_FIELDS
from urban_ml.storage.models import Station, StationStatusRecord
from urban_ml.storage.repository import (
    fetch_recent_station_status,
    get_station,
    latest_stations,
    station_status_history_query,
)

OBSERVED_AT = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _station(**overrides: object) -> Station:
    defaults: dict[str, object] = dict(
        system_id="toronto",
        station_id="a",
        station_name="Alpha",
        address="1 Main St",
        lat=43.6,
        lon=-79.4,
        capacity=20,
        is_charging_station=False,
        observed_at=OBSERVED_AT,
    )
    defaults.update(overrides)
    return Station(**defaults)


def _status(**overrides: object) -> StationStatusRecord:
    defaults: dict[str, object] = dict(
        observed_at=OBSERVED_AT,
        system_id="toronto",
        station_id="station-1",
        num_vehicles_available=7,
        num_docks_available=13,
        is_installed=True,
        is_renting=True,
        is_returning=True,
    )
    defaults.update(overrides)
    return StationStatusRecord(**defaults)


def test_get_station_returns_none_when_missing(session: Session) -> None:
    assert get_station(session, system_id="toronto", station_id="missing") is None


def test_fetch_recent_station_status_filters_by_station_and_since(
    session: Session,
) -> None:
    session.add_all([_status(), _status(station_id="station-2")])
    session.commit()

    since = OBSERVED_AT - timedelta(minutes=1)
    rows = fetch_recent_station_status(
        session, system_id="toronto", station_id="station-1", since=since
    )
    assert len(rows) == 1
    assert rows[0].station_id == "station-1"

    too_recent = OBSERVED_AT + timedelta(minutes=1)
    assert (
        fetch_recent_station_status(
            session, system_id="toronto", station_id="station-1", since=too_recent
        )
        == []
    )


def test_station_status_history_query_selects_expected_columns(
    session: Session,
) -> None:
    session.add(_status())
    session.commit()

    query = station_status_history_query(system_id="toronto")
    rows = session.execute(query).all()

    assert len(rows) == 1
    row = rows[0]
    assert row.station_id == "station-1"
    assert row.num_vehicles_available == 7
    assert row.is_renting is True


def test_latest_stations_returns_only_the_newest_row(session: Session) -> None:
    """`stations` is a change log, so "current" is the newest row per
    station, not the only row."""

    session.add_all(
        [
            _station(),
            _station(capacity=30, observed_at=OBSERVED_AT + timedelta(days=1)),
        ]
    )
    session.commit()

    current = latest_stations(session, system_id="toronto")

    assert len(current) == 1
    assert current[0].capacity == 30


def test_get_station_reads_through_the_change_log(session: Session) -> None:
    session.add_all(
        [
            _station(),
            _station(capacity=44, observed_at=OBSERVED_AT + timedelta(days=1)),
        ]
    )
    session.commit()

    station = get_station(session, system_id="toronto", station_id="a")

    assert station is not None
    assert station.capacity == 44


def test_every_station_column_is_either_tracked_or_deliberately_excluded() -> None:
    """The archive collapses each day's station snapshots to distinct states
    by comparing TRACKED_STATION_FIELDS. A column added to the model without
    a decision here would silently never count as a change."""

    columns = {c.name for c in Station.__table__.columns}
    identity_and_bookkeeping = {"id", "system_id", "station_id", "observed_at"}

    assert columns - identity_and_bookkeeping == set(TRACKED_STATION_FIELDS)
