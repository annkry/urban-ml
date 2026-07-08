from __future__ import annotations

from datetime import datetime

from urban_ml.domain.station_snapshot import StationSnapshot
from urban_ml.ingestion.gbfs_client import GbfsRawStationFeeds
from urban_ml.schemas.gbfs import LocalizedString, StationInformationStation


class GbfsTransformError(Exception):
    """Raised when validated GBFS feeds cannot be transformed safely."""


def build_station_snapshots(
    raw_feeds: GbfsRawStationFeeds,
    *,
    system_id: str,
    observed_at: datetime,
) -> list[StationSnapshot]:
    """Join station metadata and live status into flat station snapshots.

    The GBFS API separates relatively static station metadata
    (`station_information`) from live availability (`station_status`). For ML,
    we want one flat row per station per ingestion timestamp.
    """

    station_information_by_id = {
        station.station_id: station
        for station in raw_feeds.station_information.data.stations
    }

    snapshots: list[StationSnapshot] = []
    missing_station_ids: list[str] = []

    for status in raw_feeds.station_status.data.stations:
        station_information = station_information_by_id.get(status.station_id)
        if station_information is None:
            missing_station_ids.append(status.station_id)
            continue

        snapshots.append(
            StationSnapshot(
                observed_at=observed_at,
                system_id=system_id,
                station_id=status.station_id,
                station_name=_station_name(station_information),
                lat=station_information.lat,
                lon=station_information.lon,
                capacity=station_information.capacity,
                num_vehicles_available=status.num_vehicles_available,
                num_docks_available=status.num_docks_available,
                is_installed=status.is_installed,
                is_renting=status.is_renting,
                is_returning=status.is_returning,
                last_reported=status.last_reported,
            )
        )

    if missing_station_ids:
        preview = ", ".join(sorted(missing_station_ids)[:5])
        raise GbfsTransformError(
            "Station status contains station IDs missing from "
            f"station_information: {preview}"
        )

    return snapshots


def _station_name(station: StationInformationStation) -> str:
    return _preferred_localized_text(station.name)


def _preferred_localized_text(values: list[LocalizedString]) -> str:
    if not values:
        raise GbfsTransformError("Station has no localized name values")

    for value in values:
        if value.language.lower() == "en":
            return value.text

    return values[0].text
