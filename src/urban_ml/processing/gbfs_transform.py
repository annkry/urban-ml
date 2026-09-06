from __future__ import annotations

from datetime import datetime

from urban_ml.domain.station import Station
from urban_ml.domain.station_status import StationStatus
from urban_ml.domain.vehicle_type import VehicleType
from urban_ml.ingestion.gbfs_client import GbfsRawFeeds
from urban_ml.schemas.gbfs import (
    LocalizedString,
    StationInformationStation,
    StationStatusStation,
)

_ELECTRIC_PROPULSION = {"electric", "electric_assist"}


class GbfsTransformError(Exception):
    """Raised when validated GBFS feeds cannot be transformed safely."""


def build_stations(
    raw_feeds: GbfsRawFeeds,
    *,
    system_id: str,
    observed_at: datetime,
) -> list[Station]:
    """Extract station details from station_information, as of observed_at."""

    return [
        Station(
            system_id=system_id,
            station_id=station.station_id,
            station_name=_station_name(station),
            address=station.address,
            lat=station.lat,
            lon=station.lon,
            capacity=station.capacity,
            is_charging_station=station.is_charging_station,
            observed_at=observed_at,
        )
        for station in raw_feeds.station_information.data.stations
    ]


def build_vehicle_types(
    raw_feeds: GbfsRawFeeds,
    *,
    system_id: str,
) -> list[VehicleType]:
    """Extract vehicle type metadata (dimension data) from vehicle_types."""

    return [
        VehicleType(
            system_id=system_id,
            vehicle_type_id=vehicle_type.vehicle_type_id,
            form_factor=vehicle_type.form_factor,
            propulsion_type=vehicle_type.propulsion_type,
            name=_optional_localized_text(vehicle_type.name),
        )
        for vehicle_type in raw_feeds.vehicle_types.data.vehicle_types
    ]


def _electric_vehicle_type_ids(raw_feeds: GbfsRawFeeds) -> set[str]:
    return {
        vehicle_type.vehicle_type_id
        for vehicle_type in raw_feeds.vehicle_types.data.vehicle_types
        if vehicle_type.propulsion_type in _ELECTRIC_PROPULSION
    }


def _split_by_propulsion(
    status: StationStatusStation, electric_type_ids: set[str], known_type_ids: set[str]
) -> tuple[int | None, int | None, list[str]]:
    """Split a station's vehicles into electric and human-powered.

    Returns (electric, human, unknown_type_ids). Both counts are None when the
    feed reports no per-type breakdown at all, which is different from a
    genuine zero.
    """

    entries = status.vehicle_types_available
    if entries is None:
        return None, None, []

    unknown = [
        e.vehicle_type_id for e in entries if e.vehicle_type_id not in known_type_ids
    ]
    electric = sum(e.count for e in entries if e.vehicle_type_id in electric_type_ids)
    human = sum(
        e.count
        for e in entries
        if e.vehicle_type_id in known_type_ids
        and e.vehicle_type_id not in electric_type_ids
    )
    return electric, human, unknown


def build_station_status(
    raw_feeds: GbfsRawFeeds,
    *,
    system_id: str,
    known_station_ids: set[str],
    observed_at: datetime,
) -> list[StationStatus]:
    """Build one status row per station, flattened from station_status."""

    known_type_ids = {
        vehicle_type.vehicle_type_id
        for vehicle_type in raw_feeds.vehicle_types.data.vehicle_types
    }
    electric_type_ids = _electric_vehicle_type_ids(raw_feeds)

    records: list[StationStatus] = []
    missing_station_ids: list[str] = []
    unknown_vehicle_type_ids: list[str] = []

    for status in raw_feeds.station_status.data.stations:
        if status.station_id not in known_station_ids:
            missing_station_ids.append(status.station_id)
            continue

        electric, human, unknown = _split_by_propulsion(
            status, electric_type_ids, known_type_ids
        )
        unknown_vehicle_type_ids.extend(unknown)

        records.append(
            StationStatus(
                observed_at=observed_at,
                system_id=system_id,
                station_id=status.station_id,
                num_vehicles_available=status.num_vehicles_available,
                num_vehicles_disabled=status.num_vehicles_disabled,
                num_docks_available=status.num_docks_available,
                num_docks_disabled=status.num_docks_disabled,
                is_installed=status.is_installed,
                is_renting=status.is_renting,
                is_returning=status.is_returning,
                num_vehicles_electric=electric,
                num_vehicles_human=human,
            )
        )

    if missing_station_ids:
        preview = ", ".join(sorted(missing_station_ids)[:5])
        raise GbfsTransformError(
            "Station status contains station IDs missing from "
            f"station_information: {preview}"
        )

    if unknown_vehicle_type_ids:
        preview = ", ".join(sorted(set(unknown_vehicle_type_ids))[:5])
        raise GbfsTransformError(
            "Station status references vehicle type IDs missing from "
            f"vehicle_types: {preview}"
        )

    return records


def _station_name(station: StationInformationStation) -> str:
    return _preferred_localized_text(station.name)


def _preferred_localized_text(values: list[LocalizedString]) -> str:
    if not values:
        raise GbfsTransformError("Station has no localized name values")

    for value in values:
        if value.language.lower() == "en":
            return value.text

    return values[0].text


def _optional_localized_text(values: list[LocalizedString] | None) -> str | None:
    if not values:
        return None

    for value in values:
        if value.language.lower() == "en":
            return value.text

    return values[0].text
