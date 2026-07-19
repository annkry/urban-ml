from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field


class GbfsModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class FeedName(StrEnum):
    GBFS = "gbfs"
    GBFS_VERSIONS = "gbfs_versions"
    GEOFENCING_ZONES = "geofencing_zones"
    STATION_INFORMATION = "station_information"
    STATION_STATUS = "station_status"
    SYSTEM_ALERTS = "system_alerts"
    SYSTEM_INFORMATION = "system_information"
    SYSTEM_PRICING_PLANS = "system_pricing_plans"
    SYSTEM_REGIONS = "system_regions"
    VEHICLE_STATUS = "vehicle_status"
    VEHICLE_TYPES = "vehicle_types"


class RentalMethod(StrEnum):
    ACCOUNT_NUMBER = "accountnumber"
    ANDROID_PAY = "androidpay"
    APPLE_PAY = "applepay"
    CREDIT_CARD = "creditcard"
    KEY = "key"
    PAYPASS = "paypass"
    PHONE = "phone"
    TRANSIT_CARD = "transitcard"


class ParkingType(StrEnum):
    OTHER = "other"
    PARKING_LOT = "parking_lot"
    SIDEWALK_PARKING = "sidewalk_parking"
    STREET_PARKING = "street_parking"
    UNDERGROUND_PARKING = "underground_parking"


class FormFactor(StrEnum):
    BICYCLE = "bicycle"
    CARGO_BICYCLE = "cargo_bicycle"
    CAR = "car"
    MOPED = "moped"
    SCOOTER_STANDING = "scooter_standing"
    SCOOTER_SEATED = "scooter_seated"
    OTHER = "other"


class PropulsionType(StrEnum):
    HUMAN = "human"
    ELECTRIC_ASSIST = "electric_assist"
    ELECTRIC = "electric"
    COMBUSTION = "combustion"
    COMBUSTION_DIESEL = "combustion_diesel"
    HYBRID = "hybrid"
    PLUG_IN_HYBRID = "plug_in_hybrid"
    HYDROGEN_FUEL_CELL = "hydrogen_fuel_cell"


class LocalizedString(GbfsModel):
    text: str
    language: str


class RentalUris(GbfsModel):
    android: str | None = None
    ios: str | None = None
    web: str | None = None


class GeoJsonMultiPolygon(GbfsModel):
    type: Literal["MultiPolygon"]
    coordinates: list[list[list[list[float]]]]


class VehicleTypesCapacity(GbfsModel):
    vehicle_type_ids: list[str]
    count: int = Field(ge=0)


class VehicleDocksCapacity(GbfsModel):
    vehicle_type_ids: list[str]
    count: int = Field(ge=0)


class VehicleTypesAvailable(GbfsModel):
    vehicle_type_id: str
    count: int = Field(ge=0)


class VehicleDocksAvailable(GbfsModel):
    vehicle_type_ids: list[str]
    count: int = Field(ge=0)


class GbfsFeed(GbfsModel):
    name: FeedName
    url: AnyUrl


class GbfsDiscoveryData(GbfsModel):
    feeds: list[GbfsFeed]


class GbfsDiscoveryResponse(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: Literal["3.0"]
    data: GbfsDiscoveryData


class StationInformationStation(GbfsModel):
    station_id: str
    name: list[LocalizedString]
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    short_name: list[LocalizedString] | None = None
    address: str | None = None
    cross_street: str | None = None
    region_id: str | None = None
    post_code: str | None = None
    station_opening_hours: str | None = None
    rental_methods: list[RentalMethod] | None = None
    is_virtual_station: bool | None = None
    station_area: GeoJsonMultiPolygon | None = None
    parking_type: ParkingType | None = None
    parking_hoop: bool | None = None
    contact_phone: str | None = None
    capacity: int | None = Field(default=None, ge=0)
    vehicle_types_capacity: list[VehicleTypesCapacity] | None = None
    vehicle_docks_capacity: list[VehicleDocksCapacity] | None = None
    is_valet_station: bool | None = None
    is_charging_station: bool | None = None
    rental_uris: RentalUris | None = None


class StationInformationData(GbfsModel):
    stations: list[StationInformationStation]


class StationInformationResponse(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: Literal["3.0"]
    data: StationInformationData


class StationStatusStation(GbfsModel):
    station_id: str
    num_vehicles_available: int = Field(ge=0)
    vehicle_types_available: list[VehicleTypesAvailable] | None = None
    num_vehicles_disabled: int | None = Field(default=None, ge=0)
    num_docks_available: int | None = Field(default=None, ge=0)
    vehicle_docks_available: list[VehicleDocksAvailable] | None = None
    num_docks_disabled: int | None = Field(default=None, ge=0)
    is_installed: bool
    is_renting: bool
    is_returning: bool
    last_reported: datetime


class StationStatusData(GbfsModel):
    stations: list[StationStatusStation]


class StationStatusResponse(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: Literal["3.0"]
    data: StationStatusData


class SystemInformationData(GbfsModel):
    system_id: str


class SystemInformationResponse(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: Literal["3.0"]
    data: SystemInformationData


class VehicleType(GbfsModel):
    vehicle_type_id: str
    form_factor: FormFactor
    propulsion_type: PropulsionType
    name: list[LocalizedString] | None = None


class VehicleTypesData(GbfsModel):
    vehicle_types: list[VehicleType]


class VehicleTypesResponse(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: Literal["3.0"]
    data: VehicleTypesData
