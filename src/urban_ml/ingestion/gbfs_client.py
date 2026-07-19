from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from urban_ml.schemas.gbfs import (
    FeedName,
    GbfsDiscoveryResponse,
    StationInformationResponse,
    StationStatusResponse,
    SystemInformationResponse,
    VehicleTypesResponse,
)

JsonObject = dict[str, Any]
JsonFetcher = Callable[[str, float], JsonObject]


class GbfsClientError(Exception):
    """Base exception for GBFS client failures."""


class GbfsFetchError(GbfsClientError):
    """Raised when a GBFS feed cannot be fetched or decoded."""


class GbfsFeedNotFoundError(GbfsClientError):
    """Raised when a required feed is missing from gbfs.json."""


class GbfsValidationError(GbfsClientError):
    """Raised when a GBFS response does not match the expected schema."""


@dataclass(frozen=True)
class GbfsRawFeeds:
    discovery_url: str
    system_information_url: str
    vehicle_types_url: str
    station_information_url: str
    station_status_url: str
    discovery_payload: JsonObject
    station_information_payload: JsonObject
    station_status_payload: JsonObject
    discovery: GbfsDiscoveryResponse
    system_information: SystemInformationResponse
    vehicle_types: VehicleTypesResponse
    station_information: StationInformationResponse
    station_status: StationStatusResponse


def fetch_json(url: str, timeout_seconds: float) -> JsonObject:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "urban-ml-platform/0.1",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise GbfsFetchError(
            f"GBFS request failed with HTTP {exc.code}: {url}"
        ) from exc
    except URLError as exc:
        raise GbfsFetchError(f"GBFS request failed: {url}") from exc
    except json.JSONDecodeError as exc:
        raise GbfsFetchError(f"GBFS response was not valid JSON: {url}") from exc

    if not isinstance(payload, dict):
        raise GbfsFetchError(f"GBFS response must be a JSON object: {url}")

    return payload


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0


def fetch_json_with_retry(
    url: str,
    timeout_seconds: float,
    *,
    fetcher: JsonFetcher = fetch_json,
    retry_policy: RetryPolicy = RetryPolicy(),
    sleep: Callable[[float], None] = time.sleep,
) -> JsonObject:
    """Fetch JSON, retrying transient failures with exponential backoff."""

    delay = retry_policy.initial_backoff_seconds
    last_error: GbfsFetchError | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            return fetcher(url, timeout_seconds)
        except GbfsFetchError as exc:
            last_error = exc
            if attempt == retry_policy.max_attempts:
                break
            sleep(delay)
            delay *= retry_policy.backoff_multiplier

    assert last_error is not None
    raise last_error


class GbfsClient:
    def __init__(
        self,
        discovery_url: str,
        *,
        timeout_seconds: float = 10.0,
        json_fetcher: JsonFetcher = fetch_json_with_retry,
    ) -> None:
        self.discovery_url = discovery_url
        self.timeout_seconds = timeout_seconds
        self._json_fetcher = json_fetcher

    def get_feed_url(
        self,
        feed_name: FeedName,
        discovery: GbfsDiscoveryResponse,
    ) -> str:
        for feed in discovery.data.feeds:
            if feed.name == feed_name:
                return str(feed.url)

        raise GbfsFeedNotFoundError(
            f"GBFS discovery feed does not include {feed_name.value!r}"
        )

    def fetch_raw_feeds(self) -> GbfsRawFeeds:
        """Fetch discovery, system_information, vehicle_types,
        station_information, and station_status.
        """

        discovery_payload = self._fetch(self.discovery_url)
        discovery = self._validate(
            GbfsDiscoveryResponse,
            discovery_payload,
            self.discovery_url,
        )

        system_information_url = self.get_feed_url(
            FeedName.SYSTEM_INFORMATION,
            discovery,
        )
        vehicle_types_url = self.get_feed_url(FeedName.VEHICLE_TYPES, discovery)
        station_information_url = self.get_feed_url(
            FeedName.STATION_INFORMATION,
            discovery,
        )
        station_status_url = self.get_feed_url(FeedName.STATION_STATUS, discovery)

        system_information_payload = self._fetch(system_information_url)
        vehicle_types_payload = self._fetch(vehicle_types_url)
        station_information_payload = self._fetch(station_information_url)
        station_status_payload = self._fetch(station_status_url)

        system_information = self._validate(
            SystemInformationResponse,
            system_information_payload,
            system_information_url,
        )
        vehicle_types = self._validate(
            VehicleTypesResponse,
            vehicle_types_payload,
            vehicle_types_url,
        )
        station_information = self._validate(
            StationInformationResponse,
            station_information_payload,
            station_information_url,
        )
        station_status = self._validate(
            StationStatusResponse,
            station_status_payload,
            station_status_url,
        )

        return GbfsRawFeeds(
            discovery_url=self.discovery_url,
            system_information_url=system_information_url,
            vehicle_types_url=vehicle_types_url,
            station_information_url=station_information_url,
            station_status_url=station_status_url,
            discovery_payload=discovery_payload,
            station_information_payload=station_information_payload,
            station_status_payload=station_status_payload,
            discovery=discovery,
            system_information=system_information,
            vehicle_types=vehicle_types,
            station_information=station_information,
            station_status=station_status,
        )

    def _fetch(self, url: str) -> JsonObject:
        return self._json_fetcher(url, self.timeout_seconds)

    @staticmethod
    def _validate[T](model: type[T], payload: Mapping[str, Any], url: str) -> T:
        try:
            return model.model_validate(payload)  # type: ignore[attr-defined]
        except ValidationError as exc:
            raise GbfsValidationError(
                f"GBFS response failed validation: {url}"
            ) from exc
