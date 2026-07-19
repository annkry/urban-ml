import pytest

from urban_ml.ingestion.gbfs_client import (
    GbfsClient,
    GbfsFeedNotFoundError,
    GbfsFetchError,
    GbfsValidationError,
    RetryPolicy,
    fetch_json_with_retry,
)
from urban_ml.schemas.gbfs import FeedName, GbfsDiscoveryResponse


DISCOVERY_URL = "https://example.com/gbfs/3/gbfs"
SYSTEM_INFORMATION_URL = "https://example.com/gbfs/3/system_information"
VEHICLE_TYPES_URL = "https://example.com/gbfs/3/vehicle_types"
STATION_INFORMATION_URL = "https://example.com/gbfs/3/station_information"
STATION_STATUS_URL = "https://example.com/gbfs/3/station_status"


def _discovery_payload(feed_names: set[str]) -> dict:
    all_feeds = {
        "system_information": SYSTEM_INFORMATION_URL,
        "vehicle_types": VEHICLE_TYPES_URL,
        "station_information": STATION_INFORMATION_URL,
        "station_status": STATION_STATUS_URL,
    }
    return {
        "last_updated": "2023-07-17T13:34:13+02:00",
        "ttl": 0,
        "version": "3.0",
        "data": {
            "feeds": [
                {"name": name, "url": all_feeds[name]}
                for name in feed_names
                if name in all_feeds
            ]
        },
    }


def _valid_payloads() -> dict[str, dict]:
    return {
        DISCOVERY_URL: _discovery_payload(
            {
                "system_information",
                "vehicle_types",
                "station_information",
                "station_status",
            }
        ),
        SYSTEM_INFORMATION_URL: {
            "last_updated": "2023-07-17T13:34:13+02:00",
            "ttl": 60,
            "version": "3.0",
            "data": {"system_id": "example_system"},
        },
        VEHICLE_TYPES_URL: {
            "last_updated": "2023-07-17T13:34:13+02:00",
            "ttl": 60,
            "version": "3.0",
            "data": {
                "vehicle_types": [
                    {
                        "vehicle_type_id": "classic",
                        "form_factor": "bicycle",
                        "propulsion_type": "human",
                    }
                ]
            },
        },
        STATION_INFORMATION_URL: {
            "last_updated": "2023-07-17T13:34:13+02:00",
            "ttl": 60,
            "version": "3.0",
            "data": {
                "stations": [
                    {
                        "station_id": "station-1",
                        "name": [{"text": "Main Station", "language": "en"}],
                        "lat": 47.3769,
                        "lon": 8.5417,
                        "capacity": 20,
                    }
                ]
            },
        },
        STATION_STATUS_URL: {
            "last_updated": "2023-07-17T13:34:13+02:00",
            "ttl": 60,
            "version": "3.0",
            "data": {
                "stations": [
                    {
                        "station_id": "station-1",
                        "num_vehicles_available": 7,
                        "num_docks_available": 13,
                        "is_installed": True,
                        "is_renting": True,
                        "is_returning": True,
                        "last_reported": "2023-07-17T13:34:13+02:00",
                    }
                ]
            },
        },
    }


def test_fetch_raw_feeds_discovers_urls_and_validates_all_payloads() -> None:
    payloads = _valid_payloads()
    requested_urls = []

    def fake_fetcher(url: str, timeout_seconds: float) -> dict:
        requested_urls.append(url)
        return payloads[url]

    client = GbfsClient(DISCOVERY_URL, json_fetcher=fake_fetcher)

    raw_feeds = client.fetch_raw_feeds()

    assert requested_urls == [
        DISCOVERY_URL,
        SYSTEM_INFORMATION_URL,
        VEHICLE_TYPES_URL,
        STATION_INFORMATION_URL,
        STATION_STATUS_URL,
    ]
    assert raw_feeds.system_information.data.system_id == "example_system"
    assert raw_feeds.vehicle_types.data.vehicle_types[0].vehicle_type_id == "classic"
    assert raw_feeds.station_information.data.stations[0].station_id == "station-1"
    assert raw_feeds.station_status.data.stations[0].num_vehicles_available == 7


def test_get_feed_url_raises_when_feed_is_not_discovered() -> None:
    discovery = GbfsDiscoveryResponse.model_validate(
        _discovery_payload({"station_information"})
    )
    client = GbfsClient(DISCOVERY_URL)

    with pytest.raises(GbfsFeedNotFoundError):
        client.get_feed_url(FeedName.STATION_STATUS, discovery)


def test_fetch_raw_feeds_wraps_validation_errors() -> None:
    payloads = _valid_payloads()
    payloads[STATION_STATUS_URL] = {
        **payloads[STATION_STATUS_URL],
        "data": {
            "stations": [
                {
                    **payloads[STATION_STATUS_URL]["data"]["stations"][0],
                    "num_vehicles_available": -1,
                }
            ]
        },
    }

    def fake_fetcher(url: str, timeout_seconds: float) -> dict:
        return payloads[url]

    client = GbfsClient(DISCOVERY_URL, json_fetcher=fake_fetcher)

    with pytest.raises(GbfsValidationError):
        client.fetch_raw_feeds()


def test_fetch_json_with_retry_succeeds_after_transient_failures() -> None:
    attempts = []
    sleeps: list[float] = []

    def flaky_fetcher(url: str, timeout_seconds: float) -> dict:
        attempts.append(url)
        if len(attempts) < 3:
            raise GbfsFetchError("transient failure")
        return {"ok": True}

    result = fetch_json_with_retry(
        DISCOVERY_URL,
        10.0,
        fetcher=flaky_fetcher,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.1),
        sleep=sleeps.append,
    )

    assert result == {"ok": True}
    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]


def test_fetch_json_with_retry_raises_last_error_after_max_attempts() -> None:
    attempts = []

    def always_fails(url: str, timeout_seconds: float) -> dict:
        attempts.append(url)
        raise GbfsFetchError(f"failure {len(attempts)}")

    with pytest.raises(GbfsFetchError, match="failure 3"):
        fetch_json_with_retry(
            DISCOVERY_URL,
            10.0,
            fetcher=always_fails,
            retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0),
            sleep=lambda _seconds: None,
        )

    assert len(attempts) == 3


def test_fetch_json_with_retry_does_not_retry_non_fetch_errors() -> None:
    attempts = []

    def buggy_fetcher(url: str, timeout_seconds: float) -> dict:
        attempts.append(url)
        raise ValueError("not a retryable fetch failure")

    with pytest.raises(ValueError, match="not a retryable fetch failure"):
        fetch_json_with_retry(
            DISCOVERY_URL,
            10.0,
            fetcher=buggy_fetcher,
            retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0),
            sleep=lambda _seconds: None,
        )

    assert len(attempts) == 1
