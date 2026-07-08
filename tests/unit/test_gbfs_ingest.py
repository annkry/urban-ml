from urban_ml.ingestion.gbfs_ingest import (
    GbfsIngestionSummary,
    format_ingestion_summary,
)


def test_format_ingestion_summary_includes_counts() -> None:
    summary = GbfsIngestionSummary(
        discovery_url="https://example.com/gbfs/3/gbfs",
        station_information_last_updated="2023-07-17T13:34:13+02:00",
        station_status_last_updated="2023-07-17T13:35:13+02:00",
        station_count=2,
        status_count=3,
        matched_station_status_count=2,
        raw_snapshot_dir="/tmp/gbfs_snapshot_2023-07-17T13:34:13+02:00",
        processed_snapshot_dir="/tmp/gbfs_processed_2023-07-17T13:35:13+02:00",
        processed_station_snapshot_count=2,
    )

    formatted = format_ingestion_summary(summary)

    assert "GBFS ingestion completed" in formatted
    assert "Stations discovered: 2" in formatted
    assert "Station statuses discovered: 3" in formatted
    assert "Stations with matching status: 2" in formatted
