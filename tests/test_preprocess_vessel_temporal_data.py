import pandas as pd
from datetime import time
from assignment_1.location_anomalies import preprocess_vessel_temporal_data


def test_basic_time_binning():
    """Test basic time binning functionality"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": ["01/01/2023 12:30:15", "01/01/2023 12:45:30"],
            "Latitude": [50.0, 50.0],
            "Longitude": [10.0, 10.0],
        }
    )

    result = preprocess_vessel_temporal_data(data)

    # Check time bins are correct (should be floored to the minute)
    assert (
        result[result["MMSI"] == 1].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "12:30:00"
    )
    assert (
        result[result["MMSI"] == 2].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "12:45:00"
    )


def test_boundary_time_values():
    """Test behavior with timestamps near time bin boundaries"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": [
                "01/01/2023 12:00:55",
                "01/01/2023 12:29:55",
            ],  # 5 seconds before next minute
            "Latitude": [50.0, 50.0],
            "Longitude": [10.0, 10.0],
        }
    )

    result = preprocess_vessel_temporal_data(data)

    # With default 10 second threshold, these should appear in both bins
    mmsi_counts = result["MMSI"].value_counts()
    assert mmsi_counts[1] == 2  # Should be in two bins
    assert mmsi_counts[2] == 2  # Should be in two bins

    # Check both primary and next bins are present
    bins_for_mmsi_1 = set(
        row["time_bin"].strftime("%H:%M:%S")
        for _, row in result[result["MMSI"] == 1].iterrows()
    )
    assert "12:00:00" in bins_for_mmsi_1
    assert "12:01:00" in bins_for_mmsi_1


def test_disable_time_overlap():
    """Test behavior when overlap is disabled"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2, 3, 4],
            "Timestamp": [
                "01/01/2023 12:00:55",  # 5 sec before next min
                "01/01/2023 12:15:55",  # 5 sec before next min
                "01/01/2023 12:30:05",  # 5 sec after min
                "01/01/2023 12:45:05",  # 5 sec after min
            ],
            "Latitude": [50.0, 50.0, 50.0, 50.0],
            "Longitude": [10.0, 10.0, 10.0, 10.0],
        }
    )

    # Without overlap, each point should be in exactly one bin
    result = preprocess_vessel_temporal_data(data, overlap=False)

    assert len(result) == len(
        data
    )  # Should have the same number of rows as original data

    # Check bin assignments (should be floored to the minute)
    assert (
        result[result["MMSI"] == 1].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "12:00:00"
    )
    assert (
        result[result["MMSI"] == 2].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "12:15:00"
    )
    assert (
        result[result["MMSI"] == 3].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "12:30:00"
    )
    assert (
        result[result["MMSI"] == 4].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "12:45:00"
    )


def test_custom_time_bin_sizes():
    """Test behavior with custom time bin sizes"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:45:30"],
            "Latitude": [50.0],
            "Longitude": [10.0],
        }
    )

    # Test with different bin sizes
    result_1min = preprocess_vessel_temporal_data(data, time_bin_size="1min")
    result_15min = preprocess_vessel_temporal_data(data, time_bin_size="15min")
    result_1hour = preprocess_vessel_temporal_data(data, time_bin_size="1h")

    # 1min should bin to 12:45:00
    assert result_1min.iloc[0]["time_bin"].strftime("%H:%M:%S") == "12:45:00"

    # 15min should bin to 12:45:00 (since 12:45 falls on a 15-min boundary)
    assert result_15min.iloc[0]["time_bin"].strftime("%H:%M:%S") == "12:45:00"

    # 1hour should bin to 12:00:00
    assert result_1hour.iloc[0]["time_bin"].strftime("%H:%M:%S") == "12:00:00"


def test_boundary_threshold_zero():
    """Test behavior when boundary_threshold is set to 0"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:59:59"],  # 1 second before next hour
            "Latitude": [50.0],
            "Longitude": [10.0],
        }
    )

    # With threshold of 0, no points should be considered near boundary
    result = preprocess_vessel_temporal_data(data, time_boundary_threshold=0)

    # Should only be in one bin
    assert len(result) == 1
    assert result.iloc[0]["time_bin"].strftime("%H:%M:%S") == "12:59:00"


def test_boundary_threshold_large():
    """Test behavior when boundary_threshold is set very large (30 seconds)"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:30:35"],  # 25 seconds after the minute
            "Latitude": [50.0],
            "Longitude": [10.0],
        }
    )

    # With threshold of 30, this point should be considered near boundary to next minute
    result = preprocess_vessel_temporal_data(data, time_boundary_threshold=30)

    # Should be in both bins
    assert len(result) == 2

    # Check both bins are present
    bins = set(row["time_bin"].strftime("%H:%M") for _, row in result.iterrows())
    assert "12:30" in bins
    assert "12:31" in bins


def test_midnight_crossing():
    """Test behavior with timestamps near midnight"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": [
                "01/01/2023 23:59:55",
                "01/01/2023 00:00:05",
            ],  # Near midnight
            "Latitude": [50.0, 50.0],
            "Longitude": [10.0, 10.0],
        }
    )

    result = preprocess_vessel_temporal_data(data)

    # First point should be in 23:59 bin only (since we're assuming same-day data)
    bins_for_mmsi_1 = set(
        row["time_bin"].strftime("%H:%M:%S")
        for _, row in result[result["MMSI"] == 1].iterrows()
    )
    assert "23:59:00" in bins_for_mmsi_1
    # Should not have any bin in the next day
    assert not any("00:00:00" in bin_time for bin_time in bins_for_mmsi_1)

    # Second point should be in 00:00 bin
    assert (
        result[result["MMSI"] == 2].iloc[0]["time_bin"].strftime("%H:%M:%S")
        == "00:00:00"
    )


def test_duplicate_timestamps():
    """Test behavior with duplicate timestamps but different vessels"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": ["01/01/2023 12:30:00", "01/01/2023 12:30:00"],  # Same time
            "Latitude": [50.0, 51.0],  # Different locations
            "Longitude": [10.0, 11.0],
        }
    )

    result = preprocess_vessel_temporal_data(data)

    # Both points should be processed independently
    assert len(result) == 2

    # Check MMSI values are preserved
    assert set(result["MMSI"]) == {1, 2}

    # Check coordinates are preserved
    latitudes = set(result["Latitude"])
    assert len(latitudes) == 2
    assert 50.0 in latitudes
    assert 51.0 in latitudes
