import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from assignment_1.vessel_anomalies import detect_vessel_anomalies


# Fixture for test data
@pytest.fixture
def normal_vessel_data():
    """Create a sample dataframe with normal vessel data (no anomalies)"""
    base_timestamp = datetime(2025, 1, 1, 12, 0, 0)

    return pd.DataFrame(
        {
            "# Timestamp": [
                base_timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                (base_timestamp + timedelta(minutes=15)).strftime("%d/%m/%Y %H:%M:%S"),
                (base_timestamp + timedelta(minutes=30)).strftime("%d/%m/%Y %H:%M:%S"),
                (base_timestamp + timedelta(minutes=45)).strftime("%d/%m/%Y %H:%M:%S"),
            ],
            "MMSI": [123456789, 123456789, 123456789, 123456789],
            "Latitude": [40.0, 40.1, 40.2, 40.3],
            "Longitude": [-74.0, -74.1, -74.2, -74.3],
            "SOG": [10.0, 12.0, 11.0, 10.5],
        }
    )


def test_no_anomalies(normal_vessel_data):
    """Test case with no anomalies"""
    # Process data
    result = detect_vessel_anomalies(normal_vessel_data, return_entire_dataframe=True)

    # Check that no rows are flagged as anomalies
    assert not result["anomaly"].any()

    # Check that the resulting dataframe has expected columns
    expected_columns = [
        "Timestamp",
        "MMSI",
        "Latitude",
        "Longitude",
        "SOG",
        "distance",
        "delta_time_seconds",
        "delta_time_hours",
        "speed",
        "speed_anomaly",
        "ais_gap",
        "anomaly",
    ]
    for col in expected_columns:
        assert col in result.columns

    # Check distance calculation (not testing exact values, but should be reasonable)
    assert (result["distance"][1:] > 0).all()


def test_speed_anomaly(normal_vessel_data):
    """Test case with speed anomalies"""
    # Create data with a speed anomaly
    speed_anomaly_data = normal_vessel_data.copy()
    # Make a large position jump (from 40.1 to 45.0 latitude in 15 minutes)
    speed_anomaly_data.loc[2, "Latitude"] = 45.0

    # Process data
    result = detect_vessel_anomalies(speed_anomaly_data, return_entire_dataframe=True)

    # Check for anomalies
    assert result["speed_anomaly"].any()
    assert result["anomaly"].any()

    # Test with return_entire_dataframe=False
    anomaly_only = detect_vessel_anomalies(
        speed_anomaly_data, return_entire_dataframe=False
    )
    assert len(anomaly_only) > 0
    assert anomaly_only["anomaly"].all()


def test_ais_gap(normal_vessel_data):
    """Test case with AIS time gaps"""
    # Create data with a time gap
    ais_gap_data = normal_vessel_data.copy()

    # Create a large time gap (3 hours) between points 2 and 3
    base_timestamp = datetime.strptime(
        ais_gap_data.loc[0, "# Timestamp"], "%d/%m/%Y %H:%M:%S"
    )
    ais_gap_data.loc[3, "# Timestamp"] = (
        base_timestamp + timedelta(hours=3, minutes=30)
    ).strftime("%d/%m/%Y %H:%M:%S")

    # Process data
    result = detect_vessel_anomalies(ais_gap_data, return_entire_dataframe=True)

    # Check for AIS gap anomaly
    assert result["ais_gap"].any()
    assert result["anomaly"].any()


def test_latitude_longitude_clipping(normal_vessel_data):
    """Test that latitude and longitude values are properly clipped"""
    # Create data with out-of-range coordinates
    invalid_coords_data = normal_vessel_data.copy()
    invalid_coords_data.loc[1, "Latitude"] = 100.0  # Invalid latitude
    invalid_coords_data.loc[2, "Longitude"] = 200.0  # Invalid longitude
    invalid_coords_data.loc[3, "Latitude"] = -100.0  # Invalid latitude

    # Process data
    result = detect_vessel_anomalies(invalid_coords_data, return_entire_dataframe=True)

    # Check that values were clipped
    assert result.loc[1, "Latitude"] == 90.0
    assert result.loc[2, "Longitude"] == 180.0
    assert result.loc[3, "Latitude"] == -90.0


def test_duplicate_removal(normal_vessel_data):
    """Test that duplicate rows are removed"""
    # Create data with duplicate rows
    duplicate_data = pd.concat([normal_vessel_data, normal_vessel_data.iloc[1:2]])

    # Process data
    result = detect_vessel_anomalies(duplicate_data, return_entire_dataframe=True)

    # Check that duplicates were removed
    assert len(result) == len(normal_vessel_data)


def test_custom_thresholds(normal_vessel_data):
    """Test using custom thresholds for anomaly detection"""
    # Set a very low speed threshold to force a speed anomaly
    low_speed_result = detect_vessel_anomalies(
        normal_vessel_data, max_vessel_speed=5.0, return_entire_dataframe=True
    )

    # Should detect speed anomalies with lower threshold
    assert low_speed_result["speed_anomaly"].any()

    # Set a very low time gap threshold to force an AIS gap anomaly
    low_gap_result = detect_vessel_anomalies(
        normal_vessel_data, max_time_gap=0.1, return_entire_dataframe=True
    )

    # Should detect AIS gaps with lower threshold (15 min = 0.25 hr > 0.1 hr)
    assert low_gap_result["ais_gap"].any()


def test_empty_dataframe():
    """Test function behavior with an empty dataframe"""
    empty_df = pd.DataFrame(
        columns=["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG"]
    )

    # Process empty dataframe
    result = detect_vessel_anomalies(empty_df)

    # Should return an empty dataframe with the expected columns
    assert len(result) == 0


def test_single_row_dataframe(normal_vessel_data):
    """Test function behavior with a dataframe containing only one row"""
    single_row = normal_vessel_data.iloc[0:1]

    # Process single row dataframe
    result = detect_vessel_anomalies(single_row, return_entire_dataframe=True)

    # Should return a dataframe with one row and no anomalies
    assert len(result) == 1
    assert not result["anomaly"].any()

    # First row should have distance = 0
    assert result.iloc[0]["distance"] == 0.0


def test_timestamp_conversion(normal_vessel_data):
    """Test that timestamps are correctly converted"""
    result = detect_vessel_anomalies(normal_vessel_data, return_entire_dataframe=True)

    # Check that timestamps were converted to datetime
    assert pd.api.types.is_datetime64_dtype(result["Timestamp"])


def test_zero_time_difference(normal_vessel_data):
    """Test handling of zero time difference between points"""
    # Create data with identical timestamps
    zero_time_data = normal_vessel_data.copy()
    zero_time_data.loc[2, "# Timestamp"] = zero_time_data.loc[1, "# Timestamp"]

    # Process data
    result = detect_vessel_anomalies(zero_time_data, return_entire_dataframe=True)

    # Speed should be 0 for the row with zero time difference
    assert result.loc[2, "speed"] == 0.0


def _test_backwards_time(normal_vessel_data):
    """Test handling of timestamps that go backwards"""
    # Create data with a timestamp that goes backwards in time
    backwards_time_data = normal_vessel_data.copy()
    base_timestamp = datetime.strptime(
        backwards_time_data.loc[0, "# Timestamp"], "%d/%m/%Y %H:%M:%S"
    )
    backwards_time_data.loc[2, "# Timestamp"] = (
        base_timestamp - timedelta(minutes=30)
    ).strftime("%d/%m/%Y %H:%M:%S")

    # Process data
    result = detect_vessel_anomalies(backwards_time_data, return_entire_dataframe=True)

    # Time difference should be negative
    assert result.loc[2, "delta_time_seconds"] < 0
    assert result.loc[2, "delta_time_hours"] < 0

    # Speed should be 0 for negative time differences (as per the function logic)
    assert result.loc[2, "speed"] == 0.0


def test_distance_calculation(normal_vessel_data):
    """Test the distance calculation using a mock"""
    with patch("geopy.distance.distance") as mock_distance:
        # Mock the distance.distance function to return a known value
        distance_instance = MagicMock()
        distance_instance.miles = 10.0
        mock_distance.return_value = distance_instance

        # Process data
        result = detect_vessel_anomalies(
            normal_vessel_data, return_entire_dataframe=True
        )

        # Check that distance was calculated using the mocked function
        assert mock_distance.called

        # All distances (except the first row) should be the mocked value
        for i in range(1, len(result)):
            assert result.iloc[i]["distance"] == 10.0


def test_expected_anomalies_with_real_data():
    """Test with more realistic vessel data to ensure anomalies are correctly identified"""
    # Create a dataframe with realistic vessel movement including an anomaly
    timestamps = [
        "01/01/2025 08:00:00",
        "01/01/2025 08:15:00",
        "01/01/2025 08:30:00",
        "01/01/2025 08:45:00",
        "01/01/2025 09:00:00",
        # Large time gap (3 hours)
        "01/01/2025 12:00:00",
        # Position jump (should trigger speed anomaly)
        "01/01/2025 12:15:00",
    ]

    latitudes = [40.0, 40.05, 40.1, 40.15, 40.2, 40.3, 45.0]
    longitudes = [-74.0, -74.05, -74.1, -74.15, -74.2, -74.3, -74.4]

    test_data = pd.DataFrame(
        {
            "# Timestamp": timestamps,
            "MMSI": [123456789] * len(timestamps),
            "Latitude": latitudes,
            "Longitude": longitudes,
            "SOG": [10.0] * len(timestamps),
        }
    )

    # Process data
    result = detect_vessel_anomalies(test_data)

    # Should have detected two anomalies:
    # 1. AIS gap between 09:00 and 12:00
    # 2. Speed anomaly between 12:00 and 12:15
    assert len(result) == 2

    # Check that both types of anomalies were detected
    assert result["ais_gap"].any()
    assert result["speed_anomaly"].any()


def test_max_speed_calculation():
    """Test that speed calculations are correct and the max_vessel_speed threshold works"""
    # Create a simple dataset with known distances and times
    timestamps = [
        "01/01/2025 08:00:00",
        "01/01/2025 09:00:00",  # 1 hour later
    ]

    # These coordinates should be approximately 69 miles apart (1 degree latitude ≈ 69 miles)
    latitudes = [40.0, 41.0]
    longitudes = [-74.0, -74.0]

    test_data = pd.DataFrame(
        {
            "# Timestamp": timestamps,
            "MMSI": [123456789, 123456789],
            "Latitude": latitudes,
            "Longitude": longitudes,
            "SOG": [10.0, 10.0],
        }
    )

    # With default max_vessel_speed=50.0, this should trigger a speed anomaly
    result_default = detect_vessel_anomalies(test_data, return_entire_dataframe=True)
    assert result_default["speed_anomaly"].any()

    # The calculated speed should be close to 69 mph
    assert 65 <= result_default.iloc[1]["speed"] <= 75

    # With max_vessel_speed=100.0, this should NOT trigger a speed anomaly
    result_high_threshold = detect_vessel_anomalies(
        test_data, max_vessel_speed=100.0, return_entire_dataframe=True
    )
    assert not result_high_threshold["speed_anomaly"].any()
