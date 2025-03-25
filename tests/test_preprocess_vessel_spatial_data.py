import pandas as pd
from assignment_1.location_anomalies import preprocess_vessel_spatial_data


def test_negative_coordinates():
    """Test behavior with negative latitude/longitude"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": ["01/01/2023 12:00:00", "01/01/2023 12:00:00"],
            "Latitude": [-50.005, 50.005],  # Negative latitude
            "Longitude": [10.005, -10.005],  # Negative longitude
        }
    )

    result = preprocess_vessel_spatial_data(data)

    # Check negative coordinates are binned correctly
    assert result[result["MMSI"] == 1].iloc[0]["lat_bin"] == -50.01
    assert result[result["MMSI"] == 2].iloc[0]["lon_bin"] == -10.01


def test_float_precision_issues():
    """Test behavior with coordinates that could cause float precision issues"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": ["01/01/2023 12:00:00", "01/01/2023 12:00:00"],
            "Latitude": [
                50.009999999999,
                50.019999999999,
            ],  # Very close to bin boundaries
            "Longitude": [10.009999999999, 10.019999999999],
        }
    )

    result = preprocess_vessel_spatial_data(data)

    # Check binning is consistent despite float precision issues
    assert result[result["MMSI"] == 1].iloc[0]["lat_bin"] == 50.00
    assert result[result["MMSI"] == 2].iloc[0]["lat_bin"] == 50.01


def test_disable_overlap():
    """Test behavior when overlap is disabled (more comprehensive)"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2, 3, 4],
            "Timestamp": ["01/01/2023 12:00:00"] * 4,
            "Latitude": [50.005, 50.019, 50.005, 50.019],
            "Longitude": [10.005, 10.005, 10.019, 10.019],
        }
    )

    # Without overlap, each point should be in exactly one bin
    result = preprocess_vessel_spatial_data(data, overlap=False)

    assert len(result) == len(
        data
    )  # Should have the same number of rows as original data

    # Check bin assignments
    assert result[result["MMSI"] == 1].iloc[0]["lat_bin"] == 50.00
    assert result[result["MMSI"] == 1].iloc[0]["lon_bin"] == 10.00

    assert result[result["MMSI"] == 2].iloc[0]["lat_bin"] == 50.01
    assert result[result["MMSI"] == 2].iloc[0]["lon_bin"] == 10.00

    assert result[result["MMSI"] == 3].iloc[0]["lat_bin"] == 50.00
    assert result[result["MMSI"] == 3].iloc[0]["lon_bin"] == 10.01

    assert result[result["MMSI"] == 4].iloc[0]["lat_bin"] == 50.01
    assert result[result["MMSI"] == 4].iloc[0]["lon_bin"] == 10.01


def test_custom_bin_sizes():
    """Test behavior with custom bin sizes"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.025],
            "Longitude": [10.025],
        }
    )

    # Test with different bin sizes
    result_default = preprocess_vessel_spatial_data(
        data, lat_bin_size=0.01, lon_bin_size=0.01
    )
    result_custom = preprocess_vessel_spatial_data(
        data, lat_bin_size=0.05, lon_bin_size=0.05
    )

    # Default should bin to 50.02, 10.02
    assert result_default.iloc[0]["lat_bin"] == 50.02
    assert result_default.iloc[0]["lon_bin"] == 10.02

    # Custom should bin to 50.00, 10.00 (since 50.025 / 0.05 = 1000.5, floored to 1000, then 1000 * 0.05 = 50.00)
    assert result_custom.iloc[0]["lat_bin"] == 50.00
    assert result_custom.iloc[0]["lon_bin"] == 10.00


def test_extreme_values():
    """Test behavior with extreme latitude/longitude values"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": ["01/01/2023 12:00:00", "01/01/2023 12:00:00"],
            "Latitude": [89.9999, -89.9999],  # Very close to poles
            "Longitude": [179.9999, -179.9999],  # Very close to date line
        }
    )

    result = preprocess_vessel_spatial_data(data)

    # Check extreme values are binned correctly
    assert result[result["MMSI"] == 1].iloc[0]["lat_bin"] == 89.99
    assert result[result["MMSI"] == 1].iloc[0]["lon_bin"] == 179.99

    assert result[result["MMSI"] == 2].iloc[0]["lat_bin"] == -90.00
    assert result[result["MMSI"] == 2].iloc[0]["lon_bin"] == -180.00


def test_boundary_threshold_zero():
    """Test behavior when boundary_threshold is set to 0"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.019],  # Very close to boundary
            "Longitude": [10.019],  # Very close to boundary
        }
    )

    # With threshold of 0, no points should be considered near boundary
    result = preprocess_vessel_spatial_data(data, boundary_threshold=0)

    # Should only be in one bin
    assert len(result) == 1
    assert result.iloc[0]["lat_bin"] == 50.01
    assert result.iloc[0]["lon_bin"] == 10.01


def test_boundary_threshold_one():
    """Test behavior when boundary_threshold is set to 1 (entire bin is 'near boundary')"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.005],  # Not usually considered near boundary
            "Longitude": [10.005],  # Not usually considered near boundary
        }
    )

    # With threshold of 1, all points should be considered near boundary
    result = preprocess_vessel_spatial_data(data, boundary_threshold=1)

    # Should be in all 4 bins
    assert len(result) == 4

    # Check all combinations exist
    bins = set([(row["lat_bin"], row["lon_bin"]) for _, row in result.iterrows()])
    assert (50.00, 10.00) in bins
    assert (50.00, 10.01) in bins
    assert (50.01, 10.00) in bins
    assert (50.01, 10.01) in bins


def test_duplicate_rows():
    """Test behavior with duplicate coordinates"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "Timestamp": ["01/01/2023 12:00:00", "01/01/2023 12:05:00"],
            "Latitude": [50.005, 50.005],  # Same coordinates
            "Longitude": [10.005, 10.005],  # Same coordinates
        }
    )

    result = preprocess_vessel_spatial_data(data)

    # Both points should be processed independently
    assert len(result) == 2

    # Check MMSI values are preserved
    assert set(result["MMSI"]) == {1, 2}

    # Check timestamps are preserved
    timestamps = result["Timestamp"].dt.strftime("%H:%M:%S").unique()
    assert len(timestamps) == 2
    assert set(timestamps) == {"12:00:00", "12:05:00"}
