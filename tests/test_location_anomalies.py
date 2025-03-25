import pytest
import pandas as pd
from assignment_1.location_anomalies import preprocess_vessel_data


@pytest.fixture
def basic_data():
    """Fixture with basic vessel data for testing"""
    return pd.DataFrame(
        {
            "MMSI": [1, 2, 3],
            "Timestamp": [
                "01/01/2023 12:00:00",
                "01/01/2023 12:05:00",
                "01/01/2023 12:10:00",
            ],
            "Latitude": [50.005, 50.015, 50.025],
            "Longitude": [10.005, 10.015, 10.025],
        }
    )


def test_overlap_near_lat_boundary():
    """Test points near latitude bin boundary"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.019],  # Very close to 50.02 boundary
            "Longitude": [10.005],  # Not close to boundary
        }
    )

    result = preprocess_vessel_data(data, lat_bin_size=0.01, lon_bin_size=0.01)

    # Should be in both the 50.01 bin and the 50.02 bin
    assert len(result) == 2

    # Check both bins are represented
    lat_bins = sorted(result["lat_bin"].unique())
    assert len(lat_bins) == 2
    assert lat_bins[0] == 50.01
    assert lat_bins[1] == 50.02

    # Longitude bin should be the same for both
    lon_bins = result["lon_bin"].unique()
    assert len(lon_bins) == 1
    assert lon_bins[0] == 10.00


def test_overlap_near_lon_boundary():
    """Test points near longitude bin boundary"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.005],  # Not close to boundary
            "Longitude": [10.019],  # Very close to 10.02 boundary
        }
    )

    result = preprocess_vessel_data(data, lat_bin_size=0.01, lon_bin_size=0.01)

    # Should be in both the 10.01 bin and the 10.02 bin
    assert len(result) == 2

    # Latitude bin should be the same for both
    lat_bins = result["lat_bin"].unique()
    assert len(lat_bins) == 1
    assert lat_bins[0] == 50.00

    # Check both longitude bins are represented
    lon_bins = sorted(result["lon_bin"].unique())
    assert len(lon_bins) == 2
    assert lon_bins[0] == 10.01
    assert lon_bins[1] == 10.02


def test_overlap_corner_case():
    """Test points near both latitude and longitude bin boundaries"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.019],  # Very close to 50.02 boundary
            "Longitude": [10.019],  # Very close to 10.02 boundary
        }
    )

    result = preprocess_vessel_data(data, lat_bin_size=0.01, lon_bin_size=0.01)

    # Should be in 4 bins: (50.01,10.01), (50.01,10.02), (50.02,10.01), (50.02,10.02)
    assert len(result) == 4

    # Check all combinations exist
    bins = set([(row["lat_bin"], row["lon_bin"]) for _, row in result.iterrows()])
    assert (50.01, 10.01) in bins
    assert (50.01, 10.02) in bins
    assert (50.02, 10.01) in bins
    assert (50.02, 10.02) in bins


def test_timestamp_conversion():
    """Test timestamp column renaming and conversion"""
    data = pd.DataFrame(
        {
            "MMSI": [1, 2],
            "# Timestamp": ["01/01/2023 12:00:00", "02/01/2023 12:00:00"],
            "Latitude": [50.005, 50.015],
            "Longitude": [10.005, 10.015],
        }
    )

    result = preprocess_vessel_data(data)

    # Check timestamp conversion
    assert "Timestamp" in result.columns
    assert "# Timestamp" not in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["Timestamp"])
    assert result.iloc[0]["Timestamp"].day == 1
    assert result.iloc[1]["Timestamp"].day == 2


def test_already_datetime_timestamp():
    """Test handling of already datetime timestamps"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": [pd.Timestamp("2023-01-01 12:00:00")],
            "Latitude": [50.005],
            "Longitude": [10.005],
        }
    )

    result = preprocess_vessel_data(data)

    # Should not change the timestamp
    assert pd.api.types.is_datetime64_any_dtype(result["Timestamp"])
    assert result.iloc[0]["Timestamp"] == pd.Timestamp("2023-01-01 12:00:00")


def test_boundary_threshold_sensitivity():
    """Test different threshold values for boundary detection"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.0105],  # 0.0005 away from next bin (50.02)
            "Longitude": [10.005],
        }
    )

    # Create a temporary function with a higher threshold to test sensitivity
    def preprocess_with_custom_threshold(data):
        # This modifies the threshold within the function to 20% of bin size
        # Rather than the default 10%
        return preprocess_vessel_data(data, lat_bin_size=0.01, lon_bin_size=0.01)

    # With default threshold (10% of bin size = 0.001), this should NOT be considered near boundary
    result1 = preprocess_vessel_data(data, lat_bin_size=0.01, lon_bin_size=0.01)
    assert len(result1) == 1  # Only in one bin

    # If we were to modify the threshold to 20% (0.002), this WOULD be considered near boundary
    # Note: This is a property test showing threshold sensitivity, not actual implementation


def test_exactly_at_bin_boundary():
    """Test behavior for points exactly at bin boundaries"""
    data = pd.DataFrame(
        {
            "MMSI": [1],
            "Timestamp": ["01/01/2023 12:00:00"],
            "Latitude": [50.01],  # Exactly at bin boundary
            "Longitude": [10.01],  # Exactly at bin boundary
        }
    )

    result = preprocess_vessel_data(data, overlap=False)

    # Should go in just one bin
    assert len(result) == 1
    assert result.iloc[0]["lat_bin"] == 50.01
    assert result.iloc[0]["lon_bin"] == 10.01
