import pandas as pd
from pathlib import Path
from utils.timer_wrapper import timeit
from assignment_1.vessel_anomalies import (
    detect_vessel_anomalies_single_process,
    detect_vessel_anomalies_multi_process,
)
from assignment_1.location_anomalies import (
    detect_conflicting_locations_single_process,
    detect_conflicting_locations_multi_process,
)


BASE_PATH = Path("assets").resolve()


def normalize_coordinates(data, precision=6):
    """
    Normalize latitude and longitude values in a DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Latitude' and 'Longitude' columns
    precision (int): Decimal precision for rounding coordinates

    Returns:
    pd.DataFrame: DataFrame with normalized and rounded coordinates
    """
    import numpy as np

    # Make a copy to avoid modifying the original DataFrame
    df = data.copy()

    # Normalize latitude (-90 to 90)
    lat = df["Latitude"].copy()
    # Handle values outside valid range with modulo arithmetic and reflection
    lat = lat % 360  # Normalize to 0-360 range
    # Convert to -180 to 180 range
    lat = np.where(lat > 180, lat - 360, lat)
    # Apply reflection at the poles
    lat = np.where(lat > 90, 180 - lat, lat)
    lat = np.where(lat < -90, -180 - lat, lat)

    # Normalize longitude (-180 to 180)
    lon = df["Longitude"].copy()
    lon = lon % 360  # Normalize to 0-360 range
    # Convert to -180 to 180 range
    lon = np.where(lon > 180, lon - 360, lon)

    # Update the DataFrame with normalized values
    df["Latitude"] = lat
    df["Longitude"] = lon

    # Optionally round the coordinates to specified precision
    if precision is not None:
        df["Latitude"] = df["Latitude"].round(precision)
        df["Longitude"] = df["Longitude"].round(precision)

    return df


@timeit
def main_multiprocess():
    data = pd.read_csv(
        # BASE_PATH / "test.csv",
        BASE_PATH / "aisdk-2024-06-30.csv",
        usecols=[
            "# Timestamp",
            "Type of mobile",
            "MMSI",
            "Latitude",
            "Longitude",
            "SOG",
        ],
    )
    # Remove all Type of mobile that are "Base Station"
    data = data[data["Type of mobile"] != "Base Station"]
    # Remove column "Type of mobile", not needed anymore
    data = data.drop(columns=["Type of mobile"])
    data = normalize_coordinates(data)

    vessel_anomalies = detect_vessel_anomalies_multi_process(data)
    location_anomalies = detect_conflicting_locations_multi_process(
        data, lat_bin_size=0.1, lon_bin_size=0.1, time_bin_size="1H"
    )
    return vessel_anomalies, location_anomalies


@timeit
def main_single_process():
    data = pd.read_csv(
        # BASE_PATH / "test.csv",
        BASE_PATH / "aisdk-2024-06-30.csv",
        usecols=[
            "# Timestamp",
            "Type of mobile",
            "MMSI",
            "Latitude",
            "Longitude",
            "SOG",
        ],
    )
    # Remove all Type of mobile that are "Base Station"
    data = data[data["Type of mobile"] != "Base Station"]
    # Remove column "Type of mobile", not needed anymore
    data = data.drop(columns=["Type of mobile"])
    data = normalize_coordinates(data)

    vessel_anomalies = detect_vessel_anomalies_single_process(data)
    location_anomalies = detect_conflicting_locations_single_process(
        data, lat_bin_size=0.1, lon_bin_size=0.1, time_bin_size="1H"
    )
    return vessel_anomalies, location_anomalies


def main(mono=False, multi=False):
    pd.set_option("display.max_rows", None)

    if multi:
        vessel_anomalies_multi, location_anomalies_multi = main_multiprocess()
        vessel_anomalies_multi.sort_values(
            ["MMSI", "Timestamp", "Latitude", "Longitude"]
        )
        location_anomalies_multi.sort_values(
            ["MMSI1", "MMSI2", "Timestamp1", "Timestamp2"]
        )

    if mono:
        vessel_anomalies_mono, location_anomalies_mono = main_single_process()
        vessel_anomalies_mono.sort_values(
            ["MMSI", "Timestamp", "Latitude", "Longitude"]
        )
        location_anomalies_mono.sort_values(
            ["MMSI1", "MMSI2", "Timestamp1", "Timestamp2"]
        )


if __name__ == "__main__":
    main(multi=True, mono=True)
