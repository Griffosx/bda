import pandas as pd
from pathlib import Path
from assignment_1.vessel_anomalies import detect_vessel_anomalies_multi_process
from assignment_1.location_anomalies import detect_conflicting_locations_single_process


BASE_PATH = Path("assets").resolve()


def normalize_coordinates(data, precision=4):
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


def main():
    pd.set_option("display.max_rows", None)

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
    # Remove column "Type of mobile"
    data = data.drop(columns=["Type of mobile"])
    data = normalize_coordinates(data)

    # r1 = detect_vessel_anomalies_single_process(data)
    # vessel_anomalies = detect_vessel_anomalies_multi_process(data)

    location_anomalies = detect_conflicting_locations_single_process(
        data, lat_bin_size=0.1, lon_bin_size=0.1
    )
    # print(location_anomalies)

    # key_cols = ["MMSI", "Timestamp"]
    # df1_sorted = r1.sort_values(key_cols).reset_index(drop=True)
    # df2_sorted = r2.sort_values(key_cols).reset_index(drop=True)
    # print(df1_sorted.equals(df2_sorted))

    # detect_vessel_anomalies_multi_process(data)
    # print(detect_vessel_anomalies_multi_process(data))

    # vessels = data.groupby("MMSI")
    # vessles_ids = list(vessels.groups.keys())

    # vessel_id = 111257003
    # anomalies = detect_vessel_anomalies(
    #     vessels.get_group(vessel_id), return_entire_dataframe=True
    # )
    # print(anomalies)

    # for vessel_id in vessles_ids:
    #     print(f"Vessel ID: {vessel_id}")
    #     anomalies = detect_vessels_anomalies(
    #         vessels.get_group(vessel_id), detect_intrasecond_anomalies=False
    #     )
    #     # If there is at least one SOG > 0 print the anomalies
    #     if anomalies["SOG"].max() > 1:
    #         print(anomalies)


if __name__ == "__main__":
    main()
