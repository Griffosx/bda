from pathlib import Path
import numpy as np
import pandas as pd
from geopy import distance as geopy_distance

BASE_PATH = Path("~/Projects/bda").expanduser()
MAX_VESSEL_SPEED = 50.0  # Maximum expected speed in km/h


def detect_vessels_anomalies(
    vessel_data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
    detect_intrasecond_anomalies: bool = False,
) -> pd.DataFrame:
    # Make a copy of the data to avoid modifying the original DataFrame and to be thread-safe
    vessel_data = vessel_data.copy()

    # Add columns distance, delta_time, and speed
    vessel_data["distance"] = 0.0
    vessel_data["delta_time_seconds"] = 0.0
    vessel_data["delta_time_hours"] = 0.0
    vessel_data["speed"] = 0.0
    vessel_data["intrasecond_anomaly"] = False

    # Rename "# Timestamp" to "Timestamp" and convert to datetime
    vessel_data = vessel_data.rename(columns={"# Timestamp": "Timestamp"})
    vessel_data["Timestamp"] = pd.to_datetime(
        vessel_data["Timestamp"], format="%d/%m/%Y %H:%M:%S"
    )

    # Drop duplicate rows
    vessel_data = vessel_data.drop_duplicates()

    if detect_intrasecond_anomalies:
        # TODO count number or rows and number of unique timestamps
        pass

    # vessel_data = vessel_data.sort_values(by="Timestamp")

    # Calculate distance, time difference, and speed between consecutive points
    for i in range(1, len(vessel_data)):
        prev_row = vessel_data.iloc[i - 1]
        curr_row = vessel_data.iloc[i]

        distance = geopy_distance.distance(
            (prev_row["Latitude"], prev_row["Longitude"]),
            (curr_row["Latitude"], curr_row["Longitude"]),
        ).miles
        # Calculate time difference in hours
        time_diff = (curr_row["Timestamp"] - prev_row["Timestamp"]).total_seconds()
        time_diff_hours = time_diff / 3600

        vessel_data.loc[vessel_data.index[i], "distance"] = distance
        vessel_data.loc[vessel_data.index[i], "delta_time_seconds"] = time_diff
        vessel_data.loc[vessel_data.index[i], "delta_time_hours"] = time_diff_hours

        # Calculate speed in km/h (distance/time)
        if time_diff > 0:  # Avoid division by zero
            vessel_data.loc[vessel_data.index[i], "speed"] = distance / time_diff_hours

    # Detect anomalies
    # 1. Speed anomalies / position jumps
    vessel_data["speed_anomaly"] = vessel_data["speed"] > max_vessel_speed

    # 2. AIS gaps (significant time gaps between transmissions)
    vessel_data["ais_gap"] = (
        vessel_data["delta_time_hours"] > 1.0
    )  # Gap greater than 1 hour

    # Flag as anomaly if any of the individual anomaly types are detected
    vessel_data["anomaly"] = (
        vessel_data["speed_anomaly"]
        | vessel_data["ais_gap"]
        | vessel_data["intrasecond_anomaly"]
    )
    # Only keep rows with anomalies
    # vessel_data = vessel_data[vessel_data["anomaly"]]

    return vessel_data


def detect_conflicting_locations(data: pd.DataFrame) -> pd.DataFrame: ...


def main():
    # pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    data = pd.read_csv(
        BASE_PATH / "assets/test.csv",
        usecols=["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG"],
    )
    vessels = data.groupby("MMSI")
    vessles_ids = list(vessels.groups.keys())

    vessel_id = 211718360
    anomalies = detect_vessels_anomalies(
        vessels.get_group(vessel_id), detect_intrasecond_anomalies=False
    )
    print(anomalies)

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
