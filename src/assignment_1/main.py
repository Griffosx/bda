from pathlib import Path
import numpy as np
import pandas as pd


BASE_PATH = Path("~/Projects/bda").expanduser()
MAX_VESSEL_SPEED = 50.0  # Maximum expected speed in km/h
MAX_ACCELERATION = 25.0  # Maximum expected acceleration in km/h^2


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.
    Returns the distance in kilometers.
    """
    # Earth's radius in kilometers
    earth_radius = 6371

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the square of half the chord length between the points
    # This is the Haversine formula
    half_chord_squared = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(
        lat2
    ) * np.sin(dlon / 2) * np.sin(dlon / 2)

    # Calculate the angular distance in radians
    angular_distance = 2 * np.arctan2(
        np.sqrt(half_chord_squared), np.sqrt(1 - half_chord_squared)
    )

    # Calculate the distance in kilometers
    return earth_radius * angular_distance


def detect_vessels_anomalies(
    vessel_data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
    max_acceleration: float = MAX_ACCELERATION,
    detect_intrasecond_anomalies: bool = False,
) -> pd.DataFrame:
    # Make a copy of the data to avoid modifying the original DataFrame and to be thread-safe
    vessel_data = vessel_data.copy()

    # Add columns distance, delta_time, and speed
    vessel_data["distance"] = 0.0
    vessel_data["delta_time"] = 0.0
    vessel_data["speed"] = 0.0
    vessel_data["acceleration"] = 0.0
    vessel_data["intrasecond_anomaly"] = False

    # Rename "# Timestamp" to "Timestamp" and convert to datetime
    vessel_data = vessel_data.rename(columns={"# Timestamp": "Timestamp"})
    vessel_data["Timestamp"] = pd.to_datetime(
        vessel_data["Timestamp"], format="%d/%m/%Y %H:%M:%S"
    )

    if detect_intrasecond_anomalies:
        # Group by timestamp and calculate variance for lat and lon
        timestamp_groups = vessel_data.groupby("Timestamp")

        # Calculate variance for each group
        for timestamp, group in timestamp_groups:
            if len(group) > 1:  # Only process groups with multiple records
                lat_variance = group["Latitude"].var()
                lon_variance = group["Longitude"].var()

                # Set threshold for variance (adjust as needed)
                variance_threshold = 1e-6  # Example threshold

                # Mark records as anomalies if variance exceeds threshold
                if (
                    lat_variance > variance_threshold
                    or lon_variance > variance_threshold
                ):
                    vessel_data.loc[group.index, "intrasecond_anomaly"] = True

    # vessel_data = vessel_data.sort_values(by="Timestamp")

    # Calculate distance, time difference, and speed between consecutive points
    for i in range(1, len(vessel_data)):
        prev_row = vessel_data.iloc[i - 1]
        curr_row = vessel_data.iloc[i]

        # Calculate distance between consecutive points
        distance = calculate_distance(
            prev_row["Latitude"],
            prev_row["Longitude"],
            curr_row["Latitude"],
            curr_row["Longitude"],
        )
        # Calculate time difference in hours
        time_diff = (
            curr_row["Timestamp"] - prev_row["Timestamp"]
        ).total_seconds() / 3600

        vessel_data.loc[vessel_data.index[i], "distance"] = distance
        vessel_data.loc[vessel_data.index[i], "delta_time"] = time_diff

        # Calculate speed in km/h (distance/time)
        if time_diff > 0:  # Avoid division by zero
            vessel_data.loc[vessel_data.index[i], "speed"] = distance / time_diff

    # Calculate acceleration between consecutive points (change in speed over time)
    for i in range(2, len(vessel_data)):
        prev_speed = vessel_data.iloc[i - 1]["speed"]
        curr_speed = vessel_data.iloc[i]["speed"]
        time_diff = vessel_data.iloc[i]["delta_time"]

        # Calculate acceleration in km/h^2 (change in speed / time)
        if time_diff > 0:  # Avoid division by zero
            acceleration = (curr_speed - prev_speed) / time_diff
            vessel_data.loc[vessel_data.index[i], "acceleration"] = acceleration

    # Detect anomalies
    # 1. Speed anomalies / position jumps
    vessel_data["speed_anomaly"] = vessel_data["speed"] > max_vessel_speed

    # 2. Stationary but moving (SOG > 0 but position doesn't change)
    vessel_data["stationary_moving"] = (vessel_data["distance"] < 0.1) & (
        vessel_data["SOG"] > 2.0
    )

    # 3. AIS gaps (significant time gaps between transmissions)
    vessel_data["ais_gap"] = vessel_data["delta_time"] > 1.0  # Gap greater than 1 hour

    # 4. Acceleration anomalies (unrealistic acceleration or deceleration)
    vessel_data["acceleration_anomaly"] = (
        abs(vessel_data["acceleration"]) > max_acceleration
    )

    # Flag as anomaly if any of the individual anomaly types are detected
    vessel_data["anomaly"] = (
        vessel_data["speed_anomaly"]
        | vessel_data["stationary_moving"]
        | vessel_data["ais_gap"]
        | vessel_data["acceleration_anomaly"]
        | vessel_data["intrasecond_anomaly"]
    )
    # Only keep rows with anomalies
    vessel_data = vessel_data[vessel_data["anomaly"]]

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

    for vessel_id in vessles_ids[:1]:
        print(f"Vessel ID: {vessel_id}")
        anomalies = detect_vessels_anomalies(
            vessels.get_group(vessel_id), detect_intrasecond_anomalies=True
        )
        print(anomalies)


if __name__ == "__main__":
    main()
