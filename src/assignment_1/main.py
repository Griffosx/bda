import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from geopy import distance as geopy_distance
from utils.timer_wrapper import timeit


BASE_PATH = Path("~/Projects/bda").expanduser()
MAX_VESSEL_SPEED = 50.0  # Maximum expected speed in miles/h
MAX_TIME_GAP = 1.0  # Maximum expected time gap between AIS transmissions in hours


def detect_vessel_anomalies(
    vessel_data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
    max_time_gap: float = MAX_TIME_GAP,
    return_entire_dataframe: bool = False,
) -> pd.DataFrame:
    # Make a copy of the data to avoid modifying the original DataFrame and to be thread-safe
    vessel_data = vessel_data.copy()

    # Cap latitude between -90 and 90
    vessel_data["Latitude"] = vessel_data["Latitude"].clip(lower=-90, upper=90)

    # Cap longitude between -180 and 180
    vessel_data["Longitude"] = vessel_data["Longitude"].clip(lower=-180, upper=180)

    # Add columns distance, delta_time, and speed
    vessel_data["distance"] = 0.0
    vessel_data["delta_time_seconds"] = 0.0
    vessel_data["delta_time_hours"] = 0.0
    vessel_data["speed"] = 0.0

    # Rename "# Timestamp" to "Timestamp" and convert to datetime
    vessel_data = vessel_data.rename(columns={"# Timestamp": "Timestamp"})
    vessel_data["Timestamp"] = pd.to_datetime(
        vessel_data["Timestamp"], format="%d/%m/%Y %H:%M:%S"
    )

    # Drop duplicate rows
    vessel_data = vessel_data.drop_duplicates()

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

        # Calculate speed in miles/h (distance/time)
        if time_diff > 0:  # Avoid division by zero
            vessel_data.loc[vessel_data.index[i], "speed"] = distance / time_diff_hours

    # Detect anomalies
    # 1. Speed anomalies / position jumps
    vessel_data["speed_anomaly"] = vessel_data["speed"] > max_vessel_speed

    # 2. AIS gaps (significant time gaps between transmissions)
    vessel_data["ais_gap"] = (
        vessel_data["delta_time_hours"] > max_time_gap
    )  # Gap greater than 1 hour

    # Flag as anomaly if any of the individual anomaly types are detected
    vessel_data["anomaly"] = vessel_data["speed_anomaly"] | vessel_data["ais_gap"]

    # Only keep rows with anomalies
    if not return_entire_dataframe:
        vessel_data = vessel_data[vessel_data["anomaly"]]

    return vessel_data


@timeit
def group_by_vessel(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("MMSI")


@timeit
def detect_vessel_anomalies_single_process(
    data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
) -> pd.DataFrame:
    all_anomalies = pd.DataFrame()
    vessels = data.groupby("MMSI")
    vessles_ids = list(vessels.groups.keys())

    with tqdm(
        total=len(vessels.groups), unit="vessel", desc="Processing vessels"
    ) as pbar:
        for vessel_id in vessles_ids:
            anomalies = detect_vessel_anomalies(
                vessels.get_group(vessel_id), max_vessel_speed
            )
            all_anomalies = pd.concat([all_anomalies, anomalies])
            pbar.update(1)

    return all_anomalies


@timeit
def detect_vessel_anomalies_multi_process(
    data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
) -> pd.DataFrame:
    number_of_processes = max(mp.cpu_count() - 1, 1)
    print(f"Using {number_of_processes} processes.")

    # Group data by vessel MMSI
    vessels = data.groupby("MMSI")
    vessels_data = [(group.copy(), max_vessel_speed) for _, group in vessels]

    # Initialize multiprocessing pool
    all_results = []
    with mp.Pool(number_of_processes) as pool:
        # Create a progress bar for the total number of vessels
        with tqdm(
            total=len(vessels_data), unit="vessel", desc="Processing vessels"
        ) as pbar:
            # Process each vessel and update the progress bar when a result is ready
            for result in pool.starmap_async(
                detect_vessel_anomalies, vessels_data
            ).get():
                all_results.append(result)
                pbar.update()

    # Combine results from all processes
    all_anomalies = pd.concat(all_results, ignore_index=True)

    return all_anomalies


def detect_conflicting_locations(data: pd.DataFrame) -> pd.DataFrame: ...


def main():
    # pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    data = pd.read_csv(
        BASE_PATH / "assets/test.csv",
        usecols=["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG"],
    )

    # print(detect_vessel_anomalies_single_process(data))
    detect_vessel_anomalies_multi_process(data)
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
