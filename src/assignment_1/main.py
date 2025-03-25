import numpy as np
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
    # Make a copy of the data to avoid modifying the original DataFrame
    vessel_data = vessel_data.copy()

    # Preprocessing - do all column operations at once for better vectorization
    vessel_data["Latitude"] = vessel_data["Latitude"].clip(lower=-90, upper=90)
    vessel_data["Longitude"] = vessel_data["Longitude"].clip(lower=-180, upper=180)

    # Rename timestamp column if needed and convert to datetime
    if "# Timestamp" in vessel_data.columns:
        vessel_data = vessel_data.rename(columns={"# Timestamp": "Timestamp"})

    if not pd.api.types.is_datetime64_any_dtype(vessel_data["Timestamp"]):
        vessel_data["Timestamp"] = pd.to_datetime(
            vessel_data["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )

    # Drop duplicates and sort by timestamp
    vessel_data = vessel_data.drop_duplicates().sort_values("Timestamp")

    # Initialize calculation columns
    vessel_data["distance"] = 0.0
    vessel_data["delta_time_seconds"] = 0.0
    vessel_data["delta_time_hours"] = 0.0
    vessel_data["speed"] = 0.0

    # Vectorized calculation of time differences
    vessel_data["delta_time_seconds"] = (
        vessel_data["Timestamp"].diff().dt.total_seconds()
    )
    vessel_data["delta_time_hours"] = vessel_data["delta_time_seconds"] / 3600

    # Use NumPy for faster distance calculations if possible
    # Pre-allocate arrays for lat/lon points
    coords_prev = np.column_stack(
        (
            vessel_data["Latitude"].iloc[:-1].values,
            vessel_data["Longitude"].iloc[:-1].values,
        )
    )
    coords_curr = np.column_stack(
        (
            vessel_data["Latitude"].iloc[1:].values,
            vessel_data["Longitude"].iloc[1:].values,
        )
    )

    # Calculate distances using vectorized operations
    distances = np.zeros(len(vessel_data))

    for i in range(len(coords_prev)):
        distances[i + 1] = geopy_distance.distance(
            (coords_prev[i, 0], coords_prev[i, 1]),
            (coords_curr[i, 0], coords_curr[i, 1]),
        ).miles

    vessel_data["distance"] = distances

    # Calculate speed (vectorized)
    mask = vessel_data["delta_time_hours"] > 0
    vessel_data.loc[mask, "speed"] = (
        vessel_data.loc[mask, "distance"] / vessel_data.loc[mask, "delta_time_hours"]
    )

    # Detect anomalies (vectorized operations)
    vessel_data["speed_anomaly"] = vessel_data["speed"] > max_vessel_speed
    vessel_data["ais_gap"] = vessel_data["delta_time_hours"] > max_time_gap
    vessel_data["anomaly"] = vessel_data["speed_anomaly"] | vessel_data["ais_gap"]

    # Only keep rows with anomalies if requested
    if not return_entire_dataframe:
        vessel_data = vessel_data[vessel_data["anomaly"]]

    return vessel_data


def detect_vessel_anomalies_kwwargs(kwargs):
    """Helper function to unpack arguments for detect_vessel_anomalies"""
    return detect_vessel_anomalies(**kwargs)


@timeit
def group_by_vessel(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("MMSI")


@timeit
def detect_vessel_anomalies_single_process(
    data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
    max_time_gap: float = MAX_TIME_GAP,
) -> pd.DataFrame:
    all_anomalies = pd.DataFrame()
    vessels = group_by_vessel(data)
    vessles_ids = list(vessels.groups.keys())

    with tqdm(
        total=len(vessels.groups), unit="vessel", desc="Processing vessels"
    ) as pbar:
        for vessel_id in vessles_ids:
            anomalies = detect_vessel_anomalies(
                vessels.get_group(vessel_id), max_vessel_speed, max_time_gap
            )
            all_anomalies = pd.concat([all_anomalies, anomalies])
            pbar.update(1)

    return all_anomalies


@timeit
def detect_vessel_anomalies_multi_process(
    data: pd.DataFrame,
    max_vessel_speed: float = MAX_VESSEL_SPEED,
    max_time_gap: float = MAX_TIME_GAP,
) -> pd.DataFrame:
    number_of_processes = mp.cpu_count() - 1
    print(f"Using {number_of_processes} processes.")

    # Group data by vessel MMSI
    vessels = group_by_vessel(data)
    vessels_data = [
        {
            "vessel_data": group.copy(),
            "max_vessel_speed": max_vessel_speed,
            "max_time_gap": max_time_gap,
        }
        for _, group in vessels
    ]

    # Initialize multiprocessing pool
    all_results = []
    with mp.Pool(number_of_processes) as pool:
        results = pool.imap(detect_vessel_anomalies_kwwargs, vessels_data)

        # Create a progress bar for the total number of vessels
        with tqdm(
            total=len(vessels_data), unit="vessel", desc="Processing vessels"
        ) as pbar:
            # Process each result as it completes
            for result in results:
                all_results.append(result)
                pbar.update(1)

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
    r1 = detect_vessel_anomalies_single_process(data)
    r2 = detect_vessel_anomalies_multi_process(data)

    key_cols = ["MMSI", "Timestamp"]
    df1_sorted = r1.sort_values(key_cols).reset_index(drop=True)
    df2_sorted = r2.sort_values(key_cols).reset_index(drop=True)
    print(df1_sorted.equals(df2_sorted))

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
