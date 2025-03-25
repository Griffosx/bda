import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from utils.timer_wrapper import timeit


DISTANCE_THRESHOLD = 100  # meters
TIME_THRESHOLD = 30  # seconds
LAT_BIN_SIZE = 0.01  # ~1km on average
LON_BIN_SIZE = 0.01  # varies with latitude
TIME_BIN_SIZE = "1min"
TIME_BOUNDARY_THRESHOLD = 10  # seconds


@timeit
def preprocess_vessel_spatial_data(
    data, lat_bin_size=0.01, lon_bin_size=0.01, boundary_threshold=0.1, overlap=True
):
    """
    Vectorized version of the vessel data preprocessing function.

    Parameters:
    data (pd.DataFrame): Input vessel data
    lat_bin_size (float): Size of latitude bins
    lon_bin_size (float): Size of longitude bins
    boundary_threshold (float): Threshold as a percentage of bin size to consider a point near the boundary
    overlap (bool): Whether to handle points near bin boundaries by placing them in multiple bins

    Returns:
    pd.DataFrame: Processed data with bin assignments
    """
    # Make a copy of the data to avoid modifying the original DataFrame
    data = data.copy()

    # Ensure timestamp is in datetime format
    if "# Timestamp" in data.columns:
        data = data.rename(columns={"# Timestamp": "Timestamp"})

    if not pd.api.types.is_datetime64_any_dtype(data["Timestamp"]):
        data["Timestamp"] = pd.to_datetime(
            data["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )

    # If overlap is disabled, just assign primary bins and return
    if not overlap:
        # Calculate the primary bin for each point
        data["lat_bin"] = (
            np.floor(data["Latitude"] / lat_bin_size) * lat_bin_size
        ).round(2)
        data["lon_bin"] = (
            np.floor(data["Longitude"] / lon_bin_size) * lon_bin_size
        ).round(2)
        return data

    # Calculate primary bins (rounded to 2 decimal places)
    data["lat_bin"] = (np.floor(data["Latitude"] / lat_bin_size) * lat_bin_size).round(
        2
    )
    data["lon_bin"] = (np.floor(data["Longitude"] / lon_bin_size) * lon_bin_size).round(
        2
    )

    # Calculate next bins
    data["next_lat_bin"] = (data["lat_bin"] + lat_bin_size).round(2)
    data["next_lon_bin"] = (data["lon_bin"] + lon_bin_size).round(2)

    # Calculate distance to next bin boundaries
    data["lat_distance"] = (data["next_lat_bin"] - data["Latitude"]).round(10)
    data["lon_distance"] = (data["next_lon_bin"] - data["Longitude"]).round(10)

    # Thresholds for boundary proximity
    lat_threshold = lat_bin_size * boundary_threshold
    lon_threshold = lon_bin_size * boundary_threshold

    # Check if near boundaries
    data["near_lat_boundary"] = data["lat_distance"] <= (lat_threshold + 1e-10)
    data["near_lon_boundary"] = data["lon_distance"] <= (lon_threshold + 1e-10)

    # Create a list to store DataFrames for each case
    result_dfs = []

    # Case 1: Primary bin (all points)
    primary_df = data[
        ["MMSI", "Timestamp", "Latitude", "Longitude", "lat_bin", "lon_bin"]
    ].copy()
    result_dfs.append(primary_df)

    # Case 2: Upper latitude bin (points near latitude boundary)
    lat_boundary_df = data[data["near_lat_boundary"]].copy()
    if not lat_boundary_df.empty:
        lat_boundary_df["lat_bin"] = lat_boundary_df["next_lat_bin"]
        lat_boundary_df = lat_boundary_df[
            ["MMSI", "Timestamp", "Latitude", "Longitude", "lat_bin", "lon_bin"]
        ]
        result_dfs.append(lat_boundary_df)

    # Case 3: Upper longitude bin (points near longitude boundary)
    lon_boundary_df = data[data["near_lon_boundary"]].copy()
    if not lon_boundary_df.empty:
        lon_boundary_df["lon_bin"] = lon_boundary_df["next_lon_bin"]
        lon_boundary_df = lon_boundary_df[
            ["MMSI", "Timestamp", "Latitude", "Longitude", "lat_bin", "lon_bin"]
        ]
        result_dfs.append(lon_boundary_df)

    # Case 4: Diagonal bin (points near both boundaries)
    diagonal_df = data[data["near_lat_boundary"] & data["near_lon_boundary"]].copy()
    if not diagonal_df.empty:
        diagonal_df["lat_bin"] = diagonal_df["next_lat_bin"]
        diagonal_df["lon_bin"] = diagonal_df["next_lon_bin"]
        diagonal_df = diagonal_df[
            ["MMSI", "Timestamp", "Latitude", "Longitude", "lat_bin", "lon_bin"]
        ]
        result_dfs.append(diagonal_df)

    # Combine all DataFrames
    result_df = pd.concat(result_dfs, ignore_index=True)

    return result_df


@timeit
def preprocess_vessel_temporal_data(
    data,
    time_bin_size=TIME_BIN_SIZE,
    time_boundary_threshold=TIME_BOUNDARY_THRESHOLD,
    overlap=True,
):
    """
    Cluster vessel data into temporal bins based on Timestamp.
    Assumes all data are from the same day.

    Parameters:
    data (pd.DataFrame): Input vessel data
    time_bin_size (str): Size of time bins as pandas frequency string (e.g., '1min' for minutes, '1H' for hourly)
    boundary_threshold (int): Threshold in seconds to consider a point near the boundary
    overlap (bool): Whether to handle points near bin boundaries by placing them in multiple bins

    Returns:
    pd.DataFrame: Processed data with time bin assignments
    """
    # Make a copy of the data to avoid modifying the original DataFrame
    data = data.copy()

    # Handle deprecated 'H' in time_bin_size
    if isinstance(time_bin_size, str) and "H" in time_bin_size:
        time_bin_size = time_bin_size.replace("H", "h")

    # Ensure timestamp is in datetime format
    if "# Timestamp" in data.columns:
        data = data.rename(columns={"# Timestamp": "Timestamp"})

    if not pd.api.types.is_datetime64_any_dtype(data["Timestamp"]):
        data["Timestamp"] = pd.to_datetime(
            data["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )

    # Since all data are from the same day, extract the base date once
    base_date = data["Timestamp"].dt.floor("D").iloc[0]

    # Add time-of-day column for easier time bin calculations
    data["time_of_day"] = data["Timestamp"].dt.time

    # Parse time_bin_size to get timedelta
    bin_timedelta = pd.Timedelta(time_bin_size)

    # If overlap is disabled, just assign primary bins and return
    if not overlap:
        # Calculate the primary bin for each point using dt.floor()
        data["time_bin"] = data["Timestamp"].dt.floor(time_bin_size)
        return data

    # Calculate primary time bins (using time-of-day aware flooring)
    data["time_bin"] = data["Timestamp"].dt.floor(time_bin_size)

    # Calculate next time bins
    data["next_time_bin"] = data["time_bin"] + bin_timedelta

    # Handle the case of midnight crossing if bins go to the next day
    midnight = pd.Timestamp(base_date.date() + pd.Timedelta(days=1))
    data.loc[data["next_time_bin"] >= midnight, "next_time_bin"] = (
        midnight - pd.Timedelta(microseconds=1)
    )

    # Calculate distance to next bin boundary (in seconds for easier threshold comparison)
    data["time_distance"] = (
        data["next_time_bin"] - data["Timestamp"]
    ).dt.total_seconds()

    # Boundary threshold is already in seconds, so we use it directly

    # Check if near boundary
    data["near_time_boundary"] = data["time_distance"] <= time_boundary_threshold

    # Create a list to store DataFrames for each case
    result_dfs = []

    # Define columns to select: all original columns plus time_bin
    # (excluding internal calculation columns)
    internal_cols = [
        "next_time_bin",
        "time_distance",
        "near_time_boundary",
        "time_of_day",
    ]
    output_cols = [
        col for col in data.columns if col not in internal_cols and col != "time_bin"
    ] + ["time_bin"]

    # Case 1: Primary bin (all points)
    primary_df = data[output_cols].copy()
    result_dfs.append(primary_df)

    # Case 2: Upper time bin (points near time boundary)
    time_boundary_df = data[data["near_time_boundary"]].copy()
    if not time_boundary_df.empty:
        time_boundary_df["time_bin"] = time_boundary_df["next_time_bin"]
        time_boundary_df = time_boundary_df[output_cols]
        result_dfs.append(time_boundary_df)

    # Combine all DataFrames
    result_df = pd.concat(result_dfs, ignore_index=True)

    return result_df


def preprocess_data(
    data,
    lat_bin_size: float = LAT_BIN_SIZE,
    lon_bin_size: float = LON_BIN_SIZE,
    time_bin_size: str = TIME_BIN_SIZE,
    boundary_threshold: int = TIME_BOUNDARY_THRESHOLD,
):
    """
    Preprocess vessel location data by spatial and temporal binning and create chunks for processing.
    """
    data = preprocess_vessel_spatial_data(
        data, lat_bin_size=lat_bin_size, lon_bin_size=lon_bin_size
    )
    data = preprocess_vessel_temporal_data(
        data, time_bin_size=time_bin_size, time_boundary_threshold=boundary_threshold
    )
    # Group by spatial and temporal bins
    chunks = data.groupby(["lat_bin", "lon_bin", "time_bin"])
    print(f"Created {len(chunks)} bins.")
    return chunks


def process_chunk(chunk_data):
    """
    Process a chunk of vessel location data to find conflicts with exact position and time matches.
    Highly optimized version that only identifies different vessels (MMSIs) at identical positions.

    Parameters:
    chunk_data (pd.DataFrame): DataFrame with vessel location data for a specific bin

    Returns:
    list: List of dictionaries containing pairs of conflicting vessel records
    """
    # Initialize results list
    conflicts = []

    # Round coordinates to a fixed precision to handle minor floating-point differences
    precision = 6
    chunk_data["Lat_rounded"] = chunk_data["Latitude"].round(precision)
    chunk_data["Lon_rounded"] = chunk_data["Longitude"].round(precision)

    # Create a position-time hash for efficient lookup
    chunk_data["pos_time_hash"] = (
        chunk_data["Timestamp"].astype(str)
        + "_"
        + chunk_data["Lat_rounded"].astype(str)
        + "_"
        + chunk_data["Lon_rounded"].astype(str)
    )

    # Find duplicated position-time hashes (exact same position and time)
    # Get counts of each position-time hash
    hash_counts = chunk_data["pos_time_hash"].value_counts()

    # Filter to only include hashes that appear more than once
    conflicting_hashes = hash_counts[hash_counts > 1].index

    # If no conflicts found, return empty list
    if len(conflicting_hashes) == 0:
        return conflicts

    # Filter to only records with conflicting position-time hashes
    potential_conflicts = chunk_data[
        chunk_data["pos_time_hash"].isin(conflicting_hashes)
    ]

    # Group by position-time hash
    for pos_time_hash, group in potential_conflicts.groupby("pos_time_hash"):
        # Only proceed if we have different MMSIs
        if len(group["MMSI"].unique()) > 1:
            # Extract timestamp from first record (all records in group have same timestamp)
            timestamp = group["Timestamp"].iloc[0]

            # Get all pairs of vessels in conflict (with different MMSIs)
            vessels = group[["MMSI", "Latitude", "Longitude"]].values
            for i in range(len(vessels)):
                for j in range(i + 1, len(vessels)):
                    mmsi1, lat1, lon1 = vessels[i]
                    mmsi2, lat2, lon2 = vessels[j]

                    # Skip if same MMSI (shouldn't happen with our groupby, but just to be safe)
                    if mmsi1 == mmsi2:
                        continue

                    # Create conflict record
                    conflict = {
                        "MMSI1": int(mmsi1),
                        "MMSI2": int(mmsi2),
                        "Timestamp1": timestamp,
                        "Timestamp2": timestamp,
                        "Latitude1": lat1,
                        "Longitude1": lon1,
                        "Latitude2": lat2,
                        "Longitude2": lon2,
                        "TimeDifference": 0.0,
                        "DistanceMeters": 0.0,
                    }
                    conflicts.append(conflict)

    return conflicts


@timeit
def detect_conflicting_locations_single_process(
    data: pd.DataFrame,
    lat_bin_size: float = LAT_BIN_SIZE,
    lon_bin_size: float = LON_BIN_SIZE,
    time_bin_size: str = TIME_BIN_SIZE,
    time_boundary_threshold: int = TIME_BOUNDARY_THRESHOLD,
) -> pd.DataFrame:
    """
    Detect vessel locations that are suspiciously close in both space and time
    but correspond to different vessel identifiers (MMSI).

    Single-process implementation that processes spatial bins sequentially.

    Returns:
    pd.DataFrame: DataFrame with pairs of conflicting vessel locations
    """
    chunks = preprocess_data(
        data, lat_bin_size, lon_bin_size, time_bin_size, time_boundary_threshold
    )

    # Initialize results
    all_results = []

    # Process each spatial bin sequentially with progress bar
    with tqdm(total=len(chunks), unit="bin", desc="Processing spatial bins") as pbar:
        for _, chunk_data in chunks:
            chunk_results = process_chunk(chunk_data)
            all_results.extend(chunk_results)
            pbar.update(1)

    # Convert results to DataFrame
    if all_results:
        result_df = pd.DataFrame(all_results)

        # Remove duplicates based on the vessel pairs
        # Two vessels might conflict multiple times, so create a unique ID for each pair
        result_df["pair_id"] = result_df.apply(
            lambda row: tuple(sorted([row["MMSI1"], row["MMSI2"]])), axis=1
        )

        # Get the conflict with the smallest distance for each pair
        result_df = result_df.loc[
            result_df.groupby("pair_id")["DistanceMeters"].idxmin()
        ]

        # Drop the helper column
        result_df = result_df.drop(columns=["pair_id"])

        # Sort results by distance
        result_df = result_df.sort_values("DistanceMeters")

        # Reset index for clean output
        result_df = result_df.reset_index(drop=True)
    else:
        # Create an empty DataFrame with the expected columns if no conflicts found
        result_df = pd.DataFrame(
            columns=[
                "MMSI1",
                "MMSI2",
                "Timestamp1",
                "Timestamp2",
                "Latitude1",
                "Longitude1",
                "Latitude2",
                "Longitude2",
                "TimeDifference",
                "DistanceMeters",
            ]
        )

    print(f"Found {len(result_df)} unique conflicting vessel pairs")
    return result_df


@timeit
def detect_conflicting_locations_multi_process(
    data: pd.DataFrame,
    lat_bin_size: float = LAT_BIN_SIZE,
    lon_bin_size: float = LON_BIN_SIZE,
    time_bin_size: str = TIME_BIN_SIZE,
    time_boundary_threshold: int = TIME_BOUNDARY_THRESHOLD,
) -> pd.DataFrame:
    """
    Detect vessel locations that are suspiciously close in both space and time
    but correspond to different vessel identifiers (MMSI).

    Multi-process implementation that processes spatial bins in parallel.

    Returns:
    pd.DataFrame: DataFrame with pairs of conflicting vessel locations
    """
    # Determine number of processes
    number_of_processes = mp.cpu_count() - 1
    print(f"Using {number_of_processes} processes.")

    chunks = preprocess_data(
        data, lat_bin_size, lon_bin_size, time_bin_size, time_boundary_threshold
    )

    # Group by spatial and temporal bins and extract dataframes only
    chunk_dataframes = []
    for _, chunk_df in chunks:
        chunk_dataframes.append(chunk_df)

    print(f"Processing {len(chunk_dataframes)} bins")

    # Initialize multiprocessing pool
    all_results = []
    with mp.Pool(number_of_processes) as pool:
        # Use imap directly with dataframes
        results_iter = pool.imap(process_chunk, chunk_dataframes)

        # Create a progress bar for the total number of chunks
        with tqdm(
            total=len(chunk_dataframes), unit="bin", desc="Processing spatial bins"
        ) as pbar:
            # Process each result as it completes
            for result in results_iter:
                all_results.extend(result)
                pbar.update(1)

    # Convert results to DataFrame
    if all_results:
        result_df = pd.DataFrame(all_results)

        # Remove duplicates based on the vessel pairs
        # Two vessels might conflict multiple times, so create a unique ID for each pair
        result_df["pair_id"] = result_df.apply(
            lambda row: tuple(sorted([row["MMSI1"], row["MMSI2"]])), axis=1
        )

        # Get the conflict with the smallest distance for each pair
        result_df = result_df.loc[
            result_df.groupby("pair_id")["DistanceMeters"].idxmin()
        ]

        # Drop the helper column
        result_df = result_df.drop(columns=["pair_id"])

        # Sort results by distance
        result_df = result_df.sort_values("DistanceMeters")

        # Reset index for clean output
        result_df = result_df.reset_index(drop=True)
    else:
        # Create an empty DataFrame with the expected columns if no conflicts found
        result_df = pd.DataFrame(
            columns=[
                "MMSI1",
                "MMSI2",
                "Timestamp1",
                "Timestamp2",
                "Latitude1",
                "Longitude1",
                "Latitude2",
                "Longitude2",
                "TimeDifference",
                "DistanceMeters",
            ]
        )

    print(f"Found {len(result_df)} unique conflicting vessel pairs")
    return result_df
