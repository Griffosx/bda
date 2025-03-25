import pandas as pd
import numpy as np
import networkx as nx
from geopy import distance as geopy_distance
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from utils.timer_wrapper import timeit


def process_spatial_chunk(chunk_data, distance_threshold=100, time_threshold=30):
    """
    Process a spatial chunk of data to find conflicting vessel locations.
    This function is designed to be run in parallel.

    Parameters:
    chunk_data (pd.DataFrame): DataFrame containing vessel data for a specific spatial bin
    distance_threshold (float): Maximum distance in meters to consider conflicting
    time_threshold (float): Maximum time difference in seconds to consider conflicting

    Returns:
    list: List of dictionaries containing conflicting vessel pairs
    """
    # Skip if only one vessel in this area
    if chunk_data["MMSI"].nunique() <= 1:
        return []

    # Create time bins (1-minute intervals)
    min_time = chunk_data["Timestamp"].min()
    max_time = chunk_data["Timestamp"].max()

    # Generate minute intervals with buffer
    start_times = pd.date_range(
        start=min_time - pd.Timedelta(seconds=10), end=max_time, freq="1min"
    )

    conflict_results = []

    # Process each time interval
    for start_time in start_times:
        end_time = start_time + pd.Timedelta(seconds=70)  # 1 min + 10 sec buffer

        # Get data points in this time window
        time_window_data = chunk_data[
            (chunk_data["Timestamp"] >= start_time)
            & (chunk_data["Timestamp"] <= end_time)
        ]

        # Skip if only one vessel in this time window
        if time_window_data["MMSI"].nunique() <= 1:
            continue

        # Use NetworkX for finding conflicts
        G = nx.Graph()

        # Add nodes
        for idx, row in time_window_data.iterrows():
            G.add_node(
                idx,
                mmsi=row["MMSI"],
                lat=row["Latitude"],
                lon=row["Longitude"],
                timestamp=row["Timestamp"],
            )

        # Add edges between different vessels that are close in time
        node_list = list(G.nodes(data=True))
        for i in range(len(node_list)):
            node1_id, node1_data = node_list[i]

            for j in range(i + 1, len(node_list)):
                node2_id, node2_data = node_list[j]

                # Skip if same vessel
                if node1_data["mmsi"] == node2_data["mmsi"]:
                    continue

                # Calculate time difference
                time_diff = abs(
                    (node1_data["timestamp"] - node2_data["timestamp"]).total_seconds()
                )

                # Skip if time difference > threshold
                if time_diff > time_threshold:
                    continue

                # Calculate spatial distance
                point1 = (node1_data["lat"], node1_data["lon"])
                point2 = (node2_data["lat"], node2_data["lon"])

                # Calculate distance in meters
                dist_meters = geopy_distance.distance(point1, point2).meters

                # Add edge if distance < threshold
                if dist_meters < distance_threshold:
                    G.add_edge(
                        node1_id, node2_id, distance=dist_meters, time_diff=time_diff
                    )

                    # Add to results
                    conflict_pair = {
                        "mmsi1": node1_data["mmsi"],
                        "mmsi2": node2_data["mmsi"],
                        "lat1": node1_data["lat"],
                        "lon1": node1_data["lon"],
                        "lat2": node2_data["lat"],
                        "lon2": node2_data["lon"],
                        "timestamp1": node1_data["timestamp"],
                        "timestamp2": node2_data["timestamp"],
                        "distance_meters": dist_meters,
                        "time_diff_seconds": time_diff,
                    }
                    conflict_results.append(conflict_pair)

    return conflict_results


"""
The `preprocess_vessel_data` function:

It receives a dataframe that contains vessel tracking information, typically with columns:
- `MMSI`: Maritime Mobile Service Identity (unique vessel identifier)
- `Timestamp` (or `# Timestamp`): Time of the position report
- `Latitude`: Geographical latitude position
- `Longitude`: Geographical longitude position

It returns a dataframe with the same information plus:
- Standardized timestamps in datetime format
- Added `lat_bin` and `lon_bin` columns that assign each vessel position to geographic grid cells
- Vessels near bin boundaries appear in multiple bins to ensure no vessel-to-vessel proximity comparisons are missed

This preprocessing is essential for efficiently detecting potential vessel conflicts by limiting comparisons to only vessels in the same or adjacent spatial regions.
"""


def preprocess_vessel_data(
    data, lat_bin_size=0.01, lon_bin_size=0.01, boundary_threshold=0.1, overlap=True
):
    """
    Common preprocessing steps for vessel data.

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
        # Calculate the primary bin for each point and round to 2 decimals to ensure consistency
        data["lat_bin"] = (
            np.floor(data["Latitude"] / lat_bin_size) * lat_bin_size
        ).round(2)
        data["lon_bin"] = (
            np.floor(data["Longitude"] / lon_bin_size) * lon_bin_size
        ).round(2)
        return data

    # Results container
    result_rows = []

    # Thresholds for boundary proximity
    lat_threshold = lat_bin_size * boundary_threshold
    lon_threshold = lon_bin_size * boundary_threshold

    for _, row in data.iterrows():
        # Get original values
        original_values = {
            "MMSI": row["MMSI"],
            "Timestamp": row["Timestamp"],
            "Latitude": row["Latitude"],
            "Longitude": row["Longitude"],
        }

        # Calculate primary bin (rounded to 2 decimal places)
        lat_bin = round(np.floor(row["Latitude"] / lat_bin_size) * lat_bin_size, 2)
        lon_bin = round(np.floor(row["Longitude"] / lon_bin_size) * lon_bin_size, 2)

        # Calculate next bins (rounded to 2 decimal places to ensure exact 50.02, not 50.019999...)
        next_lat_bin = round(lat_bin + lat_bin_size, 2)
        next_lon_bin = round(lon_bin + lon_bin_size, 2)

        # Distance to next bin boundaries
        lat_distance = round(next_lat_bin - row["Latitude"], 10)
        lon_distance = round(next_lon_bin - row["Longitude"], 10)

        # Check if near boundaries (with a tiny epsilon for floating point comparison)
        near_lat_boundary = lat_distance <= (lat_threshold + 1e-10)
        near_lon_boundary = lon_distance <= (lon_threshold + 1e-10)

        # Always add to primary bin
        bin_combinations = [(lat_bin, lon_bin)]

        # If near latitude boundary, add to upper latitude bin
        if near_lat_boundary:
            bin_combinations.append((next_lat_bin, lon_bin))

        # If near longitude boundary, add to upper longitude bin
        if near_lon_boundary:
            bin_combinations.append((lat_bin, next_lon_bin))

        # If near both boundaries, add to diagonal bin
        if near_lat_boundary and near_lon_boundary:
            bin_combinations.append((next_lat_bin, next_lon_bin))

        # Create a row for each bin combination
        for lat_bin_val, lon_bin_val in bin_combinations:
            row_data = original_values.copy()
            row_data["lat_bin"] = lat_bin_val
            row_data["lon_bin"] = lon_bin_val
            result_rows.append(row_data)

    # Convert to DataFrame
    result_df = pd.DataFrame(result_rows)

    return result_df


@timeit
def detect_conflicting_locations_single_process(
    data: pd.DataFrame,
    distance_threshold: float = 100,  # meters
    time_threshold: float = 30,  # seconds
    lat_bin_size: float = 0.01,  # ~1.1km at equator
    lon_bin_size: float = 0.01,  # varies with latitude
) -> pd.DataFrame:
    """
    Detect vessel locations that are suspiciously close in both space and time
    but correspond to different vessel identifiers (MMSI).

    Single-process implementation that processes spatial bins sequentially.

    Parameters:
    data (pd.DataFrame): DataFrame with vessel location data including 'MMSI',
                        'Timestamp', 'Latitude', 'Longitude' columns
    distance_threshold (float): Maximum distance in meters to consider conflicting
    time_threshold (float): Maximum time difference in seconds to consider conflicting

    Returns:
    pd.DataFrame: DataFrame with pairs of conflicting vessel locations
    """
    # Preprocess the data
    data = preprocess_vessel_data(
        data, lat_bin_size=lat_bin_size, lon_bin_size=lon_bin_size
    )

    # Group by spatial bins
    spatial_groups = data.groupby(["lat_bin", "lon_bin"])
    print(f"Processing {len(spatial_groups)} spatial bins")

    # Initialize results
    all_results = []

    # Process each spatial bin sequentially with progress bar
    with tqdm(
        total=len(spatial_groups), unit="bin", desc="Processing spatial bins"
    ) as pbar:
        for _, chunk_data in spatial_groups:
            chunk_results = process_spatial_chunk(
                chunk_data, distance_threshold, time_threshold
            )
            all_results.extend(chunk_results)
            pbar.update(1)

    # If no conflicts found, return empty DataFrame
    if not all_results:
        return pd.DataFrame(
            columns=[
                "mmsi1",
                "mmsi2",
                "lat1",
                "lon1",
                "lat2",
                "lon2",
                "timestamp1",
                "timestamp2",
                "distance_meters",
                "time_diff_seconds",
            ]
        )

    # Convert results to DataFrame and remove duplicates
    result_df = pd.DataFrame(all_results)

    # Remove duplicates by standardizing vessel ordering
    result_df["mmsi_min"] = result_df[["mmsi1", "mmsi2"]].min(axis=1)
    result_df["mmsi_max"] = result_df[["mmsi1", "mmsi2"]].max(axis=1)
    result_df["minute1"] = result_df["timestamp1"].dt.floor("min")
    result_df["minute2"] = result_df["timestamp2"].dt.floor("min")

    # Keep only unique combinations
    result_df = result_df.drop_duplicates(
        subset=["mmsi_min", "mmsi_max", "minute1", "minute2"]
    )

    # Drop helper columns
    result_df = result_df.drop(columns=["mmsi_min", "mmsi_max", "minute1", "minute2"])

    print(f"Found {len(result_df)} unique conflicting vessel pairs")
    return result_df


@timeit
def detect_conflicting_locations_multi_process(
    data: pd.DataFrame,
    distance_threshold: float = 100,  # meters
    time_threshold: float = 30,  # seconds
    lat_bin_size: float = 0.01,  # ~1.1km at equator
    lon_bin_size: float = 0.01,  # varies with latitude
) -> pd.DataFrame:
    """
    Detect vessel locations that are suspiciously close in both space and time
    but correspond to different vessel identifiers (MMSI).

    Multi-process implementation that processes spatial bins in parallel.

    Parameters:
    data (pd.DataFrame): DataFrame with vessel location data including 'MMSI',
                        'Timestamp', 'Latitude', 'Longitude' columns
    distance_threshold (float): Maximum distance in meters to consider conflicting
    time_threshold (float): Maximum time difference in seconds to consider conflicting

    Returns:
    pd.DataFrame: DataFrame with pairs of conflicting vessel locations
    """
    # Determine the number of processes
    number_of_processes = mp.cpu_count() - 1
    print(f"Using {number_of_processes} processes.")

    # Preprocess the data
    data = preprocess_vessel_data(data)

    # Group by spatial bins
    spatial_groups = data.groupby(["lat_bin", "lon_bin"])
    print(f"Processing {len(spatial_groups)} spatial bins")

    # Prepare data chunks for parallel processing
    chunks = [group for _, group in spatial_groups]

    # Define the process function with fixed parameters
    process_fn = partial(
        process_spatial_chunk,
        distance_threshold=distance_threshold,
        time_threshold=time_threshold,
    )

    # Process chunks in parallel
    all_results = []
    with mp.Pool(number_of_processes) as pool:
        # Map function to all chunks
        results_iter = pool.imap(process_fn, chunks)

        # Process results with progress bar
        with tqdm(total=len(chunks), desc="Processing spatial bins") as pbar:
            for chunk_results in results_iter:
                all_results.extend(chunk_results)
                pbar.update(1)

    # If no conflicts found, return empty DataFrame
    if not all_results:
        return pd.DataFrame(
            columns=[
                "mmsi1",
                "mmsi2",
                "lat1",
                "lon1",
                "lat2",
                "lon2",
                "timestamp1",
                "timestamp2",
                "distance_meters",
                "time_diff_seconds",
            ]
        )

    # Convert results to DataFrame
    result_df = pd.DataFrame(all_results)

    # Remove duplicates by standardizing vessel ordering
    result_df["mmsi_min"] = result_df[["mmsi1", "mmsi2"]].min(axis=1)
    result_df["mmsi_max"] = result_df[["mmsi1", "mmsi2"]].max(axis=1)
    result_df["minute1"] = result_df["timestamp1"].dt.floor("min")
    result_df["minute2"] = result_df["timestamp2"].dt.floor("min")

    # Keep only unique combinations
    result_df = result_df.drop_duplicates(
        subset=["mmsi_min", "mmsi_max", "minute1", "minute2"]
    )

    # Drop helper columns
    result_df = result_df.drop(columns=["mmsi_min", "mmsi_max", "minute1", "minute2"])

    print(f"Found {len(result_df)} unique conflicting vessel pairs")
    return result_df
