import pandas as pd
from pathlib import Path
from assignment_1.vessel_anomalies import detect_vessel_anomalies_multi_process


BASE_PATH = Path("assets").resolve()


def main():
    pd.set_option("display.max_rows", None)

    data = pd.read_csv(
        BASE_PATH / "test.csv",
        usecols=["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG"],
    )

    # r1 = detect_vessel_anomalies_single_process(data)
    vessel_anomalies = detect_vessel_anomalies_multi_process(data)

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
