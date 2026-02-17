import argparse
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute average time for two CSVs and show SW/HW speedup.")
    p.add_argument("sw_csv", nargs="?", help="CSV file for SW times (must have column 'time').")
    p.add_argument("hw_csv", nargs="?", help="CSV file for HW times (must have column 'time').")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sw_data = pd.read_csv(args.sw_csv)
    hw_data = pd.read_csv(args.hw_csv)

    mean_sw_time = sw_data["time"].mean()
    mean_hw_time = hw_data["time"].mean()

    speedup = mean_sw_time / mean_hw_time

    print(f"Mean SW time: {float(mean_sw_time)}")
    print(f"Mean HW time: {float(mean_hw_time)}")

    print(f"Speedup: {float(speedup) * 100}%")

    from scipy.stats import gmean


    geomean_speedup = gmean(sw_data["time"].div(hw_data["time"]))
    print(f"Geomean Speedup: {geomean_speedup}")
