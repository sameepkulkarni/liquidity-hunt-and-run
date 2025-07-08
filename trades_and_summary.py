import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import concurrent.futures
import warnings

from backtesting_without_scaling import SwingBacktesterWithoutScaling  # assuming the class is in this file/module
from generate_summary import generate_summary
from run_backtest import run_backtest_for_params

warnings.filterwarnings('ignore')

def main():
    input_dir = "./data"
    output_dir = "./results_without_scaling"
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Load the data once
    df_path = os.path.join(input_dir, "gold.csv")
    df = pd.read_csv(df_path)

    # ✅ Preprocess timestamp and select relevant columns
    if df['t'].iloc[0] > 1e12:  # UNIX ms timestamp
        df['t'] = pd.to_datetime(df['t'], unit='ms')
    else:
        df['t'] = pd.to_datetime(df['t'], format='%d-%m-%Y %H:%M')

    df.set_index('t', inplace=True)
    df = df[['o', 'h', 'l', 'c']]  # Keep only OHLC

    window_values = [1, 2, 3, 4, 5, 6, 7, 8]
    lag_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    max_workers = max(1, os.cpu_count())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for lag in lag_values:
            for window in window_values:
                # ✅ Pass the same df to all threads
                futures.append(executor.submit(run_backtest_for_params, df, window, lag, output_dir))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error: {e}")

    summary_path = os.path.join("./Summary", "Backtesting_results_without_scaling.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    generate_summary(output_dir, summary_path)

if __name__ == "__main__":
    main()
