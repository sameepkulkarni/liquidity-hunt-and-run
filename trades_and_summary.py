import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import concurrent.futures
import warnings

from backtesting_v1 import SwingBacktesterWithScaling  # assuming the class is in this file/module
from generate_summary import generate_summary
from run_backtest import run_backtest_for_file

warnings.filterwarnings('ignore')


def safe_divide(a, b):
    return a / b if b != 0 else np.nan


def main():
    input_dir = "./data"
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    for f in glob.glob(os.path.join(output_dir, "*.csv")):
        os.remove(f)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    window_values = [1,2, 3, 4, 5,6,7,8]  # test multiple swing window sizes
    lag_values = [1, 2, 3,4,5,6,7,8,9,10,11,12]
    max_workers = max(1, os.cpu_count())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for lag in lag_values:
            for window in window_values:
                for csv_file in csv_files:
                    futures.append(
                        executor.submit(run_backtest_for_file, csv_file, window, lag, output_dir)
                    )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error: {e}")

    summary_path = os.path.join("./Summary", "Backtesting_results_v1.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    generate_summary(output_dir, summary_path)

if __name__ == "__main__":
    main()
