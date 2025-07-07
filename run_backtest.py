import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import concurrent.futures
import warnings
from backtesting_v1 import SwingBacktesterWithScaling  # assuming the class is in this file/module


def run_backtest_for_file(df_path, window, lag, output_dir):
    symbol_name = os.path.basename(df_path).replace(".csv", "")
    print(f"📂 Processing: {df_path.replace('\\', '/')}")

    df = pd.read_csv(df_path)

    # Convert 't' column correctly depending on format
    if df['t'].iloc[0] > 1e12:  # likely ms timestamp
        df['t'] = pd.to_datetime(df['t'], unit='ms')
    else:  # likely string timestamp like "14-06-2023 00:00"
        df['t'] = pd.to_datetime(df['t'], format='%d-%m-%Y %H:%M')

    df.set_index('t', inplace=True)
    df = df[['o', 'h', 'l', 'c']]

    bt = SwingBacktesterWithScaling(data=df, lag=lag, window=window)
    bt.run_backtest()

    if bt.bt is not None and not bt.bt.empty:
        bt.calculate_mae_mfe()
        output_csv = os.path.join(output_dir, f"{symbol_name}_lag{lag}_win{window}.csv")
        bt.bt.to_csv(output_csv, index=False)
        print(f"📅 Saved to: {output_csv}")
    else:
        print(f"⚠️ No trades generated for {symbol_name} (Lag: {lag}, Window: {window})")
