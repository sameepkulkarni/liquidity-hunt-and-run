import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import concurrent.futures
import warnings
from backtesting_without_scaling import SwingBacktesterWithoutScaling  # assuming the class is in this file/module
# ✅ Backtest function accepts DataFrame as argument
def run_backtest_for_params(df, window, lag, output_dir):
    symbol_name = "gold"  # From df_path, hardcoded here

    bt = SwingBacktesterWithoutScaling(data=df.copy(), lag=lag, window=window)
    bt.run_backtest()

    if bt.bt is not None and not bt.bt.empty:
        bt.calculate_mae_mfe()
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{symbol_name}_lag{lag}_win{window}.csv")
        bt.bt.to_csv(output_csv, index=False)
        print(f"📅 Saved to: {output_csv}")
    else:
        print(f"⚠️ No trades generated for {symbol_name} (Lag: {lag}, Window: {window})")
