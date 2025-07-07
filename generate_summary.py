import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import concurrent.futures
import warnings


from backtesting_v1 import SwingBacktesterWithScaling  # assuming the class is in this file/module

def generate_summary(trades_folder, output_file):
    pattern = os.path.join(trades_folder, "*.csv")
    summary_rows = []

    for file in glob.glob(pattern):
        filename = os.path.basename(file)
        parts = filename.replace(".csv", "").split("_")
        if len(parts) < 4:
            continue

        try:
            lag = int(parts[1].replace("lag", ""))
            window = int(parts[2].replace("win", ""))
        except ValueError:
            continue

        df = pd.read_csv(file)
        if df.empty:
            continue

        total_trades = len(df)
        win_trades = (df['PnL'] > 0).sum()
        win_rate = win_trades / total_trades * 100
        total_pnl = df['PnL'].sum()
        avg_pnl = df['PnL'].mean()
        max_drawdown = (df['Cumulative PnL'].cummax() - df['Cumulative PnL']).max()
        avg_win = df.loc[df['PnL'] > 0, 'PnL'].mean()
        avg_loss = df.loc[df['PnL'] < 0, 'PnL'].mean()
        rr = abs(avg_win) / abs(avg_loss) if avg_loss != 0 else float('inf')
        avg_mae = df['MAE'].mean()
        max_mae = df['MAE'].min()
        avg_mfe = df['MFE'].mean()
        max_mfe = df['MFE'].max()
        winning_pnl = df.loc[df['PnL'] > 0, 'PnL'].sum()
        losing_pnl = df.loc[df['PnL'] < 0, 'PnL'].sum()
        efficiency = (df['PnL'] / df['MFE'].replace(0, float('nan'))) * 100
        avg_efficiency = efficiency.mean()
        total_units = df['Units'].sum()

        sharpe_like = safe_divide(avg_pnl, np.std([avg_win, abs(avg_loss)]))
        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss
        normalized_pnl = safe_divide(total_pnl, total_trades)
        pnl_per_unit = safe_divide(total_pnl, total_units)
        efficiency_ratio = safe_divide(avg_pnl, abs(avg_mae))

        downside_returns = df[df['PnL'] < 0]['PnL']
        downside_deviation = downside_returns.std() if not downside_returns.empty else 0
        sortino_ratio = safe_divide(avg_pnl, downside_deviation)

        summary_rows.append({
            'Lag': lag,
            'Window': window,
            'Total Trades': total_trades,
            'Total Units': total_units,
            'Win Rate (%)': win_rate,
            'Total PnL': total_pnl,
            'Average PnL': avg_pnl,
            'Max Drawdown': max_drawdown,
            'Risk-Reward': rr,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Average MAE': avg_mae,
            'Max MAE': max_mae,
            'Average MFE': avg_mfe,
            'Max MFE': max_mfe,
            'Winning PnL': winning_pnl,
            'Losing PnL': losing_pnl,
            'Avg Trade Efficiency (%)': avg_efficiency,
            'Sharpe-like Ratio': sharpe_like,
            'Sortino Ratio': sortino_ratio,
            'Expectancy': expectancy,
            'Normalized PnL': normalized_pnl,
            'PnL per Unit': pnl_per_unit,
            'Efficiency Ratio': efficiency_ratio
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by=["Lag", "Window"], inplace=True)
    summary_df.to_csv(output_file, index=False)
    print(f"\n🚀 Final summary saved to {output_file}")
