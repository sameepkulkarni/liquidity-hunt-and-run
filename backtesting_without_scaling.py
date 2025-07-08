import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SwingBacktesterWithoutScaling:
    """
    Backtester with vectorized swing detection (minor pivots) and configurable lag/window.
    """
    def __init__(self, data: pd.DataFrame, lag: int, window: int, ):
        """
        :param data: DataFrame with ['o','h','l','c'] indexed by datetime
        :param lag: number of bars to lag swing detection (avoids lookahead)
        :param window: half-window size for centered swing detection
        """
        self.lag = lag
        self.window = window
        self.data = data.copy()
        self.bt = None
        self.results = []
        self.process_all()

    def process_all(self):
        self.compute_vectorized_swings()
        self.map_swing_levels()
        self.calculate_liquidity_grabs()
        self.generate_entry_signals()

    def compute_vectorized_swings(self):
        span = 2 * self.window + 1
        max_h = self.data['h'].rolling(window=span, center=True, min_periods=1).max()
        min_l = self.data['l'].rolling(window=span, center=True, min_periods=1).min()
        self.data['is_swing_high'] = (self.data['h'] == max_h).shift(self.lag)
        self.data['is_swing_low']  = (self.data['l'] == min_l).shift(self.lag)

    def map_swing_levels(self):
        self.data['swing_high_level'] = (
            pd.Series(np.where(self.data['is_swing_high'], self.data['h'], np.nan),index=self.data.index).ffill()
        )
        self.data['swing_low_level'] = (
            pd.Series(np.where(self.data['is_swing_low'], self.data['l'], np.nan),index=self.data.index).ffill()
        )

    def calculate_liquidity_grabs(self):
        conds = [
            self.data['l'] < self.data['swing_low_level'],
            self.data['h'] > self.data['swing_high_level']
        ]
        choices = ['Bearish_Grab', 'Bullish_Grab']
        self.data['liquidity_grab'] = np.select(conds, choices, default=None)

    def generate_entry_signals(self):
        grab_prev = self.data['liquidity_grab'].shift(1)
        bull = (grab_prev == 'Bearish_Grab') & (self.data['c'] > self.data['o'])
        bear = (grab_prev == 'Bullish_Grab') & (self.data['c'] < self.data['o'])
        self.data['entry_signal'] = np.where(bull, 1, np.where(bear, -1, 0))

    def run_backtest(self):
        results = []
        in_position = False
        position = 0
        entry_price = None
        entry_time = None
        SL_price = None
        data = self.data
        for i in range(1, len(data)):
            signal = data['entry_signal'].iloc[i]
            open_price = data['o'].iloc[i]
            high = data['h'].iloc[i]
            low = data['l'].iloc[i]
            time_now = data.index[i]
            entry_candle_height = open_price - low if signal == 1 else high - open_price
            if not in_position and signal != 0:
                position = signal
                entry_price = open_price
                entry_time = time_now
                SL_price = entry_price - (entry_candle_height) if position == 1 else entry_price + (entry_candle_height)
                in_position = True
            elif in_position:
                sl_hit = (low <= SL_price) if position == 1 else (high >= SL_price)
                reverse_signal = (signal != 0 and signal != position)
                if sl_hit:
                    exit_price = SL_price
                    exit_time = time_now
                    pnl = (exit_price - entry_price) * position
                    results.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Direction': 'Long' if position == 1 else 'Short',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'PnL': pnl,
                        'Exit Reason': 'SL Hit',
                        'SL Price': SL_price
                    })
                    in_position = False
                    position = 0
                elif reverse_signal:
                    exit_price = open_price
                    exit_time = time_now
                    pnl = (exit_price - entry_price) * position
                    results.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Direction': 'Long' if position == 1 else 'Short',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'PnL': pnl,
                        'Exit Reason': 'Signal Reversed',
                        'SL Price': SL_price
                    })
                    position = signal
                    entry_price = open_price
                    entry_time = time_now
                    SL_price = low if position == 1 else high
                    in_position = True
        self.results = results
        self.bt = pd.DataFrame(results)
        if not self.bt.empty:
            self.bt['Cumulative PnL'] = self.bt['PnL'].cumsum()
            self.bt['Entry Time'] = pd.to_datetime(self.bt['Entry Time'])
            self.bt['Exit Time'] = pd.to_datetime(self.bt['Exit Time'])
            self.bt['Duration'] = (self.bt['Exit Time'] - self.bt['Entry Time']).dt.total_seconds() / 60

    
        
    def calculate_mae_mfe(self):
        if self.bt is None or self.bt.empty:
            print("Run backtest first.")
            return
        mae_list = []
        mfe_list = []
        for idx, row in self.bt.iterrows():
            entry_time = row['Entry Time']
            exit_time = row['Exit Time']
            entry_price = row['Entry Price']
            direction = 1 if row['Direction'] == 'Long' else -1
            trade_data = self.data.loc[entry_time:exit_time]
            if direction == 1:
                min_low = trade_data['l'].min()
                max_high = trade_data['h'].max()
                mae = min_low - entry_price
                mfe = max_high - entry_price
            else:
                max_high = trade_data['h'].max()
                min_low = trade_data['l'].min()
                mae = entry_price - max_high
                mfe = entry_price - min_low
            mae_list.append(mae * direction)
            mfe_list.append(mfe * direction)
        self.bt['MAE'] = mae_list
        self.bt['MFE'] = mfe_list
    def plot_trades(self, n_trades: int = 5):
        """
        Plot the close price and overlay the first n_trades entry/exit points.
        """
        if self.bt is None or self.bt.empty:
            raise RuntimeError("No trades to plot – run run_backtest() first.")

        # pick the first n_trades
        trades = self.bt.head(n_trades)

        fig, ax = plt.subplots(figsize=(12, 6))
        # plot the close series
        ax.plot(self.data.index, self.data['c'], label='Close')

        # for each trade, plot entry & exit
        for _, row in trades.iterrows():
            t_entry = row['Entry Time']
            t_exit  = row['Exit Time']
            p_entry = row['Entry Price']
            p_exit  = row['Exit Price']
            direction = row['Direction']

            # entry marker: up‑triangle for Long, down‑triangle for Short
            marker_entry = '^' if direction == 'Long' else 'v'
            ax.scatter(t_entry, p_entry,
                       marker=marker_entry,
                       s=100,
                       edgecolors='k',
                       label=f"{direction} Entry" if _ == 0 else "")

            # exit marker: circle
            ax.scatter(t_exit, p_exit,
                       marker='o',
                       s=100,
                       edgecolors='k',
                       label="Exit" if _ == 0 else "")

        ax.set_title(f"First {n_trades} Trades on Price Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        plt.tight_layout()
        plt.show()