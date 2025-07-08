import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SwingBacktesterWithScaling:
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

        # 1) center=True so each max/min at t uses [t-window ... t+window]
        # 2) shift by (window + lag) so the pivot only flags at t + window + lag
        max_h = (
            self.data['h']
                .rolling(window=span, center=True, min_periods=1)
                .max()
                .shift(self.window + self.lag)
        )
        min_l = (
            self.data['l']
                .rolling(window=span, center=True, min_periods=1)
                .min()
                .shift(self.window + self.lag)
        )

        self.data['is_swing_high'] = (self.data['h'] == max_h)
        self.data['is_swing_low']  = (self.data['l'] == min_l)

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
    # previously you did shift(1).  Now you want to shift by exactly window+lag+1
        grab_prev = self.data['liquidity_grab'].shift(self.window + self.lag + 1)

        bull = (grab_prev == 'Bearish_Grab') & (self.data['c'] > self.data['o'])
        bear = (grab_prev == 'Bullish_Grab') & (self.data['c'] < self.data['o'])

        self.data['entry_signal'] = np.where(bull, 1,
                                    np.where(bear, -1, 0))
    def run_backtest(self):
        results = []
        in_position = False
        position = 0
        bullish_entry_prices = []
        bullish_entry_times = []
        bearish_entry_prices = []
        bearish_entry_times = []
        bullish_SL_price = None
        bearish_SL_price = None
        data = self.data

        for i in range(1, len(data)):
            signal = data['entry_signal'].iloc[i]
            open_price = data['o'].iloc[i]
            high = data['h'].iloc[i]
            low = data['l'].iloc[i]
            time_now = data.index[i]

            swing_low_flag = data['is_swing_low'].iloc[i - 1] if i - 1 >= 0 else False
            recent_swing_low = data['l'].iloc[i - 1] if swing_low_flag else None
            swing_high_flag = data['is_swing_high'].iloc[i - 1] if i - 1 >= 0 else False
            recent_swing_high = data['h'].iloc[i - 1] if swing_high_flag else None

            entry_candle_height = abs(open_price - low) if signal == 1 else abs(high - open_price) if signal == -1 else None

            if not in_position:
                if signal == 1:
                    position = 1
                    bullish_entry_prices = [open_price]
                    bullish_entry_times = [time_now]
                    bullish_SL_price = open_price - (entry_candle_height)
                    in_position = True
                elif signal == -1:
                    position = -1
                    bearish_entry_prices = [open_price]
                    bearish_entry_times = [time_now]
                    bearish_SL_price = open_price + (entry_candle_height)
                    in_position = True

            elif in_position:
                if position == 1:
                    if swing_low_flag:
                        bullish_entry_prices.append(open_price)
                        bullish_entry_times.append(time_now)
                        bullish_SL_price = recent_swing_low

                    if low <= bullish_SL_price and signal != -1:
                        avg_price = np.mean(bullish_entry_prices)
                        pnl = (bullish_SL_price - avg_price) * len(bullish_entry_prices)
                        results.append({
                            'Entry Time': bullish_entry_times[0],
                            'Exit Time': time_now,
                            'Direction': 'Long',
                            'Entry Price': avg_price,
                            'Exit Price': bullish_SL_price,
                            'PnL': pnl,
                            'Exit Reason': 'SL Hit',
                            'SL Price': bullish_SL_price,
                            'Units': len(bullish_entry_prices)
                        })
                        in_position = False
                        position = 0
                        bullish_entry_prices = []
                        bullish_entry_times = []
                        bullish_SL_price = None
                    elif signal == -1:
                        avg_price = np.mean(bullish_entry_prices)
                        pnl = (bullish_SL_price - avg_price) * len(bullish_entry_prices)
                        results.append({
                            'Entry Time': bullish_entry_times[0],
                            'Exit Time': time_now,
                            'Direction': 'Long',
                            'Entry Price': avg_price,
                            'Exit Price': bullish_SL_price,
                            'PnL': pnl,
                            'Exit Reason': 'Bearish Trade Reversal',
                            'SL Price': bullish_SL_price,
                            'Units': len(bullish_entry_prices)
                        })
                        position = -1
                        bearish_entry_prices = [open_price]
                        bearish_entry_times = [time_now]
                        bearish_SL_price = open_price + (entry_candle_height)
                        in_position = True

                elif position == -1:
                    if swing_high_flag:
                        bearish_entry_prices.append(open_price)
                        bearish_entry_times.append(time_now)
                        bearish_SL_price = recent_swing_high

                    if high >= bearish_SL_price and signal != 1:
                        avg_price = np.mean(bearish_entry_prices)
                        pnl = (avg_price - bearish_SL_price) * len(bearish_entry_prices)
                        results.append({
                            'Entry Time': bearish_entry_times[0],
                            'Exit Time': time_now,
                            'Direction': 'Short',
                            'Entry Price': avg_price,
                            'Exit Price': bearish_SL_price,
                            'PnL': pnl,
                            'Exit Reason': 'SL Hit',
                            'SL Price': bearish_SL_price,
                            'Units': len(bearish_entry_prices)
                        })
                        in_position = False
                        position = 0
                        bearish_entry_prices = []
                        bearish_entry_times = []
                        bearish_SL_price = None
                    elif signal == 1:
                        avg_price = np.mean(bearish_entry_prices)
                        pnl = (avg_price - bearish_SL_price) * len(bearish_entry_prices)
                        results.append({
                            'Entry Time': bearish_entry_times[0],
                            'Exit Time': time_now,
                            'Direction': 'Short',
                            'Entry Price': avg_price,
                            'Exit Price': bearish_SL_price,
                            'PnL': pnl,
                            'Exit Reason': 'Bullish Trade Reversal',
                            'SL Price': bearish_SL_price,
                            'Units': len(bearish_entry_prices)
                        })
                        position = 1
                        bullish_entry_prices = [open_price]
                        bullish_entry_times = [time_now]
                        bullish_SL_price = open_price - (entry_candle_height)
                        in_position = True

        self.results = results
        self.bt = pd.DataFrame(results)
        print(f"✅ Total trades generated: {len(self.bt)}")

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
