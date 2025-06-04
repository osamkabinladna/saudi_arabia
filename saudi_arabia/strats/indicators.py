import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class HurstGoldSilverStrategy:
    def __init__(self,
                 hurst_window: int = 100,
                 signal_window: int = 20,
                 hurst_threshold: float = 0.5,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.5,
                 transaction_cost: float = 0.0001):
        """
        Args:
            hurst_window: Window size for Hurst exponent calculation
            signal_window: Window size for z-score calculation
            hurst_threshold: Hurst threshold for mean reversion (< 0.5 = mean reverting)
            zscore_entry: Z-score threshold for trade entry
            zscore_exit: Z-score threshold for trade exit
            transaction_cost: Transaction cost per trade (as fraction)
        """
        self.hurst_window = hurst_window
        self.signal_window = signal_window
        self.hurst_threshold = hurst_threshold
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.transaction_cost = transaction_cost

    def calculate_hurst_exponent(self, ts: np.array) -> float:
        if len(ts) < 10:
            return 0.5

        # Demean and calculate cumsum
        ts = ts - np.mean(ts)
        cumsum = np.cumsum(ts)

        # Calculate R/S for different lags
        lags = range(2, min(len(ts) // 2, 50))
        rs_values = []

        for lag in lags:
            n_windows = len(ts) // lag
            if n_windows < 2:
                continue

            rs_lag = []
            for i in range(n_windows):
                start = i * lag
                end = start + lag

                window = ts[start:end]
                cumsum_window = cumsum[start:end] - cumsum[start]

                R = np.max(cumsum_window) - np.min(cumsum_window)
                S = np.std(window, ddof=1) if len(window) > 1 else 1e-8

                if S > 1e-8:
                    rs_lag.append(R / S)

            if rs_lag:
                rs_values.append((lag, np.mean(rs_lag)))

        if len(rs_values) < 3:
            return 0.5

        lags_log = np.log([x[0] for x in rs_values])
        rs_log = np.log([x[1] for x in rs_values if x[1] > 0])

        if len(lags_log) != len(rs_log) or len(rs_log) < 3:
            return 0.5

        try:
            hurst = np.polyfit(lags_log, rs_log, 1)[0]
            return max(0.1, min(0.9, hurst))  # Bound between 0.1 and 0.9
        except:
            return 0.5

    def calculate_spread(self, gold_prices: pd.Series, silver_prices: pd.Series) -> pd.Series:
        return gold_prices / silver_prices

    def calculate_zscore(self, series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Hurst exponent and z-score.
        """
        df = data.copy()

        df['spread'] = self.calculate_spread(df['Gold'], df['Silver'])
        df['hurst'] = np.nan
        for i in range(self.hurst_window, len(df)):
            spread_window = df['spread'].iloc[i - self.hurst_window:i].values
            df.loc[df.index[i], 'hurst'] = self.calculate_hurst_exponent(spread_window)

        df['zscore'] = self.calculate_zscore(df['spread'], self.signal_window)

        df['regime'] = df['hurst'] < self.hurst_threshold

        df['position'] = 0
        df['signal'] = 0

        position = 0

        for i in range(1, len(df)):
            if pd.isna(df.iloc[i]['zscore']) or pd.isna(df.iloc[i]['hurst']):
                df.iloc[i, df.columns.get_loc('position')] = position
                continue

            zscore = df.iloc[i]['zscore']
            is_mean_reverting = df.iloc[i]['regime']

            if is_mean_reverting and position == 0:
                if zscore > self.zscore_entry:
                    # short spread (short gold, long silver)
                    position = -1
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                elif zscore < -self.zscore_entry:
                    # long spread (long gold, short silver)
                    position = 1
                    df.iloc[i, df.columns.get_loc('signal')] = 1

            # Exit signals
            elif position != 0:
                if (position == 1 and zscore > -self.zscore_exit) or \
                        (position == -1 and zscore < self.zscore_exit) or \
                        not is_mean_reverting:
                    # Exit position
                    df.iloc[i, df.columns.get_loc('signal')] = -position
                    position = 0
            df.iloc[i, df.columns.get_loc('position')] = position

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate spread returns
        df['spread_return'] = df['spread'].pct_change()

        # Calculate strategy returns (position * spread_return)
        df['strategy_return'] = df['position'].shift(1) * df['spread_return']

        # Apply transaction costs
        df['trades'] = df['signal'].abs()
        df['transaction_costs'] = df['trades'] * self.transaction_cost
        df['net_strategy_return'] = df['strategy_return'] - df['transaction_costs']

        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['net_strategy_return'].fillna(0)).cumprod()
        df['spread_cumulative'] = (1 + df['spread_return'].fillna(0)).cumprod()

        return df

    def backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        results = self.generate_signals(data)
        results = self.calculate_returns(results)
        metrics = self.calculate_performance_metrics(results)
        return results, metrics

    def calculate_performance_metrics(self, df: pd.DataFrame) -> dict:
        returns = df['net_strategy_return'].dropna()

        if len(returns) == 0:
            return {}

        total_return = df['cumulative_return'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 * 288 / len(df)) - 1  # 5-min bars
        volatility = returns.std() * np.sqrt(252 * 288)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        cumulative = df['cumulative_return']
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        trades = df[df['signal'] != 0]
        num_trades = len(trades)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        mean_hurst = df['hurst'].mean()
        mean_reverting_pct = (df['regime'].sum() / len(df)) * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'mean_hurst': mean_hurst,
            'mean_reverting_regime_pct': mean_reverting_pct
        }



# To use with your data:
# results, metrics = run_backtest_example(data)