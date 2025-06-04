import pandas as pd
from typing import Tuple
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class HurstGoldSilverStrategy:
    def __init__(self,
                 hurst_window: int = 100,
                 signal_window: int = 20,
                 hurst_threshold: float = 0.5,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.5,
                 transaction_cost: float = 0.0001,
                 initial_capital: float = 100000):
        """
        Args:
            hurst_window: Window size for Hurst exponent calculation
            signal_window: Window size for z-score calculation
            hurst_threshold: Hurst threshold for mean reversion (< 0.5 = mean reverting)
            zscore_entry: Z-score threshold for trade entry
            zscore_exit: Z-score threshold for trade exit
            transaction_cost: Transaction cost per trade (as fraction)
            initial_capital: Starting capital for dollar-neutral positioning
        """
        self.hurst_window = hurst_window
        self.signal_window = signal_window
        self.hurst_threshold = hurst_threshold
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

    def calculate_hurst_exponent(self, ts: np.array) -> float:
        if len(ts) < 10:
            return 0.5

        ts = ts - np.mean(ts)
        cumsum = np.cumsum(ts)

        # Calculate R/S for different lags
        lags = range(2, min(len(ts) // 2, 50))
        rs_values = []

        for lag in lags:
            # Split series into non-overlapping windows
            n_windows = len(ts) // lag
            if n_windows < 2:
                continue

            rs_lag = []
            for i in range(n_windows):
                start = i * lag
                end = start + lag

                window = ts[start:end]
                cumsum_window = cumsum[start:end] - cumsum[start]

                # Range
                R = np.max(cumsum_window) - np.min(cumsum_window)

                # Standard deviation
                S = np.std(window, ddof=1) if len(window) > 1 else 1e-8

                if S > 1e-8:
                    rs_lag.append(R / S)

            if rs_lag:
                rs_values.append((lag, np.mean(rs_lag)))

        if len(rs_values) < 3:
            return 0.5

        # Linear regression to find Hurst exponent
        lags_log = np.log([x[0] for x in rs_values])
        rs_log = np.log([x[1] for x in rs_values if x[1] > 0])

        if len(lags_log) != len(rs_log) or len(rs_log) < 3:
            return 0.5

        try:
            hurst = np.polyfit(lags_log, rs_log, 1)[0]
            return max(0.1, min(0.9, hurst))  # Bound between 0.1 and 0.9
        except:
            return 0.5

    def backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        # Generate signals
        results = self.generate_signals(data)

        # Calculate returns
        results = self.calculate_returns(results)

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results)

        return results, metrics

    def calculate_dollar_neutral_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate notional exposure (half of capital per leg)
        notional_per_leg = self.initial_capital / 2

        # Calculate shares for each asset to maintain dollar neutrality
        df['gold_shares'] = (notional_per_leg / df['Gold']) * df['position']
        df['silver_shares'] = -(notional_per_leg / df['Silver']) * df['position']  # Opposite position

        # Calculate dollar exposures (should be equal and opposite)
        df['gold_exposure'] = df['gold_shares'] * df['Gold']
        df['silver_exposure'] = df['silver_shares'] * df['Silver']
        df['net_exposure'] = df['gold_exposure'] + df['silver_exposure']  # Should be ~0

        return df

    def calculate_spread(self, gold_prices: pd.Series, silver_prices: pd.Series) -> pd.Series:
        """Calculate the gold-silver ratio spread."""
        return gold_prices / silver_prices

    def calculate_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Hurst exponent and z-score.

        Args:
            data: DataFrame with 'Gold' and 'Silver' columns

        Returns:
            DataFrame with signals and indicators
        """
        df = data.copy()

        # Calculate spread (Gold/Silver ratio)
        df['spread'] = self.calculate_spread(df['Gold'], df['Silver'])

        # Calculate rolling Hurst exponent
        df['hurst'] = np.nan
        for i in range(self.hurst_window, len(df)):
            spread_window = df['spread'].iloc[i - self.hurst_window:i].values
            df.loc[df.index[i], 'hurst'] = self.calculate_hurst_exponent(spread_window)

        # Calculate z-score of spread
        df['zscore'] = self.calculate_zscore(df['spread'], self.signal_window)

        # Generate signals only when Hurst < threshold (mean reverting regime)
        df['regime'] = df['hurst'] < self.hurst_threshold

        # Initialize position and signals
        df['position'] = 0
        df['signal'] = 0

        position = 0

        for i in range(1, len(df)):
            if pd.isna(df.iloc[i]['zscore']) or pd.isna(df.iloc[i]['hurst']):
                df.iloc[i, df.columns.get_loc('position')] = position
                continue

            zscore = df.iloc[i]['zscore']
            is_mean_reverting = df.iloc[i]['regime']

            # Entry signals (only in mean-reverting regime)
            if is_mean_reverting and position == 0:
                if zscore > self.zscore_entry:
                    # Spread too high, short spread (short gold, long silver)
                    position = -1
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                elif zscore < -self.zscore_entry:
                    # Spread too low, long spread (long gold, short silver)
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

        # Add dollar-neutral positions
        df = self.calculate_dollar_neutral_positions(df)

        # Calculate individual asset returns
        df['gold_return'] = df['Gold'].pct_change()
        df['silver_return'] = df['Silver'].pct_change()

        # Calculate P&L from each leg (lagged positions to avoid look-ahead bias)
        df['gold_pnl'] = df['gold_shares'].shift(1) * df['Gold'].diff()
        df['silver_pnl'] = df['silver_shares'].shift(1) * df['Silver'].diff()

        # Total strategy P&L
        df['strategy_pnl'] = df['gold_pnl'].fillna(0) + df['silver_pnl'].fillna(0)

        # Calculate transaction costs based on position changes
        df['gold_trades'] = df['gold_shares'].diff().abs()
        df['silver_trades'] = df['silver_shares'].diff().abs()
        df['total_transaction_costs'] = (df['gold_trades'] * df['Gold'] +
                                         df['silver_trades'] * df['Silver']) * self.transaction_cost

        # Net P&L after costs
        df['net_pnl'] = df['strategy_pnl'] - df['total_transaction_costs'].fillna(0)

        # Calculate returns as percentage of capital
        df['strategy_return'] = df['net_pnl'] / self.initial_capital

        # Cumulative P&L and returns
        df['cumulative_pnl'] = df['net_pnl'].fillna(0).cumsum()
        df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()

        # Mark-to-market portfolio value
        df['portfolio_value'] = self.initial_capital + df['cumulative_pnl']

        # Calculate spread returns for comparison
        df['spread_return'] = df['spread'].pct_change()
        df['spread_cumulative'] = (1 + df['spread_return'].fillna(0)).cumprod()

        return df

    def calculate_performance_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics."""
        returns = df['strategy_return'].dropna()

        if len(returns) == 0:
            return {}

        # Basic metrics
        total_pnl = df['cumulative_pnl'].iloc[-1]
        total_return = total_pnl / self.initial_capital
        annual_return = (1 + total_return) ** (252 * 288 / len(df)) - 1  # 5-min bars
        volatility = returns.std() * np.sqrt(252 * 288)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Drawdown analysis
        portfolio_values = df['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade metrics
        trades = df[df['signal'] != 0]
        num_trades = len(trades)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Hurst regime analysis
        mean_hurst = df['hurst'].mean()
        mean_reverting_pct = (df['regime'].sum() / len(df)) * 100

        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0

        return {
            'total_pnl': total_pnl,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'mean_hurst': mean_hurst,
            'mean_reverting_regime_pct': mean_reverting_pct
        }

    def plot_performance(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 12)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Cumulative P&L
        axes[0, 0].plot(df.index, df['cumulative_pnl'], label='Strategy P&L', linewidth=2)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0, 0].set_title('Cumulative P&L ($)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('P&L ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 2. Portfolio Value vs Spread
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()

        ax1.plot(df.index, df['portfolio_value'], 'b-', label='Portfolio Value', linewidth=2)
        ax2.plot(df.index, df['spread'], 'r-', alpha=0.7, label='Gold/Silver Ratio')

        ax1.set_ylabel('Portfolio Value ($)', color='b')
        ax2.set_ylabel('Gold/Silver Ratio', color='r')
        ax1.set_title('Portfolio Value vs Gold/Silver Spread', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 3. Drawdown
        portfolio_values = df['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max * 100

        axes[1, 0].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].plot(df.index, drawdown, color='red', linewidth=1)
        axes[1, 0].set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Hurst Exponent and Regime
        ax3 = axes[1, 1]
        ax4 = ax3.twinx()

        ax3.plot(df.index, df['hurst'], 'g-', alpha=0.7, label='Hurst Exponent')
        ax3.axhline(y=self.hurst_threshold, color='orange', linestyle='--',
                    label=f'Threshold ({self.hurst_threshold})')
        ax4.fill_between(df.index, 0, df['regime'].astype(int), alpha=0.2, color='blue',
                         label='Mean Reverting Regime')

        ax3.set_ylabel('Hurst Exponent', color='g')
        ax4.set_ylabel('Mean Reverting Regime', color='b')
        ax3.set_title('Hurst Exponent & Trading Regime', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_trade_analysis(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 8)):
        """
        Plot detailed trade analysis.

        Args:
            df: Results dataframe from backtest
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Position sizes over time
        axes[0, 0].plot(df.index, df['gold_exposure'], label='Gold Exposure', alpha=0.7)
        axes[0, 0].plot(df.index, df['silver_exposure'], label='Silver Exposure', alpha=0.7)
        axes[0, 0].plot(df.index, df['net_exposure'], label='Net Exposure', linewidth=2, color='red')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0, 0].set_title('Dollar Exposures (Should be Dollar Neutral)', fontweight='bold')
        axes[0, 0].set_ylabel('Exposure ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Z-score and positions
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()

        ax1.plot(df.index, df['zscore'], 'purple', alpha=0.7, label='Z-Score')
        ax1.axhline(y=self.zscore_entry, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=-self.zscore_entry, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=self.zscore_exit, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y=-self.zscore_exit, color='green', linestyle='--', alpha=0.5)

        ax2.plot(df.index, df['position'], 'orange', linewidth=2, label='Position')

        ax1.set_ylabel('Z-Score', color='purple')
        ax2.set_ylabel('Position', color='orange')
        ax1.set_title('Z-Score vs Position', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 3. Rolling returns distribution
        rolling_returns = df['strategy_return'].rolling(20).sum().dropna()
        axes[1, 0].hist(rolling_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=rolling_returns.mean(), color='red', linestyle='--',
                           label=f'Mean: {rolling_returns.mean():.4f}')
        axes[1, 0].set_title('20-Period Rolling Returns Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Rolling Returns')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Trade P&L distribution
        trade_returns = df[df['signal'] != 0]['strategy_return']
        if len(trade_returns) > 0:
            axes[1, 1].hist(trade_returns, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=trade_returns.mean(), color='red', linestyle='--',
                               label=f'Mean: {trade_returns.mean():.4f}')
            axes[1, 1].set_title('Individual Trade Returns Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Trade Returns')
            axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        """
        Run complete backtest.

        Args:
            data: DataFrame with 'Gold' and 'Silver' columns

        Returns:
            Tuple of (results_df, performance_metrics)
        """
        # Generate signals
        results = self.generate_signals(df)

        # Calculate returns
        results = self.calculate_returns(results)

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results)

        return results, metrics