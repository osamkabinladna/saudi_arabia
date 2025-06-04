import pandas as pd
from optuna.samplers import TPESampler
import numpy as np
from saudi_arabia.strats.hurst import HurstGoldSilverStrategy
import matplotlib.pyplot as plt
from typing import Dict, Any
import optuna


class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.initial_capital = initial_capital

    def objective(self, trial) -> float:
        hurst_window = trial.suggest_int('hurst_window', 50, 500, step=10)
        signal_window = trial.suggest_int('signal_window', 10, 200, step=5)
        hurst_threshold = trial.suggest_float('hurst_threshold', 0.15, 0.45, step=0.01)
        zscore_entry = trial.suggest_float('zscore_entry', 1.0, 3.0, step=0.05)
        zscore_exit = trial.suggest_float('zscore_exit', 0.25, 1.0, step=0.05)

        if signal_window >= hurst_window:
            return -10  # Negative penalty for maximization

        try:
            # Create strategy with trial parameters
            strategy = HurstGoldSilverStrategy(
                hurst_window=hurst_window,
                signal_window=signal_window,
                hurst_threshold=hurst_threshold,
                zscore_entry=zscore_entry,
                zscore_exit=zscore_exit,
                initial_capital=self.initial_capital
            )

            # Run backtest
            results, metrics = strategy.backtest(self.data)

            # Get Sharpe ratio (positive value)
            sharpe = metrics.get('sharpe_ratio', -10)

            # Add constraints - return negative values for bad performance
            if metrics.get('num_trades', 0) < 10:  # Minimum trade requirement
                return -10
            if metrics.get('max_drawdown', -1) < -0.5:  # Max 50% drawdown
                return -10

            return sharpe  # Return positive Sharpe ratio for maximization

        except Exception as e:
            return -10  # Negative penalty for failed backtests

    def optimize(self, n_trials: int = 100, n_jobs: int = 1) -> Dict[str, Any]:
        study = optuna.create_study(
            direction='maximize',  # FIXED: Maximize Sharpe ratio
            sampler=TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)

        # Get best parameters
        best_params = study.best_params

        # Run backtest with best parameters
        best_strategy = HurstGoldSilverStrategy(
            hurst_window=best_params['hurst_window'],
            signal_window=best_params['signal_window'],
            hurst_threshold=best_params['hurst_threshold'],
            zscore_entry=best_params['zscore_entry'],
            zscore_exit=best_params['zscore_exit'],
            initial_capital=self.initial_capital
        )

        best_results, best_metrics = best_strategy.backtest(self.data)

        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_strategy': best_strategy,
            'best_results': best_results,
            'study': study
        }

    def plot_optimization_results(self, study):
        """Plot optimization results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Optimization history
        axes[0, 0].plot([trial.value for trial in study.trials])
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Objective Value (Sharpe Ratio)')  # FIXED: Updated label
        axes[0, 0].grid(True, alpha=0.3)

        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())

            axes[0, 1].barh(params, values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available',
                            ha='center', va='center', transform=axes[0, 1].transAxes)

        # 3. Best parameters visualization
        best_params = study.best_params
        param_names = list(best_params.keys())
        param_values = list(best_params.values())

        axes[1, 0].bar(param_names, param_values)
        axes[1, 0].set_title('Best Parameters')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Objective value distribution
        objective_values = [trial.value for trial in study.trials if trial.value is not None]
        axes[1, 1].hist(objective_values, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=study.best_value, color='red', linestyle='--',
                           label=f'Best: {study.best_value:.3f}')
        axes[1, 1].set_title('Objective Value Distribution')
        axes[1, 1].set_xlabel('Objective Value (Sharpe Ratio)')  # FIXED: Updated label
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def calculate_performance_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics."""
        returns = df['strategy_return'].dropna()

        if len(returns) == 0:
            return {}

        total_pnl = df['cumulative_pnl'].iloc[-1]
        total_return = total_pnl / self.initial_capital
        annual_return = (1 + total_return) ** (252 * 288 / len(df)) - 1
        volatility = returns.std() * np.sqrt(252 * 288)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

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

