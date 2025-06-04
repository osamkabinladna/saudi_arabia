import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from saudi_arabia.optimizers.hurst import StrategyOptimizer
from pathlib import Path
import sys


def run_strategy_optimization(data, n_trials=50, initial_capital=10_000_000):
    print(f"Starting optimization with {n_trials} trials...")
    print("This may take a few minutes depending on data size and n_trials...")

    # 1. Initialize optimizer
    optimizer = StrategyOptimizer(data, initial_capital=initial_capital)

    # 2. Run optimization
    optimization_results = optimizer.optimize(n_trials=n_trials, n_jobs=1)

    # 3. Extract results
    best_params = optimization_results['best_params']
    best_metrics = optimization_results['best_metrics']
    best_strategy = optimization_results['best_strategy']
    best_results = optimization_results['best_results']
    study = optimization_results['study']

    # 4. Print optimization results
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)

    print("\nBest Parameters Found:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.3f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nBest Performance Metrics:")
    print(f"  Total P&L: ${best_metrics['total_pnl']:,.2f}")
    print(f"  Total Return: {best_metrics['total_return']:.2%}")
    print(f"  Annual Return: {best_metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Calmar Ratio: {best_metrics['calmar_ratio']:.2f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']:.2%}")
    print(f"  Volatility: {best_metrics['volatility']:.2%}")
    print(f"  Number of Trades: {best_metrics['num_trades']}")
    print(f"  Win Rate: {best_metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {best_metrics['profit_factor']:.2f}")
    print(f"  Mean Hurst Exponent: {best_metrics['mean_hurst']:.3f}")
    print(f"  Mean Reverting Regime %: {best_metrics['mean_reverting_regime_pct']:.1f}%")

    #TODO: This is broken
    # print("\nGenerating optimization plots...")
    # optimizer.plot_optimization_results(study)
    # print("Generating strategy performance plots...")
    # best_strategy.plot_performance(best_results)
    # best_strategy.plot_trade_analysis(best_results)

    best_params.to_csv('best_params.csv')

    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_strategy': best_strategy,
        'best_results': best_results,
        'study': study,
        'optimizer': optimizer
    }


def optimize(data):
    """
    Quick optimization example with sensible defaults.
    """
    print("Running quick optimization (50 trials)...")

    # Run optimization
    results = run_strategy_optimization(
        data=data,
        n_trials=50,
        initial_capital=1_000_000
    )

    # Get best strategy for further use
    best_strategy = results['best_strategy']
    best_params = results['best_params']

    print(f"\nOptimization complete!")
    print(f"Best Sharpe Ratio: {results['best_metrics']['sharpe_ratio']:.2f}")

    return best_strategy, best_params

if __name__ == '__main__':
    root = Path().resolve().parent.parent
    print(f'CURRENT ROOT: {root}; root should be top level saudi_arabia')

    data = pd.read_csv(root / 'data' / 'goldsilver.csv', index_col=0)
    data = data.iloc[:, :2]
    data.index = pd.to_datetime(data.index)
    data.dropna(how='all', inplace=True)
    optimize(data)