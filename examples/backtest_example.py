"""
Example: Using the Backtesting Framework

This example demonstrates how to use the backtesting framework to:
1. Set up a backtest configuration
2. Run a complete backtest with the full trading pipeline
3. Analyze the results
4. Compare multiple strategies
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtester import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    create_backtest_config,
    run_simple_backtest,
    compare_strategies,
    PerformanceAnalyzer,
    create_performance_config
)


async def basic_backtest_example():
    """
    Example 1: Basic backtest with minimal configuration.
    """
    print("=" * 60)
    print("Example 1: Basic Backtest")
    print("=" * 60)
    
    # Define backtest parameters
    symbols = ["EURUSD", "GBPUSD"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)  # 1 week of data
    initial_capital = 100000.0
    
    try:
        # Run a simple backtest
        results = await run_simple_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # Display results
        print(f"Backtest completed successfully!")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        
        performance = results['performance']
        print(f"Total Return: {performance['total_return']:.2%}")
        print(f"Annualized Return: {performance['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.2%}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")


async def advanced_backtest_example():
    """
    Example 2: Advanced backtest with custom configuration.
    """
    print("\n" + "=" * 60)
    print("Example 2: Advanced Backtest Configuration")
    print("=" * 60)
    
    # Create detailed configuration
    config = BacktestConfig(
        symbols=["EURUSD", "GBPUSD", "USDJPY"],
        timeframe="5m",  # 5-minute bars
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 15),  # 2 weeks
        initial_capital=250000.0,
        mode=BacktestMode.FULL_PIPELINE,
        
        # Strategy configuration
        strategies=["grid_martingale", "breakout_trend", "mean_reversion"],
        strategy_weights={"grid_martingale": 0.4, "breakout_trend": 0.4, "mean_reversion": 0.2},
        
        # Risk management
        max_position_size=0.15,  # 15% per position
        max_total_exposure=0.6,  # 60% total exposure
        stop_loss_pct=0.025,     # 2.5% stop loss
        take_profit_pct=0.05,    # 5% take profit
        
        # Execution settings
        enable_slippage=True,
        enable_commission=True,
        enable_latency=True,
        
        # Results
        save_results=True,
        save_trades=True,
        save_equity_curve=True,
        results_path="outputs/advanced_backtest"
    )
    
    try:
        # Create and run backtest engine
        engine = BacktestEngine(config)
        await engine.initialize()
        
        print(f"Running advanced backtest...")
        print(f"Symbols: {config.symbols}")
        print(f"Timeframe: {config.timeframe}")
        print(f"Period: {config.start_date.date()} to {config.end_date.date()}")
        print(f"Strategies: {config.strategies}")
        
        results = await engine.run()
        
        # Display comprehensive results
        print(f"\nBacktest Results:")
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        
        performance = results['performance']
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {performance['total_return']:.2%}")
        print(f"  Annualized Return: {performance['annualized_return']:.2%}")
        print(f"  Volatility: {performance['volatility']:.2%}")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {performance['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {performance['calmar_ratio']:.2f}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"  Max DD Duration: {performance['max_drawdown_duration']:.1f} hours")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {performance['total_trades']}")
        print(f"  Winning Trades: {performance['winning_trades']}")
        print(f"  Losing Trades: {performance['losing_trades']}")
        print(f"  Win Rate: {performance['win_rate']:.2%}")
        print(f"  Profit Factor: {performance['profit_factor']:.2f}")
        print(f"  Average Win: ${performance['avg_win']:.2f}")
        print(f"  Average Loss: ${performance['avg_loss']:.2f}")
        
        print(f"\nExecution Statistics:")
        execution = results['execution']
        print(f"  Total Orders: {execution['total_orders']}")
        print(f"  Fill Rate: {execution['fill_rate']:.2%}")
        print(f"  Total Commission: ${execution['total_commission']:.2f}")
        print(f"  Average Slippage: {execution['avg_slippage_per_fill']:.4f}")
        
        # Regime performance
        if 'regime_performance' in results:
            print(f"\nRegime Performance:")
            for regime, metrics in results['regime_performance'].items():
                if metrics['num_periods'] > 0:
                    print(f"  {regime.title()}:")
                    print(f"    Return: {metrics['total_return']:.2%}")
                    print(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
                    print(f"    Win Rate: {metrics['win_rate']:.2%}")
                    print(f"    Periods: {metrics['num_periods']}")
        
        print(f"\nResults saved to: {config.results_path}")
        
    except Exception as e:
        print(f"Error running advanced backtest: {e}")


async def strategy_comparison_example():
    """
    Example 3: Compare multiple strategies.
    """
    print("\n" + "=" * 60)
    print("Example 3: Strategy Comparison")
    print("=" * 60)
    
    strategies_to_test = [
        ("Conservative Grid", ["grid_martingale"], {"max_position_size": 0.05}),
        ("Aggressive Breakout", ["breakout_trend"], {"max_position_size": 0.20}),
        ("Balanced Mix", ["grid_martingale", "breakout_trend"], {"max_position_size": 0.10})
    ]
    
    results = {}
    
    for strategy_name, strategies, custom_params in strategies_to_test:
        print(f"\nTesting {strategy_name}...")
        
        # Create configuration for this strategy
        config = create_backtest_config(
            symbols=["EURUSD", "GBPUSD"],
            timeframe="1m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),  # 5 days for faster testing
            initial_capital=100000.0,
            mode=BacktestMode.FULL_PIPELINE
        )
        
        # Apply custom parameters
        config.strategies = strategies
        for param, value in custom_params.items():
            setattr(config, param, value)
        
        config.save_results = False  # Don't save individual results
        
        try:
            engine = BacktestEngine(config)
            await engine.initialize()
            result = await engine.run()
            results[strategy_name] = result
            
            performance = result['performance']
            print(f"  Return: {performance['total_return']:.2%}")
            print(f"  Sharpe: {performance['sharpe_ratio']:.2f}")
            print(f"  Max DD: {performance['max_drawdown']:.2%}")
            print(f"  Trades: {performance['total_trades']}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Create comparison table
    if len(results) > 1:
        print(f"\nStrategy Comparison Summary:")
        print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8}")
        print("-" * 60)
        
        for strategy_name, result in results.items():
            perf = result['performance']
            print(f"{strategy_name:<20} {perf['total_return']:>8.2%} "
                  f"{perf['sharpe_ratio']:>7.2f} {perf['max_drawdown']:>7.2%} "
                  f"{perf['total_trades']:>7}")
        
        # Find best strategy by Sharpe ratio
        best_strategy = max(results.items(), 
                          key=lambda x: x[1]['performance']['sharpe_ratio'])
        print(f"\nBest Strategy (by Sharpe): {best_strategy[0]}")


async def performance_analysis_example():
    """
    Example 4: Detailed performance analysis.
    """
    print("\n" + "=" * 60)
    print("Example 4: Detailed Performance Analysis")
    print("=" * 60)
    
    # Create a performance analyzer for demonstration
    config = create_performance_config(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        trading_days_per_year=252
    )
    
    analyzer = PerformanceAnalyzer(config)
    
    # Simulate some equity curve data
    import numpy as np
    
    base_time = datetime(2024, 1, 1)
    equity = 100000.0
    
    print("Simulating 30 days of trading data...")
    
    for i in range(30):
        timestamp = base_time + timedelta(days=i)
        
        # Simulate daily return with some volatility
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
        equity *= (1 + daily_return)
        
        analyzer.update_equity(timestamp, equity)
        
        # Simulate regime changes
        if i % 10 == 0:
            regimes = ["trending", "ranging", "volatile"]
            regime = regimes[i // 10]
            analyzer.set_regime(regime)
    
    # Calculate comprehensive metrics
    metrics = analyzer.calculate_metrics()
    
    print(f"\nPerformance Analysis Results:")
    print(f"Period: {metrics.start_date.date()} to {metrics.end_date.date()}")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Volatility: {metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    
    # Calculate risk metrics
    var_95 = analyzer.calculate_var(0.95)
    var_99 = analyzer.calculate_var(0.99)
    expected_shortfall = analyzer.calculate_expected_shortfall(0.95)
    
    print(f"\nRisk Metrics:")
    print(f"VaR (95%): {var_95:.2%}")
    print(f"VaR (99%): {var_99:.2%}")
    print(f"Expected Shortfall (95%): {expected_shortfall:.2%}")
    
    # Regime performance
    regime_performance = analyzer.calculate_regime_performance()
    print(f"\nRegime Performance:")
    for regime, metrics in regime_performance.items():
        if metrics['num_periods'] > 0:
            print(f"  {regime.title()}:")
            print(f"    Return: {metrics['total_return']:.2%}")
            print(f"    Volatility: {metrics['volatility']:.2%}")
            print(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"    Periods: {metrics['num_periods']}")
    
    # Rolling metrics
    rolling_metrics = analyzer.calculate_rolling_metrics(window_days=7)
    if not rolling_metrics.empty:
        print(f"\nRolling Metrics (7-day window):")
        print(f"  Average Rolling Return: {rolling_metrics['rolling_return'].mean():.2%}")
        print(f"  Average Rolling Sharpe: {rolling_metrics['rolling_sharpe'].mean():.2f}")
        print(f"  Max Rolling Drawdown: {rolling_metrics['rolling_drawdown'].max():.2%}")


async def main():
    """
    Run all examples.
    """
    print("FX Quant System - Backtesting Framework Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        await basic_backtest_example()
        await advanced_backtest_example()
        await strategy_comparison_example()
        await performance_analysis_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Run the examples
    asyncio.run(main()) 