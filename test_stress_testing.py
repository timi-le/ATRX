#!/usr/bin/env python3
"""
Test script for Monte Carlo and Stress Testing module (Task 18)

This script validates all the stress testing functionality including:
- Bootstrap resampling and Monte Carlo simulation
- Shock scenario application
- VaR/CVaR calculations
- Comprehensive stress test analysis
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtester.stress_tester import (
    MonteCarloStressTester,
    ShockScenario,
    StressTestConfig,
    create_default_shock_scenarios,
    run_comprehensive_stress_test,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_trade_data(num_trades: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create sample trade data for testing."""
    np.random.seed(seed)

    # Generate realistic trade data
    start_date = datetime(2024, 1, 1)

    trades = []
    for i in range(num_trades):
        # Random trade timing
        trade_time = start_date + timedelta(
            hours=np.random.randint(0, 24 * 30)
        )  # 30 days

        # Random trade characteristics
        symbol = np.random.choice(["EURUSD", "GBPUSD", "USDJPY", "USDCHF"])
        quantity = np.random.choice([10000, 25000, 50000, 100000])  # Standard lot sizes

        # Price around typical FX levels
        if symbol == "USDJPY":
            price = 150.0 + np.random.normal(0, 2)
        else:
            price = 1.1000 + np.random.normal(0, 0.05)

        # PnL with some bias toward profitability (60% win rate)
        if np.random.random() < 0.6:  # Winning trade
            pnl = abs(np.random.normal(50, 30))  # Positive PnL
        else:  # Losing trade
            pnl = -abs(np.random.normal(40, 25))  # Negative PnL

        # Execution costs
        slippage = np.random.normal(0, 0.0001)  # Small slippage
        commission = quantity * 0.00002  # 2 pips per 100k

        trades.append(
            {
                "timestamp": trade_time,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
                "slippage": slippage,
                "commission": commission,
            }
        )

    df = pd.DataFrame(trades)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Created sample trade data: {len(df)} trades, "
        f"Total PnL: ${df['pnl'].sum():.2f}, "
        f"Win Rate: {(df['pnl'] > 0).mean():.1%}"
    )

    return df


def test_monte_carlo_simulation():
    """Test Monte Carlo equity path simulation."""
    logger.info("=== Testing Monte Carlo Simulation ===")

    # Create sample data
    trades_df = create_sample_trade_data(50)

    # Initialize stress tester
    config = StressTestConfig(num_monte_carlo_paths=100, random_seed=42)
    stress_tester = MonteCarloStressTester(config)

    # Generate Monte Carlo paths
    equity_paths = stress_tester.simulate_equity_paths(trades_df)

    # Validate results
    assert len(equity_paths.columns) == 100, "Should have 100 Monte Carlo paths"
    assert len(equity_paths) > 0, "Should have time series data"

    # Check that paths are different (bootstrap resampling working)
    path_finals = equity_paths.iloc[-1]
    assert path_finals.std() > 0, "Paths should have variation"

    logger.info(
        f"✓ Monte Carlo simulation successful: {len(equity_paths)} timesteps, "
        f"{len(equity_paths.columns)} paths"
    )
    logger.info(
        f"  Final equity range: [{path_finals.min():.2f}, {path_finals.max():.2f}]"
    )

    return equity_paths


def test_shock_scenarios():
    """Test shock scenario application."""
    logger.info("=== Testing Shock Scenarios ===")

    # Create sample data
    trades_df = create_sample_trade_data(30)
    original_pnl = trades_df["pnl"].sum()

    # Initialize stress tester
    stress_tester = MonteCarloStressTester()

    # Test individual shock scenarios
    scenarios = [
        ShockScenario(name="High Slippage", slippage_multiplier=3.0),
        ShockScenario(name="High Commission", commission_multiplier=2.0),
        ShockScenario(name="Poor Fill Rate", fill_rate_reduction=0.2),
        ShockScenario(name="High Volatility", volatility_multiplier=2.0),
    ]

    for scenario in scenarios:
        shocked_trades = stress_tester.apply_shocks(trades_df, scenario)
        shocked_pnl = shocked_trades["pnl"].sum()

        logger.info(
            f"  {scenario.name}: {len(trades_df)} -> {len(shocked_trades)} trades, "
            f"PnL: ${original_pnl:.2f} -> ${shocked_pnl:.2f}"
        )

        # Validate shock effects
        if scenario.fill_rate_reduction > 0:
            assert len(shocked_trades) < len(
                trades_df
            ), "Fill rate reduction should remove trades"

        if scenario.commission_multiplier > 1:
            assert shocked_pnl < original_pnl, "Higher commission should reduce PnL"

    logger.info("✓ Shock scenarios applied successfully")


def test_var_cvar_calculation():
    """Test VaR and CVaR calculations."""
    logger.info("=== Testing VaR/CVaR Calculations ===")

    # Create sample equity curve
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    equity_curve = pd.Series((1 + returns).cumprod() * 100000)  # Start with $100k

    # Initialize stress tester
    stress_tester = MonteCarloStressTester()

    # Calculate VaR and CVaR
    var_95, cvar_95 = stress_tester.calculate_var_cvar(equity_curve, confidence=0.95)
    var_99, cvar_99 = stress_tester.calculate_var_cvar(equity_curve, confidence=0.99)

    # Validate results
    assert var_95 < 0, "VaR should be negative (loss)"
    assert cvar_95 < var_95, "CVaR should be more negative than VaR"
    assert var_99 < var_95, "99% VaR should be more negative than 95% VaR"

    logger.info(f"✓ VaR/CVaR calculations successful:")
    logger.info(f"  VaR(95%): {var_95:.4f}, CVaR(95%): {cvar_95:.4f}")
    logger.info(f"  VaR(99%): {var_99:.4f}, CVaR(99%): {cvar_99:.4f}")


def test_stress_test_scenarios():
    """Test comprehensive stress testing with multiple scenarios."""
    logger.info("=== Testing Comprehensive Stress Testing ===")

    # Create sample data
    trades_df = create_sample_trade_data(75)

    # Initialize stress tester
    stress_tester = MonteCarloStressTester()

    # Get default scenarios
    scenarios = create_default_shock_scenarios()

    # Run stress tests
    results_df = stress_tester.run_stress_tests(trades_df, scenarios)

    # Validate results
    assert len(results_df) > 0, "Should have stress test results"
    assert (
        "Baseline" in results_df["scenario"].values
    ), "Should include baseline scenario"

    # Check that we have expected columns
    expected_columns = [
        "scenario",
        "total_trades",
        "final_equity",
        "max_drawdown",
        "sharpe_ratio",
        "var_95",
        "cvar_95",
    ]
    for col in expected_columns:
        assert col in results_df.columns, f"Missing column: {col}"

    # Display results
    logger.info(f"✓ Stress testing completed with {len(results_df)} scenarios:")
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['scenario']}: Final Equity=${row['final_equity']:.2f}, "
            f"Max DD={row['max_drawdown']:.3f}, Sharpe={row['sharpe_ratio']:.3f}"
        )

    return results_df


def test_comprehensive_analysis():
    """Test the comprehensive stress test analysis function."""
    logger.info("=== Testing Comprehensive Analysis ===")

    # Create sample data
    trades_df = create_sample_trade_data(60)

    # Run comprehensive analysis
    output_dir = "outputs/test_stress_testing"
    results = run_comprehensive_stress_test(trades_df, output_dir)

    # Validate results structure
    assert "monte_carlo" in results, "Should have Monte Carlo results"
    assert "stress_tests" in results, "Should have stress test results"
    assert "summary" in results, "Should have summary"

    # Check Monte Carlo results
    mc_results = results["monte_carlo"]
    assert "final_equity_stats" in mc_results, "Should have final equity statistics"
    assert "risk_metrics" in mc_results, "Should have risk metrics"

    # Check that files were created
    output_path = Path(output_dir)
    assert (
        output_path / "monte_carlo_results.json"
    ).exists(), "Should create JSON results"
    assert (
        output_path / "stress_test_results.csv"
    ).exists(), "Should create CSV results"

    logger.info("✓ Comprehensive analysis completed successfully")
    logger.info(f"  Monte Carlo paths: {results['summary']['num_monte_carlo_paths']}")
    logger.info(f"  Stress scenarios: {results['summary']['num_stress_scenarios']}")
    logger.info(
        f"  Baseline final equity: ${results['summary']['baseline_final_equity']:.2f}"
    )
    logger.info(
        f"  Monte Carlo mean final: ${results['summary']['monte_carlo_mean_final']:.2f}"
    )
    logger.info(f"  Worst case scenario: {results['summary']['worst_case_scenario']}")

    return results


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("=== Testing Edge Cases ===")

    stress_tester = MonteCarloStressTester()

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    try:
        equity_paths = stress_tester.simulate_equity_paths(empty_df)
        assert len(equity_paths) == 0, "Empty input should return empty result"
    except ValueError as e:
        # Expected behavior - empty DataFrame should raise ValueError for missing columns
        assert "Missing required columns" in str(
            e
        ), "Should raise ValueError for missing columns"
        logger.info("✓ Empty DataFrame correctly raises ValueError")

    # Test with minimal data
    minimal_df = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "pnl": [100.0],
            "slippage": [0.0001],
            "commission": [2.0],
        }
    )

    equity_paths = stress_tester.simulate_equity_paths(minimal_df, num_paths=10)
    assert len(equity_paths.columns) == 10, "Should handle minimal data"

    # Test VaR/CVaR with insufficient data
    short_series = pd.Series([100])
    var, cvar = stress_tester.calculate_var_cvar(short_series)
    assert var == 0.0 and cvar == 0.0, "Should handle insufficient data gracefully"

    logger.info("✓ Edge cases handled correctly")


def main():
    """Run all stress testing validation tests."""
    logger.info("Starting Monte Carlo and Stress Testing validation...")

    try:
        # Run all tests
        test_monte_carlo_simulation()
        test_shock_scenarios()
        test_var_cvar_calculation()
        test_stress_test_scenarios()
        test_comprehensive_analysis()
        test_edge_cases()

        logger.info("🎉 ALL STRESS TESTING TESTS PASSED! 🎉")
        logger.info(
            "Task 18: Monte Carlo and Stress Testing module is working correctly"
        )

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
