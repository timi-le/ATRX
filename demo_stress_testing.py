#!/usr/bin/env python3
"""
Demonstration script for Monte Carlo and Stress Testing integration

This script shows how to use the stress testing module with real backtest results
from the FX Quant Trading System.
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


def load_backtest_results(
    results_file: str = "outputs/backtest_results.csv",
) -> pd.DataFrame:
    """
    Load backtest results from CSV file.

    This function attempts to load real backtest results, but falls back to
    generating sample data if the file doesn't exist.
    """
    try:
        # Try to load real backtest results
        if Path(results_file).exists():
            df = pd.read_csv(results_file)
            logger.info(f"Loaded real backtest results from {results_file}")
            return df
        else:
            logger.warning(
                f"Backtest results file {results_file} not found. Generating sample data."
            )
            return generate_realistic_sample_data()
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}. Generating sample data.")
        return generate_realistic_sample_data()


def generate_realistic_sample_data(num_trades: int = 200) -> pd.DataFrame:
    """Generate realistic sample trade data for demonstration."""
    np.random.seed(42)

    logger.info(f"Generating {num_trades} realistic sample trades...")

    # Generate realistic trade data over 3 months
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)

    trades = []
    current_time = start_date

    for i in range(num_trades):
        # Random trade timing (avoid weekends)
        while current_time.weekday() >= 5:  # Skip weekends
            current_time += timedelta(hours=1)

        # Random trade characteristics
        symbol = np.random.choice(
            ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
            p=[0.3, 0.25, 0.2, 0.15, 0.1],
        )

        # Realistic lot sizes
        quantity = np.random.choice(
            [10000, 25000, 50000, 100000], p=[0.4, 0.3, 0.2, 0.1]
        )

        # Price around typical FX levels
        if symbol == "USDJPY":
            price = 150.0 + np.random.normal(0, 2)
        elif symbol == "AUDUSD":
            price = 0.6500 + np.random.normal(0, 0.02)
        else:
            price = 1.1000 + np.random.normal(0, 0.05)

        # PnL with realistic distribution (slight positive bias)
        if np.random.random() < 0.58:  # 58% win rate
            pnl = abs(
                np.random.lognormal(3.5, 0.8)
            )  # Positive PnL, log-normal distribution
        else:  # Losing trade
            pnl = -abs(np.random.lognormal(3.3, 0.7))  # Negative PnL

        # Realistic execution costs
        spread = np.random.uniform(0.00008, 0.00025)  # 0.8-2.5 pips
        slippage = np.random.normal(0, spread / 4)  # Slippage around 1/4 of spread
        commission = quantity * np.random.uniform(0.00001, 0.00003)  # 1-3 pips per 100k

        trades.append(
            {
                "timestamp": current_time,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
                "slippage": slippage,
                "commission": commission,
                "spread": spread,
            }
        )

        # Advance time (trades every 2-8 hours on average)
        current_time += timedelta(hours=np.random.randint(2, 9))
        if current_time > end_date:
            break

    df = pd.DataFrame(trades)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calculate some summary statistics
    total_pnl = df["pnl"].sum()
    win_rate = (df["pnl"] > 0).mean()
    avg_trade = df["pnl"].mean()

    logger.info(f"Generated sample data: {len(df)} trades")
    logger.info(f"  Total PnL: ${total_pnl:.2f}")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    logger.info(f"  Average Trade: ${avg_trade:.2f}")

    return df


def create_custom_shock_scenarios() -> list:
    """Create custom shock scenarios relevant to FX trading."""
    scenarios = [
        ShockScenario(
            name="Market Crisis",
            slippage_multiplier=4.0,
            commission_multiplier=1.5,
            fill_rate_reduction=0.20,
            volatility_multiplier=3.0,
            description="2008-style financial crisis conditions",
        ),
        ShockScenario(
            name="Flash Crash",
            slippage_multiplier=8.0,
            fill_rate_reduction=0.35,
            volatility_multiplier=5.0,
            description="Sudden liquidity evaporation (flash crash)",
        ),
        ShockScenario(
            name="High Frequency Competition",
            slippage_multiplier=2.5,
            latency_ms_add=50,
            fill_rate_reduction=0.10,
            description="Increased HFT competition",
        ),
        ShockScenario(
            name="Broker Issues",
            commission_multiplier=3.0,
            latency_ms_add=300,
            fill_rate_reduction=0.15,
            description="Broker technical problems and higher costs",
        ),
        ShockScenario(
            name="Central Bank Intervention",
            volatility_multiplier=4.0,
            slippage_multiplier=3.0,
            description="Unexpected central bank intervention",
        ),
    ]

    return scenarios


def analyze_monte_carlo_results(equity_paths_df: pd.DataFrame) -> dict:
    """Analyze Monte Carlo results and extract key insights."""
    final_values = equity_paths_df.iloc[-1]

    # Calculate key statistics
    mean_final = final_values.mean()
    std_final = final_values.std()
    median_final = final_values.median()

    # Risk metrics
    prob_loss = (final_values < 0).mean()
    prob_large_loss = (final_values < -1000).mean()

    # Percentiles
    percentiles = {
        "worst_5pct": np.percentile(final_values, 5),
        "worst_10pct": np.percentile(final_values, 10),
        "best_10pct": np.percentile(final_values, 90),
        "best_5pct": np.percentile(final_values, 95),
    }

    # Path analysis
    max_equity_paths = equity_paths_df.max(axis=0)
    min_equity_paths = equity_paths_df.min(axis=0)

    max_drawdowns = []
    for col in equity_paths_df.columns:
        path = equity_paths_df[col]
        running_max = path.expanding().max()
        drawdowns = (path - running_max) / running_max
        max_drawdowns.append(abs(drawdowns.min()))

    avg_max_drawdown = np.mean(max_drawdowns)
    worst_drawdown = max(max_drawdowns)

    return {
        "final_equity": {
            "mean": mean_final,
            "median": median_final,
            "std": std_final,
            **percentiles,
        },
        "risk_metrics": {
            "probability_of_loss": prob_loss,
            "probability_of_large_loss": prob_large_loss,
            "avg_max_drawdown": avg_max_drawdown,
            "worst_drawdown": worst_drawdown,
        },
        "path_stats": {
            "avg_peak_equity": np.mean(max_equity_paths),
            "avg_trough_equity": np.mean(min_equity_paths),
        },
    }


def main():
    """Run comprehensive stress testing demonstration."""
    logger.info("🚀 Starting FX Quant Trading System Stress Testing Demo")
    logger.info("=" * 60)

    # Step 1: Load or generate trade data
    logger.info("📊 Step 1: Loading trade data...")
    trades_df = load_backtest_results()

    # Step 2: Configure stress tester
    logger.info("⚙️  Step 2: Configuring stress tester...")
    config = StressTestConfig(
        num_monte_carlo_paths=500,  # Reduced for demo speed
        confidence_levels=[0.90, 0.95, 0.99],
        random_seed=42,
    )
    stress_tester = MonteCarloStressTester(config)

    # Step 3: Run Monte Carlo simulation
    logger.info("🎲 Step 3: Running Monte Carlo simulation...")
    equity_paths = stress_tester.simulate_equity_paths(trades_df)

    # Step 4: Analyze Monte Carlo results
    logger.info("📈 Step 4: Analyzing Monte Carlo results...")
    mc_analysis = analyze_monte_carlo_results(equity_paths)

    print("\n" + "=" * 50)
    print("📊 MONTE CARLO ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Number of simulated paths: {len(equity_paths.columns)}")
    print(f"Mean final equity: ${mc_analysis['final_equity']['mean']:,.2f}")
    print(f"Median final equity: ${mc_analysis['final_equity']['median']:,.2f}")
    print(f"Standard deviation: ${mc_analysis['final_equity']['std']:,.2f}")
    print(f"Worst 5% outcome: ${mc_analysis['final_equity']['worst_5pct']:,.2f}")
    print(f"Best 5% outcome: ${mc_analysis['final_equity']['best_5pct']:,.2f}")
    print(
        f"Probability of loss: {mc_analysis['risk_metrics']['probability_of_loss']:.1%}"
    )
    print(
        f"Average max drawdown: {mc_analysis['risk_metrics']['avg_max_drawdown']:.1%}"
    )

    # Step 5: Run stress test scenarios
    logger.info("💥 Step 5: Running stress test scenarios...")

    # Use both default and custom scenarios
    default_scenarios = create_default_shock_scenarios()
    custom_scenarios = create_custom_shock_scenarios()
    all_scenarios = default_scenarios + custom_scenarios

    stress_results = stress_tester.run_stress_tests(trades_df, all_scenarios)

    # Step 6: Display stress test results
    print("\n" + "=" * 50)
    print("💥 STRESS TEST RESULTS")
    print("=" * 50)

    # Sort by final equity to show worst-case scenarios first
    stress_results_sorted = stress_results.sort_values("final_equity")

    for _, row in stress_results_sorted.iterrows():
        if row["scenario"] == "Baseline":
            continue  # Skip baseline for now

        baseline_equity = stress_results[stress_results["scenario"] == "Baseline"][
            "final_equity"
        ].iloc[0]
        equity_change = (
            (row["final_equity"] - baseline_equity) / baseline_equity
        ) * 100

        print(f"\n{row['scenario']}:")
        print(f"  Description: {row['description']}")
        print(f"  Final Equity: ${row['final_equity']:,.2f} ({equity_change:+.1f}%)")
        print(f"  Max Drawdown: {row['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        print(f"  Trades: {row['total_trades']}")

    # Step 7: Generate comprehensive report
    logger.info("📋 Step 7: Generating comprehensive report...")
    output_dir = "outputs/stress_testing_demo"
    run_comprehensive_stress_test(trades_df, output_dir)

    print(f"\n📁 Comprehensive results saved to: {output_dir}")
    print("   - monte_carlo_results.json")
    print("   - stress_test_results.csv")
    print("   - monte_carlo_paths.png")

    # Step 8: Risk management recommendations
    print("\n" + "=" * 50)
    print("🛡️  RISK MANAGEMENT RECOMMENDATIONS")
    print("=" * 50)

    worst_case = stress_results_sorted.iloc[0]
    prob_loss = mc_analysis["risk_metrics"]["probability_of_loss"]
    avg_drawdown = mc_analysis["risk_metrics"]["avg_max_drawdown"]

    print(f"1. Worst-case scenario: {worst_case['scenario']}")
    print(f"   - Potential loss: ${abs(worst_case['final_equity']):,.2f}")
    print(f"   - Recommendation: Maintain emergency reserves")

    print(f"\n2. Monte Carlo risk assessment:")
    print(f"   - Probability of loss: {prob_loss:.1%}")
    print(f"   - Average max drawdown: {avg_drawdown:.1%}")
    if prob_loss > 0.3:
        print("   - Recommendation: Consider reducing position sizes")
    if avg_drawdown > 0.2:
        print("   - Recommendation: Implement stricter stop-losses")

    print(f"\n3. Stress testing insights:")
    high_slippage_impact = stress_results[
        stress_results["scenario"] == "High Slippage"
    ]["final_equity"].iloc[0]
    baseline_equity = stress_results[stress_results["scenario"] == "Baseline"][
        "final_equity"
    ].iloc[0]
    slippage_sensitivity = abs(
        (high_slippage_impact - baseline_equity) / baseline_equity
    )

    if slippage_sensitivity > 0.15:
        print("   - High sensitivity to slippage - consider limit orders")

    print("\n🎯 Stress testing analysis complete!")
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
