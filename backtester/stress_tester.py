"""
Monte Carlo and Stress Testing Module for FX Quant Trading System

This module provides comprehensive stress testing and Monte Carlo simulation
capabilities for evaluating strategy robustness under uncertainty.

Features:
- Bootstrap resampling of trades for equity simulation
- Monte Carlo parameter sampling
- Stress testing (slippage, latency, execution failures)
- Tail risk metrics (VaR, CVaR)
- Scenario analysis and comparison
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress testing scenarios."""

    num_monte_carlo_paths: int = 1000
    confidence_levels: list[float] = None
    random_seed: int = 42

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


@dataclass
class ShockScenario:
    """Definition of a shock scenario for stress testing."""

    name: str
    slippage_multiplier: float = 1.0  # 1.0 = no change, 2.0 = double slippage
    latency_ms_add: int = 0  # Additional latency in milliseconds
    commission_multiplier: float = 1.0  # 1.0 = no change, 2.0 = double commission
    fill_rate_reduction: float = 0.0  # 0.0 = no change, 0.1 = 10% fewer fills
    volatility_multiplier: float = 1.0  # Market volatility shock
    description: str = ""


class MonteCarloStressTester:
    """
    Monte Carlo and Stress Testing engine for backtesting results.

    This class provides comprehensive tools for evaluating strategy robustness
    through various simulation and stress testing techniques.
    """

    def __init__(self, config: StressTestConfig = None):
        """Initialize the stress tester with configuration."""
        self.config = config or StressTestConfig()
        self.results_cache = {}

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        logger.info(
            f"MonteCarloStressTester initialized with {self.config.num_monte_carlo_paths} paths"
        )

    def simulate_equity_paths(
        self, trades_df: pd.DataFrame, num_paths: int = None, seed: int = None
    ) -> pd.DataFrame:
        """
        Generate multiple equity curves using bootstrap resampling of trades.

        Args:
            trades_df: DataFrame with trade data including 'pnl' and 'timestamp' columns
            num_paths: Number of Monte Carlo paths to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with shape (timesteps, num_paths) containing equity curves
        """
        if num_paths is None:
            num_paths = self.config.num_monte_carlo_paths

        if seed is not None:
            np.random.seed(seed)

        logger.info(
            f"Generating {num_paths} Monte Carlo equity paths from {len(trades_df)} trades"
        )

        # Validate input data
        required_columns = ["pnl", "timestamp"]
        missing_columns = [
            col for col in required_columns if col not in trades_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(trades_df) == 0:
            logger.warning("No trades provided for Monte Carlo simulation")
            return pd.DataFrame()

        # Extract trade returns
        trade_returns = trades_df["pnl"].values
        trade_timestamps = pd.to_datetime(trades_df["timestamp"])

        # Create time index for equity curves
        start_time = trade_timestamps.min()
        end_time = trade_timestamps.max()
        time_index = pd.date_range(start=start_time, end=end_time, freq="1h")

        # Initialize equity paths matrix
        equity_paths = np.zeros((len(time_index), num_paths))

        # Generate Monte Carlo paths
        for path_idx in range(num_paths):
            # Bootstrap resample trades (with replacement)
            resampled_indices = np.random.choice(
                len(trade_returns), size=len(trade_returns), replace=True
            )
            resampled_returns = trade_returns[resampled_indices]
            resampled_times = trade_timestamps.iloc[resampled_indices].sort_values()

            # Create equity curve for this path
            equity_curve = np.zeros(len(time_index))
            current_equity = 0

            for i, timestamp in enumerate(time_index):
                # Add returns that occurred before this timestamp
                applicable_returns = resampled_returns[resampled_times <= timestamp]
                if len(applicable_returns) > 0:
                    current_equity = applicable_returns.sum()

                equity_curve[i] = current_equity

            equity_paths[:, path_idx] = equity_curve

        # Convert to DataFrame
        path_columns = [f"path_{i}" for i in range(num_paths)]
        equity_df = pd.DataFrame(equity_paths, index=time_index, columns=path_columns)

        logger.info(
            f"Generated Monte Carlo paths with final equity range: "
            f"[{equity_df.iloc[-1].min():.2f}, {equity_df.iloc[-1].max():.2f}]"
        )

        return equity_df

    def apply_shocks(
        self, trades_df: pd.DataFrame, scenario: ShockScenario
    ) -> pd.DataFrame:
        """
        Apply shock scenario to trade data to simulate execution degradation.

        Args:
            trades_df: Original trade data
            scenario: Shock scenario to apply

        Returns:
            Modified trade data with shocks applied
        """
        logger.info(f"Applying shock scenario: {scenario.name}")

        # Create a copy to avoid modifying original data
        shocked_trades = trades_df.copy()

        # Apply slippage shock
        if "slippage" in shocked_trades.columns and scenario.slippage_multiplier != 1.0:
            original_slippage = shocked_trades["slippage"].abs().mean()
            shocked_trades["slippage"] *= scenario.slippage_multiplier
            new_slippage = shocked_trades["slippage"].abs().mean()
            logger.debug(
                f"Slippage shock: {original_slippage:.6f} -> {new_slippage:.6f}"
            )

        # Apply commission shock
        if (
            "commission" in shocked_trades.columns
            and scenario.commission_multiplier != 1.0
        ):
            original_commission = shocked_trades["commission"].sum()
            shocked_trades["commission"] *= scenario.commission_multiplier
            new_commission = shocked_trades["commission"].sum()
            logger.debug(
                f"Commission shock: {original_commission:.2f} -> {new_commission:.2f}"
            )

        # Apply fill rate reduction (randomly remove some fills)
        if scenario.fill_rate_reduction > 0:
            num_to_remove = int(len(shocked_trades) * scenario.fill_rate_reduction)
            if num_to_remove > 0:
                indices_to_remove = np.random.choice(
                    shocked_trades.index, size=num_to_remove, replace=False
                )
                shocked_trades = shocked_trades.drop(indices_to_remove)
                logger.debug(
                    f"Fill rate shock: Removed {num_to_remove} trades "
                    f"({scenario.fill_rate_reduction:.1%} reduction)"
                )

        # Recalculate PnL with shocks applied
        # Instead of recalculating from scratch, adjust the existing PnL by shock impacts
        if (
            "slippage" in shocked_trades.columns
            or "commission" in shocked_trades.columns
        ):
            # Calculate additional costs from shocks
            pass

            if "slippage" in shocked_trades.columns:
                # Slippage impact: additional cost per unit traded
                # For FX, slippage is typically in price units, so multiply by quantity
                slippage_cost_adjustment = (
                    shocked_trades["slippage"]
                    * (scenario.slippage_multiplier - 1.0)
                    * abs(shocked_trades.get("quantity", 1))
                )
                shocked_trades["pnl"] -= slippage_cost_adjustment

            if "commission" in shocked_trades.columns:
                # Commission impact: additional commission costs
                commission_cost_adjustment = shocked_trades["commission"] * (
                    scenario.commission_multiplier - 1.0
                )
                shocked_trades["pnl"] -= commission_cost_adjustment

        # Apply volatility shock to price movements (if applicable)
        if scenario.volatility_multiplier != 1.0 and "price" in shocked_trades.columns:
            price_changes = shocked_trades["price"].pct_change().fillna(0)
            shocked_price_changes = price_changes * scenario.volatility_multiplier

            # Reconstruct prices with shocked volatility
            base_price = shocked_trades["price"].iloc[0]
            shocked_prices = [base_price]
            for change in shocked_price_changes[1:]:
                new_price = shocked_prices[-1] * (1 + change)
                shocked_prices.append(new_price)

            shocked_trades["price"] = shocked_prices
            logger.debug(
                f"Volatility shock applied: {scenario.volatility_multiplier}x multiplier"
            )

        logger.info(
            f"Shock scenario '{scenario.name}' applied. "
            f"Trades: {len(trades_df)} -> {len(shocked_trades)}"
        )

        return shocked_trades

    def calculate_var_cvar(
        self, equity_curve: pd.Series, confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Calculate Value-at-Risk (VaR) and Conditional VaR (CVaR) from equity curve.

        Args:
            equity_curve: Time series of equity values
            confidence: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            Tuple of (VaR, CVaR) values
        """
        if len(equity_curve) < 2:
            logger.warning("Insufficient data for VaR/CVaR calculation")
            return 0.0, 0.0

        # Calculate daily returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0:
            logger.warning("No valid returns for VaR/CVaR calculation")
            return 0.0, 0.0

        # Calculate VaR (percentile of loss distribution)
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)

        # Calculate CVaR (expected value of losses beyond VaR)
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var

        logger.debug(
            f"VaR/CVaR calculated: VaR({confidence:.1%}) = {var:.4f}, "
            f"CVaR({confidence:.1%}) = {cvar:.4f}"
        )

        return var, cvar

    def run_stress_tests(
        self, trades_df: pd.DataFrame, scenarios: list[ShockScenario]
    ) -> pd.DataFrame:
        """
        Run multiple stress test scenarios and compare results.

        Args:
            trades_df: Original trade data
            scenarios: List of shock scenarios to test

        Returns:
            DataFrame with performance metrics for each scenario
        """
        logger.info(f"Running stress tests with {len(scenarios)} scenarios")

        results = []

        # Calculate baseline metrics (no shock)
        baseline_equity = trades_df["pnl"].cumsum()
        baseline_var, baseline_cvar = self.calculate_var_cvar(baseline_equity)
        baseline_metrics = self._calculate_performance_metrics(
            trades_df, baseline_equity
        )
        baseline_metrics.update(
            {
                "scenario": "Baseline",
                "var_95": baseline_var,
                "cvar_95": baseline_cvar,
                "description": "Original backtest results",
            }
        )
        results.append(baseline_metrics)

        # Run each shock scenario
        for scenario in scenarios:
            try:
                # Apply shocks
                shocked_trades = self.apply_shocks(trades_df, scenario)

                if len(shocked_trades) == 0:
                    logger.warning(
                        f"No trades remaining after shock scenario: {scenario.name}"
                    )
                    continue

                # Calculate metrics for shocked scenario
                shocked_equity = shocked_trades["pnl"].cumsum()
                var, cvar = self.calculate_var_cvar(shocked_equity)
                metrics = self._calculate_performance_metrics(
                    shocked_trades, shocked_equity
                )

                metrics.update(
                    {
                        "scenario": scenario.name,
                        "var_95": var,
                        "cvar_95": cvar,
                        "description": scenario.description,
                        "slippage_multiplier": scenario.slippage_multiplier,
                        "latency_ms_add": scenario.latency_ms_add,
                        "commission_multiplier": scenario.commission_multiplier,
                        "fill_rate_reduction": scenario.fill_rate_reduction,
                        "volatility_multiplier": scenario.volatility_multiplier,
                    }
                )

                results.append(metrics)

            except Exception as e:
                logger.error(f"Error in stress test scenario '{scenario.name}': {e}")
                continue

        results_df = pd.DataFrame(results)
        logger.info(f"Stress testing completed. {len(results_df)} scenarios analyzed.")

        return results_df

    def _calculate_performance_metrics(
        self, trades_df: pd.DataFrame, equity_curve: pd.Series
    ) -> dict[str, float]:
        """Calculate standard performance metrics for a trade set."""
        if len(trades_df) == 0 or len(equity_curve) == 0:
            return {
                "total_trades": 0,
                "final_equity": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0,
            }

        # Basic metrics
        total_trades = len(trades_df)
        final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0
        initial_equity = equity_curve.iloc[0] if len(equity_curve) > 0 else 0
        total_return = (
            (final_equity - initial_equity) / abs(initial_equity)
            if initial_equity != 0
            else 0
        )

        # Drawdown calculation
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Returns-based metrics
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = (
            returns.mean() / returns.std()
            if len(returns) > 0 and returns.std() > 0
            else 0
        )

        # Trade-based metrics
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
            if gross_profit > 0
            else 0
        )

        avg_trade = trades_df["pnl"].mean() if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
        }

    def generate_monte_carlo_report(
        self, equity_paths_df: pd.DataFrame, output_dir: str = "outputs/stress_testing"
    ) -> dict[str, Any]:
        """
        Generate comprehensive Monte Carlo analysis report.

        Args:
            equity_paths_df: DataFrame with Monte Carlo equity paths
            output_dir: Directory to save outputs

        Returns:
            Dictionary with analysis results
        """
        logger.info("Generating Monte Carlo analysis report")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate statistics across paths
        final_values = equity_paths_df.iloc[-1]

        # Percentile analysis
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = {
            f"p{p}": np.percentile(final_values, p) for p in percentiles
        }

        # Risk metrics
        mean_final = final_values.mean()
        std_final = final_values.std()

        # Probability of loss
        prob_loss = (final_values < 0).mean()

        # VaR and CVaR for final values
        var_95 = np.percentile(final_values, 5)
        cvar_95 = final_values[final_values <= var_95].mean()

        # Path-wise Sharpe ratios
        path_sharpes = []
        for col in equity_paths_df.columns:
            returns = equity_paths_df[col].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std()
                path_sharpes.append(sharpe)

        avg_sharpe = np.mean(path_sharpes) if path_sharpes else 0

        # Compile results
        results = {
            "num_paths": len(equity_paths_df.columns),
            "final_equity_stats": {
                "mean": mean_final,
                "std": std_final,
                "min": final_values.min(),
                "max": final_values.max(),
                **percentile_values,
            },
            "risk_metrics": {
                "probability_of_loss": prob_loss,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "avg_sharpe_ratio": avg_sharpe,
            },
            "confidence_bands": {
                "lower_5pct": equity_paths_df.quantile(0.05, axis=1),
                "upper_95pct": equity_paths_df.quantile(0.95, axis=1),
                "median": equity_paths_df.quantile(0.5, axis=1),
            },
        }

        # Save results to JSON
        results_file = output_path / "monte_carlo_results.json"
        with open(results_file, "w") as f:
            # Convert pandas Series to lists for JSON serialization
            json_results = results.copy()
            for band_name, series in json_results["confidence_bands"].items():
                json_results["confidence_bands"][band_name] = series.tolist()

            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Monte Carlo report saved to {results_file}")

        return results

    def plot_monte_carlo_paths(
        self,
        equity_paths_df: pd.DataFrame,
        output_dir: str = "outputs/stress_testing",
        show_individual_paths: bool = False,
        max_paths_to_show: int = 100,
    ) -> str:
        """
        Create Monte Carlo equity paths visualization.

        Args:
            equity_paths_df: DataFrame with Monte Carlo equity paths
            output_dir: Directory to save plot
            show_individual_paths: Whether to show individual paths
            max_paths_to_show: Maximum number of individual paths to display

        Returns:
            Path to saved plot file
        """
        logger.info("Creating Monte Carlo paths visualization")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set up the plot
        plt.figure(figsize=(12, 8))

        # Calculate confidence bands
        lower_5 = equity_paths_df.quantile(0.05, axis=1)
        upper_95 = equity_paths_df.quantile(0.95, axis=1)
        median = equity_paths_df.quantile(0.5, axis=1)
        mean_path = equity_paths_df.mean(axis=1)

        # Plot confidence bands
        plt.fill_between(
            equity_paths_df.index,
            lower_5,
            upper_95,
            alpha=0.2,
            color="blue",
            label="90% Confidence Band",
        )

        # Plot individual paths (sample if too many)
        if show_individual_paths:
            paths_to_plot = min(max_paths_to_show, len(equity_paths_df.columns))
            sample_columns = np.random.choice(
                equity_paths_df.columns, size=paths_to_plot, replace=False
            )

            for col in sample_columns:
                plt.plot(
                    equity_paths_df.index,
                    equity_paths_df[col],
                    alpha=0.1,
                    color="gray",
                    linewidth=0.5,
                )

        # Plot key statistics
        plt.plot(
            equity_paths_df.index,
            median,
            color="blue",
            linewidth=2,
            label="Median Path",
        )
        plt.plot(
            equity_paths_df.index,
            mean_path,
            color="red",
            linewidth=2,
            label="Mean Path",
        )

        # Formatting
        plt.title(
            f"Monte Carlo Equity Paths ({len(equity_paths_df.columns)} simulations)"
        )
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics text box
        final_mean = equity_paths_df.iloc[-1].mean()
        final_std = equity_paths_df.iloc[-1].std()
        prob_loss = (equity_paths_df.iloc[-1] < 0).mean()

        stats_text = f"""Final Equity Statistics:
Mean: ${final_mean:,.0f}
Std: ${final_std:,.0f}
P(Loss): {prob_loss:.1%}
90% Range: [${lower_5.iloc[-1]:,.0f}, ${upper_95.iloc[-1]:,.0f}]"""

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Save plot
        plot_file = output_path / "monte_carlo_paths.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Monte Carlo paths plot saved to {plot_file}")

        return str(plot_file)


def create_default_shock_scenarios() -> list[ShockScenario]:
    """Create a set of default shock scenarios for stress testing."""
    scenarios = [
        ShockScenario(
            name="High Slippage",
            slippage_multiplier=3.0,
            description="3x normal slippage (poor liquidity conditions)",
        ),
        ShockScenario(
            name="High Latency",
            latency_ms_add=200,
            description="Additional 200ms execution latency",
        ),
        ShockScenario(
            name="High Commission",
            commission_multiplier=2.0,
            description="Double commission costs",
        ),
        ShockScenario(
            name="Poor Fill Rate",
            fill_rate_reduction=0.15,
            description="15% reduction in fill rate",
        ),
        ShockScenario(
            name="High Volatility",
            volatility_multiplier=2.0,
            description="2x market volatility",
        ),
        ShockScenario(
            name="Combined Stress",
            slippage_multiplier=2.0,
            latency_ms_add=100,
            commission_multiplier=1.5,
            fill_rate_reduction=0.10,
            volatility_multiplier=1.5,
            description="Combined moderate stress across all factors",
        ),
        ShockScenario(
            name="Extreme Stress",
            slippage_multiplier=5.0,
            latency_ms_add=500,
            commission_multiplier=3.0,
            fill_rate_reduction=0.25,
            volatility_multiplier=3.0,
            description="Extreme stress scenario (market crisis conditions)",
        ),
    ]

    return scenarios


# Example usage and CLI integration functions
def run_comprehensive_stress_test(
    trades_df: pd.DataFrame, output_dir: str = "outputs/stress_testing"
) -> dict[str, Any]:
    """
    Run a comprehensive stress test analysis on trade data.

    Args:
        trades_df: DataFrame with trade data
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    logger.info("Starting comprehensive stress test analysis")

    # Initialize stress tester
    config = StressTestConfig(num_monte_carlo_paths=1000)
    stress_tester = MonteCarloStressTester(config)

    # Generate Monte Carlo paths
    equity_paths = stress_tester.simulate_equity_paths(trades_df)

    # Generate Monte Carlo report
    mc_results = stress_tester.generate_monte_carlo_report(equity_paths, output_dir)

    # Create visualization
    plot_path = stress_tester.plot_monte_carlo_paths(
        equity_paths, output_dir, show_individual_paths=True
    )

    # Run stress test scenarios
    scenarios = create_default_shock_scenarios()
    stress_results = stress_tester.run_stress_tests(trades_df, scenarios)

    # Save stress test results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stress_results.to_csv(output_path / "stress_test_results.csv", index=False)

    # Compile comprehensive results
    comprehensive_results = {
        "monte_carlo": mc_results,
        "stress_tests": stress_results.to_dict("records"),
        "plot_path": plot_path,
        "summary": {
            "num_monte_carlo_paths": len(equity_paths.columns),
            "num_stress_scenarios": len(stress_results),
            "baseline_final_equity": trades_df["pnl"].sum(),
            "monte_carlo_mean_final": mc_results["final_equity_stats"]["mean"],
            "worst_case_scenario": stress_results.loc[
                stress_results["final_equity"].idxmin(), "scenario"
            ],
        },
    }

    logger.info("Comprehensive stress test analysis completed")

    return comprehensive_results
