#!/usr/bin/env python3
"""
Demonstration script for Prometheus Metrics Monitoring (Task 19)

This script shows how to integrate the monitoring infrastructure with
a simulated trading session, demonstrating real-time metrics collection.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from monitoring.metrics_server import (
    record_trade_execution,
    start_metrics_server,
    update_trading_session_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TradingSessionSimulator:
    """
    Simulates a live trading session with real-time metrics updates.
    """

    def __init__(self):
        self.metrics = None
        self.running = False
        self.start_time = datetime.now()
        self.current_equity = 100000.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.positions = {}
        self.regime_states = {
            "EURUSD": {"trending": 0.6, "ranging": 0.4},
            "GBPUSD": {"trending": 0.4, "ranging": 0.6},
            "USDJPY": {"trending": 0.7, "ranging": 0.3},
        }

    def start_monitoring(self, port: int = 9000):
        """Start the metrics server and begin monitoring."""
        logger.info("🚀 Starting Trading Session Monitoring Demo")
        logger.info("=" * 60)

        # Start metrics server
        self.metrics = start_metrics_server(port=port)

        logger.info(f"📊 Metrics server running on http://localhost:{port}/metrics")
        logger.info("🌐 Open in browser to see real-time metrics")
        logger.info("")

        # Initialize starting metrics
        self._initialize_session_metrics()

        self.running = True

    def _initialize_session_metrics(self):
        """Initialize session metrics."""
        # Set initial equity and system info
        self.metrics.update_pnl(self.current_equity, 0.0, 0.0)

        # Initialize regime distributions
        for symbol, regimes in self.regime_states.items():
            for regime, ratio in regimes.items():
                self.metrics.update_regime(symbol, regime, ratio)

        # Initialize system resource tracking
        self._update_system_metrics()

        logger.info("✓ Session metrics initialized")

    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Get current system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)

            self.metrics.update_system_resources(
                memory_mb=memory_info.used / (1024 * 1024), cpu_percent=cpu_percent
            )

        except Exception as e:
            logger.debug(f"Could not update system metrics: {e}")

    def simulate_trade_execution(self, symbol: str, side: str, quantity: int = 10000):
        """Simulate executing a trade with realistic timing and slippage."""

        # Simulate order processing time
        execution_start = time.time()

        # Random execution characteristics
        will_fill = np.random.random() > 0.05  # 95% fill rate
        execution_latency = np.random.uniform(0.010, 0.150)  # 10-150ms
        slippage_pips = np.random.uniform(0.1, 2.5) if will_fill else 0

        # Simulate processing delay
        time.sleep(execution_latency)

        if will_fill:
            # Calculate PnL (simplified)
            if side == "BUY":
                pnl = np.random.normal(50, 200)  # Mean profit with variance
            else:
                pnl = np.random.normal(45, 195)  # Slightly lower for sells

            # Adjust for slippage
            pnl -= slippage_pips * (quantity / 10000) * 2  # Rough slippage cost

            status = "filled"
            self.total_trades += 1
            self.daily_pnl += pnl
            self.current_equity += pnl

            if pnl > 0:
                self.winning_trades += 1

            # Update position tracking
            pos_key = f"{symbol}_{side}"
            if pos_key not in self.positions:
                self.positions[pos_key] = {"quantity": 0, "exposure": 0}

            self.positions[pos_key]["quantity"] += quantity
            self.positions[pos_key]["exposure"] += abs(pnl * 10)  # Rough exposure calc

        else:
            pnl = None
            status = "rejected"

        execution_time = time.time() - execution_start

        # Record all metrics
        record_trade_execution(
            symbol=symbol,
            side=side,
            status=status,
            latency_seconds=execution_time,
            slippage_pips=slippage_pips if will_fill else None,
            pnl=pnl,
        )

        # Update position metrics
        if will_fill:
            total_exposure = sum(pos["exposure"] for pos in self.positions.values())
            self.metrics.update_total_exposure(total_exposure)

            for pos_key, pos_data in self.positions.items():
                symbol_part, side_part = pos_key.split("_")
                self.metrics.update_position_exposure(
                    symbol_part, side_part, pos_data["exposure"]
                )

        # Log trade result
        if will_fill:
            logger.info(
                f"✅ {side} {symbol}: ${pnl:+.2f} PnL, "
                f"{execution_time*1000:.1f}ms, {slippage_pips:.1f} pips slippage"
            )
        else:
            logger.info(
                f"❌ {side} {symbol}: REJECTED after {execution_time*1000:.1f}ms"
            )

        return will_fill, pnl

    def simulate_strategy_signals(self):
        """Simulate strategy signal generation."""
        strategies = ["momentum", "mean_reversion", "breakout"]
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        signal_types = ["buy", "sell", "hold"]

        # Generate random signals
        for _ in range(np.random.randint(1, 4)):  # 1-3 signals
            strategy = np.random.choice(strategies)
            symbol = np.random.choice(symbols)
            signal_type = np.random.choice(signal_types)

            self.metrics.record_strategy_signal(strategy, signal_type, symbol)
            logger.debug(f"📡 Signal: {strategy} -> {signal_type} {symbol}")

        # Update strategy performance (mock)
        for strategy in strategies:
            # Random performance attribution
            perf = np.random.normal(0, 100)
            self.metrics.update_strategy_performance(strategy, perf)

    def simulate_market_data_updates(self):
        """Simulate market data feed updates."""
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
        data_types = ["tick", "quote", "trade"]

        for symbol in symbols:
            # Simulate data lag (realistic network latency)
            lag = np.random.uniform(0.005, 0.100)  # 5-100ms
            self.metrics.update_market_data_lag(symbol, lag)

            # Record data updates
            data_type = np.random.choice(data_types)
            self.metrics.record_market_data_update(symbol, data_type)

    def update_regime_analysis(self):
        """Update regime classification (simulate regime detection)."""
        for symbol in self.regime_states:
            # Slowly evolving regime probabilities
            current_trending = self.regime_states[symbol]["trending"]

            # Add some random walk to regime
            change = np.random.normal(0, 0.02)  # Small changes
            new_trending = np.clip(current_trending + change, 0.1, 0.9)
            new_ranging = 1.0 - new_trending

            self.regime_states[symbol]["trending"] = new_trending
            self.regime_states[symbol]["ranging"] = new_ranging

            # Update metrics
            self.metrics.update_regime(symbol, "trending", new_trending)
            self.metrics.update_regime(symbol, "ranging", new_ranging)

    def calculate_and_update_performance_metrics(self):
        """Calculate and update various performance metrics."""
        if self.total_trades > 0:
            # Win rate
            self.winning_trades / self.total_trades

            # Rough Sharpe ratio calculation
            if self.total_trades > 10:
                # Simulate some returns variance
                returns_std = abs(self.daily_pnl) * 0.1
                sharpe = (self.daily_pnl / returns_std) if returns_std > 0 else 0
            else:
                sharpe = 0

            # Drawdown (simplified)
            peak_equity = max(self.current_equity, 100000)
            current_drawdown = (peak_equity - self.current_equity) / peak_equity

            # Update metrics
            self.metrics.update_performance_ratios(sharpe)
            self.metrics.update_risk_metrics(
                current_drawdown=current_drawdown,
                max_drawdown=max(current_drawdown, 0.001),  # At least some small DD
            )

            # Update win rates per symbol
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            for symbol in symbols:
                # Random win rate per symbol (would be calculated from actual trades)
                symbol_win_rate = np.random.uniform(0.45, 0.65)
                self.metrics.update_win_rate(symbol, symbol_win_rate)

    def simulate_random_errors(self):
        """Simulate occasional system errors."""
        if np.random.random() < 0.05:  # 5% chance of error
            error_types = ["connection_timeout", "invalid_price", "insufficient_margin"]
            components = ["broker_api", "price_feed", "order_manager"]

            error_type = np.random.choice(error_types)
            component = np.random.choice(components)

            self.metrics.record_error(error_type, component)
            logger.warning(f"⚠️  Simulated error: {error_type} in {component}")

    def run_trading_session(
        self, duration_minutes: int = 10, trades_per_minute: float = 2.0
    ):
        """Run a simulated trading session with real-time metrics updates."""

        if not self.running:
            logger.error("Monitoring not started. Call start_monitoring() first.")
            return

        logger.info(f"📈 Starting {duration_minutes}-minute trading session simulation")
        logger.info(f"🎯 Target: {trades_per_minute:.1f} trades per minute")
        logger.info("")

        end_time = time.time() + (duration_minutes * 60)
        last_update = time.time()

        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        sides = ["BUY", "SELL"]

        iteration = 0

        while time.time() < end_time and self.running:
            iteration += 1

            # Simulate trading activity
            if np.random.random() < (trades_per_minute / 60):  # Probability per second
                symbol = np.random.choice(symbols)
                side = np.random.choice(sides)

                self.simulate_trade_execution(symbol, side)

            # Update various metrics periodically
            if time.time() - last_update > 5:  # Every 5 seconds
                self.simulate_strategy_signals()
                self.simulate_market_data_updates()
                self.update_regime_analysis()
                self.calculate_and_update_performance_metrics()
                self.simulate_random_errors()
                self._update_system_metrics()

                # Update session metrics
                update_trading_session_metrics(
                    equity=self.current_equity,
                    daily_pnl=self.daily_pnl,
                    total_exposure=sum(
                        pos["exposure"] for pos in self.positions.values()
                    ),
                    sharpe_ratio=np.random.uniform(0.5, 2.5),  # Mock Sharpe
                    current_drawdown=max(0, (100000 - self.current_equity) / 100000),
                )

                last_update = time.time()

                # Print session summary
                elapsed_minutes = (time.time() - self.start_time.timestamp()) / 60
                logger.info(
                    f"📊 Session update ({elapsed_minutes:.1f}min): "
                    f"Equity=${self.current_equity:,.2f}, "
                    f"Daily PnL=${self.daily_pnl:+,.2f}, "
                    f"Trades={self.total_trades}"
                )

            # Sleep briefly to simulate real-time
            time.sleep(0.1)

        # Final session summary
        self._print_session_summary()

    def _print_session_summary(self):
        """Print final session summary."""
        logger.info("")
        logger.info("🎯 TRADING SESSION SUMMARY")
        logger.info("=" * 40)
        logger.info(
            f"Duration: {(datetime.now() - self.start_time).total_seconds()/60:.1f} minutes"
        )
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        logger.info(
            f"Win Rate: {(self.winning_trades/self.total_trades)*100 if self.total_trades > 0 else 0:.1f}%"
        )
        logger.info(f"Starting Equity: $100,000.00")
        logger.info(f"Final Equity: ${self.current_equity:,.2f}")
        logger.info(f"Daily PnL: ${self.daily_pnl:+,.2f}")
        logger.info(f"Return: {((self.current_equity - 100000) / 100000) * 100:+.2f}%")
        logger.info("")

    def stop_monitoring(self):
        """Stop the trading session monitoring."""
        self.running = False
        if self.metrics:
            self.metrics.stop_server()
        logger.info("🛑 Monitoring stopped")


def main():
    """Run the comprehensive monitoring demonstration."""

    # Create simulator
    simulator = TradingSessionSimulator()

    try:
        # Start monitoring
        simulator.start_monitoring(port=9000)

        print("\n" + "=" * 60)
        print("🚀 FX QUANT TRADING SYSTEM - MONITORING DEMO")
        print("=" * 60)
        print("📊 Metrics Endpoint: http://localhost:9000/metrics")
        print("🌐 Open in browser to see real-time Prometheus metrics")
        print("")
        print("📈 Starting live trading session simulation...")
        print("⏱️  The session will run for 30 minutes with live updates")
        print("")
        print("📋 Metrics being tracked:")
        print("   • Real-time PnL and equity")
        print("   • Trade execution (latency, slippage, fills)")
        print("   • Position exposures and counts")
        print("   • Strategy signals and performance")
        print("   • Market regime classification")
        print("   • System health and errors")
        print("   • Risk metrics (drawdown, VaR)")
        print("")

        # Run trading session
        simulator.run_trading_session(duration_minutes=30, trades_per_minute=3.0)

        print("\n" + "=" * 60)
        print("✅ MONITORING DEMONSTRATION COMPLETED")
        print("=" * 60)
        print("📊 Key Achievements:")
        print("   ✓ Prometheus metrics server running")
        print("   ✓ Real-time trading metrics collection")
        print("   ✓ System health monitoring active")
        print("   ✓ Performance analytics updated")
        print("   ✓ Risk metrics calculated")
        print("")
        print("🔗 Next Steps:")
        print("   • Configure Prometheus to scrape http://localhost:9000/metrics")
        print("   • Set up Grafana dashboards for visualization")
        print("   • Configure alerting rules for critical metrics")
        print("   • Integrate with live trading system")
        print("")
        print("⏳ Server will continue running for 30 seconds...")
        print("   Visit http://localhost:9000/metrics to see final metrics")

        # Keep server running briefly for inspection
        time.sleep(30)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise
    finally:
        # Clean shutdown
        simulator.stop_monitoring()
        print("�� Demo completed!")


if __name__ == "__main__":
    main()
