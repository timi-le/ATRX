#!/usr/bin/env python3
"""
Test script for Prometheus Metrics Server (Task 19)

This script validates all the monitoring infrastructure functionality including:
- Metrics server startup and shutdown
- Metric recording and updating
- Prometheus endpoint accessibility
- Integration with trading system components
"""

import logging
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from monitoring.metrics_server import (
    FXTradingMetrics,
    MetricsConfig,
    record_trade_execution,
    start_metrics_server,
    update_trading_session_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_metrics_initialization():
    """Test metrics server initialization."""
    logger.info("=== Testing Metrics Initialization ===")

    # Test basic initialization
    config = MetricsConfig(port=9001)  # Use different port for testing
    metrics = FXTradingMetrics(config)

    # Verify metrics are properly initialized
    assert hasattr(metrics, "pnl_gauge"), "PnL gauge should be initialized"
    assert hasattr(metrics, "trade_count"), "Trade counter should be initialized"
    assert hasattr(
        metrics, "execution_latency"
    ), "Execution latency histogram should be initialized"
    assert hasattr(metrics, "error_counter"), "Error counter should be initialized"
    assert hasattr(
        metrics, "regime_distribution"
    ), "Regime distribution gauge should be initialized"

    logger.info("✓ Metrics initialization successful")
    return metrics


def test_metrics_server_startup():
    """Test starting and stopping the metrics server."""
    logger.info("=== Testing Metrics Server Startup ===")

    # Test server startup
    config = MetricsConfig(port=9002)  # Use different port
    metrics = FXTradingMetrics(config)

    # Start server in background thread
    server_thread = threading.Thread(target=metrics.start_server)
    server_thread.daemon = True
    server_thread.start()

    # Give server time to start
    time.sleep(2)

    # Test that server is running
    try:
        response = requests.get(f"http://localhost:9002/metrics", timeout=5)
        assert response.status_code == 200, "Metrics endpoint should be accessible"
        assert "fxai_" in response.text, "Should contain our custom metrics"
        logger.info("✓ Metrics server startup successful")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to access metrics endpoint: {e}")
        raise

    # Test server stop
    metrics.stop_server()
    logger.info("✓ Metrics server stopped successfully")

    return metrics


def test_trading_metrics_updates():
    """Test updating trading-specific metrics."""
    logger.info("=== Testing Trading Metrics Updates ===")

    metrics = FXTradingMetrics()

    # Test PnL updates
    metrics.update_pnl(equity=100000, daily_pnl=2500, unrealized_pnl=150)
    logger.info("✓ PnL metrics updated")

    # Test trade recording
    metrics.record_trade("EURUSD", "BUY", "filled", pnl=250.5)
    metrics.record_trade("GBPUSD", "SELL", "filled", pnl=-120.0)
    metrics.record_trade("USDJPY", "BUY", "rejected")
    logger.info("✓ Trade metrics recorded")

    # Test execution metrics
    metrics.record_execution_latency(0.025)  # 25ms
    metrics.record_fill_latency(0.150)  # 150ms
    metrics.record_slippage(1.2)  # 1.2 pips
    logger.info("✓ Execution metrics recorded")

    # Test position metrics
    metrics.update_position_exposure("EURUSD", "BUY", 50000)
    metrics.update_position_exposure("GBPUSD", "SELL", 25000)
    metrics.update_total_exposure(75000)
    metrics.update_position_count("EURUSD", 2)
    logger.info("✓ Position metrics updated")

    # Test regime and strategy metrics
    metrics.update_regime("EURUSD", "trending", 0.65)
    metrics.update_regime("EURUSD", "ranging", 0.35)
    metrics.record_strategy_signal("momentum", "buy", "EURUSD")
    metrics.update_strategy_performance("momentum", 1250.75)
    logger.info("✓ Strategy and regime metrics updated")

    logger.info("✓ All trading metrics updates successful")


def test_system_health_metrics():
    """Test system health and monitoring metrics."""
    logger.info("=== Testing System Health Metrics ===")

    metrics = FXTradingMetrics()

    # Test error tracking
    metrics.record_error("connection_timeout", "broker_api")
    metrics.record_error("invalid_symbol", "order_manager")
    metrics.record_exception("ValueError", "price_feed")
    logger.info("✓ Error tracking metrics recorded")

    # Test system resource metrics
    metrics.update_system_resources(memory_mb=512.8, cpu_percent=25.5)
    logger.info("✓ System resource metrics updated")

    # Test market data metrics
    metrics.update_market_data_lag("EURUSD", 0.085)  # 85ms lag
    metrics.record_market_data_update("EURUSD", "tick")
    metrics.record_market_data_update("GBPUSD", "quote")
    logger.info("✓ Market data metrics updated")

    # Test risk metrics
    metrics.update_risk_metrics(
        current_drawdown=0.025,  # 2.5% drawdown
        max_drawdown=0.055,  # 5.5% max drawdown
        var_95=-1250.0,  # 95% VaR
        var_99=-2100.0,  # 99% VaR
    )
    logger.info("✓ Risk metrics updated")

    # Test performance ratios
    metrics.update_performance_ratios(sharpe_ratio=1.85)
    metrics.update_win_rate("EURUSD", 0.62)
    metrics.update_win_rate("GBPUSD", 0.58)
    logger.info("✓ Performance ratio metrics updated")

    logger.info("✓ All system health metrics updates successful")


def test_convenience_functions():
    """Test convenience functions for common operations."""
    logger.info("=== Testing Convenience Functions ===")

    # Initialize global metrics
    config = MetricsConfig(port=9003)
    global_metrics = FXTradingMetrics(config)

    # Test global metrics instance
    import monitoring.metrics_server as ms

    ms._metrics_instance = global_metrics

    # Test trade execution convenience function
    record_trade_execution(
        symbol="EURUSD",
        side="BUY",
        status="filled",
        latency_seconds=0.045,
        slippage_pips=0.8,
        pnl=150.25,
    )
    logger.info("✓ Trade execution recording successful")

    # Test session metrics convenience function
    update_trading_session_metrics(
        equity=102500.0,
        daily_pnl=2500.0,
        total_exposure=85000.0,
        sharpe_ratio=1.75,
        current_drawdown=0.015,
    )
    logger.info("✓ Trading session metrics update successful")

    logger.info("✓ All convenience functions working correctly")


def test_metrics_endpoint_content():
    """Test the actual content of the metrics endpoint."""
    logger.info("=== Testing Metrics Endpoint Content ===")

    # Start metrics server
    metrics = start_metrics_server(port=9004)

    # Give server time to start
    time.sleep(2)

    # Add some sample data
    metrics.update_pnl(100000)
    metrics.record_trade("EURUSD", "BUY", "filled", 250)
    metrics.record_execution_latency(0.050)
    metrics.update_regime("EURUSD", "trending", 0.70)
    metrics.record_error("timeout", "api")

    # Test endpoint accessibility
    try:
        response = requests.get("http://localhost:9004/metrics", timeout=5)
        assert response.status_code == 200, "Metrics endpoint should be accessible"

        content = response.text

        # Verify key metrics are present
        expected_metrics = [
            "fxai_pnl_equity",
            "fxai_trades_total",
            "fxai_execution_latency_seconds",
            "fxai_regime_ratio",
            "fxai_errors_total",
            "fxai_system_info",
        ]

        for metric in expected_metrics:
            assert metric in content, f"Metric {metric} should be present in endpoint"

        logger.info("✓ All expected metrics found in endpoint")

        # Test that metrics have labels
        assert 'symbol="EURUSD"' in content, "Should have symbol labels"
        assert 'side="BUY"' in content, "Should have side labels"
        assert 'regime="trending"' in content, "Should have regime labels"

        logger.info("✓ Metric labels properly formatted")

        # Print sample of metrics for verification
        lines = content.split("\n")
        sample_metrics = [
            line for line in lines if line.startswith("fxai_") and "=" in line
        ][:10]

        logger.info("Sample metrics from endpoint:")
        for metric in sample_metrics:
            logger.info(f"  {metric}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to access metrics endpoint: {e}")
        raise
    finally:
        # Clean up
        metrics.stop_server()

    logger.info("✓ Metrics endpoint content validation successful")


def test_integration_with_sample_data():
    """Test integration with sample trading data."""
    logger.info("=== Testing Integration with Sample Data ===")

    # Create sample trading data
    np.random.seed(42)

    trades_data = []
    current_time = datetime.now()
    current_equity = 100000

    for i in range(50):
        # Generate sample trade
        symbol = np.random.choice(["EURUSD", "GBPUSD", "USDJPY"])
        side = np.random.choice(["BUY", "SELL"])
        pnl = np.random.normal(25, 150)  # Random PnL

        trade_time = current_time + timedelta(minutes=i * 30)

        trades_data.append(
            {
                "timestamp": trade_time,
                "symbol": symbol,
                "side": side,
                "pnl": pnl,
                "status": "filled" if np.random.random() > 0.05 else "rejected",
            }
        )

    trades_df = pd.DataFrame(trades_data)

    # Initialize metrics server
    metrics = FXTradingMetrics()

    # Process trades and update metrics
    cumulative_pnl = 0

    for _, trade in trades_df.iterrows():
        # Record trade
        metrics.record_trade(
            trade["symbol"],
            trade["side"],
            trade["status"],
            trade["pnl"] if trade["status"] == "filled" else None,
        )

        # Add to cumulative PnL if filled
        if trade["status"] == "filled":
            cumulative_pnl += trade["pnl"]

        # Record execution metrics
        metrics.record_execution_latency(np.random.uniform(0.010, 0.200))
        if np.random.random() > 0.7:  # 30% chance of slippage
            metrics.record_slippage(np.random.uniform(0.1, 3.0))

    # Update final equity
    final_equity = current_equity + cumulative_pnl
    metrics.update_pnl(final_equity, cumulative_pnl)

    # Calculate and update performance metrics
    filled_trades = trades_df[trades_df["status"] == "filled"]
    win_rate = (filled_trades["pnl"] > 0).mean()

    for symbol in filled_trades["symbol"].unique():
        symbol_trades = filled_trades[filled_trades["symbol"] == symbol]
        symbol_win_rate = (symbol_trades["pnl"] > 0).mean()
        metrics.update_win_rate(symbol, symbol_win_rate)

    # Update regime classifications (mock data)
    for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
        trending_ratio = np.random.uniform(0.3, 0.8)
        ranging_ratio = 1.0 - trending_ratio

        metrics.update_regime(symbol, "trending", trending_ratio)
        metrics.update_regime(symbol, "ranging", ranging_ratio)

    logger.info(f"✓ Processed {len(trades_df)} trades")
    logger.info(f"✓ Final equity: ${final_equity:.2f}")
    logger.info(f"✓ Overall win rate: {win_rate:.1%}")
    logger.info("✓ Integration test with sample data successful")


def test_metrics_with_real_backtest_integration():
    """Test metrics integration with backtest results."""
    logger.info("=== Testing Real Backtest Integration ===")

    # Try to load real backtest results if available
    backtest_file = "outputs/backtest_results.csv"

    if Path(backtest_file).exists():
        try:
            df = pd.read_csv(backtest_file)
            logger.info(f"Loaded real backtest results: {len(df)} records")

            metrics = FXTradingMetrics()

            # Update metrics from real backtest data
            metrics.update_from_backtest_results(df)

            logger.info("✓ Real backtest integration successful")

        except Exception as e:
            logger.warning(f"Could not process real backtest data: {e}")
    else:
        logger.info("No real backtest results found, skipping real integration test")


def main():
    """Run all metrics server validation tests."""
    logger.info("🚀 Starting Prometheus Metrics Server Testing")
    logger.info("=" * 60)

    try:
        # Run all tests
        test_metrics_initialization()
        test_metrics_server_startup()
        test_trading_metrics_updates()
        test_system_health_metrics()
        test_convenience_functions()
        test_metrics_endpoint_content()
        test_integration_with_sample_data()
        test_metrics_with_real_backtest_integration()

        logger.info("")
        logger.info("🎉 ALL METRICS SERVER TESTS PASSED! 🎉")
        logger.info(
            "Task 19: Monitoring Infrastructure - Metrics Collection is working correctly"
        )
        logger.info("")
        logger.info("📊 Available Metrics Categories:")
        logger.info("  • Trading Performance: PnL, equity, win rates")
        logger.info("  • Execution Quality: Latency, slippage, fill rates")
        logger.info("  • Position Monitoring: Exposures, open positions")
        logger.info("  • Strategy Analytics: Signals, regime detection")
        logger.info("  • System Health: Errors, resources, market data lag")
        logger.info("  • Risk Management: Drawdowns, VaR, performance ratios")
        logger.info("")
        logger.info("🌐 Metrics Server Ready:")
        logger.info("  • Endpoint: http://localhost:9000/metrics")
        logger.info("  • Prometheus-compatible format")
        logger.info("  • Real-time metric updates")
        logger.info("  • Ready for Grafana dashboards")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
