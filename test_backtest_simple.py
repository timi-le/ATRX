"""
Simple Backtest Test - Verify Framework Works
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtester import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    ExecutionConfig,
    ExecutionSimulator
)

async def test_simple_backtest():
    """Test basic backtesting functionality with guaranteed fills."""
    
    print("Testing Simple Backtest...")
    
    # Create a very simple config
    config = BacktestConfig(
        symbols=["EURUSD"],
        timeframe="5m",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),  # Just 1 day
        initial_capital=100000.0,
        mode=BacktestMode.EXECUTION_ONLY,  # Skip complex pipeline
        enable_slippage=False,
        enable_commission=False,
        enable_latency=False
    )
    
    # Create execution simulator with guaranteed fills
    execution_config = ExecutionConfig(
        min_latency_ms=0,
        max_latency_ms=0,
        base_slippage_bps=0.0,
        commission_per_lot=0.0,
        rejection_rate=0.0,  # No rejections
        partial_fill_probability=0.0  # No partial fills
    )
    
    # Create and run backtest
    engine = BacktestEngine(config)
    engine.execution_simulator = ExecutionSimulator(execution_config)
    
    # Initialize
    await engine.initialize()
    
    # Run backtest
    results = await engine.run()
    
    print(f"Results: {results}")
    print(f"Total orders: {results.get('execution', {}).get('total_orders', 0)}")
    print(f"Filled orders: {results.get('execution', {}).get('filled_orders', 0)}")
    print(f"Fill rate: {results.get('execution', {}).get('fill_rate', 0)}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_simple_backtest()) 