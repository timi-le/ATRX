"""
Debug test to trace execution flow from signals to fills
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.interfaces.trading_interfaces import Signal, OrderSide, Order, OrderType
from core.interfaces.data_interfaces import OHLCV
from backtester.market_replay import DataPoint
import uuid

async def test_execution_flow():
    """Test the execution flow to find where it breaks"""
    
    print("=== Testing Signal to Fill Flow ===")
    
    # 3. Test ExecutionSimulator
    from backtester.execution_simulator import ExecutionSimulator, ExecutionConfig
    import structlog
    
    logger = structlog.get_logger(__name__)
    execution_config = ExecutionConfig(
        min_latency_ms=10,
        max_latency_ms=50,
        base_slippage_bps=0.5,
        commission_per_lot=7.0,
        rejection_rate=0.01,
        partial_fill_probability=0.05
    )
    
    execution_simulator = ExecutionSimulator(execution_config, logger)
    print(f"✓ ExecutionSimulator created")
    
    # 4. Provide market data BEFORE submitting orders
    ohlcv = OHLCV(
        symbol="EURUSD",
        timestamp=datetime.now(),
        open=1.1000,
        high=1.1005,
        low=1.0995,
        close=1.1002,
        volume=10000.0
    )
    
    data_point = DataPoint(
        symbol="EURUSD",
        timestamp=datetime.now(),
        data=ohlcv,
        data_type="bar"
    )
    
    await execution_simulator.update_market_data(data_point)
    print(f"✓ Market data provided for {data_point.symbol}")
    
    # 5. Test PerformanceAnalyzer
    from backtester.performance_metrics import PerformanceAnalyzer, PerformanceConfig
    
    performance_config = PerformanceConfig(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        trading_days_per_year=252
    )
    
    performance_analyzer = PerformanceAnalyzer(performance_config, logger)
    print(f"✓ PerformanceAnalyzer created")
    
    # 6. Test complete round-trip trade (BUY then SELL)
    print("\n--- Testing Round-Trip Trade ---")
    
    # ORDER 1: BUY (Open Long Position)
    buy_order = Order(
        order_id=f"buy_order_{uuid.uuid4().hex[:8]}",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100000.0,  # 1 lot
        timestamp=datetime.now()
    )
    
    print(f"1. Submitting BUY order: {buy_order.order_id}")
    await execution_simulator.submit_order(buy_order)
    await asyncio.sleep(0.1)  # Wait for execution
    
    # Check for fills and record them
    fills = execution_simulator.get_fills()
    print(f"   BUY fills generated: {len(fills)}")
    
    positions = {}
    for fill in fills:
        positions = execution_simulator.get_positions()  # Get updated positions
        performance_analyzer.record_fill(fill, positions)
        print(f"   Recorded BUY fill: {fill.symbol} {fill.side.value} {fill.quantity}")
    
    # Clear processed fills
    execution_simulator.mark_fills_processed(len(fills))
    
    # ORDER 2: SELL (Close Long Position)
    # Wait a bit and update market data to simulate time passing
    await asyncio.sleep(0.1)
    
    # Update market data with a slightly different price
    ohlcv2 = OHLCV(
        symbol="EURUSD",
        timestamp=datetime.now() + timedelta(minutes=1),
        open=1.1002,
        high=1.1008,
        low=1.0998,
        close=1.1005,  # Slightly higher price for profit
        volume=10000.0
    )
    
    data_point2 = DataPoint(
        symbol="EURUSD",
        timestamp=datetime.now() + timedelta(minutes=1),
        data=ohlcv2,
        data_type="bar"
    )
    
    await execution_simulator.update_market_data(data_point2)
    
    sell_order = Order(
        order_id=f"sell_order_{uuid.uuid4().hex[:8]}",
        symbol="EURUSD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100000.0,  # Close the full position
        timestamp=datetime.now() + timedelta(minutes=1)
    )
    
    print(f"2. Submitting SELL order: {sell_order.order_id}")
    await execution_simulator.submit_order(sell_order)
    await asyncio.sleep(0.1)  # Wait for execution
    
    # Check for new fills and record them
    new_fills = execution_simulator.get_fills()
    print(f"   SELL fills generated: {len(new_fills)}")
    
    for fill in new_fills:
        positions = execution_simulator.get_positions()  # Get updated positions
        performance_analyzer.record_fill(fill, positions)
        print(f"   Recorded SELL fill: {fill.symbol} {fill.side.value} {fill.quantity}")
    
    # 7. Check final metrics
    print("\n--- Final Results ---")
    metrics = performance_analyzer.calculate_metrics()
    print(f"✓ Total completed trades: {metrics.total_trades}")
    print(f"✓ Current equity: ${performance_analyzer.current_equity:.2f}")
    print(f"✓ Total commission: ${metrics.total_commission:.2f}")
    print(f"✓ Win rate: {metrics.win_rate:.2%}")
    
    if metrics.total_trades > 0:
        print("✅ SUCCESS: Trades were properly recorded!")
    else:
        print("❌ ISSUE: No trades recorded despite fills")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_execution_flow()) 