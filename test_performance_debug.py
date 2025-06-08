"""
Test performance tracking to debug why 22,722 fills don't create any trades
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import structlog

from backtester.execution_simulator import Fill, FillType
from backtester.performance_metrics import PerformanceAnalyzer, PerformanceConfig
from core.interfaces.trading_interfaces import OrderSide, Position

# Set up logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_performance_tracking():
    """Test why fills aren't creating trades"""

    print("=== Testing Performance Tracking ===")

    # Create performance analyzer with debug logging
    logger = structlog.get_logger(__name__)
    performance_config = PerformanceConfig(
        initial_capital=100000.0, risk_free_rate=0.02, trading_days_per_year=252
    )

    performance_analyzer = PerformanceAnalyzer(performance_config, logger)
    print(
        f"✓ PerformanceAnalyzer created, initial equity: ${performance_analyzer.current_equity:.2f}"
    )

    # Test sequence: BUY to open long, then SELL to close long
    print("\n--- Test 1: Long Position Round Trip ---")

    # Step 1: BUY fill to open long position
    buy_fill = Fill(
        order_id="test_buy",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=100000,
        price=1.1000,
        timestamp=datetime.now(),
        commission=7.0,
        slippage=0.0001,
        fill_type=FillType.FULL,
    )

    # Simulate position after BUY
    long_position = Position(
        symbol="EURUSD",
        quantity=100000,  # Long 1 lot
        avg_price=1.1000,
        unrealized_pnl=0.0,
    )
    positions_after_buy = {"EURUSD": long_position}

    print(f"1. Recording BUY fill: {buy_fill.quantity} @ {buy_fill.price}")
    performance_analyzer.record_fill(buy_fill, positions_after_buy)
    print(f"   Trades after BUY: {len(performance_analyzer.trades)}")
    print(f"   Current equity: ${performance_analyzer.current_equity:.2f}")

    # Step 2: SELL fill to close long position
    sell_fill = Fill(
        order_id="test_sell",
        symbol="EURUSD",
        side=OrderSide.SELL,
        quantity=100000,
        price=1.1020,  # 20 pips profit
        timestamp=datetime.now() + timedelta(hours=1),
        commission=7.0,
        slippage=0.0001,
        fill_type=FillType.FULL,
    )

    # Simulate position after SELL (closed)
    positions_after_sell = {}  # No position remaining

    print(f"2. Recording SELL fill: {sell_fill.quantity} @ {sell_fill.price}")
    performance_analyzer.record_fill(sell_fill, positions_after_sell)
    print(f"   Trades after SELL: {len(performance_analyzer.trades)}")
    print(f"   Current equity: ${performance_analyzer.current_equity:.2f}")

    if len(performance_analyzer.trades) > 0:
        trade = performance_analyzer.trades[0]
        print(
            f"   ✓ Trade created: {trade.symbol} {trade.side.value} P&L=${trade.pnl:.2f}"
        )
    else:
        print("   ❌ No trade created!")

    # Test sequence: SELL to open short, then BUY to close short
    print("\n--- Test 2: Short Position Round Trip ---")

    # Reset for clean test
    performance_analyzer.reset()

    # Step 1: SELL fill to open short position
    sell_short_fill = Fill(
        order_id="test_sell_short",
        symbol="GBPUSD",
        side=OrderSide.SELL,
        quantity=100000,
        price=1.3000,
        timestamp=datetime.now(),
        commission=7.0,
        slippage=0.0001,
        fill_type=FillType.FULL,
    )

    # Simulate position after SELL (short)
    short_position = Position(
        symbol="GBPUSD",
        quantity=-100000,  # Short 1 lot
        avg_price=1.3000,
        unrealized_pnl=0.0,
    )
    positions_after_sell_short = {"GBPUSD": short_position}

    print(
        f"1. Recording SELL SHORT fill: {sell_short_fill.quantity} @ {sell_short_fill.price}"
    )
    performance_analyzer.record_fill(sell_short_fill, positions_after_sell_short)
    print(f"   Trades after SELL SHORT: {len(performance_analyzer.trades)}")
    print(f"   Current equity: ${performance_analyzer.current_equity:.2f}")

    # Step 2: BUY fill to close short position
    buy_cover_fill = Fill(
        order_id="test_buy_cover",
        symbol="GBPUSD",
        side=OrderSide.BUY,
        quantity=100000,
        price=1.2980,  # 20 pips profit (price went down)
        timestamp=datetime.now() + timedelta(hours=1),
        commission=7.0,
        slippage=0.0001,
        fill_type=FillType.FULL,
    )

    # Simulate position after BUY (closed)
    positions_after_buy_cover = {}  # No position remaining

    print(
        f"2. Recording BUY COVER fill: {buy_cover_fill.quantity} @ {buy_cover_fill.price}"
    )
    performance_analyzer.record_fill(buy_cover_fill, positions_after_buy_cover)
    print(f"   Trades after BUY COVER: {len(performance_analyzer.trades)}")
    print(f"   Current equity: ${performance_analyzer.current_equity:.2f}")

    if len(performance_analyzer.trades) > 0:
        trade = performance_analyzer.trades[0]
        print(
            f"   ✓ Trade created: {trade.symbol} {trade.side.value} P&L=${trade.pnl:.2f}"
        )
    else:
        print("   ❌ No trade created!")

    # Test what happens with random fills like in our backtest
    print("\n--- Test 3: Random Fills (Like Our Backtest) ---")

    # Reset for clean test
    performance_analyzer.reset()

    # Simulate what happens in our backtest - random BUY/SELL without proper closures
    fills_and_positions = [
        # Random BUY
        (
            Fill(
                "order1",
                "EURUSD",
                OrderSide.BUY,
                100000,
                1.1000,
                datetime.now(),
                7.0,
                0.0001,
                FillType.FULL,
            ),
            {"EURUSD": Position("EURUSD", 100000, 1.1000, 0.0)},
        ),
        # Another random BUY (add to position)
        (
            Fill(
                "order2",
                "EURUSD",
                OrderSide.BUY,
                100000,
                1.1010,
                datetime.now(),
                7.0,
                0.0001,
                FillType.FULL,
            ),
            {"EURUSD": Position("EURUSD", 200000, 1.1005, 0.0)},
        ),
        # Random SELL on different symbol
        (
            Fill(
                "order3",
                "GBPUSD",
                OrderSide.SELL,
                100000,
                1.3000,
                datetime.now(),
                7.0,
                0.0001,
                FillType.FULL,
            ),
            {
                "EURUSD": Position("EURUSD", 200000, 1.1005, 0.0),
                "GBPUSD": Position("GBPUSD", -100000, 1.3000, 0.0),
            },
        ),
        # Random BUY on different symbol
        (
            Fill(
                "order4",
                "USDJPY",
                OrderSide.BUY,
                100000,
                145.00,
                datetime.now(),
                7.0,
                0.0001,
                FillType.FULL,
            ),
            {
                "EURUSD": Position("EURUSD", 200000, 1.1005, 0.0),
                "GBPUSD": Position("GBPUSD", -100000, 1.3000, 0.0),
                "USDJPY": Position("USDJPY", 100000, 145.00, 0.0),
            },
        ),
    ]

    for i, (fill, positions) in enumerate(fills_and_positions, 1):
        print(
            f"{i}. Recording {fill.side.value} fill: {fill.symbol} {fill.quantity} @ {fill.price}"
        )
        performance_analyzer.record_fill(fill, positions)
        print(f"   Trades after fill {i}: {len(performance_analyzer.trades)}")

    print(f"\nFinal Result:")
    print(f"   Total fills processed: {len(fills_and_positions)}")
    print(f"   Total trades created: {len(performance_analyzer.trades)}")
    print(f"   Current equity: ${performance_analyzer.current_equity:.2f}")

    # Final metrics
    metrics = performance_analyzer.calculate_metrics()
    print(f"   Final metrics - Total trades: {metrics.total_trades}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_performance_tracking())
