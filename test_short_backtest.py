#!/usr/bin/env python3
"""
Short backtest test to verify our fix for performance tracking
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtester import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
)

from core.interfaces.trading_interfaces import Signal, OrderSide
from core.interfaces.data_interfaces import OHLCV
from backtester.market_replay import DataPoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('short_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def run_short_test():
    """Run a short backtest test to verify our fix."""
    logger.info("Starting Short Backtest Test")
    logger.info("=" * 50)
    
    # Test configuration - very short
    config = BacktestConfig(
        symbols=["EURUSD"],
        timeframe="5m",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),  # Just 1 day
        initial_capital=100000.0,
        mode=BacktestMode.FULL_PIPELINE,
        
        # Strategy configuration
        strategies=["test_strategy"],
        strategy_weights={"test_strategy": 1.0},
        
        # Risk management
        max_position_size=0.10,
        max_total_exposure=0.40,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        
        # Execution settings
        enable_slippage=True,
        enable_commission=True,
        enable_latency=True,
        
        # Results
        save_results=False,
        save_trades=False,
        save_equity_curve=False,
    )
    
    # Create mock components
    feature_engine = MockFeatureEngine()
    regime_detector = MockRegimeDetector()
    strategy_switcher = MockStrategySwitcher(["test_strategy"])
    position_sizer = MockPositionSizer()
    risk_manager = MockRiskManager()
    
    # Create mock market replay with limited data
    mock_market_replay = MockMarketReplay(
        symbols=["EURUSD"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        timeframe="5m"
    )
    
    # Create and run backtest engine
    engine = BacktestEngine(
        config=config,
        feature_engine=feature_engine,
        regime_detector=regime_detector,
        strategy_switcher=strategy_switcher,
        position_sizer=position_sizer,
        risk_manager=risk_manager
    )
    
    # Replace the market replay with our mock
    engine.market_replay = mock_market_replay
    
    # Initialize and run
    await engine.initialize()
    result = await engine.run()
    
    # Analyze results
    logger.info("\n" + "=" * 50)
    logger.info("SHORT BACKTEST RESULTS:")
    logger.info("=" * 50)
    
    perf = result['performance']
    exec_stats = result['execution']
    
    logger.info(f"Initial Capital: ${config.initial_capital:.2f}")
    logger.info(f"Final Equity: ${result['final_equity']:.2f}")
    logger.info(f"Total Return: {perf['total_return']:.2%}")
    logger.info(f"Total Trades: {perf['total_trades']}")
    logger.info(f"Win Rate: {perf['win_rate']:.2%}")
    logger.info(f"Profit Factor: {perf['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    
    logger.info(f"\nExecution Stats:")
    logger.info(f"Total Orders: {exec_stats['total_orders']}")
    logger.info(f"Total Fills: {exec_stats['total_fills']}")
    logger.info(f"Fill Rate: {exec_stats['fill_rate']:.2%}")
    logger.info(f"Avg Slippage: {exec_stats['avg_slippage_per_fill']:.4f}")
    
    # Verify our fix worked
    logger.info(f"\n" + "=" * 50)
    if perf['total_trades'] > 0:
        logger.info("✅ SUCCESS: Completed trades were recorded!")
        logger.info(f"✅ Generated {perf['total_trades']} completed trades from {exec_stats['total_fills']} fills")
        logger.info("✅ Performance tracking is working correctly!")
    else:
        logger.info("❌ ISSUE: No completed trades recorded despite fills")
        logger.info("❌ Performance tracking still has issues")
    
    logger.info("=" * 50)
    return result


class MockFeatureEngine:
    """Mock feature engine for testing."""
    async def compute_features(self, data, symbol):
        return {
            'sma_20': np.random.uniform(0.8, 1.2),
            'rsi_14': np.random.uniform(30, 70),
            'macd_signal': np.random.uniform(-0.1, 0.1),
            'bollinger_position': np.random.uniform(0, 1),
            'volatility': np.random.uniform(0.01, 0.05),
            'momentum': np.random.uniform(-0.02, 0.02)
        }


class MockRegimeDetector:
    """Mock regime detector for testing."""
    async def detect_regime(self, data, symbol):
        regimes = ["trending", "ranging", "volatile"]
        return np.random.choice(regimes)


class MockStrategySwitcher:
    """Mock strategy that generates both opening and closing signals"""
    def __init__(self, strategies):
        self.strategies = strategies
        self.signal_counter = 0
        self.signals_generated = 0
        self.current_position = 0  # Track net position size
        self.position_history = []  # Track position changes
        self.trades_completed = 0
        
    async def generate_signal(self, market_data, features, regime):
        # Generate signals more frequently for testing
        self.signal_counter += 1
        if self.signal_counter % 5 == 0:  # Signal every 5 data points
            
            # Strategy: Create complete round-trip trades
            # If no position, open a position (buy or sell)
            # If have position, close it after some time
            
            if self.current_position == 0:
                # Open new position (random direction)
                side = random.choice([OrderSide.BUY, OrderSide.SELL])
                self.current_position = 45454 if side == OrderSide.BUY else -45454
                logger.info(f"Opening position: {side.value}, net position now: {self.current_position}")
            else:
                # Close current position (25% chance to close, 75% to continue)
                if random.random() < 0.25:  # 25% chance to close position
                    # Close position with opposite signal
                    if self.current_position > 0:
                        side = OrderSide.SELL
                        self.current_position = 0  # Position will be closed
                        self.trades_completed += 1
                        logger.info(f"Closing LONG position with SELL signal, trade #{self.trades_completed} completed")
                    else:
                        side = OrderSide.BUY
                        self.current_position = 0  # Position will be closed  
                        self.trades_completed += 1
                        logger.info(f"Closing SHORT position with BUY signal, trade #{self.trades_completed} completed")
                else:
                    # Continue in same direction or don't generate signal
                    return None
            
            # Create signal
            signal = Signal(
                symbol=market_data.symbol,
                side=side,
                strength=random.uniform(0.6, 0.9),
                confidence=random.uniform(0.7, 0.95),
                strategy_name=random.choice(self.strategies),
                timestamp=datetime.now()
            )
            
            self.signals_generated += 1
            logger.info(f"Signal #{self.signals_generated}: {signal.symbol} {signal.side.value} (trades completed: {self.trades_completed})")
            return signal
        
        return None


class MockPositionSizer:
    """Mock position sizer for testing."""
    async def calculate_position_size(self, signal, current_equity, current_positions):
        # Much smaller position sizes for testing that will pass risk checks
        # Calculate size based on a much smaller fraction of equity (e.g., 10% of capital)
        max_exposure = current_equity * 0.1  # 10% of capital per trade
        estimated_price = 1.1000  # EURUSD approximate price
        position_size = int(max_exposure / estimated_price)  # Convert to units
        
        # Ensure it's reasonable size (at least 1,000 units for valid FX trade)
        position_size = max(position_size, 1000)
        
        logger.info(f"Position sizing: equity=${current_equity:.2f}, max_exposure=${max_exposure:.2f}, price={estimated_price}, units={position_size}")
        return position_size


class MockRiskManager:
    """Mock risk manager for testing."""
    async def check_pre_trade_risk(self, order, current_positions, current_equity):
        # Very permissive risk check for testing
        # Allow up to 95% total exposure (was 80%)
        total_exposure = sum(abs(pos.quantity * pos.avg_price) for pos in current_positions.values())
        
        # Use a more realistic price estimate
        estimated_price = 1.1000  # EURUSD approximate price
        new_exposure = abs(order.quantity) * estimated_price
        total_after_trade = total_exposure + new_exposure
        max_allowed = current_equity * 0.95  # 95% max exposure (very permissive)
        
        # Debug logging
        logger.info(f"Risk check: current_exposure=${total_exposure:.2f}, new_trade=${new_exposure:.2f}, total=${total_after_trade:.2f}, max_allowed=${max_allowed:.2f}")
        
        passed = total_after_trade <= max_allowed
        logger.info(f"Risk check result: {'PASSED' if passed else 'FAILED'}")
        return passed


class MockMarketReplay:
    """Mock market replay that generates limited synthetic data."""
    def __init__(self, symbols, start_date, end_date, timeframe):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.data_points = []
        
    async def load_data(self):
        """Generate limited synthetic market data for short test."""
        logger.info(f"MockMarketReplay: Generating data for {self.symbols} from {self.start_date} to {self.end_date}")
        
        # Generate data points every 5 minutes but only for a few hours
        current_time = self.start_date
        end_time = self.start_date + timedelta(hours=6)  # Just 6 hours of data
        prices = {symbol: 1.1000 + np.random.uniform(-0.01, 0.01) for symbol in self.symbols}
        
        point_count = 0
        while current_time <= end_time:
            for symbol in self.symbols:
                # Generate realistic price movement
                price_change = np.random.uniform(-0.0005, 0.0005)
                prices[symbol] += price_change
                prices[symbol] = max(0.5, min(2.0, prices[symbol]))
                
                # Create OHLCV data
                base_price = prices[symbol]
                high = base_price + np.random.uniform(0, 0.0002)
                low = base_price - np.random.uniform(0, 0.0002)
                open_price = base_price + np.random.uniform(-0.0001, 0.0001)
                close_price = base_price + np.random.uniform(-0.0001, 0.0001)
                volume = np.random.uniform(1000, 10000)
                
                ohlcv = OHLCV(
                    symbol=symbol,
                    timestamp=current_time,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume
                )
                
                data_point = DataPoint(
                    symbol=symbol,
                    timestamp=current_time,
                    data=ohlcv,
                    data_type="bar"
                )
                
                self.data_points.append(data_point)
                point_count += 1
            
            # Move to next time interval (5 minutes)
            current_time += timedelta(minutes=5)
        
        logger.info(f"MockMarketReplay: Generated {point_count} data points")
        
    async def stream(self):
        """Stream the generated data points."""
        logger.info(f"MockMarketReplay: Starting to stream {len(self.data_points)} data points")
        for i, data_point in enumerate(self.data_points):
            if i % 20 == 0:
                logger.info(f"MockMarketReplay: Streaming data point {i+1}/{len(self.data_points)}")
            yield data_point
            await asyncio.sleep(0.001)
        logger.info("MockMarketReplay: Finished streaming all data points")


if __name__ == "__main__":
    asyncio.run(run_short_test()) 