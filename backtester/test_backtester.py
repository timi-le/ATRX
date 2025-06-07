"""
Comprehensive Test Suite for Backtesting Framework.

Tests all components of the backtesting system:
- Market replay functionality
- Execution simulation
- Performance metrics calculation
- End-to-end backtesting
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional

from core.interfaces.trading_interfaces import (
    Order, OrderType, OrderSide, OrderStatus, Position, Signal
)
from core.interfaces.data_interfaces import MarketData, OHLCV

from backtester.market_replay import (
    MarketReplay, ReplayConfig, DataPoint, TimeFrame, create_replay_config
)
from backtester.execution_simulator import (
    ExecutionSimulator, ExecutionConfig, Fill, FillType, create_execution_config
)
from backtester.performance_metrics import (
    PerformanceAnalyzer, PerformanceConfig, TradeMetrics, PeriodMetrics,
    create_performance_config, compare_strategies
)
from backtester.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestMode, BacktestState,
    create_backtest_config, run_simple_backtest
)


class TestMarketReplay:
    """Test suite for MarketReplay functionality."""
    
    @pytest.fixture
    def replay_config(self):
        """Create test replay configuration."""
        return ReplayConfig(
            data_path="test_data",
            symbols=["EURUSD", "GBPUSD"],
            timeframe=TimeFrame.M1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            replay_speed=0.0,
            validate_data=True,
            fill_gaps=False
        )
    
    @pytest.fixture
    def market_replay(self, replay_config):
        """Create MarketReplay instance."""
        return MarketReplay(replay_config)
    
    def test_replay_config_creation(self):
        """Test replay configuration creation."""
        config = create_replay_config(
            symbols=["EURUSD"],
            timeframe="1m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2)
        )
        
        assert config.symbols == ["EURUSD"]
        assert config.timeframe == TimeFrame.M1
        assert config.start_date == datetime(2024, 1, 1)
        assert config.end_date == datetime(2024, 1, 2)
        assert config.replay_speed == 0.0
    
    def test_market_replay_initialization(self, market_replay, replay_config):
        """Test MarketReplay initialization."""
        assert market_replay.config == replay_config
        assert market_replay.data_queue == []
        assert market_replay.total_points == 0
        assert market_replay.points_streamed == 0
        assert not market_replay.is_running
    
    @pytest.mark.asyncio
    async def test_generate_mock_data(self, market_replay):
        """Test mock data generation."""
        symbol = "EURUSD"
        data_points = await market_replay._generate_mock_data(symbol)
        
        assert len(data_points) > 0
        assert all(point.symbol == symbol for point in data_points)
        assert all(point.data_type == "bar" for point in data_points)
        
        # Check data ordering
        timestamps = [point.timestamp for point in data_points]
        assert timestamps == sorted(timestamps)
    
    def test_timestamp_parsing(self, market_replay):
        """Test timestamp parsing with various formats."""
        test_cases = [
            "2024-01-01 12:00:00",
            "2024-01-01T12:00:00",
            "2024-01-01T12:00:00Z",
            "01.01.2024 12:00:00",
            "01/01/2024 12:00:00"
        ]
        
        for timestamp_str in test_cases:
            result = market_replay._parse_timestamp(timestamp_str)
            assert result is not None
            assert isinstance(result, datetime)
    
    @pytest.mark.asyncio
    async def test_data_validation(self, market_replay):
        """Test data validation functionality."""
        # Create test data points with some invalid data
        valid_point = DataPoint(
            timestamp=datetime(2024, 1, 1, 12, 0),
            symbol="EURUSD",
            data=OHLCV(
                symbol="EURUSD",
                timestamp=datetime(2024, 1, 1, 12, 0),
                open=1.1000,
                high=1.1010,
                low=1.0990,
                close=1.1005,
                volume=1000,
                timeframe="1m"
            ),
            data_type="bar"
        )
        
        invalid_point = DataPoint(
            timestamp=datetime(2024, 1, 1, 12, 1),
            symbol="EURUSD",
            data=OHLCV(
                symbol="EURUSD",
                timestamp=datetime(2024, 1, 1, 12, 1),
                open=1.1000,
                high=1.0990,  # Invalid: high < open
                low=1.1010,   # Invalid: low > open
                close=1.1005,
                volume=1000,
                timeframe="1m"
            ),
            data_type="bar"
        )
        
        test_data = [valid_point, invalid_point]
        validated_data = await market_replay._validate_data(test_data, "EURUSD")
        
        assert len(validated_data) == 1
        assert validated_data[0] == valid_point
        assert market_replay.validation_errors == 1
    
    @pytest.mark.asyncio
    async def test_data_streaming(self, market_replay):
        """Test data streaming functionality."""
        # Mock the load_data method to provide test data
        test_data = []
        for i in range(5):
            timestamp = datetime(2024, 1, 1, 12, i)
            data_point = DataPoint(
                timestamp=timestamp,
                symbol="EURUSD",
                data=OHLCV(
                    symbol="EURUSD",
                    timestamp=timestamp,
                    open=1.1000 + i * 0.0001,
                    high=1.1010 + i * 0.0001,
                    low=1.0990 + i * 0.0001,
                    close=1.1005 + i * 0.0001,
                    volume=1000,
                    timeframe="1m"
                ),
                data_type="bar"
            )
            test_data.append(data_point)
        
        market_replay.data_queue = test_data
        market_replay.total_points = len(test_data)
        
        streamed_points = []
        async for point in market_replay.stream():
            streamed_points.append(point)
        
        assert len(streamed_points) == len(test_data)
        assert streamed_points == test_data
        assert market_replay.points_streamed == len(test_data)


class TestExecutionSimulator:
    """Test suite for ExecutionSimulator functionality."""
    
    @pytest.fixture
    def execution_config(self):
        """Create test execution configuration."""
        return ExecutionConfig(
            min_latency_ms=0,
            max_latency_ms=0,
            base_slippage_bps=0.5,
            commission_per_lot=7.0,
            rejection_rate=0.0,
            partial_fill_probability=0.0
        )
    
    @pytest.fixture
    def execution_simulator(self, execution_config):
        """Create ExecutionSimulator instance."""
        return ExecutionSimulator(execution_config)
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            order_id="test_order_1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            bid=1.1000,
            ask=1.1002,
            volume=1000,
            source="test"
        )
    
    def test_execution_config_creation(self):
        """Test execution configuration creation."""
        config = create_execution_config(
            latency_ms=(10, 100),
            slippage_bps=1.0,
            commission_per_lot=5.0,
            rejection_rate=0.02
        )
        
        assert config.min_latency_ms == 10
        assert config.max_latency_ms == 100
        assert config.base_slippage_bps == 1.0
        assert config.commission_per_lot == 5.0
        assert config.rejection_rate == 0.02
    
    def test_execution_simulator_initialization(self, execution_simulator, execution_config):
        """Test ExecutionSimulator initialization."""
        assert execution_simulator.config == execution_config
        assert execution_simulator.pending_orders == {}
        assert execution_simulator.positions == {}
        assert execution_simulator.fills == []
        assert execution_simulator.total_orders == 0
    
    @pytest.mark.asyncio
    async def test_order_validation(self, execution_simulator, sample_order):
        """Test order validation."""
        # Valid order
        assert await execution_simulator._validate_order(sample_order)
        
        # Invalid order (too large)
        large_order = Order(
            order_id="large_order",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=execution_simulator.config.max_order_size + 1,
            timestamp=datetime.now()
        )
        assert not await execution_simulator._validate_order(large_order)
    
    @pytest.mark.asyncio
    async def test_slippage_calculation(self, execution_simulator, sample_order, sample_market_data):
        """Test slippage calculation."""
        execution_price = 1.1001
        slippage = await execution_simulator._calculate_slippage(
            sample_order, execution_price, sample_market_data
        )
        
        assert slippage >= 0
        assert isinstance(slippage, float)
    
    @pytest.mark.asyncio
    async def test_commission_calculation(self, execution_simulator, sample_order):
        """Test commission calculation."""
        quantity = 100000
        price = 1.1001
        commission = await execution_simulator._calculate_commission(sample_order, quantity, price)
        
        assert commission >= execution_simulator.config.minimum_commission
        assert isinstance(commission, float)
    
    @pytest.mark.asyncio
    async def test_order_execution(self, execution_simulator, sample_order, sample_market_data):
        """Test order execution process."""
        # Add market data
        data_point = DataPoint(
            timestamp=datetime.now(),
            symbol="EURUSD",
            data=sample_market_data,
            data_type="tick"
        )
        await execution_simulator.update_market_data(data_point)
        
        # Submit order
        order_id = await execution_simulator.submit_order(sample_order)
        assert order_id == sample_order.order_id
        assert sample_order.order_id in execution_simulator.pending_orders
        
        # Wait for execution (simulate delay)
        await asyncio.sleep(0.1)
        
        # Check if order was executed
        fills = execution_simulator.get_fills()
        assert len(fills) > 0
        
        fill = fills[0]
        assert fill.order_id == sample_order.order_id
        assert fill.symbol == sample_order.symbol
        assert fill.side == sample_order.side
        assert fill.quantity > 0
    
    @pytest.mark.asyncio
    async def test_position_tracking(self, execution_simulator):
        """Test position tracking after fills."""
        # Create and execute buy order
        buy_order = Order(
            order_id="buy_order",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000,
            timestamp=datetime.now()
        )
        
        buy_fill = Fill(
            order_id="buy_order",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=100000,
            price=1.1001,
            timestamp=datetime.now(),
            commission=7.0,
            slippage=0.0001
        )
        
        await execution_simulator._update_position(buy_fill)
        
        positions = execution_simulator.get_positions()
        assert "EURUSD" in positions
        
        position = positions["EURUSD"]
        assert position.quantity == 100000
        assert position.avg_price == 1.1001
        
        # Create and execute sell order
        sell_fill = Fill(
            order_id="sell_order",
            symbol="EURUSD",
            side=OrderSide.SELL,
            quantity=50000,
            price=1.1005,
            timestamp=datetime.now(),
            commission=3.5,
            slippage=0.0001
        )
        
        await execution_simulator._update_position(sell_fill)
        
        positions = execution_simulator.get_positions()
        position = positions["EURUSD"]
        assert position.quantity == 50000  # Reduced position
    
    def test_statistics_calculation(self, execution_simulator):
        """Test execution statistics calculation."""
        # Simulate some orders and fills
        execution_simulator.total_orders = 10
        execution_simulator.filled_orders = 8
        execution_simulator.rejected_orders = 2
        execution_simulator.partial_fills = 1
        execution_simulator.total_commission = 56.0
        execution_simulator.total_slippage = 0.008
        
        # Add some fills
        for i in range(8):
            fill = Fill(
                order_id=f"order_{i}",
                symbol="EURUSD",
                side=OrderSide.BUY,
                quantity=100000,
                price=1.1001,
                timestamp=datetime.now(),
                commission=7.0,
                slippage=0.001
            )
            execution_simulator.fills.append(fill)
        
        stats = execution_simulator.get_statistics()
        
        assert stats["total_orders"] == 10
        assert stats["filled_orders"] == 8
        assert stats["rejected_orders"] == 2
        assert stats["fill_rate"] == 0.8
        assert stats["total_fills"] == 8
        assert stats["avg_commission_per_fill"] == 7.0
        assert stats["avg_slippage_per_fill"] == 0.001


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics functionality."""
    
    @pytest.fixture
    def performance_config(self):
        """Create test performance configuration."""
        return PerformanceConfig(
            initial_capital=100000.0,
            risk_free_rate=0.02,
            trading_days_per_year=252
        )
    
    @pytest.fixture
    def performance_analyzer(self, performance_config):
        """Create PerformanceAnalyzer instance."""
        return PerformanceAnalyzer(performance_config)
    
    @pytest.fixture
    def sample_fills(self):
        """Create sample fills for testing."""
        fills = []
        
        # Winning trade
        fills.append(Fill(
            order_id="trade1_buy",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=100000,
            price=1.1000,
            timestamp=datetime(2024, 1, 1, 12, 0),
            commission=7.0,
            slippage=0.0001
        ))
        
        fills.append(Fill(
            order_id="trade1_sell",
            symbol="EURUSD",
            side=OrderSide.SELL,
            quantity=100000,
            price=1.1050,
            timestamp=datetime(2024, 1, 1, 13, 0),
            commission=7.0,
            slippage=0.0001
        ))
        
        # Losing trade
        fills.append(Fill(
            order_id="trade2_buy",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=100000,
            price=1.1100,
            timestamp=datetime(2024, 1, 1, 14, 0),
            commission=7.0,
            slippage=0.0001
        ))
        
        fills.append(Fill(
            order_id="trade2_sell",
            symbol="EURUSD",
            side=OrderSide.SELL,
            quantity=100000,
            price=1.1080,
            timestamp=datetime(2024, 1, 1, 15, 0),
            commission=7.0,
            slippage=0.0001
        ))
        
        return fills
    
    def test_performance_config_creation(self):
        """Test performance configuration creation."""
        config = create_performance_config(
            initial_capital=50000.0,
            risk_free_rate=0.03,
            trading_days_per_year=250
        )
        
        assert config.initial_capital == 50000.0
        assert config.risk_free_rate == 0.03
        assert config.trading_days_per_year == 250
    
    def test_performance_analyzer_initialization(self, performance_analyzer, performance_config):
        """Test PerformanceAnalyzer initialization."""
        assert performance_analyzer.config == performance_config
        assert performance_analyzer.current_equity == performance_config.initial_capital
        assert performance_analyzer.peak_equity == performance_config.initial_capital
        assert performance_analyzer.max_drawdown == 0.0
        assert len(performance_analyzer.trades) == 0
    
    def test_equity_curve_tracking(self, performance_analyzer):
        """Test equity curve tracking and drawdown calculation."""
        timestamps = [
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0),
            datetime(2024, 1, 1, 14, 0),
            datetime(2024, 1, 1, 15, 0),
            datetime(2024, 1, 1, 16, 0)
        ]
        
        equity_values = [100000, 105000, 103000, 108000, 106000]
        
        for timestamp, equity in zip(timestamps, equity_values):
            performance_analyzer.update_equity(timestamp, equity)
        
        assert len(performance_analyzer.equity_curve) == 5
        assert performance_analyzer.current_equity == 106000
        assert performance_analyzer.peak_equity == 108000
        
        # Check drawdown calculation
        expected_drawdown = (108000 - 106000) / 108000
        assert abs(performance_analyzer.current_drawdown - expected_drawdown) < 1e-6
    
    def test_trade_tracking(self, performance_analyzer, sample_fills):
        """Test trade tracking from fills."""
        positions = {}
        
        for fill in sample_fills:
            # Simulate position updates
            if fill.symbol not in positions:
                positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=0.0,
                    avg_price=0.0
                )
            
            position = positions[fill.symbol]
            
            if fill.side == OrderSide.BUY:
                new_quantity = position.quantity + fill.quantity
                if position.quantity == 0:
                    position.avg_price = fill.price
                else:
                    total_cost = position.quantity * position.avg_price + fill.quantity * fill.price
                    position.avg_price = total_cost / new_quantity
                position.quantity = new_quantity
            else:
                position.quantity -= fill.quantity
                if abs(position.quantity) < 1e-6:
                    position.quantity = 0.0
            
            performance_analyzer.record_fill(fill, positions)
        
        # Should have recorded 2 completed trades
        assert len(performance_analyzer.trades) == 2
        
        # Check trade details
        trade1 = performance_analyzer.trades[0]
        assert trade1.symbol == "EURUSD"
        assert trade1.side == OrderSide.BUY
        assert trade1.is_winner  # Profit of 500 pips minus commission
        
        trade2 = performance_analyzer.trades[1]
        assert trade2.symbol == "EURUSD"
        assert trade2.side == OrderSide.BUY
        assert not trade2.is_winner  # Loss of 200 pips plus commission
    
    def test_metrics_calculation(self, performance_analyzer):
        """Test performance metrics calculation."""
        # Add some equity data
        timestamps = []
        equity_values = []
        
        base_time = datetime(2024, 1, 1)
        for i in range(30):  # 30 days of data
            timestamp = base_time + timedelta(days=i)
            # Simulate some volatility with overall upward trend
            equity = 100000 * (1 + 0.001 * i + 0.01 * np.sin(i * 0.5))
            
            timestamps.append(timestamp)
            equity_values.append(equity)
            performance_analyzer.update_equity(timestamp, equity)
        
        # Calculate metrics
        metrics = performance_analyzer.calculate_metrics()
        
        assert isinstance(metrics, PeriodMetrics)
        assert metrics.total_return > 0  # Should be positive due to upward trend
        assert metrics.volatility >= 0
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
    
    def test_regime_performance_tracking(self, performance_analyzer):
        """Test regime-based performance tracking."""
        # Set different regimes and add returns
        regimes = ["trending", "ranging", "volatile"]
        
        for i, regime in enumerate(regimes):
            performance_analyzer.set_regime(regime)
            
            # Add some returns for this regime
            for j in range(5):
                timestamp = datetime(2024, 1, 1) + timedelta(days=i*5 + j)
                equity = 100000 * (1 + (i+1) * 0.01 * (j+1))
                performance_analyzer.update_equity(timestamp, equity)
        
        regime_performance = performance_analyzer.calculate_regime_performance()
        
        assert len(regime_performance) == 3
        for regime in regimes:
            assert regime in regime_performance
            assert "total_return" in regime_performance[regime]
            assert "volatility" in regime_performance[regime]
            assert "sharpe_ratio" in regime_performance[regime]
    
    def test_rolling_metrics(self, performance_analyzer):
        """Test rolling metrics calculation."""
        # Add daily returns data
        base_time = datetime(2024, 1, 1)
        for i in range(60):  # 60 days of data
            timestamp = base_time + timedelta(days=i)
            daily_return = 0.001 + 0.01 * np.random.normal()  # 0.1% average with 1% volatility
            equity = 100000 * (1 + daily_return * (i + 1))
            
            performance_analyzer.update_equity(timestamp, equity)
        
        rolling_metrics = performance_analyzer.calculate_rolling_metrics(window_days=30)
        
        assert not rolling_metrics.empty
        assert "rolling_return" in rolling_metrics.columns
        assert "rolling_volatility" in rolling_metrics.columns
        assert "rolling_sharpe" in rolling_metrics.columns
        assert "rolling_drawdown" in rolling_metrics.columns
    
    def test_var_calculation(self, performance_analyzer):
        """Test Value at Risk calculation."""
        # Add some returns with known distribution
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% mean, 2% volatility
        
        base_time = datetime(2024, 1, 1)
        equity = 100000
        
        for i, ret in enumerate(returns):
            timestamp = base_time + timedelta(days=i)
            equity *= (1 + ret)
            performance_analyzer.update_equity(timestamp, equity)
        
        var_95 = performance_analyzer.calculate_var(0.95)
        var_99 = performance_analyzer.calculate_var(0.99)
        
        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher than 95% VaR
    
    def test_strategy_comparison(self):
        """Test strategy comparison functionality."""
        # Create multiple analyzers for different strategies
        analyzers = {}
        
        for strategy in ["strategy_a", "strategy_b", "strategy_c"]:
            config = create_performance_config(initial_capital=100000.0)
            analyzer = PerformanceAnalyzer(config)
            
            # Add some performance data
            base_time = datetime(2024, 1, 1)
            for i in range(30):
                timestamp = base_time + timedelta(days=i)
                # Different performance for each strategy
                multiplier = {"strategy_a": 1.002, "strategy_b": 1.001, "strategy_c": 1.003}[strategy]
                equity = 100000 * (multiplier ** i)
                analyzer.update_equity(timestamp, equity)
            
            analyzers[strategy] = analyzer
        
        comparison = compare_strategies(analyzers)
        
        assert not comparison.empty
        assert len(comparison) == 3
        assert "total_return" in comparison.columns
        assert "sharpe_ratio" in comparison.columns
        assert "max_drawdown" in comparison.columns
        
        # Strategy C should have the highest return
        assert comparison.loc["strategy_c", "total_return"] > comparison.loc["strategy_a", "total_return"]


class TestBacktestEngine:
    """Test suite for BacktestEngine functionality."""
    
    @pytest.fixture
    def backtest_config(self):
        """Create test backtest configuration."""
        return BacktestConfig(
            symbols=["EURUSD"],
            timeframe="1m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_capital=100000.0,
            mode=BacktestMode.FULL_PIPELINE,
            enable_slippage=False,
            enable_commission=False,
            enable_latency=False,
            save_results=False
        )
    
    @pytest.fixture
    def backtest_engine(self, backtest_config):
        """Create BacktestEngine instance."""
        return BacktestEngine(backtest_config)
    
    def test_backtest_config_creation(self):
        """Test backtest configuration creation."""
        config = create_backtest_config(
            symbols=["EURUSD", "GBPUSD"],
            timeframe="5m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=50000.0,
            mode=BacktestMode.STRATEGY_ONLY
        )
        
        assert config.symbols == ["EURUSD", "GBPUSD"]
        assert config.timeframe == "5m"
        assert config.start_date == datetime(2024, 1, 1)
        assert config.end_date == datetime(2024, 1, 31)
        assert config.initial_capital == 50000.0
        assert config.mode == BacktestMode.STRATEGY_ONLY
    
    def test_backtest_engine_initialization(self, backtest_engine, backtest_config):
        """Test BacktestEngine initialization."""
        assert backtest_engine.config == backtest_config
        assert backtest_engine.market_replay is None
        assert backtest_engine.execution_simulator is None
        assert backtest_engine.performance_analyzer is None
        assert not backtest_engine.is_running
        assert backtest_engine.results == {}
    
    @pytest.mark.asyncio
    async def test_backtest_engine_initialization_async(self, backtest_engine):
        """Test BacktestEngine async initialization."""
        await backtest_engine.initialize()
        
        assert backtest_engine.market_replay is not None
        assert backtest_engine.execution_simulator is not None
        assert backtest_engine.performance_analyzer is not None
        assert backtest_engine.state is not None
        
        assert backtest_engine.state.current_equity == backtest_engine.config.initial_capital
        assert backtest_engine.state.current_positions == {}
        assert backtest_engine.state.pending_orders == {}
    
    @pytest.mark.asyncio
    async def test_equity_calculation(self, backtest_engine):
        """Test current equity calculation."""
        await backtest_engine.initialize()
        
        # Add some fills to the execution simulator
        fill1 = Fill(
            order_id="order1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=100000,
            price=1.1000,
            timestamp=datetime.now(),
            commission=7.0,
            slippage=0.0001
        )
        
        backtest_engine.execution_simulator.fills.append(fill1)
        
        # Add current market data
        market_data = MarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            bid=1.1010,
            ask=1.1012,
            volume=1000,
            source="test"
        )
        
        backtest_engine.state.last_market_data["EURUSD"] = market_data
        
        # Add position
        position = Position(
            symbol="EURUSD",
            quantity=100000,
            avg_price=1.1000
        )
        
        backtest_engine.execution_simulator.positions["EURUSD"] = position
        
        equity = await backtest_engine._calculate_current_equity()
        
        # Should be initial capital minus buy cost plus unrealized profit
        expected_equity = (100000 - 100000 * 1.1000 - 7.0 + 
                          (1.1011 - 1.1000) * 100000)  # Using mid price
        
        assert abs(equity - expected_equity) < 1.0  # Allow small rounding differences
    
    @pytest.mark.asyncio
    async def test_signal_processing(self, backtest_engine):
        """Test signal processing pipeline."""
        await backtest_engine.initialize()
        
        # Create a test signal
        signal = Signal(
            symbol="EURUSD",
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.9,
            strategy_name="test_strategy",
            timestamp=datetime.now()
        )
        
        # Mock position sizer to return a specific size
        mock_position_sizer = AsyncMock()
        mock_position_sizer.calculate_position_size.return_value = 100000
        backtest_engine.position_sizer = mock_position_sizer
        
        # Process the signal
        await backtest_engine._process_signal(signal)
        
        # Check that an order was created and submitted
        assert len(backtest_engine.state.pending_orders) > 0
        
        # Verify position sizer was called
        mock_position_sizer.calculate_position_size.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_risk_limit_checking(self, backtest_engine):
        """Test risk limit checking."""
        await backtest_engine.initialize()
        
        # Create a signal that would exceed risk limits
        signal = Signal(
            symbol="EURUSD",
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.9,
            strategy_name="test_strategy",
            timestamp=datetime.now()
        )
        
        # Set a large position size that exceeds limits
        large_position_size = backtest_engine.config.initial_capital * 2  # 200% of capital
        
        # Should fail risk check
        risk_check = await backtest_engine._check_risk_limits(signal, large_position_size)
        assert not risk_check
        
        # Normal position size should pass
        normal_position_size = backtest_engine.config.initial_capital * 0.05  # 5% of capital
        risk_check = await backtest_engine._check_risk_limits(signal, normal_position_size)
        assert risk_check
    
    def test_callback_system(self, backtest_engine):
        """Test event callback system."""
        data_callback = Mock()
        signal_callback = Mock()
        order_callback = Mock()
        fill_callback = Mock()
        
        backtest_engine.add_data_callback(data_callback)
        backtest_engine.add_signal_callback(signal_callback)
        backtest_engine.add_order_callback(order_callback)
        backtest_engine.add_fill_callback(fill_callback)
        
        assert data_callback in backtest_engine.on_data_callbacks
        assert signal_callback in backtest_engine.on_signal_callbacks
        assert order_callback in backtest_engine.on_order_callbacks
        assert fill_callback in backtest_engine.on_fill_callbacks
    
    @pytest.mark.asyncio
    async def test_simple_backtest_function(self):
        """Test the simple backtest utility function."""
        symbols = ["EURUSD"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        initial_capital = 50000.0
        
        # Mock the backtest engine to avoid actual execution
        with patch('backtester.backtest_engine.BacktestEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.initialize = AsyncMock()
            mock_engine.run = AsyncMock(return_value={"test": "results"})
            mock_engine_class.return_value = mock_engine
            
            results = await run_simple_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            assert results == {"test": "results"}
            mock_engine.initialize.assert_called_once()
            mock_engine.run.assert_called_once()


class TestIntegration:
    """Integration tests for the complete backtesting system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_backtest(self):
        """Test complete end-to-end backtesting process."""
        # Create a minimal backtest configuration
        config = BacktestConfig(
            symbols=["EURUSD"],
            timeframe="1m",
            start_date=datetime(2024, 1, 1, 12, 0),
            end_date=datetime(2024, 1, 1, 12, 5),  # 5 minutes of data
            initial_capital=100000.0,
            mode=BacktestMode.FULL_PIPELINE,
            enable_slippage=True,
            enable_commission=True,
            enable_latency=False,  # Disable for faster testing
            save_results=False
        )
        
        # Create backtest engine
        engine = BacktestEngine(config)
        await engine.initialize()
        
        # Mock the market replay to provide test data
        test_data = []
        for i in range(5):
            timestamp = datetime(2024, 1, 1, 12, i)
            data_point = DataPoint(
                timestamp=timestamp,
                symbol="EURUSD",
                data=OHLCV(
                    symbol="EURUSD",
                    timestamp=timestamp,
                    open=1.1000 + i * 0.0001,
                    high=1.1010 + i * 0.0001,
                    low=1.0990 + i * 0.0001,
                    close=1.1005 + i * 0.0001,
                    volume=1000,
                    timeframe="1m"
                ),
                data_type="bar"
            )
            test_data.append(data_point)
        
        engine.market_replay.data_queue = test_data
        engine.market_replay.total_points = len(test_data)
        
        # Run the backtest
        results = await engine.run()
        
        # Verify results structure
        assert "config" in results
        assert "performance" in results
        assert "execution" in results
        assert "final_equity" in results
        
        # Verify config was preserved
        assert results["config"]["symbols"] == ["EURUSD"]
        assert results["config"]["initial_capital"] == 100000.0
        
        # Verify performance metrics exist
        performance = results["performance"]
        assert "total_return" in performance
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance
        assert "total_trades" in performance
        
        # Verify execution statistics exist
        execution = results["execution"]
        assert "total_orders" in execution
        assert "fill_rate" in execution
    
    @pytest.mark.asyncio
    async def test_strategy_comparison_integration(self):
        """Test strategy comparison using multiple backtest runs."""
        symbols = ["EURUSD"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        strategies = ["strategy_a", "strategy_b"]
        results = {}
        
        for strategy in strategies:
            config = BacktestConfig(
                symbols=symbols,
                timeframe="1m",
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                strategies=[strategy],
                mode=BacktestMode.STRATEGY_ONLY,  # Faster for testing
                save_results=False
            )
            
            engine = BacktestEngine(config)
            await engine.initialize()
            
            # Mock some test data
            test_data = []
            for i in range(10):
                timestamp = start_date + timedelta(minutes=i)
                data_point = DataPoint(
                    timestamp=timestamp,
                    symbol="EURUSD",
                    data=OHLCV(
                        symbol="EURUSD",
                        timestamp=timestamp,
                        open=1.1000,
                        high=1.1010,
                        low=1.0990,
                        close=1.1005,
                        volume=1000,
                        timeframe="1m"
                    ),
                    data_type="bar"
                )
                test_data.append(data_point)
            
            engine.market_replay.data_queue = test_data
            engine.market_replay.total_points = len(test_data)
            
            result = await engine.run()
            results[strategy] = result
        
        # Verify we got results for both strategies
        assert len(results) == 2
        assert "strategy_a" in results
        assert "strategy_b" in results
        
        # Each result should have the expected structure
        for strategy, result in results.items():
            assert "performance" in result
            assert "config" in result
            assert result["config"]["strategies"] == [strategy]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 