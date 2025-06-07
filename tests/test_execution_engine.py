"""
Comprehensive Test Suite for Execution Engine - Order Management System.

This module provides extensive testing for the execution engine including:
- Unit tests for core functionality
- Integration tests with brokers
- Performance and latency testing
- Slippage validation and quality metrics
- Algorithm-specific testing
- Error handling and edge cases
"""

import pytest
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional, Any
import numpy as np
import statistics

from core.execution_engine import (
    CoreExecutionEngine, ExecutionEngineConfig, ExecutionAlgorithm,
    OrderPriority, SliceStatus, ExecutionConstraints, OrderSlice,
    ExecutionResult, ExecutionMetrics, create_execution_engine
)
from core.order_router import (
    OrderRouter, BrokerType, ConnectionStatus, BrokerOrder,
    MT5Interface, IBKRInterface, MockBrokerInterface, create_order_router
)
from core.execution_algorithms import (
    TWAPAlgorithm, POVAlgorithm, VWAPAlgorithm, DirectExecutionAlgorithm,
    ExecutionAlgorithmManager, MarketData, VolumeProfile, ExecutionParameters,
    ExecutionState, MarketCondition, ExecutionUrgency, create_execution_algorithm_manager
)
from core.interfaces.trading_interfaces import (
    Order, Position, OrderSide, OrderType, OrderStatus
)
from core.pubsub import ZMQPublisher
import structlog


class TestExecutionEngineCore:
    """Test core execution engine functionality."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration for testing."""
        config_file = tmp_path / "test_execution_settings.yaml"
        config_content = """
execution_algorithms:
  twap:
    slice_interval_seconds: 30
    max_slices: 10
  pov:
    target_participation_rate: 0.1
    max_participation_rate: 0.2
  direct:
    size_threshold: 5000
order_management:
  max_order_age_hours: 1
  retry_attempts: 2
  retry_delay_seconds: 1
  partial_fill_timeout_seconds: 60
slippage_control:
  max_slippage_bps: 15
  warning_threshold_bps: 8
quality_monitoring:
  track_execution_metrics: true
  metrics_retention_days: 7
risk_controls:
  enable_pre_trade_checks: true
  max_order_value: 500000
brokers:
  mock:
    enabled: true
    priority: 1
"""
        config_file.write_text(config_content)
        return ExecutionEngineConfig(str(config_file))
    
    @pytest.fixture
    def execution_engine(self, mock_config):
        """Create execution engine for testing."""
        logger = structlog.get_logger("test")
        return CoreExecutionEngine(config=mock_config, logger=logger)
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_order_submission(self, execution_engine, sample_order):
        """Test basic order submission."""
        order_id = await execution_engine.submit_order(sample_order)
        
        assert order_id == sample_order.order_id
        assert sample_order.order_id in execution_engine.active_orders
        assert execution_engine.orders_processed == 1
    
    @pytest.mark.asyncio
    async def test_order_validation(self, execution_engine):
        """Test order validation logic."""
        # Test invalid quantity
        invalid_order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=-1000.0,  # Invalid negative quantity
            price=1.1000,
            timestamp=datetime.now()
        )
        
        order_id = await execution_engine.submit_order(invalid_order)
        assert invalid_order.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_engine, sample_order):
        """Test order cancellation."""
        # Submit order first
        await execution_engine.submit_order(sample_order)
        
        # Cancel order
        success = await execution_engine.cancel_order(sample_order.order_id)
        
        assert success
        assert sample_order.status == OrderStatus.CANCELLED
        assert sample_order.order_id not in execution_engine.active_orders
        assert len(execution_engine.order_history) == 1
    
    @pytest.mark.asyncio
    async def test_order_slicing(self, execution_engine, sample_order):
        """Test order slicing functionality."""
        slice_size = 2000.0
        time_interval = 30
        
        slices = await execution_engine.slice_order(sample_order, slice_size, time_interval)
        
        assert len(slices) == 5  # 10000 / 2000 = 5 slices
        assert sum(slice.quantity for slice in slices) == sample_order.quantity
        
        for i, slice_order in enumerate(slices):
            assert slice_order.order_id == f"{sample_order.order_id}_slice_{i}"
            assert slice_order.symbol == sample_order.symbol
            assert slice_order.quantity == slice_size
    
    @pytest.mark.asyncio
    async def test_execution_quality_calculation(self, execution_engine, sample_order):
        """Test execution quality metrics calculation."""
        # Mock fills
        fills = [
            {'quantity': 3000, 'price': 1.1001, 'timestamp': datetime.now()},
            {'quantity': 4000, 'price': 1.1002, 'timestamp': datetime.now()},
            {'quantity': 3000, 'price': 1.1000, 'timestamp': datetime.now()}
        ]
        
        metrics = await execution_engine.calculate_execution_quality(sample_order, fills)
        
        assert 'avg_fill_price' in metrics
        assert 'slippage_bps' in metrics
        assert 'execution_time_ms' in metrics
        assert 'fill_rate' in metrics
        assert metrics['total_quantity'] == 10000
        assert metrics['fills_count'] == 3
    
    def test_slippage_calculation(self, execution_engine):
        """Test slippage calculation."""
        # Test buy order slippage
        reference_price = 1.1000
        fill_price = 1.1005
        slippage = execution_engine._calculate_slippage_bps(
            reference_price, fill_price, OrderSide.BUY
        )
        
        expected_slippage = (1.1005 - 1.1000) / 1.1000 * 10000
        assert abs(slippage - expected_slippage) < 0.01
        
        # Test sell order slippage
        slippage_sell = execution_engine._calculate_slippage_bps(
            reference_price, fill_price, OrderSide.SELL
        )
        
        expected_slippage_sell = (1.1000 - 1.1005) / 1.1000 * 10000
        assert abs(slippage_sell - expected_slippage_sell) < 0.01
    
    def test_execution_statistics(self, execution_engine):
        """Test execution statistics tracking."""
        # Simulate some activity
        execution_engine.orders_processed = 10
        execution_engine.orders_filled = 8
        execution_engine.orders_cancelled = 1
        execution_engine.orders_failed = 1
        execution_engine.slippage_history.extend([1.5, 2.0, 1.8, 2.2])
        
        stats = execution_engine.get_execution_statistics()
        
        assert stats['orders_processed'] == 10
        assert stats['orders_filled'] == 8
        assert stats['fill_rate'] == 0.8
        assert stats['avg_slippage_bps'] == statistics.mean([1.5, 2.0, 1.8, 2.2])


class TestOrderRouter:
    """Test order routing functionality."""
    
    @pytest.fixture
    def mock_router_config(self, tmp_path):
        """Create mock router configuration."""
        config_file = tmp_path / "test_router_config.yaml"
        config_content = """
brokers:
  mock:
    enabled: true
    priority: 1
    max_orders_per_second: 10
    connection_timeout: 5
    retry_attempts: 2
    credentials: {}
    settings:
      simulate_latency: true
      latency_ms: 10
"""
        config_file.write_text(config_content)
        return str(config_file)
    
    @pytest.fixture
    def order_router(self, mock_router_config):
        """Create order router for testing."""
        logger = structlog.get_logger("test")
        return OrderRouter(config_path=mock_router_config, logger=logger)
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for routing tests."""
        return Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_router_initialization(self, order_router):
        """Test order router initialization."""
        assert BrokerType.MOCK in order_router.brokers
        assert BrokerType.MOCK in order_router.broker_configs
        assert order_router.broker_configs[BrokerType.MOCK].enabled
    
    @pytest.mark.asyncio
    async def test_broker_connection(self, order_router):
        """Test broker connection process."""
        await order_router.start()
        
        # Check that mock broker is connected
        mock_health = order_router.broker_health[BrokerType.MOCK]
        assert mock_health['status'] == ConnectionStatus.CONNECTED
    
    @pytest.mark.asyncio
    async def test_order_routing(self, order_router, sample_order):
        """Test order routing to broker."""
        await order_router.start()
        
        broker_order = await order_router.route_order(sample_order)
        
        assert broker_order.internal_order_id == sample_order.order_id
        assert broker_order.broker_type == BrokerType.MOCK
        assert broker_order.symbol == sample_order.symbol
        assert broker_order.quantity == sample_order.quantity
        assert sample_order.order_id in order_router.routed_orders
    
    @pytest.mark.asyncio
    async def test_order_cancellation_routing(self, order_router, sample_order):
        """Test order cancellation through router."""
        await order_router.start()
        
        # Route order first
        await order_router.route_order(sample_order)
        
        # Cancel order
        success = await order_router.cancel_order(sample_order.order_id)
        
        assert success
    
    def test_routing_statistics(self, order_router):
        """Test routing statistics collection."""
        stats = order_router.get_routing_statistics()
        
        assert 'routing_stats' in stats
        assert 'broker_health' in stats
        assert 'active_orders' in stats
        assert 'connected_brokers' in stats


class TestExecutionAlgorithms:
    """Test execution algorithms."""
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            last=1.1001,
            volume=50000,
            spread_bps=1.8,
            volatility=0.015,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def execution_parameters(self):
        """Create execution parameters."""
        return ExecutionParameters(
            algorithm=ExecutionAlgorithm.TWAP,
            urgency=ExecutionUrgency.NORMAL,
            max_participation_rate=0.2,
            target_completion_time=timedelta(hours=1),
            price_tolerance_bps=5.0
        )
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for algorithm testing."""
        return Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50000.0,
            price=1.1001,
            timestamp=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_twap_algorithm(self, market_data, execution_parameters, sample_order):
        """Test TWAP algorithm execution."""
        twap_algo = TWAPAlgorithm()
        
        slices = await twap_algo.execute_order(sample_order, execution_parameters, market_data)
        
        assert len(slices) > 0
        assert sample_order.order_id in twap_algo.execution_states
        
        execution_state = twap_algo.execution_states[sample_order.order_id]
        assert execution_state.algorithm == ExecutionAlgorithm.TWAP
        assert execution_state.total_quantity == sample_order.quantity
    
    @pytest.mark.asyncio
    async def test_pov_algorithm(self, market_data, sample_order):
        """Test POV algorithm execution."""
        pov_algo = POVAlgorithm()
        pov_params = ExecutionParameters(
            algorithm=ExecutionAlgorithm.POV,
            max_participation_rate=0.15
        )
        
        slices = await pov_algo.execute_order(sample_order, pov_params, market_data)
        
        assert len(slices) > 0
        execution_state = pov_algo.execution_states[sample_order.order_id]
        assert execution_state.algorithm == ExecutionAlgorithm.POV
    
    @pytest.mark.asyncio
    async def test_direct_algorithm(self, market_data, sample_order):
        """Test Direct execution algorithm."""
        direct_algo = DirectExecutionAlgorithm()
        direct_params = ExecutionParameters(
            algorithm=ExecutionAlgorithm.DIRECT,
            allow_market_orders=True
        )
        
        slices = await direct_algo.execute_order(sample_order, direct_params, market_data)
        
        assert len(slices) == 1  # Direct execution should create single slice
        assert slices[0].quantity == sample_order.quantity
    
    @pytest.mark.asyncio
    async def test_algorithm_manager(self, market_data, execution_parameters, sample_order):
        """Test execution algorithm manager."""
        manager = ExecutionAlgorithmManager()
        
        slices = await manager.execute_order(sample_order, execution_parameters, market_data)
        
        assert len(slices) > 0
        assert sample_order.order_id in manager.active_executions
        
        # Test market data update
        manager.update_market_data(market_data)
        
        # Test execution state retrieval
        state = manager.get_execution_state(sample_order.order_id)
        assert state is not None
        assert state.order_id == sample_order.order_id


class TestPerformanceAndLatency:
    """Test performance and latency requirements."""
    
    @pytest.fixture
    def execution_engine(self):
        """Create execution engine for performance testing."""
        return create_execution_engine()
    
    @pytest.mark.asyncio
    async def test_order_submission_latency(self, execution_engine):
        """Test order submission latency (target: <10ms)."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
        
        start_time = time.perf_counter()
        await execution_engine.submit_order(order)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 10, f"Order submission latency {latency_ms:.2f}ms exceeds 10ms target"
    
    @pytest.mark.asyncio
    async def test_order_cancellation_latency(self, execution_engine):
        """Test order cancellation latency (target: <5ms)."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
        
        await execution_engine.submit_order(order)
        
        start_time = time.perf_counter()
        await execution_engine.cancel_order(order.order_id)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 5, f"Order cancellation latency {latency_ms:.2f}ms exceeds 5ms target"
    
    @pytest.mark.asyncio
    async def test_concurrent_order_processing(self, execution_engine):
        """Test concurrent order processing capability."""
        num_orders = 100
        orders = []
        
        # Create multiple orders
        for i in range(num_orders):
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1000.0,
                price=1.1000,
                timestamp=datetime.now()
            )
            orders.append(order)
        
        # Submit all orders concurrently
        start_time = time.perf_counter()
        tasks = [execution_engine.submit_order(order) for order in orders]
        await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        orders_per_second = num_orders / total_time
        
        assert orders_per_second > 50, f"Processing rate {orders_per_second:.1f} orders/sec below target"
        assert execution_engine.orders_processed == num_orders
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, execution_engine):
        """Test memory usage under high load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Submit many orders
        for i in range(1000):
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1000.0,
                price=1.1000,
                timestamp=datetime.now()
            )
            await execution_engine.submit_order(order)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 1000 orders)
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"


class TestSlippageValidation:
    """Test slippage validation and control."""
    
    @pytest.fixture
    def execution_engine(self):
        """Create execution engine with strict slippage controls."""
        return create_execution_engine()
    
    def test_slippage_calculation_accuracy(self, execution_engine):
        """Test accuracy of slippage calculations."""
        test_cases = [
            # (reference_price, fill_price, side, expected_slippage_bps)
            (1.1000, 1.1005, OrderSide.BUY, 45.45),  # Positive slippage for buy
            (1.1000, 1.0995, OrderSide.BUY, -45.45),  # Negative slippage for buy
            (1.1000, 1.0995, OrderSide.SELL, 45.45),  # Positive slippage for sell
            (1.1000, 1.1005, OrderSide.SELL, -45.45),  # Negative slippage for sell
        ]
        
        for ref_price, fill_price, side, expected in test_cases:
            slippage = execution_engine._calculate_slippage_bps(ref_price, fill_price, side)
            assert abs(slippage - expected) < 0.1, f"Slippage calculation error: {slippage} vs {expected}"
    
    @pytest.mark.asyncio
    async def test_slippage_monitoring(self, execution_engine):
        """Test slippage monitoring and alerting."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
        
        # Simulate high slippage scenario
        fills = [
            {'quantity': 10000, 'price': 1.1015, 'timestamp': datetime.now()}  # 13.6 bps slippage
        ]
        
        metrics = await execution_engine.calculate_execution_quality(order, fills)
        
        # Should detect high slippage
        assert metrics['slippage_bps'] > execution_engine.config.slippage_warning_bps
    
    def test_slippage_history_tracking(self, execution_engine):
        """Test slippage history tracking."""
        # Add some slippage data
        slippage_values = [2.5, 3.1, 1.8, 4.2, 2.9]
        execution_engine.slippage_history.extend(slippage_values)
        
        stats = execution_engine.get_execution_statistics()
        expected_avg = statistics.mean(slippage_values)
        
        assert abs(stats['avg_slippage_bps'] - expected_avg) < 0.01


class TestIntegrationScenarios:
    """Test integration scenarios and end-to-end workflows."""
    
    @pytest.fixture
    def full_system(self, tmp_path):
        """Create full system for integration testing."""
        # Create test config
        config_file = tmp_path / "integration_config.yaml"
        config_content = """
execution_algorithms:
  twap:
    slice_interval_seconds: 10
    max_slices: 5
  pov:
    target_participation_rate: 0.1
  direct:
    size_threshold: 5000
order_management:
  max_order_age_hours: 1
  retry_attempts: 2
slippage_control:
  max_slippage_bps: 10
brokers:
  mock:
    enabled: true
    priority: 1
"""
        config_file.write_text(config_content)
        
        # Create execution engine and router
        config = ExecutionEngineConfig(str(config_file))
        execution_engine = CoreExecutionEngine(config=config)
        order_router = OrderRouter(config_path=str(config_file))
        
        return execution_engine, order_router
    
    @pytest.mark.asyncio
    async def test_end_to_end_order_flow(self, full_system):
        """Test complete order flow from submission to execution."""
        execution_engine, order_router = full_system
        
        # Start router
        await order_router.start()
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=25000.0,
            price=1.1001,
            timestamp=datetime.now()
        )
        
        # Submit through execution engine
        order_id = await execution_engine.submit_order(order)
        
        # Verify order is active
        assert order_id in execution_engine.active_orders
        
        # Route through order router
        broker_order = await order_router.route_order(order)
        
        # Verify routing
        assert broker_order.internal_order_id == order_id
        assert broker_order.broker_type == BrokerType.MOCK
        
        # Check order status
        status = await order_router.get_order_status(order_id)
        assert status in [OrderStatus.PENDING, OrderStatus.FILLED]
    
    @pytest.mark.asyncio
    async def test_algorithm_switching_scenario(self, full_system):
        """Test switching between execution algorithms."""
        execution_engine, _ = full_system
        
        # Test different order sizes triggering different algorithms
        test_orders = [
            (3000, ExecutionAlgorithm.DIRECT),    # Small order -> Direct
            (15000, ExecutionAlgorithm.TWAP),     # Medium order -> TWAP
            (100000, ExecutionAlgorithm.POV),     # Large order -> POV
        ]
        
        for quantity, expected_algo in test_orders:
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=1.1000,
                timestamp=datetime.now()
            )
            
            # Submit order and check algorithm selection
            await execution_engine.submit_order(order)
            selected_algo = await execution_engine._select_execution_algorithm(order)
            
            assert selected_algo == expected_algo, f"Wrong algorithm for quantity {quantity}"
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, full_system):
        """Test error recovery and retry mechanisms."""
        execution_engine, order_router = full_system
        
        # Start router
        await order_router.start()
        
        # Create order that might fail
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="INVALID",  # Invalid symbol to trigger error
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
        
        # Submit order (should handle error gracefully)
        order_id = await execution_engine.submit_order(order)
        
        # Check that error was handled properly
        assert order.status in [OrderStatus.REJECTED, OrderStatus.PENDING]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def execution_engine(self):
        """Create execution engine for edge case testing."""
        return create_execution_engine()
    
    @pytest.mark.asyncio
    async def test_zero_quantity_order(self, execution_engine):
        """Test handling of zero quantity orders."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.0,  # Zero quantity
            price=1.1000,
            timestamp=datetime.now()
        )
        
        await execution_engine.submit_order(order)
        assert order.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_negative_quantity_order(self, execution_engine):
        """Test handling of negative quantity orders."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=-1000.0,  # Negative quantity
            price=1.1000,
            timestamp=datetime.now()
        )
        
        await execution_engine.submit_order(order)
        assert order.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_duplicate_order_id(self, execution_engine):
        """Test handling of duplicate order IDs."""
        order_id = str(uuid.uuid4())
        
        order1 = Order(
            order_id=order_id,
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
        
        order2 = Order(
            order_id=order_id,  # Same ID
            symbol="GBPUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=5000.0,
            price=1.3000,
            timestamp=datetime.now()
        )
        
        # Submit first order
        await execution_engine.submit_order(order1)
        
        # Submit second order with same ID
        await execution_engine.submit_order(order2)
        
        # Should handle gracefully (implementation dependent)
        assert len(execution_engine.active_orders) <= 2
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, execution_engine):
        """Test cancelling non-existent order."""
        fake_order_id = str(uuid.uuid4())
        
        success = await execution_engine.cancel_order(fake_order_id)
        assert not success
    
    @pytest.mark.asyncio
    async def test_extremely_large_order(self, execution_engine):
        """Test handling of extremely large orders."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1e12,  # Extremely large quantity
            price=1.1000,
            timestamp=datetime.now()
        )
        
        await execution_engine.submit_order(order)
        # Should be rejected due to value limits
        assert order.status == OrderStatus.REJECTED


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for execution engine."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_order_throughput_benchmark(self):
        """Benchmark order processing throughput."""
        execution_engine = create_execution_engine()
        
        num_orders = 1000
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(num_orders):
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1000.0,
                price=1.1000,
                timestamp=datetime.now()
            )
            tasks.append(execution_engine.submit_order(order))
        
        await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = num_orders / total_time
        
        print(f"Order throughput: {throughput:.1f} orders/second")
        assert throughput > 100, f"Throughput {throughput:.1f} below target of 100 orders/sec"
    
    @pytest.mark.benchmark
    def test_slippage_calculation_performance(self):
        """Benchmark slippage calculation performance."""
        execution_engine = create_execution_engine()
        
        num_calculations = 10000
        start_time = time.perf_counter()
        
        for i in range(num_calculations):
            execution_engine._calculate_slippage_bps(1.1000, 1.1005, OrderSide.BUY)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        calculations_per_second = num_calculations / total_time
        
        print(f"Slippage calculations: {calculations_per_second:.0f} calculations/second")
        assert calculations_per_second > 50000, "Slippage calculation performance too slow"


# Fixtures for test data
@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return MarketData(
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1002,
        last=1.1001,
        volume=25000,
        spread_bps=1.8,
        volatility=0.012,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_volume_profile():
    """Sample volume profile for testing."""
    return VolumeProfile(
        symbol="EURUSD",
        interval_minutes=60,
        historical_volumes=[20000, 35000, 45000, 30000, 25000],
        avg_volume=31000,
        volume_std=8500,
        peak_hours=[8, 9, 10, 14, 15, 16],
        quiet_hours=[0, 1, 2, 3, 22, 23],
        last_update=datetime.now()
    )


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=core.execution_engine",
        "--cov=core.order_router", 
        "--cov=core.execution_algorithms",
        "--cov-report=html",
        "--cov-report=term-missing"
    ]) 