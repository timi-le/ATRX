"""
Execution Engine Integration Example - FX AI-Quant Trading System.

This example demonstrates how to use the execution engine with different
algorithms, order types, and integration scenarios.

Features demonstrated:
- Basic order submission and management
- Different execution algorithms (TWAP, POV, VWAP, Direct)
- Order routing and broker integration
- Performance monitoring and quality metrics
- Error handling and edge cases
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta

import structlog

from core.execution_algorithms import (
    ExecutionParameters,
    ExecutionUrgency,
    MarketData,
    create_execution_algorithm_manager,
)
from core.execution_engine import (
    ExecutionAlgorithm,
    create_execution_engine,
)
from core.interfaces.trading_interfaces import Order, OrderSide, OrderType
from core.order_router import create_order_router


class ExecutionEngineDemo:
    """Demonstration of execution engine capabilities."""

    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.execution_engine = None
        self.order_router = None
        self.algorithm_manager = None

    async def initialize(self):
        """Initialize the execution engine components."""
        try:
            # Create execution engine
            self.execution_engine = create_execution_engine()

            # Create order router
            self.order_router = create_order_router()

            # Create algorithm manager
            self.algorithm_manager = create_execution_algorithm_manager()

            # Start order router
            await self.order_router.start()

            self.logger.info("Execution engine demo initialized successfully")

        except Exception as e:
            self.logger.error(
                "Failed to initialize execution engine demo", error=str(e)
            )
            raise

    async def demo_basic_order_submission(self):
        """Demonstrate basic order submission and management."""
        print("\n=== Basic Order Submission Demo ===")

        # Create a simple market order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now(),
        )

        print(f"Submitting order: {order.order_id} for {order.quantity} {order.symbol}")

        # Submit order
        start_time = time.perf_counter()
        order_id = await self.execution_engine.submit_order(order)
        end_time = time.perf_counter()

        submission_time = (end_time - start_time) * 1000
        print(f"Order submitted in {submission_time:.2f}ms")

        # Check order status
        status = await self.execution_engine.get_order_status(order_id)
        print(f"Order status: {status.value}")

        # Get order details
        details = self.execution_engine.get_order_details(order_id)
        if details:
            print(f"Order details: {details}")

        return order_id

    async def demo_order_cancellation(self):
        """Demonstrate order cancellation."""
        print("\n=== Order Cancellation Demo ===")

        # Create a limit order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="GBPUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=5000.0,
            price=1.3050,
            timestamp=datetime.now(),
        )

        print(f"Submitting limit order: {order.order_id}")

        # Submit order
        order_id = await self.execution_engine.submit_order(order)

        # Wait a moment
        await asyncio.sleep(0.1)

        # Cancel order
        print(f"Cancelling order: {order_id}")
        start_time = time.perf_counter()
        success = await self.execution_engine.cancel_order(order_id)
        end_time = time.perf_counter()

        cancellation_time = (end_time - start_time) * 1000
        print(f"Order cancelled in {cancellation_time:.2f}ms, success: {success}")

        # Check final status
        status = await self.execution_engine.get_order_status(order_id)
        print(f"Final order status: {status.value}")

    async def demo_twap_algorithm(self):
        """Demonstrate TWAP algorithm execution."""
        print("\n=== TWAP Algorithm Demo ===")

        # Create market data
        market_data = MarketData(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            last=1.1001,
            volume=50000,
            spread_bps=1.8,
            volatility=0.015,
            timestamp=datetime.now(),
        )

        # Create execution parameters
        parameters = ExecutionParameters(
            algorithm=ExecutionAlgorithm.TWAP,
            urgency=ExecutionUrgency.NORMAL,
            max_participation_rate=0.2,
            target_completion_time=timedelta(minutes=30),
            price_tolerance_bps=5.0,
        )

        # Create large order for TWAP
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100000.0,
            price=1.1001,
            timestamp=datetime.now(),
        )

        print(f"Executing TWAP order: {order.quantity} {order.symbol}")
        print(f"Target completion time: {parameters.target_completion_time}")

        # Execute using algorithm manager
        slices = await self.algorithm_manager.execute_order(
            order, parameters, market_data
        )

        print(f"TWAP generated {len(slices)} initial slices")
        for i, slice_obj in enumerate(slices):
            print(f"  Slice {i+1}: {slice_obj.quantity} @ {slice_obj.price}")

        # Get execution state
        state = self.algorithm_manager.get_execution_state(order.order_id)
        if state:
            print(f"Execution state: {state.total_slices} total slices planned")

    async def demo_pov_algorithm(self):
        """Demonstrate POV algorithm execution."""
        print("\n=== POV Algorithm Demo ===")

        # Create market data with higher volume
        market_data = MarketData(
            symbol="GBPUSD",
            bid=1.3000,
            ask=1.3003,
            last=1.3001,
            volume=75000,
            spread_bps=2.3,
            volatility=0.018,
            timestamp=datetime.now(),
        )

        # Create POV parameters
        parameters = ExecutionParameters(
            algorithm=ExecutionAlgorithm.POV,
            urgency=ExecutionUrgency.HIGH,
            max_participation_rate=0.15,
            min_participation_rate=0.05,
            price_tolerance_bps=8.0,
        )

        # Create order for POV
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="GBPUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50000.0,
            price=1.3002,
            timestamp=datetime.now(),
        )

        print(f"Executing POV order: {order.quantity} {order.symbol}")
        print(f"Target participation: {parameters.max_participation_rate*100:.1f}%")

        # Execute using algorithm manager
        slices = await self.algorithm_manager.execute_order(
            order, parameters, market_data
        )

        print(f"POV generated {len(slices)} initial slices")
        for i, slice_obj in enumerate(slices):
            print(f"  Slice {i+1}: {slice_obj.quantity} @ {slice_obj.price}")

    async def demo_direct_execution(self):
        """Demonstrate direct execution algorithm."""
        print("\n=== Direct Execution Demo ===")

        # Create market data
        market_data = MarketData(
            symbol="USDJPY",
            bid=150.00,
            ask=150.03,
            last=150.01,
            volume=30000,
            spread_bps=2.0,
            volatility=0.012,
            timestamp=datetime.now(),
        )

        # Create direct execution parameters
        parameters = ExecutionParameters(
            algorithm=ExecutionAlgorithm.DIRECT,
            urgency=ExecutionUrgency.URGENT,
            allow_market_orders=True,
            price_tolerance_bps=10.0,
        )

        # Create small order for direct execution
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="USDJPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=3000.0,
            price=150.01,
            timestamp=datetime.now(),
        )

        print(f"Executing direct order: {order.quantity} {order.symbol}")

        # Execute using algorithm manager
        start_time = time.perf_counter()
        slices = await self.algorithm_manager.execute_order(
            order, parameters, market_data
        )
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000
        print(f"Direct execution completed in {execution_time:.2f}ms")
        print(f"Generated {len(slices)} slice(s)")

        if slices:
            slice_obj = slices[0]
            print(f"  Single slice: {slice_obj.quantity} @ {slice_obj.price}")

    async def demo_order_routing(self):
        """Demonstrate order routing through brokers."""
        print("\n=== Order Routing Demo ===")

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=15000.0,
            price=1.1001,
            timestamp=datetime.now(),
        )

        print(f"Routing order: {order.order_id}")

        # Route order
        broker_order = await self.order_router.route_order(order)

        print(f"Order routed to: {broker_order.broker_type.value}")
        print(f"Broker order ID: {broker_order.broker_order_id}")

        # Check routing statistics
        stats = self.order_router.get_routing_statistics()
        print(f"Routing statistics: {stats}")

    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and metrics."""
        print("\n=== Performance Monitoring Demo ===")

        # Submit multiple orders to generate statistics
        orders = []
        for i in range(10):
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol="EURUSD",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=1000.0 + i * 500,
                price=1.1000 + i * 0.0001,
                timestamp=datetime.now(),
            )
            orders.append(order)

        print(f"Submitting {len(orders)} orders for performance testing...")

        # Submit all orders
        start_time = time.perf_counter()
        tasks = [self.execution_engine.submit_order(order) for order in orders]
        await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        throughput = len(orders) / total_time

        print(f"Processed {len(orders)} orders in {total_time:.3f}s")
        print(f"Throughput: {throughput:.1f} orders/second")

        # Get execution statistics
        stats = self.execution_engine.get_execution_statistics()
        print(f"Execution statistics:")
        print(f"  Orders processed: {stats['orders_processed']}")
        print(f"  Orders filled: {stats['orders_filled']}")
        print(f"  Fill rate: {stats['fill_rate']:.2%}")
        print(f"  Average slippage: {stats['avg_slippage_bps']:.2f} bps")
        print(f"  Average execution time: {stats['avg_execution_time_ms']:.2f} ms")

    async def demo_error_handling(self):
        """Demonstrate error handling scenarios."""
        print("\n=== Error Handling Demo ===")

        # Test invalid order
        invalid_order = Order(
            order_id=str(uuid.uuid4()),
            symbol="INVALID",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=-1000.0,  # Invalid negative quantity
            price=1.1000,
            timestamp=datetime.now(),
        )

        print("Submitting invalid order (negative quantity)...")
        order_id = await self.execution_engine.submit_order(invalid_order)
        status = await self.execution_engine.get_order_status(order_id)
        print(f"Invalid order status: {status.value}")

        # Test cancelling non-existent order
        fake_order_id = str(uuid.uuid4())
        print(f"Attempting to cancel non-existent order: {fake_order_id}")
        success = await self.execution_engine.cancel_order(fake_order_id)
        print(f"Cancellation result: {success}")

        # Test extremely large order
        large_order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1e10,  # Extremely large
            price=1.1000,
            timestamp=datetime.now(),
        )

        print("Submitting extremely large order...")
        order_id = await self.execution_engine.submit_order(large_order)
        status = await self.execution_engine.get_order_status(order_id)
        print(f"Large order status: {status.value}")

    async def demo_slippage_monitoring(self):
        """Demonstrate slippage monitoring and quality metrics."""
        print("\n=== Slippage Monitoring Demo ===")

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now(),
        )

        # Simulate fills with different slippage scenarios
        fills_scenarios = [
            # Good execution (minimal slippage)
            [{"quantity": 10000, "price": 1.1001, "timestamp": datetime.now()}],
            # Moderate slippage
            [{"quantity": 10000, "price": 1.1005, "timestamp": datetime.now()}],
            # High slippage
            [{"quantity": 10000, "price": 1.1012, "timestamp": datetime.now()}],
        ]

        for i, fills in enumerate(fills_scenarios):
            print(f"\nScenario {i+1}: Fill price {fills[0]['price']}")

            metrics = await self.execution_engine.calculate_execution_quality(
                order, fills
            )

            if metrics:
                print(f"  Slippage: {metrics['slippage_bps']:.2f} bps")
                print(f"  Execution time: {metrics['execution_time_ms']:.2f} ms")
                print(f"  Fill rate: {metrics['fill_rate']:.2%}")

                # Check if slippage exceeds warning threshold
                if (
                    metrics["slippage_bps"]
                    > self.execution_engine.config.slippage_warning_bps
                ):
                    print(f"  ⚠️  High slippage detected!")

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.order_router:
                await self.order_router.stop()

            self.logger.info("Execution engine demo cleanup completed")

        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))

    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        try:
            await self.initialize()

            print("🚀 Starting Execution Engine Demonstration")
            print("=" * 50)

            # Run all demos
            await self.demo_basic_order_submission()
            await self.demo_order_cancellation()
            await self.demo_twap_algorithm()
            await self.demo_pov_algorithm()
            await self.demo_direct_execution()
            await self.demo_order_routing()
            await self.demo_performance_monitoring()
            await self.demo_error_handling()
            await self.demo_slippage_monitoring()

            print("\n" + "=" * 50)
            print("✅ All demonstrations completed successfully!")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the execution engine demo."""
    demo = ExecutionEngineDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Run the demo
    asyncio.run(main())
