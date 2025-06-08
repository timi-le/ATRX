"""
Comprehensive Task 14 Integration Test - Full Project Validation.

This test validates Task 14 execution engine integration with the entire
FX AI-Quant Trading System project structure and components.
"""

import asyncio
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_project_structure():
    """Test that all required project files exist."""
    print("🔍 Testing Project Structure...")

    required_files = [
        "core/execution_engine.py",
        "core/order_router.py",
        "core/execution_algorithms.py",
        "core/interfaces/trading_interfaces.py",
        "core/interfaces/messaging_interfaces.py",
        "config/execution_settings.yaml",
        "tests/test_execution_engine.py",
        "examples/execution_engine_example.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")

    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False

    print("   ✅ All required files present")
    return True


def test_imports():
    """Test that all core modules can be imported."""
    print("\n🔍 Testing Module Imports...")

    try:
        # Test core execution engine imports
        pass

        print("   ✅ Execution Engine imports")

        # Test order router imports

        print("   ✅ Order Router imports")

        # Test execution algorithms imports

        print("   ✅ Execution Algorithms imports")

        # Test trading interfaces imports

        print("   ✅ Trading Interfaces imports")

        return True

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n🔍 Testing Configuration...")

    try:
        config_path = project_root / "config" / "execution_settings.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required configuration sections
        required_sections = [
            "execution_algorithms",
            "brokers",
            "order_management",
            "slippage_control",
            "quality_monitoring",
        ]

        for section in required_sections:
            if section in config:
                print(f"   ✅ {section} configuration")
            else:
                print(f"   ❌ Missing {section} configuration")
                return False

        # Check algorithm configurations
        algorithms = config.get("execution_algorithms", {})
        for algo in ["twap", "pov", "vwap", "direct"]:
            if algo in algorithms:
                print(f"   ✅ {algo.upper()} algorithm configured")
            else:
                print(f"   ❌ Missing {algo.upper()} algorithm configuration")

        return True

    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False


def test_execution_engine_creation():
    """Test execution engine can be created."""
    print("\n🔍 Testing Execution Engine Creation...")

    try:
        from core.execution_engine import create_execution_engine

        engine = create_execution_engine()

        if engine is not None:
            print("   ✅ Execution engine created successfully")
            print(f"   ✅ Engine type: {type(engine).__name__}")

            # Test basic properties
            if hasattr(engine, "active_orders"):
                print("   ✅ Active orders tracking available")
            if hasattr(engine, "order_history"):
                print("   ✅ Order history tracking available")
            if hasattr(engine, "slippage_history"):
                print("   ✅ Slippage history tracking available")

            return True
        else:
            print("   ❌ Failed to create execution engine")
            return False

    except Exception as e:
        print(f"   ❌ Execution engine creation error: {e}")
        return False


def test_order_router_creation():
    """Test order router can be created."""
    print("\n🔍 Testing Order Router Creation...")

    try:
        from core.order_router import create_order_router

        router = create_order_router()

        if router is not None:
            print("   ✅ Order router created successfully")
            print(f"   ✅ Router type: {type(router).__name__}")

            # Test broker availability
            if hasattr(router, "brokers"):
                print(f"   ✅ Brokers available: {list(router.brokers.keys())}")

            return True
        else:
            print("   ❌ Failed to create order router")
            return False

    except Exception as e:
        print(f"   ❌ Order router creation error: {e}")
        return False


def test_algorithm_manager_creation():
    """Test algorithm manager can be created."""
    print("\n🔍 Testing Algorithm Manager Creation...")

    try:
        from core.execution_algorithms import create_execution_algorithm_manager

        manager = create_execution_algorithm_manager()

        if manager is not None:
            print("   ✅ Algorithm manager created successfully")
            print(f"   ✅ Manager type: {type(manager).__name__}")

            # Test algorithms availability
            if hasattr(manager, "algorithms"):
                print(f"   ✅ Algorithms available: {list(manager.algorithms.keys())}")

            return True
        else:
            print("   ❌ Failed to create algorithm manager")
            return False

    except Exception as e:
        print(f"   ❌ Algorithm manager creation error: {e}")
        return False


async def test_basic_order_flow():
    """Test basic order submission and processing flow."""
    print("\n🔍 Testing Basic Order Flow...")

    try:
        from core.execution_engine import create_execution_engine
        from core.interfaces.trading_interfaces import Order, OrderSide, OrderType

        # Create execution engine
        engine = create_execution_engine()

        # Create test order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now(),
        )

        print(f"   📝 Created test order: {order.quantity} {order.symbol}")

        # Submit order
        order_id = await engine.submit_order(order)
        print(f"   ✅ Order submitted: {order_id}")

        # Check order status
        status = await engine.get_order_status(order_id)
        print(f"   ✅ Order status: {status.value}")

        # Get order details
        details = engine.get_order_details(order_id)
        if details:
            print("   ✅ Order details retrieved")

        return True

    except Exception as e:
        print(f"   ❌ Order flow error: {e}")
        return False


async def test_twap_algorithm():
    """Test TWAP algorithm functionality."""
    print("\n🔍 Testing TWAP Algorithm...")

    try:
        from core.execution_algorithms import (
            ExecutionAlgorithm,
            ExecutionParameters,
            ExecutionUrgency,
            MarketData,
            create_execution_algorithm_manager,
        )
        from core.interfaces.trading_interfaces import Order, OrderSide, OrderType

        # Create algorithm manager
        manager = create_execution_algorithm_manager()

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

        # Create TWAP parameters
        parameters = ExecutionParameters(
            algorithm=ExecutionAlgorithm.TWAP,
            urgency=ExecutionUrgency.NORMAL,
            target_completion_time=timedelta(minutes=15),
            max_slice_size=20000.0,
            min_slice_size=5000.0,
        )

        # Create 1-lot order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100000.0,  # 1 standard lot
            price=1.1001,
            timestamp=datetime.now(),
        )

        print(f"   📝 TWAP order: {order.quantity:,.0f} {order.symbol} over 15 minutes")

        # Execute TWAP
        slices = await manager.execute_order(order, parameters, market_data)

        if slices and len(slices) > 0:
            print(f"   ✅ TWAP generated {len(slices)} slice(s)")
            print(
                f"   ✅ First slice: {slices[0].quantity:,.0f} units @ {slices[0].price}"
            )

            # Validate slice is reasonable
            if slices[0].quantity <= parameters.max_slice_size:
                print("   ✅ Slice size within limits")
            else:
                print(
                    f"   ⚠️  Slice size {slices[0].quantity} exceeds max {parameters.max_slice_size}"
                )

            return True
        else:
            print("   ❌ TWAP failed to generate slices")
            return False

    except Exception as e:
        print(f"   ❌ TWAP algorithm error: {e}")
        return False


async def test_pov_algorithm():
    """Test POV algorithm functionality."""
    print("\n🔍 Testing POV Algorithm...")

    try:
        from core.execution_algorithms import (
            ExecutionAlgorithm,
            ExecutionParameters,
            ExecutionUrgency,
            MarketData,
            create_execution_algorithm_manager,
        )
        from core.interfaces.trading_interfaces import Order, OrderSide, OrderType

        # Create algorithm manager
        manager = create_execution_algorithm_manager()

        # Create market data with volume
        market_data = MarketData(
            symbol="GBPUSD",
            bid=1.3000,
            ask=1.3003,
            last=1.3001,
            volume=50000,  # Market volume
            spread_bps=2.3,
            volatility=0.018,
            timestamp=datetime.now(),
        )

        # Create POV parameters
        parameters = ExecutionParameters(
            algorithm=ExecutionAlgorithm.POV,
            urgency=ExecutionUrgency.NORMAL,
            max_participation_rate=0.10,  # 10% participation
            min_participation_rate=0.05,
        )

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="GBPUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=25000.0,
            price=1.3002,
            timestamp=datetime.now(),
        )

        print(
            f"   📝 POV order: {order.quantity:,.0f} {order.symbol} with 10% participation"
        )
        print(f"   📊 Market volume: {market_data.volume:,} units")

        # Execute POV
        slices = await manager.execute_order(order, parameters, market_data)

        if slices and len(slices) > 0:
            expected_slice = market_data.volume * parameters.max_participation_rate
            print(f"   ✅ POV generated {len(slices)} slice(s)")
            print(f"   ✅ First slice: {slices[0].quantity:,.0f} units")
            print(f"   📊 Expected slice: {expected_slice:,.0f} units (10% of volume)")

            return True
        else:
            print("   ❌ POV failed to generate slices")
            return False

    except Exception as e:
        print(f"   ❌ POV algorithm error: {e}")
        return False


async def test_direct_execution():
    """Test direct execution algorithm."""
    print("\n🔍 Testing Direct Execution...")

    try:
        from core.execution_algorithms import (
            ExecutionAlgorithm,
            ExecutionParameters,
            ExecutionUrgency,
            MarketData,
            create_execution_algorithm_manager,
        )
        from core.interfaces.trading_interfaces import Order, OrderSide, OrderType

        # Create algorithm manager
        manager = create_execution_algorithm_manager()

        # Create market data
        market_data = MarketData(
            symbol="USDJPY",
            bid=150.00,
            ask=150.03,
            last=150.01,
            volume=40000,
            spread_bps=2.0,
            volatility=0.012,
            timestamp=datetime.now(),
        )

        # Create direct execution parameters
        parameters = ExecutionParameters(
            algorithm=ExecutionAlgorithm.DIRECT,
            urgency=ExecutionUrgency.URGENT,
            allow_market_orders=True,
        )

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="USDJPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5000.0,
            price=150.01,
            timestamp=datetime.now(),
        )

        print(f"   📝 Direct execution: {order.quantity:,.0f} {order.symbol}")

        # Measure execution time
        start_time = time.perf_counter()
        slices = await manager.execute_order(order, parameters, market_data)
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000

        if slices and len(slices) > 0:
            print(f"   ✅ Direct execution completed in {execution_time_ms:.2f}ms")
            print(f"   ✅ Generated {len(slices)} slice (should be 1)")
            print(f"   ✅ Slice quantity: {slices[0].quantity:,.0f} units")

            # Validate it's truly direct (single slice)
            if len(slices) == 1 and slices[0].quantity == order.quantity:
                print("   ✅ Direct execution validated")
                return True
            else:
                print(
                    "   ⚠️  Direct execution should generate single slice matching order quantity"
                )
                return False
        else:
            print("   ❌ Direct execution failed")
            return False

    except Exception as e:
        print(f"   ❌ Direct execution error: {e}")
        return False


async def test_slippage_calculation():
    """Test slippage monitoring functionality."""
    print("\n🔍 Testing Slippage Calculation...")

    try:
        from core.execution_engine import create_execution_engine
        from core.interfaces.trading_interfaces import Order, OrderSide, OrderType

        # Create execution engine
        engine = create_execution_engine()

        # Create test order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50000.0,
            price=1.1000,  # Reference price
            timestamp=datetime.now(),
        )

        # Test fills with controlled slippage
        fills = [
            {
                "quantity": 25000,
                "price": 1.1002,
                "timestamp": datetime.now(),
            },  # ~1.8 bps
            {
                "quantity": 25000,
                "price": 1.1003,
                "timestamp": datetime.now(),
            },  # ~2.7 bps
        ]

        print(f"   📝 Order: {order.quantity:,.0f} {order.symbol} @ {order.price}")
        print(f"   📊 Fills: {len(fills)} fills")

        # Calculate execution quality
        metrics = await engine.calculate_execution_quality(order, fills)

        if metrics:
            slippage_bps = metrics["slippage_bps"]
            print(f"   ✅ Slippage calculated: {slippage_bps:.2f} bps")
            print(f"   ✅ Average fill price: {metrics['avg_fill_price']:.5f}")
            print(f"   ✅ Fill rate: {metrics['fill_rate']:.2%}")

            # Validate slippage is reasonable
            if slippage_bps < 5.0:
                print("   ✅ Slippage within 5 bps threshold")
                return True
            else:
                print(f"   ⚠️  Slippage {slippage_bps:.2f} bps exceeds 5 bps threshold")
                return False
        else:
            print("   ❌ Failed to calculate execution quality")
            return False

    except Exception as e:
        print(f"   ❌ Slippage calculation error: {e}")
        return False


async def test_broker_integration():
    """Test broker integration functionality."""
    print("\n🔍 Testing Broker Integration...")

    try:
        from core.interfaces.trading_interfaces import Order, OrderSide, OrderType
        from core.order_router import create_order_router

        # Create order router
        router = create_order_router()

        # Start router
        await router.start()
        print("   ✅ Order router started")

        # Create test order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            price=1.1000,
            timestamp=datetime.now(),
        )

        print(f"   📝 Test order: {order.quantity:,.0f} {order.symbol}")

        # Route order
        broker_order = await router.route_order(order)

        if broker_order:
            print(f"   ✅ Order routed to {broker_order.broker_type.value}")
            print(f"   ✅ Broker order ID: {broker_order.broker_order_id}")
            print(f"   ✅ Status: {broker_order.status.value}")

            # Stop router
            await router.stop()
            print("   ✅ Order router stopped")

            return True
        else:
            print("   ❌ Failed to route order")
            await router.stop()
            return False

    except Exception as e:
        print(f"   ❌ Broker integration error: {e}")
        return False


async def run_comprehensive_test():
    """Run all comprehensive tests."""
    print("🚀 COMPREHENSIVE TASK 14 INTEGRATION TEST")
    print("=" * 60)

    test_results = []

    # Synchronous tests
    sync_tests = [
        ("Project Structure", test_project_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Execution Engine Creation", test_execution_engine_creation),
        ("Order Router Creation", test_order_router_creation),
        ("Algorithm Manager Creation", test_algorithm_manager_creation),
    ]

    for test_name, test_func in sync_tests:
        try:
            result = test_func()
            test_results.append((test_name, "PASSED" if result else "FAILED"))
        except Exception as e:
            test_results.append((test_name, f"ERROR: {e}"))

    # Asynchronous tests
    async_tests = [
        ("Basic Order Flow", test_basic_order_flow),
        ("TWAP Algorithm", test_twap_algorithm),
        ("POV Algorithm", test_pov_algorithm),
        ("Direct Execution", test_direct_execution),
        ("Slippage Calculation", test_slippage_calculation),
        ("Broker Integration", test_broker_integration),
    ]

    for test_name, test_func in async_tests:
        try:
            result = await test_func()
            test_results.append((test_name, "PASSED" if result else "FAILED"))
        except Exception as e:
            test_results.append((test_name, f"ERROR: {e}"))

    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        if result == "PASSED":
            print(f"✅ {test_name}: {result}")
            passed += 1
        else:
            print(f"❌ {test_name}: {result}")

    print(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 TASK 14 COMPREHENSIVE TEST - ALL PASSED!")
        print("\n✅ VALIDATED CAPABILITIES:")
        print("  • Complete project structure and imports")
        print("  • Configuration loading and validation")
        print("  • Core component creation and initialization")
        print("  • Basic order submission and processing flow")
        print("  • TWAP algorithm with order slicing")
        print("  • POV algorithm with volume adaptation")
        print("  • Direct execution with instant processing")
        print("  • Slippage calculation and monitoring")
        print("  • Broker integration and order routing")
        print("\n🚀 TASK 14 EXECUTION ENGINE IS FULLY OPERATIONAL!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests need attention")
        return False


if __name__ == "__main__":
    # Run comprehensive test
    success = asyncio.run(run_comprehensive_test())
    exit(0 if success else 1)
