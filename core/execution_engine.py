"""
Execution Engine - Order Management System (OMS) for FX AI-Quant Trading System.

This module implements a robust execution engine that handles trade instructions,
manages order states, implements execution algorithms (TWAP, POV, Direct),
and monitors execution quality with slippage control.

Features:
- Order lifecycle management (pending, filled, partial, canceled, failed)
- Execution algorithms: TWAP, POV, Direct execution
- Order slicing for large orders
- Slippage monitoring and control
- Execution quality metrics
- Real-time order status updates
- Integration with risk manager and broker interfaces
"""

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


import numpy as np
import structlog
import yaml

from core.interfaces.messaging_interfaces import Message, Topics
from core.interfaces.trading_interfaces import (
    ExecutionEngine as ExecutionEngineInterface,
)
from core.interfaces.trading_interfaces import Order
from core.interfaces.trading_interfaces import OrderManager as OrderManagerInterface
from core.interfaces.trading_interfaces import (
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from core.pubsub import ZMQPublisher


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    DIRECT = "direct"
    TWAP = "twap"
    POV = "pov"
    VWAP = "vwap"


class OrderPriority(Enum):
    """Order priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class SliceStatus(Enum):
    """Order slice status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionConstraints:
    """Execution constraints for orders."""

    max_participation_rate: float = 0.20  # Max 20% of volume
    time_limit: timedelta | None = None
    urgency: OrderPriority = OrderPriority.NORMAL
    min_fill_size: float = 0.0
    max_slice_size: float = float("inf")
    allow_partial_fills: bool = True
    price_tolerance: float = 0.0005  # 5 bps price tolerance


@dataclass
class OrderSlice:
    """Individual order slice for execution."""

    slice_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float | None
    order_type: OrderType
    status: SliceStatus = SliceStatus.PENDING
    submitted_time: datetime | None = None
    filled_time: datetime | None = None
    filled_quantity: float = 0.0
    filled_price: float | None = None
    broker_order_id: str | None = None
    slippage: float = 0.0
    commission: float = 0.0


@dataclass
class ExecutionResult:
    """Execution result for an order."""

    order_id: str
    symbol: str
    status: OrderStatus
    total_quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_slippage: float
    execution_time: float
    algorithm_used: ExecutionAlgorithm
    slices_count: int
    commission: float
    execution_quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Execution quality metrics."""

    timestamp: datetime
    symbol: str
    order_id: str
    target_price: float
    avg_fill_price: float
    slippage_bps: float
    execution_time_ms: float
    fill_rate: float
    participation_rate: float
    market_impact_bps: float
    timing_risk_bps: float
    implementation_shortfall_bps: float


class ExecutionEngineConfig:
    """Configuration for execution engine."""

    def __init__(self, config_path: str = "config/execution_settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Execution algorithms
        algo_config = self.config["execution_algorithms"]
        self.twap_slice_interval = algo_config["twap"]["slice_interval_seconds"]
        self.twap_max_slices = algo_config["twap"]["max_slices"]
        self.pov_target_participation = algo_config["pov"]["target_participation_rate"]
        self.pov_max_participation = algo_config["pov"]["max_participation_rate"]
        self.direct_size_threshold = algo_config["direct"]["size_threshold"]

        # Order management
        order_config = self.config["order_management"]
        self.max_order_age_hours = order_config["max_order_age_hours"]
        self.retry_attempts = order_config["retry_attempts"]
        self.retry_delay_seconds = order_config["retry_delay_seconds"]
        self.partial_fill_timeout = order_config["partial_fill_timeout_seconds"]

        # Slippage control
        slippage_config = self.config["slippage_control"]
        self.max_slippage_bps = slippage_config["max_slippage_bps"]
        self.slippage_warning_bps = slippage_config["warning_threshold_bps"]
        self.price_improvement_threshold = slippage_config[
            "price_improvement_threshold_bps"
        ]

        # Quality monitoring
        quality_config = self.config["quality_monitoring"]
        self.track_execution_metrics = quality_config["track_execution_metrics"]
        self.metrics_retention_days = quality_config["metrics_retention_days"]
        self.benchmark_against_arrival = quality_config[
            "benchmark_against_arrival_price"
        ]

        # Risk controls
        risk_config = self.config["risk_controls"]
        self.enable_pre_trade_checks = risk_config["enable_pre_trade_checks"]
        self.max_order_value = risk_config["max_order_value"]
        self.position_limit_check = risk_config["position_limit_check"]


class CoreExecutionEngine(ExecutionEngineInterface, OrderManagerInterface):
    """
    Core Execution Engine implementation with comprehensive order management.

    Features:
    - Multiple execution algorithms (TWAP, POV, Direct)
    - Order slicing and lifecycle management
    - Real-time execution quality monitoring
    - Slippage control and optimization
    - Integration with risk management
    - Broker abstraction layer
    """

    def __init__(
        self,
        config: ExecutionEngineConfig | None = None,
        publisher: ZMQPublisher | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config or ExecutionEngineConfig()
        self.publisher = publisher
        self.logger = logger or structlog.get_logger(__name__)

        # Order tracking
        self.active_orders: dict[str, Order] = {}
        self.order_slices: dict[str, list[OrderSlice]] = {}
        self.execution_results: dict[str, ExecutionResult] = {}
        self.order_history: list[Order] = []

        # Execution metrics
        self.execution_metrics: deque = deque(maxlen=10000)
        self.slippage_history: deque = deque(maxlen=1000)
        self.execution_times: deque = deque(maxlen=1000)

        # Algorithm state
        self.twap_schedules: dict[str, list[datetime]] = {}
        self.pov_monitors: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self.orders_processed = 0
        self.orders_filled = 0
        self.orders_cancelled = 0
        self.orders_failed = 0
        self.total_slippage = 0.0
        self.avg_execution_time = 0.0

        # Broker interface (would be injected in real implementation)
        self.broker_interface = None

        self.logger.info(
            "CoreExecutionEngine initialized",
            max_slippage_bps=self.config.max_slippage_bps,
            twap_slice_interval=self.config.twap_slice_interval,
            pov_target_participation=self.config.pov_target_participation,
        )

    async def submit_order(self, order: Order) -> str:
        """Submit an order for execution."""
        try:
            self.orders_processed += 1

            # Validate order
            if not await self._validate_order(order):
                order.status = OrderStatus.REJECTED
                await self._publish_order_update(order)
                return order.order_id

            # Store order
            self.active_orders[order.order_id] = order

            # Determine execution algorithm
            algorithm = await self._select_execution_algorithm(order)

            # Execute based on algorithm
            if algorithm == ExecutionAlgorithm.DIRECT:
                await self._execute_direct(order)
            elif algorithm == ExecutionAlgorithm.TWAP:
                await self._execute_twap(order)
            elif algorithm == ExecutionAlgorithm.POV:
                await self._execute_pov(order)
            else:
                await self._execute_direct(order)  # Fallback

            await self._publish_order_update(order)

            self.logger.info(
                "Order submitted",
                order_id=order.order_id,
                symbol=order.symbol,
                algorithm=algorithm.value,
                quantity=order.quantity,
            )

            return order.order_id

        except Exception as e:
            self.logger.error(
                "Error submitting order", order_id=order.order_id, error=str(e)
            )
            order.status = OrderStatus.REJECTED
            await self._publish_order_update(order)
            return order.order_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if order_id not in self.active_orders:
                self.logger.warning(
                    "Order not found for cancellation", order_id=order_id
                )
                return False

            order = self.active_orders[order_id]

            # Cancel all slices
            if order_id in self.order_slices:
                for slice_order in self.order_slices[order_id]:
                    if slice_order.status in [
                        SliceStatus.PENDING,
                        SliceStatus.SUBMITTED,
                    ]:
                        slice_order.status = SliceStatus.CANCELLED
                        # Would cancel with broker here

            order.status = OrderStatus.CANCELLED
            self.orders_cancelled += 1

            # Move to history
            self.order_history.append(order)
            del self.active_orders[order_id]

            await self._publish_order_update(order)

            self.logger.info("Order cancelled", order_id=order_id)
            return True

        except Exception as e:
            self.logger.error("Error cancelling order", order_id=order_id, error=str(e))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id in self.active_orders:
            return self.active_orders[order_id].status

        # Check history
        for order in self.order_history:
            if order.order_id == order_id:
                return order.status

        return OrderStatus.REJECTED  # Not found

    async def get_positions(self) -> dict[str, Position]:
        """Get current positions (would interface with broker)."""
        # This would interface with the broker to get actual positions
        # For now, return empty dict as placeholder
        return {}

    async def get_account_balance(self) -> float:
        """Get account balance (would interface with broker)."""
        # This would interface with the broker to get actual balance
        # For now, return placeholder value
        return 100000.0

    async def slice_order(
        self, order: Order, slice_size: float, time_interval: int
    ) -> list[Order]:
        """Slice a large order into smaller pieces."""
        try:
            slices = []
            remaining_quantity = order.quantity
            slice_count = 0

            while remaining_quantity > 0 and slice_count < self.config.twap_max_slices:
                current_slice_size = min(slice_size, remaining_quantity)

                slice_order = Order(
                    order_id=f"{order.order_id}_slice_{slice_count}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=current_slice_size,
                    price=order.price,
                    stop_price=order.stop_price,
                )

                slices.append(slice_order)
                remaining_quantity -= current_slice_size
                slice_count += 1

            self.logger.debug(
                "Order sliced",
                order_id=order.order_id,
                slice_count=len(slices),
                slice_size=slice_size,
            )

            return slices

        except Exception as e:
            self.logger.error(
                "Error slicing order", order_id=order.order_id, error=str(e)
            )
            return [order]  # Return original order if slicing fails

    async def manage_order_lifecycle(self, order: Order) -> None:
        """Manage the complete lifecycle of an order."""
        try:
            start_time = time.time()

            # Monitor order until completion
            while order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                # Check for timeout
                if time.time() - start_time > self.config.max_order_age_hours * 3600:
                    await self.cancel_order(order.order_id)
                    break

                # Update order status (would check with broker)
                await self._update_order_status(order)

                # Handle partial fills
                if order.status == OrderStatus.PARTIALLY_FILLED:
                    await self._handle_partial_fill(order)

                await asyncio.sleep(1)  # Check every second

            # Calculate final execution metrics
            if order.status == OrderStatus.FILLED:
                await self._calculate_execution_metrics(order)
                self.orders_filled += 1
            elif order.status == OrderStatus.CANCELLED:
                self.orders_cancelled += 1
            else:
                self.orders_failed += 1

            # Move to history
            if order.order_id in self.active_orders:
                self.order_history.append(order)
                del self.active_orders[order.order_id]

        except Exception as e:
            self.logger.error(
                "Error managing order lifecycle", order_id=order.order_id, error=str(e)
            )

    async def calculate_execution_quality(
        self, order: Order, fills: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate execution quality metrics."""
        try:
            if not fills:
                return {}

            # Calculate metrics
            total_quantity = sum(fill["quantity"] for fill in fills)
            weighted_price = (
                sum(fill["price"] * fill["quantity"] for fill in fills) / total_quantity
            )

            # Slippage calculation
            reference_price = order.price or fills[0]["price"]
            slippage_bps = self._calculate_slippage_bps(
                reference_price, weighted_price, order.side
            )

            # Timing metrics
            min(fill["timestamp"] for fill in fills)
            last_fill_time = max(fill["timestamp"] for fill in fills)
            execution_time_ms = (
                last_fill_time - order.timestamp
            ).total_seconds() * 1000

            # Fill rate
            fill_rate = total_quantity / order.quantity

            # Market impact (simplified)
            market_impact_bps = abs(slippage_bps) * 0.5  # Simplified calculation

            metrics = {
                "avg_fill_price": weighted_price,
                "slippage_bps": slippage_bps,
                "execution_time_ms": execution_time_ms,
                "fill_rate": fill_rate,
                "market_impact_bps": market_impact_bps,
                "total_quantity": total_quantity,
                "fills_count": len(fills),
            }

            # Store metrics
            execution_metric = ExecutionMetrics(
                timestamp=datetime.now(),
                symbol=order.symbol,
                order_id=order.order_id,
                target_price=reference_price,
                avg_fill_price=weighted_price,
                slippage_bps=slippage_bps,
                execution_time_ms=execution_time_ms,
                fill_rate=fill_rate,
                participation_rate=0.0,  # Would calculate from market data
                market_impact_bps=market_impact_bps,
                timing_risk_bps=0.0,  # Would calculate from price movement
                implementation_shortfall_bps=slippage_bps + market_impact_bps,
            )

            self.execution_metrics.append(execution_metric)

            return metrics

        except Exception as e:
            self.logger.error(
                "Error calculating execution quality",
                order_id=order.order_id,
                error=str(e),
            )
            return {}

    async def get_slippage(self, order: Order, reference_price: float) -> float:
        """Calculate slippage for an order."""
        if order.avg_fill_price == 0:
            return 0.0

        return self._calculate_slippage_bps(
            reference_price, order.avg_fill_price, order.side
        )

    # Private helper methods

    async def _validate_order(self, order: Order) -> bool:
        """Validate order before execution."""
        try:
            # Basic validation
            if order.quantity <= 0:
                self.logger.warning(
                    "Invalid order quantity",
                    order_id=order.order_id,
                    quantity=order.quantity,
                )
                return False

            if not order.symbol:
                self.logger.warning("Missing symbol", order_id=order.order_id)
                return False

            # Value check
            order_value = order.quantity * (order.price or 1.0)
            if order_value > self.config.max_order_value:
                self.logger.warning(
                    "Order value exceeds limit",
                    order_id=order.order_id,
                    value=order_value,
                )
                return False

            return True

        except Exception as e:
            self.logger.error(
                "Error validating order", order_id=order.order_id, error=str(e)
            )
            return False

    async def _select_execution_algorithm(self, order: Order) -> ExecutionAlgorithm:
        """Select appropriate execution algorithm."""
        try:
            order_value = order.quantity * (order.price or 1.0)

            # Direct execution for small orders
            if order_value < self.config.direct_size_threshold:
                return ExecutionAlgorithm.DIRECT

            # TWAP for medium orders
            if order_value < self.config.direct_size_threshold * 10:
                return ExecutionAlgorithm.TWAP

            # POV for large orders
            return ExecutionAlgorithm.POV

        except Exception as e:
            self.logger.error(
                "Error selecting algorithm", order_id=order.order_id, error=str(e)
            )
            return ExecutionAlgorithm.DIRECT

    async def _execute_direct(self, order: Order) -> None:
        """Execute order directly without slicing."""
        try:
            # Simulate direct execution
            order.status = OrderStatus.PENDING

            # Would submit to broker here
            await asyncio.sleep(0.1)  # Simulate network latency

            # Simulate fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = order.price or 1.1000  # Would come from broker

            # Calculate slippage
            if order.price:
                slippage = self._calculate_slippage_bps(
                    order.price, order.avg_fill_price, order.side
                )
                self.slippage_history.append(slippage)
                self.total_slippage += abs(slippage)

            self.logger.debug("Direct execution completed", order_id=order.order_id)

        except Exception as e:
            self.logger.error(
                "Error in direct execution", order_id=order.order_id, error=str(e)
            )
            order.status = OrderStatus.REJECTED

    async def _execute_twap(self, order: Order) -> None:
        """Execute order using Time Weighted Average Price algorithm."""
        try:
            # Calculate slice parameters
            slice_size = order.quantity / self.config.twap_max_slices
            slice_interval = self.config.twap_slice_interval

            # Create slices
            slices = await self.slice_order(order, slice_size, slice_interval)

            # Store slices
            order_slices = []
            for i, slice_order in enumerate(slices):
                slice_obj = OrderSlice(
                    slice_id=slice_order.order_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_order.quantity,
                    price=slice_order.price,
                    order_type=slice_order.order_type,
                )
                order_slices.append(slice_obj)

            self.order_slices[order.order_id] = order_slices

            # Execute slices over time
            order.status = OrderStatus.PENDING
            total_filled = 0.0
            total_value = 0.0

            for i, slice_obj in enumerate(order_slices):
                # Wait for slice interval
                if i > 0:
                    await asyncio.sleep(slice_interval)

                # Execute slice
                slice_obj.status = SliceStatus.SUBMITTED
                slice_obj.submitted_time = datetime.now()

                # Simulate execution
                await asyncio.sleep(0.05)  # Simulate execution time

                slice_obj.status = SliceStatus.FILLED
                slice_obj.filled_time = datetime.now()
                slice_obj.filled_quantity = slice_obj.quantity
                slice_obj.filled_price = (
                    slice_obj.price or 1.1000
                )  # Would come from broker

                total_filled += slice_obj.filled_quantity
                total_value += slice_obj.filled_quantity * slice_obj.filled_price

            # Update parent order
            order.filled_quantity = total_filled
            order.avg_fill_price = total_value / total_filled if total_filled > 0 else 0
            order.status = (
                OrderStatus.FILLED
                if total_filled == order.quantity
                else OrderStatus.PARTIALLY_FILLED
            )

            self.logger.debug(
                "TWAP execution completed",
                order_id=order.order_id,
                slices=len(order_slices),
                filled_quantity=total_filled,
            )

        except Exception as e:
            self.logger.error(
                "Error in TWAP execution", order_id=order.order_id, error=str(e)
            )
            order.status = OrderStatus.REJECTED

    async def _execute_pov(self, order: Order) -> None:
        """Execute order using Percent of Volume algorithm."""
        try:
            # POV execution would monitor market volume and adjust slice sizes
            # For simulation, we'll use adaptive slicing

            target_participation = self.config.pov_target_participation
            self.config.pov_max_participation

            order.status = OrderStatus.PENDING
            remaining_quantity = order.quantity
            total_filled = 0.0
            total_value = 0.0

            while remaining_quantity > 0:
                # Simulate market volume (would come from market data)
                market_volume = np.random.uniform(10000, 50000)

                # Calculate slice size based on participation rate
                slice_size = min(
                    remaining_quantity,
                    market_volume * target_participation,
                    remaining_quantity * 0.2,  # Max 20% per slice
                )

                if slice_size < order.quantity * 0.01:  # Min 1% of order
                    slice_size = min(remaining_quantity, order.quantity * 0.01)

                # Create and execute slice
                slice_obj = OrderSlice(
                    slice_id=f"{order.order_id}_pov_{len(self.order_slices.get(order.order_id, []))}",
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    price=order.price,
                    order_type=order.order_type,
                    status=SliceStatus.SUBMITTED,
                    submitted_time=datetime.now(),
                )

                # Simulate execution
                await asyncio.sleep(0.1)

                slice_obj.status = SliceStatus.FILLED
                slice_obj.filled_time = datetime.now()
                slice_obj.filled_quantity = slice_size
                slice_obj.filled_price = order.price or 1.1000

                # Update totals
                total_filled += slice_size
                total_value += slice_size * slice_obj.filled_price
                remaining_quantity -= slice_size

                # Store slice
                if order.order_id not in self.order_slices:
                    self.order_slices[order.order_id] = []
                self.order_slices[order.order_id].append(slice_obj)

                # Adaptive delay based on market conditions
                await asyncio.sleep(np.random.uniform(1, 5))

            # Update parent order
            order.filled_quantity = total_filled
            order.avg_fill_price = total_value / total_filled if total_filled > 0 else 0
            order.status = OrderStatus.FILLED

            self.logger.debug(
                "POV execution completed",
                order_id=order.order_id,
                slices=len(self.order_slices[order.order_id]),
                avg_participation=target_participation,
            )

        except Exception as e:
            self.logger.error(
                "Error in POV execution", order_id=order.order_id, error=str(e)
            )
            order.status = OrderStatus.REJECTED

    async def _update_order_status(self, order: Order) -> None:
        """Update order status from broker."""
        # This would query the broker for actual order status
        # For simulation, we'll just maintain current status

    async def _handle_partial_fill(self, order: Order) -> None:
        """Handle partial fill scenarios."""
        try:
            # Check if we should continue or cancel remaining
            if order.filled_quantity / order.quantity > 0.9:  # 90% filled
                # Accept partial fill
                order.status = OrderStatus.FILLED
                self.logger.info(
                    "Accepting partial fill as complete",
                    order_id=order.order_id,
                    fill_rate=order.filled_quantity / order.quantity,
                )

        except Exception as e:
            self.logger.error(
                "Error handling partial fill", order_id=order.order_id, error=str(e)
            )

    async def _calculate_execution_metrics(self, order: Order) -> None:
        """Calculate and store execution metrics."""
        try:
            if order.price and order.avg_fill_price:
                slippage = self._calculate_slippage_bps(
                    order.price, order.avg_fill_price, order.side
                )
                execution_time = (
                    datetime.now() - order.timestamp
                ).total_seconds() * 1000

                # Update running averages
                self.slippage_history.append(slippage)
                self.execution_times.append(execution_time)

                if len(self.execution_times) > 0:
                    self.avg_execution_time = statistics.mean(self.execution_times)

                self.logger.debug(
                    "Execution metrics calculated",
                    order_id=order.order_id,
                    slippage_bps=slippage,
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            self.logger.error(
                "Error calculating execution metrics",
                order_id=order.order_id,
                error=str(e),
            )

    def _calculate_slippage_bps(
        self, reference_price: float, fill_price: float, side: OrderSide
    ) -> float:
        """Calculate slippage in basis points."""
        if reference_price == 0:
            return 0.0

        if side == OrderSide.BUY:
            # For buy orders, positive slippage means paying more
            slippage = (fill_price - reference_price) / reference_price * 10000
        else:
            # For sell orders, positive slippage means receiving less
            slippage = (reference_price - fill_price) / reference_price * 10000

        return slippage

    async def _publish_order_update(self, order: Order) -> None:
        """Publish order status update."""
        try:
            if self.publisher:
                update_data = {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity,
                    "avg_fill_price": order.avg_fill_price,
                    "timestamp": datetime.now().isoformat(),
                }

                message = Message(
                    topic=Topics.ORDERS,
                    data=update_data,
                    timestamp=datetime.now(),
                    source="execution_engine",
                )

                # Would publish message here
                # await self.publisher.publish_message(message)

        except Exception as e:
            self.logger.error(
                "Error publishing order update", order_id=order.order_id, error=str(e)
            )

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get execution engine statistics."""
        return {
            "orders_processed": self.orders_processed,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "orders_failed": self.orders_failed,
            "fill_rate": self.orders_filled / max(self.orders_processed, 1),
            "avg_slippage_bps": statistics.mean(self.slippage_history)
            if self.slippage_history
            else 0.0,
            "avg_execution_time_ms": self.avg_execution_time,
            "active_orders_count": len(self.active_orders),
            "total_slices": sum(len(slices) for slices in self.order_slices.values()),
        }

    def get_order_details(self, order_id: str) -> dict[str, Any] | None:
        """Get detailed information about an order."""
        order = self.active_orders.get(order_id)
        if not order:
            # Check history
            for hist_order in self.order_history:
                if hist_order.order_id == order_id:
                    order = hist_order
                    break

        if not order:
            return None

        details = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
            "timestamp": order.timestamp.isoformat(),
            "slices": [],
        }

        # Add slice information
        if order_id in self.order_slices:
            for slice_obj in self.order_slices[order_id]:
                slice_details = {
                    "slice_id": slice_obj.slice_id,
                    "quantity": slice_obj.quantity,
                    "status": slice_obj.status.value,
                    "filled_quantity": slice_obj.filled_quantity,
                    "filled_price": slice_obj.filled_price,
                    "submitted_time": slice_obj.submitted_time.isoformat()
                    if slice_obj.submitted_time
                    else None,
                    "filled_time": slice_obj.filled_time.isoformat()
                    if slice_obj.filled_time
                    else None,
                }
                details["slices"].append(slice_details)

        return details


def create_execution_engine(
    config_path: str = "config/execution_settings.yaml",
    publisher: ZMQPublisher | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> CoreExecutionEngine:
    """
    Factory function to create a configured execution engine.

    Args:
        config_path: Path to execution settings configuration file
        publisher: Optional message publisher for order updates
        logger: Optional logger instance

    Returns:
        Configured CoreExecutionEngine instance
    """
    config = ExecutionEngineConfig(config_path)
    return CoreExecutionEngine(config=config, publisher=publisher, logger=logger)
