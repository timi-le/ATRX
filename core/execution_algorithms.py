"""
Execution Algorithms - Advanced Order Execution Strategies for FX AI-Quant Trading System.

This module implements sophisticated execution algorithms including TWAP, POV, VWAP,
and Direct execution with advanced order slicing, market impact optimization,
and adaptive execution strategies.

Features:
- Time Weighted Average Price (TWAP) with adaptive slicing
- Percent of Volume (POV) with dynamic participation rates
- Volume Weighted Average Price (VWAP) with historical volume patterns
- Direct execution with smart order routing
- Market impact modeling and optimization
- Adaptive execution based on market conditions
- Real-time execution quality monitoring
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


import structlog

from core.execution_engine import ExecutionAlgorithm, OrderSlice, SliceStatus
from core.interfaces.trading_interfaces import (
    Order,
    OrderSide,
    OrderType,
)


class MarketCondition(Enum):
    """Market condition types."""

    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    TRENDING = "trending"
    RANGING = "ranging"


class ExecutionUrgency(Enum):
    """Execution urgency levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class MarketData:
    """Market data snapshot."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    spread_bps: float
    volatility: float
    timestamp: datetime
    depth: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )  # price, size


@dataclass
class VolumeProfile:
    """Historical volume profile."""

    symbol: str
    interval_minutes: int
    historical_volumes: list[float]
    avg_volume: float
    volume_std: float
    peak_hours: list[int]
    quiet_hours: list[int]
    last_update: datetime


@dataclass
class ExecutionParameters:
    """Execution algorithm parameters."""

    algorithm: ExecutionAlgorithm
    urgency: ExecutionUrgency = ExecutionUrgency.NORMAL
    max_participation_rate: float = 0.20
    min_participation_rate: float = 0.05
    target_completion_time: timedelta | None = None
    max_slice_size: float | None = None
    min_slice_size: float | None = None
    price_tolerance_bps: float = 5.0
    allow_market_orders: bool = True
    adaptive_sizing: bool = True
    market_impact_limit_bps: float = 10.0


@dataclass
class ExecutionState:
    """Current execution state."""

    order_id: str
    algorithm: ExecutionAlgorithm
    total_quantity: float
    remaining_quantity: float
    executed_quantity: float
    avg_execution_price: float
    start_time: datetime
    target_completion_time: datetime | None
    slices_executed: int
    total_slices: int
    current_participation_rate: float
    market_impact_bps: float
    execution_quality_score: float
    last_slice_time: datetime | None = None


class ExecutionAlgorithmBase(ABC):
    """Abstract base class for execution algorithms."""

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or structlog.get_logger(__name__)
        self.execution_states: dict[str, ExecutionState] = {}
        self.market_data_cache: dict[str, MarketData] = {}
        self.volume_profiles: dict[str, VolumeProfile] = {}

    @abstractmethod
    async def execute_order(
        self, order: Order, parameters: ExecutionParameters, market_data: MarketData
    ) -> list[OrderSlice]:
        """Execute order using the specific algorithm."""

    @abstractmethod
    async def update_execution(
        self, order_id: str, market_data: MarketData, filled_slices: list[OrderSlice]
    ) -> list[OrderSlice]:
        """Update ongoing execution based on market conditions."""

    @abstractmethod
    def calculate_next_slice(
        self,
        execution_state: ExecutionState,
        market_data: MarketData,
        parameters: ExecutionParameters,
    ) -> OrderSlice | None:
        """Calculate the next order slice."""

    def get_execution_state(self, order_id: str) -> ExecutionState | None:
        """Get execution state for an order."""
        return self.execution_states.get(order_id)

    def update_market_data(self, market_data: MarketData) -> None:
        """Update market data cache."""
        self.market_data_cache[market_data.symbol] = market_data

    def update_volume_profile(self, volume_profile: VolumeProfile) -> None:
        """Update volume profile cache."""
        self.volume_profiles[volume_profile.symbol] = volume_profile


class TWAPAlgorithm(ExecutionAlgorithmBase):
    """
    Time Weighted Average Price (TWAP) execution algorithm.

    Features:
    - Uniform time-based slicing
    - Adaptive slice sizing based on market conditions
    - Market impact optimization
    - Completion time targeting
    """

    async def execute_order(
        self, order: Order, parameters: ExecutionParameters, market_data: MarketData
    ) -> list[OrderSlice]:
        """Execute order using TWAP algorithm."""
        try:
            # Calculate execution timeline
            completion_time = parameters.target_completion_time or timedelta(hours=1)
            slice_interval = self._calculate_slice_interval(
                order.quantity, completion_time, market_data, parameters
            )

            # Initialize execution state
            execution_state = ExecutionState(
                order_id=order.order_id,
                algorithm=ExecutionAlgorithm.TWAP,
                total_quantity=order.quantity,
                remaining_quantity=order.quantity,
                executed_quantity=0.0,
                avg_execution_price=0.0,
                start_time=datetime.now(),
                target_completion_time=datetime.now() + completion_time,
                slices_executed=0,
                total_slices=int(
                    completion_time.total_seconds() / slice_interval.total_seconds()
                ),
                current_participation_rate=0.0,
                market_impact_bps=0.0,
                execution_quality_score=0.0,
            )

            self.execution_states[order.order_id] = execution_state

            # Generate initial slice
            first_slice = self.calculate_next_slice(
                execution_state, market_data, parameters
            )

            self.logger.info(
                "TWAP execution started",
                order_id=order.order_id,
                total_slices=execution_state.total_slices,
                slice_interval_seconds=slice_interval.total_seconds(),
                completion_time=completion_time.total_seconds(),
            )

            return [first_slice] if first_slice else []

        except Exception as e:
            self.logger.error(
                "Error in TWAP execution", order_id=order.order_id, error=str(e)
            )
            return []

    async def update_execution(
        self, order_id: str, market_data: MarketData, filled_slices: list[OrderSlice]
    ) -> list[OrderSlice]:
        """Update TWAP execution."""
        try:
            execution_state = self.execution_states.get(order_id)
            if not execution_state:
                return []

            # Update execution state with filled slices
            for slice_obj in filled_slices:
                if slice_obj.status == SliceStatus.FILLED:
                    execution_state.executed_quantity += slice_obj.filled_quantity
                    execution_state.remaining_quantity -= slice_obj.filled_quantity
                    execution_state.slices_executed += 1

                    # Update average execution price
                    if execution_state.executed_quantity > 0:
                        total_value = (
                            execution_state.avg_execution_price
                            * (
                                execution_state.executed_quantity
                                - slice_obj.filled_quantity
                            )
                            + slice_obj.filled_price * slice_obj.filled_quantity
                        )
                        execution_state.avg_execution_price = (
                            total_value / execution_state.executed_quantity
                        )

            # Check if execution is complete
            if execution_state.remaining_quantity <= 0:
                self.logger.info("TWAP execution completed", order_id=order_id)
                return []

            # Check if we're behind schedule
            now = datetime.now()
            time_elapsed = (now - execution_state.start_time).total_seconds()
            target_time = (
                execution_state.target_completion_time - execution_state.start_time
            ).total_seconds()
            progress_ratio = time_elapsed / target_time if target_time > 0 else 1.0

            expected_executed = execution_state.total_quantity * progress_ratio
            execution_deficit = expected_executed - execution_state.executed_quantity

            # Generate next slice with adaptive sizing
            next_slice = self._calculate_adaptive_slice(
                execution_state, market_data, execution_deficit
            )

            return [next_slice] if next_slice else []

        except Exception as e:
            self.logger.error(
                "Error updating TWAP execution", order_id=order_id, error=str(e)
            )
            return []

    def calculate_next_slice(
        self,
        execution_state: ExecutionState,
        market_data: MarketData,
        parameters: ExecutionParameters,
    ) -> OrderSlice | None:
        """Calculate next TWAP slice."""
        try:
            if execution_state.remaining_quantity <= 0:
                return None

            # Calculate base slice size
            remaining_time = (
                execution_state.target_completion_time - datetime.now()
            ).total_seconds()
            remaining_slices = max(
                1, int(remaining_time / 60)
            )  # Assume 1-minute intervals

            base_slice_size = execution_state.remaining_quantity / remaining_slices

            # Apply market condition adjustments
            market_condition = self._assess_market_condition(market_data)
            slice_size = self._adjust_slice_for_market_condition(
                base_slice_size, market_condition, parameters
            )

            # Ensure slice size limits
            slice_size = max(
                parameters.min_slice_size or execution_state.total_quantity * 0.01,
                min(
                    slice_size,
                    parameters.max_slice_size or execution_state.total_quantity * 0.1,
                    execution_state.remaining_quantity,
                ),
            )

            # Create slice
            slice_obj = OrderSlice(
                slice_id=f"{execution_state.order_id}_twap_{execution_state.slices_executed + 1}",
                parent_order_id=execution_state.order_id,
                symbol=market_data.symbol,
                side=OrderSide.BUY,  # Would be determined from original order
                quantity=slice_size,
                price=self._calculate_slice_price(market_data, parameters),
                order_type=OrderType.LIMIT,
                status=SliceStatus.PENDING,
            )

            return slice_obj

        except Exception as e:
            self.logger.error("Error calculating TWAP slice", error=str(e))
            return None

    def _calculate_slice_interval(
        self,
        total_quantity: float,
        completion_time: timedelta,
        market_data: MarketData,
        parameters: ExecutionParameters,
    ) -> timedelta:
        """Calculate optimal slice interval."""
        # Base interval calculation
        base_interval_minutes = max(
            1, completion_time.total_seconds() / 3600 * 10
        )  # 10 slices per hour

        # Adjust for market conditions
        if market_data.volatility > 0.02:  # High volatility
            base_interval_minutes *= 0.5  # More frequent slices
        elif market_data.spread_bps > 5:  # Wide spreads
            base_interval_minutes *= 1.5  # Less frequent slices

        return timedelta(minutes=max(1, base_interval_minutes))

    def _calculate_adaptive_slice(
        self,
        execution_state: ExecutionState,
        market_data: MarketData,
        execution_deficit: float,
    ) -> OrderSlice | None:
        """Calculate adaptive slice size based on execution progress."""
        try:
            # Base slice size
            base_size = execution_state.remaining_quantity / max(
                1,
                (
                    execution_state.target_completion_time - datetime.now()
                ).total_seconds()
                / 60,
            )

            # Adjust for execution deficit
            if execution_deficit > 0:
                # Behind schedule, increase slice size
                adjustment_factor = min(
                    2.0, 1.0 + execution_deficit / execution_state.total_quantity
                )
                adjusted_size = base_size * adjustment_factor
            else:
                # Ahead of schedule, maintain normal pace
                adjusted_size = base_size

            # Apply limits
            slice_size = max(
                execution_state.total_quantity * 0.005,  # Min 0.5% of total
                min(
                    adjusted_size,
                    execution_state.total_quantity * 0.1,  # Max 10% of total
                    execution_state.remaining_quantity,
                ),
            )

            return OrderSlice(
                slice_id=f"{execution_state.order_id}_twap_adaptive_{execution_state.slices_executed + 1}",
                parent_order_id=execution_state.order_id,
                symbol=market_data.symbol,
                side=OrderSide.BUY,  # Would be from original order
                quantity=slice_size,
                price=self._calculate_slice_price(
                    market_data, ExecutionParameters(ExecutionAlgorithm.TWAP)
                ),
                order_type=OrderType.LIMIT,
                status=SliceStatus.PENDING,
            )

        except Exception as e:
            self.logger.error("Error calculating adaptive TWAP slice", error=str(e))
            return None

    def _assess_market_condition(self, market_data: MarketData) -> MarketCondition:
        """Assess current market condition."""
        if market_data.volatility > 0.03:
            return MarketCondition.VOLATILE
        elif market_data.spread_bps > 10:
            return MarketCondition.ILLIQUID
        elif market_data.volume < 1000:
            return MarketCondition.ILLIQUID
        else:
            return MarketCondition.NORMAL

    def _adjust_slice_for_market_condition(
        self,
        base_slice_size: float,
        market_condition: MarketCondition,
        parameters: ExecutionParameters,
    ) -> float:
        """Adjust slice size based on market condition."""
        if market_condition == MarketCondition.VOLATILE:
            return base_slice_size * 0.7  # Smaller slices in volatile markets
        elif market_condition == MarketCondition.ILLIQUID:
            return base_slice_size * 0.5  # Much smaller slices in illiquid markets
        else:
            return base_slice_size

    def _calculate_slice_price(
        self, market_data: MarketData, parameters: ExecutionParameters
    ) -> float:
        """Calculate optimal slice price."""
        # Use mid-price with small improvement
        mid_price = (market_data.bid + market_data.ask) / 2

        # Add small price improvement to increase fill probability
        price_improvement_bps = min(
            parameters.price_tolerance_bps / 2, market_data.spread_bps / 4
        )
        price_improvement = mid_price * price_improvement_bps / 10000

        return mid_price + price_improvement


class POVAlgorithm(ExecutionAlgorithmBase):
    """
    Percent of Volume (POV) execution algorithm.

    Features:
    - Dynamic participation rate adjustment
    - Volume-based slice sizing
    - Market impact monitoring
    - Liquidity-aware execution
    """

    async def execute_order(
        self, order: Order, parameters: ExecutionParameters, market_data: MarketData
    ) -> list[OrderSlice]:
        """Execute order using POV algorithm."""
        try:
            # Initialize execution state
            execution_state = ExecutionState(
                order_id=order.order_id,
                algorithm=ExecutionAlgorithm.POV,
                total_quantity=order.quantity,
                remaining_quantity=order.quantity,
                executed_quantity=0.0,
                avg_execution_price=0.0,
                start_time=datetime.now(),
                target_completion_time=None,  # POV doesn't have fixed completion time
                slices_executed=0,
                total_slices=0,  # Dynamic
                current_participation_rate=parameters.max_participation_rate
                / 2,  # Start conservative
                market_impact_bps=0.0,
                execution_quality_score=0.0,
            )

            self.execution_states[order.order_id] = execution_state

            # Generate initial slice based on current volume
            first_slice = self.calculate_next_slice(
                execution_state, market_data, parameters
            )

            self.logger.info(
                "POV execution started",
                order_id=order.order_id,
                target_participation=parameters.max_participation_rate,
                initial_slice_size=first_slice.quantity if first_slice else 0,
            )

            return [first_slice] if first_slice else []

        except Exception as e:
            self.logger.error(
                "Error in POV execution", order_id=order.order_id, error=str(e)
            )
            return []

    async def update_execution(
        self, order_id: str, market_data: MarketData, filled_slices: list[OrderSlice]
    ) -> list[OrderSlice]:
        """Update POV execution."""
        try:
            execution_state = self.execution_states.get(order_id)
            if not execution_state:
                return []

            # Update execution state
            for slice_obj in filled_slices:
                if slice_obj.status == SliceStatus.FILLED:
                    execution_state.executed_quantity += slice_obj.filled_quantity
                    execution_state.remaining_quantity -= slice_obj.filled_quantity
                    execution_state.slices_executed += 1
                    execution_state.last_slice_time = datetime.now()

                    # Update average execution price
                    if execution_state.executed_quantity > 0:
                        total_value = (
                            execution_state.avg_execution_price
                            * (
                                execution_state.executed_quantity
                                - slice_obj.filled_quantity
                            )
                            + slice_obj.filled_price * slice_obj.filled_quantity
                        )
                        execution_state.avg_execution_price = (
                            total_value / execution_state.executed_quantity
                        )

            # Check if execution is complete
            if execution_state.remaining_quantity <= 0:
                self.logger.info("POV execution completed", order_id=order_id)
                return []

            # Adjust participation rate based on market impact
            self._adjust_participation_rate(execution_state, market_data)

            # Generate next slice
            next_slice = self.calculate_next_slice(
                execution_state,
                market_data,
                ExecutionParameters(ExecutionAlgorithm.POV),
            )

            return [next_slice] if next_slice else []

        except Exception as e:
            self.logger.error(
                "Error updating POV execution", order_id=order_id, error=str(e)
            )
            return []

    def calculate_next_slice(
        self,
        execution_state: ExecutionState,
        market_data: MarketData,
        parameters: ExecutionParameters,
    ) -> OrderSlice | None:
        """Calculate next POV slice."""
        try:
            if execution_state.remaining_quantity <= 0:
                return None

            # Get volume profile for better estimation
            volume_profile = self.volume_profiles.get(market_data.symbol)
            expected_volume = self._estimate_expected_volume(
                market_data, volume_profile
            )

            # Calculate slice size based on participation rate
            participation_rate = min(
                execution_state.current_participation_rate,
                parameters.max_participation_rate,
            )

            slice_size = expected_volume * participation_rate

            # Apply constraints
            slice_size = max(
                parameters.min_slice_size or execution_state.total_quantity * 0.005,
                min(
                    slice_size,
                    parameters.max_slice_size or execution_state.total_quantity * 0.1,
                    execution_state.remaining_quantity,
                ),
            )

            # Create slice
            slice_obj = OrderSlice(
                slice_id=f"{execution_state.order_id}_pov_{execution_state.slices_executed + 1}",
                parent_order_id=execution_state.order_id,
                symbol=market_data.symbol,
                side=OrderSide.BUY,  # Would be from original order
                quantity=slice_size,
                price=self._calculate_aggressive_price(market_data, parameters),
                order_type=OrderType.LIMIT,
                status=SliceStatus.PENDING,
            )

            return slice_obj

        except Exception as e:
            self.logger.error("Error calculating POV slice", error=str(e))
            return None

    def _estimate_expected_volume(
        self, market_data: MarketData, volume_profile: VolumeProfile | None
    ) -> float:
        """Estimate expected volume for next period."""
        if volume_profile:
            # Use historical patterns
            current_hour = datetime.now().hour
            if current_hour in volume_profile.peak_hours:
                return volume_profile.avg_volume * 1.5
            elif current_hour in volume_profile.quiet_hours:
                return volume_profile.avg_volume * 0.5
            else:
                return volume_profile.avg_volume
        else:
            # Fallback to current volume
            return max(market_data.volume, 1000)  # Minimum assumption

    def _adjust_participation_rate(
        self, execution_state: ExecutionState, market_data: MarketData
    ) -> None:
        """Adjust participation rate based on market impact."""
        # Calculate market impact (simplified)
        if execution_state.avg_execution_price > 0:
            mid_price = (market_data.bid + market_data.ask) / 2
            impact_bps = (
                abs(execution_state.avg_execution_price - mid_price) / mid_price * 10000
            )
            execution_state.market_impact_bps = impact_bps

            # Adjust participation rate
            if impact_bps > 5:  # High impact
                execution_state.current_participation_rate *= (
                    0.8  # Reduce aggressiveness
                )
            elif impact_bps < 2:  # Low impact
                execution_state.current_participation_rate *= (
                    1.1  # Increase aggressiveness
                )

            # Apply bounds
            execution_state.current_participation_rate = max(
                0.01,  # Minimum 1%
                min(execution_state.current_participation_rate, 0.25),  # Maximum 25%
            )

    def _calculate_aggressive_price(
        self, market_data: MarketData, parameters: ExecutionParameters
    ) -> float:
        """Calculate more aggressive price for POV execution."""
        # POV typically uses more aggressive pricing to ensure fills
        mid_price = (market_data.bid + market_data.ask) / 2

        # Use larger price improvement for POV
        price_improvement_bps = min(
            parameters.price_tolerance_bps, market_data.spread_bps / 2
        )
        price_improvement = mid_price * price_improvement_bps / 10000

        return mid_price + price_improvement


class VWAPAlgorithm(ExecutionAlgorithmBase):
    """
    Volume Weighted Average Price (VWAP) execution algorithm.

    Features:
    - Historical volume pattern matching
    - Intraday volume curve following
    - Benchmark tracking
    - Adaptive execution timing
    """

    async def execute_order(
        self, order: Order, parameters: ExecutionParameters, market_data: MarketData
    ) -> list[OrderSlice]:
        """Execute order using VWAP algorithm."""
        try:
            # Get volume profile
            volume_profile = self.volume_profiles.get(market_data.symbol)
            if not volume_profile:
                # Fallback to TWAP if no volume profile available
                self.logger.warning(
                    "No volume profile available, falling back to TWAP",
                    symbol=market_data.symbol,
                )
                twap_algo = TWAPAlgorithm(self.logger)
                return await twap_algo.execute_order(order, parameters, market_data)

            # Initialize execution state
            execution_state = ExecutionState(
                order_id=order.order_id,
                algorithm=ExecutionAlgorithm.VWAP,
                total_quantity=order.quantity,
                remaining_quantity=order.quantity,
                executed_quantity=0.0,
                avg_execution_price=0.0,
                start_time=datetime.now(),
                target_completion_time=datetime.now()
                + (parameters.target_completion_time or timedelta(hours=4)),
                slices_executed=0,
                total_slices=0,  # Dynamic based on volume curve
                current_participation_rate=0.0,
                market_impact_bps=0.0,
                execution_quality_score=0.0,
            )

            self.execution_states[order.order_id] = execution_state

            # Generate initial slice
            first_slice = self.calculate_next_slice(
                execution_state, market_data, parameters
            )

            self.logger.info(
                "VWAP execution started",
                order_id=order.order_id,
                completion_time=execution_state.target_completion_time,
                volume_profile_available=True,
            )

            return [first_slice] if first_slice else []

        except Exception as e:
            self.logger.error(
                "Error in VWAP execution", order_id=order.order_id, error=str(e)
            )
            return []

    async def update_execution(
        self, order_id: str, market_data: MarketData, filled_slices: list[OrderSlice]
    ) -> list[OrderSlice]:
        """Update VWAP execution."""
        try:
            execution_state = self.execution_states.get(order_id)
            if not execution_state:
                return []

            # Update execution state
            for slice_obj in filled_slices:
                if slice_obj.status == SliceStatus.FILLED:
                    execution_state.executed_quantity += slice_obj.filled_quantity
                    execution_state.remaining_quantity -= slice_obj.filled_quantity
                    execution_state.slices_executed += 1

                    # Update average execution price
                    if execution_state.executed_quantity > 0:
                        total_value = (
                            execution_state.avg_execution_price
                            * (
                                execution_state.executed_quantity
                                - slice_obj.filled_quantity
                            )
                            + slice_obj.filled_price * slice_obj.filled_quantity
                        )
                        execution_state.avg_execution_price = (
                            total_value / execution_state.executed_quantity
                        )

            # Check if execution is complete
            if execution_state.remaining_quantity <= 0:
                self.logger.info("VWAP execution completed", order_id=order_id)
                return []

            # Generate next slice based on volume curve
            next_slice = self.calculate_next_slice(
                execution_state,
                market_data,
                ExecutionParameters(ExecutionAlgorithm.VWAP),
            )

            return [next_slice] if next_slice else []

        except Exception as e:
            self.logger.error(
                "Error updating VWAP execution", order_id=order_id, error=str(e)
            )
            return []

    def calculate_next_slice(
        self,
        execution_state: ExecutionState,
        market_data: MarketData,
        parameters: ExecutionParameters,
    ) -> OrderSlice | None:
        """Calculate next VWAP slice."""
        try:
            if execution_state.remaining_quantity <= 0:
                return None

            # Get volume profile
            volume_profile = self.volume_profiles.get(market_data.symbol)
            if not volume_profile:
                return None

            # Calculate expected volume for current time
            current_hour = datetime.now().hour
            expected_volume_ratio = self._get_volume_ratio_for_hour(
                current_hour, volume_profile
            )

            # Calculate slice size based on volume curve
            remaining_time = (
                execution_state.target_completion_time - datetime.now()
            ).total_seconds()
            total_time = (
                execution_state.target_completion_time - execution_state.start_time
            ).total_seconds()

            if remaining_time <= 0:
                # Execute remaining quantity immediately
                slice_size = execution_state.remaining_quantity
            else:
                # Calculate proportional slice based on expected volume
                1.0 - (remaining_time / total_time)
                volume_weighted_progress = self._calculate_volume_weighted_progress(
                    execution_state.start_time, datetime.now(), volume_profile
                )

                # Adjust for volume curve
                expected_executed = (
                    execution_state.total_quantity * volume_weighted_progress
                )
                execution_deficit = (
                    expected_executed - execution_state.executed_quantity
                )

                # Base slice size
                base_slice = execution_state.remaining_quantity * expected_volume_ratio

                # Adjust for deficit
                if execution_deficit > 0:
                    slice_size = base_slice + execution_deficit * 0.5
                else:
                    slice_size = base_slice

            # Apply constraints
            slice_size = max(
                parameters.min_slice_size or execution_state.total_quantity * 0.005,
                min(
                    slice_size,
                    parameters.max_slice_size or execution_state.total_quantity * 0.15,
                    execution_state.remaining_quantity,
                ),
            )

            # Create slice
            slice_obj = OrderSlice(
                slice_id=f"{execution_state.order_id}_vwap_{execution_state.slices_executed + 1}",
                parent_order_id=execution_state.order_id,
                symbol=market_data.symbol,
                side=OrderSide.BUY,  # Would be from original order
                quantity=slice_size,
                price=self._calculate_vwap_price(market_data, parameters),
                order_type=OrderType.LIMIT,
                status=SliceStatus.PENDING,
            )

            return slice_obj

        except Exception as e:
            self.logger.error("Error calculating VWAP slice", error=str(e))
            return None

    def _get_volume_ratio_for_hour(
        self, hour: int, volume_profile: VolumeProfile
    ) -> float:
        """Get expected volume ratio for specific hour."""
        if hour in volume_profile.peak_hours:
            return 0.15  # 15% of daily volume in peak hours
        elif hour in volume_profile.quiet_hours:
            return 0.02  # 2% of daily volume in quiet hours
        else:
            return 0.04  # 4% of daily volume in normal hours

    def _calculate_volume_weighted_progress(
        self,
        start_time: datetime,
        current_time: datetime,
        volume_profile: VolumeProfile,
    ) -> float:
        """Calculate volume-weighted execution progress."""
        total_expected_volume = 0.0
        elapsed_expected_volume = 0.0

        # Calculate for each hour in the execution period
        start_time.hour
        end_hour = current_time.hour

        for hour in range(24):  # Full day calculation
            volume_ratio = self._get_volume_ratio_for_hour(hour, volume_profile)
            total_expected_volume += volume_ratio

            if hour <= end_hour:
                elapsed_expected_volume += volume_ratio

        return (
            elapsed_expected_volume / total_expected_volume
            if total_expected_volume > 0
            else 0.0
        )

    def _calculate_vwap_price(
        self, market_data: MarketData, parameters: ExecutionParameters
    ) -> float:
        """Calculate VWAP-optimized price."""
        # VWAP typically uses conservative pricing to track the benchmark
        mid_price = (market_data.bid + market_data.ask) / 2

        # Small price improvement to ensure fills while tracking VWAP
        price_improvement_bps = min(
            parameters.price_tolerance_bps / 3, market_data.spread_bps / 6
        )
        price_improvement = mid_price * price_improvement_bps / 10000

        return mid_price + price_improvement


class DirectExecutionAlgorithm(ExecutionAlgorithmBase):
    """
    Direct execution algorithm for immediate order execution.

    Features:
    - Immediate market execution
    - Smart order routing
    - Minimal market impact
    - Optimal timing
    """

    async def execute_order(
        self, order: Order, parameters: ExecutionParameters, market_data: MarketData
    ) -> list[OrderSlice]:
        """Execute order directly."""
        try:
            # For direct execution, create a single slice
            slice_obj = OrderSlice(
                slice_id=f"{order.order_id}_direct_1",
                parent_order_id=order.order_id,
                symbol=market_data.symbol,
                side=OrderSide.BUY,  # Would be from original order
                quantity=order.quantity,
                price=self._calculate_direct_price(market_data, parameters),
                order_type=OrderType.MARKET
                if parameters.allow_market_orders
                else OrderType.LIMIT,
                status=SliceStatus.PENDING,
            )

            # Initialize execution state
            execution_state = ExecutionState(
                order_id=order.order_id,
                algorithm=ExecutionAlgorithm.DIRECT,
                total_quantity=order.quantity,
                remaining_quantity=order.quantity,
                executed_quantity=0.0,
                avg_execution_price=0.0,
                start_time=datetime.now(),
                target_completion_time=datetime.now()
                + timedelta(seconds=30),  # Quick execution
                slices_executed=0,
                total_slices=1,
                current_participation_rate=1.0,  # Full participation for direct
                market_impact_bps=0.0,
                execution_quality_score=0.0,
            )

            self.execution_states[order.order_id] = execution_state

            self.logger.info(
                "Direct execution started",
                order_id=order.order_id,
                order_type=slice_obj.order_type.value,
                price=slice_obj.price,
            )

            return [slice_obj]

        except Exception as e:
            self.logger.error(
                "Error in direct execution", order_id=order.order_id, error=str(e)
            )
            return []

    async def update_execution(
        self, order_id: str, market_data: MarketData, filled_slices: list[OrderSlice]
    ) -> list[OrderSlice]:
        """Update direct execution (usually completes immediately)."""
        execution_state = self.execution_states.get(order_id)
        if not execution_state:
            return []

        # Update execution state
        for slice_obj in filled_slices:
            if slice_obj.status == SliceStatus.FILLED:
                execution_state.executed_quantity += slice_obj.filled_quantity
                execution_state.remaining_quantity -= slice_obj.filled_quantity
                execution_state.avg_execution_price = slice_obj.filled_price
                execution_state.slices_executed += 1

        return []  # Direct execution typically completes in one slice

    def calculate_next_slice(
        self,
        execution_state: ExecutionState,
        market_data: MarketData,
        parameters: ExecutionParameters,
    ) -> OrderSlice | None:
        """Direct execution doesn't typically need additional slices."""
        return None

    def _calculate_direct_price(
        self, market_data: MarketData, parameters: ExecutionParameters
    ) -> float:
        """Calculate price for direct execution."""
        if parameters.allow_market_orders:
            # Market order - use current ask/bid
            return market_data.ask  # Assuming buy order
        else:
            # Aggressive limit order
            mid_price = (market_data.bid + market_data.ask) / 2
            aggressive_improvement = mid_price * parameters.price_tolerance_bps / 10000
            return mid_price + aggressive_improvement


class ExecutionAlgorithmFactory:
    """Factory for creating execution algorithms."""

    @staticmethod
    def create_algorithm(
        algorithm_type: ExecutionAlgorithm,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> ExecutionAlgorithmBase:
        """Create execution algorithm instance."""
        if algorithm_type == ExecutionAlgorithm.TWAP:
            return TWAPAlgorithm(logger)
        elif algorithm_type == ExecutionAlgorithm.POV:
            return POVAlgorithm(logger)
        elif algorithm_type == ExecutionAlgorithm.VWAP:
            return VWAPAlgorithm(logger)
        elif algorithm_type == ExecutionAlgorithm.DIRECT:
            return DirectExecutionAlgorithm(logger)
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")


class ExecutionAlgorithmManager:
    """
    Manager for execution algorithms with market data integration.

    Features:
    - Algorithm selection and management
    - Market data integration
    - Performance monitoring
    - Adaptive algorithm switching
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or structlog.get_logger(__name__)
        self.algorithms: dict[ExecutionAlgorithm, ExecutionAlgorithmBase] = {}
        self.active_executions: dict[str, ExecutionAlgorithmBase] = {}
        self.performance_stats: dict[
            ExecutionAlgorithm, dict[str, float]
        ] = defaultdict(dict)

        # Initialize algorithms
        for algo_type in ExecutionAlgorithm:
            self.algorithms[algo_type] = ExecutionAlgorithmFactory.create_algorithm(
                algo_type, logger
            )

    async def execute_order(
        self, order: Order, parameters: ExecutionParameters, market_data: MarketData
    ) -> list[OrderSlice]:
        """Execute order using specified algorithm."""
        try:
            algorithm = self.algorithms[parameters.algorithm]
            self.active_executions[order.order_id] = algorithm

            slices = await algorithm.execute_order(order, parameters, market_data)

            self.logger.info(
                "Order execution started",
                order_id=order.order_id,
                algorithm=parameters.algorithm.value,
                initial_slices=len(slices),
            )

            return slices

        except Exception as e:
            self.logger.error(
                "Error executing order", order_id=order.order_id, error=str(e)
            )
            return []

    async def update_execution(
        self, order_id: str, market_data: MarketData, filled_slices: list[OrderSlice]
    ) -> list[OrderSlice]:
        """Update ongoing execution."""
        try:
            algorithm = self.active_executions.get(order_id)
            if not algorithm:
                return []

            return await algorithm.update_execution(
                order_id, market_data, filled_slices
            )

        except Exception as e:
            self.logger.error(
                "Error updating execution", order_id=order_id, error=str(e)
            )
            return []

    def update_market_data(self, market_data: MarketData) -> None:
        """Update market data for all algorithms."""
        for algorithm in self.algorithms.values():
            algorithm.update_market_data(market_data)

    def update_volume_profile(self, volume_profile: VolumeProfile) -> None:
        """Update volume profile for all algorithms."""
        for algorithm in self.algorithms.values():
            algorithm.update_volume_profile(volume_profile)

    def get_execution_state(self, order_id: str) -> ExecutionState | None:
        """Get execution state for an order."""
        algorithm = self.active_executions.get(order_id)
        if algorithm:
            return algorithm.get_execution_state(order_id)
        return None

    def complete_execution(self, order_id: str) -> None:
        """Mark execution as complete and clean up."""
        if order_id in self.active_executions:
            del self.active_executions[order_id]

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance statistics for all algorithms."""
        return {
            "active_executions": len(self.active_executions),
            "algorithm_performance": dict(self.performance_stats),
            "algorithms_available": [algo.value for algo in self.algorithms.keys()],
        }


def create_execution_algorithm_manager(
    logger: structlog.stdlib.BoundLogger | None = None,
) -> ExecutionAlgorithmManager:
    """
    Factory function to create execution algorithm manager.

    Args:
        logger: Optional logger instance

    Returns:
        Configured ExecutionAlgorithmManager instance
    """
    return ExecutionAlgorithmManager(logger=logger)
