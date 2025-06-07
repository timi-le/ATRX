"""
Execution Simulator - Realistic Order Execution for Backtesting.

This module simulates realistic order execution including:
- Market impact and slippage
- Execution latency and delays
- Partial fills and order rejection
- Realistic spread and liquidity modeling
- Commission and fee calculation
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog

from core.interfaces.trading_interfaces import (
    Order, OrderType, OrderSide, OrderStatus, Position
)
from core.interfaces.data_interfaces import MarketData, OHLCV
from backtester.market_replay import DataPoint


class FillType(Enum):
    """Types of order fills."""
    FULL = "full"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Fill:
    """Represents an order fill."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    fill_type: FillType = FillType.FULL


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""
    # Latency simulation
    min_latency_ms: int = 10
    max_latency_ms: int = 100
    network_jitter_ms: int = 5
    
    # Slippage modeling
    base_slippage_bps: float = 0.5  # Base slippage in basis points
    market_impact_factor: float = 0.1  # Impact per unit size
    volatility_slippage_factor: float = 2.0  # Multiplier during high volatility
    
    # Spread modeling
    min_spread_bps: float = 0.8  # Minimum spread in basis points
    max_spread_bps: float = 5.0  # Maximum spread in basis points
    spread_volatility_factor: float = 1.5  # Spread widens with volatility
    
    # Liquidity modeling
    max_position_size: float = 1000000.0  # Maximum position size
    liquidity_impact_threshold: float = 100000.0  # Size threshold for impact
    
    # Rejection modeling
    rejection_rate: float = 0.01  # 1% rejection rate
    max_order_size: float = 10000000.0  # Maximum single order size
    
    # Commission structure
    commission_per_lot: float = 7.0  # Commission per standard lot
    commission_percentage: float = 0.0  # Percentage-based commission
    minimum_commission: float = 2.0  # Minimum commission per trade
    
    # Partial fill modeling
    partial_fill_probability: float = 0.05  # 5% chance of partial fill
    min_fill_percentage: float = 0.3  # Minimum fill percentage
    
    # Market hours
    market_open_hour: int = 0  # 24/7 for FX
    market_close_hour: int = 24
    weekend_trading: bool = True  # Enable for backtesting (FX is global 24/7 except for brief weekend gaps)


class ExecutionSimulator:
    """
    Simulates realistic order execution for backtesting.
    
    Provides realistic modeling of:
    - Order latency and execution delays
    - Market impact and slippage
    - Partial fills and rejections
    - Commission and fees
    - Spread dynamics
    """
    
    def __init__(self, config: ExecutionConfig, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.execution_tasks: List[asyncio.Task] = []  # Track execution tasks
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        
        # Fill tracking
        self.fills: List[Fill] = []
        self.processed_fill_count = 0  # Track how many fills have been consumed
        
        # Market data
        self.current_market_data: Dict[str, Union[MarketData, OHLCV]] = {}
        
        # Volatility tracking
        self.volatility_window: Dict[str, List[float]] = {}
        self.volatility_lookback = 100  # Number of price points to track
        
        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.partial_fills = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.logger.info("ExecutionSimulator initialized", config=config.__dict__)
    
    async def submit_order(self, order: Order) -> str:
        """Submit an order for execution simulation."""
        self.total_orders += 1
        
        # Validate order
        if not await self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.rejected_orders += 1
            self.logger.warning("Order rejected", order_id=order.order_id, reason="validation_failed")
            return order.order_id
        
        # Schedule execution with latency
        latency = await self._calculate_latency()
        execution_task = asyncio.create_task(self._execute_order_after_delay(order, latency))
        self.execution_tasks.append(execution_task)  # Track the task
        
        # Store as pending order
        self.pending_orders[order.order_id] = order
        
        self.logger.info(
            "Order submitted",
            order_id=order.order_id,
            order_type=order.order_type.value,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            latency_ms=latency * 1000
        )
        
        return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            
            self.logger.info("Order cancelled", order_id=order_id)
            return True
        
        return False
    
    async def update_market_data(self, data_point: DataPoint) -> None:
        """Update current market data for execution simulation."""
        self.current_market_data[data_point.symbol] = data_point.data
        
        # Update volatility tracking
        if data_point.data_type == "bar":
            bar_data = data_point.data
            self._update_volatility(data_point.symbol, bar_data.close)
        elif data_point.data_type == "tick":
            tick_data = data_point.data
            self._update_volatility(data_point.symbol, tick_data.mid)
        
        # Check for pending order executions
        await self._check_pending_executions(data_point.symbol)
    
    async def _validate_order(self, order: Order) -> bool:
        """Validate order before execution."""
        # Check order size limits
        if order.quantity > self.config.max_order_size:
            self.logger.debug("Order rejected: quantity too large", 
                            order_id=order.order_id, 
                            quantity=order.quantity, 
                            max_size=self.config.max_order_size)
            return False
        
        # Check market hours (if applicable)
        if not self._is_market_open(order.timestamp):
            self.logger.debug("Order rejected: market closed", 
                            order_id=order.order_id, 
                            timestamp=order.timestamp,
                            weekday=order.timestamp.weekday(),
                            hour=order.timestamp.hour)
            return False
        
        # Random rejection simulation
        if np.random.random() < self.config.rejection_rate:
            self.logger.debug("Order rejected: random rejection", 
                            order_id=order.order_id, 
                            rejection_rate=self.config.rejection_rate)
            return False
        
        # Check if we have current market data
        if order.symbol not in self.current_market_data:
            self.logger.debug("Order rejected: no market data", 
                            order_id=order.order_id, 
                            symbol=order.symbol,
                            available_symbols=list(self.current_market_data.keys()))
            return False
        
        self.logger.debug("Order validation passed", 
                        order_id=order.order_id, 
                        symbol=order.symbol)
        return True
    
    def _is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open for trading."""
        if not self.config.weekend_trading:
            # FX markets are closed on weekends
            if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
        
        hour = timestamp.hour
        return self.config.market_open_hour <= hour < self.config.market_close_hour
    
    async def _calculate_latency(self) -> float:
        """Calculate execution latency in seconds."""
        base_latency = np.random.uniform(
            self.config.min_latency_ms,
            self.config.max_latency_ms
        )
        
        # Add network jitter
        jitter = np.random.normal(0, self.config.network_jitter_ms)
        
        total_latency_ms = max(0, base_latency + jitter)
        return total_latency_ms / 1000.0  # Convert to seconds
    
    async def _execute_order_after_delay(self, order: Order, delay: float) -> None:
        """Execute order after specified delay."""
        await asyncio.sleep(delay)
        
        # Check if order is still pending (not cancelled)
        if order.order_id in self.pending_orders:
            await self._execute_order(order)
    
    async def _execute_order(self, order: Order) -> None:
        """Execute a pending order."""
        try:
            # Get current market data
            market_data = self.current_market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.REJECTED
                self.rejected_orders += 1
                return
            
            # Calculate execution price
            execution_price = await self._calculate_execution_price(order, market_data)
            
            # Determine fill quantity
            fill_quantity = await self._calculate_fill_quantity(order)
            
            if fill_quantity <= 0:
                order.status = OrderStatus.REJECTED
                self.rejected_orders += 1
                return
            
            # Calculate slippage
            slippage = await self._calculate_slippage(order, execution_price, market_data)
            
            # Apply slippage to execution price
            if order.side == OrderSide.BUY:
                final_price = execution_price + slippage
            else:
                final_price = execution_price - slippage
            
            # Calculate commission
            commission = await self._calculate_commission(order, fill_quantity, final_price)
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=final_price,
                timestamp=datetime.now(),
                commission=commission,
                slippage=slippage,
                fill_type=FillType.FULL if fill_quantity == order.quantity else FillType.PARTIAL
            )
            
            # Update order
            order.filled_quantity += fill_quantity
            order.avg_fill_price = (
                (order.avg_fill_price * (order.filled_quantity - fill_quantity) + 
                 final_price * fill_quantity) / order.filled_quantity
            )
            
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                self.filled_orders += 1
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                self.partial_fills += 1
            
            # Update position
            await self._update_position(fill)
            
            # Store fill
            self.fills.append(fill)
            self.logger.debug("Fill appended in ExecutionSimulator", fill=fill, total_fills_now=len(self.fills), processed_fill_count_before_this_fill=self.processed_fill_count)
            self.total_commission += commission
            self.total_slippage += abs(slippage)
            
            self.logger.info(
                "Order executed",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=fill_quantity,
                price=final_price,
                commission=commission,
                slippage=slippage,
                status=order.status.value
            )
            
        except Exception as e:
            self.logger.error("Error executing order", order_id=order.order_id, error=str(e))
            order.status = OrderStatus.REJECTED
            self.rejected_orders += 1
    
    async def _calculate_execution_price(self, order: Order, market_data: Union[MarketData, OHLCV]) -> float:
        """Calculate the base execution price before slippage."""
        if isinstance(market_data, MarketData):
            # Tick data
            if order.order_type == OrderType.MARKET:
                return market_data.ask if order.side == OrderSide.BUY else market_data.bid
            elif order.order_type == OrderType.LIMIT:
                return order.price or market_data.mid
            else:
                return market_data.mid
        
        else:
            # Bar data (OHLCV)
            if order.order_type == OrderType.MARKET:
                # Use close price with simulated spread
                spread = await self._calculate_spread(order.symbol, market_data.close)
                if order.side == OrderSide.BUY:
                    return market_data.close + spread / 2
                else:
                    return market_data.close - spread / 2
            elif order.order_type == OrderType.LIMIT:
                return order.price or market_data.close
            else:
                return market_data.close
    
    async def _calculate_spread(self, symbol: str, price: float) -> float:
        """Calculate dynamic spread based on volatility and market conditions."""
        # Get current volatility
        volatility = self._get_current_volatility(symbol)
        
        # Base spread
        base_spread_abs = price * self.config.min_spread_bps / 10000
        
        # Volatility adjustment
        volatility_multiplier = 1 + (volatility * self.config.spread_volatility_factor)
        
        # Calculate final spread
        spread = base_spread_abs * volatility_multiplier
        
        # Apply limits
        max_spread_abs = price * self.config.max_spread_bps / 10000
        return min(spread, max_spread_abs)
    
    async def _calculate_fill_quantity(self, order: Order) -> float:
        """Calculate how much of the order gets filled."""
        # Check for partial fill
        if np.random.random() < self.config.partial_fill_probability:
            # Partial fill
            fill_percentage = np.random.uniform(
                self.config.min_fill_percentage,
                1.0
            )
            return order.quantity * fill_percentage
        
        # Full fill
        return order.quantity
    
    async def _calculate_slippage(
        self, 
        order: Order, 
        execution_price: float, 
        market_data: Union[MarketData, OHLCV]
    ) -> float:
        """Calculate slippage based on order size and market conditions."""
        # Base slippage
        base_slippage = execution_price * self.config.base_slippage_bps / 10000
        
        # Market impact based on order size
        if order.quantity > self.config.liquidity_impact_threshold:
            impact_factor = (order.quantity / self.config.liquidity_impact_threshold) ** 0.5
            market_impact = base_slippage * self.config.market_impact_factor * impact_factor
        else:
            market_impact = 0
        
        # Volatility impact
        volatility = self._get_current_volatility(order.symbol)
        volatility_impact = base_slippage * volatility * self.config.volatility_slippage_factor
        
        # Total slippage
        total_slippage = base_slippage + market_impact + volatility_impact
        
        # Add random component
        random_factor = np.random.normal(1.0, 0.2)  # 20% random variation
        total_slippage *= max(0.1, random_factor)  # Ensure positive
        
        return total_slippage
    
    async def _calculate_commission(self, order: Order, quantity: float, price: float) -> float:
        """Calculate commission for the trade."""
        # Lot-based commission
        lot_size = 100000  # Standard FX lot size
        lots = quantity / lot_size
        lot_commission = lots * self.config.commission_per_lot
        
        # Percentage-based commission
        notional_value = quantity * price
        percentage_commission = notional_value * self.config.commission_percentage / 100
        
        # Total commission
        total_commission = lot_commission + percentage_commission
        
        # Apply minimum
        return max(total_commission, self.config.minimum_commission)
    
    async def _update_position(self, fill: Fill) -> None:
        """Update position based on fill."""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0
            )
        
        position = self.positions[symbol]
        
        # Calculate new position
        if fill.side == OrderSide.BUY:
            new_quantity = position.quantity + fill.quantity
        else:
            new_quantity = position.quantity - fill.quantity
        
        # Update average price
        if new_quantity != 0:
            if (position.quantity >= 0 and fill.side == OrderSide.BUY) or \
               (position.quantity <= 0 and fill.side == OrderSide.SELL):
                # Adding to position
                total_cost = (position.quantity * position.avg_price + 
                             fill.quantity * fill.price)
                position.avg_price = total_cost / abs(new_quantity)
            else:
                # Reducing or reversing position
                if abs(new_quantity) < abs(position.quantity):
                    # Reducing position - keep same avg price
                    pass
                else:
                    # Reversing position - use new fill price
                    position.avg_price = fill.price
        
        position.quantity = new_quantity
        
        # Clean up zero positions
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]
    
    def _update_volatility(self, symbol: str, price: float) -> None:
        """Update volatility tracking for a symbol."""
        if symbol not in self.volatility_window:
            self.volatility_window[symbol] = []
        
        window = self.volatility_window[symbol]
        window.append(price)
        
        # Keep only recent prices
        if len(window) > self.volatility_lookback:
            window.pop(0)
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility estimate for a symbol."""
        if symbol not in self.volatility_window or len(self.volatility_window[symbol]) < 2:
            return 0.01  # Default volatility
        
        prices = self.volatility_window[symbol]
        returns = np.diff(np.log(prices))
        
        if len(returns) == 0:
            return 0.01
        
        # Annualized volatility (assuming minute data)
        volatility = np.std(returns) * np.sqrt(525600)  # Minutes in a year
        return max(0.001, min(0.1, volatility))  # Clamp between 0.1% and 10%
    
    async def _check_pending_executions(self, symbol: str) -> None:
        """Check if any pending orders should be executed based on current market data."""
        # This would handle limit orders, stop orders, etc.
        # For now, we execute market orders immediately
        pass
    
    async def wait_for_pending_orders(self, timeout: float = 10.0) -> None:
        """Wait for all pending orders to complete execution."""
        if not self.execution_tasks:
            return
        
        try:
            # Wait for all execution tasks to complete
            await asyncio.wait_for(
                asyncio.gather(*self.execution_tasks, return_exceptions=True),
                timeout=timeout
            )
            self.logger.info(f"All {len(self.execution_tasks)} pending orders processed")
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for {len(self.execution_tasks)} pending orders")
        finally:
            # Clear completed tasks
            self.execution_tasks.clear()
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_fills(self) -> List[Fill]:
        """Get all fills that haven't been processed yet."""
        # Return only new fills since last call
        new_fills = self.fills[self.processed_fill_count:].copy()
        self.logger.debug("ExecutionSimulator.get_fills called", total_fills_in_list=len(self.fills), processed_fill_count=self.processed_fill_count, num_new_fills_returned=len(new_fills))
        return new_fills
    
    def mark_fills_processed(self, count: int) -> None:
        """Mark a number of fills as processed by the performance analyzer."""
        old_processed_count = self.processed_fill_count
        self.processed_fill_count = min(self.processed_fill_count + count, len(self.fills))
        self.logger.debug("ExecutionSimulator.mark_fills_processed called", count_to_mark=count, old_processed_count=old_processed_count, new_processed_count=self.processed_fill_count, total_fills_in_list=len(self.fills))
    
    def get_all_fills(self) -> List[Fill]:
        """Get all fills (for statistics purposes)."""
        return self.fills.copy()
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get execution statistics."""
        fill_rate = self.filled_orders / self.total_orders if self.total_orders > 0 else 0
        all_fills = self.get_all_fills()  # Get all fills for accurate statistics
        avg_commission = self.total_commission / len(all_fills) if all_fills else 0
        avg_slippage = self.total_slippage / len(all_fills) if all_fills else 0
        
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "rejected_orders": self.rejected_orders,
            "partial_fills": self.partial_fills,
            "fill_rate": fill_rate,
            "total_fills": len(all_fills),
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "avg_commission_per_fill": avg_commission,
            "avg_slippage_per_fill": avg_slippage,
            "pending_orders": len(self.pending_orders)
        }
    
    def reset(self) -> None:
        """Reset simulator state."""
        # Cancel any pending execution tasks
        for task in self.execution_tasks:
            if not task.done():
                task.cancel()
        self.execution_tasks.clear()
        
        self.pending_orders.clear()
        self.positions.clear()
        self.fills.clear()
        self.processed_fill_count = 0  # Reset processed count
        self.current_market_data.clear()
        self.volatility_window.clear()
        
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.partial_fills = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.logger.info("ExecutionSimulator reset")


# Utility functions
def create_execution_config(
    latency_ms: Tuple[int, int] = (10, 100),
    slippage_bps: float = 0.5,
    commission_per_lot: float = 7.0,
    rejection_rate: float = 0.01
) -> ExecutionConfig:
    """Create execution configuration with common defaults."""
    return ExecutionConfig(
        min_latency_ms=latency_ms[0],
        max_latency_ms=latency_ms[1],
        base_slippage_bps=slippage_bps,
        commission_per_lot=commission_per_lot,
        rejection_rate=rejection_rate
    )


class RealisticExecutionSimulator(ExecutionSimulator):
    """
    Enhanced execution simulator with more realistic modeling.
    
    Includes:
    - Time-of-day effects on spreads and liquidity
    - News event impact simulation
    - Market regime-dependent execution quality
    """
    
    def __init__(self, config: ExecutionConfig, logger: Optional[structlog.stdlib.BoundLogger] = None):
        super().__init__(config, logger)
        
        # Enhanced modeling parameters
        self.time_of_day_factors = {
            "asian_session": (0, 8),      # Lower liquidity
            "london_session": (8, 16),    # High liquidity
            "ny_session": (13, 21),       # Highest liquidity
            "overlap": (13, 16),          # Peak liquidity
        }
        
        self.session_multipliers = {
            "asian_session": 1.5,    # Higher spreads/slippage
            "london_session": 0.8,   # Lower spreads/slippage
            "ny_session": 0.7,       # Lowest spreads/slippage
            "overlap": 0.6,          # Best execution
        }
    
    def _get_session_multiplier(self, timestamp: datetime) -> float:
        """Get execution quality multiplier based on time of day."""
        hour = timestamp.hour
        
        # Check for overlap first (best execution)
        if 13 <= hour < 16:
            return self.session_multipliers["overlap"]
        elif 8 <= hour < 16:
            return self.session_multipliers["london_session"]
        elif 13 <= hour < 21:
            return self.session_multipliers["ny_session"]
        else:
            return self.session_multipliers["asian_session"]
    
    async def _calculate_slippage(
        self, 
        order: Order, 
        execution_price: float, 
        market_data: Union[MarketData, OHLCV]
    ) -> float:
        """Enhanced slippage calculation with time-of-day effects."""
        base_slippage = await super()._calculate_slippage(order, execution_price, market_data)
        
        # Apply session multiplier
        session_multiplier = self._get_session_multiplier(order.timestamp)
        
        return base_slippage * session_multiplier 