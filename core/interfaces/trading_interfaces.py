"""
Trading interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum
import pandas as pd
from dataclasses import dataclass


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionType(Enum):
    """Position types."""
    
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalType(Enum):
    """Signal types."""
    
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class Order:
    """Standardized order structure."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.timestamp = timestamp or datetime.now()
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0


class Position:
    """Standardized position structure."""

    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.total_pnl = unrealized_pnl + realized_pnl


@dataclass
class Signal:
    """Represents a trading signal from a strategy."""
    symbol: str
    side: OrderSide
    size: float  # The quantity or amount to trade
    order_type: OrderType
    strength: float  # Signal strength (0-1)
    confidence: float  # Model confidence (0-1)
    strategy_name: str
    timestamp: datetime
    price: Optional[float] = None
    sl: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    take_profit_pips: Optional[float] = None
    stop_loss_pips: Optional[float] = None
    win_probability: Optional[float] = None


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    async def generate_signal(
        self,
        market_data: Any,
        features: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
    ) -> Optional[Signal]:
        """Generate a trading signal."""
        pass

    @abstractmethod
    async def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class PositionSizer(ABC):
    """Abstract base class for position sizing."""

    @abstractmethod
    async def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_positions: Dict[str, Position],
        volatility: Optional[float] = None,
    ) -> float:
        """Calculate optimal position size using Kelly criterion."""
        pass

    @abstractmethod
    async def calculate_kelly_fraction(
        self, win_probability: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate Kelly fraction: f* = (bp - q) / b"""
        pass

    @abstractmethod
    async def apply_risk_limits(
        self, position_size: float, symbol: str, account_balance: float
    ) -> float:
        """Apply maximum allocation caps and risk limits."""
        pass


class RiskManager(ABC):
    """Abstract base class for risk management."""

    @abstractmethod
    async def check_pre_trade_risk(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_balance: float,
    ) -> bool:
        """Check if order passes pre-trade risk checks."""
        pass

    @abstractmethod
    async def monitor_drawdown(self, current_pnl: float, peak_pnl: float) -> bool:
        """Monitor drawdown levels."""
        pass

    @abstractmethod
    async def calculate_var(
        self,
        positions: Dict[str, Position],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Calculate Value at Risk."""
        pass

    @abstractmethod
    async def check_position_limits(
        self, symbol: str, new_quantity: float, current_positions: Dict[str, Position]
    ) -> bool:
        """Check position size limits."""
        pass

    @abstractmethod
    async def emergency_stop(self, reason: str) -> None:
        """Emergency stop trading."""
        pass


class ExecutionEngine(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_account_balance(self) -> float:
        """Get current account balance."""
        pass


class OrderManager(ABC):
    """Abstract base class for order management."""

    @abstractmethod
    async def slice_order(
        self, order: Order, slice_size: float, time_interval: int
    ) -> List[Order]:
        """Slice large order using TWAP/POV."""
        pass

    @abstractmethod
    async def manage_order_lifecycle(self, order: Order) -> None:
        """Manage complete order lifecycle."""
        pass

    @abstractmethod
    async def calculate_execution_quality(
        self, order: Order, fills: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate execution quality metrics."""
        pass

    @abstractmethod
    async def get_slippage(self, order: Order, reference_price: float) -> float:
        """Calculate execution slippage."""
        pass
