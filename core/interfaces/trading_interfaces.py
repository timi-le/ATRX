"""
Trading interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any



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
        price: float | None = None,
        stop_price: float | None = None,
        timestamp: datetime | None = None,
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
    price: float | None = None
    sl: float | None = None
    features: dict[str, float] | None = None
    take_profit_pips: float | None = None
    stop_loss_pips: float | None = None
    win_probability: float | None = None


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    async def generate_signal(
        self,
        market_data: Any,
        features: dict[str, float] | None = None,
        regime: str | None = None,
    ) -> Signal | None:
        """Generate a trading signal."""

    @abstractmethod
    async def update_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters."""

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters."""

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""


class PositionSizer(ABC):
    """Abstract base class for position sizing."""

    @abstractmethod
    async def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_positions: dict[str, Position],
        volatility: float | None = None,
    ) -> float:
        """Calculate optimal position size using Kelly criterion."""

    @abstractmethod
    async def calculate_kelly_fraction(
        self, win_probability: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate Kelly fraction: f* = (bp - q) / b"""

    @abstractmethod
    async def apply_risk_limits(
        self, position_size: float, symbol: str, account_balance: float
    ) -> float:
        """Apply maximum allocation caps and risk limits."""


class RiskManager(ABC):
    """Abstract base class for risk management."""

    @abstractmethod
    async def check_pre_trade_risk(
        self,
        order: Order,
        current_positions: dict[str, Position],
        account_balance: float,
    ) -> bool:
        """Check if order passes pre-trade risk checks."""

    @abstractmethod
    async def monitor_drawdown(self, current_pnl: float, peak_pnl: float) -> bool:
        """Monitor drawdown levels."""

    @abstractmethod
    async def calculate_var(
        self,
        positions: dict[str, Position],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Calculate Value at Risk."""

    @abstractmethod
    async def check_position_limits(
        self, symbol: str, new_quantity: float, current_positions: dict[str, Position]
    ) -> bool:
        """Check position size limits."""

    @abstractmethod
    async def emergency_stop(self, reason: str) -> None:
        """Emergency stop trading."""


class ExecutionEngine(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        """Get current positions."""

    @abstractmethod
    async def get_account_balance(self) -> float:
        """Get current account balance."""


class OrderManager(ABC):
    """Abstract base class for order management."""

    @abstractmethod
    async def slice_order(
        self, order: Order, slice_size: float, time_interval: int
    ) -> list[Order]:
        """Slice large order using TWAP/POV."""

    @abstractmethod
    async def manage_order_lifecycle(self, order: Order) -> None:
        """Manage complete order lifecycle."""

    @abstractmethod
    async def calculate_execution_quality(
        self, order: Order, fills: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate execution quality metrics."""

    @abstractmethod
    async def get_slippage(self, order: Order, reference_price: float) -> float:
        """Calculate execution slippage."""
