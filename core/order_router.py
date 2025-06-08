"""
Order Router - Broker Integration for FX AI-Quant Trading System.

This module implements order routing logic and broker integration interfaces
for MT5 (MetaTrader 5) and IBKR (Interactive Brokers) platforms.

Features:
- Broker abstraction layer
- MT5 and IBKR specific implementations
- Connection management and health monitoring
- Order routing and format conversion
- Error handling and retry logic
- Real-time order status synchronization
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


import structlog
import yaml

from core.interfaces.trading_interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from core.pubsub import ZMQPublisher


class BrokerType(Enum):
    """Supported broker types."""

    MT5 = "mt5"
    IBKR = "ibkr"
    MOCK = "mock"


class ConnectionStatus(Enum):
    """Broker connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class OrderRouteStatus(Enum):
    """Order routing status."""

    PENDING = "pending"
    ROUTED = "routed"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class BrokerConfig:
    """Broker configuration."""

    broker_type: BrokerType
    enabled: bool = True
    priority: int = 1
    max_orders_per_second: float = 10.0
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 60
    order_timeout: int = 300
    position_sync_interval: int = 30
    credentials: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerOrder:
    """Broker-specific order representation."""

    broker_order_id: str
    internal_order_id: str
    broker_type: BrokerType
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None
    stop_price: float | None
    status: OrderStatus
    submitted_time: datetime
    filled_time: datetime | None = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPosition:
    """Broker-specific position representation."""

    broker_position_id: str
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    commission: float
    swap: float
    last_update: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class BrokerInterface(ABC):
    """Abstract base class for broker interfaces."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to broker."""

    @abstractmethod
    async def submit_order(self, order: Order) -> BrokerOrder:
        """Submit order to broker."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at broker."""

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status from broker."""

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Get all positions from broker."""

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        """Get account information from broker."""

    @abstractmethod
    async def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get current market data for symbol."""


class MT5Interface(BrokerInterface):
    """MetaTrader 5 broker interface."""

    def __init__(
        self,
        config: BrokerConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.mt5 = None  # Would import MetaTrader5 package
        self.last_health_check = datetime.now()
        self.order_counter = 0

    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            self.connection_status = ConnectionStatus.CONNECTING

            # In real implementation, would use:
            # import MetaTrader5 as mt5
            # if not mt5.initialize():
            #     raise Exception("MT5 initialization failed")

            # Simulate connection
            await asyncio.sleep(1)

            # Login with credentials
            login = self.config.credentials.get("login")
            password = self.config.credentials.get("password")
            server = self.config.credentials.get("server")

            if not all([login, password, server]):
                raise ValueError("Missing MT5 credentials")

            # In real implementation:
            # if not mt5.login(login, password, server):
            #     raise Exception("MT5 login failed")

            self.connection_status = ConnectionStatus.CONNECTED
            self.logger.info("Connected to MT5", server=server, login=login)
            return True

        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            self.logger.error("Failed to connect to MT5", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from MT5."""
        try:
            # In real implementation:
            # if self.mt5:
            #     self.mt5.shutdown()

            self.connection_status = ConnectionStatus.DISCONNECTED
            self.logger.info("Disconnected from MT5")

        except Exception as e:
            self.logger.error("Error disconnecting from MT5", error=str(e))

    async def is_connected(self) -> bool:
        """Check MT5 connection status."""
        return self.connection_status == ConnectionStatus.CONNECTED

    async def submit_order(self, order: Order) -> BrokerOrder:
        """Submit order to MT5."""
        try:
            if not await self.is_connected():
                raise Exception("Not connected to MT5")

            # Convert to MT5 order format
            self._convert_to_mt5_order(order)

            # Submit order (simulated)
            # In real implementation:
            # result = mt5.order_send(mt5_request)
            # if result.retcode != mt5.TRADE_RETCODE_DONE:
            #     raise Exception(f"Order failed: {result.comment}")

            # Simulate successful submission
            broker_order = BrokerOrder(
                broker_order_id=f"MT5_{self.order_counter}",
                internal_order_id=order.order_id,
                broker_type=BrokerType.MT5,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                status=OrderStatus.PENDING,
                submitted_time=datetime.now(),
            )

            self.order_counter += 1

            self.logger.info(
                "Order submitted to MT5",
                order_id=order.order_id,
                broker_order_id=broker_order.broker_order_id,
                symbol=order.symbol,
            )

            return broker_order

        except Exception as e:
            self.logger.error(
                "Error submitting order to MT5", order_id=order.order_id, error=str(e)
            )
            raise

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order in MT5."""
        try:
            if not await self.is_connected():
                raise Exception("Not connected to MT5")

            # In real implementation:
            # request = {
            #     "action": mt5.TRADE_ACTION_REMOVE,
            #     "order": int(broker_order_id.split('_')[1])
            # }
            # result = mt5.order_send(request)
            # return result.retcode == mt5.TRADE_RETCODE_DONE

            # Simulate cancellation
            self.logger.info("Order cancelled in MT5", broker_order_id=broker_order_id)
            return True

        except Exception as e:
            self.logger.error(
                "Error cancelling order in MT5",
                broker_order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status from MT5."""
        try:
            if not await self.is_connected():
                return OrderStatus.REJECTED

            # In real implementation:
            # orders = mt5.orders_get(ticket=int(broker_order_id.split('_')[1]))
            # if orders:
            #     return self._convert_mt5_status(orders[0].state)

            # Simulate status check
            return OrderStatus.FILLED  # Simplified

        except Exception as e:
            self.logger.error(
                "Error getting order status from MT5",
                broker_order_id=broker_order_id,
                error=str(e),
            )
            return OrderStatus.REJECTED

    async def get_positions(self) -> list[BrokerPosition]:
        """Get positions from MT5."""
        try:
            if not await self.is_connected():
                return []

            # In real implementation:
            # positions = mt5.positions_get()
            # return [self._convert_mt5_position(pos) for pos in positions]

            # Simulate positions
            return []

        except Exception as e:
            self.logger.error("Error getting positions from MT5", error=str(e))
            return []

    async def get_account_info(self) -> dict[str, Any]:
        """Get account info from MT5."""
        try:
            if not await self.is_connected():
                return {}

            # In real implementation:
            # account_info = mt5.account_info()
            # return account_info._asdict() if account_info else {}

            # Simulate account info
            return {
                "balance": 100000.0,
                "equity": 100000.0,
                "margin": 0.0,
                "free_margin": 100000.0,
                "currency": "USD",
            }

        except Exception as e:
            self.logger.error("Error getting account info from MT5", error=str(e))
            return {}

    async def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data from MT5."""
        try:
            if not await self.is_connected():
                return {}

            # In real implementation:
            # tick = mt5.symbol_info_tick(symbol)
            # return tick._asdict() if tick else {}

            # Simulate market data
            return {
                "bid": 1.1000,
                "ask": 1.1002,
                "spread": 2,
                "volume": 1000,
                "time": datetime.now(),
            }

        except Exception as e:
            self.logger.error(
                "Error getting market data from MT5", symbol=symbol, error=str(e)
            )
            return {}

    def _convert_to_mt5_order(self, order: Order) -> dict[str, Any]:
        """Convert internal order to MT5 format."""
        # In real implementation, would convert to MT5 request format
        return {
            "symbol": order.symbol,
            "volume": order.quantity / 100000,  # Convert to lots
            "type": self._convert_order_type(order.order_type),
            "price": order.price,
            "sl": order.stop_price,
            "comment": f"Order_{order.order_id}",
        }

    def _convert_order_type(self, order_type: OrderType) -> int:
        """Convert order type to MT5 format."""
        # In real implementation, would use MT5 constants
        mapping = {
            OrderType.MARKET: 0,  # ORDER_TYPE_BUY/SELL
            OrderType.LIMIT: 2,  # ORDER_TYPE_BUY_LIMIT/SELL_LIMIT
            OrderType.STOP: 4,  # ORDER_TYPE_BUY_STOP/SELL_STOP
        }
        return mapping.get(order_type, 0)


class IBKRInterface(BrokerInterface):
    """Interactive Brokers interface using FIX protocol."""

    def __init__(
        self,
        config: BrokerConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_health_check = datetime.now()

        # Import and initialize FIX order router
        try:
            from services.execution.fix_connector.fix_order_router import (
                create_fix_order_router,
            )

            self.fix_router = create_fix_order_router()

            # Setup callbacks
            self.fix_router.set_order_update_callback(self._on_order_update)
            self.fix_router.set_execution_callback(self._on_execution)
            self.fix_router.set_error_callback(self._on_error)

            self.logger.info("IBKR FIX interface initialized")

        except ImportError as e:
            self.logger.error("Failed to import FIX connector", error=str(e))
            self.fix_router = None

        # Order tracking
        self.active_orders: dict[str, BrokerOrder] = {}
        self.order_id_mapping: dict[
            str, str
        ] = {}  # broker_order_id -> internal_order_id

    async def connect(self) -> bool:
        """Connect to IBKR via FIX protocol."""
        try:
            if not self.fix_router:
                raise Exception("FIX router not available")

            self.connection_status = ConnectionStatus.CONNECTING

            success = await self.fix_router.connect()

            if success:
                self.connection_status = ConnectionStatus.CONNECTED
                self.logger.info("Connected to IBKR via FIX")
                return True
            else:
                self.connection_status = ConnectionStatus.ERROR
                self.logger.error("Failed to connect to IBKR via FIX")
                return False

        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            self.logger.error("Error connecting to IBKR", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        try:
            if self.fix_router:
                await self.fix_router.disconnect()

            self.connection_status = ConnectionStatus.DISCONNECTED
            self.logger.info("Disconnected from IBKR")

        except Exception as e:
            self.logger.error("Error disconnecting from IBKR", error=str(e))

    async def is_connected(self) -> bool:
        """Check IBKR connection status."""
        if not self.fix_router:
            return False
        return self.fix_router.is_connected()

    async def submit_order(self, order: Order) -> BrokerOrder:
        """Submit order to IBKR via FIX."""
        try:
            if not await self.is_connected():
                raise Exception("Not connected to IBKR")

            # Submit order via FIX
            result = await self.fix_router.submit_order(order)

            if result["success"]:
                # Create broker order
                broker_order = BrokerOrder(
                    broker_order_id=result["fix_client_order_id"],
                    internal_order_id=order.order_id,
                    broker_type=BrokerType.IBKR,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=order.price,
                    stop_price=getattr(order, "stop_price", None),
                    status=OrderStatus.PENDING,
                    submitted_time=datetime.now(),
                )

                # Store order mapping
                self.active_orders[order.order_id] = broker_order
                self.order_id_mapping[result["fix_client_order_id"]] = order.order_id

                self.logger.info(
                    "Order submitted to IBKR",
                    order_id=order.order_id,
                    fix_client_order_id=result["fix_client_order_id"],
                    symbol=order.symbol,
                )

                return broker_order
            else:
                raise Exception(
                    f"Order submission failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            self.logger.error(
                "Error submitting order to IBKR", order_id=order.order_id, error=str(e)
            )
            raise

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order in IBKR via FIX."""
        try:
            if not await self.is_connected():
                raise Exception("Not connected to IBKR")

            # Find internal order ID
            internal_order_id = self.order_id_mapping.get(broker_order_id)
            if not internal_order_id:
                self.logger.warning(
                    "Order not found for cancellation", broker_order_id=broker_order_id
                )
                return False

            # Cancel via FIX
            result = await self.fix_router.cancel_order(internal_order_id)

            if result["success"]:
                self.logger.info(
                    "Order cancelled in IBKR", broker_order_id=broker_order_id
                )
                return True
            else:
                self.logger.error(
                    "Order cancellation failed",
                    broker_order_id=broker_order_id,
                    error=result.get("error"),
                )
                return False

        except Exception as e:
            self.logger.error(
                "Error cancelling order in IBKR",
                broker_order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status from IBKR."""
        try:
            if not await self.is_connected():
                return OrderStatus.REJECTED

            # Find internal order ID
            internal_order_id = self.order_id_mapping.get(broker_order_id)
            if not internal_order_id:
                return OrderStatus.REJECTED

            # Get status from FIX router
            order_status = self.fix_router.get_order_status(internal_order_id)
            if order_status:
                # Convert FIX status to internal status
                fix_status = order_status.get("status", "NEW")
                return self._convert_fix_status_to_internal(fix_status)

            return OrderStatus.PENDING

        except Exception as e:
            self.logger.error(
                "Error getting order status from IBKR",
                broker_order_id=broker_order_id,
                error=str(e),
            )
            return OrderStatus.REJECTED

    async def get_positions(self) -> list[BrokerPosition]:
        """Get positions from IBKR."""
        try:
            if not await self.is_connected():
                return []

            # Get positions from FIX router
            positions = self.fix_router.get_positions()

            # Convert to broker positions
            broker_positions = []
            for pos in positions:
                broker_pos = BrokerPosition(
                    broker_position_id=f"IBKR_{pos.symbol}",
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    avg_price=pos.avg_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    commission=0.0,  # Would need to be calculated
                    swap=0.0,
                    last_update=datetime.now(),
                )
                broker_positions.append(broker_pos)

            return broker_positions

        except Exception as e:
            self.logger.error("Error getting positions from IBKR", error=str(e))
            return []

    async def get_account_info(self) -> dict[str, Any]:
        """Get account info from IBKR."""
        try:
            if not await self.is_connected():
                return {}

            # Get account info from FIX router
            account_info = self.fix_router.get_account_info()

            return {
                "broker": "IBKR",
                "connection_status": account_info.get("connection_status", "unknown"),
                "session_start_time": account_info.get("session_start_time"),
                "total_orders_sent": account_info.get("total_orders_sent", 0),
                "total_executions_received": account_info.get(
                    "total_executions_received", 0
                ),
                "active_orders_count": account_info.get("active_orders_count", 0),
                "fix_stats": account_info.get("fix_stats", {}),
            }

        except Exception as e:
            self.logger.error("Error getting account info from IBKR", error=str(e))
            return {}

    async def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data from IBKR."""
        try:
            if not await self.is_connected():
                return {}

            # IBKR FIX doesn't typically provide market data
            # Would need separate market data connection
            # For now, return empty dict
            return {}

        except Exception as e:
            self.logger.error(
                "Error getting market data from IBKR", symbol=symbol, error=str(e)
            )
            return {}

    # FIX callback handlers
    def _on_order_update(
        self, order_id: str, status: "OrderStatus", details: dict[str, Any]
    ):
        """Handle order update from FIX."""
        try:
            if order_id in self.active_orders:
                broker_order = self.active_orders[order_id]
                broker_order.status = status
                broker_order.filled_quantity = details.get("filled_quantity", 0.0)
                broker_order.avg_fill_price = details.get("avg_price", 0.0)

                if status == OrderStatus.FILLED:
                    broker_order.filled_time = datetime.now()

                self.logger.info(
                    "Order status updated",
                    order_id=order_id,
                    status=status.value,
                    filled_qty=broker_order.filled_quantity,
                )

        except Exception as e:
            self.logger.error(
                "Error handling order update", order_id=order_id, error=str(e)
            )

    def _on_execution(
        self, order_id: str, fill_qty: float, fill_price: float, timestamp: datetime
    ):
        """Handle execution report from FIX."""
        try:
            self.logger.info(
                "Order execution received",
                order_id=order_id,
                fill_qty=fill_qty,
                fill_price=fill_price,
                timestamp=timestamp.isoformat(),
            )

        except Exception as e:
            self.logger.error(
                "Error handling execution", order_id=order_id, error=str(e)
            )

    def _on_error(self, order_id: str, error_message: str):
        """Handle error from FIX."""
        try:
            if order_id in self.active_orders:
                broker_order = self.active_orders[order_id]
                broker_order.status = OrderStatus.REJECTED
                broker_order.error_message = error_message

            self.logger.error(
                "Order error received", order_id=order_id, error=error_message
            )

        except Exception as e:
            self.logger.error(
                "Error handling order error", order_id=order_id, error=str(e)
            )

    def _convert_fix_status_to_internal(self, fix_status: str) -> OrderStatus:
        """Convert FIX status to internal status."""
        mapping = {
            "NEW": OrderStatus.PENDING,
            "PENDING_NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "PENDING_CANCEL": OrderStatus.PENDING,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.CANCELLED,
        }
        return mapping.get(fix_status, OrderStatus.PENDING)


class MockBrokerInterface(BrokerInterface):
    """Mock broker interface for testing."""

    def __init__(
        self,
        config: BrokerConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.orders: dict[str, BrokerOrder] = {}
        self.positions: list[BrokerPosition] = []
        self.order_counter = 0

    async def connect(self) -> bool:
        """Mock connection."""
        await asyncio.sleep(0.1)
        self.connection_status = ConnectionStatus.CONNECTED
        self.logger.info("Connected to Mock Broker")
        return True

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.logger.info("Disconnected from Mock Broker")

    async def is_connected(self) -> bool:
        """Check mock connection."""
        return self.connection_status == ConnectionStatus.CONNECTED

    async def submit_order(self, order: Order) -> BrokerOrder:
        """Submit mock order."""
        broker_order = BrokerOrder(
            broker_order_id=f"MOCK_{self.order_counter}",
            internal_order_id=order.order_id,
            broker_type=BrokerType.MOCK,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            status=OrderStatus.PENDING,
            submitted_time=datetime.now(),
        )

        self.orders[broker_order.broker_order_id] = broker_order
        self.order_counter += 1

        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            await asyncio.sleep(0.05)
            broker_order.status = OrderStatus.FILLED
            broker_order.filled_time = datetime.now()
            broker_order.filled_quantity = order.quantity
            broker_order.avg_fill_price = order.price or 1.1000

        return broker_order

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel mock order."""
        if broker_order_id in self.orders:
            self.orders[broker_order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get mock order status."""
        if broker_order_id in self.orders:
            return self.orders[broker_order_id].status
        return OrderStatus.REJECTED

    async def get_positions(self) -> list[BrokerPosition]:
        """Get mock positions."""
        return self.positions

    async def get_account_info(self) -> dict[str, Any]:
        """Get mock account info."""
        return {
            "balance": 100000.0,
            "equity": 100000.0,
            "margin": 0.0,
            "free_margin": 100000.0,
            "currency": "USD",
        }

    async def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get mock market data."""
        return {
            "bid": 1.1000,
            "ask": 1.1002,
            "spread": 2,
            "volume": 1000,
            "time": datetime.now(),
        }


class OrderRouter:
    """
    Order router that manages multiple broker connections and routes orders.

    Features:
    - Multi-broker support (MT5, IBKR, Mock)
    - Intelligent order routing based on broker capabilities
    - Connection health monitoring and failover
    - Order synchronization across brokers
    - Performance monitoring and statistics
    """

    def __init__(
        self,
        config_path: str = "config/execution_settings.yaml",
        publisher: ZMQPublisher | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self.publisher = publisher

        # Load configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Initialize brokers
        self.brokers: dict[BrokerType, BrokerInterface] = {}
        self.broker_configs: dict[BrokerType, BrokerConfig] = {}
        self.broker_health: dict[BrokerType, dict[str, Any]] = {}

        # Order tracking
        self.routed_orders: dict[str, BrokerOrder] = {}
        self.order_routing_map: dict[str, BrokerType] = {}

        # Performance tracking
        self.routing_stats = {
            "orders_routed": 0,
            "orders_failed": 0,
            "avg_routing_time": 0.0,
            "broker_performance": defaultdict(dict),
        }

        # Initialize brokers from config
        self._initialize_brokers(config.get("brokers", {}))

        self.logger.info(
            "OrderRouter initialized",
            brokers=list(self.brokers.keys()),
            enabled_brokers=[
                bt for bt, bc in self.broker_configs.items() if bc.enabled
            ],
        )

    def _initialize_brokers(self, broker_configs: dict[str, Any]) -> None:
        """Initialize broker interfaces from configuration."""
        for broker_name, broker_config in broker_configs.items():
            try:
                broker_type = BrokerType(broker_name.lower())

                config = BrokerConfig(
                    broker_type=broker_type,
                    enabled=broker_config.get("enabled", True),
                    priority=broker_config.get("priority", 1),
                    max_orders_per_second=broker_config.get(
                        "max_orders_per_second", 10.0
                    ),
                    connection_timeout=broker_config.get("connection_timeout", 30),
                    retry_attempts=broker_config.get("retry_attempts", 3),
                    retry_delay=broker_config.get("retry_delay", 1.0),
                    credentials=broker_config.get("credentials", {}),
                    settings=broker_config.get("settings", {}),
                )

                self.broker_configs[broker_type] = config

                # Create broker interface
                if broker_type == BrokerType.MT5:
                    self.brokers[broker_type] = MT5Interface(config, self.logger)
                elif broker_type == BrokerType.IBKR:
                    self.brokers[broker_type] = IBKRInterface(config, self.logger)
                elif broker_type == BrokerType.MOCK:
                    self.brokers[broker_type] = MockBrokerInterface(config, self.logger)

                # Initialize health tracking
                self.broker_health[broker_type] = {
                    "status": ConnectionStatus.DISCONNECTED,
                    "last_check": datetime.now(),
                    "error_count": 0,
                    "orders_processed": 0,
                    "avg_latency": 0.0,
                }

            except Exception as e:
                self.logger.error(
                    f"Failed to initialize broker {broker_name}", error=str(e)
                )

    async def start(self) -> None:
        """Start the order router and connect to brokers."""
        try:
            # Connect to enabled brokers
            for broker_type, broker in self.brokers.items():
                config = self.broker_configs[broker_type]
                if config.enabled:
                    success = await broker.connect()
                    if success:
                        self.broker_health[broker_type][
                            "status"
                        ] = ConnectionStatus.CONNECTED
                        self.logger.info(f"Connected to {broker_type.value}")
                    else:
                        self.broker_health[broker_type][
                            "status"
                        ] = ConnectionStatus.ERROR
                        self.logger.error(f"Failed to connect to {broker_type.value}")

            # Start health monitoring
            asyncio.create_task(self._health_monitor_loop())

            self.logger.info("OrderRouter started")

        except Exception as e:
            self.logger.error("Failed to start OrderRouter", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the order router and disconnect from brokers."""
        try:
            for broker_type, broker in self.brokers.items():
                await broker.disconnect()
                self.broker_health[broker_type][
                    "status"
                ] = ConnectionStatus.DISCONNECTED

            self.logger.info("OrderRouter stopped")

        except Exception as e:
            self.logger.error("Error stopping OrderRouter", error=str(e))

    async def route_order(self, order: Order) -> BrokerOrder:
        """Route order to appropriate broker."""
        try:
            start_time = time.time()

            # Select broker for order
            broker_type = await self._select_broker(order)
            if not broker_type:
                raise Exception("No available broker for order")

            broker = self.brokers[broker_type]

            # Submit order to broker
            broker_order = await broker.submit_order(order)

            # Track routing
            self.routed_orders[order.order_id] = broker_order
            self.order_routing_map[order.order_id] = broker_type

            # Update statistics
            routing_time = (time.time() - start_time) * 1000
            self.routing_stats["orders_routed"] += 1
            self.routing_stats["avg_routing_time"] = (
                self.routing_stats["avg_routing_time"]
                * (self.routing_stats["orders_routed"] - 1)
                + routing_time
            ) / self.routing_stats["orders_routed"]

            self.broker_health[broker_type]["orders_processed"] += 1

            self.logger.info(
                "Order routed successfully",
                order_id=order.order_id,
                broker=broker_type.value,
                broker_order_id=broker_order.broker_order_id,
                routing_time_ms=routing_time,
            )

            return broker_order

        except Exception as e:
            self.routing_stats["orders_failed"] += 1
            self.logger.error(
                "Failed to route order", order_id=order.order_id, error=str(e)
            )
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a routed order."""
        try:
            if order_id not in self.order_routing_map:
                self.logger.warning("Order not found in routing map", order_id=order_id)
                return False

            broker_type = self.order_routing_map[order_id]
            broker_order = self.routed_orders[order_id]
            broker = self.brokers[broker_type]

            success = await broker.cancel_order(broker_order.broker_order_id)

            if success:
                broker_order.status = OrderStatus.CANCELLED
                self.logger.info(
                    "Order cancelled", order_id=order_id, broker=broker_type.value
                )

            return success

        except Exception as e:
            self.logger.error("Error cancelling order", order_id=order_id, error=str(e))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of a routed order."""
        try:
            if order_id not in self.order_routing_map:
                return OrderStatus.REJECTED

            broker_type = self.order_routing_map[order_id]
            broker_order = self.routed_orders[order_id]
            broker = self.brokers[broker_type]

            status = await broker.get_order_status(broker_order.broker_order_id)
            broker_order.status = status

            return status

        except Exception as e:
            self.logger.error(
                "Error getting order status", order_id=order_id, error=str(e)
            )
            return OrderStatus.REJECTED

    async def get_all_positions(self) -> dict[BrokerType, list[BrokerPosition]]:
        """Get positions from all connected brokers."""
        positions = {}

        for broker_type, broker in self.brokers.items():
            if self.broker_health[broker_type]["status"] == ConnectionStatus.CONNECTED:
                try:
                    broker_positions = await broker.get_positions()
                    positions[broker_type] = broker_positions
                except Exception as e:
                    self.logger.error(
                        f"Error getting positions from {broker_type.value}",
                        error=str(e),
                    )
                    positions[broker_type] = []

        return positions

    async def get_account_info(self, broker_type: BrokerType) -> dict[str, Any]:
        """Get account information from specific broker."""
        try:
            if broker_type not in self.brokers:
                return {}

            broker = self.brokers[broker_type]
            return await broker.get_account_info()

        except Exception as e:
            self.logger.error(
                f"Error getting account info from {broker_type.value}", error=str(e)
            )
            return {}

    async def _select_broker(self, order: Order) -> BrokerType | None:
        """Select the best broker for an order."""
        try:
            # Get available brokers
            available_brokers = []
            for broker_type, config in self.broker_configs.items():
                if (
                    config.enabled
                    and self.broker_health[broker_type]["status"]
                    == ConnectionStatus.CONNECTED
                ):
                    available_brokers.append(broker_type)

            if not available_brokers:
                return None

            # Simple selection by priority (could be enhanced with load balancing)
            available_brokers.sort(key=lambda bt: self.broker_configs[bt].priority)

            return available_brokers[0]

        except Exception as e:
            self.logger.error("Error selecting broker", error=str(e))
            return None

    async def _health_monitor_loop(self) -> None:
        """Monitor broker health continuously."""
        while True:
            try:
                for broker_type, broker in self.brokers.items():
                    health = self.broker_health[broker_type]

                    # Check connection status
                    is_connected = await broker.is_connected()

                    if is_connected:
                        health["status"] = ConnectionStatus.CONNECTED
                        health["error_count"] = 0
                    else:
                        health["status"] = ConnectionStatus.DISCONNECTED
                        health["error_count"] += 1

                        # Attempt reconnection if enabled
                        config = self.broker_configs[broker_type]
                        if (
                            config.enabled
                            and health["error_count"] <= config.retry_attempts
                        ):
                            self.logger.info(
                                f"Attempting to reconnect to {broker_type.value}"
                            )
                            await broker.connect()

                    health["last_check"] = datetime.now()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error("Error in health monitor", error=str(e))
                await asyncio.sleep(60)

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get order routing statistics."""
        return {
            "routing_stats": self.routing_stats.copy(),
            "broker_health": {
                bt.value: health.copy() for bt, health in self.broker_health.items()
            },
            "active_orders": len(self.routed_orders),
            "connected_brokers": [
                bt.value
                for bt, health in self.broker_health.items()
                if health["status"] == ConnectionStatus.CONNECTED
            ],
        }


def create_order_router(
    config_path: str = "config/execution_settings.yaml",
    publisher: ZMQPublisher | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> OrderRouter:
    """
    Factory function to create a configured order router.

    Args:
        config_path: Path to execution settings configuration file
        publisher: Optional message publisher for order updates
        logger: Optional logger instance

    Returns:
        Configured OrderRouter instance
    """
    return OrderRouter(config_path=config_path, publisher=publisher, logger=logger)
