"""
MetaTrader 5 Connector - Real-time Trading Integration.

This module provides comprehensive MT5 integration for the FX AI-Quant Trading System.
Handles connection management, order execution, position monitoring, and execution reporting.

Features:
- Automatic connection management with retry logic
- Support for all MT5 order types (buy, sell, modify, close, stop, limit)
- Real-time position and order monitoring
- Execution quality reporting and slippage tracking
- Demo and live account support
- Comprehensive error handling and logging
"""

import MetaTrader5 as mt5
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import yaml
from pathlib import Path

# Import our existing interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.interfaces.trading_interfaces import (
    Order, OrderSide, OrderType, OrderStatus, Position
)


class MT5OrderType(Enum):
    """MT5 specific order types."""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP


class MT5TradeAction(Enum):
    """MT5 trade actions."""
    DEAL = mt5.TRADE_ACTION_DEAL
    PENDING = mt5.TRADE_ACTION_PENDING
    SLTP = mt5.TRADE_ACTION_SLTP
    MODIFY = mt5.TRADE_ACTION_MODIFY
    REMOVE = mt5.TRADE_ACTION_REMOVE


class MT5ConnectionStatus(Enum):
    """MT5 connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class MT5Config:
    """MT5 configuration settings."""
    login: int
    password: str
    server: str
    path: Optional[str] = None
    timeout: int = 60000  # milliseconds
    portable: bool = False
    
    # Trading settings
    magic_number: int = 12345
    default_deviation: int = 20
    default_volume: float = 0.01
    max_volume: float = 10.0
    min_volume: float = 0.01
    
    # Connection settings
    max_retries: int = 5
    retry_delay: float = 5.0
    heartbeat_interval: float = 30.0
    
    # Symbol settings
    allowed_symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])


@dataclass
class MT5ExecutionResult:
    """MT5 execution result."""
    success: bool
    order_id: Optional[str] = None
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    slippage: Optional[float] = None
    execution_time: Optional[float] = None
    error_code: Optional[int] = None
    error_description: Optional[str] = None
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MT5Position:
    """MT5 position information."""
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    comment: str
    magic: int
    time: datetime
    identifier: int


class MT5Connector:
    """
    MetaTrader 5 Connector for real-time trading integration.
    
    Provides comprehensive MT5 integration with automatic connection management,
    order execution, position monitoring, and execution reporting.
    """
    
    def __init__(
        self,
        config: Optional[MT5Config] = None,
        config_path: str = "services/execution/config_mt5.yaml",
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        """Initialize MT5 connector."""
        self.config = config or self._load_config(config_path)
        self.logger = logger or structlog.get_logger(__name__)
        
        # Connection state
        self.status = MT5ConnectionStatus.DISCONNECTED
        self.last_heartbeat = None
        self.connection_attempts = 0
        self.last_error = None
        
        # Trading state
        self.active_orders: Dict[str, Any] = {}
        self.positions: Dict[str, MT5Position] = {}
        self.execution_history: List[MT5ExecutionResult] = []
        
        # Performance tracking
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.avg_execution_time = 0.0
        self.total_slippage = 0.0
        
        # Background tasks
        self._heartbeat_task = None
        self._monitor_task = None
        self._running = False
        
        self.logger.info(
            "MT5Connector initialized",
            login=self.config.login,
            server=self.config.server,
            magic_number=self.config.magic_number
        )
    
    def _load_config(self, config_path: str) -> MT5Config:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                return MT5Config(**config_data.get('mt5', {}))
            else:
                self.logger.warning("Config file not found, using defaults", path=config_path)
                return MT5Config(
                    login=0,  # Must be set
                    password="",  # Must be set
                    server=""  # Must be set
                )
        except Exception as e:
            self.logger.error("Error loading config", error=str(e))
            raise
    
    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            self.status = MT5ConnectionStatus.CONNECTING
            self.logger.info("Connecting to MT5", login=self.config.login, server=self.config.server)
            
            # Initialize MT5 (only path and portable parameters)
            init_params = {}
            if self.config.path:
                init_params['path'] = self.config.path
            if self.config.portable:
                init_params['portable'] = self.config.portable
            
            if not mt5.initialize(**init_params):
                error = mt5.last_error()
                self.last_error = f"MT5 initialization failed: {error}"
                self.logger.error("MT5 initialization failed", error=error)
                self.status = MT5ConnectionStatus.ERROR
                return False
            
            # Login to account
            if not mt5.login(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout
            ):
                error = mt5.last_error()
                self.last_error = f"MT5 login failed: {error}"
                self.logger.error("MT5 login failed", error=error)
                mt5.shutdown()
                self.status = MT5ConnectionStatus.ERROR
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                error = mt5.last_error()
                self.last_error = f"Failed to get account info: {error}"
                self.logger.error("Failed to get account info", error=error)
                mt5.shutdown()
                self.status = MT5ConnectionStatus.ERROR
                return False
            
            self.status = MT5ConnectionStatus.CONNECTED
            self.last_heartbeat = datetime.now()
            self.connection_attempts = 0
            
            self.logger.info(
                "MT5 connected successfully",
                account=account_info.login,
                balance=account_info.balance,
                equity=account_info.equity,
                server=account_info.server,
                company=account_info.company
            )
            
            # Start background tasks
            await self._start_background_tasks()
            
            return True
            
        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            self.logger.error("MT5 connection error", error=str(e))
            self.status = MT5ConnectionStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        try:
            self.logger.info("Disconnecting from MT5")
            self._running = False
            
            # Stop background tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._monitor_task:
                self._monitor_task.cancel()
            
            # Shutdown MT5
            mt5.shutdown()
            self.status = MT5ConnectionStatus.DISCONNECTED
            
            self.logger.info("MT5 disconnected")
            
        except Exception as e:
            self.logger.error("Error during disconnect", error=str(e))
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to monitor connection."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.status == MT5ConnectionStatus.CONNECTED:
                    # Check connection by getting account info
                    account_info = mt5.account_info()
                    if account_info is None:
                        self.logger.warning("Heartbeat failed, attempting reconnect")
                        await self._reconnect()
                    else:
                        self.last_heartbeat = datetime.now()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Heartbeat error", error=str(e))
    
    async def _monitor_loop(self) -> None:
        """Monitor positions and orders."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Monitor every second
                
                if self.status == MT5ConnectionStatus.CONNECTED:
                    await self._update_positions()
                    await self._update_orders()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Monitor error", error=str(e))
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect to MT5."""
        if self.connection_attempts >= self.config.max_retries:
            self.logger.error("Max reconnection attempts reached")
            self.status = MT5ConnectionStatus.ERROR
            return False
        
        self.status = MT5ConnectionStatus.RECONNECTING
        self.connection_attempts += 1
        
        self.logger.info(
            "Attempting reconnection",
            attempt=self.connection_attempts,
            max_retries=self.config.max_retries
        )
        
        await asyncio.sleep(self.config.retry_delay)
        return await self.connect()
    
    async def _update_positions(self) -> None:
        """Update current positions."""
        try:
            positions = mt5.positions_get()
            if positions is not None:
                current_positions = {}
                for pos in positions:
                    mt5_pos = MT5Position(
                        ticket=pos.ticket,
                        symbol=pos.symbol,
                        type=pos.type,
                        volume=pos.volume,
                        price_open=pos.price_open,
                        price_current=pos.price_current,
                        profit=pos.profit,
                        swap=pos.swap,
                        commission=pos.commission,
                        comment=pos.comment,
                        magic=pos.magic,
                        time=datetime.fromtimestamp(pos.time),
                        identifier=pos.identifier
                    )
                    current_positions[str(pos.ticket)] = mt5_pos
                
                self.positions = current_positions
                
        except Exception as e:
            self.logger.error("Error updating positions", error=str(e))
    
    async def _update_orders(self) -> None:
        """Update pending orders."""
        try:
            orders = mt5.orders_get()
            if orders is not None:
                for order in orders:
                    order_id = str(order.ticket)
                    if order_id in self.active_orders:
                        # Update existing order
                        self.active_orders[order_id].update({
                            'current_price': order.price_current,
                            'state': order.state,
                            'last_update': datetime.now()
                        })
                        
        except Exception as e:
            self.logger.error("Error updating orders", error=str(e))
    
    async def submit_order(self, order: Order) -> MT5ExecutionResult:
        """Submit an order to MT5."""
        start_time = time.perf_counter()
        
        try:
            self.total_orders += 1
            
            # Validate connection
            if self.status != MT5ConnectionStatus.CONNECTED:
                return MT5ExecutionResult(
                    success=False,
                    error_description="MT5 not connected"
                )
            
            # Validate symbol
            if order.symbol not in self.config.allowed_symbols:
                return MT5ExecutionResult(
                    success=False,
                    error_description=f"Symbol {order.symbol} not allowed"
                )
            
            # Get symbol info
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                return MT5ExecutionResult(
                    success=False,
                    error_description=f"Symbol {order.symbol} not found"
                )
            
            # Prepare trade request
            request = self._prepare_trade_request(order, symbol_info)
            
            self.logger.info(
                "Submitting order to MT5",
                order_id=order.order_id,
                symbol=order.symbol,
                type=order.order_type.value,
                volume=order.quantity,
                price=order.price
            )
            
            # Send order
            result = mt5.order_send(request)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            if result is None:
                error = mt5.last_error()
                self.failed_orders += 1
                return MT5ExecutionResult(
                    success=False,
                    execution_time=execution_time,
                    error_code=error[0] if error else None,
                    error_description=error[1] if error else "Unknown error"
                )
            
            # Process result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.successful_orders += 1
                
                # Calculate slippage
                slippage = 0.0
                if order.price and result.price:
                    slippage = abs(result.price - order.price) / order.price * 10000  # bps
                    self.total_slippage += slippage
                
                # Update average execution time
                self.avg_execution_time = (
                    (self.avg_execution_time * (self.successful_orders - 1) + execution_time) 
                    / self.successful_orders
                )
                
                # Store order info
                self.active_orders[order.order_id] = {
                    'ticket': result.order,
                    'symbol': order.symbol,
                    'volume': result.volume,
                    'price': result.price,
                    'type': order.order_type.value,
                    'time': datetime.now(),
                    'magic': self.config.magic_number
                }
                
                execution_result = MT5ExecutionResult(
                    success=True,
                    order_id=order.order_id,
                    ticket=result.order,
                    price=result.price,
                    volume=result.volume,
                    slippage=slippage,
                    execution_time=execution_time,
                    comment=result.comment
                )
                
                self.execution_history.append(execution_result)
                
                self.logger.info(
                    "Order executed successfully",
                    order_id=order.order_id,
                    ticket=result.order,
                    price=result.price,
                    volume=result.volume,
                    slippage_bps=slippage,
                    execution_time_ms=execution_time
                )
                
                return execution_result
                
            else:
                self.failed_orders += 1
                return MT5ExecutionResult(
                    success=False,
                    execution_time=execution_time,
                    error_code=result.retcode,
                    error_description=self._get_error_description(result.retcode),
                    comment=result.comment
                )
                
        except Exception as e:
            self.failed_orders += 1
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error("Order submission error", order_id=order.order_id, error=str(e))
            return MT5ExecutionResult(
                success=False,
                execution_time=execution_time,
                error_description=str(e)
            )
    
    def _prepare_trade_request(self, order: Order, symbol_info) -> Dict[str, Any]:
        """Prepare MT5 trade request."""
        # Convert order type
        if order.order_type == OrderType.MARKET:
            action = MT5TradeAction.DEAL.value
            order_type = MT5OrderType.BUY.value if order.side == OrderSide.BUY else MT5OrderType.SELL.value
            price = symbol_info.ask if order.side == OrderSide.BUY else symbol_info.bid
        elif order.order_type == OrderType.LIMIT:
            action = MT5TradeAction.PENDING.value
            if order.side == OrderSide.BUY:
                order_type = MT5OrderType.BUY_LIMIT.value
            else:
                order_type = MT5OrderType.SELL_LIMIT.value
            price = order.price
        elif order.order_type == OrderType.STOP:
            action = MT5TradeAction.PENDING.value
            if order.side == OrderSide.BUY:
                order_type = MT5OrderType.BUY_STOP.value
            else:
                order_type = MT5OrderType.SELL_STOP.value
            price = order.stop_price or order.price
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
        
        # Calculate volume (convert from units to lots)
        volume = min(max(order.quantity / 100000, self.config.min_volume), self.config.max_volume)
        
        request = {
            "action": action,
            "symbol": order.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.config.default_deviation,
            "magic": self.config.magic_number,
            "comment": f"AI-Quant-{order.order_id[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add stop loss and take profit if specified
        if hasattr(order, 'stop_loss') and order.stop_loss:
            request["sl"] = order.stop_loss
        if hasattr(order, 'take_profit') and order.take_profit:
            request["tp"] = order.take_profit
        
        return request
    
    def _get_error_description(self, retcode: int) -> str:
        """Get human-readable error description."""
        error_codes = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Request partially completed",
            mt5.TRADE_RETCODE_ERROR: "Common error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "No money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "Off quotes",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid expiration",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
            mt5.TRADE_RETCODE_NO_CHANGES: "No changes",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "Autotrading disabled by server",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "Autotrading disabled by client",
            mt5.TRADE_RETCODE_LOCKED: "Request locked",
            mt5.TRADE_RETCODE_FROZEN: "Order or position frozen",
            mt5.TRADE_RETCODE_INVALID_FILL: "Invalid fill",
            mt5.TRADE_RETCODE_CONNECTION: "No connection",
            mt5.TRADE_RETCODE_ONLY_REAL: "Only real accounts allowed",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Limit orders limit reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
        }
        return error_codes.get(retcode, f"Unknown error code: {retcode}")
    
    async def cancel_order(self, order_id: str) -> MT5ExecutionResult:
        """Cancel a pending order."""
        try:
            if order_id not in self.active_orders:
                return MT5ExecutionResult(
                    success=False,
                    error_description="Order not found"
                )
            
            order_info = self.active_orders[order_id]
            ticket = order_info['ticket']
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                del self.active_orders[order_id]
                self.logger.info("Order cancelled", order_id=order_id, ticket=ticket)
                return MT5ExecutionResult(success=True, ticket=ticket)
            else:
                error_desc = self._get_error_description(result.retcode) if result else "Unknown error"
                return MT5ExecutionResult(
                    success=False,
                    error_code=result.retcode if result else None,
                    error_description=error_desc
                )
                
        except Exception as e:
            self.logger.error("Cancel order error", order_id=order_id, error=str(e))
            return MT5ExecutionResult(success=False, error_description=str(e))
    
    async def modify_order(self, order_id: str, new_price: float, new_sl: Optional[float] = None, new_tp: Optional[float] = None) -> MT5ExecutionResult:
        """Modify a pending order."""
        try:
            if order_id not in self.active_orders:
                return MT5ExecutionResult(
                    success=False,
                    error_description="Order not found"
                )
            
            order_info = self.active_orders[order_id]
            ticket = order_info['ticket']
            
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "price": new_price,
            }
            
            if new_sl is not None:
                request["sl"] = new_sl
            if new_tp is not None:
                request["tp"] = new_tp
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update stored order info
                order_info['price'] = new_price
                self.logger.info("Order modified", order_id=order_id, ticket=ticket, new_price=new_price)
                return MT5ExecutionResult(success=True, ticket=ticket, price=new_price)
            else:
                error_desc = self._get_error_description(result.retcode) if result else "Unknown error"
                return MT5ExecutionResult(
                    success=False,
                    error_code=result.retcode if result else None,
                    error_description=error_desc
                )
                
        except Exception as e:
            self.logger.error("Modify order error", order_id=order_id, error=str(e))
            return MT5ExecutionResult(success=False, error_description=str(e))
    
    async def close_position(self, symbol: str, volume: Optional[float] = None) -> MT5ExecutionResult:
        """Close a position."""
        try:
            # Get position info
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return MT5ExecutionResult(
                    success=False,
                    error_description=f"No position found for {symbol}"
                )
            
            position = positions[0]  # Get first position
            
            # Determine close parameters
            close_volume = volume or position.volume
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return MT5ExecutionResult(
                    success=False,
                    error_description=f"Symbol {symbol} not found"
                )
            
            price = symbol_info.bid if close_type == mt5.ORDER_TYPE_SELL else symbol_info.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": close_volume,
                "type": close_type,
                "position": position.ticket,
                "price": price,
                "deviation": self.config.default_deviation,
                "magic": self.config.magic_number,
                "comment": f"Close-{position.ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    "Position closed",
                    symbol=symbol,
                    ticket=position.ticket,
                    volume=close_volume,
                    price=result.price
                )
                return MT5ExecutionResult(
                    success=True,
                    ticket=result.order,
                    price=result.price,
                    volume=result.volume
                )
            else:
                error_desc = self._get_error_description(result.retcode) if result else "Unknown error"
                return MT5ExecutionResult(
                    success=False,
                    error_code=result.retcode if result else None,
                    error_description=error_desc
                )
                
        except Exception as e:
            self.logger.error("Close position error", symbol=symbol, error=str(e))
            return MT5ExecutionResult(success=False, error_description=str(e))
    
    def get_positions(self) -> List[MT5Position]:
        """Get all current positions."""
        return list(self.positions.values())
    
    def get_position(self, symbol: str) -> Optional[MT5Position]:
        """Get position for specific symbol."""
        for position in self.positions.values():
            if position.symbol == symbol:
                return position
        return None
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders."""
        try:
            orders = mt5.orders_get()
            if orders is None:
                return []
            
            order_list = []
            for order in orders:
                order_dict = {
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': order.type,
                    'volume': order.volume,
                    'price_open': order.price_open,
                    'price_current': order.price_current,
                    'sl': order.sl,
                    'tp': order.tp,
                    'time_setup': order.time_setup,
                    'state': order.state,
                    'magic': order.magic,
                    'comment': order.comment
                }
                order_list.append(order_dict)
            
            return order_list
            
        except Exception as e:
            self.logger.error("Error getting orders", error=str(e))
            return []
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit,
                'server': account_info.server,
                'company': account_info.company,
                'currency': account_info.currency,
                'leverage': account_info.leverage,
                'trade_allowed': account_info.trade_allowed,
                'trade_expert': account_info.trade_expert
            }
            
        except Exception as e:
            self.logger.error("Error getting account info", error=str(e))
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_mode': symbol_info.trade_mode,
                'trade_allowed': symbol_info.visible,
                'margin_initial': symbol_info.margin_initial,
                'margin_maintenance': symbol_info.margin_maintenance
            }
            
        except Exception as e:
            self.logger.error("Error getting symbol info", symbol=symbol, error=str(e))
            return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        success_rate = (self.successful_orders / self.total_orders * 100) if self.total_orders > 0 else 0
        avg_slippage = (self.total_slippage / self.successful_orders) if self.successful_orders > 0 else 0
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'success_rate': success_rate,
            'avg_execution_time_ms': self.avg_execution_time,
            'avg_slippage_bps': avg_slippage,
            'connection_status': self.status.value,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'connection_attempts': self.connection_attempts,
            'active_orders_count': len(self.active_orders),
            'active_positions_count': len(self.positions)
        }
    
    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self.status == MT5ConnectionStatus.CONNECTED
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for symbol."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            # Check if trading is allowed
            return symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED
            
        except Exception as e:
            self.logger.error("Error checking market status", symbol=symbol, error=str(e))
            return False


def create_mt5_connector(
    config_path: str = "services/execution/config_mt5.yaml",
    logger: Optional[structlog.stdlib.BoundLogger] = None
) -> MT5Connector:
    """Factory function to create MT5 connector."""
    return MT5Connector(config_path=config_path, logger=logger) 