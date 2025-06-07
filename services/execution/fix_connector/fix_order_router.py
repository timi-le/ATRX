"""
FIX Order Router - Integration with Execution Engine.

This module provides the order routing layer between the execution engine
and the FIX client, handling order translation, execution reporting,
and status updates.

Features:
- Order translation from internal format to FIX messages
- Execution report processing and status updates
- Order lifecycle management
- Real-time position tracking
- Comprehensive error handling and reporting
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

# Import our existing interfaces
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.interfaces.trading_interfaces import (
    Order, OrderSide, OrderType, OrderStatus, Position, BrokerInterface
)
from .fix_client import FIXClient, FIXSessionConfig, load_fix_config
from .fix_message_utils import FIXMessage, FIXMessageUtils


@dataclass
class FIXOrderMapping:
    """Mapping between internal order and FIX order."""
    internal_order_id: str
    fix_client_order_id: str
    fix_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    order_type: str = ""
    price: Optional[float] = None
    status: str = "NEW"
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_price: float = 0.0
    last_update: Optional[datetime] = None
    execution_reports: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FIXExecutionReport:
    """FIX execution report data."""
    exec_id: str
    order_id: str
    cl_ord_id: str
    exec_type: str
    ord_status: str
    symbol: str
    side: str
    last_qty: float = 0.0
    last_px: float = 0.0
    leaves_qty: float = 0.0
    cum_qty: float = 0.0
    avg_px: float = 0.0
    transact_time: Optional[datetime] = None
    text: Optional[str] = None


class FIXOrderRouter(BrokerInterface):
    """
    FIX Order Router for IBKR integration.
    
    Provides order routing between the execution engine and IBKR via FIX protocol,
    handling order translation, execution reporting, and status management.
    """
    
    def __init__(
        self,
        config_path: str = "services/execution/fix_connector/fix_session_config.yaml",
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        """Initialize FIX order router."""
        self.logger = logger or structlog.get_logger(__name__)
        
        # Load configuration
        self.config = load_fix_config(config_path)
        
        # Initialize FIX client
        self.fix_client = FIXClient(self.config, self.logger)
        
        # Order management
        self.active_orders: Dict[str, FIXOrderMapping] = {}
        self.order_id_mapping: Dict[str, str] = {}  # FIX order ID -> internal order ID
        self.client_order_id_mapping: Dict[str, str] = {}  # FIX client order ID -> internal order ID
        
        # Message utilities
        self.message_utils = self.fix_client.message_utils
        
        # Callbacks
        self.order_update_callback: Optional[Callable[[str, OrderStatus, Dict[str, Any]], None]] = None
        self.execution_callback: Optional[Callable[[str, float, float, datetime], None]] = None
        self.error_callback: Optional[Callable[[str, str], None]] = None
        
        # Statistics
        self.total_orders_sent = 0
        self.total_executions_received = 0
        self.total_rejections = 0
        
        # Setup FIX message handlers
        self._setup_message_handlers()
        
        self.logger.info("FIX order router initialized")
    
    def _setup_message_handlers(self):
        """Setup FIX message handlers."""
        # Register execution report handler
        self.fix_client.register_message_handler("8", self._handle_execution_report)
        
        # Register order cancel reject handler
        self.fix_client.register_message_handler("9", self._handle_order_cancel_reject)
        
        # Register business message reject handler
        self.fix_client.register_message_handler("j", self._handle_business_message_reject)
    
    async def connect(self) -> bool:
        """Connect to IBKR via FIX protocol."""
        try:
            self.logger.info("Connecting to IBKR FIX gateway")
            success = await self.fix_client.connect()
            
            if success:
                self.logger.info("FIX connection established successfully")
                return True
            else:
                self.logger.error("Failed to establish FIX connection")
                return False
                
        except Exception as e:
            self.logger.error("Error connecting to FIX gateway", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from IBKR FIX gateway."""
        try:
            self.logger.info("Disconnecting from IBKR FIX gateway")
            await self.fix_client.disconnect()
            self.logger.info("FIX connection closed")
            
        except Exception as e:
            self.logger.error("Error disconnecting from FIX gateway", error=str(e))
    
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Submit an order via FIX protocol."""
        try:
            self.logger.info(
                "Submitting order via FIX",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value
            )
            
            # Generate FIX client order ID
            fix_client_order_id = self.message_utils.generate_client_order_id()
            
            # Convert internal order to FIX format
            fix_side = self._convert_side_to_fix(order.side)
            fix_order_type = self._convert_order_type_to_fix(order.order_type)
            fix_time_in_force = self._convert_time_in_force_to_fix(getattr(order, 'time_in_force', 'DAY'))
            
            # Create order mapping
            order_mapping = FIXOrderMapping(
                internal_order_id=order.order_id,
                fix_client_order_id=fix_client_order_id,
                symbol=order.symbol,
                side=fix_side,
                quantity=order.quantity,
                order_type=fix_order_type,
                price=order.price,
                status="PENDING_NEW",
                remaining_quantity=order.quantity,
                last_update=datetime.now(timezone.utc)
            )
            
            # Store order mapping
            self.active_orders[order.order_id] = order_mapping
            self.client_order_id_mapping[fix_client_order_id] = order.order_id
            
            # Build FIX order message
            message = self.message_utils.build_order_single_message(
                sender_comp_id=self.config.sender_comp_id,
                target_comp_id=self.config.target_comp_id,
                sequence_number=self.fix_client.outgoing_seq_num,
                cl_ord_id=fix_client_order_id,
                symbol=order.symbol,
                side=fix_side,
                order_qty=order.quantity,
                ord_type=fix_order_type,
                price=order.price if order.order_type != OrderType.MARKET else None,
                stop_px=getattr(order, 'stop_price', None),
                time_in_force=fix_time_in_force,
                currency=getattr(order, 'currency', 'USD'),
                exchange=getattr(order, 'exchange', 'SMART'),
                sender_sub_id=self.config.sender_sub_id,
                target_sub_id=self.config.target_sub_id
            )
            
            # Send order
            success = await self.fix_client.send_message(message)
            
            if success:
                self.fix_client.outgoing_seq_num += 1
                self.total_orders_sent += 1
                
                self.logger.info(
                    "Order submitted successfully",
                    order_id=order.order_id,
                    fix_client_order_id=fix_client_order_id
                )
                
                return {
                    "success": True,
                    "order_id": order.order_id,
                    "fix_client_order_id": fix_client_order_id,
                    "message": "Order submitted successfully"
                }
            else:
                # Remove failed order from tracking
                del self.active_orders[order.order_id]
                del self.client_order_id_mapping[fix_client_order_id]
                
                return {
                    "success": False,
                    "order_id": order.order_id,
                    "error": "Failed to send FIX message"
                }
                
        except Exception as e:
            self.logger.error("Error submitting order", order_id=order.order_id, error=str(e))
            return {
                "success": False,
                "order_id": order.order_id,
                "error": str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order via FIX protocol."""
        try:
            if order_id not in self.active_orders:
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": "Order not found"
                }
            
            order_mapping = self.active_orders[order_id]
            
            self.logger.info(
                "Cancelling order via FIX",
                order_id=order_id,
                fix_client_order_id=order_mapping.fix_client_order_id
            )
            
            # Generate new client order ID for cancel request
            cancel_client_order_id = self.message_utils.generate_client_order_id()
            
            # Build cancel request message
            message = self.message_utils.build_order_cancel_request_message(
                sender_comp_id=self.config.sender_comp_id,
                target_comp_id=self.config.target_comp_id,
                sequence_number=self.fix_client.outgoing_seq_num,
                orig_cl_ord_id=order_mapping.fix_client_order_id,
                cl_ord_id=cancel_client_order_id,
                symbol=order_mapping.symbol,
                side=order_mapping.side,
                order_qty=order_mapping.quantity,
                sender_sub_id=self.config.sender_sub_id,
                target_sub_id=self.config.target_sub_id
            )
            
            # Send cancel request
            success = await self.fix_client.send_message(message)
            
            if success:
                self.fix_client.outgoing_seq_num += 1
                order_mapping.status = "PENDING_CANCEL"
                order_mapping.last_update = datetime.now(timezone.utc)
                
                self.logger.info(
                    "Cancel request submitted",
                    order_id=order_id,
                    cancel_client_order_id=cancel_client_order_id
                )
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": "Cancel request submitted"
                }
            else:
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": "Failed to send cancel request"
                }
                
        except Exception as e:
            self.logger.error("Error cancelling order", order_id=order_id, error=str(e))
            return {
                "success": False,
                "order_id": order_id,
                "error": str(e)
            }
    
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Modify an order (not directly supported in basic FIX, requires cancel/replace)."""
        try:
            if order_id not in self.active_orders:
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": "Order not found"
                }
            
            # For now, return not supported
            # Full implementation would require Order Cancel/Replace Request (35=G)
            return {
                "success": False,
                "order_id": order_id,
                "error": "Order modification not implemented - use cancel and resubmit"
            }
            
        except Exception as e:
            self.logger.error("Error modifying order", order_id=order_id, error=str(e))
            return {
                "success": False,
                "order_id": order_id,
                "error": str(e)
            }
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        if order_id not in self.active_orders:
            return None
        
        order_mapping = self.active_orders[order_id]
        
        return {
            "order_id": order_id,
            "fix_client_order_id": order_mapping.fix_client_order_id,
            "fix_order_id": order_mapping.fix_order_id,
            "symbol": order_mapping.symbol,
            "side": order_mapping.side,
            "quantity": order_mapping.quantity,
            "order_type": order_mapping.order_type,
            "price": order_mapping.price,
            "status": order_mapping.status,
            "filled_quantity": order_mapping.filled_quantity,
            "remaining_quantity": order_mapping.remaining_quantity,
            "avg_price": order_mapping.avg_price,
            "last_update": order_mapping.last_update.isoformat() if order_mapping.last_update else None,
            "execution_count": len(order_mapping.execution_reports)
        }
    
    def get_all_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders."""
        return [self.get_order_status(order_id) for order_id in self.active_orders.keys()]
    
    def get_positions(self) -> List[Position]:
        """Get current positions (calculated from executions)."""
        # This is a simplified implementation
        # In practice, you might want to request position reports from IBKR
        positions = {}
        
        for order_mapping in self.active_orders.values():
            symbol = order_mapping.symbol
            
            if symbol not in positions:
                positions[symbol] = {
                    "symbol": symbol,
                    "quantity": 0.0,
                    "avg_price": 0.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0
                }
            
            # Calculate position from filled orders
            if order_mapping.filled_quantity > 0:
                side_multiplier = 1 if order_mapping.side == "1" else -1  # 1=Buy, 2=Sell
                quantity_change = order_mapping.filled_quantity * side_multiplier
                
                positions[symbol]["quantity"] += quantity_change
                # Simplified avg price calculation
                if positions[symbol]["quantity"] != 0:
                    positions[symbol]["avg_price"] = order_mapping.avg_price
        
        # Convert to Position objects
        position_list = []
        for pos_data in positions.values():
            if pos_data["quantity"] != 0:  # Only include non-zero positions
                position = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    avg_price=pos_data["avg_price"],
                    unrealized_pnl=pos_data["unrealized_pnl"],
                    realized_pnl=pos_data["realized_pnl"]
                )
                position_list.append(position)
        
        return position_list
    
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self.fix_client.is_connected()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        # Return FIX session stats as account info
        stats = self.fix_client.get_session_stats()
        
        return {
            "broker": "IBKR",
            "connection_status": stats["state"],
            "session_start_time": stats["session_start_time"],
            "total_orders_sent": self.total_orders_sent,
            "total_executions_received": self.total_executions_received,
            "total_rejections": self.total_rejections,
            "active_orders_count": len(self.active_orders),
            "fix_stats": stats
        }
    
    # FIX message handlers
    def _handle_execution_report(self, message: FIXMessage):
        """Handle execution report from IBKR."""
        try:
            # Parse execution report
            exec_report = self._parse_execution_report(message)
            
            self.logger.info(
                "Received execution report",
                exec_id=exec_report.exec_id,
                cl_ord_id=exec_report.cl_ord_id,
                exec_type=exec_report.exec_type,
                ord_status=exec_report.ord_status,
                symbol=exec_report.symbol
            )
            
            # Find corresponding internal order
            internal_order_id = self.client_order_id_mapping.get(exec_report.cl_ord_id)
            if not internal_order_id:
                self.logger.warning("Received execution report for unknown order", cl_ord_id=exec_report.cl_ord_id)
                return
            
            order_mapping = self.active_orders.get(internal_order_id)
            if not order_mapping:
                self.logger.warning("Order mapping not found", order_id=internal_order_id)
                return
            
            # Update order mapping
            order_mapping.fix_order_id = exec_report.order_id
            order_mapping.status = exec_report.ord_status
            order_mapping.filled_quantity = exec_report.cum_qty
            order_mapping.remaining_quantity = exec_report.leaves_qty
            order_mapping.avg_price = exec_report.avg_px
            order_mapping.last_update = datetime.now(timezone.utc)
            
            # Store execution report
            order_mapping.execution_reports.append({
                "exec_id": exec_report.exec_id,
                "exec_type": exec_report.exec_type,
                "last_qty": exec_report.last_qty,
                "last_px": exec_report.last_px,
                "transact_time": exec_report.transact_time.isoformat() if exec_report.transact_time else None,
                "text": exec_report.text
            })
            
            # Update statistics
            if exec_report.exec_type in ["1", "2", "F"]:  # Partial fill, Fill, Trade
                self.total_executions_received += 1
            elif exec_report.exec_type == "8":  # Rejected
                self.total_rejections += 1
            
            # Map FIX order status to internal status
            internal_status = self._convert_fix_status_to_internal(exec_report.ord_status)
            
            # Call order update callback
            if self.order_update_callback:
                self.order_update_callback(
                    internal_order_id,
                    internal_status,
                    {
                        "filled_quantity": exec_report.cum_qty,
                        "remaining_quantity": exec_report.leaves_qty,
                        "avg_price": exec_report.avg_px,
                        "last_fill_qty": exec_report.last_qty,
                        "last_fill_price": exec_report.last_px,
                        "exec_id": exec_report.exec_id,
                        "text": exec_report.text
                    }
                )
            
            # Call execution callback for fills
            if exec_report.last_qty > 0 and self.execution_callback:
                self.execution_callback(
                    internal_order_id,
                    exec_report.last_qty,
                    exec_report.last_px,
                    exec_report.transact_time or datetime.now(timezone.utc)
                )
            
            # Remove completed orders from tracking
            if exec_report.ord_status in ["2", "4", "8", "C"]:  # Filled, Canceled, Rejected, Expired
                if exec_report.order_id in self.order_id_mapping:
                    del self.order_id_mapping[exec_report.order_id]
                if exec_report.cl_ord_id in self.client_order_id_mapping:
                    del self.client_order_id_mapping[exec_report.cl_ord_id]
                # Keep in active_orders for historical reference
            
        except Exception as e:
            self.logger.error("Error handling execution report", error=str(e))
    
    def _handle_order_cancel_reject(self, message: FIXMessage):
        """Handle order cancel reject."""
        try:
            cl_ord_id = message.fields.get(11, "")
            orig_cl_ord_id = message.fields.get(41, "")
            text = message.fields.get(58, "")
            
            self.logger.warning(
                "Order cancel rejected",
                cl_ord_id=cl_ord_id,
                orig_cl_ord_id=orig_cl_ord_id,
                text=text
            )
            
            # Find internal order
            internal_order_id = self.client_order_id_mapping.get(orig_cl_ord_id)
            if internal_order_id and self.error_callback:
                self.error_callback(internal_order_id, f"Cancel rejected: {text}")
                
        except Exception as e:
            self.logger.error("Error handling order cancel reject", error=str(e))
    
    def _handle_business_message_reject(self, message: FIXMessage):
        """Handle business message reject."""
        try:
            ref_msg_type = message.fields.get(372, "")
            text = message.fields.get(58, "")
            business_reject_reason = message.fields.get(380, "")
            
            self.logger.error(
                "Business message rejected",
                ref_msg_type=ref_msg_type,
                text=text,
                reason=business_reject_reason
            )
            
            if self.error_callback:
                self.error_callback("", f"Business reject: {text}")
                
        except Exception as e:
            self.logger.error("Error handling business message reject", error=str(e))
    
    def _parse_execution_report(self, message: FIXMessage) -> FIXExecutionReport:
        """Parse FIX execution report message."""
        fields = message.fields
        
        # Parse timestamp
        transact_time = None
        if 60 in fields:  # TransactTime
            transact_time = self.message_utils._parse_timestamp(fields[60])
        
        return FIXExecutionReport(
            exec_id=fields.get(17, ""),
            order_id=fields.get(37, ""),
            cl_ord_id=fields.get(11, ""),
            exec_type=fields.get(150, ""),
            ord_status=fields.get(39, ""),
            symbol=fields.get(55, ""),
            side=fields.get(54, ""),
            last_qty=float(fields.get(32, "0")),
            last_px=float(fields.get(31, "0")),
            leaves_qty=float(fields.get(151, "0")),
            cum_qty=float(fields.get(14, "0")),
            avg_px=float(fields.get(6, "0")),
            transact_time=transact_time,
            text=fields.get(58)
        )
    
    # Conversion utilities
    def _convert_side_to_fix(self, side: OrderSide) -> str:
        """Convert internal side to FIX side."""
        mapping = {
            OrderSide.BUY: "1",
            OrderSide.SELL: "2"
        }
        return mapping.get(side, "1")
    
    def _convert_order_type_to_fix(self, order_type: OrderType) -> str:
        """Convert internal order type to FIX order type."""
        mapping = {
            OrderType.MARKET: "1",
            OrderType.LIMIT: "2",
            OrderType.STOP: "3",
            OrderType.STOP_LIMIT: "4"
        }
        return mapping.get(order_type, "1")
    
    def _convert_time_in_force_to_fix(self, tif: str) -> str:
        """Convert time in force to FIX format."""
        mapping = {
            "DAY": "0",
            "GTC": "1",
            "IOC": "3",
            "FOK": "4"
        }
        return mapping.get(tif.upper(), "0")
    
    def _convert_fix_status_to_internal(self, fix_status: str) -> OrderStatus:
        """Convert FIX order status to internal status."""
        mapping = {
            "0": OrderStatus.PENDING,      # New
            "1": OrderStatus.PARTIALLY_FILLED,  # Partially filled
            "2": OrderStatus.FILLED,       # Filled
            "4": OrderStatus.CANCELLED,    # Canceled
            "6": OrderStatus.PENDING,      # Pending Cancel
            "8": OrderStatus.REJECTED,     # Rejected
            "A": OrderStatus.PENDING,      # Pending New
            "C": OrderStatus.CANCELLED,    # Expired
            "E": OrderStatus.PENDING,      # Pending Replace
        }
        return mapping.get(fix_status, OrderStatus.PENDING)
    
    # Callback registration
    def set_order_update_callback(self, callback: Callable[[str, OrderStatus, Dict[str, Any]], None]):
        """Set order update callback."""
        self.order_update_callback = callback
    
    def set_execution_callback(self, callback: Callable[[str, float, float, datetime], None]):
        """Set execution callback."""
        self.execution_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        """Set error callback."""
        self.error_callback = callback


# Factory function
def create_fix_order_router(
    config_path: str = "services/execution/fix_connector/fix_session_config.yaml"
) -> FIXOrderRouter:
    """Create FIX order router instance."""
    return FIXOrderRouter(config_path) 