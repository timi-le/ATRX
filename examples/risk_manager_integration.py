"""
Risk Manager Integration Example

This example demonstrates how to integrate the Risk Manager with the existing
FX AI-Quant Trading System components including the Strategy Switcher and
Position Sizer.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from core.risk_manager import create_risk_manager, RiskLevel
from core.position_sizer import create_position_sizer, TradeSignalInput, PortfolioState
from core.strategy_switcher import StrategySwitcher
from core.interfaces.trading_interfaces import Order, Position, OrderSide, OrderType, Signal
from core.pubsub import ZMQPublisher


class TradingSystemWithRiskManager:
    """
    Example trading system integration with comprehensive risk management.
    
    This class demonstrates how to integrate the Risk Manager with other
    system components to create a complete trading system with risk controls.
    """
    
    def __init__(self):
        # Initialize components
        self.risk_manager = create_risk_manager(
            config_path="config/risk_limits.yaml",
            initial_capital=100000.0
        )
        
        self.position_sizer = create_position_sizer(
            config_path="config/risk_settings.yaml"
        )
        
        self.strategy_switcher = StrategySwitcher()
        
        # Initialize publisher for alerts
        self.publisher = ZMQPublisher()
        
        # System state
        self.current_positions: Dict[str, Position] = {}
        self.account_balance = 100000.0
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the trading system."""
        try:
            await self.publisher.start()
            self.is_running = True
            self.logger.info("Trading system with risk management started")
            
            # Start monitoring tasks
            asyncio.create_task(self._risk_monitoring_loop())
            asyncio.create_task(self._position_monitoring_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            raise
    
    async def stop(self):
        """Stop the trading system."""
        self.is_running = False
        await self.publisher.stop()
        self.logger.info("Trading system stopped")
    
    async def process_signal(self, signal: Signal) -> bool:
        """
        Process a trading signal with full risk management.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            bool: True if trade was executed, False if rejected
        """
        try:
            self.logger.info(f"Processing signal: {signal.symbol} {signal.side.value}")
            
            # Step 1: Create position sizing input
            sizing_input = TradeSignalInput(
                symbol=signal.symbol,
                side=signal.side,
                signal_confidence=signal.confidence,
                take_profit_pips=None,  # Would be calculated from strategy
                stop_loss_pips=None,    # Would be calculated from strategy
                reward_risk_ratio=1.5,  # Default ratio
                win_probability=None,   # Would come from ML model
                current_price=1.1000,   # Would come from market data
                volatility_atr=0.0015,  # Would come from market data
                timestamp=datetime.now(),
                strategy_name=signal.strategy_name,
                features=signal.features
            )
            
            # Step 2: Calculate position size
            portfolio_state = PortfolioState(
                total_capital=self.account_balance,
                current_drawdown=0.0,  # Would be calculated from P&L
                daily_pnl=0.0,         # Would be tracked
                open_positions=[],      # Would be converted from current_positions
                volatility_history=[],  # Would be maintained
                performance_history=[]  # Would be maintained
            )
            
            sizing_result = await self.position_sizer.calculate_position_size(
                sizing_input, portfolio_state
            )
            
            if sizing_result.position_size <= 0:
                self.logger.warning("Position sizer rejected trade - zero size")
                return False
            
            # Step 3: Create order
            order = Order(
                order_id=f"order_{datetime.now().timestamp()}",
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                quantity=sizing_result.position_size * self.account_balance,  # Convert % to absolute
                price=1.1000  # Would come from market data
            )
            
            # Step 4: Pre-trade risk check
            risk_approved = await self.risk_manager.check_pre_trade_risk(
                order, self.current_positions, self.account_balance
            )
            
            if not risk_approved:
                self.logger.warning(f"Risk manager rejected trade: {order.symbol}")
                await self._send_risk_alert(
                    RiskLevel.WARNING,
                    f"Trade rejected by risk manager: {order.symbol} {order.side.value}"
                )
                return False
            
            # Step 5: Execute trade (simulated)
            success = await self._execute_order(order)
            
            if success:
                self.logger.info(f"Trade executed successfully: {order.order_id}")
                return True
            else:
                self.logger.error(f"Trade execution failed: {order.order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return False
    
    async def _execute_order(self, order: Order) -> bool:
        """
        Simulate order execution and update positions.
        
        Args:
            order: Order to execute
            
        Returns:
            bool: True if execution successful
        """
        try:
            # Simulate execution
            if order.symbol in self.current_positions:
                # Update existing position
                position = self.current_positions[order.symbol]
                if order.side == OrderSide.BUY:
                    new_quantity = position.quantity + order.quantity
                else:
                    new_quantity = position.quantity - order.quantity
                
                if new_quantity == 0:
                    # Close position
                    del self.current_positions[order.symbol]
                else:
                    # Update position
                    position.quantity = new_quantity
                    # Would update avg_price based on execution price
            else:
                # Create new position
                quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
                self.current_positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=quantity,
                    avg_price=order.price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                )
            
            # Update risk manager with new positions
            await self.risk_manager.update_positions(self.current_positions)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return False
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop."""
        while self.is_running:
            try:
                # Calculate current P&L
                total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.current_positions.values())
                realized_pnl = sum(pos.realized_pnl for pos in self.current_positions.values())
                unrealized_pnl = sum(pos.unrealized_pnl for pos in self.current_positions.values())
                
                # Update risk manager
                await self.risk_manager.update_pnl(total_pnl, realized_pnl, unrealized_pnl)
                
                # Calculate VaR
                if self.current_positions:
                    var = await self.risk_manager.calculate_var(self.current_positions)
                    self.logger.debug(f"Portfolio VaR: {var:.4f}")
                
                # Get risk metrics
                metrics = await self.risk_manager.get_risk_metrics()
                
                # Log status periodically
                if int(datetime.now().timestamp()) % 60 == 0:  # Every minute
                    self.logger.info(
                        f"Risk Status: {metrics.risk_status.value} | "
                        f"Positions: {metrics.position_count} | "
                        f"Exposure: {metrics.total_exposure:.2%} | "
                        f"Drawdown: {metrics.current_drawdown:.2%}"
                    )
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _position_monitoring_loop(self):
        """Monitor positions and update P&L."""
        while self.is_running:
            try:
                # Simulate position P&L updates
                for symbol, position in self.current_positions.items():
                    # Simulate price movement and update unrealized P&L
                    # In real system, this would come from market data
                    import random
                    price_change = random.uniform(-0.001, 0.001)  # ±0.1% price change
                    pnl_change = position.quantity * price_change
                    position.unrealized_pnl += pnl_change
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _send_risk_alert(self, level: RiskLevel, message: str):
        """Send risk alert through the system."""
        try:
            alert_data = {
                'level': level.value,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'source': 'trading_system'
            }
            
            # Would publish to risk alerts topic
            self.logger.warning(f"RISK ALERT [{level.value.upper()}]: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send risk alert: {e}")
    
    async def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop."""
        try:
            await self.risk_manager.emergency_stop(reason)
            
            # Close all positions (simulated)
            for symbol in list(self.current_positions.keys()):
                del self.current_positions[symbol]
            
            self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    async def set_manual_override(self, duration_hours: int = 24, reason: str = "Manual override"):
        """Set manual risk override."""
        await self.risk_manager.set_manual_override(duration_hours, reason)
        self.logger.warning(f"Manual override activated: {reason}")
    
    async def clear_manual_override(self):
        """Clear manual risk override."""
        await self.risk_manager.clear_manual_override()
        self.logger.info("Manual override cleared")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        risk_status = self.risk_manager.get_status()
        
        return {
            'is_running': self.is_running,
            'account_balance': self.account_balance,
            'position_count': len(self.current_positions),
            'risk_manager': risk_status,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.current_positions.items()
            }
        }


async def main():
    """Example usage of the integrated trading system."""
    # Create trading system
    trading_system = TradingSystemWithRiskManager()
    
    try:
        # Start the system
        await trading_system.start()
        
        # Create sample signals
        signals = [
            Signal(
                symbol="EURUSD",
                side=OrderSide.BUY,
                strength=0.8,
                confidence=0.75,
                strategy_name="breakout_trend",
                timestamp=datetime.now(),
                features={'rsi': 65, 'macd': 0.002}
            ),
            Signal(
                symbol="GBPUSD",
                side=OrderSide.SELL,
                strength=0.6,
                confidence=0.65,
                strategy_name="grid_martingale",
                timestamp=datetime.now(),
                features={'bollinger_position': 0.8, 'atr': 0.0012}
            )
        ]
        
        # Process signals
        for signal in signals:
            success = await trading_system.process_signal(signal)
            print(f"Signal processed: {signal.symbol} - {'SUCCESS' if success else 'REJECTED'}")
            await asyncio.sleep(1)
        
        # Monitor for a short time
        print("Monitoring system for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get system status
        status = trading_system.get_system_status()
        print(f"System Status: {status}")
        
        # Demonstrate emergency stop
        print("Triggering emergency stop...")
        await trading_system.emergency_stop("Demo emergency stop")
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await trading_system.stop()


if __name__ == "__main__":
    asyncio.run(main()) 