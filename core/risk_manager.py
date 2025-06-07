"""
Risk Manager - Core Risk Controls for FX AI-Quant Trading System.

This module implements comprehensive risk management with:
- Real-time P&L and drawdown monitoring
- Position exposure limits and concentration controls
- Value at Risk (VaR) calculations
- Emergency stop mechanisms and kill switches
- Manual override capabilities
- Risk alert publishing
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import yaml
import structlog
import numpy as np
from collections import deque, defaultdict
import statistics

from core.interfaces.trading_interfaces import (
    RiskManager as RiskManagerInterface,
    Order, Position, OrderSide, OrderStatus
)
from core.interfaces.messaging_interfaces import Message, Topics
from core.pubsub import ZMQPublisher


class RiskLevel(Enum):
    """Risk alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskStatus(Enum):
    """Risk manager status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    HALTED = "halted"
    EMERGENCY = "emergency"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    current_drawdown: float
    max_drawdown: float
    total_exposure: float
    position_count: int
    portfolio_var: float
    leverage: float
    largest_position: float
    risk_status: RiskStatus
    active_alerts: List[str] = field(default_factory=list)


@dataclass
class RiskAlert:
    """Risk alert structure."""
    alert_id: str
    level: RiskLevel
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    action_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionExposure:
    """Position exposure tracking."""
    symbol: str
    strategy: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    percentage_of_capital: float
    currency_exposure: Dict[str, float] = field(default_factory=dict)


class RiskManagerConfig:
    """Configuration for risk manager."""
    
    def __init__(self, config_path: str = "config/risk_limits.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Drawdown limits
        dd_config = self.config['drawdown_limits']
        self.max_daily_drawdown = dd_config['max_daily_drawdown']
        self.max_total_drawdown = dd_config['max_total_drawdown']
        self.warning_daily_drawdown = dd_config['warning_daily_drawdown']
        self.warning_total_drawdown = dd_config['warning_total_drawdown']
        self.recovery_threshold = dd_config['recovery_threshold']
        
        # Position limits
        pos_config = self.config['position_limits']
        self.max_position_per_symbol = pos_config['max_position_per_symbol']
        self.max_total_exposure = pos_config['max_total_exposure']
        self.max_concurrent_positions = pos_config['max_concurrent_positions']
        self.max_strategy_exposure = pos_config['max_strategy_exposure']
        self.max_currency_exposure = pos_config['max_currency_exposure']
        self.max_correlated_exposure = pos_config['max_correlated_exposure']
        self.correlation_threshold = pos_config['correlation_threshold']
        self.max_leverage = pos_config['max_leverage']
        self.warning_leverage = pos_config['warning_leverage']
        
        # VaR parameters
        var_config = self.config['var_parameters']
        self.var_confidence_level = var_config['confidence_level']
        self.var_horizon_days = var_config['horizon_days']
        self.var_lookback_period = var_config['lookback_period']
        self.max_portfolio_var = var_config['max_portfolio_var']
        self.warning_portfolio_var = var_config['warning_portfolio_var']
        self.var_calculation_method = var_config['calculation_method']
        
        # Monitoring settings
        mon_config = self.config['monitoring']
        self.pnl_update_frequency = mon_config['pnl_update_frequency']
        self.risk_metrics_frequency = mon_config['risk_metrics_frequency']
        self.var_calculation_frequency = mon_config['var_calculation_frequency']
        self.rapid_loss_threshold = mon_config['rapid_loss_threshold']
        self.rapid_loss_timeframe = mon_config['rapid_loss_timeframe']
        self.volatility_spike_threshold = mon_config['volatility_spike_threshold']
        
        # Emergency controls
        emergency_config = self.config['emergency_controls']
        self.kill_switch_triggers = emergency_config['kill_switch_triggers']
        self.emergency_actions = emergency_config['emergency_actions']
        self.manual_override = emergency_config['manual_override']
        self.recovery_conditions = emergency_config['recovery_conditions']
        
        # Alert configuration
        alert_config = self.config['alerts']
        self.alert_channels = alert_config['channels']
        self.alert_levels = alert_config['levels']
        self.alert_frequency_limits = alert_config['frequency_limits']


class CoreRiskManager(RiskManagerInterface):
    """
    Core Risk Manager implementation with comprehensive risk controls.
    
    Features:
    - Real-time P&L and drawdown monitoring
    - Position exposure and concentration limits
    - Value at Risk (VaR) calculations
    - Emergency stop mechanisms
    - Manual override capabilities
    - Risk alert publishing
    """
    
    def __init__(
        self,
        config: Optional[RiskManagerConfig] = None,
        initial_capital: float = 100000.0,
        publisher: Optional[ZMQPublisher] = None,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        self.config = config or RiskManagerConfig()
        self.initial_capital = initial_capital
        self.publisher = publisher
        self.logger = logger or structlog.get_logger(__name__)
        
        # Risk state tracking
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.total_realized_pnl = 0.0
        self.daily_realized_pnl = 0.0
        
        # Position tracking
        self.current_positions: Dict[str, Position] = {}
        self.position_exposures: Dict[str, PositionExposure] = {}
        self.strategy_exposures: Dict[str, float] = defaultdict(float)
        self.currency_exposures: Dict[str, float] = defaultdict(float)
        
        # Risk metrics history
        self.pnl_history: deque = deque(maxlen=1000)
        self.var_history: deque = deque(maxlen=100)
        self.drawdown_history: deque = deque(maxlen=1000)
        self.volatility_history: deque = deque(maxlen=252)
        
        # Alert management
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Emergency controls
        self.is_emergency_mode = False
        self.is_trading_halted = False
        self.manual_override_active = False
        self.manual_override_until: Optional[datetime] = None
        self.kill_switch_triggered = False
        
        # Performance tracking
        self.last_pnl_update = datetime.now()
        self.last_risk_calculation = datetime.now()
        self.last_var_calculation = datetime.now()
        self.risk_checks_performed = 0
        self.alerts_sent = 0
        
        self.logger.info(
            "CoreRiskManager initialized",
            initial_capital=initial_capital,
            max_daily_drawdown=self.config.max_daily_drawdown,
            max_total_drawdown=self.config.max_total_drawdown
        )
    
    async def check_pre_trade_risk(
        self,
        order: Order,
        current_positions: Dict[str, Position],
        account_balance: float,
    ) -> bool:
        """Check if order passes pre-trade risk checks."""
        try:
            self.risk_checks_performed += 1
            
            # Update current state
            self.current_positions = current_positions
            self.current_capital = account_balance
            
            # Emergency mode check
            if self.is_emergency_mode or self.kill_switch_triggered:
                await self._send_alert(
                    RiskLevel.CRITICAL,
                    f"Trade rejected - Emergency mode active",
                    "emergency_mode",
                    1.0, 0.0
                )
                return False
            
            # Trading halt check
            if self.is_trading_halted and not self.manual_override_active:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Trade rejected - Trading halted",
                    "trading_halted",
                    1.0, 0.0
                )
                return False
            
            # Position count limit
            if len(current_positions) >= self.config.max_concurrent_positions:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Trade rejected - Max positions limit ({self.config.max_concurrent_positions})",
                    "max_positions",
                    len(current_positions), self.config.max_concurrent_positions
                )
                return False
            
            # Calculate position size impact
            position_value = abs(order.quantity * order.price) if order.price else 0
            position_percentage = position_value / account_balance
            
            # Position size limit per symbol
            current_symbol_exposure = self._calculate_symbol_exposure(order.symbol, current_positions)
            new_symbol_exposure = current_symbol_exposure + position_percentage
            
            if new_symbol_exposure > self.config.max_position_per_symbol:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Trade rejected - Symbol exposure limit ({order.symbol}): {new_symbol_exposure:.2%} > {self.config.max_position_per_symbol:.2%}",
                    "symbol_exposure",
                    new_symbol_exposure, self.config.max_position_per_symbol
                )
                return False
            
            # Total exposure limit
            current_total_exposure = self._calculate_total_exposure(current_positions)
            new_total_exposure = current_total_exposure + position_percentage
            
            if new_total_exposure > self.config.max_total_exposure:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Trade rejected - Total exposure limit: {new_total_exposure:.2%} > {self.config.max_total_exposure:.2%}",
                    "total_exposure",
                    new_total_exposure, self.config.max_total_exposure
                )
                return False
            
            # Leverage check
            current_leverage = self._calculate_leverage(current_positions, account_balance)
            if current_leverage > self.config.max_leverage:
                await self._send_alert(
                    RiskLevel.CRITICAL,
                    f"Trade rejected - Leverage limit: {current_leverage:.1f}x > {self.config.max_leverage:.1f}x",
                    "leverage",
                    current_leverage, self.config.max_leverage
                )
                return False
            
            self.logger.debug(
                "Pre-trade risk check passed",
                order_id=order.order_id,
                symbol=order.symbol,
                position_percentage=position_percentage,
                total_exposure=new_total_exposure
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Error in pre-trade risk check", error=str(e))
            return False
    
    async def monitor_drawdown(self, current_pnl: float, peak_pnl: float) -> bool:
        """Monitor drawdown levels."""
        try:
            # Update peak if necessary
            if current_pnl > self.peak_capital - self.initial_capital:
                self.peak_capital = self.initial_capital + current_pnl
            
            # Calculate drawdowns
            total_drawdown = (self.peak_capital - (self.initial_capital + current_pnl)) / self.peak_capital
            daily_drawdown = (self.daily_start_capital - (self.initial_capital + current_pnl)) / self.daily_start_capital
            
            # Store in history
            self.drawdown_history.append({
                'timestamp': datetime.now(),
                'total_drawdown': total_drawdown,
                'daily_drawdown': daily_drawdown,
                'current_pnl': current_pnl
            })
            
            # Check daily drawdown limits
            if daily_drawdown >= self.config.max_daily_drawdown:
                await self._trigger_emergency_stop("Daily drawdown limit exceeded")
                return False
            elif daily_drawdown >= self.config.warning_daily_drawdown:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Daily drawdown warning: {daily_drawdown:.2%}",
                    "daily_drawdown",
                    daily_drawdown, self.config.warning_daily_drawdown
                )
            
            # Check total drawdown limits
            if total_drawdown >= self.config.max_total_drawdown:
                await self._trigger_emergency_stop("Total drawdown limit exceeded")
                return False
            elif total_drawdown >= self.config.warning_total_drawdown:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Total drawdown warning: {total_drawdown:.2%}",
                    "total_drawdown",
                    total_drawdown, self.config.warning_total_drawdown
                )
            
            return True
            
        except Exception as e:
            self.logger.error("Error monitoring drawdown", error=str(e))
            return False
    
    async def calculate_var(
        self,
        positions: Dict[str, Position],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            if not positions or len(self.pnl_history) < 30:
                return 0.0
            
            # Get recent P&L changes
            pnl_changes = []
            for i in range(1, min(len(self.pnl_history), self.config.var_lookback_period)):
                current_pnl = self.pnl_history[i]['total_pnl']
                previous_pnl = self.pnl_history[i-1]['total_pnl']
                pnl_change = current_pnl - previous_pnl
                pnl_changes.append(pnl_change)
            
            if len(pnl_changes) < 10:
                return 0.0
            
            # Calculate VaR using historical simulation
            if self.config.var_calculation_method == "historical":
                var_percentile = (1 - confidence_level) * 100
                var_value = np.percentile(pnl_changes, var_percentile)
                
                # Scale for horizon
                var_scaled = abs(var_value) * np.sqrt(horizon_days)
                
            elif self.config.var_calculation_method == "parametric":
                # Parametric VaR using normal distribution
                mean_return = np.mean(pnl_changes)
                std_return = np.std(pnl_changes)
                z_score = stats.norm.ppf(1 - confidence_level)
                var_scaled = abs(mean_return + z_score * std_return) * np.sqrt(horizon_days)
            
            else:
                # Default to historical method
                var_percentile = (1 - confidence_level) * 100
                var_value = np.percentile(pnl_changes, var_percentile)
                var_scaled = abs(var_value) * np.sqrt(horizon_days)
            
            # Convert to percentage of capital
            portfolio_var = var_scaled / self.current_capital
            
            # Store in history
            self.var_history.append({
                'timestamp': datetime.now(),
                'var_absolute': var_scaled,
                'var_percentage': portfolio_var,
                'confidence_level': confidence_level
            })
            
            # Check VaR limits
            if portfolio_var >= self.config.max_portfolio_var:
                await self._send_alert(
                    RiskLevel.CRITICAL,
                    f"Portfolio VaR limit exceeded: {portfolio_var:.2%} > {self.config.max_portfolio_var:.2%}",
                    "portfolio_var",
                    portfolio_var, self.config.max_portfolio_var
                )
            elif portfolio_var >= self.config.warning_portfolio_var:
                await self._send_alert(
                    RiskLevel.WARNING,
                    f"Portfolio VaR warning: {portfolio_var:.2%}",
                    "portfolio_var",
                    portfolio_var, self.config.warning_portfolio_var
                )
            
            self.last_var_calculation = datetime.now()
            return portfolio_var
            
        except Exception as e:
            self.logger.error("Error calculating VaR", error=str(e))
            return 0.0
    
    async def check_position_limits(
        self, symbol: str, new_quantity: float, current_positions: Dict[str, Position]
    ) -> bool:
        """Check if new position would violate limits."""
        try:
            # Calculate new position impact
            current_price = 1.0  # Would need market data for actual price
            position_value = abs(new_quantity * current_price)
            position_percentage = position_value / self.current_capital
            
            # Symbol exposure check
            current_symbol_exposure = self._calculate_symbol_exposure(symbol, current_positions)
            new_symbol_exposure = current_symbol_exposure + position_percentage
            
            if new_symbol_exposure > self.config.max_position_per_symbol:
                return False
            
            # Total exposure check
            current_total_exposure = self._calculate_total_exposure(current_positions)
            new_total_exposure = current_total_exposure + position_percentage
            
            if new_total_exposure > self.config.max_total_exposure:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Error checking position limits", error=str(e))
            return False
    
    async def emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop."""
        await self._trigger_emergency_stop(reason)
    
    async def update_positions(self, positions: Dict[str, Position]) -> None:
        """Update current positions and calculate exposures."""
        try:
            self.current_positions = positions
            
            # Calculate position exposures
            self.position_exposures.clear()
            self.strategy_exposures.clear()
            self.currency_exposures.clear()
            
            for symbol, position in positions.items():
                exposure = PositionExposure(
                    symbol=symbol,
                    strategy="unknown",  # Would need strategy mapping
                    position_size=position.quantity,
                    market_value=abs(position.quantity * position.avg_price),
                    unrealized_pnl=position.unrealized_pnl,
                    percentage_of_capital=abs(position.quantity * position.avg_price) / self.current_capital
                )
                self.position_exposures[symbol] = exposure
                
                # Extract currency exposures for FX pairs
                if len(symbol) == 6:  # FX pair format like EURUSD
                    base_currency = symbol[:3]
                    quote_currency = symbol[3:]
                    
                    if position.quantity > 0:  # Long position
                        self.currency_exposures[base_currency] += exposure.percentage_of_capital
                        self.currency_exposures[quote_currency] -= exposure.percentage_of_capital
                    else:  # Short position
                        self.currency_exposures[base_currency] -= exposure.percentage_of_capital
                        self.currency_exposures[quote_currency] += exposure.percentage_of_capital
            
        except Exception as e:
            self.logger.error("Error updating positions", error=str(e))
    
    async def update_pnl(self, total_pnl: float, realized_pnl: float, unrealized_pnl: float) -> None:
        """Update P&L tracking."""
        try:
            # Store P&L history
            pnl_record = {
                'timestamp': datetime.now(),
                'total_pnl': total_pnl,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'current_capital': self.current_capital
            }
            self.pnl_history.append(pnl_record)
            
            # Update running totals
            self.total_realized_pnl = realized_pnl
            
            # Check for rapid losses
            await self._check_rapid_loss()
            
            # Monitor drawdown
            await self.monitor_drawdown(total_pnl, self.peak_capital - self.initial_capital)
            
            self.last_pnl_update = datetime.now()
            
        except Exception as e:
            self.logger.error("Error updating P&L", error=str(e))
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics snapshot."""
        try:
            # Calculate current metrics
            total_pnl = self.total_realized_pnl + sum(pos.unrealized_pnl for pos in self.current_positions.values())
            daily_pnl = total_pnl  # Simplified - would need daily tracking
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.current_positions.values())
            
            current_drawdown = max(0, (self.peak_capital - (self.initial_capital + total_pnl)) / self.peak_capital)
            max_drawdown = max([dd['total_drawdown'] for dd in self.drawdown_history]) if self.drawdown_history else 0
            
            total_exposure = self._calculate_total_exposure(self.current_positions)
            leverage = self._calculate_leverage(self.current_positions, self.current_capital)
            
            largest_position = max([
                abs(pos.quantity * pos.avg_price) / self.current_capital 
                for pos in self.current_positions.values()
            ]) if self.current_positions else 0
            
            portfolio_var = self.var_history[-1]['var_percentage'] if self.var_history else 0
            
            # Determine risk status
            risk_status = self._determine_risk_status(current_drawdown, total_exposure, leverage, portfolio_var)
            
            return RiskMetrics(
                timestamp=datetime.now(),
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.total_realized_pnl,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                total_exposure=total_exposure,
                position_count=len(self.current_positions),
                portfolio_var=portfolio_var,
                leverage=leverage,
                largest_position=largest_position,
                risk_status=risk_status,
                active_alerts=list(self.active_alerts.keys())
            )
            
        except Exception as e:
            self.logger.error("Error getting risk metrics", error=str(e))
            return RiskMetrics(
                timestamp=datetime.now(),
                total_pnl=0, daily_pnl=0, unrealized_pnl=0, realized_pnl=0,
                current_drawdown=0, max_drawdown=0, total_exposure=0,
                position_count=0, portfolio_var=0, leverage=0, largest_position=0,
                risk_status=RiskStatus.HEALTHY
            )
    
    async def set_manual_override(self, duration_hours: int = 24, reason: str = "Manual override") -> None:
        """Set manual override for risk controls."""
        try:
            self.manual_override_active = True
            self.manual_override_until = datetime.now() + timedelta(hours=duration_hours)
            
            await self._send_alert(
                RiskLevel.WARNING,
                f"Manual override activated: {reason} (Duration: {duration_hours}h)",
                "manual_override",
                1.0, 0.0
            )
            
            self.logger.warning(
                "Manual override activated",
                reason=reason,
                duration_hours=duration_hours,
                until=self.manual_override_until
            )
            
        except Exception as e:
            self.logger.error("Error setting manual override", error=str(e))
    
    async def clear_manual_override(self) -> None:
        """Clear manual override."""
        try:
            self.manual_override_active = False
            self.manual_override_until = None
            
            await self._send_alert(
                RiskLevel.INFO,
                "Manual override cleared",
                "manual_override",
                0.0, 1.0
            )
            
            self.logger.info("Manual override cleared")
            
        except Exception as e:
            self.logger.error("Error clearing manual override", error=str(e))
    
    async def reset_emergency_mode(self) -> None:
        """Reset emergency mode if conditions allow."""
        try:
            if not self.is_emergency_mode:
                return
            
            # Check recovery conditions
            if await self._check_recovery_conditions():
                self.is_emergency_mode = False
                self.kill_switch_triggered = False
                self.is_trading_halted = False
                
                await self._send_alert(
                    RiskLevel.INFO,
                    "Emergency mode reset - conditions normalized",
                    "emergency_reset",
                    0.0, 1.0
                )
                
                self.logger.info("Emergency mode reset")
            else:
                await self._send_alert(
                    RiskLevel.WARNING,
                    "Emergency mode reset attempted but conditions not met",
                    "emergency_reset_failed",
                    1.0, 0.0
                )
                
        except Exception as e:
            self.logger.error("Error resetting emergency mode", error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk manager status."""
        return {
            "is_emergency_mode": self.is_emergency_mode,
            "is_trading_halted": self.is_trading_halted,
            "manual_override_active": self.manual_override_active,
            "manual_override_until": self.manual_override_until.isoformat() if self.manual_override_until else None,
            "kill_switch_triggered": self.kill_switch_triggered,
            "active_alerts_count": len(self.active_alerts),
            "risk_checks_performed": self.risk_checks_performed,
            "alerts_sent": self.alerts_sent,
            "last_pnl_update": self.last_pnl_update.isoformat(),
            "last_risk_calculation": self.last_risk_calculation.isoformat(),
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "position_count": len(self.current_positions)
        }
    
    # Private helper methods
    
    def _calculate_symbol_exposure(self, symbol: str, positions: Dict[str, Position]) -> float:
        """Calculate current exposure for a symbol."""
        if symbol not in positions:
            return 0.0
        
        position = positions[symbol]
        position_value = abs(position.quantity * position.avg_price)
        return position_value / self.current_capital
    
    def _calculate_total_exposure(self, positions: Dict[str, Position]) -> float:
        """Calculate total portfolio exposure."""
        total_value = sum(abs(pos.quantity * pos.avg_price) for pos in positions.values())
        return total_value / self.current_capital
    
    def _calculate_leverage(self, positions: Dict[str, Position], capital: float) -> float:
        """Calculate current leverage."""
        total_notional = sum(abs(pos.quantity * pos.avg_price) for pos in positions.values())
        return total_notional / capital if capital > 0 else 0.0
    
    def _determine_risk_status(self, drawdown: float, exposure: float, leverage: float, var: float) -> RiskStatus:
        """Determine overall risk status."""
        if self.is_emergency_mode or self.kill_switch_triggered:
            return RiskStatus.EMERGENCY
        
        if self.is_trading_halted:
            return RiskStatus.HALTED
        
        critical_conditions = [
            drawdown >= self.config.max_total_drawdown * 0.8,
            exposure >= self.config.max_total_exposure * 0.9,
            leverage >= self.config.max_leverage * 0.9,
            var >= self.config.max_portfolio_var * 0.8
        ]
        
        warning_conditions = [
            drawdown >= self.config.warning_total_drawdown,
            exposure >= self.config.max_total_exposure * 0.7,
            leverage >= self.config.warning_leverage,
            var >= self.config.warning_portfolio_var
        ]
        
        if any(critical_conditions):
            return RiskStatus.CRITICAL
        elif any(warning_conditions):
            return RiskStatus.WARNING
        else:
            return RiskStatus.HEALTHY
    
    async def _check_rapid_loss(self) -> None:
        """Check for rapid losses."""
        try:
            if len(self.pnl_history) < 2:
                return
            
            current_time = datetime.now()
            rapid_loss_cutoff = current_time - timedelta(seconds=self.config.rapid_loss_timeframe)
            
            # Get recent P&L changes
            recent_pnl = [
                record for record in self.pnl_history 
                if record['timestamp'] >= rapid_loss_cutoff
            ]
            
            if len(recent_pnl) < 2:
                return
            
            # Calculate loss over timeframe
            start_pnl = recent_pnl[0]['total_pnl']
            end_pnl = recent_pnl[-1]['total_pnl']
            loss_percentage = (start_pnl - end_pnl) / self.current_capital
            
            if loss_percentage >= self.config.rapid_loss_threshold:
                await self._send_alert(
                    RiskLevel.CRITICAL,
                    f"Rapid loss detected: {loss_percentage:.2%} in {self.config.rapid_loss_timeframe}s",
                    "rapid_loss",
                    loss_percentage, self.config.rapid_loss_threshold
                )
                
        except Exception as e:
            self.logger.error("Error checking rapid loss", error=str(e))
    
    async def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop procedures."""
        try:
            self.is_emergency_mode = True
            self.kill_switch_triggered = True
            self.is_trading_halted = True
            
            await self._send_alert(
                RiskLevel.EMERGENCY,
                f"EMERGENCY STOP TRIGGERED: {reason}",
                "emergency_stop",
                1.0, 0.0,
                action_required=True
            )
            
            self.logger.critical(
                "Emergency stop triggered",
                reason=reason,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error("Error triggering emergency stop", error=str(e))
    
    async def _check_recovery_conditions(self) -> bool:
        """Check if recovery conditions are met."""
        try:
            # Check stability period
            stability_cutoff = datetime.now() - timedelta(minutes=self.config.recovery_conditions['stability_period_minutes'])
            
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= stability_cutoff and alert.level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]
            ]
            
            if recent_alerts:
                return False
            
            # Check current metrics
            metrics = await self.get_risk_metrics()
            
            recovery_conditions = [
                metrics.current_drawdown < self.config.max_total_drawdown * 0.8,
                metrics.total_exposure < self.config.max_total_exposure * 0.8,
                metrics.leverage < self.config.max_leverage * 0.8,
                metrics.portfolio_var < self.config.max_portfolio_var * 0.8
            ]
            
            return all(recovery_conditions)
            
        except Exception as e:
            self.logger.error("Error checking recovery conditions", error=str(e))
            return False
    
    async def _send_alert(
        self,
        level: RiskLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        action_required: bool = False
    ) -> None:
        """Send risk alert."""
        try:
            # Check alert frequency limits
            alert_key = f"{level.value}_{metric_name}"
            now = datetime.now()
            
            if alert_key in self.last_alert_times:
                time_since_last = (now - self.last_alert_times[alert_key]).total_seconds()
                cooldown = self.config.alert_frequency_limits['duplicate_alert_cooldown']
                
                if level == RiskLevel.EMERGENCY:
                    cooldown = self.config.alert_frequency_limits['emergency_alert_cooldown']
                
                if time_since_last < cooldown:
                    return
            
            # Create alert
            alert = RiskAlert(
                alert_id=f"{alert_key}_{int(now.timestamp())}",
                level=level,
                message=message,
                timestamp=now,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                action_required=action_required
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self.last_alert_times[alert_key] = now
            self.alerts_sent += 1
            
            # Log alert
            self.logger.log(
                level.value.upper(),
                message,
                alert_id=alert.alert_id,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                action_required=action_required
            )
            
            # Publish alert if publisher available
            if self.publisher and self.config.alert_channels['pubsub']:
                alert_message = Message(
                    topic=Topics.RISK_ALERTS,
                    data={
                        'alert_id': alert.alert_id,
                        'level': level.value,
                        'message': message,
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'threshold_value': threshold_value,
                        'action_required': action_required,
                        'timestamp': now.isoformat()
                    },
                    timestamp=now,
                    source="risk_manager"
                )
                
                # Note: Would need to implement proper message publishing
                # await self.publisher.publish_message(alert_message)
            
        except Exception as e:
            self.logger.error("Error sending alert", error=str(e))


def create_risk_manager(
    config_path: str = "config/risk_limits.yaml",
    initial_capital: float = 100000.0,
    publisher: Optional[ZMQPublisher] = None,
    logger: Optional[structlog.stdlib.BoundLogger] = None
) -> CoreRiskManager:
    """
    Factory function to create a configured risk manager.
    
    Args:
        config_path: Path to risk limits configuration file
        initial_capital: Initial trading capital
        publisher: Optional message publisher for alerts
        logger: Optional logger instance
        
    Returns:
        Configured CoreRiskManager instance
    """
    config = RiskManagerConfig(config_path)
    return CoreRiskManager(
        config=config,
        initial_capital=initial_capital,
        publisher=publisher,
        logger=logger
    ) 