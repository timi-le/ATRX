"""
Unit tests for Risk Manager - Core Risk Controls.

Tests cover:
- Risk limit enforcement
- Drawdown calculations
- VaR computations
- Emergency stop mechanisms
- Override functionality
- Alert generation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import tempfile
import yaml
import os

from core.risk_manager import (
    CoreRiskManager, RiskManagerConfig, RiskLevel, RiskStatus,
    RiskMetrics, RiskAlert, PositionExposure
)
from core.interfaces.trading_interfaces import Order, Position, OrderSide, OrderType


@pytest.fixture
def sample_config():
    """Create a sample risk configuration for testing."""
    config_data = {
        'drawdown_limits': {
            'max_daily_drawdown': 0.05,
            'max_total_drawdown': 0.15,
            'warning_daily_drawdown': 0.03,
            'warning_total_drawdown': 0.10,
            'recovery_threshold': 0.02
        },
        'position_limits': {
            'max_position_per_symbol': 0.10,
            'max_total_exposure': 0.50,
            'max_concurrent_positions': 8,
            'max_strategy_exposure': 0.30,
            'max_currency_exposure': 0.40,
            'max_correlated_exposure': 0.25,
            'correlation_threshold': 0.7,
            'max_leverage': 10.0,
            'warning_leverage': 7.0
        },
        'var_parameters': {
            'confidence_level': 0.95,
            'horizon_days': 1,
            'lookback_period': 252,
            'max_portfolio_var': 0.03,
            'warning_portfolio_var': 0.02,
            'calculation_method': 'historical'
        },
        'monitoring': {
            'pnl_update_frequency': 1,
            'risk_metrics_frequency': 30,
            'var_calculation_frequency': 300,
            'rapid_loss_threshold': 0.02,
            'rapid_loss_timeframe': 300,
            'volatility_spike_threshold': 3.0
        },
        'emergency_controls': {
            'kill_switch_triggers': {
                'daily_loss_threshold': 0.05,
                'total_loss_threshold': 0.15,
                'var_breach_multiplier': 2.0
            },
            'emergency_actions': {
                'close_all_positions': True,
                'cancel_all_orders': True,
                'disable_new_trades': True,
                'send_alerts': True
            },
            'manual_override': {
                'enabled': True,
                'require_confirmation': True,
                'override_duration_hours': 24
            },
            'recovery_conditions': {
                'stability_period_minutes': 60,
                'max_loss_improvement': 0.02
            }
        },
        'alerts': {
            'channels': {
                'log': True,
                'console': True,
                'pubsub': True
            },
            'levels': {
                'info': True,
                'warning': True,
                'critical': True,
                'emergency': True
            },
            'frequency_limits': {
                'max_alerts_per_minute': 10,
                'duplicate_alert_cooldown': 300,
                'emergency_alert_cooldown': 60
            }
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


@pytest.fixture
def risk_manager(sample_config):
    """Create a risk manager instance for testing."""
    config = RiskManagerConfig(sample_config)
    manager = CoreRiskManager(
        config=config,
        initial_capital=100000.0,
        publisher=None,
        logger=Mock()
    )
    yield manager
    
    # Cleanup
    os.unlink(sample_config)


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id="test_order_1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10000,
        price=1.1000
    )


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return {
        "EURUSD": Position(
            symbol="EURUSD",
            quantity=10000,
            avg_price=1.1000,
            unrealized_pnl=100.0,
            realized_pnl=0.0
        ),
        "GBPUSD": Position(
            symbol="GBPUSD",
            quantity=-5000,
            avg_price=1.3000,
            unrealized_pnl=-50.0,
            realized_pnl=0.0
        )
    }


class TestRiskManagerConfig:
    """Test risk manager configuration loading."""
    
    def test_config_loading(self, sample_config):
        """Test configuration file loading."""
        config = RiskManagerConfig(sample_config)
        
        assert config.max_daily_drawdown == 0.05
        assert config.max_total_drawdown == 0.15
        assert config.max_position_per_symbol == 0.10
        assert config.max_concurrent_positions == 8
        assert config.var_confidence_level == 0.95
    
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            RiskManagerConfig("nonexistent_config.yaml")


class TestRiskManagerInitialization:
    """Test risk manager initialization."""
    
    def test_initialization(self, risk_manager):
        """Test basic initialization."""
        assert risk_manager.initial_capital == 100000.0
        assert risk_manager.current_capital == 100000.0
        assert risk_manager.peak_capital == 100000.0
        assert not risk_manager.is_emergency_mode
        assert not risk_manager.is_trading_halted
        assert not risk_manager.manual_override_active
    
    def test_initialization_with_custom_capital(self, sample_config):
        """Test initialization with custom capital."""
        config = RiskManagerConfig(sample_config)
        manager = CoreRiskManager(config=config, initial_capital=50000.0)
        
        assert manager.initial_capital == 50000.0
        assert manager.current_capital == 50000.0


class TestPreTradeRiskChecks:
    """Test pre-trade risk checking functionality."""
    
    @pytest.mark.asyncio
    async def test_emergency_mode_rejection(self, risk_manager, sample_order, sample_positions):
        """Test trade rejection during emergency mode."""
        risk_manager.is_emergency_mode = True
        
        result = await risk_manager.check_pre_trade_risk(
            sample_order, sample_positions, 100000.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_trading_halt_rejection(self, risk_manager, sample_order, sample_positions):
        """Test trade rejection during trading halt."""
        risk_manager.is_trading_halted = True
        
        result = await risk_manager.check_pre_trade_risk(
            sample_order, sample_positions, 100000.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_max_positions_limit(self, risk_manager, sample_order):
        """Test maximum positions limit enforcement."""
        # Create positions at the limit
        positions = {}
        for i in range(8):  # max_concurrent_positions = 8
            positions[f"PAIR{i}"] = Position(
                symbol=f"PAIR{i}",
                quantity=1000,
                avg_price=1.0,
                unrealized_pnl=0.0
            )
        
        result = await risk_manager.check_pre_trade_risk(
            sample_order, positions, 100000.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_symbol_exposure_limit(self, risk_manager, sample_positions):
        """Test symbol exposure limit enforcement."""
        # Create order that would exceed symbol limit (10% of capital)
        large_order = Order(
            order_id="large_order",
            symbol="EURUSD",  # Already has position
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50000,  # Large quantity
            price=1.1000
        )
        
        result = await risk_manager.check_pre_trade_risk(
            large_order, sample_positions, 100000.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_total_exposure_limit(self, risk_manager, sample_order):
        """Test total exposure limit enforcement."""
        # Create positions near total exposure limit (50% of capital)
        large_positions = {}
        for i in range(4):
            large_positions[f"PAIR{i}"] = Position(
                symbol=f"PAIR{i}",
                quantity=12000,  # 12% each = 48% total
                avg_price=1.0,
                unrealized_pnl=0.0
            )
        
        result = await risk_manager.check_pre_trade_risk(
            sample_order, large_positions, 100000.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_leverage_limit(self, risk_manager, sample_order):
        """Test leverage limit enforcement."""
        # Create high leverage positions
        high_leverage_positions = {
            "EURUSD": Position(
                symbol="EURUSD",
                quantity=500000,  # 5x leverage
                avg_price=1.1000,
                unrealized_pnl=0.0
            ),
            "GBPUSD": Position(
                symbol="GBPUSD",
                quantity=400000,  # 4x leverage
                avg_price=1.3000,
                unrealized_pnl=0.0
            )
        }
        
        result = await risk_manager.check_pre_trade_risk(
            sample_order, high_leverage_positions, 100000.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_valid_trade_approval(self, risk_manager, sample_order, sample_positions):
        """Test approval of valid trade."""
        result = await risk_manager.check_pre_trade_risk(
            sample_order, sample_positions, 100000.0
        )
        
        assert result is True


class TestDrawdownMonitoring:
    """Test drawdown monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_daily_drawdown_warning(self, risk_manager):
        """Test daily drawdown warning threshold."""
        # Simulate 3% daily loss (warning threshold)
        current_pnl = -3000.0  # 3% of 100k
        peak_pnl = 0.0
        
        result = await risk_manager.monitor_drawdown(current_pnl, peak_pnl)
        
        assert result is True  # Should continue but warn
        assert len(risk_manager.drawdown_history) > 0
    
    @pytest.mark.asyncio
    async def test_daily_drawdown_limit_exceeded(self, risk_manager):
        """Test daily drawdown limit exceeded."""
        # Simulate 6% daily loss (exceeds 5% limit)
        current_pnl = -6000.0  # 6% of 100k
        peak_pnl = 0.0
        
        result = await risk_manager.monitor_drawdown(current_pnl, peak_pnl)
        
        assert result is False
        assert risk_manager.is_emergency_mode
        assert risk_manager.kill_switch_triggered
    
    @pytest.mark.asyncio
    async def test_total_drawdown_calculation(self, risk_manager):
        """Test total drawdown calculation with peak updates."""
        # First, simulate profit to set new peak
        await risk_manager.monitor_drawdown(5000.0, 0.0)  # 5% profit
        assert risk_manager.peak_capital == 105000.0
        
        # Then simulate drawdown from peak
        await risk_manager.monitor_drawdown(-5000.0, 5000.0)  # 10% drawdown from peak
        
        drawdown_record = risk_manager.drawdown_history[-1]
        expected_drawdown = (105000.0 - 95000.0) / 105000.0  # ~9.5%
        assert abs(drawdown_record['total_drawdown'] - expected_drawdown) < 0.001
    
    @pytest.mark.asyncio
    async def test_total_drawdown_limit_exceeded(self, risk_manager):
        """Test total drawdown limit exceeded."""
        # Set peak higher first
        risk_manager.peak_capital = 120000.0
        
        # Simulate 16% total drawdown (exceeds 15% limit)
        current_pnl = -20000.0  # From 120k to 100k = 16.7% drawdown
        
        result = await risk_manager.monitor_drawdown(current_pnl, 20000.0)
        
        assert result is False
        assert risk_manager.is_emergency_mode


class TestVaRCalculation:
    """Test Value at Risk calculation functionality."""
    
    @pytest.mark.asyncio
    async def test_var_insufficient_data(self, risk_manager, sample_positions):
        """Test VaR calculation with insufficient data."""
        var = await risk_manager.calculate_var(sample_positions)
        assert var == 0.0
    
    @pytest.mark.asyncio
    async def test_var_historical_method(self, risk_manager, sample_positions):
        """Test VaR calculation using historical method."""
        # Populate P&L history with sample data
        for i in range(50):
            pnl_change = np.random.normal(0, 1000)  # Random P&L changes
            risk_manager.pnl_history.append({
                'timestamp': datetime.now(),
                'total_pnl': pnl_change,
                'realized_pnl': 0,
                'unrealized_pnl': pnl_change,
                'current_capital': 100000
            })
        
        var = await risk_manager.calculate_var(sample_positions)
        
        assert var >= 0.0
        assert len(risk_manager.var_history) > 0
    
    @pytest.mark.asyncio
    async def test_var_limit_exceeded(self, risk_manager, sample_positions):
        """Test VaR limit exceeded alert."""
        # Create high volatility P&L history
        for i in range(50):
            pnl_change = np.random.normal(0, 5000)  # High volatility
            risk_manager.pnl_history.append({
                'timestamp': datetime.now(),
                'total_pnl': pnl_change,
                'realized_pnl': 0,
                'unrealized_pnl': pnl_change,
                'current_capital': 100000
            })
        
        var = await risk_manager.calculate_var(sample_positions)
        
        # Should generate alert if VaR is high
        if var >= risk_manager.config.warning_portfolio_var:
            assert len(risk_manager.active_alerts) > 0


class TestPositionLimits:
    """Test position limit checking functionality."""
    
    @pytest.mark.asyncio
    async def test_check_position_limits_valid(self, risk_manager, sample_positions):
        """Test position limits check for valid position."""
        result = await risk_manager.check_position_limits(
            "USDJPY", 5000, sample_positions
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_position_limits_symbol_exceeded(self, risk_manager, sample_positions):
        """Test position limits check when symbol limit would be exceeded."""
        # Try to add large position to existing symbol
        result = await risk_manager.check_position_limits(
            "EURUSD", 50000, sample_positions  # Would exceed 10% symbol limit
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_position_limits_total_exceeded(self, risk_manager):
        """Test position limits check when total exposure would be exceeded."""
        # Create positions near total limit
        large_positions = {}
        for i in range(4):
            large_positions[f"PAIR{i}"] = Position(
                symbol=f"PAIR{i}",
                quantity=12000,  # 12% each
                avg_price=1.0,
                unrealized_pnl=0.0
            )
        
        result = await risk_manager.check_position_limits(
            "NEWPAIR", 5000, large_positions  # Would exceed 50% total limit
        )
        assert result is False


class TestEmergencyControls:
    """Test emergency control functionality."""
    
    @pytest.mark.asyncio
    async def test_emergency_stop_trigger(self, risk_manager):
        """Test emergency stop triggering."""
        await risk_manager.emergency_stop("Test emergency")
        
        assert risk_manager.is_emergency_mode
        assert risk_manager.kill_switch_triggered
        assert risk_manager.is_trading_halted
        assert len(risk_manager.active_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_manual_override_activation(self, risk_manager):
        """Test manual override activation."""
        await risk_manager.set_manual_override(12, "Test override")
        
        assert risk_manager.manual_override_active
        assert risk_manager.manual_override_until is not None
        assert len(risk_manager.active_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_manual_override_clearing(self, risk_manager):
        """Test manual override clearing."""
        # First activate override
        await risk_manager.set_manual_override(12, "Test override")
        assert risk_manager.manual_override_active
        
        # Then clear it
        await risk_manager.clear_manual_override()
        assert not risk_manager.manual_override_active
        assert risk_manager.manual_override_until is None
    
    @pytest.mark.asyncio
    async def test_emergency_mode_reset_conditions_not_met(self, risk_manager):
        """Test emergency mode reset when conditions not met."""
        # Trigger emergency mode
        await risk_manager.emergency_stop("Test emergency")
        
        # Try to reset immediately (conditions not met)
        await risk_manager.reset_emergency_mode()
        
        # Should still be in emergency mode
        assert risk_manager.is_emergency_mode
    
    @pytest.mark.asyncio
    async def test_emergency_mode_reset_success(self, risk_manager):
        """Test successful emergency mode reset."""
        # Mock the recovery conditions check to return True
        with patch.object(risk_manager, '_check_recovery_conditions', return_value=True):
            # Trigger emergency mode
            await risk_manager.emergency_stop("Test emergency")
            assert risk_manager.is_emergency_mode
            
            # Reset should succeed
            await risk_manager.reset_emergency_mode()
            assert not risk_manager.is_emergency_mode
            assert not risk_manager.kill_switch_triggered
            assert not risk_manager.is_trading_halted


class TestRiskMetrics:
    """Test risk metrics calculation and reporting."""
    
    @pytest.mark.asyncio
    async def test_get_risk_metrics_basic(self, risk_manager, sample_positions):
        """Test basic risk metrics calculation."""
        # Update positions and P&L
        await risk_manager.update_positions(sample_positions)
        await risk_manager.update_pnl(50.0, 0.0, 50.0)
        
        metrics = await risk_manager.get_risk_metrics()
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.position_count == len(sample_positions)
        assert metrics.total_pnl == 50.0
        assert metrics.unrealized_pnl == 50.0
        assert metrics.risk_status in [status for status in RiskStatus]
    
    @pytest.mark.asyncio
    async def test_risk_status_determination(self, risk_manager):
        """Test risk status determination logic."""
        # Test healthy status
        status = risk_manager._determine_risk_status(0.01, 0.1, 2.0, 0.01)
        assert status == RiskStatus.HEALTHY
        
        # Test warning status
        status = risk_manager._determine_risk_status(0.08, 0.35, 6.0, 0.015)
        assert status == RiskStatus.WARNING
        
        # Test critical status
        status = risk_manager._determine_risk_status(0.12, 0.45, 9.0, 0.025)
        assert status == RiskStatus.CRITICAL
        
        # Test emergency status
        risk_manager.is_emergency_mode = True
        status = risk_manager._determine_risk_status(0.01, 0.1, 2.0, 0.01)
        assert status == RiskStatus.EMERGENCY


class TestPositionTracking:
    """Test position tracking and exposure calculation."""
    
    @pytest.mark.asyncio
    async def test_update_positions(self, risk_manager, sample_positions):
        """Test position updates and exposure calculations."""
        await risk_manager.update_positions(sample_positions)
        
        assert len(risk_manager.current_positions) == 2
        assert len(risk_manager.position_exposures) == 2
        
        # Check currency exposures for FX pairs
        assert "EUR" in risk_manager.currency_exposures
        assert "USD" in risk_manager.currency_exposures
        assert "GBP" in risk_manager.currency_exposures
    
    def test_calculate_symbol_exposure(self, risk_manager, sample_positions):
        """Test symbol exposure calculation."""
        exposure = risk_manager._calculate_symbol_exposure("EURUSD", sample_positions)
        expected = abs(10000 * 1.1000) / 100000  # 11%
        assert abs(exposure - expected) < 0.001
    
    def test_calculate_total_exposure(self, risk_manager, sample_positions):
        """Test total exposure calculation."""
        exposure = risk_manager._calculate_total_exposure(sample_positions)
        expected = (abs(10000 * 1.1000) + abs(-5000 * 1.3000)) / 100000  # 17.5%
        assert abs(exposure - expected) < 0.001
    
    def test_calculate_leverage(self, risk_manager, sample_positions):
        """Test leverage calculation."""
        leverage = risk_manager._calculate_leverage(sample_positions, 100000.0)
        expected = (abs(10000 * 1.1000) + abs(-5000 * 1.3000)) / 100000  # 0.175x
        assert abs(leverage - expected) < 0.001


class TestPnLTracking:
    """Test P&L tracking and rapid loss detection."""
    
    @pytest.mark.asyncio
    async def test_update_pnl(self, risk_manager):
        """Test P&L update functionality."""
        await risk_manager.update_pnl(1000.0, 500.0, 500.0)
        
        assert len(risk_manager.pnl_history) == 1
        assert risk_manager.total_realized_pnl == 500.0
        
        pnl_record = risk_manager.pnl_history[0]
        assert pnl_record['total_pnl'] == 1000.0
        assert pnl_record['realized_pnl'] == 500.0
        assert pnl_record['unrealized_pnl'] == 500.0
    
    @pytest.mark.asyncio
    async def test_rapid_loss_detection(self, risk_manager):
        """Test rapid loss detection."""
        # Create P&L history with rapid loss
        base_time = datetime.now()
        
        # Add initial P&L
        risk_manager.pnl_history.append({
            'timestamp': base_time - timedelta(seconds=100),
            'total_pnl': 1000.0,
            'realized_pnl': 0,
            'unrealized_pnl': 1000.0,
            'current_capital': 100000
        })
        
        # Add rapid loss
        risk_manager.pnl_history.append({
            'timestamp': base_time,
            'total_pnl': -1500.0,  # 2.5% rapid loss
            'realized_pnl': 0,
            'unrealized_pnl': -1500.0,
            'current_capital': 100000
        })
        
        await risk_manager._check_rapid_loss()
        
        # Should generate rapid loss alert
        rapid_loss_alerts = [
            alert for alert in risk_manager.active_alerts.values()
            if alert.metric_name == "rapid_loss"
        ]
        assert len(rapid_loss_alerts) > 0


class TestAlertSystem:
    """Test alert generation and management."""
    
    @pytest.mark.asyncio
    async def test_send_alert(self, risk_manager):
        """Test alert sending functionality."""
        await risk_manager._send_alert(
            RiskLevel.WARNING,
            "Test alert message",
            "test_metric",
            0.05,
            0.03
        )
        
        assert len(risk_manager.active_alerts) == 1
        assert len(risk_manager.alert_history) == 1
        assert risk_manager.alerts_sent == 1
        
        alert = list(risk_manager.active_alerts.values())[0]
        assert alert.level == RiskLevel.WARNING
        assert alert.message == "Test alert message"
        assert alert.metric_name == "test_metric"
    
    @pytest.mark.asyncio
    async def test_alert_frequency_limiting(self, risk_manager):
        """Test alert frequency limiting."""
        # Send first alert
        await risk_manager._send_alert(
            RiskLevel.WARNING,
            "Test alert 1",
            "test_metric",
            0.05,
            0.03
        )
        
        # Send duplicate alert immediately (should be blocked)
        await risk_manager._send_alert(
            RiskLevel.WARNING,
            "Test alert 2",
            "test_metric",
            0.06,
            0.03
        )
        
        # Should only have one alert due to frequency limiting
        assert len(risk_manager.active_alerts) == 1
        assert risk_manager.alerts_sent == 1
    
    @pytest.mark.asyncio
    async def test_emergency_alert_frequency(self, risk_manager):
        """Test emergency alert frequency (shorter cooldown)."""
        # Send first emergency alert
        await risk_manager._send_alert(
            RiskLevel.EMERGENCY,
            "Emergency alert 1",
            "emergency_metric",
            1.0,
            0.0
        )
        
        # Emergency alerts have shorter cooldown, but still limited
        await risk_manager._send_alert(
            RiskLevel.EMERGENCY,
            "Emergency alert 2",
            "emergency_metric",
            1.0,
            0.0
        )
        
        # Should still be limited
        assert risk_manager.alerts_sent == 1


class TestStatusAndReporting:
    """Test status reporting and system state."""
    
    def test_get_status(self, risk_manager):
        """Test status reporting."""
        status = risk_manager.get_status()
        
        required_fields = [
            "is_emergency_mode", "is_trading_halted", "manual_override_active",
            "kill_switch_triggered", "active_alerts_count", "risk_checks_performed",
            "alerts_sent", "current_capital", "peak_capital", "position_count"
        ]
        
        for field in required_fields:
            assert field in status
        
        assert isinstance(status["is_emergency_mode"], bool)
        assert isinstance(status["current_capital"], (int, float))
        assert isinstance(status["risk_checks_performed"], int)
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, risk_manager, sample_order, sample_positions):
        """Test performance tracking metrics."""
        # Perform some risk checks
        await risk_manager.check_pre_trade_risk(sample_order, sample_positions, 100000.0)
        await risk_manager.check_pre_trade_risk(sample_order, {}, 100000.0)
        
        status = risk_manager.get_status()
        assert status["risk_checks_performed"] == 2
        
        # Send some alerts
        await risk_manager._send_alert(RiskLevel.INFO, "Test", "test", 1.0, 0.0)
        
        status = risk_manager.get_status()
        assert status["alerts_sent"] == 1


class TestFactoryFunction:
    """Test factory function for creating risk manager."""
    
    def test_create_risk_manager(self, sample_config):
        """Test factory function."""
        from core.risk_manager import create_risk_manager
        
        manager = create_risk_manager(
            config_path=sample_config,
            initial_capital=50000.0
        )
        
        assert isinstance(manager, CoreRiskManager)
        assert manager.initial_capital == 50000.0
        assert manager.config is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_positions_handling(self, risk_manager):
        """Test handling of empty positions."""
        metrics = await risk_manager.get_risk_metrics()
        assert metrics.position_count == 0
        assert metrics.total_exposure == 0.0
        assert metrics.leverage == 0.0
    
    @pytest.mark.asyncio
    async def test_zero_capital_handling(self, risk_manager, sample_order):
        """Test handling of zero capital."""
        result = await risk_manager.check_pre_trade_risk(
            sample_order, {}, 0.0
        )
        # Should handle gracefully without crashing
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_negative_pnl_handling(self, risk_manager):
        """Test handling of negative P&L."""
        await risk_manager.update_pnl(-5000.0, -2000.0, -3000.0)
        
        metrics = await risk_manager.get_risk_metrics()
        assert metrics.total_pnl == -5000.0
        assert metrics.realized_pnl == -2000.0
        assert metrics.unrealized_pnl == -3000.0
    
    def test_invalid_order_handling(self, risk_manager):
        """Test handling of invalid orders."""
        invalid_order = Order(
            order_id="invalid",
            symbol="INVALID",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,  # Invalid quantity
            price=None
        )
        
        # Should handle gracefully
        exposure = risk_manager._calculate_symbol_exposure("INVALID", {})
        assert exposure == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 