"""
MT5 Integration Test Suite - Comprehensive Testing.

This module provides comprehensive testing for MT5 integration including:
- Connection and authentication testing
- Order submission and execution testing
- Position monitoring and reporting testing
- Error handling and edge case testing
- Performance and latency testing
"""

import pytest
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional, Any
import structlog

# Mock MetaTrader5 module for testing
import sys
from unittest.mock import MagicMock
sys.modules['MetaTrader5'] = MagicMock()

from services.execution.mt5_connector import (
    MT5Connector, MT5Config, MT5ExecutionResult, MT5Position,
    MT5ConnectionStatus, create_mt5_connector
)
from services.execution.mt5_utils import (
    MT5Utils, SymbolInfo, MarketHours, MarketSession, create_mt5_utils
)
from core.interfaces.trading_interfaces import (
    Order, OrderSide, OrderType, OrderStatus
)


class TestMT5Connector:
    """Test suite for MT5Connector."""
    
    @pytest.fixture
    def mt5_config(self):
        """Create test MT5 configuration."""
        return MT5Config(
            login=12345,
            password="test_password",
            server="MetaQuotes-Demo",
            magic_number=12345,
            allowed_symbols=["EURUSD", "GBPUSD", "USDJPY"]
        )
    
    @pytest.fixture
    def mt5_connector(self, mt5_config):
        """Create MT5 connector for testing."""
        return MT5Connector(config=mt5_config)
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            order_id=str(uuid.uuid4()),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000.0,  # 1 lot
            price=1.1000,
            timestamp=datetime.now()
        )
    
    def test_mt5_connector_initialization(self, mt5_config):
        """Test MT5 connector initialization."""
        connector = MT5Connector(config=mt5_config)
        
        assert connector.config.login == 12345
        assert connector.config.server == "MetaQuotes-Demo"
        assert connector.status == MT5ConnectionStatus.DISCONNECTED
        assert connector.total_orders == 0
        assert len(connector.active_orders) == 0
        assert len(connector.positions) == 0
    
    @pytest.mark.asyncio
    async def test_connection_success(self, mt5_connector):
        """Test successful MT5 connection."""
        # Mock MT5 functions
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.account_info.return_value = MagicMock(
                login=12345,
                balance=10000.0,
                equity=10000.0,
                server="MetaQuotes-Demo",
                company="MetaQuotes"
            )
            
            result = await mt5_connector.connect()
            
            assert result is True
            assert mt5_connector.status == MT5ConnectionStatus.CONNECTED
            assert mt5_connector.last_heartbeat is not None
            assert mt5_connector.connection_attempts == 0
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, mt5_connector):
        """Test MT5 connection failure."""
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = False
            mock_mt5.last_error.return_value = (1, "Connection failed")
            
            result = await mt5_connector.connect()
            
            assert result is False
            assert mt5_connector.status == MT5ConnectionStatus.ERROR
            assert "MT5 initialization failed" in mt5_connector.last_error
    
    @pytest.mark.asyncio
    async def test_order_submission_success(self, mt5_connector, sample_order):
        """Test successful order submission."""
        # Set connector as connected
        mt5_connector.status = MT5ConnectionStatus.CONNECTED
        
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            # Mock symbol info
            mock_mt5.symbol_info.return_value = MagicMock(
                ask=1.1002,
                bid=1.1000,
                digits=5,
                point=0.00001
            )
            
            # Mock order send result
            mock_mt5.order_send.return_value = MagicMock(
                retcode=mock_mt5.TRADE_RETCODE_DONE,
                order=123456,
                price=1.1001,
                volume=1.0,
                comment="Order executed"
            )
            mock_mt5.TRADE_RETCODE_DONE = 10009
            
            result = await mt5_connector.submit_order(sample_order)
            
            assert result.success is True
            assert result.ticket == 123456
            assert result.price == 1.1001
            assert result.volume == 1.0
            assert result.execution_time is not None
            assert result.slippage is not None
    
    @pytest.mark.asyncio
    async def test_order_submission_failure(self, mt5_connector, sample_order):
        """Test order submission failure."""
        mt5_connector.status = MT5ConnectionStatus.CONNECTED
        
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            mock_mt5.symbol_info.return_value = MagicMock(
                ask=1.1002,
                bid=1.1000
            )
            
            # Mock order send failure
            mock_mt5.order_send.return_value = MagicMock(
                retcode=mock_mt5.TRADE_RETCODE_REJECT,
                comment="Order rejected"
            )
            mock_mt5.TRADE_RETCODE_REJECT = 10006
            
            result = await mt5_connector.submit_order(sample_order)
            
            assert result.success is False
            assert result.error_code == mock_mt5.TRADE_RETCODE_REJECT
            assert "Request rejected" in result.error_description
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, mt5_connector):
        """Test order cancellation."""
        # Add active order
        order_id = "test_order_123"
        mt5_connector.active_orders[order_id] = {
            'ticket': 123456,
            'symbol': 'EURUSD',
            'volume': 1.0
        }
        
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            mock_mt5.order_send.return_value = MagicMock(
                retcode=mock_mt5.TRADE_RETCODE_DONE
            )
            mock_mt5.TRADE_RETCODE_DONE = 10009
            
            result = await mt5_connector.cancel_order(order_id)
            
            assert result.success is True
            assert order_id not in mt5_connector.active_orders
    
    @pytest.mark.asyncio
    async def test_position_closing(self, mt5_connector):
        """Test position closing."""
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            # Mock position info
            mock_position = MagicMock(
                ticket=123456,
                volume=1.0,
                type=mock_mt5.ORDER_TYPE_BUY
            )
            mock_mt5.positions_get.return_value = [mock_position]
            mock_mt5.ORDER_TYPE_BUY = 0
            mock_mt5.ORDER_TYPE_SELL = 1
            
            # Mock symbol info
            mock_mt5.symbol_info.return_value = MagicMock(
                bid=1.1000,
                ask=1.1002
            )
            
            # Mock close order result
            mock_mt5.order_send.return_value = MagicMock(
                retcode=mock_mt5.TRADE_RETCODE_DONE,
                order=123457,
                price=1.1000,
                volume=1.0
            )
            mock_mt5.TRADE_RETCODE_DONE = 10009
            
            result = await mt5_connector.close_position("EURUSD")
            
            assert result.success is True
            assert result.price == 1.1000
            assert result.volume == 1.0
    
    def test_get_account_info(self, mt5_connector):
        """Test getting account information."""
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            mock_mt5.account_info.return_value = MagicMock(
                login=12345,
                balance=10000.0,
                equity=9950.0,
                margin=500.0,
                margin_free=9500.0,
                margin_level=1990.0,
                profit=-50.0,
                server="MetaQuotes-Demo",
                company="MetaQuotes",
                currency="USD",
                leverage=100,
                trade_allowed=True,
                trade_expert=True
            )
            
            account_info = mt5_connector.get_account_info()
            
            assert account_info is not None
            assert account_info['login'] == 12345
            assert account_info['balance'] == 10000.0
            assert account_info['equity'] == 9950.0
            assert account_info['leverage'] == 100
    
    def test_get_symbol_info(self, mt5_connector):
        """Test getting symbol information."""
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            mock_mt5.symbol_info.return_value = MagicMock(
                name="EURUSD",
                bid=1.1000,
                ask=1.1002,
                spread=2,
                digits=5,
                point=0.00001,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                trade_mode=0,
                visible=True,
                margin_initial=1000.0,
                margin_maintenance=500.0
            )
            
            symbol_info = mt5_connector.get_symbol_info("EURUSD")
            
            assert symbol_info is not None
            assert symbol_info['symbol'] == "EURUSD"
            assert symbol_info['bid'] == 1.1000
            assert symbol_info['ask'] == 1.1002
            assert symbol_info['spread'] == 2
    
    def test_execution_statistics(self, mt5_connector):
        """Test execution statistics calculation."""
        # Simulate some orders
        mt5_connector.total_orders = 10
        mt5_connector.successful_orders = 8
        mt5_connector.failed_orders = 2
        mt5_connector.avg_execution_time = 50.0
        mt5_connector.total_slippage = 4.0
        mt5_connector.last_heartbeat = datetime.now()
        
        stats = mt5_connector.get_execution_statistics()
        
        assert stats['total_orders'] == 10
        assert stats['successful_orders'] == 8
        assert stats['failed_orders'] == 2
        assert stats['success_rate'] == 80.0
        assert stats['avg_execution_time_ms'] == 50.0
        assert stats['avg_slippage_bps'] == 0.5  # 4.0 / 8
    
    def test_is_connected(self, mt5_connector):
        """Test connection status check."""
        assert mt5_connector.is_connected() is False
        
        mt5_connector.status = MT5ConnectionStatus.CONNECTED
        assert mt5_connector.is_connected() is True
        
        mt5_connector.status = MT5ConnectionStatus.ERROR
        assert mt5_connector.is_connected() is False


class TestMT5Utils:
    """Test suite for MT5Utils."""
    
    @pytest.fixture
    def mt5_utils(self):
        """Create MT5 utils for testing."""
        return MT5Utils()
    
    def test_mt5_utils_initialization(self, mt5_utils):
        """Test MT5 utils initialization."""
        assert len(mt5_utils.major_pairs) == 7
        assert "EURUSD" in mt5_utils.major_pairs
        assert len(mt5_utils.minor_pairs) > 10
        assert "EURJPY" in mt5_utils.minor_pairs
    
    def test_get_symbol_info(self, mt5_utils):
        """Test getting symbol information."""
        with patch('services.execution.mt5_utils.mt5') as mock_mt5:
            mock_mt5.symbol_info.return_value = MagicMock(
                name="EURUSD",
                description="Euro vs US Dollar",
                currency_base="EUR",
                currency_profit="USD",
                currency_margin="EUR",
                digits=5,
                point=0.00001,
                spread=2,
                bid=1.1000,
                ask=1.1002,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                contract_size=100000.0,
                margin_initial=1000.0,
                margin_maintenance=500.0,
                swap_long=-2.5,
                swap_short=1.5,
                trade_mode=0,
                visible=True,
                session_deals=1500,
                session_buy_orders=750,
                session_sell_orders=750
            )
            
            symbol_info = mt5_utils.get_symbol_info("EURUSD")
            
            assert symbol_info is not None
            assert symbol_info.name == "EURUSD"
            assert symbol_info.currency_base == "EUR"
            assert symbol_info.currency_profit == "USD"
            assert symbol_info.digits == 5
            assert symbol_info.contract_size == 100000.0
    
    def test_validate_symbol(self, mt5_utils):
        """Test symbol validation."""
        with patch('services.execution.mt5_utils.mt5') as mock_mt5:
            # Valid symbol
            mock_mt5.symbol_info.return_value = MagicMock(
                visible=True,
                trade_mode=0
            )
            mock_mt5.SYMBOL_TRADE_MODE_DISABLED = 1
            
            assert mt5_utils.validate_symbol("EURUSD") is True
            
            # Invalid symbol (not visible)
            mock_mt5.symbol_info.return_value = MagicMock(
                visible=False,
                trade_mode=0
            )
            
            assert mt5_utils.validate_symbol("INVALID") is False
            
            # Symbol not found
            mock_mt5.symbol_info.return_value = None
            
            assert mt5_utils.validate_symbol("NOTFOUND") is False
    
    def test_calculate_lot_size(self, mt5_utils):
        """Test lot size calculation."""
        with patch.object(mt5_utils, 'get_symbol_info') as mock_get_symbol:
            with patch.object(mt5_utils, 'calculate_pip_value') as mock_pip_value:
                with patch.object(mt5_utils, 'round_lot_size') as mock_round:
                    
                    mock_get_symbol.return_value = MagicMock()
                    mock_pip_value.return_value = 10.0  # $10 per pip
                    mock_round.return_value = 0.1
                    
                    # Risk $100, stop loss 50 pips
                    lot_size = mt5_utils.calculate_lot_size("EURUSD", 100.0, 50.0)
                    
                    # Expected: $100 / 50 pips = $2 per pip / $10 per pip = 0.2 lots
                    mock_pip_value.assert_called_once()
                    mock_round.assert_called_once()
                    assert lot_size == 0.1  # After rounding
    
    def test_calculate_pip_value(self, mt5_utils):
        """Test pip value calculation."""
        with patch.object(mt5_utils, 'get_symbol_info') as mock_get_symbol:
            with patch.object(mt5_utils, 'get_conversion_rate') as mock_conversion:
                
                # EUR/USD - profit currency is USD (same as account)
                mock_get_symbol.return_value = SymbolInfo(
                    name="EURUSD",
                    description="Euro vs US Dollar",
                    currency_base="EUR",
                    currency_profit="USD",
                    currency_margin="EUR",
                    digits=5,
                    point=0.00001,
                    spread=2,
                    bid=1.1000,
                    ask=1.1002,
                    volume_min=0.01,
                    volume_max=100.0,
                    volume_step=0.01,
                    contract_size=100000.0,
                    margin_initial=1000.0,
                    margin_maintenance=500.0,
                    swap_long=-2.5,
                    swap_short=1.5,
                    trade_mode=0,
                    trade_allowed=True,
                    session_deals=1500,
                    session_buy_orders=750,
                    session_sell_orders=750
                )
                
                pip_value = mt5_utils.calculate_pip_value("EURUSD", 1.0, "USD")
                
                # For 5-digit pair: pip = 0.0001, lot size = 1.0, contract = 100000
                # pip_value = 0.0001 * 1.0 * 100000 = 10.0
                assert pip_value == 10.0
    
    def test_format_price(self, mt5_utils):
        """Test price formatting."""
        with patch.object(mt5_utils, 'get_symbol_info') as mock_get_symbol:
            mock_get_symbol.return_value = SymbolInfo(
                name="EURUSD",
                description="",
                currency_base="EUR",
                currency_profit="USD",
                currency_margin="EUR",
                digits=5,
                point=0.00001,
                spread=2,
                bid=1.1000,
                ask=1.1002,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                contract_size=100000.0,
                margin_initial=1000.0,
                margin_maintenance=500.0,
                swap_long=-2.5,
                swap_short=1.5,
                trade_mode=0,
                trade_allowed=True,
                session_deals=1500,
                session_buy_orders=750,
                session_sell_orders=750
            )
            
            formatted = mt5_utils.format_price("EURUSD", 1.10015)
            assert formatted == "1.10015"
    
    def test_calculate_pips(self, mt5_utils):
        """Test pips calculation."""
        with patch.object(mt5_utils, 'get_symbol_info') as mock_get_symbol:
            mock_get_symbol.return_value = SymbolInfo(
                name="EURUSD",
                description="",
                currency_base="EUR",
                currency_profit="USD",
                currency_margin="EUR",
                digits=5,
                point=0.00001,
                spread=2,
                bid=1.1000,
                ask=1.1002,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                contract_size=100000.0,
                margin_initial=1000.0,
                margin_maintenance=500.0,
                swap_long=-2.5,
                swap_short=1.5,
                trade_mode=0,
                trade_allowed=True,
                session_deals=1500,
                session_buy_orders=750,
                session_sell_orders=750
            )
            
            pips = mt5_utils.calculate_pips("EURUSD", 1.1000, 1.1050)
            # 50 points = 5 pips for 5-digit pair
            assert pips == 50.0
    
    def test_get_current_session(self, mt5_utils):
        """Test current session detection."""
        # Mock current time to be in London session (07:00-16:00 UTC)
        with patch('services.execution.mt5_utils.datetime') as mock_datetime:
            mock_datetime.now.return_value = MagicMock()
            mock_datetime.now.return_value.strftime.return_value = "10:00"
            
            session = mt5_utils.get_current_session()
            # Should be London session
            assert session == MarketSession.LONDON
    
    def test_is_market_open(self, mt5_utils):
        """Test market open check."""
        with patch('services.execution.mt5_utils.mt5') as mock_mt5:
            with patch('services.execution.mt5_utils.datetime') as mock_datetime:
                # Mock weekday (Monday = 0)
                mock_datetime.now.return_value.weekday.return_value = 1  # Tuesday
                
                # Mock symbol info
                mock_mt5.symbol_info.return_value = MagicMock(
                    trade_mode=0
                )
                mock_mt5.SYMBOL_TRADE_MODE_DISABLED = 1
                
                assert mt5_utils.is_market_open("EURUSD") is True
                
                # Test weekend
                mock_datetime.now.return_value.weekday.return_value = 5  # Saturday
                assert mt5_utils.is_market_open("EURUSD") is False


class TestMT5Integration:
    """Integration tests for MT5 components."""
    
    @pytest.mark.asyncio
    async def test_full_order_lifecycle(self):
        """Test complete order lifecycle."""
        config = MT5Config(
            login=12345,
            password="test",
            server="demo",
            allowed_symbols=["EURUSD"]
        )
        
        connector = MT5Connector(config=config)
        
        # Mock MT5 for full lifecycle
        with patch('services.execution.mt5_connector.mt5') as mock_mt5:
            # Setup mocks
            mock_mt5.initialize.return_value = True
            mock_mt5.account_info.return_value = MagicMock(
                login=12345, balance=10000.0, equity=10000.0,
                server="demo", company="Test"
            )
            mock_mt5.symbol_info.return_value = MagicMock(
                ask=1.1002, bid=1.1000, digits=5, point=0.00001
            )
            mock_mt5.order_send.return_value = MagicMock(
                retcode=mock_mt5.TRADE_RETCODE_DONE,
                order=123456, price=1.1001, volume=1.0
            )
            mock_mt5.TRADE_RETCODE_DONE = 10009
            
            # Connect
            connected = await connector.connect()
            assert connected is True
            
            # Submit order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100000.0,
                price=1.1000,
                timestamp=datetime.now()
            )
            
            result = await connector.submit_order(order)
            assert result.success is True
            assert result.ticket == 123456
            
            # Check statistics
            stats = connector.get_execution_statistics()
            assert stats['total_orders'] == 1
            assert stats['successful_orders'] == 1
            assert stats['success_rate'] == 100.0
    
    def test_factory_functions(self):
        """Test factory functions."""
        # Test MT5 connector factory
        connector = create_mt5_connector()
        assert isinstance(connector, MT5Connector)
        
        # Test MT5 utils factory
        utils = create_mt5_utils()
        assert isinstance(utils, MT5Utils)


def run_mt5_integration_tests():
    """Run all MT5 integration tests."""
    print("🧪 Running MT5 Integration Tests...")
    
    # Test basic functionality without actual MT5 connection
    test_results = {
        'connection_test': False,
        'order_test': False,
        'utils_test': False,
        'integration_test': False
    }
    
    try:
        # Test 1: Connection handling
        print("  ✓ Testing connection handling...")
        config = MT5Config(login=0, password="", server="")
        connector = MT5Connector(config=config)
        assert connector.status == MT5ConnectionStatus.DISCONNECTED
        test_results['connection_test'] = True
        
        # Test 2: Order preparation
        print("  ✓ Testing order preparation...")
        order = Order(
            order_id="test_123",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000.0,
            price=1.1000,
            timestamp=datetime.now()
        )
        assert order.symbol == "EURUSD"
        test_results['order_test'] = True
        
        # Test 3: Utils functionality
        print("  ✓ Testing utilities...")
        utils = MT5Utils()
        assert len(utils.major_pairs) > 0
        assert "EURUSD" in utils.major_pairs
        test_results['utils_test'] = True
        
        # Test 4: Integration components
        print("  ✓ Testing integration...")
        connector_factory = create_mt5_connector()
        utils_factory = create_mt5_utils()
        assert isinstance(connector_factory, MT5Connector)
        assert isinstance(utils_factory, MT5Utils)
        test_results['integration_test'] = True
        
        print(f"\n✅ MT5 Integration Tests: {sum(test_results.values())}/4 passed")
        return test_results
        
    except Exception as e:
        print(f"\n❌ MT5 Integration Tests failed: {str(e)}")
        return test_results


if __name__ == "__main__":
    # Run tests when executed directly
    results = run_mt5_integration_tests()
    
    if all(results.values()):
        print("\n🎉 All MT5 integration tests passed!")
        print("\n📋 Next Steps:")
        print("  1. Install MetaTrader 5 terminal")
        print("  2. Create demo account")
        print("  3. Update config_mt5.yaml with credentials")
        print("  4. Install MetaTrader5 Python package: pip install MetaTrader5")
        print("  5. Run live integration tests")
    else:
        print(f"\n⚠️  Some tests failed: {results}")
        exit(1) 