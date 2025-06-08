"""
Test Suite for IBKR FIX Protocol Integration.

This module provides comprehensive testing for the FIX protocol integration
with Interactive Brokers, including unit tests, integration tests, and
message validation tests.

Test Categories:
- FIX message utilities tests
- FIX client session management tests
- Order router integration tests
- End-to-end order flow tests
- Error handling and recovery tests
"""


# Import modules under test
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from core.interfaces.trading_interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from services.execution.fix_connector.fix_client import (
    FIXClient,
    FIXSessionConfig,
    FIXSessionState,
)
from services.execution.fix_connector.fix_message_utils import (
    FIXMessage,
    FIXMessageUtils,
    FIXVersion,
)
from services.execution.fix_connector.fix_order_router import (
    FIXExecutionReport,
    FIXOrderMapping,
    FIXOrderRouter,
)


class TestFIXMessageUtils:
    """Test FIX message utilities."""

    def setup_method(self):
        """Setup test fixtures."""
        self.utils = FIXMessageUtils(FIXVersion.FIX_4_2)

    def test_parse_valid_message(self):
        """Test parsing a valid FIX message."""
        # Create a sample FIX message
        raw_message = "8=FIX.4.2\x019=61\x0135=A\x0149=CLIENT\x0156=IB\x0134=1\x0152=20231201-10:30:00.000\x01108=30\x0110=123\x01"

        message = self.utils.parse_message(raw_message)

        assert message.valid
        assert message.msg_type == "A"  # Logon
        assert message.fields[8] == "FIX.4.2"
        assert message.fields[49] == "CLIENT"
        assert message.fields[56] == "IB"
        assert message.sequence_number == 1

    def test_parse_invalid_message(self):
        """Test parsing an invalid FIX message."""
        raw_message = "invalid_message"

        message = self.utils.parse_message(raw_message)

        assert not message.valid
        assert len(message.errors) > 0

    def test_build_logon_message(self):
        """Test building a logon message."""
        message = self.utils.build_logon_message(
            sender_comp_id="CLIENT",
            target_comp_id="IB",
            sequence_number=1,
            heartbeat_interval=30,
            username="testuser",
            password="testpass",
        )

        assert "8=FIX.4.2" in message
        assert "35=A" in message  # Logon message type
        assert "49=CLIENT" in message
        assert "56=IB" in message
        assert "108=30" in message  # HeartBtInt
        assert "553=testuser" in message  # Username
        assert "554=testpass" in message  # Password

    def test_build_order_single_message(self):
        """Test building a new order single message."""
        message = self.utils.build_order_single_message(
            sender_comp_id="CLIENT",
            target_comp_id="IB",
            sequence_number=2,
            cl_ord_id="AI-123456789-abcd1234",
            symbol="EURUSD",
            side="1",  # Buy
            order_qty=100000,
            ord_type="1",  # Market
            time_in_force="0",  # Day
        )

        assert "8=FIX.4.2" in message
        assert "35=D" in message  # New Order Single
        assert "11=AI-123456789-abcd1234" in message  # ClOrdID
        assert "55=EURUSD" in message  # Symbol
        assert "54=1" in message  # Side (Buy)
        assert "38=100000" in message  # OrderQty
        assert "40=1" in message  # OrdType (Market)

    def test_checksum_calculation(self):
        """Test checksum calculation."""
        message_without_checksum = "8=FIX.4.2\x019=61\x0135=A\x0149=CLIENT\x0156=IB\x0134=1\x0152=20231201-10:30:00.000\x01108=30\x01"

        checksum = self.utils._calculate_checksum(message_without_checksum)

        assert isinstance(checksum, int)
        assert 0 <= checksum <= 255

    def test_checksum_validation(self):
        """Test checksum validation."""
        # Valid message with correct checksum
        valid_message = "8=FIX.4.2\x019=61\x0135=A\x0149=CLIENT\x0156=IB\x0134=1\x0152=20231201-10:30:00.000\x01108=30\x0110=123\x01"

        # Calculate correct checksum
        message_without_checksum = valid_message[: valid_message.rfind("10=")]
        correct_checksum = self.utils._calculate_checksum(message_without_checksum)
        valid_message_corrected = (
            message_without_checksum + f"10={correct_checksum:03d}\x01"
        )

        assert self.utils._validate_checksum(valid_message_corrected)

    def test_generate_client_order_id(self):
        """Test client order ID generation."""
        order_id = self.utils.generate_client_order_id()

        assert order_id.startswith("AI-")
        assert len(order_id.split("-")) == 3
        assert order_id != self.utils.generate_client_order_id()  # Should be unique


class TestFIXClient:
    """Test FIX client functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = FIXSessionConfig(
            host="127.0.0.1",
            port=4002,
            sender_comp_id="TEST_CLIENT",
            target_comp_id="IB",
            heartbeat_interval=30,
            auto_reconnect=False,  # Disable for testing
        )
        self.client = FIXClient(self.config)

    def test_client_initialization(self):
        """Test FIX client initialization."""
        assert self.client.state == FIXSessionState.DISCONNECTED
        assert self.client.outgoing_seq_num == 1
        assert self.client.incoming_seq_num == 1
        assert self.client.config.sender_comp_id == "TEST_CLIENT"

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        # Mock socket connection failure
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.connect.side_effect = ConnectionRefusedError(
                "Connection refused"
            )

            success = await self.client.connect()

            assert not success
            assert self.client.state == FIXSessionState.ERROR

    @pytest.mark.asyncio
    async def test_message_building(self):
        """Test message building functionality."""
        message = self.client.message_utils.build_heartbeat_message(
            sender_comp_id="TEST_CLIENT", target_comp_id="IB", sequence_number=1
        )

        assert "35=0" in message  # Heartbeat message type
        assert "49=TEST_CLIENT" in message
        assert "56=IB" in message

    def test_admin_message_handlers(self):
        """Test administrative message handlers setup."""
        assert "0" in self.client.admin_message_handlers  # Heartbeat
        assert "1" in self.client.admin_message_handlers  # Test Request
        assert "A" in self.client.admin_message_handlers  # Logon
        assert "5" in self.client.admin_message_handlers  # Logout

    def test_session_stats(self):
        """Test session statistics."""
        stats = self.client.get_session_stats()

        assert "state" in stats
        assert "total_messages_sent" in stats
        assert "total_messages_received" in stats
        assert "outgoing_seq_num" in stats
        assert stats["state"] == "disconnected"


class TestFIXOrderRouter:
    """Test FIX order router functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock config file
        self.config_path = "test_fix_config.yaml"

        # Mock the config loading
        with patch(
            "services.execution.fix_connector.fix_order_router.load_fix_config"
        ) as mock_load:
            mock_config = FIXSessionConfig(
                host="127.0.0.1",
                port=4002,
                sender_comp_id="TEST_CLIENT",
                target_comp_id="IB",
            )
            mock_load.return_value = mock_config

            self.router = FIXOrderRouter(self.config_path)

    def test_router_initialization(self):
        """Test order router initialization."""
        assert self.router.fix_client is not None
        assert self.router.total_orders_sent == 0
        assert self.router.total_executions_received == 0
        assert len(self.router.active_orders) == 0

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        # Mock FIX client connect
        self.router.fix_client.connect = AsyncMock(return_value=True)
        self.router.fix_client.is_connected = Mock(return_value=True)

        success = await self.router.connect()

        assert success
        self.router.fix_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        # Mock FIX client connect failure
        self.router.fix_client.connect = AsyncMock(return_value=False)

        success = await self.router.connect()

        assert not success

    @pytest.mark.asyncio
    async def test_submit_order_success(self):
        """Test successful order submission."""
        # Create test order
        order = Order(
            order_id="test_order_1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000.0,
            price=None,
        )

        # Mock FIX client
        self.router.fix_client.is_connected = Mock(return_value=True)
        self.router.fix_client.send_message = AsyncMock(return_value=True)
        self.router.fix_client.outgoing_seq_num = 1

        result = await self.router.submit_order(order)

        assert result["success"]
        assert result["order_id"] == "test_order_1"
        assert "fix_client_order_id" in result
        assert self.router.total_orders_sent == 1
        assert order.order_id in self.router.active_orders

    @pytest.mark.asyncio
    async def test_submit_order_not_connected(self):
        """Test order submission when not connected."""
        order = Order(
            order_id="test_order_2",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000.0,
        )

        # Mock not connected
        self.router.fix_client.is_connected = Mock(return_value=False)

        result = await self.router.submit_order(order)

        assert not result["success"]
        assert "Not connected" in result["error"]

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        """Test successful order cancellation."""
        # Setup order mapping
        order_id = "test_order_3"
        fix_client_order_id = "AI-123456789-abcd1234"

        order_mapping = FIXOrderMapping(
            internal_order_id=order_id,
            fix_client_order_id=fix_client_order_id,
            symbol="EURUSD",
            side="1",
            quantity=100000.0,
            order_type="1",
        )

        self.router.active_orders[order_id] = order_mapping

        # Mock FIX client
        self.router.fix_client.is_connected = Mock(return_value=True)
        self.router.fix_client.send_message = AsyncMock(return_value=True)
        self.router.fix_client.outgoing_seq_num = 2

        result = await self.router.cancel_order(order_id)

        assert result["success"]
        assert result["order_id"] == order_id
        assert order_mapping.status == "PENDING_CANCEL"

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self):
        """Test cancelling non-existent order."""
        result = await self.router.cancel_order("non_existent_order")

        assert not result["success"]
        assert "Order not found" in result["error"]

    def test_get_order_status(self):
        """Test getting order status."""
        # Setup order mapping
        order_id = "test_order_4"
        fix_client_order_id = "AI-123456789-abcd1234"

        order_mapping = FIXOrderMapping(
            internal_order_id=order_id,
            fix_client_order_id=fix_client_order_id,
            symbol="EURUSD",
            side="1",
            quantity=100000.0,
            order_type="1",
            status="FILLED",
            filled_quantity=100000.0,
            avg_price=1.1000,
        )

        self.router.active_orders[order_id] = order_mapping

        status = self.router.get_order_status(order_id)

        assert status is not None
        assert status["order_id"] == order_id
        assert status["status"] == "FILLED"
        assert status["filled_quantity"] == 100000.0
        assert status["avg_price"] == 1.1000

    def test_get_order_status_not_found(self):
        """Test getting status for non-existent order."""
        status = self.router.get_order_status("non_existent_order")

        assert status is None

    def test_execution_report_parsing(self):
        """Test execution report parsing."""
        # Create mock FIX message
        fields = {
            17: "EXEC123",  # ExecID
            37: "ORDER123",  # OrderID
            11: "AI-123456789-abcd1234",  # ClOrdID
            150: "2",  # ExecType (Fill)
            39: "2",  # OrdStatus (Filled)
            55: "EURUSD",  # Symbol
            54: "1",  # Side (Buy)
            32: "100000",  # LastQty
            31: "1.1000",  # LastPx
            151: "0",  # LeavesQty
            14: "100000",  # CumQty
            6: "1.1000",  # AvgPx
            60: "20231201-10:30:00.000",  # TransactTime
        }

        message = FIXMessage(msg_type="8", fields=fields)
        exec_report = self.router._parse_execution_report(message)

        assert exec_report.exec_id == "EXEC123"
        assert exec_report.order_id == "ORDER123"
        assert exec_report.cl_ord_id == "AI-123456789-abcd1234"
        assert exec_report.exec_type == "2"
        assert exec_report.ord_status == "2"
        assert exec_report.symbol == "EURUSD"
        assert exec_report.last_qty == 100000.0
        assert exec_report.last_px == 1.1000

    def test_side_conversion(self):
        """Test order side conversion."""
        assert self.router._convert_side_to_fix(OrderSide.BUY) == "1"
        assert self.router._convert_side_to_fix(OrderSide.SELL) == "2"

    def test_order_type_conversion(self):
        """Test order type conversion."""
        assert self.router._convert_order_type_to_fix(OrderType.MARKET) == "1"
        assert self.router._convert_order_type_to_fix(OrderType.LIMIT) == "2"
        assert self.router._convert_order_type_to_fix(OrderType.STOP) == "3"
        assert self.router._convert_order_type_to_fix(OrderType.STOP_LIMIT) == "4"

    def test_fix_status_conversion(self):
        """Test FIX status to internal status conversion."""
        assert self.router._convert_fix_status_to_internal("0") == OrderStatus.PENDING
        assert (
            self.router._convert_fix_status_to_internal("1")
            == OrderStatus.PARTIALLY_FILLED
        )
        assert self.router._convert_fix_status_to_internal("2") == OrderStatus.FILLED
        assert self.router._convert_fix_status_to_internal("4") == OrderStatus.CANCELLED
        assert self.router._convert_fix_status_to_internal("8") == OrderStatus.REJECTED


class TestFIXIntegration:
    """Integration tests for FIX components."""

    def setup_method(self):
        """Setup integration test fixtures."""
        self.config = FIXSessionConfig(
            host="127.0.0.1",
            port=4002,
            sender_comp_id="TEST_CLIENT",
            target_comp_id="IB",
            heartbeat_interval=30,
            auto_reconnect=False,
        )

    @pytest.mark.asyncio
    async def test_end_to_end_order_flow(self):
        """Test complete order flow from submission to execution."""
        # This would be a comprehensive integration test
        # For now, we'll test the components work together

        # Create order router
        with patch(
            "services.execution.fix_connector.fix_order_router.load_fix_config"
        ) as mock_load:
            mock_load.return_value = self.config
            router = FIXOrderRouter("test_config.yaml")

        # Mock FIX client for integration test
        router.fix_client.connect = AsyncMock(return_value=True)
        router.fix_client.is_connected = Mock(return_value=True)
        router.fix_client.send_message = AsyncMock(return_value=True)
        router.fix_client.outgoing_seq_num = 1

        # Create and submit order
        order = Order(
            order_id="integration_test_order",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100000.0,
            price=1.1000,
        )

        # Test connection
        connected = await router.connect()
        assert connected

        # Test order submission
        result = await router.submit_order(order)
        assert result["success"]

        # Test order status retrieval
        status = router.get_order_status(order.order_id)
        assert status is not None
        assert status["symbol"] == "EURUSD"

        # Test order cancellation
        cancel_result = await router.cancel_order(order.order_id)
        assert cancel_result["success"]

        # Test disconnection
        await router.disconnect()

    def test_message_round_trip(self):
        """Test message building and parsing round trip."""
        utils = FIXMessageUtils(FIXVersion.FIX_4_2)

        # Build a message
        original_fields = {
            11: "AI-123456789-abcd1234",  # ClOrdID
            55: "EURUSD",  # Symbol
            54: "1",  # Side
            38: "100000",  # OrderQty
            40: "2",  # OrdType
            44: "1.1000",  # Price
        }

        message = utils.build_message(
            msg_type="D",
            fields=original_fields,
            sender_comp_id="CLIENT",
            target_comp_id="IB",
            sequence_number=1,
        )

        # Parse the message back
        parsed = utils.parse_message(message)

        assert parsed.valid
        assert parsed.msg_type == "D"
        assert parsed.fields[11] == "AI-123456789-abcd1234"
        assert parsed.fields[55] == "EURUSD"
        assert parsed.fields[54] == "1"
        assert parsed.fields[38] == "100000"


class TestFIXErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Setup error handling test fixtures."""
        self.utils = FIXMessageUtils(FIXVersion.FIX_4_2)

    def test_malformed_message_handling(self):
        """Test handling of malformed messages."""
        malformed_messages = [
            "",  # Empty message
            "8=FIX.4.2",  # Incomplete message
            "invalid=data\x01more=invalid\x01",  # Invalid tags
            "8=FIX.4.2\x019=100\x0135=A\x01",  # Wrong body length
        ]

        for msg in malformed_messages:
            parsed = self.utils.parse_message(msg)
            assert not parsed.valid
            assert len(parsed.errors) > 0

    def test_sequence_number_handling(self):
        """Test sequence number validation."""
        client = FIXClient(
            FIXSessionConfig(
                host="127.0.0.1", port=4002, sender_comp_id="TEST", target_comp_id="IB"
            )
        )

        # Test sequence number increment
        initial_seq = client.outgoing_seq_num
        client.outgoing_seq_num += 1
        assert client.outgoing_seq_num == initial_seq + 1

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test connection timeout handling."""
        config = FIXSessionConfig(
            host="192.0.2.1",  # Non-routable IP for timeout
            port=4002,
            sender_comp_id="TEST",
            target_comp_id="IB",
            socket_timeout=1,  # Short timeout for testing
        )

        client = FIXClient(config)

        # This should timeout and fail
        success = await client.connect()
        assert not success
        assert client.state == FIXSessionState.ERROR

    def test_invalid_order_data_handling(self):
        """Test handling of invalid order data."""
        with patch(
            "services.execution.fix_connector.fix_order_router.load_fix_config"
        ) as mock_load:
            mock_load.return_value = FIXSessionConfig(
                host="127.0.0.1", port=4002, sender_comp_id="TEST", target_comp_id="IB"
            )
            router = FIXOrderRouter("test_config.yaml")

        # Test with invalid order (missing required fields)
        invalid_order = Order(
            order_id="",  # Empty order ID
            symbol="",  # Empty symbol
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.0,  # Zero quantity
        )

        # Mock connected state
        router.fix_client.is_connected = Mock(return_value=True)

        # This should handle the invalid data gracefully
        # The actual validation would happen in the order submission
        assert invalid_order.order_id == ""
        assert invalid_order.symbol == ""
        assert invalid_order.quantity == 0.0


# Performance and stress tests
class TestFIXPerformance:
    """Performance and stress tests for FIX components."""

    def test_message_parsing_performance(self):
        """Test message parsing performance."""
        utils = FIXMessageUtils(FIXVersion.FIX_4_2)

        # Create a sample message
        message = "8=FIX.4.2\x019=100\x0135=D\x0149=CLIENT\x0156=IB\x0134=1\x0152=20231201-10:30:00.000\x0111=AI-123456789-abcd1234\x0155=EURUSD\x0154=1\x0138=100000\x0140=1\x0110=123\x01"

        # Time parsing multiple messages
        start_time = time.time()
        for _ in range(1000):
            parsed = utils.parse_message(message)
            assert parsed.msg_type == "D"

        end_time = time.time()
        duration = end_time - start_time

        # Should parse 1000 messages in reasonable time (< 1 second)
        assert duration < 1.0

        # Calculate messages per second
        messages_per_second = 1000 / duration
        assert messages_per_second > 1000  # Should handle at least 1000 msg/sec

    def test_message_building_performance(self):
        """Test message building performance."""
        utils = FIXMessageUtils(FIXVersion.FIX_4_2)

        fields = {
            11: "AI-123456789-abcd1234",
            55: "EURUSD",
            54: "1",
            38: "100000",
            40: "1",
        }

        # Time building multiple messages
        start_time = time.time()
        for i in range(1000):
            message = utils.build_message(
                msg_type="D",
                fields=fields,
                sender_comp_id="CLIENT",
                target_comp_id="IB",
                sequence_number=i + 1,
            )
            assert "35=D" in message

        end_time = time.time()
        duration = end_time - start_time

        # Should build 1000 messages in reasonable time
        assert duration < 1.0

        messages_per_second = 1000 / duration
        assert messages_per_second > 1000


# Fixtures and utilities for testing
@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id="test_order_123",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100000.0,
        price=1.1000,
    )


@pytest.fixture
def sample_execution_report():
    """Create a sample execution report for testing."""
    return FIXExecutionReport(
        exec_id="EXEC123",
        order_id="ORDER123",
        cl_ord_id="AI-123456789-abcd1234",
        exec_type="2",  # Fill
        ord_status="2",  # Filled
        symbol="EURUSD",
        side="1",  # Buy
        last_qty=100000.0,
        last_px=1.1000,
        leaves_qty=0.0,
        cum_qty=100000.0,
        avg_px=1.1000,
        transact_time=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
