"""
Unit tests for data ingestion module.
"""

from datetime import datetime

import pytest

from core.interfaces.data_interfaces import MarketData
from data.base_connector import ConnectionConfig, ConnectionStatus
from data.mock_provider import MockFXConnector, MockMarketDataGenerator


class TestMockMarketDataGenerator:
    """Test cases for mock market data generator."""

    def test_generator_initialization(self):
        """Test generator creates with correct initial state."""
        generator = MockMarketDataGenerator("EUR/USD", 1.0850)

        assert generator.symbol == "EUR/USD"
        assert generator.current_price == 1.0850
        assert generator.initial_price == 1.0850
        assert generator.volatility > 0
        assert generator.spread > 0

    def test_market_data_generation(self):
        """Test market data generation produces valid data."""
        generator = MockMarketDataGenerator("EUR/USD", 1.0850)

        market_data = generator.generate_market_data()

        assert market_data.symbol == "EUR/USD"
        assert isinstance(market_data.timestamp, datetime)
        assert market_data.bid > 0
        assert market_data.ask > 0
        assert market_data.ask > market_data.bid  # Spread should exist
        assert market_data.volume > 0
        assert market_data.source == "mock_provider"


class TestMockFXConnector:
    """Test cases for mock FX connector."""

    @pytest.fixture
    def connector_config(self):
        """Create a test connector configuration."""
        return ConnectionConfig(
            symbols=["EUR/USD", "GBP/USD"], timeout=10, max_retries=3
        )

    @pytest.fixture
    def mock_connector(self, connector_config):
        """Create a mock connector for testing."""
        return MockFXConnector(connector_config)

    def test_connector_initialization(self, mock_connector):
        """Test connector initializes correctly."""
        assert mock_connector.status == ConnectionStatus.DISCONNECTED
        assert not mock_connector.is_connected
        assert len(mock_connector.generators) == 2  # EUR/USD and GBP/USD
        assert "EUR/USD" in mock_connector.generators
        assert "GBP/USD" in mock_connector.generators

    @pytest.mark.asyncio
    async def test_connector_connection(self, mock_connector):
        """Test connector connection process."""
        # Test connection
        success = await mock_connector.connect()

        assert success
        assert mock_connector.status == ConnectionStatus.CONNECTED
        assert mock_connector.is_connected
        assert mock_connector.uptime is not None

        # Test disconnection
        await mock_connector.disconnect()

        assert mock_connector.status == ConnectionStatus.DISCONNECTED
        assert not mock_connector.is_connected

    @pytest.mark.asyncio
    async def test_data_streaming(self, mock_connector):
        """Test data streaming functionality."""
        await mock_connector.connect()

        # Collect some market data
        market_data_list = []
        count = 0

        async for market_data in mock_connector.stream():
            market_data_list.append(market_data)
            count += 1
            if count >= 5:  # Collect 5 ticks
                break

        await mock_connector.disconnect()

        assert len(market_data_list) == 5

        # Verify market data properties
        for md in market_data_list:
            assert isinstance(md, MarketData)
            assert md.bid > 0
            assert md.ask > md.bid
            assert md.volume > 0
