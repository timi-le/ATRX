"""
Unit tests for OANDA connector.
"""

import re
from datetime import datetime

import pytest
from aioresponses import aioresponses

from data.base_connector import ConnectionStatus
from data.oanda_connector import OandaConfig, OandaConnector


class TestOandaConfig:
    """Test cases for OANDA configuration."""

    def test_config_initialization_practice(self):
        """Test configuration for practice environment."""
        config = OandaConfig(
            api_key="test-key",
            account_id="test-account",
            environment="practice",
            symbols=["EUR/USD", "GBP/USD"],
        )

        assert config.environment == "practice"
        assert config.base_url == "https://api-fxpractice.oanda.com"
        assert config.stream_url == "https://stream-fxpractice.oanda.com"
        assert config.account_id == "test-account"
        assert config.symbols == ["EUR/USD", "GBP/USD"]

    def test_config_initialization_live(self):
        """Test configuration for live environment."""
        config = OandaConfig(
            api_key="test-key",
            account_id="test-account",
            environment="live",
            symbols=["EUR/USD"],
        )

        assert config.environment == "live"
        assert config.base_url == "https://api-fxtrade.oanda.com"
        assert config.stream_url == "https://stream-fxtrade.oanda.com"


class TestOandaConnector:
    """Test cases for OANDA connector."""

    @pytest.fixture
    def oanda_config(self):
        """Create test OANDA configuration."""
        return OandaConfig(
            api_key="test-api-key",
            account_id="123-456-789",
            environment="practice",
            symbols=["EUR/USD", "GBP/USD"],
            timeout=10,
        )

    @pytest.fixture
    def oanda_connector(self, oanda_config):
        """Create OANDA connector for testing."""
        return OandaConnector(oanda_config)

    def test_connector_initialization(self, oanda_connector):
        """Test connector initializes correctly."""
        assert oanda_connector.status == ConnectionStatus.DISCONNECTED
        assert not oanda_connector.is_connected
        assert oanda_connector._instruments_map == {
            "EUR/USD": "EUR_USD",
            "GBP/USD": "GBP_USD",
        }

    def test_headers_property(self, oanda_connector):
        """Test authentication headers."""
        headers = oanda_connector.headers

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_successful_connection(self, oanda_connector):
        """Test successful connection to OANDA."""
        account_response = {
            "account": {
                "id": "123-456-789",
                "currency": "USD",
                "balance": "10000.0000",
                "instruments": [{"name": "EUR_USD"}, {"name": "GBP_USD"}],
            }
        }

        with aioresponses() as m:
            m.get(
                "https://api-fxpractice.oanda.com/v3/accounts/123-456-789",
                payload=account_response,
                status=200,
            )

            success = await oanda_connector.connect()

            assert success
            assert oanda_connector.status == ConnectionStatus.CONNECTED
            assert oanda_connector.is_connected
            assert oanda_connector.uptime is not None

    @pytest.mark.asyncio
    async def test_connection_failure(self, oanda_connector):
        """Test connection failure handling."""
        with aioresponses() as m:
            m.get(
                "https://api-fxpractice.oanda.com/v3/accounts/123-456-789",
                status=401,
                payload={"errorMessage": "Unauthorized"},
            )

            success = await oanda_connector.connect()

            assert not success
            assert oanda_connector.status == ConnectionStatus.ERROR
            assert not oanda_connector.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, oanda_connector):
        """Test graceful disconnection."""
        # First connect
        with aioresponses() as m:
            m.get(
                "https://api-fxpractice.oanda.com/v3/accounts/123-456-789",
                payload={"account": {"id": "123-456-789"}},
                status=200,
            )
            await oanda_connector.connect()

        # Then disconnect
        await oanda_connector.disconnect()

        assert oanda_connector.status == ConnectionStatus.DISCONNECTED
        assert not oanda_connector.is_connected
        assert oanda_connector.connection_start_time is None

    def test_parse_price_data(self, oanda_connector):
        """Test parsing OANDA price data."""
        price_data = {
            "type": "PRICE",
            "instrument": "EUR_USD",
            "time": "2024-01-15T12:30:45.123456789Z",
            "bids": [{"price": "1.08500", "liquidity": "1000000"}],
            "asks": [{"price": "1.08520", "liquidity": "1000000"}],
        }

        market_data = oanda_connector._parse_price_data(price_data)

        assert market_data is not None
        assert market_data.symbol == "EUR/USD"
        assert market_data.bid == 1.08500
        assert market_data.ask == 1.08520
        assert market_data.volume == 1000000.0
        assert market_data.source == "oanda"
        assert isinstance(market_data.timestamp, datetime)

    def test_parse_price_data_invalid(self, oanda_connector):
        """Test parsing invalid price data."""
        invalid_data = {
            "type": "PRICE",
            "instrument": "EUR_USD",
            # Missing required fields
        }

        market_data = oanda_connector._parse_price_data(invalid_data)

        assert market_data is None

    def test_parse_price_data_unsupported_symbol(self, oanda_connector):
        """Test parsing data for unsupported symbol."""
        price_data = {
            "type": "PRICE",
            "instrument": "USD_JPY",  # Not in configured symbols
            "time": "2024-01-15T12:30:45.123456789Z",
            "bids": [{"price": "149.500"}],
            "asks": [{"price": "149.520"}],
        }

        market_data = oanda_connector._parse_price_data(price_data)

        assert market_data is None

    @pytest.mark.asyncio
    async def test_get_historical_data(self, oanda_connector):
        """Test retrieving historical data."""
        candles_response = {
            "candles": [
                {
                    "time": "2024-01-15T12:00:00.000000000Z",
                    "mid": {
                        "o": "1.08400",
                        "h": "1.08450",
                        "l": "1.08380",
                        "c": "1.08420",
                    },
                    "volume": 1000,
                    "complete": True,
                },
                {
                    "time": "2024-01-15T12:01:00.000000000Z",
                    "mid": {
                        "o": "1.08420",
                        "h": "1.08470",
                        "l": "1.08400",
                        "c": "1.08460",
                    },
                    "volume": 1200,
                    "complete": True,
                },
            ]
        }

        # Mock connection
        with aioresponses() as m:
            # Mock account verification
            m.get(
                "https://api-fxpractice.oanda.com/v3/accounts/123-456-789",
                payload={"account": {"id": "123-456-789"}},
                status=200,
            )

            # Mock historical data request
            candles_url_pattern = re.compile(
                r"https://api-fxpractice\.oanda\.com/v3/instruments/EUR_USD/candles.*"
            )
            m.get(candles_url_pattern, payload=candles_response, status=200)

            await oanda_connector.connect()

            ohlcv_data = await oanda_connector.get_historical_data(
                symbol="EUR/USD", timeframe="1m", count=2
            )

            assert len(ohlcv_data) == 2

            first_candle = ohlcv_data[0]
            assert first_candle.symbol == "EUR/USD"
            assert first_candle.open == 1.08400
            assert first_candle.high == 1.08450
            assert first_candle.low == 1.08380
            assert first_candle.close == 1.08420
            assert first_candle.volume == 1000
            assert first_candle.timeframe == "1m"

    def test_convert_timeframe(self, oanda_connector):
        """Test timeframe conversion."""
        assert oanda_connector._convert_timeframe("1m") == "M1"
        assert oanda_connector._convert_timeframe("5m") == "M5"
        assert oanda_connector._convert_timeframe("1h") == "H1"
        assert oanda_connector._convert_timeframe("1d") == "D"
        assert (
            oanda_connector._convert_timeframe("M1") == "M1"
        )  # Already in OANDA format
        assert oanda_connector._convert_timeframe("invalid") == "M1"  # Default

    def test_parse_candle_data(self, oanda_connector):
        """Test parsing OANDA candle data."""
        candle_data = {
            "time": "2024-01-15T12:30:00.000000000Z",
            "mid": {"o": "1.08400", "h": "1.08450", "l": "1.08380", "c": "1.08420"},
            "volume": 1500,
        }

        ohlcv = oanda_connector._parse_candle_data(candle_data, "EUR/USD", "1m")

        assert ohlcv is not None
        assert ohlcv.symbol == "EUR/USD"
        assert ohlcv.open == 1.08400
        assert ohlcv.high == 1.08450
        assert ohlcv.low == 1.08380
        assert ohlcv.close == 1.08420
        assert ohlcv.volume == 1500
        assert ohlcv.timeframe == "1m"

    def test_get_supported_symbols(self, oanda_connector):
        """Test getting supported symbols."""
        symbols = oanda_connector.get_supported_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "EUR/USD" in symbols
        assert "GBP/USD" in symbols
        assert "USD/JPY" in symbols

        # All symbols should be in standard format
        for symbol in symbols:
            assert "/" in symbol
