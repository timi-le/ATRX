"""
Unit tests for Dukascopy connector.
"""

import asyncio
import struct
import lzma
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

from data.dukascopy_connector import DukascopyConnector, DukascopyConfig
from data.base_connector import ConnectionStatus
from core.interfaces.data_interfaces import MarketData, OHLCV


class TestDukascopyConfig:
    """Test cases for Dukascopy configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = DukascopyConfig(
            symbols=["EUR/USD", "GBP/USD"],
            timeout=30
        )
        
        assert config.base_url == "https://datafeed.dukascopy.com/datafeed"
        assert config.tick_base_url == "https://www.dukascopy.com/datafeed"
        assert config.symbols == ["EUR/USD", "GBP/USD"]
        assert config.max_concurrent_downloads == 5
        assert "Mozilla" in config.user_agent
    
    def test_config_custom_settings(self):
        """Test configuration with custom settings."""
        config = DukascopyConfig(
            symbols=["EUR/USD"],
            max_concurrent_downloads=10,
            user_agent="Custom Agent"
        )
        
        assert config.max_concurrent_downloads == 10
        assert config.user_agent == "Custom Agent"


class TestDukascopyConnector:
    """Test cases for Dukascopy connector."""
    
    @pytest.fixture
    def dukascopy_config(self):
        """Create test Dukascopy configuration."""
        return DukascopyConfig(
            symbols=["EUR/USD", "GBP/USD"],
            timeout=10
        )
    
    @pytest.fixture
    def dukascopy_connector(self, dukascopy_config):
        """Create Dukascopy connector for testing."""
        return DukascopyConnector(dukascopy_config)
    
    def test_connector_initialization(self, dukascopy_connector):
        """Test connector initializes correctly."""
        assert dukascopy_connector.status == ConnectionStatus.DISCONNECTED
        assert not dukascopy_connector.is_connected
        assert dukascopy_connector._symbols_map == {
            "EUR/USD": "EURUSD",
            "GBP/USD": "GBPUSD"
        }
    
    def test_headers_property(self, dukascopy_connector):
        """Test HTTP headers."""
        headers = dukascopy_connector.headers
        
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Connection" in headers
        assert headers["Connection"] == "keep-alive"
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, dukascopy_connector):
        """Test successful connection to Dukascopy."""
        with aioresponses() as m:
            m.get(
                "https://datafeed.dukascopy.com/datafeed/EURUSD/metadata/AvailableDays",
                status=200,
                payload=["2024-01-15", "2024-01-16"]
            )
            
            success = await dukascopy_connector.connect()
            
            assert success
            assert dukascopy_connector.status == ConnectionStatus.CONNECTED
            assert dukascopy_connector.is_connected
            assert dukascopy_connector.uptime is not None
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, dukascopy_connector):
        """Test connection failure handling."""
        with aioresponses() as m:
            m.get(
                "https://datafeed.dukascopy.com/datafeed/EURUSD/metadata/AvailableDays",
                status=500
            )
            
            success = await dukascopy_connector.connect()
            
            assert not success
            assert dukascopy_connector.status == ConnectionStatus.ERROR
            assert not dukascopy_connector.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, dukascopy_connector):
        """Test graceful disconnection."""
        # First connect
        with aioresponses() as m:
            m.get(
                "https://datafeed.dukascopy.com/datafeed/EURUSD/metadata/AvailableDays",
                status=200,
                payload=["2024-01-15"]
            )
            await dukascopy_connector.connect()
        
        # Then disconnect
        await dukascopy_connector.disconnect()
        
        assert dukascopy_connector.status == ConnectionStatus.DISCONNECTED
        assert not dukascopy_connector.is_connected
        assert dukascopy_connector.connection_start_time is None
    
    def test_get_point_value(self, dukascopy_connector):
        """Test point value calculation for different symbols."""
        # Regular FX pairs (5 decimal places)
        assert dukascopy_connector._get_point_value("EURUSD") == 100000.0
        assert dukascopy_connector._get_point_value("GBPUSD") == 100000.0
        
        # JPY pairs (3 decimal places)
        assert dukascopy_connector._get_point_value("USDJPY") == 1000.0
        assert dukascopy_connector._get_point_value("EURJPY") == 1000.0
    
    def test_parse_tick_data(self, dukascopy_connector):
        """Test parsing Dukascopy binary tick data."""
        # Create sample binary tick data
        # Format: timestamp(4), ask(4), bid(4), ask_volume(4), bid_volume(4)
        timestamp_ms = 30000  # 30 seconds into the hour
        ask = 108520  # 1.08520 * 100000
        bid = 108500  # 1.08500 * 100000  
        ask_vol = 1000000
        bid_vol = 1500000
        
        tick_data = struct.pack('>IIIII', timestamp_ms, ask, bid, ask_vol, bid_vol)
        
        hour_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        ticks = dukascopy_connector._parse_tick_data(tick_data, "EURUSD", hour_time)
        
        assert len(ticks) == 1
        
        tick = ticks[0]
        assert tick.symbol == "EUR/USD"
        assert tick.bid == 1.08500
        assert tick.ask == 1.08520
        assert tick.volume == 2500000.0  # Combined volumes
        assert tick.source == "dukascopy"
        
        # Check timestamp calculation
        expected_timestamp = hour_time + timedelta(milliseconds=30000)
        assert tick.timestamp == expected_timestamp
    
    def test_parse_tick_data_jpy_pair(self, dukascopy_connector):
        """Test parsing tick data for JPY pairs."""
        # JPY pair with 3 decimal places
        timestamp_ms = 15000
        ask = 149520  # 149.520 * 1000
        bid = 149500  # 149.500 * 1000
        ask_vol = 500000
        bid_vol = 750000
        
        tick_data = struct.pack('>IIIII', timestamp_ms, ask, bid, ask_vol, bid_vol)
        
        hour_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        ticks = dukascopy_connector._parse_tick_data(tick_data, "USDJPY", hour_time)
        
        assert len(ticks) == 1
        
        tick = ticks[0]
        assert tick.symbol == "USD/JPY"
        assert tick.bid == 149.500
        assert tick.ask == 149.520
        assert tick.volume == 1250000.0
    
    def test_parse_tick_data_multiple_ticks(self, dukascopy_connector):
        """Test parsing multiple ticks in one data block."""
        # Create data for 3 ticks
        ticks_data = []
        
        for i in range(3):
            timestamp_ms = (i + 1) * 10000  # 10, 20, 30 seconds
            ask = 108520 + i  # Incrementing prices
            bid = 108500 + i
            ask_vol = 1000000
            bid_vol = 1000000
            
            tick_data = struct.pack('>IIIII', timestamp_ms, ask, bid, ask_vol, bid_vol)
            ticks_data.append(tick_data)
        
        combined_data = b''.join(ticks_data)
        hour_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        ticks = dukascopy_connector._parse_tick_data(combined_data, "EURUSD", hour_time)
        
        assert len(ticks) == 3
        
        # Check first tick
        assert ticks[0].timestamp == hour_time + timedelta(milliseconds=10000)
        assert ticks[0].ask == 1.08520
        
        # Check last tick  
        assert ticks[2].timestamp == hour_time + timedelta(milliseconds=30000)
        assert ticks[2].ask == 1.08522
    
    @pytest.mark.asyncio
    async def test_download_hour_ticks_success(self, dukascopy_connector):
        """Test successful download of hour tick data."""
        # Create sample tick data
        timestamp_ms = 30000
        ask = 108520
        bid = 108500
        ask_vol = 1000000
        bid_vol = 1000000
        
        tick_data = struct.pack('>IIIII', timestamp_ms, ask, bid, ask_vol, bid_vol)
        compressed_data = lzma.compress(tick_data)
        
        hour_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        with aioresponses() as m:
            # Mock connection first
            m.get(
                "https://datafeed.dukascopy.com/datafeed/EURUSD/metadata/AvailableDays",
                status=200,
                payload=["2024-01-15"]
            )
            
            # Mock tick data download
            m.get(
                "https://www.dukascopy.com/datafeed/EURUSD/2024/00/15/12h_ticks.bi5",
                body=compressed_data,
                status=200
            )
            
            await dukascopy_connector.connect()
            
            ticks = await dukascopy_connector._download_hour_ticks("EURUSD", hour_time)
            
            assert len(ticks) == 1
            assert ticks[0].symbol == "EUR/USD"
    
    @pytest.mark.asyncio
    async def test_download_hour_ticks_no_data(self, dukascopy_connector):
        """Test download when no data is available (404)."""
        hour_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        with aioresponses() as m:
            # Mock connection
            m.get(
                "https://datafeed.dukascopy.com/datafeed/EURUSD/metadata/AvailableDays",
                status=200,
                payload=["2024-01-15"]
            )
            
            # Mock 404 for tick data
            m.get(
                "https://www.dukascopy.com/datafeed/EURUSD/2024/00/15/12h_ticks.bi5",
                status=404
            )
            
            await dukascopy_connector.connect()
            
            ticks = await dukascopy_connector._download_hour_ticks("EURUSD", hour_time)
            
            assert len(ticks) == 0
    
    def test_parse_timeframe_minutes(self, dukascopy_connector):
        """Test timeframe parsing to minutes."""
        assert dukascopy_connector._parse_timeframe_minutes("1m") == 1
        assert dukascopy_connector._parse_timeframe_minutes("5m") == 5
        assert dukascopy_connector._parse_timeframe_minutes("1h") == 60
        assert dukascopy_connector._parse_timeframe_minutes("4h") == 240
        assert dukascopy_connector._parse_timeframe_minutes("1d") == 1440
    
    def test_get_candle_start_time(self, dukascopy_connector):
        """Test candle start time calculation."""
        timestamp = datetime(2024, 1, 15, 12, 37, 45, tzinfo=timezone.utc)
        
        # 1-minute candles
        start_1m = dukascopy_connector._get_candle_start_time(timestamp, 1)
        assert start_1m == datetime(2024, 1, 15, 12, 37, 0, tzinfo=timezone.utc)
        
        # 5-minute candles
        start_5m = dukascopy_connector._get_candle_start_time(timestamp, 5)
        assert start_5m == datetime(2024, 1, 15, 12, 35, 0, tzinfo=timezone.utc)
        
        # 1-hour candles
        start_1h = dukascopy_connector._get_candle_start_time(timestamp, 60)
        assert start_1h == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Daily candles
        start_1d = dukascopy_connector._get_candle_start_time(timestamp, 1440)
        assert start_1d == datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    
    def test_create_ohlcv_candle(self, dukascopy_connector):
        """Test OHLCV candle creation from ticks."""
        base_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        
        # Create sample ticks
        ticks = []
        prices = [1.08500, 1.08520, 1.08480, 1.08510]  # O, H, L, C pattern
        
        for i, price in enumerate(prices):
            tick = MarketData(
                symbol="EUR/USD",
                timestamp=base_time + timedelta(seconds=i * 15),
                bid=price - 0.00010,
                ask=price + 0.00010,
                volume=1000000,
                source="dukascopy"
            )
            ticks.append(tick)
        
        candle = dukascopy_connector._create_ohlcv_candle(ticks, "EUR/USD", "1m")
        
        assert candle is not None
        assert candle.symbol == "EUR/USD"
        assert candle.timestamp == base_time
        assert candle.open == 1.08500
        assert candle.high == 1.08520
        assert candle.low == 1.08480
        assert candle.close == 1.08510
        assert candle.volume == 4000000  # Sum of all volumes
        assert candle.timeframe == "1m"
    
    def test_create_ohlcv_candle_empty_ticks(self, dukascopy_connector):
        """Test OHLCV candle creation with empty tick list."""
        candle = dukascopy_connector._create_ohlcv_candle([], "EUR/USD", "1m")
        assert candle is None
    
    def test_convert_ticks_to_ohlcv(self, dukascopy_connector):
        """Test converting tick data to OHLCV candles."""
        base_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        
        # Create ticks spanning multiple 1-minute candles
        ticks = []
        
        # First minute: 12:30:00 - 12:30:59
        for i in range(3):
            tick = MarketData(
                symbol="EUR/USD",
                timestamp=base_time + timedelta(seconds=i * 20),
                bid=1.08500 + i * 0.00005,
                ask=1.08520 + i * 0.00005,
                volume=1000000,
                source="dukascopy"
            )
            ticks.append(tick)
        
        # Second minute: 12:31:00 - 12:31:59
        for i in range(2):
            tick = MarketData(
                symbol="EUR/USD",
                timestamp=base_time + timedelta(minutes=1, seconds=i * 30),
                bid=1.08510 + i * 0.00010,
                ask=1.08530 + i * 0.00010,
                volume=1500000,
                source="dukascopy"
            )
            ticks.append(tick)
        
        candles = dukascopy_connector._convert_ticks_to_ohlcv(ticks, "EUR/USD", "1m")
        
        assert len(candles) == 2
        
        # Check first candle
        first_candle = candles[0]
        assert first_candle.timestamp == base_time
        assert first_candle.open == 1.08510  # Mid price of first tick
        
        # Check second candle
        second_candle = candles[1]
        assert second_candle.timestamp == base_time + timedelta(minutes=1)
        assert second_candle.volume == 3000000  # Sum of 2 ticks
    
    def test_get_supported_symbols(self, dukascopy_connector):
        """Test getting supported symbols."""
        symbols = dukascopy_connector.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "EUR/USD" in symbols
        assert "GBP/USD" in symbols
        assert "USD/JPY" in symbols
        assert "USD/SEK" in symbols  # Additional pairs
        
        # All symbols should be in standard format
        for symbol in symbols:
            assert "/" in symbol 