"""
Dukascopy FX Connector

Production-ready connector for Dukascopy historical data API providing 
historical FX tick and candle data retrieval capabilities.
"""

import asyncio
import struct
import time
import lzma
import json
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import calendar

import aiohttp
import structlog
from data.base_connector import BaseFXConnector, ConnectionConfig, ConnectionStatus, retry_on_connection_error
from core.interfaces.data_interfaces import MarketData, OHLCV


@dataclass
class DukascopyConfig(ConnectionConfig):
    """Extended configuration for Dukascopy connector."""
    base_url: str = "https://datafeed.dukascopy.com/datafeed"
    tick_base_url: str = "https://www.dukascopy.com/datafeed"
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    max_concurrent_downloads: int = 5
    chunk_size: int = 8192
    
    def __post_init__(self):
        super().__post_init__()


class DukascopyConnector(BaseFXConnector):
    """
    Dukascopy historical data connector.
    
    Provides historical tick and candle data retrieval from Dukascopy's
    free historical data feed using HTTP requests.
    """
    
    def __init__(self, config: DukascopyConfig, logger: Optional[structlog.stdlib.BoundLogger] = None):
        super().__init__(config, logger)
        self.dukascopy_config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._symbols_map = {}
        
        # Dukascopy uses specific symbol format
        self._setup_symbol_mapping()
        
        # Available timeframes
        self.supported_timeframes = {
            "tick": "tick",
            "1m": "m1", 
            "5m": "m5",
            "15m": "m15", 
            "30m": "m30",
            "1h": "h1",
            "4h": "h4", 
            "1d": "d1"
        }
    
    def _setup_symbol_mapping(self):
        """Set up mapping between standard FX notation and Dukascopy format."""
        for symbol in self.config.symbols:
            # Dukascopy format: EURUSD, GBPUSD, etc.
            dukascopy_symbol = symbol.replace("/", "")
            self._symbols_map[symbol] = dukascopy_symbol
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for Dukascopy HTTP requests."""
        return {
            "User-Agent": self.dukascopy_config.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
    
    async def connect(self) -> bool:
        """Establish connection to Dukascopy servers."""
        self.status = ConnectionStatus.CONNECTING
        self.logger.info("Connecting to Dukascopy historical data feed")
        
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(limit=self.dukascopy_config.max_concurrent_downloads)
            
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout,
                connector=connector
            )
            
            # Verify connection by testing a simple request
            await self._verify_connection()
            
            self.status = ConnectionStatus.CONNECTED
            self.connection_start_time = time.time()
            self.logger.info("Connected to Dukascopy successfully",
                           symbols=list(self._symbols_map.values()))
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error("Failed to connect to Dukascopy", error=str(e))
            if self.session:
                await self.session.close()
                self.session = None
            return False
    
    async def _verify_connection(self) -> None:
        """Verify connection by making a test request."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        # Test with a simple metadata request
        test_url = f"{self.dukascopy_config.base_url}/EURUSD/metadata/AvailableDays"
        
        async with self.session.get(test_url) as response:
            if response.status != 200:
                raise RuntimeError(f"Dukascopy connection test failed: {response.status}")
            
            self.logger.info("Dukascopy connection verified")
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from Dukascopy."""
        self.logger.info("Disconnecting from Dukascopy")
        
        if self.session:
            await self.session.close()
            self.session = None
        
        self.status = ConnectionStatus.DISCONNECTED
        self.connection_start_time = None
        self.logger.info("Disconnected from Dukascopy")
    
    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """
        Dukascopy doesn't provide real-time streaming, only historical data.
        This method simulates streaming by fetching recent data periodically.
        """
        if not self.is_connected:
            raise RuntimeError("Connector not connected")
        
        self.logger.info("Starting Dukascopy pseudo-stream (historical data simulation)")
        
        # Simulate streaming by fetching recent tick data
        last_fetch_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        while self.is_connected and not self._should_stop:
            try:
                current_time = datetime.now(timezone.utc)
                
                for symbol in self.config.symbols:
                    # Fetch recent tick data for each symbol
                    ticks = await self._get_tick_data(
                        symbol, 
                        last_fetch_time, 
                        current_time,
                        max_ticks=100
                    )
                    
                    for tick in ticks:
                        yield tick
                
                last_fetch_time = current_time
                
                # Wait before next fetch (simulate real-time frequency)
                await asyncio.sleep(5)  # 5 second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in Dukascopy pseudo-stream", error=str(e))
                await asyncio.sleep(1)
    
    @retry_on_connection_error(max_retries=3, delay=2.0)
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        count: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Retrieve historical OHLCV data from Dukascopy.
        
        Args:
            symbol: Trading pair symbol (e.g., "EUR/USD")
            timeframe: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            count: Number of bars to retrieve
            
        Returns:
            List of OHLCV data
        """
        if not self.is_connected or not self.session:
            raise RuntimeError("Connector not connected")
        
        # Convert symbol to Dukascopy format
        dukascopy_symbol = self._symbols_map.get(symbol)
        if not dukascopy_symbol:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        # Parse time range
        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        elif count:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(hours=count)  # Simple approximation
        else:
            # Default: last 24 hours
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(hours=24)
        
        self.logger.info("Fetching Dukascopy historical data",
                        symbol=symbol,
                        dukascopy_symbol=dukascopy_symbol,
                        timeframe=timeframe,
                        start_time=start_dt.isoformat(),
                        end_time=end_dt.isoformat())
        
        if timeframe == "tick":
            # Get tick data and convert to OHLCV
            ticks = await self._get_tick_data(symbol, start_dt, end_dt)
            return self._convert_ticks_to_ohlcv(ticks, symbol, "1m")  # Convert to 1-minute bars
        else:
            # Get candle data directly
            return await self._get_candle_data(symbol, timeframe, start_dt, end_dt)
    
    async def _get_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        max_ticks: Optional[int] = None
    ) -> List[MarketData]:
        """Retrieve tick data from Dukascopy for the specified time range."""
        dukascopy_symbol = self._symbols_map[symbol]
        ticks = []
        
        # Dukascopy tick data is organized by hour
        current_time = start_time.replace(minute=0, second=0, microsecond=0)
        
        while current_time < end_time:
            try:
                hour_ticks = await self._download_hour_ticks(dukascopy_symbol, current_time)
                
                # Filter ticks within the requested range
                for tick in hour_ticks:
                    if start_time <= tick.timestamp <= end_time:
                        ticks.append(tick)
                        
                        if max_ticks and len(ticks) >= max_ticks:
                            return ticks[:max_ticks]
                
                current_time += timedelta(hours=1)
                
            except Exception as e:
                self.logger.warning("Failed to download hour ticks", 
                                  symbol=symbol,
                                  hour=current_time.isoformat(),
                                  error=str(e))
                current_time += timedelta(hours=1)
                continue
        
        return ticks
    
    async def _download_hour_ticks(self, symbol: str, hour_time: datetime) -> List[MarketData]:
        """Download and parse tick data for a specific hour."""
        # Dukascopy URL format: /symbol/year/month-1/day/hour.bi5
        year = hour_time.year
        month = hour_time.month - 1  # Dukascopy uses 0-based months
        day = hour_time.day
        hour = hour_time.hour
        
        url = f"{self.dukascopy_config.tick_base_url}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 404:
                    # No data for this hour (common for weekends/holidays)
                    return []
                
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
                
                # Download compressed data
                compressed_data = await response.read()
                
                # Decompress using LZMA
                decompressed_data = lzma.decompress(compressed_data)
                
                # Parse binary tick data
                return self._parse_tick_data(decompressed_data, symbol, hour_time)
                
        except Exception as e:
            self.logger.debug("No tick data available", 
                            symbol=symbol, 
                            hour=hour_time.isoformat(),
                            error=str(e))
            return []
    
    def _parse_tick_data(self, data: bytes, symbol: str, hour_time: datetime) -> List[MarketData]:
        """Parse Dukascopy binary tick data format."""
        ticks = []
        
        # Each tick is 20 bytes: timestamp(4), ask(4), bid(4), ask_volume(4), bid_volume(4)
        tick_size = 20
        num_ticks = len(data) // tick_size
        
        # Get point value for the symbol (most FX pairs use 5 decimal places)
        point_value = self._get_point_value(symbol)
        
        for i in range(num_ticks):
            offset = i * tick_size
            
            # Unpack tick data (big-endian format)
            timestamp_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIII', data[offset:offset + tick_size])
            
            # Convert to actual prices (Dukascopy stores as integers)
            ask_price = ask / point_value
            bid_price = bid / point_value
            
            # Calculate timestamp
            tick_timestamp = hour_time + timedelta(milliseconds=timestamp_ms)
            
            # Create MarketData object
            tick = MarketData(
                symbol=self._convert_symbol_to_standard(symbol),
                timestamp=tick_timestamp,
                bid=bid_price,
                ask=ask_price,
                volume=float(bid_vol + ask_vol),  # Combine volumes
                source="dukascopy"
            )
            
            ticks.append(tick)
        
        return ticks
    
    def _get_point_value(self, symbol: str) -> float:
        """Get the point value for price conversion."""
        # Most FX pairs use 5 decimal places (point = 0.00001)
        # JPY pairs use 3 decimal places (point = 0.001)
        if "JPY" in symbol:
            return 1000.0
        else:
            return 100000.0
    
    def _convert_symbol_to_standard(self, dukascopy_symbol: str) -> str:
        """Convert Dukascopy symbol format (EURUSD) to standard format (EUR/USD)."""
        # Find the reverse mapping
        for standard_symbol, duka_symbol in self._symbols_map.items():
            if duka_symbol == dukascopy_symbol:
                return standard_symbol
        
        # Fallback: insert slash in the middle for 6-character pairs
        if len(dukascopy_symbol) == 6:
            return f"{dukascopy_symbol[:3]}/{dukascopy_symbol[3:]}"
        
        # Return as-is if no conversion found
        return dukascopy_symbol
    
    async def _get_candle_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[OHLCV]:
        """Get candle data by aggregating tick data."""
        # For now, we'll get tick data and aggregate it into candles
        # This is a simplified approach - in production, you might want to
        # implement direct candle data retrieval if Dukascopy provides it
        
        ticks = await self._get_tick_data(symbol, start_time, end_time)
        return self._convert_ticks_to_ohlcv(ticks, symbol, timeframe)
    
    def _convert_ticks_to_ohlcv(
        self, 
        ticks: List[MarketData], 
        symbol: str, 
        timeframe: str
    ) -> List[OHLCV]:
        """Convert tick data to OHLCV candles."""
        if not ticks:
            return []
        
        # Determine timeframe in minutes
        timeframe_minutes = self._parse_timeframe_minutes(timeframe)
        
        candles = []
        current_candle_data = []
        current_candle_start = None
        
        for tick in sorted(ticks, key=lambda t: t.timestamp):
            # Determine which candle this tick belongs to
            candle_start = self._get_candle_start_time(tick.timestamp, timeframe_minutes)
            
            if current_candle_start != candle_start:
                # Finalize previous candle
                if current_candle_data:
                    candle = self._create_ohlcv_candle(current_candle_data, symbol, timeframe)
                    if candle:
                        candles.append(candle)
                
                # Start new candle
                current_candle_start = candle_start
                current_candle_data = []
            
            current_candle_data.append(tick)
        
        # Finalize last candle
        if current_candle_data:
            candle = self._create_ohlcv_candle(current_candle_data, symbol, timeframe)
            if candle:
                candles.append(candle)
        
        return candles
    
    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return timeframe_map.get(timeframe.lower(), 1)
    
    def _get_candle_start_time(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Get the start time of the candle that contains the given timestamp."""
        if timeframe_minutes >= 1440:  # Daily or longer
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe_minutes >= 60:  # Hourly
            hours_interval = timeframe_minutes // 60
            hour = (timestamp.hour // hours_interval) * hours_interval
            return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:  # Minutes
            minute = (timestamp.minute // timeframe_minutes) * timeframe_minutes
            return timestamp.replace(minute=minute, second=0, microsecond=0)
    
    def _create_ohlcv_candle(
        self, 
        ticks: List[MarketData], 
        symbol: str, 
        timeframe: str
    ) -> Optional[OHLCV]:
        """Create an OHLCV candle from a list of ticks."""
        if not ticks:
            return None
        
        # Sort ticks by timestamp
        sorted_ticks = sorted(ticks, key=lambda t: t.timestamp)
        
        # Use mid prices for OHLCV
        prices = [tick.mid for tick in sorted_ticks]
        volumes = [tick.volume or 0 for tick in sorted_ticks]
        
        return OHLCV(
            symbol=symbol,
            timestamp=sorted_ticks[0].timestamp,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            timeframe=timeframe
        )
    
    async def _send_heartbeat(self) -> None:
        """Dukascopy doesn't require heartbeats for HTTP requests."""
        pass
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols supported by Dukascopy."""
        return [
            # Major FX pairs
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD", "USD/CAD",
            # Cross pairs  
            "EUR/GBP", "EUR/JPY", "GBP/JPY", "EUR/CHF", "GBP/CHF", "AUD/JPY", "CHF/JPY",
            "EUR/AUD", "GBP/AUD", "AUD/CHF", "AUD/CAD", "EUR/CAD", "GBP/CAD", "CAD/CHF",
            "NZD/JPY", "EUR/NZD", "GBP/NZD", "AUD/NZD", "CAD/JPY", "NZD/CHF", "NZD/CAD",
            # Additional pairs
            "USD/SGD", "USD/HKD", "USD/SEK", "USD/NOK", "USD/DKK", "USD/PLN", "USD/CZK",
            "EUR/SEK", "EUR/NOK", "EUR/DKK", "EUR/PLN", "EUR/CZK", "GBP/SEK", "GBP/NOK"
        ] 