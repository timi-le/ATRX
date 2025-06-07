"""
Mock FX Data Provider

Generates realistic market data for development and testing purposes.
Simulates real-world FX price movements with configurable parameters.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Optional

import numpy as np
import structlog
from data.base_connector import BaseFXConnector, ConnectionConfig, ConnectionStatus
from core.interfaces.data_interfaces import MarketData, OHLCV


class MockMarketDataGenerator:
    """
    Generates realistic mock market data using various price models.
    """
    
    def __init__(self, symbol: str = "EUR/USD", initial_price: float = 1.0850):
        self.symbol = symbol
        self.current_price = initial_price
        self.initial_price = initial_price
        
        # Market simulation parameters
        self.volatility = 0.0001  # Price volatility per tick
        self.trend_strength = 0.00001  # Trend component
        self.spread = 0.0002  # Bid-ask spread
        self.tick_interval = 0.1  # Seconds between ticks
        
        # Price bounds to keep realistic
        self.price_bounds = (initial_price * 0.95, initial_price * 1.05)
        
        # Market hours simulation
        self.market_hours = True
        
    def generate_price_movement(self) -> float:
        """Generate the next price movement using random walk with drift."""
        # Random component (Gaussian noise)
        random_component = np.random.normal(0, self.volatility)
        
        # Trend component (slight bias)
        trend_component = np.random.normal(0, self.trend_strength)
        
        # Mean reversion (pull back towards initial price)
        reversion_strength = 0.00001
        reversion_component = -(self.current_price - self.initial_price) * reversion_strength
        
        # Combine components
        price_change = random_component + trend_component + reversion_component
        
        new_price = self.current_price + price_change
        
        # Keep within realistic bounds
        new_price = max(self.price_bounds[0], min(self.price_bounds[1], new_price))
        
        self.current_price = new_price
        return new_price
    
    def generate_market_data(self) -> MarketData:
        """Generate a single MarketData tick."""
        # Generate new price
        mid_price = self.generate_price_movement()
        
        # Calculate bid/ask from mid price
        half_spread = self.spread / 2
        bid = round(mid_price - half_spread, 5)
        ask = round(mid_price + half_spread, 5)
        
        # Generate realistic volume (higher during market hours)
        base_volume = 100000
        volume_multiplier = 2.0 if self.market_hours else 0.3
        volume = int(base_volume * volume_multiplier * (0.5 + random.random()))
        
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.now(),
            bid=bid,
            ask=ask,
            volume=volume,
            source="mock_provider"
        )
    
    def generate_ohlcv_bar(self, timeframe_minutes: int = 1) -> OHLCV:
        """Generate an OHLCV bar by aggregating ticks."""
        start_time = datetime.now().replace(second=0, microsecond=0)
        start_price = self.current_price
        
        # Generate multiple ticks for the bar
        num_ticks = max(10, timeframe_minutes * 10)  # More ticks for longer timeframes
        prices = []
        volumes = []
        
        for _ in range(num_ticks):
            tick = self.generate_market_data()
            prices.append(tick.mid)
            volumes.append(tick.volume)
        
        # Calculate OHLCV values
        open_price = start_price
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]
        volume = sum(volumes)
        
        return OHLCV(
            symbol=self.symbol,
            timestamp=start_time,
            open=round(open_price, 5),
            high=round(high_price, 5),
            low=round(low_price, 5),
            close=round(close_price, 5),
            volume=volume,
            timeframe=f"{timeframe_minutes}m"
        )


class MockFXConnector(BaseFXConnector):
    """
    Mock FX connector for development and testing.
    
    Generates realistic market data without requiring external API connections.
    """
    
    def __init__(self, config: ConnectionConfig, logger: Optional[structlog.stdlib.BoundLogger] = None):
        super().__init__(config, logger)
        self.generators = {}
        self._streaming_task = None
        self._stop_streaming = False
        
        # Initialize generators for each symbol
        for symbol in config.symbols:
            initial_prices = {
                "EUR/USD": 1.0850,
                "GBP/USD": 1.2650,
                "USD/JPY": 149.50,
                "AUD/USD": 0.6750,
                "USD/CHF": 0.8950,
                "NZD/USD": 0.6150,
                "USD/CAD": 1.3450,
            }
            initial_price = initial_prices.get(symbol, 1.0000)
            self.generators[symbol] = MockMarketDataGenerator(symbol, initial_price)
    
    async def connect(self) -> bool:
        """Simulate connection to mock data provider."""
        self.status = ConnectionStatus.CONNECTING
        self.logger.info("Connecting to mock FX provider", symbols=self.config.symbols)
        
        try:
            # Simulate connection delay
            await asyncio.sleep(0.5)
            
            self.status = ConnectionStatus.CONNECTED
            self.connection_start_time = time.time()
            self.logger.info("Connected to mock FX provider", 
                           symbols=list(self.generators.keys()))
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error("Failed to connect to mock provider", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from mock provider."""
        self._stop_streaming = True
        
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
        
        self.status = ConnectionStatus.DISCONNECTED
        self.connection_start_time = None
        self.logger.info("Disconnected from mock FX provider")
    
    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """
        Stream mock market data for all configured symbols.
        
        Yields:
            MarketData: Generated market data ticks
        """
        if not self.is_connected:
            raise RuntimeError("Connector not connected")
        
        self.logger.info("Starting mock data stream", symbols=list(self.generators.keys()))
        self._stop_streaming = False
        
        symbol_cycle = 0
        symbols = list(self.generators.keys())
        
        while self.is_connected and not self._stop_streaming:
            try:
                # Rotate through symbols to simulate realistic interleaving
                symbol = symbols[symbol_cycle % len(symbols)]
                generator = self.generators[symbol]
                
                # Generate market data
                market_data = generator.generate_market_data()
                
                yield market_data
                
                # Move to next symbol
                symbol_cycle += 1
                
                # Variable delay to simulate realistic timing
                base_delay = 0.05  # 50ms base
                jitter = random.uniform(0.01, 0.1)  # 10-100ms jitter
                await asyncio.sleep(base_delay + jitter)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in mock data stream", error=str(e))
                await asyncio.sleep(0.1)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        count: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Generate mock historical OHLCV data.
        
        Args:
            symbol: Trading pair symbol (e.g., "EUR/USD")
            timeframe: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time (ISO format)
            end_time: End time (ISO format)  
            count: Number of bars to retrieve
            
        Returns:
            List of OHLCV data
        """
        if symbol not in self.generators:
            raise ValueError(f"Symbol {symbol} not configured")
        
        generator = self.generators[symbol]
        
        # Parse timeframe
        timeframe_minutes = self._parse_timeframe(timeframe)
        
        # Determine number of bars to generate
        if count:
            num_bars = count
        else:
            # Default to 100 bars
            num_bars = 100
        
        self.logger.info("Generating historical data", 
                        symbol=symbol, 
                        timeframe=timeframe,
                        bars=num_bars)
        
        bars = []
        current_time = datetime.now()
        
        for i in range(num_bars):
            # Calculate timestamp for this bar (going backwards in time)
            bar_time = current_time - timedelta(minutes=timeframe_minutes * (num_bars - i - 1))
            
            # Generate OHLCV bar
            bar = generator.generate_ohlcv_bar(timeframe_minutes)
            bar.timestamp = bar_time
            
            bars.append(bar)
        
        return bars
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        timeframe = timeframe.lower()
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    async def _send_heartbeat(self) -> None:
        """Mock heartbeat - always succeeds."""
        pass
    
    def set_market_conditions(self, symbol: str, **kwargs) -> None:
        """
        Adjust market conditions for testing different scenarios.
        
        Args:
            symbol: Symbol to adjust
            **kwargs: Market parameters (volatility, trend_strength, spread, etc.)
        """
        if symbol in self.generators:
            generator = self.generators[symbol]
            for param, value in kwargs.items():
                if hasattr(generator, param):
                    setattr(generator, param, value)
                    self.logger.info("Updated market condition", 
                                   symbol=symbol, 
                                   parameter=param, 
                                   value=value)
    
    def simulate_market_event(self, symbol: str, event_type: str, magnitude: float = 0.001) -> None:
        """
        Simulate market events like news releases or economic data.
        
        Args:
            symbol: Symbol to affect
            event_type: Type of event ("spike", "drop", "volatility_increase")
            magnitude: Size of the impact
        """
        if symbol not in self.generators:
            return
        
        generator = self.generators[symbol]
        
        if event_type == "spike":
            generator.current_price += magnitude
        elif event_type == "drop":
            generator.current_price -= magnitude
        elif event_type == "volatility_increase":
            generator.volatility *= (1 + magnitude * 10)
        
        self.logger.info("Simulated market event", 
                        symbol=symbol, 
                        event=event_type, 
                        magnitude=magnitude) 