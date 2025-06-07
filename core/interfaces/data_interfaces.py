"""
Data interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, AsyncIterator, Any
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    volume: Optional[float] = None
    source: Optional[str] = None
    mid: float = field(init=False)
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.mid = (self.bid + self.ask) / 2.0
        self.spread = self.ask - self.bid


@dataclass
class OHLCV:
    """OHLCV bar data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    async def subscribe_ticks(self, symbols: List[str]) -> AsyncIterator[MarketData]:
        """Subscribe to real-time tick data."""
        pass

    @abstractmethod
    async def subscribe_bars(
        self, symbols: List[str], timeframe: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        """Subscribe to real-time bar data."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Retrieve historical market data."""
        pass


class DataPublisher(ABC):
    """Abstract base class for data publishers."""

    @abstractmethod
    async def publish_tick(self, topic: str, data: MarketData) -> None:
        """Publish tick data to a topic."""
        pass

    @abstractmethod
    async def publish_bar(self, topic: str, data: OHLCV) -> None:
        """Publish bar data to a topic."""
        pass

    @abstractmethod
    async def publish_features(self, topic: str, features: Dict[str, Any]) -> None:
        """Publish computed features."""
        pass


class DataSubscriber(ABC):
    """Abstract base class for data subscribers."""

    @abstractmethod
    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        """Subscribe to a data topic."""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a data topic."""
        pass


class MarketDataFeed(ABC):
    """Abstract base class for market data feeds."""

    @abstractmethod
    async def start_feed(self, symbols: List[str]) -> None:
        """Start the market data feed."""
        pass

    @abstractmethod
    async def stop_feed(self) -> None:
        """Stop the market data feed."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if feed is connected."""
        pass


class HistoricalDataProvider(ABC):
    """Abstract base class for historical data providers."""

    @abstractmethod
    async def get_fx_data(
        self, pair: str, start_date: datetime, end_date: datetime, timeframe: str = "1m"
    ) -> pd.DataFrame:
        """Get historical FX data."""
        pass

    @abstractmethod
    async def get_economic_data(
        self, indicator: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get historical economic indicator data."""
        pass

    @abstractmethod
    async def cache_data(
        self, key: str, data: pd.DataFrame, ttl: Optional[int] = None
    ) -> None:
        """Cache data for future use."""
        pass
