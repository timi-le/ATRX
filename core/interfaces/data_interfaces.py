"""
Data interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from collections.abc import AsyncIterator

import pandas as pd


@dataclass
class MarketData:
    """Standardized market data structure."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    volume: float | None = None
    source: str | None = None
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

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""

    @abstractmethod
    async def subscribe_ticks(self, symbols: list[str]) -> AsyncIterator[MarketData]:
        """Subscribe to real-time tick data."""

    @abstractmethod
    async def subscribe_bars(
        self, symbols: list[str], timeframe: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        """Subscribe to real-time bar data."""

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Retrieve historical market data."""


class DataPublisher(ABC):
    """Abstract base class for data publishers."""

    @abstractmethod
    async def publish_tick(self, topic: str, data: MarketData) -> None:
        """Publish tick data to a topic."""

    @abstractmethod
    async def publish_bar(self, topic: str, data: OHLCV) -> None:
        """Publish bar data to a topic."""

    @abstractmethod
    async def publish_features(self, topic: str, features: dict[str, Any]) -> None:
        """Publish computed features."""


class DataSubscriber(ABC):
    """Abstract base class for data subscribers."""

    @abstractmethod
    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        """Subscribe to a data topic."""

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a data topic."""


class MarketDataFeed(ABC):
    """Abstract base class for market data feeds."""

    @abstractmethod
    async def start_feed(self, symbols: list[str]) -> None:
        """Start the market data feed."""

    @abstractmethod
    async def stop_feed(self) -> None:
        """Stop the market data feed."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if feed is connected."""


class HistoricalDataProvider(ABC):
    """Abstract base class for historical data providers."""

    @abstractmethod
    async def get_fx_data(
        self, pair: str, start_date: datetime, end_date: datetime, timeframe: str = "1m"
    ) -> pd.DataFrame:
        """Get historical FX data."""

    @abstractmethod
    async def get_economic_data(
        self, indicator: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get historical economic indicator data."""

    @abstractmethod
    async def cache_data(
        self, key: str, data: pd.DataFrame, ttl: int | None = None
    ) -> None:
        """Cache data for future use."""
