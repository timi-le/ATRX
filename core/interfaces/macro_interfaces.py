"""
Macro Economic Data interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd


class ImpactLevel(Enum):
    """Economic event impact classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Currency(Enum):
    """Major currencies for FX trading."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    NZD = "NZD"


class EventType(Enum):
    """Types of economic events."""

    GDP = "gdp"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    INTEREST_RATE = "interest_rate"
    PMI = "pmi"
    RETAIL_SALES = "retail_sales"
    TRADE_BALANCE = "trade_balance"
    CENTRAL_BANK = "central_bank"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    MANUFACTURING = "manufacturing"


@dataclass
class EconomicEvent:
    """Economic calendar event data."""

    event_id: str
    timestamp: datetime
    currency: Currency
    event_type: EventType
    name: str
    impact: ImpactLevel
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    unit: str = ""
    source: str = ""

    @property
    def surprise(self) -> float | None:
        """Calculate surprise as actual - forecast."""
        if self.actual is not None and self.forecast is not None:
            return self.actual - self.forecast
        return None

    @property
    def surprise_pct(self) -> float | None:
        """Calculate percentage surprise."""
        if self.surprise is not None and self.forecast != 0:
            return (self.surprise / abs(self.forecast)) * 100
        return None


@dataclass
class NewsEvent:
    """News sentiment event data."""

    news_id: str
    timestamp: datetime
    headline: str
    content: str
    source: str
    currencies_mentioned: list[Currency]
    sentiment_score: float | None = None  # -1 to 1
    confidence: float | None = None
    relevance: float | None = None
    impact_estimate: float | None = None


@dataclass
class MacroFeatureVector:
    """Combined macro economic feature vector."""

    timestamp: datetime
    currency_surprises: dict[Currency, float]  # Normalized surprise scores
    sentiment_scores: dict[Currency, float]  # Sentiment scores by currency
    rolling_surprise_means: dict[Currency, float]  # Rolling averages
    rolling_surprise_stds: dict[Currency, float]  # Rolling standard deviations
    event_counts: dict[Currency, int]  # Recent event counts
    high_impact_flags: dict[Currency, bool]  # High impact event flags

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dictionary for ML models."""
        features = {}

        # Add currency-specific features
        for currency in Currency:
            curr_str = currency.value
            features[f"{curr_str}_surprise"] = self.currency_surprises.get(
                currency, 0.0
            )
            features[f"{curr_str}_sentiment"] = self.sentiment_scores.get(currency, 0.0)
            features[f"{curr_str}_surprise_mean"] = self.rolling_surprise_means.get(
                currency, 0.0
            )
            features[f"{curr_str}_surprise_std"] = self.rolling_surprise_stds.get(
                currency, 1.0
            )
            features[f"{curr_str}_event_count"] = float(
                self.event_counts.get(currency, 0)
            )
            features[f"{curr_str}_high_impact"] = float(
                self.high_impact_flags.get(currency, False)
            )

        return features


class MacroDataProvider(ABC):
    """Abstract base class for macro economic data providers."""

    @abstractmethod
    async def get_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        impact_levels: list[ImpactLevel] | None = None,
    ) -> list[EconomicEvent]:
        """Fetch economic calendar events."""

    @abstractmethod
    async def get_live_events(self) -> list[EconomicEvent]:
        """Get live/real-time economic events."""

    @abstractmethod
    async def update_event_actual(self, event_id: str, actual_value: float) -> bool:
        """Update an event with actual released value."""


class NewsProvider(ABC):
    """Abstract base class for news data providers."""

    @abstractmethod
    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch news events."""

    @abstractmethod
    async def get_live_news(self) -> list[NewsEvent]:
        """Get live/real-time news."""


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analysis."""

    @abstractmethod
    async def analyze_sentiment(self, text: str) -> tuple[float, float]:
        """Analyze sentiment of text. Returns (sentiment_score, confidence)."""

    @abstractmethod
    async def analyze_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        """Analyze sentiment for batch of texts."""

    @abstractmethod
    def extract_currencies(self, text: str) -> list[Currency]:
        """Extract mentioned currencies from text."""


class MacroEngine(ABC):
    """Abstract base class for macro economic feature engine."""

    @abstractmethod
    async def update_economic_event(self, event: EconomicEvent) -> None:
        """Update with new economic event."""

    @abstractmethod
    async def update_news_event(self, news: NewsEvent) -> None:
        """Update with new news event."""

    @abstractmethod
    async def get_latest_macro_vector(
        self, currency: Currency | None = None
    ) -> MacroFeatureVector:
        """Get latest macro feature vector."""

    @abstractmethod
    async def get_surprise_history(
        self, currency: Currency, start_date: datetime, end_date: datetime
    ) -> list[EconomicEvent]:
        """Get historical surprise data."""

    @abstractmethod
    async def calculate_surprise_score(self, event: EconomicEvent) -> float:
        """Calculate normalized surprise score for an event."""

    @abstractmethod
    async def get_currency_correlations(self) -> pd.DataFrame:
        """Get correlation matrix between currency surprise scores."""
