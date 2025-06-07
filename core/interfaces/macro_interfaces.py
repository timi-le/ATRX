"""
Macro Economic Data interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


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
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: str = ""
    source: str = ""
    
    @property
    def surprise(self) -> Optional[float]:
        """Calculate surprise as actual - forecast."""
        if self.actual is not None and self.forecast is not None:
            return self.actual - self.forecast
        return None
    
    @property
    def surprise_pct(self) -> Optional[float]:
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
    currencies_mentioned: List[Currency]
    sentiment_score: Optional[float] = None  # -1 to 1
    confidence: Optional[float] = None
    relevance: Optional[float] = None
    impact_estimate: Optional[float] = None


@dataclass
class MacroFeatureVector:
    """Combined macro economic feature vector."""
    timestamp: datetime
    currency_surprises: Dict[Currency, float]  # Normalized surprise scores
    sentiment_scores: Dict[Currency, float]    # Sentiment scores by currency
    rolling_surprise_means: Dict[Currency, float]  # Rolling averages
    rolling_surprise_stds: Dict[Currency, float]   # Rolling standard deviations
    event_counts: Dict[Currency, int]          # Recent event counts
    high_impact_flags: Dict[Currency, bool]    # High impact event flags
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for ML models."""
        features = {}
        
        # Add currency-specific features
        for currency in Currency:
            curr_str = currency.value
            features[f"{curr_str}_surprise"] = self.currency_surprises.get(currency, 0.0)
            features[f"{curr_str}_sentiment"] = self.sentiment_scores.get(currency, 0.0)
            features[f"{curr_str}_surprise_mean"] = self.rolling_surprise_means.get(currency, 0.0)
            features[f"{curr_str}_surprise_std"] = self.rolling_surprise_stds.get(currency, 1.0)
            features[f"{curr_str}_event_count"] = float(self.event_counts.get(currency, 0))
            features[f"{curr_str}_high_impact"] = float(self.high_impact_flags.get(currency, False))
        
        return features


class MacroDataProvider(ABC):
    """Abstract base class for macro economic data providers."""
    
    @abstractmethod
    async def get_economic_calendar(
        self, 
        start_date: datetime, 
        end_date: datetime,
        currencies: Optional[List[Currency]] = None,
        impact_levels: Optional[List[ImpactLevel]] = None
    ) -> List[EconomicEvent]:
        """Fetch economic calendar events."""
        pass
    
    @abstractmethod
    async def get_live_events(self) -> List[EconomicEvent]:
        """Get live/real-time economic events."""
        pass
    
    @abstractmethod
    async def update_event_actual(self, event_id: str, actual_value: float) -> bool:
        """Update an event with actual released value."""
        pass


class NewsProvider(ABC):
    """Abstract base class for news data providers."""
    
    @abstractmethod
    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[Currency]] = None,
        keywords: Optional[List[str]] = None
    ) -> List[NewsEvent]:
        """Fetch news events."""
        pass
    
    @abstractmethod
    async def get_live_news(self) -> List[NewsEvent]:
        """Get live/real-time news."""
        pass


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analysis."""
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of text. Returns (sentiment_score, confidence)."""
        pass
    
    @abstractmethod
    async def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for batch of texts."""
        pass
    
    @abstractmethod
    def extract_currencies(self, text: str) -> List[Currency]:
        """Extract mentioned currencies from text."""
        pass


class MacroEngine(ABC):
    """Abstract base class for macro economic feature engine."""
    
    @abstractmethod
    async def update_economic_event(self, event: EconomicEvent) -> None:
        """Update with new economic event."""
        pass
    
    @abstractmethod
    async def update_news_event(self, news: NewsEvent) -> None:
        """Update with new news event."""
        pass
    
    @abstractmethod
    async def get_latest_macro_vector(self, currency: Optional[Currency] = None) -> MacroFeatureVector:
        """Get latest macro feature vector."""
        pass
    
    @abstractmethod
    async def get_surprise_history(
        self, 
        currency: Currency, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EconomicEvent]:
        """Get historical surprise data."""
        pass
    
    @abstractmethod
    async def calculate_surprise_score(self, event: EconomicEvent) -> float:
        """Calculate normalized surprise score for an event."""
        pass
    
    @abstractmethod
    async def get_currency_correlations(self) -> pd.DataFrame:
        """Get correlation matrix between currency surprise scores."""
        pass 