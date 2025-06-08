"""
Macro Economic Feature Engine for FX AI-Quant Trading System.

This module combines economic surprises and news sentiment to create
unified macro feature vectors for ML models and regime detection.
"""

import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
import structlog

from core.interfaces.macro_interfaces import (
    Currency,
    EconomicEvent,
    EventType,
    ImpactLevel,
    MacroDataProvider,
    MacroEngine,
    MacroFeatureVector,
    NewsEvent,
    NewsProvider,
    SentimentAnalyzer,
)
from data.econ_calendar_loader import EconomicCalendarLoader, MockEconomicDataProvider
from data.news_sentiment_parser import (
    MockNewsProvider,
    NewsSentimentParser,
    SimpleSentimentAnalyzer,
)


class HighPerformanceMacroEngine(MacroEngine):
    """
    High-performance macro economic feature engine that combines:
    1. Economic surprise indices (actual vs forecast)
    2. News sentiment analysis
    3. Rolling statistics and z-score normalization
    """

    def __init__(
        self,
        economic_providers: list[MacroDataProvider] | None = None,
        news_providers: list[NewsProvider] | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        surprise_window: int = 252,  # 1 year of daily data
        sentiment_window: int = 30,  # 30 days for sentiment rolling average
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)

        # Initialize data providers
        self.economic_loader = EconomicCalendarLoader(
            providers=economic_providers or [MockEconomicDataProvider(self.logger)],
            logger=self.logger,
        )

        self.news_parser = NewsSentimentParser(
            news_providers=news_providers or [MockNewsProvider(self.logger)],
            sentiment_analyzer=sentiment_analyzer
            or SimpleSentimentAnalyzer(self.logger),
            logger=self.logger,
        )

        # Configuration
        self.surprise_window = surprise_window
        self.sentiment_window = sentiment_window

        # Data storage
        self.economic_events: dict[Currency, deque] = defaultdict(
            lambda: deque(maxlen=surprise_window)
        )
        self.news_events: dict[Currency, deque] = defaultdict(
            lambda: deque(maxlen=sentiment_window)
        )
        self.surprise_history: dict[Currency, deque] = defaultdict(
            lambda: deque(maxlen=surprise_window)
        )
        self.sentiment_history: dict[Currency, deque] = defaultdict(
            lambda: deque(maxlen=sentiment_window)
        )

        # Cache for latest features
        self._latest_macro_vector: MacroFeatureVector | None = None
        self._last_update_time: datetime | None = None

        # Event type weights for impact calculation
        self.event_type_weights = {
            EventType.GDP: 1.0,
            EventType.INFLATION: 0.9,
            EventType.EMPLOYMENT: 0.8,
            EventType.INTEREST_RATE: 1.0,
            EventType.PMI: 0.6,
            EventType.RETAIL_SALES: 0.5,
            EventType.TRADE_BALANCE: 0.4,
            EventType.CENTRAL_BANK: 0.9,
            EventType.CONSUMER_CONFIDENCE: 0.4,
            EventType.MANUFACTURING: 0.5,
        }

        self.impact_level_weights = {
            ImpactLevel.HIGH: 1.0,
            ImpactLevel.MEDIUM: 0.6,
            ImpactLevel.LOW: 0.3,
        }

        self.logger.info(
            "HighPerformanceMacroEngine initialized",
            surprise_window=surprise_window,
            sentiment_window=sentiment_window,
        )

    async def update_economic_event(self, event: EconomicEvent) -> None:
        """Update with new economic event and calculate surprise."""
        try:
            # Store the event
            self.economic_events[event.currency].append(event)

            # Calculate and store surprise score
            surprise_score = await self.calculate_surprise_score(event)

            self.surprise_history[event.currency].append(
                {
                    "timestamp": event.timestamp,
                    "surprise_score": surprise_score,
                    "event_type": event.event_type,
                    "impact": event.impact,
                    "raw_surprise": event.surprise,
                    "surprise_pct": event.surprise_pct,
                }
            )

            # Invalidate cache
            self._latest_macro_vector = None

            self.logger.debug(
                f"Updated economic event for {event.currency.value}",
                event_name=event.name,
                surprise_score=surprise_score,
                raw_surprise=event.surprise,
            )

        except Exception as e:
            self.logger.error(
                f"Error updating economic event: {e}", event_id=event.event_id
            )

    async def update_news_event(self, news: NewsEvent) -> None:
        """Update with new news event."""
        try:
            # Store news for each mentioned currency
            for currency in news.currencies_mentioned:
                self.news_events[currency].append(news)

                # Calculate weighted sentiment score
                impact_weight = news.impact_estimate or 1.0
                confidence_weight = news.confidence or 0.5
                weighted_sentiment = (
                    (news.sentiment_score or 0.0) * impact_weight * confidence_weight
                )

                self.sentiment_history[currency].append(
                    {
                        "timestamp": news.timestamp,
                        "sentiment_score": weighted_sentiment,
                        "raw_sentiment": news.sentiment_score,
                        "confidence": news.confidence,
                        "impact_estimate": news.impact_estimate,
                        "source": news.source,
                    }
                )

            # Invalidate cache
            self._latest_macro_vector = None

            self.logger.debug(
                f"Updated news event affecting {len(news.currencies_mentioned)} currencies",
                headline=news.headline[:100],
                sentiment=news.sentiment_score,
                currencies=[c.value for c in news.currencies_mentioned],
            )

        except Exception as e:
            self.logger.error(f"Error updating news event: {e}", news_id=news.news_id)

    async def calculate_surprise_score(self, event: EconomicEvent) -> float:
        """Calculate normalized surprise score for an event."""
        if event.surprise is None:
            return 0.0

        try:
            # Get historical surprises for this currency and event type
            currency_history = list(self.surprise_history[event.currency])
            same_type_surprises = [
                h["raw_surprise"]
                for h in currency_history
                if h.get("event_type") == event.event_type
                and h.get("raw_surprise") is not None
            ]

            if (
                len(same_type_surprises) < 5
            ):  # Not enough history, use simple normalization
                # Use percentage surprise if available, otherwise simple normalization
                if event.surprise_pct is not None:
                    normalized_surprise = (
                        event.surprise_pct / 100.0
                    )  # Convert to decimal
                else:
                    # Simple normalization based on forecast magnitude
                    if event.forecast and abs(event.forecast) > 0:
                        normalized_surprise = event.surprise / abs(event.forecast)
                    else:
                        normalized_surprise = 0.0
            else:
                # Z-score normalization using historical data
                mean_surprise = statistics.mean(same_type_surprises)
                if len(same_type_surprises) > 1:
                    std_surprise = statistics.stdev(same_type_surprises)
                    if std_surprise > 0:
                        normalized_surprise = (
                            event.surprise - mean_surprise
                        ) / std_surprise
                    else:
                        normalized_surprise = 0.0
                else:
                    normalized_surprise = event.surprise - mean_surprise

            # Apply event type and impact weights
            type_weight = self.event_type_weights.get(event.event_type, 0.5)
            impact_weight = self.impact_level_weights.get(event.impact, 0.5)

            weighted_score = normalized_surprise * type_weight * impact_weight

            # Clamp to reasonable range
            return max(-5.0, min(5.0, weighted_score))

        except Exception as e:
            self.logger.error(f"Error calculating surprise score: {e}")
            return 0.0

    async def get_latest_macro_vector(
        self, currency: Currency | None = None
    ) -> MacroFeatureVector:
        """Get latest macro feature vector."""

        # Use cache if recent enough (within 5 minutes)
        now = datetime.now(timezone.utc)
        if (
            self._latest_macro_vector
            and self._last_update_time
            and (now - self._last_update_time).total_seconds() < 300
        ):
            return self._latest_macro_vector

        # Calculate features for all currencies
        currency_surprises = {}
        sentiment_scores = {}
        rolling_surprise_means = {}
        rolling_surprise_stds = {}
        event_counts = {}
        high_impact_flags = {}

        for curr in Currency:
            # Calculate surprise features
            recent_surprises = [
                h["surprise_score"]
                for h in list(self.surprise_history[curr])
                if h["surprise_score"] is not None
                and (now - h["timestamp"]).total_seconds() < 86400 * 7  # Last 7 days
            ]

            if recent_surprises:
                currency_surprises[curr] = recent_surprises[-1]  # Most recent
                rolling_surprise_means[curr] = statistics.mean(recent_surprises)
                rolling_surprise_stds[curr] = (
                    statistics.stdev(recent_surprises)
                    if len(recent_surprises) > 1
                    else 1.0
                )
            else:
                currency_surprises[curr] = 0.0
                rolling_surprise_means[curr] = 0.0
                rolling_surprise_stds[curr] = 1.0

            # Calculate sentiment features
            recent_sentiments = [
                h["sentiment_score"]
                for h in list(self.sentiment_history[curr])
                if h["sentiment_score"] is not None
                and (now - h["timestamp"]).total_seconds() < 86400 * 3  # Last 3 days
            ]

            if recent_sentiments:
                sentiment_scores[curr] = statistics.mean(recent_sentiments)
            else:
                sentiment_scores[curr] = 0.0

            # Event counts (last 24 hours)
            recent_econ_events = [
                e
                for e in list(self.economic_events[curr])
                if (now - e.timestamp).total_seconds() < 86400
            ]

            recent_news_events = [
                n
                for n in list(self.news_events[curr])
                if (now - n.timestamp).total_seconds() < 86400
            ]

            event_counts[curr] = len(recent_econ_events) + len(recent_news_events)

            # High impact events flag (last 24 hours)
            high_impact_events = [
                e for e in recent_econ_events if e.impact == ImpactLevel.HIGH
            ]

            high_impact_flags[curr] = len(high_impact_events) > 0

        # Create feature vector
        macro_vector = MacroFeatureVector(
            timestamp=now,
            currency_surprises=currency_surprises,
            sentiment_scores=sentiment_scores,
            rolling_surprise_means=rolling_surprise_means,
            rolling_surprise_stds=rolling_surprise_stds,
            event_counts=event_counts,
            high_impact_flags=high_impact_flags,
        )

        # Cache the result
        self._latest_macro_vector = macro_vector
        self._last_update_time = now

        self.logger.debug(
            "Generated latest macro feature vector", timestamp=now.isoformat()
        )

        return macro_vector

    async def get_surprise_history(
        self, currency: Currency, start_date: datetime, end_date: datetime
    ) -> list[EconomicEvent]:
        """Get historical surprise data."""

        # First check local cache
        cached_events = [
            e
            for e in list(self.economic_events[currency])
            if start_date <= e.timestamp <= end_date
        ]

        if cached_events:
            return sorted(cached_events, key=lambda x: x.timestamp)

        # Fetch from data providers
        try:
            events = await self.economic_loader.get_events(
                start_date, end_date, currencies=[currency]
            )

            # Update cache
            for event in events:
                await self.update_economic_event(event)

            return events

        except Exception as e:
            self.logger.error(f"Error fetching surprise history: {e}")
            return []

    async def get_currency_correlations(self) -> pd.DataFrame:
        """Get correlation matrix between currency surprise scores."""
        try:
            # Collect surprise scores for all currencies
            data = {}

            for currency in Currency:
                surprise_data = [
                    h["surprise_score"]
                    for h in list(self.surprise_history[currency])
                    if h["surprise_score"] is not None
                ]

                # Pad or truncate to same length
                min_length = min(50, len(surprise_data)) if surprise_data else 0
                if min_length > 0:
                    data[currency.value] = surprise_data[-min_length:]

            if len(data) < 2:
                # Return empty correlation matrix
                return pd.DataFrame()

            # Create DataFrame and calculate correlations
            df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
            correlation_matrix = df.corr()

            return correlation_matrix.fillna(0.0)

        except Exception as e:
            self.logger.error(f"Error calculating currency correlations: {e}")
            return pd.DataFrame()

    async def backfill_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
    ) -> None:
        """Backfill historical economic and news data."""
        try:
            target_currencies = currencies or list(Currency)

            self.logger.info(
                f"Backfilling historical data from {start_date.date()} to {end_date.date()}"
            )

            # Fetch economic events
            economic_events = await self.economic_loader.get_events(
                start_date, end_date, currencies=target_currencies
            )

            self.logger.info(f"Backfilling {len(economic_events)} economic events")

            for event in economic_events:
                await self.update_economic_event(event)

            # Fetch news events
            news_events = await self.news_parser.get_news(
                start_date, end_date, currencies=target_currencies
            )

            self.logger.info(f"Backfilling {len(news_events)} news events")

            for news in news_events:
                await self.update_news_event(news)

            self.logger.info("Historical data backfill completed")

        except Exception as e:
            self.logger.error(f"Error backfilling historical data: {e}")

    async def get_feature_importance(self) -> dict[str, float]:
        """Get importance scores for different macro features."""

        # Calculate feature importance based on historical volatility
        importance_scores = {}

        try:
            for currency in Currency:
                curr_str = currency.value

                # Surprise volatility
                surprise_data = [
                    h["surprise_score"]
                    for h in list(self.surprise_history[currency])
                    if h["surprise_score"] is not None
                ]

                if len(surprise_data) > 1:
                    surprise_vol = statistics.stdev(surprise_data)
                    importance_scores[f"{curr_str}_surprise"] = surprise_vol
                else:
                    importance_scores[f"{curr_str}_surprise"] = 0.0

                # Sentiment volatility
                sentiment_data = [
                    h["sentiment_score"]
                    for h in list(self.sentiment_history[currency])
                    if h["sentiment_score"] is not None
                ]

                if len(sentiment_data) > 1:
                    sentiment_vol = statistics.stdev(sentiment_data)
                    importance_scores[f"{curr_str}_sentiment"] = sentiment_vol
                else:
                    importance_scores[f"{curr_str}_sentiment"] = 0.0

            # Normalize scores
            max_score = (
                max(importance_scores.values()) if importance_scores.values() else 1.0
            )
            if max_score > 0:
                for key in importance_scores:
                    importance_scores[key] /= max_score

            return importance_scores

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}

    async def get_live_updates(self) -> tuple[list[EconomicEvent], list[NewsEvent]]:
        """Get live economic and news updates."""
        try:
            # Fetch live data from providers
            live_economic = await self.economic_loader.get_live_events()
            live_news = await self.news_parser.get_live_news()

            # Update with new events
            for event in live_economic:
                await self.update_economic_event(event)

            for news in live_news:
                await self.update_news_event(news)

            return live_economic, live_news

        except Exception as e:
            self.logger.error(f"Error getting live updates: {e}")
            return [], []

    async def close(self):
        """Clean up resources."""
        try:
            await self.economic_loader.close()
            await self.news_parser.close()
            self.logger.info("MacroEngine closed")
        except Exception as e:
            self.logger.error(f"Error closing MacroEngine: {e}")


# Factory function for easy instantiation
def create_macro_engine(
    use_mock_data: bool = True,
    surprise_window: int = 252,
    sentiment_window: int = 30,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> HighPerformanceMacroEngine:
    """Factory function to create macro engine with appropriate providers."""

    if use_mock_data:
        # Use mock providers for development/testing
        economic_providers = [MockEconomicDataProvider(logger)]
        news_providers = [MockNewsProvider(logger)]
    else:
        # Use real providers (would need API keys)
        from data.econ_calendar_loader import (
            ForexFactoryDataProvider,
            TradingEconomicsProvider,
        )
        from data.news_sentiment_parser import NewsAPIProvider, RSSNewsProvider

        economic_providers = [
            TradingEconomicsProvider(logger=logger),
            ForexFactoryDataProvider(logger=logger),
        ]
        news_providers = [
            RSSNewsProvider(logger=logger),
            NewsAPIProvider(logger=logger),
        ]

    sentiment_analyzer = SimpleSentimentAnalyzer(logger)

    return HighPerformanceMacroEngine(
        economic_providers=economic_providers,
        news_providers=news_providers,
        sentiment_analyzer=sentiment_analyzer,
        surprise_window=surprise_window,
        sentiment_window=sentiment_window,
        logger=logger,
    )
