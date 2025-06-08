"""
News Sentiment Parser for FX AI-Quant Trading System.

This module provides functionality to fetch news from various sources,
analyze sentiment, and extract currency-relevant information.
"""

import asyncio
import re
from datetime import datetime, timedelta, timezone

import aiohttp
import feedparser
import structlog

from core.interfaces.macro_interfaces import (
    Currency,
    NewsEvent,
    NewsProvider,
    SentimentAnalyzer,
)


class SimpleSentimentAnalyzer(SentimentAnalyzer):
    """Simple rule-based sentiment analyzer for quick prototyping."""

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or structlog.get_logger(__name__)

        # Simple sentiment word lists
        self.positive_words = {
            "bullish",
            "positive",
            "rise",
            "rising",
            "up",
            "increase",
            "growth",
            "gains",
            "stronger",
            "strengthen",
            "boost",
            "rally",
            "surged",
            "jumped",
            "climbed",
            "optimistic",
            "confident",
            "recovery",
            "improve",
            "improving",
            "outperform",
            "exceed",
            "beat",
            "better",
            "good",
            "strong",
            "robust",
            "solid",
            "healthy",
        }

        self.negative_words = {
            "bearish",
            "negative",
            "fall",
            "falling",
            "down",
            "decrease",
            "decline",
            "losses",
            "weaker",
            "weaken",
            "drop",
            "plunge",
            "tumbled",
            "crashed",
            "slumped",
            "pessimistic",
            "concern",
            "worried",
            "recession",
            "crisis",
            "deteriorate",
            "miss",
            "below",
            "worse",
            "bad",
            "weak",
            "fragile",
            "poor",
            "disappointing",
        }

        # Currency-specific keywords
        self.currency_keywords = {
            Currency.USD: [
                "dollar",
                "usd",
                "federal reserve",
                "fed",
                "fomc",
                "united states",
                "us economy",
                "treasury",
                "yellen",
                "powell",
                "america",
                "american",
            ],
            Currency.EUR: [
                "euro",
                "eur",
                "ecb",
                "european central bank",
                "eurozone",
                "eu",
                "lagarde",
                "european union",
                "germany",
                "france",
                "italy",
                "spain",
            ],
            Currency.GBP: [
                "pound",
                "sterling",
                "gbp",
                "bank of england",
                "boe",
                "uk",
                "britain",
                "british",
                "england",
                "bailey",
                "brexit",
            ],
            Currency.JPY: [
                "yen",
                "jpy",
                "bank of japan",
                "boj",
                "japan",
                "japanese",
                "tokyo",
                "kuroda",
                "ueda",
            ],
            Currency.AUD: [
                "australian dollar",
                "aud",
                "rba",
                "reserve bank australia",
                "australia",
                "aussie",
                "sydney",
                "canberra",
            ],
            Currency.CAD: [
                "canadian dollar",
                "cad",
                "bank of canada",
                "boc",
                "canada",
                "loonie",
                "ottawa",
                "toronto",
            ],
            Currency.CHF: [
                "swiss franc",
                "chf",
                "snb",
                "swiss national bank",
                "switzerland",
                "swiss",
                "zurich",
                "geneva",
            ],
            Currency.NZD: [
                "new zealand dollar",
                "nzd",
                "rbnz",
                "reserve bank new zealand",
                "new zealand",
                "kiwi",
                "wellington",
            ],
        }

    async def analyze_sentiment(self, text: str) -> tuple[float, float]:
        """Analyze sentiment of text. Returns (sentiment_score, confidence)."""
        if not text:
            return 0.0, 0.0

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0, 0.1  # Neutral with low confidence

        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_sentiment_words

        # Calculate confidence based on the proportion of sentiment words
        confidence = min(total_sentiment_words / len(words), 1.0) if words else 0.0

        return sentiment_score, confidence

    async def analyze_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        """Analyze sentiment for batch of texts."""
        results = []
        for text in texts:
            sentiment, confidence = await self.analyze_sentiment(text)
            results.append((sentiment, confidence))
        return results

    def extract_currencies(self, text: str) -> list[Currency]:
        """Extract mentioned currencies from text."""
        text_lower = text.lower()
        mentioned_currencies = []

        for currency, keywords in self.currency_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if currency not in mentioned_currencies:
                        mentioned_currencies.append(currency)
                    break

        return mentioned_currencies


class MockNewsProvider(NewsProvider):
    """Mock news provider for testing and development."""

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or structlog.get_logger(__name__)
        self.sentiment_analyzer = SimpleSentimentAnalyzer(logger)

        # Generate some mock news data
        self.mock_news_data = self._generate_mock_news()

    def _generate_mock_news(self) -> list[NewsEvent]:
        """Generate realistic mock news data."""
        import random

        random.seed(42)

        base_time = datetime.now(timezone.utc) - timedelta(days=7)

        # Sample headlines with different sentiments
        news_templates = [
            {
                "headline": "Federal Reserve Signals Potential Rate Cuts Amid Economic Uncertainty",
                "content": "The Federal Reserve indicated today that it may consider cutting interest rates in response to slowing economic indicators and increased market volatility.",
                "currencies": [Currency.USD],
                "expected_sentiment": -0.3,
            },
            {
                "headline": "European Central Bank Maintains Hawkish Stance on Inflation",
                "content": "ECB President Christine Lagarde reaffirmed the bank's commitment to fighting inflation, suggesting rates will remain elevated for the foreseeable future.",
                "currencies": [Currency.EUR],
                "expected_sentiment": 0.2,
            },
            {
                "headline": "UK GDP Growth Exceeds Expectations in Latest Quarter",
                "content": "The UK economy showed stronger than expected growth, with GDP rising 0.4% quarter-over-quarter, beating forecasts of 0.2%.",
                "currencies": [Currency.GBP],
                "expected_sentiment": 0.6,
            },
            {
                "headline": "Bank of Japan Intervenes to Support Weakening Yen",
                "content": "The BoJ stepped into currency markets today to support the yen after it reached new multi-year lows against the dollar.",
                "currencies": [Currency.JPY, Currency.USD],
                "expected_sentiment": -0.4,
            },
            {
                "headline": "Australian Employment Data Shows Strong Labor Market Resilience",
                "content": "Australia's unemployment rate dropped to 3.5%, its lowest level in decades, signaling continued strength in the labor market.",
                "currencies": [Currency.AUD],
                "expected_sentiment": 0.7,
            },
            {
                "headline": "Swiss National Bank Concerned About Global Financial Stability",
                "content": "SNB officials expressed concerns about potential spillovers from global financial market tensions to the Swiss economy.",
                "currencies": [Currency.CHF],
                "expected_sentiment": -0.2,
            },
            {
                "headline": "Oil Prices Surge on Supply Concerns, Boosting Canadian Dollar",
                "content": "Rising oil prices on supply disruption fears have provided strong support for the Canadian dollar against major currencies.",
                "currencies": [Currency.CAD],
                "expected_sentiment": 0.5,
            },
            {
                "headline": "New Zealand Central Bank Hints at Dovish Policy Shift",
                "content": "RBNZ Governor Adrian Orr suggested the central bank may need to reassess its monetary policy stance given changing economic conditions.",
                "currencies": [Currency.NZD],
                "expected_sentiment": -0.1,
            },
        ]

        mock_events = []

        for i in range(50):  # Generate 50 mock news events
            template = random.choice(news_templates)

            # Add time variation
            event_time = base_time + timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )

            # Add some variation to headlines
            variations = [
                template["headline"],
                template["headline"].replace("Signals", "Indicates"),
                template["headline"].replace("Shows", "Demonstrates"),
                template["headline"].replace("Exceeds", "Beats"),
            ]

            headline = random.choice(variations)

            # Calculate sentiment
            sentiment_score = template["expected_sentiment"] + random.gauss(0, 0.1)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]

            event = NewsEvent(
                news_id=f"mock_news_{i}",
                timestamp=event_time,
                headline=headline,
                content=template["content"],
                source="MockNewsProvider",
                currencies_mentioned=template["currencies"],
                sentiment_score=sentiment_score,
                confidence=random.uniform(0.6, 0.9),
                relevance=random.uniform(0.7, 1.0),
                impact_estimate=abs(sentiment_score) * random.uniform(0.8, 1.2),
            )

            mock_events.append(event)

        # Sort by timestamp
        mock_events.sort(key=lambda x: x.timestamp)

        self.logger.info(f"Generated {len(mock_events)} mock news events")
        return mock_events

    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch news events from mock data."""

        filtered_events = []

        for event in self.mock_news_data:
            # Date filter
            if not (start_date <= event.timestamp <= end_date):
                continue

            # Currency filter
            if currencies:
                if not any(curr in event.currencies_mentioned for curr in currencies):
                    continue

            # Keywords filter
            if keywords:
                text_to_search = f"{event.headline} {event.content}".lower()
                if not any(keyword.lower() in text_to_search for keyword in keywords):
                    continue

            filtered_events.append(event)

        self.logger.debug(
            f"Retrieved {len(filtered_events)} news events",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        return filtered_events

    async def get_live_news(self) -> list[NewsEvent]:
        """Get live/real-time news (mock implementation)."""
        now = datetime.now(timezone.utc)

        # Return news from the last 2 hours as "live"
        recent_news = [
            event
            for event in self.mock_news_data
            if (now - event.timestamp).total_seconds() < 7200  # 2 hours
        ]

        return recent_news


class RSSNewsProvider(NewsProvider):
    """RSS feed news provider for real news sources."""

    def __init__(
        self,
        rss_feeds: dict[str, str] | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentimentAnalyzer(logger)

        # Default RSS feeds for financial news
        self.rss_feeds = rss_feeds or {
            "Reuters": "http://feeds.reuters.com/reuters/businessNews",
            "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "MarketWatch": "http://feeds.marketwatch.com/marketwatch/topstories",
            "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "Financial Times": "https://www.ft.com/rss/home",
        }

        self.session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def _fetch_rss_feed(self, source: str, url: str) -> list[NewsEvent]:
        """Fetch and parse RSS feed."""
        try:
            await self._ensure_session()

            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    self.logger.warning(
                        f"Failed to fetch RSS feed from {source}: HTTP {response.status}"
                    )
                    return []

                content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(content)

            events = []

            for entry in feed.entries:
                # Extract basic information
                headline = getattr(entry, "title", "")
                content = getattr(entry, "summary", getattr(entry, "description", ""))

                # Parse timestamp
                try:
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        timestamp = datetime(
                            *entry.published_parsed[:6], tzinfo=timezone.utc
                        )
                    else:
                        timestamp = datetime.now(timezone.utc)
                except:
                    timestamp = datetime.now(timezone.utc)

                # Extract currencies mentioned
                currencies_mentioned = self.sentiment_analyzer.extract_currencies(
                    f"{headline} {content}"
                )

                # Analyze sentiment
                (
                    sentiment_score,
                    confidence,
                ) = await self.sentiment_analyzer.analyze_sentiment(
                    f"{headline} {content}"
                )

                # Create news event
                event = NewsEvent(
                    news_id=f"{source}_{hash(headline)}",
                    timestamp=timestamp,
                    headline=headline,
                    content=content,
                    source=source,
                    currencies_mentioned=currencies_mentioned,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    relevance=len(currencies_mentioned) * 0.2
                    + 0.5,  # Simple relevance scoring
                    impact_estimate=abs(sentiment_score) * confidence,
                )

                events.append(event)

            self.logger.debug(f"Fetched {len(events)} news items from {source}")
            return events

        except Exception as e:
            self.logger.error(f"Error fetching RSS feed from {source}: {e}")
            return []

    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch news events from RSS feeds."""

        all_events = []

        # Fetch from all RSS feeds concurrently
        tasks = [
            self._fetch_rss_feed(source, url) for source, url in self.rss_feeds.items()
        ]

        feed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for source, result in zip(self.rss_feeds.keys(), feed_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching from {source}: {result}")
            else:
                all_events.extend(result)

        # Filter events
        filtered_events = []

        for event in all_events:
            # Date filter
            if not (start_date <= event.timestamp <= end_date):
                continue

            # Currency filter
            if currencies:
                if not any(curr in event.currencies_mentioned for curr in currencies):
                    continue

            # Keywords filter
            if keywords:
                text_to_search = f"{event.headline} {event.content}".lower()
                if not any(keyword.lower() in text_to_search for keyword in keywords):
                    continue

            filtered_events.append(event)

        # Deduplicate by news_id
        unique_events = {}
        for event in filtered_events:
            unique_events[event.news_id] = event

        final_events = list(unique_events.values())
        final_events.sort(key=lambda x: x.timestamp)

        self.logger.info(
            f"Retrieved {len(final_events)} unique news events from RSS feeds"
        )
        return final_events

    async def get_live_news(self) -> list[NewsEvent]:
        """Get live/real-time news from RSS feeds."""
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(hours=6)  # Last 6 hours

        return await self.get_news_feed(start_date, now)

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org provider (placeholder for real implementation)."""

    def __init__(
        self,
        api_key: str | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.api_key = api_key
        self.logger = logger or structlog.get_logger(__name__)
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentimentAnalyzer(logger)
        self.base_url = "https://newsapi.org/v2"
        self.session: aiohttp.ClientSession | None = None

    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch news from NewsAPI (placeholder)."""

        if not self.api_key:
            self.logger.warning("No API key provided for NewsAPI - using mock data")
            mock_provider = MockNewsProvider(self.logger)
            return await mock_provider.get_news_feed(
                start_date, end_date, currencies, keywords
            )

        # Placeholder for real API implementation
        self.logger.warning("NewsAPIProvider not yet implemented")
        return []

    async def get_live_news(self) -> list[NewsEvent]:
        """Get live news from NewsAPI (placeholder)."""
        self.logger.warning("NewsAPIProvider live news not yet implemented")
        return []


class NewsSentimentParser:
    """Main news sentiment parser that aggregates multiple providers."""

    def __init__(
        self,
        news_providers: list[NewsProvider] | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentimentAnalyzer(logger)
        self.news_providers = news_providers or [MockNewsProvider(logger)]

        self.logger.info(f"Initialized with {len(self.news_providers)} news providers")

    async def get_news(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Get news from all providers and deduplicate."""

        all_news = []

        for provider in self.news_providers:
            try:
                news = await provider.get_news_feed(
                    start_date, end_date, currencies, keywords
                )
                all_news.extend(news)

                self.logger.debug(
                    f"Retrieved {len(news)} news items from {provider.__class__.__name__}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to get news from {provider.__class__.__name__}: {e}"
                )

        # Deduplicate by news_id and headline similarity
        unique_news = {}
        for news in all_news:
            # Use headline as key for deduplication
            key = news.headline.lower().strip()
            if key not in unique_news:
                unique_news[key] = news

        final_news = list(unique_news.values())
        final_news.sort(key=lambda x: x.timestamp)

        self.logger.info(
            f"Retrieved {len(final_news)} unique news items from {len(self.news_providers)} providers"
        )

        return final_news

    async def get_live_news(self) -> list[NewsEvent]:
        """Get live news from all providers."""
        all_news = []

        for provider in self.news_providers:
            try:
                news = await provider.get_live_news()
                all_news.extend(news)
            except Exception as e:
                self.logger.error(
                    f"Failed to get live news from {provider.__class__.__name__}: {e}"
                )

        # Deduplicate
        unique_news = {}
        for news in all_news:
            unique_news[news.news_id] = news

        return list(unique_news.values())

    async def analyze_sentiment_batch(
        self, texts: list[str]
    ) -> list[tuple[float, float]]:
        """Analyze sentiment for a batch of texts."""
        return await self.sentiment_analyzer.analyze_batch(texts)

    def add_provider(self, provider: NewsProvider):
        """Add a new news provider."""
        self.news_providers.append(provider)
        self.logger.info(f"Added provider: {provider.__class__.__name__}")

    def remove_provider(self, provider_class: type):
        """Remove a news provider by class."""
        self.news_providers = [
            p for p in self.news_providers if not isinstance(p, provider_class)
        ]
        self.logger.info(f"Removed provider: {provider_class.__name__}")

    async def close(self):
        """Close all providers that support cleanup."""
        for provider in self.news_providers:
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception as e:
                    self.logger.error(
                        f"Error closing provider {provider.__class__.__name__}: {e}"
                    )


# Utility functions
def extract_financial_entities(text: str) -> dict[str, list[str]]:
    """Extract financial entities from text."""
    entities = {
        "currencies": [],
        "central_banks": [],
        "economic_indicators": [],
        "institutions": [],
    }

    text_upper = text.upper()

    # Currency patterns
    currency_patterns = {
        "USD": r"\b(USD|DOLLAR|US\s*DOLLAR)\b",
        "EUR": r"\b(EUR|EURO)\b",
        "GBP": r"\b(GBP|POUND|STERLING)\b",
        "JPY": r"\b(JPY|YEN)\b",
    }

    for currency, pattern in currency_patterns.items():
        if re.search(pattern, text_upper):
            entities["currencies"].append(currency)

    # Central bank patterns
    cb_patterns = [
        "FEDERAL RESERVE",
        "FED",
        "ECB",
        "EUROPEAN CENTRAL BANK",
        "BANK OF ENGLAND",
        "BOE",
        "BANK OF JAPAN",
        "BOJ",
    ]

    for pattern in cb_patterns:
        if pattern in text_upper:
            entities["central_banks"].append(pattern)

    return entities


def calculate_news_impact_score(
    sentiment_score: float,
    confidence: float,
    relevance: float,
    source_weight: float = 1.0,
) -> float:
    """Calculate overall news impact score."""
    return abs(sentiment_score) * confidence * relevance * source_weight
