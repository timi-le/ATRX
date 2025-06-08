"""
Real NewsAPI Provider for FX AI-Quant Trading System.

This module provides real-time financial news integration using NewsAPI.org
and RSS feeds from major financial news sources.
"""

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin

import aiohttp
import feedparser
import structlog

from core.interfaces.macro_interfaces import (
    Currency,
    NewsEvent,
    NewsProvider,
)


@dataclass
class NewsAPIConfig:
    """Configuration for NewsAPI integration."""

    api_key: str
    base_url: str = "https://newsapi.org/v2/"
    sources: list[str] = None
    language: str = "en"
    page_size: int = 100

    def __post_init__(self):
        if self.sources is None:
            self.sources = [
                "bloomberg",
                "reuters",
                "financial-times",
                "the-wall-street-journal",
                "cnbc",
                "marketwatch",
                "business-insider",
            ]


class RealNewsAPIProvider(NewsProvider):
    """Real NewsAPI.org provider for financial news."""

    def __init__(
        self,
        api_key: str | None = None,
        config: NewsAPIConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.api_key = api_key or os.getenv(
            "NEWS_API_KEY", "ec45f3866330462db1c4e49c60ea22cd"
        )
        if not self.api_key:
            raise ValueError(
                "NewsAPI key is required. Set NEWS_API_KEY environment variable."
            )

        self.config = config or NewsAPIConfig(api_key=self.api_key)
        self.logger = logger or structlog.get_logger(__name__)
        self.session: aiohttp.ClientSession | None = None

        # Currency keywords for extraction
        self.currency_keywords = {
            Currency.USD: ["dollar", "usd", "federal reserve", "fed", "us economy"],
            Currency.EUR: ["euro", "eur", "european central bank", "ecb", "eurozone"],
            Currency.GBP: [
                "pound",
                "gbp",
                "sterling",
                "bank of england",
                "boe",
                "uk economy",
            ],
            Currency.JPY: ["yen", "jpy", "bank of japan", "boj", "japan economy"],
            Currency.CHF: ["franc", "chf", "swiss national bank", "snb", "switzerland"],
            Currency.CAD: ["canadian dollar", "cad", "bank of canada", "boc", "canada"],
            Currency.AUD: [
                "australian dollar",
                "aud",
                "reserve bank",
                "rba",
                "australia",
            ],
            Currency.NZD: ["new zealand dollar", "nzd", "rbnz", "new zealand"],
        }

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch news from NewsAPI for specified time range."""

        if not self.session:
            self.session = aiohttp.ClientSession()

        currencies = currencies or list(Currency)
        all_news = []

        # Build search query for financial news
        financial_keywords = keywords or [
            "forex",
            "currency",
            "central bank",
            "interest rates",
            "inflation",
            "gdp",
            "employment",
            "trade",
            "monetary policy",
        ]

        # Add currency-specific keywords
        currency_terms = []
        for currency in currencies:
            currency_terms.extend(self.currency_keywords.get(currency, []))

        # Combine keywords
        query = " OR ".join(
            financial_keywords + currency_terms[:10]
        )  # Limit query length

        try:
            # Fetch from NewsAPI
            news_data = await self._fetch_newsapi_data(query, start_date, end_date)

            # Process articles
            for article in news_data.get("articles", []):
                news_event = await self._process_article(article, currencies)
                if news_event:
                    all_news.append(news_event)

            self.logger.info(f"Fetched {len(all_news)} news events from NewsAPI")
            return all_news

        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI data: {e}")
            return []

    async def get_live_news(self) -> list[NewsEvent]:
        """Get live/real-time news."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)  # Last hour
        return await self.get_news_feed(start_time, end_time)

    async def _fetch_newsapi_data(
        self, query: str, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Fetch data from NewsAPI."""

        url = urljoin(self.config.base_url, "everything")

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": self.config.language,
            "pageSize": self.config.page_size,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
        }

        # Add sources if specified
        if self.config.sources:
            params["sources"] = ",".join(self.config.sources)

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"NewsAPI error {response.status}: {error_text}")
                return {"articles": []}

    async def _process_article(
        self, article: dict[str, Any], currencies: list[Currency]
    ) -> NewsEvent | None:
        """Process a single article into a NewsEvent."""

        try:
            # Extract basic information
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            article.get("url", "")
            source = article.get("source", {}).get("name", "Unknown")

            # Parse publication time
            published_str = article.get("publishedAt", "")
            if published_str:
                published_at = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            else:
                published_at = datetime.now(timezone.utc)

            # Combine text for analysis
            full_text = f"{title} {description} {content}".lower()

            # Extract currencies mentioned
            mentioned_currencies = []
            for currency in currencies:
                keywords = self.currency_keywords.get(currency, [])
                if any(keyword in full_text for keyword in keywords):
                    mentioned_currencies.append(currency)

            # Skip if no relevant currencies
            if not mentioned_currencies:
                return None

            # Simple sentiment analysis (can be enhanced)
            sentiment_score = self._analyze_sentiment(full_text)

            return NewsEvent(
                news_id=str(uuid.uuid4()),
                timestamp=published_at,
                headline=title,
                content=description or content,
                source=source,
                currencies_mentioned=mentioned_currencies,
                sentiment_score=sentiment_score,
                confidence=0.7,  # Default confidence
                relevance=self._calculate_relevance(title, description, source),
                impact_estimate=self._calculate_importance(title, description, source),
            )

        except Exception as e:
            self.logger.warning(f"Error processing article: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis."""

        positive_words = [
            "growth",
            "increase",
            "rise",
            "boost",
            "strong",
            "positive",
            "gain",
            "improve",
            "recovery",
            "optimistic",
            "bullish",
        ]

        negative_words = [
            "decline",
            "fall",
            "drop",
            "weak",
            "negative",
            "loss",
            "concern",
            "worry",
            "crisis",
            "recession",
            "bearish",
        ]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        # Normalize to -1 to 1 scale
        sentiment = (positive_count - negative_count) / max(total_words / 100, 1)
        return max(-1.0, min(1.0, sentiment))

    def _calculate_relevance(self, title: str, description: str, source: str) -> float:
        """Calculate news relevance score."""

        relevance = 0.5  # Base relevance

        # Source credibility
        high_credibility_sources = ["reuters", "bloomberg", "financial times"]
        if any(src in source.lower() for src in high_credibility_sources):
            relevance += 0.2

        # Title keywords
        important_keywords = [
            "central bank",
            "interest rate",
            "inflation",
            "gdp",
            "employment",
            "trade war",
            "brexit",
            "fed",
            "ecb",
        ]

        text = f"{title} {description}".lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        relevance += min(0.3, keyword_count * 0.1)

        return min(1.0, relevance)

    def _calculate_importance(self, title: str, description: str, source: str) -> float:
        """Calculate news importance score."""

        importance = 0.5  # Base importance

        # Source credibility
        high_credibility_sources = ["reuters", "bloomberg", "financial times"]
        if any(src in source.lower() for src in high_credibility_sources):
            importance += 0.2

        # Title keywords
        important_keywords = [
            "central bank",
            "interest rate",
            "inflation",
            "gdp",
            "employment",
            "trade war",
            "brexit",
            "fed",
            "ecb",
        ]

        text = f"{title} {description}".lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        importance += min(0.3, keyword_count * 0.1)

        return min(1.0, importance)


class RSSNewsProvider(NewsProvider):
    """RSS feed provider for financial news."""

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or structlog.get_logger(__name__)

        # RSS feeds for major financial news sources
        self.rss_feeds = {
            "forex_factory": "https://www.forexfactory.com/rss.php",
            "investing_com": "https://www.investing.com/rss/news.rss",
            "dailyfx": "https://www.dailyfx.com/feeds/market-news",
            "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
            "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        }

        # Currency keywords for RSS content
        self.currency_keywords = {
            Currency.USD: ["dollar", "usd", "federal reserve", "fed"],
            Currency.EUR: ["euro", "eur", "ecb", "eurozone"],
            Currency.GBP: ["pound", "gbp", "sterling", "boe"],
            Currency.JPY: ["yen", "jpy", "boj", "japan"],
            Currency.CHF: ["franc", "chf", "snb", "swiss"],
            Currency.CAD: ["canadian dollar", "cad", "boc"],
            Currency.AUD: ["australian dollar", "aud", "rba"],
            Currency.NZD: ["new zealand dollar", "nzd", "rbnz"],
        }

    async def get_news_feed(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: list[Currency] | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch news from RSS feeds."""

        currencies = currencies or list(Currency)
        all_news = []

        for feed_name, feed_url in self.rss_feeds.items():
            try:
                self.logger.info(f"Fetching RSS feed: {feed_name}")

                # Parse RSS feed
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    news_event = self._process_rss_entry(
                        entry, feed_name, currencies, start_date, end_date
                    )
                    if news_event:
                        all_news.append(news_event)

            except Exception as e:
                self.logger.warning(f"Error fetching RSS feed {feed_name}: {e}")
                continue

        self.logger.info(f"Fetched {len(all_news)} news events from RSS feeds")
        return all_news

    async def get_live_news(self) -> list[NewsEvent]:
        """Get live/real-time news."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)  # Last hour
        return await self.get_news_feed(start_time, end_time)

    def _process_rss_entry(
        self,
        entry: Any,
        source: str,
        currencies: list[Currency],
        start_date: datetime,
        end_date: datetime,
    ) -> NewsEvent | None:
        """Process RSS entry into NewsEvent."""

        try:
            # Extract basic info
            title = getattr(entry, "title", "")
            description = getattr(entry, "description", "") or getattr(
                entry, "summary", ""
            )
            getattr(entry, "link", "")

            # Parse publication time
            published_at = datetime.now(timezone.utc)
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(
                    *entry.published_parsed[:6], tzinfo=timezone.utc
                )
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

            # Filter by time range
            if published_at < start_date or published_at > end_date:
                return None

            # Combine text for analysis
            full_text = f"{title} {description}".lower()

            # Extract currencies
            mentioned_currencies = []
            for currency in currencies:
                keywords = self.currency_keywords.get(currency, [])
                if any(keyword in full_text for keyword in keywords):
                    mentioned_currencies.append(currency)

            # Skip if no relevant currencies
            if not mentioned_currencies:
                return None

            # Simple sentiment analysis
            sentiment_score = self._analyze_sentiment(full_text)

            return NewsEvent(
                news_id=str(uuid.uuid4()),
                timestamp=published_at,
                headline=title,
                content=description,
                source=source,
                currencies_mentioned=mentioned_currencies,
                sentiment_score=sentiment_score,
                confidence=0.6,  # Default confidence for RSS
                relevance=self._calculate_relevance(title, description, source),
                impact_estimate=self._calculate_importance(title, description, source),
            )

        except Exception as e:
            self.logger.warning(f"Error processing RSS entry: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis for RSS content."""

        positive_words = ["growth", "rise", "gain", "strong", "positive", "boost"]
        negative_words = ["fall", "decline", "weak", "negative", "concern", "crisis"]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / max(total_words / 50, 1)
        return max(-1.0, min(1.0, sentiment))

    def _calculate_relevance(self, title: str, description: str, source: str) -> float:
        """Calculate relevance for RSS news."""

        relevance = 0.4  # Base for RSS

        # Source boost
        if "reuters" in source or "bloomberg" in source:
            relevance += 0.3
        elif "forex_factory" in source or "dailyfx" in source:
            relevance += 0.2

        # Keyword boost
        important_terms = ["central bank", "interest rate", "inflation", "gdp"]
        text = f"{title} {description}".lower()

        for term in important_terms:
            if term in text:
                relevance += 0.1

        return min(1.0, relevance)

    def _calculate_importance(self, title: str, description: str, source: str) -> float:
        """Calculate importance for RSS news."""

        importance = 0.4  # Base for RSS

        # Source boost
        if "reuters" in source or "bloomberg" in source:
            importance += 0.3
        elif "forex_factory" in source or "dailyfx" in source:
            importance += 0.2

        # Keyword boost
        important_terms = ["central bank", "interest rate", "inflation", "gdp"]
        text = f"{title} {description}".lower()

        for term in important_terms:
            if term in text:
                importance += 0.1

        return min(1.0, importance)


# Factory function for easy creation
def create_news_provider(
    provider_type: str = "newsapi",
    api_key: str | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> NewsProvider:
    """Create a news provider instance."""

    if provider_type == "newsapi":
        return RealNewsAPIProvider(api_key=api_key, logger=logger)
    elif provider_type == "rss":
        return RSSNewsProvider(logger=logger)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Demo usage
async def demo_news_providers():
    """Demonstrate news providers."""

    import structlog

    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(__name__)

    # Time range for news
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)

    currencies = [Currency.USD, Currency.EUR, Currency.GBP]

    # Test RSS provider
    logger.info("Testing RSS News Provider")
    rss_provider = RSSNewsProvider(logger=logger)

    try:
        rss_news = await rss_provider.get_news_feed(start_time, end_time, currencies)
        logger.info(f"RSS Provider: Found {len(rss_news)} news events")

        if rss_news:
            sample = rss_news[0]
            logger.info(f"Sample RSS news: {sample.headline[:100]}...")
            logger.info(
                f"Sentiment: {sample.sentiment_score:.2f}, Relevance: {sample.relevance:.2f}"
            )

    except Exception as e:
        logger.error(f"RSS provider failed: {e}")

    # Test NewsAPI provider (if API key available)
    api_key = os.getenv("NEWS_API_KEY", "ec45f3866330462db1c4e49c60ea22cd")
    if api_key:
        logger.info("Testing NewsAPI Provider")

        try:
            async with RealNewsAPIProvider(
                api_key=api_key, logger=logger
            ) as newsapi_provider:
                newsapi_news = await newsapi_provider.get_news_feed(
                    start_time, end_time, currencies
                )
                logger.info(f"NewsAPI Provider: Found {len(newsapi_news)} news events")

                if newsapi_news:
                    sample = newsapi_news[0]
                    logger.info(f"Sample NewsAPI news: {sample.headline[:100]}...")
                    logger.info(
                        f"Sentiment: {sample.sentiment_score:.2f}, Relevance: {sample.relevance:.2f}"
                    )

        except Exception as e:
            logger.error(f"NewsAPI provider failed: {e}")
    else:
        logger.warning("NEWS_API_KEY not found, skipping NewsAPI test")


if __name__ == "__main__":
    asyncio.run(demo_news_providers())
