"""
Unit tests for the Macro Economic Feature Engine.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock
import structlog

from core.macro_engine import HighPerformanceMacroEngine, create_macro_engine
from core.interfaces.macro_interfaces import (
    EconomicEvent,
    NewsEvent,
    MacroFeatureVector,
    Currency,
    EventType,
    ImpactLevel
)
from data.econ_calendar_loader import MockEconomicDataProvider
from data.news_sentiment_parser import MockNewsProvider, SimpleSentimentAnalyzer


class TestMacroInterfaces:
    """Test the macro interface data structures."""
    
    def test_economic_event_surprise_calculation(self):
        """Test economic event surprise calculations."""
        event = EconomicEvent(
            event_id="test_1",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.USD,
            event_type=EventType.EMPLOYMENT,
            name="Non-Farm Payrolls",
            impact=ImpactLevel.HIGH,
            actual=250.0,
            forecast=200.0,
            previous=180.0,
            unit="K",
            source="test"
        )
        
        # Test surprise calculation
        assert event.surprise == 50.0  # actual - forecast
        assert event.surprise_pct == 25.0  # (50/200) * 100
    
    def test_economic_event_no_surprise(self):
        """Test economic event with no forecast."""
        event = EconomicEvent(
            event_id="test_2",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.EUR,
            event_type=EventType.GDP,
            name="GDP",
            impact=ImpactLevel.HIGH,
            actual=2.1,
            forecast=None,
            unit="%",
            source="test"
        )
        
        assert event.surprise is None
        assert event.surprise_pct is None
    
    def test_news_event_creation(self):
        """Test news event creation."""
        news = NewsEvent(
            news_id="news_1",
            timestamp=datetime.now(timezone.utc),
            headline="Federal Reserve Cuts Rates",
            content="The Fed announced a 50bp rate cut today.",
            source="Reuters",
            currencies_mentioned=[Currency.USD],
            sentiment_score=-0.3,
            confidence=0.8,
            relevance=0.9,
            impact_estimate=0.7
        )
        
        assert news.news_id == "news_1"
        assert Currency.USD in news.currencies_mentioned
        assert news.sentiment_score == -0.3
    
    def test_macro_feature_vector_to_dict(self):
        """Test converting macro feature vector to dictionary."""
        vector = MacroFeatureVector(
            timestamp=datetime.now(timezone.utc),
            currency_surprises={Currency.USD: 0.5, Currency.EUR: -0.2},
            sentiment_scores={Currency.USD: 0.3, Currency.EUR: -0.1},
            rolling_surprise_means={Currency.USD: 0.1, Currency.EUR: 0.0},
            rolling_surprise_stds={Currency.USD: 1.0, Currency.EUR: 0.8},
            event_counts={Currency.USD: 3, Currency.EUR: 1},
            high_impact_flags={Currency.USD: True, Currency.EUR: False}
        )
        
        feature_dict = vector.to_dict()
        
        # Check that all currencies are represented
        assert "USD_surprise" in feature_dict
        assert "EUR_surprise" in feature_dict
        assert feature_dict["USD_surprise"] == 0.5
        assert feature_dict["EUR_sentiment"] == -0.1
        assert feature_dict["USD_high_impact"] == 1.0  # Boolean converted to float


class TestSimpleSentimentAnalyzer:
    """Test the simple sentiment analyzer."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """Create sentiment analyzer for testing."""
        return SimpleSentimentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_positive_sentiment(self, sentiment_analyzer):
        """Test positive sentiment analysis."""
        text = "The economy is showing strong growth and robust recovery with positive outlook."
        
        sentiment, confidence = await sentiment_analyzer.analyze_sentiment(text)
        
        assert sentiment > 0  # Should be positive
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_negative_sentiment(self, sentiment_analyzer):
        """Test negative sentiment analysis."""
        text = "The economy is in decline with weak performance and disappointing results."
        
        sentiment, confidence = await sentiment_analyzer.analyze_sentiment(text)
        
        assert sentiment < 0  # Should be negative
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_neutral_sentiment(self, sentiment_analyzer):
        """Test neutral sentiment analysis."""
        text = "The weather is nice today and I like coffee."
        
        sentiment, confidence = await sentiment_analyzer.analyze_sentiment(text)
        
        assert abs(sentiment) < 0.5  # Should be close to neutral
        assert confidence < 0.5  # Low confidence due to no financial sentiment words
    
    @pytest.mark.asyncio
    async def test_batch_sentiment_analysis(self, sentiment_analyzer):
        """Test batch sentiment analysis."""
        texts = [
            "Strong economic growth expected",
            "Market crash imminent",
            "Neutral economic conditions"
        ]
        
        results = await sentiment_analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_currency_extraction(self, sentiment_analyzer):
        """Test currency extraction from text."""
        text = "The Federal Reserve cut rates, boosting the dollar against the euro and yen."
        
        currencies = sentiment_analyzer.extract_currencies(text)
        
        assert Currency.USD in currencies
        assert Currency.EUR in currencies
        assert Currency.JPY in currencies


class TestMockDataProviders:
    """Test the mock data providers."""
    
    @pytest.mark.asyncio
    async def test_mock_economic_provider(self):
        """Test mock economic data provider."""
        provider = MockEconomicDataProvider()
        
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)
        
        events = await provider.get_economic_calendar(start_date, end_date)
        
        assert len(events) > 0
        assert all(isinstance(e, EconomicEvent) for e in events)
        assert all(start_date <= e.timestamp <= end_date for e in events)
    
    @pytest.mark.asyncio
    async def test_mock_economic_provider_currency_filter(self):
        """Test mock economic provider with currency filter."""
        provider = MockEconomicDataProvider()
        
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)
        
        events = await provider.get_economic_calendar(
            start_date, end_date, currencies=[Currency.USD]
        )
        
        assert all(e.currency == Currency.USD for e in events)
    
    @pytest.mark.asyncio
    async def test_mock_news_provider(self):
        """Test mock news data provider."""
        provider = MockNewsProvider()
        
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)
        
        news = await provider.get_news_feed(start_date, end_date)
        
        assert len(news) > 0
        assert all(isinstance(n, NewsEvent) for n in news)
        assert all(start_date <= n.timestamp <= end_date for n in news)


class TestHighPerformanceMacroEngine:
    """Test the main macro engine."""
    
    @pytest.fixture
    def macro_engine(self):
        """Create macro engine for testing."""
        logger = structlog.get_logger("test")
        return HighPerformanceMacroEngine(logger=logger)
    
    @pytest.mark.asyncio
    async def test_macro_engine_initialization(self, macro_engine):
        """Test macro engine initialization."""
        assert macro_engine.economic_loader is not None
        assert macro_engine.news_parser is not None
        assert macro_engine.surprise_window == 252
        assert macro_engine.sentiment_window == 30
    
    @pytest.mark.asyncio
    async def test_update_economic_event(self, macro_engine):
        """Test updating with economic event."""
        event = EconomicEvent(
            event_id="test_event",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.USD,
            event_type=EventType.EMPLOYMENT,
            name="Non-Farm Payrolls",
            impact=ImpactLevel.HIGH,
            actual=250.0,
            forecast=200.0,
            previous=180.0,
            unit="K",
            source="test"
        )
        
        await macro_engine.update_economic_event(event)
        
        # Check that event was stored
        assert len(macro_engine.economic_events[Currency.USD]) == 1
        assert len(macro_engine.surprise_history[Currency.USD]) == 1
        
        # Check surprise calculation
        surprise_record = macro_engine.surprise_history[Currency.USD][0]
        assert surprise_record['surprise_score'] is not None
        assert surprise_record['event_type'] == EventType.EMPLOYMENT
    
    @pytest.mark.asyncio
    async def test_update_news_event(self, macro_engine):
        """Test updating with news event."""
        news = NewsEvent(
            news_id="test_news",
            timestamp=datetime.now(timezone.utc),
            headline="Fed Cuts Rates",
            content="The Federal Reserve cut interest rates by 50 basis points.",
            source="Reuters",
            currencies_mentioned=[Currency.USD, Currency.EUR],
            sentiment_score=-0.3,
            confidence=0.8,
            relevance=0.9,
            impact_estimate=0.7
        )
        
        await macro_engine.update_news_event(news)
        
        # Check that news was stored for both currencies
        assert len(macro_engine.news_events[Currency.USD]) == 1
        assert len(macro_engine.news_events[Currency.EUR]) == 1
        assert len(macro_engine.sentiment_history[Currency.USD]) == 1
        assert len(macro_engine.sentiment_history[Currency.EUR]) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_surprise_score(self, macro_engine):
        """Test surprise score calculation."""
        # Test with no historical data
        event = EconomicEvent(
            event_id="test_1",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.USD,
            event_type=EventType.EMPLOYMENT,
            name="Non-Farm Payrolls",
            impact=ImpactLevel.HIGH,
            actual=250.0,
            forecast=200.0,
            unit="K",
            source="test"
        )
        
        score = await macro_engine.calculate_surprise_score(event)
        
        assert isinstance(score, float)
        assert -5.0 <= score <= 5.0  # Should be clamped
    
    @pytest.mark.asyncio
    async def test_calculate_surprise_score_with_history(self, macro_engine):
        """Test surprise score calculation with historical data."""
        # Add some historical events first
        for i in range(10):
            historical_event = EconomicEvent(
                event_id=f"hist_{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(days=i),
                currency=Currency.USD,
                event_type=EventType.EMPLOYMENT,
                name="Non-Farm Payrolls",
                impact=ImpactLevel.HIGH,
                actual=200.0 + i * 5,  # Varying actuals
                forecast=200.0,
                unit="K",
                source="test"
            )
            await macro_engine.update_economic_event(historical_event)
        
        # Now test with new event
        new_event = EconomicEvent(
            event_id="new_event",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.USD,
            event_type=EventType.EMPLOYMENT,
            name="Non-Farm Payrolls",
            impact=ImpactLevel.HIGH,
            actual=300.0,  # Large surprise
            forecast=200.0,
            unit="K",
            source="test"
        )
        
        score = await macro_engine.calculate_surprise_score(new_event)
        
        # Should use z-score normalization now
        assert isinstance(score, float)
        assert score != 0.0  # Should have non-zero score for large surprise
    
    @pytest.mark.asyncio
    async def test_get_latest_macro_vector(self, macro_engine):
        """Test getting latest macro feature vector."""
        # Add some test data
        economic_event = EconomicEvent(
            event_id="test_econ",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.USD,
            event_type=EventType.EMPLOYMENT,
            name="NFP",
            impact=ImpactLevel.HIGH,
            actual=250.0,
            forecast=200.0,
            unit="K",
            source="test"
        )
        
        news_event = NewsEvent(
            news_id="test_news",
            timestamp=datetime.now(timezone.utc),
            headline="Positive Economic News",
            content="Economy shows strong growth",
            source="Reuters",
            currencies_mentioned=[Currency.USD],
            sentiment_score=0.5,
            confidence=0.8,
            impact_estimate=0.7
        )
        
        await macro_engine.update_economic_event(economic_event)
        await macro_engine.update_news_event(news_event)
        
        # Get macro vector
        vector = await macro_engine.get_latest_macro_vector()
        
        assert isinstance(vector, MacroFeatureVector)
        assert Currency.USD in vector.currency_surprises
        assert Currency.USD in vector.sentiment_scores
        assert vector.sentiment_scores[Currency.USD] != 0.0  # Should have sentiment
        
        # Test caching
        vector2 = await macro_engine.get_latest_macro_vector()
        assert vector.timestamp == vector2.timestamp  # Should use cache
    
    @pytest.mark.asyncio
    async def test_backfill_historical_data(self, macro_engine):
        """Test backfilling historical data."""
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)
        
        await macro_engine.backfill_historical_data(
            start_date, end_date, currencies=[Currency.USD, Currency.EUR]
        )
        
        # Should have loaded some economic events
        assert len(macro_engine.economic_events[Currency.USD]) > 0
        assert len(macro_engine.news_events[Currency.USD]) > 0
    
    @pytest.mark.asyncio
    async def test_get_currency_correlations(self, macro_engine):
        """Test currency correlation calculation."""
        # Add data for multiple currencies
        currencies = [Currency.USD, Currency.EUR, Currency.GBP]
        
        for i in range(20):  # Add 20 events for each currency
            for currency in currencies:
                event = EconomicEvent(
                    event_id=f"test_{currency.value}_{i}",
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                    currency=currency,
                    event_type=EventType.EMPLOYMENT,
                    name="Test Event",
                    impact=ImpactLevel.MEDIUM,
                    actual=200.0 + i,
                    forecast=200.0,
                    unit="K",
                    source="test"
                )
                await macro_engine.update_economic_event(event)
        
        correlations = await macro_engine.get_currency_correlations()
        
        assert isinstance(correlations, pd.DataFrame)
        if not correlations.empty:
            assert correlations.shape[0] > 0
            assert correlations.shape[1] > 0
    
    @pytest.mark.asyncio
    async def test_get_feature_importance(self, macro_engine):
        """Test feature importance calculation."""
        # Add some varied data
        for i in range(10):
            event = EconomicEvent(
                event_id=f"test_{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                currency=Currency.USD,
                event_type=EventType.EMPLOYMENT,
                name="Test Event",
                impact=ImpactLevel.MEDIUM,
                actual=200.0 + i * 10,  # Varying surprises
                forecast=200.0,
                unit="K",
                source="test"
            )
            await macro_engine.update_economic_event(event)
        
        importance = await macro_engine.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert "USD_surprise" in importance
        assert all(0 <= v <= 1 for v in importance.values())  # Normalized scores
    
    @pytest.mark.asyncio
    async def test_get_live_updates(self, macro_engine):
        """Test getting live updates."""
        live_economic, live_news = await macro_engine.get_live_updates()
        
        assert isinstance(live_economic, list)
        assert isinstance(live_news, list)
        # Live data might be empty in mock providers, that's OK
    
    @pytest.mark.asyncio
    async def test_macro_engine_cleanup(self, macro_engine):
        """Test proper cleanup of macro engine."""
        await macro_engine.close()
        # Should not raise any exceptions


class TestMacroEngineIntegration:
    """Integration tests for the macro engine."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create engine
        engine = create_macro_engine(use_mock_data=True)
        
        try:
            # Backfill some historical data
            start_date = datetime.now(timezone.utc) - timedelta(days=3)
            end_date = datetime.now(timezone.utc)
            
            await engine.backfill_historical_data(start_date, end_date)
            
            # Get initial macro vector
            vector1 = await engine.get_latest_macro_vector()
            assert isinstance(vector1, MacroFeatureVector)
            
            # Add a new high-impact economic event
            major_event = EconomicEvent(
                event_id="major_surprise",
                timestamp=datetime.now(timezone.utc),
                currency=Currency.USD,
                event_type=EventType.EMPLOYMENT,
                name="Non-Farm Payrolls",
                impact=ImpactLevel.HIGH,
                actual=500.0,  # Major positive surprise
                forecast=200.0,
                unit="K",
                source="test"
            )
            
            await engine.update_economic_event(major_event)
            
            # Add corresponding news
            major_news = NewsEvent(
                news_id="major_news",
                timestamp=datetime.now(timezone.utc),
                headline="Jobs Surge Beats All Expectations",
                content="Employment data shows massive job creation, beating all forecasts",
                source="Reuters",
                currencies_mentioned=[Currency.USD],
                sentiment_score=0.8,  # Very positive
                confidence=0.9,
                impact_estimate=0.9
            )
            
            await engine.update_news_event(major_news)
            
            # Get updated macro vector
            vector2 = await engine.get_latest_macro_vector()
            
            # Should show impact of new events
            assert vector2.currency_surprises[Currency.USD] != vector1.currency_surprises[Currency.USD]
            assert vector2.sentiment_scores[Currency.USD] > 0  # Positive sentiment
            assert vector2.high_impact_flags[Currency.USD] is True
            
            # Convert to feature dictionary for ML use
            features = vector2.to_dict()
            assert isinstance(features, dict)
            assert len(features) > 0
            assert all(isinstance(v, (int, float)) for v in features.values())
            
        finally:
            await engine.close()
    
    @pytest.mark.asyncio
    async def test_multiple_currency_interactions(self):
        """Test macro engine with multiple currency interactions."""
        engine = create_macro_engine(use_mock_data=True)
        
        try:
            # Add events affecting multiple currencies
            ecb_event = EconomicEvent(
                event_id="ecb_rates",
                timestamp=datetime.now(timezone.utc),
                currency=Currency.EUR,
                event_type=EventType.INTEREST_RATE,
                name="ECB Rate Decision",
                impact=ImpactLevel.HIGH,
                actual=4.5,
                forecast=4.0,
                unit="%",
                source="ECB"
            )
            
            cross_currency_news = NewsEvent(
                news_id="cross_news",
                timestamp=datetime.now(timezone.utc),
                headline="ECB Rate Hike Affects Global Markets",
                content="European Central Bank raises rates, impacting USD and GBP",
                source="Bloomberg",
                currencies_mentioned=[Currency.EUR, Currency.USD, Currency.GBP],
                sentiment_score=-0.2,
                confidence=0.7,
                impact_estimate=0.8
            )
            
            await engine.update_economic_event(ecb_event)
            await engine.update_news_event(cross_currency_news)
            
            vector = await engine.get_latest_macro_vector()
            
            # Check that multiple currencies are affected
            assert vector.currency_surprises[Currency.EUR] != 0
            assert vector.sentiment_scores[Currency.EUR] != 0
            assert vector.sentiment_scores[Currency.USD] != 0
            assert vector.sentiment_scores[Currency.GBP] != 0
            
        finally:
            await engine.close()


class TestMacroEnginePerformance:
    """Performance tests for the macro engine."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test macro engine performance with large datasets."""
        engine = create_macro_engine(use_mock_data=True)
        
        try:
            import time
            start_time = time.time()
            
            # Add 1000 economic events
            for i in range(1000):
                event = EconomicEvent(
                    event_id=f"perf_test_{i}",
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                    currency=Currency.USD if i % 2 == 0 else Currency.EUR,
                    event_type=EventType.EMPLOYMENT,
                    name="Performance Test Event",
                    impact=ImpactLevel.MEDIUM,
                    actual=200.0 + (i % 100),
                    forecast=200.0,
                    unit="K",
                    source="test"
                )
                await engine.update_economic_event(event)
            
            elapsed = time.time() - start_time
            
            # Should process 1000 events reasonably quickly
            events_per_second = 1000 / elapsed
            print(f"Processed {events_per_second:.0f} events/second")
            
            assert events_per_second > 50  # Should handle at least 50 events/second
            
            # Test feature vector generation performance
            start_time = time.time()
            vector = await engine.get_latest_macro_vector()
            vector_time = time.time() - start_time
            
            assert vector_time < 1.0  # Should generate vector in under 1 second
            assert isinstance(vector, MacroFeatureVector)
            
        finally:
            await engine.close()


class TestMacroEngineFactory:
    """Test the macro engine factory function."""
    
    def test_create_macro_engine_mock(self):
        """Test creating macro engine with mock data."""
        engine = create_macro_engine(use_mock_data=True)
        
        assert isinstance(engine, HighPerformanceMacroEngine)
        assert engine.surprise_window == 252
        assert engine.sentiment_window == 30
        
        # Clean up
        asyncio.run(engine.close())
    
    def test_create_macro_engine_custom_params(self):
        """Test creating macro engine with custom parameters."""
        logger = structlog.get_logger("test")
        
        engine = create_macro_engine(
            use_mock_data=True,
            surprise_window=100,
            sentiment_window=15,
            logger=logger
        )
        
        assert engine.surprise_window == 100
        assert engine.sentiment_window == 15
        
        # Clean up
        asyncio.run(engine.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 