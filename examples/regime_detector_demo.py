"""
Macro Economic Feature Engine Demonstration.

This script demonstrates how to use the macro economic feature engine
to process economic events and news sentiment for trading signals.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import structlog

from core.interfaces.macro_interfaces import (
    Currency,
    EconomicEvent,
    EventType,
    ImpactLevel,
    NewsEvent,
)
from core.macro_engine import create_macro_engine


async def demo_macro_engine():
    """Demonstrate the macro economic feature engine capabilities."""

    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("macro_demo")
    logger.info("Starting Macro Economic Feature Engine Demo")

    # Create macro engine with mock data providers
    engine = create_macro_engine(use_mock_data=True, logger=logger)

    try:
        # 1. Backfill some historical data
        logger.info("Backfilling historical economic and news data...")
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        await engine.backfill_historical_data(start_date, end_date)

        # 2. Get initial macro feature vector
        logger.info("Getting initial macro feature vector...")
        initial_vector = await engine.get_latest_macro_vector()

        print("\n=== Initial Macro Features ===")
        features = initial_vector.to_dict()
        for key, value in features.items():
            if abs(value) > 0.01:  # Only show non-zero features
                print(f"{key}: {value:.4f}")

        # 3. Add a major economic surprise
        logger.info("Adding major economic surprise event...")
        major_surprise = EconomicEvent(
            event_id="demo_nfp_surprise",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.USD,
            event_type=EventType.EMPLOYMENT,
            name="Non-Farm Payrolls",
            impact=ImpactLevel.HIGH,
            actual=450.0,  # Much higher than forecast
            forecast=200.0,  # Expected
            previous=180.0,
            unit="K",
            source="demo",
        )

        await engine.update_economic_event(major_surprise)

        print(f"\n=== Major Economic Surprise Added ===")
        print(f"Event: {major_surprise.name}")
        print(f"Currency: {major_surprise.currency.value}")
        print(f"Forecast: {major_surprise.forecast}")
        print(f"Actual: {major_surprise.actual}")
        print(f"Surprise: {major_surprise.surprise}")
        print(f"Surprise %: {major_surprise.surprise_pct:.1f}%")

        # 4. Add corresponding positive news
        logger.info("Adding positive news event...")
        positive_news = NewsEvent(
            news_id="demo_positive_jobs",
            timestamp=datetime.now(timezone.utc),
            headline="Jobs Report Beats Expectations by Massive Margin",
            content="The latest employment data shows exceptional job creation, "
            "with non-farm payrolls surging well above forecasts, "
            "signaling robust economic strength and growth momentum.",
            source="Demo News",
            currencies_mentioned=[Currency.USD],
            sentiment_score=0.8,  # Very positive
            confidence=0.9,
            relevance=1.0,
            impact_estimate=0.8,
        )

        await engine.update_news_event(positive_news)

        print(f"\n=== Positive News Added ===")
        print(f"Headline: {positive_news.headline}")
        print(f"Sentiment Score: {positive_news.sentiment_score}")
        print(f"Confidence: {positive_news.confidence}")

        # 5. Get updated macro vector
        logger.info("Getting updated macro feature vector...")
        updated_vector = await engine.get_latest_macro_vector()

        print(f"\n=== Updated Macro Features (After Major Events) ===")
        updated_features = updated_vector.to_dict()
        for key, value in updated_features.items():
            if abs(value) > 0.01:  # Only show non-zero features
                change = value - features.get(key, 0)
                change_str = f" (Δ{change:+.4f})" if abs(change) > 0.001 else ""
                print(f"{key}: {value:.4f}{change_str}")

        # 6. Demonstrate currency correlations
        logger.info("Calculating currency correlations...")
        correlations = await engine.get_currency_correlations()

        if not correlations.empty:
            print(f"\n=== Currency Surprise Correlations ===")
            print(correlations.round(3))

        # 7. Show feature importance
        logger.info("Calculating feature importance...")
        importance = await engine.get_feature_importance()

        print(f"\n=== Feature Importance (Top 10) ===")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_importance[:10]:
            if score > 0.01:
                print(f"{feature}: {score:.4f}")

        # 8. Add a negative event for EUR
        logger.info("Adding negative European event for contrast...")
        eur_negative = EconomicEvent(
            event_id="demo_eur_inflation",
            timestamp=datetime.now(timezone.utc),
            currency=Currency.EUR,
            event_type=EventType.INFLATION,
            name="Eurozone CPI",
            impact=ImpactLevel.HIGH,
            actual=4.8,  # Higher than expected inflation
            forecast=4.2,  # Expected
            previous=4.1,
            unit="%",
            source="demo",
        )

        await engine.update_economic_event(eur_negative)

        eur_negative_news = NewsEvent(
            news_id="demo_eur_inflation_concern",
            timestamp=datetime.now(timezone.utc),
            headline="Eurozone Inflation Accelerates Above Forecasts",
            content="European inflation data shows concerning acceleration, "
            "raising questions about ECB policy effectiveness and "
            "economic stability in the region.",
            source="Demo News",
            currencies_mentioned=[Currency.EUR],
            sentiment_score=-0.6,  # Negative
            confidence=0.8,
            relevance=0.9,
            impact_estimate=0.7,
        )

        await engine.update_news_event(eur_negative_news)

        # 9. Get final macro vector showing multi-currency impact
        final_vector = await engine.get_latest_macro_vector()

        print(f"\n=== Final Macro Features (Multi-Currency Impact) ===")
        final_features = final_vector.to_dict()

        # Group by currency for better readability
        currency_features = {}
        for key, value in final_features.items():
            if abs(value) > 0.01:
                parts = key.split("_", 1)
                if len(parts) == 2:
                    curr, feature_type = parts
                    if curr not in currency_features:
                        currency_features[curr] = {}
                    currency_features[curr][feature_type] = value

        for currency, features in currency_features.items():
            print(f"\n{currency}:")
            for feature_type, value in features.items():
                print(f"  {feature_type}: {value:.4f}")

        # 10. Show high impact events summary
        print(f"\n=== High Impact Events Summary ===")
        for currency in Currency:
            if final_vector.high_impact_flags[currency]:
                surprise = final_vector.currency_surprises[currency]
                sentiment = final_vector.sentiment_scores[currency]
                events = final_vector.event_counts[currency]
                print(
                    f"{currency.value}: Surprise={surprise:.3f}, "
                    f"Sentiment={sentiment:.3f}, Events={events}"
                )

        logger.info("Macro engine demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error in macro engine demo: {e}")
        raise
    finally:
        await engine.close()


async def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities."""

    from data.news_sentiment_parser import SimpleSentimentAnalyzer

    analyzer = SimpleSentimentAnalyzer()

    print("\n=== Sentiment Analysis Demo ===")

    test_headlines = [
        "Federal Reserve cuts interest rates to support economic growth",
        "GDP growth exceeds expectations with strong consumer spending",
        "Market crashes as inflation concerns mount",
        "Central bank maintains hawkish stance amid uncertainty",
        "Employment data shows robust job creation and wage growth",
        "Economic indicators suggest potential recession ahead",
    ]

    for headline in test_headlines:
        sentiment, confidence = await analyzer.analyze_sentiment(headline)
        currencies = analyzer.extract_currencies(headline)

        sentiment_label = (
            "Positive"
            if sentiment > 0.1
            else "Negative"
            if sentiment < -0.1
            else "Neutral"
        )
        currencies_str = (
            ", ".join([c.value for c in currencies]) if currencies else "None"
        )

        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {sentiment:.3f} ({sentiment_label})")
        print(f"Confidence: {confidence:.3f}")
        print(f"Currencies: {currencies_str}")


if __name__ == "__main__":
    print("FX AI-Quant Trading System - Macro Economic Feature Engine Demo")
    print("=" * 70)

    # Run the main demo
    asyncio.run(demo_macro_engine())

    # Run sentiment analysis demo
    asyncio.run(demo_sentiment_analysis())

    print("\nDemo completed successfully!")
