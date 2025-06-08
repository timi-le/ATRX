# Macro Economic Feature Engine Guide

The Macro Economic Feature Engine is a sophisticated component of the FX AI-Quant Trading System that processes economic surprises and news sentiment to create unified feature vectors for machine learning models and regime detection.

## Overview

The engine combines two key data sources:
1. **Economic Calendar Events** - Official economic data releases with surprise calculations
2. **News Sentiment** - Financial news articles with sentiment analysis and currency extraction

## Key Features

### Economic Surprise Processing
- **Z-score normalization** using historical surprise data
- **Event type weighting** (GDP, inflation, employment, etc.)
- **Impact level weighting** (high, medium, low impact events)
- **Real-time surprise calculation** as actual values are released

### News Sentiment Analysis
- **Rule-based sentiment analysis** with financial keyword dictionaries
- **Currency extraction** from news content using pattern matching
- **Confidence scoring** based on sentiment word density
- **Impact estimation** based on sentiment strength and relevance

### Feature Vector Generation
- **Multi-currency features** for all major FX pairs
- **Rolling statistics** (means, standard deviations)
- **Event counts** and high-impact flags
- **ML-ready feature dictionaries** with consistent naming

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Economic Data   │    │ News Providers   │    │ Sentiment       │
│ Providers       │    │                  │    │ Analyzer        │
│                 │    │ - RSS Feeds      │    │                 │
│ - Trading Econ  │────│ - NewsAPI        │────│ - Rule-based    │
│ - Forex Factory │    │ - Mock Provider  │    │ - Keyword Dict  │
│ - Mock Provider │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                            ┌─────▼─────┐
                            │ Macro     │
                            │ Engine    │
                            │           │
                            │ - Combine │
                            │ - Analyze │
                            │ - Generate│
                            └─────┬─────┘
                                  │
                         ┌────────▼─────────┐
                         │ Feature Vector   │
                         │                  │
                         │ - Surprises      │
                         │ - Sentiments     │
                         │ - Rolling Stats  │
                         │ - Event Counts   │
                         └──────────────────┘
```

## Usage Examples

### Basic Usage

```python
from core.macro_engine import create_macro_engine
from datetime import datetime, timezone, timedelta

# Create engine with mock data for testing
engine = create_macro_engine(use_mock_data=True)

try:
    # Backfill historical data
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    await engine.backfill_historical_data(start_date, end_date)

    # Get latest macro features
    macro_vector = await engine.get_latest_macro_vector()
    features = macro_vector.to_dict()

    # Use features in ML models
    for feature_name, value in features.items():
        print(f"{feature_name}: {value}")

finally:
    await engine.close()
```

### Adding Custom Economic Events

```python
from core.interfaces.macro_interfaces import EconomicEvent, Currency, EventType, ImpactLevel

# Create a major economic surprise
nfp_surprise = EconomicEvent(
    event_id="nfp_2024_01",
    timestamp=datetime.now(timezone.utc),
    currency=Currency.USD,
    event_type=EventType.EMPLOYMENT,
    name="Non-Farm Payrolls",
    impact=ImpactLevel.HIGH,
    actual=350.0,    # Actual value released
    forecast=200.0,  # Market expectation
    previous=180.0,  # Previous month's value
    unit="K",
    source="BLS"
)

await engine.update_economic_event(nfp_surprise)
```

### Adding News Events

```python
from core.interfaces.macro_interfaces import NewsEvent

# Add news with sentiment
fed_news = NewsEvent(
    news_id="fed_meeting_2024",
    timestamp=datetime.now(timezone.utc),
    headline="Fed Signals Dovish Pivot Amid Economic Concerns",
    content="Federal Reserve officials indicated potential policy easing...",
    source="Reuters",
    currencies_mentioned=[Currency.USD],
    sentiment_score=-0.3,  # Slightly negative
    confidence=0.8,
    relevance=0.9,
    impact_estimate=0.7
)

await engine.update_news_event(fed_news)
```

## Feature Vector Structure

The macro feature vector contains the following components for each currency:

### Core Features
- `{CURRENCY}_surprise` - Latest surprise score (z-score normalized)
- `{CURRENCY}_sentiment` - Weighted average sentiment from recent news
- `{CURRENCY}_surprise_mean` - Rolling mean of surprise scores
- `{CURRENCY}_surprise_std` - Rolling standard deviation of surprises
- `{CURRENCY}_event_count` - Number of recent events (24h window)
- `{CURRENCY}_high_impact` - Boolean flag for high-impact events

### Example Feature Names
```
USD_surprise: 0.5234
USD_sentiment: 0.2156
USD_surprise_mean: 0.1234
USD_surprise_std: 0.8765
USD_event_count: 3.0000
USD_high_impact: 1.0000
```

## Data Providers

### Economic Data Providers

#### MockEconomicDataProvider
- **Purpose**: Testing and development
- **Features**: Generates realistic mock economic events
- **Usage**: Default provider for testing

#### TradingEconomicsProvider (Planned)
- **Purpose**: Real economic data via API
- **Features**: Comprehensive economic calendar
- **Requirements**: API key needed

#### ForexFactoryDataProvider (Planned)
- **Purpose**: Popular economic calendar
- **Features**: Web scraping based
- **Requirements**: Rate limiting considerations

### News Providers

#### MockNewsProvider
- **Purpose**: Testing and development
- **Features**: Realistic mock news with sentiment
- **Usage**: Default for testing

#### RSSNewsProvider
- **Purpose**: Real-time news via RSS feeds
- **Features**: Multiple financial news sources
- **Configuration**: Customizable RSS feed list

#### NewsAPIProvider (Planned)
- **Purpose**: Professional news API
- **Features**: Filtered financial news
- **Requirements**: API key needed

## Configuration

### Engine Parameters
```python
engine = HighPerformanceMacroEngine(
    surprise_window=252,      # Days of surprise history
    sentiment_window=30,      # Days of sentiment history
    economic_providers=[...], # List of economic data providers
    news_providers=[...],     # List of news providers
    sentiment_analyzer=...,   # Custom sentiment analyzer
    logger=...               # Structured logger
)
```

### Event Type Weights
The engine applies different weights to event types based on market impact:

- **GDP**: 1.0 (highest impact)
- **Interest Rates**: 1.0 (highest impact)
- **Inflation (CPI)**: 0.9 (very high impact)
- **Central Bank Decisions**: 0.9 (very high impact)
- **Employment**: 0.8 (high impact)
- **PMI**: 0.6 (medium impact)
- **Retail Sales**: 0.5 (medium impact)
- **Trade Balance**: 0.4 (lower impact)
- **Consumer Confidence**: 0.4 (lower impact)
- **Manufacturing**: 0.5 (medium impact)

### Impact Level Weights
- **High Impact**: 1.0
- **Medium Impact**: 0.6
- **Low Impact**: 0.3

## Performance Considerations

### Memory Usage
- **Rolling Windows**: Configurable history length
- **Deque Storage**: Efficient memory management
- **Cache Management**: 5-minute feature vector cache

### Processing Speed
- **Async Operations**: Non-blocking data processing
- **Batch Processing**: Efficient bulk operations
- **Z-score Caching**: Pre-calculated statistical measures

### Scalability
- **Provider Abstraction**: Easy to add new data sources
- **Multi-currency Support**: Scales to all major currencies
- **Event Deduplication**: Handles overlapping data sources

## Testing

The engine includes comprehensive unit tests covering:

### Core Functionality Tests
- Economic event processing
- News sentiment analysis
- Feature vector generation
- Surprise score calculations

### Integration Tests
- Multi-provider data aggregation
- Cross-currency event impacts
- End-to-end workflows

### Performance Tests
- Large dataset processing
- Concurrent event handling
- Memory usage optimization

Run tests with:
```bash
python -m pytest tests/unit/test_macro_engine.py -v
```

## Monitoring and Debugging

### Structured Logging
The engine uses structured logging for comprehensive monitoring:

```python
import structlog

logger = structlog.get_logger("macro_engine")
engine = create_macro_engine(logger=logger)
```

### Key Metrics to Monitor
- **Event Processing Rate**: Events per second
- **Feature Generation Latency**: Time to create feature vectors
- **Provider Health**: Data source availability
- **Memory Usage**: Historical data storage
- **Cache Hit Rate**: Feature vector cache efficiency

### Common Issues and Solutions

#### Missing API Keys
```
Error: No API key provided for Trading Economics
Solution: Set TRADING_ECONOMICS_API_KEY in environment
```

#### Network Connectivity
```
Error: Failed to fetch RSS feed
Solution: Check network connectivity and RSS URL validity
```

#### Memory Usage
```
Issue: High memory usage with large surprise_window
Solution: Reduce window size or implement data compression
```

## Integration with Trading System

### ML Model Integration
```python
# Get features for ML model
macro_vector = await engine.get_latest_macro_vector()
features = macro_vector.to_dict()

# Combine with technical features
all_features = {**technical_features, **features}

# Feed to ML model
prediction = model.predict(all_features)
```

### Regime Detection
```python
# Use macro features for regime detection
if features['USD_high_impact'] and abs(features['USD_surprise']) > 2.0:
    regime = "high_volatility"
elif features['USD_sentiment'] > 0.5:
    regime = "risk_on"
else:
    regime = "normal"
```

### Risk Management
```python
# Adjust position sizing based on macro events
if any(features[f'{curr}_high_impact'] for curr in major_currencies):
    position_multiplier = 0.5  # Reduce size during major events
else:
    position_multiplier = 1.0
```

## Future Enhancements

### Planned Features
- **Machine Learning Sentiment**: Replace rule-based analyzer
- **Event Impact Modeling**: Predictive event impact scoring
- **Cross-Asset Analysis**: Correlations with bonds, equities
- **Real-time Streaming**: WebSocket-based live data
- **Alternative Data**: Social media sentiment, satellite data

### API Integrations
- **Bloomberg Terminal**: Professional data feed
- **Refinitiv Eikon**: Comprehensive market data
- **Twitter API**: Social sentiment analysis
- **Central Bank APIs**: Direct access to official data

## Conclusion

The Macro Economic Feature Engine provides a robust foundation for incorporating fundamental analysis into quantitative trading strategies. Its modular design allows for easy extension and customization while maintaining high performance and reliability.

For more examples and advanced usage, see the `examples/macro_engine_demo.py` script.
