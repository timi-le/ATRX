"""
Feature Engine Demonstration Script

This script demonstrates the capabilities of the High-Performance Feature Engine
for FX AI-Quant Trading System.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import structlog

from core.feature_engine import (
    FeatureConfig,
    HighPerformanceFeatureEngine,
)
from core.interfaces.data_interfaces import OHLCV


def generate_sample_fx_data(
    symbol: str = "EUR/USD", periods: int = 1000
) -> list[OHLCV]:
    """Generate realistic sample FX data for demonstration."""

    # Start with a base price
    base_price = 1.1000

    # Generate timestamps
    start_time = datetime.now(timezone.utc) - timedelta(hours=periods)
    timestamps = [start_time + timedelta(hours=i) for i in range(periods)]

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results

    data = []
    current_price = base_price

    for i, timestamp in enumerate(timestamps):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i * 0.01)  # Slow trend
        volatility = np.random.normal(0, 0.001)  # Random volatility

        # Price change
        price_change = trend + volatility
        current_price *= 1 + price_change

        # Generate OHLC from current price
        open_price = current_price
        close_price = current_price * (1 + np.random.normal(0, 0.0005))
        high_price = max(open_price, close_price) * (
            1 + abs(np.random.normal(0, 0.0003))
        )
        low_price = min(open_price, close_price) * (
            1 - abs(np.random.normal(0, 0.0003))
        )
        volume = np.random.uniform(100000, 2000000)

        ohlcv = OHLCV(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timeframe="1h",
        )

        data.append(ohlcv)
        current_price = close_price  # Update for next iteration

    return data


async def demonstrate_feature_engine():
    """Demonstrate the feature engine capabilities."""

    print("🚀 FX AI-Quant Trading System - Feature Engine Demo")
    print("=" * 60)

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("demo")

    # Create feature engine with custom configuration
    config = FeatureConfig(
        momentum_periods=[5, 10, 20],
        volatility_windows=[10, 20, 50],
        bollinger_periods=[20],
        max_workers=4,
    )

    engine = HighPerformanceFeatureEngine(config=config, logger=logger)

    print(f"\n📊 Available Features: {', '.join(engine.get_available_features())}")
    print(f"🔧 Buffer Size: {engine.max_buffer_size}")

    # Generate sample data
    print("\n📈 Generating sample EUR/USD data...")
    sample_data = generate_sample_fx_data("EUR/USD", 200)

    # Demonstrate streaming feature computation
    print("\n⚡ Streaming Feature Computation Demo:")
    print("-" * 40)

    features_history = []

    # Process data in streaming fashion
    for i, bar in enumerate(sample_data):
        features = engine.update(bar)

        if i % 50 == 0 and features:  # Print every 50th update
            print(f"Bar {i:3d}: {len(features)} features computed")
            if i == 100:  # Show detailed features at bar 100
                print("  Sample features:")
                for key, value in list(features.items())[:10]:  # Show first 10 features
                    print(f"    {key}: {value:.6f}")

        if features:
            features_history.append(
                {"timestamp": bar.timestamp, "close": bar.close, **features}
            )

    print(f"\n✅ Processed {len(sample_data)} bars successfully!")

    # Demonstrate batch computation
    print("\n🔄 Batch Feature Computation Demo:")
    print("-" * 40)

    # Convert sample data to DataFrame
    df_data = pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in sample_data[-100:]  # Last 100 bars
        ]
    )

    # Compute technical features
    print("Computing technical indicators...")
    technical_features = await engine.compute_technical_features(df_data)
    print(f"Technical features shape: {technical_features.shape}")

    # Compute volatility features
    print("Computing volatility features...")
    volatility_features = await engine.compute_volatility_features(df_data, [10, 20])
    print(f"Volatility features shape: {volatility_features.shape}")

    # Compute momentum features
    print("Computing momentum features...")
    momentum_features = await engine.compute_momentum_features(df_data, [5, 10])
    print(f"Momentum features shape: {momentum_features.shape}")

    # Show feature statistics
    print("\n📊 Feature Statistics:")
    print("-" * 40)

    if features_history:
        features_df = pd.DataFrame(features_history[-50:])  # Last 50 records

        # Show some key features
        key_features = ["sma_20", "rsi_14", "macd_line", "volatility_20", "momentum_10"]
        available_features = [f for f in key_features if f in features_df.columns]

        if available_features:
            stats = features_df[available_features].describe()
            print(stats.round(6))

    # Demonstrate custom feature addition
    print("\n🔧 Custom Feature Demo:")
    print("-" * 40)

    def custom_price_velocity(data):
        """Custom feature: price velocity (rate of change of rate of change)."""
        close_prices = data["close"].values
        if len(close_prices) < 3:
            return {"price_velocity": np.array([])}

        # First derivative (rate of change)
        roc = np.diff(close_prices)
        # Second derivative (acceleration)
        velocity = np.diff(roc)

        # Pad to match input length
        velocity_padded = np.full(len(close_prices), np.nan)
        velocity_padded[2:] = velocity

        return {"price_velocity": velocity_padded}

    # Add custom feature
    engine.add_custom_feature("velocity", custom_price_velocity)
    print(
        f"Added custom feature. Available features: {len(engine.get_available_features())}"
    )

    # Test custom feature
    test_bar = sample_data[-1]
    features_with_custom = engine.update(test_bar)

    if "price_velocity" in features_with_custom:
        print(f"Custom price velocity: {features_with_custom['price_velocity']:.8f}")

    # Performance demonstration
    print("\n⚡ Performance Test:")
    print("-" * 40)

    import time

    # Test streaming performance
    start_time = time.time()
    test_data = generate_sample_fx_data("GBP/USD", 1000)

    for bar in test_data:
        engine.update(bar)

    elapsed = time.time() - start_time
    throughput = len(test_data) / elapsed

    print(f"Processed {len(test_data)} bars in {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.0f} bars/second")

    # Cleanup
    engine.close()
    print("\n✅ Demo completed successfully!")


def main():
    """Run the feature engine demonstration."""
    try:
        asyncio.run(demonstrate_feature_engine())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
