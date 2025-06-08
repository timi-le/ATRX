"""
Unit tests for the High-Performance Feature Engine.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import structlog

from core.feature_engine import (
    FeatureConfig,
    HighPerformanceFeatureEngine,
    _compute_atr,
    _compute_bollinger_bands,
    _compute_bollinger_width,
    _compute_ema,
    _compute_momentum,
    _compute_rolling_entropy,
    _compute_rolling_hurst,
    _compute_rolling_kurtosis_skew,
    _compute_rolling_volatility,
    _compute_rsi,
    _compute_sma,
    _compute_spread_vs_range,
    _compute_stochastic_oscillator,
    create_feature_engine,
)
from core.interfaces.data_interfaces import OHLCV


class TestNumbaTechnicalIndicators:
    """Test the Numba-optimized technical indicator functions."""

    def test_compute_sma(self):
        """Test Simple Moving Average computation."""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        sma = _compute_sma(prices, period)

        # Check that first (period-1) values are NaN
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])

        # Check computed values
        assert sma[2] == 2.0  # (1+2+3)/3
        assert sma[3] == 3.0  # (2+3+4)/3
        assert sma[9] == 9.0  # (8+9+10)/3

    def test_compute_ema(self):
        """Test Exponential Moving Average computation."""
        prices = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        period = 3

        ema = _compute_ema(prices, period)

        # Check that first (period-1) values are NaN
        assert np.isnan(ema[0])
        assert np.isnan(ema[1])

        # First EMA should be SMA
        assert ema[2] == np.mean(prices[:3])

        # Subsequent values should follow EMA formula
        alpha = 2.0 / (period + 1)
        expected_ema_3 = alpha * prices[3] + (1 - alpha) * ema[2]
        assert abs(ema[3] - expected_ema_3) < 1e-10

    def test_compute_rsi(self):
        """Test RSI computation."""
        # Create test data with clear trend
        prices = np.array(
            [
                44.0,
                44.3,
                44.1,
                44.2,
                44.5,
                43.9,
                44.5,
                44.9,
                44.5,
                44.6,
                44.8,
                44.2,
                45.1,
                45.3,
                45.4,
                45.8,
                46.0,
                45.9,
            ]
        )
        period = 14

        rsi = _compute_rsi(prices, period)

        # Check that first period values are NaN
        for i in range(period):
            assert np.isnan(rsi[i])

        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_compute_bollinger_bands(self):
        """Test Bollinger Bands computation."""
        prices = np.array([20.0, 21.0, 22.0, 21.5, 20.5, 19.5, 20.0, 21.0, 22.5, 23.0])
        period = 5
        std_dev = 2.0

        upper, middle, lower = _compute_bollinger_bands(prices, period, std_dev)

        # Check that first (period-1) values are NaN
        for i in range(period - 1):
            assert np.isnan(upper[i])
            assert np.isnan(middle[i])
            assert np.isnan(lower[i])

        # Check that middle band is SMA
        for i in range(period - 1, len(prices)):
            window_mean = np.mean(prices[i - period + 1 : i + 1])
            assert abs(middle[i] - window_mean) < 1e-10

        # Check that bands are properly spaced
        for i in range(period - 1, len(prices)):
            if not np.isnan(upper[i]):
                assert upper[i] > middle[i]
                assert middle[i] > lower[i]

    def test_compute_atr(self):
        """Test Average True Range computation."""
        high = np.array([22.0, 23.0, 24.0, 23.5, 22.5, 21.0, 22.0, 24.0, 25.0, 24.5])
        low = np.array([20.0, 21.0, 22.0, 21.5, 20.5, 19.0, 20.0, 22.0, 23.0, 22.5])
        close = np.array([21.0, 22.0, 23.0, 22.5, 21.5, 20.0, 21.0, 23.0, 24.0, 23.5])
        period = 5

        atr = _compute_atr(high, low, close, period)

        # Check that first period values are NaN
        for i in range(period):
            assert np.isnan(atr[i])

        # ATR should be positive
        valid_atr = atr[~np.isnan(atr)]
        assert np.all(valid_atr >= 0)

    def test_compute_stochastic_oscillator(self):
        """Test Stochastic Oscillator computation."""
        high = np.array([23.0, 24.0, 25.0, 24.5, 23.5, 22.0, 23.0, 25.0, 26.0, 25.5])
        low = np.array([21.0, 22.0, 23.0, 22.5, 21.5, 20.0, 21.0, 23.0, 24.0, 23.5])
        close = np.array([22.0, 23.0, 24.0, 23.5, 22.5, 21.0, 22.0, 24.0, 25.0, 24.5])
        k_period = 5
        d_period = 3

        k_percent, d_percent = _compute_stochastic_oscillator(
            high, low, close, k_period, d_period
        )

        # Check that %K has proper range
        valid_k = k_percent[~np.isnan(k_percent)]
        assert np.all(valid_k >= 0)
        assert np.all(valid_k <= 100)

        # Check that %D has proper range
        valid_d = d_percent[~np.isnan(d_percent)]
        assert np.all(valid_d >= 0)
        assert np.all(valid_d <= 100)

    def test_compute_rolling_volatility(self):
        """Test rolling volatility computation."""
        # Create returns with known volatility pattern
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        window = 20

        vol = _compute_rolling_volatility(returns, window)

        # Check that first (window-1) values are NaN
        for i in range(window - 1):
            assert np.isnan(vol[i])

        # Volatility should be positive
        valid_vol = vol[~np.isnan(vol)]
        assert np.all(valid_vol >= 0)

    def test_compute_momentum(self):
        """Test momentum computation."""
        prices = np.array(
            [100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 109.0]
        )
        period = 3

        momentum = _compute_momentum(prices, period)

        # Check that first period values are NaN
        for i in range(period):
            assert np.isnan(momentum[i])

        # Check specific calculation
        expected = (103.0 / 100.0 - 1.0) * 100.0  # 3% momentum
        assert abs(momentum[3] - expected) < 1e-10


class TestStatisticalFeatures:
    """Test the statistical feature functions."""

    def test_compute_rolling_hurst(self):
        """Test rolling Hurst Exponent computation."""
        # A perfect random walk should have a Hurst exponent of 0.5
        np.random.seed(42)
        prices = np.random.randn(200).cumsum() + 50
        window = 100

        hurst = _compute_rolling_hurst(prices, window)

        assert np.isnan(hurst[0])
        # The value should be around 0.5 for a random walk
        assert 0.4 < hurst[-1] < 0.6

    def test_compute_rolling_kurtosis_skew(self):
        """Test rolling kurtosis and skewness computation."""
        prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
        window = 20

        kurt, skew = _compute_rolling_kurtosis_skew(prices, window)

        assert np.isnan(kurt[0])
        assert np.isnan(skew[0])
        # For a uniform distribution, skewness is ~0 and kurtosis is negative
        assert abs(skew[-1]) < 0.1
        assert kurt[-1] < 0

    def test_compute_rolling_entropy(self):
        """Test rolling entropy computation."""
        # Highly ordered (linear) data should have low entropy
        prices = np.arange(100)
        window = 50

        entropy_val = _compute_rolling_entropy(prices, window)
        low_entropy = entropy_val[-2]  # last valid value

        # Highly random data should have high entropy
        np.random.seed(42)
        prices_random = np.random.rand(100)
        entropy_val_random = _compute_rolling_entropy(prices_random, window)
        high_entropy = entropy_val_random[-2]

        assert low_entropy < high_entropy

    def test_compute_bollinger_width(self):
        """Test Bollinger Bandwidth computation."""
        upper = np.array([110, 112, 115])
        middle = np.array([100, 102, 105])
        lower = np.array([90, 92, 95])

        width = _compute_bollinger_width(upper, lower, middle)

        expected_width = (upper - lower) / middle
        np.testing.assert_array_almost_equal(width, expected_width)

    def test_compute_spread_vs_range(self):
        """Test spread vs. candle range computation."""
        high = np.array([10, 20, 30, 40, 50])
        low = np.array([0, 10, 20, 30, 40])
        spread = np.array([1, 1, 1, 1, 1])
        window = 3

        ratio = _compute_spread_vs_range(high, low, spread, window)

        # Candle range is always 10. Spread is 1. Ratio should be 0.1
        expected_ratio = 0.1
        assert abs(ratio[-1] - expected_ratio) < 1e-9


class TestFeatureConfig:
    """Test feature configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()

        assert config.momentum_periods == [5, 10, 20, 50]
        assert config.volatility_windows == [10, 20, 50, 100]
        assert config.trend_periods == [10, 20, 50]
        assert config.bollinger_periods == [20, 50]
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.max_workers == 4
        assert config.use_numba is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            momentum_periods=[5, 15],
            volatility_windows=[10, 30],
            rsi_period=21,
            max_workers=2,
        )

        assert config.momentum_periods == [5, 15]
        assert config.volatility_windows == [10, 30]
        assert config.rsi_period == 21
        assert config.max_workers == 2


class TestHighPerformanceFeatureEngine:
    """Test the main feature engine class."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        np.random.seed(42)

        # Generate realistic price data
        price = 1.1000
        prices = [price]

        for _ in range(99):
            change = np.random.normal(0, 0.001)  # 0.1% volatility
            price *= 1 + change
            prices.append(price)

        data = []
        for i, date in enumerate(dates):
            open_price = prices[i]
            close_price = prices[i] * (1 + np.random.normal(0, 0.0005))
            high_price = max(open_price, close_price) * (
                1 + abs(np.random.normal(0, 0.0005))
            )
            low_price = min(open_price, close_price) * (
                1 - abs(np.random.normal(0, 0.0005))
            )
            volume = np.random.uniform(100000, 1000000)

            ohlcv = OHLCV(
                symbol="EUR/USD",
                timestamp=date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timeframe="1h",
            )
            data.append(ohlcv)

        return data

    @pytest.fixture
    def feature_engine(self):
        """Create feature engine for testing."""
        config = FeatureConfig(max_workers=2)  # Reduce workers for testing
        logger = structlog.get_logger("test")
        return HighPerformanceFeatureEngine(config=config, logger=logger)

    def test_feature_engine_initialization(self, feature_engine):
        """Test feature engine initialization."""
        assert feature_engine.config is not None
        assert feature_engine.logger is not None
        assert len(feature_engine.feature_registry) > 0
        assert feature_engine.max_buffer_size > 0

    def test_update_with_ohlcv(self, feature_engine, sample_ohlcv_data):
        """Test updating feature engine with OHLCV data."""

        # Process first 20 bars to build up buffer
        for i in range(20):
            features = feature_engine.update(sample_ohlcv_data[i])

            # After enough data, should have some features
            if i >= 10:
                assert isinstance(features, dict)
                # Should have at least some SMA features
                sma_features = [k for k in features.keys() if "sma" in k]
                assert len(sma_features) > 0

    def test_get_latest_features(self, feature_engine, sample_ohlcv_data):
        """Test getting latest features."""
        symbol = "EUR/USD"

        # Add data to buffer
        for bar in sample_ohlcv_data[:30]:
            feature_engine.update(bar)

        # Get latest features
        features = feature_engine.get_latest_features(symbol)

        assert isinstance(features, dict)
        assert len(features) > 0

        # Check that all values are numeric and not NaN
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not pd.isna(value)

    def test_buffer_management(self, feature_engine, sample_ohlcv_data):
        """Test that buffer size is properly managed."""
        symbol = "EUR/USD"
        max_size = feature_engine.max_buffer_size

        # Add more data than max buffer size
        for bar in sample_ohlcv_data:
            feature_engine.update(bar)

            # Check buffer doesn't exceed max size
            if symbol in feature_engine.rolling_buffers:
                buffer_len = len(feature_engine.rolling_buffers[symbol])
                assert buffer_len <= max_size

    @pytest.mark.asyncio
    async def test_compute_technical_features(self, feature_engine):
        """Test async technical features computation."""
        # Create test DataFrame
        data = pd.DataFrame(
            {
                "open": [1.1000, 1.1010, 1.1020, 1.1015, 1.1025] * 20,
                "high": [1.1005, 1.1015, 1.1025, 1.1020, 1.1030] * 20,
                "low": [1.0995, 1.1005, 1.1015, 1.1010, 1.1020] * 20,
                "close": [1.1002, 1.1012, 1.1022, 1.1017, 1.1027] * 20,
                "volume": [100000] * 100,
            }
        )

        result = await feature_engine.compute_technical_features(data, [10, 20])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert len(result.columns) > 0

    @pytest.mark.asyncio
    async def test_compute_volatility_features(self, feature_engine):
        """Test async volatility features computation."""
        data = pd.DataFrame(
            {
                "close": [
                    1.1000 + 0.001 * np.sin(i * 0.1) + np.random.normal(0, 0.0005)
                    for i in range(100)
                ]
            }
        )

        result = await feature_engine.compute_volatility_features(data, [10, 20])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)

    @pytest.mark.asyncio
    async def test_compute_momentum_features(self, feature_engine):
        """Test async momentum features computation."""
        data = pd.DataFrame(
            {"close": [1.1000 * (1.001**i) for i in range(50)]}  # Trending data
        )

        result = await feature_engine.compute_momentum_features(data, [5, 10])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)

    def test_add_custom_feature(self, feature_engine):
        """Test adding custom features."""

        def custom_feature(data):
            return {"custom_indicator": data["close"].rolling(5).mean().values}

        feature_engine.add_custom_feature("custom", custom_feature)

        assert "custom" in feature_engine.feature_registry
        assert len(feature_engine.get_available_features()) == len(
            feature_engine.feature_registry
        )

    def test_remove_feature(self, feature_engine):
        """Test removing features."""
        initial_count = len(feature_engine.feature_registry)

        feature_engine.remove_feature("sma")

        assert "sma" not in feature_engine.feature_registry
        assert len(feature_engine.feature_registry) == initial_count - 1

    def test_edge_cases(self, feature_engine):
        """Test edge cases and error handling."""
        # Empty symbol buffer
        features = feature_engine.get_latest_features("NONEXISTENT")
        assert features == {}

        # Single data point
        bar = OHLCV(
            symbol="TEST/USD",
            timestamp=datetime.now(timezone.utc),
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.05,
            volume=1000,
            timeframe="1h",
        )

        features = feature_engine.update(bar)
        # Should return empty dict or dict with NaN values
        assert isinstance(features, dict)

    def test_feature_engine_cleanup(self, feature_engine):
        """Test proper cleanup of resources."""
        feature_engine.close()
        # Should not raise any exceptions


class TestFeatureEnginePerformance:
    """Test performance characteristics of the feature engine."""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        dates = pd.date_range(start="2020-01-01", periods=10000, freq="1h")
        np.random.seed(42)

        price = 1.1000
        data = []

        for date in dates:
            change = np.random.normal(0, 0.001)
            price *= 1 + change

            ohlcv = OHLCV(
                symbol="EUR/USD",
                timestamp=date,
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=100000,
                timeframe="1h",
            )
            data.append(ohlcv)

        return data

    @pytest.mark.slow
    def test_streaming_performance(self, large_dataset):
        """Test performance with large streaming dataset."""
        config = FeatureConfig(max_workers=4)
        engine = HighPerformanceFeatureEngine(config=config)

        import time

        start_time = time.time()

        # Process all data
        for bar in large_dataset:
            engine.update(bar)

        elapsed_time = time.time() - start_time

        # Should process 10k bars in reasonable time (adjust threshold as needed)
        bars_per_second = len(large_dataset) / elapsed_time
        print(f"Processed {bars_per_second:.0f} bars/second")

        assert (
            bars_per_second > 100
        )  # Should be able to process at least 100 bars/second

        engine.close()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_parallel_computation_performance(self):
        """Test parallel computation performance."""
        config = FeatureConfig(max_workers=4)
        engine = HighPerformanceFeatureEngine(config=config)

        # Create large DataFrame
        data = pd.DataFrame(
            {
                "open": np.random.normal(1.1, 0.01, 5000),
                "high": np.random.normal(1.102, 0.01, 5000),
                "low": np.random.normal(1.098, 0.01, 5000),
                "close": np.random.normal(1.1, 0.01, 5000),
                "volume": np.random.uniform(100000, 1000000, 5000),
            }
        )

        import time

        start_time = time.time()

        result = await engine.compute_technical_features(data)

        elapsed_time = time.time() - start_time
        print(f"Computed features for {len(data)} bars in {elapsed_time:.2f} seconds")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)

        engine.close()


class TestFeatureEngineFactory:
    """Test the feature engine factory function."""

    def test_create_feature_engine_default(self):
        """Test creating feature engine with default parameters."""
        engine = create_feature_engine()

        assert isinstance(engine, HighPerformanceFeatureEngine)
        assert engine.config.use_numba is True
        assert engine.config.max_workers == 4

        engine.close()

    def test_create_feature_engine_custom(self):
        """Test creating feature engine with custom parameters."""
        logger = structlog.get_logger("test")

        engine = create_feature_engine(
            momentum_periods=[5, 15],
            volatility_windows=[10, 30],
            use_numba=False,
            max_workers=2,
            logger=logger,
        )

        assert isinstance(engine, HighPerformanceFeatureEngine)
        assert engine.config.momentum_periods == [5, 15]
        assert engine.config.volatility_windows == [10, 30]
        assert engine.config.use_numba is False
        assert engine.config.max_workers == 2

        engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
