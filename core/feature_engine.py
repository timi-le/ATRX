"""
High-Performance Feature Engine for FX AI-Quant Trading System.

This module implements a modular, extensible feature computation engine
that generates technical indicators and features for ML models and regime detection.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Callable

import nolds
import numpy as np
import pandas as pd
import structlog
from numba import jit
from scipy.stats import entropy, kurtosis, skew



@dataclass
class FeatureConfig:
    """Configuration for feature computation."""

    momentum_periods: list[int] = None
    volatility_windows: list[int] = None
    trend_periods: list[int] = None
    bollinger_periods: list[int] = None
    hurst_windows: list[int] = None
    entropy_windows: list[int] = None
    kurtosis_windows: list[int] = None
    spread_windows: list[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    adx_period: int = 14
    cci_period: int = 20
    atr_period: int = 14
    max_workers: int = 4
    use_numba: bool = True
    
    def __post_init__(self):
        if self.momentum_periods is None:
            self.momentum_periods = [5, 10, 20, 50]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50, 100]
        if self.trend_periods is None:
            self.trend_periods = [10, 20, 50]
        if self.bollinger_periods is None:
            self.bollinger_periods = [20, 50]
        if self.hurst_windows is None:
            self.hurst_windows = [50, 100, 200]
        if self.entropy_windows is None:
            self.entropy_windows = [20, 50, 100]
        if self.kurtosis_windows is None:
            self.kurtosis_windows = [20, 50, 100]
        if self.spread_windows is None:
            self.spread_windows = [20, 50, 100]


# Numba-optimized technical indicator functions
@jit(nopython=True, cache=True)
def _compute_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute Simple Moving Average using Numba."""
    n = len(prices)
    sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(prices[i - period + 1 : i + 1])
    
    return sma


def _compute_rolling_hurst(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling Hurst Exponent."""
    n = len(prices)
    hurst = np.full(n, np.nan)
    for i in range(window, n):
        hurst[i] = nolds.hurst_rs(prices[i - window : i])
    return hurst


def _compute_rolling_kurtosis_skew(
    prices: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling kurtosis and skewness."""
    n = len(prices)
    kurt = np.full(n, np.nan)
    sk = np.full(n, np.nan)
    for i in range(window, n):
        data = prices[i - window : i]
        kurt[i] = kurtosis(data)
        sk[i] = skew(data)
    return kurt, sk


def _compute_rolling_entropy(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling entropy of price changes."""
    n = len(prices)
    ent = np.full(n, np.nan)
    price_changes = np.diff(prices)
    for i in range(window, n - 1):
        # Discretize the price changes into bins
        hist, _ = np.histogram(price_changes[i - window : i], bins="auto", density=True)
        ent[i] = entropy(hist)
    return ent


def _compute_bollinger_width(
    upper_band: np.ndarray, lower_band: np.ndarray, middle_band: np.ndarray
) -> np.ndarray:
    """Compute normalized Bollinger Bandwidth."""
    return (upper_band - lower_band) / middle_band


def _compute_spread_vs_range(
    high: np.ndarray, low: np.ndarray, spread: np.ndarray, window: int
) -> np.ndarray:
    """Compute rolling average of spread vs. candle range."""
    n = len(high)
    ratio = np.full(n, np.nan)
    candle_range = high - low

    # Avoid division by zero
    candle_range[candle_range == 0] = 1e-9

    spread_over_range = spread / candle_range

    for i in range(window, n):
        ratio[i] = np.mean(spread_over_range[i - window : i])

    return ratio


@jit(nopython=True, cache=True)
def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average using Numba."""
    n = len(prices)
    ema = np.full(n, np.nan)
    
    if n == 0:
        return ema
    
    # Initialize first EMA with SMA
    alpha = 2.0 / (period + 1)
    
    # First value
    ema[period - 1] = np.mean(prices[:period])
    
    # Calculate subsequent EMAs
    for i in range(period, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@jit(nopython=True, cache=True)
def _compute_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute Relative Strength Index using Numba."""
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Calculate RSI for initial period
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi[period] = 100.0
    
    # Calculate subsequent RSI values using smoothed averages
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi


@jit(nopython=True, cache=True)
def _compute_bollinger_bands(
    prices: np.ndarray, period: int, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bollinger Bands using Numba."""
    n = len(prices)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        
        middle[i] = mean_val
        upper[i] = mean_val + std_dev * std_val
        lower[i] = mean_val - std_dev * std_val
    
    return upper, middle, lower


@jit(nopython=True, cache=True)
def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Compute Average True Range using Numba."""
    n = len(high)
    atr = np.full(n, np.nan)
    
    if n < 2:
        return atr
    
    # Calculate true ranges
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR using Wilder's smoothing
    if period <= n - 1:
        atr[period] = np.mean(tr[1 : period + 1])
        
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


@jit(nopython=True, cache=True)
def _compute_stochastic_oscillator(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Stochastic Oscillator using Numba."""
    n = len(high)
    k_percent = np.full(n, np.nan)
    d_percent = np.full(n, np.nan)
    
    for i in range(k_period - 1, n):
        window_high = np.max(high[i - k_period + 1 : i + 1])
        window_low = np.min(low[i - k_period + 1 : i + 1])
        
        if window_high != window_low:
            k_percent[i] = (close[i] - window_low) / (window_high - window_low) * 100.0
        else:
            k_percent[i] = 50.0
    
    # Calculate %D (SMA of %K)
    for i in range(k_period + d_period - 2, n):
        d_percent[i] = np.mean(k_percent[i - d_period + 1 : i + 1])
    
    return k_percent, d_percent


@jit(nopython=True, cache=True)
def _compute_rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling volatility using Numba."""
    n = len(returns)
    vol = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_returns = returns[i - window + 1 : i + 1]
        vol[i] = np.std(window_returns) * np.sqrt(252)  # Annualized
    
    return vol


@jit(nopython=True, cache=True)
def _compute_momentum(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute price momentum using Numba."""
    n = len(prices)
    momentum = np.full(n, np.nan)
    
    for i in range(period, n):
        momentum[i] = (prices[i] / prices[i - period] - 1.0) * 100.0
    
    return momentum


class HighPerformanceFeatureEngine:
    """
    A high-performance feature engine capable of computing features across
    multiple timeframes and aligning them to a primary timeframe for use
    in sophisticated financial models.
    """

    def __init__(
        self,
        config: FeatureConfig = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        """
        Initializes the feature engine with a given configuration.

        Args:
            config (FeatureConfig, optional): Configuration object for feature parameters.
                                              Defaults to a standard configuration.
            logger (structlog.stdlib.BoundLogger, optional): A structured logger instance.
        """
        self.config = config or FeatureConfig()
        self.logger = logger or structlog.get_logger(__name__)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._custom_features: dict[str, Callable] = {}
        self.logger.info("HighPerformanceFeatureEngine initialized", config=self.config)

    async def compute_technical_features(
        self, data: pd.DataFrame, window_sizes: list[int] = None
    ) -> pd.DataFrame:
        """Compute all technical features. The interface is async, but core logic is sync for MTF."""
        if window_sizes:
            self.logger.warning(
                "`window_sizes` is ignored; using FeatureConfig.",
                provided_windows=window_sizes,
            )
        return self._compute_technical_features_sync(data)

    async def compute_volatility_features(
        self, data: pd.DataFrame, lookback_periods: list[int] = None
    ) -> pd.DataFrame:
        """Compute volatility-based features."""
        if lookback_periods:
            self.logger.warning(
                "`lookback_periods` is ignored; using FeatureConfig.",
                provided_periods=lookback_periods,
            )

        feature_dict = self._compute_volatility_features(data)
        return pd.DataFrame(feature_dict, index=data.index)

    async def compute_momentum_features(
        self, data: pd.DataFrame, periods: list[int] = None
    ) -> pd.DataFrame:
        """Compute momentum indicators."""
        if periods:
            self.logger.warning(
                "`periods` is ignored; using FeatureConfig.", provided_periods=periods
            )

        feature_dict = self._compute_momentum_features(data)
        return pd.DataFrame(feature_dict, index=data.index)

    async def compute_carry_features(
        self, fx_data: pd.DataFrame, interest_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """This engine does not compute carry features."""
        self.logger.warning(
            "compute_carry_features is not implemented in HighPerformanceFeatureEngine."
        )
        return pd.DataFrame(index=fx_data.index)

    async def compute_macro_surprises(
        self, economic_data: pd.DataFrame, expectations: pd.DataFrame
    ) -> pd.DataFrame:
        """This engine does not compute macro surprise features."""
        self.logger.warning(
            "compute_macro_surprises is not implemented in HighPerformanceFeatureEngine."
        )
        return pd.DataFrame(index=economic_data.index)

    def _compute_volatility_features(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        features = {}
        returns = data["close"].pct_change().to_numpy()
        for window in self.config.volatility_windows:
            features[f"volatility_{window}"] = _compute_rolling_volatility(
                returns, window
            )
        return features

    def _compute_momentum_features(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        features = {}
        prices_np = data["close"].to_numpy()
        for period in self.config.momentum_periods:
            features[f"momentum_{period}"] = _compute_momentum(prices_np, period)
        return features
    
    def _compute_hurst_features(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        features = {}
        prices_np = data["close"].to_numpy()
        for window in self.config.hurst_windows:
            features[f"hurst_{window}"] = _compute_rolling_hurst(prices_np, window)
        return features
    
    def _compute_kurtosis_skew_features(
        self, data: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        features = {}
        prices_np = data["close"].to_numpy()
        for window in self.config.kurtosis_windows:
            kurt, skew_val = _compute_rolling_kurtosis_skew(prices_np, window)
            features[f"kurtosis_{window}"] = kurt
            features[f"skew_{window}"] = skew_val
        return features
    
    def _compute_entropy_features(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        features = {}
        prices_np = data["close"].to_numpy()
        for window in self.config.entropy_windows:
            features[f"entropy_{window}"] = _compute_rolling_entropy(prices_np, window)
        return features
    
    def _compute_bollinger_features(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Compute Bollinger Bands features."""
        close_prices = data["close"].values
        features = {}
        
        for period in self.config.bollinger_periods:
            if len(close_prices) >= period:
                upper, middle, lower = _compute_bollinger_bands(close_prices, period)
                features[f"bb_width_{period}"] = (upper - lower) / middle
        
        return features
    
    def _compute_spread_features(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        # This method is simplified and assumes 'spread' is not a primary feature for now
            return {}
        
    def _compute_technical_features_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronous version of feature computation for a single DataFrame.
        This is useful for batch processing and MTF alignment.
        """
        # Ensure data has a datetime index for time-based features
        # Handle both cases: when 'time' is a column or already the index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "time" in data.columns:
                data = data.set_index("time")
            else:
                raise ValueError("Data must have a datetime index or a 'time' column.")

        all_feature_sets = []
        computations = {
            "volatility": self._compute_volatility_features,
            "momentum": self._compute_momentum_features,
            "hurst": self._compute_hurst_features,
            "kurtosis_skew": self._compute_kurtosis_skew_features,
            "entropy": self._compute_entropy_features,
            "bollinger": self._compute_bollinger_features,
        }

        for name, func in computations.items():
            try:
                feature_dict = func(data)
                if feature_dict:
                    all_feature_sets.append(
                        pd.DataFrame(feature_dict, index=data.index)
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to compute '{name}' features", error=e, exc_info=False
                )

        if not all_feature_sets:
            return pd.DataFrame(index=data.index)

        return pd.concat(all_feature_sets, axis=1)

    async def compute_mtf_features(
        self, mtf_data: dict[str, pd.DataFrame], primary_tf: str = "M1"
    ) -> pd.DataFrame:
        """
        Computes features on multiple timeframes and aligns them to a primary timeframe.
        """
        if primary_tf not in mtf_data:
            raise ValueError(
                f"Primary timeframe '{primary_tf}' not found in mtf_data keys."
            )

        # Handle both cases: when 'time' is a column or already the index
        primary_df = mtf_data[primary_tf]
        if "time" in primary_df.columns:
            primary_df = primary_df.set_index("time")
        elif not isinstance(primary_df.index, pd.DatetimeIndex):
            raise ValueError(
                "Primary dataframe must have a datetime index or 'time' column"
            )

        final_df = primary_df.copy()

        for tf_str, df in mtf_data.items():
            self.logger.info(f"Computing features for timeframe: {tf_str}")

            # Handle both cases: when 'time' is a column or already the index
            tf_df = df
            if "time" in tf_df.columns:
                tf_df = tf_df.set_index("time")
            elif not isinstance(tf_df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"Timeframe {tf_str} dataframe must have a datetime index or 'time' column"
                )

            tf_features = self._compute_technical_features_sync(tf_df)

            tf_features.columns = [f"{col}_{tf_str}" for col in tf_features.columns]

            if tf_str == primary_tf:
                final_df = final_df.join(tf_features)
            else:
                # Align higher timeframe features to primary timeframe
                aligned_features = tf_features.reindex(final_df.index, method="ffill")
                final_df = final_df.join(aligned_features)

        self.logger.info(
            f"Successfully computed and aligned features from {list(mtf_data.keys())}."
        )
        return final_df.dropna()
    
    def add_custom_feature(self, name: str, compute_func: Callable) -> None:
        """Adds a custom feature computation function."""
        self.logger.warning(
            "Custom features are not fully supported with the new MTF engine yet."
        )
        self._custom_features[name] = compute_func
    
    def remove_feature(self, name: str) -> None:
        """Remove a feature from computation."""
        if name in self._custom_features:
            del self._custom_features[name]
            self.logger.info(f"Removed custom feature: {name}")

    def get_available_features(self) -> list[str]:
        """Get list of available feature types."""
        return list(self._custom_features.keys())
    
    def close(self):
        """Clean up resources."""
        self.thread_executor.shutdown(wait=False)
        self.logger.info("HighPerformanceFeatureEngine closed.")


def create_feature_engine(
    momentum_periods: list[int] = None,
    volatility_windows: list[int] = None,
    use_numba: bool = True,
    max_workers: int = 4,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> HighPerformanceFeatureEngine:
    """Factory function to create feature engine with custom configuration."""
    
    config = FeatureConfig(
        momentum_periods=momentum_periods,
        volatility_windows=volatility_windows,
        use_numba=use_numba,
        max_workers=max_workers,
    )
    
    return HighPerformanceFeatureEngine(config=config, logger=logger) 
