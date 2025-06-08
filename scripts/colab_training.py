"""
Professional Multi-Currency Econophysics-Inspired FX Trading Model Training
Advanced ML approach with regime detection and sophisticated feature engineering
Supports EURUSD, GBPUSD, USDJPY with pre-downloaded data from GitHub
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# Try to import MT5, fallback to synthetic data if not available
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
    print("✅ MetaTrader5 available for real data")
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available, will use synthetic data")

try:
    import yfinance as yf

    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


class EconophysicsFeatureEngine:
    """Professional econophysics-inspired feature engineering for FX markets"""

    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.volatility_windows = [10, 20, 50]

    def hurst_exponent(self, ts, max_lag=20):
        """Calculate Hurst exponent for trend persistence analysis"""
        try:
            if len(ts) < max_lag:
                return 0.5
            lags = range(2, min(max_lag, len(ts)))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            if len(tau) < 2:
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except Exception:
            return 0.5

    def fractal_dimension(self, ts, n=20):
        """Calculate fractal dimension using box counting"""
        try:
            if len(ts) < n:
                return 1.5
            ts_norm = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-8)
            scales = np.logspace(0.01, 0.2, n, endpoint=False, base=10)
            counts = []

            for scale in scales:
                hist, _ = np.histogram(ts_norm, bins=max(1, int(1 / scale)))
                counts.append(np.sum(hist > 0))

            if len(counts) < 2:
                return 1.5
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return -coeffs[0]
        except Exception:
            return 1.5

    def regime_volatility_clustering(self, returns, window=50):
        """Detect volatility clustering regimes"""
        vol = returns.rolling(window).std()
        vol_mean = vol.rolling(window * 2).mean()
        vol_std = vol.rolling(window * 2).std()

        low_vol = vol < (vol_mean - 0.5 * vol_std)
        high_vol = vol > (vol_mean + 0.5 * vol_std)

        regime = pd.Series(1, index=returns.index)
        regime[low_vol] = 0
        regime[high_vol] = 2

        return regime

    def market_microstructure_features(self, ohlcv_data):
        """Extract microstructure features from OHLCV data"""
        df = ohlcv_data.copy()

        # Avoid division by zero
        range_val = df["high"] - df["low"]
        range_val = range_val.replace(0, np.nan)

        df["price_efficiency"] = np.abs(df["close"] - df["open"]) / range_val
        df["bid_ask_spread_proxy"] = range_val / df["close"]
        df["buying_pressure"] = (df["close"] - df["low"]) / range_val
        df["selling_pressure"] = (df["high"] - df["close"]) / range_val

        # Volume-price relationship (handle zero volume)
        price_change = (df["close"] - df["open"]) / df["open"]
        df["volume_price_trend"] = df["volume"] * price_change

        return df

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Avoid division by zero
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def multi_timeframe_features(self, data):
        """Create multi-timeframe technical features"""
        df = data.copy()

        # Multiple moving averages
        for window in self.lookback_periods:
            if len(df) > window:
                df[f"sma_{window}"] = df["close"].rolling(window).mean()
                df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
                sma_val = df[f"sma_{window}"]
                df[f"price_vs_sma_{window}"] = (df["close"] - sma_val) / (
                    sma_val + 1e-8
                )

        # RSI with multiple periods
        for period in [14, 21, 50]:
            if len(df) > period:
                df[f"rsi_{period}"] = self.calculate_rsi(df["close"], period)

        # Bollinger Bands
        for window in [20, 50]:
            if len(df) > window:
                sma = df["close"].rolling(window).mean()
                std = df["close"].rolling(window).std()
                df[f"bb_upper_{window}"] = sma + (2 * std)
                df[f"bb_lower_{window}"] = sma - (2 * std)
                bb_range = df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
                df[f"bb_position_{window}"] = (
                    df["close"] - df[f"bb_lower_{window}"]
                ) / (bb_range + 1e-8)

        # MACD family
        if len(df) > 26:
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

        return df

    def volatility_features(self, returns):
        """Advanced volatility modeling features"""
        df = pd.DataFrame(index=returns.index)

        for window in self.volatility_windows:
            if len(returns) > window:
                df[f"realized_vol_{window}"] = returns.rolling(window).std() * np.sqrt(
                    252
                )
                df[f"vol_of_vol_{window}"] = (
                    df[f"realized_vol_{window}"].rolling(window).std()
                )

        df["volatility_regime"] = self.regime_volatility_clustering(returns)

        # Avoid division by zero
        vol_20 = df.get("realized_vol_20", pd.Series(1, index=returns.index))
        vol_50 = df.get("realized_vol_50", pd.Series(1, index=returns.index))
        df["vol_momentum"] = vol_20 / (vol_50 + 1e-8)

        return df

    def create_econophysics_features(self, ohlcv_data):
        """Main feature engineering pipeline"""
        print("🔬 Creating advanced econophysics features...")

        df = ohlcv_data.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Market microstructure
        df = self.market_microstructure_features(df)

        # Multi-timeframe technical features
        df = self.multi_timeframe_features(df)

        # Advanced volatility features
        vol_features = self.volatility_features(df["returns"])
        df = pd.concat([df, vol_features], axis=1)

        # Econophysics-inspired features
        print("📊 Computing fractal and complexity measures...")
        for window in [50, 100, 200]:
            if len(df) > window:
                df[f"hurst_{window}"] = (
                    df["close"]
                    .rolling(window)
                    .apply(lambda x: self.hurst_exponent(x.values), raw=False)
                )
                df[f"fractal_dim_{window}"] = (
                    df["close"]
                    .rolling(window)
                    .apply(lambda x: self.fractal_dimension(x.values), raw=False)
                )

        # Cross-asset correlation features
        df["market_beta"] = df["returns"].rolling(50).corr(df["returns"].shift(1))

        # Momentum and mean reversion features
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
                mean_val = df["close"].rolling(period).mean()
                std_val = df["close"].rolling(period).std()
                df[f"mean_reversion_{period}"] = (df["close"] - mean_val) / (
                    std_val + 1e-8
                )

        print(f"✅ Created {len(df.columns)} total features")
        return df


def get_mt5_data(
    symbol="EURUSD", timeframe="H1", start_date="2018-01-01", end_date="2025-03-31"
):
    """Download data from MetaTrader 5 for specified date range."""
    if not MT5_AVAILABLE:
        return None

    print(
        f"📊 Attempting to get MT5 data for {symbol} on {timeframe} from {start_date} to {end_date}..."
    )

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return None

    # Timeframe mapping
    timeframe_map = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

    try:
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get data for the specified date range
        rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            print(
                f"❌ No data received for {symbol} in date range {start_date} to {end_date}"
            )
            print("🔄 Trying with recent data instead...")
            # Fallback to recent data
            rates = mt5.copy_rates_from_pos(
                symbol, tf, 0, 50000
            )  # Get more recent data

        if rates is None or len(rates) == 0:
            print(f"❌ No data available for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Rename columns to match expected format
        df.columns = [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]
        df = df[["open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Filter to requested date range if we got more data
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        print(f"✅ Downloaded {len(df)} bars of {symbol} data from MT5")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        return df

    except Exception as e:
        print(f"❌ MT5 data download failed: {e}")
        return None
    finally:
        mt5.shutdown()


def generate_professional_forex_data(symbol="EURUSD", timeframe="H1"):
    """Download real forex data from MT5 or generate sophisticated synthetic data (2018-2025)."""
    print(f"📈 Fetching professional forex data for {symbol} (2018-2025)...")

    # Try MT5 first with extended date range
    if MT5_AVAILABLE:
        data = get_mt5_data(symbol, timeframe, "2018-01-01", "2025-03-31")
        if data is not None and len(data) > 1000:
            return data

    # Try Yahoo Finance as backup with extended period
    if YF_AVAILABLE:
        try:
            symbol_yf = f"{symbol}=X"
            ticker = yf.Ticker(symbol_yf)

            # Try to get maximum available data
            print("📈 Attempting to download extended Yahoo Finance data...")
            data = ticker.history(period="max", interval="1h")

            # If that fails, try different periods
            if len(data) < 1000:
                print("🔄 Trying 5-year period...")
                data = ticker.history(period="5y", interval="1h")

            if len(data) < 1000:
                print("🔄 Trying 2-year period...")
                data = ticker.history(period="2y", interval="1h")

            if len(data) > 1000:
                data.columns = data.columns.str.lower()
                print(f"✅ Downloaded {len(data)} Yahoo Finance data points")
                print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
                return data
        except Exception as e:
            print(f"⚠️ Yahoo Finance failed: {e}")

    # Generate sophisticated synthetic data for 2018-2025
    print("🎲 Generating sophisticated synthetic forex data (2018-2025)...")

    # Calculate samples for ~7 years of hourly data
    start_date = "2018-01-01"
    end_date = "2025-03-31"
    dates = pd.date_range(start=start_date, end=end_date, freq="1H")
    n_samples = len(dates)

    print(f"📊 Generating {n_samples:,} data points from {start_date} to {end_date}")

    np.random.seed(42)

    # Create more realistic regime transitions over 7 years
    # Model major market cycles: Normal (60%), Trending (25%), Volatile (15%)
    regimes = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])

    # Add some persistence to regimes (markets stay in regimes for periods)
    for i in range(1, n_samples):
        if np.random.random() < 0.8:  # 80% chance to stay in same regime
            regimes[i] = regimes[i - 1]

    returns = np.zeros(n_samples)

    # Generate returns with different characteristics per regime
    for i in range(n_samples):
        if regimes[i] == 0:  # Normal market regime
            returns[i] = np.random.normal(0, 0.0003)
        elif regimes[i] == 1:  # Trending regime
            # Trending markets have momentum
            momentum = 0.0001 if i > 0 and returns[i - 1] > 0 else -0.0001
            returns[i] = np.random.normal(momentum, 0.0002)
        else:  # Volatile/Crisis regime
            returns[i] = np.random.normal(0, 0.001)

    # Add volatility clustering (realistic market behavior)
    for i in range(1, n_samples):
        if abs(returns[i - 1]) > 0.0005:  # Previous high volatility
            returns[i] *= 1.5  # Increase current volatility

    # Add some long-term trends and cycles
    # Add weekly cycles (weaker on weekends)
    for i in range(n_samples):
        day_of_week = dates[i].dayofweek
        if day_of_week >= 5:  # Weekend (Saturday=5, Sunday=6)
            returns[i] *= 0.3  # Much lower activity on weekends

    # Add some realistic price levels with long-term drift
    base_price = 1.1000
    trend_component = np.linspace(
        0, 0.15, n_samples
    )  # Slight upward trend over 7 years
    trend_component += 0.05 * np.sin(
        np.linspace(0, 14 * np.pi, n_samples)
    )  # Add cycles

    # Generate prices
    prices = base_price * np.exp(np.cumsum(returns) + trend_component * 0.1)

    # Add realistic noise
    noise = np.random.normal(0, 0.00001, n_samples)

    data = pd.DataFrame(
        {
            "open": prices + noise,
            "high": prices + np.abs(np.random.normal(0, 0.0001, n_samples)),
            "low": prices - np.abs(np.random.normal(0, 0.0001, n_samples)),
            "close": prices,
            "volume": np.random.lognormal(10, 0.5, n_samples),
        },
        index=dates,
    )

    # Ensure OHLC relationships are valid
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    # Add some realistic volume patterns
    # Higher volume during market hours (approximate)
    for i in range(len(data)):
        hour = data.index[i].hour
        if 7 <= hour <= 18:  # Market hours (approximate)
            data.iloc[i, data.columns.get_loc("volume")] *= 1.5
        if hour in [8, 9, 14, 15]:  # Peak trading hours
            data.iloc[i, data.columns.get_loc("volume")] *= 2.0

    print(f"✅ Generated {len(data):,} sophisticated synthetic data points")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    print(f"💹 Price range: {data['close'].min():.5f} to {data['close'].max():.5f}")

    return data


def load_predownloaded_data(symbol="EURUSD"):
    """Load pre-downloaded data from GitHub repository"""
    try:
        # Check if we're in Colab environment
        if "/content" in os.getcwd():
            data_path = f"data/forex/{symbol}_2018_2025.csv"
        else:
            data_path = f"data/forex/{symbol}_2018_2025.csv"

        print(f"📂 Loading pre-downloaded {symbol} data from: {data_path}")

        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"✅ Loaded {len(data):,} pre-downloaded {symbol} data points")
            print(f"📅 Date range: {data.index.min()} to {data.index.max()}")

            # Ensure proper column names
            if "source" in data.columns:
                source = data["source"].iloc[0]
                data = data.drop("source", axis=1)
                print(f"📊 Data source: {source}")

            return data
        else:
            print(f"❌ Pre-downloaded file not found: {data_path}")
            return None

    except Exception as e:
        print(f"❌ Error loading pre-downloaded {symbol} data: {e}")
        return None


def load_multi_currency_data(currency_pairs=['EURUSD', 'GBPUSD', 'USDJPY'], timeframes=['H1'], data_dir='data/forex'):
    """
    Load multi-currency, multi-timeframe data with flexible timeframe selection
    
    Args:
        currency_pairs: List of currency pairs to load
        timeframes: List of timeframes to load ('M5', 'M15', 'M30', 'H1', 'H4', 'D1')
        data_dir: Directory containing the forex data files
    
    Returns:
        Dictionary with structure: {currency: {timeframe: dataframe}}
    """
    print(f"📊 Loading multi-timeframe data for {len(currency_pairs)} currencies, {len(timeframes)} timeframes")
    
    all_data = {}
    total_samples = 0
    
    for symbol in currency_pairs:
        all_data[symbol] = {}
        
        for timeframe in timeframes:
            print(f"📈 Loading {symbol} {timeframe} data...")
            
            # Try to load pre-downloaded data from GitHub first
            try:
                filename = f"{data_dir}/{symbol}_{timeframe}_2018_2025.csv"
                
                if os.path.exists(filename):
                    data = pd.read_csv(filename, index_col=0, parse_dates=True)
                    
                    # Ensure we have the required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        all_data[symbol][timeframe] = data[required_cols].copy()
                        samples = len(data)
                        total_samples += samples
                        print(f"✅ Loaded {symbol} {timeframe}: {samples:,} samples from GitHub data")
                        continue
                    else:
                        print(f"⚠️ Missing required columns in {filename}")
                        
                else:
                    print(f"⚠️ File not found: {filename}")
                    
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
            
            # Fallback to live download/synthetic generation
            print(f"🔄 Falling back to live data download for {symbol} {timeframe}...")
            fallback_data = get_fallback_data(symbol, timeframe)
            
            if fallback_data is not None and len(fallback_data) > 100:
                all_data[symbol][timeframe] = fallback_data
                samples = len(fallback_data)
                total_samples += samples
                print(f"✅ Generated {symbol} {timeframe}: {samples:,} samples (fallback)")
            else:
                print(f"❌ Failed to get {symbol} {timeframe} data")
                
    print(f"\n📊 MULTI-TIMEFRAME DATA SUMMARY:")
    print(f"🎯 Total samples loaded: {total_samples:,}")
    print(f"🎯 Currencies: {len(all_data)}")
    
    for symbol in all_data:
        print(f"   🔹 {symbol}: {list(all_data[symbol].keys())} timeframes")
        for tf in all_data[symbol]:
            print(f"      - {tf}: {len(all_data[symbol][tf]):,} samples")
    
    return all_data


def get_fallback_data(symbol, timeframe, start_date='2018-01-01', end_date='2025-03-31'):
    """Generate fallback data when pre-downloaded files aren't available"""
    
    # Try MT5 first if available
    if MT5_AVAILABLE:
        try:
            data = get_mt5_data(symbol, timeframe, start_date, end_date)
            if data is not None and len(data) > 100:
                return data[['open', 'high', 'low', 'close', 'volume']].copy()
        except Exception as e:
            print(f"⚠️ MT5 fallback failed for {symbol} {timeframe}: {e}")
    
    # Try Yahoo Finance for H1 and resample if needed
    try:
        if timeframe == 'H1':
            data = get_yahoo_data(symbol, start_date, end_date)
            if data is not None and len(data) > 100:
                return data[['open', 'high', 'low', 'close', 'volume']].copy()
        elif timeframe in ['H4', 'D1']:
            # Try to get H1 and resample up
            h1_data = get_yahoo_data(symbol, start_date, end_date)
            if h1_data is not None:
                return resample_timeframe_data(h1_data, timeframe)
    except Exception as e:
        print(f"⚠️ Yahoo Finance fallback failed for {symbol} {timeframe}: {e}")
    
    # Generate sophisticated synthetic data as last resort
    return generate_advanced_synthetic_data(symbol, timeframe, start_date, end_date)


def resample_timeframe_data(df, target_timeframe):
    """Resample data to target timeframe"""
    timeframe_rules = {
        'M5': '5T', 'M15': '15T', 'M30': '30T',
        'H1': '1H', 'H4': '4H', 'D1': '1D'
    }
    
    if target_timeframe not in timeframe_rules:
        return None
        
    rule = timeframe_rules[target_timeframe]
    
    try:
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    except Exception as e:
        print(f"⚠️ Resampling failed: {e}")
        return None


def generate_advanced_synthetic_data(symbol, timeframe, start_date='2018-01-01', end_date='2025-03-31'):
    """Generate sophisticated synthetic forex data"""
    
    # Timeframe frequency mapping
    freq_map = {
        'M5': '5T', 'M15': '15T', 'M30': '30T',
        'H1': '1H', 'H4': '4H', 'D1': '1D'
    }
    
    freq = freq_map.get(timeframe, '1H')
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_samples = len(dates)

    # Currency-specific parameters
    params = {
        "EURUSD": {"base_price": 1.1000, "volatility": 0.0003, "trend": 0.10},
        "GBPUSD": {"base_price": 1.2500, "volatility": 0.0004, "trend": -0.05},
        "USDJPY": {"base_price": 110.00, "volatility": 0.0003, "trend": 0.15},
    }

    param = params.get(symbol, params["EURUSD"])
    
    # Adjust volatility by timeframe
    vol_multiplier = {
        'M5': 0.3, 'M15': 0.5, 'M30': 0.7,
        'H1': 1.0, 'H4': 1.8, 'D1': 3.0
    }
    
    base_vol = param["volatility"] * vol_multiplier.get(timeframe, 1.0)
    np.random.seed(hash(f"{symbol}_{timeframe}") % 1000)

    # Generate regime-based market data
    regimes = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])
    
    # Add regime persistence
    for i in range(1, n_samples):
        if np.random.random() < 0.85:
            regimes[i] = regimes[i - 1]

    returns = np.zeros(n_samples)

    for i in range(n_samples):
        if regimes[i] == 0:  # Normal regime
            returns[i] = np.random.normal(0, base_vol)
        elif regimes[i] == 1:  # Trending regime
            momentum = base_vol * 0.5 if i > 0 and returns[i - 1] > 0 else -base_vol * 0.5
            returns[i] = np.random.normal(momentum, base_vol * 0.7)
        else:  # Volatile regime
            returns[i] = np.random.normal(0, base_vol * 3)

    # Add volatility clustering
    for i in range(1, n_samples):
        if abs(returns[i - 1]) > base_vol * 2:
            returns[i] *= 1.4

    # Generate realistic price series
    trend_component = np.linspace(0, param["trend"], n_samples)
    cycle_component = 0.03 * np.sin(np.linspace(0, 12 * np.pi, n_samples))
    
    prices = param["base_price"] * np.exp(
        np.cumsum(returns) + (trend_component + cycle_component) * 0.1
    )

    # Create OHLC data
    noise = np.random.normal(0, base_vol * 0.1, n_samples)
    spread = np.random.normal(0, base_vol * 0.3, n_samples)

    data = pd.DataFrame({
        "open": prices + noise,
        "high": prices + np.abs(spread) + np.abs(noise),
        "low": prices - np.abs(spread) - np.abs(noise),
        "close": prices,
        "volume": np.random.lognormal(8 + np.log(vol_multiplier.get(timeframe, 1.0)), 0.5, n_samples),
    }, index=dates)

    # Ensure OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    return data


def combine_multi_currency_features(all_data):
    """Combine features from multiple currency pairs"""
    print("\n🔗 Combining multi-currency features...")

    if not all_data:
        raise ValueError("No currency data available")

    feature_engine = EconophysicsFeatureEngine()
    combined_features = None
    combined_labels = None

    for symbol, data in all_data.items():
        print(f"🔬 Processing {symbol} features...")

        # Create features for this currency
        enriched_data = feature_engine.create_econophysics_features(data)
        labeled_data = create_professional_labels(enriched_data, symbol=symbol)

        # Prepare features
        exclude_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "future_return",
            "signal",
            "label",
        ]
        feature_columns = [
            col
            for col in labeled_data.columns
            if col not in exclude_cols and not labeled_data[col].isna().all()
        ]

        # Add currency prefix to avoid conflicts
        feature_columns_renamed = [f"{symbol}_{col}" for col in feature_columns]

        clean_data = labeled_data[feature_columns + ["label_simple"]].dropna()

        if len(clean_data) == 0:
            print(f"⚠️ No clean data for {symbol}, skipping...")
            continue

        # Rename features with currency prefix
        currency_features = clean_data[feature_columns].copy()
        currency_features.columns = feature_columns_renamed

        # Align timestamps (use intersection)
        if combined_features is None:
            combined_features = currency_features
            combined_labels = clean_data["label_simple"]
            print(f"📊 Base dataset: {symbol} ({len(combined_features)} samples)")
        else:
            # Find common timestamps
            common_idx = combined_features.index.intersection(currency_features.index)

            if len(common_idx) > 1000:  # Ensure sufficient overlap
                combined_features = combined_features.loc[common_idx]
                currency_features = currency_features.loc[common_idx]
                combined_labels = combined_labels.loc[common_idx]

                # Concatenate features
                combined_features = pd.concat(
                    [combined_features, currency_features], axis=1
                )
                print(
                    f"🔗 Added {symbol} features ({len(common_idx)} overlapping samples)"
                )
            else:
                print(
                    f"⚠️ Insufficient overlap with {symbol} ({len(common_idx)} samples), skipping..."
                )

    if combined_features is None:
        raise ValueError("No features could be combined")

    print(f"✅ Combined features shape: {combined_features.shape}")
    print(f"🎯 Total features: {len(combined_features.columns)}")

    return combined_features, combined_labels


def create_professional_labels(
    data, prediction_horizon=5, threshold=0.0001, symbol="EURUSD"
):
    """Create sophisticated multi-class labels for trading"""
    print(f"🎯 Creating professional trading labels for {symbol}...")

    df = data.copy()
    df["future_return"] = df["close"].shift(-prediction_horizon) / df["close"] - 1

    # Adjust threshold based on currency volatility
    volatility_multipliers = {
        "EURUSD": 1.0,
        "GBPUSD": 1.2,  # GBP is more volatile
        "USDJPY": 0.8,  # JPY is less volatile
    }

    adj_threshold = threshold * volatility_multipliers.get(symbol, 1.0)

    conditions = [
        df["future_return"] > adj_threshold * 2,
        df["future_return"] > adj_threshold,
        df["future_return"] < -adj_threshold * 2,
        df["future_return"] < -adj_threshold,
    ]
    choices = [2, 1, -2, -1]

    df["signal"] = np.select(conditions, choices, default=0)
    df["label"] = df["signal"] + 2
    df["label"] = np.clip(df["label"], 0, 4)

    df["label_simple"] = np.where(df["label"] >= 3, 2, np.where(df["label"] <= 1, 0, 1))

    print(f"✅ Created sophisticated trading labels for {symbol}")
    return df


def train_advanced_xgboost(X_train, y_train, X_test, y_test, feature_names, models_dir):
    """Train professional XGBoost with advanced hyperparameters"""
    print("\n🚀 Training Advanced XGBoost Model...")

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
    }

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False,
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"🎯 XGBoost Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n🔍 Top 10 Most Important Features:")
    print(feature_importance.head(10))

    model_path = os.path.join(models_dir, "advanced_xgboost_model.pkl")
    model_data = {
        "model": model,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "accuracy": accuracy,
        "params": xgb_params,
        "training_date": datetime.now().isoformat(),
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"💾 Advanced XGBoost model saved to: {model_path}")
    return accuracy, feature_importance


def train_professional_lstm(
    X_train, y_train, X_test, y_test, models_dir, time_steps=20
):
    """Train professional LSTM model"""
    print("\n🧠 Training Professional LSTM Model...")

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical

    def create_sequences(X, y, time_steps):
        X_seq, y_seq = [], []
        for i in range(time_steps, len(X)):
            X_seq.append(X[i - time_steps : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    y_train_cat = to_categorical(y_train_seq, num_classes=3)
    y_test_cat = to_categorical(y_test_seq, num_classes=3)

    model = Sequential(
        [
            LSTM(
                128, return_sequences=True, input_shape=(time_steps, X_train.shape[1])
            ),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6),
    ]

    print("🏗️ LSTM Architecture:")
    model.summary()

    model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_test_seq, y_test_cat),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"🎯 LSTM Accuracy: {accuracy:.4f}")

    model_path = os.path.join(models_dir, "professional_lstm_model.h5")
    model.save(model_path)
    print(f"💾 Professional LSTM model saved to: {model_path}")

    return accuracy


def train_advanced_cnn(X_train, y_train, X_test, y_test, models_dir, time_steps=20):
    """Train advanced CNN with dilated convolutions"""
    print("\n🌊 Training Advanced CNN Model...")

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        GlobalMaxPooling1D,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical

    def create_sequences(X, y, time_steps):
        X_seq, y_seq = [], []
        for i in range(time_steps, len(X)):
            X_seq.append(X[i - time_steps : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    y_train_cat = to_categorical(y_train_seq, num_classes=3)
    y_test_cat = to_categorical(y_test_seq, num_classes=3)

    model = Sequential(
        [
            Conv1D(
                64, 3, activation="relu", input_shape=(time_steps, X_train.shape[1])
            ),
            BatchNormalization(),
            Conv1D(64, 3, activation="relu", dilation_rate=2),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            Conv1D(128, 3, activation="relu", dilation_rate=4),
            BatchNormalization(),
            Conv1D(128, 3, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            GlobalMaxPooling1D(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=8, factor=0.5),
    ]

    print("🏗️ CNN Architecture:")
    model.summary()

    model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_test_seq, y_test_cat),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"🎯 CNN Accuracy: {accuracy:.4f}")

    model_path = os.path.join(models_dir, "advanced_cnn_model.h5")
    model.save(model_path)
    print(f"💾 Advanced CNN model saved to: {model_path}")

    return accuracy


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 Starting Professional Multi-Currency Multi-Timeframe Econophysics-Inspired FX Model Training")
    print("="*80)
    
    MODELS_DIR = '/content/drive/MyDrive/FX_Models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Configuration for multi-timeframe training
    CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Timeframe configurations for different training approaches
    TIMEFRAME_CONFIGS = {
        'quick_test': ['H1'],                            # Fast testing
        'balanced': ['H1', 'H4'],                        # Balanced approach
        'comprehensive': ['M30', 'H1', 'H4', 'D1'],      # Comprehensive training
        'maximum': ['M5', 'M15', 'M30', 'H1', 'H4', 'D1'] # Maximum data utilization
    }
    
    # Select training configuration (modify as needed)
    TRAINING_MODE = 'comprehensive'  # Change to 'maximum' for full multi-timeframe training
    SELECTED_TIMEFRAMES = TIMEFRAME_CONFIGS[TRAINING_MODE]
    
    print(f"🎯 Training Mode: {TRAINING_MODE.upper()}")
    print(f"🎯 Selected Timeframes: {SELECTED_TIMEFRAMES}")
    print(f"🎯 Currency Pairs: {CURRENCY_PAIRS}")
    
    # 1. Multi-Currency Multi-Timeframe Data Loading
    print(f"\n📊 PHASE 1: Multi-Currency Multi-Timeframe Data Acquisition")
    all_currency_data = load_multi_currency_data(
        currency_pairs=CURRENCY_PAIRS,
        timeframes=SELECTED_TIMEFRAMES
    )
    
    if not all_currency_data:
        print("❌ No currency data loaded! Exiting...")
        exit(1)
    
    # Verify all currencies have data for all selected timeframes
    valid_data = {}
    for currency in CURRENCY_PAIRS:
        if currency in all_currency_data:
            valid_data[currency] = {}
            for timeframe in SELECTED_TIMEFRAMES:
                if timeframe in all_currency_data[currency]:
                    if len(all_currency_data[currency][timeframe]) > 1000:
                        valid_data[currency][timeframe] = all_currency_data[currency][timeframe]
                        print(f"✅ {currency} {timeframe}: {len(all_currency_data[currency][timeframe]):,} samples")
                    else:
                        print(f"⚠️ {currency} {timeframe}: Insufficient data ({len(all_currency_data[currency][timeframe])} samples)")
                else:
                    print(f"❌ {currency} {timeframe}: Missing timeframe data")
    
    if not valid_data:
        print("❌ No valid multi-timeframe data! Exiting...")
        exit(1)
    
    print(f"\n✅ Successfully validated multi-timeframe data:")
    total_samples = 0
    for currency in valid_data:
        for timeframe in valid_data[currency]:
            samples = len(valid_data[currency][timeframe])
            total_samples += samples
            print(f"   📊 {currency} {timeframe}: {samples:,} samples")
    
    print(f"\n🎯 TOTAL DATASET: {total_samples:,} samples across all currencies and timeframes")
    
    # 2. Train models for each currency-timeframe combination
    print(f"\n🧠 PHASE 2: Multi-Timeframe Model Training")
    
    all_results = {}
    
    for currency in valid_data:
        all_results[currency] = {}
        
        for timeframe in valid_data[currency]:
            print(f"\n{'='*60}")
            print(f"🎯 TRAINING {currency} {timeframe} MODELS")
            print(f"{'='*60}")
            
            data = valid_data[currency][timeframe]
            
            # 2a. Feature Engineering with Econophysics
            print(f"\n🔬 PHASE 2a: Econophysics Feature Engineering for {currency} {timeframe}")
            feature_engineer = EconophysicsFeatureEngine()
            X, y = feature_engineer.create_econophysics_features(data)
            
            if X is None or len(X) < 100:
                print(f"❌ Feature engineering failed for {currency} {timeframe}")
                continue
            
            print(f"✅ Features created: {X.shape[0]:,} samples, {X.shape[1]} features")
            
            # 2b. Train Models
            print(f"\n🎯 PHASE 2b: Model Training for {currency} {timeframe}")
            trainer = ProfessionalModelTrainer()
            results = trainer.train_all_models(X, y, f"{currency}_{timeframe}")
            
            if results:
                all_results[currency][timeframe] = results
                
                # Save models with currency and timeframe info
                model_prefix = f"{currency}_{timeframe}"
                for model_name, model_data in results.items():
                    if 'model' in model_data:
                        model_filename = f"{MODELS_DIR}/{model_prefix}_{model_name}_model.pkl"
                        with open(model_filename, 'wb') as f:
                            pickle.dump(model_data['model'], f)
                        print(f"💾 Saved {model_name} model: {model_filename}")
            else:
                print(f"❌ Model training failed for {currency} {timeframe}")
    
    # 3. Comprehensive Results Analysis
    print(f"\n📊 PHASE 3: Multi-Timeframe Results Analysis")
    print(f"{'='*80}")
    
    if all_results:
        best_performers = {}
        
        for currency in all_results:
            best_performers[currency] = {}
            
            for timeframe in all_results[currency]:
                timeframe_results = all_results[currency][timeframe]
                
                print(f"\n🔹 {currency} {timeframe} RESULTS:")
                
                best_accuracy = 0
                best_model = None
                
                for model_name, metrics in timeframe_results.items():
                    accuracy = metrics.get('accuracy', 0)
                    precision = metrics.get('precision', 0)
                    
                    print(f"   📈 {model_name}:")
                    print(f"      Accuracy: {accuracy:.3f}")
                    print(f"      Precision: {precision:.3f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
                
                if best_model:
                    best_performers[currency][timeframe] = {
                        'model': best_model,
                        'accuracy': best_accuracy
                    }
                    print(f"   🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.3f})")
        
        # Overall summary
        print(f"\n{'='*80}")
        print(f"🏆 MULTI-TIMEFRAME TRAINING SUMMARY")
        print(f"{'='*80}")
        
        total_models = 0
        avg_accuracy = 0
        
        for currency in best_performers:
            print(f"\n🔸 {currency} BEST PERFORMERS:")
            for timeframe in best_performers[currency]:
                info = best_performers[currency][timeframe]
                print(f"   {timeframe}: {info['model']} (Accuracy: {info['accuracy']:.3f})")
                total_models += 1
                avg_accuracy += info['accuracy']
        
        if total_models > 0:
            avg_accuracy /= total_models
            print(f"\n📊 OVERALL STATISTICS:")
            print(f"   🎯 Total Models Trained: {total_models}")
            print(f"   📈 Average Best Accuracy: {avg_accuracy:.3f}")
            print(f"   🎯 Training Mode Used: {TRAINING_MODE}")
            print(f"   📊 Total Data Points: {total_samples:,}")
        
        # Save comprehensive results
        results_file = f"{MODELS_DIR}/multi_timeframe_results_{TRAINING_MODE}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'all_results': all_results,
                'best_performers': best_performers,
                'config': {
                    'currency_pairs': CURRENCY_PAIRS,
                    'timeframes': SELECTED_TIMEFRAMES,
                    'training_mode': TRAINING_MODE,
                    'total_samples': total_samples
                }
            }, f)
        
        print(f"\n💾 Comprehensive results saved to: {results_file}")
        
    else:
        print("❌ No successful model training results!")
    
    print(f"\n{'='*80}")
    print(f"✅ MULTI-TIMEFRAME PROFESSIONAL FX MODEL TRAINING COMPLETE!")
    print(f"{'='*80}")
