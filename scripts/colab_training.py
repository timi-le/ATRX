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


def load_multi_currency_data(symbols=["EURUSD", "GBPUSD", "USDJPY"]):
    """Load data for multiple currency pairs"""
    print(f"🌍 Loading multi-currency data: {symbols}")

    all_data = {}

    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"📊 LOADING {symbol} DATA")
        print(f"{'='*40}")

        # Try pre-downloaded data first
        data = load_predownloaded_data(symbol)

        # Fallback to live download if needed
        if data is None:
            print(f"🔄 Falling back to live download for {symbol}...")
            data = generate_professional_forex_data(symbol, "H1")

        if data is not None and len(data) > 1000:
            all_data[symbol] = data
            print(f"✅ {symbol} data loaded successfully: {len(data):,} samples")
        else:
            print(f"❌ Failed to load {symbol} data")

    return all_data


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
    print(
        "🚀 Starting Professional Multi-Currency Econophysics-Inspired FX Model Training"
    )
    print("=" * 80)

    MODELS_DIR = "/content/drive/MyDrive/FX_Models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Currency pairs to train on
    CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

    # 1. Multi-Currency Data Loading
    print("\n📊 PHASE 1: Multi-Currency Data Acquisition")
    all_currency_data = load_multi_currency_data(CURRENCY_PAIRS)

    if not all_currency_data:
        print("❌ No currency data loaded! Exiting...")
        exit(1)

    print(f"\n✅ Successfully loaded {len(all_currency_data)} currency pairs:")
    for symbol, data in all_currency_data.items():
        print(
            f"   📈 {symbol}: {len(data):,} samples ({data.index.min()} to {data.index.max()})"
        )

    # 2. Multi-Currency Feature Engineering & Combination
    print("\n🔬 PHASE 2: Multi-Currency Econophysics Feature Engineering")
    combined_features, combined_labels = combine_multi_currency_features(
        all_currency_data
    )

    # 3. Data Preparation
    print("\n⚙️ PHASE 3: Multi-Currency Data Preprocessing")

    X = combined_features.values
    y = combined_labels.values

    # Fix the np.bincount error by ensuring y is proper integer array
    y = y.astype(int)
    y = y[~np.isnan(y)]  # Remove any NaN values
    X = X[: len(y)]  # Ensure X and y have same length

    print(f"📈 Combined dataset shape: {X.shape}")
    print(f"🎯 Total features: {combined_features.shape[1]}")
    print(f"📊 Label distribution: {np.bincount(y)}")
    print(f"🌍 Currency pairs: {list(all_currency_data.keys())}")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Professional scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for future use
    scaler_path = os.path.join(MODELS_DIR, "multi_currency_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {scaler_path}")

    # 4. Advanced Model Training
    print("\n🤖 PHASE 4: Advanced Multi-Currency Model Training")

    feature_names = list(combined_features.columns)

    xgb_acc, feature_importance = train_advanced_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_names, MODELS_DIR
    )

    lstm_acc = train_professional_lstm(
        X_train_scaled, y_train, X_test_scaled, y_test, MODELS_DIR
    )

    cnn_acc = train_advanced_cnn(
        X_train_scaled, y_train, X_test_scaled, y_test, MODELS_DIR
    )

    # 5. Multi-Currency Feature Analysis
    print("\n📊 PHASE 5: Multi-Currency Feature Analysis")

    # Analyze feature importance by currency
    currency_importance = {}
    for currency in CURRENCY_PAIRS:
        currency_features = [f for f in feature_names if f.startswith(f"{currency}_")]
        currency_imp = feature_importance[
            feature_importance["feature"].isin(currency_features)
        ]["importance"].sum()
        currency_importance[currency] = currency_imp
        print(f"🔍 {currency} total importance: {currency_imp:.4f}")

    # Save comprehensive training metadata
    training_metadata = {
        "currency_pairs": CURRENCY_PAIRS,
        "training_date": datetime.now().isoformat(),
        "data_samples": {
            symbol: len(data) for symbol, data in all_currency_data.items()
        },
        "combined_features": len(feature_names),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "model_performance": {"xgboost": xgb_acc, "lstm": lstm_acc, "cnn": cnn_acc},
        "currency_importance": currency_importance,
        "feature_names": feature_names,
    }

    metadata_path = os.path.join(MODELS_DIR, "training_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(training_metadata, f)

    # 6. Results Summary
    print("\n" + "=" * 80)
    print("🎉 PROFESSIONAL MULTI-CURRENCY ECONOPHYSICS TRAINING COMPLETE!")
    print("=" * 80)
    print(f"📁 Models saved to: {MODELS_DIR}")
    print(f"🌍 Currency pairs trained: {', '.join(CURRENCY_PAIRS)}")
    print(f"📊 Training samples: {len(X_train):,}")
    print(f"🧪 Test samples: {len(X_test):,}")
    print(f"🔍 Total features: {len(feature_names):,}")

    print("\n🏆 PERFORMANCE SUMMARY:")
    print(f"  🌲 Advanced XGBoost: {xgb_acc:.4f}")
    print(f"  🧠 Professional LSTM: {lstm_acc:.4f}")
    print(f"  🌊 Advanced CNN: {cnn_acc:.4f}")

    best_model = max(
        [("XGBoost", xgb_acc), ("LSTM", lstm_acc), ("CNN", cnn_acc)], key=lambda x: x[1]
    )
    print(f"\n🥇 Best performing model: {best_model[0]} ({best_model[1]:.4f})")

    print("\n📊 CURRENCY IMPORTANCE RANKING:")
    sorted_currencies = sorted(
        currency_importance.items(), key=lambda x: x[1], reverse=True
    )
    for i, (currency, importance) in enumerate(sorted_currencies, 1):
        print(f"  {i}. {currency}: {importance:.4f}")

    print(f"\n💾 Training metadata saved to: {metadata_path}")
    print("\n✅ Ready for professional multi-currency FX trading deployment!")
    print("=" * 80)
