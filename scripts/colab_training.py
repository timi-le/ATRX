"""
Professional Econophysics-Inspired FX Trading Model Training
Advanced ML approach with regime detection and sophisticated feature engineering
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


class EconophysicsFeatureEngine:
    """Professional econophysics-inspired feature engineering for FX markets"""

    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.volatility_windows = [10, 20, 50]

    def hurst_exponent(self, ts, max_lag=20):
        """Calculate Hurst exponent for trend persistence analysis"""
        try:
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5

    def fractal_dimension(self, ts, n=20):
        """Calculate fractal dimension using box counting"""
        try:
            ts_norm = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
            scales = np.logspace(0.01, 0.2, n, endpoint=False, base=10)
            counts = []

            for scale in scales:
                hist, _ = np.histogram(ts_norm, bins=int(1 / scale))
                counts.append(np.sum(hist > 0))

            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return -coeffs[0]
        except:
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

        df["price_efficiency"] = np.abs(df["close"] - df["open"]) / (
            df["high"] - df["low"]
        )
        df["bid_ask_spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        df["buying_pressure"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        df["selling_pressure"] = (df["high"] - df["close"]) / (df["high"] - df["low"])
        df["volume_price_trend"] = df["volume"] * (
            (df["close"] - df["open"]) / df["open"]
        )

        return df

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def multi_timeframe_features(self, data):
        """Create multi-timeframe technical features"""
        df = data.copy()

        # Multiple moving averages
        for window in self.lookback_periods:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
            df[f"price_vs_sma_{window}"] = (df["close"] - df[f"sma_{window}"]) / df[
                f"sma_{window}"
            ]

        # RSI with multiple periods
        for period in [14, 21, 50]:
            df[f"rsi_{period}"] = self.calculate_rsi(df["close"], period)

        # Bollinger Bands
        for window in [20, 50]:
            sma = df["close"].rolling(window).mean()
            std = df["close"].rolling(window).std()
            df[f"bb_upper_{window}"] = sma + (2 * std)
            df[f"bb_lower_{window}"] = sma - (2 * std)
            df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (
                df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
            )

        # MACD family
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
            df[f"realized_vol_{window}"] = returns.rolling(window).std() * np.sqrt(252)
            df[f"vol_of_vol_{window}"] = (
                df[f"realized_vol_{window}"].rolling(window).std()
            )

        df["volatility_regime"] = self.regime_volatility_clustering(returns)
        df["vol_momentum"] = df["realized_vol_20"] / df["realized_vol_50"]

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
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
            df[f"mean_reversion_{period}"] = (
                df["close"] - df["close"].rolling(period).mean()
            ) / df["close"].rolling(period).std()

        print(f"✅ Created {len(df.columns)} total features")
        return df


def generate_professional_forex_data(symbol="EURUSD=X", period="2y"):
    """Download real forex data or generate sophisticated synthetic data"""
    print(f"📈 Fetching professional forex data for {symbol}...")

    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1h")

        if len(data) < 1000:
            raise Exception("Insufficient real data, generating synthetic")

        data.columns = data.columns.str.lower()
        print(f"✅ Downloaded {len(data)} real market data points")
        return data

    except Exception as e:
        print(f"⚠️ Real data unavailable: {e}")
        print("🎲 Generating sophisticated synthetic forex data...")

        n_samples = 8760
        dates = pd.date_range(start="2022-01-01", periods=n_samples, freq="1H")

        np.random.seed(42)
        regimes = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])

        returns = np.zeros(n_samples)
        for i in range(n_samples):
            if regimes[i] == 0:
                returns[i] = np.random.normal(0, 0.0003)
            elif regimes[i] == 1:
                returns[i] = np.random.normal(
                    0.0001 if i > 0 and returns[i - 1] > 0 else -0.0001, 0.0002
                )
            else:
                returns[i] = np.random.normal(0, 0.001)

        for i in range(1, n_samples):
            if abs(returns[i - 1]) > 0.0005:
                returns[i] *= 1.5

        prices = 1.1000 * np.exp(np.cumsum(returns))
        noise = np.random.normal(0, 0.00002, n_samples)

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

        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        print(f"✅ Generated {len(data)} sophisticated synthetic data points")
        return data


def create_professional_labels(data, prediction_horizon=5, threshold=0.0001):
    """Create sophisticated multi-class labels for trading"""
    print("🎯 Creating professional trading labels...")

    df = data.copy()
    df["future_return"] = df["close"].shift(-prediction_horizon) / df["close"] - 1

    conditions = [
        df["future_return"] > threshold * 2,
        df["future_return"] > threshold,
        df["future_return"] < -threshold * 2,
        df["future_return"] < -threshold,
    ]
    choices = [2, 1, -2, -1]

    df["signal"] = np.select(conditions, choices, default=0)
    df["label"] = df["signal"] + 2
    df["label"] = np.clip(df["label"], 0, 4)

    df["label_simple"] = np.where(df["label"] >= 3, 2, np.where(df["label"] <= 1, 0, 1))

    print("✅ Created sophisticated trading labels")
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

    history = model.fit(
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

    history = model.fit(
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
    print("🚀 Starting Professional Econophysics-Inspired FX Model Training")
    print("=" * 80)

    MODELS_DIR = "/content/drive/MyDrive/FX_Models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Data Generation/Loading
    print("\n📊 PHASE 1: Data Acquisition")
    data = generate_professional_forex_data()

    # 2. Advanced Feature Engineering
    print("\n🔬 PHASE 2: Econophysics Feature Engineering")
    feature_engine = EconophysicsFeatureEngine()
    enriched_data = feature_engine.create_econophysics_features(data)

    # 3. Professional Label Creation
    print("\n🎯 PHASE 3: Professional Label Generation")
    labeled_data = create_professional_labels(enriched_data)

    # 4. Data Preparation
    print("\n⚙️ PHASE 4: Data Preprocessing")

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

    clean_data = labeled_data[feature_columns + ["label_simple"]].dropna()

    X = clean_data[feature_columns].values
    y = clean_data["label_simple"].values

    print(f"📈 Dataset shape: {X.shape}")
    print(f"🎯 Features: {len(feature_columns)}")
    print(f"📊 Label distribution: {np.bincount(y)}")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Professional scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training
    print("\n🤖 PHASE 5: Advanced Model Training")

    xgb_acc, feature_importance = train_advanced_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_columns, MODELS_DIR
    )

    lstm_acc = train_professional_lstm(
        X_train_scaled, y_train, X_test_scaled, y_test, MODELS_DIR
    )

    cnn_acc = train_advanced_cnn(
        X_train_scaled, y_train, X_test_scaled, y_test, MODELS_DIR
    )

    # 6. Results Summary
    print("\n" + "=" * 80)
    print("🎉 PROFESSIONAL ECONOPHYSICS TRAINING COMPLETE!")
    print("=" * 80)
    print(f"📁 Models saved to: {MODELS_DIR}")
    print(f"📊 Training samples: {len(X_train):,}")
    print(f"🧪 Test samples: {len(X_test):,}")
    print(f"🔍 Features engineered: {len(feature_columns)}")
    print("\n🏆 PERFORMANCE SUMMARY:")
    print(f"  🌲 Advanced XGBoost: {xgb_acc:.4f}")
    print(f"  🧠 Professional LSTM: {lstm_acc:.4f}")
    print(f"  🌊 Advanced CNN: {cnn_acc:.4f}")

    best_model = max(
        [("XGBoost", xgb_acc), ("LSTM", lstm_acc), ("CNN", cnn_acc)], key=lambda x: x[1]
    )
    print(f"\n🥇 Best performing model: {best_model[0]} ({best_model[1]:.4f})")

    print("\n✅ Ready for professional FX trading deployment!")
    print("=" * 80)
