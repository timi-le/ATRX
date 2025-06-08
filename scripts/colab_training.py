#!/usr/bin/env python3
"""
Enhanced Multi-Currency, Multi-Timeframe FX Trading System - Google Colab Version
Supports combined synthetic and real MT5 data for maximum training coverage
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Scientific computing and ML
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available for deep learning models")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using traditional ML models only")

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib available for technical indicators")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available - using custom technical indicators")

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Set random seeds for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    symbols: List[str]
    timeframes: List[str]
    training_mode: str
    models: List[str]
    sequence_length: int
    prediction_horizon: int
    train_split: float
    validation_split: float
    epochs: int
    batch_size: int
    data_source: str  # 'original', 'mt5_live', or 'combined'

class EconophysicsFeatureEngineer:
    """Advanced feature engineering inspired by econophysics and market microstructure"""
    
    def __init__(self):
        self.scalers = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        if TALIB_AVAILABLE:
            # TA-Lib indicators
            df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
            df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        else:
            # Custom implementations
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def add_econophysics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add econophysics-inspired features"""
        df = df.copy()
        
        # Price returns and log returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility measures
        df['realized_volatility'] = df['log_returns'].rolling(window=20).std() * np.sqrt(252)
        df['parkinson_volatility'] = np.sqrt(252) * np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['High'] / df['Low']).rolling(window=20).mean()
        )
        
        # Market microstructure
        df['spread'] = (df['High'] - df['Low']) / df['Close']
        df['price_efficiency'] = df['Close'] / df[['Open', 'High', 'Low']].mean(axis=1)
        
        # Regime detection features
        df['volatility_regime'] = pd.cut(df['realized_volatility'], 
                                       bins=3, labels=['Low', 'Medium', 'High'])
        df['volatility_regime_num'] = pd.cut(df['realized_volatility'], 
                                          bins=3, labels=[0, 1, 2]).astype(float)
        
        # Momentum and mean reversion
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['mean_reversion'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
        
        # Volume analysis (if available)
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            df['price_volume'] = df['Close'] * df['Volume']
            df['vwap'] = (df['price_volume'].rolling(window=20).sum() / 
                         df['Volume'].rolling(window=20).sum())
        
        # Fractal and complexity measures
        df['hurst_5'] = self.calculate_hurst_exponent(df['log_returns'], window=20)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def calculate_hurst_exponent(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling Hurst exponent for fractal analysis"""
        def hurst_window(x):
            try:
                if len(x) < 10:
                    return 0.5
                lags = range(2, min(len(x)//2, 10))
                tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        return series.rolling(window=window).apply(hurst_window, raw=False)
    
    def add_target_variables(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Add prediction targets"""
        df = df.copy()
        
        # Price direction (classification)
        df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['direction'] = (df['future_return'] > 0).astype(int)
        
        # Price level (regression)
        df['future_price'] = df['Close'].shift(-horizon)
        df['price_change'] = df['future_price'] - df['Close']
        df['price_change_pct'] = df['future_return']
        
        # Volatility prediction
        df['future_volatility'] = df['log_returns'].shift(-horizon).rolling(window=5).std()
        
        # Multi-class direction (strong down, down, neutral, up, strong up)
        thresholds = [-0.002, -0.0005, 0.0005, 0.002]
        df['direction_multiclass'] = pd.cut(df['future_return'], 
                                          bins=[-np.inf] + thresholds + [np.inf],
                                          labels=[0, 1, 2, 3, 4]).astype(float)
        
        return df

class CombinedDataLoader:
    """Load and manage combined synthetic and real MT5 data"""
    
    def __init__(self, data_source: str = 'combined'):
        """
        Initialize data loader
        
        Args:
            data_source: 'original', 'mt5_live', or 'combined'
        """
        self.data_source = data_source
        self.base_paths = {
            'original': 'data/forex/',
            'mt5_live': 'data/forex/mt5_live/',
            'combined': 'data/forex/combined/'
        }
        self.loaded_data = {}
        
    def get_filename_pattern(self, symbol: str, timeframe: str) -> str:
        """Get the appropriate filename pattern based on data source"""
        if self.data_source == 'original':
            return f"{symbol}_{timeframe}_2018_2025.csv"
        elif self.data_source == 'mt5_live':
            return f"{symbol}_{timeframe}_mt5_live.csv"
        else:  # combined
            return f"{symbol}_{timeframe}_combined.csv"
    
    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data for a specific symbol and timeframe"""
        try:
            filename = self.get_filename_pattern(symbol, timeframe)
            filepath = os.path.join(self.base_paths[self.data_source], filename)
            
            # Try to download if in Colab and file doesn't exist
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                if 'COLAB_GPU' in os.environ:
                    logger.info(f"Running in Colab - please upload {filename} manually")
                return None
            
            df = pd.read_csv(filepath)
            
            # Handle different date column formats
            date_columns = ['Date', 'date', 'Time', 'time', 'Unnamed: 0']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df['Date'] = pd.to_datetime(df[date_col])
                if date_col != 'Date':
                    df = df.drop(columns=[date_col])
            else:
                # Try using index as date
                df['Date'] = pd.to_datetime(df.index)
            
            df = df.set_index('Date').sort_index()
            
            # Standardize column names
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column '{col}' in {filepath}")
                    return None
            
            # Add Volume if missing
            if 'Volume' not in df.columns:
                df['Volume'] = 1000  # Default volume
            
            # Remove any extra columns that might cause issues
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Data quality checks
            df = self.clean_data(df)
            
            logger.info(f"Loaded {symbol} {timeframe}: {len(df)} rows from {self.data_source} data")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {symbol} {timeframe} from {self.data_source}: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        df = df.copy()
        
        # Remove rows with NaN values
        initial_len = len(df)
        df = df.dropna()
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} rows with NaN values")
        
        # Remove rows with zero or negative prices
        invalid_mask = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if invalid_mask.any():
            df = df[~invalid_mask]
            logger.info(f"Removed {invalid_mask.sum()} rows with invalid prices")
        
        # Basic consistency checks
        consistency_mask = (
            (df['High'] >= df['Low']) & 
            (df['High'] >= df['Open']) & 
            (df['High'] >= df['Close']) & 
            (df['Low'] <= df['Open']) & 
            (df['Low'] <= df['Close'])
        )
        
        if not consistency_mask.all():
            df = df[consistency_mask]
            logger.info(f"Removed {(~consistency_mask).sum()} rows with inconsistent OHLC data")
        
        return df
    
    def get_data_info(self, symbol: str, timeframe: str) -> Dict:
        """Get information about the loaded data"""
        df = self.load_data(symbol, timeframe)
        if df is None:
            return {}
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data_source': self.data_source,
            'rows': len(df),
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'date_range_days': (df.index.max() - df.index.min()).days,
            'columns': list(df.columns)
        }

class MultiTimeframeTrainer:
    """Enhanced trainer supporting multiple data sources and architectures"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_loader = CombinedDataLoader(config.data_source)
        self.feature_engineer = EconophysicsFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_dataset(self, symbol: str, timeframe: str) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        """Prepare dataset for training"""
        logger.info(f"Preparing dataset for {symbol} {timeframe} using {self.config.data_source} data")
        
        # Load data
        df = self.data_loader.load_data(symbol, timeframe)
        if df is None:
            return None
        
        logger.info(f"Loaded {len(df)} samples for {symbol} {timeframe}")
        
        # Feature engineering
        df = self.feature_engineer.add_technical_indicators(df)
        df = self.feature_engineer.add_econophysics_features(df)
        df = self.feature_engineer.add_target_variables(df, self.config.prediction_horizon)
        
        # Remove rows with NaN values (from feature engineering)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"After feature engineering: {len(df)} samples (removed {initial_len - len(df)} NaN rows)")
        
        if len(df) < 100:
            logger.error(f"Insufficient data for {symbol} {timeframe}: {len(df)} samples")
            return None
        
        # Prepare features and targets
        feature_cols = [col for col in df.columns 
                       if not col.startswith('future_') and 
                          col not in ['direction', 'direction_multiclass', 'price_change', 'price_change_pct']]
        
        X = df[feature_cols].values
        y_direction = df['direction'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y_direction.astype(int))}")
        
        return X, y_direction, df
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        
        for i in range(self.config.sequence_length, len(X)):
            X_seq.append(X[i-self.config.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.2),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.LSTM(32, dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build CNN model for time series"""
        model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.Dropout(0.2),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.Dropout(0.2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_symbol_timeframe(self, symbol: str, timeframe: str) -> Dict:
        """Train models for a specific symbol and timeframe"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol} {timeframe} - {self.config.data_source} data")
        logger.info(f"{'='*60}")
        
        # Prepare dataset
        data_result = self.prepare_dataset(symbol, timeframe)
        if data_result is None:
            return {'error': f'Failed to prepare dataset for {symbol} {timeframe}'}
        
        X, y, df = data_result
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler
        scaler_key = f"{symbol}_{timeframe}"
        self.scalers[scaler_key] = scaler
        
        # Train-test split
        if self.config.training_mode == 'quick_test':
            # Use only recent data for quick testing
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[-2000:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[-2000:split_idx], y[split_idx:]
        else:
            split_idx = int(len(X_scaled) * self.config.train_split)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        results = {'symbol': symbol, 'timeframe': timeframe, 'data_source': self.config.data_source}
        
        # Train models
        for model_name in self.config.models:
            try:
                logger.info(f"\nTraining {model_name}...")
                
                if model_name == 'xgboost':
                    model = xgb.XGBClassifier(
                        n_estimators=100 if self.config.training_mode == 'quick_test' else 200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                elif model_name == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=50 if self.config.training_mode == 'quick_test' else 100,
                        max_depth=10,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                elif model_name in ['lstm', 'cnn'] and TENSORFLOW_AVAILABLE:
                    # Create sequences for deep learning
                    X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
                    X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
                    
                    if len(X_train_seq) == 0:
                        logger.warning(f"Insufficient data for sequence creation in {model_name}")
                        continue
                    
                    if model_name == 'lstm':
                        model = self.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
                    else:  # cnn
                        model = self.build_cnn_model((X_train_seq.shape[1], X_train_seq.shape[2]))
                    
                    # Callbacks
                    early_stopping = callbacks.EarlyStopping(
                        monitor='val_loss', patience=10, restore_best_weights=True
                    )
                    reduce_lr = callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
                    )
                    
                    # Train
                    epochs = min(self.config.epochs, 20 if self.config.training_mode == 'quick_test' else self.config.epochs)
                    history = model.fit(
                        X_train_seq, y_train_seq,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=self.config.batch_size,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                    
                    # Predictions
                    y_pred_proba = model.predict(X_test_seq).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    y_test = y_test_seq
                    
                else:
                    logger.warning(f"Model {model_name} not available or not supported")
                    continue
                
                # Calculate metrics
                accuracy = np.mean(y_pred == y_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Classification report
                try:
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    precision = class_report['1']['precision']
                    recall = class_report['1']['recall']
                    f1 = class_report['1']['f1-score']
                except:
                    precision = recall = f1 = 0.0
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mse': mse,
                    'mae': mae,
                    'predictions': y_pred.tolist()[:100],  # Store first 100 predictions
                    'probabilities': y_pred_proba.tolist()[:100] if hasattr(y_pred_proba, 'tolist') else []
                }
                
                # Store model
                model_key = f"{symbol}_{timeframe}_{model_name}"
                self.models[model_key] = model
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} for {symbol} {timeframe}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def train_all(self) -> Dict:
        """Train models for all symbol-timeframe combinations"""
        logger.info(f"\n{'='*80}")
        logger.info(f"ENHANCED MULTI-CURRENCY FX TRAINING - {self.config.data_source.upper()} DATA")
        logger.info(f"{'='*80}")
        logger.info(f"Training Mode: {self.config.training_mode}")
        logger.info(f"Models: {', '.join(self.config.models)}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Timeframes: {', '.join(self.config.timeframes)}")
        logger.info(f"Data Source: {self.config.data_source}")
        
        all_results = {
            'config': {
                'training_mode': self.config.training_mode,
                'data_source': self.config.data_source,
                'models': self.config.models,
                'symbols': self.config.symbols,
                'timeframes': self.config.timeframes,
                'timestamp': datetime.now().isoformat()
            },
            'results': {},
            'summary': {}
        }
        
        total_combinations = len(self.config.symbols) * len(self.config.timeframes)
        current_combination = 0
        
        for symbol in self.config.symbols:
            all_results['results'][symbol] = {}
            
            for timeframe in self.config.timeframes:
                current_combination += 1
                logger.info(f"\nProgress: {current_combination}/{total_combinations}")
                
                # Get data info
                data_info = self.data_loader.get_data_info(symbol, timeframe)
                logger.info(f"Data info: {data_info}")
                
                # Train models
                result = self.train_symbol_timeframe(symbol, timeframe)
                result['data_info'] = data_info
                all_results['results'][symbol][timeframe] = result
        
        # Generate summary
        all_results['summary'] = self.generate_summary(all_results['results'])
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def generate_summary(self, results: Dict) -> Dict:
        """Generate training summary"""
        summary = {
            'total_combinations': 0,
            'successful_combinations': 0,
            'failed_combinations': 0,
            'model_performance': {},
            'best_performers': {},
            'data_coverage': {}
        }
        
        all_accuracies = {model: [] for model in self.config.models}
        
        for symbol, timeframes in results.items():
            for timeframe, result in timeframes.items():
                summary['total_combinations'] += 1
                
                if 'error' in result:
                    summary['failed_combinations'] += 1
                    continue
                
                summary['successful_combinations'] += 1
                
                # Collect data coverage info
                data_info = result.get('data_info', {})
                if data_info:
                    key = f"{symbol}_{timeframe}"
                    summary['data_coverage'][key] = {
                        'rows': data_info.get('rows', 0),
                        'date_range_days': data_info.get('date_range_days', 0),
                        'start_date': str(data_info.get('start_date', '')),
                        'end_date': str(data_info.get('end_date', ''))
                    }
                
                # Collect model performances
                for model_name in self.config.models:
                    if model_name in result and 'accuracy' in result[model_name]:
                        accuracy = result[model_name]['accuracy']
                        all_accuracies[model_name].append(accuracy)
        
        # Calculate model performance statistics
        for model_name, accuracies in all_accuracies.items():
            if accuracies:
                summary['model_performance'][model_name] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'successful_trainings': len(accuracies)
                }
        
        # Find best performers
        for model_name, perf in summary['model_performance'].items():
            if not summary['best_performers'] or perf['mean_accuracy'] > summary['best_performers']['mean_accuracy']:
                summary['best_performers'] = {
                    'model': model_name,
                    'mean_accuracy': perf['mean_accuracy']
                }
        
        return summary
    
    def save_results(self, results: Dict) -> None:
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"training_results_{self.config.data_source}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save models and scalers
        models_file = f"models_{self.config.data_source}_{timestamp}.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'config': self.config
            }, f)
        
        logger.info(f"Models saved to {models_file}")

def get_training_mode_config(mode: str) -> dict:
    """Get configuration based on training mode"""
    configs = {
        'quick_test': {
            'epochs': 5,
            'batch_size': 64,
            'models': ['xgboost'],
            'timeframes': ['H1']
        },
        'balanced': {
            'epochs': 20,
            'batch_size': 32,
            'models': ['xgboost', 'random_forest'],
            'timeframes': ['M15', 'H1', 'H4']
        },
        'comprehensive': {
            'epochs': 50,
            'batch_size': 32,
            'models': ['xgboost', 'random_forest', 'lstm'],
            'timeframes': ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        },
        'maximum': {
            'epochs': 100,
            'batch_size': 16,
            'models': ['xgboost', 'random_forest', 'lstm', 'cnn'],
            'timeframes': ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        }
    }
    
    return configs.get(mode, configs['balanced'])

def main():
    """Main training function optimized for Google Colab"""
    print("="*80)
    print("ENHANCED MULTI-CURRENCY FX TRADING SYSTEM")
    print("Combined Synthetic + Real MT5 Data Training")
    print("="*80)
    
    # Configuration
    TRAINING_MODE = 'balanced'  # quick_test, balanced, comprehensive, maximum
    DATA_SOURCE = 'combined'    # original, mt5_live, combined
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Get mode-specific configuration
    mode_config = get_training_mode_config(TRAINING_MODE)
    
    # Create training configuration
    config = TrainingConfig(
        symbols=SYMBOLS,
        timeframes=mode_config['timeframes'],
        training_mode=TRAINING_MODE,
        models=mode_config['models'],
        sequence_length=20,
        prediction_horizon=1,
        train_split=0.8,
        validation_split=0.2,
        epochs=mode_config['epochs'],
        batch_size=mode_config['batch_size'],
        data_source=DATA_SOURCE
    )
    
    print(f"\nTraining Configuration:")
    print(f"├── Mode: {config.training_mode}")
    print(f"├── Data Source: {config.data_source}")
    print(f"├── Symbols: {', '.join(config.symbols)}")
    print(f"├── Timeframes: {', '.join(config.timeframes)}")
    print(f"├── Models: {', '.join(config.models)}")
    print(f"├── Epochs: {config.epochs}")
    print(f"└── Batch Size: {config.batch_size}")
    
    # Check if in Colab
    if 'COLAB_GPU' in os.environ:
        print(f"\n🔥 Running in Google Colab!")
        print(f"📁 Please ensure your {config.data_source} data files are uploaded to:")
        print(f"   data/forex/{config.data_source}/")
        print(f"   Required files: {len(config.symbols)} symbols × {len(config.timeframes)} timeframes = {len(config.symbols) * len(config.timeframes)} files")
    
    # Create trainer
    trainer = MultiTimeframeTrainer(config)
    
    # Run training
    print(f"\n🚀 Starting enhanced training with {config.data_source} data...")
    results = trainer.train_all()
    
    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED - SUMMARY")
    print(f"{'='*80}")
    
    summary = results['summary']
    print(f"Total combinations: {summary['total_combinations']}")
    print(f"Successful: {summary['successful_combinations']}")
    print(f"Failed: {summary['failed_combinations']}")
    
    if summary['model_performance']:
        print(f"\nModel Performance:")
        for model, perf in summary['model_performance'].items():
            print(f"├── {model}: {perf['mean_accuracy']:.4f} ± {perf['std_accuracy']:.4f}")
    
    if summary['best_performers']:
        print(f"\nBest Performer: {summary['best_performers']['model']} ({summary['best_performers']['mean_accuracy']:.4f})")
    
    print(f"\nData Coverage Summary:")
    total_rows = sum(info['rows'] for info in summary['data_coverage'].values())
    print(f"├── Total training samples: {total_rows:,}")
    print(f"├── Data source: {config.data_source}")
    
    if config.data_source == 'combined':
        print(f"└── Enhanced with real MT5 + synthetic data!")
    
    print(f"\n✅ Training completed successfully!")
    print(f"📊 Results saved with timestamp")
    print(f"🎯 Models ready for forex trading predictions!")
    
    return results

if __name__ == "__main__":
    results = main()
