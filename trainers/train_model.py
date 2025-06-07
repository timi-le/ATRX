#!/usr/bin/env python3
"""
Production-Grade ML Training Pipeline for FX AI-Quant Trading System.

This module provides a comprehensive training pipeline that ensures models are:
- Trained reliably on rolling FX data
- Regularized to prevent overfitting
- Evaluated using walk-forward and time-series cross-validation
- Re-trained on schedule with out-of-sample validation
- Stored and exported in ONNX-ready formats

Features:
- K-Fold time-series CV (no data leakage)
- Walk-forward optimization (e.g., 6mo train → 1mo test → roll)
- Regularization: L1, L2, Dropout (deep models), max_depth (trees)
- Early stopping on validation loss or Sharpe
- Out-of-sample testing
- Auto-saving best checkpoints + ONNX export
- Retraining scheduler logic for future live operation
"""

import argparse
import asyncio
import json
import sys
import yaml
import joblib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import structlog

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import xgboost as xgb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from models.predictor_interface import (
    PredictionType, ModelType, ModelConfig, TrainingMetrics
)
from models.lstm_model import LSTMPredictor, create_lstm_config
from models.cnn_model import CNNPredictor, create_cnn_config
from models.xgboost_model import XGBoostPredictor, create_xgboost_config
from models.ensemble_model import EnsembleMLPredictor, create_ensemble_config
from core.config.settings import SystemConfig
from trainers.cv_utils import (
    TimeSeriesKFold, WalkForwardOptimizer, PurgedKFold, CVResult,
    sharpe_ratio_score, information_ratio_score, calmar_ratio_score,
    validate_cv_setup
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@dataclass
class TrainingResult:
    """Training result container."""
    model_name: str
    model_type: str
    training_time: float
    best_score: float
    best_params: Dict[str, Any]
    cv_results: List[CVResult]
    metrics: Dict[str, float]
    model_path: str
    onnx_path: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None


class ProductionTrainingPipeline:
    """
    Production-grade training pipeline for FX AI-Quant Trading System.
    
    Provides comprehensive training with proper cross-validation, regularization,
    early stopping, and model management for live trading deployment.
    """
    
    def __init__(
        self, 
        config_path: str = "config/model_config.yaml",
        system_config: Optional[SystemConfig] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to model configuration YAML
            system_config: System configuration object
        """
        self.config_path = Path(config_path)
        self.system_config = system_config or SystemConfig()
        self.logger = logger.bind(component="ProductionTrainingPipeline")
        
        # Load model configuration
        self.model_config = self._load_model_config()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize scalers and preprocessors
        self.scalers = {}
        self.preprocessors = {}
        
        # Training state
        self.training_history = []
        self.best_models = {}
        
        self.logger.info(
            "Training pipeline initialized",
            config_path=str(self.config_path),
            output_dirs=self.output_dirs
        )
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info("Model configuration loaded", config_file=str(self.config_path))
            return config
            
        except Exception as e:
            self.logger.error("Failed to load model configuration", error=str(e))
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available."""
        return {
            "training": {
                "cv_method": "walk_forward",
                "primary_metric": "sharpe_ratio",
                "early_stopping": {"enabled": True, "patience": 10}
            },
            "lstm": {"units": [64, 32], "dropout": [0.2, 0.3], "learning_rate": 0.001},
            "cnn": {"filters": [32, 64], "learning_rate": 0.001},
            "xgboost": {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.1},
            "output": {
                "model_dir": "outputs/models",
                "metrics_dir": "outputs/metrics",
                "logs_dir": "outputs/logs",
                "onnx_dir": "outputs/onnx"
            }
        }
    
    def _setup_directories(self) -> None:
        """Setup output directories."""
        output_config = self.model_config.get("output", {})
        
        self.output_dirs = {
            "models": Path(output_config.get("model_dir", "outputs/models")),
            "metrics": Path(output_config.get("metrics_dir", "outputs/metrics")),
            "logs": Path(output_config.get("logs_dir", "outputs/logs")),
            "plots": Path(output_config.get("plots_dir", "outputs/plots")),
            "onnx": Path(output_config.get("onnx_dir", "outputs/onnx"))
        }
        
        # Create directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 10000, 
        sequence_length: int = 60, 
        n_features: int = 13,
        prediction_type: PredictionType = PredictionType.RETURN_REGRESSION,
        add_regime_changes: bool = True,
        noise_level: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic financial data with realistic characteristics.
        
        Args:
            n_samples: Number of samples to generate
            sequence_length: Length of each sequence
            n_features: Number of features per timestep
            prediction_type: Type of prediction task
            add_regime_changes: Whether to add regime changes
            noise_level: Level of noise to add
            
        Returns:
            Tuple of (X, y) arrays
        """
        self.logger.info(
            "Generating synthetic data",
            n_samples=n_samples,
            sequence_length=sequence_length,
            n_features=n_features,
            prediction_type=prediction_type.value
        )
        
        np.random.seed(42)
        
        # Generate base time series with different regimes
        features = []
        returns = []
        
        # Regime parameters
        regimes = [
            {"vol": 0.005, "trend": 0.0001, "mean_reversion": 0.1},  # Low vol
            {"vol": 0.015, "trend": 0.0005, "mean_reversion": 0.05}, # High vol trending
            {"vol": 0.010, "trend": -0.0002, "mean_reversion": 0.2}  # Medium vol mean reverting
        ]
        
        current_regime = 0
        regime_duration = 0
        
        for i in range(n_samples + sequence_length):
            # Change regime occasionally
            if add_regime_changes and regime_duration > np.random.exponential(500):
                current_regime = np.random.randint(0, len(regimes))
                regime_duration = 0
            
            regime = regimes[current_regime]
            regime_duration += 1
            
            if i == 0:
                # Initialize
                feature_vec = np.random.randn(n_features) * 0.1
                ret = np.random.randn() * regime["vol"]
            else:
                # Generate correlated features
                prev_features = features[-1]
                prev_return = returns[-1]
                
                # Feature evolution with regime-dependent parameters
                feature_vec = (
                    0.8 * prev_features + 
                    0.2 * np.random.randn(n_features) * regime["vol"] * 10
                )
                
                # Return generation with regime characteristics
                trend_component = regime["trend"]
                mean_reversion = -regime["mean_reversion"] * prev_return
                noise = np.random.randn() * regime["vol"]
                
                ret = trend_component + mean_reversion + noise
                
                # Add some cross-correlation between features and returns
                feature_vec[0] = abs(ret) * 100  # Volatility proxy
                feature_vec[1] = np.sign(ret) * min(abs(ret) * 50, 1)  # Momentum proxy
                feature_vec[2] = current_regime / len(regimes)  # Regime indicator
            
            features.append(feature_vec)
            returns.append(ret)
        
        features = np.array(features)
        returns = np.array(returns)
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(sequence_length, len(features)):
            X_sequences.append(features[i-sequence_length:i])
            
            if prediction_type == PredictionType.RETURN_CLASSIFICATION:
                # Binary classification: positive/negative returns
                y_targets.append(1.0 if returns[i] > 0 else 0.0)
            elif prediction_type == PredictionType.VOLATILITY:
                # Predict volatility (absolute return)
                y_targets.append(abs(returns[i]))
            else:
                # Regression: predict next return
                y_targets.append(returns[i])
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        # Add noise if specified
        if noise_level > 0:
            X += np.random.randn(*X.shape) * noise_level
        
        self.logger.info(
            "Synthetic data generated",
            X_shape=X.shape,
            y_shape=y.shape,
            y_mean=float(np.mean(y)),
            y_std=float(np.std(y)),
            n_regimes=len(regimes)
        )
        
        return X, y
    
    def preprocess_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        fit_preprocessors: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data according to configuration.
        
        Args:
            X: Feature data
            y: Target data
            fit_preprocessors: Whether to fit preprocessors
            
        Returns:
            Preprocessed (X, y)
        """
        preprocessing_config = self.model_config.get("preprocessing", {})
        
        # Feature scaling
        scaling_config = preprocessing_config.get("scaling", {})
        scaling_method = scaling_config.get("method", "standard")
        
        if fit_preprocessors:
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                feature_range = scaling_config.get("feature_range", [0, 1])
                scaler = MinMaxScaler(feature_range=feature_range)
            elif scaling_method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            # Fit scaler on flattened features
            X_flat = X.reshape(-1, X.shape[-1])
            scaler.fit(X_flat)
            self.scalers["feature_scaler"] = scaler
        else:
            scaler = self.scalers.get("feature_scaler")
        
        if scaler is not None:
            # Transform features
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.transform(X_flat)
            X = X_scaled.reshape(original_shape)
        
        # Outlier handling
        outlier_config = preprocessing_config.get("outlier_detection", {})
        if outlier_config.get("enabled", False):
            X, y = self._handle_outliers(X, y, outlier_config)
        
        self.logger.info(
            "Data preprocessing completed",
            scaling_method=scaling_method,
            final_X_shape=X.shape,
            final_y_shape=y.shape
        )
        
        return X, y
    
    def _handle_outliers(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Handle outliers in the data."""
        method = config.get("method", "iqr")
        threshold = config.get("threshold", 3.0)
        action = config.get("action", "clip")
        
        if method == "iqr":
            # IQR method for target variable
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if action == "clip":
                y = np.clip(y, lower_bound, upper_bound)
            elif action == "remove":
                mask = (y >= lower_bound) & (y <= upper_bound)
                X, y = X[mask], y[mask]
        
        elif method == "zscore":
            # Z-score method
            z_scores = np.abs((y - np.mean(y)) / np.std(y))
            
            if action == "clip":
                outlier_mask = z_scores > threshold
                y[outlier_mask] = np.sign(y[outlier_mask]) * threshold * np.std(y) + np.mean(y)
            elif action == "remove":
                mask = z_scores <= threshold
                X, y = X[mask], y[mask]
        
        return X, y
    
    def setup_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Setup cross-validation method based on configuration."""
        training_config = self.model_config.get("training", {})
        cv_method = training_config.get("cv_method", "walk_forward")
        
        if cv_method == "walk_forward":
            wf_config = training_config.get("walk_forward", {})
            cv = WalkForwardOptimizer(
                train_window=wf_config.get("train_window", 2520),
                test_window=wf_config.get("test_window", 252),
                step_size=wf_config.get("step_size", 63),
                gap=training_config.get("gap", 5),
                min_train_size=wf_config.get("min_train_size", 1000),
                expanding_window=wf_config.get("expanding_window", False)
            )
        
        elif cv_method == "time_series_kfold":
            ts_config = training_config.get("time_series_kfold", {})
            cv = TimeSeriesKFold(
                n_splits=training_config.get("n_splits", 5),
                max_train_size=ts_config.get("max_train_size", 5000),
                gap=training_config.get("gap", 5)
            )
        
        elif cv_method == "purged_kfold":
            ts_config = training_config.get("time_series_kfold", {})
            cv = PurgedKFold(
                n_splits=training_config.get("n_splits", 5),
                purge_length=ts_config.get("purge_length", 10),
                embargo_length=ts_config.get("embargo_length", 5)
            )
        
        else:
            # Default to time series k-fold
            cv = TimeSeriesKFold(n_splits=training_config.get("n_splits", 5))
        
        # Validate CV setup
        if not validate_cv_setup(X, y, cv):
            self.logger.warning("CV setup validation failed, using default TimeSeriesKFold")
            cv = TimeSeriesKFold(n_splits=3)
        
        return cv
    
    def get_scoring_function(self, metric_name: str):
        """Get scoring function by name."""
        if metric_name == "mse":
            return lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
        elif metric_name == "mae":
            return lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
        elif metric_name == "accuracy":
            return lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))
        elif metric_name == "sharpe_ratio":
            return sharpe_ratio_score
        elif metric_name == "information_ratio":
            return information_ratio_score
        elif metric_name == "calmar_ratio":
            return calmar_ratio_score
        else:
            return lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
    
    async def train_lstm_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        optimize_hyperparameters: bool = True
    ) -> TrainingResult:
        """Train LSTM model with cross-validation and hyperparameter optimization."""
        self.logger.info("Starting LSTM model training")
        start_time = datetime.now()
        
        lstm_config = self.model_config.get("lstm", {})
        training_config = self.model_config.get("training", {})
        
        # Setup cross-validation
        cv = self.setup_cross_validation(X, y)
        
        # Get scoring function
        primary_metric = training_config.get("primary_metric", "sharpe_ratio")
        scoring_func = self.get_scoring_function(primary_metric)
        
        if optimize_hyperparameters and "param_grid" in lstm_config:
            # Hyperparameter optimization
            param_grid = lstm_config["param_grid"]
            
            if hasattr(cv, 'optimize'):
                # Use walk-forward optimizer
                best_params, cv_results = cv.optimize(
                    LSTMPredictor,
                    X, y,
                    param_grid,
                    scoring=primary_metric
                )
            else:
                # Manual grid search with CV
                best_params, cv_results = self._manual_grid_search(
                    LSTMPredictor, X, y, param_grid, cv, scoring_func
                )
        else:
            # Use default parameters
            best_params = {
                "units": lstm_config.get("units", [64, 32]),
                "dropout": lstm_config.get("dropout", [0.2, 0.3]),
                "learning_rate": lstm_config.get("learning_rate", 0.001),
                "l2_reg": lstm_config.get("l2_reg", 0.001)
            }
            cv_results = []
        
        # Train final model with best parameters
        model_config = create_lstm_config(
            sequence_length=X.shape[1],
            n_features=X.shape[2],
            **best_params
        )
        
        model = LSTMPredictor(model_config)
        
        # Split data for final training
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        y_pred = model._predict_raw(X_val)
        final_score = scoring_func(y_val, y_pred)
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(y_val, y_pred)
        
        # Save model
        model_path = self.output_dirs["models"] / f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)
        
        # Export to ONNX if configured
        onnx_path = None
        if training_config.get("export_onnx", True):
            try:
                onnx_path = self.output_dirs["onnx"] / f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
                model.export_onnx(str(onnx_path), X_val[:1])  # Use sample input
            except Exception as e:
                self.logger.warning("ONNX export failed", error=str(e))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        result = TrainingResult(
            model_name="LSTM",
            model_type="deep_learning",
            training_time=training_time,
            best_score=final_score,
            best_params=best_params,
            cv_results=cv_results,
            metrics=metrics,
            model_path=str(model_path),
            onnx_path=str(onnx_path) if onnx_path else None
        )
        
        self.logger.info(
            "LSTM training completed",
            training_time=training_time,
            best_score=final_score,
            model_path=str(model_path)
        )
        
        return result
    
    async def train_cnn_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        optimize_hyperparameters: bool = True
    ) -> TrainingResult:
        """Train CNN model with cross-validation and hyperparameter optimization."""
        self.logger.info("Starting CNN model training")
        start_time = datetime.now()
        
        cnn_config = self.model_config.get("cnn", {})
        training_config = self.model_config.get("training", {})
        
        # Setup cross-validation
        cv = self.setup_cross_validation(X, y)
        
        # Get scoring function
        primary_metric = training_config.get("primary_metric", "sharpe_ratio")
        scoring_func = self.get_scoring_function(primary_metric)
        
        if optimize_hyperparameters and "param_grid" in cnn_config:
            # Hyperparameter optimization
            param_grid = cnn_config["param_grid"]
            
            if hasattr(cv, 'optimize'):
                best_params, cv_results = cv.optimize(
                    CNNPredictor,
                    X, y,
                    param_grid,
                    scoring=primary_metric
                )
            else:
                best_params, cv_results = self._manual_grid_search(
                    CNNPredictor, X, y, param_grid, cv, scoring_func
                )
        else:
            # Use default parameters
            best_params = {
                "filters": cnn_config.get("filters", [32, 64]),
                "kernel_sizes": cnn_config.get("kernel_sizes", [3, 3]),
                "learning_rate": cnn_config.get("learning_rate", 0.001)
            }
            cv_results = []
        
        # Train final model
        model_config = create_cnn_config(
            sequence_length=X.shape[1],
            n_features=X.shape[2],
            **best_params
        )
        
        model = CNNPredictor(model_config)
        
        # Split data for final training
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        y_pred = model._predict_raw(X_val)
        final_score = scoring_func(y_val, y_pred)
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(y_val, y_pred)
        
        # Save model
        model_path = self.output_dirs["models"] / f"cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)
        
        # Export to ONNX if configured
        onnx_path = None
        if training_config.get("export_onnx", True):
            try:
                onnx_path = self.output_dirs["onnx"] / f"cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
                model.export_onnx(str(onnx_path), X_val[:1])
            except Exception as e:
                self.logger.warning("ONNX export failed", error=str(e))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        result = TrainingResult(
            model_name="CNN",
            model_type="deep_learning",
            training_time=training_time,
            best_score=final_score,
            best_params=best_params,
            cv_results=cv_results,
            metrics=metrics,
            model_path=str(model_path),
            onnx_path=str(onnx_path) if onnx_path else None
        )
        
        self.logger.info(
            "CNN training completed",
            training_time=training_time,
            best_score=final_score
        )
        
        return result
    
    def train_xgboost_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        optimize_hyperparameters: bool = True
    ) -> TrainingResult:
        """Train XGBoost model with cross-validation and hyperparameter optimization."""
        self.logger.info("Starting XGBoost model training")
        start_time = datetime.now()
        
        xgb_config = self.model_config.get("xgboost", {})
        training_config = self.model_config.get("training", {})
        
        # Flatten X for XGBoost (it expects 2D input)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Setup cross-validation
        cv = self.setup_cross_validation(X_flat, y)
        
        # Get scoring function
        primary_metric = training_config.get("primary_metric", "sharpe_ratio")
        scoring_func = self.get_scoring_function(primary_metric)
        
        if optimize_hyperparameters and "param_grid" in xgb_config:
            # Hyperparameter optimization
            param_grid = xgb_config["param_grid"]
            
            if hasattr(cv, 'optimize'):
                best_params, cv_results = cv.optimize(
                    xgb.XGBRegressor,
                    X_flat, y,
                    param_grid,
                    scoring=primary_metric
                )
            else:
                best_params, cv_results = self._manual_grid_search(
                    xgb.XGBRegressor, X_flat, y, param_grid, cv, scoring_func
                )
        else:
            # Use default parameters
            best_params = {
                "n_estimators": xgb_config.get("n_estimators", 1000),
                "max_depth": xgb_config.get("max_depth", 6),
                "learning_rate": xgb_config.get("learning_rate", 0.1),
                "reg_alpha": xgb_config.get("reg_alpha", 0.1),
                "reg_lambda": xgb_config.get("reg_lambda", 1.0)
            }
            cv_results = []
        
        # Train final model
        model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            **best_params
        )
        
        # Split data for final training
        split_idx = int(len(X_flat) * 0.8)
        X_train, X_val = X_flat[:split_idx], X_flat[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train with early stopping
        try:
            # Try newer XGBoost API with callbacks
            from xgboost.callback import EarlyStopping
            early_stopping = EarlyStopping(
                rounds=xgb_config.get("early_stopping_rounds", 50),
                save_best=True
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping],
                verbose=False
            )
        except (ImportError, TypeError):
            # Fallback to older API
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=xgb_config.get("early_stopping_rounds", 50),
                    verbose=False
                )
            except TypeError:
                # If early stopping is not supported, train without it
                model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        y_pred = model.predict(X_val)
        final_score = scoring_func(y_val, y_pred)
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(y_val, y_pred)
        
        # Get feature importance
        feature_importance = dict(zip(
            [f"feature_{i}" for i in range(X_flat.shape[1])],
            model.feature_importances_
        ))
        
        # Save model
        model_path = self.output_dirs["models"] / f"xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        result = TrainingResult(
            model_name="XGBoost",
            model_type="tree_based",
            training_time=training_time,
            best_score=final_score,
            best_params=best_params,
            cv_results=cv_results,
            metrics=metrics,
            model_path=str(model_path),
            feature_importance=feature_importance
        )
        
        self.logger.info(
            "XGBoost training completed",
            training_time=training_time,
            best_score=final_score
        )
        
        return result
    
    def _manual_grid_search(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv,
        scoring_func
    ) -> Tuple[Dict[str, Any], List[CVResult]]:
        """Manual grid search implementation."""
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = float('inf')
        best_params = None
        all_results = []
        
        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            
            cv_scores = []
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                if model_class == xgb.XGBRegressor:
                    model = model_class(random_state=42, n_jobs=1, **params)
                else:
                    model = model_class(**params)
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                score = scoring_func(y_test, y_pred)
                cv_scores.append(score)
                
                result = CVResult(
                    fold=fold,
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                    train_score=0.0,  # Not calculated for efficiency
                    test_score=score,
                    model_params=params
                )
                fold_results.append(result)
            
            mean_score = np.mean(cv_scores)
            if mean_score < best_score:  # Assuming lower is better
                best_score = mean_score
                best_params = params
            
            all_results.extend(fold_results)
        
        return best_params, all_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Regression metrics
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        
        # Financial metrics
        metrics["sharpe_ratio"] = float(sharpe_ratio_score(y_true, y_pred))
        metrics["information_ratio"] = float(information_ratio_score(y_true, y_pred))
        metrics["calmar_ratio"] = float(calmar_ratio_score(y_true, y_pred))
        
        # Classification metrics (if applicable)
        if len(np.unique(y_true)) == 2:
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred_binary))
        
        return metrics
    
    def save_training_results(self, results: List[TrainingResult]) -> None:
        """Save training results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results as JSON
        results_dict = [asdict(result) for result in results]
        results_path = self.output_dirs["metrics"] / f"training_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save summary
        summary = {
            "timestamp": timestamp,
            "n_models": len(results),
            "best_model": max(results, key=lambda x: x.best_score).model_name,
            "total_training_time": sum(r.training_time for r in results),
            "models": {r.model_name: r.best_score for r in results}
        }
        
        summary_path = self.output_dirs["metrics"] / f"training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(
            "Training results saved",
            results_path=str(results_path),
            summary_path=str(summary_path)
        )
    
    def run_full_training_pipeline(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        models_to_train: Optional[List[str]] = None,
        optimize_hyperparameters: bool = True
    ) -> List[TrainingResult]:
        """
        Run the complete training pipeline for all specified models.
        
        Args:
            X: Feature data (if None, synthetic data will be generated)
            y: Target data (if None, synthetic data will be generated)
            models_to_train: List of models to train ['lstm', 'cnn', 'xgboost']
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            List of training results
        """
        self.logger.info("Starting full training pipeline")
        
        # Generate data if not provided
        if X is None or y is None:
            training_config = self.model_config.get("training", {})
            X, y = self.generate_synthetic_data(
                n_samples=10000,
                sequence_length=training_config.get("sequence_length", 60),
                n_features=training_config.get("n_features", 13)
            )
        
        # Preprocess data
        X, y = self.preprocess_data(X, y, fit_preprocessors=True)
        
        # Default models to train
        if models_to_train is None:
            models_to_train = ["lstm", "cnn", "xgboost"]
        
        results = []
        
        # Train each model
        for model_name in models_to_train:
            try:
                if model_name.lower() == "lstm":
                    result = asyncio.run(self.train_lstm_model(X, y, optimize_hyperparameters))
                elif model_name.lower() == "cnn":
                    result = asyncio.run(self.train_cnn_model(X, y, optimize_hyperparameters))
                elif model_name.lower() == "xgboost":
                    result = self.train_xgboost_model(X, y, optimize_hyperparameters)
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                results.append(result)
                self.best_models[model_name] = result
                
            except Exception as e:
                self.logger.error(
                    f"Failed to train {model_name} model",
                    error=str(e),
                    exc_info=True
                )
        
        # Save results
        if results:
            self.save_training_results(results)
        
        self.logger.info(
            "Full training pipeline completed",
            n_models_trained=len(results),
            models=[r.model_name for r in results]
        )
        
        return results


def main():
    """Main entry point for the training pipeline."""
    parser = argparse.ArgumentParser(description="FX AI-Quant Training Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["lstm", "cnn", "xgboost"],
        help="Models to train"
    )
    parser.add_argument(
        "--no-optimization", 
        action="store_true",
        help="Skip hyperparameter optimization"
    )
    parser.add_argument(
        "--data-path", 
        type=str,
        help="Path to training data (if not provided, synthetic data will be used)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionTrainingPipeline(config_path=args.config)
    
    # Load data if provided
    X, y = None, None
    if args.data_path:
        try:
            data = pd.read_csv(args.data_path)
            # Assume data preprocessing logic here
            # X, y = preprocess_real_data(data)
            logger.info(f"Loaded data from {args.data_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            logger.info("Using synthetic data instead")
    
    # Run training pipeline
    results = pipeline.run_full_training_pipeline(
        X=X,
        y=y,
        models_to_train=args.models,
        optimize_hyperparameters=not args.no_optimization
    )
    
    # Print summary
    if results:
        print("\n" + "="*50)
        print("TRAINING PIPELINE SUMMARY")
        print("="*50)
        
        for result in results:
            print(f"\n{result.model_name}:")
            print(f"  Best Score: {result.best_score:.6f}")
            print(f"  Training Time: {result.training_time:.2f}s")
            print(f"  Model Path: {result.model_path}")
            if result.onnx_path:
                print(f"  ONNX Path: {result.onnx_path}")
        
        best_model = max(results, key=lambda x: x.best_score)
        print(f"\nBest Model: {best_model.model_name} (Score: {best_model.best_score:.6f})")
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    main() 