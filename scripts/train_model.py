#!/usr/bin/env python3
"""
Model Training Script for FX AI-Quant Trading System.

This script provides a comprehensive training pipeline for ML models including
LSTM, CNN, XGBoost, and ensemble models with proper data preparation,
training, evaluation, and model saving.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.predictor_interface import (
    PredictionType, ModelType, ModelConfig, TrainingMetrics
)
from models.lstm_model import LSTMPredictor, create_lstm_config
from models.cnn_model import CNNPredictor, create_cnn_config
from models.ensemble_model import (
    XGBoostPredictor, EnsembleMLPredictor, 
    create_xgboost_config, create_ensemble_config
)
from core.config.settings import SystemConfig


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


class ModelTrainer:
    """Main model training class."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logger.bind(component="ModelTrainer")
        
        # Create model directories
        self.model_dir = Path(config.ml.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ONNX directory
        self.onnx_dir = Path(config.ml.model_path) / "onnx"
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 10000, 
        sequence_length: int = 60, 
        n_features: int = 13,
        prediction_type: PredictionType = PredictionType.RETURN_REGRESSION
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic financial data for testing."""
        self.logger.info(
            "Generating synthetic data",
            n_samples=n_samples,
            sequence_length=sequence_length,
            n_features=n_features,
            prediction_type=prediction_type.value
        )
        
        # Generate random walk-like price data
        np.random.seed(42)
        
        # Create base features (technical indicators, volatility, etc.)
        features = []
        returns = []
        
        for i in range(n_samples + sequence_length):
            # Generate correlated features that might represent:
            # ATR, BB_width, realized_vol, vol_ratio, MACD, RSI, etc.
            if i == 0:
                feature_vec = np.random.randn(n_features) * 0.1
                ret = np.random.randn() * 0.01
            else:
                # Add some autocorrelation and cross-correlation
                feature_vec = 0.7 * features[-1] + 0.3 * np.random.randn(n_features) * 0.1
                ret = 0.1 * returns[-1] + 0.9 * np.random.randn() * 0.01
                
                # Add some regime-like behavior
                if i % 500 < 100:  # High volatility regime
                    feature_vec *= 2.0
                    ret *= 2.0
                elif i % 500 < 200:  # Trending regime
                    ret += 0.001 * np.sign(ret)
            
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
        
        self.logger.info(
            "Synthetic data generated",
            X_shape=X.shape,
            y_shape=y.shape,
            y_mean=float(np.mean(y)),
            y_std=float(np.std(y))
        )
        
        return X, y
    
    def load_real_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load real financial data (placeholder for actual implementation)."""
        # This would load actual market data and features
        # For now, use synthetic data
        self.logger.warning("Real data loading not implemented, using synthetic data")
        return self.generate_synthetic_data()
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        validation_size: float = 0.2,
        time_series_split: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with proper time series splits."""
        
        if time_series_split:
            # For time series, use chronological splits
            n_samples = len(X)
            
            # Calculate split indices
            train_end = int(n_samples * (1 - test_size - validation_size))
            val_end = int(n_samples * (1 - test_size))
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            
            X_test = X[val_end:]
            y_test = y[val_end:]
            
        else:
            # Random splits (not recommended for time series)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size/(1-test_size), random_state=42
            )
        
        self.logger.info(
            "Data prepared",
            train_shape=X_train.shape,
            val_shape=X_val.shape,
            test_shape=X_test.shape,
            time_series_split=time_series_split
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_lstm_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        config_params: Dict[str, Any]
    ) -> LSTMPredictor:
        """Train LSTM model."""
        self.logger.info("Training LSTM model")
        
        # Create LSTM configuration
        config = create_lstm_config(**config_params)
        
        # Initialize model
        model = LSTMPredictor(config, self.logger)
        
        # Set feature names
        model.feature_names = [f"feature_{i}" for i in range(config.features_dim)]
        
        # Train model
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        self.logger.info(
            "LSTM training completed",
            train_loss=metrics.train_loss,
            val_loss=metrics.val_loss,
            training_time=metrics.training_time
        )
        
        return model
    
    def train_cnn_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        config_params: Dict[str, Any]
    ) -> CNNPredictor:
        """Train CNN model."""
        self.logger.info("Training CNN model")
        
        # Create CNN configuration
        config = create_cnn_config(**config_params)
        
        # Initialize model
        model = CNNPredictor(config, self.logger)
        
        # Set feature names
        model.feature_names = [f"feature_{i}" for i in range(config.features_dim)]
        
        # Train model
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        self.logger.info(
            "CNN training completed",
            train_loss=metrics.train_loss,
            val_loss=metrics.val_loss,
            training_time=metrics.training_time
        )
        
        return model
    
    def train_xgboost_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        config_params: Dict[str, Any]
    ) -> XGBoostPredictor:
        """Train XGBoost model."""
        self.logger.info("Training XGBoost model")
        
        # Create XGBoost configuration
        config = create_xgboost_config(**config_params)
        
        # Initialize model
        model = XGBoostPredictor(config, self.logger)
        
        # Set feature names (for flattened input)
        if len(X_train.shape) == 3:
            n_features = X_train.shape[1] * X_train.shape[2]
        else:
            n_features = X_train.shape[1]
        
        model.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Train model
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        self.logger.info(
            "XGBoost training completed",
            train_loss=metrics.train_loss,
            val_loss=metrics.val_loss,
            training_time=metrics.training_time
        )
        
        return model
    
    async def train_ensemble_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        config_params: Dict[str, Any],
        individual_models: Optional[List[str]] = None
    ) -> EnsembleMLPredictor:
        """Train ensemble model."""
        self.logger.info("Training ensemble model")
        
        if individual_models is None:
            individual_models = ['lstm', 'cnn', 'xgboost']
        
        # Create ensemble configuration
        config = create_ensemble_config(**config_params)
        
        # Initialize ensemble
        ensemble = EnsembleMLPredictor(config, self.logger)
        
        # Train individual models and add to ensemble
        for model_type in individual_models:
            self.logger.info(f"Training {model_type} for ensemble")
            
            if model_type == 'lstm':
                model = self.train_lstm_model(X_train, y_train, X_val, y_val, config_params)
                await ensemble.add_model(model, weight=1.0)
                
            elif model_type == 'cnn':
                model = self.train_cnn_model(X_train, y_train, X_val, y_val, config_params)
                await ensemble.add_model(model, weight=1.0)
                
            elif model_type == 'xgboost':
                model = self.train_xgboost_model(X_train, y_train, X_val, y_val, config_params)
                await ensemble.add_model(model, weight=1.0)
        
        # Train ensemble (meta-model if using stacking)
        metrics = ensemble.train(X_train, y_train, X_val, y_val)
        
        self.logger.info(
            "Ensemble training completed",
            train_loss=metrics.train_loss,
            val_loss=metrics.val_loss,
            training_time=metrics.training_time
        )
        
        return ensemble
    
    def evaluate_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        self.logger.info(f"Evaluating {model_name} model")
        
        # Get predictions
        y_pred = model._predict_raw(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
        }
        
        # Add classification metrics if applicable
        if model.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            y_pred_binary = (y_pred > 0.5).astype(int)
            y_test_binary = y_test.astype(int)
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            metrics["accuracy"] = float(accuracy)
        
        # Calculate financial metrics
        if len(y_test) > 1:
            # Calculate Sharpe ratio
            returns = y_pred if model.config.prediction_type == PredictionType.RETURN_REGRESSION else y_test
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                metrics["sharpe_ratio"] = float(sharpe_ratio)
            
            # Calculate max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(np.min(drawdown))
            metrics["max_drawdown"] = max_drawdown
        
        # Calculate baseline comparison (AR(1) model)
        baseline_pred = np.roll(y_test, 1)[1:]  # Simple lag-1 prediction
        baseline_mse = mean_squared_error(y_test[1:], baseline_pred)
        improvement = (baseline_mse - mse) / baseline_mse * 100
        metrics["improvement_over_baseline"] = float(improvement)
        
        self.logger.info(
            f"{model_name} evaluation completed",
            **metrics
        )
        
        return metrics
    
    def save_model_and_results(
        self, 
        model, 
        model_name: str, 
        metrics: Dict[str, float],
        export_onnx: bool = True
    ) -> None:
        """Save model and evaluation results."""
        
        # Save model
        model_path = self.model_dir / f"{model_name}_model"
        model.save_model(str(model_path))
        
        # Export to ONNX if requested
        if export_onnx:
            try:
                onnx_path = self.onnx_dir / f"{model_name}_model.onnx"
                model.export_to_onnx(str(onnx_path))
                self.logger.info(f"Model exported to ONNX: {onnx_path}")
            except Exception as e:
                self.logger.warning(f"Failed to export {model_name} to ONNX: {e}")
        
        # Save evaluation results
        results = {
            "model_name": model_name,
            "model_type": model.config.model_type.value,
            "prediction_type": model.config.prediction_type.value,
            "metrics": metrics,
            "model_info": model.get_model_info()
        }
        
        results_path = self.model_dir / f"{model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML models for FX trading")
    
    parser.add_argument(
        "--model-type", 
        choices=["lstm", "cnn", "xgboost", "ensemble", "all"],
        default="all",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--prediction-type",
        choices=["return_regression", "return_classification", "volatility"],
        default="return_regression",
        help="Type of prediction task"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data (if not provided, synthetic data will be used)"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples for synthetic data"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for time series models"
    )
    
    parser.add_argument(
        "--n-features",
        type=int,
        default=13,
        help="Number of features"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs for neural networks"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export models to ONNX format"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = SystemConfig()
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Convert prediction type
    prediction_type = PredictionType(args.prediction_type)
    
    # Load or generate data
    if args.data_path:
        X, y = trainer.load_real_data(args.data_path)
    else:
        X, y = trainer.generate_synthetic_data(
            n_samples=args.n_samples,
            sequence_length=args.sequence_length,
            n_features=args.n_features,
            prediction_type=prediction_type
        )
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    # Common configuration parameters
    config_params = {
        "prediction_type": prediction_type,
        "sequence_length": args.sequence_length,
        "features_dim": args.n_features,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    }
    
    # Train models based on selection
    models_to_train = []
    
    if args.model_type == "all":
        models_to_train = ["lstm", "cnn", "xgboost", "ensemble"]
    else:
        models_to_train = [args.model_type]
    
    trained_models = {}
    
    for model_type in models_to_train:
        try:
            logger.info(f"Starting training for {model_type}")
            
            if model_type == "lstm":
                model = trainer.train_lstm_model(X_train, y_train, X_val, y_val, config_params)
                
            elif model_type == "cnn":
                model = trainer.train_cnn_model(X_train, y_train, X_val, y_val, config_params)
                
            elif model_type == "xgboost":
                model = trainer.train_xgboost_model(X_train, y_train, X_val, y_val, config_params)
                
            elif model_type == "ensemble":
                model = await trainer.train_ensemble_model(
                    X_train, y_train, X_val, y_val, config_params
                )
            
            # Evaluate model
            metrics = trainer.evaluate_model(model, X_test, y_test, model_type)
            
            # Save model and results
            trainer.save_model_and_results(model, model_type, metrics, args.export_onnx)
            
            trained_models[model_type] = {
                "model": model,
                "metrics": metrics
            }
            
            logger.info(f"Completed training for {model_type}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}", exc_info=True)
    
    # Print summary
    logger.info("Training completed!")
    logger.info("Model Performance Summary:")
    
    for model_type, results in trained_models.items():
        metrics = results["metrics"]
        logger.info(
            f"{model_type.upper()}: "
            f"MSE={metrics.get('mse', 'N/A'):.6f}, "
            f"RMSE={metrics.get('rmse', 'N/A'):.6f}, "
            f"Improvement={metrics.get('improvement_over_baseline', 'N/A'):.2f}%"
        )


if __name__ == "__main__":
    asyncio.run(main()) 