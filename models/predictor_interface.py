"""
ML Predictor Interface for the FX AI-Quant Trading System.

This module defines the common interface and base classes for all ML predictors,
including LSTM, CNN, and ensemble models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
import structlog

from core.interfaces.ml_interfaces import MLPredictor, Prediction, Features


class PredictionType(Enum):
    """Types of predictions the model can make."""
    RETURN_REGRESSION = "return_regression"
    RETURN_CLASSIFICATION = "return_classification"
    VOLATILITY = "volatility"
    REGIME_CONFIDENCE = "regime_confidence"


class ModelType(Enum):
    """Types of ML models."""
    LSTM = "lstm"
    CNN = "cnn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_type: ModelType
    prediction_type: PredictionType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    sequence_length: int = 60
    features_dim: int = 13
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_path: str = "models/"
    onnx_path: str = "models/onnx/"


@dataclass
class TrainingMetrics:
    """Training metrics and evaluation results."""
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0


class BasePredictorModel(MLPredictor):
    """Base class for all ML predictor models."""
    
    def __init__(
        self,
        config: ModelConfig,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.scaler = None
        self.training_metrics: Optional[TrainingMetrics] = None
        
    @abstractmethod
    def build_model(self) -> Any:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingMetrics:
        """Train the model."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        pass
    
    @abstractmethod
    def export_to_onnx(self, onnx_path: str) -> None:
        """Export model to ONNX format."""
        pass
    
    async def predict(
        self, 
        features: Union[Features, np.ndarray, pd.DataFrame]
    ) -> Prediction:
        """Make a single prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to numpy array
        if isinstance(features, Features):
            # For Features object, create proper sequence shape
            feature_vector = features.feature_vector
            if len(feature_vector.shape) == 1:
                # Check if this is an XGBoost model that needs flattened input
                if hasattr(self.config, 'model_type') and self.config.model_type.value == 'xgboost':
                    # For XGBoost, repeat the feature vector to create flattened sequence
                    X = np.tile(feature_vector, self.config.sequence_length).reshape(1, -1)
                elif hasattr(self.config, 'sequence_length'):
                    # For sequence models (LSTM/CNN), create proper sequence shape
                    X = np.tile(feature_vector, (self.config.sequence_length, 1)).reshape(1, self.config.sequence_length, -1)
                else:
                    X = feature_vector.reshape(1, -1)
            else:
                X = feature_vector
            symbol = features.symbol
            timestamp = features.timestamp
        elif isinstance(features, pd.DataFrame):
            X = features.values
            symbol = "UNKNOWN"
            timestamp = datetime.now()
        else:
            X = np.array(features)
            if X.ndim == 1:
                # For 1D input, create proper shape based on model type
                if hasattr(self.config, 'model_type') and self.config.model_type.value == 'xgboost':
                    # For XGBoost, repeat to create flattened sequence
                    X = np.tile(X, self.config.sequence_length).reshape(1, -1)
                elif hasattr(self.config, 'sequence_length') and self.config.model_type.value in ['lstm', 'cnn']:
                    # For sequence models, repeat the features to create a sequence
                    X = np.tile(X, (self.config.sequence_length, 1)).reshape(1, self.config.sequence_length, -1)
                else:
                    X = X.reshape(1, -1)
            symbol = "UNKNOWN"
            timestamp = datetime.now()
        
        # Preprocess if needed
        if self.scaler is not None:
            original_shape = X.shape
            if len(X.shape) == 3:  # (batch, sequence, features)
                X_flat = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.transform(X_flat)
                X = X_scaled.reshape(original_shape)
            else:
                X = self.scaler.transform(X)
        
        # Make prediction
        pred_value = self._predict_raw(X)[0]
        confidence = self._calculate_confidence(X, pred_value)
        
        return Prediction(
            symbol=symbol,
            timestamp=timestamp,
            prediction=float(pred_value),
            confidence=float(confidence),
            model_name=self.config.model_type.value,
            horizon=1
        )
    
    async def predict_batch(
        self, 
        features_batch: Union[List[Features], np.ndarray, pd.DataFrame]
    ) -> List[Prediction]:
        """Make predictions for a batch of features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        if isinstance(features_batch, list) and isinstance(features_batch[0], Features):
            # Handle list of Features objects
            for features in features_batch:
                pred = await self.predict(features)
                predictions.append(pred)
        else:
            # Handle numpy array or DataFrame
            if isinstance(features_batch, pd.DataFrame):
                X = features_batch.values
            else:
                X = np.array(features_batch)
            
            # Preprocess if needed
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Make batch predictions
            pred_values = self._predict_raw(X)
            confidences = [self._calculate_confidence(X[i:i+1], pred_values[i]) 
                          for i in range(len(pred_values))]
            
            for i, (pred_value, confidence) in enumerate(zip(pred_values, confidences)):
                predictions.append(Prediction(
                    symbol="UNKNOWN",
                    timestamp=datetime.now(),
                    prediction=float(pred_value),
                    confidence=float(confidence),
                    model_name=self.config.model_type.value,
                    horizon=1
                ))
        
        return predictions
    
    @abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw prediction method to be implemented by subclasses."""
        pass
    
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence (to be overridden by subclasses)."""
        # Default implementation - can be overridden
        return 0.5
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        # Default implementation - to be overridden by models that support it
        if not self.feature_names:
            return {}
        
        # Return uniform importance as default
        n_features = len(self.feature_names)
        return {name: 1.0 / n_features for name in self.feature_names}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information."""
        info = {
            "model_type": self.config.model_type.value,
            "prediction_type": self.config.prediction_type.value,
            "input_shape": self.config.input_shape,
            "output_shape": self.config.output_shape,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
        }
        
        if self.training_metrics:
            info["training_metrics"] = {
                "train_loss": self.training_metrics.train_loss,
                "val_loss": self.training_metrics.val_loss,
                "train_accuracy": self.training_metrics.train_accuracy,
                "val_accuracy": self.training_metrics.val_accuracy,
                "mse": self.training_metrics.mse,
                "mae": self.training_metrics.mae,
                "training_time": self.training_metrics.training_time,
                "inference_time": self.training_metrics.inference_time,
                "memory_usage_mb": self.training_metrics.memory_usage_mb,
            }
        
        return info
    
    def prepare_sequences(
        self, 
        data: np.ndarray, 
        sequence_length: Optional[int] = None
    ) -> np.ndarray:
        """Prepare sequential data for LSTM/CNN models."""
        seq_len = sequence_length or self.config.sequence_length
        
        if len(data) < seq_len:
            raise ValueError(f"Data length {len(data)} is less than sequence length {seq_len}")
        
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
        
        return np.array(sequences)
    
    def evaluate_model(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self._predict_raw(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(np.sqrt(mse)),
        }
        
        # Add classification metrics if applicable
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            # Convert to binary classification (positive/negative returns)
            y_test_binary = (y_test > 0).astype(int)
            y_pred_binary = (y_pred > 0).astype(int)
            
            accuracy = np.mean(y_test_binary == y_pred_binary)
            metrics["accuracy"] = float(accuracy)
        
        # Calculate financial metrics
        if len(y_test) > 1:
            returns = y_pred if self.config.prediction_type == PredictionType.RETURN_REGRESSION else y_test
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            metrics["sharpe_ratio"] = float(sharpe_ratio)
            metrics["max_drawdown"] = float(max_drawdown)
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown)) 