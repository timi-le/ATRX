"""
Ensemble Model for FX AI-Quant Trading System.

This module implements an ensemble predictor that combines multiple ML models
(LSTM, CNN, XGBoost/LightGBM) using weighted voting or stacking approaches.
"""

import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import structlog

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from core.interfaces.ml_interfaces import EnsemblePredictor
from models.predictor_interface import (
    BasePredictorModel, ModelConfig, TrainingMetrics, 
    PredictionType, ModelType
)
from models.xgboost_model import XGBoostPredictor


class EnsembleMLPredictor(BasePredictorModel, EnsemblePredictor):
    """Ensemble predictor combining multiple ML models."""
    
    def __init__(
        self,
        config: ModelConfig,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        super().__init__(config, logger)
        
        # Ensemble-specific configuration
        self.ensemble_method = getattr(config, 'ensemble_method', 'weighted_average')
        self.meta_model_type = getattr(config, 'meta_model_type', 'linear')
        
        # Model storage
        self.models: Dict[str, BasePredictorModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.meta_model = None
        
        self.logger.info(
            "Ensemble Predictor initialized",
            ensemble_method=self.ensemble_method,
            meta_model_type=self.meta_model_type
        )
    
    def build_model(self) -> Any:
        """Build ensemble meta-model if using stacking."""
        if self.ensemble_method == 'stacking':
            if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
                return LogisticRegression()
            else:
                return LinearRegression()
        return None
    
    async def add_model(self, model: BasePredictorModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        model_name = f"{model.config.model_type.value}_{len(self.models)}"
        self.models[model_name] = model
        self.model_weights[model_name] = weight
        
        self.logger.info(f"Added model to ensemble: {model_name}")
    
    async def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            del self.model_weights[model_name]
            self.logger.info(f"Removed model from ensemble: {model_name}")
    
    async def update_weights(self, weights: Dict[str, float]) -> None:
        """Update model weights."""
        for model_name, weight in weights.items():
            if model_name in self.model_weights:
                self.model_weights[model_name] = weight
        
        self.logger.info("Updated ensemble weights", weights=self.model_weights)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingMetrics:
        """Train all models in the ensemble."""
        start_time = time.time()
        
        self.logger.info(
            "Starting ensemble training",
            X_shape=X.shape,
            y_shape=y.shape,
            num_models=len(self.models)
        )
        
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Train all individual models
        model_metrics = {}
        for model_name, model in self.models.items():
            self.logger.info(f"Training model: {model_name}")
            metrics = model.train(X, y, X_val, y_val)
            model_metrics[model_name] = metrics
        
        # Train meta-model if using stacking
        if self.ensemble_method == 'stacking':
            self.meta_model = self.build_model()
            
            # Get predictions from all models for meta-training
            meta_features = []
            for model_name, model in self.models.items():
                pred = model._predict_raw(X)
                meta_features.append(pred.reshape(-1, 1))
            
            meta_X = np.hstack(meta_features)
            self.meta_model.fit(meta_X, y)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate ensemble metrics
        y_pred_train = self._ensemble_predict_raw(X)
        train_loss = mean_squared_error(y, y_pred_train)
        
        val_loss = train_loss
        train_accuracy = None
        val_accuracy = None
        
        if X_val is not None and y_val is not None:
            y_pred_val = self._ensemble_predict_raw(X_val)
            val_loss = mean_squared_error(y_val, y_pred_val)
            
            if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
                train_accuracy = accuracy_score(y, (y_pred_train > 0.5).astype(int))
                val_accuracy = accuracy_score(y_val, (y_pred_val > 0.5).astype(int))
        
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage()
        
        self.training_metrics = TrainingMetrics(
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            training_time=training_time,
            memory_usage_mb=memory_usage
        )
        
        self.logger.info(
            "Ensemble training completed",
            train_loss=train_loss,
            val_loss=val_loss,
            training_time=training_time,
            model_metrics=model_metrics
        )
        
        return self.training_metrics
    
    def _ensemble_predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            predictions = []
            weights = []
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    pred = model._predict_raw(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[model_name])
            
            if not predictions:
                raise ValueError("No trained models in ensemble")
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted average
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, weights):
                ensemble_pred += weight * pred
            
            return ensemble_pred
            
        elif self.ensemble_method == 'stacking':
            # Use meta-model for final prediction
            if self.meta_model is None:
                raise ValueError("Meta-model not trained")
            
            meta_features = []
            for model_name, model in self.models.items():
                if model.is_trained:
                    pred = model._predict_raw(X)
                    meta_features.append(pred.reshape(-1, 1))
            
            if not meta_features:
                raise ValueError("No trained models in ensemble")
            
            meta_X = np.hstack(meta_features)
            return self.meta_model.predict(meta_X)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw prediction method."""
        return self._ensemble_predict_raw(X)
    
    async def ensemble_predict(self, features) -> Any:
        """Make ensemble prediction with confidence."""
        return await self.predict(features)
    
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calculate ensemble prediction confidence."""
        if not self.models:
            return 0.0
        
        # Calculate confidence as agreement between models
        predictions = []
        for model_name, model in self.models.items():
            if model.is_trained:
                pred = model._predict_raw(X)
                predictions.append(pred[0] if len(pred) > 0 else 0.0)
        
        if len(predictions) < 2:
            return 0.8  # Default confidence for single model
        
        # Calculate standard deviation as measure of disagreement
        std_dev = np.std(predictions)
        max_std = np.abs(np.mean(predictions))  # Normalize by mean
        
        if max_std == 0:
            return 1.0
        
        # Convert to confidence (lower std = higher confidence)
        confidence = max(0.1, 1.0 - (std_dev / max_std))
        return min(1.0, confidence)
    
    def save_model(self, filepath: str) -> None:
        """Save the ensemble model."""
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        model_files = {}
        for model_name, model in self.models.items():
            model_file = model_path.parent / f"{model_name}.pkl"
            model.save_model(str(model_file))
            model_files[model_name] = str(model_file)
        
        # Save ensemble metadata
        ensemble_data = {
            'config': self.config,
            'ensemble_method': self.ensemble_method,
            'meta_model_type': self.meta_model_type,
            'model_weights': self.model_weights,
            'model_files': model_files,
            'meta_model': self.meta_model,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
            'model_type': ModelType.ENSEMBLE
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        self.logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the ensemble model."""
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(model_path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        # Load ensemble metadata
        self.ensemble_method = ensemble_data['ensemble_method']
        self.meta_model_type = ensemble_data['meta_model_type']
        self.model_weights = ensemble_data['model_weights']
        self.meta_model = ensemble_data.get('meta_model')
        self.training_metrics = ensemble_data.get('training_metrics')
        self.is_trained = ensemble_data.get('is_trained', True)
        
        # Load individual models (would need model factory)
        # This is simplified - in practice, you'd need a model factory
        self.models = {}  # Would load from model_files
        
        self.logger.info(f"Ensemble model loaded from {filepath}")
    
    def export_to_onnx(self, onnx_path: str) -> None:
        """Export ensemble model to ONNX format."""
        raise NotImplementedError("Ensemble ONNX export not implemented")
    
    def _calculate_memory_usage(self) -> float:
        """Calculate total memory usage of all models."""
        total_memory = 0.0
        for model in self.models.values():
            total_memory += model._calculate_memory_usage()
        return total_memory
    
    async def get_feature_importance(self) -> dict:
        """Get aggregated feature importance from all models."""
        if not self.models:
            return {}
        
        all_importance = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                importance = await model.get_feature_importance()
                for feature, imp in importance.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(imp)
        
        # Average importance across models
        avg_importance = {}
        for feature, importances in all_importance.items():
            avg_importance[feature] = np.mean(importances)
        
        return avg_importance


def create_ensemble_config(
    prediction_type: PredictionType = PredictionType.RETURN_REGRESSION,
    ensemble_method: str = 'weighted_average',
    meta_model_type: str = 'linear',
    **kwargs
) -> ModelConfig:
    """Create ensemble model configuration."""
    
    # Extract standard ModelConfig parameters
    sequence_length = kwargs.get('sequence_length', 60)
    features_dim = kwargs.get('features_dim', 13)
    
    # Create base config with standard parameters
    config = ModelConfig(
        model_type=ModelType.ENSEMBLE,
        prediction_type=prediction_type,
        input_shape=(sequence_length, features_dim),
        output_shape=(1,),
        sequence_length=sequence_length,
        features_dim=features_dim,
        learning_rate=kwargs.get('learning_rate', 0.001),
        batch_size=kwargs.get('batch_size', 32),
        epochs=kwargs.get('epochs', 100),
        validation_split=kwargs.get('validation_split', 0.2),
        early_stopping_patience=kwargs.get('early_stopping_patience', 10),
        model_path=kwargs.get('model_path', "models/"),
        onnx_path=kwargs.get('onnx_path', "models/onnx/")
    )
    
    # Add ensemble-specific parameters as attributes
    config.ensemble_method = ensemble_method
    config.meta_model_type = meta_model_type
    
    return config 