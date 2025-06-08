"""
XGBoost Model for FX AI-Quant Trading System.

This module implements an XGBoost-based predictor for tabular financial data.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from models.predictor_interface import (
    BasePredictorModel,
    ModelConfig,
    ModelType,
    PredictionType,
    TrainingMetrics,
)


class XGBoostPredictor(BasePredictorModel):
    """XGBoost-based predictor for tabular financial data."""

    def __init__(
        self, config: ModelConfig, logger: structlog.stdlib.BoundLogger | None = None
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required but not installed")

        super().__init__(config, logger)

        model_file_path = getattr(config, "model_file", None)

        if model_file_path and Path(model_file_path).exists():
            self.load_model(model_file_path)
            self.logger.info(
                f"XGBoost Predictor initialized and loaded from {model_file_path}",
                prediction_type=self.config.prediction_type.value,
            )
        else:
            # XGBoost-specific configuration for training
            self.n_estimators = getattr(config, "n_estimators", 100)
            self.max_depth = getattr(config, "max_depth", 6)
            self.learning_rate = config.learning_rate
            self.subsample = getattr(config, "subsample", 0.8)
            self.colsample_bytree = getattr(config, "colsample_bytree", 0.8)
            self.random_state = getattr(config, "random_state", 42)

            # Initialize scaler
            self.scaler = StandardScaler()

            self.logger.info(
                "XGBoost Predictor initialized for training",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                prediction_type=self.config.prediction_type.value,
            )

    def build_model(self) -> Any:
        """Build XGBoost model."""
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                objective="reg:squarederror",
                eval_metric="rmse",
            )

        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainingMetrics:
        """Train the XGBoost model."""
        start_time = time.time()

        self.logger.info("Starting XGBoost training", X_shape=X.shape, y_shape=y.shape)

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Flatten sequences for XGBoost (it expects 2D input)
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        if X_val is not None and len(X_val.shape) == 3:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_val_flat = X_val

        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        X_val_scaled = None
        if X_val_flat is not None:
            X_val_scaled = self.scaler.transform(X_val_flat)

        # Prepare evaluation set
        eval_set = []
        if X_val_scaled is not None and y_val is not None:
            eval_set = [(X_scaled, y), (X_val_scaled, y_val)]
        else:
            eval_set = [(X_scaled, y)]

        # Train model
        self.model.fit(X_scaled, y, eval_set=eval_set, verbose=False)

        training_time = time.time() - start_time
        self.is_trained = True

        # Calculate training metrics
        y_pred_train = self.model.predict(X_scaled)
        train_loss = mean_squared_error(y, y_pred_train)

        val_loss = train_loss
        train_accuracy = None
        val_accuracy = None

        if X_val_scaled is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val_scaled)
            val_loss = mean_squared_error(y_val, y_pred_val)

            if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
                train_accuracy = accuracy_score(y, (y_pred_train > 0.5).astype(int))
                val_accuracy = accuracy_score(y_val, (y_pred_val > 0.5).astype(int))

        # Calculate memory usage (approximate)
        memory_usage = self._calculate_memory_usage()

        self.training_metrics = TrainingMetrics(
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            training_time=training_time,
            memory_usage_mb=memory_usage,
        )

        self.logger.info(
            "XGBoost training completed",
            train_loss=train_loss,
            val_loss=val_loss,
            training_time=training_time,
        )

        return self.training_metrics

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw prediction method."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Flatten sequences for XGBoost - handle both single and batch predictions
        if len(X.shape) == 3:
            # Batch of sequences: (batch_size, sequence_length, features)
            X_flat = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 2:
            # Single sequence: (sequence_length, features) -> reshape to (1, sequence_length * features)
            X_flat = X.reshape(1, -1)
        else:
            # Already flattened
            X_flat = X

        # Scale features
        X_scaled = self.scaler.transform(X_flat)

        return self.model.predict(X_scaled)

    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence for XGBoost."""
        try:
            # Flatten sequences for XGBoost - handle both single and batch predictions
            if len(X.shape) == 3:
                # Batch of sequences: (batch_size, sequence_length, features)
                X_flat = X.reshape(X.shape[0], -1)
            elif len(X.shape) == 2:
                # Single sequence: (sequence_length, features) -> reshape to (1, sequence_length * features)
                X_flat = X.reshape(1, -1)
            else:
                # Already flattened
                X_flat = X

            # Scale features
            X_scaled = self.scaler.transform(X_flat)

            # For XGBoost, use prediction probability as confidence
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_scaled)
                return float(np.max(proba, axis=1)[0])
            else:
                # For regression, use a simple heuristic
                return 0.8  # Default confidence
        except Exception:
            # Fallback confidence
            return 0.8

    def save_model(self, filepath: str) -> None:
        """Save the XGBoost model."""
        if self.model is None:
            raise ValueError("No model to save")

        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and scaler
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "training_metrics": self.training_metrics,
            "is_trained": self.is_trained,
            "model_type": ModelType.XGBOOST,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info(f"XGBoost model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the XGBoost model."""
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Validate model type
        if model_data.get("model_type") != ModelType.XGBOOST:
            raise TypeError("Saved model is not an XGBoost model")

        # Restore state
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.config = model_data["config"]
        self.training_metrics = model_data.get("training_metrics")
        self.is_trained = model_data.get("is_trained", True)

        self.logger.info(f"XGBoost model loaded from {filepath}")

    def export_to_onnx(self, onnx_path: str) -> None:
        """Export XGBoost model to ONNX format."""
        # XGBoost ONNX export would require additional dependencies
        raise NotImplementedError("XGBoost ONNX export not implemented")

    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB."""
        if self.model is None:
            return 0.0

        # Rough estimate for XGBoost model size
        return 10.0  # MB

    async def get_feature_importance(self) -> dict:
        """Get feature importance from XGBoost model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            feature_names = [f"feature_{i}" for i in range(len(importance))]

            return {name: float(imp) for name, imp in zip(feature_names, importance)}

        return {}

    async def predict(self, features) -> "Prediction":
        """Override predict method to handle XGBoost-specific preprocessing."""
        from datetime import datetime

        from models.predictor_interface import Prediction

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Handle different input types
        if hasattr(features, "values"):  # Features object
            X = features.values
            symbol = getattr(features, "symbol", "UNKNOWN")
            timestamp = getattr(features, "timestamp", datetime.now())
        else:
            X = np.array(features)
            if X.ndim == 1:
                # For 1D input, create proper flattened shape for XGBoost
                X = X.reshape(1, -1)
            symbol = "UNKNOWN"
            timestamp = datetime.now()

        # XGBoost-specific preprocessing: flatten then scale
        if len(X.shape) == 3:
            # Batch of sequences: (batch_size, sequence_length, features)
            X_flat = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 2 and X.shape[0] == 1:
            # Single sequence: (1, sequence_length, features) -> already handled above
            # or (sequence_length, features) -> flatten to (1, sequence_length * features)
            if X.shape[1] == self.config.features_dim:
                # This is (sequence_length, features)
                X_flat = X.reshape(1, -1)
            else:
                # This is already flattened (1, flattened_features)
                X_flat = X
        else:
            X_flat = X

        # Scale features
        X_scaled = self.scaler.transform(X_flat)

        # Make prediction
        pred_value = self.model.predict(X_scaled)[0]
        confidence = self._calculate_confidence(X, pred_value)

        return Prediction(
            symbol=symbol,
            timestamp=timestamp,
            prediction=float(pred_value),
            confidence=float(confidence),
            model_name=self.config.model_type.value,
            horizon=1,
        )


def create_xgboost_config(
    prediction_type: PredictionType = PredictionType.RETURN_REGRESSION,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    model_file: str | None = None,
    **kwargs,
) -> ModelConfig:
    """Create XGBoost model configuration."""

    # Extract standard ModelConfig parameters
    sequence_length = kwargs.get("sequence_length", 60)
    features_dim = kwargs.get("features_dim", 13)

    # For XGBoost, input is flattened (sequence_length * features_dim)
    flattened_input_dim = sequence_length * features_dim

    # Create base config with standard parameters
    config = ModelConfig(
        model_type=ModelType.XGBOOST,
        prediction_type=prediction_type,
        input_shape=(flattened_input_dim,),
        output_shape=(1,),
        sequence_length=sequence_length,
        features_dim=features_dim,
        learning_rate=learning_rate,
        batch_size=kwargs.get("batch_size", 32),
        epochs=kwargs.get("epochs", 100),
        validation_split=kwargs.get("validation_split", 0.2),
        early_stopping_patience=kwargs.get("early_stopping_patience", 10),
        model_path=kwargs.get("model_path", "models/"),
        onnx_path=kwargs.get("onnx_path", "models/onnx/"),
    )

    # Add XGBoost-specific parameters as attributes
    config.n_estimators = n_estimators
    config.max_depth = max_depth
    config.subsample = kwargs.get("subsample", 0.8)
    config.colsample_bytree = kwargs.get("colsample_bytree", 0.8)
    config.random_state = kwargs.get("random_state", 42)
    config.model_file = model_file

    return config
