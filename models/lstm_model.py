"""
LSTM Model for FX AI-Quant Trading System.

This module implements an optimized LSTM-based predictor for sequential financial data,
capable of predicting returns, volatility, or regime confidence scores.
Optimized for: Fast inference (<100ms), Efficient training (<300s), Reduced memory usage.
"""

import os
import time
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import structlog

# Disable oneDNN optimizations for stability
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers, Model, Input
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.regularizers import l1_l2
    import tf2onnx
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy objects to prevent NameError
    keras = None
    tf = None

from models.predictor_interface import (
    BasePredictorModel, ModelConfig, TrainingMetrics, 
    PredictionType, ModelType
)


class LSTMPredictor(BasePredictorModel):
    """Optimized LSTM-based predictor for sequential financial data."""
    
    def __init__(
        self,
        config: ModelConfig,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but not installed")
        
        super().__init__(config, logger)
        
        # Optimized LSTM configuration
        self.lstm_units = getattr(config, 'lstm_units', 32)  # Reduced from 50
        self.dropout_rate = getattr(config, 'dropout_rate', 0.2)  # Reduced from 0.3
        self.recurrent_dropout = getattr(config, 'recurrent_dropout', 0.1)  # Reduced
        self.num_layers = getattr(config, 'num_layers', 1)  # Reduced from 2
        self.use_bidirectional = getattr(config, 'use_bidirectional', False)  # Disabled for speed
        self.use_attention = getattr(config, 'use_attention', False)  # Optional attention
        
        # Optimization settings
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', True)
        self.use_xla = getattr(config, 'use_xla', True)
        self.batch_size = max(config.batch_size, 64)  # Larger batch for efficiency
        
        # Enable mixed precision for speed
        if self.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            except:
                self.logger.warning("Mixed precision not available")
        
        # Enable XLA compilation
        if self.use_xla:
            tf.config.optimizer.set_jit(True)
        
        # Use more efficient scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.logger.info(
            "Optimized LSTM Predictor initialized",
            lstm_units=self.lstm_units,
            num_layers=self.num_layers,
            use_bidirectional=self.use_bidirectional,
            use_attention=self.use_attention,
            use_mixed_precision=self.use_mixed_precision,
            sequence_length=self.config.sequence_length,
            prediction_type=self.config.prediction_type.value
        )
    
    def build_model(self) -> "keras.Model":
        """Build optimized LSTM model architecture."""
        # Input layer
        inputs = Input(
            shape=(self.config.sequence_length, self.config.features_dim),
            name="lstm_input"
        )
        
        x = inputs
        
        # Optimized LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1) or self.use_attention
            
            lstm_layer = layers.LSTM(
                self.lstm_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),  # Light regularization
                recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                implementation=2,  # GPU-optimized implementation
                name=f"lstm_{i}"
            )
            
            if self.use_bidirectional and i == 0:  # Only first layer bidirectional
                x = layers.Bidirectional(lstm_layer, name=f"bidirectional_lstm_{i}")(x)
            else:
                x = lstm_layer(x)
            
            # Batch normalization for faster convergence
            if i < self.num_layers - 1:
                x = layers.BatchNormalization(name=f"batch_norm_{i}")(x)
        
        # Optional attention mechanism
        if self.use_attention:
            attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.lstm_units // 4,
                name="attention"
            )
            x = attention(x, x)
            x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # Optimized dense layers
        x = layers.Dense(
            self.lstm_units // 2,
            activation='relu',
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name="dense_1"
        )(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_1")(x)
        
        # Output layer
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            outputs = layers.Dense(
                1,
                activation='sigmoid',
                name=f"lstm_{self.config.prediction_type.value}"
            )(x)
        else:
            outputs = layers.Dense(
                1,
                activation='linear',
                dtype='float32',  # Ensure float32 output for mixed precision
                name=f"lstm_{self.config.prediction_type.value}"
            )(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Optimized optimizer
        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate * 2,  # Higher LR for faster convergence
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping
        )
        
        # Compile with optimized settings
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'],
                jit_compile=self.use_xla
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae'],
                jit_compile=self.use_xla
            )
        
        # Build the model with a dummy input to get parameter counts
        dummy_input = tf.zeros((1, self.config.sequence_length, self.config.features_dim))
        _ = model(dummy_input)
        
        self.logger.info(
            "Optimized LSTM model built",
            total_params=model.count_params(),
            trainable_params=sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        )
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingMetrics:
        """Train the optimized LSTM model."""
        start_time = time.time()
        
        self.logger.info(
            "Starting optimized LSTM training",
            X_shape=X.shape,
            y_shape=y.shape,
            epochs=self.config.epochs
        )
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Prepare data with optimized scaling
        X_scaled = self._prepare_data(X)
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
        
        # Optimized callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val_scaled is not None else 'loss',
                patience=5,  # Reduced patience for faster training
                restore_best_weights=True,
                verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val_scaled is not None else 'loss',
                factor=0.5,
                patience=3,  # Reduced patience
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val_scaled is not None and y_val is not None:
            validation_data = (X_val_scaled, y_val)
        
        # Train with optimized settings
        history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=min(self.config.epochs, 50),  # Cap epochs for speed
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=0,
            shuffle=True,
            use_multiprocessing=True,
            workers=4
        )
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate metrics
        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history.get('val_loss', [train_loss])[-1])
        
        train_accuracy = None
        val_accuracy = None
        if 'accuracy' in history.history:
            train_accuracy = float(history.history['accuracy'][-1])
            val_accuracy = float(history.history.get('val_accuracy', [train_accuracy])[-1])
        
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage()
        
        self.training_metrics = TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            training_time=training_time,
            memory_usage_mb=memory_usage
        )
        
        self.logger.info(
            "Optimized LSTM training completed",
            train_loss=train_loss,
            val_loss=val_loss,
            training_time=training_time,
            memory_usage_mb=memory_usage
        )
        
        return self.training_metrics
    
    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare and scale input data efficiently."""
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        return X_scaled.reshape(original_shape)
    
    def _prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with optimized scaling."""
        # Use the new optimized method
        X_scaled = self._prepare_data(X)
        return X_scaled, y
    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Optimized raw prediction method."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Scale input data
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Make prediction with optimized batch processing
        predictions = self.model.predict(X_scaled, batch_size=self.batch_size, verbose=0)
        
        return predictions.flatten()
    
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence using dropout-based uncertainty."""
        if self.model is None:
            return 0.0
        
        try:
            # Enable dropout during inference for uncertainty estimation
            f = tf.keras.backend.function(
                [self.model.input, tf.keras.backend.learning_phase()],
                [self.model.output]
            )
            
            # Multiple forward passes with dropout
            predictions = []
            for _ in range(10):  # Reduced from 20 for speed
                pred = f([X, 1])[0]  # 1 = training mode (dropout enabled)
                predictions.append(pred[0, 0])
            
            # Calculate uncertainty as standard deviation
            std_dev = np.std(predictions)
            mean_pred = np.mean(predictions)
            
            # Convert to confidence (lower std = higher confidence)
            if abs(mean_pred) > 0:
                confidence = max(0.1, 1.0 - (std_dev / abs(mean_pred)))
            else:
                confidence = 0.5
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.8  # Default confidence
    
    def save_model(self, filepath: str) -> None:
        """Save the trained LSTM model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = filepath.with_suffix('.h5')
        self.model.save(str(model_path))
        
        # Save scaler and metadata
        metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        scaler_path = filepath.with_suffix('.scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained LSTM model."""
        filepath = Path(filepath)
        
        # Load Keras model
        model_path = filepath.with_suffix('.h5')
        self.model = load_model(str(model_path))
        
        # Load scaler
        scaler_path = filepath.with_suffix('.scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_metrics = metadata['training_metrics']
        self.is_trained = metadata['is_trained']
        
        self.logger.info(f"LSTM model loaded from {filepath}")
    
    def export_to_onnx(self, onnx_path: str) -> None:
        """Export LSTM model to ONNX format."""
        if self.model is None:
            raise ValueError("No model to export")
        
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create input signature
        input_signature = [tf.TensorSpec(
            shape=(None, self.config.sequence_length, self.config.features_dim),
            dtype=tf.float32,
            name="input"
        )]
        
        # Convert to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=input_signature,
            opset=13
        )
        
        # Save ONNX model
        with open(onnx_path, 'wb') as f:
            f.write(model_proto.SerializeToString())
        
        self.logger.info(f"LSTM model exported to ONNX: {onnx_path}")
    
    def _calculate_memory_usage(self) -> float:
        """Calculate model memory usage in MB."""
        if self.model is None:
            return 0.0
        
        # Estimate memory usage based on model parameters
        total_params = self.model.count_params()
        # Assume 4 bytes per parameter (float32)
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        return float(memory_mb)
    
    async def get_feature_importance(self) -> dict:
        """Get feature importance using gradient-based attribution."""
        if not self.is_trained or not self.feature_names:
            return {}
        
        # For LSTM, feature importance is complex due to temporal dependencies
        # Return uniform importance as a placeholder
        # In practice, you might use techniques like SHAP or integrated gradients
        n_features = len(self.feature_names)
        return {name: 1.0 / n_features for name in self.feature_names}


def create_lstm_config(
    prediction_type: PredictionType = PredictionType.RETURN_REGRESSION,
    sequence_length: int = 30,  # Reduced from 60 for speed
    features_dim: int = 13,
    lstm_units: int = 32,  # Reduced from 50 for speed
    num_layers: int = 1,  # Reduced from 2 for speed
    learning_rate: float = 0.002,  # Higher for faster convergence
    **kwargs
) -> ModelConfig:
    """Create optimized LSTM model configuration."""
    
    # Extract standard ModelConfig parameters with optimized defaults
    standard_params = {
        'model_type': ModelType.LSTM,
        'prediction_type': prediction_type,
        'input_shape': (sequence_length, features_dim),
        'output_shape': (1,),
        'sequence_length': sequence_length,
        'features_dim': features_dim,
        'learning_rate': learning_rate,
        'batch_size': kwargs.get('batch_size', 64),  # Larger batch for efficiency
        'epochs': kwargs.get('epochs', 50),  # Reduced epochs
        'validation_split': kwargs.get('validation_split', 0.2),
        'early_stopping_patience': kwargs.get('early_stopping_patience', 5),  # Reduced patience
        'model_path': kwargs.get('model_path', "models/"),
        'onnx_path': kwargs.get('onnx_path', "models/onnx/")
    }
    
    config = ModelConfig(**standard_params)
    
    # Add LSTM-specific optimized parameters
    config.lstm_units = lstm_units
    config.num_layers = num_layers
    config.dropout_rate = kwargs.get('dropout_rate', 0.2)  # Reduced from 0.3
    config.recurrent_dropout = kwargs.get('recurrent_dropout', 0.1)  # Reduced
    config.use_bidirectional = kwargs.get('use_bidirectional', False)  # Disabled for speed
    config.use_attention = kwargs.get('use_attention', False)  # Optional attention
    config.use_mixed_precision = kwargs.get('use_mixed_precision', True)  # Enable mixed precision
    config.use_xla = kwargs.get('use_xla', True)  # Enable XLA compilation
    
    return config 