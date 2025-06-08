"""
CNN Model for FX AI-Quant Trading System.

This module implements an optimized CNN-based predictor for detecting localized patterns
in financial time series data, particularly effective for volatility clustering
and pattern recognition. Optimized for FX trading patterns and fast inference.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import structlog
from sklearn.preprocessing import MinMaxScaler

# Disable oneDNN optimizations for stability
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import tensorflow as tf
    import tf2onnx
    from tensorflow import keras
    from tensorflow.keras import Input, Model, callbacks, layers, optimizers
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.regularizers import l1_l2

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy objects to prevent NameError
    keras = None
    tf = None

from models.predictor_interface import (
    BasePredictorModel,
    ModelConfig,
    ModelType,
    PredictionType,
    TrainingMetrics,
)


class CNNPredictor(BasePredictorModel):
    """Optimized CNN-based predictor for FX pattern detection."""

    def __init__(
        self, config: ModelConfig, logger: structlog.stdlib.BoundLogger | None = None
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN model but not installed")

        super().__init__(config, logger)

        # Optimized CNN configuration for FX patterns
        self.cnn_filters = getattr(
            config, "cnn_filters", 64
        )  # Increased for better pattern detection
        self.kernel_sizes = getattr(
            config, "kernel_sizes", [2, 3, 5, 8]
        )  # FX-specific timeframes
        self.pool_size = getattr(config, "pool_size", 2)
        self.num_conv_layers = getattr(
            config, "num_conv_layers", 2
        )  # Reduced for speed
        self.dropout_rate = getattr(config, "dropout_rate", 0.3)
        self.use_batch_norm = getattr(config, "use_batch_norm", True)
        self.use_residual = getattr(
            config, "use_residual", True
        )  # Residual connections
        self.use_attention = getattr(
            config, "use_attention", False
        )  # Optional attention

        # Optimization settings
        self.use_mixed_precision = getattr(config, "use_mixed_precision", True)
        self.use_xla = getattr(config, "use_xla", True)
        self.batch_size = max(config.batch_size, 64)  # Larger batch for efficiency

        # Enable mixed precision for speed
        if self.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
            except:
                self.logger.warning("Mixed precision not available")

        # Enable XLA compilation
        if self.use_xla:
            tf.config.optimizer.set_jit(True)

        # Use more efficient scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.logger.info(
            "Optimized CNN Predictor initialized",
            cnn_filters=self.cnn_filters,
            kernel_sizes=self.kernel_sizes,
            num_conv_layers=self.num_conv_layers,
            use_residual=self.use_residual,
            use_attention=self.use_attention,
            use_mixed_precision=self.use_mixed_precision,
            sequence_length=self.config.sequence_length,
            prediction_type=self.config.prediction_type.value,
        )

    def build_model(self) -> "keras.Model":
        """Build optimized CNN model architecture for FX patterns."""
        # Input layer
        inputs = Input(
            shape=(self.config.sequence_length, self.config.features_dim),
            name="cnn_input",
        )

        # Multi-scale convolutional branches for different FX timeframes
        conv_outputs = []

        for i, kernel_size in enumerate(self.kernel_sizes):
            x = inputs

            # First conv layer for this kernel size
            x = layers.Conv1D(
                filters=self.cnn_filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name=f"conv1d_k{kernel_size}_1",
            )(x)

            if self.use_batch_norm:
                x = layers.BatchNormalization(name=f"bn_k{kernel_size}_1")(x)

            # Residual connection for deeper networks
            if self.use_residual and self.num_conv_layers > 1:
                residual = x

            # Additional conv layers
            for j in range(1, self.num_conv_layers):
                x = layers.Conv1D(
                    filters=self.cnn_filters,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    name=f"conv1d_k{kernel_size}_{j+1}",
                )(x)

                if self.use_batch_norm:
                    x = layers.BatchNormalization(name=f"bn_k{kernel_size}_{j+1}")(x)

                # Add residual connection
                if self.use_residual and j == self.num_conv_layers - 1:
                    x = layers.Add(name=f"residual_k{kernel_size}")([x, residual])

                # Pooling only on alternate layers to preserve information
                if j % 2 == 0:
                    x = layers.MaxPooling1D(
                        pool_size=self.pool_size, name=f"maxpool_k{kernel_size}_{j+1}"
                    )(x)

            x = layers.Dropout(self.dropout_rate, name=f"dropout_k{kernel_size}")(x)

            # Global pooling for this branch
            x = layers.GlobalMaxPooling1D(name=f"global_maxpool_k{kernel_size}")(x)
            conv_outputs.append(x)

        # Concatenate all branches
        if len(conv_outputs) > 1:
            merged = layers.Concatenate(name="concat_branches")(conv_outputs)
        else:
            merged = conv_outputs[0]

        # Optional attention mechanism
        if self.use_attention:
            # Reshape for attention
            attention_input = layers.Reshape(
                (len(self.kernel_sizes), self.cnn_filters)
            )(merged)
            attention = layers.MultiHeadAttention(
                num_heads=4, key_dim=self.cnn_filters // 4, name="attention"
            )(attention_input, attention_input)
            merged = layers.Flatten(name="flatten_attention")(attention)

        # Optimized dense layers
        x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name="dense_1",
        )(merged)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense_1")(x)

        x = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name="dense_2",
        )(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense_2")(x)

        # Output layer
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            outputs = layers.Dense(
                1, activation="sigmoid", name=f"cnn_{self.config.prediction_type.value}"
            )(x)
        else:
            outputs = layers.Dense(
                1,
                activation="linear",
                dtype="float32",  # Ensure float32 output for mixed precision
                name=f"cnn_{self.config.prediction_type.value}",
            )(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Optimized optimizer
        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate * 1.5,  # Slightly higher LR
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,  # Gradient clipping
        )

        # Compile with optimized settings
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"],
                jit_compile=self.use_xla,
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mae"],
                jit_compile=self.use_xla,
            )

        # Build the model with a dummy input to get parameter counts
        dummy_input = tf.zeros(
            (1, self.config.sequence_length, self.config.features_dim)
        )
        _ = model(dummy_input)

        self.logger.info(
            "Optimized CNN model built",
            total_params=model.count_params(),
            trainable_params=sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            ),
        )

        return model

    def build_model_simple(self) -> "keras.Model":
        """Build a simpler CNN model architecture (fallback)."""
        model = Sequential(name=f"cnn_simple_{self.config.prediction_type.value}")

        # Input shape for 1D convolution
        input_shape = (self.config.sequence_length, self.config.features_dim)

        # First convolutional block
        model.add(
            layers.Conv1D(
                filters=self.cnn_filters,
                kernel_size=5,
                activation="relu",
                padding="same",
                input_shape=input_shape,
                name="conv1d_1",
            )
        )

        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name="bn_1"))

        model.add(layers.MaxPooling1D(pool_size=2, name="maxpool_1"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_1"))

        # Second convolutional block
        model.add(
            layers.Conv1D(
                filters=self.cnn_filters * 2,
                kernel_size=3,
                activation="relu",
                padding="same",
                name="conv1d_2",
            )
        )

        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name="bn_2"))

        model.add(layers.MaxPooling1D(pool_size=2, name="maxpool_2"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_2"))

        # Third convolutional block
        model.add(
            layers.Conv1D(
                filters=self.cnn_filters * 4,
                kernel_size=3,
                activation="relu",
                padding="same",
                name="conv1d_3",
            )
        )

        if self.use_batch_norm:
            model.add(layers.BatchNormalization(name="bn_3"))

        model.add(layers.GlobalMaxPooling1D(name="global_maxpool"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_3"))

        # Dense layers
        model.add(layers.Dense(128, activation="relu", name="dense_1"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_dense_1"))

        model.add(layers.Dense(64, activation="relu", name="dense_2"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_dense_2"))

        # Output layer
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            model.add(layers.Dense(1, activation="sigmoid", name="output"))
        else:
            model.add(layers.Dense(1, activation="linear", name="output"))

        # Compile model
        if self.config.prediction_type == PredictionType.RETURN_CLASSIFICATION:
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
                loss="mse",
                metrics=["mae"],
            )

        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainingMetrics:
        """Train the optimized CNN model."""
        start_time = time.time()

        self.logger.info(
            "Starting optimized CNN training",
            X_shape=X.shape,
            y_shape=y.shape,
            epochs=self.config.epochs,
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
                monitor="val_loss" if X_val_scaled is not None else "loss",
                patience=7,  # Slightly more patience for CNN
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val_scaled is not None else "loss",
                factor=0.5,
                patience=4,  # Reduced patience
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        # Prepare validation data
        validation_data = None
        if X_val_scaled is not None and y_val is not None:
            validation_data = (X_val_scaled, y_val)

        # Train with optimized settings
        history = self.model.fit(
            X_scaled,
            y,
            batch_size=self.batch_size,
            epochs=min(self.config.epochs, 60),  # Cap epochs for speed
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=0,
            shuffle=True,
            use_multiprocessing=True,
            workers=4,
        )

        training_time = time.time() - start_time
        self.is_trained = True

        # Calculate metrics
        train_loss = float(history.history["loss"][-1])
        val_loss = float(history.history.get("val_loss", [train_loss])[-1])

        train_accuracy = None
        val_accuracy = None
        if "accuracy" in history.history:
            train_accuracy = float(history.history["accuracy"][-1])
            val_accuracy = float(
                history.history.get("val_accuracy", [train_accuracy])[-1]
            )

        # Calculate memory usage
        memory_usage = self._calculate_memory_usage()

        self.training_metrics = TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            training_time=training_time,
            memory_usage_mb=memory_usage,
        )

        self.logger.info(
            "Optimized CNN training completed",
            train_loss=train_loss,
            val_loss=val_loss,
            training_time=training_time,
            memory_usage_mb=memory_usage,
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

    def _prepare_training_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        predictions = self.model.predict(
            X_scaled, batch_size=self.batch_size, verbose=0
        )

        return predictions.flatten()

    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence using dropout-based uncertainty."""
        if self.model is None:
            return 0.0

        try:
            # Enable dropout during inference for uncertainty estimation
            f = tf.keras.backend.function(
                [self.model.input, tf.keras.backend.learning_phase()],
                [self.model.output],
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
        """Save the trained CNN model."""
        if self.model is None:
            raise ValueError("No model to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = filepath.with_suffix(".h5")
        self.model.save(str(model_path))

        # Save scaler and metadata
        metadata = {
            "config": self.config,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "is_trained": self.is_trained,
        }

        scaler_path = filepath.with_suffix(".scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        metadata_path = filepath.with_suffix(".metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self.logger.info(f"CNN model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained CNN model."""
        filepath = Path(filepath)

        # Load Keras model
        model_path = filepath.with_suffix(".h5")
        self.model = load_model(str(model_path))

        # Load scaler
        scaler_path = filepath.with_suffix(".scaler.pkl")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load metadata
        metadata_path = filepath.with_suffix(".metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.feature_names = metadata["feature_names"]
        self.training_metrics = metadata["training_metrics"]
        self.is_trained = metadata["is_trained"]

        self.logger.info(f"CNN model loaded from {filepath}")

    def export_to_onnx(self, onnx_path: str) -> None:
        """Export CNN model to ONNX format."""
        if self.model is None:
            raise ValueError("No model to export")

        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        # Create input signature
        input_signature = [
            tf.TensorSpec(
                shape=(None, self.config.sequence_length, self.config.features_dim),
                dtype=tf.float32,
                name="input",
            )
        ]

        # Convert to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=input_signature, opset=13
        )

        # Save ONNX model
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        self.logger.info(f"CNN model exported to ONNX: {onnx_path}")

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

        # For CNN, feature importance can be calculated using gradients
        # This is a simplified implementation
        n_features = len(self.feature_names)
        return {name: 1.0 / n_features for name in self.feature_names}


def create_cnn_config(
    prediction_type: PredictionType = PredictionType.RETURN_REGRESSION,
    sequence_length: int = 30,  # Reduced from 60 for speed
    features_dim: int = 13,
    cnn_filters: int = 64,  # Increased for better pattern detection
    kernel_sizes: list[int] = None,
    num_conv_layers: int = 2,  # Reduced from 3 for speed
    learning_rate: float = 0.0015,  # Slightly higher for faster convergence
    **kwargs,
) -> ModelConfig:
    """Create optimized CNN model configuration for FX trading."""

    if kernel_sizes is None:
        kernel_sizes = [2, 3, 5, 8]  # FX-specific timeframes

    # Extract standard ModelConfig parameters with optimized defaults
    standard_params = {
        "model_type": ModelType.CNN,
        "prediction_type": prediction_type,
        "input_shape": (sequence_length, features_dim),
        "output_shape": (1,),
        "sequence_length": sequence_length,
        "features_dim": features_dim,
        "learning_rate": learning_rate,
        "batch_size": kwargs.get("batch_size", 64),  # Larger batch for efficiency
        "epochs": kwargs.get("epochs", 60),  # Slightly more epochs for CNN
        "validation_split": kwargs.get("validation_split", 0.2),
        "early_stopping_patience": kwargs.get(
            "early_stopping_patience", 7
        ),  # More patience for CNN
        "model_path": kwargs.get("model_path", "models/"),
        "onnx_path": kwargs.get("onnx_path", "models/onnx/"),
    }

    config = ModelConfig(**standard_params)

    # Add CNN-specific optimized parameters
    config.cnn_filters = cnn_filters
    config.kernel_sizes = kernel_sizes
    config.num_conv_layers = num_conv_layers
    config.pool_size = kwargs.get("pool_size", 2)
    config.dropout_rate = kwargs.get("dropout_rate", 0.3)
    config.use_batch_norm = kwargs.get("use_batch_norm", True)
    config.use_residual = kwargs.get("use_residual", True)  # Residual connections
    config.use_attention = kwargs.get("use_attention", False)  # Optional attention
    config.use_mixed_precision = kwargs.get(
        "use_mixed_precision", True
    )  # Enable mixed precision
    config.use_xla = kwargs.get("use_xla", True)  # Enable XLA compilation

    return config
