"""
Machine Learning Predictor Module for FX AI-Quant Trading System.

This module defines the interface for all ML prediction models and includes
implementations for various model types (e.g., XGBoost, LSTM, CNN).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model


@dataclass
class Prediction:
    """Represents the output of a prediction model."""

    signal: int  # -1 for short, 0 for neutral, 1 for long
    probability: float  # Confidence of the prediction (e.g., probability of class 1)
    meta: dict[str, Any] | None = None  # Optional metadata


class MLPredictor(ABC):
    """Abstract base class for all ML prediction models."""

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.model = None
        self.logger = logger or structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on the given data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Prediction:
        """Make a prediction on new data."""

    def save(self, path: str | Path):
        """Save the trained model to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str | Path):
        """Load a model from a file."""
        path = Path(path)
        if not path.exists():
            self.logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model found at {path}")
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")


class XGBoostPredictor(MLPredictor):
    """An ML predictor using the XGBoost library."""

    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
        model_path: str | None = None,
    ):
        super().__init__(logger)
        if model_path:
            self.load(model_path)
        else:
            default_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.1,
                "random_state": 42,
            }
            if model_params:
                default_params.update(model_params)

            self.model = xgb.XGBClassifier(**default_params)
            self.logger.info(
                "Initialized XGBoostPredictor for training", params=default_params
            )

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the XGBoost model."""
        self.logger.info("Starting XGBoost model training...", shape=X.shape)
        # XGBoost handles binary classification with 0/1, but our labels are -1, 0, 1.
        # We'll treat this as a multi-class problem for now.
        y_mapped = y.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)

        self.model.fit(X, y_mapped)
        self.logger.info("XGBoost model training complete.")

    def predict(self, X: pd.DataFrame) -> Prediction:
        """Make a prediction using the trained XGBoost model."""
        if self.model is None:
            self.logger.error("Model is not trained. Cannot predict.")
            return Prediction(signal=0, probability=0.0)

        # Predict probabilities for each class [0, 1, 2]
        probabilities = self.model.predict_proba(X.head(1))

        # Get the probability of the most likely class
        prob_of_max_class = np.max(probabilities)

        # Get the predicted class
        predicted_class = np.argmax(probabilities)

        # Map back to our signal format
        signal = {0: -1, 1: 0, 2: 1}.get(predicted_class, 0)

        return Prediction(signal=signal, probability=float(prob_of_max_class))


class LSTMPredictor(MLPredictor):
    """An ML predictor using a Long Short-Term Memory (LSTM) network."""

    def __init__(
        self,
        timesteps: int = 10,
        n_features: int = 13,  # Should match the number of features in RegimeFeatures
        logger: structlog.stdlib.BoundLogger | None = None,
        model_path: str | None = None,
    ):
        super().__init__(logger)
        if model_path:
            self.load(model_path)
        else:
            self.timesteps = timesteps
            self.n_features = n_features
            self.model = self._build_model()
            self.logger.info(
                "Initialized LSTMPredictor for training",
                timesteps=timesteps,
                n_features=n_features,
            )

    def _build_model(self):
        model = Sequential(
            [
                LSTM(
                    50,
                    input_shape=(self.timesteps, self.n_features),
                    return_sequences=True,
                ),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(3, activation="softmax"),  # 3 classes: -1, 0, 1
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _create_dataset(self, X: pd.DataFrame, y: pd.Series):
        """Create sequences for LSTM training."""
        Xs, ys = [], []
        for i in range(len(X) - self.timesteps):
            v = X.iloc[i : (i + self.timesteps)].values
            Xs.append(v)
            ys.append(y.iloc[i + self.timesteps])
        return np.array(Xs), np.array(ys)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the LSTM model."""
        self.logger.info("Creating LSTM sequences...")
        y_mapped = y.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)

        X_seq, y_seq = self._create_dataset(X, y_mapped)

        if X_seq.shape[0] == 0:
            self.logger.warning(
                "Not enough data to create sequences for training.", data_points=len(X)
            )
            return

        self.logger.info("Starting LSTM model training...", shape=X_seq.shape)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        self.model.fit(
            X_seq,
            y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
        self.logger.info("LSTM model training complete.")

    def predict(self, X: pd.DataFrame) -> Prediction:
        """Make a prediction using the trained LSTM model."""
        if len(X) < self.timesteps:
            self.logger.warning(
                "Not enough data for prediction.",
                required=self.timesteps,
                available=len(X),
            )
            return Prediction(signal=0, probability=0.0)

        # Use the most recent `timesteps`
        X_pred = X.tail(self.timesteps).values.reshape(
            1, self.timesteps, self.n_features
        )

        probabilities = self.model.predict(X_pred)[0]
        prob_of_max_class = np.max(probabilities)
        predicted_class = np.argmax(probabilities)

        signal = {0: -1, 1: 0, 2: 1}.get(predicted_class, 0)

        return Prediction(signal=signal, probability=float(prob_of_max_class))

    def save(self, path: str | Path):
        """Save the trained Keras model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.logger.info(f"LSTM model saved to {path}")

    def load(self, path: str | Path):
        """Load a Keras model from a file."""
        path = Path(path)
        if not path.exists():
            self.logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model found at {path}")
        self.model = load_model(path)
        self.logger.info(f"LSTM model loaded from {path}")


class CNNPredictor(MLPredictor):
    """An ML predictor using a Convolutional Neural Network (CNN)."""

    def __init__(
        self,
        image_size: int = 10,  # Creates a 10x10 "image" from features
        n_features: int = 100,  # Input feature count, must be image_size * image_size
        logger: structlog.stdlib.BoundLogger | None = None,
        model_path: str | None = None,
    ):
        super().__init__(logger)
        if model_path:
            self.load(model_path)
        else:
            if n_features != image_size * image_size:
                raise ValueError("n_features must equal image_size * image_size")

            self.image_size = image_size
            self.n_features = n_features
            self.model = self._build_model()
            self.logger.info(
                "Initialized CNNPredictor for training",
                image_size=image_size,
                n_features=n_features,
            )

    def _build_model(self):
        model = Sequential(
            [
                Conv2D(
                    32,
                    (2, 2),
                    activation="relu",
                    input_shape=(self.image_size, self.image_size, 1),
                ),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(3, activation="softmax"),  # 3 classes: -1, 0, 1
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _create_dataset(self, X: pd.DataFrame, y: pd.Series):
        """Convert tabular data into image-like format."""
        num_samples = len(X)

        # Pad features if necessary to reach n_features
        if X.shape[1] < self.n_features:
            padding = pd.DataFrame(
                np.zeros((num_samples, self.n_features - X.shape[1])), index=X.index
            )
            X_padded = pd.concat([X, padding], axis=1)
        else:
            X_padded = X.iloc[:, : self.n_features]

        X_images = X_padded.values.reshape(
            num_samples, self.image_size, self.image_size, 1
        )

        y_mapped = y.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)

        return X_images, y_mapped.values

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the CNN model."""
        self.logger.info("Creating CNN image dataset...")
        X_img, y_img = self._create_dataset(X, y)

        self.logger.info("Starting CNN model training...", shape=X_img.shape)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        self.model.fit(
            X_img,
            y_img,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
        self.logger.info("CNN model training complete.")

    def predict(self, X: pd.DataFrame) -> Prediction:
        """Make a prediction using the trained CNN model."""
        if X.shape[1] < self.n_features:
            padding = pd.DataFrame(
                np.zeros((len(X), self.n_features - X.shape[1])), index=X.index
            )
            X_padded = pd.concat([X, padding], axis=1)
        else:
            X_padded = X.iloc[:, : self.n_features]

        X_pred = X_padded.head(1).values.reshape(1, self.image_size, self.image_size, 1)

        probabilities = self.model.predict(X_pred)[0]
        prob_of_max_class = np.max(probabilities)
        predicted_class = np.argmax(probabilities)

        signal = {0: -1, 1: 0, 2: 1}.get(predicted_class, 0)

        return Prediction(signal=signal, probability=float(prob_of_max_class))

    def save(self, path: str | Path):
        """Save the trained Keras model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.logger.info(f"CNN model saved to {path}")

    def load(self, path: str | Path):
        """Load a Keras model from a file."""
        path = Path(path)
        if not path.exists():
            self.logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model found at {path}")
        self.model = load_model(path)
        self.logger.info(f"CNN model loaded from {path}")
