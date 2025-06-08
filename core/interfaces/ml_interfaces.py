"""
Machine Learning interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


class Features:
    """Standardized feature data structure."""

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        features: dict[str, float],
        feature_names: list[str],
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.features = features
        self.feature_names = feature_names
        self.feature_vector = np.array([features[name] for name in feature_names])


class Prediction:
    """Standardized prediction structure."""

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        prediction: float,
        confidence: float,
        model_name: str,
        horizon: int = 1,
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.prediction = prediction
        self.confidence = confidence
        self.model_name = model_name
        self.horizon = horizon


class RegimeLabel:
    """Market regime classification."""

    def __init__(
        self,
        timestamp: datetime,
        regime: str,  # 'trending', 'mean_reverting', 'choppy'
        confidence: float,
        features: dict[str, float] | None = None,
    ):
        self.timestamp = timestamp
        self.regime = regime
        self.confidence = confidence
        self.features = features or {}


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering."""

    @abstractmethod
    async def compute_technical_features(
        self, data: pd.DataFrame, window_sizes: list[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """Compute technical indicators and features."""

    @abstractmethod
    async def compute_volatility_features(
        self, data: pd.DataFrame, lookback_periods: list[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """Compute volatility-based features."""

    @abstractmethod
    async def compute_momentum_features(
        self, data: pd.DataFrame, periods: list[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """Compute momentum indicators."""

    @abstractmethod
    async def compute_carry_features(
        self, fx_data: pd.DataFrame, interest_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute carry trade features."""

    @abstractmethod
    async def compute_macro_surprises(
        self, economic_data: pd.DataFrame, expectations: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute macro economic surprise features."""


class MLPredictor(ABC):
    """Abstract base class for ML prediction models."""

    @abstractmethod
    async def predict(
        self, features: Features | np.ndarray | pd.DataFrame
    ) -> Prediction:
        """Make a prediction based on input features."""

    @abstractmethod
    async def predict_batch(
        self, features_batch: list[Features] | np.ndarray | pd.DataFrame
    ) -> list[Prediction]:
        """Make predictions for a batch of feature sets."""

    @abstractmethod
    async def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata and information."""


class ModelTrainer(ABC):
    """Abstract base class for model training."""

    @abstractmethod
    async def train(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> dict[str, Any]:
        """Train the model with given data."""

    @abstractmethod
    async def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        time_series_split: bool = True,
    ) -> dict[str, float]:
        """Perform cross-validation."""

    @abstractmethod
    async def walk_forward_validation(
        self, X: pd.DataFrame, y: pd.Series, train_window: int, test_window: int
    ) -> dict[str, Any]:
        """Perform walk-forward validation."""

    @abstractmethod
    async def save_model(self, filepath: str) -> None:
        """Save trained model to file."""

    @abstractmethod
    async def load_model(self, filepath: str) -> None:
        """Load model from file."""


class RegimeDetector(ABC):
    """Abstract base class for regime detection."""

    @abstractmethod
    async def detect_regime(
        self, market_data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> RegimeLabel:
        """Detect current market regime."""

    @abstractmethod
    async def get_regime_history(
        self, start_date: datetime, end_date: datetime
    ) -> list[RegimeLabel]:
        """Get historical regime classifications."""

    @abstractmethod
    async def get_transition_probabilities(self) -> pd.DataFrame:
        """Get regime transition probability matrix."""

    @abstractmethod
    async def fit(self, data: pd.DataFrame, n_regimes: int = 3) -> None:
        """Fit regime detection model to historical data."""


class EnsemblePredictor(ABC):
    """Abstract base class for ensemble prediction models."""

    @abstractmethod
    async def add_model(self, model: MLPredictor, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""

    @abstractmethod
    async def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""

    @abstractmethod
    async def update_weights(self, weights: dict[str, float]) -> None:
        """Update model weights in the ensemble."""

    @abstractmethod
    async def ensemble_predict(self, features: Features) -> Prediction:
        """Make ensemble prediction."""
