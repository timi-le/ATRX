"""
Regime Detection Module for FX AI-Quant Trading System.

This module implements multiple approaches for detecting market regimes:
- Hidden Markov Models (HMM) using GaussianHMM
- Unsupervised clustering (KMeans, DBSCAN, GMM)
- Rule-based fallback logic

The system classifies market conditions as:
- trending: Strong directional movement
- mean-reverting: Range-bound oscillation
- choppy: Low momentum, high noise
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

# ML and statistical libraries
from sklearn.preprocessing import StandardScaler

from core.ml_predictor import CNNPredictor, LSTMPredictor, XGBoostPredictor


class RegimeType(Enum):
    """Market regime classifications."""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CHOPPY = "choppy"
    SQUEEZE = "squeeze"
    COMPRESSION = "compression"
    NOISE = "noise"


@dataclass
class RegimeFeatures:
    """
    Feature vector for regime detection.

    Contains 13 normalized features across 4 categories:
    - Volatility: atr, bb_width, realized_vol, vol_ratio
    - Momentum: macd_signal, macd_histogram, adx, rsi, momentum
    - Macro: macro_surprise, macro_sentiment
    - Market Structure: trend_strength, mean_reversion
    """

    # Volatility features (4)
    atr: float  # Average True Range (normalized)
    bb_width: float  # Bollinger Band width (normalized)
    realized_vol: float  # Realized volatility (normalized)
    vol_ratio: float  # Current vol / historical vol

    # Momentum features (5)
    macd_signal: float  # MACD signal strength (-1 to 1)
    macd_histogram: float  # MACD histogram (-1 to 1)
    adx: float  # Average Directional Index (0-100, normalized to 0-1)
    rsi: float  # RSI (0-100, normalized to 0-1)
    momentum: float  # Price momentum (-1 to 1)

    # Macro features (2)
    macro_surprise: float  # Economic surprise index (-1 to 1)
    macro_sentiment: float  # News sentiment (-1 to 1)

    # Market structure features (2)
    trend_strength: float  # Trend strength (0 to 1)
    mean_reversion: float  # Mean reversion tendency (0 to 1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array(
            [
                self.atr,
                self.bb_width,
                self.realized_vol,
                self.vol_ratio,
                self.macd_signal,
                self.macd_histogram,
                self.adx,
                self.rsi,
                self.momentum,
                self.macro_surprise,
                self.macro_sentiment,
                self.trend_strength,
                self.mean_reversion,
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for analysis."""
        return {
            "atr": self.atr,
            "bb_width": self.bb_width,
            "realized_vol": self.realized_vol,
            "vol_ratio": self.vol_ratio,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "adx": self.adx,
            "rsi": self.rsi,
            "momentum": self.momentum,
            "macro_surprise": self.macro_surprise,
            "macro_sentiment": self.macro_sentiment,
            "trend_strength": self.trend_strength,
            "mean_reversion": self.mean_reversion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "RegimeFeatures":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RegimeFeatures":
        """Create from numpy array."""
        if len(arr) != 13:
            raise ValueError(f"Expected 13 features, got {len(arr)}")

        return cls(
            atr=arr[0],
            bb_width=arr[1],
            realized_vol=arr[2],
            vol_ratio=arr[3],
            macd_signal=arr[4],
            macd_histogram=arr[5],
            adx=arr[6],
            rsi=arr[7],
            momentum=arr[8],
            macro_surprise=arr[9],
            macro_sentiment=arr[10],
            trend_strength=arr[11],
            mean_reversion=arr[12],
        )


@dataclass
class RegimeOutput:
    """Output from regime detection."""

    regime: RegimeType  # Detected regime
    confidence: float  # Confidence score (0-1)
    probabilities: dict[RegimeType, float]  # Probability for each regime
    transition_prob: float | None = None  # Probability of regime transition
    features_used: list[str] | None = None  # Features that contributed most

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "probabilities": {k.value: v for k, v in self.probabilities.items()},
            "transition_prob": self.transition_prob,
            "features_used": self.features_used,
        }


class RegimeDetector(ABC):
    """Abstract base class for regime detectors."""

    @abstractmethod
    def fit(
        self, features: list[RegimeFeatures], labels: list[RegimeType] | None = None
    ) -> None:
        """Train the detector on historical data."""

    @abstractmethod
    async def predict(self, features: RegimeFeatures) -> RegimeOutput:
        """Predict regime for given features."""

    @abstractmethod
    async def predict_proba(self, features: RegimeFeatures) -> dict[RegimeType, float]:
        """Get probabilities for each regime."""


class RuleBasedRegimeDetector(RegimeDetector):
    """
    Rule-based regime detector using configurable thresholds.

    This serves as a baseline and fallback when ML models fail.
    Uses simple heuristics based on volatility, momentum, and trend strength.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)

        # Default thresholds (can be overridden by config)
        self.thresholds = {
            "volatility": {"high": 0.7, "low": 0.3},
            "momentum": {"threshold": 0.6},
            "adx": {"trending": 25, "strong_trend": 40},
            "rsi": {"oversold": 30, "overbought": 70, "neutral_range": 20},
        }

        if config:
            self._update_thresholds(config)

        self.logger.info(
            "Initialized RuleBasedRegimeDetector", thresholds=self.thresholds
        )

    def _update_thresholds(self, config: dict[str, Any]) -> None:
        """Update thresholds from configuration."""
        if "rules" in config:
            rules = config["rules"]
            for category, values in rules.items():
                if category in self.thresholds:
                    self.thresholds[category].update(values)

    def fit(
        self, features: list[RegimeFeatures], labels: list[RegimeType] | None = None
    ) -> None:
        """Rule-based detector doesn't require training."""
        self.logger.info(f"Rule-based detector fit called with {len(features)} samples")
        # Could potentially optimize thresholds based on historical performance

    async def predict(self, features: RegimeFeatures) -> RegimeOutput:
        """Predict regime using rule-based logic."""

        # Get probabilities for each regime
        probabilities = await self.predict_proba(features)

        # Select regime with highest probability
        regime = max(probabilities.keys(), key=lambda k: probabilities[k])
        confidence = probabilities[regime]

        # Determine which features contributed most
        features_used = self._get_contributing_features(features, regime)

        return RegimeOutput(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            features_used=features_used,
        )

    async def predict_proba(self, features: RegimeFeatures) -> dict[RegimeType, float]:
        """Calculate probabilities for each regime using rules."""

        scores = {
            RegimeType.TRENDING: 0.0,
            RegimeType.MEAN_REVERTING: 0.0,
            RegimeType.CHOPPY: 0.0,
        }

        # Trending indicators
        if features.adx > self.thresholds["adx"]["trending"] / 100:
            scores[RegimeType.TRENDING] += 0.3

        if features.momentum > self.thresholds["momentum"]["threshold"]:
            scores[RegimeType.TRENDING] += 0.2

        if features.trend_strength > 0.6:
            scores[RegimeType.TRENDING] += 0.2

        if abs(features.macd_signal) > 0.5:
            scores[RegimeType.TRENDING] += 0.15

        # Mean-reverting indicators
        if features.adx < self.thresholds["adx"]["trending"] / 100:
            scores[RegimeType.MEAN_REVERTING] += 0.2

        if features.mean_reversion > 0.6:
            scores[RegimeType.MEAN_REVERTING] += 0.3

        if (
            features.rsi < self.thresholds["rsi"]["oversold"] / 100
            or features.rsi > self.thresholds["rsi"]["overbought"] / 100
        ):
            scores[RegimeType.MEAN_REVERTING] += 0.2

        if features.bb_width < self.thresholds["volatility"]["low"]:
            scores[RegimeType.MEAN_REVERTING] += 0.15

        # Choppy indicators
        if features.realized_vol > self.thresholds["volatility"]["high"]:
            scores[RegimeType.CHOPPY] += 0.3

        if features.vol_ratio > 1.3:
            scores[RegimeType.CHOPPY] += 0.2

        if abs(features.momentum) < 0.2 and features.adx < 20 / 100:
            scores[RegimeType.CHOPPY] += 0.25

        if features.trend_strength < 0.3:
            scores[RegimeType.CHOPPY] += 0.15

        # Normalize to probabilities
        total_score = sum(scores.values())
        if total_score == 0:
            # Default to equal probabilities if no clear signals
            return {regime: 1 / 3 for regime in scores.keys()}

        probabilities = {
            regime: score / total_score for regime, score in scores.items()
        }

        return probabilities

    def _get_contributing_features(
        self, features: RegimeFeatures, regime: RegimeType
    ) -> list[str]:
        """Identify which features contributed most to the regime classification."""

        contributing = []

        if regime == RegimeType.TRENDING:
            if features.adx > self.thresholds["adx"]["trending"] / 100:
                contributing.append("adx")
            if features.momentum > self.thresholds["momentum"]["threshold"]:
                contributing.append("momentum")
            if features.trend_strength > 0.6:
                contributing.append("trend_strength")

        elif regime == RegimeType.MEAN_REVERTING:
            if features.mean_reversion > 0.6:
                contributing.append("mean_reversion")
            if (
                features.rsi < self.thresholds["rsi"]["oversold"] / 100
                or features.rsi > self.thresholds["rsi"]["overbought"] / 100
            ):
                contributing.append("rsi")
            if features.bb_width < self.thresholds["volatility"]["low"]:
                contributing.append("bb_width")

        elif regime == RegimeType.CHOPPY:
            if features.realized_vol > self.thresholds["volatility"]["high"]:
                contributing.append("realized_vol")
            if features.vol_ratio > 1.3:
                contributing.append("vol_ratio")
            if features.trend_strength < 0.3:
                contributing.append("trend_strength")

        return contributing


class StatisticalRegimeDetector(RegimeDetector):
    """
    Regime detector using unsupervised clustering (Gaussian Mixture Model).

    This detector fits a GMM to the feature space and maps the resulting
    clusters to market regimes based on the cluster centroids.
    """

    def __init__(
        self,
        n_components: int = 3,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        # We only map to the first 3 regimes for this statistical model
        if n_components > 3:
            self.logger.warning(
                "StatisticalRegimeDetector only supports up to 3 regimes (T, MR, C). Adjusting n_components.",
                requested=n_components,
            )
            n_components = 3

        self.n_components = n_components
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)
        self.scaler = StandardScaler()
        self.cluster_regime_map: dict[int, RegimeType] = {}

        self.logger.info(
            "Initialized StatisticalRegimeDetector", n_components=n_components
        )

    def fit(
        self, features: list[RegimeFeatures], labels: list[RegimeType] | None = None
    ) -> None:
        """
        Fit the GMM to the historical feature data and map clusters to regimes.
        """
        if not features:
            self.logger.warning("fit called with no features. Model cannot be trained.")
            return

        feature_array = np.array([f.to_array() for f in features])

        # Scale features
        scaled_features = self.scaler.fit_transform(feature_array)

        # Fit GMM
        self.model.fit(scaled_features)
        self.logger.info("GMM model fitted.", inertia=self.model.lower_bound_)

        # Map clusters to regimes by analyzing centroids
        self._map_clusters_to_regimes(scaled_features)

    def _map_clusters_to_regimes(self, scaled_features: np.ndarray) -> None:
        """
        Map GMM clusters to regimes by analyzing cluster characteristics.
        """
        centroids = self.model.means_

        feature_names = list(RegimeFeatures.__annotations__.keys())

        regime_scores = {}
        for i in range(self.n_components):
            centroid_features = dict(zip(feature_names, centroids[i]))

            trending_score = (
                centroid_features.get("adx", 0)
                + centroid_features.get("momentum", 0)
                + centroid_features.get("trend_strength", 0)
            )
            mr_score = centroid_features.get(
                "mean_reversion", 0
            ) - centroid_features.get("adx", 0)
            choppy_score = (
                centroid_features.get("realized_vol", 0)
                + centroid_features.get("bb_width", 0)
                - centroid_features.get("trend_strength", 0)
            )

            regime_scores[i] = {
                RegimeType.TRENDING: trending_score,
                RegimeType.MEAN_REVERTING: mr_score,
                RegimeType.CHOPPY: choppy_score,
            }

        assigned_regimes = {}
        for cluster_id in sorted(
            regime_scores,
            key=lambda cid: max(regime_scores[cid].values()),
            reverse=True,
        ):
            best_regime = max(
                regime_scores[cluster_id], key=regime_scores[cluster_id].get
            )
            if best_regime not in assigned_regimes.values():
                assigned_regimes[cluster_id] = best_regime

        self.cluster_regime_map = assigned_regimes
        self.logger.info(
            "Mapped clusters to regimes",
            mapping={k: v.value for k, v in self.cluster_regime_map.items()},
        )

        if len(self.cluster_regime_map) != self.n_components:
            self.logger.warning(
                "Could not create a 1-to-1 mapping for all regimes. Some regimes may not be detected."
            )

    async def predict(self, features: RegimeFeatures) -> RegimeOutput:
        """
        Predict the regime for a single feature vector.
        """
        if not hasattr(self.model, "means_"):
            self.logger.warning(
                "Model not fitted yet. Returning default CHOPPY regime."
            )
            return RegimeOutput(
                regime=RegimeType.CHOPPY,
                confidence=0.0,
                probabilities={
                    r: 1 / 3
                    for r in [
                        RegimeType.TRENDING,
                        RegimeType.MEAN_REVERTING,
                        RegimeType.CHOPPY,
                    ]
                },
            )

        probabilities = await self.predict_proba(features)

        regime = max(probabilities.keys(), key=lambda k: probabilities[k])
        confidence = probabilities[regime]

        return RegimeOutput(
            regime=regime, confidence=confidence, probabilities=probabilities
        )

    async def predict_proba(self, features: RegimeFeatures) -> dict[RegimeType, float]:
        """
        Calculate probabilities for each regime.
        """
        if not self.cluster_regime_map:
            return {
                regime: 1 / 3
                for regime in [
                    RegimeType.TRENDING,
                    RegimeType.MEAN_REVERTING,
                    RegimeType.CHOPPY,
                ]
            }

        feature_array = features.to_array().reshape(1, -1)
        scaled_feature = self.scaler.transform(feature_array)

        cluster_probabilities = self.model.predict_proba(scaled_feature)[0]

        regime_probs = {
            regime: 0.0
            for regime in [
                RegimeType.TRENDING,
                RegimeType.MEAN_REVERTING,
                RegimeType.CHOPPY,
            ]
        }
        for cluster_id, prob in enumerate(cluster_probabilities):
            if cluster_id in self.cluster_regime_map:
                regime = self.cluster_regime_map[cluster_id]
                if regime in regime_probs:
                    regime_probs[regime] += prob

        total_prob = sum(regime_probs.values())
        if total_prob > 0:
            regime_probs = {k: v / total_prob for k, v in regime_probs.items()}

        return regime_probs


class MetaLearnerRegimeDetector(RegimeDetector):
    """
    A sophisticated regime detector that uses a meta-learner (stacking) approach.
    It combines predictions from multiple base models (XGBoost, LSTM, CNN)
    and uses a final estimator (e.g., Logistic Regression) to make the final
    regime prediction. This approach aims to improve accuracy and robustness.
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or structlog.get_logger(__name__)
        # Initialize predictors without loading models. Loading happens on-demand.
        self.base_models: dict[
            str, XGBoostPredictor | LSTMPredictor | CNNPredictor
        ] = {
            "xgboost": XGBoostPredictor(),
            "lstm": LSTMPredictor(
                n_features=13
            ),  # Explicitly set n_features for clarity
            # CNNPredictor pads the 13 features to 100 (10x10 image). This is suboptimal
            # but functional. A future improvement would be a 1D CNN architecture.
            "cnn": CNNPredictor(n_features=100, image_size=10),
        }
        self.meta_learner = LogisticRegression()
        self.is_fitted = False
        self.models_loaded = False
        self.logger.info("Initialized MetaLearnerRegimeDetector")

    async def _load_base_models(self):
        """Loads the base models from disk if they haven't been loaded yet."""
        if self.models_loaded:
            return

        self.logger.info("Loading base models for MetaLearner...")
        try:
            self.base_models["xgboost"].load("models/xgboost_model.json")
            self.base_models["lstm"].load("models/lstm_model.h5")
            self.base_models["cnn"].load("models/cnn_model.h5")
            self.models_loaded = True
            self.logger.info("All base models loaded successfully.")
        except FileNotFoundError as e:
            self.logger.error(
                "Failed to load a base model. Models may need to be trained first.",
                error=str(e),
            )
            raise

    def fit(
        self, features: list[RegimeFeatures], labels: list[RegimeType] | None = None
    ) -> None:
        """
        Trains both the base models and the meta-learner.
        This method should be called with a substantial amount of historical data.
        """
        self.logger.info(
            f"Fitting MetaLearnerRegimeDetector with {len(features)} samples."
        )

        pd.DataFrame([f.to_dict() for f in features])

        if labels is None:
            self.logger.warning("No labels provided for fitting. Skipping training.")
            return

        # Placeholder for a full training pipeline:
        # 1. Train each base model:
        #    self.base_models["xgboost"].train(feature_df, labels)
        #    ...
        # 2. Get predictions from base models to create a meta-dataset.
        # 3. Train the meta-learner on the meta-dataset.

        self.is_fitted = True
        self.logger.info("MetaLearnerRegimeDetector fitting complete (placeholder).")

    async def predict(self, features: RegimeFeatures) -> RegimeOutput:
        """
        Predict the regime using the full meta-learner stack.
        Ensures models are loaded, gets predictions from base models,
        and then uses the meta-learner for the final prediction.
        """
        await self._load_base_models()
        probabilities = await self.predict_proba(features)

        if not probabilities:
            self.logger.warning("Predict_proba returned empty probabilities.")
            return RegimeOutput(
                regime=RegimeType.CHOPPY,
                confidence=0.0,
                probabilities={rt: 0.0 for rt in RegimeType},
            )

        best_regime = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_regime]

        return RegimeOutput(
            regime=best_regime, confidence=confidence, probabilities=probabilities
        )

    async def predict_proba(self, features: RegimeFeatures) -> dict[RegimeType, float]:
        """Get probabilities for each regime from the meta-learner."""
        await self._load_base_models()

        feature_df = pd.DataFrame([features.to_dict()])

        base_model_outputs = []
        for name, model in self.base_models.items():
            try:
                # This logic is complex because of different model interfaces.
                # A better long-term solution would be a standard `predict_proba` on each predictor.
                if name == "xgboost":
                    probs = model.model.predict_proba(feature_df)
                elif name == "lstm":
                    # LSTM needs sequences. We create a sequence of length `timesteps`
                    # by repeating the current feature vector. This is a simplification.
                    X_seq = np.tile(feature_df.values, (model.timesteps, 1))
                    X_seq = X_seq.reshape(1, model.timesteps, model.n_features)
                    probs = model.model.predict(X_seq)
                elif name == "cnn":
                    # CNN needs a padded image
                    if feature_df.shape[1] < model.n_features:
                        padding = pd.DataFrame(
                            np.zeros(
                                (
                                    len(feature_df),
                                    model.n_features - feature_df.shape[1],
                                )
                            ),
                            index=feature_df.index,
                        )
                        X_padded = pd.concat([feature_df, padding], axis=1)
                    else:
                        X_padded = feature_df.iloc[:, : model.n_features]
                    X_pred = X_padded.head(1).values.reshape(
                        1, model.image_size, model.image_size, 1
                    )
                    probs = model.model.predict(X_pred)
                else:
                    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])  # Fallback

                base_model_outputs.append(probs[0])
            except Exception as e:
                self.logger.error(
                    f"Error predicting with base model '{name}'",
                    error=str(e),
                    exc_info=True,
                )
                num_classes = 3
                base_model_outputs.append(np.full(num_classes, 1.0 / num_classes))

        if not base_model_outputs:
            self.logger.error("No base models returned predictions.")
            return {regime: 1.0 / len(RegimeType) for regime in RegimeType}

        # Combine predictions for meta-learner
        meta_features = np.concatenate(base_model_outputs).reshape(1, -1)

        if not self.is_fitted:
            self.logger.warning(
                "Meta-learner is not fitted. Returning averaged base model probabilities."
            )
            avg_probs = np.mean(np.array(base_model_outputs), axis=0)

            target_regimes = [
                RegimeType.TRENDING,
                RegimeType.MEAN_REVERTING,
                RegimeType.CHOPPY,
            ]
            if len(avg_probs) != len(target_regimes):
                self.logger.error(
                    f"Mismatch in probability vector length. Expected {len(target_regimes)}, got {len(avg_probs)}"
                )
                return {regime: 1.0 / len(target_regimes) for regime in target_regimes}
            return {
                regime: float(prob) for regime, prob in zip(target_regimes, avg_probs)
            }

        final_probs = self.meta_learner.predict_proba(meta_features)[0]

        num_classes = len(final_probs)
        regime_map = list(RegimeType)[:num_classes]

        return {regime: float(prob) for regime, prob in zip(regime_map, final_probs)}


def create_sample_features(
    n_samples: int = 100,
    regime_type: RegimeType | None = None,
    noise_level: float = 0.1,
) -> list[RegimeFeatures]:
    """
    Generate sample feature data for testing and validation.

    Args:
        n_samples: Number of samples to generate
        regime_type: Target regime type (None for mixed)
        noise_level: Amount of random noise to add

    Returns:
        List of RegimeFeatures
    """

    np.random.seed(42)  # For reproducible results
    samples = []

    for i in range(n_samples):
        if regime_type is None:
            # Random regime selection
            target_regime = np.random.choice(list(RegimeType))
        else:
            target_regime = regime_type

        # Generate features based on regime type
        if target_regime == RegimeType.TRENDING:
            features = RegimeFeatures(
                atr=np.random.normal(0.5, 0.1),
                bb_width=np.random.normal(0.6, 0.1),
                realized_vol=np.random.normal(0.4, 0.1),
                vol_ratio=np.random.normal(1.1, 0.2),
                macd_signal=np.random.normal(0.6, 0.2),
                macd_histogram=np.random.normal(0.4, 0.2),
                adx=np.random.normal(0.6, 0.1),  # Normalized ADX
                rsi=np.random.normal(0.65, 0.1),  # Normalized RSI
                momentum=np.random.normal(0.7, 0.2),
                macro_surprise=np.random.normal(0.1, 0.1),
                macro_sentiment=np.random.normal(0.2, 0.2),
                trend_strength=np.random.normal(0.8, 0.1),
                mean_reversion=np.random.normal(0.2, 0.1),
            )

        elif target_regime == RegimeType.MEAN_REVERTING:
            features = RegimeFeatures(
                atr=np.random.normal(0.3, 0.1),
                bb_width=np.random.normal(0.2, 0.1),
                realized_vol=np.random.normal(0.2, 0.1),
                vol_ratio=np.random.normal(0.9, 0.1),
                macd_signal=np.random.normal(0.1, 0.1),
                macd_histogram=np.random.normal(0.05, 0.1),
                adx=np.random.normal(0.2, 0.1),  # Low ADX
                rsi=np.random.normal(0.5, 0.2),  # Neutral RSI
                momentum=np.random.normal(0.1, 0.1),
                macro_surprise=np.random.normal(0.0, 0.1),
                macro_sentiment=np.random.normal(0.0, 0.1),
                trend_strength=np.random.normal(0.2, 0.1),
                mean_reversion=np.random.normal(0.8, 0.1),
            )

        else:  # CHOPPY or other new regimes
            features = RegimeFeatures(
                atr=np.random.normal(0.8, 0.1),
                bb_width=np.random.normal(0.9, 0.1),
                realized_vol=np.random.normal(0.7, 0.1),
                vol_ratio=np.random.normal(1.4, 0.2),
                macd_signal=np.random.normal(0.0, 0.2),
                macd_histogram=np.random.normal(0.0, 0.2),
                adx=np.random.normal(0.15, 0.1),  # Very low ADX
                rsi=np.random.normal(0.5, 0.2),
                momentum=np.random.normal(0.0, 0.2),
                macro_surprise=np.random.normal(0.2, 0.2),
                macro_sentiment=np.random.normal(-0.1, 0.2),
                trend_strength=np.random.normal(0.1, 0.1),
                mean_reversion=np.random.normal(0.3, 0.2),
            )

        # Add noise and clip to valid ranges
        feature_dict = features.to_dict()
        for key, value in feature_dict.items():
            noise = np.random.normal(0, noise_level)
            feature_dict[key] = np.clip(value + noise, 0, 1)

        samples.append(RegimeFeatures.from_dict(feature_dict))

    return samples


# Demo function
async def demo_regime_detection():
    """Demonstrate regime detection functionality."""

    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(__name__)
    logger.info("Starting Regime Detection Demo")

    # Initialize detector
    detector = RuleBasedRegimeDetector(logger=logger)

    # Test with different market scenarios
    scenarios = [
        (
            "Strong Uptrend",
            RegimeFeatures(
                atr=0.4,
                bb_width=0.5,
                realized_vol=0.3,
                vol_ratio=1.0,
                macd_signal=0.8,
                macd_histogram=0.6,
                adx=0.7,
                rsi=0.75,
                momentum=0.9,
                macro_surprise=0.2,
                macro_sentiment=0.4,
                trend_strength=0.9,
                mean_reversion=0.1,
            ),
        ),
        (
            "Range-bound Market",
            RegimeFeatures(
                atr=0.2,
                bb_width=0.15,
                realized_vol=0.1,
                vol_ratio=0.8,
                macd_signal=0.1,
                macd_histogram=0.05,
                adx=0.15,
                rsi=0.5,
                momentum=0.1,
                macro_surprise=0.0,
                macro_sentiment=0.0,
                trend_strength=0.2,
                mean_reversion=0.9,
            ),
        ),
        (
            "High Volatility Chop",
            RegimeFeatures(
                atr=0.9,
                bb_width=0.8,
                realized_vol=0.8,
                vol_ratio=1.6,
                macd_signal=0.1,
                macd_histogram=0.0,
                adx=0.1,
                rsi=0.45,
                momentum=0.0,
                macro_surprise=0.3,
                macro_sentiment=-0.2,
                trend_strength=0.1,
                mean_reversion=0.3,
            ),
        ),
    ]

    for scenario_name, features in scenarios:
        logger.info(f"\n--- {scenario_name} ---")
        result = await detector.predict(features)

        logger.info(f"Detected Regime: {result.regime.value}")
        logger.info(f"Confidence: {result.confidence:.2f}")
        logger.info(
            f"Probabilities: {[(k.value, f'{v:.2f}') for k, v in result.probabilities.items()]}"
        )
        logger.info(f"Key Features: {result.features_used}")

    # Generate and test sample data
    logger.info("\n--- Testing Sample Data Generation ---")
    for regime_type in RegimeType:
        samples = create_sample_features(n_samples=5, regime_type=regime_type)
        logger.info(f"Generated {len(samples)} samples for {regime_type.value}")

        # Test first sample
        result = await detector.predict(samples[0])
        logger.info(
            f"Sample prediction: {result.regime.value} (confidence: {result.confidence:.2f})"
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_regime_detection())
