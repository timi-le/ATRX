"""
Strategy Switcher - Core Logic for FX AI-Quant Trading System.

This module implements the central decision layer that dynamically selects and configures
trading strategies based on market regime, ML prediction strength, and feature context.
Uses edge-enhanced strategy logic for optimal performance.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import structlog
import yaml

from core.interfaces import Signal, Strategy
from core.interfaces.trading_interfaces import OrderSide
from core.ml_predictor import Prediction
from core.regime_detector import RegimeOutput, RegimeType
from strategies.breakout_trend import BreakoutTrendStrategy
from strategies.grid_martingale import GridMartingaleStrategy
from strategies.time_scalping import TimeScalpingStrategy


@dataclass
class StrategyDecision:
    """Strategy selection decision output."""

    selected_strategy: str
    signal: Signal | None
    confidence: float
    regime_used: RegimeType
    ml_prediction_used: Prediction | None
    decision_time_ms: float
    ensemble_signals: dict[str, Signal] | None = None


class StrategySwitcherConfig:
    """Configuration for strategy switcher."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.switcher_config = self.config["strategy_switcher"]
        self.decision_timeout_ms = self.switcher_config["decision_timeout_ms"]
        self.ensemble_mode = self.switcher_config["ensemble_mode"]
        self.confidence_threshold = self.switcher_config["confidence_threshold"]
        self.regime_mapping = self.switcher_config["regime_mapping"]
        self.ensemble_weights = self.switcher_config["ensemble_weights"]


class StrategySwitcher:
    """
    Central strategy switcher that maps regimes and ML signals to trading strategies.

    Implements edge-enhanced strategy logic with:
    - Dynamic strategy selection based on market regime
    - ML prediction integration with confidence weighting
    - Optional ensemble signal blending
    - Sub-100ms decision latency
    - Comprehensive logging and metrics
    """

    def __init__(
        self,
        config: StrategySwitcherConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config or StrategySwitcherConfig()
        self.logger = logger or structlog.get_logger(__name__)

        # Initialize strategies
        self.strategies: dict[str, Strategy] = {}
        self._initialize_strategies()

        # Performance tracking
        self.decision_count = 0
        self.total_decision_time_ms = 0.0
        self.last_decision: StrategyDecision | None = None

        self.logger.info(
            "StrategySwitcher initialized",
            ensemble_mode=self.config.ensemble_mode,
            strategies=list(self.strategies.keys()),
            decision_timeout_ms=self.config.decision_timeout_ms,
        )

    def _initialize_strategies(self) -> None:
        """Initialize all available strategies."""
        try:
            # Grid Martingale for mean-reverting regime
            self.strategies["grid_martingale"] = GridMartingaleStrategy(
                config=self.config.config["grid_martingale"], logger=self.logger
            )

            # Breakout Trend for trending regime
            self.strategies["breakout_trend"] = BreakoutTrendStrategy(
                config=self.config.config["breakout_trend"], logger=self.logger
            )

            # Time Scalping for choppy regime
            self.strategies["time_scalping"] = TimeScalpingStrategy(
                config=self.config.config["time_scalping"], logger=self.logger
            )

            self.logger.info("All strategies initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize strategies: {e}")
            raise

    async def choose_strategy(
        self,
        regime: RegimeOutput,
        ml_prediction: Prediction | None = None,
        features: dict[str, float] | None = None,
        market_data: Any | None = None,
    ) -> StrategyDecision:
        """
        Choose and execute the most appropriate strategy based on inputs.

        Args:
            regime: Current market regime from regime detector
            ml_prediction: ML model prediction with confidence
            features: Feature vector from feature engine
            market_data: Current market data (OHLCV)

        Returns:
            StrategyDecision with selected strategy and generated signal
        """
        start_time = time.perf_counter()

        try:
            # Validate inputs
            if regime.confidence < self.config.confidence_threshold:
                self.logger.warning(
                    "Regime confidence below threshold",
                    regime_confidence=regime.confidence,
                    threshold=self.config.confidence_threshold,
                )
                decision_time_ms = (time.perf_counter() - start_time) * 1000
                return self._create_no_signal_decision(
                    regime, ml_prediction, decision_time_ms
                )

            if self.config.ensemble_mode:
                decision = await self._ensemble_strategy_selection(
                    regime, ml_prediction, features, market_data
                )
            else:
                decision = await self._single_strategy_selection(
                    regime, ml_prediction, features, market_data
                )

            # Calculate decision time
            decision_time_ms = (time.perf_counter() - start_time) * 1000
            decision.decision_time_ms = decision_time_ms

            # Update performance metrics
            self._update_performance_metrics(decision_time_ms)

            # Check timeout
            if decision_time_ms > self.config.decision_timeout_ms:
                self.logger.warning(
                    "Decision timeout exceeded",
                    decision_time_ms=decision_time_ms,
                    timeout_ms=self.config.decision_timeout_ms,
                )

            self.last_decision = decision

            self.logger.info(
                "Strategy decision completed",
                selected_strategy=decision.selected_strategy,
                regime=regime.regime.value,
                confidence=decision.confidence,
                decision_time_ms=decision_time_ms,
                has_signal=decision.signal is not None,
            )

            return decision

        except Exception as e:
            decision_time_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Strategy selection failed: {e}", decision_time_ms=decision_time_ms
            )
            return self._create_no_signal_decision(
                regime, ml_prediction, decision_time_ms
            )

    async def _single_strategy_selection(
        self,
        regime: RegimeOutput,
        ml_prediction: Prediction | None,
        features: dict[str, float] | None,
        market_data: Any | None,
    ) -> StrategyDecision:
        """Select single strategy based on regime."""

        # Map regime to strategy
        strategy_name = self._map_regime_to_strategy(regime.regime)
        strategy = self.strategies[strategy_name]

        # Generate signal from selected strategy
        signal = await strategy.generate_signal(
            market_data=market_data, features=features, regime=regime.regime.value
        )

        # Apply ML prediction weighting if available
        if signal and ml_prediction:
            signal = self._apply_ml_weighting(signal, ml_prediction)

        # Calculate overall confidence
        confidence = self._calculate_confidence(regime, ml_prediction, signal)

        return StrategyDecision(
            selected_strategy=strategy_name,
            signal=signal,
            confidence=confidence,
            regime_used=regime.regime,
            ml_prediction_used=ml_prediction,
            decision_time_ms=0.0,  # Will be set by caller
        )

    async def _ensemble_strategy_selection(
        self,
        regime: RegimeOutput,
        ml_prediction: Prediction | None,
        features: dict[str, float] | None,
        market_data: Any | None,
    ) -> StrategyDecision:
        """Generate ensemble signal from multiple strategies."""

        ensemble_signals = {}

        # Generate signals from all strategies
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = await strategy.generate_signal(
                    market_data=market_data,
                    features=features,
                    regime=regime.regime.value,
                )
                if signal:
                    ensemble_signals[strategy_name] = signal
            except Exception as e:
                self.logger.warning(f"Strategy {strategy_name} failed: {e}")

        # Blend signals using ensemble weights
        blended_signal = self._blend_ensemble_signals(
            ensemble_signals, regime, ml_prediction
        )

        # Calculate ensemble confidence
        confidence = self._calculate_ensemble_confidence(
            ensemble_signals, regime, ml_prediction
        )

        return StrategyDecision(
            selected_strategy="ensemble",
            signal=blended_signal,
            confidence=confidence,
            regime_used=regime.regime,
            ml_prediction_used=ml_prediction,
            decision_time_ms=0.0,
            ensemble_signals=ensemble_signals,
        )

    def _map_regime_to_strategy(self, regime: RegimeType) -> str:
        """Map regime to its designated strategy from config."""
        return self.config.regime_mapping.get(regime.value, "default_strategy_name")

    def _apply_ml_weighting(self, signal: Signal, ml_prediction: Prediction) -> Signal:
        """
        Adjust signal properties based on ML model confidence.
        For example, scale order size by prediction probability.
        """
        # Map integer signal to OrderSide enum
        ml_side = OrderSide.BUY if ml_prediction.signal == 1 else OrderSide.SELL

        # Ensure the ML signal direction aligns with the strategy signal
        if ml_side != signal.side:
            # If they don't align, we can reduce confidence or size, or even veto.
            # For now, let's reduce the size.
            signal.size *= 0.5  # Halve the size if signals conflict
            self.logger.warning(
                "ML signal conflicts with strategy signal.",
                ml_signal=ml_side.value,
                strategy_signal=signal.side.value,
            )
            return signal

        # Scale size by ML confidence
        signal.size *= ml_prediction.probability

        # We could also adjust other parameters like take profit/stop loss
        # based on the ML prediction's metadata if available.

        return signal

    def _calculate_confidence(
        self,
        regime: RegimeOutput,
        ml_prediction: Prediction | None,
        signal: Signal | None,
    ) -> float:
        """Calculate overall confidence score."""

        if not signal:
            return 0.0

        base_confidence = regime.confidence * signal.strength

        if ml_prediction:
            # Use probability from the new Prediction object
            ml_weight = ml_prediction.probability

            # Simple average, could be more sophisticated
            return (base_confidence + ml_weight) / 2

        return base_confidence

    def _blend_ensemble_signals(
        self,
        ensemble_signals: dict[str, Signal],
        regime: RegimeOutput,
        ml_prediction: Prediction | None,
    ) -> Signal | None:
        """Blend multiple strategy signals using weighted ensemble."""

        if not ensemble_signals:
            return None

        # Calculate weighted averages
        total_weight = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        signal_votes = {"BUY": 0.0, "SELL": 0.0}

        for strategy_name, signal in ensemble_signals.items():
            weight = self.config.ensemble_weights.get(strategy_name, 0.33)

            # Apply regime-based weight adjustment
            if strategy_name == self._map_regime_to_strategy(regime.regime):
                weight *= 1.5  # Boost primary strategy for current regime

            total_weight += weight
            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight

            # Vote for direction
            signal_votes[signal.side.value.upper()] += weight

        if total_weight == 0:
            return None

        # Normalize weights
        weighted_strength /= total_weight
        weighted_confidence /= total_weight

        # Determine final direction
        final_side = (
            OrderSide.BUY
            if signal_votes["BUY"] > signal_votes["SELL"]
            else OrderSide.SELL
        )

        # Create blended signal
        reference_signal = list(ensemble_signals.values())[0]

        blended_signal = Signal(
            symbol=reference_signal.symbol,
            side=final_side,
            strength=weighted_strength,
            confidence=weighted_confidence,
            strategy_name="ensemble",
            timestamp=datetime.now(),
            features=reference_signal.features,
        )

        return blended_signal

    def _calculate_ensemble_confidence(
        self,
        ensemble_signals: dict[str, Signal],
        regime: RegimeOutput,
        ml_prediction: Prediction | None,
    ) -> float:
        """Calculate confidence for ensemble decision."""

        if not ensemble_signals:
            return 0.0

        # Agreement factor (how much strategies agree)
        buy_votes = sum(1 for s in ensemble_signals.values() if s.side == OrderSide.BUY)
        total_votes = len(ensemble_signals)
        agreement = max(buy_votes, total_votes - buy_votes) / total_votes

        # Average signal confidence
        avg_confidence = np.mean([s.confidence for s in ensemble_signals.values()])

        # Regime confidence
        regime_confidence = regime.confidence

        # Combined confidence
        ensemble_confidence = (
            agreement * 0.4 + avg_confidence * 0.4 + regime_confidence * 0.2
        )

        return min(1.0, ensemble_confidence)

    def _create_no_signal_decision(
        self,
        regime: RegimeOutput,
        ml_prediction: Prediction | None,
        decision_time_ms: float,
    ) -> StrategyDecision:
        """Create decision with no signal."""

        return StrategyDecision(
            selected_strategy="no_signal",
            signal=None,
            confidence=0.0,
            regime_used=regime.regime,
            ml_prediction_used=ml_prediction,
            decision_time_ms=decision_time_ms,
        )

    def _update_performance_metrics(self, decision_time_ms: float) -> None:
        """Update performance tracking metrics."""
        self.decision_count += 1
        self.total_decision_time_ms += decision_time_ms

    def get_performance_metrics(self) -> dict[str, float]:
        """Get performance metrics."""
        if self.decision_count == 0:
            return {"avg_decision_time_ms": 0.0, "total_decisions": 0}

        return {
            "avg_decision_time_ms": self.total_decision_time_ms / self.decision_count,
            "total_decisions": self.decision_count,
            "last_decision_time_ms": self.last_decision.decision_time_ms
            if self.last_decision
            else 0.0,
        }

    async def update_strategy_parameters(
        self, strategy_name: str, params: dict[str, Any]
    ) -> None:
        """Update parameters for a specific strategy."""
        if strategy_name in self.strategies:
            await self.strategies[strategy_name].update_parameters(params)
            self.logger.info(f"Updated parameters for {strategy_name}", params=params)
        else:
            self.logger.warning(f"Strategy {strategy_name} not found")

    def get_strategy_parameters(self, strategy_name: str) -> dict[str, Any]:
        """Get current parameters for a strategy."""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].get_parameters()
        return {}

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategies."""
        return list(self.strategies.keys())


# Factory function for easy instantiation
def create_strategy_switcher(
    config_path: str = "config/strategy_params.yaml",
    logger: structlog.stdlib.BoundLogger | None = None,
) -> StrategySwitcher:
    """Create and initialize strategy switcher."""
    config = StrategySwitcherConfig(config_path)
    return StrategySwitcher(config, logger)
