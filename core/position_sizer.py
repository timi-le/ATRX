"""
Kelly Criterion Position Sizer for FX AI-Quant Trading System.

This module implements optimal position sizing using the Kelly criterion with:
- Dynamic Kelly fraction calculation: f* = (b × p - q) / b
- Volatility-based position scaling
- Drawdown protection mechanisms
- Risk profile management
- Emergency controls and constraints
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import structlog
import yaml

from core.interfaces.trading_interfaces import OrderSide


class RiskProfile(Enum):
    """Risk profile types."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class VolatilityRegime(Enum):
    """Volatility regime types."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class DrawdownLevel(Enum):
    """Drawdown severity levels."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class PositionSizeResult:
    """Position sizing calculation result."""

    position_size: float  # Final position size (% of capital)
    confidence: float  # Overall confidence in sizing
    volatility_scaled: float  # Volatility scaling factor applied
    kelly_raw: float  # Raw Kelly fraction before adjustments
    kelly_scaled: float  # Kelly fraction after scaling
    risk_profile: str  # Risk profile used
    volatility_regime: str  # Current volatility regime
    drawdown_level: str  # Current drawdown level
    constraints_applied: list[str] = field(default_factory=list)  # Applied constraints
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class TradeSignalInput:
    """Input signal for position sizing."""

    symbol: str
    side: OrderSide
    signal_confidence: float  # Signal confidence (0.0 to 1.0)
    take_profit_pips: float | None  # Take profit in pips
    stop_loss_pips: float | None  # Stop loss in pips
    reward_risk_ratio: float | None  # TP/SL ratio (if pips not available)
    win_probability: float | None  # Historical or ML-estimated win rate
    current_price: float
    volatility_atr: float  # Current ATR
    timestamp: datetime
    strategy_name: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Current portfolio state for risk management."""

    total_capital: float
    current_drawdown: float  # Current drawdown (% of capital)
    daily_pnl: float  # Today's P&L
    open_positions: list[dict[str, Any]]  # Current open positions
    volatility_history: list[float]  # Recent volatility measurements
    performance_history: list[dict[str, Any]]  # Historical performance data


class PositionSizerConfig:
    """Configuration for position sizer."""

    def __init__(self, config_path: str = "config/risk_settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Kelly Criterion parameters
        kelly_config = self.config["kelly_criterion"]
        self.max_position_cap = kelly_config["max_position_cap"]
        self.min_position_size = kelly_config["min_position_size"]
        self.kelly_scaling_factor = kelly_config["kelly_scaling_factor"]
        self.default_win_probability = kelly_config["default_win_probability"]
        self.default_reward_risk_ratio = kelly_config["default_reward_risk_ratio"]

        # Volatility scaling
        vol_config = self.config["volatility_scaling"]
        self.volatility_scaling_enabled = vol_config["enabled"]
        self.volatility_lookback = vol_config["lookback_period"]
        self.base_volatility_threshold = vol_config["base_volatility_threshold"]
        self.max_volatility_threshold = vol_config["max_volatility_threshold"]
        self.low_volatility_multiplier = vol_config["low_volatility_multiplier"]
        self.high_volatility_multiplier = vol_config["high_volatility_multiplier"]
        self.low_vol_percentile = vol_config["low_vol_percentile"]
        self.high_vol_percentile = vol_config["high_vol_percentile"]

        # Drawdown protection
        dd_config = self.config["drawdown_protection"]
        self.drawdown_protection_enabled = dd_config["enabled"]
        self.mild_drawdown_threshold = dd_config["mild_drawdown_threshold"]
        self.moderate_drawdown_threshold = dd_config["moderate_drawdown_threshold"]
        self.severe_drawdown_threshold = dd_config["severe_drawdown_threshold"]
        self.mild_drawdown_scaling = dd_config["mild_drawdown_scaling"]
        self.moderate_drawdown_scaling = dd_config["moderate_drawdown_scaling"]
        self.severe_drawdown_scaling = dd_config["severe_drawdown_scaling"]
        self.recovery_threshold = dd_config["recovery_threshold"]

        # Risk profiles
        self.risk_profiles = self.config["risk_profiles"]

        # Confidence mapping
        conf_config = self.config["confidence_mapping"]
        self.confidence_boost_factor = conf_config["confidence_boost_factor"]
        self.confidence_penalty_factor = conf_config["confidence_penalty_factor"]
        self.high_confidence_threshold = conf_config["high_confidence_threshold"]
        self.low_confidence_threshold = conf_config["low_confidence_threshold"]

        # Constraints
        constraints_config = self.config["constraints"]
        self.max_concurrent_positions = constraints_config["max_concurrent_positions"]
        self.max_total_exposure = constraints_config["max_total_exposure"]
        self.max_correlation_exposure = constraints_config["max_correlation_exposure"]
        self.correlation_threshold = constraints_config["correlation_threshold"]
        self.max_currency_exposure = constraints_config["max_currency_exposure"]

        # Emergency controls
        emergency_config = self.config["emergency_controls"]
        self.emergency_max_position = emergency_config["emergency_max_position"]
        self.max_daily_loss_threshold = emergency_config["max_daily_loss_threshold"]
        self.volatility_spike_threshold = emergency_config["volatility_spike_threshold"]
        self.emergency_mode_duration_hours = emergency_config[
            "emergency_mode_duration_hours"
        ]


class KellyPositionSizer:
    """
    Kelly Criterion Position Sizer with advanced risk management.

    Implements optimal position sizing using the Kelly criterion:
    f* = (b × p - q) / b

    Where:
    - f* = optimal fraction of capital to risk
    - b = reward/risk ratio (TP/SL)
    - p = probability of winning
    - q = probability of losing (1 - p)

    Features:
    - Volatility-based scaling
    - Drawdown protection
    - Risk profile management
    - Emergency controls
    - Portfolio constraints
    """

    def __init__(
        self,
        config: PositionSizerConfig | None = None,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config or PositionSizerConfig()
        self.risk_profile = risk_profile
        self.logger = logger or structlog.get_logger(__name__)

        # State tracking
        self.volatility_history: deque = deque(maxlen=self.config.volatility_lookback)
        self.sizing_history: list[PositionSizeResult] = []
        self.emergency_mode_until: datetime | None = None
        self.last_drawdown_check: datetime | None = None

        # Performance tracking
        self.total_sizing_decisions = 0
        self.successful_sizings = 0
        self.average_kelly_fraction = 0.0

        self.logger.info(
            "KellyPositionSizer initialized",
            risk_profile=risk_profile.value,
            kelly_scaling_factor=self.config.kelly_scaling_factor,
            max_position_cap=self.config.max_position_cap,
        )

    async def calculate_position_size(
        self, signal: TradeSignalInput, portfolio_state: PortfolioState
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Kelly criterion.

        Args:
            signal: Trading signal with confidence and risk parameters
            portfolio_state: Current portfolio state for risk management

        Returns:
            PositionSizeResult with optimal position size and metadata
        """
        start_time = time.perf_counter()

        try:
            # Update volatility history
            self._update_volatility_history(signal.volatility_atr)

            # Check emergency mode
            if self._is_emergency_mode(portfolio_state):
                return self._create_emergency_result(signal, "Emergency mode active")

            # Calculate win probability
            win_probability = self._calculate_win_probability(signal)

            # Calculate reward/risk ratio
            reward_risk_ratio = self._calculate_reward_risk_ratio(signal)

            # Calculate raw Kelly fraction
            kelly_raw = self._calculate_kelly_fraction(
                win_probability, reward_risk_ratio
            )

            # Apply Kelly scaling
            kelly_scaled = self._apply_kelly_scaling(kelly_raw)

            # Apply volatility scaling
            volatility_regime, volatility_scaling = self._calculate_volatility_scaling(
                signal
            )

            # Apply drawdown protection
            drawdown_level, drawdown_scaling = self._calculate_drawdown_scaling(
                portfolio_state
            )

            # Calculate base position size
            base_position_size = kelly_scaled * volatility_scaling * drawdown_scaling

            # Apply constraints
            final_position_size, constraints_applied = self._apply_constraints(
                base_position_size, signal, portfolio_state
            )

            # Calculate overall confidence
            confidence = self._calculate_sizing_confidence(
                signal,
                win_probability,
                reward_risk_ratio,
                volatility_regime,
                drawdown_level,
            )

            # Create result
            result = PositionSizeResult(
                position_size=final_position_size,
                confidence=confidence,
                volatility_scaled=volatility_scaling,
                kelly_raw=kelly_raw,
                kelly_scaled=kelly_scaled,
                risk_profile=self.risk_profile.value,
                volatility_regime=volatility_regime.value,
                drawdown_level=drawdown_level.value,
                constraints_applied=constraints_applied,
                metadata={
                    "win_probability": win_probability,
                    "reward_risk_ratio": reward_risk_ratio,
                    "drawdown_scaling": drawdown_scaling,
                    "signal_confidence": signal.signal_confidence,
                    "calculation_time_ms": (time.perf_counter() - start_time) * 1000,
                    "symbol": signal.symbol,
                    "strategy": signal.strategy_name,
                },
            )

            # Update tracking
            self._update_performance_tracking(result)

            self.logger.info(
                "Position size calculated",
                symbol=signal.symbol,
                position_size=final_position_size,
                kelly_raw=kelly_raw,
                confidence=confidence,
                volatility_regime=volatility_regime.value,
                drawdown_level=drawdown_level.value,
            )

            return result

        except Exception as e:
            self.logger.error(f"Position sizing calculation failed: {e}")
            return self._create_emergency_result(signal, f"Calculation error: {e}")

    def _calculate_win_probability(self, signal: TradeSignalInput) -> float:
        """Calculate win probability from signal and historical data."""

        # Start with provided win probability or default
        if signal.win_probability is not None:
            base_probability = signal.win_probability
        else:
            base_probability = self.config.default_win_probability

        # Adjust based on signal confidence
        confidence_adjustment = 0.0

        if signal.signal_confidence >= self.config.high_confidence_threshold:
            # High confidence signals get a boost
            confidence_adjustment = (
                (signal.signal_confidence - self.config.high_confidence_threshold)
                / (1.0 - self.config.high_confidence_threshold)
                * self.config.confidence_boost_factor
            )
        elif signal.signal_confidence <= self.config.low_confidence_threshold:
            # Low confidence signals get penalized
            confidence_adjustment = -(
                (self.config.low_confidence_threshold - signal.signal_confidence)
                / self.config.low_confidence_threshold
                * self.config.confidence_penalty_factor
            )

        # Apply adjustment and clamp to valid range
        adjusted_probability = base_probability + confidence_adjustment
        return max(0.01, min(0.99, adjusted_probability))

    def _calculate_reward_risk_ratio(self, signal: TradeSignalInput) -> float:
        """Calculate reward/risk ratio from signal parameters."""

        if signal.reward_risk_ratio is not None:
            return max(0.1, signal.reward_risk_ratio)

        if signal.take_profit_pips is not None and signal.stop_loss_pips is not None:
            if signal.stop_loss_pips > 0:
                return signal.take_profit_pips / signal.stop_loss_pips

        # Use default if no ratio can be calculated
        return self.config.default_reward_risk_ratio

    def _calculate_kelly_fraction(
        self, win_probability: float, reward_risk_ratio: float
    ) -> float:
        """Calculate raw Kelly fraction: f* = (b × p - q) / b"""

        p = win_probability
        q = 1.0 - p
        b = reward_risk_ratio

        # Kelly formula
        kelly_fraction = (b * p - q) / b

        # Ensure non-negative (don't take trades with negative edge)
        return max(0.0, kelly_fraction)

    def _apply_kelly_scaling(self, kelly_raw: float) -> float:
        """Apply Kelly scaling factor based on risk profile."""

        # Get scaling factor from risk profile
        profile_config = self.config.risk_profiles[self.risk_profile.value]
        scaling_factor = profile_config.get(
            "kelly_scaling_factor", self.config.kelly_scaling_factor
        )

        return kelly_raw * scaling_factor

    def _calculate_volatility_scaling(
        self, signal: TradeSignalInput
    ) -> tuple[VolatilityRegime, float]:
        """Calculate volatility-based position scaling."""

        if (
            not self.config.volatility_scaling_enabled
            or len(self.volatility_history) < 5
        ):
            return VolatilityRegime.NORMAL, 1.0

        current_volatility = signal.volatility_atr
        volatility_array = np.array(list(self.volatility_history))

        # Calculate percentiles
        low_percentile = np.percentile(volatility_array, self.config.low_vol_percentile)
        high_percentile = np.percentile(
            volatility_array, self.config.high_vol_percentile
        )

        # Determine volatility regime
        if current_volatility <= low_percentile:
            regime = VolatilityRegime.LOW
            scaling = self.config.low_volatility_multiplier
        elif current_volatility >= high_percentile:
            regime = VolatilityRegime.HIGH
            scaling = self.config.high_volatility_multiplier
        else:
            regime = VolatilityRegime.NORMAL
            scaling = 1.0

        # Apply risk profile volatility multiplier
        profile_config = self.config.risk_profiles[self.risk_profile.value]
        volatility_multiplier = profile_config.get("volatility_multiplier", 1.0)
        scaling *= volatility_multiplier

        return regime, scaling

    def _calculate_drawdown_scaling(
        self, portfolio_state: PortfolioState
    ) -> tuple[DrawdownLevel, float]:
        """Calculate drawdown-based position scaling."""

        if not self.config.drawdown_protection_enabled:
            return DrawdownLevel.NONE, 1.0

        current_drawdown = abs(portfolio_state.current_drawdown)

        # Determine drawdown level and scaling
        if current_drawdown >= self.config.severe_drawdown_threshold:
            return DrawdownLevel.SEVERE, self.config.severe_drawdown_scaling
        elif current_drawdown >= self.config.moderate_drawdown_threshold:
            return DrawdownLevel.MODERATE, self.config.moderate_drawdown_scaling
        elif current_drawdown >= self.config.mild_drawdown_threshold:
            return DrawdownLevel.MILD, self.config.mild_drawdown_scaling
        else:
            return DrawdownLevel.NONE, 1.0

    def _apply_constraints(
        self,
        base_position_size: float,
        signal: TradeSignalInput,
        portfolio_state: PortfolioState,
    ) -> tuple[float, list[str]]:
        """Apply position sizing constraints."""

        constraints_applied = []
        position_size = base_position_size

        # Apply maximum position cap
        profile_config = self.config.risk_profiles[self.risk_profile.value]
        max_cap = profile_config.get("max_position_cap", self.config.max_position_cap)

        if position_size > max_cap:
            position_size = max_cap
            constraints_applied.append(f"max_position_cap_{max_cap}")

        # Apply minimum position size
        if position_size < self.config.min_position_size:
            position_size = self.config.min_position_size
            constraints_applied.append(
                f"min_position_size_{self.config.min_position_size}"
            )

        # Check maximum concurrent positions
        if len(portfolio_state.open_positions) >= self.config.max_concurrent_positions:
            position_size = 0.0
            constraints_applied.append("max_concurrent_positions")

        # Check total exposure limit
        current_exposure = sum(
            pos.get("size", 0) for pos in portfolio_state.open_positions
        )
        if current_exposure + position_size > self.config.max_total_exposure:
            max_allowed = max(0, self.config.max_total_exposure - current_exposure)
            position_size = min(position_size, max_allowed)
            constraints_applied.append("max_total_exposure")

        # Check daily loss limit (emergency trigger)
        daily_loss_pct = abs(portfolio_state.daily_pnl) / portfolio_state.total_capital
        if daily_loss_pct >= self.config.max_daily_loss_threshold:
            position_size = min(position_size, self.config.emergency_max_position)
            constraints_applied.append("daily_loss_limit")

        return position_size, constraints_applied

    def _calculate_sizing_confidence(
        self,
        signal: TradeSignalInput,
        win_probability: float,
        reward_risk_ratio: float,
        volatility_regime: VolatilityRegime,
        drawdown_level: DrawdownLevel,
    ) -> float:
        """Calculate overall confidence in the position sizing decision."""

        # Base confidence from signal
        base_confidence = signal.signal_confidence

        # Adjust for Kelly parameters quality
        if signal.win_probability is not None and signal.reward_risk_ratio is not None:
            parameter_quality = 1.0  # High quality - both parameters provided
        elif signal.win_probability is not None or signal.reward_risk_ratio is not None:
            parameter_quality = 0.8  # Medium quality - one parameter provided
        else:
            parameter_quality = 0.6  # Low quality - using defaults

        # Adjust for volatility regime
        if volatility_regime == VolatilityRegime.HIGH:
            volatility_adjustment = 0.9  # Slightly lower confidence in high vol
        elif volatility_regime == VolatilityRegime.LOW:
            volatility_adjustment = 1.1  # Slightly higher confidence in low vol
        else:
            volatility_adjustment = 1.0

        # Adjust for drawdown level
        if drawdown_level == DrawdownLevel.SEVERE:
            drawdown_adjustment = 0.7
        elif drawdown_level == DrawdownLevel.MODERATE:
            drawdown_adjustment = 0.8
        elif drawdown_level == DrawdownLevel.MILD:
            drawdown_adjustment = 0.9
        else:
            drawdown_adjustment = 1.0

        # Calculate final confidence
        confidence = (
            base_confidence
            * parameter_quality
            * volatility_adjustment
            * drawdown_adjustment
        )

        return max(0.0, min(1.0, confidence))

    def _is_emergency_mode(self, portfolio_state: PortfolioState) -> bool:
        """Check if emergency mode should be activated."""

        # Check if already in emergency mode
        if self.emergency_mode_until and datetime.now() < self.emergency_mode_until:
            return True

        # Check daily loss threshold
        daily_loss_pct = abs(portfolio_state.daily_pnl) / portfolio_state.total_capital
        if daily_loss_pct >= self.config.max_daily_loss_threshold:
            self._activate_emergency_mode("Daily loss threshold exceeded")
            return True

        # Check volatility spike
        if len(self.volatility_history) >= 5:
            recent_volatility = np.mean(list(self.volatility_history)[-5:])
            historical_volatility = (
                np.mean(list(self.volatility_history)[:-5])
                if len(self.volatility_history) > 5
                else recent_volatility
            )

            if (
                historical_volatility > 0
                and recent_volatility / historical_volatility
                >= self.config.volatility_spike_threshold
            ):
                self._activate_emergency_mode("Volatility spike detected")
                return True

        return False

    def _activate_emergency_mode(self, reason: str) -> None:
        """Activate emergency mode."""
        self.emergency_mode_until = datetime.now() + timedelta(
            hours=self.config.emergency_mode_duration_hours
        )
        self.logger.warning(
            f"Emergency mode activated: {reason}", until=self.emergency_mode_until
        )

    def _create_emergency_result(
        self, signal: TradeSignalInput, reason: str
    ) -> PositionSizeResult:
        """Create emergency position sizing result."""
        return PositionSizeResult(
            position_size=self.config.emergency_max_position,
            confidence=0.1,
            volatility_scaled=0.1,
            kelly_raw=0.0,
            kelly_scaled=0.0,
            risk_profile=self.risk_profile.value,
            volatility_regime="emergency",
            drawdown_level="emergency",
            constraints_applied=["emergency_mode"],
            metadata={"emergency_reason": reason},
        )

    def _update_volatility_history(self, volatility: float) -> None:
        """Update volatility history for regime detection."""
        if volatility > 0:
            self.volatility_history.append(volatility)

    def _update_performance_tracking(self, result: PositionSizeResult) -> None:
        """Update performance tracking metrics."""
        self.total_sizing_decisions += 1

        if result.position_size > 0:
            self.successful_sizings += 1

        # Update average Kelly fraction
        if result.kelly_raw > 0:
            self.average_kelly_fraction = (
                self.average_kelly_fraction * (self.total_sizing_decisions - 1)
                + result.kelly_raw
            ) / self.total_sizing_decisions

        # Store sizing history
        self.sizing_history.append(result)

        # Keep only recent history
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get position sizer performance metrics."""
        if self.total_sizing_decisions == 0:
            return {"total_decisions": 0, "success_rate": 0.0}

        success_rate = self.successful_sizings / self.total_sizing_decisions

        recent_results = (
            self.sizing_history[-100:]
            if len(self.sizing_history) >= 100
            else self.sizing_history
        )

        return {
            "total_decisions": self.total_sizing_decisions,
            "successful_sizings": self.successful_sizings,
            "success_rate": success_rate,
            "average_kelly_fraction": self.average_kelly_fraction,
            "average_position_size": np.mean([r.position_size for r in recent_results])
            if recent_results
            else 0.0,
            "average_confidence": np.mean([r.confidence for r in recent_results])
            if recent_results
            else 0.0,
            "emergency_mode_active": self.emergency_mode_until is not None
            and datetime.now() < self.emergency_mode_until,
            "volatility_history_length": len(self.volatility_history),
        }

    def update_risk_profile(self, new_profile: RiskProfile) -> None:
        """Update risk profile."""
        old_profile = self.risk_profile
        self.risk_profile = new_profile
        self.logger.info(
            f"Risk profile updated from {old_profile.value} to {new_profile.value}"
        )

    def reset_emergency_mode(self) -> None:
        """Manually reset emergency mode."""
        self.emergency_mode_until = None
        self.logger.info("Emergency mode manually reset")


# Factory function for easy instantiation
def create_position_sizer(
    config_path: str = "config/risk_settings.yaml",
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> KellyPositionSizer:
    """Create and initialize Kelly position sizer."""
    config = PositionSizerConfig(config_path)
    return KellyPositionSizer(config, risk_profile, logger)
