"""
Unit tests for the Strategy Switcher module.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from core.interfaces import Signal
from core.interfaces.trading_interfaces import OrderSide
from core.ml_predictor import Prediction
from core.regime_detector import RegimeOutput, RegimeType
from core.strategy_switcher import StrategySwitcher


@pytest.fixture
def mock_config():
    """Fixture for a mock StrategySwitcherConfig."""
    config = Mock()
    config.confidence_threshold = 0.6
    config.decision_timeout_ms = 100.0
    config.ensemble_mode = False
    config.regime_mapping = {
        "trending": "breakout_trend",
        "mean_reverting": "grid_martingale",
        "choppy": "time_scalping",
    }
    config.config = {
        "grid_martingale": {"param1": "value1"},
        "breakout_trend": {"param1": "value2"},
        "time_scalping": {"param1": "value3"},
    }
    return config


@pytest.fixture
def mock_strategies():
    """Fixture for mock strategies."""
    # Create async mock for generate_signal
    trend_signal = Signal(
        symbol="EUR/USD",
        side=OrderSide.BUY,
        size=1.0,
        order_type="market",
        strength=0.8,
        confidence=0.7,
        strategy_name="breakout_trend",
        timestamp=datetime.now(),
    )
    trend_strategy = MagicMock()
    trend_strategy.generate_signal = AsyncMock(return_value=trend_signal)

    mr_signal = Signal(
        symbol="EUR/USD",
        side=OrderSide.SELL,
        size=0.5,
        order_type="market",
        strength=0.6,
        confidence=0.65,
        strategy_name="grid_martingale",
        timestamp=datetime.now(),
    )
    mr_strategy = MagicMock()
    mr_strategy.generate_signal = AsyncMock(return_value=mr_signal)

    return {
        "breakout_trend": trend_strategy,
        "grid_martingale": mr_strategy,
        "time_scalping": MagicMock(),
    }


@pytest.fixture
def strategy_switcher(mock_config, mock_strategies):
    """Fixture for a configured StrategySwitcher."""
    with patch.object(StrategySwitcher, "_initialize_strategies", return_value=None):
        switcher = StrategySwitcher(config=mock_config)
        switcher.strategies = mock_strategies
        return switcher


@pytest.mark.asyncio
async def test_trending_regime_selects_correct_strategy(strategy_switcher):
    """Test that a TRENDING regime selects the breakout_trend strategy."""
    regime = RegimeOutput(regime=RegimeType.TRENDING, confidence=0.8, probabilities={})

    decision = await strategy_switcher.choose_strategy(regime=regime)

    assert decision.selected_strategy == "breakout_trend"
    assert decision.signal.side == OrderSide.BUY


@pytest.mark.asyncio
async def test_mean_reverting_regime_selects_correct_strategy(strategy_switcher):
    """Test that a MEAN_REVERTING regime selects the grid_martingale strategy."""
    regime = RegimeOutput(
        regime=RegimeType.MEAN_REVERTING, confidence=0.9, probabilities={}
    )

    decision = await strategy_switcher.choose_strategy(regime=regime)

    assert decision.selected_strategy == "grid_martingale"
    assert decision.signal.side == OrderSide.SELL


@pytest.mark.asyncio
async def test_low_confidence_regime_produces_no_signal(strategy_switcher):
    """Test that a low confidence regime results in no signal."""
    regime = RegimeOutput(
        regime=RegimeType.TRENDING, confidence=0.5, probabilities={}
    )  # Below threshold 0.6

    decision = await strategy_switcher.choose_strategy(regime=regime)

    assert decision.selected_strategy == "no_signal"
    assert decision.signal is None


@pytest.mark.asyncio
async def test_ml_weighting_scales_signal_size(strategy_switcher):
    """Test that ML prediction probability correctly scales the signal size."""
    regime = RegimeOutput(regime=RegimeType.TRENDING, confidence=0.8, probabilities={})
    # ML prediction agrees with strategy signal
    ml_pred = Prediction(signal=1, probability=0.75)  # signal=1 maps to OrderSide.BUY

    decision = await strategy_switcher.choose_strategy(
        regime=regime, ml_prediction=ml_pred
    )

    assert decision.signal.size == 1.0 * 0.75  # Initial size * probability


@pytest.mark.asyncio
async def test_ml_weighting_with_conflicting_signal(strategy_switcher):
    """Test that a conflicting ML signal reduces the signal size."""
    regime = RegimeOutput(regime=RegimeType.TRENDING, confidence=0.8, probabilities={})
    # ML prediction disagrees with strategy signal
    ml_pred = Prediction(signal=-1, probability=0.9)  # signal=-1 maps to OrderSide.SELL

    decision = await strategy_switcher.choose_strategy(
        regime=regime, ml_prediction=ml_pred
    )

    assert decision.signal.size == 1.0 * 0.5  # Halved due to conflict
