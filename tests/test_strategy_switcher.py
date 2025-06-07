"""
Comprehensive test suite for Strategy Switcher - Core Logic.

Tests the central decision layer that dynamically selects trading strategies
based on market regime, ML prediction strength, and feature context.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional

from core.strategy_switcher import (
    StrategySwitcher, 
    StrategySwitcherConfig,
    MLPrediction,
    StrategyDecision
)
from core.regime_detector import RegimeType, RegimeOutput
from core.interfaces import Signal, Strategy
from core.interfaces.trading_interfaces import OrderSide
from strategies.grid_martingale import GridMartingaleStrategy
from strategies.breakout_trend import BreakoutTrendStrategy
from strategies.time_scalping import TimeScalpingStrategy


class TestStrategySwitcherConfig:
    """Test strategy switcher configuration."""
    
    def test_config_loading(self):
        """Test configuration loading from YAML."""
        config = StrategySwitcherConfig()
        
        assert config.decision_timeout_ms == 100
        assert config.confidence_threshold == 0.6
        assert config.regime_mapping['trending'] == 'breakout_trend'
        assert config.regime_mapping['mean_reverting'] == 'grid_martingale'
        assert config.regime_mapping['choppy'] == 'time_scalping'
    
    def test_ensemble_weights(self):
        """Test ensemble weight configuration."""
        config = StrategySwitcherConfig()
        
        weights = config.ensemble_weights
        assert 'breakout_trend' in weights
        assert 'grid_martingale' in weights
        assert 'time_scalping' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.1  # Should sum to ~1.0


class TestMLPrediction:
    """Test ML prediction dataclass."""
    
    def test_ml_prediction_creation(self):
        """Test ML prediction creation."""
        prediction = MLPrediction(
            direction=OrderSide.BUY,
            confidence=0.8,
            strength=0.7,
            features={'rsi': 30, 'atr': 0.0001},
            timestamp=datetime.now()
        )
        
        assert prediction.direction == OrderSide.BUY
        assert prediction.confidence == 0.8
        assert prediction.strength == 0.7
        assert 'rsi' in prediction.features


class TestStrategyDecision:
    """Test strategy decision dataclass."""
    
    def test_strategy_decision_creation(self):
        """Test strategy decision creation."""
        signal = Signal(
            symbol='EURUSD',
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7,
            strategy_name='grid_martingale',
            timestamp=datetime.now()
        )
        
        decision = StrategyDecision(
            selected_strategy='grid_martingale',
            signal=signal,
            confidence=0.8,
            regime_used=RegimeType.MEAN_REVERTING,
            ml_prediction_used=None,
            decision_time_ms=50.0
        )
        
        assert decision.selected_strategy == 'grid_martingale'
        assert decision.signal == signal
        assert decision.regime_used == RegimeType.MEAN_REVERTING
        assert decision.decision_time_ms == 50.0


class TestStrategySwitcher:
    """Test the main strategy switcher functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=StrategySwitcherConfig)
        config.decision_timeout_ms = 100
        config.ensemble_mode = False
        config.confidence_threshold = 0.6
        config.regime_mapping = {
            'trending': 'breakout_trend',
            'mean_reverting': 'grid_martingale',
            'choppy': 'time_scalping'
        }
        config.ensemble_weights = {
            'breakout_trend': 0.4,
            'grid_martingale': 0.3,
            'time_scalping': 0.3
        }
        config.config = {
            'grid_martingale': {},
            'breakout_trend': {},
            'time_scalping': {}
        }
        return config
    
    @pytest.fixture
    def switcher(self, mock_config):
        """Create strategy switcher with mocked dependencies."""
        with patch('core.strategy_switcher.GridMartingaleStrategy'), \
             patch('core.strategy_switcher.BreakoutTrendStrategy'), \
             patch('core.strategy_switcher.TimeScalpingStrategy'):
            return StrategySwitcher(config=mock_config)
    
    def test_initialization(self, switcher):
        """Test strategy switcher initialization."""
        assert len(switcher.strategies) == 3
        assert 'grid_martingale' in switcher.strategies
        assert 'breakout_trend' in switcher.strategies
        assert 'time_scalping' in switcher.strategies
        assert switcher.decision_count == 0
    
    @pytest.mark.asyncio
    async def test_regime_to_strategy_mapping(self, switcher):
        """Test regime to strategy mapping."""
        # Test trending regime
        trending_regime = RegimeOutput(
            regime=RegimeType.TRENDING,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.8,
                RegimeType.MEAN_REVERTING: 0.1,
                RegimeType.CHOPPY: 0.1
            }
        )
        
        strategy_name = switcher._map_regime_to_strategy(trending_regime.regime)
        assert strategy_name == 'breakout_trend'
        
        # Test mean-reverting regime
        mean_reverting_regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.1,
                RegimeType.MEAN_REVERTING: 0.8,
                RegimeType.CHOPPY: 0.1
            }
        )
        
        strategy_name = switcher._map_regime_to_strategy(mean_reverting_regime.regime)
        assert strategy_name == 'grid_martingale'
        
        # Test choppy regime
        choppy_regime = RegimeOutput(
            regime=RegimeType.CHOPPY,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.1,
                RegimeType.MEAN_REVERTING: 0.1,
                RegimeType.CHOPPY: 0.8
            }
        )
        
        strategy_name = switcher._map_regime_to_strategy(choppy_regime.regime)
        assert strategy_name == 'time_scalping'
    
    @pytest.mark.asyncio
    async def test_single_strategy_selection(self, switcher):
        """Test single strategy selection mode."""
        # Mock strategy to return a signal
        mock_signal = Signal(
            symbol='EURUSD',
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7,
            strategy_name='grid_martingale',
            timestamp=datetime.now()
        )
        
        switcher.strategies['grid_martingale'].generate_signal = AsyncMock(return_value=mock_signal)
        
        regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.1,
                RegimeType.MEAN_REVERTING: 0.8,
                RegimeType.CHOPPY: 0.1
            }
        )
        
        decision = await switcher.choose_strategy(regime)
        
        assert decision.selected_strategy == 'grid_martingale'
        assert decision.signal == mock_signal
        assert decision.regime_used == RegimeType.MEAN_REVERTING
        assert decision.decision_time_ms < switcher.config.decision_timeout_ms
    
    @pytest.mark.asyncio
    async def test_ml_prediction_weighting(self, switcher):
        """Test ML prediction weighting of signals."""
        original_signal = Signal(
            symbol='EURUSD',
            side=OrderSide.BUY,
            strength=0.6,
            confidence=0.7,
            strategy_name='grid_martingale',
            timestamp=datetime.now()
        )
        
        # Store original values since method modifies in place
        original_strength = original_signal.strength
        original_confidence = original_signal.confidence
        
        # Test agreeing ML prediction (should boost signal)
        agreeing_prediction = MLPrediction(
            direction=OrderSide.BUY,
            confidence=0.8,
            strength=0.9,
            features={},
            timestamp=datetime.now()
        )
        
        weighted_signal = switcher._apply_ml_weighting(original_signal, agreeing_prediction)
        # Signal strength should be boosted (but capped at 1.0)
        assert weighted_signal.strength >= original_strength
        assert weighted_signal.confidence >= original_confidence
        
        # Test disagreeing ML prediction (should reduce signal)
        disagreeing_prediction = MLPrediction(
            direction=OrderSide.SELL,
            confidence=0.8,
            strength=0.9,
            features={},
            timestamp=datetime.now()
        )
        
        # Create a fresh signal for disagreement test
        fresh_signal = Signal(
            symbol='EURUSD',
            side=OrderSide.BUY,
            strength=0.6,
            confidence=0.7,
            strategy_name='grid_martingale',
            timestamp=datetime.now()
        )
        
        # Store original values
        fresh_strength = fresh_signal.strength
        fresh_confidence = fresh_signal.confidence
        
        weighted_signal = switcher._apply_ml_weighting(fresh_signal, disagreeing_prediction)
        assert weighted_signal.strength < fresh_strength
        assert weighted_signal.confidence < fresh_confidence
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, switcher):
        """Test that low confidence regimes are filtered out."""
        low_confidence_regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.3,  # Below threshold of 0.6
            probabilities={
                RegimeType.TRENDING: 0.4,
                RegimeType.MEAN_REVERTING: 0.3,
                RegimeType.CHOPPY: 0.3
            }
        )
        
        decision = await switcher.choose_strategy(low_confidence_regime)
        
        assert decision.signal is None
        assert decision.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_decision_latency_requirement(self, switcher):
        """Test that decisions are made within 100ms."""
        regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.1,
                RegimeType.MEAN_REVERTING: 0.8,
                RegimeType.CHOPPY: 0.1
            }
        )
        
        # Mock strategy to return quickly
        mock_signal = Signal(
            symbol='EURUSD',
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7,
            strategy_name='grid_martingale',
            timestamp=datetime.now()
        )
        
        switcher.strategies['grid_martingale'].generate_signal = AsyncMock(return_value=mock_signal)
        
        start_time = time.perf_counter()
        decision = await switcher.choose_strategy(regime)
        end_time = time.perf_counter()
        
        actual_time_ms = (end_time - start_time) * 1000
        
        assert decision.decision_time_ms < 100  # Should be under 100ms
        assert actual_time_ms < 100  # Actual measurement should also be under 100ms
    
    @pytest.mark.asyncio
    async def test_ensemble_mode(self, switcher):
        """Test ensemble mode signal blending."""
        switcher.config.ensemble_mode = True
        
        # Mock all strategies to return signals
        mock_signals = {
            'grid_martingale': Signal(
                symbol='EURUSD', side=OrderSide.BUY, strength=0.8, confidence=0.7,
                strategy_name='grid_martingale', timestamp=datetime.now()
            ),
            'breakout_trend': Signal(
                symbol='EURUSD', side=OrderSide.BUY, strength=0.6, confidence=0.8,
                strategy_name='breakout_trend', timestamp=datetime.now()
            ),
            'time_scalping': Signal(
                symbol='EURUSD', side=OrderSide.SELL, strength=0.4, confidence=0.6,
                strategy_name='time_scalping', timestamp=datetime.now()
            )
        }
        
        for strategy_name, signal in mock_signals.items():
            switcher.strategies[strategy_name].generate_signal = AsyncMock(return_value=signal)
        
        regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.2,
                RegimeType.MEAN_REVERTING: 0.6,
                RegimeType.CHOPPY: 0.2
            }
        )
        
        decision = await switcher.choose_strategy(regime)
        
        assert decision.selected_strategy == 'ensemble'
        assert decision.ensemble_signals is not None
        assert len(decision.ensemble_signals) == 3
        assert decision.signal is not None  # Should have blended signal
    
    @pytest.mark.asyncio
    async def test_error_handling(self, switcher):
        """Test error handling in strategy selection."""
        # Mock strategy to raise exception
        switcher.strategies['grid_martingale'].generate_signal = AsyncMock(
            side_effect=Exception("Strategy error")
        )
        
        regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.1,
                RegimeType.MEAN_REVERTING: 0.8,
                RegimeType.CHOPPY: 0.1
            }
        )
        
        decision = await switcher.choose_strategy(regime)
        
        assert decision.signal is None
        assert decision.confidence == 0.0
    
    def test_performance_metrics_tracking(self, switcher):
        """Test performance metrics tracking."""
        # Simulate some decisions
        switcher._update_performance_metrics(50.0)
        switcher._update_performance_metrics(75.0)
        switcher._update_performance_metrics(60.0)
        
        metrics = switcher.get_performance_metrics()
        
        assert metrics['total_decisions'] == 3
        assert metrics['avg_decision_time_ms'] == (50.0 + 75.0 + 60.0) / 3
    
    @pytest.mark.asyncio
    async def test_strategy_parameter_updates(self, switcher):
        """Test strategy parameter updates."""
        new_params = {'grid_spacing_atr_multiplier': 0.7}
        
        switcher.strategies['grid_martingale'].update_parameters = AsyncMock()
        
        await switcher.update_strategy_parameters('grid_martingale', new_params)
        
        switcher.strategies['grid_martingale'].update_parameters.assert_called_once_with(new_params)
    
    def test_available_strategies(self, switcher):
        """Test getting available strategies."""
        strategies = switcher.get_available_strategies()
        
        assert len(strategies) == 3
        assert 'grid_martingale' in strategies
        assert 'breakout_trend' in strategies
        assert 'time_scalping' in strategies


class TestIntegrationScenarios:
    """Test integration scenarios with realistic market conditions."""
    
    @pytest.fixture
    def real_switcher(self):
        """Create a real strategy switcher for integration tests."""
        return StrategySwitcher()
    
    @pytest.mark.asyncio
    async def test_trending_market_scenario(self, real_switcher):
        """Test strategy selection in trending market conditions."""
        trending_regime = RegimeOutput(
            regime=RegimeType.TRENDING,
            confidence=0.85,
            probabilities={
                RegimeType.TRENDING: 0.85,
                RegimeType.MEAN_REVERTING: 0.10,
                RegimeType.CHOPPY: 0.05
            }
        )
        
        ml_prediction = MLPrediction(
            direction=OrderSide.BUY,
            confidence=0.8,
            strength=0.7,
            features={'adx': 35, 'atr': 0.0002, 'rsi': 65},
            timestamp=datetime.now()
        )
        
        # Mock market data
        market_data = Mock()
        market_data.symbol = 'EURUSD'
        market_data.close = 1.1000
        
        features = {
            'close': 1.1000,
            'atr_14': 0.0002,
            'adx': 35,
            'rsi_14': 65,
            'macd_signal': 0.5
        }
        
        decision = await real_switcher.choose_strategy(
            regime=trending_regime,
            ml_prediction=ml_prediction,
            features=features,
            market_data=market_data
        )
        
        assert decision.selected_strategy == 'breakout_trend'
        assert decision.regime_used == RegimeType.TRENDING
        assert decision.decision_time_ms < 100
    
    @pytest.mark.asyncio
    async def test_mean_reverting_market_scenario(self, real_switcher):
        """Test strategy selection in mean-reverting market conditions."""
        mean_reverting_regime = RegimeOutput(
            regime=RegimeType.MEAN_REVERTING,
            confidence=0.9,
            probabilities={
                RegimeType.TRENDING: 0.05,
                RegimeType.MEAN_REVERTING: 0.90,
                RegimeType.CHOPPY: 0.05
            }
        )
        
        ml_prediction = MLPrediction(
            direction=OrderSide.BUY,
            confidence=0.75,
            strength=0.6,
            features={'rsi': 25, 'bb_width': 0.0001, 'atr': 0.0001},
            timestamp=datetime.now()
        )
        
        market_data = Mock()
        market_data.symbol = 'EURUSD'
        market_data.close = 1.1000
        
        features = {
            'close': 1.1000,
            'atr_14': 0.0001,
            'rsi_14': 25,
            'bb_width': 0.0001,
            'adx': 15
        }
        
        decision = await real_switcher.choose_strategy(
            regime=mean_reverting_regime,
            ml_prediction=ml_prediction,
            features=features,
            market_data=market_data
        )
        
        assert decision.selected_strategy == 'grid_martingale'
        assert decision.regime_used == RegimeType.MEAN_REVERTING
        assert decision.decision_time_ms < 100
    
    @pytest.mark.asyncio
    async def test_choppy_market_scenario(self, real_switcher):
        """Test strategy selection in choppy market conditions."""
        choppy_regime = RegimeOutput(
            regime=RegimeType.CHOPPY,
            confidence=0.8,
            probabilities={
                RegimeType.TRENDING: 0.1,
                RegimeType.MEAN_REVERTING: 0.1,
                RegimeType.CHOPPY: 0.8
            }
        )
        
        ml_prediction = MLPrediction(
            direction=OrderSide.SELL,
            confidence=0.7,
            strength=0.5,
            features={'rsi': 55, 'atr': 0.00015, 'adx': 20},
            timestamp=datetime.now()
        )
        
        market_data = Mock()
        market_data.symbol = 'EURUSD'
        market_data.close = 1.1000
        
        features = {
            'close': 1.1000,
            'atr_14': 0.00015,
            'rsi_14': 55,
            'adx': 20,
            'momentum': 0.1
        }
        
        decision = await real_switcher.choose_strategy(
            regime=choppy_regime,
            ml_prediction=ml_prediction,
            features=features,
            market_data=market_data
        )
        
        assert decision.selected_strategy == 'time_scalping'
        assert decision.regime_used == RegimeType.CHOPPY
        assert decision.decision_time_ms < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 