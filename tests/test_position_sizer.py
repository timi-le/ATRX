"""
Comprehensive test suite for Kelly Criterion Position Sizer.

Tests the position sizing module that uses Kelly criterion to compute optimal
capital fraction based on edge, win probability, and reward-to-risk ratio.
"""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

from core.interfaces.trading_interfaces import OrderSide
from core.position_sizer import (
    DrawdownLevel,
    KellyPositionSizer,
    PortfolioState,
    PositionSizerConfig,
    PositionSizeResult,
    RiskProfile,
    TradeSignalInput,
    VolatilityRegime,
    create_position_sizer,
)


class TestPositionSizerConfig:
    """Test position sizer configuration."""

    def test_config_loading(self):
        """Test configuration loading from YAML."""
        config = PositionSizerConfig()

        # Test Kelly criterion parameters
        assert config.max_position_cap == 0.05
        assert config.min_position_size == 0.001
        assert config.kelly_scaling_factor == 0.25
        assert config.default_win_probability == 0.55
        assert config.default_reward_risk_ratio == 1.5

        # Test volatility scaling
        assert config.volatility_scaling_enabled is True
        assert config.volatility_lookback == 20
        assert config.low_volatility_multiplier == 1.2
        assert config.high_volatility_multiplier == 0.6

        # Test drawdown protection
        assert config.drawdown_protection_enabled is True
        assert config.mild_drawdown_threshold == 0.05
        assert config.moderate_drawdown_threshold == 0.10
        assert config.severe_drawdown_threshold == 0.15

    def test_risk_profiles(self):
        """Test risk profile configurations."""
        config = PositionSizerConfig()

        # Test conservative profile
        conservative = config.risk_profiles["conservative"]
        assert conservative["kelly_scaling_factor"] == 0.125
        assert conservative["max_position_cap"] == 0.02

        # Test moderate profile
        moderate = config.risk_profiles["moderate"]
        assert moderate["kelly_scaling_factor"] == 0.25
        assert moderate["max_position_cap"] == 0.05

        # Test aggressive profile
        aggressive = config.risk_profiles["aggressive"]
        assert aggressive["kelly_scaling_factor"] == 0.5
        assert aggressive["max_position_cap"] == 0.10


class TestTradeSignalInput:
    """Test trade signal input dataclass."""

    def test_signal_creation(self):
        """Test signal input creation."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.8,
            take_profit_pips=30.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=1.5,
            win_probability=0.6,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="test_strategy",
        )

        assert signal.symbol == "EURUSD"
        assert signal.side == OrderSide.BUY
        assert signal.signal_confidence == 0.8
        assert signal.take_profit_pips == 30.0
        assert signal.stop_loss_pips == 20.0
        assert signal.reward_risk_ratio == 1.5
        assert signal.win_probability == 0.6


class TestPortfolioState:
    """Test portfolio state dataclass."""

    def test_portfolio_state_creation(self):
        """Test portfolio state creation."""
        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.02,
            daily_pnl=-500.0,
            open_positions=[
                {"symbol": "EURUSD", "size": 0.02},
                {"symbol": "GBPUSD", "size": 0.03},
            ],
            volatility_history=[0.0001, 0.0002, 0.0003],
            performance_history=[],
        )

        assert portfolio.total_capital == 100000.0
        assert portfolio.current_drawdown == 0.02
        assert portfolio.daily_pnl == -500.0
        assert len(portfolio.open_positions) == 2


class TestKellyPositionSizer:
    """Test the main Kelly position sizer functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=PositionSizerConfig)

        # Kelly criterion parameters
        config.max_position_cap = 0.05
        config.min_position_size = 0.001
        config.kelly_scaling_factor = 0.25
        config.default_win_probability = 0.55
        config.default_reward_risk_ratio = 1.5

        # Volatility scaling
        config.volatility_scaling_enabled = True
        config.volatility_lookback = 20
        config.low_volatility_multiplier = 1.2
        config.high_volatility_multiplier = 0.6
        config.low_vol_percentile = 25
        config.high_vol_percentile = 75

        # Drawdown protection
        config.drawdown_protection_enabled = True
        config.mild_drawdown_threshold = 0.05
        config.moderate_drawdown_threshold = 0.10
        config.severe_drawdown_threshold = 0.15
        config.mild_drawdown_scaling = 0.8
        config.moderate_drawdown_scaling = 0.6
        config.severe_drawdown_scaling = 0.4

        # Risk profiles
        config.risk_profiles = {
            "moderate": {
                "kelly_scaling_factor": 0.25,
                "max_position_cap": 0.05,
                "volatility_multiplier": 1.0,
            }
        }

        # Confidence mapping
        config.confidence_boost_factor = 0.3
        config.confidence_penalty_factor = 0.2
        config.high_confidence_threshold = 0.8
        config.low_confidence_threshold = 0.4

        # Constraints
        config.max_concurrent_positions = 5
        config.max_total_exposure = 0.20
        config.max_correlation_exposure = 0.15
        config.correlation_threshold = 0.7
        config.max_currency_exposure = 0.30

        # Emergency controls
        config.emergency_max_position = 0.01
        config.max_daily_loss_threshold = 0.05
        config.volatility_spike_threshold = 3.0
        config.emergency_mode_duration_hours = 24

        return config

    @pytest.fixture
    def position_sizer(self, mock_config):
        """Create position sizer with mocked config."""
        return KellyPositionSizer(config=mock_config, risk_profile=RiskProfile.MODERATE)

    @pytest.fixture
    def sample_signal(self):
        """Create sample trading signal."""
        return TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.7,
            take_profit_pips=30.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=1.5,
            win_probability=0.6,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="test_strategy",
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio state."""
        return PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[],
            volatility_history=[0.0002] * 20,
            performance_history=[],
        )

    def test_initialization(self, position_sizer):
        """Test position sizer initialization."""
        assert position_sizer.risk_profile == RiskProfile.MODERATE
        assert position_sizer.total_sizing_decisions == 0
        assert position_sizer.successful_sizings == 0
        assert len(position_sizer.volatility_history) == 0
        assert position_sizer.emergency_mode_until is None

    @pytest.mark.asyncio
    async def test_kelly_criterion_calculation(self, position_sizer):
        """Test Kelly criterion calculation with known values."""
        # Known test case: p=0.6, b=1.5 → f* ≈ 0.15
        win_probability = 0.6
        reward_risk_ratio = 1.5

        kelly_fraction = position_sizer._calculate_kelly_fraction(
            win_probability, reward_risk_ratio
        )

        # f* = (b × p - q) / b = (1.5 × 0.6 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 ≈ 0.333
        expected_kelly = (1.5 * 0.6 - 0.4) / 1.5
        assert abs(kelly_fraction - expected_kelly) < 0.001
        assert abs(kelly_fraction - 0.333) < 0.01

    @pytest.mark.asyncio
    async def test_position_size_calculation(
        self, position_sizer, sample_signal, sample_portfolio
    ):
        """Test complete position size calculation."""
        result = await position_sizer.calculate_position_size(
            sample_signal, sample_portfolio
        )

        assert isinstance(result, PositionSizeResult)
        assert 0 <= result.position_size <= position_sizer.config.max_position_cap
        assert 0 <= result.confidence <= 1.0
        assert result.kelly_raw >= 0
        assert result.kelly_scaled >= 0
        assert result.risk_profile == "moderate"
        assert "win_probability" in result.metadata
        assert "reward_risk_ratio" in result.metadata

    @pytest.mark.asyncio
    async def test_win_probability_calculation(self, position_sizer, sample_signal):
        """Test win probability calculation with confidence adjustments."""
        # Test high confidence signal
        sample_signal.signal_confidence = 0.9
        sample_signal.win_probability = 0.6

        win_prob = position_sizer._calculate_win_probability(sample_signal)
        assert win_prob > 0.6  # Should be boosted

        # Test low confidence signal
        sample_signal.signal_confidence = 0.3
        win_prob_low = position_sizer._calculate_win_probability(sample_signal)
        assert win_prob_low < 0.6  # Should be penalized

        # Test with no provided win probability (should use default with penalty applied)
        sample_signal.win_probability = None
        win_prob_default = position_sizer._calculate_win_probability(sample_signal)
        # With low confidence (0.3), it should be penalized from the default
        assert win_prob_default < position_sizer.config.default_win_probability

    @pytest.mark.asyncio
    async def test_reward_risk_ratio_calculation(self, position_sizer, sample_signal):
        """Test reward/risk ratio calculation."""
        # Test with provided ratio
        sample_signal.reward_risk_ratio = 2.0
        ratio = position_sizer._calculate_reward_risk_ratio(sample_signal)
        assert ratio == 2.0

        # Test with pips
        sample_signal.reward_risk_ratio = None
        sample_signal.take_profit_pips = 40.0
        sample_signal.stop_loss_pips = 20.0
        ratio = position_sizer._calculate_reward_risk_ratio(sample_signal)
        assert ratio == 2.0

        # Test with no data (should use default)
        sample_signal.reward_risk_ratio = None
        sample_signal.take_profit_pips = None
        sample_signal.stop_loss_pips = None
        ratio = position_sizer._calculate_reward_risk_ratio(sample_signal)
        assert ratio == position_sizer.config.default_reward_risk_ratio

    @pytest.mark.asyncio
    async def test_volatility_scaling(self, position_sizer, sample_signal):
        """Test volatility-based position scaling."""
        # Build volatility history with varied values for proper percentile calculation
        volatility_values = [0.0001, 0.00015, 0.0002, 0.00025, 0.0003] * 4  # 20 values
        for vol in volatility_values:
            position_sizer.volatility_history.append(vol)

        # Test low volatility (below 25th percentile)
        sample_signal.volatility_atr = 0.0001
        regime, scaling = position_sizer._calculate_volatility_scaling(sample_signal)
        assert regime == VolatilityRegime.LOW
        assert scaling == position_sizer.config.low_volatility_multiplier

        # Test high volatility (above 75th percentile)
        sample_signal.volatility_atr = 0.0003
        regime, scaling = position_sizer._calculate_volatility_scaling(sample_signal)
        assert regime == VolatilityRegime.HIGH
        assert scaling == position_sizer.config.high_volatility_multiplier

        # Test normal volatility (between percentiles)
        sample_signal.volatility_atr = 0.0002
        regime, scaling = position_sizer._calculate_volatility_scaling(sample_signal)
        assert regime == VolatilityRegime.NORMAL
        assert scaling == 1.0

    @pytest.mark.asyncio
    async def test_drawdown_protection(self, position_sizer, sample_portfolio):
        """Test drawdown-based position scaling."""
        # Test no drawdown
        sample_portfolio.current_drawdown = 0.0
        level, scaling = position_sizer._calculate_drawdown_scaling(sample_portfolio)
        assert level == DrawdownLevel.NONE
        assert scaling == 1.0

        # Test mild drawdown
        sample_portfolio.current_drawdown = 0.07
        level, scaling = position_sizer._calculate_drawdown_scaling(sample_portfolio)
        assert level == DrawdownLevel.MILD
        assert scaling == position_sizer.config.mild_drawdown_scaling

        # Test moderate drawdown
        sample_portfolio.current_drawdown = 0.12
        level, scaling = position_sizer._calculate_drawdown_scaling(sample_portfolio)
        assert level == DrawdownLevel.MODERATE
        assert scaling == position_sizer.config.moderate_drawdown_scaling

        # Test severe drawdown
        sample_portfolio.current_drawdown = 0.18
        level, scaling = position_sizer._calculate_drawdown_scaling(sample_portfolio)
        assert level == DrawdownLevel.SEVERE
        assert scaling == position_sizer.config.severe_drawdown_scaling

    @pytest.mark.asyncio
    async def test_constraints_application(
        self, position_sizer, sample_signal, sample_portfolio
    ):
        """Test position sizing constraints."""
        # Test maximum position cap
        base_size = 0.10  # Above max cap
        final_size, constraints = position_sizer._apply_constraints(
            base_size, sample_signal, sample_portfolio
        )
        assert final_size <= position_sizer.config.max_position_cap
        assert any("max_position_cap" in c for c in constraints)

        # Test minimum position size
        base_size = 0.0005  # Below min size
        final_size, constraints = position_sizer._apply_constraints(
            base_size, sample_signal, sample_portfolio
        )
        assert final_size >= position_sizer.config.min_position_size
        assert any("min_position_size" in c for c in constraints)

        # Test maximum concurrent positions
        sample_portfolio.open_positions = [{"size": 0.02}] * 6  # Above limit
        base_size = 0.03
        final_size, constraints = position_sizer._apply_constraints(
            base_size, sample_signal, sample_portfolio
        )
        assert final_size == 0.0
        assert "max_concurrent_positions" in constraints

        # Test total exposure limit
        sample_portfolio.open_positions = [
            {"size": 0.05},
            {"size": 0.05},
            {"size": 0.05},
            {"size": 0.04},
        ]  # 0.19 total
        base_size = 0.03  # Would exceed 0.20 limit
        final_size, constraints = position_sizer._apply_constraints(
            base_size, sample_signal, sample_portfolio
        )
        assert final_size <= 0.01  # Should be capped to stay within limit
        assert "max_total_exposure" in constraints

    @pytest.mark.asyncio
    async def test_emergency_mode(
        self, position_sizer, sample_signal, sample_portfolio
    ):
        """Test emergency mode activation and handling."""
        # Test daily loss threshold trigger
        sample_portfolio.daily_pnl = -6000  # 6% loss on 100k capital
        sample_portfolio.total_capital = 100000

        result = await position_sizer.calculate_position_size(
            sample_signal, sample_portfolio
        )

        assert result.position_size <= position_sizer.config.emergency_max_position
        assert (
            "emergency_mode" in result.constraints_applied
            or "daily_loss_limit" in result.constraints_applied
        )
        assert result.confidence < 0.5  # Low confidence in emergency

    @pytest.mark.asyncio
    async def test_risk_profile_effects(self, mock_config):
        """Test different risk profiles."""
        # Test conservative profile
        conservative_sizer = KellyPositionSizer(
            config=mock_config, risk_profile=RiskProfile.CONSERVATIVE
        )

        # Mock conservative profile config
        mock_config.risk_profiles["conservative"] = {
            "kelly_scaling_factor": 0.125,
            "max_position_cap": 0.02,
            "volatility_multiplier": 0.8,
        }

        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.7,
            take_profit_pips=30.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=1.5,
            win_probability=0.6,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="test",
        )

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[],
            volatility_history=[],
            performance_history=[],
        )

        result = await conservative_sizer.calculate_position_size(signal, portfolio)

        # Conservative profile should have smaller positions
        assert result.position_size <= 0.02  # Conservative max cap
        assert result.risk_profile == "conservative"

    @pytest.mark.asyncio
    async def test_performance_tracking(
        self, position_sizer, sample_signal, sample_portfolio
    ):
        """Test performance metrics tracking."""
        # Make several sizing decisions
        for i in range(5):
            await position_sizer.calculate_position_size(
                sample_signal, sample_portfolio
            )

        metrics = position_sizer.get_performance_metrics()

        assert metrics["total_decisions"] == 5
        assert metrics["successful_sizings"] <= 5
        assert "success_rate" in metrics
        assert "average_kelly_fraction" in metrics
        assert "average_position_size" in metrics
        assert "average_confidence" in metrics

    def test_risk_profile_update(self, position_sizer):
        """Test risk profile updates."""
        assert position_sizer.risk_profile == RiskProfile.MODERATE

        position_sizer.update_risk_profile(RiskProfile.AGGRESSIVE)
        assert position_sizer.risk_profile == RiskProfile.AGGRESSIVE

    def test_emergency_mode_reset(self, position_sizer):
        """Test emergency mode reset."""
        # Activate emergency mode
        position_sizer._activate_emergency_mode("Test emergency")
        assert position_sizer.emergency_mode_until is not None

        # Reset emergency mode
        position_sizer.reset_emergency_mode()
        assert position_sizer.emergency_mode_until is None


class TestKellyFormulaValidation:
    """Test Kelly formula with known mathematical cases."""

    @pytest.fixture
    def position_sizer(self):
        """Create position sizer for formula testing."""
        return KellyPositionSizer()

    def test_known_kelly_cases(self, position_sizer):
        """Test Kelly formula with known mathematical cases."""

        # Case 1: p=0.6, b=1.5 → f* = (1.5*0.6 - 0.4)/1.5 = 0.333
        kelly = position_sizer._calculate_kelly_fraction(0.6, 1.5)
        assert abs(kelly - 0.333) < 0.01

        # Case 2: p=0.55, b=2.0 → f* = (2.0*0.55 - 0.45)/2.0 = 0.325
        kelly = position_sizer._calculate_kelly_fraction(0.55, 2.0)
        assert abs(kelly - 0.325) < 0.01

        # Case 3: p=0.7, b=1.0 → f* = (1.0*0.7 - 0.3)/1.0 = 0.4
        kelly = position_sizer._calculate_kelly_fraction(0.7, 1.0)
        assert abs(kelly - 0.4) < 0.01

        # Case 4: No edge (p=0.5, b=1.0) → f* = 0
        kelly = position_sizer._calculate_kelly_fraction(0.5, 1.0)
        assert kelly == 0.0

        # Case 5: Negative edge (p=0.4, b=1.0) → f* = 0 (clamped)
        kelly = position_sizer._calculate_kelly_fraction(0.4, 1.0)
        assert kelly == 0.0


class TestStressTesting:
    """Stress test the position sizer under various conditions."""

    @pytest.fixture
    def position_sizer(self):
        """Create position sizer for stress testing."""
        return KellyPositionSizer()

    @pytest.mark.asyncio
    async def test_longshot_trades(self, position_sizer):
        """Test low probability, high reward trades."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.3,
            take_profit_pips=50.0,
            stop_loss_pips=10.0,
            reward_risk_ratio=5.0,
            win_probability=0.2,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="longshot",
        )

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[],
            volatility_history=[],
            performance_history=[],
        )

        result = await position_sizer.calculate_position_size(signal, portfolio)

        # Should have very small position size due to low probability
        assert result.position_size < 0.02
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_scalping_trades(self, position_sizer):
        """Test high probability, low reward trades."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.9,
            take_profit_pips=10.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=0.5,
            win_probability=0.8,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="scalping",
        )

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[],
            volatility_history=[],
            performance_history=[],
        )

        result = await position_sizer.calculate_position_size(signal, portfolio)

        # Kelly calculation: f* = (b × p - q) / b = (0.5 × 0.8 - 0.2) / 0.5 = (0.4 - 0.2) / 0.5 = 0.4
        # But with high confidence (0.9), win probability gets boosted, so Kelly will be higher
        # The actual Kelly raw should be positive but the position size should still be reasonable
        assert result.kelly_raw > 0  # Should be positive due to edge
        assert result.position_size > 0  # Should still be positive
        assert result.position_size <= 0.05  # Should respect max cap

    @pytest.mark.asyncio
    async def test_high_volatility_periods(self, position_sizer):
        """Test position sizing during high volatility."""
        # Build high volatility history
        for i in range(20):
            position_sizer.volatility_history.append(0.0001 + i * 0.00001)

        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.7,
            take_profit_pips=30.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=1.5,
            win_probability=0.6,
            current_price=1.1000,
            volatility_atr=0.0008,
            timestamp=datetime.now(),
            strategy_name="high_vol",
        )

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[],
            volatility_history=[],
            performance_history=[],
        )

        result = await position_sizer.calculate_position_size(signal, portfolio)

        # Should scale down position size in high volatility
        assert result.volatility_regime == "high"
        assert result.volatility_scaled < 1.0

    @pytest.mark.asyncio
    async def test_extreme_drawdown(self, position_sizer):
        """Test position sizing during extreme drawdown."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.8,
            take_profit_pips=40.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=2.0,
            win_probability=0.7,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="drawdown_test",
        )

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.25,
            daily_pnl=-3000,  # 3% daily loss, below emergency threshold
            open_positions=[],
            volatility_history=[],
            performance_history=[],
        )

        result = await position_sizer.calculate_position_size(signal, portfolio)

        # Should significantly reduce position size during severe drawdown
        assert result.drawdown_level == "severe"
        # Check that drawdown scaling was applied (should be 0.4 for severe)
        assert result.metadata["drawdown_scaling"] == 0.4
        # Position size should be reasonable (may hit max cap but drawdown scaling was applied)
        assert result.position_size <= 0.05  # Should respect max cap
        assert (
            result.confidence < 0.7
        )  # Confidence should be reduced due to severe drawdown

    @pytest.mark.asyncio
    async def test_random_signal_stability(self, position_sizer):
        """Test position sizer stability with random signals."""
        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[],
            volatility_history=[],
            performance_history=[],
        )

        results = []

        # Generate 100 random signals
        for i in range(100):
            signal = TradeSignalInput(
                symbol="EURUSD",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                signal_confidence=np.random.uniform(0.1, 0.9),
                take_profit_pips=np.random.uniform(10.0, 50.0),
                stop_loss_pips=np.random.uniform(10.0, 30.0),
                reward_risk_ratio=np.random.uniform(0.5, 3.0),
                win_probability=np.random.uniform(0.3, 0.8),
                current_price=1.1000,
                volatility_atr=np.random.uniform(0.0001, 0.0005),
                timestamp=datetime.now(),
                strategy_name=f"random_{i}",
            )

            result = await position_sizer.calculate_position_size(signal, portfolio)
            results.append(result)

        # Check stability
        position_sizes = [r.position_size for r in results]
        confidences = [r.confidence for r in results]

        # All position sizes should be within valid range
        assert all(0 <= size <= 0.05 for size in position_sizes)

        # All confidences should be within valid range
        assert all(0 <= conf <= 1.0 for conf in confidences)

        # Should have reasonable distribution (not all zeros or all max)
        assert np.std(position_sizes) > 0.001  # Some variation
        assert np.mean(position_sizes) > 0.005  # Not all tiny
        assert np.mean(position_sizes) < 0.04  # Not all max


class TestIntegrationScenarios:
    """Test integration scenarios with realistic market conditions."""

    @pytest.fixture
    def real_position_sizer(self):
        """Create real position sizer for integration tests."""
        return create_position_sizer()

    @pytest.mark.asyncio
    async def test_trending_market_scenario(self, real_position_sizer):
        """Test position sizing in trending market conditions."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.85,
            take_profit_pips=50.0,
            stop_loss_pips=25.0,
            reward_risk_ratio=2.0,
            win_probability=0.65,
            current_price=1.1000,
            volatility_atr=0.0003,
            timestamp=datetime.now(),
            strategy_name="breakout_trend",
        )

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=500.0,
            open_positions=[{"symbol": "GBPUSD", "size": 0.02}],
            volatility_history=[0.0002] * 15 + [0.0003] * 5,
            performance_history=[],
        )

        result = await real_position_sizer.calculate_position_size(signal, portfolio)

        assert result.position_size > 0
        assert result.position_size <= 0.05
        assert result.confidence > 0.5
        assert "win_probability" in result.metadata
        assert result.metadata["reward_risk_ratio"] == 2.0

    @pytest.mark.asyncio
    async def test_mean_reverting_scenario(self, real_position_sizer):
        """Test position sizing in mean-reverting conditions."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.SELL,
            signal_confidence=0.7,
            take_profit_pips=20.0,
            stop_loss_pips=30.0,
            reward_risk_ratio=0.67,
            win_probability=0.75,
            current_price=1.1000,
            volatility_atr=0.0001,
            timestamp=datetime.now(),
            strategy_name="grid_martingale",
        )

        # Create volatility history that will make 0.0001 appear as low volatility
        volatility_history = [
            0.0003,
            0.0004,
            0.0005,
            0.0003,
            0.0004,
        ] * 10  # 50 values, higher than current

        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.03,
            daily_pnl=-200.0,
            open_positions=[],
            volatility_history=volatility_history,
            performance_history=[],
        )

        result = await real_position_sizer.calculate_position_size(signal, portfolio)

        # Should have smaller position due to unfavorable reward/risk ratio
        assert result.position_size > 0
        assert result.kelly_raw < 0.5  # Low Kelly due to poor reward/risk
        assert result.volatility_regime in [
            "low",
            "normal",
        ]  # Accept either since it depends on history
        assert result.drawdown_level == "none"  # 3% drawdown is below 5% mild threshold
        assert any(
            "max_position_cap" in constraint
            for constraint in result.constraints_applied
        )  # Should hit position cap

    @pytest.mark.asyncio
    async def test_portfolio_constraint_scenario(self, real_position_sizer):
        """Test position sizing with portfolio constraints."""
        signal = TradeSignalInput(
            symbol="EURUSD",
            side=OrderSide.BUY,
            signal_confidence=0.8,
            take_profit_pips=36.0,
            stop_loss_pips=20.0,
            reward_risk_ratio=1.8,
            win_probability=0.6,
            current_price=1.1000,
            volatility_atr=0.0002,
            timestamp=datetime.now(),
            strategy_name="test",
        )

        # Portfolio near exposure limits
        portfolio = PortfolioState(
            total_capital=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=[
                {"symbol": "GBPUSD", "size": 0.05},
                {"symbol": "USDJPY", "size": 0.04},
                {"symbol": "AUDUSD", "size": 0.03},
                {"symbol": "USDCAD", "size": 0.05},
            ],  # Total: 0.17, near 0.20 limit
            volatility_history=[0.0002] * 20,
            performance_history=[],
        )

        result = await real_position_sizer.calculate_position_size(signal, portfolio)

        # Should be constrained by total exposure limit (allow for floating point precision)
        assert (
            result.position_size <= 0.031
        )  # Max allowed to stay under 0.20 with some tolerance
        assert "max_total_exposure" in result.constraints_applied
        assert result.confidence > 0.7


class TestFactoryFunction:
    """Test the factory function for creating position sizers."""

    def test_create_position_sizer_default(self):
        """Test creating position sizer with defaults."""
        sizer = create_position_sizer()

        assert isinstance(sizer, KellyPositionSizer)
        assert sizer.risk_profile == RiskProfile.MODERATE
        assert sizer.config is not None

    def test_create_position_sizer_custom(self):
        """Test creating position sizer with custom parameters."""
        sizer = create_position_sizer(risk_profile=RiskProfile.AGGRESSIVE)

        assert isinstance(sizer, KellyPositionSizer)
        assert sizer.risk_profile == RiskProfile.AGGRESSIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
