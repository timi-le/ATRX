"""
Unit tests for core interfaces in the FX AI-Quant Trading System.

These tests verify the behavior of our abstract base classes and data structures.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from core.interfaces.data_interfaces import MarketData, OHLCV
from core.interfaces.ml_interfaces import Features, Prediction, RegimeLabel
from core.interfaces.trading_interfaces import (
    Order,
    Position,
    Signal,
    OrderType,
    OrderSide,
    OrderStatus,
)
from core.interfaces.messaging_interfaces import Message, Topics


class TestMarketDataStructures:
    """Test cases for market data structures."""

    def test_market_data_creation(self):
        """Test MarketData object creation and properties."""
        timestamp = datetime.now()
        market_data = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=1.0850,
            ask=1.0852,
            volume=1000000,
            source="test_provider",
        )

        assert market_data.symbol == "EURUSD"
        assert market_data.timestamp == timestamp
        assert market_data.bid == 1.0850
        assert market_data.ask == 1.0852
        assert market_data.volume == 1000000
        assert market_data.source == "test_provider"

        # Calculated properties
        assert market_data.mid == 1.0851  # (1.0850 + 1.0852) / 2
        assert (
            abs(market_data.spread - 0.0002) < 1e-10
        )  # Handle floating point precision

    def test_ohlcv_creation(self):
        """Test OHLCV object creation and properties."""
        timestamp = datetime.now()
        ohlcv = OHLCV(
            symbol="GBPUSD",
            timestamp=timestamp,
            open=1.2500,
            high=1.2520,
            low=1.2490,
            close=1.2510,
            volume=500000,
            timeframe="5m",
        )

        assert ohlcv.symbol == "GBPUSD"
        assert ohlcv.timestamp == timestamp
        assert ohlcv.open == 1.2500
        assert ohlcv.high == 1.2520
        assert ohlcv.low == 1.2490
        assert ohlcv.close == 1.2510
        assert ohlcv.volume == 500000
        assert ohlcv.timeframe == "5m"


class TestMLInterfaces:
    """Test cases for machine learning interfaces."""

    def test_features_creation(self):
        """Test Features object creation."""
        features = Features(
            symbol="EURUSD",
            timestamp=datetime.now(),
            features={"rsi": 0.6, "ma_20": 1.0851, "volatility": 0.015},
            feature_names=["rsi", "ma_20", "volatility"],
        )

        assert features.symbol == "EURUSD"
        assert isinstance(features.features, dict)
        assert features.features["rsi"] == 0.6
        assert "rsi" in features.feature_names

    def test_prediction_creation(self):
        """Test Prediction object creation."""
        prediction = Prediction(
            symbol="EURUSD",
            timestamp=datetime.now(),
            prediction=0.0025,
            confidence=0.85,
            model_name="lstm_v1",
            horizon=60,
        )

        assert prediction.symbol == "EURUSD"
        assert prediction.horizon == 60
        assert prediction.prediction == 0.0025
        assert prediction.confidence == 0.85

    def test_regime_label_creation(self):
        """Test RegimeLabel object creation."""
        regime = RegimeLabel(
            timestamp=datetime.now(),
            regime="trending",
            confidence=0.92,
            features={"momentum": 0.0012, "volatility": 0.008},
        )

        assert regime.regime == "trending"
        assert regime.confidence == 0.92
        assert regime.features["momentum"] == 0.0012


class TestTradingInterfaces:
    """Test cases for trading interfaces."""

    def test_order_creation(self):
        """Test Order object creation."""
        order = Order(
            order_id="ORD_12345",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100000,
            price=None,
        )

        assert order.symbol == "EURUSD"
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 100000
        assert order.status == OrderStatus.PENDING

    def test_position_creation(self):
        """Test Position object creation."""
        position = Position(
            symbol="GBPUSD",
            quantity=50000,
            avg_price=1.2500,
            unrealized_pnl=100.0,
            realized_pnl=50.0,
        )

        assert position.symbol == "GBPUSD"
        assert position.quantity == 50000
        assert position.avg_price == 1.2500
        assert position.total_pnl == 150.0  # unrealized + realized

    def test_signal_creation(self):
        """Test Signal object creation."""
        signal = Signal(
            symbol="USDJPY",
            side=OrderSide.BUY,
            strength=0.75,
            confidence=0.80,
            strategy_name="breakout_v1",
            timestamp=datetime.now(),
            features={"momentum": 0.0015},
        )

        assert signal.symbol == "USDJPY"
        assert signal.side == OrderSide.BUY
        assert signal.strength == 0.75
        assert signal.confidence == 0.80


class TestMessagingInterfaces:
    """Test cases for messaging interfaces."""

    def test_message_creation(self):
        """Test Message object creation."""
        message = Message(
            topic=Topics.MARKET_DATA_TICKS,
            data={"symbol": "EURUSD", "price": 1.0851},
            timestamp=datetime.now(),
            message_id="msg_123",
            source="data_provider",
        )

        assert message.topic == Topics.MARKET_DATA_TICKS
        assert isinstance(message.data, dict)
        assert message.source == "data_provider"
        assert message.message_id == "msg_123"

    def test_topics_constants(self):
        """Test Topics constants are properly defined."""
        assert hasattr(Topics, "MARKET_DATA_TICKS")
        assert hasattr(Topics, "SIGNALS")
        assert hasattr(Topics, "ORDERS")
        assert hasattr(Topics, "POSITIONS")
        assert hasattr(Topics, "REGIME_DETECTION")


class TestEnumValues:
    """Test cases for enum values."""

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"


# Integration-style tests for interface compliance
@pytest.mark.integration
class TestInterfaceCompliance:
    """Test that our concrete classes will properly implement interfaces."""

    def test_data_provider_interface_methods(self):
        """Test that DataProvider interface has required methods."""
        from core.interfaces.data_interfaces import DataProvider

        # Check that abstract methods are defined
        abstract_methods = DataProvider.__abstractmethods__
        expected_methods = {
            "connect",
            "disconnect",
            "subscribe_ticks",
            "subscribe_bars",
            "get_historical_data",
        }

        assert expected_methods.issubset(abstract_methods)

    def test_strategy_interface_methods(self):
        """Test that Strategy interface has required methods."""
        from core.interfaces.trading_interfaces import Strategy

        abstract_methods = Strategy.__abstractmethods__
        expected_methods = {
            "generate_signal",
            "update_parameters",
            "get_parameters",
            "get_name",
        }

        assert expected_methods.issubset(abstract_methods)

    def test_ml_predictor_interface_methods(self):
        """Test that MLPredictor interface has required methods."""
        from core.interfaces.ml_interfaces import MLPredictor

        abstract_methods = MLPredictor.__abstractmethods__
        expected_methods = {
            "predict",
            "predict_batch",
            "get_feature_importance",
            "get_model_info",
        }

        assert expected_methods.issubset(abstract_methods)
