"""
Core interface definitions for the FX AI-Quant Trading System.

This module defines the abstract base classes and protocols that all
components must implement to ensure proper integration.
"""

# Data interfaces
from .data_interfaces import (
    OHLCV,
    DataProvider,
    DataPublisher,
    DataSubscriber,
    HistoricalDataProvider,
    MarketData,
    MarketDataFeed,
)

# Macro economic interfaces
from .macro_interfaces import (
    Currency,
    EconomicEvent,
    EventType,
    ImpactLevel,
    MacroDataProvider,
    MacroEngine,
    MacroFeatureVector,
    NewsEvent,
    NewsProvider,
    SentimentAnalyzer,
)

# Messaging interfaces
from .messaging_interfaces import (
    Message,
    MessageBus,
    Publisher,
    RedisMessageBus,
    Subscriber,
    Topics,
    ZeroMQMessageBus,
)

# ML interfaces
from .ml_interfaces import (
    EnsemblePredictor,
    FeatureEngineer,
    Features,
    MLPredictor,
    ModelTrainer,
    Prediction,
    RegimeDetector,
    RegimeLabel,
)

# Trading interfaces
from .trading_interfaces import (
    ExecutionEngine,
    Order,
    OrderManager,
    OrderStatus,
    OrderType,
    Position,
    PositionSizer,
    PositionType,
    RiskManager,
    Signal,
    SignalType,
    Strategy,
)

__all__ = [
    # Data interfaces
    "MarketData",
    "OHLCV",
    "DataProvider",
    "DataPublisher",
    "DataSubscriber",
    "MarketDataFeed",
    "HistoricalDataProvider",
    # Trading interfaces
    "Order",
    "Position",
    "Signal",
    "Strategy",
    "PositionSizer",
    "RiskManager",
    "ExecutionEngine",
    "OrderManager",
    "OrderType",
    "OrderStatus",
    "PositionType",
    "SignalType",
    # ML interfaces
    "Features",
    "Prediction",
    "RegimeLabel",
    "FeatureEngineer",
    "MLPredictor",
    "ModelTrainer",
    "RegimeDetector",
    "EnsemblePredictor",
    # Messaging interfaces
    "Message",
    "MessageBus",
    "Publisher",
    "Subscriber",
    "ZeroMQMessageBus",
    "RedisMessageBus",
    "Topics",
    # Macro economic interfaces
    "EconomicEvent",
    "NewsEvent",
    "MacroFeatureVector",
    "MacroDataProvider",
    "NewsProvider",
    "SentimentAnalyzer",
    "MacroEngine",
    "ImpactLevel",
    "Currency",
    "EventType",
]
