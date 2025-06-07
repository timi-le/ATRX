"""
Core interface definitions for the FX AI-Quant Trading System.

This module defines the abstract base classes and protocols that all
components must implement to ensure proper integration.
"""

# Data interfaces
from .data_interfaces import (
    MarketData,
    OHLCV,
    DataProvider,
    DataPublisher,
    DataSubscriber,
    MarketDataFeed,
    HistoricalDataProvider,
)

# Trading interfaces  
from .trading_interfaces import (
    Order,
    Position,
    Signal,
    Strategy,
    PositionSizer,
    RiskManager,
    ExecutionEngine,
    OrderManager,
    OrderType,
    OrderStatus,
    PositionType,
    SignalType,
)

# ML interfaces
from .ml_interfaces import (
    Features,
    Prediction,
    RegimeLabel,
    FeatureEngineer,
    MLPredictor,
    ModelTrainer,
    RegimeDetector,
    EnsemblePredictor,
)

# Messaging interfaces
from .messaging_interfaces import (
    Message,
    MessageBus,
    Publisher,
    Subscriber,
    ZeroMQMessageBus,
    RedisMessageBus,
    Topics,
)

# Macro economic interfaces
from .macro_interfaces import (
    EconomicEvent,
    NewsEvent,
    MacroFeatureVector,
    MacroDataProvider,
    NewsProvider,
    SentimentAnalyzer,
    MacroEngine,
    ImpactLevel,
    Currency,
    EventType,
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
