"""
Messaging interface definitions for the FX AI-Quant Trading System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, field
import asyncio


@dataclass
class Message:
    """Standardized message structure."""
    topic: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None
    source: Optional[str] = None


class MessageBus(ABC):
    """Abstract base class for message bus implementation."""

    @abstractmethod
    async def start(self) -> None:
        """Start the message bus."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the message bus."""
        pass

    @abstractmethod
    async def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic."""
        pass

    @abstractmethod
    async def subscribe(self, topic: str) -> AsyncIterator[Message]:
        """Subscribe to a topic and receive messages."""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if message bus is connected."""
        pass


class Publisher(ABC):
    """Abstract base class for message publishers."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the messaging system."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the messaging system."""
        pass

    @abstractmethod
    async def publish(
        self, topic: str, data: Any, routing_key: Optional[str] = None
    ) -> None:
        """Publish data to a topic."""
        pass

    @abstractmethod
    async def publish_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Publish multiple messages at once."""
        pass


class Subscriber(ABC):
    """Abstract base class for message subscribers."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the messaging system."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the messaging system."""
        pass

    @abstractmethod
    async def subscribe(
        self, topic: str, callback: Optional[Callable[[Message], None]] = None
    ) -> AsyncIterator[Message]:
        """Subscribe to a topic."""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        pass

    @abstractmethod
    async def subscribe_pattern(
        self, pattern: str, callback: Optional[Callable[[Message], None]] = None
    ) -> AsyncIterator[Message]:
        """Subscribe to topics matching a pattern."""
        pass


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle_message(self, message: Message) -> None:
        """Process an incoming message."""
        pass

    @abstractmethod
    async def handle_error(self, error: Exception, message: Message) -> None:
        """Handle message processing errors."""
        pass

    @abstractmethod
    def get_supported_topics(self) -> List[str]:
        """Get list of topics this handler supports."""
        pass


class ZeroMQPublisher(Publisher):
    """ZeroMQ-specific publisher interface."""

    @abstractmethod
    async def bind(self, address: str) -> None:
        """Bind to a ZeroMQ address."""
        pass

    @abstractmethod
    async def connect_to(self, address: str) -> None:
        """Connect to a ZeroMQ address."""
        pass


class ZeroMQSubscriber(Subscriber):
    """ZeroMQ-specific subscriber interface."""

    @abstractmethod
    async def bind(self, address: str) -> None:
        """Bind to a ZeroMQ address."""
        pass

    @abstractmethod
    async def connect_to(self, address: str) -> None:
        """Connect to a ZeroMQ address."""
        pass

    @abstractmethod
    async def set_subscription_filter(self, filter_prefix: str) -> None:
        """Set message filter for subscriptions."""
        pass


class RedisPublisher(Publisher):
    """Redis-specific publisher interface."""

    @abstractmethod
    async def set_connection_params(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ) -> None:
        """Set Redis connection parameters."""
        pass

    @abstractmethod
    async def publish_stream(
        self, stream_name: str, data: Dict[str, Any], max_length: Optional[int] = None
    ) -> str:
        """Publish to Redis Stream."""
        pass


class RedisSubscriber(Subscriber):
    """Redis-specific subscriber interface."""

    @abstractmethod
    async def set_connection_params(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ) -> None:
        """Set Redis connection parameters."""
        pass

    @abstractmethod
    async def subscribe_stream(
        self,
        stream_name: str,
        consumer_group: str,
        consumer_name: str,
        start_id: str = ">",
    ) -> AsyncIterator[Message]:
        """Subscribe to Redis Stream."""
        pass

    @abstractmethod
    async def acknowledge_message(
        self, stream_name: str, consumer_group: str, message_id: str
    ) -> None:
        """Acknowledge message processing."""
        pass


# Topic constants for the system
class Topics:
    """Standard topics used throughout the system."""

    # Market data topics
    MARKET_DATA_TICKS = "market.data.ticks"
    MARKET_DATA_BARS = "market.data.bars"
    MARKET_DATA_L2 = "market.data.level2"
    
    # Provider-specific market data topics
    MARKET_DATA_OANDA = "market.data.oanda"
    MARKET_DATA_DUKASCOPY = "market.data.dukascopy" 
    MARKET_DATA_MOCK = "market.data.mock"

    # Feature topics
    FEATURES_TECHNICAL = "features.technical"
    FEATURES_VOLATILITY = "features.volatility"
    FEATURES_MOMENTUM = "features.momentum"
    FEATURES_CARRY = "features.carry"
    FEATURES_MACRO = "features.macro"

    # ML topics
    ML_PREDICTIONS = "ml.predictions"
    ML_TRAINING = "ml.training"
    REGIME_DETECTION = "regime.detection"

    # Trading topics
    SIGNALS = "trading.signals"
    ORDERS = "trading.orders"
    POSITIONS = "trading.positions"
    EXECUTIONS = "trading.executions"

    # Risk topics
    RISK_ALERTS = "risk.alerts"
    RISK_LIMITS = "risk.limits"

    # System topics
    SYSTEM_HEALTH = "system.health"
    SYSTEM_METRICS = "system.metrics"
    SYSTEM_ALERTS = "system.alerts"
    
    # Provider health topics
    PROVIDER_HEALTH_OANDA = "provider.health.oanda"
    PROVIDER_HEALTH_DUKASCOPY = "provider.health.dukascopy"
    
    @classmethod
    def get_provider_data_topic(cls, provider: str) -> str:
        """Get the market data topic for a specific provider."""
        provider_topics = {
            "oanda": cls.MARKET_DATA_OANDA,
            "dukascopy": cls.MARKET_DATA_DUKASCOPY,
            "mock": cls.MARKET_DATA_MOCK
        }
        return provider_topics.get(provider.lower(), cls.MARKET_DATA_TICKS)
    
    @classmethod
    def get_provider_health_topic(cls, provider: str) -> str:
        """Get the health topic for a specific provider."""
        provider_health_topics = {
            "oanda": cls.PROVIDER_HEALTH_OANDA,
            "dukascopy": cls.PROVIDER_HEALTH_DUKASCOPY
        }
        return provider_health_topics.get(provider.lower(), cls.SYSTEM_HEALTH)


class ZeroMQMessageBus(MessageBus):
    """ZeroMQ implementation of MessageBus."""
    
    def __init__(self, bind_address: str = "tcp://*:5555"):
        self.bind_address = bind_address
        self._connected = False
    
    async def start(self) -> None:
        """Start the ZeroMQ message bus."""
        self._connected = True
    
    async def stop(self) -> None:
        """Stop the ZeroMQ message bus."""
        self._connected = False
    
    async def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic via ZeroMQ."""
        pass  # Implementation would use pyzmq
    
    async def subscribe(self, topic: str) -> AsyncIterator[Message]:
        """Subscribe to a topic via ZeroMQ."""
        while self._connected:
            yield Message(topic=topic, data={})
            await asyncio.sleep(0.1)
    
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        pass
    
    def is_connected(self) -> bool:
        """Check if ZeroMQ message bus is connected."""
        return self._connected


class RedisMessageBus(MessageBus):
    """Redis implementation of MessageBus."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._connected = False
    
    async def start(self) -> None:
        """Start the Redis message bus."""
        self._connected = True
    
    async def stop(self) -> None:
        """Stop the Redis message bus."""
        self._connected = False
    
    async def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic via Redis."""
        pass  # Implementation would use aioredis
    
    async def subscribe(self, topic: str) -> AsyncIterator[Message]:
        """Subscribe to a topic via Redis."""
        while self._connected:
            yield Message(topic=topic, data={})
            await asyncio.sleep(0.1)
    
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        pass
    
    def is_connected(self) -> bool:
        """Check if Redis message bus is connected."""
        return self._connected
