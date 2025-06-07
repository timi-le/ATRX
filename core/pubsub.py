"""
ZeroMQ Publisher/Subscriber Wrapper

Provides high-level interface for streaming market data using ZeroMQ.
Handles connection management, message serialization, and error recovery.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import asdict
from datetime import datetime

import zmq
import zmq.asyncio
import structlog
import ujson
from core.interfaces.data_interfaces import MarketData, OHLCV
from core.interfaces.messaging_interfaces import Message, Topics


def serialize_message_data(obj: Any) -> Any:
    """
    Custom serialization function to handle datetime objects and other non-JSON types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_message_data(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_message_data(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle dataclass or other objects with __dict__
        return serialize_message_data(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
    else:
        return obj


class ZMQPublisher:
    """
    ZeroMQ Publisher for streaming market data.
    
    Publishes market data to multiple subscribers with topic-based routing.
    """
    
    def __init__(self, 
                 bind_address: str = "tcp://*:5555",
                 logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.bind_address = bind_address
        self.logger = logger or structlog.get_logger(__name__)
        
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.is_running = False
        self._message_count = 0
        self._start_time: Optional[float] = None
        
    async def start(self) -> None:
        """Start the publisher."""
        try:
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(self.bind_address)
            self.is_running = True
            self._start_time = time.time()
            self.logger.info("ZMQ Publisher started", address=self.bind_address)
            
            # Give subscribers time to connect
            await asyncio.sleep(0.1)
            
        except Exception as e:
            self.logger.error("Failed to start ZMQ Publisher", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the publisher and cleanup resources."""
        self.is_running = False
        
        if self.socket:
            self.socket.close()
            
        self.context.term()
        self.logger.info("ZMQ Publisher stopped", 
                        messages_sent=self._message_count,
                        uptime=time.time() - self._start_time if self._start_time else 0)
    
    async def publish_market_data(self, data: MarketData, topic: str = Topics.MARKET_DATA_TICKS) -> None:
        """
        Publish market data to subscribers.
        
        Args:
            data: MarketData object to publish
            topic: Topic for message routing
        """
        if not self.is_running or not self.socket:
            raise RuntimeError("Publisher not started")
        
        try:
            message = Message(
                topic=topic,
                data=serialize_message_data(data),
                timestamp=data.timestamp,
                message_id=str(uuid.uuid4()),
                source="data_ingestion"
            )
            
            # Serialize message
            message_bytes = ujson.dumps(serialize_message_data(message)).encode('utf-8')
            
            # Send with topic prefix
            topic_bytes = topic.encode('utf-8')
            await self.socket.send_multipart([topic_bytes, message_bytes])
            
            self._message_count += 1
            
            if self._message_count % 1000 == 0:
                self.logger.debug("Published messages", count=self._message_count)
                
        except Exception as e:
            self.logger.error("Failed to publish market data", 
                            error=str(e), 
                            symbol=data.symbol)
            raise
    
    async def publish_ohlcv_data(self, data: OHLCV, topic: str = Topics.MARKET_DATA_BARS) -> None:
        """
        Publish OHLCV data to subscribers.
        
        Args:
            data: OHLCV object to publish
            topic: Topic for message routing
        """
        if not self.is_running or not self.socket:
            raise RuntimeError("Publisher not started")
        
        try:
            message = Message(
                topic=topic,
                data=serialize_message_data(data),
                timestamp=data.timestamp,
                message_id=str(uuid.uuid4()),
                source="data_ingestion"
            )
            
            # Serialize message
            message_bytes = ujson.dumps(serialize_message_data(message)).encode('utf-8')
            
            # Send with topic prefix
            topic_bytes = topic.encode('utf-8')
            await self.socket.send_multipart([topic_bytes, message_bytes])
            
            self._message_count += 1
            
        except Exception as e:
            self.logger.error("Failed to publish OHLCV data", 
                            error=str(e), 
                            symbol=data.symbol)
            raise
    
    def get_stats(self) -> Dict[str, Union[int, float, bool]]:
        """Get publisher statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "is_running": self.is_running,
            "messages_sent": self._message_count,
            "uptime": uptime,
            "messages_per_second": self._message_count / uptime if uptime > 0 else 0,
            "bind_address": self.bind_address,
        }


class ZMQSubscriber:
    """
    ZeroMQ Subscriber for receiving market data.
    
    Subscribes to market data streams with topic filtering.
    """
    
    def __init__(self, 
                 connect_address: str = "tcp://localhost:5555",
                 topics: List[str] = None,
                 logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.connect_address = connect_address
        self.topics = topics or [Topics.MARKET_DATA_TICKS, Topics.MARKET_DATA_BARS]
        self.logger = logger or structlog.get_logger(__name__)
        
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.is_running = False
        self._message_count = 0
        self._start_time: Optional[float] = None
        
    async def start(self) -> None:
        """Start the subscriber."""
        try:
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(self.connect_address)
            
            # Subscribe to specified topics
            for topic in self.topics:
                self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
            
            self.is_running = True
            self._start_time = time.time()
            self.logger.info("ZMQ Subscriber started", 
                           address=self.connect_address,
                           topics=self.topics)
            
        except Exception as e:
            self.logger.error("Failed to start ZMQ Subscriber", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the subscriber and cleanup resources."""
        self.is_running = False
        
        if self.socket:
            self.socket.close()
            
        self.context.term()
        self.logger.info("ZMQ Subscriber stopped", 
                        messages_received=self._message_count,
                        uptime=time.time() - self._start_time if self._start_time else 0)
    
    async def receive_messages(self, 
                             message_handler: Optional[Callable[[Message], None]] = None) -> None:
        """
        Receive and process messages from publishers.
        
        Args:
            message_handler: Optional callback function to handle received messages
        """
        if not self.is_running or not self.socket:
            raise RuntimeError("Subscriber not started")
        
        while self.is_running:
            try:
                # Receive message with topic
                topic_bytes, message_bytes = await self.socket.recv_multipart()
                topic = topic_bytes.decode('utf-8')
                
                # Deserialize message
                message_data = ujson.loads(message_bytes.decode('utf-8'))
                message = Message(**message_data)
                
                self._message_count += 1
                
                if message_handler:
                    try:
                        await asyncio.create_task(message_handler(message))
                    except Exception as e:
                        self.logger.error("Message handler failed", 
                                        error=str(e), 
                                        topic=topic)
                
                if self._message_count % 1000 == 0:
                    self.logger.debug("Received messages", count=self._message_count)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Failed to receive message", error=str(e))
                await asyncio.sleep(0.1)  # Brief pause before retry
    
    def get_stats(self) -> Dict[str, Union[int, float, bool]]:
        """Get subscriber statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "is_running": self.is_running,
            "messages_received": self._message_count,
            "uptime": uptime,
            "messages_per_second": self._message_count / uptime if uptime > 0 else 0,
            "connect_address": self.connect_address,
            "topics": self.topics,
        }


class MarketDataStreamer:
    """
    High-level market data streaming coordinator.
    
    Manages publisher/subscriber lifecycle and provides unified interface.
    """
    
    def __init__(self, 
                 publisher_address: str = "tcp://*:5555",
                 logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.publisher_address = publisher_address
        self.logger = logger or structlog.get_logger(__name__)
        
        self.publisher = ZMQPublisher(publisher_address, logger)
        self.subscribers: List[ZMQSubscriber] = []
        self.is_running = False
        
    async def start_publisher(self) -> None:
        """Start the market data publisher."""
        await self.publisher.start()
        self.is_running = True
        self.logger.info("Market data streamer started")
    
    async def stop_publisher(self) -> None:
        """Stop the publisher."""
        await self.publisher.stop()
        self.is_running = False
        self.logger.info("Market data streamer stopped")
    
    async def create_subscriber(self, 
                              topics: List[str] = None,
                              message_handler: Optional[Callable[[Message], None]] = None) -> ZMQSubscriber:
        """
        Create and start a new subscriber.
        
        Args:
            topics: List of topics to subscribe to
            message_handler: Optional message handler function
            
        Returns:
            Configured and started ZMQSubscriber
        """
        # Convert publisher bind address to connect address
        connect_address = self.publisher_address.replace("*", "localhost")
        
        subscriber = ZMQSubscriber(connect_address, topics, self.logger)
        await subscriber.start()
        
        if message_handler:
            # Start message receiving loop
            asyncio.create_task(subscriber.receive_messages(message_handler))
        
        self.subscribers.append(subscriber)
        return subscriber
    
    async def stop_all_subscribers(self) -> None:
        """Stop all active subscribers."""
        for subscriber in self.subscribers:
            await subscriber.stop()
        self.subscribers.clear()
    
    async def publish_market_data(self, data: MarketData, topic: str = Topics.MARKET_DATA_TICKS) -> None:
        """Publish market data through the publisher."""
        await self.publisher.publish_market_data(data, topic)
    
    async def publish_ohlcv_data(self, data: OHLCV, topic: str = Topics.MARKET_DATA_BARS) -> None:
        """Publish OHLCV data through the publisher."""
        await self.publisher.publish_ohlcv_data(data, topic)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get statistics for publisher and all subscribers."""
        return {
            "publisher": self.publisher.get_stats(),
            "subscribers": [sub.get_stats() for sub in self.subscribers],
            "total_subscribers": len(self.subscribers),
            "is_running": self.is_running,
        } 