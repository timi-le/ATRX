"""
Base FX Connector Interface

Defines the standard interface that all FX data providers must implement.
Provides connection management, streaming, and error handling capabilities.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Union

import structlog
from core.interfaces.data_interfaces import MarketData, OHLCV


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """Configuration for FX connector connections."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 1.0
    heartbeat_interval: int = 30
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]


class BaseFXConnector(ABC):
    """
    Abstract base class for all FX data connectors.
    
    Provides standard interface for connecting to data providers,
    streaming market data, and handling connection issues.
    """
    
    def __init__(self, config: ConnectionConfig, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        self.status = ConnectionStatus.DISCONNECTED
        self.connection_start_time: Optional[float] = None
        self.last_heartbeat: Optional[float] = None
        self.retry_count = 0
        self._should_stop = False
        self._connection_lock = asyncio.Lock()
        
    @property
    def is_connected(self) -> bool:
        """Check if the connector is currently connected."""
        return self.status == ConnectionStatus.CONNECTED
    
    @property
    def uptime(self) -> Optional[float]:
        """Get connection uptime in seconds."""
        if self.connection_start_time and self.is_connected:
            return time.time() - self.connection_start_time
        return None
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data provider.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the data provider."""
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """
        Stream market data from the provider.
        
        Yields:
            MarketData: Real-time market data objects
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        count: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Retrieve historical OHLCV data.
        
        Args:
            symbol: Trading pair symbol (e.g., "EUR/USD")
            timeframe: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time (ISO format)
            end_time: End time (ISO format)  
            count: Number of bars to retrieve
            
        Returns:
            List of OHLCV data
        """
        pass
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the data provider with exponential backoff.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        async with self._connection_lock:
            if self.status == ConnectionStatus.RECONNECTING:
                return False
                
            self.status = ConnectionStatus.RECONNECTING
            self.logger.info("Attempting to reconnect", 
                           retry_count=self.retry_count,
                           max_retries=self.config.max_retries)
            
            while self.retry_count < self.config.max_retries and not self._should_stop:
                try:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2 ** self.retry_count)
                    await asyncio.sleep(delay)
                    
                    if await self.connect():
                        self.retry_count = 0
                        self.logger.info("Reconnection successful")
                        return True
                        
                except Exception as e:
                    self.logger.error("Reconnection attempt failed", 
                                    error=str(e), 
                                    retry_count=self.retry_count)
                
                self.retry_count += 1
            
            self.status = ConnectionStatus.ERROR
            self.logger.error("All reconnection attempts failed")
            return False
    
    async def start_heartbeat(self) -> None:
        """Start heartbeat monitoring."""
        while self.is_connected and not self._should_stop:
            await asyncio.sleep(self.config.heartbeat_interval)
            try:
                await self._send_heartbeat()
                self.last_heartbeat = time.time()
            except Exception as e:
                self.logger.error("Heartbeat failed", error=str(e))
                await self.reconnect()
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat to provider. Override in subclasses if needed."""
        pass
    
    def stop(self) -> None:
        """Signal the connector to stop operations."""
        self._should_stop = True
        self.logger.info("Connector stop signal sent")
    
    async def health_check(self) -> Dict[str, Union[str, float, bool]]:
        """
        Get connector health status.
        
        Returns:
            Dict containing health metrics
        """
        return {
            "status": self.status.value,
            "is_connected": self.is_connected,
            "uptime": self.uptime,
            "retry_count": self.retry_count,
            "last_heartbeat": self.last_heartbeat,
            "symbols": self.config.symbols,
        }


def retry_on_connection_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying operations that might fail due to connection issues.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    break
                except Exception as e:
                    # Don't retry for non-connection errors
                    raise e
            
            raise last_exception
        return wrapper
    return decorator 