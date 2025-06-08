"""
Data Stream Manager

Main entry point for data ingestion. Manages multiple data connectors
and coordinates streaming to ZeroMQ publishers.
"""

import asyncio
import time
from typing import Any

import structlog

from core.interfaces.messaging_interfaces import Topics
from core.pubsub import MarketDataStreamer
from data.base_connector import BaseFXConnector, ConnectionConfig
from data.mock_provider import MockFXConnector


class DataStreamManager:
    """
    Main coordinator for data ingestion and streaming.

    Manages multiple data connectors and handles streaming to message bus.
    """

    def __init__(
        self,
        publisher_address: str = "tcp://*:5555",
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.publisher_address = publisher_address
        self.logger = logger or structlog.get_logger(__name__)

        self.connectors: dict[str, BaseFXConnector] = {}
        self.streamer = MarketDataStreamer(publisher_address, logger)
        self.streaming_tasks: dict[str, asyncio.Task] = {}

        self.is_running = False
        self._shutdown_event = asyncio.Event()

        # Statistics
        self.stats = {
            "start_time": None,
            "messages_processed": 0,
            "errors": 0,
            "reconnections": 0,
        }

    def add_connector(
        self,
        name: str,
        connector_class: type[BaseFXConnector],
        config: ConnectionConfig,
    ) -> None:
        """Add a new data connector."""
        if name in self.connectors:
            raise ValueError(f"Connector '{name}' already exists")

        connector = connector_class(config, self.logger)
        self.connectors[name] = connector

        self.logger.info(
            "Added connector",
            name=name,
            connector_type=connector_class.__name__,
            symbols=config.symbols,
        )

    async def start(self) -> None:
        """Start the data stream manager."""
        try:
            self.logger.info("Starting data stream manager")

            # Start the message publisher
            await self.streamer.start_publisher()

            # Connect all connectors
            for name, connector in self.connectors.items():
                success = await connector.connect()
                if success:
                    # Start streaming task
                    task = asyncio.create_task(
                        self._stream_from_connector(name, connector)
                    )
                    self.streaming_tasks[name] = task

            self.is_running = True
            self.stats["start_time"] = time.time()

            self.logger.info(
                "Data stream manager started",
                active_connectors=len(self.streaming_tasks),
            )

        except Exception as e:
            self.logger.error("Failed to start data stream manager", error=str(e))
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the data stream manager and all connectors."""
        self.logger.info("Stopping data stream manager")
        self.is_running = False

        # Cancel all streaming tasks
        for task in self.streaming_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.streaming_tasks:
            await asyncio.gather(*self.streaming_tasks.values(), return_exceptions=True)

        self.streaming_tasks.clear()

        # Disconnect all connectors
        for connector in self.connectors.values():
            if connector.is_connected:
                await connector.disconnect()

        # Stop the message publisher
        await self.streamer.stop_publisher()

        self._shutdown_event.set()

        uptime = (
            time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        )
        self.logger.info(
            "Data stream manager stopped",
            uptime=uptime,
            messages_processed=self.stats["messages_processed"],
        )

    async def _stream_from_connector(
        self, name: str, connector: BaseFXConnector
    ) -> None:
        """Stream data from a specific connector."""
        self.logger.info("Starting stream from connector", name=name)

        try:
            async for market_data in connector.stream():
                if not self.is_running:
                    break

                try:
                    # Publish to message bus
                    await self.streamer.publish_market_data(
                        market_data, Topics.MARKET_DATA_TICKS
                    )

                    self.stats["messages_processed"] += 1

                    # Log progress periodically
                    if self.stats["messages_processed"] % 100 == 0:
                        self.logger.debug(
                            "Messages processed",
                            count=self.stats["messages_processed"],
                            connector=name,
                        )

                except Exception as e:
                    self.logger.error(
                        "Failed to publish market data", error=str(e), connector=name
                    )
                    self.stats["errors"] += 1

        except asyncio.CancelledError:
            self.logger.info("Stream cancelled", connector=name)
            raise
        except Exception as e:
            self.logger.error("Stream error", connector=name, error=str(e))
            self.stats["errors"] += 1

    def get_status(self) -> dict[str, Any]:
        """Get current status of the stream manager."""
        connector_status = {}
        for name, connector in self.connectors.items():
            connector_status[name] = {
                "connected": connector.is_connected,
                "status": connector.status.value,
                "uptime": connector.uptime,
                "retry_count": connector.retry_count,
            }

        uptime = (
            time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        )

        return {
            "is_running": self.is_running,
            "uptime": uptime,
            "connectors": connector_status,
            "streaming_tasks": len(self.streaming_tasks),
            "statistics": self.stats,
        }


async def create_mock_data_stream(
    symbols: list[str] = None, publisher_address: str = "tcp://*:5555"
) -> DataStreamManager:
    """
    Create a data stream manager with mock data provider.

    Args:
        symbols: List of symbols to stream (default: major FX pairs)
        publisher_address: ZeroMQ publisher address

    Returns:
        Configured and started DataStreamManager
    """
    if symbols is None:
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]

    manager = DataStreamManager(publisher_address)

    # Add mock connector
    config = ConnectionConfig(symbols=symbols)
    manager.add_connector("mock", MockFXConnector, config)

    await manager.start()
    return manager
