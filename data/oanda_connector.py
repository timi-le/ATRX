"""
OANDA FX Connector

Production-ready connector for OANDA v20 API providing live FX data streaming
and historical data retrieval capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime

from collections.abc import AsyncGenerator

import aiohttp
import structlog

from core.interfaces.data_interfaces import OHLCV, MarketData
from data.base_connector import (
    BaseFXConnector,
    ConnectionConfig,
    ConnectionStatus,
    retry_on_connection_error,
)


@dataclass
class OandaConfig(ConnectionConfig):
    """Extended configuration for OANDA connector."""

    account_id: str | None = None
    environment: str = "practice"  # "practice" or "live"
    stream_timeout: int = 60
    rest_timeout: int = 30

    def __post_init__(self):
        super().__post_init__()
        if self.environment == "practice":
            self.base_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"


class OandaConnector(BaseFXConnector):
    """
    OANDA v20 API connector for live FX data streaming.

    Provides real-time price streaming and historical data retrieval
    using the OANDA v20 REST and streaming APIs.
    """

    def __init__(
        self, config: OandaConfig, logger: structlog.stdlib.BoundLogger | None = None
    ):
        super().__init__(config, logger)
        self.oanda_config = config
        self.session: aiohttp.ClientSession | None = None
        self.stream_session: aiohttp.ClientSession | None = None
        self._stream_response: aiohttp.ClientResponse | None = None
        self._instruments_map = {}

        # OANDA requires specific instrument format (e.g., EUR_USD instead of EUR/USD)
        self._setup_instrument_mapping()

    def _setup_instrument_mapping(self):
        """Set up mapping between standard FX notation and OANDA format."""
        for symbol in self.config.symbols:
            # Convert EUR/USD to EUR_USD format
            oanda_symbol = symbol.replace("/", "_")
            self._instruments_map[symbol] = oanda_symbol

    @property
    def headers(self) -> dict[str, str]:
        """Get authentication headers for OANDA API."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def connect(self) -> bool:
        """Establish connection to OANDA API."""
        self.status = ConnectionStatus.CONNECTING
        self.logger.info(
            "Connecting to OANDA",
            environment=self.oanda_config.environment,
            account_id=self.oanda_config.account_id,
        )

        try:
            # Create HTTP sessions
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)

            # Verify connection by getting account info
            await self._verify_connection()

            # Set up streaming session
            stream_timeout = aiohttp.ClientTimeout(
                total=self.oanda_config.stream_timeout
            )
            self.stream_session = aiohttp.ClientSession(
                headers=self.headers, timeout=stream_timeout
            )

            self.status = ConnectionStatus.CONNECTED
            self.connection_start_time = time.time()
            self.logger.info(
                "Connected to OANDA successfully",
                instruments=list(self._instruments_map.values()),
            )
            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error("Failed to connect to OANDA", error=str(e))
            await self._cleanup_sessions()
            return False

    async def _verify_connection(self) -> None:
        """Verify API connection by retrieving account information."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        url = f"{self.oanda_config.base_url}/v3/accounts/{self.oanda_config.account_id}"

        async with self.session.get(url) as response:
            if response.status != 200:
                error_data = await response.text()
                raise RuntimeError(f"OANDA API error {response.status}: {error_data}")

            data = await response.json()
            account_info = data.get("account", {})

            self.logger.info(
                "OANDA account verified",
                currency=account_info.get("currency"),
                balance=account_info.get("balance"),
                instruments_count=len(account_info.get("instruments", [])),
            )

    async def disconnect(self) -> None:
        """Gracefully disconnect from OANDA API."""
        self.logger.info("Disconnecting from OANDA")

        # Stop streaming
        if self._stream_response:
            self._stream_response.close()
            self._stream_response = None

        # Cleanup sessions
        await self._cleanup_sessions()

        self.status = ConnectionStatus.DISCONNECTED
        self.connection_start_time = None
        self.logger.info("Disconnected from OANDA")

    async def _cleanup_sessions(self) -> None:
        """Clean up HTTP sessions."""
        if self.stream_session:
            await self.stream_session.close()
            self.stream_session = None

        if self.session:
            await self.session.close()
            self.session = None

    @retry_on_connection_error(max_retries=3, delay=1.0)
    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """
        Stream real-time market data from OANDA.

        Yields:
            MarketData: Real-time market data objects
        """
        if not self.is_connected or not self.stream_session:
            raise RuntimeError("Connector not connected")

        # Build instruments parameter for OANDA API
        instruments = ",".join(self._instruments_map.values())

        url = f"{self.oanda_config.stream_url}/v3/accounts/{self.oanda_config.account_id}/pricing/stream"
        params = {
            "instruments": instruments,
            "snapshot": "true",  # Include initial snapshot
        }

        self.logger.info(
            "Starting OANDA price stream", instruments=instruments, url=url
        )

        try:
            async with self.stream_session.get(url, params=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise RuntimeError(
                        f"OANDA stream error {response.status}: {error_data}"
                    )

                self._stream_response = response

                async for line in response.content:
                    if self._should_stop:
                        break

                    try:
                        # Parse JSON line
                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue

                        data = json.loads(line_str)

                        # Process different message types
                        if data.get("type") == "PRICE":
                            market_data = self._parse_price_data(data)
                            if market_data:
                                yield market_data

                        elif data.get("type") == "HEARTBEAT":
                            self.last_heartbeat = time.time()
                            self.logger.debug("OANDA heartbeat received")

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            "Failed to parse OANDA stream data",
                            error=str(e),
                            data=line_str[:100],
                        )
                        continue
                    except Exception as e:
                        self.logger.error(
                            "Error processing OANDA stream data", error=str(e)
                        )
                        continue

        except asyncio.CancelledError:
            self.logger.info("OANDA stream cancelled")
            raise
        except Exception as e:
            self.logger.error("OANDA stream error", error=str(e))
            raise

    def _parse_price_data(self, data: dict) -> MarketData | None:
        """Parse OANDA price data into MarketData format."""
        try:
            instrument = data.get("instrument", "")
            # Convert OANDA format back to standard (EUR_USD -> EUR/USD)
            symbol = instrument.replace("_", "/")

            if symbol not in self.config.symbols:
                return None

            # Parse timestamp
            timestamp_str = data.get("time", "")
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # Get bid/ask prices
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                return None

            # Use the first (best) bid/ask prices
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])

            # Volume is optional in OANDA streaming
            volume = None
            if bids[0].get("liquidity"):
                volume = float(bids[0]["liquidity"])

            return MarketData(
                symbol=symbol,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                volume=volume,
                source="oanda",
            )

        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(
                "Failed to parse OANDA price data", error=str(e), data=data
            )
            return None

    @retry_on_connection_error(max_retries=3, delay=1.0)
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: str | None = None,
        end_time: str | None = None,
        count: int | None = None,
    ) -> list[OHLCV]:
        """
        Retrieve historical OHLCV data from OANDA.

        Args:
            symbol: Trading pair symbol (e.g., "EUR/USD")
            timeframe: Time interval (e.g., "M1", "M5", "H1", "D")
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            count: Number of candles to retrieve

        Returns:
            List of OHLCV data
        """
        if not self.is_connected or not self.session:
            raise RuntimeError("Connector not connected")

        # Convert symbol to OANDA format
        instrument = self._instruments_map.get(symbol)
        if not instrument:
            raise ValueError(f"Unsupported symbol: {symbol}")

        # Convert timeframe to OANDA format
        granularity = self._convert_timeframe(timeframe)

        url = f"{self.oanda_config.base_url}/v3/instruments/{instrument}/candles"

        # Build parameters
        params = {"granularity": granularity, "price": "MBA"}  # Mid, Bid, Ask prices

        if count:
            params["count"] = min(count, 5000)  # OANDA limit
        else:
            params["count"] = 500  # Default

        if start_time:
            params["from"] = start_time
        if end_time:
            params["to"] = end_time

        self.logger.info(
            "Fetching OANDA historical data",
            symbol=symbol,
            instrument=instrument,
            granularity=granularity,
            params=params,
        )

        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise RuntimeError(
                        f"OANDA API error {response.status}: {error_data}"
                    )

                data = await response.json()
                candles = data.get("candles", [])

                ohlcv_data = []
                for candle in candles:
                    if candle.get("complete"):  # Only include completed candles
                        ohlcv = self._parse_candle_data(candle, symbol, timeframe)
                        if ohlcv:
                            ohlcv_data.append(ohlcv)

                self.logger.info(
                    "Retrieved OANDA historical data",
                    symbol=symbol,
                    candles_count=len(ohlcv_data),
                )

                return ohlcv_data

        except Exception as e:
            self.logger.error(
                "Failed to get OANDA historical data", symbol=symbol, error=str(e)
            )
            raise

    def _parse_candle_data(
        self, candle: dict, symbol: str, timeframe: str
    ) -> OHLCV | None:
        """Parse OANDA candle data into OHLCV format."""
        try:
            # Parse timestamp
            timestamp_str = candle.get("time", "")
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # Use mid prices
            mid = candle.get("mid", {})

            return OHLCV(
                symbol=symbol,
                timestamp=timestamp,
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=float(candle.get("volume", 0)),
                timeframe=timeframe,
            )

        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(
                "Failed to parse OANDA candle data", error=str(e), data=candle
            )
            return None

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to OANDA granularity format."""
        timeframe_map = {
            "1m": "M1",
            "5m": "M5",
            "15m": "M15",
            "30m": "M30",
            "1h": "H1",
            "4h": "H4",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }

        # Handle variations
        if timeframe.upper() in ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "M"]:
            return timeframe.upper()

        return timeframe_map.get(timeframe.lower(), "M1")

    async def _send_heartbeat(self) -> None:
        """OANDA handles heartbeats automatically in the stream."""

    def get_supported_symbols(self) -> list[str]:
        """Get list of symbols supported by this connector."""
        return [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "AUD/USD",
            "USD/CHF",
            "NZD/USD",
            "USD/CAD",
            "EUR/GBP",
            "EUR/JPY",
            "GBP/JPY",
            "EUR/CHF",
            "GBP/CHF",
            "AUD/JPY",
            "CHF/JPY",
            "EUR/AUD",
            "GBP/AUD",
            "AUD/CHF",
            "AUD/CAD",
            "EUR/CAD",
            "GBP/CAD",
            "CAD/CHF",
            "NZD/JPY",
            "EUR/NZD",
            "GBP/NZD",
            "AUD/NZD",
            "CAD/JPY",
            "NZD/CHF",
            "NZD/CAD",
        ]
