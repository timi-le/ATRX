"""
Dual Provider Demo

Demonstrates using both OANDA and Dukascopy connectors together
with failover capabilities and data consistency validation.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog
from data import OandaConnector, OandaConfig, DukascopyConnector, DukascopyConfig
from data.stream_feed import DataStreamManager
from core.pubsub import MarketDataStreamer
from core.interfaces.messaging_interfaces import Message, Topics
from core.interfaces.data_interfaces import MarketData


# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class DualProviderManager:
    """
    Manages both OANDA and Dukascopy connectors with failover capabilities.
    """
    
    def __init__(self, 
                 oanda_config: Optional[OandaConfig] = None,
                 dukascopy_config: Optional[DukascopyConfig] = None,
                 publisher_address: str = "tcp://*:5556"):
        self.oanda_config = oanda_config
        self.dukascopy_config = dukascopy_config
        self.publisher_address = publisher_address
        
        self.stream_manager = DataStreamManager(publisher_address)
        self.oanda_connector: Optional[OandaConnector] = None
        self.dukascopy_connector: Optional[DukascopyConnector] = None
        
        self.price_cache: Dict[str, MarketData] = {}
        self.provider_status = {
            "oanda": False,
            "dukascopy": False
        }
        
        self.logger = logger
    
    async def setup_connectors(self):
        """Set up both connectors."""
        try:
            # Setup OANDA connector (live streaming)
            if self.oanda_config:
                self.oanda_connector = OandaConnector(self.oanda_config, self.logger)
                self.stream_manager.add_connector("oanda", OandaConnector, self.oanda_config)
                self.logger.info("OANDA connector configured")
            
            # Setup Dukascopy connector (historical/backup)
            if self.dukascopy_config:
                self.dukascopy_connector = DukascopyConnector(self.dukascopy_config, self.logger)
                self.stream_manager.add_connector("dukascopy", DukascopyConnector, self.dukascopy_config)
                self.logger.info("Dukascopy connector configured")
                
        except Exception as e:
            self.logger.error("Failed to setup connectors", error=str(e))
            raise
    
    async def start(self):
        """Start the dual provider system."""
        self.logger.info("Starting dual provider system")
        
        try:
            await self.stream_manager.start()
            
            # Check connector status
            status = self.stream_manager.get_status()
            for name, connector_info in status["connectors"].items():
                self.provider_status[name] = connector_info["connected"]
                
            self.logger.info("Dual provider system started", 
                           provider_status=self.provider_status)
            
        except Exception as e:
            self.logger.error("Failed to start dual provider system", error=str(e))
            raise
    
    async def stop(self):
        """Stop the dual provider system."""
        self.logger.info("Stopping dual provider system")
        await self.stream_manager.stop()
    
    async def get_historical_comparison(self, symbol: str, hours: int = 1) -> Dict:
        """
        Compare historical data from both providers for consistency validation.
        """
        self.logger.info("Fetching historical data for comparison", 
                        symbol=symbol, hours=hours)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        results = {
            "symbol": symbol,
            "timeframe": f"{hours}h",
            "oanda_data": None,
            "dukascopy_data": None,
            "comparison": None
        }
        
        try:
            # Get OANDA historical data
            if self.oanda_connector:
                oanda_data = await self.oanda_connector.get_historical_data(
                    symbol=symbol,
                    timeframe="1h",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )
                results["oanda_data"] = {
                    "count": len(oanda_data),
                    "first_candle": oanda_data[0].__dict__ if oanda_data else None,
                    "last_candle": oanda_data[-1].__dict__ if oanda_data else None
                }
                self.logger.info("OANDA historical data retrieved", count=len(oanda_data))
            
            # Get Dukascopy historical data
            if self.dukascopy_connector:
                dukascopy_data = await self.dukascopy_connector.get_historical_data(
                    symbol=symbol,
                    timeframe="1h",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )
                results["dukascopy_data"] = {
                    "count": len(dukascopy_data),
                    "first_candle": dukascopy_data[0].__dict__ if dukascopy_data else None,
                    "last_candle": dukascopy_data[-1].__dict__ if dukascopy_data else None
                }
                self.logger.info("Dukascopy historical data retrieved", count=len(dukascopy_data))
            
            # Compare data if both available
            if results["oanda_data"] and results["dukascopy_data"]:
                results["comparison"] = self._compare_historical_data(
                    results["oanda_data"], 
                    results["dukascopy_data"]
                )
            
        except Exception as e:
            self.logger.error("Failed to get historical comparison", error=str(e))
            results["error"] = str(e)
        
        return results
    
    def _compare_historical_data(self, oanda_data: Dict, dukascopy_data: Dict) -> Dict:
        """Compare historical data from both providers."""
        comparison = {
            "count_match": oanda_data["count"] == dukascopy_data["count"],
            "oanda_count": oanda_data["count"],
            "dukascopy_count": dukascopy_data["count"],
            "price_correlation": None
        }
        
        # Basic comparison - in production you'd do more sophisticated analysis
        if oanda_data["last_candle"] and dukascopy_data["last_candle"]:
            oanda_close = oanda_data["last_candle"]["close"]
            dukascopy_close = dukascopy_data["last_candle"]["close"]
            
            price_diff = abs(oanda_close - dukascopy_close)
            price_diff_pct = (price_diff / oanda_close) * 100
            
            comparison["price_difference"] = price_diff
            comparison["price_difference_pct"] = price_diff_pct
            comparison["prices_similar"] = price_diff_pct < 0.1  # Within 0.1%
        
        return comparison
    
    async def test_failover(self, symbol: str = "EUR/USD"):
        """Test failover capabilities between providers."""
        self.logger.info("Testing failover capabilities", symbol=symbol)
        
        # Test OANDA first
        if self.oanda_connector:
            try:
                await self.oanda_connector.connect()
                self.logger.info("OANDA connection test: SUCCESS")
                
                # Test streaming for a few seconds
                count = 0
                async for market_data in self.oanda_connector.stream():
                    self.logger.info("OANDA stream data", 
                                   symbol=market_data.symbol,
                                   bid=market_data.bid,
                                   ask=market_data.ask)
                    count += 1
                    if count >= 3:  # Get 3 ticks
                        break
                
                await self.oanda_connector.disconnect()
                
            except Exception as e:
                self.logger.error("OANDA connection test: FAILED", error=str(e))
        
        # Test Dukascopy as backup
        if self.dukascopy_connector:
            try:
                await self.dukascopy_connector.connect()
                self.logger.info("Dukascopy connection test: SUCCESS")
                
                # Test historical data retrieval
                historical_data = await self.dukascopy_connector.get_historical_data(
                    symbol=symbol,
                    timeframe="1h",
                    count=5
                )
                
                self.logger.info("Dukascopy historical data test", 
                               count=len(historical_data))
                
                await self.dukascopy_connector.disconnect()
                
            except Exception as e:
                self.logger.error("Dukascopy connection test: FAILED", error=str(e))


async def run_demo():
    """Run the dual provider demonstration."""
    logger.info("Starting Dual Provider Demo")
    
    # Note: These are demo configurations - replace with real credentials
    oanda_config = OandaConfig(
        api_key="demo-api-key",  # Replace with real API key
        account_id="demo-account",  # Replace with real account ID
        environment="practice",  # Use "practice" for demo
        symbols=["EUR/USD", "GBP/USD"],
        timeout=30
    )
    
    dukascopy_config = DukascopyConfig(
        symbols=["EUR/USD", "GBP/USD"],
        timeout=30
    )
    
    # Create dual provider manager
    manager = DualProviderManager(
        oanda_config=oanda_config,
        dukascopy_config=dukascopy_config,
        publisher_address="tcp://*:5556"
    )
    
    try:
        # Setup connectors
        await manager.setup_connectors()
        
        # Test individual connections
        await manager.test_failover("EUR/USD")
        
        # Compare historical data
        comparison = await manager.get_historical_comparison("EUR/USD", hours=2)
        
        print("\n" + "="*80)
        print("HISTORICAL DATA COMPARISON")
        print("="*80)
        print(f"Symbol: {comparison['symbol']}")
        print(f"Timeframe: {comparison['timeframe']}")
        
        if comparison.get("oanda_data"):
            print(f"OANDA candles: {comparison['oanda_data']['count']}")
        
        if comparison.get("dukascopy_data"):
            print(f"Dukascopy candles: {comparison['dukascopy_data']['count']}")
        
        if comparison.get("comparison"):
            comp = comparison["comparison"]
            print(f"Count match: {comp['count_match']}")
            if comp.get("prices_similar"):
                print(f"Price similarity: {comp['prices_similar']} "
                      f"(diff: {comp.get('price_difference_pct', 0):.4f}%)")
        
        print("="*80)
        
        # Start streaming (commented out for demo - requires real credentials)
        # await manager.start()
        # await asyncio.sleep(10)  # Stream for 10 seconds
        # await manager.stop()
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"\nDemo failed: {e}")
        print("\nNote: This demo requires valid OANDA API credentials.")
        print("For Dukascopy, it will attempt to download real historical data.")
        
    finally:
        await manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1) 