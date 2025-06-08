"""
Data Stream Demo

Demonstrates the data ingestion system by starting a mock data stream
and displaying real-time market data.
"""

import asyncio
import sys
from datetime import datetime

import structlog

from core.interfaces.messaging_interfaces import Message, Topics
from core.pubsub import ZMQSubscriber
from data.stream_feed import create_mock_data_stream

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MarketDataDisplay:
    """Display market data in real-time."""

    def __init__(self):
        self.message_count = 0
        self.last_prices = {}
        self.start_time = datetime.now()

    async def handle_market_data(self, message: Message):
        """Handle incoming market data messages."""
        try:
            if message.topic == Topics.MARKET_DATA_TICKS:
                data = message.data
                symbol = data["symbol"]
                bid = data["bid"]
                ask = data["ask"]
                volume = data["volume"]

                # Track price changes
                if symbol in self.last_prices:
                    last_bid = self.last_prices[symbol]["bid"]
                    direction = (
                        "↑" if bid > last_bid else "↓" if bid < last_bid else "→"
                    )
                else:
                    direction = "→"

                self.last_prices[symbol] = {"bid": bid, "ask": ask}
                self.message_count += 1

                # Display update
                spread = ask - bid
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                print(
                    f"[{timestamp}] {symbol:>8} | "
                    f"Bid: {bid:.5f} | Ask: {ask:.5f} | "
                    f"Spread: {spread:.5f} | Vol: {volume:>8,} | {direction}"
                )

                # Show summary every 50 messages
                if self.message_count % 50 == 0:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    rate = self.message_count / elapsed if elapsed > 0 else 0
                    print(
                        f"\n--- Summary: {self.message_count} messages, "
                        f"{rate:.1f} msg/sec ---\n"
                    )

        except Exception as e:
            logger.error("Error handling market data", error=str(e))


async def run_demo():
    """Run the data stream demonstration."""
    logger.info("Starting FX Data Stream Demo")

    # Create display handler
    display = MarketDataDisplay()

    # Start mock data stream
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
    logger.info("Starting mock data stream", symbols=symbols)

    stream_manager = await create_mock_data_stream(
        symbols=symbols, publisher_address="tcp://*:5555"
    )

    # Create subscriber to display data
    subscriber = ZMQSubscriber(
        connect_address="tcp://localhost:5555", topics=[Topics.MARKET_DATA_TICKS]
    )

    await subscriber.start()
    logger.info("Subscriber started, receiving market data...")

    print("\n" + "=" * 80)
    print("FX MARKET DATA STREAM - Live Feed")
    print("=" * 80)
    print(
        f"{'Time':>12} | {'Symbol':>8} | {'Bid':>10} | {'Ask':>10} | "
        f"{'Spread':>8} | {'Volume':>10} | Trend"
    )
    print("-" * 80)

    try:
        # Start receiving messages
        receive_task = asyncio.create_task(
            subscriber.receive_messages(display.handle_market_data)
        )

        # Let it run for demo purposes (30 seconds)
        await asyncio.sleep(30)

        # Cancel the receive task
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        print("\n" + "=" * 80)
        print("Shutting down...")

        await subscriber.stop()
        await stream_manager.stop()

        # Show final statistics
        elapsed = (datetime.now() - display.start_time).total_seconds()
        rate = display.message_count / elapsed if elapsed > 0 else 0

        print(f"\nFinal Statistics:")
        print(f"  Total Messages: {display.message_count}")
        print(f"  Runtime: {elapsed:.1f} seconds")
        print(f"  Average Rate: {rate:.1f} messages/sec")
        print(f"  Symbols Tracked: {len(display.last_prices)}")

        if display.last_prices:
            print(f"\nFinal Prices:")
            for symbol, prices in display.last_prices.items():
                print(f"  {symbol}: Bid {prices['bid']:.5f}, Ask {prices['ask']:.5f}")

        logger.info("Demo completed")


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)
