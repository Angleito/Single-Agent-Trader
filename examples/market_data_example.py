#!/usr/bin/env python3
"""
Market Data Integration Usage Example

This example demonstrates how to use the MarketDataProvider and MarketDataClient
classes for real-time and historical market data from Coinbase.

Features demonstrated:
- Historical data fetching
- Real-time price updates via WebSocket
- Order book data retrieval
- Data caching and validation
- Error handling and reconnection
"""

import asyncio
import logging
from datetime import datetime, timedelta

from bot.data.market import (
    MarketDataProvider,
    create_market_data_client,
)
from bot.types import MarketData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example of basic market data usage."""
    print("=== Basic Market Data Usage ===")

    # Create a market data client using context manager (recommended)
    async with create_market_data_client("BTC-USD", "1m") as client:

        # Get current price
        current_price = await client.get_current_price()
        if current_price:
            print(f"Current BTC-USD price: ${current_price:,}")

        # Get historical data as DataFrame
        historical_df = await client.get_historical_data(lookback_hours=6)
        if not historical_df.empty:
            print(f"Historical data: {len(historical_df)} candles")
            print(
                f"Price range: ${historical_df['low'].min():,.2f} - ${historical_df['high'].max():,.2f}"
            )
            print(f"Latest close: ${historical_df['close'].iloc[-1]:,.2f}")

        # Get order book snapshot
        orderbook = await client.get_orderbook_snapshot(level=2)
        if orderbook:
            best_bid = orderbook["bids"][0][0] if orderbook["bids"] else None
            best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None
            if best_bid and best_ask:
                spread = best_ask - best_bid
                print(
                    f"Best bid/ask: ${best_bid:,.2f} / ${best_ask:,.2f} (spread: ${spread:.2f})"
                )

        # Get connection status
        status = client.get_connection_status()
        print(f"Connection status: {status}")


async def example_real_time_updates():
    """Example of real-time market data updates."""
    print("\n=== Real-time Market Data Updates ===")

    update_count = 0
    max_updates = 10  # Limit updates for example

    def price_update_handler(market_data: MarketData):
        """Handle real-time price updates."""
        nonlocal update_count
        update_count += 1

        print(
            f"Price update #{update_count}: {market_data.symbol} @ ${market_data.close:,.2f} "
            f"(Vol: {market_data.volume:.2f}) at {market_data.timestamp}"
        )

        if update_count >= max_updates:
            print(f"Received {max_updates} updates, stopping...")

    # Create provider directly for more control
    provider = MarketDataProvider("BTC-USD", "1m")

    try:
        # Connect and subscribe to updates
        await provider.connect()
        provider.subscribe_to_updates(price_update_handler)

        print(
            "Listening for real-time updates... (this would run indefinitely in real usage)"
        )

        # In a real application, this would run continuously
        # For this example, we'll wait a short time
        await asyncio.sleep(30)  # Wait 30 seconds for updates

    except Exception as e:
        logger.error(f"Error in real-time updates: {e}")
    finally:
        await provider.disconnect()


async def example_advanced_usage():
    """Example of advanced market data features."""
    print("\n=== Advanced Market Data Features ===")

    provider = MarketDataProvider("ETH-USD", "5m")

    try:
        await provider.connect()

        # Fetch historical data with custom parameters
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        historical_data = await provider.fetch_historical_data(
            start_time=start_time, end_time=end_time, granularity="5m"
        )

        if historical_data:
            print(f"24-hour ETH-USD data: {len(historical_data)} 5-minute candles")

            # Calculate some basic statistics
            prices = [float(candle.close) for candle in historical_data]
            volumes = [float(candle.volume) for candle in historical_data]

            print(
                f"Price stats: Min=${min(prices):,.2f}, Max=${max(prices):,.2f}, "
                f"Avg=${sum(prices)/len(prices):,.2f}"
            )
            print(
                f"Volume stats: Total={sum(volumes):,.2f}, Avg={sum(volumes)/len(volumes):,.2f}"
            )

        # Test data validation
        print("\nTesting data validation...")
        if historical_data:
            valid_candles = 0
            for candle in historical_data:
                if provider._validate_market_data(candle):
                    valid_candles += 1

            print(
                f"Data quality: {valid_candles}/{len(historical_data)} candles passed validation "
                f"({valid_candles/len(historical_data)*100:.1f}%)"
            )

        # Test caching
        print("\nTesting cache functionality...")
        cache_status = provider.get_data_status()
        print(f"Cache status: {cache_status}")

        # Clear cache and check
        provider.clear_cache()
        print("Cache cleared")

        # Get latest OHLCV data as DataFrame
        df = provider.to_dataframe(limit=20)
        if not df.empty:
            print(f"Latest 20 candles DataFrame shape: {df.shape}")

    except Exception as e:
        logger.error(f"Error in advanced usage: {e}")
    finally:
        await provider.disconnect()


async def example_error_handling():
    """Example of error handling and reconnection."""
    print("\n=== Error Handling and Reconnection ===")

    # Create provider with invalid symbol to test error handling
    provider = MarketDataProvider("INVALID-SYMBOL", "1m")

    try:
        await provider.connect()

        # Try to fetch data for invalid symbol
        price = await provider.fetch_latest_price()
        if price:
            print(f"Price: ${price}")
        else:
            print("No price data available (expected for invalid symbol)")

        # Check connection status
        status = provider.get_data_status()
        print(
            f"Status: Connected={status['connected']}, "
            f"WebSocket={status.get('websocket_connected', False)}"
        )

    except Exception as e:
        print(f"Expected error for invalid symbol: {e}")
    finally:
        await provider.disconnect()

    # Test with valid symbol
    provider = MarketDataProvider("BTC-USD", "1m")
    try:
        await provider.connect()
        print("Successfully connected with valid symbol")

        # Test reconnection attempts status
        status = provider.get_data_status()
        print(f"Reconnection attempts: {status['reconnect_attempts']}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await provider.disconnect()


async def example_multiple_symbols():
    """Example of handling multiple symbols."""
    print("\n=== Multiple Symbols Example ===")

    symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
    clients = []

    try:
        # Create clients for multiple symbols
        for symbol in symbols:
            client = create_market_data_client(symbol, "1m")
            await client.connect()
            clients.append((symbol, client))

        # Get current prices for all symbols
        prices = {}
        for symbol, client in clients:
            try:
                price = await client.get_current_price()
                if price:
                    prices[symbol] = price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")

        print("Current prices:")
        for symbol, price in prices.items():
            print(f"  {symbol}: ${price:,.2f}")

        # Get historical data for comparison
        print("\nHistorical data comparison:")
        for symbol, client in clients:
            try:
                df = await client.get_historical_data(lookback_hours=1)
                if not df.empty:
                    change = (
                        (df["close"].iloc[-1] - df["close"].iloc[0])
                        / df["close"].iloc[0]
                    ) * 100
                    print(f"  {symbol}: {change:+.2f}% (1-hour change)")
            except Exception as e:
                logger.warning(f"Failed to get historical data for {symbol}: {e}")

    finally:
        # Disconnect all clients
        for symbol, client in clients:
            await client.disconnect()


async def main():
    """Run all examples."""
    print("Market Data Integration Examples")
    print("=" * 50)

    # Run examples
    await example_basic_usage()
    await example_advanced_usage()
    await example_error_handling()
    await example_multiple_symbols()

    # Note: Real-time updates example commented out as it requires extended runtime
    # await example_real_time_updates()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNotes:")
    print("- Some features require valid Coinbase API credentials")
    print(
        "- Real-time WebSocket updates work with both public and authenticated connections"
    )
    print("- Data caching reduces API calls and improves performance")
    print("- Error handling ensures robust operation in production")


if __name__ == "__main__":
    asyncio.run(main())
