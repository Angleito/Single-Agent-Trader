#!/usr/bin/env python3
"""Test script to validate price conversion from 18-decimal format."""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.utils.price_conversion import (
    convert_candle_data,
    convert_from_18_decimal,
    convert_ticker_price,
    is_likely_18_decimal,
    is_price_valid,
)


def test_price_conversions():
    """Test various price conversion scenarios."""
    print("=" * 80)
    print("PRICE CONVERSION TESTS")
    print("=" * 80)

    # Test astronomical price values (18-decimal format)
    test_prices = [
        ("2648900000000000000", "SUI-PERP", "Astronomical SUI price"),
        ("69420000000000000000000", "BTC-PERP", "Astronomical BTC price"),
        ("3450000000000000000", "ETH-PERP", "Astronomical ETH price"),
        ("2.64", "SUI-PERP", "Normal SUI price"),
        ("69420", "BTC-PERP", "Normal BTC price"),
        ("3450", "ETH-PERP", "Normal ETH price"),
    ]

    print("\n1. Testing individual price conversions:")
    print("-" * 80)
    for price_str, symbol, description in test_prices:
        price = Decimal(price_str)
        is_18_dec = is_likely_18_decimal(price)
        converted = convert_from_18_decimal(price, symbol, "test_price")
        is_valid = is_price_valid(converted, symbol)

        print(f"\n{description}:")
        print(f"  Original: {price}")
        print(f"  Is 18-decimal: {is_18_dec}")
        print(f"  Converted: {converted}")
        print(f"  Is valid: {is_valid}")
        print(f"  Human readable: ${float(converted):,.2f}")

    # Test ticker data conversion
    print("\n\n2. Testing ticker data conversion:")
    print("-" * 80)
    ticker_data = {
        "price": "2648900000000000000",
        "bestBid": "2648800000000000000",
        "bestAsk": "2649000000000000000",
        "high": "2700000000000000000",
        "low": "2600000000000000000",
        "volume": "50000000000000000000000",
    }

    converted_ticker = convert_ticker_price(ticker_data, "SUI-PERP")
    print("\nOriginal ticker data:")
    for key, value in ticker_data.items():
        print(f"  {key}: {value}")

    print("\nConverted ticker data:")
    for key, value in converted_ticker.items():
        if key in ["price", "bestBid", "bestAsk", "high", "low"]:
            print(f"  {key}: {value} (${float(Decimal(value)):,.4f})")
        else:
            print(f"  {key}: {value}")

    # Test candle data conversion
    print("\n\n3. Testing candle data conversion:")
    print("-" * 80)
    candle = [
        1234567890,  # timestamp
        "2650000000000000000",  # open
        "2700000000000000000",  # high
        "2600000000000000000",  # low
        "2648900000000000000",  # close
        "10000000000000000000000",  # volume
    ]

    print("\nOriginal candle:")
    print(f"  Timestamp: {candle[0]}")
    print(f"  OHLCV: {candle[1:]}")

    converted_candle = convert_candle_data(candle, "SUI-PERP")
    print("\nConverted candle:")
    print(f"  Timestamp: {converted_candle[0]}")
    print(f"  Open:  ${converted_candle[1]:,.4f}")
    print(f"  High:  ${converted_candle[2]:,.4f}")
    print(f"  Low:   ${converted_candle[3]:,.4f}")
    print(f"  Close: ${converted_candle[4]:,.4f}")
    print(f"  Volume: {converted_candle[5]:,.2f}")

    # Test balance conversion
    print("\n\n4. Testing balance conversion:")
    print("-" * 80)
    test_balances = [
        ("10000000000000000000000", "Astronomical balance"),
        ("10000", "Normal balance"),
        ("0", "Zero balance"),
    ]

    for balance_str, description in test_balances:
        balance = Decimal(balance_str)
        is_18_dec = is_likely_18_decimal(balance)
        converted = convert_from_18_decimal(balance, "USDC", "balance")

        print(f"\n{description}:")
        print(f"  Original: {balance}")
        print(f"  Is 18-decimal: {is_18_dec}")
        print(f"  Converted: {converted}")
        print(f"  Human readable: ${float(converted):,.2f}")


async def test_live_data_integration():
    """Test price conversion with live Bluefin data."""
    print("\n\n" + "=" * 80)
    print("LIVE DATA INTEGRATION TEST")
    print("=" * 80)

    try:
        from bot.data.bluefin_market import BluefinMarketDataProvider

        print("\nTesting Bluefin market data provider...")
        provider = BluefinMarketDataProvider("SUI-PERP", "1m")

        # Initialize provider
        await provider.initialize()

        # Fetch latest price
        print("\nFetching latest price...")
        latest_price = await provider.fetch_latest_price()
        if latest_price:
            print(f"Latest price: ${float(latest_price):,.4f}")
            print(f"Is valid: {is_price_valid(latest_price, 'SUI-PERP')}")
        else:
            print("Failed to fetch latest price")

        # Get data status
        status = provider.get_data_status()
        print("\nData status:")
        print(f"  Symbol: {status.get('symbol')}")
        print(f"  Latest price: {status.get('latest_price')}")
        print(f"  Connected: {status.get('connected')}")

        # Cleanup
        await provider.cleanup()

    except Exception as e:
        print(f"Error testing live data: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    # Run synchronous tests
    test_price_conversions()

    # Run async tests
    await test_live_data_integration()

    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
