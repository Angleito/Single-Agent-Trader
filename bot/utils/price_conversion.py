"""
Price conversion utilities for handling 18-decimal format prices from Bluefin DEX.

This module provides safe conversion functions that can detect if a price value
is in 18-decimal format and convert it appropriately, with validation and logging.
"""

import logging
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Price validation ranges for different symbols
PRICE_RANGES = {
    "SUI-PERP": {"min": 0.5, "max": 20.0},
    "BTC-PERP": {"min": 10000.0, "max": 200000.0},
    "ETH-PERP": {"min": 1000.0, "max": 20000.0},
    "SOL-PERP": {"min": 10.0, "max": 1000.0},
    # Default range for unknown symbols
    "default": {"min": 0.0001, "max": 1000000.0},
}


def is_likely_18_decimal(value: float | int | str | Decimal) -> bool:
    """
    Determine if a numeric value is likely in 18-decimal format.

    18-decimal format values are typically very large numbers (>1e10).
    Using 1e10 as threshold instead of 1e15 to catch more edge cases.

    Args:
        value: The numeric value to check

    Returns:
        bool: True if the value appears to be in 18-decimal format
    """
    try:
        numeric_value = float(value)
        # Values larger than 1e10 are likely in 18-decimal format
        # This catches values like 3.45e12 which should be ~3.45 after conversion
        return numeric_value > 1e10
    except (ValueError, TypeError):
        return False


def convert_from_18_decimal(
    value: float | int | str | Decimal,
    symbol: str | None = None,
    field_name: str | None = None,
) -> Decimal:
    """
    Safely convert a value from 18-decimal format to regular decimal.

    Args:
        value: The value to convert
        symbol: Trading symbol for validation (e.g., "SUI-PERP")
        field_name: Name of the field being converted (for logging)

    Returns:
        Decimal: The converted value

    Raises:
        ValueError: If the input value is invalid
    """
    if value is None:
        return Decimal("0")

    try:
        # Convert to Decimal for precision
        decimal_value = Decimal(str(value))

        # Check if conversion is needed
        if is_likely_18_decimal(decimal_value):
            converted_value = decimal_value / Decimal("1e18")
            logger.debug("Converted %s from 18-decimal: %s -> %s (symbol: %s)", field_name or 'value', decimal_value, converted_value, symbol)
        else:
            converted_value = decimal_value
            logger.debug("No conversion needed for %s: %s (symbol: %s)", field_name or 'value', decimal_value, symbol)

        # Validate the result
        if symbol and not is_price_valid(converted_value, symbol):
            logger.warning("Price %s for %s is outside expected range. Original value: %s, Field: %s", converted_value, symbol, value, field_name)

        return converted_value

    except (ValueError, TypeError, ArithmeticError) as e:
        logger.exception("Error converting value %s: %s", value, e)
        raise ValueError(f"Invalid numeric value: {value}") from e


def is_price_valid(price: float | Decimal, symbol: str) -> bool:
    """
    Validate if a price is within expected range for a given symbol.

    Args:
        price: The price to validate
        symbol: Trading symbol (e.g., "SUI-PERP")

    Returns:
        bool: True if price is within expected range
    """
    try:
        price_float = float(price)

        # Get price range for symbol
        price_range = PRICE_RANGES.get(symbol, PRICE_RANGES["default"])

        return price_range["min"] <= price_float <= price_range["max"]

    except (ValueError, TypeError):
        return False


def convert_candle_data(candle: list, symbol: str | None = None) -> list:
    """
    Convert candle data from 18-decimal format to regular decimal.

    Args:
        candle: List containing [timestamp, open, high, low, close, volume]
        symbol: Trading symbol for validation

    Returns:
        list: Converted candle data
    """
    if not isinstance(candle, list) or len(candle) < 6:
        raise ValueError(f"Invalid candle format: {candle}")

    try:
        converted_candle = [
            int(candle[0]) if candle[0] else 0,  # timestamp
            float(convert_from_18_decimal(candle[1], symbol, "open")),  # open
            float(convert_from_18_decimal(candle[2], symbol, "high")),  # high
            float(convert_from_18_decimal(candle[3], symbol, "low")),  # low
            float(convert_from_18_decimal(candle[4], symbol, "close")),  # close
            float(convert_from_18_decimal(candle[5], symbol, "volume")),  # volume
        ]

        logger.debug("Converted candle for %s: OHLCV = %s", symbol, converted_candle[1:6])
        return converted_candle

    except (ValueError, TypeError, IndexError) as e:
        logger.exception("Error converting candle data: %s", e)
        raise ValueError(f"Failed to convert candle data: {candle}") from e


def convert_ticker_price(
    price_data: dict[str, Any], symbol: str | None = None
) -> dict[str, Any]:
    """
    Convert ticker price data from 18-decimal format.

    Args:
        price_data: Dictionary containing price information
        symbol: Trading symbol for validation

    Returns:
        Dict: Converted price data
    """
    converted_data = {}

    for key, value in price_data.items():
        if key in [
            "price",
            "lastPrice",
            "bestBid",
            "bestAsk",
            "high",
            "low",
            "open",
            "close",
        ]:
            try:
                converted_data[key] = str(convert_from_18_decimal(value, symbol, key))
            except ValueError:
                logger.warning("Failed to convert %s: %s", key, value)
                converted_data[key] = str(value)
        else:
            converted_data[key] = value

    return converted_data


def log_price_conversion_stats(
    original_value: float | Decimal,
    converted_value: float | Decimal,
    symbol: str,
    field_name: str,
) -> None:
    """
    Log detailed price conversion statistics for debugging.

    Args:
        original_value: Original value before conversion
        converted_value: Value after conversion
        symbol: Trading symbol
        field_name: Name of the field being converted
    """
    logger.info("Price conversion stats for %s:%s - Original: %s, Converted: %s, Ratio: %s, Valid: %s", 
                symbol, field_name, original_value, converted_value, 
                float(original_value) / float(converted_value) if converted_value != 0 else 'N/A',
                is_price_valid(converted_value, symbol))


def get_current_real_price(symbol: str) -> float | None:
    """
    Get current real market price for a symbol.

    This function attempts to fetch the current real market price from available
    market data sources. It's used by the paper trading system to ensure
    simulations use real market data instead of mock prices.

    Args:
        symbol: Trading symbol (e.g., 'SUI-PERP', 'BTC-USD')

    Returns:
        Current market price as float, or None if unavailable
    """
    try:
        # Try to import and use the market data providers
        from bot.data.market import MarketDataProvider

        # Create a temporary market data provider to get current price
        market_provider = MarketDataProvider(symbol)

        # Get latest OHLCV data
        latest_data = market_provider.get_latest_ohlcv(limit=1)
        if latest_data and len(latest_data) > 0:
            current_price = float(latest_data[-1].close)
            logger.debug("Retrieved real market price for %s: $%s", symbol, current_price)
            return current_price

    except Exception as e:
        logger.debug("Could not fetch real market price for %s: %s", symbol, e)

    try:
        # Try Bluefin market data provider if available
        from bot.data.bluefin_market import BluefinMarketDataProvider

        bluefin_provider = BluefinMarketDataProvider(symbol)
        latest_data = bluefin_provider.get_latest_ohlcv(limit=1)
        if latest_data and len(latest_data) > 0:
            current_price = float(latest_data[-1].close)
            logger.debug("Retrieved real Bluefin price for %s: $%s", symbol, current_price)
            return current_price

    except Exception as e:
        logger.debug("Could not fetch real Bluefin price for %s: %s", symbol, e)

    # Return None if no real price could be fetched
    logger.debug("No real market price available for %s, using fallback", symbol)
    return None
