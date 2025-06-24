"""
Price conversion utilities for handling 18-decimal format prices from Bluefin DEX.

This module provides safe conversion functions that can detect if a price value
is in 18-decimal format and convert it appropriately, with validation and logging.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite, isnan
from typing import Any

logger = logging.getLogger(__name__)

# Counter for rate-limiting astronomical value logging
_ASTRONOMICAL_LOG_COUNTER = {}
_LOG_EVERY_N_INSTANCES = 25

# Circuit breaker for consecutive astronomical price detections
_CIRCUIT_BREAKER_STATE = {
    "consecutive_failures": {},  # Track failures per symbol/field
    "is_open": {},  # Circuit breaker open state per symbol
    "last_known_good_prices": {},  # Store last valid prices per symbol
    "failure_timestamps": {},  # Track failure timing
}
_CIRCUIT_BREAKER_THRESHOLD = 5  # Max consecutive failures before opening circuit
_CIRCUIT_BREAKER_TIMEOUT = 300  # Seconds before attempting to close circuit

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

    Enhanced detection for better edge case handling:
    - Values > 1e15 are very likely 18-decimal format
    - Values between 1e10-1e15 are checked more carefully with pattern detection
    - Special handling for suspicious patterns like 12345000000000000
    - Consider the magnitude relative to expected price ranges

    Args:
        value: The numeric value to check

    Returns:
        bool: True if the value appears to be in 18-decimal format
    """
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return False
    else:
        # Enhanced thresholds for better edge case detection
        if numeric_value > 1e15:
            # Definitely 18-decimal format
            is_18_decimal = True
        elif numeric_value > 1e10:
            # Enhanced edge case detection for values between 1e10-1e15
            # Check for suspicious patterns that indicate 18-decimal format

            # Pattern 1: Check if the value has many trailing zeros (like 12345000000000000)
            value_str = (
                str(int(numeric_value))
                if numeric_value == int(numeric_value)
                else str(numeric_value)
            )
            trailing_zeros = len(value_str) - len(value_str.rstrip("0"))

            # Pattern 2: Check magnitude relative to typical crypto prices
            # Most legitimate crypto prices are < $100k, so anything > 1M is suspicious
            magnitude_suspicious = numeric_value > 1e6

            # Pattern 3: Check for values that end in many zeros (common in 18-decimal)
            # Values like 12345000000000000 have 9+ trailing zeros
            pattern_suspicious = trailing_zeros >= 9

            # Pattern 4: Check if value would make sense as 18-decimal conversion
            # Convert and see if result falls in reasonable price range
            potential_converted = numeric_value / 1e18
            reasonable_after_conversion = 0.0001 <= potential_converted <= 100000

            is_18_decimal = (
                magnitude_suspicious
                or pattern_suspicious
                or (reasonable_after_conversion and numeric_value > 5e10)
            )

            if is_18_decimal:
                logger.debug(
                    "Edge case 18-decimal detected: %s (mag_sus=%s, pat_sus=%s, trails=%d, conv=%s)",
                    value,
                    magnitude_suspicious,
                    pattern_suspicious,
                    trailing_zeros,
                    potential_converted,
                )
        else:
            is_18_decimal = False

        # Log detection for debugging with enhanced information
        if is_18_decimal:
            magnitude = "very_high" if numeric_value > 1e15 else "high"
            logger.debug(
                "Detected likely 18-decimal value (%s magnitude): %s (%.2e)",
                magnitude,
                value,
                numeric_value,
            )

        return is_18_decimal


def convert_from_18_decimal(
    value: float | int | str | Decimal,
    symbol: str | None = None,
    field_name: str | None = None,
    use_circuit_breaker: bool = True,
) -> Decimal:
    """
    Safely convert a value from 18-decimal format to regular decimal.

    Enhanced with circuit breaker functionality and fallback mechanisms.
    When consecutive conversion failures occur, the circuit breaker opens
    and fallback to last known good prices is used.

    Args:
        value: The value to convert
        symbol: Trading symbol for validation (e.g., "SUI-PERP")
        field_name: Name of the field being converted (for logging)
        use_circuit_breaker: Enable circuit breaker functionality

    Returns:
        Decimal: The converted value

    Raises:
        ValueError: If the input value is invalid and no fallback available
    """
    if value is None:
        return Decimal(0)

    # Pre-processing data sanitization
    sanitized_value = _sanitize_price_input(value)
    if sanitized_value is None:
        logger.warning(
            "Failed to sanitize input value %s for %s:%s, using fallback",
            value,
            symbol,
            field_name,
        )
        return _get_fallback_price(symbol, field_name)

    # Check circuit breaker state
    circuit_key = f"{symbol or 'unknown'}_{field_name or 'value'}"

    # Additional pre-conversion validation
    if not _validate_price_before_conversion(sanitized_value, symbol, field_name):
        logger.warning(
            "Pre-conversion validation failed for %s:%s value %s, using fallback",
            symbol,
            field_name,
            sanitized_value,
        )
        if use_circuit_breaker:
            _record_conversion_failure(circuit_key)
        return _get_fallback_price(symbol, field_name)
    if use_circuit_breaker and _is_circuit_breaker_open(circuit_key):
        logger.warning(
            "Circuit breaker OPEN for %s - using fallback price", circuit_key
        )
        return _get_fallback_price(symbol, field_name)

    try:
        # Convert to Decimal for precision
        decimal_value = Decimal(str(sanitized_value))

        # Edge case handling for zero or negative values
        if decimal_value <= 0:
            if field_name in ["volume", "quantity"]:
                # Volume can be zero, that's valid
                return decimal_value
            if decimal_value < 0:
                logger.warning(
                    "Negative %s value detected: %s (symbol: %s)",
                    field_name or "value",
                    decimal_value,
                    symbol,
                )
                # For prices, negative values are invalid, but we'll return absolute value
                if field_name in ["price", "open", "high", "low", "close"]:
                    raise ValueError(f"Invalid negative price: {value}")

        # Check if conversion is needed
        if is_likely_18_decimal(decimal_value):
            converted_value = decimal_value / Decimal("1e18")

            # Rate-limited logging for astronomical value detection
            if decimal_value > 1e18:
                # Create unique key for this symbol+field combination
                log_key = f"{symbol or 'unknown'}_{field_name or 'value'}"

                # Increment counter for this combination
                _ASTRONOMICAL_LOG_COUNTER[log_key] = (
                    _ASTRONOMICAL_LOG_COUNTER.get(log_key, 0) + 1
                )
                current_count = _ASTRONOMICAL_LOG_COUNTER[log_key]

                # Only log every N instances
                if current_count % _LOG_EVERY_N_INSTANCES == 1:
                    logger.info(
                        "Astronomical value detected for %s (instance %d/%d): %s -> %s (symbol: %s)",
                        field_name or "value",
                        int(current_count),
                        int(current_count + _LOG_EVERY_N_INSTANCES - 1),
                        decimal_value,
                        converted_value,
                        symbol,
                    )
            else:
                logger.debug(
                    "Converted %s from 18-decimal: %s -> %s (symbol: %s)",
                    field_name or "value",
                    decimal_value,
                    converted_value,
                    symbol,
                )
        else:
            converted_value = decimal_value
            logger.debug(
                "No conversion needed for %s: %s (symbol: %s)",
                field_name or "value",
                decimal_value,
                symbol,
            )

        # Enhanced validation and circuit breaker logic
        if symbol and field_name in ["price", "open", "high", "low", "close"]:
            if not is_price_valid(converted_value, symbol):
                logger.warning(
                    "Price %s for %s is outside expected range. Original value: %s, Field: %s",
                    converted_value,
                    symbol,
                    value,
                    field_name,
                )
                if use_circuit_breaker:
                    _record_conversion_failure(circuit_key)
                    # Check if we should open circuit breaker
                    if _should_open_circuit_breaker(circuit_key):
                        logger.error(
                            "Opening circuit breaker for %s after repeated validation failures",
                            circuit_key,
                        )
                        _open_circuit_breaker(circuit_key)
                        return _get_fallback_price(symbol, field_name)
            # Successful conversion - reset circuit breaker and update last known good price
            elif use_circuit_breaker:
                _record_conversion_success(circuit_key, converted_value)
                _update_last_known_good_price(symbol, field_name, converted_value)

    except (ValueError, TypeError, ArithmeticError) as e:
        logger.exception(
            "Error converting value %s for field %s (symbol: %s)",
            value,
            field_name,
            symbol,
        )

        # Record failure and potentially use fallback
        if use_circuit_breaker:
            _record_conversion_failure(circuit_key)
            if _should_open_circuit_breaker(circuit_key):
                logger.error(
                    "Opening circuit breaker for %s after conversion error", circuit_key
                )
                _open_circuit_breaker(circuit_key)
                return _get_fallback_price(symbol, field_name)

        raise ValueError(
            f"Invalid numeric value: {value} for field {field_name}"
        ) from e
    else:
        return converted_value


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
    except (ValueError, TypeError):
        return False
    else:
        # Basic bounds checking for all prices
        if price_float <= 0 or price_float > 1e6:
            return False

        # Get price range for symbol
        price_range = PRICE_RANGES.get(symbol, PRICE_RANGES["default"])

        return price_range["min"] <= price_float <= price_range["max"]


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

        logger.debug(
            "Converted candle for %s: OHLCV = %s", symbol, converted_candle[1:6]
        )
    except (ValueError, TypeError, IndexError) as e:
        logger.exception("Error converting candle data")
        raise ValueError(f"Failed to convert candle data: {candle}") from e
    else:
        return converted_candle


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
    logger.info(
        "Price conversion stats for %s:%s - Original: %s, Converted: %s, Ratio: %s, Valid: %s",
        symbol,
        field_name,
        original_value,
        converted_value,
        (
            float(original_value) / float(converted_value)
            if converted_value != 0
            else "N/A"
        ),
        is_price_valid(converted_value, symbol),
    )


def format_price_for_display(
    price: float | Decimal | str | None, symbol: str | None = None, decimals: int = 2
) -> str:
    """
    Format a price value for display in logs or UI.

    This function ensures prices are displayed in human-readable format,
    automatically detecting and converting 18-decimal values.

    Args:
        price: The price value to format
        symbol: Optional trading symbol for context
        decimals: Number of decimal places to display

    Returns:
        Formatted price string with $ prefix
    """
    if price is None:
        return "$N/A"

    try:
        # Convert to Decimal for precision
        price_decimal = Decimal(str(price))

        # Check if conversion from 18-decimal is needed
        if is_likely_18_decimal(price_decimal):
            price_decimal = convert_from_18_decimal(
                price_decimal, symbol, "display_price"
            )

        # Format with specified decimals
        formatted = f"${float(price_decimal):,.{decimals}f}"

        # Add symbol context if provided
        if symbol:
            formatted = f"{formatted} ({symbol})"

        return formatted

    except Exception as e:
        logger.warning("Failed to format price %s: %s", price, e)
        return f"${price}"


def _sanitize_price_input(value: Any) -> Any:
    """
    Enhanced sanitization of price input before processing to catch malformed data.

    Performs comprehensive validation including:
    - Format validation and cleanup
    - Range checking for extreme values
    - Pattern detection for corrupted data
    - Type safety validation

    Args:
        value: Raw input value

    Returns:
        Sanitized value or None if unsanitizable
    """
    try:
        # Handle None/empty values
        if value is None or value == "":
            return None

        # Handle string values with enhanced validation
        if isinstance(value, str):
            # Remove whitespace and common formatting
            cleaned = value.strip().replace(",", "").replace("$", "")
            if not cleaned or cleaned.lower() in [
                "nan",
                "inf",
                "-inf",
                "null",
                "undefined",
                "none",
            ]:
                return None

            # Check for obviously corrupted string patterns
            if len(cleaned) > 50:  # Unreasonably long number string
                logger.warning(
                    "Suspiciously long numeric string: %s", value[:50] + "..."
                )
                return None

            # Check for multiple decimal points or other formatting issues
            if cleaned.count(".") > 1:
                logger.warning("Multiple decimal points detected: %s", value)
                return None

            # Check for scientific notation abuse or corruption
            if "e+" in cleaned.lower():
                if cleaned.lower().count("e+") > 1:
                    logger.warning("Malformed scientific notation detected: %s", value)
                    return None
                # Validate scientific notation format
                try:
                    test_val = float(cleaned)
                    if abs(test_val) > 1e20:
                        logger.warning("Scientific notation value too large: %s", value)
                        return None
                except ValueError:
                    logger.warning("Invalid scientific notation: %s", value)
                    return None

            # Check for suspicious patterns in the string
            # Like repeated sequences that might indicate corruption
            if len(cleaned) >= 10:
                # Check for repeated digit patterns (like 11111111111)
                if len(set(cleaned.replace(".", "").replace("-", ""))) <= 2:
                    logger.warning("Suspicious repeated digit pattern: %s", value)
                    return None

            return cleaned

        # Handle numeric values with enhanced validation
        if isinstance(value, (int, float, Decimal)):
            # Check for NaN, infinity, or obviously corrupted values
            if isinstance(value, float) and (not isfinite(value) or isnan(value)):
                return None

            # Convert to float for range checking
            float_val = float(value)

            # Check for suspiciously large numbers that might be corrupted
            if abs(float_val) > 1e20:  # Beyond reasonable 18-decimal values
                logger.warning("Suspiciously large value detected: %s", value)
                return None

            # Check for precision issues that might indicate corruption
            # Very precise decimals with many digits might be corrupted
            if isinstance(value, float):
                str_repr = str(value)
                if (
                    "e+" not in str_repr.lower()
                    and len(str_repr.replace(".", "").replace("-", "")) > 15
                ):
                    logger.debug("High precision float detected, monitoring: %s", value)

            # Check for values that are exactly zero but represented strangely
            if float_val == 0.0 and str(value) not in ["0", "0.0", "0.00"]:
                logger.debug("Unusual zero representation: %s", value)

            return value

        # Handle Decimal objects
        if hasattr(value, "__class__") and "Decimal" in str(value.__class__):
            try:
                float_val = float(value)
                if abs(float_val) > 1e20:
                    logger.warning("Decimal value too large: %s", value)
                    return None
                return value
            except (ValueError, OverflowError):
                logger.warning("Invalid Decimal object: %s", value)
                return None

        # Try to convert other types with validation
        try:
            converted = str(value)
            if len(converted) > 100:  # Prevent memory issues
                logger.warning("Converted string too long: %s", converted[:50] + "...")
                return None
            return converted
        except Exception as conv_e:
            logger.debug("Failed to convert type %s to string: %s", type(value), conv_e)
            return None

    except Exception as e:
        logger.debug("Failed to sanitize input %s: %s", value, e)
        return None


def _validate_price_before_conversion(
    value: Any, symbol: str | None, field_name: str | None
) -> bool:
    """
    Validate price value before attempting conversion.

    Performs additional safety checks beyond sanitization:
    - Checks for patterns that historically cause issues
    - Validates against symbol-specific ranges
    - Detects potential data corruption indicators

    Args:
        value: Sanitized value to validate
        symbol: Trading symbol for context
        field_name: Field name for context

    Returns:
        True if value passes pre-conversion validation
    """
    try:
        # Convert to float for analysis
        float_val = float(value)

        # Check for basic invalid conditions
        if not isfinite(float_val) or isnan(float_val):
            logger.debug("Pre-validation failed: non-finite value %s", value)
            return False

        # Check for suspiciously round numbers that might indicate placeholder data
        # Like exactly 1000000000000000000 (1e18) which is often a default/error
        if float_val in [1e18, 1e17, 1e16, 1e15]:
            logger.warning("Suspicious round 18-decimal value detected: %s", value)
            return False

        # For price fields, do enhanced validation
        if field_name in ["price", "open", "high", "low", "close"]:
            # Check for values that are exactly powers of 10 (often indicates errors)
            if float_val > 1e10 and float_val in [10**i for i in range(10, 21)]:
                logger.warning("Suspicious power-of-10 price detected: %s", value)
                return False

        # Symbol-specific pre-validation
        if symbol and field_name in ["price", "open", "high", "low", "close"]:
            # Get expected range for symbol
            price_range = PRICE_RANGES.get(symbol, PRICE_RANGES["default"])

            # If value is too large even for 18-decimal conversion, reject it
            max_18_decimal = price_range["max"] * 1e18
            if float_val > max_18_decimal * 10:  # Allow 10x margin for safety
                logger.warning(
                    "Value %s exceeds maximum possible 18-decimal for %s (max: %s)",
                    value,
                    symbol,
                    max_18_decimal,
                )
                return False

        # Enhanced pattern detection for suspicious digit repetition
        if isinstance(value, str):
            # Look for patterns like repeated sequences of digits
            cleaned_str = value.replace(".", "").replace("-", "")
            if len(cleaned_str) > 8:
                # Check for too many repeated digits (like 1111111111111111)
                digit_counts = {}
                for digit in cleaned_str:
                    digit_counts[digit] = digit_counts.get(digit, 0) + 1

                # If any single digit appears more than 80% of the time, it's suspicious
                # This catches clearly corrupted data while allowing legitimate values
                max_count = max(digit_counts.values())
                repetition_ratio = max_count / len(cleaned_str)
                if repetition_ratio > 0.8:
                    logger.warning(
                        "Suspicious repeated digit pattern: %s (%.1f%% repetition)",
                        value,
                        repetition_ratio * 100,
                    )
                    return False

                # Additional check for suspicious ending patterns (many zeros)
                # Only flag if there are 12+ trailing zeros (more conservative)
                if cleaned_str.endswith("000000000000"):  # 12+ trailing zeros
                    trailing_zeros = len(cleaned_str) - len(cleaned_str.rstrip("0"))
                    if trailing_zeros >= 12:
                        logger.warning(
                            "Suspicious trailing zeros pattern: %s (%d zeros)",
                            value,
                            trailing_zeros,
                        )
                        return False

        return True

    except Exception as e:
        logger.debug("Pre-validation check failed for %s: %s", value, e)
        return False


def _is_circuit_breaker_open(circuit_key: str) -> bool:
    """
    Check if circuit breaker is open for a given key.

    Args:
        circuit_key: Circuit breaker key (symbol_field)

    Returns:
        True if circuit breaker is open
    """
    if circuit_key not in _CIRCUIT_BREAKER_STATE["is_open"]:
        return False

    is_open = _CIRCUIT_BREAKER_STATE["is_open"][circuit_key]
    if not is_open:
        return False

    # Check if timeout has passed and we should attempt to close
    if circuit_key in _CIRCUIT_BREAKER_STATE["failure_timestamps"]:
        last_failure = _CIRCUIT_BREAKER_STATE["failure_timestamps"][circuit_key]
        current_time = datetime.now(UTC).timestamp()
        if current_time - last_failure > _CIRCUIT_BREAKER_TIMEOUT:
            logger.info(
                "Circuit breaker timeout passed for %s, attempting to close",
                circuit_key,
            )
            _CIRCUIT_BREAKER_STATE["is_open"][circuit_key] = False
            return False

    return True


def _record_conversion_failure(circuit_key: str) -> None:
    """
    Record a conversion failure for circuit breaker tracking.

    Args:
        circuit_key: Circuit breaker key (symbol_field)
    """
    if circuit_key not in _CIRCUIT_BREAKER_STATE["consecutive_failures"]:
        _CIRCUIT_BREAKER_STATE["consecutive_failures"][circuit_key] = 0

    _CIRCUIT_BREAKER_STATE["consecutive_failures"][circuit_key] += 1
    _CIRCUIT_BREAKER_STATE["failure_timestamps"][circuit_key] = datetime.now(
        UTC
    ).timestamp()

    logger.debug(
        "Recorded conversion failure for %s (count: %d)",
        circuit_key,
        _CIRCUIT_BREAKER_STATE["consecutive_failures"][circuit_key],
    )


def _record_conversion_success(circuit_key: str, converted_value: Decimal) -> None:
    """
    Record a successful conversion, resetting failure counters.

    Args:
        circuit_key: Circuit breaker key (symbol_field)
        converted_value: Successfully converted value
    """
    _CIRCUIT_BREAKER_STATE["consecutive_failures"][circuit_key] = 0
    if circuit_key in _CIRCUIT_BREAKER_STATE["is_open"]:
        _CIRCUIT_BREAKER_STATE["is_open"][circuit_key] = False

    logger.debug(
        "Reset circuit breaker for %s after successful conversion", circuit_key
    )


def _should_open_circuit_breaker(circuit_key: str) -> bool:
    """
    Check if circuit breaker should be opened based on failure count.

    Args:
        circuit_key: Circuit breaker key (symbol_field)

    Returns:
        True if circuit breaker should be opened
    """
    failure_count = _CIRCUIT_BREAKER_STATE["consecutive_failures"].get(circuit_key, 0)
    return failure_count >= _CIRCUIT_BREAKER_THRESHOLD


def _open_circuit_breaker(circuit_key: str) -> None:
    """
    Open circuit breaker for a given key.

    Args:
        circuit_key: Circuit breaker key (symbol_field)
    """
    _CIRCUIT_BREAKER_STATE["is_open"][circuit_key] = True
    logger.warning("Circuit breaker OPENED for %s", circuit_key)


def _update_last_known_good_price(
    symbol: str | None, field_name: str | None, price: Decimal
) -> None:
    """
    Update the last known good price for fallback purposes.

    Args:
        symbol: Trading symbol
        field_name: Field name
        price: Valid price value
    """
    if not symbol or not field_name:
        return

    price_key = f"{symbol}_{field_name}"
    _CIRCUIT_BREAKER_STATE["last_known_good_prices"][price_key] = {
        "price": price,
        "timestamp": datetime.now(UTC).timestamp(),
    }

    logger.debug("Updated last known good price for %s: %s", price_key, price)


def _get_fallback_price(symbol: str | None, field_name: str | None) -> Decimal:
    """
    Enhanced fallback price mechanism when conversion fails or circuit breaker is open.

    Tries multiple fallback strategies in order:
    1. Last known good price (if recent)
    2. Current real-time market price
    3. Historical average price (if available)
    4. Symbol-specific reasonable default
    5. Ultimate safe fallback

    Args:
        symbol: Trading symbol
        field_name: Field name

    Returns:
        Fallback price value
    """
    if symbol and field_name:
        price_key = f"{symbol}_{field_name}"

        # Strategy 1: Try to get last known good price with extended age tolerance
        if price_key in _CIRCUIT_BREAKER_STATE["last_known_good_prices"]:
            good_price_data = _CIRCUIT_BREAKER_STATE["last_known_good_prices"][
                price_key
            ]
            age = datetime.now(UTC).timestamp() - good_price_data["timestamp"]

            # Use last known good price if it's not too old
            # Extended tolerance: 2 hours for prices, 6 hours for non-critical fields
            max_age = 7200 if field_name in ["price", "close"] else 21600
            if age < max_age:
                logger.info(
                    "Using last known good price for %s: %s (age: %.1f minutes)",
                    price_key,
                    good_price_data["price"],
                    age / 60,
                )
                return good_price_data["price"]
            logger.debug(
                "Last known good price for %s is too old (%.1f hours), trying other fallbacks",
                price_key,
                age / 3600,
            )

        # Strategy 2: Try to get current real price as fallback
        try:
            current_price = get_current_real_price(symbol)
            if current_price and current_price > 0:
                fallback_price = Decimal(str(current_price))
                logger.info(
                    "Using current real price as fallback for %s: %s",
                    symbol,
                    fallback_price,
                )
                # Update last known good price with this current price
                _update_last_known_good_price(symbol, field_name, fallback_price)
                return fallback_price
        except Exception as e:
            logger.debug("Could not get current real price for fallback: %s", e)

        # Strategy 3: Try to get historical average from stored good prices
        try:
            related_prices = []
            for key, price_data in _CIRCUIT_BREAKER_STATE[
                "last_known_good_prices"
            ].items():
                if key.startswith(f"{symbol}_") and key.endswith(
                    ("_price", "_close", "_open")
                ):
                    age = datetime.now(UTC).timestamp() - price_data["timestamp"]
                    if age < 86400:  # Within 24 hours
                        related_prices.append(float(price_data["price"]))

            if related_prices:
                avg_price = sum(related_prices) / len(related_prices)
                fallback_price = Decimal(str(avg_price))
                logger.info(
                    "Using historical average as fallback for %s: %s (from %d prices)",
                    symbol,
                    fallback_price,
                    len(related_prices),
                )
                return fallback_price
        except Exception as e:
            logger.debug("Could not calculate historical average for fallback: %s", e)

        # Strategy 4: Use symbol-specific reasonable default
        if symbol in PRICE_RANGES:
            price_range = PRICE_RANGES[symbol]
            # Use midpoint of expected range, but prefer lower end for safety
            if field_name in ["price", "close", "open"]:
                # For price fields, use midpoint
                fallback_price = Decimal(
                    str((price_range["min"] + price_range["max"]) / 2)
                )
            else:
                # For other fields, use lower end for safety
                fallback_price = Decimal(str(price_range["min"] * 2))

            logger.warning(
                "Using symbol-specific fallback price for %s:%s: %s",
                symbol,
                field_name,
                fallback_price,
            )
            return fallback_price

    # Strategy 5: Ultimate fallback with field-specific defaults
    if field_name in ["volume", "quantity"]:
        # Volume can be zero, that's safe
        fallback_value = Decimal(0)
    elif field_name in ["price", "close", "open", "high", "low"]:
        # Price fields need positive values
        fallback_value = Decimal("1.0")  # More reasonable than 0.01 for most cryptos
    else:
        # Other fields get small positive value
        fallback_value = Decimal("0.01")

    logger.error(
        "No fallback price available for %s:%s, using ultimate fallback: %s",
        symbol,
        field_name,
        fallback_value,
    )
    return fallback_value


def get_circuit_breaker_status() -> dict[str, Any]:
    """
    Get current circuit breaker status for monitoring.

    Returns:
        Dictionary with circuit breaker status information
    """
    return {
        "open_circuits": {
            k: v for k, v in _CIRCUIT_BREAKER_STATE["is_open"].items() if v
        },
        "failure_counts": dict(_CIRCUIT_BREAKER_STATE["consecutive_failures"]),
        "last_known_good_prices": {
            k: {
                "price": float(v["price"]),
                "age_minutes": (datetime.now(UTC).timestamp() - v["timestamp"]) / 60,
            }
            for k, v in _CIRCUIT_BREAKER_STATE["last_known_good_prices"].items()
        },
        "threshold": _CIRCUIT_BREAKER_THRESHOLD,
        "timeout_seconds": _CIRCUIT_BREAKER_TIMEOUT,
    }


def reset_circuit_breaker(circuit_key: str | None = None) -> None:
    """
    Reset circuit breaker state for debugging/recovery.

    Args:
        circuit_key: Specific circuit to reset, or None for all
    """
    if circuit_key:
        _CIRCUIT_BREAKER_STATE["consecutive_failures"].pop(circuit_key, None)
        _CIRCUIT_BREAKER_STATE["is_open"].pop(circuit_key, None)
        _CIRCUIT_BREAKER_STATE["failure_timestamps"].pop(circuit_key, None)
        logger.info("Reset circuit breaker for %s", circuit_key)
    else:
        _CIRCUIT_BREAKER_STATE["consecutive_failures"].clear()
        _CIRCUIT_BREAKER_STATE["is_open"].clear()
        _CIRCUIT_BREAKER_STATE["failure_timestamps"].clear()
        logger.info("Reset all circuit breakers")


def get_conversion_statistics() -> dict[str, Any]:
    """
    Get statistics about price conversions and astronomical value detections.

    Returns:
        Dictionary with conversion statistics
    """
    total_astronomical = sum(_ASTRONOMICAL_LOG_COUNTER.values())

    return {
        "astronomical_detections": dict(_ASTRONOMICAL_LOG_COUNTER),
        "total_astronomical": total_astronomical,
        "circuit_breaker_stats": get_circuit_breaker_status(),
        "log_every_n_instances": _LOG_EVERY_N_INSTANCES,
        "supported_symbols": list(PRICE_RANGES.keys()),
    }


def validate_conversion_performance(
    test_value: float = 12345000000000000,
) -> dict[str, Any]:
    """
    Test conversion performance and accuracy for debugging.

    Args:
        test_value: Value to test conversion with

    Returns:
        Performance and accuracy statistics
    """
    import time

    # Test basic conversion
    start_time = time.perf_counter()
    try:
        converted = convert_from_18_decimal(test_value, "SUI-PERP", "test_price")
        conversion_time = time.perf_counter() - start_time
        conversion_success = True
        conversion_error = None
    except Exception as e:
        conversion_time = time.perf_counter() - start_time
        converted = None
        conversion_success = False
        conversion_error = str(e)

    # Test detection accuracy
    detection_result = is_likely_18_decimal(test_value)

    # Test sanitization
    sanitized = _sanitize_price_input(test_value)

    # Test validation
    validated = _validate_price_before_conversion(test_value, "SUI-PERP", "test_price")

    return {
        "test_value": test_value,
        "conversion_time_ms": conversion_time * 1000,
        "conversion_success": conversion_success,
        "conversion_error": conversion_error,
        "converted_value": float(converted) if converted else None,
        "detection_result": detection_result,
        "sanitized_value": sanitized,
        "validation_passed": validated,
        "circuit_breaker_open": _is_circuit_breaker_open("SUI-PERP_test_price"),
    }


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
        from bot.data import MarketDataProvider

        # Create a temporary market data provider to get current price
        market_provider = MarketDataProvider(symbol)

        # Get latest OHLCV data
        latest_data = market_provider.get_latest_ohlcv(limit=1)
        if latest_data and len(latest_data) > 0:
            current_price = float(latest_data[-1].close)
            logger.debug(
                "Retrieved real market price for %s: $%s", symbol, current_price
            )
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
            logger.debug(
                "Retrieved real Bluefin price for %s: $%s", symbol, current_price
            )
            return current_price

    except Exception as e:
        logger.debug("Could not fetch real Bluefin price for %s: %s", symbol, e)

    # Return None if no real price could be fetched
    logger.debug("No real market price available for %s, using fallback", symbol)
    return None
