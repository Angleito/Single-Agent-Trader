"""
Optimized price precision management for Bluefin trading.

This module provides high-precision price handling with minimal conversions,
reducing precision loss and improving performance in trading calculations.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any, Literal

# Set high precision for all Decimal operations
getcontext().prec = 50

logger = logging.getLogger(__name__)

PriceFormat = Literal["raw", "decimal_18", "display"]


@dataclass
class PriceContext:
    """Context for price operations to track conversion history."""

    symbol: str
    field_name: str
    source_format: PriceFormat
    target_format: PriceFormat
    timestamp: datetime
    conversion_count: int = 0


class PrecisionManager:
    """
    High-precision price manager that minimizes conversions and maintains accuracy.

    Key principles:
    - Keep prices in Decimal format throughout calculation pipelines
    - Convert only at API boundaries (input/output)
    - Track conversion history to detect unnecessary round-trips
    - Provide validation for precision loss detection
    """

    def __init__(self):
        self._conversion_stats = defaultdict(int)
        self._precision_warnings = defaultdict(int)
        self._last_prices = {}  # symbol -> last known good price
        self._conversion_history = defaultdict(list)

        # Performance metrics
        self._conversion_times = []
        self._validation_times = []

        # Precision thresholds
        self.PRECISION_WARNING_THRESHOLD = Decimal("1e-12")  # Warn if loss > this
        self.MAX_CONVERSION_ROUNDS = 3  # Max conversions in pipeline

    def detect_format(self, value: str | int | float | Decimal) -> PriceFormat:
        """
        Intelligently detect price format without conversion.

        Args:
            value: Price value to analyze

        Returns:
            Detected format type
        """
        try:
            if isinstance(value, Decimal):
                # Already in preferred format
                return "raw"

            numeric_value = float(value)

            # Fast detection for 18-decimal format
            if numeric_value > 1e15:
                return "decimal_18"
            if numeric_value > 1e10:
                # Borderline case - use pattern analysis
                if isinstance(value, str):
                    # Count trailing zeros as heuristic
                    clean_str = str(value).replace(".", "").rstrip("0")
                    trailing_zeros = len(str(value).replace(".", "")) - len(clean_str)
                    if trailing_zeros >= 9:
                        return "decimal_18"
                return "raw"
            return "raw"

        except (ValueError, TypeError):
            return "raw"  # Default to raw if uncertain

    def convert_price(
        self,
        value: str | int | float | Decimal,
        context: PriceContext,
        target_format: PriceFormat | None = None,
    ) -> Decimal:
        """
        Convert price with minimal precision loss and conversion tracking.

        Args:
            value: Input price value
            context: Conversion context for tracking
            target_format: Target format (auto-detected if None)

        Returns:
            Converted Decimal price
        """
        start_time = time.perf_counter()

        try:
            # Early return if already Decimal and no conversion needed
            if isinstance(value, Decimal) and (
                target_format is None or target_format == "raw"
            ):
                return value

            # Detect source format if not specified
            source_format = context.source_format or self.detect_format(value)

            # Track conversion attempt
            self._conversion_stats[f"{context.symbol}_{context.field_name}"] += 1

            # Convert to Decimal with appropriate precision
            if source_format == "decimal_18":
                # High precision conversion from 18-decimal
                decimal_value = Decimal(str(value))
                converted = decimal_value / Decimal(
                    1000000000000000000
                )  # Exact division

                # Validate precision loss
                self._validate_precision_loss(value, converted, context)

            else:
                # Direct conversion for raw values
                converted = Decimal(str(value))

            # Update conversion history
            context.conversion_count += 1
            self._conversion_history[context.symbol].append(
                {
                    "timestamp": context.timestamp,
                    "field": context.field_name,
                    "source_format": source_format,
                    "conversion_count": context.conversion_count,
                    "value": float(converted),
                }
            )

            # Warn if too many conversions in pipeline
            if context.conversion_count > self.MAX_CONVERSION_ROUNDS:
                logger.warning(
                    "Excessive conversions detected for %s.%s: %d rounds",
                    context.symbol,
                    context.field_name,
                    context.conversion_count,
                )

            # Update last known price
            self._last_prices[context.symbol] = converted

            return converted

        except (ValueError, TypeError, ArithmeticError) as e:
            logger.error(
                "Precision conversion failed for %s.%s: %s",
                context.symbol,
                context.field_name,
                str(e),
            )
            # Return last known good price as fallback
            return self._get_fallback_price(context.symbol)

        finally:
            # Track performance
            conversion_time = time.perf_counter() - start_time
            self._conversion_times.append(conversion_time)
            if len(self._conversion_times) > 1000:
                self._conversion_times = self._conversion_times[-500:]  # Keep recent

    def _validate_precision_loss(
        self,
        original: str | int | float | Decimal,
        converted: Decimal,
        context: PriceContext,
    ) -> None:
        """Validate and warn about precision loss during conversion."""
        try:
            # Calculate relative precision loss
            original_decimal = Decimal(str(original))
            if original_decimal == 0:
                return

            # For 18-decimal conversion, check round-trip accuracy
            if context.source_format == "decimal_18":
                round_trip = converted * Decimal(1000000000000000000)
                relative_error = abs(round_trip - original_decimal) / original_decimal

                if relative_error > self.PRECISION_WARNING_THRESHOLD:
                    warning_key = f"{context.symbol}_{context.field_name}"
                    self._precision_warnings[warning_key] += 1

                    # Rate-limited logging
                    if self._precision_warnings[warning_key] % 10 == 1:
                        logger.warning(
                            "Precision loss detected for %s.%s: %.2e relative error (instance #%d)",
                            context.symbol,
                            context.field_name,
                            float(relative_error),
                            self._precision_warnings[warning_key],
                        )
        except Exception as e:
            logger.debug("Precision validation failed: %s", str(e))

    def _get_fallback_price(self, symbol: str) -> Decimal:
        """Get fallback price for symbol when conversion fails."""
        if symbol in self._last_prices:
            return self._last_prices[symbol]

        # Symbol-specific reasonable defaults
        defaults = {
            "SUI-PERP": Decimal("2.5"),
            "BTC-PERP": Decimal(50000),
            "ETH-PERP": Decimal(3000),
            "SOL-PERP": Decimal(100),
        }

        return defaults.get(symbol, Decimal("1.0"))

    def batch_convert_ohlcv(
        self, ohlcv_data: dict[str, Any], symbol: str, timestamp: datetime | None = None
    ) -> dict[str, Decimal]:
        """
        Efficiently convert OHLCV data with minimal conversions.

        Args:
            ohlcv_data: Dictionary with OHLCV fields
            symbol: Trading symbol
            timestamp: Conversion timestamp

        Returns:
            Dictionary with converted Decimal values
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        converted = {}
        base_context = PriceContext(
            symbol=symbol,
            field_name="",
            source_format="raw",
            target_format="raw",
            timestamp=timestamp,
        )

        # Batch conversion for efficiency
        price_fields = ["open", "high", "low", "close", "price", "lastPrice"]
        volume_fields = ["volume", "baseVolume", "quoteVolume"]

        for field, value in ohlcv_data.items():
            if value is None:
                continue

            context = PriceContext(
                symbol=symbol,
                field_name=field,
                source_format=self.detect_format(value),
                target_format="raw",
                timestamp=timestamp,
            )

            try:
                if field in price_fields or field in volume_fields:
                    converted[field] = self.convert_price(value, context)
                else:
                    # Non-price fields - direct conversion
                    converted[field] = Decimal(str(value)) if value else Decimal(0)
            except (ValueError, TypeError):
                logger.debug("Failed to convert field %s: %s", field, value)
                converted[field] = Decimal(0)

        return converted

    def format_for_display(self, price: Decimal, symbol: str, decimals: int = 4) -> str:
        """
        Format price for display without affecting calculation precision.

        Args:
            price: Price in calculation format (Decimal)
            symbol: Trading symbol for context
            decimals: Number of decimal places for display

        Returns:
            Formatted price string
        """
        try:
            # Round for display only - don't modify the original
            display_price = price.quantize(
                Decimal("0.1") ** decimals, rounding=ROUND_HALF_UP
            )

            return f"${display_price:,.{decimals}f}"

        except Exception as e:
            logger.debug("Display formatting failed for %s: %s", symbol, str(e))
            return f"${float(price):,.{decimals}f}"

    def format_for_api(
        self,
        price: Decimal,
        symbol: str,
        api_format: Literal["18_decimal", "string", "float"] = "string",
    ) -> str | int | float:
        """
        Format price for API calls with proper precision.

        Args:
            price: Price in calculation format
            symbol: Trading symbol
            api_format: Target API format

        Returns:
            Price in API format
        """
        try:
            if api_format == "18_decimal":
                # Convert to 18-decimal integer for Bluefin API
                return int(price * Decimal(1000000000000000000))
            if api_format == "string":
                # High-precision string representation
                return str(price)
            # float
            return float(price)

        except Exception as e:
            logger.error("API formatting failed for %s: %s", symbol, str(e))
            raise ValueError(f"Cannot format price {price} for API") from e

    def get_precision_stats(self) -> dict[str, Any]:
        """Get precision and performance statistics."""
        avg_conversion_time = (
            sum(self._conversion_times) / len(self._conversion_times)
            if self._conversion_times
            else 0
        )

        return {
            "conversion_counts": dict(self._conversion_stats),
            "precision_warnings": dict(self._precision_warnings),
            "avg_conversion_time_ms": avg_conversion_time * 1000,
            "total_conversions": sum(self._conversion_stats.values()),
            "symbols_tracked": len(self._last_prices),
            "last_prices": {k: float(v) for k, v in self._last_prices.items()},
        }

    def reset_stats(self) -> None:
        """Reset statistics for fresh tracking."""
        self._conversion_stats.clear()
        self._precision_warnings.clear()
        self._conversion_times.clear()
        self._conversion_history.clear()


# Global precision manager instance
_precision_manager = PrecisionManager()


def get_precision_manager() -> PrecisionManager:
    """Get global precision manager instance."""
    return _precision_manager


def convert_price_optimized(
    value: str | int | float | Decimal,
    symbol: str,
    field_name: str,
    source_format: PriceFormat | None = None,
) -> Decimal:
    """
    Optimized price conversion with minimal precision loss.

    Args:
        value: Price value to convert
        symbol: Trading symbol
        field_name: Field name for tracking
        source_format: Source format hint

    Returns:
        Converted Decimal price
    """
    context = PriceContext(
        symbol=symbol,
        field_name=field_name,
        source_format=source_format or "raw",
        target_format="raw",
        timestamp=datetime.now(UTC),
    )

    return _precision_manager.convert_price(value, context)


def batch_convert_market_data(data: dict[str, Any], symbol: str) -> dict[str, Decimal]:
    """
    Efficiently convert market data with batch processing.

    Args:
        data: Market data dictionary
        symbol: Trading symbol

    Returns:
        Converted data with Decimal values
    """
    return _precision_manager.batch_convert_ohlcv(data, symbol)


def format_price_for_display(price: Decimal, symbol: str, decimals: int = 4) -> str:
    """Format price for display without precision loss."""
    return _precision_manager.format_for_display(price, symbol, decimals)


def format_price_for_api(
    price: Decimal,
    symbol: str,
    api_format: Literal["18_decimal", "string", "float"] = "string",
) -> str | int | float:
    """Format price for API calls with proper precision."""
    return _precision_manager.format_for_api(price, symbol, api_format)
