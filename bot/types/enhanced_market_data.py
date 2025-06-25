"""
Enhanced MarketData implementation showing how to integrate new validation.

This module demonstrates how the existing MarketData class can be enhanced
with the comprehensive validation from market_data.py
"""

from decimal import Decimal

from pydantic import Field, model_validator

from bot.trading_types import MarketData as BaseMarketData
from bot.types.market_data import is_valid_price, is_valid_volume

# Type alias to resolve import conflicts
ValidatedMarketData = BaseMarketData


class EnhancedMarketData(BaseMarketData):
    """
    Enhanced MarketData with comprehensive validation.

    This extends the existing MarketData class with additional validation
    while maintaining backward compatibility.
    """

    # Additional optional fields
    trades_count: int | None = Field(
        default=None, ge=0, description="Number of trades in this candle"
    )
    vwap: Decimal | None = Field(
        default=None, gt=0, description="Volume-weighted average price"
    )

    @model_validator(mode="after")
    def validate_ohlcv_relationships(self) -> "EnhancedMarketData":
        """
        Validate OHLCV price relationships.

        Ensures:
        - High is the highest price
        - Low is the lowest price
        - Open and Close are within High/Low range
        - All prices are valid according to market rules
        """
        # Validate individual prices
        for price_name, price_value in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ]:
            if not is_valid_price(price_value):
                raise ValueError(f"Invalid {price_name} price: {price_value}")

        # Validate volume
        if not is_valid_volume(self.volume):
            raise ValueError(f"Invalid volume: {self.volume}")

        # Validate price relationships
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(
                f"High price {self.high} must be >= all other prices. "
                f"Open: {self.open}, Close: {self.close}, Low: {self.low}"
            )

        if self.low > min(self.open, self.close, self.high):
            raise ValueError(
                f"Low price {self.low} must be <= all other prices. "
                f"Open: {self.open}, Close: {self.close}, High: {self.high}"
            )

        # Validate VWAP if provided
        if self.vwap is not None:
            if self.vwap > self.high or self.vwap < self.low:
                raise ValueError(
                    f"VWAP {self.vwap} must be between Low {self.low} and High {self.high}"
                )

        return self

    def get_price_range(self) -> Decimal:
        """Calculate the price range (high - low)."""
        return self.high - self.low

    def get_body_size(self) -> Decimal:
        """Calculate the candle body size (abs(close - open))."""
        return abs(self.close - self.open)

    def is_bullish(self) -> bool:
        """Check if this is a bullish candle."""
        return self.close > self.open

    def is_bearish(self) -> bool:
        """Check if this is a bearish candle."""
        return self.close < self.open

    def is_doji(self, threshold_pct: float = 0.1) -> bool:
        """
        Check if this is a doji candle.

        Args:
            threshold_pct: Body size threshold as percentage of range

        Returns:
            True if candle body is very small relative to range
        """
        price_range = self.get_price_range()
        if price_range == 0:
            return True
        body_ratio = self.get_body_size() / price_range
        return float(body_ratio) < threshold_pct

    def is_hammer(self, body_ratio: float = 0.3, wick_ratio: float = 2.0) -> bool:
        """
        Check if this is a hammer pattern.

        Args:
            body_ratio: Maximum body size as ratio of total range
            wick_ratio: Minimum lower wick to body ratio

        Returns:
            True if candle matches hammer pattern
        """
        price_range = self.get_price_range()
        if price_range == 0:
            return False

        body_size = self.get_body_size()
        body_position = min(self.open, self.close)
        lower_wick = body_position - self.low

        # Body should be small relative to range
        if body_size / price_range > body_ratio:
            return False

        # Lower wick should be long relative to body
        if body_size > 0 and lower_wick / body_size < wick_ratio:
            return False

        # Upper wick should be small
        upper_wick = self.high - max(self.open, self.close)
        return not upper_wick > body_size

    def is_shooting_star(
        self, body_ratio: float = 0.3, wick_ratio: float = 2.0
    ) -> bool:
        """
        Check if this is a shooting star pattern.

        Args:
            body_ratio: Maximum body size as ratio of total range
            wick_ratio: Minimum upper wick to body ratio

        Returns:
            True if candle matches shooting star pattern
        """
        price_range = self.get_price_range()
        if price_range == 0:
            return False

        body_size = self.get_body_size()
        body_top = max(self.open, self.close)
        upper_wick = self.high - body_top

        # Body should be small relative to range
        if body_size / price_range > body_ratio:
            return False

        # Upper wick should be long relative to body
        if body_size > 0 and upper_wick / body_size < wick_ratio:
            return False

        # Lower wick should be small
        lower_wick = min(self.open, self.close) - self.low
        return not lower_wick > body_size

    def get_momentum(self) -> Decimal:
        """
        Calculate price momentum (close - open).

        Returns:
            Positive for bullish momentum, negative for bearish
        """
        return self.close - self.open

    def get_volatility(self) -> Decimal:
        """
        Calculate simple volatility metric (range as % of close).

        Returns:
            Volatility as decimal (0.01 = 1%)
        """
        if self.close == 0:
            return Decimal(0)
        return self.get_price_range() / self.close


def create_validated_market_data(**kwargs) -> EnhancedMarketData:
    """
    Factory function to create validated market data.

    This ensures all market data goes through validation before use.
    """
    return EnhancedMarketData(**kwargs)


# Example usage in existing code:
def migrate_to_enhanced_market_data(old_data: BaseMarketData) -> EnhancedMarketData:
    """
    Migrate existing MarketData to enhanced version with validation.

    Args:
        old_data: Existing MarketData instance

    Returns:
        EnhancedMarketData with validation

    Raises:
        ValueError: If data fails validation
    """
    return EnhancedMarketData(
        symbol=old_data.symbol,
        timestamp=old_data.timestamp,
        open=old_data.open,
        high=old_data.high,
        low=old_data.low,
        close=old_data.close,
        volume=old_data.volume,
    )
