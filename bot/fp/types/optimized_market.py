"""
Optimized Functional Market Data Types

Agent 8: Performance-optimized immutable market data types with __slots__
for memory efficiency and faster access patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class OptimizedOHLCV:
    """Memory-optimized OHLCV candle data with __slots__ for performance.

    Uses __slots__ to reduce memory overhead by ~40% and improve access speed.
    """

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate OHLCV data consistency."""
        if any(price <= 0 for price in [self.open, self.high, self.low, self.close]):
            raise ValueError("All prices must be positive")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"High {self.high} must be >= all other prices")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Low {self.low} must be <= all other prices")

    @property
    def price_range(self) -> Decimal:
        """Calculate the price range of the candle."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (green) candle."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish (red) candle."""
        return self.close < self.open

    @property
    def body_size(self) -> Decimal:
        """Calculate the size of the candle body."""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> Decimal:
        """Calculate the upper shadow length."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> Decimal:
        """Calculate the lower shadow length."""
        return min(self.open, self.close) - self.low

    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / Decimal(3)

    def weighted_close(self) -> Decimal:
        """Calculate weighted close price (OHLC/4)."""
        return (self.open + self.high + self.low + self.close) / Decimal(4)


@dataclass(frozen=True, slots=True)
class OptimizedTrade:
    """Memory-optimized trade data with __slots__."""

    id: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: str  # "BUY" or "SELL"
    symbol: str

    def __post_init__(self) -> None:
        """Validate trade data."""
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")
        if self.side not in ["BUY", "SELL"]:
            raise ValueError(f"Side must be BUY or SELL, got {self.side}")

    @property
    def value(self) -> Decimal:
        """Calculate trade value (price * size)."""
        return self.price * self.size

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side == "BUY"

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side == "SELL"


@dataclass(frozen=True, slots=True)
class OptimizedMarketSnapshot:
    """Memory-optimized market snapshot with __slots__."""

    timestamp: datetime
    symbol: str
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        """Calculate derived fields and validate data."""
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("Bid and ask prices must be positive")
        if self.bid > self.ask:
            raise ValueError(f"Bid {self.bid} cannot be greater than ask {self.ask}")

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid-point price."""
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * Decimal(10000)
        return Decimal(0)


class OptimizedDataFactory:
    """Factory for creating optimized data types with caching and pooling."""

    def __init__(self, enable_caching: bool = True, cache_size: int = 1000):
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._ohlcv_cache: dict[str, OptimizedOHLCV] = {}
        self._trade_cache: dict[str, OptimizedTrade] = {}

    def create_ohlcv(
        self,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        timestamp: datetime | None = None,
    ) -> OptimizedOHLCV:
        """Create optimized OHLCV with optional caching."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        if self.enable_caching:
            # Create cache key
            cache_key = f"{open}_{high}_{low}_{close}_{volume}_{timestamp.isoformat()}"

            if cache_key in self._ohlcv_cache:
                return self._ohlcv_cache[cache_key]

            # Create new instance
            ohlcv = OptimizedOHLCV(
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timestamp=timestamp,
            )

            # Add to cache (with size limit)
            if len(self._ohlcv_cache) < self.cache_size:
                self._ohlcv_cache[cache_key] = ohlcv
            elif len(self._ohlcv_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._ohlcv_cache))
                del self._ohlcv_cache[oldest_key]
                self._ohlcv_cache[cache_key] = ohlcv

            return ohlcv
        return OptimizedOHLCV(
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timestamp=timestamp,
        )

    def create_trade(
        self,
        id: str,
        price: Decimal,
        size: Decimal,
        side: str,
        symbol: str,
        timestamp: datetime | None = None,
    ) -> OptimizedTrade:
        """Create optimized trade with optional caching."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        if self.enable_caching:
            cache_key = f"{id}_{price}_{size}_{side}_{symbol}"

            if cache_key in self._trade_cache:
                return self._trade_cache[cache_key]

            trade = OptimizedTrade(
                id=id,
                timestamp=timestamp,
                price=price,
                size=size,
                side=side,
                symbol=symbol,
            )

            # Add to cache (with size limit)
            if len(self._trade_cache) < self.cache_size:
                self._trade_cache[cache_key] = trade

            return trade
        return OptimizedTrade(
            id=id, timestamp=timestamp, price=price, size=size, side=side, symbol=symbol
        )

    def create_market_snapshot(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        bid: Decimal,
        ask: Decimal,
        timestamp: datetime | None = None,
    ) -> OptimizedMarketSnapshot:
        """Create optimized market snapshot."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        return OptimizedMarketSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
        )

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._ohlcv_cache.clear()
        self._trade_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "ohlcv_cache_size": len(self._ohlcv_cache),
            "trade_cache_size": len(self._trade_cache),
            "cache_enabled": self.enable_caching,
            "max_cache_size": self.cache_size,
        }


class OptimizedDataProcessor:
    """High-performance data processing with vectorized operations."""

    @staticmethod
    def calculate_sma(prices: list[Decimal], period: int) -> list[Decimal]:
        """Calculate Simple Moving Average with optimized algorithm."""
        if len(prices) < period:
            return []

        sma_values = []
        # Calculate first SMA value
        first_sum = sum(prices[:period])
        sma_values.append(first_sum / Decimal(period))

        # Use sliding window for subsequent values
        for i in range(period, len(prices)):
            # Remove oldest price, add newest price
            first_sum = first_sum - prices[i - period] + prices[i]
            sma_values.append(first_sum / Decimal(period))

        return sma_values

    @staticmethod
    def calculate_rsi(prices: list[Decimal], period: int = 14) -> list[Decimal]:
        """Calculate RSI with optimized algorithm."""
        if len(prices) < period + 1:
            return []

        # Calculate price changes
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(delta, Decimal(0)) for delta in deltas]
        losses = [abs(min(delta, Decimal(0))) for delta in deltas]

        rsi_values = []

        # Calculate initial averages
        avg_gain = sum(gains[:period]) / Decimal(period)
        avg_loss = sum(losses[:period]) / Decimal(period)

        # Calculate first RSI value
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi = Decimal(100) - (Decimal(100) / (Decimal(1) + rs))
        else:
            rsi = Decimal(100)
        rsi_values.append(rsi)

        # Calculate subsequent RSI values using Wilder's smoothing
        for i in range(period, len(deltas)):
            avg_gain = ((avg_gain * Decimal(period - 1)) + gains[i]) / Decimal(period)
            avg_loss = ((avg_loss * Decimal(period - 1)) + losses[i]) / Decimal(period)

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = Decimal(100) - (Decimal(100) / (Decimal(1) + rs))
            else:
                rsi = Decimal(100)
            rsi_values.append(rsi)

        return rsi_values

    @staticmethod
    def batch_validate_ohlcv(
        ohlcv_list: list[OptimizedOHLCV],
    ) -> tuple[list[OptimizedOHLCV], list[str]]:
        """Batch validate OHLCV data for performance."""
        valid_data = []
        errors = []

        for i, ohlcv in enumerate(ohlcv_list):
            try:
                # Quick validation checks
                if (
                    ohlcv.high >= max(ohlcv.open, ohlcv.close, ohlcv.low)
                    and ohlcv.low <= min(ohlcv.open, ohlcv.close, ohlcv.high)
                    and all(
                        price > 0
                        for price in [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]
                    )
                    and ohlcv.volume >= 0
                ):
                    valid_data.append(ohlcv)
                else:
                    errors.append(f"Invalid OHLCV at index {i}")
            except Exception as e:
                errors.append(f"Error at index {i}: {e}")

        return valid_data, errors


# Factory instance for global use
default_data_factory = OptimizedDataFactory(enable_caching=True, cache_size=1000)


# Convenience functions
def create_optimized_ohlcv(
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal,
    timestamp: datetime | None = None,
) -> OptimizedOHLCV:
    """Convenience function to create optimized OHLCV."""
    return default_data_factory.create_ohlcv(open, high, low, close, volume, timestamp)


def create_optimized_trade(
    id: str,
    price: Decimal,
    size: Decimal,
    side: str,
    symbol: str,
    timestamp: datetime | None = None,
) -> OptimizedTrade:
    """Convenience function to create optimized trade."""
    return default_data_factory.create_trade(id, price, size, side, symbol, timestamp)


def create_optimized_snapshot(
    symbol: str,
    price: Decimal,
    volume: Decimal,
    bid: Decimal,
    ask: Decimal,
    timestamp: datetime | None = None,
) -> OptimizedMarketSnapshot:
    """Convenience function to create optimized market snapshot."""
    return default_data_factory.create_market_snapshot(
        symbol, price, volume, bid, ask, timestamp
    )
