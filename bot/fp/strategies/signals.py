"""
Functional Trading Signals

This module provides functional programming patterns for trading signals,
ensuring immutable and composable signal generation and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from decimal import Decimal


class SignalType(str, Enum):
    """Type of trading signal"""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    CLOSE_ALL = "CLOSE_ALL"


class SignalStrength(str, Enum):
    """Strength of trading signal"""

    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


@dataclass(frozen=True)
class Signal:
    """Immutable trading signal"""

    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    timestamp: datetime
    price: Decimal | None = None
    confidence: float = 0.0
    source: str = "unknown"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate signal data"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")

    @property
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal"""
        return self.signal_type in {SignalType.LONG, SignalType.SHORT}

    @property
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal"""
        return self.signal_type in {
            SignalType.CLOSE_LONG,
            SignalType.CLOSE_SHORT,
            SignalType.CLOSE_ALL,
        }

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish"""
        return self.signal_type == SignalType.LONG

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish"""
        return self.signal_type == SignalType.SHORT

    def with_price(self, price: Decimal) -> Signal:
        """Create new signal with updated price"""
        return Signal(
            symbol=self.symbol,
            signal_type=self.signal_type,
            strength=self.strength,
            timestamp=self.timestamp,
            price=price,
            confidence=self.confidence,
            source=self.source,
            metadata=self.metadata,
        )

    def with_confidence(self, confidence: float) -> Signal:
        """Create new signal with updated confidence"""
        return Signal(
            symbol=self.symbol,
            signal_type=self.signal_type,
            strength=self.strength,
            timestamp=self.timestamp,
            price=self.price,
            confidence=confidence,
            source=self.source,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class SignalContext:
    """Context information for signal generation"""

    market_conditions: dict[str, Any]
    technical_indicators: dict[str, float]
    timestamp: datetime
    volatility: float = 0.0
    volume_profile: dict[str, float] | None = None


# Factory functions for common signals


def create_long_signal(
    symbol: str,
    strength: SignalStrength = SignalStrength.MODERATE,
    confidence: float = 0.5,
    price: Decimal | None = None,
    source: str = "strategy",
) -> Signal:
    """Create a long signal"""
    return Signal(
        symbol=symbol,
        signal_type=SignalType.LONG,
        strength=strength,
        timestamp=datetime.utcnow(),
        price=price,
        confidence=confidence,
        source=source,
    )


def create_short_signal(
    symbol: str,
    strength: SignalStrength = SignalStrength.MODERATE,
    confidence: float = 0.5,
    price: Decimal | None = None,
    source: str = "strategy",
) -> Signal:
    """Create a short signal"""
    return Signal(
        symbol=symbol,
        signal_type=SignalType.SHORT,
        strength=strength,
        timestamp=datetime.utcnow(),
        price=price,
        confidence=confidence,
        source=source,
    )


def create_hold_signal(
    symbol: str,
    confidence: float = 0.5,
    source: str = "strategy",
) -> Signal:
    """Create a hold signal"""
    return Signal(
        symbol=symbol,
        signal_type=SignalType.HOLD,
        strength=SignalStrength.WEAK,
        timestamp=datetime.utcnow(),
        confidence=confidence,
        source=source,
    )


def create_close_all_signal(
    symbol: str,
    strength: SignalStrength = SignalStrength.STRONG,
    confidence: float = 0.8,
    source: str = "risk_management",
) -> Signal:
    """Create a close all positions signal"""
    return Signal(
        symbol=symbol,
        signal_type=SignalType.CLOSE_ALL,
        strength=strength,
        timestamp=datetime.utcnow(),
        confidence=confidence,
        source=source,
    )
