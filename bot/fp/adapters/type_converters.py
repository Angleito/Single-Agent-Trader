"""
Type Conversion Utilities for Exchange Adapter Migration

This module provides conversion functions between the current trading types
and the functional programming types used in the FP system.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from bot.trading_types import MarketData as CurrentMarketData

# Use standardized imports from bot.types when available
# Fall back to bot.trading_types for types not yet in bot.types
from bot.trading_types import Order as CurrentOrder
from bot.trading_types import Position as CurrentPosition
from bot.trading_types import TradeAction as CurrentTradeAction
from bot.types.market_data import CandleData

from ..types import FPCandle
from ..types.market import ConnectionState, ConnectionStatus, DataQuality, Subscription
from ..types.market import Trade as FPTrade
from ..types.portfolio import Position as FPPosition
from ..types.trading import (
    LimitOrder as FPLimitOrder,
)
from ..types.trading import (
    MarketOrder as FPMarketOrder,
)
from ..types.trading import (
    Order as FPOrder,
)
from ..types.trading import (
    StopOrder as FPStopOrder,
)
from ..types.trading import (
    TradeSignal,
)


class TypeConversionError(Exception):
    """Error raised when type conversion fails."""


# Current Types to FP Types Conversion


def current_order_to_fp_order(order: CurrentOrder) -> FPOrder:
    """Convert current Order to functional Order type."""
    symbol = order.symbol
    side = order.side.lower()  # Convert BUY/SELL to buy/sell

    if side not in ("buy", "sell"):
        raise TypeConversionError(f"Invalid order side: {order.side}")

    side_literal: Literal["buy", "sell"] = side  # type: ignore[assignment]

    if order.type == "MARKET":
        return FPMarketOrder(
            symbol=symbol,
            side=side_literal,
            size=float(order.quantity),
            order_id=order.id,
        )
    if order.type == "LIMIT":
        if order.price is None:
            raise TypeConversionError("Limit order missing price")
        return FPLimitOrder(
            symbol=symbol,
            side=side_literal,
            price=float(order.price),
            size=float(order.quantity),
            order_id=order.id,
        )
    if order.type in ("STOP", "STOP_LIMIT"):
        if order.stop_price is None:
            raise TypeConversionError("Stop order missing stop price")
        return FPStopOrder(
            symbol=symbol,
            side=side_literal,
            stop_price=float(order.stop_price),
            size=float(order.quantity),
            order_id=order.id,
        )
    raise TypeConversionError(f"Unsupported order type: {order.type}")


def current_position_to_fp_position(position: CurrentPosition) -> FPPosition:
    """Convert current Position to functional Position type."""
    # Map LONG/SHORT to functional equivalents
    side = position.side
    if side not in ("LONG", "SHORT"):
        raise TypeConversionError("Cannot convert FLAT position to FP position")

    return FPPosition(
        symbol=position.symbol,
        side=side,
        size=position.size,
        entry_price=position.entry_price or Decimal(0),
        current_price=position.entry_price
        or Decimal(0),  # Will be updated with current market price
    )


def trade_action_to_fp_signal(action: CurrentTradeAction) -> TradeSignal:
    """Convert current TradeAction to functional TradeSignal."""
    from ..types.trading import Hold, Long, Short

    if action.action == "LONG":
        return Long(
            confidence=0.8,  # Default confidence for now
            size=action.size_pct / 100.0,  # Convert percentage to decimal
            reason=action.rationale,
        )
    if action.action == "SHORT":
        return Short(
            confidence=0.8,  # Default confidence for now
            size=action.size_pct / 100.0,  # Convert percentage to decimal
            reason=action.rationale,
        )
    if action.action in ("HOLD", "CLOSE"):
        return Hold(reason=action.rationale)
    raise TypeConversionError(f"Unsupported trade action: {action.action}")


# FP Types to Current Types Conversion


def fp_order_to_current_order(fp_order: FPOrder) -> CurrentOrder:
    """Convert functional Order to current Order type."""
    from bot.trading_types import OrderStatus

    # Convert side back to uppercase
    side: Literal["BUY", "SELL"] = fp_order.side.upper()  # type: ignore[assignment]

    if isinstance(fp_order, FPMarketOrder):
        return CurrentOrder(
            id=fp_order.order_id,
            symbol=fp_order.symbol,
            side=side,
            type="MARKET",
            quantity=Decimal(str(fp_order.size)),
            status=OrderStatus.PENDING,
        )
    if isinstance(fp_order, FPLimitOrder):
        return CurrentOrder(
            id=fp_order.order_id,
            symbol=fp_order.symbol,
            side=side,
            type="LIMIT",
            quantity=Decimal(str(fp_order.size)),
            price=Decimal(str(fp_order.price)),
            status=OrderStatus.PENDING,
        )
    if isinstance(fp_order, FPStopOrder):
        return CurrentOrder(
            id=fp_order.order_id,
            symbol=fp_order.symbol,
            side=side,
            type="STOP",
            quantity=Decimal(str(fp_order.size)),
            stop_price=Decimal(str(fp_order.stop_price)),
            status=OrderStatus.PENDING,
        )
    raise TypeConversionError(f"Unsupported FP order type: {type(fp_order)}")


def fp_position_to_current_position(fp_position: FPPosition) -> CurrentPosition:
    """Convert functional Position to current Position type."""

    return CurrentPosition(
        symbol=fp_position.symbol,
        side=fp_position.side,  # type: ignore[arg-type]
        size=fp_position.size,
        entry_price=fp_position.entry_price,
        unrealized_pnl=fp_position.unrealized_pnl,
        timestamp=datetime.now(UTC),
    )


# Account Balance Conversion


def create_fp_account_balance(balance: Decimal) -> dict[str, float]:
    """Create functional AccountBalance from Decimal balance."""
    return {
        "cash": float(balance),
        "total": float(balance),
        "margin_used": 0.0,
        "available_margin": float(balance),
    }


# Order Result Conversion


def create_order_result(
    order: CurrentOrder, success: bool = True
) -> dict[str, str | bool | float]:
    """Create functional OrderResult from current Order."""
    return {
        "order_id": order.id,
        "symbol": order.symbol,
        "side": order.side.lower(),
        "type": order.type.lower(),
        "quantity": float(order.quantity),
        "price": float(order.price) if order.price else None,
        "status": "filled" if success else "rejected",
        "success": success,
    }


# Market Data Conversion Functions


def current_market_data_to_fp_candle(market_data: CurrentMarketData) -> FPCandle:
    """Convert current MarketData to functional Candle."""
    return FPCandle(
        symbol=market_data.symbol,
        timestamp=market_data.timestamp,
        open=float(market_data.open),
        high=float(market_data.high),
        low=float(market_data.low),
        close=float(market_data.close),
        volume=float(market_data.volume),
    )


def fp_candle_to_current_market_data(candle: FPCandle) -> CurrentMarketData:
    """Convert functional Candle to current MarketData."""
    return CurrentMarketData(
        symbol=candle.symbol,
        timestamp=candle.timestamp,
        open=Decimal(str(candle.open)),
        high=Decimal(str(candle.high)),
        low=Decimal(str(candle.low)),
        close=Decimal(str(candle.close)),
        volume=Decimal(str(candle.volume)),
    )


def fp_candle_to_simple_market_data(candle: FPCandle) -> "SimpleMarketData":
    """Convert functional Candle to simple MarketData (from fp.types.market)."""
    from bot.fp.types.market import MarketData as SimpleMarketData

    return SimpleMarketData.from_ohlcv(
        symbol=candle.symbol,
        timestamp=candle.timestamp,
        open=Decimal(str(candle.open)),
        high=Decimal(str(candle.high)),
        low=Decimal(str(candle.low)),
        close=Decimal(str(candle.close)),
        volume=Decimal(str(candle.volume)),
    )


def pydantic_candle_to_fp_candle(candle_data: CandleData) -> FPCandle:
    """Convert Pydantic CandleData to functional Candle."""
    return FPCandle(
        symbol=candle_data.symbol,
        timestamp=candle_data.timestamp,
        open=float(candle_data.open),
        high=float(candle_data.high),
        low=float(candle_data.low),
        close=float(candle_data.close),
        volume=float(candle_data.volume),
    )


def fp_candle_to_pydantic_candle(candle: FPCandle) -> CandleData:
    """Convert functional Candle to Pydantic CandleData."""
    return CandleData(
        symbol=candle.symbol,
        timestamp=candle.timestamp,
        open=Decimal(str(candle.open)),
        high=Decimal(str(candle.high)),
        low=Decimal(str(candle.low)),
        close=Decimal(str(candle.close)),
        volume=Decimal(str(candle.volume)),
    )


# List conversion utilities


def convert_candle_list_to_fp(candles: list[CurrentMarketData]) -> list[FPCandle]:
    """Convert list of current MarketData to functional Candles."""
    return [current_market_data_to_fp_candle(candle) for candle in candles]


def convert_fp_candle_list_to_current(
    candles: list[FPCandle],
) -> list[CurrentMarketData]:
    """Convert list of functional Candles to current MarketData."""
    return [fp_candle_to_current_market_data(candle) for candle in candles]


def convert_pydantic_candle_list_to_fp(candles: list[CandleData]) -> list[FPCandle]:
    """Convert list of Pydantic CandleData to functional Candles."""
    return [pydantic_candle_to_fp_candle(candle) for candle in candles]


def convert_fp_candle_list_to_pydantic(candles: list[FPCandle]) -> list[CandleData]:
    """Convert list of functional Candles to Pydantic CandleData."""
    return [fp_candle_to_pydantic_candle(candle) for candle in candles]


# Helper functions for market data processing


def merge_candle_data(
    fp_candles: list[FPCandle], current_candles: list[CurrentMarketData]
) -> list[FPCandle]:
    """Merge functional and current candle data into functional format."""
    # Convert current candles to functional format
    converted_current = convert_candle_list_to_fp(current_candles)

    # Combine and sort by timestamp
    all_candles = fp_candles + converted_current
    all_candles.sort(key=lambda c: c.timestamp)

    # Remove duplicates based on timestamp
    unique_candles = []
    seen_timestamps = set()

    for candle in all_candles:
        if candle.timestamp not in seen_timestamps:
            unique_candles.append(candle)
            seen_timestamps.add(candle.timestamp)

    return unique_candles


def extract_market_data_summary(candles: list[FPCandle]) -> dict[str, Any]:
    """Extract summary statistics from functional candle data."""
    if not candles:
        return {}

    prices = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]

    return {
        "symbol": candles[0].symbol,
        "start_time": min(candle.timestamp for candle in candles),
        "end_time": max(candle.timestamp for candle in candles),
        "candle_count": len(candles),
        "price_range": {"min": min(prices), "max": max(prices)},
        "total_volume": sum(volumes),
        "average_volume": sum(volumes) / len(volumes),
        "bullish_candles": sum(1 for candle in candles if candle.is_bullish),
        "bearish_candles": sum(1 for candle in candles if candle.is_bearish),
    }


def create_subscription_from_config(
    symbol: str, channels: list[str], exchange: str | None = None
) -> Subscription:
    """Create functional Subscription from configuration."""
    return Subscription(
        symbol=symbol,
        channels=channels,
        active=True,
        created_at=datetime.now(UTC),
        exchange=exchange,
        subscription_id=f"{symbol}_{'_'.join(channels)}_{datetime.now(UTC).timestamp()}",
    )


def update_connection_state(
    current_state: ConnectionState,
    new_status: str | None = None,
    error: str | None = None,
    message_received: bool = False,
) -> ConnectionState:
    """Create updated connection state with new information."""
    status = current_state.status
    if new_status:
        try:
            status = ConnectionStatus(new_status)
        except ValueError:
            status = current_state.status

    return ConnectionState(
        status=status,
        url=current_state.url,
        reconnect_attempts=current_state.reconnect_attempts + (1 if error else 0),
        last_error=error or current_state.last_error,
        connected_at=(
            current_state.connected_at
            if status != ConnectionStatus.CONNECTED
            else datetime.now(UTC)
        ),
        last_message_at=(
            datetime.now(UTC) if message_received else current_state.last_message_at
        ),
    )


def update_data_quality(
    current_quality: DataQuality,
    new_message: bool = False,
    processed: bool = False,
    validation_failed: bool = False,
    latency_ms: float | None = None,
) -> DataQuality:
    """Create updated data quality metrics."""
    return DataQuality(
        timestamp=datetime.now(UTC),
        messages_received=current_quality.messages_received + (1 if new_message else 0),
        messages_processed=current_quality.messages_processed + (1 if processed else 0),
        validation_failures=current_quality.validation_failures
        + (1 if validation_failed else 0),
        average_latency_ms=latency_ms or current_quality.average_latency_ms,
    )


# Validation utilities


def validate_functional_candle(candle: FPCandle) -> bool:
    """Validate functional Candle data integrity."""
    try:
        # Basic validation is handled by __post_init__
        # Additional business logic validation can be added here
        if candle.price_range < 0:
            return False
        if candle.body_size < 0:
            return False
        return True
    except Exception:
        return False


def validate_functional_trade(trade: FPTrade) -> bool:
    """Validate functional Trade data integrity."""
    try:
        # Basic validation is handled by __post_init__
        # Additional business logic validation can be added here
        if trade.value <= 0:
            return False
        return True
    except Exception:
        return False


def validate_connection_health(connection: ConnectionState) -> bool:
    """Validate connection health."""
    return connection.is_healthy()


def validate_data_quality(quality: DataQuality, min_success_rate: float = 95.0) -> bool:
    """Validate data quality meets requirements."""
    return quality.success_rate >= min_success_rate and quality.error_rate <= 5.0


# Convenience functions for __init__.py imports
def convert_market_data(market_data: CurrentMarketData) -> FPCandle:
    """Convert market data to functional candle - convenience function."""
    return current_market_data_to_fp_candle(market_data)


def convert_position(position: CurrentPosition) -> FPPosition:
    """Convert position to functional position - convenience function."""
    return current_position_to_fp_position(position)


def convert_trade_action(action: CurrentTradeAction) -> TradeSignal:
    """Convert trade action to functional signal - convenience function."""
    return trade_action_to_fp_signal(action)


# ============================================================================
# MISSING FACTORY FUNCTIONS FOR MARKET DATA ADAPTER
# ============================================================================

from ..types.market import (
    ConnectionState,
    DataQuality,
    MarketDataStream,
    OrderBookMessage,
    RealtimeUpdate,
    TickerMessage,
    TradeMessage,
)


def create_connection_state(url: str, status: str = "CONNECTING") -> ConnectionState:
    """Create initial connection state."""
    try:
        connection_status = ConnectionStatus(status)
    except ValueError:
        connection_status = ConnectionStatus.CONNECTING

    return ConnectionState(
        status=connection_status,
        url=url,
        reconnect_attempts=0,
        last_error=None,
        connected_at=None,
        last_message_at=None,
    )


def create_data_quality(
    messages_received: int = 0,
    messages_processed: int = 0,
    validation_failures: int = 0,
    average_latency_ms: float | None = None,
) -> DataQuality:
    """Create initial data quality metrics."""
    return DataQuality(
        timestamp=datetime.now(UTC),
        messages_received=messages_received,
        messages_processed=messages_processed,
        validation_failures=validation_failures,
        average_latency_ms=average_latency_ms,
    )


def create_market_data_stream(symbol: str, exchanges: list[str]) -> MarketDataStream:
    """Create initial market data stream."""
    # Create initial connection states for all exchanges
    connection_states = {
        exchange: create_connection_state(f"wss://{exchange}.example.com")
        for exchange in exchanges
    }

    # Create initial data quality
    data_quality = create_data_quality()

    return MarketDataStream(
        symbol=symbol,
        exchanges=exchanges,
        connection_states=connection_states,
        data_quality=data_quality,
        active=True,
    )


def create_realtime_update(
    symbol: str,
    update_type: str,
    data: dict[str, Any],
    exchange: str | None = None,
    latency_ms: float | None = None,
) -> RealtimeUpdate:
    """Create real-time update from data."""
    return RealtimeUpdate(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        update_type=update_type,
        data=data,
        exchange=exchange,
        latency_ms=latency_ms,
    )


def create_ticker_message_from_data(
    message: dict[str, Any], symbol: str
) -> TickerMessage:
    """Create ticker message from WebSocket data."""
    # Extract price from various possible fields
    price = None
    for price_field in ["price", "last", "last_price", "close"]:
        if price_field in message:
            try:
                price = Decimal(str(message[price_field]))
                break
            except (ValueError, TypeError):
                continue

    if price is None:
        raise TypeConversionError(f"No valid price found in ticker message: {message}")

    # Extract volume (optional)
    volume_24h = None
    for volume_field in ["volume", "volume_24h", "base_volume", "volume_24hr"]:
        if volume_field in message:
            try:
                volume_24h = Decimal(str(message[volume_field]))
                break
            except (ValueError, TypeError):
                continue

    # Extract channel
    channel = message.get("channel", message.get("type", "ticker"))

    # Extract message ID
    message_id = message.get("id", message.get("message_id"))

    return TickerMessage(
        channel=channel,
        timestamp=datetime.now(UTC),
        data=message,
        price=price,
        volume_24h=volume_24h,
        message_id=message_id,
    )


def create_trade_message_from_data(
    message: dict[str, Any], symbol: str
) -> TradeMessage:
    """Create trade message from WebSocket data."""
    # Extract trade ID
    trade_id = message.get("trade_id", message.get("id", message.get("tradeId")))

    # Extract price
    price = None
    for price_field in ["price", "last", "last_price"]:
        if price_field in message:
            try:
                price = Decimal(str(message[price_field]))
                break
            except (ValueError, TypeError):
                continue

    # Extract size
    size = None
    for size_field in ["size", "amount", "quantity", "vol"]:
        if size_field in message:
            try:
                size = Decimal(str(message[size_field]))
                break
            except (ValueError, TypeError):
                continue

    # Extract side
    side = None
    for side_field in ["side", "type", "taker_side"]:
        if side_field in message:
            raw_side = str(message[side_field]).upper()
            if raw_side in ["BUY", "SELL", "B", "S"]:
                side = "BUY" if raw_side in ["BUY", "B"] else "SELL"
                break

    # Extract channel
    channel = message.get("channel", message.get("type", "trades"))

    # Extract message ID
    message_id = message.get("id", message.get("message_id"))

    return TradeMessage(
        channel=channel,
        timestamp=datetime.now(UTC),
        data=message,
        message_id=message_id,
        trade_id=trade_id,
        price=price,
        size=size,
        side=side,
    )


def create_orderbook_message_from_data(
    message: dict[str, Any], symbol: str
) -> OrderBookMessage:
    """Create orderbook message from WebSocket data."""
    # Extract bids and asks
    bids = None
    asks = None

    # Try to extract bids
    if "bids" in message:
        try:
            bids = [
                (Decimal(str(bid[0])), Decimal(str(bid[1])))
                for bid in message["bids"]
                if len(bid) >= 2
            ]
        except (ValueError, TypeError, IndexError):
            bids = None

    # Try to extract asks
    if "asks" in message:
        try:
            asks = [
                (Decimal(str(ask[0])), Decimal(str(ask[1])))
                for ask in message["asks"]
                if len(ask) >= 2
            ]
        except (ValueError, TypeError, IndexError):
            asks = None

    # Extract channel
    channel = message.get("channel", message.get("type", "orderbook"))

    # Extract message ID
    message_id = message.get("id", message.get("message_id"))

    return OrderBookMessage(
        channel=channel,
        timestamp=datetime.now(UTC),
        data=message,
        message_id=message_id,
        bids=bids,
        asks=asks,
    )
