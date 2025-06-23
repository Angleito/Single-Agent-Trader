"""
Functional validation module for type safety and data conversion.

This module provides:
- Pydantic validators for market data, orders, and configuration
- Converters from legacy types to FP types
- Result monad for validation outcomes
- Schema definitions for external data
"""

from datetime import datetime
from decimal import Decimal
from typing import Union, cast

from pydantic import BaseModel, Field, field_validator, model_validator
from returns.result import Failure, Result, Success

from bot.fp.core.types import (
    Config,
    Exchange,
    MarketCondition,
    MarketSnapshot,
    Order,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    Risk,
    Symbol,
    Timestamp,
    TradingMode,
    TradingParameters,
    ValidationError,
)


# Validation Error Types
class FieldError(BaseModel):
    """Individual field validation error."""

    field: str
    message: str
    value: str | None = None


class SchemaError(BaseModel):
    """Schema validation error."""

    schema: str
    errors: list[FieldError]


class ConversionError(BaseModel):
    """Type conversion error."""

    source_type: str
    target_type: str
    message: str


# Union type for all validation errors
ValidatorError = Union[ValidationError, SchemaError, ConversionError]


# Market Data Validators
class MarketDataValidator(BaseModel):
    """Validator for market data inputs."""

    timestamp: float = Field(..., gt=0)
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    bid: float | None = Field(None, gt=0)
    ask: float | None = Field(None, gt=0)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        """Ensure timestamp is reasonable."""
        now = datetime.now().timestamp()
        # Allow timestamps up to 1 year in the past and 1 day in the future
        min_time = now - (365 * 24 * 60 * 60)
        max_time = now + (24 * 60 * 60)

        if v < min_time or v > max_time:
            raise ValueError(f"Timestamp {v} is outside reasonable range")
        return v

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> "MarketDataValidator":
        """Ensure OHLC relationships are valid."""
        if self.high < self.low:
            raise ValueError("High price cannot be less than low price")
        if self.high < self.open or self.high < self.close:
            raise ValueError("High price must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("Low price must be <= open and close")

        # Validate spread if bid/ask present
        if self.bid and self.ask and self.bid >= self.ask:
            raise ValueError("Bid must be less than ask")

        return self


# Order Validators
class OrderValidator(BaseModel):
    """Validator for order parameters."""

    symbol: str = Field(..., min_length=1, max_length=20)
    side: str = Field(..., pattern="^(BUY|SELL)$")
    order_type: str = Field(..., pattern="^(MARKET|LIMIT|STOP|STOP_LIMIT)$")
    quantity: float = Field(..., gt=0)
    price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    leverage: int | None = Field(None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_order_type_requirements(self) -> "OrderValidator":
        """Ensure required fields are present for order types."""
        if self.order_type in ["LIMIT", "STOP_LIMIT"] and self.price is None:
            raise ValueError(f"{self.order_type} orders require a price")
        if self.order_type in ["STOP", "STOP_LIMIT"] and self.stop_price is None:
            raise ValueError(f"{self.order_type} orders require a stop price")
        return self


# Config Validators
class ConfigValidator(BaseModel):
    """Validator for configuration parameters."""

    mode: str = Field(..., pattern="^(paper|live|backtest)$")
    symbol: str = Field(..., min_length=1, max_length=20)
    interval: str = Field(..., pattern="^(1s|5s|15s|30s|1m|3m|5m|15m|30m|1h|4h|1d)$")
    leverage: int = Field(..., ge=1, le=100)
    max_position_size: float = Field(..., gt=0, le=1.0)
    max_drawdown: float = Field(..., gt=0, le=1.0)
    stop_loss_pct: float = Field(..., gt=0, le=0.5)
    take_profit_pct: float = Field(..., gt=0, le=2.0)

    @model_validator(mode="after")
    def validate_risk_parameters(self) -> "ConfigValidator":
        """Ensure risk parameters are sensible."""
        if self.stop_loss_pct >= self.take_profit_pct:
            raise ValueError("Stop loss percentage should be less than take profit")
        if self.max_drawdown < self.stop_loss_pct:
            raise ValueError("Max drawdown should be >= stop loss percentage")
        return self


# Conversion Functions
def convert_legacy_market_state(data: dict) -> Result[MarketSnapshot, ValidatorError]:
    """Convert legacy market state dict to MarketSnapshot."""
    try:
        # Validate input data
        validator = MarketDataValidator(**data)

        # Convert to FP types
        snapshot = MarketSnapshot(
            timestamp=Timestamp(
                int(validator.timestamp * 1000)
            ),  # Convert to milliseconds
            price=Price(Decimal(str(validator.close))),
            volume=Quantity(Decimal(str(validator.volume))),
            bid=Price(Decimal(str(validator.bid))) if validator.bid else None,
            ask=Price(Decimal(str(validator.ask))) if validator.ask else None,
            high_24h=Price(Decimal(str(validator.high))),
            low_24h=Price(Decimal(str(validator.low))),
            vwap_24h=Price(
                Decimal(str((validator.high + validator.low) / 2))
            ),  # Simplified
            conditions={
                MarketCondition.NORMAL: 0.7,  # Default conditions
                MarketCondition.VOLATILE: 0.2,
                MarketCondition.TRENDING: 0.1,
            },
        )

        return Success(snapshot)

    except Exception as e:
        error = ConversionError(
            source_type="legacy_market_state",
            target_type="MarketSnapshot",
            message=str(e),
        )
        return Failure(cast("ValidatorError", error))


def convert_legacy_order(data: dict) -> Result[Order, ValidatorError]:
    """Convert legacy order dict to Order type."""
    try:
        # Validate input data
        validator = OrderValidator(**data)

        # Map string values to enums
        side_map = {"BUY": OrderSide.BUY, "SELL": OrderSide.SELL}
        type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP,
            "STOP_LIMIT": OrderType.STOP_LIMIT,
        }

        # Create order
        order = Order(
            id=OrderId(data.get("id", f"order_{datetime.now().timestamp()}")),
            symbol=Symbol(validator.symbol),
            side=side_map[validator.side],
            type=type_map[validator.order_type],
            quantity=Quantity(Decimal(str(validator.quantity))),
            price=Price(Decimal(str(validator.price))) if validator.price else None,
            stop_price=(
                Price(Decimal(str(validator.stop_price)))
                if validator.stop_price
                else None
            ),
            status=OrderStatus.PENDING,
            timestamp=Timestamp(int(datetime.now().timestamp() * 1000)),
            filled_quantity=Quantity(Decimal(0)),
            average_price=None,
            fee=Quantity(Decimal(0)),
        )

        return Success(order)

    except Exception as e:
        error = ConversionError(
            source_type="legacy_order", target_type="Order", message=str(e)
        )
        return Failure(cast("ValidatorError", error))


def convert_legacy_config(data: dict) -> Result[Config, ValidatorError]:
    """Convert legacy config dict to Config type."""
    try:
        # Validate input data
        validator = ConfigValidator(**data)

        # Map mode strings
        mode_map = {
            "paper": TradingMode.PAPER,
            "live": TradingMode.LIVE,
            "backtest": TradingMode.BACKTEST,
        }

        # Create config
        config = Config(
            mode=mode_map[validator.mode],
            exchange=Exchange.COINBASE,  # Default, should be in data
            symbol=Symbol(validator.symbol),
            parameters=TradingParameters(
                interval=validator.interval,
                lookback_periods=20,  # Default
                indicator_config={
                    "rsi_period": 14,
                    "ema_fast": 12,
                    "ema_slow": 26,
                },
            ),
            risk=Risk(
                max_position_size=Decimal(str(validator.max_position_size)),
                max_leverage=validator.leverage,
                max_drawdown=Decimal(str(validator.max_drawdown)),
                stop_loss_pct=Decimal(str(validator.stop_loss_pct)),
                take_profit_pct=Decimal(str(validator.take_profit_pct)),
                position_sizing_method="fixed",
            ),
        )

        return Success(config)

    except Exception as e:
        error = ConversionError(
            source_type="legacy_config", target_type="Config", message=str(e)
        )
        return Failure(cast("ValidatorError", error))


# Schema Definitions
MARKET_DATA_SCHEMA = {
    "type": "object",
    "required": ["timestamp", "open", "high", "low", "close", "volume"],
    "properties": {
        "timestamp": {"type": "number", "minimum": 0},
        "open": {"type": "number", "exclusiveMinimum": 0},
        "high": {"type": "number", "exclusiveMinimum": 0},
        "low": {"type": "number", "exclusiveMinimum": 0},
        "close": {"type": "number", "exclusiveMinimum": 0},
        "volume": {"type": "number", "minimum": 0},
        "bid": {"type": "number", "exclusiveMinimum": 0},
        "ask": {"type": "number", "exclusiveMinimum": 0},
    },
}

ORDER_SCHEMA = {
    "type": "object",
    "required": ["symbol", "side", "order_type", "quantity"],
    "properties": {
        "symbol": {"type": "string", "minLength": 1, "maxLength": 20},
        "side": {"type": "string", "enum": ["BUY", "SELL"]},
        "order_type": {
            "type": "string",
            "enum": ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"],
        },
        "quantity": {"type": "number", "exclusiveMinimum": 0},
        "price": {"type": "number", "exclusiveMinimum": 0},
        "stop_price": {"type": "number", "exclusiveMinimum": 0},
        "leverage": {"type": "integer", "minimum": 1, "maximum": 100},
    },
}

CONFIG_SCHEMA = {
    "type": "object",
    "required": [
        "mode",
        "symbol",
        "interval",
        "leverage",
        "max_position_size",
        "max_drawdown",
        "stop_loss_pct",
        "take_profit_pct",
    ],
    "properties": {
        "mode": {"type": "string", "enum": ["paper", "live", "backtest"]},
        "symbol": {"type": "string", "minLength": 1, "maxLength": 20},
        "interval": {
            "type": "string",
            "enum": [
                "1s",
                "5s",
                "15s",
                "30s",
                "1m",
                "3m",
                "5m",
                "15m",
                "30m",
                "1h",
                "4h",
                "1d",
            ],
        },
        "leverage": {"type": "integer", "minimum": 1, "maximum": 100},
        "max_position_size": {"type": "number", "exclusiveMinimum": 0, "maximum": 1.0},
        "max_drawdown": {"type": "number", "exclusiveMinimum": 0, "maximum": 1.0},
        "stop_loss_pct": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5},
        "take_profit_pct": {"type": "number", "exclusiveMinimum": 0, "maximum": 2.0},
    },
}


# Validation Helper Functions
def validate_market_data(data: dict) -> Result[MarketSnapshot, ValidatorError]:
    """Validate and convert market data to MarketSnapshot."""
    return convert_legacy_market_state(data)


def validate_order(data: dict) -> Result[Order, ValidatorError]:
    """Validate and convert order data to Order."""
    return convert_legacy_order(data)


def validate_config(data: dict) -> Result[Config, ValidatorError]:
    """Validate and convert config data to Config."""
    return convert_legacy_config(data)


# Batch validation
def validate_batch(
    items: list[dict], validator_fn
) -> list[Result[any, ValidatorError]]:
    """Validate a batch of items using the specified validator function."""
    return [validator_fn(item) for item in items]
