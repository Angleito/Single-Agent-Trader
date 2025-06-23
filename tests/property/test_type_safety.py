"""
Property-based tests for comprehensive type safety validation.

This module uses Hypothesis to generate extensive test cases for:
- Runtime type validation and guards
- Pydantic model constraints
- Service endpoint validation
- Market data type boundaries
- Configuration type safety
- Error type hierarchies

ULTRATHINK DESIGN RATIONALE:
1. Type guards must correctly narrow types at runtime
2. Pydantic models must enforce all constraints
3. NewType definitions should prevent invalid values
4. Protocol implementations must be correctly validated
5. Union types must handle all variants safely
6. Edge cases must be thoroughly covered
"""

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import assume, example, given, settings
from pydantic import ValidationError

# Import types and guards to test
from bot.types.base_types import (
    MarketDataDict,
    Price,
)
from bot.types.exceptions import ValidationError as BotValidationError
from bot.types.guards import (
    ensure_decimal,
    ensure_positive_decimal,
    is_valid_percentage,
    is_valid_price,
    is_valid_quantity,
    is_valid_symbol,
    validate_dict_keys,
    validate_enum_value,
)
from bot.types.market_data import (
    CandleData,
    ConnectionState,
    MarketDataQuality,
    MarketDataStatus,
    OrderBook,
    OrderBookLevel,
    TickerData,
    aggregate_candles,
)
from bot.types.services import (
    ServiceEndpoint,
    ServiceHealth,
    ServiceStatus,
    create_endpoint,
    create_health_status,
    is_docker_service,
    is_healthy_service,
    is_valid_endpoint,
    validate_service_config,
)


# Custom strategies for generating test data
@st.composite
def service_endpoint_strategy(draw: st.DrawFn) -> ServiceEndpoint:
    """Generate valid service endpoints with edge cases."""
    protocol = draw(st.sampled_from(["http", "https", "ws", "wss", "tcp", "grpc"]))
    host = draw(
        st.from_regex(
            r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$",
            fullmatch=True,
        )
    )
    port = draw(st.integers(min_value=1, max_value=65535))
    health_endpoint = draw(
        st.one_of(st.none(), st.from_regex(r"^/[a-zA-Z0-9/_-]*$", fullmatch=True))
    )

    endpoint: ServiceEndpoint = {
        "host": host,
        "port": port,
        "protocol": protocol,  # type: ignore
        "health_endpoint": health_endpoint,
    }

    # Optionally add extra fields
    if draw(st.booleans()):
        endpoint["base_path"] = draw(
            st.from_regex(r"^/[a-zA-Z0-9/_-]*$", fullmatch=True)
        )
    if draw(st.booleans()):
        endpoint["timeout_seconds"] = draw(st.floats(min_value=0.1, max_value=300.0))
    if draw(st.booleans()):
        endpoint["headers"] = draw(
            st.dictionaries(
                st.from_regex(r"^[A-Za-z0-9-]+$", fullmatch=True),
                st.text(min_size=1, max_size=100),
            )
        )

    return endpoint


@st.composite
def invalid_service_endpoint_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate invalid service endpoints to test guards."""
    base = draw(service_endpoint_strategy())

    # Corrupt the endpoint in various ways
    corruption = draw(
        st.sampled_from(
            [
                "missing_host",
                "empty_host",
                "invalid_port",
                "negative_port",
                "invalid_protocol",
                "wrong_type_host",
                "wrong_type_port",
                "negative_timeout",
                "non_dict_headers",
            ]
        )
    )

    invalid: dict[str, Any] = dict(base)

    if corruption == "missing_host":
        del invalid["host"]
    elif corruption == "empty_host":
        invalid["host"] = ""
    elif corruption == "invalid_port":
        invalid["port"] = 70000
    elif corruption == "negative_port":
        invalid["port"] = -1
    elif corruption == "invalid_protocol":
        invalid["protocol"] = "ftp"
    elif corruption == "wrong_type_host":
        invalid["host"] = 12345
    elif corruption == "wrong_type_port":
        invalid["port"] = "8080"
    elif corruption == "negative_timeout":
        invalid["timeout_seconds"] = -5.0
    elif corruption == "non_dict_headers":
        invalid["headers"] = "not-a-dict"

    return invalid


@st.composite
def price_strategy(draw: st.DrawFn) -> Decimal:
    """Generate valid prices with edge cases."""
    # Test various decimal representations
    choice = draw(
        st.sampled_from(["normal", "small", "large", "precise", "scientific"])
    )

    if choice == "normal":
        return Decimal(
            draw(st.decimals(min_value="0.01", max_value="100000", places=2))
        )
    if choice == "small":
        return Decimal(
            draw(st.decimals(min_value="0.00000001", max_value="0.001", places=8))
        )
    if choice == "large":
        return Decimal(
            draw(st.decimals(min_value="10000", max_value="1000000", places=0))
        )
    if choice == "precise":
        # Test maximum precision (8 decimal places)
        return Decimal(
            f"{draw(st.integers(1, 100))}.{draw(st.integers(0, 99999999)):08d}"
        )
    # scientific
    return Decimal(f"{draw(st.integers(1, 9))}e{draw(st.integers(-8, 6))}")


@st.composite
def invalid_price_strategy(draw: st.DrawFn) -> Any:
    """Generate invalid prices to test guards."""
    return draw(
        st.one_of(
            st.none(),
            st.just(0),
            st.just(-1),
            st.just(Decimal("-0.01")),
            st.just(Decimal("Infinity")),
            st.just(Decimal("-Infinity")),
            st.just(Decimal("NaN")),
            st.just("not-a-number"),
            st.just([]),
            st.just({}),
        )
    )


@st.composite
def candle_data_strategy(draw: st.DrawFn) -> CandleData:
    """Generate valid OHLCV candles with proper constraints."""
    # Generate prices ensuring high >= all others and low <= all others
    low = draw(price_strategy())
    high = draw(
        st.decimals(min_value=str(low), max_value=str(low * Decimal(2)), places=8).map(
            Decimal
        )
    )

    # Open and close must be between low and high
    open_price = draw(
        st.decimals(min_value=str(low), max_value=str(high), places=8).map(Decimal)
    )

    close_price = draw(
        st.decimals(min_value=str(low), max_value=str(high), places=8).map(Decimal)
    )

    # Volume can be zero
    volume = draw(
        st.decimals(min_value="0", max_value="1000000", places=8).map(Decimal)
    )

    # Generate timestamp with timezone
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2025, 12, 31, tzinfo=UTC),
            timezones=st.just(UTC),
        )
    )

    candle = CandleData(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close_price,
        volume=volume,
    )

    # Optionally add extra fields
    if draw(st.booleans()):
        candle.trades_count = draw(st.integers(min_value=0, max_value=10000))

    if draw(st.booleans()):
        # VWAP must be between low and high
        candle.vwap = draw(
            st.decimals(min_value=str(low), max_value=str(high), places=8).map(Decimal)
        )

    return candle


@st.composite
def ticker_data_strategy(draw: st.DrawFn) -> TickerData:
    """Generate valid ticker data with bid/ask constraints."""
    product_id = draw(st.from_regex(r"^[A-Z]{3,4}-[A-Z]{3,4}$", fullmatch=True))
    price = draw(price_strategy())

    # Bid must be less than ask
    spread_pct = draw(st.floats(min_value=0.0001, max_value=0.01))  # 0.01% to 1% spread
    bid = price * (Decimal(1) - Decimal(str(spread_pct)))
    ask = price * (Decimal(1) + Decimal(str(spread_pct)))

    # 24h range
    range_pct = draw(st.floats(min_value=0.01, max_value=0.5))  # 1% to 50% range
    low_24h = price * (Decimal(1) - Decimal(str(range_pct)))
    high_24h = price * (Decimal(1) + Decimal(str(range_pct)))

    timestamp = draw(st.datetimes(timezones=st.just(UTC)))

    return TickerData(
        product_id=product_id,
        timestamp=timestamp,
        price=price,
        bid=bid,
        ask=ask,
        low_24h=low_24h,
        high_24h=high_24h,
        volume_24h=draw(
            st.decimals(min_value="0", max_value="10000000", places=2).map(Decimal)
        ),
    )


@st.composite
def symbol_strategy(draw: st.DrawFn) -> str:
    """Generate valid trading symbols."""
    base = draw(st.sampled_from(["BTC", "ETH", "SOL", "USDT", "USDC", "SUI"]))
    quote = draw(st.sampled_from(["USD", "USDT", "USDC", "EUR", "PERP"]))
    return f"{base}-{quote}"


class TestServiceTypeValidation:
    """Test service endpoint and health validation."""

    @given(endpoint=service_endpoint_strategy())
    def test_valid_endpoint_type_guard(self, endpoint: ServiceEndpoint) -> None:
        """Valid endpoints should pass type guard."""
        assert is_valid_endpoint(endpoint)
        # Should be able to create endpoint helper
        created = create_endpoint(
            host=endpoint["host"],
            port=endpoint["port"],
            protocol=endpoint["protocol"],
            health_endpoint=endpoint["health_endpoint"],
        )
        assert created["host"] == endpoint["host"]
        assert created["port"] == endpoint["port"]

    @given(invalid=invalid_service_endpoint_strategy())
    def test_invalid_endpoint_type_guard(self, invalid: dict[str, Any]) -> None:
        """Invalid endpoints should fail type guard."""
        assert not is_valid_endpoint(invalid)

    @given(
        status=st.sampled_from(list(ServiceStatus)),
        error=st.one_of(st.none(), st.text(min_size=1)),
        response_time=st.one_of(st.none(), st.floats(min_value=0, max_value=10000)),
    )
    def test_service_health_creation(
        self, status: ServiceStatus, error: str | None, response_time: float | None
    ) -> None:
        """Service health should be created correctly."""
        health = create_health_status(status, error, response_time)

        assert health["status"] == status.value
        assert isinstance(health["last_check"], float)
        assert health["last_check"] > 0

        if error is not None:
            assert health.get("error") == error
        else:
            assert "error" not in health

        if response_time is not None:
            assert health.get("response_time_ms") == response_time

    @given(
        health=st.fixed_dictionaries(
            {
                "status": st.sampled_from(["healthy", "unhealthy", "degraded"]),
                "last_check": st.floats(min_value=0),
            }
        )
    )
    def test_healthy_service_guard(self, health: dict[str, Any]) -> None:
        """Health guard should only pass for healthy status."""
        if health["status"] == "healthy":
            assert is_healthy_service(health)
        else:
            assert not is_healthy_service(health)

    @given(
        config=st.fixed_dictionaries(
            {
                "name": st.text(min_size=1),
                "enabled": st.booleans(),
                "required": st.booleans(),
                "endpoint": service_endpoint_strategy(),
                "startup_delay": st.one_of(
                    st.none(), st.floats(min_value=0, max_value=60)
                ),
                "dependencies": st.one_of(st.none(), st.lists(st.text(min_size=1))),
            }
        )
    )
    def test_service_config_validation(self, config: dict[str, Any]) -> None:
        """Service config validation should enforce all constraints."""
        # Remove None values
        clean_config = {k: v for k, v in config.items() if v is not None}

        try:
            validated = validate_service_config(clean_config)
            # Should pass validation
            assert validated["name"] == config["name"]
            assert validated["enabled"] == config["enabled"]
            assert is_valid_endpoint(validated["endpoint"])
        except ValueError:
            # Should only fail if config is actually invalid
            pytest.fail("Valid config failed validation")


class TestMarketDataTypeValidation:
    """Test market data type constraints and validation."""

    @given(value=price_strategy())
    def test_valid_price_guard(self, value: Decimal) -> None:
        """Valid prices should pass type guard."""
        assert is_valid_price(value)
        assert is_valid_price(str(value))
        assert is_valid_price(float(value))

    @given(value=invalid_price_strategy())
    def test_invalid_price_guard(self, value: Any) -> None:
        """Invalid prices should fail type guard."""
        assert not is_valid_price(value)

    @given(value=st.decimals(min_value="0", max_value="1000000", places=8))
    def test_valid_quantity_guard(self, value: Decimal) -> None:
        """Valid quantities should pass type guard."""
        assert is_valid_quantity(value)
        assert is_valid_quantity(str(value))

    @given(
        value=st.one_of(
            st.just(-1), st.just(Decimal("-0.1")), st.just("invalid"), st.none()
        )
    )
    def test_invalid_quantity_guard(self, value: Any) -> None:
        """Invalid quantities should fail type guard."""
        assert not is_valid_quantity(value)

    @given(candle=candle_data_strategy())
    def test_candle_data_validation(self, candle: CandleData) -> None:
        """Candle data should maintain OHLCV constraints."""
        # Price relationships
        assert candle.high >= candle.open
        assert candle.high >= candle.close
        assert candle.high >= candle.low
        assert candle.low <= candle.open
        assert candle.low <= candle.close
        assert candle.volume >= 0

        # Timezone must be present
        assert candle.timestamp.tzinfo is not None

        # Methods should work
        assert candle.get_price_range() == candle.high - candle.low
        assert candle.get_body_size() == abs(candle.close - candle.open)

        if candle.close > candle.open:
            assert candle.is_bullish()
        else:
            assert not candle.is_bullish()

        # VWAP constraints
        if candle.vwap is not None:
            assert candle.low <= candle.vwap <= candle.high

    @given(high=price_strategy(), low=price_strategy())
    def test_invalid_candle_constraints(self, high: Decimal, low: Decimal) -> None:
        """Invalid OHLCV relationships should raise validation errors."""
        assume(high < low)  # Ensure invalid relationship

        with pytest.raises(ValidationError):
            CandleData(
                timestamp=datetime.now(UTC),
                open=high,
                high=high,
                low=low,  # low > high - invalid!
                close=high,
                volume=Decimal(100),
            )

    @given(ticker=ticker_data_strategy())
    def test_ticker_data_validation(self, ticker: TickerData) -> None:
        """Ticker data should maintain bid/ask constraints."""
        # Bid/ask relationship
        if ticker.bid is not None and ticker.ask is not None:
            assert ticker.bid < ticker.ask
            spread = ticker.get_spread()
            assert spread is not None
            assert spread > 0
            assert spread == ticker.ask - ticker.bid

        # 24h range
        if ticker.low_24h is not None and ticker.high_24h is not None:
            assert ticker.low_24h <= ticker.high_24h
            # Current price should be reasonably within range
            # (with 2% tolerance for rapid moves)
            tolerance = Decimal("1.02")
            assert ticker.price <= ticker.high_24h * tolerance
            assert ticker.price >= ticker.low_24h / tolerance

    @given(levels=st.integers(min_value=1, max_value=20), base_price=price_strategy())
    def test_order_book_structure(self, levels: int, base_price: Decimal) -> None:
        """Order book should maintain proper structure."""
        # Generate bid levels (descending)
        bids = []
        for i in range(levels):
            price = base_price * (Decimal("0.999") ** i)
            size = Decimal(str(100 - i))
            bids.append(OrderBookLevel(price=price, size=size))

        # Generate ask levels (ascending)
        asks = []
        for i in range(levels):
            price = base_price * (Decimal("1.001") ** i)
            size = Decimal(str(100 - i))
            asks.append(OrderBookLevel(price=price, size=size))

        order_book = OrderBook(
            product_id="BTC-USD", timestamp=datetime.now(UTC), bids=bids, asks=asks
        )

        # Verify structure
        assert order_book.get_best_bid() == bids[0]
        assert order_book.get_best_ask() == asks[0]
        assert order_book.get_spread() == asks[0].price - bids[0].price
        assert order_book.get_mid_price() == (bids[0].price + asks[0].price) / 2

        # Check imbalance calculation
        imbalance = order_book.get_depth_imbalance(5)
        assert imbalance is not None
        assert -1 <= imbalance <= 1

    @given(candles=st.lists(candle_data_strategy(), min_size=2, max_size=100))
    def test_candle_aggregation(self, candles: list[CandleData]) -> None:
        """Candle aggregation should preserve data integrity."""
        # Sort candles by timestamp
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)

        # Aggregate to larger intervals
        aggregated = aggregate_candles(sorted_candles, 5)

        if aggregated:
            for agg_candle in aggregated:
                # Aggregated candle should maintain OHLCV constraints
                assert agg_candle.high >= agg_candle.open
                assert agg_candle.high >= agg_candle.close
                assert agg_candle.high >= agg_candle.low
                assert agg_candle.low <= agg_candle.open
                assert agg_candle.low <= agg_candle.close
                assert agg_candle.volume >= 0


class TestTypeGuardsAndValidators:
    """Test type guard functions and validators."""

    @given(value=st.floats(min_value=0, max_value=100))
    def test_valid_percentage_guard(self, value: float) -> None:
        """Valid percentages should pass type guard."""
        assert is_valid_percentage(value)

    @given(
        value=st.one_of(
            st.floats(min_value=-10, max_value=-0.1),
            st.floats(min_value=100.1, max_value=200),
            st.none(),
            st.text(),
        )
    )
    def test_invalid_percentage_guard(self, value: Any) -> None:
        """Invalid percentages should fail type guard."""
        assert not is_valid_percentage(value)

    @given(symbol=symbol_strategy())
    def test_valid_symbol_guard(self, symbol: str) -> None:
        """Valid symbols should pass type guard."""
        assert is_valid_symbol(symbol)

    @given(
        invalid=st.one_of(
            st.text(max_size=2),  # Too short
            st.from_regex(r"^[A-Z]+$", fullmatch=True),  # No hyphen
            st.just("BTC--USD"),  # Double hyphen
            st.just("btc-usd"),  # Lowercase
            st.just("BTC-"),  # Missing quote
            st.just("-USD"),  # Missing base
            st.none(),
            st.integers(),
        )
    )
    def test_invalid_symbol_guard(self, invalid: Any) -> None:
        """Invalid symbols should fail type guard."""
        assert not is_valid_symbol(invalid)

    @given(
        value=st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(alphabet="0123456789.", min_size=1),
        ),
        field_name=st.text(min_size=1),
    )
    def test_ensure_decimal_conversion(self, value: Any, field_name: str) -> None:
        """Decimal conversion should handle various inputs."""
        try:
            result = ensure_decimal(value, field_name)
            assert isinstance(result, Decimal)
            # Should be able to convert back
            assert str(result) is not None
        except BotValidationError as e:
            # Should have proper error context
            assert e.field_name == field_name
            assert e.validation_rule == "decimal_conversion"

    @given(
        value=st.one_of(
            st.just(0), st.just(-1), st.floats(max_value=0), st.just("invalid")
        ),
        field_name=st.text(min_size=1),
    )
    def test_ensure_positive_decimal_validation(
        self, value: Any, field_name: str
    ) -> None:
        """Positive decimal validation should reject non-positive values."""
        with pytest.raises(BotValidationError) as exc_info:
            ensure_positive_decimal(value, field_name)

        error = exc_info.value
        assert error.field_name == field_name
        assert error.validation_rule in ["decimal_conversion", "positive_value"]

    @given(
        data=st.dictionaries(st.text(min_size=1), st.integers()),
        required=st.sets(st.text(min_size=1), min_size=1),
        optional=st.sets(st.text(min_size=1)),
    )
    def test_dict_key_validation(
        self, data: dict[str, Any], required: set[str], optional: set[str]
    ) -> None:
        """Dictionary key validation should enforce requirements."""
        # Ensure no overlap between required and optional
        optional = optional - required

        try:
            validate_dict_keys(data, required, optional)
            # Should pass if all required keys present
            assert required.issubset(set(data.keys()))
            # And no extra keys beyond required + optional
            assert set(data.keys()).issubset(required | optional)
        except BotValidationError:
            # Should fail if missing required or has extra keys
            data_keys = set(data.keys())
            missing = required - data_keys
            extra = data_keys - (required | optional)
            assert missing or extra

    @given(
        value=st.text(min_size=1),
        enum_values=st.sets(st.text(min_size=1), min_size=1, max_size=10),
        field_name=st.text(min_size=1),
    )
    def test_enum_validation(
        self, value: str, enum_values: set[str], field_name: str
    ) -> None:
        """Enum validation should enforce allowed values."""
        if value in enum_values:
            result = validate_enum_value(value, enum_values, field_name)
            assert result == value
        else:
            with pytest.raises(BotValidationError) as exc_info:
                validate_enum_value(value, enum_values, field_name)

            error = exc_info.value
            assert error.field_name == field_name
            assert error.validation_rule == "enum_value"


class TestConfigurationTypeValidation:
    """Test configuration type validation and boundaries."""

    @pytest.mark.parametrize("leverage", [0, -1, 101, 1000])
    def test_invalid_leverage_bounds(self, leverage: int) -> None:
        """Leverage outside bounds should be rejected."""
        from bot.config import TradingSettings

        with pytest.raises(ValidationError):
            TradingSettings(leverage=leverage)

    @pytest.mark.parametrize("leverage", [1, 5, 10, 50, 100])
    def test_valid_leverage_bounds(self, leverage: int) -> None:
        """Leverage within bounds should be accepted."""
        from bot.config import TradingSettings

        settings = TradingSettings(leverage=leverage)
        assert settings.leverage == leverage

    @given(symbol=st.text(min_size=1))
    def test_symbol_pattern_validation(self, symbol: str) -> None:
        """Trading symbol must match pattern."""
        from bot.config import TradingSettings

        if re.match(r"^[A-Z]+-[A-Z]+$", symbol):
            settings = TradingSettings(symbol=symbol)
            assert settings.symbol == symbol
        else:
            with pytest.raises(ValidationError):
                TradingSettings(symbol=symbol)

    @given(host=st.text(min_size=1), port=st.integers())
    def test_endpoint_port_validation(self, host: str, port: int) -> None:
        """Endpoint port must be in valid range."""
        if 1 <= port <= 65535:
            endpoint = create_endpoint(host, port)
            assert endpoint["port"] == port
            assert is_valid_endpoint(endpoint)
        else:
            # Should not be valid
            invalid_endpoint = {
                "host": host,
                "port": port,
                "protocol": "http",
                "health_endpoint": None,
            }
            assert not is_valid_endpoint(invalid_endpoint)


class TestProtocolImplementations:
    """Test Protocol type checking and implementations."""

    def test_docker_service_protocol_check(self) -> None:
        """DockerService protocol should be correctly identified."""

        class ValidService:
            def __init__(self):
                self.name = "test-service"
                self.endpoint = create_endpoint("localhost", 8080)
                self.required = True

            def health_check(self) -> ServiceHealth:
                return create_health_status(ServiceStatus.HEALTHY)

            async def async_health_check(self) -> ServiceHealth:
                return create_health_status(ServiceStatus.HEALTHY)

            def is_ready(self) -> bool:
                return True

            async def initialize(self) -> bool:
                return True

            async def shutdown(self) -> None:
                pass

        class InvalidService:
            def __init__(self):
                self.name = "test-service"
                # Missing required attributes/methods

        valid = ValidService()
        invalid = InvalidService()

        assert is_docker_service(valid)
        assert not is_docker_service(invalid)
        assert not is_docker_service("not-a-service")
        assert not is_docker_service(None)
        assert not is_docker_service({})


class TestErrorTypeHierarchies:
    """Test error type hierarchies and exception handling."""

    def test_validation_error_context(self) -> None:
        """ValidationError should maintain proper context."""
        error = BotValidationError(
            "Test error",
            field_name="test_field",
            invalid_value=123,
            validation_rule="test_rule",
        )

        assert str(error) == "Test error"
        assert error.field_name == "test_field"
        assert error.invalid_value == 123
        assert error.validation_rule == "test_rule"

    @given(
        message=st.text(min_size=1),
        service_name=st.one_of(st.none(), st.text(min_size=1)),
        error_code=st.one_of(st.none(), st.text(min_size=1)),
    )
    def test_service_error_initialization(
        self, message: str, service_name: str | None, error_code: str | None
    ) -> None:
        """Service errors should store context properly."""
        from bot.types.services import ServiceConnectionError, ServiceError

        error = ServiceConnectionError(
            message,
            service_name=service_name,
            error_code=error_code,
            extra_field="extra_value",
        )

        assert str(error) == message
        assert error.service_name == service_name
        assert error.error_code == error_code
        assert error.details.get("extra_field") == "extra_value"
        assert isinstance(error, ServiceError)
        assert isinstance(error, Exception)


class TestNewTypeConstraints:
    """Test NewType definitions and their constraints."""

    @given(value=st.decimals(min_value="0.00000001", max_value="1000000", places=8))
    def test_price_newtype_constraints(self, value: Decimal) -> None:
        """Price NewType should work with Decimal values."""
        # NewType doesn't enforce runtime checks, but we can test usage
        price: Price = Price(value)  # type: ignore
        assert price > 0
        assert isinstance(price, Decimal)

    @given(value=st.integers(min_value=1))
    def test_timestamp_newtype_constraints(self, value: int) -> None:
        """Timestamp NewType should work with millisecond integers."""
        from bot.types.market_data import Timestamp

        ts: Timestamp = Timestamp(value)  # type: ignore
        assert ts > 0
        assert isinstance(ts, int)


class TestComplexTypeInteractions:
    """Test interactions between multiple type systems."""

    @given(
        candles=st.lists(candle_data_strategy(), min_size=1, max_size=10),
        endpoint=service_endpoint_strategy(),
    )
    def test_mixed_type_validation(
        self, candles: list[CandleData], endpoint: ServiceEndpoint
    ) -> None:
        """Different type systems should work together."""
        # Create a complex data structure mixing types
        market_status = MarketDataStatus(
            product_id="BTC-USD",
            timestamp=datetime.now(UTC),
            connection_state=ConnectionState.CONNECTED,
            data_quality=MarketDataQuality.EXCELLENT,
            messages_received=1000,
            messages_processed=995,
            validation_failures=5,
        )

        # Verify methods work
        assert market_status.is_healthy()
        success_rate = market_status.get_success_rate()
        assert success_rate is not None
        assert 0 <= success_rate <= 100

        # Service endpoint validation
        assert is_valid_endpoint(endpoint)

        # Candle validation
        for candle in candles:
            assert candle.high >= candle.low
            assert candle.volume >= 0

    @given(
        base_data=st.fixed_dictionaries(
            {
                "symbol": symbol_strategy(),
                "timestamp": st.datetimes(timezones=st.just(UTC)),
                "open": price_strategy(),
                "high": price_strategy(),
                "low": price_strategy(),
                "close": price_strategy(),
                "volume": st.decimals(min_value="0", max_value="1000000", places=2).map(
                    Decimal
                ),
            }
        )
    )
    def test_typed_dict_conversion(self, base_data: dict[str, Any]) -> None:
        """TypedDict should work with runtime data."""
        # Fix high/low constraints
        base_data["high"] = max(
            base_data["open"], base_data["high"], base_data["low"], base_data["close"]
        )
        base_data["low"] = min(
            base_data["open"], base_data["high"], base_data["low"], base_data["close"]
        )

        # Create MarketDataDict
        market_data: MarketDataDict = base_data  # type: ignore

        # Should be able to access fields
        assert isinstance(market_data["symbol"], str)
        assert isinstance(market_data["volume"], Decimal)

        # Convert to Pydantic model
        candle = CandleData(**market_data)
        assert candle.symbol == market_data["symbol"]  # type: ignore


# Performance and Edge Case Tests
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    @settings(max_examples=1000, deadline=None)
    @given(st.data())
    def test_high_frequency_type_validation(self, data: st.DataObject) -> None:
        """Type validation should be performant under load."""
        # Generate many prices rapidly
        prices = [data.draw(price_strategy()) for _ in range(100)]

        # All should validate quickly
        for price in prices:
            assert is_valid_price(price)
            assert is_valid_quantity(price)  # Prices are valid quantities

    @given(
        decimals=st.lists(
            st.decimals(min_value="0.00000001", max_value="1000000", places=8),
            min_size=100,
            max_size=100,
        )
    )
    def test_decimal_precision_preservation(self, decimals: list[Decimal]) -> None:
        """Decimal precision should be preserved through conversions."""
        for dec in decimals:
            # Convert to string and back
            str_val = str(dec)
            converted = ensure_decimal(str_val)
            assert converted == dec

            # Ensure precision is maintained
            if "." in str_val:
                decimal_places = len(str_val.split(".")[1])
                assert abs(converted.as_tuple().exponent) <= decimal_places

    @example(value=Decimal("0.00000001"))  # Minimum BTC unit (satoshi)
    @example(value=Decimal(21000000))  # Maximum BTC supply
    @example(value=Decimal("0.99999999"))  # Just under 1
    @example(value=Decimal("1e-8"))  # Scientific notation minimum
    @example(value=Decimal("1e8"))  # Scientific notation large
    @given(value=price_strategy())
    def test_extreme_price_values(self, value: Decimal) -> None:
        """Extreme but valid prices should be handled correctly."""
        assert is_valid_price(value)

        # Should work in calculations
        doubled = value * 2
        assert is_valid_price(doubled)

        halved = value / 2
        assert is_valid_price(halved)

        # Should work in Pydantic models
        ticker = TickerData(
            product_id="BTC-USD", timestamp=datetime.now(UTC), price=value
        )
        assert ticker.price == value

    def test_recursive_validation_performance(self) -> None:
        """Deeply nested structures should validate efficiently."""
        # Create a complex order book with many levels
        levels = 1000
        base_price = Decimal(50000)

        bids = [
            OrderBookLevel(
                price=base_price * Decimal(f"0.{999999 - i:06d}"),
                size=Decimal(str(100 + i)),
            )
            for i in range(levels)
        ]

        asks = [
            OrderBookLevel(
                price=base_price * Decimal(f"1.{i:06d}"), size=Decimal(str(100 + i))
            )
            for i in range(levels)
        ]

        # Should create without performance issues
        order_book = OrderBook(
            product_id="BTC-USD", timestamp=datetime.now(UTC), bids=bids, asks=asks
        )

        # Verify it's valid
        assert len(order_book.bids) == levels
        assert len(order_book.asks) == levels
        assert order_book.get_spread() > 0


if __name__ == "__main__":
    # Run with: pytest tests/property/test_type_safety.py -v
    # For detailed hypothesis statistics: add --hypothesis-show-statistics
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
