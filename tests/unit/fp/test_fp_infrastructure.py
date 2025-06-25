"""
FP Test Infrastructure

Comprehensive test utilities, base classes, and helpers for functional programming
testing patterns. This module provides the core infrastructure needed for testing
FP components and migrating imperative tests to functional patterns.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Generic, TypeVar
from unittest.mock import AsyncMock, Mock

from hypothesis import strategies as st

# FP type imports with safe fallbacks
T = TypeVar("T")
E = TypeVar("E")

try:
    from bot.fp.types.effects import IO, Err, Maybe, Nothing, Ok, Result, Some
except ImportError:
    # Fallback implementations for testing when FP types aren't available
    class Result(Generic[T, E]):
        pass

    class Ok(Result[T, E]):
        def __init__(self, value: T):
            self.value = value

        def is_ok(self) -> bool:
            return True

        def is_err(self) -> bool:
            return False

        def unwrap(self) -> T:
            return self.value

    class Err(Result[T, E]):
        def __init__(self, error: E):
            self.error = error

        def is_ok(self) -> bool:
            return False

        def is_err(self) -> bool:
            return True

    class Maybe(Generic[T]):
        pass

    class Some(Maybe[T]):
        def __init__(self, value: T):
            self.value = value

        def is_some(self) -> bool:
            return True

        def is_nothing(self) -> bool:
            return False

        def unwrap(self) -> T:
            return self.value

    class Nothing(Maybe[T]):
        def is_some(self) -> bool:
            return False

        def is_nothing(self) -> bool:
            return True

    class IO(Generic[T]):
        def __init__(self, computation: Callable[[], T]):
            self._computation = computation

        @classmethod
        def pure(cls, value: T) -> "IO[T]":
            return cls(lambda: value)

        def run(self) -> T:
            return self._computation()


# =============================================================================
# FP TEST BASE CLASSES
# =============================================================================


class FPTestCase(ABC):
    """Base class for functional programming tests."""

    @abstractmethod
    def setup_fp_fixtures(self) -> dict[str, Any]:
        """Setup FP fixtures for the test."""

    def assert_result_ok(self, result: Result[T, E], expected_value: T = None) -> T:
        """Assert Result is Ok and optionally check value."""
        assert result.is_ok(), (
            f"Expected Ok, got Err: {result.error if hasattr(result, 'error') else result}"
        )
        value = result.unwrap()
        if expected_value is not None:
            assert value == expected_value
        return value

    def assert_result_err(self, result: Result[T, E], expected_error: E = None) -> E:
        """Assert Result is Err and optionally check error."""
        assert result.is_err(), (
            f"Expected Err, got Ok: {result.unwrap() if hasattr(result, 'unwrap') else result}"
        )
        error = result.error
        if expected_error is not None:
            assert error == expected_error
        return error

    def assert_maybe_some(self, maybe: Maybe[T], expected_value: T = None) -> T:
        """Assert Maybe is Some and optionally check value."""
        assert maybe.is_some(), "Expected Some, got Nothing"
        value = maybe.unwrap()
        if expected_value is not None:
            assert value == expected_value
        return value

    def assert_maybe_nothing(self, maybe: Maybe[T]) -> None:
        """Assert Maybe is Nothing."""
        assert maybe.is_nothing(), (
            f"Expected Nothing, got Some: {maybe.unwrap() if hasattr(maybe, 'unwrap') else maybe}"
        )

    def assert_io_result(self, io: IO[T], expected_value: T = None) -> T:
        """Assert IO computation result."""
        result = io.run()
        if expected_value is not None:
            assert result == expected_value
        return result

    def run_io_test(self, io_computation: IO[T]) -> T:
        """Run IO computation and return result."""
        return io_computation.run()


class FPExchangeTestCase(FPTestCase):
    """Base class for FP exchange tests."""

    def setup_fp_fixtures(self) -> dict[str, Any]:
        """Setup FP exchange fixtures."""
        return {
            "mock_adapter": self.create_mock_exchange_adapter(),
            "test_balance": Decimal("10000.00"),
            "test_symbol": "BTC-USD",
        }

    def create_mock_exchange_adapter(self):
        """Create mock FP exchange adapter."""
        adapter = Mock()
        adapter.get_balance = Mock(return_value=IO.pure(Ok(Decimal("10000.00"))))
        adapter.place_order = Mock(return_value=IO.pure(Ok({"order_id": "test-123"})))
        adapter.get_market_data = Mock(return_value=IO.pure(Ok({})))
        adapter.supports_functional = Mock(return_value=True)
        return adapter


class FPStrategyTestCase(FPTestCase):
    """Base class for FP strategy tests."""

    def setup_fp_fixtures(self) -> dict[str, Any]:
        """Setup FP strategy fixtures."""
        return {
            "mock_strategy": self.create_mock_strategy(),
            "test_market_data": self.create_test_market_data(),
        }

    def create_mock_strategy(self):
        """Create mock FP strategy."""
        try:
            from bot.fp.types.trading import Hold

            strategy = Mock()
            strategy.generate_signal = Mock(
                return_value=IO.pure(Ok(Hold(reason="Test hold")))
            )
            strategy.update_market_data = Mock(return_value=IO.pure(Ok(None)))
            return strategy
        except ImportError:
            return Mock()

    def create_test_market_data(self):
        """Create test market data in FP format."""
        try:
            from datetime import UTC, datetime

            from bot.fp.types.market import MarketSnapshot

            return MarketSnapshot(
                timestamp=datetime.now(UTC),
                symbol="BTC-USD",
                price=Decimal("50000.00"),
                volume=Decimal("100.00"),
                bid=Decimal("49950.00"),
                ask=Decimal("50050.00"),
            )
        except ImportError:
            return {"price": 50000.0, "volume": 100.0}


# =============================================================================
# FP MOCK OBJECTS
# =============================================================================


class FPMockFactory:
    """Factory for creating FP-compatible mock objects."""

    @staticmethod
    def create_mock_exchange_client(supports_fp: bool = True):
        """Create mock exchange client with FP support."""
        client = Mock()

        # Basic methods
        client.get_balance = AsyncMock(return_value=Decimal("10000.00"))
        client.place_order = AsyncMock(return_value={"order_id": "test-123"})
        client.cancel_order = AsyncMock(return_value=True)
        client.get_positions = AsyncMock(return_value=[])

        # FP methods
        if supports_fp:
            client.supports_functional_operations = Mock(return_value=True)
            client.get_functional_adapter = Mock(
                return_value=FPMockFactory.create_mock_fp_adapter()
            )
            client.place_order_functional = Mock(
                return_value=IO.pure(Ok({"order_id": "test-123"}))
            )
        else:
            client.supports_functional_operations = Mock(return_value=False)
            client.get_functional_adapter = Mock(return_value=None)

        return client

    @staticmethod
    def create_mock_fp_adapter():
        """Create mock FP adapter."""
        adapter = Mock()
        adapter.get_balance = Mock(return_value=IO.pure(Ok(Decimal("10000.00"))))
        adapter.place_order = Mock(return_value=IO.pure(Ok({"order_id": "test-123"})))
        adapter.cancel_order = Mock(return_value=IO.pure(Ok(True)))
        adapter.get_market_data = Mock(return_value=IO.pure(Ok({})))
        adapter.get_positions = Mock(return_value=IO.pure(Ok([])))
        return adapter

    @staticmethod
    def create_mock_fp_strategy():
        """Create mock FP strategy."""
        try:
            from bot.fp.types.trading import Hold

            strategy = Mock()
            strategy.generate_signal = Mock(
                return_value=IO.pure(Ok(Hold(reason="Test")))
            )
            strategy.analyze_market = Mock(return_value=IO.pure(Ok({})))
            strategy.update_indicators = Mock(return_value=IO.pure(Ok(None)))
            return strategy
        except ImportError:
            return Mock()

    @staticmethod
    def create_mock_fp_risk_manager():
        """Create mock FP risk manager."""
        risk_manager = Mock()
        risk_manager.validate_trade = Mock(return_value=IO.pure(Ok(True)))
        risk_manager.calculate_position_size = Mock(return_value=IO.pure(Ok(0.1)))
        risk_manager.check_limits = Mock(return_value=IO.pure(Ok(True)))
        risk_manager.assess_risk = Mock(return_value=IO.pure(Ok({"risk_score": 0.3})))
        return risk_manager

    @staticmethod
    def create_mock_fp_indicator():
        """Create mock FP indicator."""
        try:
            from bot.fp.types.indicators import IndicatorResult

            indicator = Mock()
            indicator.calculate = Mock(return_value=IO.pure(Ok({})))
            indicator.update = Mock(return_value=IO.pure(Ok(None)))
            indicator.get_signals = Mock(return_value=IO.pure(Ok([])))
            return indicator
        except ImportError:
            return Mock()


# =============================================================================
# TEST MIGRATION ADAPTERS
# =============================================================================


class TestMigrationAdapter:
    """Adapter to help migrate imperative tests to FP patterns."""

    def __init__(self, legacy_test_class: type):
        """Initialize with legacy test class."""
        self.legacy_test_class = legacy_test_class
        self.fp_conversions = {}

    def add_fp_conversion(self, method_name: str, fp_converter: Callable):
        """Add FP conversion for a specific method."""
        self.fp_conversions[method_name] = fp_converter

    def create_fp_wrapper(self, instance: Any) -> Any:
        """Create FP wrapper for legacy test instance."""

        class FPWrapper:
            def __init__(self, legacy_instance):
                self.legacy_instance = legacy_instance
                self.adapter = self

        wrapper = FPWrapper(instance)

        # Copy methods from legacy instance
        for method_name in dir(instance):
            if not method_name.startswith("_"):
                method = getattr(instance, method_name)
                if callable(method):
                    if method_name in self.fp_conversions:
                        # Apply FP conversion
                        fp_method = self.fp_conversions[method_name](method)
                        setattr(wrapper, method_name, fp_method)
                    else:
                        # Keep original method
                        setattr(wrapper, method_name, method)

        return wrapper

    @staticmethod
    def convert_async_to_io(async_method: Callable) -> Callable:
        """Convert async method to IO computation."""

        def io_wrapper(*args, **kwargs):
            async def async_computation():
                return await async_method(*args, **kwargs)

            def sync_computation():
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_computation())

            return IO(sync_computation)

        return io_wrapper

    @staticmethod
    def convert_exception_to_result(method: Callable) -> Callable:
        """Convert method that raises exceptions to return Result."""

        def result_wrapper(*args, **kwargs):
            try:
                result = method(*args, **kwargs)
                return Ok(result)
            except Exception as e:
                return Err(str(e))

        return result_wrapper

    @staticmethod
    def convert_nullable_to_maybe(method: Callable) -> Callable:
        """Convert method that returns None to Maybe."""

        def maybe_wrapper(*args, **kwargs):
            result = method(*args, **kwargs)
            if result is None:
                return Nothing()
            return Some(result)

        return maybe_wrapper


# =============================================================================
# FP PROPERTY-BASED TESTING STRATEGIES
# =============================================================================


class FPPropertyStrategies:
    """Property-based testing strategies for FP types."""

    @staticmethod
    @st.composite
    def fp_decimal_strategy(draw, min_value=0, max_value=1000000, places=8):
        """Strategy for generating Decimal values."""
        value = draw(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return Decimal(str(round(value, places)))

    @staticmethod
    @st.composite
    def fp_result_strategy(
        draw, value_strategy=st.integers(), error_strategy=st.text()
    ):
        """Strategy for generating Result values."""
        is_ok = draw(st.booleans())
        if is_ok:
            value = draw(value_strategy)
            return Ok(value)
        error = draw(error_strategy)
        return Err(error)

    @staticmethod
    @st.composite
    def fp_maybe_strategy(draw, value_strategy=st.integers()):
        """Strategy for generating Maybe values."""
        is_some = draw(st.booleans())
        if is_some:
            value = draw(value_strategy)
            return Some(value)
        return Nothing()

    @staticmethod
    @st.composite
    def fp_market_snapshot_strategy(draw):
        """Strategy for generating MarketSnapshot values."""
        try:
            from datetime import UTC, datetime, timedelta

            from bot.fp.types.market import MarketSnapshot

            base_time = datetime.now(UTC)
            timestamp = base_time + timedelta(
                minutes=draw(st.integers(min_value=-1000, max_value=1000))
            )

            symbol = draw(st.sampled_from(["BTC-USD", "ETH-USD", "SOL-USD"]))

            # Generate prices ensuring bid < ask
            base_price = draw(
                FPPropertyStrategies.fp_decimal_strategy(
                    min_value=0.01, max_value=100000, places=2
                )
            )
            spread_pct = draw(st.floats(min_value=0.0001, max_value=0.01))
            half_spread = base_price * Decimal(str(spread_pct)) / 2

            bid = base_price - half_spread
            ask = base_price + half_spread
            price = (bid + ask) / 2
            volume = draw(
                FPPropertyStrategies.fp_decimal_strategy(
                    min_value=0, max_value=10000, places=2
                )
            )

            return MarketSnapshot(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
            )
        except ImportError:
            return draw(
                st.dictionaries(
                    st.text(), st.one_of(st.floats(), st.text(), st.integers())
                )
            )

    @staticmethod
    @st.composite
    def fp_trade_signal_strategy(draw):
        """Strategy for generating trade signals."""
        try:
            from bot.fp.types.trading import Hold, Long, MarketMake, Short

            signal_type = draw(
                st.sampled_from(["long", "short", "hold", "market_make"])
            )

            if signal_type == "long":
                return Long(
                    confidence=draw(st.floats(min_value=0, max_value=1)),
                    size=draw(st.floats(min_value=0.01, max_value=1)),
                    reason=draw(st.text(min_size=1, max_size=100)),
                )
            if signal_type == "short":
                return Short(
                    confidence=draw(st.floats(min_value=0, max_value=1)),
                    size=draw(st.floats(min_value=0.01, max_value=1)),
                    reason=draw(st.text(min_size=1, max_size=100)),
                )
            if signal_type == "hold":
                return Hold(reason=draw(st.text(min_size=1, max_size=100)))
            # market_make
            base_price = draw(st.floats(min_value=1, max_value=100000))
            spread = draw(st.floats(min_value=0.01, max_value=100))
            return MarketMake(
                bid_price=base_price - spread / 2,
                ask_price=base_price + spread / 2,
                bid_size=draw(st.floats(min_value=0.01, max_value=100)),
                ask_size=draw(st.floats(min_value=0.01, max_value=100)),
            )
        except ImportError:
            return draw(st.dictionaries(st.text(), st.text()))


# =============================================================================
# FP TEST DATA GENERATORS
# =============================================================================


class FPTestDataGenerator:
    """Generator for FP-compatible test data."""

    @staticmethod
    def create_market_snapshots(count: int = 100, base_price: float = 50000.0):
        """Create series of market snapshots."""
        try:
            import random
            from datetime import UTC, datetime, timedelta

            from bot.fp.types.market import MarketSnapshot

            snapshots = []
            current_price = base_price
            base_time = datetime.now(UTC)

            for i in range(count):
                # Random walk price
                price_change = random.uniform(-0.02, 0.02)
                current_price *= 1 + price_change

                # Generate bid/ask spread
                spread_pct = random.uniform(0.0005, 0.002)
                half_spread = current_price * spread_pct / 2

                bid = Decimal(str(current_price - half_spread))
                ask = Decimal(str(current_price + half_spread))
                price = (bid + ask) / 2

                snapshot = MarketSnapshot(
                    timestamp=base_time + timedelta(minutes=i),
                    symbol="BTC-USD",
                    price=price,
                    volume=Decimal(str(random.uniform(50, 200))),
                    bid=bid,
                    ask=ask,
                )
                snapshots.append(snapshot)

            return snapshots
        except ImportError:
            return []

    @staticmethod
    def create_portfolio_timeline(count: int = 50):
        """Create portfolio state timeline."""
        try:
            from bot.fp.types.portfolio import Portfolio, Position

            portfolios = []
            base_balance = Decimal("10000.00")

            for i in range(count):
                # Sometimes have position, sometimes not
                if i % 3 == 0:
                    positions = ()
                else:
                    position = Position(
                        symbol="BTC-USD",
                        side="LONG" if i % 2 == 0 else "SHORT",
                        size=Decimal("0.1"),
                        entry_price=Decimal("50000.00"),
                        current_price=Decimal(str(50000 + (i * 100))),
                    )
                    positions = (position,)

                portfolio = Portfolio(
                    positions=positions,
                    cash_balance=base_balance + Decimal(str(i * 10)),
                )
                portfolios.append(portfolio)

            return portfolios
        except ImportError:
            return []

    @staticmethod
    def create_trade_results(count: int = 20):
        """Create series of trade results."""
        try:
            import random
            from datetime import UTC, datetime, timedelta

            from bot.fp.types.portfolio import TradeResult

            results = []
            base_time = datetime.now(UTC)

            for i in range(count):
                is_profitable = random.choice([True, False])
                entry_price = Decimal("50000.00")

                if is_profitable:
                    exit_price = entry_price * Decimal("1.05")  # 5% profit
                else:
                    exit_price = entry_price * Decimal("0.97")  # 3% loss

                result = TradeResult(
                    trade_id=f"trade-{i}",
                    symbol="BTC-USD",
                    side="LONG" if i % 2 == 0 else "SHORT",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=Decimal("0.1"),
                    entry_time=base_time + timedelta(hours=i),
                    exit_time=base_time + timedelta(hours=i, minutes=30),
                )
                results.append(result)

            return results
        except ImportError:
            return []


# =============================================================================
# FP TEST COMPATIBILITY LAYER
# =============================================================================


class FPTestCompatibility:
    """Compatibility layer for running both FP and imperative tests."""

    @staticmethod
    def wrap_test_method(test_method: Callable, fp_mode: bool = True):
        """Wrap test method to support both FP and imperative modes."""
        if fp_mode:
            return FPTestCompatibility._wrap_fp_mode(test_method)
        return FPTestCompatibility._wrap_imperative_mode(test_method)

    @staticmethod
    def _wrap_fp_mode(test_method: Callable):
        """Wrap test method for FP mode."""

        def fp_wrapper(*args, **kwargs):
            # Convert results to FP types if needed
            result = test_method(*args, **kwargs)

            # Wrap exceptions in Result
            try:
                return Ok(result) if result is not None else Ok(None)
            except Exception as e:
                return Err(str(e))

        return fp_wrapper

    @staticmethod
    def _wrap_imperative_mode(test_method: Callable):
        """Wrap test method for imperative mode."""

        def imperative_wrapper(*args, **kwargs):
            # Unwrap FP types to imperative values
            converted_args = []
            for arg in args:
                if (hasattr(arg, "is_ok") and arg.is_ok()) or (
                    hasattr(arg, "is_some") and arg.is_some()
                ):
                    converted_args.append(arg.unwrap())
                elif hasattr(arg, "run"):
                    converted_args.append(arg.run())
                else:
                    converted_args.append(arg)

            converted_kwargs = {}
            for key, value in kwargs.items():
                if (hasattr(value, "is_ok") and value.is_ok()) or (
                    hasattr(value, "is_some") and value.is_some()
                ):
                    converted_kwargs[key] = value.unwrap()
                elif hasattr(value, "run"):
                    converted_kwargs[key] = value.run()
                else:
                    converted_kwargs[key] = value

            return test_method(*converted_args, **converted_kwargs)

        return imperative_wrapper


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    "FPTestCase",
    "FPExchangeTestCase",
    "FPStrategyTestCase",
    # Mock factories
    "FPMockFactory",
    # Migration adapters
    "TestMigrationAdapter",
    # Property strategies
    "FPPropertyStrategies",
    # Data generators
    "FPTestDataGenerator",
    # Compatibility layer
    "FPTestCompatibility",
]
