"""
FP Test Base Classes

Base classes and mixins for functional programming tests that provide
standardized testing patterns, utilities, and assertions for FP components.
"""

from abc import ABC
from collections.abc import Callable
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, TypeVar
from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

# Safe imports with fallbacks
T = TypeVar("T")
E = TypeVar("E")

try:
    from bot.fp.core.functional_validation import FPFailure, FPResult, FPSuccess
    from bot.fp.types.base import Money, Percentage, Symbol, TimeInterval
    from bot.fp.types.effects import IO, Err, Maybe, Nothing, Ok, Result, Some
    from bot.fp.types.market import OHLCV, MarketSnapshot
    from bot.fp.types.portfolio import Portfolio, Position, TradeResult
    from bot.fp.types.trading import Hold, Long, MarketMake, Short, TradeSignal

    FP_AVAILABLE = True
except ImportError:
    FP_AVAILABLE = False

    # Minimal fallback types for compatibility
    class Result:
        def is_ok(self):
            return False

        def is_err(self):
            return False

        def unwrap(self):
            raise NotImplementedError

    class Ok(Result):
        def __init__(self, value):
            self.value = value

        def is_ok(self):
            return True

        def unwrap(self):
            return self.value

    class Err(Result):
        def __init__(self, error):
            self.error = error

        def is_err(self):
            return True

    class Maybe:
        def is_some(self):
            return False

        def is_nothing(self):
            return False

    class Some(Maybe):
        def __init__(self, value):
            self.value = value

        def is_some(self):
            return True

        def unwrap(self):
            return self.value

    class Nothing(Maybe):
        def is_nothing(self):
            return True

    class IO:
        def __init__(self, fn):
            self._fn = fn

        @classmethod
        def pure(cls, value):
            return cls(lambda: value)

        def run(self):
            return self._fn()


# =============================================================================
# BASE FP TEST CLASS
# =============================================================================


class FPTestBase(ABC):
    """
    Base class for all functional programming tests.

    Provides common utilities, assertions, and patterns for testing FP components.
    """

    @pytest.fixture(autouse=True)
    def setup_fp_test(self):
        """Setup FP test environment."""
        self.fp_available = FP_AVAILABLE
        self.mock_registry = {}
        self.test_data_cache = {}

        # Setup method is called before each test
        self.setup_test_fixtures()

        yield

        # Cleanup after test
        self.cleanup_test_fixtures()

    def setup_test_fixtures(self):
        """Override to setup test-specific fixtures."""

    def cleanup_test_fixtures(self):
        """Override to cleanup test-specific fixtures."""
        self.mock_registry.clear()
        self.test_data_cache.clear()

    # =============================================================================
    # FP ASSERTIONS
    # =============================================================================

    def assert_result_ok(self, result: Result[T, E], expected_value: T = None) -> T:
        """Assert Result is Ok and optionally check value."""
        assert (
            hasattr(result, "is_ok") and result.is_ok()
        ), f"Expected Ok, got Err: {getattr(result, 'error', result)}"
        value = result.unwrap()
        if expected_value is not None:
            assert value == expected_value, f"Expected {expected_value}, got {value}"
        return value

    def assert_result_err(self, result: Result[T, E], expected_error: E = None) -> E:
        """Assert Result is Err and optionally check error."""
        assert (
            hasattr(result, "is_err") and result.is_err()
        ), f"Expected Err, got Ok: {getattr(result, 'value', result)}"
        error = getattr(result, "error", result)
        if expected_error is not None:
            assert (
                error == expected_error
            ), f"Expected error {expected_error}, got {error}"
        return error

    def assert_maybe_some(self, maybe: Maybe[T], expected_value: T = None) -> T:
        """Assert Maybe is Some and optionally check value."""
        assert (
            hasattr(maybe, "is_some") and maybe.is_some()
        ), "Expected Some, got Nothing"
        value = maybe.unwrap()
        if expected_value is not None:
            assert value == expected_value, f"Expected {expected_value}, got {value}"
        return value

    def assert_maybe_nothing(self, maybe: Maybe[T]) -> None:
        """Assert Maybe is Nothing."""
        assert (
            hasattr(maybe, "is_nothing") and maybe.is_nothing()
        ), f"Expected Nothing, got Some: {getattr(maybe, 'value', maybe)}"

    def assert_io_result(self, io: IO[T], expected_value: T = None) -> T:
        """Assert IO computation result."""
        result = io.run()
        if expected_value is not None:
            assert result == expected_value, f"Expected {expected_value}, got {result}"
        return result

    def assert_fp_result_success(self, fp_result, expected_value: Any = None) -> Any:
        """Assert FPResult is success."""
        assert (
            hasattr(fp_result, "is_success") and fp_result.is_success()
        ), f"Expected FPSuccess, got FPFailure: {getattr(fp_result, 'failure', fp_result)}"
        value = fp_result.success()
        if expected_value is not None:
            assert value == expected_value, f"Expected {expected_value}, got {value}"
        return value

    def assert_fp_result_failure(self, fp_result, expected_error: Any = None) -> Any:
        """Assert FPResult is failure."""
        assert (
            hasattr(fp_result, "is_failure") and fp_result.is_failure()
        ), f"Expected FPFailure, got FPSuccess: {getattr(fp_result, 'success', fp_result)}"
        error = fp_result.failure()
        if expected_error is not None:
            assert (
                error == expected_error
            ), f"Expected error {expected_error}, got {error}"
        return error

    # =============================================================================
    # FP TEST UTILITIES
    # =============================================================================

    def run_io(self, io: IO[T]) -> T:
        """Run IO computation and return result."""
        return io.run()

    def run_io_safe(self, io: IO[T], default: T = None) -> T | Exception:
        """Run IO computation with exception handling."""
        try:
            return io.run()
        except Exception as e:
            return e if default is None else default

    def create_test_result_ok(self, value: T) -> Result[T, str]:
        """Create Ok Result for testing."""
        return Ok(value)

    def create_test_result_err(self, error: E) -> Result[Any, E]:
        """Create Err Result for testing."""
        return Err(error)

    def create_test_maybe_some(self, value: T) -> Maybe[T]:
        """Create Some Maybe for testing."""
        return Some(value)

    def create_test_maybe_nothing(self) -> Maybe[Any]:
        """Create Nothing Maybe for testing."""
        return Nothing()

    def create_test_io(self, value: T) -> IO[T]:
        """Create pure IO for testing."""
        return IO.pure(value)

    def create_test_io_effect(self, effect: Callable[[], T]) -> IO[T]:
        """Create IO with side effect for testing."""
        return IO(effect)

    # =============================================================================
    # MOCK UTILITIES
    # =============================================================================

    def create_mock_with_fp_result(
        self, method_name: str, value: Any, is_error: bool = False
    ) -> Mock:
        """Create mock that returns FP Result."""
        mock = Mock()
        if is_error:
            setattr(mock, method_name, Mock(return_value=Err(value)))
        else:
            setattr(mock, method_name, Mock(return_value=Ok(value)))
        return mock

    def create_mock_with_fp_maybe(self, method_name: str, value: Any = None) -> Mock:
        """Create mock that returns FP Maybe."""
        mock = Mock()
        if value is None:
            setattr(mock, method_name, Mock(return_value=Nothing()))
        else:
            setattr(mock, method_name, Mock(return_value=Some(value)))
        return mock

    def create_mock_with_fp_io(self, method_name: str, value: Any) -> Mock:
        """Create mock that returns FP IO."""
        mock = Mock()
        setattr(mock, method_name, Mock(return_value=IO.pure(value)))
        return mock

    def register_mock(self, name: str, mock: Mock) -> Mock:
        """Register mock for cleanup."""
        self.mock_registry[name] = mock
        return mock

    # =============================================================================
    # PROPERTY-BASED TESTING HELPERS
    # =============================================================================

    def fp_decimal_strategy(
        self, min_value: float = 0, max_value: float = 1000000, places: int = 8
    ):
        """Strategy for generating Decimal values."""

        @st.composite
        def decimal_strategy(draw):
            value = draw(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            return Decimal(str(round(value, places)))

        return decimal_strategy()

    def fp_result_strategy(
        self, value_strategy=st.integers(), error_strategy=st.text()
    ):
        """Strategy for generating Result values."""

        @st.composite
        def result_strategy(draw):
            is_ok = draw(st.booleans())
            if is_ok:
                value = draw(value_strategy)
                return Ok(value)
            error = draw(error_strategy)
            return Err(error)

        return result_strategy()

    def fp_maybe_strategy(self, value_strategy=st.integers()):
        """Strategy for generating Maybe values."""

        @st.composite
        def maybe_strategy(draw):
            is_some = draw(st.booleans())
            if is_some:
                value = draw(value_strategy)
                return Some(value)
            return Nothing()

        return maybe_strategy()

    # =============================================================================
    # CONTEXT MANAGERS
    # =============================================================================

    @contextmanager
    def fp_test_context(self, **kwargs):
        """Context manager for FP testing with custom configuration."""
        old_config = getattr(self, "fp_config", {})
        self.fp_config = {**old_config, **kwargs}

        try:
            yield self.fp_config
        finally:
            self.fp_config = old_config

    @contextmanager
    def mock_fp_dependencies(self, **mocks):
        """Context manager for mocking FP dependencies."""
        patchers = []

        try:
            for module_path, mock_obj in mocks.items():
                patcher = patch(module_path, mock_obj)
                patchers.append(patcher)
                patcher.start()

            yield

        finally:
            for patcher in patchers:
                patcher.stop()


# =============================================================================
# SPECIALIZED FP TEST BASE CLASSES
# =============================================================================


class FPExchangeTestBase(FPTestBase):
    """Base class for testing FP exchange components."""

    def setup_test_fixtures(self):
        """Setup exchange-specific test fixtures."""
        super().setup_test_fixtures()
        self.test_balance = Decimal("10000.00")
        self.test_symbol = "BTC-USD"
        self.test_price = Decimal("50000.00")

    def create_mock_exchange_adapter(self, balance: Decimal = None) -> Mock:
        """Create mock FP exchange adapter."""
        balance = balance or self.test_balance

        adapter = Mock()
        adapter.get_balance = Mock(return_value=IO.pure(Ok(balance)))
        adapter.place_order = Mock(return_value=IO.pure(Ok({"order_id": "test-123"})))
        adapter.cancel_order = Mock(return_value=IO.pure(Ok(True)))
        adapter.get_market_data = Mock(return_value=IO.pure(Ok({})))
        adapter.get_positions = Mock(return_value=IO.pure(Ok([])))
        adapter.supports_functional = Mock(return_value=True)

        return self.register_mock("exchange_adapter", adapter)

    def create_test_market_snapshot(
        self, price: Decimal = None, symbol: str = None
    ) -> "MarketSnapshot":
        """Create test market snapshot."""
        if not FP_AVAILABLE:
            return {}

        from datetime import UTC, datetime

        price = price or self.test_price
        symbol = symbol or self.test_symbol
        spread = price * Decimal("0.001")

        return MarketSnapshot(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            price=price,
            volume=Decimal("100.00"),
            bid=price - spread / 2,
            ask=price + spread / 2,
        )

    def assert_order_placed(self, mock_adapter: Mock, expected_symbol: str = None):
        """Assert order was placed with expected parameters."""
        mock_adapter.place_order.assert_called()
        if expected_symbol:
            call_args = mock_adapter.place_order.call_args
            # Check if symbol is in the call arguments
            assert expected_symbol in str(
                call_args
            ), f"Expected symbol {expected_symbol} in order call"


class FPStrategyTestBase(FPTestBase):
    """Base class for testing FP strategy components."""

    def setup_test_fixtures(self):
        """Setup strategy-specific test fixtures."""
        super().setup_test_fixtures()
        self.test_confidence = 0.8
        self.test_size = 0.25

    def create_mock_strategy(self) -> Mock:
        """Create mock FP strategy."""
        if not FP_AVAILABLE:
            return Mock()

        strategy = Mock()
        strategy.generate_signal = Mock(return_value=IO.pure(Ok(Hold(reason="Test"))))
        strategy.analyze_market = Mock(return_value=IO.pure(Ok({})))
        strategy.update_indicators = Mock(return_value=IO.pure(Ok(None)))

        return self.register_mock("strategy", strategy)

    def create_test_long_signal(
        self, confidence: float = None, size: float = None
    ) -> "Long":
        """Create test Long signal."""
        if not FP_AVAILABLE:
            return {}

        return Long(
            confidence=confidence or self.test_confidence,
            size=size or self.test_size,
            reason="Test long signal",
        )

    def create_test_short_signal(
        self, confidence: float = None, size: float = None
    ) -> "Short":
        """Create test Short signal."""
        if not FP_AVAILABLE:
            return {}

        return Short(
            confidence=confidence or self.test_confidence,
            size=size or self.test_size,
            reason="Test short signal",
        )

    def create_test_hold_signal(self, reason: str = "Test hold") -> "Hold":
        """Create test Hold signal."""
        if not FP_AVAILABLE:
            return {}

        return Hold(reason=reason)

    def assert_signal_type(self, signal: Any, expected_type: type):
        """Assert signal is of expected type."""
        assert isinstance(
            signal, expected_type
        ), f"Expected {expected_type.__name__}, got {type(signal).__name__}"

    def assert_signal_confidence(
        self, signal: Any, min_confidence: float = 0.0, max_confidence: float = 1.0
    ):
        """Assert signal confidence is within range."""
        if hasattr(signal, "confidence"):
            assert (
                min_confidence <= signal.confidence <= max_confidence
            ), f"Signal confidence {signal.confidence} not in range [{min_confidence}, {max_confidence}]"


class FPRiskTestBase(FPTestBase):
    """Base class for testing FP risk management components."""

    def setup_test_fixtures(self):
        """Setup risk-specific test fixtures."""
        super().setup_test_fixtures()
        self.test_max_position_size = 0.25
        self.test_stop_loss_pct = 0.02
        self.test_take_profit_pct = 0.05

    def create_mock_risk_manager(self) -> Mock:
        """Create mock FP risk manager."""
        risk_manager = Mock()
        risk_manager.validate_trade = Mock(return_value=IO.pure(Ok(True)))
        risk_manager.calculate_position_size = Mock(return_value=IO.pure(Ok(0.1)))
        risk_manager.check_limits = Mock(return_value=IO.pure(Ok(True)))
        risk_manager.assess_risk = Mock(return_value=IO.pure(Ok({"risk_score": 0.3})))

        return self.register_mock("risk_manager", risk_manager)

    def create_test_position(
        self, side: str = "LONG", size: Decimal = None
    ) -> "Position":
        """Create test position."""
        if not FP_AVAILABLE:
            return {}

        size = size or Decimal("0.1")

        return Position(
            symbol="BTC-USD",
            side=side,
            size=size,
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50000.00"),
        )

    def assert_risk_within_limits(
        self, risk_metrics: dict[str, Any], max_risk: float = 1.0
    ):
        """Assert risk metrics are within acceptable limits."""
        risk_score = risk_metrics.get("risk_score", 0)
        assert (
            0 <= risk_score <= max_risk
        ), f"Risk score {risk_score} exceeds limit {max_risk}"


class FPIndicatorTestBase(FPTestBase):
    """Base class for testing FP indicator components."""

    def setup_test_fixtures(self):
        """Setup indicator-specific test fixtures."""
        super().setup_test_fixtures()
        self.test_period = 14
        self.test_ohlcv_data = []

    def create_mock_indicator(self, indicator_name: str = "test") -> Mock:
        """Create mock FP indicator."""
        indicator = Mock()
        indicator.calculate = Mock(return_value=IO.pure(Ok({"value": 50.0})))
        indicator.update = Mock(return_value=IO.pure(Ok(None)))
        indicator.get_signals = Mock(return_value=IO.pure(Ok([])))
        indicator.name = indicator_name

        return self.register_mock(f"indicator_{indicator_name}", indicator)

    def create_test_ohlcv_data(self, count: int = 100) -> list["OHLCV"]:
        """Create test OHLCV data series."""
        if not FP_AVAILABLE:
            return []

        from datetime import UTC, datetime, timedelta

        ohlcv_data = []
        base_time = datetime.now(UTC)
        base_price = 50000.0

        for i in range(count):
            price = base_price + (i * 10)  # Simple ascending price

            ohlcv = OHLCV(
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(str(price - 5)),
                high=Decimal(str(price + 10)),
                low=Decimal(str(price - 10)),
                close=Decimal(str(price)),
                volume=Decimal("100.00"),
            )
            ohlcv_data.append(ohlcv)

        return ohlcv_data

    def assert_indicator_value_range(
        self, value: float, min_val: float, max_val: float
    ):
        """Assert indicator value is within expected range."""
        assert (
            min_val <= value <= max_val
        ), f"Indicator value {value} not in range [{min_val}, {max_val}]"


# =============================================================================
# TEST MIXINS
# =============================================================================


class MonadLawTestMixin:
    """Mixin for testing monad laws."""

    def test_result_left_identity(self):
        """Test Result monad left identity law."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        value = 42
        f = lambda x: Ok(x * 2)

        # Left side: return a >>= f
        left = Ok(value).flat_map(f)
        # Right side: f a
        right = f(value)

        assert left.unwrap() == right.unwrap()

    def test_result_right_identity(self):
        """Test Result monad right identity law."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        m = Ok(42)

        # Left side: m >>= return
        left = m.flat_map(lambda x: Ok(x))
        # Right side: m
        right = m

        assert left.unwrap() == right.unwrap()

    def test_result_associativity(self):
        """Test Result monad associativity law."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        m = Ok(10)
        f = lambda x: Ok(x * 2)
        g = lambda x: Ok(x + 1)

        # Left side: (m >>= f) >>= g
        left = m.flat_map(f).flat_map(g)
        # Right side: m >>= (λx → f x >>= g)
        right = m.flat_map(lambda x: f(x).flat_map(g))

        assert left.unwrap() == right.unwrap()


class PropertyTestMixin:
    """Mixin for property-based testing."""

    @given(st.integers())
    def test_fp_result_properties(self, value: int):
        """Property test for Result monad."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        result = Ok(value)

        # Ok and Err are mutually exclusive
        assert result.is_ok() and not result.is_err()

        # Unwrap returns the original value
        assert result.unwrap() == value

    @given(st.integers())
    def test_fp_maybe_properties(self, value: int):
        """Property test for Maybe monad."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        maybe = Some(value)

        # Some and Nothing are mutually exclusive
        assert maybe.is_some() and not maybe.is_nothing()

        # Unwrap returns the original value
        assert maybe.unwrap() == value


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    "FPTestBase",
    "FPExchangeTestBase",
    "FPStrategyTestBase",
    "FPRiskTestBase",
    "FPIndicatorTestBase",
    # Mixins
    "MonadLawTestMixin",
    "PropertyTestMixin",
    # Availability flag
    "FP_AVAILABLE",
]
