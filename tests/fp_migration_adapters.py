"""
FP Migration Adapters for Test Infrastructure

This module provides adapters and utilities to help migrate existing imperative
tests to functional programming patterns while maintaining backward compatibility.
It supports gradual migration and dual-mode testing.
"""

import asyncio
import inspect
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeVar
from unittest.mock import Mock, patch

import pytest

# Safe imports with fallbacks
T = TypeVar("T")
E = TypeVar("E")

try:
    from bot.fp.types.effects import IO, Err, Maybe, Nothing, Ok, Result, Some
    from bot.fp.types.market import MarketSnapshot
    from bot.fp.types.portfolio import Portfolio, Position
    from bot.fp.types.trading import Hold, Long, Short, TradeSignal

    FP_AVAILABLE = True
except ImportError:
    FP_AVAILABLE = False

    # Minimal fallback types for compatibility
    class Result:
        pass

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
        pass

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
# MIGRATION CONFIGURATION
# =============================================================================


@dataclass
class MigrationConfig:
    """Configuration for test migration behavior."""

    enable_fp_mode: bool = True
    enable_dual_mode: bool = True  # Run both FP and imperative versions
    auto_convert_types: bool = True
    preserve_exceptions: bool = False
    validate_conversions: bool = True
    migration_warnings: bool = True


class MigrationRegistry:
    """Registry for tracking migration status and compatibility."""

    def __init__(self):
        self.migrated_classes: dict[str, bool] = {}
        self.migration_errors: list[str] = []
        self.compatibility_issues: list[str] = []

    def register_migration(self, class_name: str, success: bool):
        """Register migration attempt result."""
        self.migrated_classes[class_name] = success
        if not success:
            self.migration_errors.append(f"Failed to migrate {class_name}")

    def report_compatibility_issue(self, issue: str):
        """Report compatibility issue during migration."""
        self.compatibility_issues.append(issue)

    def get_migration_report(self) -> dict[str, Any]:
        """Get comprehensive migration report."""
        total_migrations = len(self.migrated_classes)
        successful_migrations = sum(self.migrated_classes.values())

        return {
            "total_migrations": total_migrations,
            "successful_migrations": successful_migrations,
            "success_rate": (
                successful_migrations / total_migrations if total_migrations > 0 else 0
            ),
            "migration_errors": self.migration_errors,
            "compatibility_issues": self.compatibility_issues,
        }


# Global migration registry
migration_registry = MigrationRegistry()


# =============================================================================
# TYPE CONVERSION ADAPTERS
# =============================================================================


class TypeConversionAdapter:
    """Adapter for converting between imperative and FP types."""

    @staticmethod
    def to_fp_result(value: Any, error_on_exception: bool = True) -> Result:
        """Convert value to FP Result type."""
        if isinstance(value, Result):
            return value

        try:
            if error_on_exception and isinstance(value, Exception):
                return Err(str(value))
            return Ok(value)
        except Exception as e:
            return Err(str(e))

    @staticmethod
    def to_fp_maybe(value: Any) -> Maybe:
        """Convert value to FP Maybe type."""
        if isinstance(value, Maybe):
            return value

        if value is None:
            return Nothing()
        return Some(value)

    @staticmethod
    def to_fp_io(value_or_callable: Any | Callable) -> IO:
        """Convert value or callable to FP IO type."""
        if isinstance(value_or_callable, IO):
            return value_or_callable

        if callable(value_or_callable):
            return IO(value_or_callable)
        return IO.pure(value_or_callable)

    @staticmethod
    def from_fp_result(result: Result, raise_on_error: bool = True) -> Any:
        """Convert FP Result to imperative value."""
        if not isinstance(result, Result):
            return result

        if hasattr(result, "is_ok") and result.is_ok():
            return result.unwrap()
        if hasattr(result, "is_err") and result.is_err():
            if raise_on_error:
                raise Exception(result.error)
            return None

        return result

    @staticmethod
    def from_fp_maybe(maybe: Maybe, default: Any = None) -> Any:
        """Convert FP Maybe to imperative value."""
        if not isinstance(maybe, Maybe):
            return maybe

        if hasattr(maybe, "is_some") and maybe.is_some():
            return maybe.unwrap()

        return default

    @staticmethod
    def from_fp_io(io: IO) -> Any:
        """Convert FP IO to imperative value by running it."""
        if not isinstance(io, IO):
            return io

        return io.run()


# =============================================================================
# TEST METHOD ADAPTERS
# =============================================================================


class TestMethodAdapter:
    """Adapter for converting test methods between imperative and FP styles."""

    def __init__(self, config: MigrationConfig = None):
        self.config = config or MigrationConfig()
        self.type_converter = TypeConversionAdapter()

    def adapt_sync_method(self, method: Callable) -> Callable:
        """Adapt synchronous test method for FP patterns."""

        def fp_wrapper(*args, **kwargs):
            # Convert arguments to FP types if needed
            if self.config.auto_convert_types:
                args, kwargs = self._convert_args_to_fp(args, kwargs)

            try:
                # Execute original method
                result = method(*args, **kwargs)

                # Convert result to FP type
                if self.config.auto_convert_types:
                    result = self.type_converter.to_fp_result(result)

                return result

            except Exception as e:
                if self.config.preserve_exceptions:
                    raise
                return self.type_converter.to_fp_result(e, error_on_exception=True)

        return fp_wrapper

    def adapt_async_method(self, method: Callable) -> Callable:
        """Adapt asynchronous test method for FP patterns."""

        async def fp_async_wrapper(*args, **kwargs):
            # Convert arguments to FP types if needed
            if self.config.auto_convert_types:
                args, kwargs = self._convert_args_to_fp(args, kwargs)

            try:
                # Execute original async method
                result = await method(*args, **kwargs)

                # Convert result to FP type
                if self.config.auto_convert_types:
                    result = self.type_converter.to_fp_result(result)

                return result

            except Exception as e:
                if self.config.preserve_exceptions:
                    raise
                return self.type_converter.to_fp_result(e, error_on_exception=True)

        def io_wrapper(*args, **kwargs):
            """Return IO computation that runs the async method."""

            async def async_computation():
                return await fp_async_wrapper(*args, **kwargs)

            def sync_computation():
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_computation())

            return IO(sync_computation)

        return io_wrapper

    def _convert_args_to_fp(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Convert method arguments to FP types."""
        converted_args = []
        for arg in args:
            if isinstance(arg, (int, float, str, bool, Decimal)):
                converted_args.append(arg)  # Keep primitives as-is
            elif arg is None:
                converted_args.append(Nothing())
            else:
                converted_args.append(arg)  # Keep complex types as-is for now

        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str, bool, Decimal)):
                converted_kwargs[key] = value  # Keep primitives as-is
            elif value is None:
                converted_kwargs[key] = Nothing()
            else:
                converted_kwargs[key] = value  # Keep complex types as-is for now

        return tuple(converted_args), converted_kwargs


# =============================================================================
# TEST CLASS MIGRATION ADAPTER
# =============================================================================


class TestClassMigrationAdapter:
    """Adapter for migrating entire test classes to FP patterns."""

    def __init__(self, config: MigrationConfig = None):
        self.config = config or MigrationConfig()
        self.method_adapter = TestMethodAdapter(config)

    def migrate_test_class(self, test_class: type) -> type:
        """Migrate entire test class to FP patterns."""
        class_name = test_class.__name__

        try:
            # Create new FP-compatible class
            fp_class_name = f"FP{class_name}"
            fp_class_dict = {}

            # Copy and adapt all test methods
            for attr_name in dir(test_class):
                if not attr_name.startswith("_"):
                    attr = getattr(test_class, attr_name)

                    if self._is_test_method(attr_name, attr):
                        # Adapt test methods
                        if inspect.iscoroutinefunction(attr):
                            adapted_method = self.method_adapter.adapt_async_method(
                                attr
                            )
                        else:
                            adapted_method = self.method_adapter.adapt_sync_method(attr)

                        fp_class_dict[attr_name] = adapted_method
                    elif callable(attr) and not inspect.ismethod(attr):
                        # Copy other methods as-is
                        fp_class_dict[attr_name] = attr
                    elif not callable(attr):
                        # Copy class attributes
                        fp_class_dict[attr_name] = attr

            # Add FP-specific helper methods
            fp_class_dict.update(self._create_fp_helpers())

            # Create new class
            fp_class = type(fp_class_name, (test_class,), fp_class_dict)

            migration_registry.register_migration(class_name, True)
            return fp_class

        except Exception as e:
            migration_registry.register_migration(class_name, False)
            migration_registry.report_compatibility_issue(
                f"Failed to migrate {class_name}: {e!s}"
            )
            return test_class

    def _is_test_method(self, name: str, attr: Any) -> bool:
        """Check if attribute is a test method."""
        return callable(attr) and (
            name.startswith("test_") or hasattr(attr, "_pytestmark")
        )

    def _create_fp_helpers(self) -> dict[str, Any]:
        """Create FP-specific helper methods for test class."""

        def assert_result_ok(self, result: Result, expected_value: Any = None):
            """Assert Result is Ok and optionally check value."""
            assert (
                hasattr(result, "is_ok") and result.is_ok()
            ), f"Expected Ok, got Err: {getattr(result, 'error', result)}"
            if expected_value is not None:
                assert result.unwrap() == expected_value

        def assert_result_err(self, result: Result, expected_error: Any = None):
            """Assert Result is Err and optionally check error."""
            assert (
                hasattr(result, "is_err") and result.is_err()
            ), f"Expected Err, got Ok: {getattr(result, 'value', result)}"
            if expected_error is not None:
                assert result.error == expected_error

        def assert_maybe_some(self, maybe: Maybe, expected_value: Any = None):
            """Assert Maybe is Some and optionally check value."""
            assert (
                hasattr(maybe, "is_some") and maybe.is_some()
            ), "Expected Some, got Nothing"
            if expected_value is not None:
                assert maybe.unwrap() == expected_value

        def assert_maybe_nothing(self, maybe: Maybe):
            """Assert Maybe is Nothing."""
            assert (
                hasattr(maybe, "is_nothing") and maybe.is_nothing()
            ), f"Expected Nothing, got Some: {getattr(maybe, 'value', maybe)}"

        def run_io(self, io: IO):
            """Run IO computation and return result."""
            return io.run()

        return {
            "assert_result_ok": assert_result_ok,
            "assert_result_err": assert_result_err,
            "assert_maybe_some": assert_maybe_some,
            "assert_maybe_nothing": assert_maybe_nothing,
            "run_io": run_io,
        }


# =============================================================================
# DUAL-MODE TEST RUNNER
# =============================================================================


class DualModeTestRunner:
    """Runner that executes tests in both imperative and FP modes."""

    def __init__(self, config: MigrationConfig = None):
        self.config = config or MigrationConfig()
        self.migration_adapter = TestClassMigrationAdapter(config)

    def create_dual_mode_class(self, test_class: type) -> type:
        """Create test class that runs in both modes."""
        if not self.config.enable_dual_mode:
            if self.config.enable_fp_mode:
                return self.migration_adapter.migrate_test_class(test_class)
            return test_class

        # Create FP version
        fp_class = self.migration_adapter.migrate_test_class(test_class)

        class_name = test_class.__name__
        dual_class_name = f"DualMode{class_name}"
        dual_class_dict = {}

        # Add both imperative and FP versions of test methods
        for attr_name in dir(test_class):
            if self._is_test_method(attr_name, getattr(test_class, attr_name)):
                # Add imperative version
                imperative_method = getattr(test_class, attr_name)
                dual_class_dict[f"{attr_name}_imperative"] = imperative_method

                # Add FP version
                fp_method = getattr(fp_class, attr_name)
                dual_class_dict[f"{attr_name}_fp"] = fp_method

                # Add comparison method
                dual_class_dict[f"{attr_name}_compare"] = (
                    self._create_comparison_method(
                        imperative_method, fp_method, attr_name
                    )
                )

        # Create dual-mode class
        dual_class = type(dual_class_name, (test_class,), dual_class_dict)
        return dual_class

    def _is_test_method(self, name: str, attr: Any) -> bool:
        """Check if attribute is a test method."""
        return callable(attr) and (
            name.startswith("test_") or hasattr(attr, "_pytestmark")
        )

    def _create_comparison_method(
        self, imperative_method: Callable, fp_method: Callable, method_name: str
    ) -> Callable:
        """Create method that compares results from both modes."""

        def comparison_method(self, *args, **kwargs):
            # Run imperative version
            try:
                imperative_result = imperative_method(self, *args, **kwargs)
                imperative_success = True
                imperative_error = None
            except Exception as e:
                imperative_result = None
                imperative_success = False
                imperative_error = e

            # Run FP version
            try:
                fp_result = fp_method(self, *args, **kwargs)
                fp_success = True
                fp_error = None
            except Exception as e:
                fp_result = None
                fp_success = False
                fp_error = e

            # Compare results
            if imperative_success and fp_success:
                # Both succeeded - compare results
                self._compare_results(imperative_result, fp_result, method_name)
            elif not imperative_success and not fp_success:
                # Both failed - compare errors
                self._compare_errors(imperative_error, fp_error, method_name)
            else:
                # One succeeded, one failed - report discrepancy
                pytest.fail(
                    f"Result discrepancy in {method_name}: "
                    f"imperative={'success' if imperative_success else 'failure'}, "
                    f"fp={'success' if fp_success else 'failure'}"
                )

        comparison_method.__name__ = f"{method_name}_compare"
        return comparison_method

    def _compare_results(
        self, imperative_result: Any, fp_result: Any, method_name: str
    ):
        """Compare results from imperative and FP modes."""
        # Convert FP result to imperative for comparison
        if hasattr(fp_result, "is_ok") and fp_result.is_ok():
            fp_unwrapped = fp_result.unwrap()
        elif hasattr(fp_result, "run"):
            fp_unwrapped = fp_result.run()
        else:
            fp_unwrapped = fp_result

        # Compare values
        if imperative_result != fp_unwrapped:
            migration_registry.report_compatibility_issue(
                f"Result mismatch in {method_name}: "
                f"imperative={imperative_result}, fp={fp_unwrapped}"
            )

            if self.config.validate_conversions:
                pytest.fail(
                    f"Result mismatch in {method_name}: "
                    f"imperative={imperative_result}, fp={fp_unwrapped}"
                )

    def _compare_errors(
        self, imperative_error: Exception, fp_error: Exception, method_name: str
    ):
        """Compare errors from imperative and FP modes."""
        if type(imperative_error) != type(fp_error):
            migration_registry.report_compatibility_issue(
                f"Error type mismatch in {method_name}: "
                f"imperative={type(imperative_error)}, fp={type(fp_error)}"
            )


# =============================================================================
# MOCK OBJECT ADAPTERS
# =============================================================================


class MockObjectAdapter:
    """Adapter for creating FP-compatible mock objects."""

    @staticmethod
    def adapt_exchange_mock(mock_exchange: Mock) -> Mock:
        """Adapt exchange mock for FP compatibility."""
        # Add FP methods to existing mock
        mock_exchange.supports_functional_operations = Mock(return_value=True)
        mock_exchange.get_functional_adapter = Mock(
            return_value=MockObjectAdapter._create_fp_adapter_mock()
        )

        # Wrap existing methods to return FP types
        original_get_balance = mock_exchange.get_balance
        mock_exchange.get_balance_fp = Mock(
            return_value=IO.pure(Ok(original_get_balance.return_value))
        )

        original_place_order = mock_exchange.place_order
        mock_exchange.place_order_fp = Mock(
            return_value=IO.pure(Ok(original_place_order.return_value))
        )

        return mock_exchange

    @staticmethod
    def _create_fp_adapter_mock() -> Mock:
        """Create FP adapter mock."""
        adapter = Mock()
        adapter.get_balance = Mock(return_value=IO.pure(Ok(Decimal("10000.00"))))
        adapter.place_order = Mock(return_value=IO.pure(Ok({"order_id": "test-123"})))
        adapter.cancel_order = Mock(return_value=IO.pure(Ok(True)))
        adapter.get_market_data = Mock(return_value=IO.pure(Ok({})))
        return adapter

    @staticmethod
    def adapt_strategy_mock(mock_strategy: Mock) -> Mock:
        """Adapt strategy mock for FP compatibility."""
        # Add FP methods
        mock_strategy.generate_signal_fp = Mock(
            return_value=IO.pure(Ok(Hold(reason="Test hold")))
        )
        mock_strategy.analyze_market_fp = Mock(return_value=IO.pure(Ok({})))

        return mock_strategy


# =============================================================================
# MIGRATION DECORATORS
# =============================================================================


def fp_migration(config: MigrationConfig = None):
    """Decorator to enable FP migration for test class."""

    def decorator(test_class: type) -> type:
        migration_config = config or MigrationConfig()

        if not FP_AVAILABLE:
            if migration_config.migration_warnings:
                pytest.warn(
                    f"FP types not available, skipping migration for {test_class.__name__}",
                    category=UserWarning,
                )
            return test_class

        adapter = TestClassMigrationAdapter(migration_config)
        return adapter.migrate_test_class(test_class)

    return decorator


def dual_mode_test(config: MigrationConfig = None):
    """Decorator to enable dual-mode testing (both imperative and FP)."""

    def decorator(test_class: type) -> type:
        migration_config = config or MigrationConfig()
        migration_config.enable_dual_mode = True

        if not FP_AVAILABLE:
            if migration_config.migration_warnings:
                pytest.warn(
                    f"FP types not available, running in imperative mode only for {test_class.__name__}",
                    category=UserWarning,
                )
            return test_class

        runner = DualModeTestRunner(migration_config)
        return runner.create_dual_mode_class(test_class)

    return decorator


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================


@contextmanager
def fp_test_mode(enable_fp: bool = True):
    """Context manager for temporarily enabling/disabling FP mode."""
    original_config = MigrationConfig()

    try:
        # Temporarily modify global config
        with patch.object(MigrationConfig, "enable_fp_mode", enable_fp):
            yield
    finally:
        pass  # Config automatically restored


@contextmanager
def migration_reporting():
    """Context manager for collecting migration reports."""
    migration_registry.migration_errors.clear()
    migration_registry.compatibility_issues.clear()

    try:
        yield migration_registry
    finally:
        report = migration_registry.get_migration_report()
        if report["migration_errors"] or report["compatibility_issues"]:
            print("\n--- Migration Report ---")
            print(f"Success Rate: {report['success_rate']:.2%}")
            if report["migration_errors"]:
                print("Migration Errors:")
                for error in report["migration_errors"]:
                    print(f"  - {error}")
            if report["compatibility_issues"]:
                print("Compatibility Issues:")
                for issue in report["compatibility_issues"]:
                    print(f"  - {issue}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "MigrationConfig",
    "MigrationRegistry",
    "migration_registry",
    # Type adapters
    "TypeConversionAdapter",
    # Method adapters
    "TestMethodAdapter",
    "TestClassMigrationAdapter",
    # Test runners
    "DualModeTestRunner",
    # Mock adapters
    "MockObjectAdapter",
    # Decorators
    "fp_migration",
    "dual_mode_test",
    # Context managers
    "fp_test_mode",
    "migration_reporting",
]
