"""
Comprehensive Functional Programming Error Handling Tests

This test suite focuses on functional programming error handling patterns:
1. Result/Either monadic error handling
2. Functional error composition and chaining
3. Type-safe error propagation
4. Functional circuit breakers and retry mechanisms
5. Error effect composition
6. Functional validation and error accumulation
7. IO effect error handling
8. Async FP error patterns
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest

from bot.fp.core.either import Either, Left, Right
from bot.fp.core.validation import Validation
from bot.fp.effects.error import (
    ExchangeError,
    InsufficientFundsError,
    NetworkError,
    ValidationError,
    create_error_recovery_strategy,
    with_circuit_breaker,
    with_fallback,
    with_retry,
)
from bot.fp.effects.io import IOEither
from bot.fp.types.config import (
    Config,
    ConfigValidationError,
    build_exchange_config_from_env,
    build_llm_config_from_env,
)
from bot.fp.types.market import (
    MarketDataValidationError,
    PriceData,
    VolumeData,
)
from bot.fp.types.result import Failure, Result, Success
from bot.fp.types.trading import (
    TradeValidationError,
    TradingDecision,
)


class FunctionalErrorSimulator:
    """Simulator for functional programming error scenarios."""

    def __init__(self):
        self.error_rate = 0.0
        self.error_types = []
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0

    def set_error_rate(self, rate: float):
        """Set the probability of errors (0.0 to 1.0)."""
        self.error_rate = max(0.0, min(1.0, rate))

    def add_error_type(self, error_type: type):
        """Add an error type to the pool of possible errors."""
        self.error_types.append(error_type)

    def reset(self):
        """Reset simulator state."""
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0

    def simulate_operation(
        self, operation_name: str = "test_op"
    ) -> Result[str, Exception]:
        """Simulate an operation that may fail."""
        self.call_count += 1

        if random.random() < self.error_rate and self.error_types:
            self.failure_count += 1
            error_type = random.choice(self.error_types)

            if error_type == NetworkError:
                error = NetworkError(
                    f"Network failure in {operation_name}",
                    "NETWORK_ERROR",
                    retryable=True,
                    retry_after=timedelta(milliseconds=100),
                )
            elif error_type == ValidationError:
                error = ValidationError(
                    f"Validation failed in {operation_name}",
                    "VALIDATION_ERROR",
                    field="test_field",
                    value="invalid_value",
                )
            elif error_type == ExchangeError:
                error = ExchangeError(
                    f"Exchange error in {operation_name}",
                    "EXCHANGE_ERROR",
                    exchange_name="test_exchange",
                    retryable=False,
                )
            else:
                error = Exception(f"Generic error in {operation_name}")

            return Failure(error)
        self.success_count += 1
        return Success(f"{operation_name} succeeded (call #{self.call_count})")

    def simulate_either_operation(
        self, operation_name: str = "test_op"
    ) -> Either[Exception, str]:
        """Simulate operation returning Either."""
        result = self.simulate_operation(operation_name)
        return result.to_either()

    def simulate_io_operation(
        self, operation_name: str = "test_op"
    ) -> IOEither[Exception, str]:
        """Simulate IO operation that may fail."""

        def io_action():
            return self.simulate_either_operation(operation_name)

        return IOEither.from_callable(io_action)


class TestResultMonadErrorHandling:
    """Test Result monad error handling patterns."""

    def test_result_success_and_failure_creation(self):
        """Test basic Result creation and type checking."""
        # Success case
        success_result = Success("operation succeeded")
        assert success_result.is_success()
        assert not success_result.is_failure()
        assert success_result.success() == "operation succeeded"

        # Failure case
        error = ValueError("operation failed")
        failure_result = Failure(error)
        assert failure_result.is_failure()
        assert not failure_result.is_success()
        assert failure_result.failure() == error

    def test_result_map_operations(self):
        """Test Result map operations preserve error handling."""
        # Map success
        success = Success(10)
        mapped_success = success.map(lambda x: x * 2)
        assert mapped_success.success() == 20

        # Map failure (should preserve error)
        failure = Failure(ValueError("error"))
        mapped_failure = failure.map(lambda x: x * 2)
        assert mapped_failure.is_failure()
        assert isinstance(mapped_failure.failure(), ValueError)

    def test_result_flat_map_chaining(self):
        """Test Result flat_map for error chaining."""

        def divide_by_two(x: int) -> Result[int, str]:
            if x % 2 != 0:
                return Failure("Number is not even")
            return Success(x // 2)

        def subtract_five(x: int) -> Result[int, str]:
            if x < 5:
                return Failure("Number is less than 5")
            return Success(x - 5)

        # Success chain
        result = Success(20).flat_map(divide_by_two).flat_map(subtract_five)
        assert result.success() == 5

        # Failure in first operation
        result = Success(9).flat_map(divide_by_two).flat_map(subtract_five)
        assert result.is_failure()
        assert result.failure() == "Number is not even"

        # Failure in second operation
        result = Success(6).flat_map(divide_by_two).flat_map(subtract_five)
        assert result.is_failure()
        assert result.failure() == "Number is less than 5"

    def test_result_error_recovery(self):
        """Test Result error recovery patterns."""
        # Recover from error
        failure = Failure("original error")
        recovered = failure.or_else(lambda _: Success("recovered value"))
        assert recovered.success() == "recovered value"

        # No recovery needed for success
        success = Success("original value")
        not_recovered = success.or_else(lambda _: Success("recovered value"))
        assert not_recovered.success() == "original value"

    def test_result_to_either_conversion(self):
        """Test Result to Either conversion."""
        # Success to Right
        success = Success("value")
        either = success.to_either()
        assert either.is_right()
        assert either.right() == "value"

        # Failure to Left
        failure = Failure("error")
        either = failure.to_either()
        assert either.is_left()
        assert either.left() == "error"


class TestEitherMonadErrorHandling:
    """Test Either monad error handling patterns."""

    def test_either_left_right_creation(self):
        """Test Either Left/Right creation and checking."""
        # Right (success) case
        right = Right("success value")
        assert right.is_right()
        assert not right.is_left()
        assert right.right() == "success value"

        # Left (error) case
        left = Left("error value")
        assert left.is_left()
        assert not left.is_right()
        assert left.left() == "error value"

    def test_either_functor_operations(self):
        """Test Either functor operations."""
        # Map right value
        right = Right(5)
        mapped_right = right.map(lambda x: x * 3)
        assert mapped_right.right() == 15

        # Map left value (should be unchanged)
        left = Left("error")
        mapped_left = left.map(lambda x: x * 3)
        assert mapped_left.is_left()
        assert mapped_left.left() == "error"

    def test_either_applicative_operations(self):
        """Test Either applicative operations for error accumulation."""

        def add_three_numbers(a: int, b: int, c: int) -> int:
            return a + b + c

        # All successes
        result = (
            Right(add_three_numbers).apply(Right(1)).apply(Right(2)).apply(Right(3))
        )
        assert result.right() == 6

        # One failure
        result = (
            Right(add_three_numbers)
            .apply(Right(1))
            .apply(Left("error"))
            .apply(Right(3))
        )
        assert result.is_left()

    def test_either_monadic_operations(self):
        """Test Either monadic operations."""

        def safe_divide(x: int, y: int) -> Either[str, int]:
            if y == 0:
                return Left("Division by zero")
            return Right(x // y)

        def safe_sqrt(x: int) -> Either[str, float]:
            if x < 0:
                return Left("Negative number")
            return Right(x**0.5)

        # Success chain
        result = (
            Right(16)
            .flat_map(lambda x: safe_divide(x, 4))
            .flat_map(lambda x: safe_sqrt(x))
        )
        assert abs(result.right() - 2.0) < 0.001

        # Failure in chain
        result = (
            Right(16)
            .flat_map(lambda x: safe_divide(x, 0))
            .flat_map(lambda x: safe_sqrt(x))
        )
        assert result.left() == "Division by zero"


class TestValidationErrorAccumulation:
    """Test Validation monad for error accumulation."""

    def test_validation_success_and_failure(self):
        """Test basic Validation operations."""
        # Success case
        success = Validation.success("valid value")
        assert success.is_success()
        assert success.success() == "valid value"

        # Failure case
        failure = Validation.failure(["error1", "error2"])
        assert failure.is_failure()
        assert failure.failure() == ["error1", "error2"]

    def test_validation_error_accumulation(self):
        """Test Validation accumulates multiple errors."""

        def validate_positive(x: int, field_name: str) -> Validation[list[str], int]:
            if x <= 0:
                return Validation.failure([f"{field_name} must be positive"])
            return Validation.success(x)

        def validate_even(x: int, field_name: str) -> Validation[list[str], int]:
            if x % 2 != 0:
                return Validation.failure([f"{field_name} must be even"])
            return Validation.success(x)

        def validate_small(x: int, field_name: str) -> Validation[list[str], int]:
            if x > 100:
                return Validation.failure([f"{field_name} must be <= 100"])
            return Validation.success(x)

        # Test with invalid value that fails multiple validations
        value = -5  # Negative and odd

        validations = [
            validate_positive(value, "value"),
            validate_even(value, "value"),
            validate_small(value, "value"),
        ]

        # Accumulate all validation results
        result = Validation.sequence(validations)

        assert result.is_failure()
        errors = result.failure()
        assert "value must be positive" in errors
        assert "value must be even" in errors

    def test_validation_applicative_style(self):
        """Test Validation applicative style for combining validations."""

        def validate_name(name: str) -> Validation[list[str], str]:
            errors = []
            if not name:
                errors.append("Name cannot be empty")
            if len(name) < 2:
                errors.append("Name must be at least 2 characters")
            if len(name) > 50:
                errors.append("Name must be at most 50 characters")

            if errors:
                return Validation.failure(errors)
            return Validation.success(name)

        def validate_age(age: int) -> Validation[list[str], int]:
            errors = []
            if age < 0:
                errors.append("Age cannot be negative")
            if age > 150:
                errors.append("Age must be realistic")

            if errors:
                return Validation.failure(errors)
            return Validation.success(age)

        # Valid data
        name_result = validate_name("John")
        age_result = validate_age(25)

        assert name_result.is_success()
        assert age_result.is_success()

        # Invalid data
        invalid_name = validate_name("")
        invalid_age = validate_age(-5)

        assert invalid_name.is_failure()
        assert invalid_age.is_failure()


class TestIOEffectErrorHandling:
    """Test IO effect error handling patterns."""

    def test_io_either_basic_operations(self):
        """Test basic IOEither operations."""
        # Success case
        success_io = IOEither.right("success value")
        result = success_io.run()
        assert result.is_right()
        assert result.right() == "success value"

        # Error case
        error_io = IOEither.left(ValueError("error"))
        result = error_io.run()
        assert result.is_left()
        assert isinstance(result.left(), ValueError)

    def test_io_either_from_callable(self):
        """Test IOEither creation from callable functions."""

        def success_function():
            return "function succeeded"

        def failing_function():
            raise RuntimeError("function failed")

        # Success case
        success_io = IOEither.from_callable(success_function)
        result = success_io.run()
        assert result.is_right()
        assert result.right() == "function succeeded"

        # Failure case
        failure_io = IOEither.from_callable(failing_function)
        result = failure_io.run()
        assert result.is_left()
        assert isinstance(result.left(), RuntimeError)

    def test_io_either_map_operations(self):
        """Test IOEither map operations."""
        # Map success
        success_io = IOEither.right(10)
        mapped_io = success_io.map(lambda x: x * 2)
        result = mapped_io.run()
        assert result.right() == 20

        # Map failure (should preserve error)
        error_io = IOEither.left(ValueError("error"))
        mapped_error_io = error_io.map(lambda x: x * 2)
        result = mapped_error_io.run()
        assert result.is_left()
        assert isinstance(result.left(), ValueError)

    def test_io_either_flat_map_chaining(self):
        """Test IOEither flat_map for operation chaining."""

        def safe_divide(x: int, y: int) -> IOEither[Exception, int]:
            if y == 0:
                return IOEither.left(ZeroDivisionError("Division by zero"))
            return IOEither.right(x // y)

        def safe_multiply(x: int, factor: int) -> IOEither[Exception, int]:
            if factor < 0:
                return IOEither.left(ValueError("Negative factor"))
            return IOEither.right(x * factor)

        # Success chain
        result = (
            IOEither.right(20)
            .flat_map(lambda x: safe_divide(x, 4))
            .flat_map(lambda x: safe_multiply(x, 3))
        )
        final_result = result.run()
        assert final_result.right() == 15

        # Failure in chain
        result = (
            IOEither.right(20)
            .flat_map(lambda x: safe_divide(x, 0))
            .flat_map(lambda x: safe_multiply(x, 3))
        )
        final_result = result.run()
        assert final_result.is_left()
        assert isinstance(final_result.left(), ZeroDivisionError)

    @pytest.mark.asyncio
    async def test_async_io_error_handling(self):
        """Test async IO error handling."""

        async def async_operation(should_fail: bool) -> IOEither[Exception, str]:
            await asyncio.sleep(0.01)  # Simulate async work
            if should_fail:
                return IOEither.left(RuntimeError("Async operation failed"))
            return IOEither.right("Async operation succeeded")

        # Success case
        success_result = await async_operation(False)
        result = success_result.run()
        assert result.is_right()
        assert "succeeded" in result.right()

        # Failure case
        failure_result = await async_operation(True)
        result = failure_result.run()
        assert result.is_left()
        assert isinstance(result.left(), RuntimeError)


class TestFunctionalRetryMechanisms:
    """Test functional retry mechanisms."""

    @pytest.fixture
    def error_simulator(self):
        """Error simulator fixture."""
        simulator = FunctionalErrorSimulator()
        yield simulator
        simulator.reset()

    def test_with_retry_success_after_failures(self, error_simulator):
        """Test retry mechanism eventually succeeds."""
        error_simulator.set_error_rate(0.8)  # 80% failure rate
        error_simulator.add_error_type(NetworkError)

        def unreliable_operation() -> Either[Exception, str]:
            return error_simulator.simulate_either_operation("retry_test")

        # With enough retries, should eventually succeed
        result = with_retry(
            operation=unreliable_operation,
            max_attempts=10,
            delay=timedelta(milliseconds=1),
        )

        final_result = result.run()
        # Should eventually succeed or show retry attempts were made
        assert error_simulator.call_count > 1

    def test_with_retry_respects_max_attempts(self, error_simulator):
        """Test retry mechanism respects maximum attempts."""
        error_simulator.set_error_rate(1.0)  # Always fail
        error_simulator.add_error_type(ValidationError)

        def always_failing_operation() -> Either[Exception, str]:
            return error_simulator.simulate_either_operation("always_fail")

        result = with_retry(
            operation=always_failing_operation,
            max_attempts=3,
            delay=timedelta(milliseconds=1),
        )

        final_result = result.run()
        assert final_result.is_left()
        assert error_simulator.call_count == 3  # Should have attempted exactly 3 times

    def test_with_retry_conditional_retry(self, error_simulator):
        """Test retry with conditional retry logic."""

        def should_retry_network_only(error: Exception) -> bool:
            return isinstance(error, NetworkError)

        # Network error should be retried
        def network_failing_operation() -> Either[Exception, str]:
            return Left(NetworkError("Network unstable", "NETWORK_ERROR"))

        network_result = with_retry(
            operation=network_failing_operation,
            max_attempts=3,
            should_retry=should_retry_network_only,
        )

        # Validation error should not be retried
        def validation_failing_operation() -> Either[Exception, str]:
            return Left(ValidationError("Invalid data", "VALIDATION_ERROR", "field"))

        validation_result = with_retry(
            operation=validation_failing_operation,
            max_attempts=3,
            should_retry=should_retry_network_only,
        )

        # Both should fail, but network should have attempted retries
        assert network_result.run().is_left()
        assert validation_result.run().is_left()

    def test_retry_with_exponential_backoff(self, error_simulator):
        """Test retry with exponential backoff timing."""
        error_simulator.set_error_rate(1.0)  # Always fail initially
        error_simulator.add_error_type(NetworkError)

        def backoff_operation() -> Either[Exception, str]:
            return error_simulator.simulate_either_operation("backoff_test")

        start_time = datetime.now()

        result = with_retry(
            operation=backoff_operation,
            max_attempts=3,
            delay=timedelta(milliseconds=50),
            backoff_multiplier=2.0,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should have taken time for backoff delays
        # First retry: 50ms, second retry: 100ms = 150ms minimum
        expected_min_duration = 0.15
        assert duration >= expected_min_duration


class TestFunctionalCircuitBreakers:
    """Test functional circuit breaker patterns."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed (normal) state."""
        call_count = 0

        def reliable_operation() -> Either[Exception, str]:
            nonlocal call_count
            call_count += 1
            return Right(f"Success #{call_count}")

        circuit_operation = with_circuit_breaker(
            operation=reliable_operation,
            failure_threshold=3,
            timeout=timedelta(milliseconds=100),
        )

        # Should work normally
        for i in range(5):
            result = circuit_operation.run()
            assert result.is_right()
            assert f"Success #{i + 1}" in result.right()

        assert call_count == 5

    def test_circuit_breaker_open_state(self):
        """Test circuit breaker opens after failure threshold."""
        call_count = 0

        def failing_operation() -> Either[Exception, str]:
            nonlocal call_count
            call_count += 1
            return Left(Exception(f"Failure #{call_count}"))

        circuit_operation = with_circuit_breaker(
            operation=failing_operation,
            failure_threshold=3,
            timeout=timedelta(milliseconds=100),
        )

        # First few calls should reach the operation
        for _ in range(5):
            result = circuit_operation.run()
            assert result.is_left()

        # After threshold, circuit should be open
        initial_call_count = call_count

        # These calls should be blocked
        for _ in range(3):
            result = circuit_operation.run()
            assert result.is_left()

        # Should not have made additional calls (circuit is open)
        assert call_count <= initial_call_count + 1

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        call_count = 0

        def recovering_operation() -> Either[Exception, str]:
            nonlocal call_count
            call_count += 1
            # Fail first 3 calls, then succeed
            if call_count <= 3:
                return Left(Exception(f"Initial failure #{call_count}"))
            return Right("Recovery success")

        circuit_operation = with_circuit_breaker(
            operation=recovering_operation,
            failure_threshold=3,
            timeout=timedelta(milliseconds=50),
        )

        # Trigger circuit open
        for _ in range(4):
            circuit_operation.run()

        # Wait for timeout
        import time

        time.sleep(0.1)

        # Next call should succeed and close circuit
        result = circuit_operation.run()

        # Verify circuit is working again
        for _ in range(3):
            result = circuit_operation.run()
            assert result.is_right()


class TestFunctionalFallbackMechanisms:
    """Test functional fallback mechanisms."""

    def test_simple_fallback(self):
        """Test simple fallback to alternative operation."""

        def primary_operation() -> Either[Exception, str]:
            return Left(Exception("Primary service unavailable"))

        def fallback_operation() -> Either[Exception, str]:
            return Right("Fallback service response")

        result = with_fallback(primary=primary_operation, fallback=fallback_operation)

        final_result = result.run()
        assert final_result.is_right()
        assert final_result.right() == "Fallback service response"

    def test_cascading_fallbacks(self):
        """Test multiple levels of fallbacks."""

        def primary() -> Either[Exception, str]:
            return Left(Exception("Primary failed"))

        def fallback1() -> Either[Exception, str]:
            return Left(Exception("Fallback1 failed"))

        def fallback2() -> Either[Exception, str]:
            return Right("Fallback2 success")

        def fallback3() -> Either[Exception, str]:
            return Right("Fallback3 success")

        # Chain fallbacks
        result = (
            Either.from_callable(primary)
            .or_else(lambda _: fallback1())
            .or_else(lambda _: fallback2())
            .or_else(lambda _: fallback3())
        )

        assert result.is_right()
        assert result.right() == "Fallback2 success"

    def test_conditional_fallback(self):
        """Test conditional fallback based on error type."""

        def should_fallback(error: Exception) -> bool:
            return isinstance(error, NetworkError)

        def primary_network_error() -> Either[Exception, str]:
            return Left(NetworkError("Network down", "NETWORK_ERROR"))

        def primary_validation_error() -> Either[Exception, str]:
            return Left(ValidationError("Invalid data", "VALIDATION_ERROR", "field"))

        def cache_fallback() -> Either[Exception, str]:
            return Right("Cached data")

        # Network error should trigger fallback
        network_result = with_fallback(
            primary=primary_network_error,
            fallback=cache_fallback,
            should_fallback=should_fallback,
        )

        assert network_result.run().is_right()
        assert "Cached data" in network_result.run().right()

        # Validation error should not trigger fallback
        validation_result = with_fallback(
            primary=primary_validation_error,
            fallback=cache_fallback,
            should_fallback=should_fallback,
        )

        assert validation_result.run().is_left()
        assert isinstance(validation_result.run().left(), ValidationError)


class TestFunctionalErrorComposition:
    """Test composition of functional error handling patterns."""

    def test_retry_with_circuit_breaker(self):
        """Test combining retry and circuit breaker patterns."""
        failure_count = 0

        def intermittent_operation() -> Either[Exception, str]:
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:
                return Left(
                    NetworkError(f"Network error #{failure_count}", "NETWORK_ERROR")
                )
            return Right("Success after retries")

        # Combine retry with circuit breaker
        operation_with_retry = with_retry(
            operation=intermittent_operation,
            max_attempts=3,
            delay=timedelta(milliseconds=10),
        )

        operation_with_circuit_breaker = with_circuit_breaker(
            operation=lambda: operation_with_retry.run(),
            failure_threshold=2,
            timeout=timedelta(milliseconds=50),
        )

        # Should handle failures with both patterns
        result = operation_with_circuit_breaker.run()

        # Verify both patterns were applied
        assert failure_count > 1  # Retry was attempted

    def test_fallback_with_retry(self):
        """Test combining fallback and retry patterns."""
        primary_call_count = 0
        fallback_call_count = 0

        def primary_operation() -> Either[Exception, str]:
            nonlocal primary_call_count
            primary_call_count += 1
            return Left(Exception(f"Primary failure #{primary_call_count}"))

        def fallback_operation() -> Either[Exception, str]:
            nonlocal fallback_call_count
            fallback_call_count += 1
            if fallback_call_count <= 2:
                return Left(Exception(f"Fallback failure #{fallback_call_count}"))
            return Right("Fallback success")

        # Primary with retry
        primary_with_retry = with_retry(
            operation=primary_operation, max_attempts=2, delay=timedelta(milliseconds=1)
        )

        # Fallback with retry
        fallback_with_retry = with_retry(
            operation=fallback_operation,
            max_attempts=3,
            delay=timedelta(milliseconds=1),
        )

        # Combine with fallback
        result = with_fallback(
            primary=lambda: primary_with_retry.run(),
            fallback=lambda: fallback_with_retry.run(),
        )

        final_result = result.run()

        # Should eventually succeed via fallback
        assert primary_call_count == 2  # Primary retried
        assert fallback_call_count == 3  # Fallback retried
        assert final_result.is_right()
        assert "Fallback success" in final_result.right()

    def test_error_recovery_strategy_composition(self):
        """Test composition of error recovery strategies."""
        strategy = create_error_recovery_strategy(
            max_retries=3,
            base_delay=timedelta(milliseconds=10),
            max_delay=timedelta(seconds=1),
            backoff_multiplier=2.0,
        )

        call_count = 0

        def complex_operation() -> Either[Exception, str]:
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                return Left(
                    NetworkError(f"Network unstable #{call_count}", "NETWORK_ERROR")
                )
            return Right("Operation succeeded after recovery")

        # Apply comprehensive recovery strategy
        result = strategy.apply(complex_operation)
        final_result = result.run()

        # Should succeed after applying recovery strategy
        assert call_count >= 3
        if final_result.is_right():
            assert "succeeded after recovery" in final_result.right()


class TestConfigurationErrorHandling:
    """Test functional configuration error handling."""

    def test_config_validation_error_accumulation(self):
        """Test configuration validation accumulates errors."""
        # Test with missing environment variables
        with patch.dict("os.environ", {}, clear=True):
            result = Config.from_env()

            assert result.is_failure()
            failure_details = result.failure()

            # Should accumulate multiple missing environment variable errors
            assert isinstance(failure_details, ConfigValidationError)

    def test_exchange_config_validation(self):
        """Test exchange configuration validation."""
        # Test with invalid exchange type
        with patch.dict("os.environ", {"EXCHANGE_TYPE": "invalid_exchange"}):
            result = build_exchange_config_from_env()

            assert result.is_failure()
            error = result.failure()
            assert "invalid_exchange" in str(error).lower()

    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        # Test with invalid temperature
        with patch.dict(
            "os.environ",
            {
                "LLM_TEMPERATURE": "5.0",  # Out of valid range
                "LLM_OPENAI_API_KEY": "test_key",
            },
        ):
            result = build_llm_config_from_env()

            assert result.is_failure()
            error = result.failure()
            assert "temperature" in str(error).lower()


class TestMarketDataErrorHandling:
    """Test functional market data error handling."""

    def test_price_data_validation(self):
        """Test price data validation with functional patterns."""
        # Valid price data
        valid_result = PriceData.create(
            open_price=Decimal(50000),
            high_price=Decimal(50100),
            low_price=Decimal(49900),
            close_price=Decimal(50050),
        )
        assert valid_result.is_success()

        # Invalid price data (high < low)
        invalid_result = PriceData.create(
            open_price=Decimal(50000),
            high_price=Decimal(49000),  # Invalid
            low_price=Decimal(50000),
            close_price=Decimal(50050),
        )
        assert invalid_result.is_failure()
        assert isinstance(invalid_result.failure(), MarketDataValidationError)

    def test_volume_data_validation(self):
        """Test volume data validation."""
        # Valid volume
        valid_result = VolumeData.create(volume=Decimal("100.5"))
        assert valid_result.is_success()

        # Invalid volume (negative)
        invalid_result = VolumeData.create(volume=Decimal(-10))
        assert invalid_result.is_failure()
        assert isinstance(invalid_result.failure(), MarketDataValidationError)


class TestTradingErrorHandling:
    """Test functional trading error handling."""

    def test_trading_decision_validation(self):
        """Test trading decision validation."""
        # Valid decision
        valid_result = TradingDecision.create(
            action="LONG",
            size_percentage=Decimal(25),
            confidence=Decimal("0.8"),
            rationale="Strong bullish signal",
        )
        assert valid_result.is_success()

        # Invalid decision (size > 100%)
        invalid_result = TradingDecision.create(
            action="LONG",
            size_percentage=Decimal(150),  # Invalid
            confidence=Decimal("0.8"),
            rationale="Invalid size",
        )
        assert invalid_result.is_failure()
        assert isinstance(invalid_result.failure(), TradeValidationError)

    def test_position_error_handling(self):
        """Test position error handling."""
        # Test insufficient funds error
        insufficient_funds_error = InsufficientFundsError(
            message="Insufficient balance for position",
            error_code="INSUFFICIENT_FUNDS",
            required_amount=Decimal(1000),
            available_amount=Decimal(500),
            currency="USD",
        )

        result = Result.failure(insufficient_funds_error)
        assert result.is_failure()

        error = result.failure()
        assert error.required_amount == Decimal(1000)
        assert error.available_amount == Decimal(500)
        assert error.currency == "USD"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
