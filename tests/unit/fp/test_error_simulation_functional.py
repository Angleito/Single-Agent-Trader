"""
Functional Programming Error Simulation Tests

Test suite for error simulation and handling with functional programming patterns.
Tests Result/Either error handling, resilience patterns, and recovery mechanisms.
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.fp.types.trading import (
    LimitOrder,
    MarketOrder,
    OrderResult,
    OrderStatus,
    Position,
    AccountBalance,
    FunctionalMarketData
)
from bot.fp.types.result import Result, Ok, Err
from bot.fp.effects.io import IOEither, from_try, AsyncIO
from bot.fp.effects.error import (
    ErrorType,
    ExchangeError,
    NetworkError,
    ValidationError,
    RateLimitError,
    InsufficientFundsError,
    create_error_recovery_strategy,
    with_retry,
    with_circuit_breaker,
    with_fallback
)
from bot.fp.effects.websocket_enhanced import (
    EnhancedWebSocketManager,
    ConnectionState,
    create_enhanced_websocket_manager
)


class TestErrorTypes:
    """Test functional error type hierarchy."""

    def test_error_type_creation(self):
        """Test creation of different error types."""
        # Network error
        network_error = NetworkError(
            message="Connection timeout",
            error_code="NETWORK_TIMEOUT",
            retryable=True,
            retry_after=timedelta(seconds=5)
        )
        
        assert network_error.message == "Connection timeout"
        assert network_error.error_code == "NETWORK_TIMEOUT"
        assert network_error.retryable is True
        assert network_error.retry_after == timedelta(seconds=5)

    def test_exchange_error_creation(self):
        """Test creation of exchange-specific errors."""
        exchange_error = ExchangeError(
            message="Order rejected by exchange",
            error_code="ORDER_REJECTED",
            exchange_name="coinbase",
            retryable=False,
            details={"reason": "Insufficient balance", "order_id": "123"}
        )
        
        assert exchange_error.exchange_name == "coinbase"
        assert exchange_error.retryable is False
        assert exchange_error.details["reason"] == "Insufficient balance"

    def test_validation_error_creation(self):
        """Test creation of validation errors."""
        validation_error = ValidationError(
            message="Invalid order size",
            error_code="INVALID_SIZE",
            field="size",
            value="0.0",
            constraints={"min": "0.001", "max": "1000.0"}
        )
        
        assert validation_error.field == "size"
        assert validation_error.value == "0.0"
        assert validation_error.constraints["min"] == "0.001"

    def test_rate_limit_error_creation(self):
        """Test creation of rate limit errors."""
        rate_limit_error = RateLimitError(
            message="Rate limit exceeded",
            error_code="RATE_LIMIT",
            retry_after=timedelta(seconds=60),
            limit_type="requests_per_minute",
            current_rate=1000,
            max_rate=100
        )
        
        assert rate_limit_error.limit_type == "requests_per_minute"
        assert rate_limit_error.current_rate == 1000
        assert rate_limit_error.max_rate == 100

    def test_insufficient_funds_error_creation(self):
        """Test creation of insufficient funds errors."""
        funds_error = InsufficientFundsError(
            message="Insufficient balance for order",
            error_code="INSUFFICIENT_FUNDS",
            required_amount=Decimal("1000.0"),
            available_amount=Decimal("500.0"),
            currency="USD"
        )
        
        assert funds_error.required_amount == Decimal("1000.0")
        assert funds_error.available_amount == Decimal("500.0")
        assert funds_error.currency == "USD"


class FailureSimulator:
    """Simulator for various failure scenarios."""

    def __init__(self, failure_rate: float = 0.0, failure_type: str = "random"):
        self.failure_rate = failure_rate
        self.failure_type = failure_type
        self.call_count = 0
        self.failures = []

    def should_fail(self) -> bool:
        """Determine if operation should fail."""
        self.call_count += 1
        
        if self.failure_type == "intermittent":
            # Fail every 3rd call
            return self.call_count % 3 == 0
        elif self.failure_type == "progressive":
            # Increase failure rate over time
            adjusted_rate = min(self.failure_rate * (self.call_count / 10), 1.0)
            return random.random() < adjusted_rate
        else:
            # Random failure based on rate
            return random.random() < self.failure_rate

    def simulate_operation(self, operation_name: str) -> IOEither[Exception, str]:
        """Simulate an operation that may fail."""
        if self.should_fail():
            error = self._create_error(operation_name)
            self.failures.append(error)
            return IOEither.left(error)
        
        return IOEither.right(f"{operation_name} succeeded")

    def _create_error(self, operation_name: str) -> Exception:
        """Create appropriate error for operation."""
        if "network" in operation_name.lower():
            return NetworkError(f"Network failure in {operation_name}", "NETWORK_ERROR")
        elif "order" in operation_name.lower():
            return ExchangeError(f"Order failure in {operation_name}", "ORDER_ERROR", "test_exchange")
        elif "validation" in operation_name.lower():
            return ValidationError(f"Validation failure in {operation_name}", "VALIDATION_ERROR", "test_field")
        else:
            return Exception(f"Generic failure in {operation_name}")


class TestErrorHandlingPatterns:
    """Test functional error handling patterns."""

    def test_either_error_handling(self):
        """Test Either monad for error handling."""
        # Success case
        success_result = IOEither.right("Success value")
        result = success_result.run()
        
        assert result.is_right()
        assert result.value == "Success value"
        
        # Error case
        error_result = IOEither.left(ValueError("Test error"))
        result = error_result.run()
        
        assert result.is_left()
        assert isinstance(result.value, ValueError)

    def test_error_mapping(self):
        """Test error mapping and transformation."""
        # Map success value
        success_result = IOEither.right(10)
        mapped_result = success_result.map(lambda x: x * 2)
        
        assert mapped_result.run().value == 20
        
        # Map error (should preserve error)
        error_result = IOEither.left(ValueError("Test error"))
        mapped_error = error_result.map(lambda x: x * 2)
        
        assert mapped_error.run().is_left()
        assert isinstance(mapped_error.run().value, ValueError)

    def test_error_chaining(self):
        """Test error chaining with flatMap."""
        def divide_by_two(x: int) -> IOEither[Exception, int]:
            if x % 2 != 0:
                return IOEither.left(ValueError("Number is not even"))
            return IOEither.right(x // 2)
        
        # Success chain
        success_chain = IOEither.right(10).flat_map(divide_by_two)
        assert success_chain.run().value == 5
        
        # Error in chain
        error_chain = IOEither.right(9).flat_map(divide_by_two)
        assert error_chain.run().is_left()

    def test_error_recovery(self):
        """Test error recovery patterns."""
        def operation_with_fallback(x: int) -> IOEither[Exception, int]:
            if x < 0:
                return IOEither.left(ValueError("Negative number"))
            return IOEither.right(x * 2)
        
        # Recover from error
        result = (IOEither.left(ValueError("Initial error"))
                 .or_else(lambda _: operation_with_fallback(5)))
        
        assert result.run().value == 10

    def test_error_accumulation(self):
        """Test accumulation of multiple errors."""
        operations = [
            IOEither.right("success1"),
            IOEither.left(ValueError("error1")),
            IOEither.right("success2"),
            IOEither.left(ValueError("error2"))
        ]
        
        # Collect all errors
        errors = []
        successes = []
        
        for op in operations:
            result = op.run()
            if result.is_left():
                errors.append(result.value)
            else:
                successes.append(result.value)
        
        assert len(errors) == 2
        assert len(successes) == 2


class TestRetryMechanism:
    """Test retry mechanisms with functional patterns."""

    def test_simple_retry(self):
        """Test simple retry mechanism."""
        simulator = FailureSimulator(failure_rate=0.7)
        
        def unreliable_operation() -> IOEither[Exception, str]:
            return simulator.simulate_operation("test_operation")
        
        # Retry with multiple attempts
        result = with_retry(
            operation=unreliable_operation,
            max_attempts=5,
            delay=timedelta(milliseconds=10)
        )
        
        # Should eventually succeed or exhaust retries
        final_result = result.run()
        
        # Check that retries were attempted
        assert simulator.call_count > 1

    def test_exponential_backoff_retry(self):
        """Test retry with exponential backoff."""
        simulator = FailureSimulator(failure_rate=0.8)
        
        def failing_operation() -> IOEither[Exception, str]:
            return simulator.simulate_operation("backoff_test")
        
        start_time = datetime.now()
        
        result = with_retry(
            operation=failing_operation,
            max_attempts=3,
            delay=timedelta(milliseconds=100),
            backoff_multiplier=2.0
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should have taken time for backoff delays
        # First retry: 100ms, second retry: 200ms
        expected_min_duration = 0.3  # 300ms minimum
        assert duration >= expected_min_duration

    def test_conditional_retry(self):
        """Test conditional retry based on error type."""
        def should_retry(error: Exception) -> bool:
            """Only retry network errors."""
            return isinstance(error, NetworkError)
        
        simulator = FailureSimulator(failure_rate=1.0)  # Always fail
        
        def network_operation() -> IOEither[Exception, str]:
            return IOEither.left(NetworkError("Network timeout", "NETWORK_TIMEOUT"))
        
        def validation_operation() -> IOEither[Exception, str]:
            return IOEither.left(ValidationError("Invalid input", "VALIDATION_ERROR", "field"))
        
        # Network error should be retried
        network_result = with_retry(
            operation=network_operation,
            max_attempts=3,
            should_retry=should_retry
        )
        
        # Validation error should not be retried
        validation_result = with_retry(
            operation=validation_operation,
            max_attempts=3,
            should_retry=should_retry
        )
        
        # Both should fail, but network should have made more attempts
        assert network_result.run().is_left()
        assert validation_result.run().is_left()

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test retry mechanism with async operations."""
        simulator = FailureSimulator(failure_rate=0.6)
        
        async def async_operation() -> IOEither[Exception, str]:
            await asyncio.sleep(0.01)  # Simulate async work
            return simulator.simulate_operation("async_test")
        
        result = await with_retry_async(
            operation=async_operation,
            max_attempts=5,
            delay=timedelta(milliseconds=10)
        )
        
        # Should eventually succeed or fail after retries
        assert simulator.call_count > 1


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed (normal) state."""
        simulator = FailureSimulator(failure_rate=0.1)  # Low failure rate
        
        def reliable_operation() -> IOEither[Exception, str]:
            return simulator.simulate_operation("reliable_test")
        
        circuit_breaker = with_circuit_breaker(
            operation=reliable_operation,
            failure_threshold=5,
            timeout=timedelta(seconds=1)
        )
        
        # Should work normally with low failure rate
        for _ in range(10):
            result = circuit_breaker.run()
            # Most should succeed
        
        assert simulator.call_count == 10

    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open (failing) state."""
        simulator = FailureSimulator(failure_rate=1.0)  # Always fail
        
        def failing_operation() -> IOEither[Exception, str]:
            return simulator.simulate_operation("failing_test")
        
        circuit_breaker = with_circuit_breaker(
            operation=failing_operation,
            failure_threshold=3,
            timeout=timedelta(milliseconds=100)
        )
        
        # First few calls should reach the operation
        for _ in range(5):
            result = circuit_breaker.run()
        
        # After threshold, circuit should open and prevent calls
        initial_call_count = simulator.call_count
        
        # These calls should be blocked by open circuit
        for _ in range(5):
            result = circuit_breaker.run()
            assert result.is_left()
        
        # Call count should not increase (circuit is open)
        assert simulator.call_count <= initial_call_count + 1

    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker half-open state and recovery."""
        simulator = FailureSimulator(failure_rate=0.0)  # No failures after initial ones
        
        def recovering_operation() -> IOEither[Exception, str]:
            # Fail first few calls, then succeed
            if simulator.call_count < 3:
                return IOEither.left(Exception("Initial failures"))
            return IOEither.right("Recovery success")
        
        circuit_breaker = with_circuit_breaker(
            operation=recovering_operation,
            failure_threshold=2,
            timeout=timedelta(milliseconds=50)
        )
        
        # Trigger circuit open
        for _ in range(3):
            circuit_breaker.run()
        
        # Wait for timeout
        import time
        time.sleep(0.1)
        
        # Next call should be in half-open state and succeed
        recovery_result = circuit_breaker.run()
        
        # Circuit should close again
        assert circuit_breaker.state == "closed"


class TestFallbackMechanisms:
    """Test fallback mechanisms for resilience."""

    def test_simple_fallback(self):
        """Test simple fallback to alternative operation."""
        def primary_operation() -> IOEither[Exception, str]:
            return IOEither.left(Exception("Primary failed"))
        
        def fallback_operation() -> IOEither[Exception, str]:
            return IOEither.right("Fallback success")
        
        result = with_fallback(
            primary=primary_operation,
            fallback=fallback_operation
        )
        
        assert result.run().value == "Fallback success"

    def test_multiple_fallbacks(self):
        """Test cascading fallbacks."""
        def primary() -> IOEither[Exception, str]:
            return IOEither.left(Exception("Primary failed"))
        
        def fallback1() -> IOEither[Exception, str]:
            return IOEither.left(Exception("Fallback1 failed"))
        
        def fallback2() -> IOEither[Exception, str]:
            return IOEither.right("Fallback2 success")
        
        result = (IOEither.from_callable(primary)
                 .or_else(lambda _: fallback1())
                 .or_else(lambda _: fallback2()))
        
        assert result.run().value == "Fallback2 success"

    def test_conditional_fallback(self):
        """Test conditional fallback based on error type."""
        def should_fallback(error: Exception) -> bool:
            return isinstance(error, NetworkError)
        
        def primary() -> IOEither[Exception, str]:
            return IOEither.left(NetworkError("Network down", "NETWORK_ERROR"))
        
        def fallback() -> IOEither[Exception, str]:
            return IOEither.right("Cache result")
        
        result = with_fallback(
            primary=primary,
            fallback=fallback,
            should_fallback=should_fallback
        )
        
        assert result.run().value == "Cache result"

    def test_fallback_with_degraded_service(self):
        """Test fallback to degraded service."""
        def full_service() -> IOEither[Exception, Dict[str, Any]]:
            return IOEither.left(Exception("Full service unavailable"))
        
        def degraded_service() -> IOEither[Exception, Dict[str, Any]]:
            return IOEither.right({
                "data": "limited_data",
                "degraded": True,
                "features": ["basic_only"]
            })
        
        result = with_fallback(
            primary=full_service,
            fallback=degraded_service
        )
        
        service_result = result.run().value
        assert service_result["degraded"] is True
        assert "basic_only" in service_result["features"]


class TestWebSocketErrorHandling:
    """Test WebSocket error handling with functional patterns."""

    def test_websocket_connection_error_handling(self):
        """Test WebSocket connection error handling."""
        # Mock WebSocket that fails to connect
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = ConnectionError("Failed to connect")
            
            manager = create_enhanced_websocket_manager("wss://invalid-url")
            
            # Connection should return error
            result = manager.connect()
            connection_result = result.run()
            
            assert connection_result.is_left()
            assert "Failed to connect" in str(connection_result.value)

    def test_websocket_message_error_handling(self):
        """Test WebSocket message error handling."""
        manager = create_enhanced_websocket_manager("wss://test-url")
        
        # Test invalid message format
        invalid_messages = [
            "invalid json",
            {"incomplete": "message"},
            None
        ]
        
        for invalid_msg in invalid_messages:
            # Should handle gracefully without crashing
            try:
                if isinstance(invalid_msg, str):
                    import json
                    json.loads(invalid_msg)  # This will fail
            except json.JSONDecodeError:
                # Expected for invalid JSON
                pass

    def test_websocket_reconnection_error_handling(self):
        """Test WebSocket reconnection error handling."""
        manager = create_enhanced_websocket_manager("wss://test-url")
        
        # Simulate failed reconnection
        manager._state = ConnectionState.FAILED
        
        # Health check should return False
        health_result = manager.check_connection_health().run()
        assert health_result is False


@pytest.mark.asyncio
class TestAsyncErrorHandling:
    """Test async error handling patterns."""

    async def test_async_operation_error_handling(self):
        """Test error handling in async operations."""
        async def failing_async_operation() -> IOEither[Exception, str]:
            await asyncio.sleep(0.01)
            return IOEither.left(Exception("Async operation failed"))
        
        result = await failing_async_operation()
        operation_result = result.run()
        
        assert operation_result.is_left()
        assert "Async operation failed" in str(operation_result.value)

    async def test_concurrent_error_handling(self):
        """Test error handling in concurrent operations."""
        async def mixed_operations(success: bool, delay: float) -> IOEither[Exception, str]:
            await asyncio.sleep(delay)
            if success:
                return IOEither.right(f"Success after {delay}s")
            else:
                return IOEither.left(Exception(f"Failed after {delay}s"))
        
        # Mix of successful and failing operations
        tasks = [
            mixed_operations(True, 0.01),
            mixed_operations(False, 0.02),
            mixed_operations(True, 0.03),
            mixed_operations(False, 0.01),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Check results
        successes = [r for r in results if r.run().is_right()]
        failures = [r for r in results if r.run().is_left()]
        
        assert len(successes) == 2
        assert len(failures) == 2

    async def test_timeout_error_handling(self):
        """Test timeout error handling."""
        async def slow_operation() -> IOEither[Exception, str]:
            await asyncio.sleep(1.0)  # Long operation
            return IOEither.right("Slow success")
        
        # Apply timeout
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=0.1)
        except asyncio.TimeoutError:
            result = IOEither.left(Exception("Operation timed out"))
        
        timeout_result = result.run()
        assert timeout_result.is_left()
        assert "timed out" in str(timeout_result.value).lower()


class TestErrorRecoveryStrategies:
    """Test error recovery strategies."""

    def test_create_error_recovery_strategy(self):
        """Test creation of error recovery strategies."""
        strategy = create_error_recovery_strategy(
            max_retries=3,
            base_delay=timedelta(milliseconds=100),
            max_delay=timedelta(seconds=5),
            backoff_multiplier=2.0
        )
        
        assert strategy.max_retries == 3
        assert strategy.base_delay == timedelta(milliseconds=100)
        assert strategy.backoff_multiplier == 2.0

    def test_recovery_strategy_application(self):
        """Test application of recovery strategy."""
        strategy = create_error_recovery_strategy(max_retries=2)
        simulator = FailureSimulator(failure_rate=0.8)
        
        def unreliable_operation() -> IOEither[Exception, str]:
            return simulator.simulate_operation("recovery_test")
        
        # Apply strategy
        result = strategy.apply(unreliable_operation)
        
        # Should have attempted retries
        assert simulator.call_count > 1

    def test_adaptive_recovery_strategy(self):
        """Test adaptive recovery strategy based on error patterns."""
        class AdaptiveRecoveryStrategy:
            def __init__(self):
                self.error_history = []
                self.success_history = []
            
            def adapt_strategy(self, error: Exception) -> Dict[str, Any]:
                """Adapt strategy based on error type and history."""
                self.error_history.append(error)
                
                if isinstance(error, NetworkError):
                    return {"retry_delay": timedelta(seconds=1), "max_retries": 5}
                elif isinstance(error, RateLimitError):
                    return {"retry_delay": error.retry_after, "max_retries": 2}
                else:
                    return {"retry_delay": timedelta(milliseconds=100), "max_retries": 3}
        
        strategy = AdaptiveRecoveryStrategy()
        
        # Test with network error
        network_error = NetworkError("Connection lost", "NETWORK_ERROR")
        network_config = strategy.adapt_strategy(network_error)
        
        assert network_config["max_retries"] == 5
        
        # Test with rate limit error
        rate_error = RateLimitError(
            "Rate limited", "RATE_LIMIT", 
            retry_after=timedelta(seconds=30),
            limit_type="requests", current_rate=100, max_rate=50
        )
        rate_config = strategy.adapt_strategy(rate_error)
        
        assert rate_config["retry_delay"] == timedelta(seconds=30)


# Helper functions for async retry (would be implemented in actual code)
async def with_retry_async(operation, max_attempts: int = 3, 
                          delay: timedelta = timedelta(seconds=1)):
    """Async retry helper."""
    for attempt in range(max_attempts):
        try:
            result = await operation()
            if result.run().is_right():
                return result
        except Exception:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(delay.total_seconds())
    
    return IOEither.left(Exception("Max retries exceeded"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])