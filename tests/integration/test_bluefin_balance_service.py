"""
Bluefin balance service integration tests.

This module tests the Bluefin SDK service integration with focus on:
- Service startup and health checks
- Balance retrieval with retry logic
- Error handling and recovery scenarios
- Performance benchmarks for balance operations
- Service resilience and circuit breaker functionality
"""

import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from aiohttp import ClientError, ClientTimeout

from services.bluefin_sdk_service import (
    BluefinAPIError,
    BluefinConnectionError,
    BluefinSDKService,
    BluefinServiceError,
)


class TestBluefinBalanceServiceIntegration:
    """Test Bluefin service integration for balance operations."""

    @pytest.fixture
    def mock_bluefin_client(self):
        """Create mock Bluefin client for testing."""
        client = Mock()
        client.initialized = True
        client.is_initialized.return_value = True

        # Mock account balance response
        client.get_account_balance.return_value = {
            "status": "success",
            "data": {
                "availableBalance": "10000.50",
                "marginBalance": "8500.25",
                "positionBalance": "1500.25",
                "equity": "10000.50",
                "totalBalance": "10000.50",
                "crossMarginAvailable": "8500.25",
                "crossMarginUsed": "1500.25",
            },
        }

        # Mock position data
        client.get_user_positions.return_value = {
            "status": "success",
            "data": [
                {
                    "symbol": "SUI-PERP",
                    "side": "LONG",
                    "size": "100.5",
                    "entryPrice": "3.45",
                    "markPrice": "3.50",
                    "unrealizedPnl": "5.025",
                    "marginRequired": "34.5",
                }
            ],
        }

        return client

    @pytest.fixture
    async def bluefin_service(self, mock_bluefin_client):
        """Create Bluefin service with mocked client."""
        service = BluefinSDKService()

        # Mock environment variables
        with patch.dict(
            "os.environ",
            {"BLUEFIN_PRIVATE_KEY": "test_private_key", "BLUEFIN_NETWORK": "testnet"},
        ):
            # Mock the client initialization
            with patch(
                "services.bluefin_sdk_service.BluefinClient",
                return_value=mock_bluefin_client,
            ):
                await service.initialize()

        yield service

        # Cleanup
        if hasattr(service, "client") and service.client:
            service.circuit_state = "CLOSED"  # Reset circuit breaker

    @pytest.mark.asyncio
    async def test_service_startup_and_health_checks(self, mock_bluefin_client):
        """Test service startup sequence and health monitoring."""
        service = BluefinSDKService()

        # Test initial state
        assert not service.initialized
        assert service.client is None
        assert service.circuit_state == "CLOSED"

        # Mock successful initialization
        with (
            patch.dict(
                "os.environ",
                {
                    "BLUEFIN_PRIVATE_KEY": "test_private_key",
                    "BLUEFIN_NETWORK": "testnet",
                },
            ),
            patch(
                "services.bluefin_sdk_service.BluefinClient",
                return_value=mock_bluefin_client,
            ),
        ):
            await service.initialize()

            # Verify initialization
            assert service.initialized
            assert service.client is not None
            assert service.network == "testnet"

            # Test health check
            health_status = await service.health_check()

            assert health_status["status"] == "healthy"
            assert health_status["client_initialized"] is True
            assert health_status["circuit_state"] == "CLOSED"
            assert "network" in health_status
            assert "uptime_seconds" in health_status

    @pytest.mark.asyncio
    async def test_balance_retrieval_with_retry_logic(self, bluefin_service):
        """Test balance retrieval with retry mechanisms."""
        # Test successful balance retrieval
        balance_data = await bluefin_service.get_account_balance()

        assert balance_data["status"] == "success"
        assert "availableBalance" in balance_data["data"]
        assert "marginBalance" in balance_data["data"]
        assert "equity" in balance_data["data"]

        # Test retry logic with transient failures
        call_count = 0
        original_method = bluefin_service.client.get_account_balance

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 calls
                raise ClientError("Temporary network error")
            return original_method()

        bluefin_service.client.get_account_balance = failing_then_success

        # Should succeed after retries
        balance_data = await bluefin_service.get_account_balance()
        assert balance_data["status"] == "success"
        assert call_count == 3  # Should have retried twice

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_scenarios(self, bluefin_service):
        """Test comprehensive error handling and recovery."""

        # Test API error scenarios
        error_scenarios = [
            # Network timeout
            ("timeout", ClientTimeout(), "timeout"),
            # Connection error
            ("connection", ClientError("Connection failed"), "connection"),
            # Server error
            ("server_error", Exception("Internal server error"), "server_error"),
        ]

        for _scenario_name, exception, expected_failure_type in error_scenarios:
            # Reset service state
            bluefin_service.failure_count = 0
            bluefin_service.circuit_state = "CLOSED"

            # Mock the failing method
            bluefin_service.client.get_account_balance.side_effect = exception

            # Test error handling
            with pytest.raises(
                (BluefinAPIError, BluefinConnectionError, BluefinServiceError)
            ):
                await bluefin_service.get_account_balance()

            # Verify failure tracking
            assert bluefin_service.failure_count > 0
            assert bluefin_service.failure_types[expected_failure_type] > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, bluefin_service):
        """Test circuit breaker for service resilience."""

        # Force multiple failures to trigger circuit breaker
        bluefin_service.client.get_account_balance.side_effect = ClientError(
            "Persistent failure"
        )

        # Make multiple failing calls
        for _i in range(bluefin_service.circuit_failure_threshold + 1):
            try:
                await bluefin_service.get_account_balance()
            except Exception:
                logger.debug("Expected circuit breaker test failure")

        # Circuit should be open now
        assert bluefin_service.circuit_state == "OPEN"
        assert (
            bluefin_service.failure_count >= bluefin_service.circuit_failure_threshold
        )

        # Further calls should fail fast
        start_time = time.perf_counter()
        with pytest.raises(BluefinServiceError, match="circuit breaker"):
            await bluefin_service.get_account_balance()

        elapsed_time = time.perf_counter() - start_time
        assert elapsed_time < 0.1  # Should fail fast

        # Test circuit recovery
        bluefin_service.circuit_open_until = time.time() - 1  # Force recovery time
        bluefin_service.circuit_state = "HALF_OPEN"

        # Mock successful response for recovery
        bluefin_service.client.get_account_balance.side_effect = None
        bluefin_service.client.get_account_balance.return_value = {
            "status": "success",
            "data": {"availableBalance": "10000.00"},
        }

        # Should succeed and close circuit
        balance_data = await bluefin_service.get_account_balance()
        assert balance_data["status"] == "success"
        assert bluefin_service.circuit_state == "CLOSED"

    @pytest.mark.asyncio
    async def test_balance_data_validation_and_parsing(self, bluefin_service):
        """Test balance data validation and parsing."""

        # Test with various response formats
        test_responses = [
            # Standard response
            {
                "status": "success",
                "data": {
                    "availableBalance": "10000.50",
                    "marginBalance": "8500.25",
                    "equity": "10000.50",
                },
            },
            # Response with string numbers
            {
                "status": "success",
                "data": {
                    "availableBalance": "10000.123456",
                    "marginBalance": "8500.987654",
                    "equity": "10000.123456",
                },
            },
            # Response with integer values
            {
                "status": "success",
                "data": {
                    "availableBalance": "10000",
                    "marginBalance": "8500",
                    "equity": "10000",
                },
            },
        ]

        for response in test_responses:
            bluefin_service.client.get_account_balance.return_value = response

            balance_data = await bluefin_service.get_account_balance()

            # Verify data structure
            assert balance_data["status"] == "success"
            assert "data" in balance_data
            assert "availableBalance" in balance_data["data"]

            # Verify numerical parsing
            available_balance = Decimal(balance_data["data"]["availableBalance"])
            assert available_balance > 0
            assert isinstance(available_balance, Decimal)

    @pytest.mark.asyncio
    async def test_position_balance_integration(self, bluefin_service):
        """Test position data integration with balance calculations."""

        # Get account balance
        balance_data = await bluefin_service.get_account_balance()

        # Get user positions
        positions_data = await bluefin_service.get_user_positions()

        # Verify data consistency
        assert balance_data["status"] == "success"
        assert positions_data["status"] == "success"

        # Calculate total unrealized P&L from positions
        positions = positions_data["data"]
        sum(Decimal(pos.get("unrealizedPnl", "0")) for pos in positions)

        # Verify position data structure
        for position in positions:
            required_fields = ["symbol", "side", "size", "entryPrice", "markPrice"]
            for field in required_fields:
                assert field in position

            # Verify numeric fields can be parsed
            assert Decimal(position["size"]) >= 0
            assert Decimal(position["entryPrice"]) > 0
            assert Decimal(position["markPrice"]) > 0

    @pytest.mark.asyncio
    async def test_service_performance_benchmarks(self, bluefin_service):
        """Test performance benchmarks for balance operations."""

        performance_metrics = {}

        # Benchmark balance retrieval
        start_time = time.perf_counter()
        for _ in range(10):
            await bluefin_service.get_account_balance()
        balance_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["balance_retrieval_10_calls"] = balance_time

        # Benchmark position retrieval
        start_time = time.perf_counter()
        for _ in range(10):
            await bluefin_service.get_user_positions()
        positions_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["positions_retrieval_10_calls"] = positions_time

        # Benchmark health checks
        start_time = time.perf_counter()
        for _ in range(50):
            await bluefin_service.health_check()
        health_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["health_check_50_calls"] = health_time

        # Performance assertions (reasonable thresholds for mocked operations)
        assert (
            performance_metrics["balance_retrieval_10_calls"] < 1000
        ), "Balance retrieval too slow"
        assert (
            performance_metrics["positions_retrieval_10_calls"] < 1000
        ), "Position retrieval too slow"
        assert (
            performance_metrics["health_check_50_calls"] < 500
        ), "Health checks too slow"

        # Log performance metrics
        print("\nBluefin Service Performance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"  {metric}: {value:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_balance_operations(self, bluefin_service):
        """Test concurrent balance operations for thread safety."""

        # Test concurrent balance retrievals
        balance_tasks = [bluefin_service.get_account_balance() for _ in range(5)]

        # Test concurrent position retrievals
        position_tasks = [bluefin_service.get_user_positions() for _ in range(5)]

        # Test concurrent health checks
        health_tasks = [bluefin_service.health_check() for _ in range(10)]

        # Execute all tasks concurrently
        all_tasks = balance_tasks + position_tasks + health_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Verify all operations completed successfully
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed with {result}"

            if i < 5:  # Balance tasks
                assert result["status"] == "success"
                assert "availableBalance" in result["data"]
            elif i < 10:  # Position tasks
                assert result["status"] == "success"
                assert isinstance(result["data"], list)
            else:  # Health tasks
                assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_service_cleanup_and_shutdown(self, bluefin_service):
        """Test proper service cleanup and shutdown procedures."""

        # Verify service is running
        assert bluefin_service.initialized
        assert bluefin_service.client is not None

        # Perform some operations to generate state
        await bluefin_service.get_account_balance()
        await bluefin_service.get_user_positions()

        # Verify health stats were updated
        assert bluefin_service.health_stats["total_requests"] > 0
        assert bluefin_service.health_stats["successful_requests"] > 0
        assert bluefin_service.health_stats["last_success_time"] > 0

        # Test graceful shutdown
        await bluefin_service.cleanup()

        # Verify cleanup completed
        assert bluefin_service._cleanup_complete

    @pytest.mark.asyncio
    async def test_real_time_balance_updates(self, bluefin_service):
        """Test real-time balance update scenarios."""

        # Simulate balance changes over time
        initial_balance = await bluefin_service.get_account_balance()
        initial_available = Decimal(initial_balance["data"]["availableBalance"])

        # Simulate a trade affecting balance
        bluefin_service.client.get_account_balance.return_value = {
            "status": "success",
            "data": {
                "availableBalance": str(initial_available - Decimal("100.50")),
                "marginBalance": "8400.00",
                "equity": "9900.00",
                "totalBalance": "9900.00",
                "crossMarginAvailable": "8400.00",
                "crossMarginUsed": "1500.00",
            },
        }

        # Get updated balance
        updated_balance = await bluefin_service.get_account_balance()
        updated_available = Decimal(updated_balance["data"]["availableBalance"])

        # Verify balance change was captured
        assert updated_available == initial_available - Decimal("100.50")
        assert updated_balance["data"]["equity"] == "9900.00"

    @pytest.mark.asyncio
    async def test_error_recovery_after_network_issues(self, bluefin_service):
        """Test error recovery after simulated network issues."""

        # Simulate network failure followed by recovery
        failure_count = 0
        original_method = bluefin_service.client.get_account_balance

        def intermittent_failure():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise ClientError(f"Network error {failure_count}")
            return original_method()

        bluefin_service.client.get_account_balance = intermittent_failure

        # Should recover after retries
        start_time = time.perf_counter()
        balance_data = await bluefin_service.get_account_balance()
        recovery_time = time.perf_counter() - start_time

        # Verify recovery
        assert balance_data["status"] == "success"
        assert failure_count == 4  # 3 failures + 1 success

        # Verify recovery was reasonably fast (within retry timeouts)
        assert recovery_time < 10.0  # Should recover within 10 seconds

        # Verify service health is restored
        health_status = await bluefin_service.health_check()
        assert health_status["status"] == "healthy"
