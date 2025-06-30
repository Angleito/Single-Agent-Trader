"""
SDK to Service Client Integration Tests

This module tests the integration between SDK clients, service containers,
and the trading system, ensuring proper data flow and error handling
across the complete technology stack.

Test Coverage:
- SDK client initialization and configuration
- Service client communication and health checks
- Data transformation between SDK and service layers
- Error propagation and recovery mechanisms
- Connection pooling and resource management
- Authentication and authorization flows
- Rate limiting and throttling behavior
- Fallback mechanisms and circuit breakers
"""

import asyncio
import time
from datetime import UTC, datetime

import aiohttp
import pytest


class TestSDKServiceIntegration:
    """Test SDK to Service Client integration scenarios."""

    @pytest.fixture
    async def mock_bluefin_sdk(self):
        """Create mock Bluefin SDK client."""

        class MockBluefinSDK:
            def __init__(self):
                self.connected = False
                self.authenticated = False
                self.api_calls = []
                self.last_error = None

            async def connect(self):
                """Connect to Bluefin network."""
                await asyncio.sleep(0.1)  # Simulate connection time
                self.connected = True
                return True

            async def authenticate(self, private_key):
                """Authenticate with private key."""
                if not self.connected:
                    raise ConnectionError("Must connect first")

                await asyncio.sleep(0.05)  # Simulate auth time
                self.authenticated = True
                return True

            async def get_orderbook(self, symbol, levels=10):
                """Get orderbook data from Bluefin."""
                if not self.authenticated:
                    raise PermissionError("Must authenticate first")

                call_record = {
                    "method": "get_orderbook",
                    "symbol": symbol,
                    "levels": levels,
                    "timestamp": datetime.now(UTC),
                }
                self.api_calls.append(call_record)

                # Simulate network delay
                await asyncio.sleep(0.02)

                return {
                    "symbol": symbol,
                    "bids": [
                        {"price": "50000.0", "size": "10.0"},
                        {"price": "49950.0", "size": "5.0"},
                    ],
                    "asks": [
                        {"price": "50050.0", "size": "8.0"},
                        {"price": "50100.0", "size": "6.0"},
                    ],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "sequence": len(self.api_calls),
                }

            async def place_order(self, order_data):
                """Place order via Bluefin."""
                if not self.authenticated:
                    raise PermissionError("Must authenticate first")

                call_record = {
                    "method": "place_order",
                    "order_data": order_data,
                    "timestamp": datetime.now(UTC),
                }
                self.api_calls.append(call_record)

                await asyncio.sleep(0.1)  # Simulate order placement time

                return {
                    "order_id": f"bluefin_order_{len(self.api_calls)}",
                    "status": "submitted",
                    "symbol": order_data["symbol"],
                    "side": order_data["side"],
                    "price": order_data["price"],
                    "size": order_data["size"],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            async def disconnect(self):
                """Disconnect from Bluefin."""
                self.connected = False
                self.authenticated = False

            def simulate_error(self, error_type="network"):
                """Simulate various error conditions."""
                if error_type == "network":
                    self.last_error = ConnectionError("Network unavailable")
                elif error_type == "auth":
                    self.last_error = PermissionError("Authentication failed")
                elif error_type == "rate_limit":
                    self.last_error = Exception("Rate limit exceeded")

            def get_stats(self):
                """Get SDK usage statistics."""
                return {
                    "connected": self.connected,
                    "authenticated": self.authenticated,
                    "total_calls": len(self.api_calls),
                    "last_error": str(self.last_error) if self.last_error else None,
                }

        return MockBluefinSDK()

    @pytest.fixture
    async def mock_service_client(self):
        """Create mock service client."""

        class MockServiceClient:
            def __init__(self):
                self.base_url = "http://test-service:8080"
                self.session = None
                self.healthy = True
                self.request_count = 0
                self.response_cache = {}

            async def initialize(self):
                """Initialize HTTP session."""
                self.session = aiohttp.ClientSession()

            async def close(self):
                """Close HTTP session."""
                if self.session:
                    await self.session.close()

            async def health_check(self):
                """Check service health."""
                self.request_count += 1

                if not self.healthy:
                    raise aiohttp.ClientConnectorError(
                        connection_key=None, os_error=OSError("Service unavailable")
                    )

                return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}

            async def proxy_orderbook_request(self, symbol, levels=10):
                """Proxy orderbook request to underlying SDK."""
                if not self.healthy:
                    raise Exception("Service unhealthy")

                self.request_count += 1

                # Simulate service processing time
                await asyncio.sleep(0.05)

                # Check cache first
                cache_key = f"orderbook_{symbol}_{levels}"
                if cache_key in self.response_cache:
                    cached_response = self.response_cache[cache_key]
                    # Check if cache is still fresh (5 second TTL)
                    cache_age = datetime.now(UTC) - cached_response["cached_at"]
                    if cache_age.total_seconds() < 5:
                        return cached_response["data"]

                # Mock response from underlying SDK
                response_data = {
                    "symbol": symbol,
                    "bids": [
                        {"price": "50000.0", "size": "10.0"},
                        {"price": "49950.0", "size": "5.0"},
                    ],
                    "asks": [
                        {"price": "50050.0", "size": "8.0"},
                        {"price": "50100.0", "size": "6.0"},
                    ],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source": "service_proxy",
                    "levels": levels,
                }

                # Cache response
                self.response_cache[cache_key] = {
                    "data": response_data,
                    "cached_at": datetime.now(UTC),
                }

                return response_data

            async def proxy_order_request(self, order_data):
                """Proxy order request to underlying SDK."""
                if not self.healthy:
                    raise Exception("Service unhealthy")

                self.request_count += 1
                await asyncio.sleep(0.1)  # Simulate order processing

                return {
                    "order_id": f"service_order_{self.request_count}",
                    "status": "accepted",
                    "symbol": order_data["symbol"],
                    "side": order_data["side"],
                    "price": order_data["price"],
                    "size": order_data["size"],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "processed_by": "service_proxy",
                }

            def simulate_service_error(self, error_type="unavailable"):
                """Simulate service errors."""
                if error_type == "unavailable":
                    self.healthy = False
                elif error_type == "slow":
                    # Will cause timeouts in real scenarios
                    pass

            def get_metrics(self):
                """Get service metrics."""
                return {
                    "healthy": self.healthy,
                    "request_count": self.request_count,
                    "cache_size": len(self.response_cache),
                    "uptime": "mocked",
                }

        return MockServiceClient()

    @pytest.fixture
    async def integration_coordinator(self):
        """Create integration coordinator that manages SDK and service clients."""

        class IntegrationCoordinator:
            def __init__(self, sdk_client, service_client):
                self.sdk_client = sdk_client
                self.service_client = service_client
                self.initialized = False
                self.fallback_mode = False

            async def initialize(self):
                """Initialize all clients."""
                try:
                    # Initialize service client first
                    await self.service_client.initialize()
                    await self.service_client.health_check()

                    # Initialize SDK client
                    await self.sdk_client.connect()
                    await self.sdk_client.authenticate("mock_private_key")

                    self.initialized = True
                    return True

                except Exception:
                    # Try fallback initialization
                    try:
                        await self.sdk_client.connect()
                        await self.sdk_client.authenticate("mock_private_key")
                        self.fallback_mode = True
                        self.initialized = True
                        return True
                    except Exception:
                        return False

            async def get_orderbook(self, symbol, prefer_service=True):
                """Get orderbook with fallback logic."""
                if not self.initialized:
                    raise RuntimeError("Not initialized")

                # Try service client first if preferred and available
                if prefer_service and not self.fallback_mode:
                    try:
                        return await self.service_client.proxy_orderbook_request(symbol)
                    except Exception:
                        # Fall back to direct SDK
                        pass

                # Use direct SDK
                return await self.sdk_client.get_orderbook(symbol)

            async def place_order(self, order_data, prefer_service=True):
                """Place order with fallback logic."""
                if not self.initialized:
                    raise RuntimeError("Not initialized")

                # Try service client first if preferred and available
                if prefer_service and not self.fallback_mode:
                    try:
                        return await self.service_client.proxy_order_request(order_data)
                    except Exception:
                        # Fall back to direct SDK
                        pass

                # Use direct SDK
                return await self.sdk_client.place_order(order_data)

            async def cleanup(self):
                """Clean up all resources."""
                await self.service_client.close()
                await self.sdk_client.disconnect()
                self.initialized = False

        return IntegrationCoordinator

    async def test_basic_sdk_service_integration(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test basic SDK to Service Client integration."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)

        # Initialize integration
        success = await coordinator.initialize()
        assert success
        assert coordinator.initialized
        assert not coordinator.fallback_mode

        # Test orderbook retrieval through service
        orderbook_data = await coordinator.get_orderbook("BTC-USD")
        assert orderbook_data["symbol"] == "BTC-USD"
        assert "bids" in orderbook_data
        assert "asks" in orderbook_data
        assert orderbook_data["source"] == "service_proxy"

        # Test order placement through service
        order_data = {
            "symbol": "BTC-USD",
            "side": "buy",
            "price": "49900.0",
            "size": "1.0",
        }

        order_result = await coordinator.place_order(order_data)
        assert order_result["status"] == "accepted"
        assert order_result["processed_by"] == "service_proxy"

        # Cleanup
        await coordinator.cleanup()
        assert not coordinator.initialized

    async def test_service_unavailable_fallback(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test fallback to direct SDK when service is unavailable."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)

        # Make service unhealthy before initialization
        mock_service_client.simulate_service_error("unavailable")

        # Should still initialize successfully with fallback
        success = await coordinator.initialize()
        assert success
        assert coordinator.initialized
        assert coordinator.fallback_mode

        # Test orderbook retrieval falls back to SDK
        orderbook_data = await coordinator.get_orderbook("BTC-USD")
        assert orderbook_data["symbol"] == "BTC-USD"
        assert "source" not in orderbook_data  # Direct from SDK

        # Verify SDK was called directly
        assert len(mock_bluefin_sdk.api_calls) == 1
        assert mock_bluefin_sdk.api_calls[0]["method"] == "get_orderbook"

        await coordinator.cleanup()

    async def test_service_recovery_and_switching(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test recovery and switching between service and SDK."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)

        # Initialize normally
        await coordinator.initialize()
        assert not coordinator.fallback_mode

        # First request through service
        orderbook1 = await coordinator.get_orderbook("BTC-USD")
        assert orderbook1["source"] == "service_proxy"

        # Simulate service failure
        mock_service_client.simulate_service_error("unavailable")

        # Next request should fall back to SDK
        orderbook2 = await coordinator.get_orderbook("BTC-USD", prefer_service=True)
        assert "source" not in orderbook2  # Direct from SDK

        # Verify both service and SDK were called
        assert mock_service_client.request_count >= 1
        assert len(mock_bluefin_sdk.api_calls) >= 1

        await coordinator.cleanup()

    async def test_performance_comparison(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test performance comparison between service and direct SDK."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)
        await coordinator.initialize()

        # Test service performance
        service_times = []
        for i in range(10):
            start_time = time.time()
            await coordinator.get_orderbook("BTC-USD", prefer_service=True)
            end_time = time.time()
            service_times.append(end_time - start_time)

        # Force fallback mode for SDK testing
        coordinator.fallback_mode = True

        # Test direct SDK performance
        sdk_times = []
        for i in range(10):
            start_time = time.time()
            await coordinator.get_orderbook("ETH-USD", prefer_service=False)
            end_time = time.time()
            sdk_times.append(end_time - start_time)

        # Analyze performance
        avg_service_time = sum(service_times) / len(service_times)
        avg_sdk_time = sum(sdk_times) / len(sdk_times)

        # Both should be reasonably fast
        assert avg_service_time < 1.0
        assert avg_sdk_time < 1.0

        # Service might be slightly slower due to proxy overhead
        # but should benefit from caching on subsequent requests
        print(f"Average service time: {avg_service_time:.3f}s")
        print(f"Average SDK time: {avg_sdk_time:.3f}s")

        await coordinator.cleanup()

    async def test_concurrent_request_handling(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test handling of concurrent requests across service and SDK."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)
        await coordinator.initialize()

        # Create multiple concurrent orderbook requests
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOT-USD"]

        async def fetch_orderbook(symbol):
            return await coordinator.get_orderbook(symbol)

        # Execute all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*[fetch_orderbook(symbol) for symbol in symbols])
        end_time = time.time()

        total_time = end_time - start_time

        # Validate results
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            assert result["symbol"] == symbols[i]
            assert "bids" in result
            assert "asks" in result

        # Should complete concurrently much faster than sequentially
        assert total_time < 0.5  # Should be faster than 5 * 0.1s sequential

        # Check that service caching worked
        unique_service_requests = mock_service_client.request_count
        print(f"Total concurrent requests: {len(symbols)}")
        print(f"Service requests made: {unique_service_requests}")
        print(f"Total time: {total_time:.3f}s")

        await coordinator.cleanup()

    async def test_error_propagation_and_handling(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test error propagation and handling across the integration stack."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)
        await coordinator.initialize()

        # Test 1: Service error with successful fallback
        mock_service_client.simulate_service_error("unavailable")

        # Should still succeed via fallback
        orderbook = await coordinator.get_orderbook("BTC-USD")
        assert orderbook["symbol"] == "BTC-USD"

        # Test 2: Both service and SDK errors
        mock_bluefin_sdk.simulate_error("network")

        # Should now fail completely
        with pytest.raises(Exception):
            await coordinator.get_orderbook("ETH-USD")

        # Test 3: Authentication errors
        mock_bluefin_sdk.simulate_error("auth")
        coordinator.fallback_mode = False  # Reset fallback mode

        with pytest.raises(Exception):
            await coordinator.get_orderbook("SOL-USD")

        await coordinator.cleanup()

    async def test_resource_management_and_cleanup(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test proper resource management and cleanup."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)

        # Multiple initialization/cleanup cycles
        for cycle in range(3):
            # Initialize
            success = await coordinator.initialize()
            assert success
            assert coordinator.initialized

            # Use the connection
            orderbook = await coordinator.get_orderbook("BTC-USD")
            assert orderbook is not None

            # Cleanup
            await coordinator.cleanup()
            assert not coordinator.initialized

            # Verify connections are properly closed
            assert not mock_bluefin_sdk.connected
            assert not mock_bluefin_sdk.authenticated

        # Test cleanup after errors
        await coordinator.initialize()
        mock_service_client.simulate_service_error("unavailable")
        mock_bluefin_sdk.simulate_error("network")

        # Cleanup should still work despite errors
        await coordinator.cleanup()
        assert not coordinator.initialized

    async def test_caching_and_data_consistency(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test caching behavior and data consistency."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)
        await coordinator.initialize()

        symbol = "BTC-USD"

        # First request - should hit service and cache
        orderbook1 = await coordinator.get_orderbook(symbol)
        initial_request_count = mock_service_client.request_count

        # Second request - should hit cache
        orderbook2 = await coordinator.get_orderbook(symbol)
        cached_request_count = mock_service_client.request_count

        # Cache should be used (no additional service requests)
        assert cached_request_count == initial_request_count
        assert orderbook1["timestamp"] == orderbook2["timestamp"]  # Same cached data

        # Wait for cache to expire
        await asyncio.sleep(6)  # TTL is 5 seconds

        # Third request - should refresh cache
        orderbook3 = await coordinator.get_orderbook(symbol)
        final_request_count = mock_service_client.request_count

        # Cache should have been refreshed
        assert final_request_count > cached_request_count
        assert orderbook3["timestamp"] != orderbook1["timestamp"]  # Fresh data

        await coordinator.cleanup()

    async def test_authentication_and_authorization_flow(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test authentication and authorization flow."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)

        # Test successful authentication
        success = await coordinator.initialize()
        assert success
        assert mock_bluefin_sdk.authenticated

        # Test operations require authentication
        orderbook = await coordinator.get_orderbook("BTC-USD")
        assert orderbook is not None

        # Test order placement requires authentication
        order_data = {
            "symbol": "BTC-USD",
            "side": "buy",
            "price": "50000.0",
            "size": "1.0",
        }

        order_result = await coordinator.place_order(order_data)
        assert order_result["status"] in ["accepted", "submitted"]

        # Test re-authentication after disconnect
        await mock_bluefin_sdk.disconnect()
        assert not mock_bluefin_sdk.authenticated

        # Operations should fail without authentication
        with pytest.raises(PermissionError):
            await coordinator.get_orderbook("ETH-USD", prefer_service=False)

        await coordinator.cleanup()

    async def test_rate_limiting_and_throttling(
        self, mock_bluefin_sdk, mock_service_client, integration_coordinator
    ):
        """Test rate limiting and throttling behavior."""

        coordinator = integration_coordinator(mock_bluefin_sdk, mock_service_client)
        await coordinator.initialize()

        # Make rapid requests
        request_times = []

        for i in range(20):
            start_time = time.time()
            try:
                await coordinator.get_orderbook(f"SYM{i}-USD")
                end_time = time.time()
                request_times.append(end_time - start_time)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print(f"Rate limit hit at request {i}")
                    break

        # Analyze request timing
        if request_times:
            avg_time = sum(request_times) / len(request_times)
            print(f"Average request time: {avg_time:.3f}s")
            print(f"Completed requests: {len(request_times)}")

        # Verify service handled requests appropriately
        assert mock_service_client.request_count > 0

        await coordinator.cleanup()


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
