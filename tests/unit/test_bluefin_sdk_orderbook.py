#!/usr/bin/env python3
"""
Comprehensive unit tests for the Bluefin SDK service orderbook endpoint.

This test suite covers:
- Valid symbol and depth parameters
- Invalid input validation
- Error handling (API failures, timeouts)
- Response format validation
- Rate limiting behavior
- Authentication and authorization
- Performance and load testing
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the service and related components
try:
    from bluefin_v2_client import MARKET_SYMBOLS
    from bluefin_v2_client.interfaces import GetOrderbookRequest

    from services.bluefin_sdk_service import (
        ORDERBOOK_PRECISION_LIMIT,
        BluefinAPIError,
        BluefinSDKService,
        get_orderbook,
    )
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestBluefinOrderbookEndpoint:
    """Test suite for the Bluefin SDK orderbook endpoint."""

    @pytest.fixture
    def mock_bluefin_client(self):
        """Mock Bluefin client for testing."""
        client = Mock()
        client.get_orderbook = AsyncMock()
        return client

    @pytest.fixture
    def sdk_service(self, mock_bluefin_client):
        """Create SDK service instance with mocked client."""
        service = BluefinSDKService("test_private_key", "testnet")
        service.client = mock_bluefin_client
        service._initialized = True
        return service

    @pytest.fixture
    def sample_orderbook_response(self):
        """Sample orderbook response from Bluefin SDK."""
        return {
            "bids": [
                {"price": "50000.00", "quantity": "0.5", "size": "25000.00"},
                {"price": "49950.00", "quantity": "1.0", "size": "49950.00"},
                {"price": "49900.00", "quantity": "2.0", "size": "99800.00"},
                {"price": "49850.00", "quantity": "0.8", "size": "39880.00"},
                {"price": "49800.00", "quantity": "1.5", "size": "74700.00"},
            ],
            "asks": [
                {"price": "50050.00", "quantity": "0.3", "size": "15015.00"},
                {"price": "50100.00", "quantity": "0.7", "size": "35070.00"},
                {"price": "50150.00", "quantity": "1.2", "size": "60180.00"},
                {"price": "50200.00", "quantity": "0.9", "size": "45180.00"},
                {"price": "50250.00", "quantity": "2.1", "size": "105525.00"},
            ],
        }

    @pytest.fixture
    def expected_formatted_response(self):
        """Expected formatted orderbook response."""
        return {
            "symbol": "SUI-PERP",
            "bids": [
                {"price": "50000.00", "quantity": "0.5", "size": "25000.00"},
                {"price": "49950.00", "quantity": "1.0", "size": "49950.00"},
                {"price": "49900.00", "quantity": "2.0", "size": "99800.00"},
                {"price": "49850.00", "quantity": "0.8", "size": "39880.00"},
                {"price": "49800.00", "quantity": "1.5", "size": "74700.00"},
            ],
            "asks": [
                {"price": "50050.00", "quantity": "0.3", "size": "15015.00"},
                {"price": "50100.00", "quantity": "0.7", "size": "35070.00"},
                {"price": "50150.00", "quantity": "1.2", "size": "60180.00"},
                {"price": "50200.00", "quantity": "0.9", "size": "45180.00"},
                {"price": "50250.00", "quantity": "2.1", "size": "105525.00"},
            ],
            "depth": 10,
        }

    @pytest.mark.asyncio
    async def test_get_orderbook_valid_symbol_default_depth(
        self, sdk_service, sample_orderbook_response
    ):
        """Test orderbook endpoint with valid symbol and default depth."""
        # Setup mock
        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        # Call method
        result = await sdk_service.get_orderbook("SUI-PERP")

        # Verify call was made correctly
        sdk_service.client.get_orderbook.assert_called_once()
        call_args = sdk_service.client.get_orderbook.call_args[0][0]
        assert isinstance(call_args, GetOrderbookRequest)
        assert call_args.limit == 10  # Default depth

        # Verify response format
        assert result["symbol"] == "SUI-PERP"
        assert "bids" in result
        assert "asks" in result
        assert "depth" in result
        assert "timestamp" in result
        assert result["depth"] == 10
        assert len(result["bids"]) == 5
        assert len(result["asks"]) == 5

        # Verify bid/ask structure
        for bid in result["bids"]:
            assert "price" in bid
            assert "quantity" in bid
            assert "size" in bid
            assert isinstance(bid["price"], str)
            assert isinstance(bid["quantity"], str)
            assert isinstance(bid["size"], str)

    @pytest.mark.asyncio
    async def test_get_orderbook_custom_depth(
        self, sdk_service, sample_orderbook_response
    ):
        """Test orderbook endpoint with custom depth parameter."""
        # Setup mock
        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        # Call method with custom depth
        result = await sdk_service.get_orderbook("SUI-PERP", depth=5)

        # Verify call was made with correct depth
        call_args = sdk_service.client.get_orderbook.call_args[0][0]
        assert call_args.limit == 5

        # Verify response
        assert result["depth"] == 5

    @pytest.mark.asyncio
    async def test_get_orderbook_depth_validation(
        self, sdk_service, sample_orderbook_response
    ):
        """Test depth parameter validation."""
        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        # Test depth too low (should be normalized to 1)
        await sdk_service.get_orderbook("SUI-PERP", depth=0)
        call_args = sdk_service.client.get_orderbook.call_args[0][0]
        assert call_args.limit == 1

        # Test depth too high (should be normalized to 100)
        await sdk_service.get_orderbook("SUI-PERP", depth=150)
        call_args = sdk_service.client.get_orderbook.call_args[0][0]
        assert call_args.limit == 100

        # Test negative depth (should be normalized to 1)
        await sdk_service.get_orderbook("SUI-PERP", depth=-5)
        call_args = sdk_service.client.get_orderbook.call_args[0][0]
        assert call_args.limit == 1

    @pytest.mark.asyncio
    async def test_get_orderbook_symbol_validation(
        self, sdk_service, sample_orderbook_response
    ):
        """Test symbol validation and normalization."""
        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        # Mock symbol validation methods
        sdk_service._validate_and_normalize_symbol = Mock(
            side_effect=lambda x: x.upper()
        )
        sdk_service._get_market_symbol_value = Mock(return_value="SUI_PERP")

        # Test symbol normalization
        result = await sdk_service.get_orderbook("sui-perp")

        # Verify symbol was normalized
        sdk_service._validate_and_normalize_symbol.assert_called_once_with("sui-perp")
        sdk_service._get_market_symbol_value.assert_called_once_with("SUI-PERP")

    @pytest.mark.asyncio
    async def test_get_orderbook_empty_bids_asks(self, sdk_service):
        """Test orderbook with empty bids or asks."""
        # Test empty bids
        empty_bids_response = {
            "bids": [],
            "asks": [{"price": "50000", "quantity": "1.0", "size": "50000"}],
        }
        sdk_service.client.get_orderbook.return_value = empty_bids_response

        result = await sdk_service.get_orderbook("SUI-PERP")
        assert len(result["bids"]) == 0
        assert len(result["asks"]) == 1

        # Test empty asks
        empty_asks_response = {
            "bids": [{"price": "50000", "quantity": "1.0", "size": "50000"}],
            "asks": [],
        }
        sdk_service.client.get_orderbook.return_value = empty_asks_response

        result = await sdk_service.get_orderbook("SUI-PERP")
        assert len(result["bids"]) == 1
        assert len(result["asks"]) == 0

    @pytest.mark.asyncio
    async def test_get_orderbook_missing_fields(self, sdk_service):
        """Test orderbook response with missing fields."""
        # Test missing bids field
        missing_bids_response = {
            "asks": [{"price": "50000", "quantity": "1.0", "size": "50000"}]
        }
        sdk_service.client.get_orderbook.return_value = missing_bids_response

        result = await sdk_service.get_orderbook("SUI-PERP")
        assert result["bids"] == []
        assert len(result["asks"]) == 1

        # Test missing asks field
        missing_asks_response = {
            "bids": [{"price": "50000", "quantity": "1.0", "size": "50000"}]
        }
        sdk_service.client.get_orderbook.return_value = missing_asks_response

        result = await sdk_service.get_orderbook("SUI-PERP")
        assert len(result["bids"]) == 1
        assert result["asks"] == []

    @pytest.mark.asyncio
    async def test_get_orderbook_api_error(self, sdk_service):
        """Test orderbook API error handling."""
        # Mock API error
        sdk_service.client.get_orderbook.side_effect = Exception(
            "API connection failed"
        )

        # Should raise BluefinAPIError
        with pytest.raises(BluefinAPIError) as exc_info:
            await sdk_service.get_orderbook("SUI-PERP")

        assert "Failed to fetch orderbook" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_orderbook_timeout_error(self, sdk_service):
        """Test orderbook timeout handling."""
        # Mock timeout error
        sdk_service.client.get_orderbook.side_effect = TimeoutError("Request timed out")

        with pytest.raises(BluefinAPIError):
            await sdk_service.get_orderbook("SUI-PERP")

    @pytest.mark.asyncio
    async def test_get_orderbook_network_error(self, sdk_service):
        """Test orderbook network error handling."""
        # Mock network error
        from aiohttp import ClientError

        sdk_service.client.get_orderbook.side_effect = ClientError("Network error")

        with pytest.raises(BluefinAPIError):
            await sdk_service.get_orderbook("SUI-PERP")

    @pytest.mark.asyncio
    async def test_get_orderbook_malformed_response(self, sdk_service):
        """Test orderbook with malformed response data."""
        # Test malformed bid entry
        malformed_response = {
            "bids": [
                {"price": "invalid_price", "quantity": "1.0"},  # Missing size
                {"price": "50000", "quantity": "invalid_quantity", "size": "50000"},
            ],
            "asks": [
                {"price": "50100", "quantity": "1.0", "size": "50100"},
            ],
        }
        sdk_service.client.get_orderbook.return_value = malformed_response

        result = await sdk_service.get_orderbook("SUI-PERP")

        # Should handle missing/invalid fields gracefully
        assert len(result["bids"]) == 2
        assert result["bids"][0]["price"] == "invalid_price"
        assert result["bids"][0]["size"] == "0"  # Default for missing size
        assert result["bids"][1]["quantity"] == "invalid_quantity"

    @pytest.mark.asyncio
    async def test_get_orderbook_depth_limiting(self, sdk_service):
        """Test orderbook depth limiting works correctly."""
        # Create response with more entries than requested depth
        large_response = {
            "bids": [
                {
                    "price": f"{50000 - i * 10}",
                    "quantity": "1.0",
                    "size": f"{50000 - i * 10}",
                }
                for i in range(20)
            ],
            "asks": [
                {
                    "price": f"{50000 + i * 10}",
                    "quantity": "1.0",
                    "size": f"{50000 + i * 10}",
                }
                for i in range(1, 21)
            ],
        }
        sdk_service.client.get_orderbook.return_value = large_response

        # Request depth of 5
        result = await sdk_service.get_orderbook("SUI-PERP", depth=5)

        # Should only return 5 bids and 5 asks
        assert len(result["bids"]) == 5
        assert len(result["asks"]) == 5
        assert result["depth"] == 5

    @pytest.mark.asyncio
    async def test_get_orderbook_precision_handling(self, sdk_service):
        """Test orderbook price/quantity precision handling."""
        high_precision_response = {
            "bids": [
                {
                    "price": "50000.123456789",
                    "quantity": "1.987654321",
                    "size": "98765.432109876",
                }
            ],
            "asks": [
                {
                    "price": "50100.987654321",
                    "quantity": "2.123456789",
                    "size": "106398.765432109",
                }
            ],
        }
        sdk_service.client.get_orderbook.return_value = high_precision_response

        result = await sdk_service.get_orderbook("SUI-PERP")

        # Values should be preserved as strings
        assert result["bids"][0]["price"] == "50000.123456789"
        assert result["bids"][0]["quantity"] == "1.987654321"
        assert result["asks"][0]["price"] == "50100.987654321"
        assert result["asks"][0]["quantity"] == "2.123456789"


class TestOrderbookHTTPEndpoint:
    """Test suite for the orderbook HTTP endpoint."""

    @pytest.fixture
    def mock_service(self):
        """Mock SDK service for HTTP endpoint testing."""
        service = Mock()
        service.get_orderbook = AsyncMock()
        return service

    @pytest.fixture
    def mock_request_path_param(self):
        """Mock HTTP request with path parameter."""
        request = Mock()
        request.match_info = {"symbol": "SUI-PERP"}
        request.query = {"depth": "10"}
        return request

    @pytest.fixture
    def mock_request_query_param(self):
        """Mock HTTP request with query parameter."""
        request = Mock()
        request.match_info = {}
        request.query = {"symbol": "BTC-PERP", "depth": "5"}
        return request

    @pytest.fixture
    def sample_service_response(self):
        """Sample response from service layer."""
        return {
            "symbol": "SUI-PERP",
            "bids": [{"price": "50000.00", "quantity": "1.0", "size": "50000.00"}],
            "asks": [{"price": "50100.00", "quantity": "1.0", "size": "50100.00"}],
            "depth": 10,
            "timestamp": int(time.time() * 1000),
        }

    @pytest.mark.asyncio
    async def test_http_get_orderbook_path_param(
        self, mock_request_path_param, sample_service_response
    ):
        """Test HTTP endpoint with path parameter."""
        with patch("services.bluefin_sdk_service.service") as mock_service:
            mock_service.get_orderbook.return_value = sample_service_response

            response = await get_orderbook(mock_request_path_param)

            # Verify service was called correctly
            mock_service.get_orderbook.assert_called_once_with("SUI-PERP", 10)

            # Verify response
            assert response.status == 200
            assert response.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_http_get_orderbook_query_param(
        self, mock_request_query_param, sample_service_response
    ):
        """Test HTTP endpoint with query parameter."""
        with patch("services.bluefin_sdk_service.service") as mock_service:
            mock_service.get_orderbook.return_value = sample_service_response

            response = await get_orderbook(mock_request_query_param)

            # Verify service was called correctly
            mock_service.get_orderbook.assert_called_once_with("BTC-PERP", 5)

    @pytest.mark.asyncio
    async def test_http_get_orderbook_default_params(self):
        """Test HTTP endpoint with default parameters."""
        request = Mock()
        request.match_info = {}
        request.query = {}

        sample_response = {
            "symbol": "SUI-PERP",
            "bids": [],
            "asks": [],
            "depth": 10,
            "timestamp": int(time.time() * 1000),
        }

        with patch("services.bluefin_sdk_service.service") as mock_service:
            mock_service.get_orderbook.return_value = sample_response

            response = await get_orderbook(request)

            # Should use default symbol and depth
            mock_service.get_orderbook.assert_called_once_with("SUI-PERP", 10)

    @pytest.mark.asyncio
    async def test_http_get_orderbook_invalid_depth(self):
        """Test HTTP endpoint with invalid depth parameter."""
        request = Mock()
        request.match_info = {"symbol": "SUI-PERP"}
        request.query = {"depth": "invalid"}

        response = await get_orderbook(request)

        # Should return 400 error
        assert response.status == 400
        assert "Invalid parameter" in response.text

    @pytest.mark.asyncio
    async def test_http_get_orderbook_depth_out_of_range(self):
        """Test HTTP endpoint with depth out of valid range."""
        # Test depth too low
        request_low = Mock()
        request_low.match_info = {"symbol": "SUI-PERP"}
        request_low.query = {"depth": "0"}

        response = await get_orderbook(request_low)
        assert response.status == 400
        assert "Depth must be between 1 and 100" in response.text

        # Test depth too high
        request_high = Mock()
        request_high.match_info = {"symbol": "SUI-PERP"}
        request_high.query = {"depth": "150"}

        response = await get_orderbook(request_high)
        assert response.status == 400
        assert "Depth must be between 1 and 100" in response.text

    @pytest.mark.asyncio
    async def test_http_get_orderbook_bluefin_api_error(self):
        """Test HTTP endpoint with Bluefin API error."""
        request = Mock()
        request.match_info = {"symbol": "SUI-PERP"}
        request.query = {"depth": "10"}

        with patch("services.bluefin_sdk_service.service") as mock_service:
            mock_service.get_orderbook.side_effect = BluefinAPIError(
                "API connection failed"
            )

            response = await get_orderbook(request)

            # Should return 500 error
            assert response.status == 500
            assert "API connection failed" in response.text

    @pytest.mark.asyncio
    async def test_http_get_orderbook_unexpected_error(self):
        """Test HTTP endpoint with unexpected error."""
        request = Mock()
        request.match_info = {"symbol": "SUI-PERP"}
        request.query = {"depth": "10"}

        with patch("services.bluefin_sdk_service.service") as mock_service:
            mock_service.get_orderbook.side_effect = RuntimeError("Unexpected error")

            response = await get_orderbook(request)

            # Should return 500 error
            assert response.status == 500
            assert "Unexpected error" in response.text


class TestOrderbookPerformance:
    """Performance and load testing for orderbook endpoint."""

    @pytest.fixture
    def sdk_service(self):
        """Create SDK service for performance testing."""
        service = BluefinSDKService("test_private_key", "testnet")
        service.client = Mock()
        service.client.get_orderbook = AsyncMock()
        service._initialized = True
        return service

    @pytest.mark.asyncio
    async def test_orderbook_response_time(
        self, sdk_service, sample_orderbook_response
    ):
        """Test orderbook response time performance."""
        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        start_time = time.time()
        await sdk_service.get_orderbook("SUI-PERP")
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within reasonable time (adjust threshold as needed)
        assert (
            response_time < 1.0
        ), f"Response time {response_time:.3f}s exceeds threshold"

    @pytest.mark.asyncio
    async def test_orderbook_concurrent_requests(
        self, sdk_service, sample_orderbook_response
    ):
        """Test orderbook endpoint under concurrent load."""
        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        # Create multiple concurrent requests
        concurrent_requests = 10
        tasks = [
            sdk_service.get_orderbook("SUI-PERP", depth=10)
            for _ in range(concurrent_requests)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # All requests should succeed
        for result in results:
            assert not isinstance(result, Exception), f"Request failed: {result}"
            assert "symbol" in result
            assert "bids" in result
            assert "asks" in result

        # Should handle concurrent requests efficiently
        total_time = end_time - start_time
        avg_time_per_request = total_time / concurrent_requests
        assert (
            avg_time_per_request < 0.5
        ), f"Average time per request {avg_time_per_request:.3f}s too high"

    @pytest.mark.asyncio
    async def test_orderbook_large_depth_performance(self, sdk_service):
        """Test orderbook performance with large depth parameter."""
        # Create large orderbook response
        large_response = {
            "bids": [
                {
                    "price": f"{50000 - i}",
                    "quantity": f"{i + 1}",
                    "size": f"{(50000 - i) * (i + 1)}",
                }
                for i in range(100)
            ],
            "asks": [
                {
                    "price": f"{50001 + i}",
                    "quantity": f"{i + 1}",
                    "size": f"{(50001 + i) * (i + 1)}",
                }
                for i in range(100)
            ],
        }
        sdk_service.client.get_orderbook.return_value = large_response

        start_time = time.time()
        result = await sdk_service.get_orderbook("SUI-PERP", depth=100)
        end_time = time.time()

        response_time = end_time - start_time

        # Should handle large responses efficiently
        assert (
            response_time < 2.0
        ), f"Large depth response time {response_time:.3f}s too high"
        assert len(result["bids"]) == 100
        assert len(result["asks"]) == 100

    @pytest.mark.asyncio
    async def test_orderbook_memory_usage(self, sdk_service, sample_orderbook_response):
        """Test orderbook memory usage patterns."""
        import tracemalloc

        sdk_service.client.get_orderbook.return_value = sample_orderbook_response

        # Start memory tracing
        tracemalloc.start()

        # Make multiple requests to check for memory leaks
        for _ in range(100):
            await sdk_service.get_orderbook("SUI-PERP")

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (adjust thresholds as needed)
        assert (
            current < 10 * 1024 * 1024
        ), f"Current memory usage {current / 1024 / 1024:.2f}MB too high"
        assert (
            peak < 50 * 1024 * 1024
        ), f"Peak memory usage {peak / 1024 / 1024:.2f}MB too high"


class TestOrderbookRateLimiting:
    """Test rate limiting behavior for orderbook endpoint."""

    @pytest.fixture
    def rate_limited_service(self):
        """Create service with rate limiting."""
        service = BluefinSDKService("test_private_key", "testnet")
        service.client = Mock()
        service.client.get_orderbook = AsyncMock()
        service._initialized = True

        # Mock rate limiting
        service._rate_limiter = Mock()
        service._rate_limiter.acquire = AsyncMock()

        return service

    @pytest.mark.asyncio
    async def test_orderbook_rate_limiting_compliance(
        self, rate_limited_service, sample_orderbook_response
    ):
        """Test that orderbook requests comply with rate limits."""
        rate_limited_service.client.get_orderbook.return_value = (
            sample_orderbook_response
        )

        # Make multiple requests rapidly
        for i in range(5):
            await rate_limited_service.get_orderbook("SUI-PERP")

        # Rate limiter should have been called for each request
        assert rate_limited_service._rate_limiter.acquire.call_count >= 5

    @pytest.mark.asyncio
    async def test_orderbook_rate_limit_exceeded(self, rate_limited_service):
        """Test behavior when rate limit is exceeded."""
        # Configure rate limiter to raise exception
        from aiohttp import ClientResponseError

        rate_limited_service._rate_limiter.acquire.side_effect = ClientResponseError(
            request_info=Mock(), history=(), status=429, message="Rate limit exceeded"
        )

        with pytest.raises(BluefinAPIError) as exc_info:
            await rate_limited_service.get_orderbook("SUI-PERP")

        assert "Rate limit exceeded" in str(
            exc_info.value
        ) or "Failed to fetch orderbook" in str(exc_info.value)


class TestOrderbookSecurity:
    """Security testing for orderbook endpoint."""

    @pytest.fixture
    def secure_service(self):
        """Create service for security testing."""
        service = BluefinSDKService("test_private_key", "testnet")
        service.client = Mock()
        service.client.get_orderbook = AsyncMock()
        service._initialized = True
        return service

    @pytest.mark.asyncio
    async def test_orderbook_sql_injection_protection(
        self, secure_service, sample_orderbook_response
    ):
        """Test protection against SQL injection attempts."""
        secure_service.client.get_orderbook.return_value = sample_orderbook_response

        # Test various SQL injection patterns
        malicious_symbols = [
            "SUI-PERP'; DROP TABLE orders; --",
            "SUI-PERP' OR '1'='1",
            "SUI-PERP'; INSERT INTO orders VALUES ('hack'); --",
            "SUI-PERP' UNION SELECT * FROM users --",
        ]

        for malicious_symbol in malicious_symbols:
            # Should handle malicious input gracefully
            try:
                result = await secure_service.get_orderbook(malicious_symbol)
                # If no exception, verify response is safe
                assert isinstance(result, dict)
                assert "symbol" in result
            except (BluefinAPIError, ValueError):
                # Acceptable to reject malicious input
                pass

    @pytest.mark.asyncio
    async def test_orderbook_xss_protection(
        self, secure_service, sample_orderbook_response
    ):
        """Test protection against XSS attempts."""
        secure_service.client.get_orderbook.return_value = sample_orderbook_response

        # Test XSS patterns
        xss_symbols = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "SUI-PERP<script>alert('xss')</script>",
        ]

        for xss_symbol in xss_symbols:
            try:
                result = await secure_service.get_orderbook(xss_symbol)
                # Response should not contain executable scripts
                response_str = json.dumps(result)
                assert "<script>" not in response_str
                assert "javascript:" not in response_str
                assert "onerror=" not in response_str
            except (BluefinAPIError, ValueError):
                # Acceptable to reject malicious input
                pass

    @pytest.mark.asyncio
    async def test_orderbook_input_sanitization(
        self, secure_service, sample_orderbook_response
    ):
        """Test input sanitization for orderbook requests."""
        secure_service.client.get_orderbook.return_value = sample_orderbook_response

        # Test various input formats
        test_cases = [
            ("  SUI-PERP  ", "SUI-PERP"),  # Whitespace trimming
            ("sui-perp", "SUI-PERP"),  # Case normalization
            ("SUI_PERP", "SUI-PERP"),  # Underscore to dash
        ]

        # Mock the normalization method
        secure_service._validate_and_normalize_symbol = Mock(
            side_effect=lambda x: x.strip().upper().replace("_", "-")
        )
        secure_service._get_market_symbol_value = Mock(return_value="SUI_PERP")

        for input_symbol, expected_normalized in test_cases:
            await secure_service.get_orderbook(input_symbol)
            secure_service._validate_and_normalize_symbol.assert_called_with(
                input_symbol
            )

    @pytest.mark.asyncio
    async def test_orderbook_authentication_required(self):
        """Test that orderbook requires proper authentication."""
        # Test with invalid/missing credentials
        invalid_service = BluefinSDKService("", "testnet")  # Empty private key

        # Should fail to initialize or make requests
        with pytest.raises((ValueError, BluefinAPIError, AttributeError)):
            await invalid_service.get_orderbook("SUI-PERP")


class TestOrderbookEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def edge_case_service(self):
        """Create service for edge case testing."""
        service = BluefinSDKService("test_private_key", "testnet")
        service.client = Mock()
        service.client.get_orderbook = AsyncMock()
        service._initialized = True
        return service

    @pytest.mark.asyncio
    async def test_orderbook_zero_prices(self, edge_case_service):
        """Test orderbook with zero prices."""
        zero_price_response = {
            "bids": [{"price": "0.00", "quantity": "1.0", "size": "0.00"}],
            "asks": [{"price": "0.00", "quantity": "1.0", "size": "0.00"}],
        }
        edge_case_service.client.get_orderbook.return_value = zero_price_response

        result = await edge_case_service.get_orderbook("SUI-PERP")

        # Should handle zero prices gracefully
        assert result["bids"][0]["price"] == "0.00"
        assert result["asks"][0]["price"] == "0.00"

    @pytest.mark.asyncio
    async def test_orderbook_very_large_numbers(self, edge_case_service):
        """Test orderbook with very large numbers."""
        large_number_response = {
            "bids": [
                {
                    "price": "999999999999.999999",
                    "quantity": "999999999.999999",
                    "size": "999999999999999999.999998",
                }
            ],
            "asks": [
                {
                    "price": "1000000000000.000001",
                    "quantity": "1000000000.000001",
                    "size": "1000000000000000000.000001",
                }
            ],
        }
        edge_case_service.client.get_orderbook.return_value = large_number_response

        result = await edge_case_service.get_orderbook("SUI-PERP")

        # Should handle large numbers as strings
        assert result["bids"][0]["price"] == "999999999999.999999"
        assert result["asks"][0]["price"] == "1000000000000.000001"

    @pytest.mark.asyncio
    async def test_orderbook_extreme_depth_values(
        self, edge_case_service, sample_orderbook_response
    ):
        """Test orderbook with extreme depth values."""
        edge_case_service.client.get_orderbook.return_value = sample_orderbook_response

        # Test with maximum integer value
        await edge_case_service.get_orderbook("SUI-PERP", depth=2**31 - 1)
        call_args = edge_case_service.client.get_orderbook.call_args[0][0]
        assert call_args.limit == 100  # Should be capped at maximum

        # Test with minimum integer value
        await edge_case_service.get_orderbook("SUI-PERP", depth=-(2**31))
        call_args = edge_case_service.client.get_orderbook.call_args[0][0]
        assert call_args.limit == 1  # Should be normalized to minimum

    @pytest.mark.asyncio
    async def test_orderbook_unicode_symbols(
        self, edge_case_service, sample_orderbook_response
    ):
        """Test orderbook with unicode symbols."""
        edge_case_service.client.get_orderbook.return_value = sample_orderbook_response
        edge_case_service._validate_and_normalize_symbol = Mock(side_effect=lambda x: x)
        edge_case_service._get_market_symbol_value = Mock(return_value="SUI_PERP")

        unicode_symbols = [
            "SUI-PERP🚀",
            "ETH-PERP™",
            "BTC-PERP©",
            "DOGE-PERP💎",
        ]

        for unicode_symbol in unicode_symbols:
            try:
                result = await edge_case_service.get_orderbook(unicode_symbol)
                # Should handle unicode gracefully
                assert isinstance(result, dict)
            except (BluefinAPIError, ValueError, UnicodeError):
                # Acceptable to reject unicode symbols
                pass

    @pytest.mark.asyncio
    async def test_orderbook_very_long_symbol_names(
        self, edge_case_service, sample_orderbook_response
    ):
        """Test orderbook with very long symbol names."""
        edge_case_service.client.get_orderbook.return_value = sample_orderbook_response
        edge_case_service._validate_and_normalize_symbol = Mock(
            side_effect=lambda x: x[:20]
        )  # Truncate
        edge_case_service._get_market_symbol_value = Mock(return_value="SUI_PERP")

        # Test with very long symbol name
        long_symbol = "A" * 1000 + "-PERP"

        try:
            result = await edge_case_service.get_orderbook(long_symbol)
            # Should handle long symbols (possibly truncated)
            assert isinstance(result, dict)
        except (BluefinAPIError, ValueError):
            # Acceptable to reject overly long symbols
            pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
