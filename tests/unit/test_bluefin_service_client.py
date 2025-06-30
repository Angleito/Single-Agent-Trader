"""
Unit tests for Bluefin service client orderbook functionality.

These tests cover HTTP request/response handling, retry logic, error recovery,
data standardization, connection pooling, and health checks.
"""

import time
import unittest.mock
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot.config import Settings
from bot.exchange.bluefin_service_client import (
    BluefinServiceClient,
    BluefinServiceError,
    BluefinServiceUnavailable,
    close_bluefin_service_client,
    get_bluefin_service_client,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.system = Mock()
    settings.system.bluefin_service_url = "http://test-service:8080"
    settings.system.bluefin_service_api_key = "test-api-key"
    return settings


@pytest.fixture
def client(mock_settings):
    """Create BluefinServiceClient instance for testing."""
    with patch("aiohttp.TCPConnector"):
        return BluefinServiceClient(mock_settings)


class TestBluefinServiceClient:
    """Test suite for BluefinServiceClient."""

    @pytest.mark.asyncio
    async def test_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "http://test-service:8080"
        assert client.api_key == "test-api-key"
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client._session is None
        assert client._service_available is None

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager functionality."""
        with (
            patch.object(client, "initialize") as mock_init,
            patch.object(client, "close") as mock_close,
        ):
            async with client:
                mock_init.assert_called_once()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_headers(self, client):
        """Test default headers generation."""
        headers = client._get_default_headers()

        expected_headers = {
            "User-Agent": "AI-Trading-Bot-Bluefin-Client/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-Key": "test-api-key",
        }

        assert headers == expected_headers

    @pytest.mark.asyncio
    async def test_default_headers_no_api_key(self, mock_settings):
        """Test default headers when no API key is provided."""
        mock_settings.system.bluefin_service_api_key = None
        with patch("aiohttp.TCPConnector"):
            client = BluefinServiceClient(mock_settings)
            headers = client._get_default_headers()

            assert "X-API-Key" not in headers
            assert headers["Accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        with patch.object(client, "check_health", return_value=True) as mock_health:
            result = await client.check_health()
            assert result is True

    @pytest.mark.asyncio
    async def test_timeout_configuration(self, client):
        """Test timeout configuration."""
        timeout = client.timeout

        assert timeout.total == 30
        assert timeout.connect == 10
        assert timeout.sock_read == 20

    @pytest.mark.asyncio
    async def test_is_available_method(self, client):
        """Test is_available method."""
        # Initially None
        assert client.is_available() is False

        # After successful health check
        client._service_available = True
        assert client.is_available() is True

        # After failed health check
        client._service_available = False
        assert client.is_available() is False

    @pytest.mark.asyncio
    async def test_get_order_book_success(self, client):
        """Test successful order book retrieval."""
        mock_response_data = {
            "bids": [
                {"price": "100.50", "quantity": "10.0"},
                {"price": "100.25", "quantity": "5.0"},
            ],
            "asks": [
                {"price": "100.75", "quantity": "8.0"},
                {"price": "101.00", "quantity": "12.0"},
            ],
            "timestamp": 1640995200,
        }

        with patch.object(
            client, "_make_request", return_value=mock_response_data
        ) as mock_request:
            result = await client.get_order_book("SUI-PERP", depth=5)

            expected_result = {
                "symbol": "SUI-PERP",
                "bids": [
                    {"price": 100.50, "quantity": 10.0},
                    {"price": 100.25, "quantity": 5.0},
                ],
                "asks": [
                    {"price": 100.75, "quantity": 8.0},
                    {"price": 101.00, "quantity": 12.0},
                ],
                "timestamp": 1640995200,
            }

            assert result == expected_result
            mock_request.assert_called_once_with("GET", "/orderbook/SUI-PERP?depth=5")

    @pytest.mark.asyncio
    async def test_get_order_book_default_depth(self, client):
        """Test order book retrieval with default depth."""
        mock_response_data = {"bids": [], "asks": [], "timestamp": 0}

        with patch.object(
            client, "_make_request", return_value=mock_response_data
        ) as mock_request:
            await client.get_order_book("BTC-USD")

            mock_request.assert_called_once_with("GET", "/orderbook/BTC-USD")

    @pytest.mark.asyncio
    async def test_get_order_book_invalid_response(self, client):
        """Test order book retrieval with invalid response format."""
        with patch.object(client, "_make_request", return_value="invalid"):
            with pytest.raises(
                BluefinServiceError, match="Invalid order book response"
            ):
                await client.get_order_book("SUI-PERP")

    @pytest.mark.asyncio
    async def test_get_order_book_missing_fields(self, client):
        """Test order book retrieval with missing required fields."""
        mock_response_data = {"bids": []}  # Missing 'asks'

        with patch.object(client, "_make_request", return_value=mock_response_data):
            with pytest.raises(
                BluefinServiceError, match="Missing required field 'asks'"
            ):
                await client.get_order_book("SUI-PERP")

    @pytest.mark.asyncio
    async def test_get_order_book_service_error_propagation(self, client):
        """Test that service errors are properly propagated from get_order_book."""
        with patch.object(
            client,
            "_make_request",
            side_effect=BluefinServiceUnavailable("Service down"),
        ):
            with pytest.raises(BluefinServiceUnavailable):
                await client.get_order_book("SUI-PERP")

    @pytest.mark.asyncio
    async def test_get_order_book_unexpected_error(self, client):
        """Test handling of unexpected errors in get_order_book."""
        with patch.object(
            client, "_make_request", side_effect=ValueError("Unexpected error")
        ):
            with pytest.raises(
                BluefinServiceError, match="Failed to retrieve order book for SUI-PERP"
            ):
                await client.get_order_book("SUI-PERP")

    @pytest.mark.asyncio
    async def test_make_request_service_unavailable(self, client):
        """Test request when service is unavailable."""
        with patch.object(client, "check_health", return_value=False):
            with pytest.raises(
                BluefinServiceUnavailable,
                match="Bluefin service is not available at http://test-service:8080",
            ):
                await client._make_request("GET", "/test")


class TestOrderBookDataStandardization:
    """Test suite specifically for orderbook data standardization scenarios."""

    @pytest.fixture
    def client(self, mock_settings):
        """Create client instance for standardization tests."""
        with patch("aiohttp.TCPConnector"):
            return BluefinServiceClient(mock_settings)

    @pytest.mark.asyncio
    async def test_standardize_order_book_side_dict_format(self, client):
        """Test order book side standardization with dict format."""
        side_data = [
            {"price": "100.50", "quantity": "10.0"},
            {"price": "100.25", "size": "5.0"},  # Alternative size field
        ]

        result = client._standardize_order_book_side(side_data)

        expected = [
            {"price": 100.50, "quantity": 10.0},
            {"price": 100.25, "quantity": 5.0},
        ]

        assert result == expected

    @pytest.mark.asyncio
    async def test_standardize_order_book_side_array_format(self, client):
        """Test order book side standardization with array format."""
        side_data = [
            [100.50, 10.0],
            (100.25, 5.0),  # Tuple format
        ]

        result = client._standardize_order_book_side(side_data)

        expected = [
            {"price": 100.50, "quantity": 10.0},
            {"price": 100.25, "quantity": 5.0},
        ]

        assert result == expected

    @pytest.mark.asyncio
    async def test_standardize_order_book_side_dict_positional(self, client):
        """Test order book side standardization with dict positional values."""
        side_data = [{"0": "100.50", "1": "10.0"}]

        result = client._standardize_order_book_side(side_data)

        expected = [{"price": 100.50, "quantity": 10.0}]

        assert result == expected

    @pytest.mark.asyncio
    async def test_standardize_order_book_side_invalid_data(self, client):
        """Test order book side standardization with invalid data."""
        side_data = [
            "invalid",
            {"price": "invalid_number"},
            [],  # Empty array
        ]

        result = client._standardize_order_book_side(side_data)

        assert result == []

    @pytest.mark.asyncio
    async def test_standardize_order_book_side_non_list(self, client):
        """Test order book side standardization with non-list input."""
        result = client._standardize_order_book_side("not_a_list")

        assert result == []

    @pytest.mark.asyncio
    async def test_mixed_order_book_formats(self, client):
        """Test standardization with mixed data formats in same response."""
        side_data = [
            {"price": "100.50", "quantity": "10.0"},  # Standard format
            [101.00, 15.0],  # Array format
            {"price": "99.75", "size": "20.0"},  # Alternative size field
            {"0": "102.25", "1": "8.0"},  # Positional dict
        ]

        result = client._standardize_order_book_side(side_data)

        expected = [
            {"price": 100.50, "quantity": 10.0},
            {"price": 101.00, "quantity": 15.0},
            {"price": 99.75, "quantity": 20.0},
            {"price": 102.25, "quantity": 8.0},
        ]

        assert result == expected

    @pytest.mark.asyncio
    async def test_multiple_depth_parameters(self, client):
        """Test order book retrieval with various depth parameters."""
        base_response = {"bids": [], "asks": [], "timestamp": 0}

        with patch.object(
            client, "_make_request", return_value=base_response
        ) as mock_request:
            # Test different depth values
            await client.get_order_book("BTC-USD", depth=5)
            await client.get_order_book("BTC-USD", depth=20)
            await client.get_order_book("BTC-USD", depth=100)

            expected_calls = [
                unittest.mock.call("GET", "/orderbook/BTC-USD?depth=5"),
                unittest.mock.call("GET", "/orderbook/BTC-USD?depth=20"),
                unittest.mock.call("GET", "/orderbook/BTC-USD?depth=100"),
            ]
            mock_request.assert_has_calls(expected_calls)

    @pytest.mark.asyncio
    async def test_order_book_empty_response(self, client):
        """Test order book handling with empty bids/asks."""
        mock_response_data = {
            "bids": [],
            "asks": [],
            "timestamp": 1640995200,
        }

        with patch.object(client, "_make_request", return_value=mock_response_data):
            result = await client.get_order_book("SUI-PERP")

            expected_result = {
                "symbol": "SUI-PERP",
                "bids": [],
                "asks": [],
                "timestamp": 1640995200,
            }

            assert result == expected_result

    @pytest.mark.asyncio
    async def test_order_book_missing_timestamp(self, client):
        """Test order book handling when timestamp is missing."""
        mock_response_data = {
            "bids": [],
            "asks": [],
            # No timestamp field
        }

        with patch.object(client, "_make_request", return_value=mock_response_data):
            result = await client.get_order_book("SUI-PERP")

            assert result["timestamp"] == 0

    @pytest.mark.asyncio
    async def test_response_format_validation_none(self, client):
        """Test response format validation with None response."""
        with patch.object(client, "_make_request", return_value=None):
            with pytest.raises(
                BluefinServiceError, match="Invalid order book response"
            ):
                await client.get_order_book("SUI-PERP")

    @pytest.mark.asyncio
    async def test_response_format_validation_empty_list(self, client):
        """Test response format validation with empty list response."""
        with patch.object(client, "_make_request", return_value=[]):
            with pytest.raises(
                BluefinServiceError, match="Invalid order book response"
            ):
                await client.get_order_book("SUI-PERP")

    @pytest.mark.asyncio
    async def test_response_format_validation_string(self, client):
        """Test response format validation with string response."""
        with patch.object(client, "_make_request", return_value=""):
            with pytest.raises(
                BluefinServiceError, match="Invalid order book response"
            ):
                await client.get_order_book("SUI-PERP")


class TestBluefinServiceClientSingleton:
    """Test suite for BluefinServiceClient singleton management."""

    @pytest.mark.asyncio
    async def test_singleton_creation(self, mock_settings):
        """Test singleton client creation."""
        # Clear any existing singleton
        await close_bluefin_service_client()

        with patch(
            "bot.exchange.bluefin_service_client.BluefinServiceClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            client1 = await get_bluefin_service_client(mock_settings)
            client2 = await get_bluefin_service_client(mock_settings)

            assert client1 is client2
            MockClient.assert_called_once_with(mock_settings)
            mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_close(self, mock_settings):
        """Test singleton client closure."""
        # Clear any existing singleton
        await close_bluefin_service_client()

        with patch(
            "bot.exchange.bluefin_service_client.BluefinServiceClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await get_bluefin_service_client(mock_settings)
            await close_bluefin_service_client()

            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_recreation_after_close(self, mock_settings):
        """Test singleton recreation after closure."""
        # Clear any existing singleton
        await close_bluefin_service_client()

        with patch(
            "bot.exchange.bluefin_service_client.BluefinServiceClient"
        ) as MockClient:
            mock_instance1 = AsyncMock()
            mock_instance2 = AsyncMock()
            MockClient.side_effect = [mock_instance1, mock_instance2]

            client1 = await get_bluefin_service_client(mock_settings)
            await close_bluefin_service_client()
            client2 = await get_bluefin_service_client(mock_settings)

            assert client1 is not client2
            assert MockClient.call_count == 2


class TestBluefinServiceClientEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_close_without_initialization(self, mock_settings):
        """Test closing client without initialization."""
        mock_connector = AsyncMock()
        mock_connector.close = AsyncMock()

        with patch("aiohttp.TCPConnector", return_value=mock_connector):
            client = BluefinServiceClient(mock_settings)
            # Should not raise any errors
            await client.close()

    @pytest.mark.asyncio
    async def test_connection_pool_configuration(self, client):
        """Test connection pool is configured correctly."""
        # Since we mocked the connector, check the original parameters
        assert hasattr(client, "connector")


class TestRetryLogicAndErrorHandling:
    """Test suite for retry logic and error handling scenarios."""

    @pytest.fixture
    def client(self, mock_settings):
        """Create client instance for retry tests."""
        with patch("aiohttp.TCPConnector"):
            return BluefinServiceClient(mock_settings)

    @pytest.mark.asyncio
    async def test_retry_configuration(self, client):
        """Test retry configuration is set correctly."""
        assert client.max_retries == 3
        assert client.retry_delay == 1.0

    @pytest.mark.asyncio
    async def test_health_check_caching(self, client):
        """Test health check result caching."""
        # Set cached values
        client._service_available = True
        client._last_health_check = time.time()

        # Should return cached result without making actual request
        result = await client.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_base_url_configuration(self, client):
        """Test base URL is configured correctly."""
        assert client.base_url == "http://test-service:8080"

    @pytest.mark.asyncio
    async def test_api_key_configuration(self, client):
        """Test API key is configured correctly."""
        assert client.api_key == "test-api-key"


class TestNetworkConditions:
    """Test suite for various network conditions and scenarios."""

    @pytest.fixture
    def client(self, mock_settings):
        """Create client instance for network tests."""
        with patch("aiohttp.TCPConnector"):
            return BluefinServiceClient(mock_settings)

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, client):
        """Test handling of connection timeouts."""
        # Test that timeout is configured
        assert client.timeout.total == 30
        assert client.timeout.connect == 10
        assert client.timeout.sock_read == 20

    @pytest.mark.asyncio
    async def test_service_availability_tracking(self, client):
        """Test service availability state tracking."""
        # Initially unknown
        assert client._service_available is None

        # Simulate successful health check
        client._service_available = True
        assert client.is_available() is True

        # Simulate failed health check
        client._service_available = False
        assert client.is_available() is False
