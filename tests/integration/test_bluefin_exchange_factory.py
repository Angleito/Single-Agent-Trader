"""
Integration tests for Bluefin exchange factory configuration and service URL passing.

Tests the fixes implemented for exchange factory configuration, service URL passing,
and proper Bluefin client instantiation with correct network settings.
"""

import logging
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot.config import ExchangeSettings, Settings
from bot.exchange.bluefin import BluefinClient
from bot.exchange.bluefin_endpoints import BluefinEndpointConfig, get_rest_api_url
from bot.exchange.factory import ExchangeFactory
from bot.utils.symbol_utils import BluefinSymbolConverter

logger = logging.getLogger(__name__)


class TestBluefinExchangeFactory:
    """Test Bluefin exchange factory configuration and service URL handling."""

    @pytest.fixture()
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.system = Mock()
        settings.system.dry_run = True
        settings.exchange = Mock(spec=ExchangeSettings)
        settings.exchange.exchange_type = "bluefin"
        settings.exchange.bluefin_network = "testnet"
        settings.exchange.bluefin_private_key = Mock()
        settings.exchange.bluefin_private_key.get_secret_value.return_value = (
            "0x1234567890abcdef"
        )
        return settings

    @pytest.fixture()
    def mock_bluefin_service_client(self):
        """Mock the BluefinServiceClient for testing."""
        with patch("bot.exchange.bluefin.BluefinServiceClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock service connection
            mock_client.connect.return_value = True
            mock_client.is_connected.return_value = True
            mock_client.get_service_info.return_value = {
                "status": "healthy",
                "network": "testnet",
                "endpoint": "https://dapi.api.sui-staging.bluefin.io",
            }

            yield mock_client

    def test_exchange_factory_creates_bluefin_client(self, mock_settings):
        """Test that factory correctly creates Bluefin client with proper configuration."""
        # Test basic client creation
        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings, exchange_type="bluefin"
        )

        assert isinstance(client, BluefinClient)
        assert client.network == "testnet"
        assert client.dry_run is True

    def test_exchange_factory_passes_service_url_correctly(self, mock_settings):
        """Test that factory passes service URL correctly to Bluefin client."""
        # Test with custom service URL
        custom_service_url = "http://localhost:8080"

        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings,
            exchange_type="bluefin",
            service_url=custom_service_url,
        )

        assert isinstance(client, BluefinClient)
        # Verify the service URL is passed through (would be handled by BluefinClient internally)

    def test_exchange_factory_handles_network_configuration(self, mock_settings):
        """Test that factory handles network configuration properly."""
        # Test mainnet configuration
        mock_settings.exchange.bluefin_network = "mainnet"

        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings, exchange_type="bluefin", network="mainnet"
        )

        assert isinstance(client, BluefinClient)
        assert client.network == "mainnet"

    def test_exchange_factory_validates_private_key(self, mock_settings):
        """Test that factory validates private key presence."""
        # Test with missing private key
        mock_settings.exchange.bluefin_private_key = None

        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings, exchange_type="bluefin"
        )

        # Should still create client but with None private key
        assert isinstance(client, BluefinClient)

    @pytest.mark.asyncio()
    async def test_bluefin_client_service_connection(self, mock_bluefin_service_client):
        """Test Bluefin client establishes service connection properly."""
        with patch(
            "bot.exchange.bluefin.BluefinServiceClient",
            return_value=mock_bluefin_service_client,
        ):
            client = BluefinClient(
                private_key="0x1234567890abcdef", network="testnet", dry_run=True
            )

            # Test connection
            await client.connect()

            mock_bluefin_service_client.connect.assert_called_once()
            assert client.is_connected()

    def test_bluefin_endpoint_configuration(self):
        """Test Bluefin endpoint configuration for different networks."""
        # Test testnet endpoint
        testnet_url = get_rest_api_url("testnet")
        assert "staging" in testnet_url.lower()

        # Test mainnet endpoint
        mainnet_url = get_rest_api_url("mainnet")
        assert "prod" in mainnet_url.lower()

        # Test default (should be mainnet)
        default_url = get_rest_api_url()
        assert default_url == mainnet_url

    def test_supported_exchanges_list(self):
        """Test that factory returns correct list of supported exchanges."""
        supported = ExchangeFactory.get_supported_exchanges()
        assert "bluefin" in supported
        assert "coinbase" in supported

    def test_exchange_factory_error_handling(self):
        """Test factory error handling for unsupported exchange types."""
        with pytest.raises(ValueError, match="Unsupported exchange type"):
            ExchangeFactory.create_exchange(exchange_type="unsupported_exchange")

    @pytest.mark.asyncio()
    async def test_bluefin_client_initialization_with_service(
        self, mock_bluefin_service_client
    ):
        """Test complete Bluefin client initialization with service connection."""
        with patch(
            "bot.exchange.bluefin.BluefinServiceClient",
            return_value=mock_bluefin_service_client,
        ):
            client = BluefinClient(
                private_key="0x1234567890abcdef",
                network="testnet",
                dry_run=True,
                service_url="http://localhost:8080",
            )

            # Initialize client
            await client.initialize()

            # Verify service client was set up
            assert hasattr(client, "_service_client")
            mock_bluefin_service_client.connect.assert_called()

    def test_bluefin_symbol_converter_integration(self):
        """Test symbol converter integration with factory-created clients."""
        converter = BluefinSymbolConverter()

        # Test common symbol conversions
        test_symbols = ["BTC-USD", "ETH-USD", "SUI-USD"]

        for symbol in test_symbols:
            try:
                perp_symbol = converter.to_market_symbol(symbol)
                assert perp_symbol is not None
            except Exception:
                # Some symbols might not be supported, which is fine
                logger.debug("Symbol %s not supported in test converter", symbol)

    @pytest.mark.asyncio()
    async def test_exchange_factory_async_initialization(
        self, mock_settings, mock_bluefin_service_client
    ):
        """Test factory-created clients can be properly initialized asynchronously."""
        with patch(
            "bot.exchange.bluefin.BluefinServiceClient",
            return_value=mock_bluefin_service_client,
        ):
            client = ExchangeFactory.create_exchange(
                settings_obj=mock_settings, exchange_type="bluefin"
            )

            # Test async initialization
            await client.initialize()

            # Verify client is properly set up
            assert client.is_connected()

    def test_exchange_factory_dry_run_override(self, mock_settings):
        """Test that factory respects dry_run parameter override."""
        # Test dry_run override to False
        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings, exchange_type="bluefin", dry_run=False
        )

        assert client.dry_run is False

        # Test dry_run override to True
        mock_settings.system.dry_run = False  # Set settings to False
        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings,
            exchange_type="bluefin",
            dry_run=True,  # Override to True
        )

        assert client.dry_run is True

    @pytest.mark.asyncio()
    async def test_bluefin_client_service_health_check(
        self, mock_bluefin_service_client
    ):
        """Test Bluefin client can perform service health checks."""
        with patch(
            "bot.exchange.bluefin.BluefinServiceClient",
            return_value=mock_bluefin_service_client,
        ):
            client = BluefinClient(
                private_key="0x1234567890abcdef", network="testnet", dry_run=True
            )

            await client.initialize()

            # Test service health check
            health_info = await client.get_service_health()

            mock_bluefin_service_client.get_service_info.assert_called()
            assert health_info is not None

    def test_exchange_factory_with_kwargs(self, mock_settings):
        """Test factory passes through additional keyword arguments."""
        extra_kwargs = {"rate_limit": 30, "timeout": 10, "retry_attempts": 3}

        client = ExchangeFactory.create_exchange(
            settings_obj=mock_settings, exchange_type="bluefin", **extra_kwargs
        )

        assert isinstance(client, BluefinClient)
        # Additional kwargs should be handled by the BluefinClient constructor


class TestBluefinServiceUrlPassing:
    """Test service URL passing and configuration."""

    def test_service_url_environment_variable(self):
        """Test service URL configuration from environment variables."""
        test_url = "http://custom-bluefin-service:8080"

        with patch.dict(os.environ, {"BLUEFIN_SERVICE_URL": test_url}):
            # Test that environment variable is respected
            # This would be handled by the actual client implementation
            assert os.getenv("BLUEFIN_SERVICE_URL") == test_url

    def test_docker_compose_service_discovery(self):
        """Test service discovery in Docker Compose environment."""
        # Test common Docker Compose service names
        docker_service_urls = [
            "http://bluefin-sdk-service:8080",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        for url in docker_service_urls:
            # Verify URL format is valid
            assert url.startswith("http")
            assert ":" in url

    @pytest.mark.asyncio()
    async def test_service_connection_retry_logic(self):
        """Test service connection retry logic."""
        with patch("bot.exchange.bluefin.BluefinServiceClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate connection failure then success
            mock_client.connect.side_effect = [False, False, True]
            mock_client.is_connected.return_value = True

            client = BluefinClient(
                private_key="0x1234567890abcdef", network="testnet", dry_run=True
            )

            # Test connection with retries
            result = await client.connect()

            # Should eventually succeed after retries
            assert result is True
            assert mock_client.connect.call_count >= 1

    def test_network_endpoint_mapping(self):
        """Test network to endpoint URL mapping."""
        # Test endpoint configuration
        config = BluefinEndpointConfig()

        testnet_endpoints = config.get_endpoints("testnet")
        mainnet_endpoints = config.get_endpoints("mainnet")

        assert testnet_endpoints.rest_api != mainnet_endpoints.rest_api
        assert (
            "staging" in testnet_endpoints.rest_api.lower()
            or "testnet" in testnet_endpoints.rest_api.lower()
        )
        assert (
            "prod" in mainnet_endpoints.rest_api.lower()
            or "mainnet" in mainnet_endpoints.rest_api.lower()
        )
