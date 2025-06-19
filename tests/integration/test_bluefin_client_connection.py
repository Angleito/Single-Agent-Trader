"""
Integration tests for Bluefin client connection and API authentication.

Tests the fixes implemented for Bluefin client connection establishment,
API authentication, and service communication reliability.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta, UTC
import aiohttp
import json

from bot.exchange.bluefin import BluefinClient
from bot.exchange.bluefin_client import BluefinServiceClient
from bot.exchange.base import ExchangeConnectionError, ExchangeAuthError
from bot.trading_types import Order, Position, MarketData, OrderStatus
from bot.utils.symbol_utils import BluefinSymbolConverter


class TestBluefinClientConnection:
    """Test Bluefin client connection establishment and management."""

    @pytest.fixture
    def mock_service_client(self):
        """Mock BluefinServiceClient for testing."""
        mock_client = AsyncMock(spec=BluefinServiceClient)
        mock_client.connect.return_value = True
        mock_client.is_connected.return_value = True
        mock_client.get_service_info.return_value = {
            "status": "healthy",
            "network": "testnet",
            "endpoint": "https://dapi.api.sui-staging.bluefin.io",
            "version": "2.0.0",
            "last_update": datetime.now(UTC).isoformat()
        }
        return mock_client

    @pytest.fixture
    def bluefin_client(self, mock_service_client):
        """Create BluefinClient instance with mocked service client."""
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True,
                service_url="http://localhost:8080"
            )
            client._service_client = mock_service_client
            return client

    @pytest.mark.asyncio
    async def test_bluefin_client_initialization(self, mock_service_client):
        """Test Bluefin client initialization with service connection."""
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test initialization
            await client.initialize()
            
            # Verify service client was created and connected
            assert hasattr(client, '_service_client')
            mock_service_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_bluefin_client_connection_success(self, bluefin_client, mock_service_client):
        """Test successful connection to Bluefin service."""
        # Test connection
        result = await bluefin_client.connect()
        
        assert result is True
        assert bluefin_client.is_connected()
        mock_service_client.connect.assert_called()

    @pytest.mark.asyncio
    async def test_bluefin_client_connection_failure(self, mock_service_client):
        """Test connection failure handling."""
        # Mock connection failure
        mock_service_client.connect.return_value = False
        mock_service_client.is_connected.return_value = False
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test connection failure
            result = await client.connect()
            
            assert result is False
            assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_bluefin_client_connection_retry(self, mock_service_client):
        """Test connection retry logic."""
        # Mock connection failure then success
        mock_service_client.connect.side_effect = [False, False, True]
        mock_service_client.is_connected.return_value = True
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test connection with retries
            result = await client.connect()
            
            assert result is True
            assert mock_service_client.connect.call_count == 3

    @pytest.mark.asyncio
    async def test_bluefin_client_authentication(self, bluefin_client, mock_service_client):
        """Test Bluefin API authentication."""
        # Mock authentication response
        mock_service_client.authenticate.return_value = {
            "success": True,
            "wallet_address": "0xabcdef1234567890",
            "authenticated": True
        }
        
        # Test authentication
        auth_result = await bluefin_client.authenticate()
        
        assert auth_result is True
        mock_service_client.authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_bluefin_client_authentication_failure(self, bluefin_client, mock_service_client):
        """Test authentication failure handling."""
        # Mock authentication failure
        mock_service_client.authenticate.side_effect = ExchangeAuthError("Invalid private key")
        
        # Test authentication failure
        with pytest.raises(ExchangeAuthError):
            await bluefin_client.authenticate()

    @pytest.mark.asyncio
    async def test_bluefin_client_service_health_check(self, bluefin_client, mock_service_client):
        """Test service health check functionality."""
        # Test health check
        health_info = await bluefin_client.get_service_health()
        
        assert health_info is not None
        assert health_info["status"] == "healthy"
        assert health_info["network"] == "testnet"
        mock_service_client.get_service_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_bluefin_client_network_configuration(self, mock_service_client):
        """Test network configuration (testnet vs mainnet)."""
        # Test testnet configuration
        testnet_client = BluefinClient(
            private_key="0x1234567890abcdef1234567890abcdef12345678",
            network="testnet",
            dry_run=True
        )
        
        assert testnet_client.network == "testnet"
        
        # Test mainnet configuration
        mainnet_client = BluefinClient(
            private_key="0x1234567890abcdef1234567890abcdef12345678",
            network="mainnet",
            dry_run=True
        )
        
        assert mainnet_client.network == "mainnet"

    @pytest.mark.asyncio
    async def test_bluefin_client_private_key_validation(self):
        """Test private key validation."""
        # Test invalid private key format
        with pytest.raises(ValueError):
            BluefinClient(
                private_key="invalid_key",
                network="testnet",
                dry_run=True
            )
        
        # Test missing private key
        with pytest.raises(ValueError):
            BluefinClient(
                private_key=None,
                network="testnet",
                dry_run=True
            )

    @pytest.mark.asyncio
    async def test_bluefin_client_service_url_configuration(self, mock_service_client):
        """Test service URL configuration."""
        custom_url = "http://custom-service:9090"
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client) as mock_client_class:
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True,
                service_url=custom_url
            )
            
            await client.initialize()
            
            # Verify service client was created with custom URL
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            assert custom_url in str(call_args) or any(custom_url in str(arg) for arg in call_args[0])


class TestBluefinServiceClient:
    """Test BluefinServiceClient functionality."""

    @pytest.fixture
    def mock_aiohttp_session(self):
        """Mock aiohttp session for HTTP requests."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "network": "testnet",
            "endpoint": "https://dapi.api.sui-staging.bluefin.io"
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aenter__.return_value = mock_response
        return mock_session

    @pytest.mark.asyncio
    async def test_service_client_connection(self, mock_aiohttp_session):
        """Test BluefinServiceClient connection."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            client = BluefinServiceClient(
                service_url="http://localhost:8080",
                network="testnet"
            )
            
            result = await client.connect()
            
            assert result is True
            mock_aiohttp_session.get.assert_called()

    @pytest.mark.asyncio
    async def test_service_client_get_service_info(self, mock_aiohttp_session):
        """Test getting service information."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            client = BluefinServiceClient(
                service_url="http://localhost:8080",
                network="testnet"
            )
            
            info = await client.get_service_info()
            
            assert info is not None
            assert info["status"] == "healthy"
            assert info["network"] == "testnet"

    @pytest.mark.asyncio
    async def test_service_client_http_error_handling(self, mock_aiohttp_session):
        """Test HTTP error handling."""
        # Mock HTTP error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            client = BluefinServiceClient(
                service_url="http://localhost:8080",
                network="testnet"
            )
            
            with pytest.raises(ExchangeConnectionError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_service_client_timeout_handling(self, mock_aiohttp_session):
        """Test timeout handling."""
        # Mock timeout
        mock_aiohttp_session.get.side_effect = asyncio.TimeoutError()
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            client = BluefinServiceClient(
                service_url="http://localhost:8080",
                network="testnet"
            )
            
            with pytest.raises(ExchangeConnectionError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_service_client_rate_limiting(self, mock_aiohttp_session):
        """Test rate limiting functionality."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            client = BluefinServiceClient(
                service_url="http://localhost:8080",
                network="testnet"
            )
            
            # Test multiple rapid requests
            tasks = []
            for _ in range(5):
                task = asyncio.create_task(client.get_service_info())
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All requests should complete successfully
            assert len(results) == 5
            assert all(result is not None for result in results)


class TestBluefinAPIAuthentication:
    """Test Bluefin API authentication mechanisms."""

    @pytest.fixture
    def mock_crypto_operations(self):
        """Mock cryptographic operations for testing."""
        with patch('bot.exchange.bluefin_client.sign_message') as mock_sign:
            mock_sign.return_value = "mocked_signature"
            yield mock_sign

    @pytest.mark.asyncio
    async def test_api_authentication_flow(self, mock_crypto_operations):
        """Test complete API authentication flow."""
        mock_service_client = AsyncMock()
        mock_service_client.authenticate.return_value = {
            "success": True,
            "wallet_address": "0xabcdef1234567890",
            "authenticated": True
        }
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test authentication
            result = await client.authenticate()
            
            assert result is True
            mock_service_client.authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_signature_validation(self, mock_crypto_operations):
        """Test API signature validation."""
        mock_service_client = AsyncMock()
        mock_service_client.validate_signature.return_value = True
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test signature validation
            is_valid = await client.validate_signature("test_message", "test_signature")
            
            assert is_valid is True
            mock_service_client.validate_signature.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_authentication_refresh(self, mock_crypto_operations):
        """Test authentication token refresh."""
        mock_service_client = AsyncMock()
        mock_service_client.refresh_authentication.return_value = {
            "success": True,
            "new_token": "new_auth_token",
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        }
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test authentication refresh
            result = await client.refresh_authentication()
            
            assert result is True
            mock_service_client.refresh_authentication.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_authentication_failure_recovery(self, mock_crypto_operations):
        """Test authentication failure recovery."""
        mock_service_client = AsyncMock()
        
        # First authentication fails, second succeeds
        mock_service_client.authenticate.side_effect = [
            ExchangeAuthError("Authentication failed"),
            {"success": True, "authenticated": True}
        ]
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # First attempt should raise exception
            with pytest.raises(ExchangeAuthError):
                await client.authenticate()
            
            # Second attempt should succeed
            result = await client.authenticate()
            assert result is True


class TestBluefinConnectionResilience:
    """Test connection resilience and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_connection_auto_recovery(self):
        """Test automatic connection recovery."""
        mock_service_client = AsyncMock()
        mock_service_client.is_connected.side_effect = [True, False, True]
        mock_service_client.connect.return_value = True
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Simulate connection check and recovery
            initial_status = client.is_connected()
            assert initial_status is True
            
            # Connection lost
            disconnected_status = client.is_connected()
            assert disconnected_status is False
            
            # Auto-recovery
            await client.connect()
            recovered_status = client.is_connected()
            assert recovered_status is True

    @pytest.mark.asyncio
    async def test_connection_heartbeat(self):
        """Test connection heartbeat mechanism."""
        mock_service_client = AsyncMock()
        mock_service_client.ping.return_value = True
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test heartbeat
            result = await client.ping()
            
            assert result is True
            mock_service_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test connection timeout handling."""
        mock_service_client = AsyncMock()
        mock_service_client.connect.side_effect = asyncio.TimeoutError()
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test timeout handling
            with pytest.raises(ExchangeConnectionError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connection_error_propagation(self):
        """Test proper error propagation from service client."""
        mock_service_client = AsyncMock()
        mock_service_client.connect.side_effect = Exception("Service unavailable")
        
        with patch('bot.exchange.bluefin.BluefinServiceClient', return_value=mock_service_client):
            client = BluefinClient(
                private_key="0x1234567890abcdef1234567890abcdef12345678",
                network="testnet",
                dry_run=True
            )
            
            # Test error propagation
            with pytest.raises(ExchangeConnectionError):
                await client.connect()