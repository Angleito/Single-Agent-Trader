"""Unit tests for exchange factory."""

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from bot.exchange.base import BaseExchange
from bot.exchange.bluefin import BluefinClient
from bot.exchange.coinbase import CoinbaseClient
from bot.exchange.factory import ExchangeFactory


def generate_fake_test_credentials():
    """Generate clearly fake test credentials to avoid security lint warnings."""
    test_id = str(uuid.uuid4())[:8]
    return {
        "api_key": f"fake_test_api_key_{test_id}",
        "api_secret": f"fake_test_api_secret_{test_id}_not_real",
        "passphrase": f"fake_test_passphrase_{test_id}_not_real",
        "cdp_api_key_name": f"fake_test_cdp_key_{test_id}",
        "cdp_private_key": f"fake_test_cdp_private_{test_id}",
    }


class TestExchangeFactory:
    """Test cases for the exchange factory."""

    @patch("bot.exchange.factory.CoinbaseClient")
    def test_create_coinbase_exchange(self, mock_coinbase_client: Any) -> None:
        """Test that ExchangeFactory correctly creates Coinbase exchange when exchange_type='coinbase'."""
        # Mock the CoinbaseClient instance
        mock_instance = MagicMock(spec=CoinbaseClient)
        mock_instance.auth_method = "cdp"
        mock_instance.sandbox = False
        mock_coinbase_client.return_value = mock_instance

        # Create exchange with explicit type
        exchange = ExchangeFactory.create_exchange(
            exchange_type="coinbase", dry_run=False
        )

        # Verify CoinbaseClient was instantiated
        mock_coinbase_client.assert_called_once_with(
            api_key=None,
            api_secret=None,
            passphrase=None,
            cdp_api_key_name=None,
            cdp_private_key=None,
            dry_run=False,
        )

        # Verify return type
        assert exchange == mock_instance
        assert isinstance(exchange, BaseExchange)

    @patch("bot.exchange.factory.BluefinClient")
    def test_create_bluefin_exchange(self, mock_bluefin_client: Any) -> None:
        """Test that ExchangeFactory correctly creates Bluefin exchange when exchange_type='bluefin'."""
        # Mock the BluefinClient instance
        mock_instance = MagicMock(spec=BluefinClient)
        mock_bluefin_client.return_value = mock_instance

        # Create exchange with explicit type
        exchange = ExchangeFactory.create_exchange(
            exchange_type="bluefin",
            dry_run=True,
            private_key="test_private_key",
            network="testnet",
        )

        # Verify BluefinClient was instantiated
        mock_bluefin_client.assert_called_once_with(
            private_key="test_private_key",
            network="testnet",
            dry_run=True,
        )

        # Verify return type
        assert exchange == mock_instance
        assert isinstance(exchange, BaseExchange)

    def test_invalid_exchange_type_raises_error(self) -> None:
        """Test that ExchangeFactory raises ValueError for invalid exchange types."""
        with pytest.raises(
            ValueError,
            match=r"Unsupported exchange type: invalid_exchange\. Supported exchanges: coinbase, bluefin",
        ):
            ExchangeFactory.create_exchange(exchange_type="invalid_exchange")

    @patch("bot.exchange.factory.settings")
    @patch("bot.exchange.factory.CoinbaseClient")
    def test_uses_settings_when_no_explicit_type(
        self, mock_coinbase_client: Any, mock_settings: Any
    ) -> None:
        """Test that the factory uses settings.exchange.exchange_type when no explicit type is provided."""
        # Mock settings
        mock_settings.exchange.exchange_type = "coinbase"
        mock_settings.system.dry_run = True

        # Mock the CoinbaseClient instance
        mock_instance = MagicMock(spec=CoinbaseClient)
        mock_instance.auth_method = "cdp"
        mock_instance.sandbox = True
        mock_coinbase_client.return_value = mock_instance

        # Create exchange without explicit type
        exchange = ExchangeFactory.create_exchange()

        # Verify settings were used
        mock_coinbase_client.assert_called_once_with(
            api_key=None,
            api_secret=None,
            passphrase=None,
            cdp_api_key_name=None,
            cdp_private_key=None,
            dry_run=True,
        )

        assert exchange == mock_instance

    @patch("bot.exchange.factory.settings")
    @patch("bot.exchange.factory.BluefinClient")
    def test_bluefin_uses_settings_private_key(
        self, mock_bluefin_client, mock_settings
    ):
        """Test that Bluefin creation uses private key from settings when not provided."""
        # Mock settings
        mock_settings.exchange.exchange_type = "bluefin"
        mock_settings.exchange.bluefin_private_key = SecretStr("settings_private_key")
        mock_settings.exchange.bluefin_network = "mainnet"
        mock_settings.system.dry_run = False

        # Mock the BluefinClient instance
        mock_instance = MagicMock(spec=BluefinClient)
        mock_bluefin_client.return_value = mock_instance

        # Create exchange without explicit parameters
        exchange = ExchangeFactory.create_exchange()

        # Verify settings were used
        mock_bluefin_client.assert_called_once_with(
            private_key="settings_private_key",
            network="mainnet",
            dry_run=False,
        )

        assert exchange == mock_instance

    @patch("bot.exchange.factory.CoinbaseClient")
    def test_coinbase_with_all_parameters(self, mock_coinbase_client):
        """Test Coinbase creation with all parameters provided."""
        # Mock the CoinbaseClient instance
        mock_instance = MagicMock(spec=CoinbaseClient)
        mock_instance.auth_method = "cdp"
        mock_instance.sandbox = False
        mock_coinbase_client.return_value = mock_instance

        # Create exchange with all parameters (using dynamically generated fake credentials)
        fake_creds = generate_fake_test_credentials()
        exchange = ExchangeFactory.create_exchange(
            exchange_type="coinbase",
            dry_run=False,
            api_key=fake_creds["api_key"],
            api_secret=fake_creds["api_secret"],
            passphrase=fake_creds["passphrase"],
            cdp_api_key_name=fake_creds["cdp_api_key_name"],
            cdp_private_key=fake_creds["cdp_private_key"],
        )

        # Verify all parameters were passed
        mock_coinbase_client.assert_called_once_with(
            api_key=fake_creds["api_key"],
            api_secret=fake_creds["api_secret"],
            passphrase=fake_creds["passphrase"],
            cdp_api_key_name=fake_creds["cdp_api_key_name"],
            cdp_private_key=fake_creds["cdp_private_key"],
            dry_run=False,
        )

        assert exchange == mock_instance

    def test_get_supported_exchanges(self):
        """Test that get_supported_exchanges returns correct list."""
        supported = ExchangeFactory.get_supported_exchanges()

        assert supported == ["coinbase", "bluefin"]
        assert len(supported) == 2
        assert "coinbase" in supported
        assert "bluefin" in supported

    @patch("bot.exchange.factory.CoinbaseClient")
    def test_case_insensitive_exchange_type(self, mock_coinbase_client):
        """Test that exchange type is case-insensitive."""
        # Mock the CoinbaseClient instance
        mock_instance = MagicMock(spec=CoinbaseClient)
        mock_instance.auth_method = "cdp"
        mock_instance.sandbox = False
        mock_coinbase_client.return_value = mock_instance

        # Test with uppercase
        exchange1 = ExchangeFactory.create_exchange(exchange_type="COINBASE")
        # Test with mixed case
        exchange2 = ExchangeFactory.create_exchange(exchange_type="CoInBaSe")

        # Both should create Coinbase clients
        assert mock_coinbase_client.call_count == 2
        assert exchange1 == mock_instance
        assert exchange2 == mock_instance

    @patch("bot.exchange.factory.logger")
    @patch("bot.exchange.factory.CoinbaseClient")
    def test_logging_coinbase_creation(self, mock_coinbase_client, mock_logger):
        """Test that appropriate logs are generated for Coinbase creation."""
        # Mock the CoinbaseClient instance
        mock_instance = MagicMock(spec=CoinbaseClient)
        mock_instance.auth_method = "cdp"
        mock_instance.sandbox = True
        mock_coinbase_client.return_value = mock_instance

        # Create exchange
        ExchangeFactory.create_exchange(exchange_type="coinbase", dry_run=True)

        # Verify logging
        mock_logger.info.assert_any_call(
            "Creating coinbase exchange client (dry_run=True)"
        )
        mock_logger.info.assert_any_call(
            "Created Coinbase client (auth_method=cdp, sandbox=True)"
        )

    @patch("bot.exchange.factory.logger")
    @patch("bot.exchange.factory.BluefinClient")
    def test_logging_bluefin_creation(self, mock_bluefin_client, mock_logger):
        """Test that appropriate logs are generated for Bluefin creation."""
        # Mock the BluefinClient instance
        mock_instance = MagicMock(spec=BluefinClient)
        mock_bluefin_client.return_value = mock_instance

        # Create exchange
        ExchangeFactory.create_exchange(
            exchange_type="bluefin",
            dry_run=False,
            private_key="test_key",
            network="testnet",
        )

        # Verify logging
        mock_logger.info.assert_any_call(
            "Creating bluefin exchange client (dry_run=False)"
        )
        mock_logger.info.assert_any_call(
            "Created Bluefin client (network=testnet, has_key=True)"
        )

    @patch("bot.exchange.factory.settings")
    @patch("bot.exchange.factory.BluefinClient")
    def test_bluefin_no_private_key_provided(self, mock_bluefin_client, mock_settings):
        """Test Bluefin creation when no private key is available."""
        # Mock settings without private key
        mock_settings.exchange.bluefin_private_key = None
        mock_settings.exchange.bluefin_network = "mainnet"
        mock_settings.system.dry_run = True

        # Mock the BluefinClient instance
        mock_instance = MagicMock(spec=BluefinClient)
        mock_bluefin_client.return_value = mock_instance

        # Create exchange
        exchange = ExchangeFactory.create_exchange(exchange_type="bluefin")

        # Verify None was passed for private_key
        mock_bluefin_client.assert_called_once_with(
            private_key=None,
            network="mainnet",
            dry_run=True,
        )

        assert exchange == mock_instance

    @patch("bot.exchange.factory.CoinbaseClient")
    def test_dry_run_override(self, mock_coinbase_client):
        """Test that dry_run parameter overrides settings."""
        # Mock the CoinbaseClient instance
        mock_instance = MagicMock(spec=CoinbaseClient)
        mock_instance.auth_method = "cdp"
        mock_instance.sandbox = False
        mock_coinbase_client.return_value = mock_instance

        # Create exchange with explicit dry_run=False
        with patch("bot.exchange.factory.settings") as mock_settings:
            mock_settings.system.dry_run = True  # Settings say dry_run=True

            ExchangeFactory.create_exchange(
                exchange_type="coinbase",
                dry_run=False,  # But we override with False
            )

            # Verify dry_run=False was used
            mock_coinbase_client.assert_called_once()
            call_args = mock_coinbase_client.call_args[1]
            assert call_args["dry_run"] is False
