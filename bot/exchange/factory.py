"""
Exchange factory for creating exchange clients based on configuration.

This module provides a factory pattern for instantiating the appropriate
exchange client based on the configuration settings.
"""

import logging
from typing import Any

from bot.config import Settings, settings

from .base import BaseExchange
from .bluefin import BluefinClient
from .coinbase import CoinbaseClient

logger = logging.getLogger(__name__)


class ExchangeFactory:
    """Factory class for creating exchange clients."""

    @staticmethod
    def create_exchange(
        settings_obj: Settings | None = None,
        exchange_type: str | None = None,
        dry_run: bool | None = None,
        **kwargs: Any,
    ) -> BaseExchange:
        """
        Create an exchange client based on configuration.

        Args:
            settings_obj: Configuration settings object
            exchange_type: Exchange type ('coinbase', 'bluefin').
                          Defaults to config setting.
            dry_run: Override dry run setting. Defaults to config setting.
            **kwargs: Additional exchange-specific parameters

        Returns:
            Exchange client instance

        Raises:
            ValueError: If exchange type is not supported
        """
        # Use config defaults if not specified
        if settings_obj is None:
            settings_obj = settings

        if exchange_type is None:
            exchange_type = getattr(settings_obj.exchange, "exchange_type", "coinbase")

        if dry_run is None:
            dry_run = settings_obj.system.dry_run

        exchange_type = exchange_type.lower()

        logger.info("Creating %s exchange client (dry_run=%s)", exchange_type, dry_run)

        if exchange_type == "coinbase":
            return ExchangeFactory._create_coinbase(dry_run, **kwargs)
        if exchange_type == "bluefin":
            return ExchangeFactory._create_bluefin(dry_run, **kwargs)
        raise ValueError(
            f"Unsupported exchange type: {exchange_type}. "
            "Supported exchanges: coinbase, bluefin"
        )

    @staticmethod
    def _create_coinbase(dry_run: bool, **kwargs: Any) -> CoinbaseClient:
        """Create a Coinbase exchange client."""
        # Extract Coinbase-specific parameters
        api_key = kwargs.get("api_key")
        api_secret = kwargs.get("api_secret")
        passphrase = kwargs.get("passphrase")
        cdp_api_key_name = kwargs.get("cdp_api_key_name")
        cdp_private_key = kwargs.get("cdp_private_key")

        # Create client
        client = CoinbaseClient(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            cdp_api_key_name=cdp_api_key_name,
            cdp_private_key=cdp_private_key,
            dry_run=dry_run,
        )

        logger.info(
            "Created Coinbase client (auth_method=%s, sandbox=%s)",
            client.auth_method,
            client.sandbox,
        )

        return client

    @staticmethod
    def _create_bluefin(dry_run: bool, **kwargs: Any) -> BluefinClient:
        """Create a Bluefin exchange client."""
        # Extract Bluefin-specific parameters
        private_key = kwargs.get("private_key")
        if not private_key:
            # Get from settings and extract string value if SecretStr or SecureString
            private_key_setting = getattr(
                settings.exchange, "bluefin_private_key", None
            )
            if private_key_setting is not None:
                if hasattr(private_key_setting, 'get_secret_value'):
                    # Pydantic SecretStr
                    private_key = private_key_setting.get_secret_value()
                elif hasattr(private_key_setting, 'get_value'):
                    # SecureString
                    private_key = private_key_setting.get_value()
                else:
                    # Fallback to string conversion
                    private_key = str(private_key_setting)
            else:
                private_key = None

        network_setting = getattr(settings.exchange, "bluefin_network", "mainnet")
        network = kwargs.get("network") or network_setting

        # Extract service URL from settings
        service_url_setting = getattr(
            settings.exchange, "bluefin_service_url", "http://bluefin-service:8080"
        )
        service_url = kwargs.get("service_url") or service_url_setting

        # Create client
        client = BluefinClient(
            private_key=private_key,
            network=network,
            service_url=service_url,
            dry_run=dry_run,
        )

        logger.info(
            "Created Bluefin client (network=%s, service_url=%s, has_key=%s)",
            network,
            service_url,
            bool(private_key),
        )

        return client

    @staticmethod
    def get_supported_exchanges() -> list[str]:
        """Get list of supported exchange types."""
        return ["coinbase", "bluefin"]
