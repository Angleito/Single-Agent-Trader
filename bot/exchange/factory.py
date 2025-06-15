"""
Exchange factory for creating exchange clients based on configuration.

This module provides a factory pattern for instantiating the appropriate
exchange client based on the configuration settings.
"""

import logging
from typing import Optional

from ..config import settings
from .base import BaseExchange
from .bluefin import BluefinClient
from .coinbase import CoinbaseClient

logger = logging.getLogger(__name__)


class ExchangeFactory:
    """Factory class for creating exchange clients."""
    
    @staticmethod
    def create_exchange(
        exchange_type: Optional[str] = None,
        dry_run: Optional[bool] = None,
        **kwargs
    ) -> BaseExchange:
        """
        Create an exchange client based on configuration.
        
        Args:
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
        if exchange_type is None:
            exchange_type = getattr(settings.exchange, 'exchange_type', 'coinbase')
        
        if dry_run is None:
            dry_run = settings.system.dry_run
        
        exchange_type = exchange_type.lower()
        
        logger.info(
            f"Creating {exchange_type} exchange client "
            f"(dry_run={dry_run})"
        )
        
        if exchange_type == "coinbase":
            return ExchangeFactory._create_coinbase(dry_run, **kwargs)
        elif exchange_type == "bluefin":
            return ExchangeFactory._create_bluefin(dry_run, **kwargs)
        else:
            raise ValueError(
                f"Unsupported exchange type: {exchange_type}. "
                "Supported exchanges: coinbase, bluefin"
            )
    
    @staticmethod
    def _create_coinbase(dry_run: bool, **kwargs) -> CoinbaseClient:
        """Create a Coinbase exchange client."""
        # Extract Coinbase-specific parameters
        api_key = kwargs.get('api_key')
        api_secret = kwargs.get('api_secret')
        passphrase = kwargs.get('passphrase')
        cdp_api_key_name = kwargs.get('cdp_api_key_name')
        cdp_private_key = kwargs.get('cdp_private_key')
        
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
            "Created Coinbase client "
            f"(auth_method={client.auth_method}, "
            f"sandbox={client.sandbox})"
        )
        
        return client
    
    @staticmethod
    def _create_bluefin(dry_run: bool, **kwargs) -> BluefinClient:
        """Create a Bluefin exchange client."""
        # Extract Bluefin-specific parameters
        private_key = kwargs.get('private_key') or getattr(
            settings.exchange, 'bluefin_private_key', None
        )
        network = kwargs.get('network') or getattr(
            settings.exchange, 'bluefin_network', 'mainnet'
        )
        
        # Create client
        client = BluefinClient(
            private_key=private_key,
            network=network,
            dry_run=dry_run,
        )
        
        logger.info(
            "Created Bluefin client "
            f"(network={network}, "
            f"has_key={bool(private_key)})"
        )
        
        return client
    
    @staticmethod
    def get_supported_exchanges() -> list[str]:
        """Get list of supported exchange types."""
        return ["coinbase", "bluefin"]