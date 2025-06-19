"""
Centralized endpoint configuration for Bluefin API services.

This module provides a single source of truth for all Bluefin API endpoints,
supporting both mainnet and testnet environments with proper URL validation.
"""

import os
from typing import Literal, NamedTuple


class BluefinEndpoints(NamedTuple):
    """Container for Bluefin API endpoints."""

    # REST API endpoints
    rest_api: str

    # WebSocket endpoints
    websocket_api: str
    websocket_notifications: str

    # Network identifier
    network: str


class BluefinEndpointConfig:
    """
    Centralized configuration for Bluefin API endpoints.

    Provides network-specific URL resolution with validation and fallback support.
    """

    # Official Bluefin API endpoints
    MAINNET_ENDPOINTS = BluefinEndpoints(
        rest_api="https://dapi.api.sui-prod.bluefin.io",
        websocket_api="wss://dapi.api.sui-prod.bluefin.io",
        websocket_notifications="wss://notifications.api.sui-prod.bluefin.io",
        network="mainnet",
    )

    TESTNET_ENDPOINTS = BluefinEndpoints(
        rest_api="https://dapi.api.sui-staging.bluefin.io",
        websocket_api="wss://dapi.api.sui-staging.bluefin.io",
        websocket_notifications="wss://notifications.api.sui-staging.bluefin.io",
        network="testnet",
    )

    @classmethod
    def get_endpoints(
        cls, network: Literal["mainnet", "testnet"] = "mainnet"
    ) -> BluefinEndpoints:
        """
        Get endpoints for the specified network.

        Args:
            network: Target network ("mainnet" or "testnet")

        Returns:
            BluefinEndpoints object with network-specific URLs

        Raises:
            ValueError: If network is not supported
        """
        if network == "mainnet":
            return cls.MAINNET_ENDPOINTS
        elif network == "testnet":
            return cls.TESTNET_ENDPOINTS
        else:
            raise ValueError(
                f"Unsupported network: {network}. Must be 'mainnet' or 'testnet'"
            )

    @classmethod
    def get_network_from_env(cls) -> str:
        """
        Get network configuration from environment variables.

        Returns:
            Network string ("mainnet" or "testnet")
        """
        network = os.getenv("EXCHANGE__BLUEFIN_NETWORK", "mainnet").lower()

        # Validate network value
        if network not in ["mainnet", "testnet"]:
            # Default to mainnet for invalid values
            network = "mainnet"

        return network

    @classmethod
    def get_current_endpoints(cls) -> BluefinEndpoints:
        """
        Get endpoints for the currently configured network.

        Uses EXCHANGE__BLUEFIN_NETWORK environment variable or defaults to mainnet.

        Returns:
            BluefinEndpoints object for current network
        """
        network = cls.get_network_from_env()
        return cls.get_endpoints(network)

    @classmethod
    def validate_endpoint_url(cls, url: str) -> bool:
        """
        Validate that a URL is a known Bluefin endpoint.

        Args:
            url: URL to validate

        Returns:
            True if URL is a known Bluefin endpoint
        """
        all_endpoints = [
            cls.MAINNET_ENDPOINTS.rest_api,
            cls.MAINNET_ENDPOINTS.websocket_api,
            cls.MAINNET_ENDPOINTS.websocket_notifications,
            cls.TESTNET_ENDPOINTS.rest_api,
            cls.TESTNET_ENDPOINTS.websocket_api,
            cls.TESTNET_ENDPOINTS.websocket_notifications,
        ]

        return url in all_endpoints

    @classmethod
    def is_public_endpoint(cls, endpoint_path: str) -> bool:
        """
        Check if an endpoint path requires authentication.

        Args:
            endpoint_path: API endpoint path (e.g., "/ticker24hr", "/candlestickData")

        Returns:
            True if endpoint is public (no auth required)
        """
        # Public market data endpoints that don't require authentication
        public_endpoints = {
            "/ticker24hr",
            "/candlestickData",
            "/klines",
            "/trades",
            "/orderbook",
            "/exchangeInfo",
            "/ping",
            "/time",
            "/depth",
            "/aggTrades",
            "/historicalTrades",
        }

        # Remove query parameters for comparison
        clean_path = endpoint_path.split("?")[0]

        return clean_path in public_endpoints


# Convenience functions for easy access
def get_rest_api_url(network: str = None) -> str:
    """Get REST API URL for specified network or current environment."""
    if network:
        return BluefinEndpointConfig.get_endpoints(network).rest_api
    return BluefinEndpointConfig.get_current_endpoints().rest_api


def get_websocket_url(network: str = None) -> str:
    """Get WebSocket API URL for specified network or current environment."""
    if network:
        return BluefinEndpointConfig.get_endpoints(network).websocket_api
    return BluefinEndpointConfig.get_current_endpoints().websocket_api


def get_notifications_url(network: str = None) -> str:
    """Get notifications WebSocket URL for specified network or current environment."""
    if network:
        return BluefinEndpointConfig.get_endpoints(network).websocket_notifications
    return BluefinEndpointConfig.get_current_endpoints().websocket_notifications


def is_public_endpoint(path: str) -> bool:
    """Check if endpoint requires authentication."""
    return BluefinEndpointConfig.is_public_endpoint(path)
