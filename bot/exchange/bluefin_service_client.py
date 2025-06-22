"""
Bluefin service client with robust error handling and fallback mechanisms.

This module provides a resilient client for interacting with the Bluefin service
container, with proper error handling, connection pooling, and graceful degradation.
"""

import asyncio
import logging
from typing import Any

import aiohttp
from aiohttp import ClientConnectorError, ClientTimeout

from bot.config import Settings

logger = logging.getLogger(__name__)


class BluefinServiceError(Exception):
    """Base exception for Bluefin service errors."""


class BluefinServiceUnavailable(BluefinServiceError):
    """Exception raised when Bluefin service is unavailable."""


class BluefinServiceClient:
    """Client for interacting with Bluefin service container."""

    def __init__(self, settings: Settings):
        """Initialize Bluefin service client."""
        self.settings = settings
        self.base_url = getattr(
            settings.system, "bluefin_service_url", "http://bluefin-service:8080"
        )
        self.api_key = getattr(settings.system, "bluefin_service_api_key", None)
        self.timeout = ClientTimeout(total=30, connect=10, sock_read=20)
        self.max_retries = 3
        self.retry_delay = 1.0

        # Connection pool settings
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool limit
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache timeout
            enable_cleanup_closed=True,
        )

        self._session: aiohttp.ClientSession | None = None
        self._service_available = None  # Cache service availability
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, _exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize the client session."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers=self._get_default_headers(),
            )

        # Perform initial health check
        await self.check_health()

    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

        if self.connector:
            await self.connector.close()

    def _get_default_headers(self) -> dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "User-Agent": "AI-Trading-Bot-Bluefin-Client/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    async def check_health(self, force: bool = False) -> bool:
        """
        Check if Bluefin service is healthy.

        Args:
            force: Force health check even if recently checked

        Returns:
            True if service is healthy, False otherwise
        """
        import time

        current_time = time.time()

        # Use cached result if available and recent
        if not force and self._service_available is not None:
            if current_time - self._last_health_check < self._health_check_interval:
                return self._service_available

        try:
            if not self._session:
                await self.initialize()

            async with self._session.get(f"{self.base_url}/health") as response:
                self._service_available = response.status == 200
                self._last_health_check = current_time

                if self._service_available:
                    logger.debug("Bluefin service health check: OK")
                else:
                    logger.warning(
                        "Bluefin service health check failed: status=%d",
                        response.status,
                    )

                return self._service_available

        except ClientConnectorError as e:
            logger.warning(
                "Cannot connect to Bluefin service at %s: %s", self.base_url, str(e)
            )
            self._service_available = False
            self._last_health_check = current_time
            return False

        except TimeoutError:
            logger.warning("Bluefin service health check timeout")
            self._service_available = False
            self._last_health_check = current_time
            return False

        except Exception:
            logger.exception("Unexpected error during Bluefin service health check")
            self._service_available = False
            self._last_health_check = current_time
            return False

    async def _make_request(
        self, method: str, endpoint: str, data: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Make HTTP request to Bluefin service with retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional request parameters

        Returns:
            Response data

        Raises:
            BluefinServiceUnavailable: If service is not available
            BluefinServiceError: For other errors
        """
        if not self._session:
            await self.initialize()

        # Check service health first
        if not await self.check_health():
            raise BluefinServiceUnavailable(
                f"Bluefin service is not available at {self.base_url}"
            )

        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                async with self._session.request(
                    method, url, json=data, **kwargs
                ) as response:
                    if response.status == 200:
                        return await response.json()

                    # Handle specific error codes
                    if response.status == 503:
                        raise BluefinServiceUnavailable(
                            "Bluefin service temporarily unavailable (503)"
                        )

                    # Get error details
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("error", "Unknown error")
                    except Exception:
                        error_msg = await response.text()

                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "Bluefin service request failed (attempt %d/%d): %s",
                            attempt + 1,
                            self.max_retries,
                            error_msg,
                        )
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue

                    raise BluefinServiceError(
                        f"Bluefin service error (status={response.status}): {error_msg}"
                    )

            except ClientConnectorError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Connection error to Bluefin service (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        str(e),
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

                raise BluefinServiceUnavailable(
                    f"Cannot connect to Bluefin service: {e!s}"
                ) from e

            except TimeoutError as err:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Timeout calling Bluefin service (attempt %d/%d)",
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

                raise BluefinServiceError("Bluefin service request timeout") from err

            except BluefinServiceUnavailable:
                # Don't retry for service unavailable
                raise

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Unexpected error calling Bluefin service (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        str(e),
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

                raise BluefinServiceError(f"Unexpected error: {e!s}") from e

    async def get_balance(self, address: str) -> dict[str, Any]:
        """Get balance from Bluefin service."""
        return await self._make_request("GET", f"/balance/{address}")

    async def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Place order through Bluefin service."""
        return await self._make_request("POST", "/order", data=order_data)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel order through Bluefin service."""
        return await self._make_request("DELETE", f"/order/{order_id}")

    async def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data from Bluefin service."""
        return await self._make_request("GET", f"/market/{symbol}")

    def is_available(self) -> bool:
        """Check if service is currently available (cached result)."""
        return self._service_available is True


# Singleton instance management
_client_instance: BluefinServiceClient | None = None
_client_lock = asyncio.Lock()


async def get_bluefin_service_client(settings: Settings) -> BluefinServiceClient:
    """
    Get or create singleton Bluefin service client.

    Args:
        settings: Application settings

    Returns:
        BluefinServiceClient instance
    """
    global _client_instance

    async with _client_lock:
        if _client_instance is None:
            _client_instance = BluefinServiceClient(settings)
            await _client_instance.initialize()

        return _client_instance


async def close_bluefin_service_client():
    """Close the singleton Bluefin service client."""
    global _client_instance

    async with _client_lock:
        if _client_instance is not None:
            await _client_instance.close()
            _client_instance = None
