"""
Bluefin Service Client for connecting to the isolated Bluefin SDK service.

This client communicates with the Bluefin service container that has the
actual Bluefin SDK installed, avoiding dependency conflicts.
"""

import logging
import os
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class BluefinServiceClient:
    """
    Client for communicating with the Bluefin SDK service container.

    The service provides a REST API wrapper around the Bluefin SDK,
    allowing the main bot to interact with Bluefin without dependency conflicts.
    """

    def __init__(
        self,
        service_url: str = "http://bluefin-service:8080",
        api_key: str | None = None,
    ):
        """
        Initialize the Bluefin service client.

        Args:
            service_url: URL of the Bluefin service container
            api_key: API key for authentication (if not provided, will use BLUEFIN_SERVICE_API_KEY env var)
        """
        self.service_url = service_url
        self._session: aiohttp.ClientSession | None = None
        self._connected = False

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("BLUEFIN_SERVICE_API_KEY")
        if not self.api_key:
            logger.warning(
                "No BLUEFIN_SERVICE_API_KEY configured - authentication may fail"
            )

        # Prepare headers with authentication
        self._headers = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    async def connect(self) -> bool:
        """
        Connect to the Bluefin service.

        Returns:
            True if connection successful
        """
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession(headers=self._headers)

            # Check service health (health endpoint doesn't require auth)
            async with self._session.get(f"{self.service_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._connected = data.get("status") == "healthy"
                    logger.info(f"Connected to Bluefin service: {data}")
                    return self._connected
                else:
                    logger.error(f"Bluefin service unhealthy: {resp.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to Bluefin service: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the Bluefin service."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def get_account_data(self) -> dict[str, Any]:
        """
        Get account data including balances.

        Returns:
            Account data dictionary
        """
        try:
            async with self._session.get(f"{self.service_url}/account") as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 401:
                    logger.error(
                        "Authentication failed - check BLUEFIN_SERVICE_API_KEY"
                    )
                    return {"error": "Authentication failed"}
                elif resp.status == 429:
                    retry_after = resp.headers.get("Retry-After", "60")
                    logger.error(
                        f"Rate limit exceeded - retry after {retry_after} seconds"
                    )
                    return {"error": f"Rate limit exceeded, retry after {retry_after}s"}
                else:
                    error_text = await resp.text()
                    logger.error(f"Failed to get account data: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return {}

    async def get_user_positions(self) -> list[dict[str, Any]]:
        """
        Get current user positions.

        Returns:
            List of position dictionaries
        """
        try:
            async with self._session.get(f"{self.service_url}/positions") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    positions = data.get("positions", [])
                    logger.info(
                        f"Retrieved {len(positions)} positions from Bluefin service"
                    )
                    return positions
                elif resp.status == 401:
                    logger.error(
                        "Authentication failed - check BLUEFIN_SERVICE_API_KEY"
                    )
                    return []
                elif resp.status == 429:
                    retry_after = resp.headers.get("Retry-After", "60")
                    logger.error(
                        f"Rate limit exceeded - retry after {retry_after} seconds"
                    )
                    return []
                else:
                    error_text = await resp.text()
                    logger.error(f"Failed to get positions: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """
        Place an order through the Bluefin service.

        Args:
            order_data: Order details including symbol, side, quantity, etc.

        Returns:
            Order response dictionary
        """
        try:
            async with self._session.post(
                f"{self.service_url}/orders", json=order_data
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 401:
                    logger.error(
                        "Authentication failed - check BLUEFIN_SERVICE_API_KEY"
                    )
                    return {"status": "error", "message": "Authentication failed"}
                elif resp.status == 429:
                    retry_after = resp.headers.get("Retry-After", "60")
                    logger.error(
                        f"Rate limit exceeded - retry after {retry_after} seconds"
                    )
                    return {
                        "status": "error",
                        "message": f"Rate limit exceeded, retry after {retry_after}s",
                    }
                else:
                    error_text = await resp.text()
                    logger.error(f"Failed to place order: {error_text}")
                    return {"status": "error", "message": error_text}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            async with self._session.delete(
                f"{self.service_url}/orders/{order_id}"
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

    async def get_market_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get market ticker data.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker data dictionary
        """
        try:
            async with self._session.get(
                f"{self.service_url}/market/ticker", params={"symbol": symbol}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"price": "0"}
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return {"price": "0"}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage value

        Returns:
            True if successful
        """
        try:
            async with self._session.post(
                f"{self.service_url}/leverage",
                json={"symbol": symbol, "leverage": leverage},
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False

    async def get_candlestick_data(self, params: dict[str, Any]) -> list[list[Any]]:
        """
        Get historical candlestick data.

        Args:
            params: Parameters including symbol, interval, and limit

        Returns:
            List of candlestick arrays [timestamp, open, high, low, close, volume]
        """
        try:
            async with self._session.get(
                f"{self.service_url}/market/candles", params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("candles", [])
                else:
                    error_text = await resp.text()
                    logger.error(f"Failed to get candlestick data: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting candlestick data: {e}")
            return []
