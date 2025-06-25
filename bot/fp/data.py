"""
Functional Data Layer

This module provides a functional replacement for the imperative data layer,
maintaining the same interface while using pure functional effects internally.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from bot.config import settings
from bot.trading_types import MarketData

from .adapters.market_data_adapter import (
    create_market_data_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from decimal import Decimal

logger = logging.getLogger(__name__)


class FunctionalMarketDataProvider:
    """
    Functional market data provider that maintains compatibility with the imperative interface
    while using functional effects internally.

    This class acts as a bridge between the imperative API expected by the rest of the system
    and the functional effects-based implementation.
    """

    def __init__(self, symbol: str | None = None, interval: str | None = None):
        """Initialize functional market data provider"""
        self.symbol = symbol or settings.trading.symbol
        self.interval = interval or settings.trading.interval

        # Create functional adapter
        self._adapter = create_market_data_adapter(
            exchange_type=settings.exchange.exchange_type,
            symbol=self.symbol,
            interval=self.interval,
        )

        # State tracking for compatibility
        self._connected = False
        self._subscribers: list[Callable[[MarketData], None]] = []

        logger.info(
            f"Initialized functional market data provider for {self.symbol} @ {self.interval}"
        )

    async def connect(self, fetch_historical: bool = True) -> None:
        """Connect to market data feeds using functional effects"""
        try:
            # Use functional effect to connect
            result = self._adapter.connect().run()

            if result.is_right():
                self._connected = True
                logger.info("Functional market data provider connected successfully")

                # Fetch initial historical data if requested
                if fetch_historical:
                    await self._fetch_initial_data()
            else:
                error = result.value if hasattr(result, "value") else "Unknown error"
                raise Exception(f"Failed to connect: {error}")

        except Exception as e:
            logger.exception(f"Connection failed: {e}")
            raise

    async def _fetch_initial_data(self) -> None:
        """Fetch initial historical data"""
        try:
            result = self._adapter.fetch_historical_data().run()
            if result.is_right():
                data_count = len(result.value)
                logger.info(f"Fetched {data_count} historical candles")
            else:
                logger.warning("Failed to fetch initial historical data")
        except Exception as e:
            logger.warning(f"Error fetching initial data: {e}")

    async def disconnect(self) -> None:
        """Disconnect from market data feeds"""
        try:
            self._adapter.disconnect().run()
            self._connected = False
            self._subscribers.clear()
            logger.info("Functional market data provider disconnected")
        except Exception as e:
            logger.exception(f"Disconnect error: {e}")

    async def fetch_historical_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        granularity: str | None = None,
    ) -> list[MarketData]:
        """Fetch historical OHLCV data"""
        try:
            result = self._adapter.fetch_historical_data(
                start_time=start_time,
                end_time=end_time,
                limit=settings.data.candle_limit,
            ).run()

            if result.is_right():
                # Convert functional OHLCV to MarketData for compatibility
                ohlcv_data = result.value
                return [
                    MarketData(
                        symbol=self.symbol,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                    )
                    for candle in ohlcv_data
                ]
            logger.error(f"Failed to fetch historical data: {result.value}")
            return []

        except Exception as e:
            logger.exception(f"Error fetching historical data: {e}")
            return []

    async def fetch_latest_price(self) -> Decimal | None:
        """Fetch latest price using functional effects"""
        try:
            result = self._adapter.get_latest_price().run()
            if result.is_right():
                return result.value
            logger.warning(f"Failed to get latest price: {result.value}")
            return None
        except Exception as e:
            logger.exception(f"Error fetching latest price: {e}")
            return None

    async def fetch_orderbook(self, level: int = 2) -> dict[str, Any] | None:
        """Fetch order book data"""
        try:
            result = self._adapter.fetch_orderbook(depth=level * 5).run()
            if result.is_right():
                orderbook = result.value
                return {
                    "bids": orderbook.bids,
                    "asks": orderbook.asks,
                    "timestamp": orderbook.timestamp,
                }
            logger.warning(f"Failed to fetch orderbook: {result.value}")
            return None
        except Exception as e:
            logger.exception(f"Error fetching orderbook: {e}")
            return None

    def get_latest_ohlcv(self, limit: int | None = None) -> list[MarketData]:
        """Get latest cached OHLCV data"""
        try:
            result = self._adapter.get_latest_ohlcv(limit=limit).run()
            if result.is_right():
                ohlcv_data = result.value
                return [
                    MarketData(
                        symbol=self.symbol,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                    )
                    for candle in ohlcv_data
                ]
            logger.warning(f"Failed to get cached OHLCV: {result.value}")
            return []
        except Exception as e:
            logger.exception(f"Error getting cached OHLCV: {e}")
            return []

    def get_latest_price(self) -> Decimal | None:
        """Get latest cached price"""
        try:
            result = self._adapter.get_latest_price().run()
            if result.is_right():
                return result.value
            return None
        except Exception:
            return None

    def to_dataframe(self, limit: int | None = None):
        """Convert OHLCV data to pandas DataFrame"""
        import pandas as pd

        data = self.get_latest_ohlcv(limit)
        if not data:
            return pd.DataFrame()

        df_data = []
        for candle in data:
            df_data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        market_df = pd.DataFrame(df_data)
        return market_df.set_index("timestamp")

    def subscribe_to_updates(self, callback: Callable[[MarketData], None]) -> None:
        """Subscribe to real-time data updates"""
        self._subscribers.append(callback)
        logger.debug(f"Added subscriber: {callback.__name__}")

        # Start real-time streaming if this is the first subscriber
        if len(self._subscribers) == 1:
            asyncio.create_task(self._start_real_time_streaming())

    def unsubscribe_from_updates(self, callback: Callable[[MarketData], None]) -> None:
        """Unsubscribe from real-time data updates"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Removed subscriber: {callback.__name__}")

    async def _start_real_time_streaming(self) -> None:
        """Start real-time data streaming using functional effects"""
        try:
            # Get the market data stream using functional effects
            stream_result = await self._adapter.stream_market_data().run()

            async for snapshot in stream_result:
                # Convert functional MarketSnapshot to MarketData for compatibility
                market_data = MarketData(
                    symbol=self.symbol,
                    timestamp=snapshot.timestamp,
                    open=snapshot.price,  # For real-time, open == current price
                    high=snapshot.price,  # For real-time, high == current price
                    low=snapshot.price,  # For real-time, low == current price
                    close=snapshot.price,
                    volume=snapshot.volume,
                )

                # Notify all subscribers
                for callback in self._subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(market_data)
                        else:
                            callback(market_data)
                    except Exception as e:
                        logger.exception(f"Error in subscriber callback: {e}")

        except Exception as e:
            logger.exception(f"Error in real-time streaming: {e}")

    def is_connected(self) -> bool:
        """Check if provider is connected"""
        try:
            result = self._adapter.is_connected().run()
            return result and self._connected
        except Exception:
            return False

    def has_websocket_data(self) -> bool:
        """Check if WebSocket data is being received"""
        return self.is_connected()

    async def wait_for_websocket_data(self, timeout: int = 30) -> bool:
        """Wait for WebSocket to start receiving data"""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.has_websocket_data():
                return True
            await asyncio.sleep(0.1)

        return False

    def get_data_status(self) -> dict[str, Any]:
        """Get comprehensive status information"""
        try:
            result = self._adapter.get_connection_status().run()
            return {
                **result,
                "functional_provider": True,
                "subscribers": len(self._subscribers),
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "functional_provider": True,
                "subscribers": len(self._subscribers),
            }

    def clear_cache(self) -> None:
        """Clear cached data (no-op for functional implementation)"""
        logger.debug("Clear cache called on functional provider (no-op)")

    def get_tick_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get recent tick data (simplified for functional implementation)"""
        # Return empty list as tick data is handled differently in functional approach
        return []


class FunctionalMarketDataClient:
    """
    Functional market data client that maintains compatibility with the imperative interface.
    """

    def __init__(self, symbol: str | None = None, interval: str | None = None):
        self.provider = FunctionalMarketDataProvider(symbol, interval)
        self._initialized = False

    async def __aenter__(self) -> FunctionalMarketDataClient:
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to market data feeds"""
        if not self._initialized:
            await self.provider.connect()
            self._initialized = True
            logger.info("FunctionalMarketDataClient connected successfully")

    async def disconnect(self) -> None:
        """Disconnect from market data feeds"""
        if self._initialized:
            await self.provider.disconnect()
            self._initialized = False
            logger.info("FunctionalMarketDataClient disconnected")

    async def get_historical_data(
        self, lookback_hours: int = 24, granularity: str | None = None
    ):
        """Get historical data as DataFrame"""
        from datetime import timedelta

        if not self._initialized:
            await self.connect()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        data = await self.provider.fetch_historical_data(
            start_time=start_time, end_time=end_time, granularity=granularity
        )

        return self._to_dataframe(data)

    async def get_current_price(self) -> Decimal | None:
        """Get current market price"""
        if not self._initialized:
            await self.connect()

        return await self.provider.fetch_latest_price()

    async def get_orderbook_snapshot(self, level: int = 2) -> dict[str, Any] | None:
        """Get orderbook snapshot"""
        if not self._initialized:
            await self.connect()

        return await self.provider.fetch_orderbook(level)

    def get_latest_ohlcv_dataframe(self, limit: int | None = None):
        """Get latest OHLCV data as DataFrame"""
        return self.provider.to_dataframe(limit)

    def subscribe_to_price_updates(
        self, callback: Callable[[MarketData], None]
    ) -> None:
        """Subscribe to real-time price updates"""
        self.provider.subscribe_to_updates(callback)

    def unsubscribe_from_price_updates(
        self, callback: Callable[[MarketData], None]
    ) -> None:
        """Unsubscribe from real-time price updates"""
        self.provider.unsubscribe_from_updates(callback)

    def get_connection_status(self) -> dict[str, Any]:
        """Get connection status"""
        return self.provider.get_data_status()

    def _to_dataframe(self, data: list[MarketData]):
        """Convert MarketData to DataFrame"""
        import pandas as pd

        if not data:
            return pd.DataFrame()

        df_data = []
        for candle in data:
            df_data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        historical_df = pd.DataFrame(df_data)
        historical_df = historical_df.set_index("timestamp")
        return historical_df.sort_index()


# Factory functions for compatibility


def create_market_data_client(
    symbol: str | None = None, interval: str | None = None
) -> FunctionalMarketDataClient:
    """Factory function to create a functional MarketDataClient"""
    return FunctionalMarketDataClient(symbol, interval)


# Export aliases for backward compatibility
MarketDataProvider = FunctionalMarketDataProvider
MarketDataClient = FunctionalMarketDataClient
