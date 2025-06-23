"""
Market Data Effects for Functional Trading Bot

This module provides functional effects for market data operations including
WebSocket connections, REST API calls, and data streaming.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import aiohttp
import websockets

from ..types.effects import RateLimit, RetryPolicy, WebSocketConnection
from ..types.market import Candle, MarketDataStream, OrderBook, Subscription, Trade
from .io import IO, AsyncIO, IOEither, from_try


@dataclass
class ConnectionConfig:
    """WebSocket connection configuration"""

    url: str
    headers: dict[str, str]
    heartbeat_interval: int = 30
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0


@dataclass
class APIConfig:
    """REST API configuration"""

    base_url: str
    headers: dict[str, str]
    timeout: int = 30
    rate_limit: RateLimit


# WebSocket Effects


def connect_websocket(
    config: ConnectionConfig,
) -> IOEither[Exception, WebSocketConnection]:
    """Create WebSocket connection effect"""

    async def connect():
        try:
            websocket = await websockets.connect(
                config.url,
                extra_headers=config.headers,
                ping_interval=config.heartbeat_interval,
            )
            return WebSocketConnection(
                websocket=websocket, config=config, is_connected=True, subscriptions=[]
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to WebSocket: {e}")

    return from_try(lambda: asyncio.run(connect()))


def subscribe_to_symbol(
    symbol: str, channels: list[str], connection: WebSocketConnection
) -> IOEither[Exception, Subscription]:
    """Subscribe to market data for a symbol"""

    def subscribe():
        if not connection.is_connected:
            raise ConnectionError("WebSocket not connected")

        subscription_msg = {
            "type": "subscribe",
            "channels": channels,
            "product_ids": [symbol],
        }

        # Send subscription message
        asyncio.run(connection.websocket.send(str(subscription_msg)))

        subscription = Subscription(
            symbol=symbol, channels=channels, active=True, created_at=datetime.utcnow()
        )

        connection.subscriptions.append(subscription)
        return subscription

    return from_try(subscribe)


def stream_market_data(
    connection: WebSocketConnection,
) -> AsyncIO[AsyncIterator[dict[str, Any]]]:
    """Stream market data from WebSocket connection"""

    async def stream():
        if not connection.is_connected:
            raise ConnectionError("WebSocket not connected")

        async for message in connection.websocket:
            yield json.loads(message)  # Parse JSON message safely

    return AsyncIO.pure(stream())


def reconnect_websocket(
    connection: WebSocketConnection, retry_policy: RetryPolicy
) -> IOEither[Exception, WebSocketConnection]:
    """Reconnect WebSocket with retry logic"""

    def reconnect():
        attempts = 0
        while attempts < retry_policy.max_attempts:
            try:
                # Close existing connection
                if connection.websocket and not connection.websocket.closed:
                    asyncio.run(connection.websocket.close())

                # Reconnect
                new_connection = connect_websocket(connection.config).run()
                if new_connection.is_right():
                    # Resubscribe to all channels
                    for sub in connection.subscriptions:
                        subscribe_to_symbol(
                            sub.symbol, sub.channels, new_connection.value
                        )
                    return new_connection.value

                attempts += 1
                if attempts < retry_policy.max_attempts:
                    asyncio.sleep(
                        retry_policy.delay * (2**attempts)
                    )  # Exponential backoff

            except Exception as e:
                attempts += 1
                if attempts >= retry_policy.max_attempts:
                    raise e

        raise ConnectionError(
            f"Failed to reconnect after {retry_policy.max_attempts} attempts"
        )

    return from_try(reconnect)


# REST API Effects


def fetch_candles(
    symbol: str, interval: str, limit: int, api_config: APIConfig
) -> IOEither[Exception, list[Candle]]:
    """Fetch historical candles from REST API"""

    async def fetch():
        url = f"{api_config.base_url}/candles"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        async with aiohttp.ClientSession(
            headers=api_config.headers,
            timeout=aiohttp.ClientTimeout(total=api_config.timeout),
        ) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")

                data = await response.json()
                return [
                    Candle(
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        open=Decimal(item["open"]),
                        high=Decimal(item["high"]),
                        low=Decimal(item["low"]),
                        close=Decimal(item["close"]),
                        volume=Decimal(item["volume"]),
                    )
                    for item in data
                ]

    return from_try(lambda: asyncio.run(fetch()))


def fetch_orderbook(
    symbol: str, depth: int, api_config: APIConfig
) -> IOEither[Exception, OrderBook]:
    """Fetch current orderbook from REST API"""

    async def fetch():
        url = f"{api_config.base_url}/orderbook"
        params = {"symbol": symbol, "depth": depth}

        async with aiohttp.ClientSession(
            headers=api_config.headers,
            timeout=aiohttp.ClientTimeout(total=api_config.timeout),
        ) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")

                data = await response.json()
                return OrderBook(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    bids=[(Decimal(p), Decimal(s)) for p, s in data["bids"]],
                    asks=[(Decimal(p), Decimal(s)) for p, s in data["asks"]],
                )

    return from_try(lambda: asyncio.run(fetch()))


def fetch_recent_trades(
    symbol: str, limit: int, api_config: APIConfig
) -> IOEither[Exception, list[Trade]]:
    """Fetch recent trades from REST API"""

    async def fetch():
        url = f"{api_config.base_url}/trades"
        params = {"symbol": symbol, "limit": limit}

        async with aiohttp.ClientSession(
            headers=api_config.headers,
            timeout=aiohttp.ClientTimeout(total=api_config.timeout),
        ) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")

                data = await response.json()
                return [
                    Trade(
                        id=item["id"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        price=Decimal(item["price"]),
                        size=Decimal(item["size"]),
                        side=item["side"],
                    )
                    for item in data
                ]

    return from_try(lambda: asyncio.run(fetch()))


# Rate Limiting Effects


def rate_limit(limit: RateLimit) -> IO[None]:
    """Apply rate limiting to API calls"""

    def apply_limit():
        if limit.requests_per_second > 0:
            delay = 1.0 / limit.requests_per_second
            asyncio.sleep(delay)

    return IO(apply_limit)


def with_rate_limit(limit: RateLimit, effect: IO[A]) -> IO[A]:
    """Apply rate limiting to an effect"""
    return rate_limit(limit).chain(effect)


# Data Aggregation Effects


def aggregate_trades_to_candles(
    trades: list[Trade], interval: timedelta
) -> IO[list[Candle]]:
    """Aggregate trades into candles"""

    def aggregate():
        if not trades:
            return []

        candles = []
        current_candle = None

        for trade in sorted(trades, key=lambda t: t.timestamp):
            candle_start = trade.timestamp.replace(second=0, microsecond=0)

            if (
                current_candle is None
                or candle_start >= current_candle.timestamp + interval
            ):
                if current_candle:
                    candles.append(current_candle)

                current_candle = Candle(
                    timestamp=candle_start,
                    open=trade.price,
                    high=trade.price,
                    low=trade.price,
                    close=trade.price,
                    volume=trade.size,
                )
            else:
                current_candle.high = max(current_candle.high, trade.price)
                current_candle.low = min(current_candle.low, trade.price)
                current_candle.close = trade.price
                current_candle.volume += trade.size

        if current_candle:
            candles.append(current_candle)

        return candles

    return IO(aggregate)


# Health Check Effects


def check_connection_health(connection: WebSocketConnection) -> IO[bool]:
    """Check if WebSocket connection is healthy"""

    def check():
        if not connection.websocket:
            return False
        return not connection.websocket.closed and connection.is_connected

    return IO(check)


def check_api_health(api_config: APIConfig) -> IOEither[Exception, bool]:
    """Check if REST API is healthy"""

    async def check():
        url = f"{api_config.base_url}/health"

        async with aiohttp.ClientSession(
            headers=api_config.headers, timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(url) as response:
                return response.status == 200

    return from_try(lambda: asyncio.run(check()))


# Multi-Exchange Support


def create_multi_exchange_stream(
    exchanges: list[str], symbol: str, channels: list[str]
) -> AsyncIO[MarketDataStream]:
    """Create unified stream from multiple exchanges"""

    async def create_stream():
        connections = []

        for exchange in exchanges:
            # Create connection for each exchange
            config = ConnectionConfig(
                url=f"wss://{exchange}.com/ws",
                headers={"User-Agent": "FunctionalTradingBot/1.0"},
            )

            connection_result = connect_websocket(config).run()
            if connection_result.is_right():
                connection = connection_result.value
                subscribe_to_symbol(symbol, channels, connection)
                connections.append(connection)

        return MarketDataStream(
            symbol=symbol, exchanges=exchanges, connections=connections, active=True
        )

    return AsyncIO.pure(await create_stream())
