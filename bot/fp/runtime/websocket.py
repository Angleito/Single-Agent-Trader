"""
WebSocket Connection Management for Functional Trading Bot

This module provides functional WebSocket connection management
with automatic reconnection and subscription handling.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..effects.error import RetryPolicy, retry
from ..effects.io import IO, AsyncIO
from ..effects.logging import error, info, warn
from ..effects.market_data import (
    ConnectionConfig,
    connect_websocket,
    subscribe_to_symbol,
)
from ..effects.monitoring import increment_counter


@dataclass
class WebSocketManager:
    """WebSocket connection manager"""

    connections: dict[str, Any]
    subscriptions: dict[str, list[str]]

    def __init__(self):
        self.connections = {}
        self.subscriptions = {}
        self.running = False

    def add_connection(self, name: str, config: ConnectionConfig) -> IO[None]:
        """Add a WebSocket connection"""

        def add():
            info(
                f"Adding WebSocket connection: {name}",
                {"url": config.url, "heartbeat_interval": config.heartbeat_interval},
            ).run()

            self.connections[name] = {
                "config": config,
                "connection": None,
                "status": "disconnected",
                "last_ping": None,
                "retry_count": 0,
            }

            increment_counter("websocket.connections.added", {"connection": name}).run()

        return IO(add)

    def connect_all(self) -> IO[None]:
        """Connect all configured WebSocket connections"""

        def connect():
            for name, conn_info in self.connections.items():
                try:
                    self.connect_single(name).run()
                except Exception as e:
                    error(f"Failed to connect {name}: {e!s}").run()

        return IO(connect)

    def connect_single(self, name: str) -> IO[None]:
        """Connect a single WebSocket"""

        def connect():
            if name not in self.connections:
                raise ValueError(f"Connection {name} not configured")

            conn_info = self.connections[name]
            config = conn_info["config"]

            info(f"Connecting WebSocket: {name}").run()

            # Use retry policy for connection
            retry_policy = RetryPolicy(max_attempts=3, delay=1.0)

            def attempt_connection():
                result = connect_websocket(config).run()
                if result.is_right():
                    conn_info["connection"] = result.value
                    conn_info["status"] = "connected"
                    conn_info["retry_count"] = 0

                    increment_counter(
                        "websocket.connections.established", {"connection": name}
                    ).run()

                    return result.value
                raise Exception(f"Connection failed: {result.value}")

            connection_effect = IO(attempt_connection)
            retry(retry_policy, connection_effect).run()

        return IO(connect)

    def subscribe(
        self, connection_name: str, symbol: str, channels: list[str]
    ) -> IO[None]:
        """Subscribe to channels on a connection"""

        def sub():
            if connection_name not in self.connections:
                raise ValueError(f"Connection {connection_name} not found")

            conn_info = self.connections[connection_name]
            if conn_info["status"] != "connected":
                raise ValueError(f"Connection {connection_name} not connected")

            connection = conn_info["connection"]

            info(
                f"Subscribing to {symbol} on {connection_name}", {"channels": channels}
            ).run()

            subscription_result = subscribe_to_symbol(
                symbol, channels, connection
            ).run()

            if subscription_result.is_right():
                # Track subscription
                if connection_name not in self.subscriptions:
                    self.subscriptions[connection_name] = []

                self.subscriptions[connection_name].extend(channels)

                increment_counter(
                    "websocket.subscriptions.added",
                    {
                        "connection": connection_name,
                        "symbol": symbol,
                        "channels": len(channels),
                    },
                ).run()
            else:
                raise Exception(f"Subscription failed: {subscription_result.value}")

        return IO(sub)

    def start_heartbeat(self, connection_name: str) -> AsyncIO[None]:
        """Start heartbeat for a connection"""

        async def heartbeat():
            if connection_name not in self.connections:
                return

            conn_info = self.connections[connection_name]
            config = conn_info["config"]

            while self.running and conn_info["status"] == "connected":
                try:
                    # Send ping
                    connection = conn_info["connection"]
                    if connection and connection.websocket:
                        await connection.websocket.ping()
                        conn_info["last_ping"] = datetime.utcnow()

                        increment_counter(
                            "websocket.heartbeat.sent", {"connection": connection_name}
                        ).run()

                    await asyncio.sleep(config.heartbeat_interval)

                except Exception as e:
                    warn(f"Heartbeat failed for {connection_name}: {e!s}").run()

                    # Mark as disconnected
                    conn_info["status"] = "disconnected"

                    # Attempt reconnection
                    await self.reconnect_single(connection_name)

        return AsyncIO.from_async(lambda: heartbeat())

    async def reconnect_single(self, connection_name: str) -> None:
        """Reconnect a single WebSocket"""
        if connection_name not in self.connections:
            return

        conn_info = self.connections[connection_name]

        warn(f"Attempting to reconnect {connection_name}").run()

        try:
            # Close existing connection
            if conn_info["connection"]:
                connection = conn_info["connection"]
                if connection.websocket and not connection.websocket.closed:
                    await connection.websocket.close()

            # Wait before reconnecting
            await asyncio.sleep(1.0 * (conn_info["retry_count"] + 1))

            # Attempt reconnection
            self.connect_single(connection_name).run()

            # Resubscribe to channels
            if connection_name in self.subscriptions:
                for channel in self.subscriptions[connection_name]:
                    # Re-subscribe logic would go here
                    pass

            info(f"Successfully reconnected {connection_name}").run()

        except Exception as e:
            conn_info["retry_count"] += 1
            error(f"Reconnection failed for {connection_name}: {e!s}").run()

            increment_counter(
                "websocket.reconnect.failed",
                {
                    "connection": connection_name,
                    "retry_count": conn_info["retry_count"],
                },
            ).run()

    async def message_handler_loop(self, connection_name: str) -> None:
        """Handle incoming messages for a connection"""
        if connection_name not in self.connections:
            return

        conn_info = self.connections[connection_name]

        try:
            while self.running and conn_info["status"] == "connected":
                connection = conn_info["connection"]
                if not connection or not connection.websocket:
                    break

                try:
                    # Receive message
                    message = await connection.websocket.recv()

                    # Parse and process message
                    data = json.loads(message)

                    # Process message (implement message routing)
                    await self.process_message(connection_name, data)

                    increment_counter(
                        "websocket.messages.received", {"connection": connection_name}
                    ).run()

                except Exception as e:
                    warn(f"Message handling error for {connection_name}: {e!s}").run()
                    break

        except Exception as e:
            error(f"Message handler loop failed for {connection_name}: {e!s}").run()

    async def process_message(self, connection_name: str, data: dict[str, Any]) -> None:
        """Process an incoming WebSocket message"""
        # Implement message processing logic
        info(
            f"Received message on {connection_name}",
            {"type": data.get("type", "unknown"), "size": len(str(data))},
        ).run()

    def start(self) -> AsyncIO[None]:
        """Start the WebSocket manager"""

        async def start():
            self.running = True
            info("WebSocket manager started").run()

            # Connect all configured connections
            self.connect_all().run()

            # Start heartbeats and message handlers
            tasks = []

            for name in self.connections:
                # Start heartbeat
                heartbeat_task = asyncio.create_task(self.start_heartbeat(name).run())
                tasks.append(heartbeat_task)

                # Start message handler
                handler_task = asyncio.create_task(self.message_handler_loop(name))
                tasks.append(handler_task)

            # Wait for all tasks
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                error(f"WebSocket manager error: {e!s}").run()
            finally:
                self.running = False
                info("WebSocket manager stopped").run()

        return AsyncIO.from_async(start())

    def stop(self) -> IO[None]:
        """Stop the WebSocket manager"""

        def stop():
            self.running = False
            info("WebSocket manager stop requested").run()

            # Close all connections
            for name, conn_info in self.connections.items():
                if conn_info["connection"]:
                    connection = conn_info["connection"]
                    if connection.websocket and not connection.websocket.closed:
                        asyncio.create_task(connection.websocket.close())

        return IO(stop)

    def get_status(self) -> dict[str, Any]:
        """Get WebSocket manager status"""
        return {
            "running": self.running,
            "connections": {
                name: {
                    "status": info["status"],
                    "last_ping": (
                        info["last_ping"].isoformat() if info["last_ping"] else None
                    ),
                    "retry_count": info["retry_count"],
                }
                for name, info in self.connections.items()
            },
            "subscriptions": self.subscriptions,
        }


# Global WebSocket manager
_ws_manager: WebSocketManager | None = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
