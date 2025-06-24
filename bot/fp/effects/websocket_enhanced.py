"""
Enhanced WebSocket Effects with Functional Programming

This module provides enhanced WebSocket connection management with functional effects,
offering improved reliability, reconnection logic, and real-time data streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import websockets
from websockets.exceptions import ConnectionClosed

from .io import IO, AsyncIO, IOEither, from_try

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConnectionState(Enum):
    """WebSocket connection states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass(frozen=True)
class EnhancedConnectionConfig:
    """Enhanced WebSocket connection configuration"""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    heartbeat_interval: int = 30
    reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    connection_timeout: float = 10.0
    message_timeout: float = 5.0
    max_message_size: int = 1024 * 1024  # 1MB
    compression: str | None = None
    enable_ping_pong: bool = True


@dataclass(frozen=True)
class ConnectionMetrics:
    """Connection performance metrics"""

    connect_time: datetime
    last_message_time: datetime | None
    messages_received: int
    messages_sent: int
    reconnection_count: int
    total_downtime: timedelta
    average_latency: float
    connection_state: ConnectionState


@dataclass(frozen=True)
class MessageEnvelope:
    """Wrapper for WebSocket messages with metadata"""

    payload: dict[str, Any]
    timestamp: datetime
    message_id: str
    message_type: str
    source: str


class EnhancedWebSocketManager:
    """
    Enhanced WebSocket manager with functional effects for improved reliability.

    Features:
    - Exponential backoff reconnection
    - Connection health monitoring
    - Message validation and queuing
    - Performance metrics
    - Circuit breaker pattern
    """

    def __init__(self, config: EnhancedConnectionConfig):
        self.config = config
        self._connection: websockets.WebSocketServerProtocol | None = None
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics(
            connect_time=datetime.utcnow(),
            last_message_time=None,
            messages_received=0,
            messages_sent=0,
            reconnection_count=0,
            total_downtime=timedelta(),
            average_latency=0.0,
            connection_state=ConnectionState.DISCONNECTED,
        )
        self._message_queue: asyncio.Queue[MessageEnvelope] = asyncio.Queue(
            maxsize=10000
        )
        self._subscriptions: dict[str, list[Callable[[MessageEnvelope], None]]] = {}
        self._connection_lock = asyncio.Lock()
        self._is_running = False
        self._background_tasks: set[asyncio.Task] = set()

    # Core Connection Management

    def connect(self) -> IOEither[Exception, bool]:
        """Establish WebSocket connection with enhanced error handling"""

        async def establish_connection():
            async with self._connection_lock:
                if self._state == ConnectionState.CONNECTED:
                    return True

                self._state = ConnectionState.CONNECTING
                logger.info(f"Connecting to WebSocket: {self.config.url}")

                try:
                    # Create connection with enhanced configuration
                    self._connection = await websockets.connect(
                        self.config.url,
                        extra_headers=self.config.headers,
                        ping_interval=(
                            self.config.heartbeat_interval
                            if self.config.enable_ping_pong
                            else None
                        ),
                        ping_timeout=self.config.connection_timeout,
                        open_timeout=self.config.connection_timeout,
                        max_size=self.config.max_message_size,
                        compression=self.config.compression,
                    )

                    self._state = ConnectionState.CONNECTED
                    self._is_running = True

                    # Update metrics
                    self._metrics = self._metrics._replace(
                        connect_time=datetime.utcnow(),
                        connection_state=ConnectionState.CONNECTED,
                    )

                    # Start background message processor
                    task = asyncio.create_task(self._message_processor())
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

                    logger.info("WebSocket connection established successfully")
                    return True

                except Exception as e:
                    self._state = ConnectionState.FAILED
                    self._metrics = self._metrics._replace(
                        connection_state=ConnectionState.FAILED
                    )
                    logger.error(f"Failed to establish WebSocket connection: {e}")
                    raise e

        return from_try(lambda: asyncio.run(establish_connection()))

    def disconnect(self) -> IOEither[Exception, None]:
        """Gracefully disconnect WebSocket"""

        async def close_connection():
            async with self._connection_lock:
                self._is_running = False

                if self._connection and not self._connection.closed:
                    await self._connection.close()
                    logger.info("WebSocket connection closed")

                self._state = ConnectionState.DISCONNECTED
                self._metrics = self._metrics._replace(
                    connection_state=ConnectionState.DISCONNECTED
                )

                # Cancel background tasks
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()

                self._background_tasks.clear()

        return from_try(lambda: asyncio.run(close_connection()))

    def reconnect_with_backoff(self) -> IOEither[Exception, bool]:
        """Reconnect with exponential backoff strategy"""

        async def reconnect():
            if self._state == ConnectionState.CONNECTED:
                return True

            self._state = ConnectionState.RECONNECTING

            for attempt in range(self.config.reconnect_attempts):
                try:
                    logger.info(
                        f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}"
                    )

                    # Close existing connection if any
                    if self._connection and not self._connection.closed:
                        await self._connection.close()

                    # Attempt to reconnect
                    result = self.connect().run()
                    if result.is_right() and result.value:
                        logger.info("Reconnection successful")
                        self._metrics = self._metrics._replace(
                            reconnection_count=self._metrics.reconnection_count + 1
                        )
                        return True

                except Exception as e:
                    logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

                # Calculate exponential backoff delay
                delay = min(
                    self.config.reconnect_delay * (2**attempt),
                    self.config.max_reconnect_delay,
                )

                logger.info(f"Waiting {delay:.2f}s before next reconnection attempt")
                await asyncio.sleep(delay)

            self._state = ConnectionState.FAILED
            logger.error("All reconnection attempts failed")
            return False

        return from_try(lambda: asyncio.run(reconnect()))

    # Message Handling

    def send_message(self, message: dict[str, Any]) -> IOEither[Exception, bool]:
        """Send message with enhanced error handling"""

        async def send():
            if not self._connection or self._connection.closed:
                raise ConnectionError("WebSocket not connected")

            try:
                message_str = json.dumps(message)
                await asyncio.wait_for(
                    self._connection.send(message_str),
                    timeout=self.config.message_timeout,
                )

                self._metrics = self._metrics._replace(
                    messages_sent=self._metrics.messages_sent + 1
                )

                return True

            except TimeoutError:
                logger.error("Message send timeout")
                raise TimeoutError("Message send timeout")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                raise e

        return from_try(lambda: asyncio.run(send()))

    def subscribe_to_messages(
        self, message_type: str, callback: Callable[[MessageEnvelope], None]
    ) -> IO[None]:
        """Subscribe to specific message types"""

        def subscribe():
            if message_type not in self._subscriptions:
                self._subscriptions[message_type] = []

            self._subscriptions[message_type].append(callback)
            logger.debug(f"Subscribed to message type: {message_type}")

        return IO(subscribe)

    def unsubscribe_from_messages(
        self, message_type: str, callback: Callable[[MessageEnvelope], None]
    ) -> IO[None]:
        """Unsubscribe from specific message types"""

        def unsubscribe():
            if message_type in self._subscriptions:
                try:
                    self._subscriptions[message_type].remove(callback)
                    if not self._subscriptions[message_type]:
                        del self._subscriptions[message_type]
                    logger.debug(f"Unsubscribed from message type: {message_type}")
                except ValueError:
                    logger.warning(
                        f"Callback not found for message type: {message_type}"
                    )

        return IO(unsubscribe)

    def stream_messages(self) -> AsyncIO[AsyncIterator[MessageEnvelope]]:
        """Stream incoming messages as async iterator"""

        async def create_stream():
            async def message_stream():
                while self._is_running:
                    try:
                        message = await asyncio.wait_for(
                            self._message_queue.get(), timeout=1.0
                        )
                        yield message
                    except TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error in message stream: {e}")
                        break

            return message_stream()

        return AsyncIO(lambda: asyncio.create_task(create_stream()))

    async def _message_processor(self) -> None:
        """Background message processor"""
        try:
            async for message in self._connection:
                try:
                    # Parse message
                    data = json.loads(message)

                    # Create message envelope
                    envelope = MessageEnvelope(
                        payload=data,
                        timestamp=datetime.utcnow(),
                        message_id=self._generate_message_id(),
                        message_type=self._extract_message_type(data),
                        source=self.config.url,
                    )

                    # Update metrics
                    self._metrics = self._metrics._replace(
                        messages_received=self._metrics.messages_received + 1,
                        last_message_time=envelope.timestamp,
                    )

                    # Queue message for processing
                    try:
                        self._message_queue.put_nowait(envelope)
                    except asyncio.QueueFull:
                        logger.warning("Message queue full, dropping message")

                    # Notify subscribers
                    self._notify_subscribers(envelope)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except ConnectionClosed:
            logger.warning("WebSocket connection closed, attempting reconnection")
            if self._is_running:
                # Attempt automatic reconnection
                asyncio.create_task(self._auto_reconnect())
        except Exception as e:
            logger.error(f"Message processor error: {e}")

    async def _auto_reconnect(self) -> None:
        """Automatic reconnection handler"""
        reconnect_result = self.reconnect_with_backoff().run()
        if reconnect_result.is_left():
            logger.error("Automatic reconnection failed")
            self._state = ConnectionState.FAILED

    def _notify_subscribers(self, message: MessageEnvelope) -> None:
        """Notify message subscribers"""
        message_type = message.message_type

        if message_type in self._subscriptions:
            for callback in self._subscriptions[message_type]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        import uuid

        return str(uuid.uuid4())

    def _extract_message_type(self, data: dict[str, Any]) -> str:
        """Extract message type from message data"""
        # Common patterns for message type extraction
        return data.get("type") or data.get("channel") or data.get("event") or "unknown"

    # Health and Monitoring

    def check_connection_health(self) -> IO[bool]:
        """Check if connection is healthy"""

        def check():
            if not self._connection or self._connection.closed:
                return False

            if self._state != ConnectionState.CONNECTED:
                return False

            # Check for recent activity
            if self._metrics.last_message_time:
                time_since_last_message = (
                    datetime.utcnow() - self._metrics.last_message_time
                )
                if time_since_last_message > timedelta(minutes=5):
                    logger.warning("No messages received in 5 minutes")
                    return False

            return True

        return IO(check)

    def get_connection_metrics(self) -> IO[ConnectionMetrics]:
        """Get current connection metrics"""
        return IO(lambda: self._metrics)

    def get_connection_state(self) -> IO[ConnectionState]:
        """Get current connection state"""
        return IO(lambda: self._state)

    # Circuit Breaker Pattern

    def with_circuit_breaker(
        self,
        operation: Callable[[], IOEither[Exception, T]],
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=1),
    ) -> IOEither[Exception, T]:
        """Execute operation with circuit breaker pattern"""

        # Simplified circuit breaker implementation
        def execute():
            if self._state == ConnectionState.FAILED:
                # Check if we should attempt recovery
                time_since_failure = datetime.utcnow() - self._metrics.connect_time
                if time_since_failure < recovery_timeout:
                    raise ConnectionError("Circuit breaker open")

            return operation().run()

        return from_try(execute)


# Factory Functions


def create_enhanced_websocket_manager(
    url: str,
    headers: dict[str, str] | None = None,
    heartbeat_interval: int = 30,
    reconnect_attempts: int = 10,
) -> EnhancedWebSocketManager:
    """Create enhanced WebSocket manager with specified configuration"""
    config = EnhancedConnectionConfig(
        url=url,
        headers=headers or {},
        heartbeat_interval=heartbeat_interval,
        reconnect_attempts=reconnect_attempts,
    )
    return EnhancedWebSocketManager(config)


def create_high_reliability_websocket_manager(url: str) -> EnhancedWebSocketManager:
    """Create WebSocket manager optimized for reliability"""
    config = EnhancedConnectionConfig(
        url=url,
        heartbeat_interval=15,  # More frequent heartbeats
        reconnect_attempts=20,  # More reconnection attempts
        reconnect_delay=0.5,  # Faster initial reconnection
        max_reconnect_delay=30.0,  # Lower max delay
        connection_timeout=5.0,  # Shorter timeout
        enable_ping_pong=True,  # Enable ping/pong
    )
    return EnhancedWebSocketManager(config)


def create_low_latency_websocket_manager(url: str) -> EnhancedWebSocketManager:
    """Create WebSocket manager optimized for low latency"""
    config = EnhancedConnectionConfig(
        url=url,
        heartbeat_interval=60,  # Less frequent heartbeats
        reconnect_attempts=5,  # Fewer reconnection attempts
        reconnect_delay=0.1,  # Very fast reconnection
        connection_timeout=2.0,  # Short timeout
        message_timeout=1.0,  # Short message timeout
        compression=None,  # No compression for speed
    )
    return EnhancedWebSocketManager(config)
