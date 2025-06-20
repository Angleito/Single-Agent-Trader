"""WebSocket publisher for sending real-time trading data to dashboard."""

import asyncio
import contextlib
import json
import logging
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException

from .config import Settings
from .trading_types import MarketState, Position, TradeAction

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class TradingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for trading data types."""

    def default(self, obj):
        """Convert non-serializable objects to JSON-compatible formats."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime | pd.Timestamp) or hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):  # pydantic models and other objects
            return obj.__dict__
        elif hasattr(obj, "model_dump"):  # pydantic v2 models
            return obj.model_dump()
        # Handle numpy integer types
        elif hasattr(obj, "dtype") and "int" in str(obj.dtype):
            return int(obj)
        # Handle numpy float types
        elif hasattr(obj, "dtype") and "float" in str(obj.dtype):
            return float(obj)
        # Handle pandas Series and numpy arrays
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class WebSocketPublisher:
    """Publishes real-time trading data to dashboard via WebSocket."""

    def __init__(self, settings: Settings):
        """Initialize WebSocket publisher with settings."""
        self.settings = settings
        self.dashboard_url = getattr(
            settings.system, "websocket_dashboard_url", "ws://dashboard-backend:8000/ws"
        )
        # Parse fallback URLs
        fallback_urls_str = getattr(settings.system, "websocket_fallback_urls", "")
        self.fallback_urls = [
            url.strip() for url in fallback_urls_str.split(",") if url.strip()
        ]

        self.publish_interval = getattr(
            settings.system, "websocket_publish_interval", 1.0
        )
        self.max_retries = getattr(settings.system, "websocket_max_retries", 15)
        self.retry_delay = getattr(settings.system, "websocket_retry_delay", 5)
        self.connection_timeout = getattr(settings.system, "websocket_timeout", 45)
        self.initial_connect_timeout = getattr(
            settings.system, "websocket_initial_connect_timeout", 60
        )
        self.connection_delay = getattr(
            settings.system, "websocket_connection_delay", 0
        )
        self.ping_interval = getattr(settings.system, "websocket_ping_interval", 20)
        self.ping_timeout = getattr(settings.system, "websocket_ping_timeout", 10)
        self.queue_size = getattr(settings.system, "websocket_queue_size", 500)
        self.health_check_interval = getattr(
            settings.system, "websocket_health_check_interval", 45
        )

        self._ws: WebSocketClientProtocol | None = None
        self._connected = False
        self._retry_count = 0
        self._publish_enabled = getattr(
            settings.system, "enable_websocket_publishing", True
        )
        self._last_pong_time: float | None = None
        self._ping_task: asyncio.Task | None = None
        self._connection_monitor_task: asyncio.Task | None = None
        self._auto_reconnect_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._priority_queue: asyncio.Queue = asyncio.Queue(
            maxsize=min(self.queue_size // 4, 500)
        )  # Priority queue for critical messages
        self._queue_worker_task: asyncio.Task | None = None
        self._connection_start_time: float | None = None
        self._consecutive_failures = 0
        self._current_url = self.dashboard_url
        self._url_index = 0  # Track which URL we're currently trying

        # Queue monitoring
        self._queue_stats = {
            "messages_sent": 0,
            "messages_dropped": 0,
            "queue_full_events": 0,
            "max_queue_size_seen": 0,
            "last_queue_warning": 0.0,
        }
        self._priority_message_types = {
            "performance_update",
            "trade_execution",
            "position_update",
            "error",
        }

        logger.info("WebSocketPublisher initialized - Primary URL: %s", self.dashboard_url)
        if self.fallback_urls:
            logger.info("Fallback URLs configured: %s", ', '.join(self.fallback_urls))
        if self.connection_delay > 0:
            logger.info("Connection delay configured: %ss", self.connection_delay)

    async def initialize(self) -> bool:
        """Initialize WebSocket connection to dashboard with diagnostics and delay."""
        if not self._publish_enabled:
            logger.info("WebSocket publishing disabled in settings")
            return False

        # Apply connection delay if configured
        if self.connection_delay > 0:
            logger.info("Applying connection delay of %ss for service startup", self.connection_delay)
            await asyncio.sleep(self.connection_delay)

        # Run network diagnostics before connecting
        await self._run_network_diagnostics()

        try:
            await self._connect_with_fallback()
            if self._connected:
                logger.info("WebSocket publisher initialized successfully")
                # Start automatic reconnection monitoring
                self._auto_reconnect_task = asyncio.create_task(
                    self._auto_reconnect_manager()
                )
                return True
        except Exception as e:
            logger.exception("Failed to initialize WebSocket publisher: %s", e)
            # Start auto-reconnect even if initial connection fails
            self._auto_reconnect_task = asyncio.create_task(
                self._auto_reconnect_manager()
            )
        return False

    async def _auto_reconnect_manager(self) -> None:
        """Background task to manage automatic reconnections."""
        while self._publish_enabled:
            try:
                if not self._connected:
                    logger.info("Connection lost, attempting to reconnect...")
                    try:
                        await self._reconnect()
                    except Exception as e:
                        logger.exception("Auto-reconnect failed: %s", e)
                        # Reset URL index to try primary URL again after failures
                        self._url_index = 0
                        self._current_url = self.dashboard_url

                # Check connection status every 10 seconds
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in auto-reconnect manager: %s", e)
                await asyncio.sleep(30)  # Wait longer on unexpected errors

    async def _run_network_diagnostics(self) -> None:
        """Run network diagnostics to check connectivity to dashboard service."""
        try:
            logger.info("Running network diagnostics for dashboard connectivity...")

            # Test DNS resolution and basic connectivity
            urls_to_test = [self.dashboard_url, *self.fallback_urls]

            for url in urls_to_test[:3]:  # Test first 3 URLs
                try:
                    # Convert WebSocket URL to HTTP for testing
                    http_url = (
                        url.replace("ws://", "http://")
                        .replace("wss://", "https://")
                        .replace("/ws", "/health")
                    )

                    # Try to connect to health endpoint
                    import aiohttp

                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as session:
                        try:
                            async with session.get(http_url) as response:
                                if response.status in [
                                    200,
                                    404,
                                ]:  # 404 is OK - service is up
                                    logger.info("✓ Network connectivity to %s confirmed", http_url)
                                else:
                                    logger.warning("⚠ Network connectivity to %s returned status %s", http_url, response.status)
                        except Exception as e:
                            logger.warning("✗ Network connectivity to %s failed: %s", http_url, e)

                except Exception as e:
                    logger.debug("Network diagnostic failed for %s: %s", url, e)

        except Exception as e:
            logger.warning("Network diagnostics failed: %s", e)

    async def _connect_with_fallback(self) -> None:
        """Try to connect using primary URL and fallback URLs if needed."""
        urls_to_try = [self.dashboard_url, *self.fallback_urls]

        for i, url in enumerate(urls_to_try):
            try:
                logger.info("Attempting WebSocket connection to %s (attempt %s/%s)", url, i+1, len(urls_to_try))
                self._current_url = url
                self._url_index = i

                await self._connect_to_url(url)

                if self._connected:
                    logger.info("✓ Successfully connected to %s", url)
                    return

            except Exception as e:
                logger.warning("✗ Connection to %s failed: %s", url, e)

                if i < len(urls_to_try) - 1:
                    logger.info("Trying next fallback URL...")
                    await asyncio.sleep(2)  # Brief delay between attempts
                else:
                    logger.exception(
                        "All WebSocket URLs failed, no more fallbacks available"
                    )
                    raise Exception(
                        "All WebSocket connection attempts failed"
                    ) from None

    async def _connect_to_url(self, url: str) -> None:
        """Connect to a specific WebSocket URL."""
        try:
            logger.info("Connecting to dashboard WebSocket at %s", url)
            self._connection_start_time = time.time()

            # Use initial connection timeout for first attempt, regular timeout for retries
            timeout = (
                self.initial_connect_timeout
                if self._retry_count == 0
                else self.connection_timeout
            )

            # Enhanced connection parameters for better stability
            self._ws = await websockets.connect(
                url,
                timeout=timeout,
                ping_interval=self.ping_interval,  # Configurable ping interval
                ping_timeout=self.ping_timeout,  # Configurable ping timeout
                close_timeout=5,  # Quick close timeout
                max_size=2**20,  # 1MB max message size
                compression=None,  # Disable compression to reduce CPU overhead
                # Additional headers for better compatibility
                extra_headers={
                    "User-Agent": "AI-Trading-Bot-WebSocket-Publisher/1.0",
                    "Accept": "*/*",
                    "Connection": "Upgrade",
                },
            )

            self._connected = True
            self._retry_count = 0
            self._consecutive_failures = 0
            self._last_pong_time = time.time()

            # Start monitoring tasks
            await self._start_monitoring_tasks()

            connection_time = time.time() - (
                self._connection_start_time
                if self._connection_start_time is not None
                else time.time()
            )
            logger.info("Successfully connected to dashboard WebSocket in %.2fs", connection_time)

        except (TimeoutError, OSError, InvalidURI) as e:
            self._connected = False
            self._consecutive_failures += 1
            logger.exception("Failed to connect to dashboard WebSocket: %s", e)
            raise
        except Exception as e:
            self._connected = False
            self._consecutive_failures += 1
            logger.exception("Unexpected error connecting to dashboard WebSocket: %s", e)
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect with enhanced exponential backoff."""
        if self._retry_count >= self.max_retries:
            logger.error("Max retries (%s) reached, giving up", self.max_retries)
            return

        # Clean up existing connection
        await self._cleanup_connection()

        self._retry_count += 1

        # Enhanced exponential backoff with jitter to prevent thundering herd
        base_delay = self.retry_delay * (2 ** (self._retry_count - 1))
        jitter = base_delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
        delay = min(base_delay + jitter, 60)  # Cap at 60 seconds

        # Additional delay for consecutive failures
        if self._consecutive_failures > 3:
            delay = min(delay * 1.5, 120)  # Extended delay for persistent issues

        logger.warning("Reconnecting in %ss (attempt %s/%s, failures: %s)", delay:.1f, self._retry_count, self.max_retries, self._consecutive_failures)

        await asyncio.sleep(delay)

        try:
            await self._connect_with_fallback()
        except Exception as e:
            logger.exception("Reconnection attempt %s failed: %s", self._retry_count, e)
            # If we've failed multiple times, check if URL is reachable
            if self._retry_count > 2:
                await self._check_endpoint_health()

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send message to dashboard with queue-based delivery and prioritization."""
        if not self._publish_enabled:
            return

        try:
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(UTC).isoformat()

            message_type = message.get("type", "unknown")

            # Update queue statistics
            current_queue_size = self._message_queue.qsize()
            self._queue_stats["max_queue_size_seen"] = max(
                self._queue_stats["max_queue_size_seen"], current_queue_size
            )

            # Check if this is a priority message
            is_priority = message_type in self._priority_message_types

            try:
                if is_priority:
                    # Try priority queue first
                    try:
                        await asyncio.wait_for(
                            self._priority_queue.put(message), timeout=0.5
                        )
                        logger.debug("Queued priority message: %s", message_type)
                        return
                    except (TimeoutError, asyncio.QueueFull):
                        # Priority queue full, fall back to regular queue
                        logger.debug("Priority queue full, using regular queue for: %s", message_type)

                # Queue message for delivery in regular queue
                await asyncio.wait_for(self._message_queue.put(message), timeout=1.0)
                logger.debug("Queued message: %s", message_type)

            except (TimeoutError, asyncio.QueueFull):
                # Queue is full, implement intelligent dropping
                self._queue_stats["messages_dropped"] += 1
                self._queue_stats["queue_full_events"] += 1

                # Log warning with throttling (max once per 30 seconds)
                current_time = time.time()
                if current_time - self._queue_stats["last_queue_warning"] > 30:
                    self._queue_stats["last_queue_warning"] = current_time
                    logger.warning("Message queue full (size: %s/%s), ", current_queue_size, self.queue_size)
                        f"dropping {message_type} message. "
                        f"Stats: sent={self._queue_stats['messages_sent']}, "
                        f"dropped={self._queue_stats['messages_dropped']}, "
                        f"queue_full_events={self._queue_stats['queue_full_events']}"
                    )

                # For critical messages, try to make room by dropping older non-priority messages
                if is_priority and current_queue_size > self.queue_size * 0.9:
                    self._make_room_for_priority_message(message)

        except Exception as e:
            logger.exception("Error queuing WebSocket message: %s", e)

    def _make_room_for_priority_message(self, priority_message: dict[str, Any]) -> None:
        """Try to make room for a priority message by dropping non-priority messages."""
        try:
            # Remove up to 3 non-priority messages to make room
            messages_removed = 0
            temp_messages = []

            while not self._message_queue.empty() and messages_removed < 3:
                try:
                    msg = self._message_queue.get_nowait()
                    msg_type = msg.get("type", "unknown")

                    # Keep priority messages, drop others
                    if msg_type in self._priority_message_types:
                        temp_messages.append(msg)
                    else:
                        messages_removed += 1
                        self._queue_stats["messages_dropped"] += 1
                        logger.debug("Dropped %s message to make room for priority message", msg_type)

                except asyncio.QueueEmpty:
                    break

            # Put back the priority messages we kept
            for msg in temp_messages:
                try:
                    self._message_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.debug(
                        "Could not restore all priority messages after cleanup"
                    )
                    break

            # Now try to queue the new priority message
            if messages_removed > 0:
                try:
                    self._message_queue.put_nowait(priority_message)
                    logger.debug(
                        "Successfully queued priority message after making room"
                    )
                except asyncio.QueueFull:
                    logger.warning(
                        "Still couldn't queue priority message after cleanup"
                    )

        except Exception as e:
            logger.exception("Error making room for priority message: %s", e)

    async def _queue_worker(self) -> None:
        """Background worker to process message queue with priority handling."""
        while self._publish_enabled:
            try:
                message = None

                # Check priority queue first
                try:
                    message = self._priority_queue.get_nowait()
                    logger.debug("Processing priority message")
                except asyncio.QueueEmpty:
                    # No priority messages, check regular queue
                    try:
                        message = await asyncio.wait_for(
                            self._message_queue.get(), timeout=5.0
                        )
                    except TimeoutError:
                        # No messages to process, continue
                        continue

                if not self._connected or not self._ws:
                    # Drop message if not connected, but log it for priority messages
                    if message and message.get("type") in self._priority_message_types:
                        logger.debug("Dropping priority message %s due to disconnection", message.get('type'))
                    continue

                if message:
                    try:
                        # Use custom encoder for JSON serialization
                        json_data = json.dumps(message, cls=TradingJSONEncoder)
                        await self._ws.send(json_data)

                        # Update statistics
                        self._queue_stats["messages_sent"] += 1

                        message_type = message.get("type", "unknown")
                        logger.debug("Sent WebSocket message: %s", message_type)

                        # Log queue health every 100 messages
                        if self._queue_stats["messages_sent"] % 100 == 0:
                            self._log_queue_health()

                    except (ConnectionClosed, WebSocketException) as e:
                        logger.warning("WebSocket connection lost during send: %s", e)
                        self._connected = False

                        # Re-queue priority messages, drop others to prevent queue buildup
                        if message.get("type") in self._priority_message_types:
                            try:
                                self._priority_queue.put_nowait(message)
                                logger.debug(
                                    "Re-queued priority message after connection loss"
                                )
                            except asyncio.QueueFull:
                                logger.warning(
                                    "Priority queue full, dropping priority message after connection loss"
                                )
                        else:
                            logger.debug("Dropping non-priority message %s after connection loss", message.get('type'))

                    except Exception as e:
                        logger.exception("Error sending WebSocket message: %s", e)

            except Exception as e:
                logger.exception("Error in queue worker: %s", e)
                await asyncio.sleep(1)

    def _log_queue_health(self) -> None:
        """Log queue health statistics."""
        regular_queue_size = self._message_queue.qsize()
        priority_queue_size = self._priority_queue.qsize()

        logger.info(
            "📊 WebSocket Queue Health: "
            f"Regular={regular_queue_size}/{self.queue_size}, "
            f"Priority={priority_queue_size}/{self._priority_queue.maxsize}, "
            f"Sent={self._queue_stats['messages_sent']}, "
            f"Dropped={self._queue_stats['messages_dropped']}, "
            f"MaxSeen={self._queue_stats['max_queue_size_seen']}"
        )

    async def publish_system_status(
        self,
        status: str,
        details: dict[str, Any] | None = None,
        health: bool | None = None,
        status_message_text: str | None = None,
        **kwargs,
    ) -> None:
        """Publish system status update."""
        status_message: dict[str, Any] = {
            "type": "system_status",
            "status": status,
            "details": details or {},
            "bot_id": "ai-trading-bot",
        }

        # Support legacy parameters
        if health is not None:
            status_message["health"] = health
        if status_message_text is not None:
            status_message["message"] = status_message_text

        # Add any additional kwargs
        status_message.update(kwargs)

        await self._send_message(status_message)

    async def publish_market_data(
        self,
        symbol: str,
        price: float,
        volume: float = 0,
        market_state: MarketState = None,
        timestamp: Any = None,
    ) -> None:
        """Publish market data update."""
        message = {
            "type": "market_data",
            "symbol": symbol,
            "price": float(price),
            "volume": float(volume),
            "market_state": market_state,  # Let the encoder handle serialization
            "data_timestamp": timestamp,  # Let the encoder handle timestamp conversion
        }
        await self._send_message(message)

    async def publish_indicator_data(self, symbol: str, indicators: Any) -> None:
        """Publish technical indicator data."""
        message = {"type": "indicator_data", "symbol": symbol, "indicators": indicators}
        await self._send_message(message)

    async def publish_ai_decision(
        self,
        action: str | None = None,
        decision: str | None = None,
        confidence: float = 0,
        reasoning: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Publish AI trading decision."""
        # Support both 'action' and 'decision' parameters for backward compatibility
        final_decision = action or decision or "HOLD"
        message = {
            "type": "ai_decision",
            "decision": final_decision,
            "action": final_decision,  # Include both for compatibility
            "confidence": float(confidence),
            "reasoning": reasoning,
            "context": context or {},
        }
        await self._send_message(message)

    async def publish_trading_decision(
        self,
        trade_action: TradeAction,
        symbol: str,
        current_price: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Publish trading decision."""
        message = {
            "type": "trading_decision",
            "action": trade_action.action,
            "symbol": symbol,
            "size_pct": float(trade_action.size_pct),
            "price": float(current_price),
            "reasoning": trade_action.rationale or "",
            "take_profit_pct": (
                float(trade_action.take_profit_pct)
                if trade_action.take_profit_pct
                else None
            ),
            "stop_loss_pct": (
                float(trade_action.stop_loss_pct)
                if trade_action.stop_loss_pct
                else None
            ),
            "leverage": trade_action.leverage,
            "context": context or {},
        }
        await self._send_message(message)

    async def publish_trade_execution(self, trade_result: dict[str, Any]) -> None:
        """Publish trade execution result."""
        message = {"type": "trade_execution", "result": trade_result}
        await self._send_message(message)

    async def publish_position_update(
        self, position: Position | None = None, positions: list[Position] | None = None
    ) -> None:
        """Publish position update."""
        message = {
            "type": "position_update",
            "position": (
                position.model_dump()
                if position and hasattr(position, "model_dump")
                else (position.__dict__ if position else None)
            ),
            "positions": [
                pos.model_dump() if hasattr(pos, "model_dump") else pos.__dict__
                for pos in (positions or [])
            ],
        }
        await self._send_message(message)

    async def publish_performance_update(self, metrics: dict[str, Any]) -> None:
        """Publish performance metrics."""
        message = {"type": "performance_update", "metrics": metrics}
        await self._send_message(message)

    async def publish_error(
        self, error_type: str, error_message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Publish error notification."""
        message = {
            "type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "details": details or {},
        }
        await self._send_message(message)

    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Start queue worker
        if self._queue_worker_task is None or self._queue_worker_task.done():
            self._queue_worker_task = asyncio.create_task(self._queue_worker())

        # Start connection monitor
        if (
            self._connection_monitor_task is None
            or self._connection_monitor_task.done()
        ):
            self._connection_monitor_task = asyncio.create_task(
                self._connection_monitor()
            )

    async def _connection_monitor(self) -> None:
        """Monitor connection health and trigger reconnections if needed."""
        while self._connected and self._publish_enabled:
            try:
                await asyncio.sleep(self.health_check_interval)

                if not self._connected or not self._ws:
                    break

                # Check if we've received a pong recently
                if self._last_pong_time is not None:
                    time_since_pong = time.time() - self._last_pong_time
                    if time_since_pong > 60:  # 60 seconds without pong
                        logger.warning("No pong received for %ss, connection may be stale", time_since_pong:.1f)
                        self._connected = False
                        break

                # Send a custom ping to test connection
                try:
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    self._last_pong_time = time.time()
                    logger.debug("Connection health check passed")
                except TimeoutError:
                    logger.warning(
                        "Health check ping timeout, marking connection as unhealthy"
                    )
                    self._connected = False
                    break
                except Exception as e:
                    logger.warning("Health check failed: %s", e)
                    self._connected = False
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in connection monitor: %s", e)
                await asyncio.sleep(5)

    async def _cleanup_connection(self) -> None:
        """Clean up existing connection and tasks."""
        self._connected = False

        # Cancel monitoring tasks
        if self._connection_monitor_task and not self._connection_monitor_task.done():
            self._connection_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connection_monitor_task

        # Close WebSocket connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._ws = None

    async def _check_endpoint_health(self) -> None:
        """Check if the WebSocket endpoints are reachable."""
        try:
            # Try a simple HTTP request to the URLs
            import aiohttp

            urls_to_check = [self.dashboard_url, *self.fallback_urls]

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                for url in urls_to_check:
                    try:
                        base_url = (
                            url.replace("ws://", "http://")
                            .replace("wss://", "https://")
                            .replace("/ws", "/health")
                        )

                        async with session.get(base_url) as response:
                            if response.status in [
                                200,
                                404,
                            ]:  # 404 is OK - service is up
                                logger.info("Dashboard endpoint %s is reachable", base_url)
                            else:
                                logger.warning("Dashboard endpoint %s returned status %s", base_url, response.status)
                    except Exception as e:
                        logger.warning("Dashboard endpoint %s health check failed: %s", url, e)

        except Exception as e:
            logger.warning("Dashboard endpoint health check failed: %s", e)

    async def disconnect(self) -> None:
        """Disconnect WebSocket and cleanup resources - alias for close()."""
        await self.close()

    async def close(self) -> None:
        """Close WebSocket connection and cleanup resources."""
        logger.info("Closing WebSocket publisher")
        self._publish_enabled = False

        # Cancel ping task
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ping_task

        # Cancel queue worker
        if self._queue_worker_task and not self._queue_worker_task.done():
            self._queue_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._queue_worker_task

        # Cancel auto-reconnect task
        if self._auto_reconnect_task and not self._auto_reconnect_task.done():
            self._auto_reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._auto_reconnect_task

        # Cleanup connection
        await self._cleanup_connection()

        # Clear message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("WebSocket publisher closed")

    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None and not self._ws.closed
