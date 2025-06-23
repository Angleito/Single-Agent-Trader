"""WebSocket publisher for sending real-time trading data to dashboard."""

import asyncio
import contextlib
import json
import socket
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException

from .config import Settings
from .trading_types import MarketState, Position, TradeAction
from .utils.logger_factory import get_logger
from .utils.typed_config import get_typed

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol

logger = get_logger(__name__)


class TradingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for trading data types."""

    def default(self, obj):
        """Convert non-serializable objects to JSON-compatible formats."""
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime | pd.Timestamp) or hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):  # pydantic models and other objects
            return obj.__dict__
        if hasattr(obj, "model_dump"):  # pydantic v2 models
            return obj.model_dump()
        # Handle numpy integer types
        if hasattr(obj, "dtype") and "int" in str(obj.dtype):
            return int(obj)
        # Handle numpy float types
        if hasattr(obj, "dtype") and "float" in str(obj.dtype):
            return float(obj)
        # Handle pandas Series and numpy arrays
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class WebSocketPublisher:
    """Publishes real-time trading data to dashboard via WebSocket."""

    def __init__(self, settings: Settings):
        """Initialize WebSocket publisher with settings."""
        self.settings = settings

        # Get configured URLs
        configured_url = getattr(
            settings.system, "websocket_dashboard_url", "ws://localhost:8000/ws"
        )
        fallback_urls_str = getattr(settings.system, "websocket_fallback_urls", "")

        # Build intelligent fallback chain based on environment
        self.dashboard_url, self.fallback_urls = self._build_url_chain(
            configured_url, fallback_urls_str
        )

        # Configuration with proper type conversion and error handling
        self.publish_interval = get_typed(
            settings.system, "websocket_publish_interval", 1.0
        )
        self.max_retries = get_typed(settings.system, "websocket_max_retries", 15)
        self.retry_delay = get_typed(settings.system, "websocket_retry_delay", 5.0)

        self.connection_timeout = get_typed(settings.system, "websocket_timeout", 45.0)
        self.initial_connect_timeout = get_typed(
            settings.system, "websocket_initial_connect_timeout", 60.0
        )
        self.connection_delay = get_typed(
            settings.system, "websocket_connection_delay", 0.0
        )
        self.ping_interval = get_typed(settings.system, "websocket_ping_interval", 20.0)
        self.ping_timeout = get_typed(settings.system, "websocket_ping_timeout", 10.0)
        self.queue_size = get_typed(settings.system, "websocket_queue_size", 500)
        self.health_check_interval = get_typed(
            settings.system, "websocket_health_check_interval", 45.0
        )

        self._ws: WebSocketClientProtocol | None = None
        self._connected = False
        self._retry_count = 0
        # Boolean configuration with proper type conversion
        self._publish_enabled = get_typed(
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

        # Null publisher mode flag
        self._null_publisher_mode = False
        self._environment_detected = self._detect_environment()

        logger.info(
            "WebSocketPublisher initialized - Environment: %s, Primary URL: %s",
            self._environment_detected,
            self.dashboard_url,
        )
        if self.fallback_urls:
            logger.info("Fallback URLs configured: %s", ", ".join(self.fallback_urls))
        if self.connection_delay > 0:
            logger.info("Connection delay configured: %ss", self.connection_delay)

    def _detect_environment(self) -> str:
        """Detect if running in Docker or local environment."""
        # Check for Docker environment indicators
        import os

        # Method 1: Check for /.dockerenv file
        if os.path.exists("/.dockerenv"):
            return "docker"

        # Method 2: Check for Docker-specific environment variables
        if os.environ.get("DOCKER_CONTAINER"):
            return "docker"

        # Method 3: Check /proc/1/cgroup for docker references
        try:
            with open("/proc/1/cgroup") as f:
                if "docker" in f.read():
                    return "docker"
        except Exception:
            pass

        # Method 4: Try to resolve Docker service names
        try:
            socket.gethostbyname("dashboard-backend")
            return "docker"
        except OSError:
            pass

        return "local"

    def _build_url_chain(
        self, configured_url: str, fallback_urls_str: str
    ) -> tuple[str, list[str]]:
        """Build intelligent URL chain based on environment."""
        # Parse configured fallback URLs
        configured_fallbacks = [
            url.strip() for url in fallback_urls_str.split(",") if url.strip()
        ]

        # Common URL patterns
        docker_urls = [
            "ws://dashboard-backend:8000/ws",
            "ws://dashboard:8000/ws",
        ]

        local_urls = [
            "ws://localhost:8000/ws",
            "ws://127.0.0.1:8000/ws",
            "ws://host.docker.internal:8000/ws",  # Works from inside Docker to host
        ]

        # Build URL chain based on environment
        if self._detect_environment() == "docker":
            # In Docker: prioritize Docker service names
            primary = (
                configured_url if "dashboard" in configured_url else docker_urls[0]
            )
            fallbacks = []

            # Add Docker URLs first
            for url in docker_urls:
                if url != primary and url not in fallbacks:
                    fallbacks.append(url)

            # Add configured fallbacks
            fallbacks.extend(configured_fallbacks)

            # Add local URLs as last resort
            for url in local_urls:
                if url not in fallbacks:
                    fallbacks.append(url)
        else:
            # Local environment: prioritize localhost
            primary = (
                configured_url
                if "localhost" in configured_url or "127.0.0.1" in configured_url
                else local_urls[0]
            )
            fallbacks = []

            # Add local URLs first
            for url in local_urls:
                if url != primary and url not in fallbacks:
                    fallbacks.append(url)

            # Add configured fallbacks
            fallbacks.extend(configured_fallbacks)

            # Docker URLs less likely to work but try anyway
            for url in docker_urls:
                if url not in fallbacks:
                    fallbacks.append(url)

        return primary, fallbacks

    def _enable_null_publisher_mode(self) -> None:
        """Enable null publisher mode for graceful degradation."""
        logger.info(
            "Enabling null publisher mode - bot will continue without dashboard connection"
        )
        self._null_publisher_mode = True
        self._connected = False
        self._publish_enabled = True  # Keep enabled but in null mode

        # Start a dummy auto-reconnect task that does nothing
        self._auto_reconnect_task = asyncio.create_task(self._null_reconnect_manager())

    async def _null_reconnect_manager(self) -> None:
        """Dummy reconnect manager for null publisher mode."""
        # Just sleep forever, we're in null mode
        while self._null_publisher_mode:
            await asyncio.sleep(3600)  # Sleep for an hour

    async def initialize(self) -> bool:
        """Initialize WebSocket connection to dashboard with diagnostics and delay."""
        if not self._publish_enabled:
            logger.info("WebSocket publishing disabled in settings")
            return True  # Return True to indicate successful initialization (in disabled state)

        # Apply connection delay if configured
        if self.connection_delay > 0:
            logger.info(
                "Applying connection delay of %ss for service startup",
                self.connection_delay,
            )
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
            logger.warning(
                "WebSocket connection failed: %s. Enabling null publisher mode.", str(e)
            )
            self._enable_null_publisher_mode()

        # Always return True to allow bot to continue
        return True

    async def _auto_reconnect_manager(self) -> None:
        """Background task to manage automatic reconnections."""
        consecutive_reconnect_failures = 0
        max_consecutive_failures = 3  # Reduced from 5 for faster fallback
        total_reconnect_attempts = 0
        max_total_attempts = 10  # Total limit across all sessions

        while self._publish_enabled and not self._null_publisher_mode:
            try:
                if not self._connected:
                    consecutive_reconnect_failures += 1
                    total_reconnect_attempts += 1

                    if (
                        consecutive_reconnect_failures > max_consecutive_failures
                        or total_reconnect_attempts > max_total_attempts
                    ):
                        logger.warning(
                            "WebSocket reconnection limit reached (consecutive: %d/%d, total: %d/%d). "
                            "Switching to null publisher mode.",
                            consecutive_reconnect_failures,
                            max_consecutive_failures,
                            total_reconnect_attempts,
                            max_total_attempts,
                        )
                        # Switch to null publisher mode instead of disabling
                        self._enable_null_publisher_mode()
                        break

                    logger.info(
                        "Connection lost, attempting to reconnect (attempt %d/%d, total: %d/%d)...",
                        consecutive_reconnect_failures,
                        max_consecutive_failures,
                        total_reconnect_attempts,
                        max_total_attempts,
                    )
                    try:
                        await self._reconnect()
                        consecutive_reconnect_failures = 0  # Reset on success
                    except Exception as e:
                        logger.debug("Auto-reconnect failed: %s", str(e))
                        # Reset URL index to try primary URL again after failures
                        self._url_index = 0
                        self._current_url = self.dashboard_url
                else:
                    # Connection is healthy, reset failure counter
                    consecutive_reconnect_failures = 0

                # Check connection status every 10 seconds
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in auto-reconnect manager")
                await asyncio.sleep(30)  # Wait longer on unexpected errors

    async def _run_network_diagnostics(self) -> None:
        """Run network diagnostics to check connectivity to dashboard service."""
        try:
            logger.debug("Running network diagnostics for dashboard connectivity...")

            # Quick DNS resolution test for Docker services
            if self._environment_detected == "docker":
                try:
                    socket.gethostbyname("dashboard-backend")
                    logger.debug("✓ Docker service name 'dashboard-backend' resolves")
                except OSError:
                    logger.debug(
                        "✗ Docker service name 'dashboard-backend' does not resolve"
                    )

            # Test only the first few URLs to avoid long delays
            urls_to_test = [self.dashboard_url, *self.fallback_urls[:2]]
            successful_urls = []

            for url in urls_to_test:
                try:
                    # Extract host and port from WebSocket URL
                    import urllib.parse

                    parsed = urllib.parse.urlparse(url)
                    host = parsed.hostname or "localhost"
                    port = parsed.port or 8000

                    # Quick TCP connection test
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)  # 2 second timeout

                    try:
                        # Try to resolve hostname first
                        ip = socket.gethostbyname(host)
                        result = sock.connect_ex((ip, port))
                        if result == 0:
                            logger.debug(
                                "✓ TCP connectivity to %s:%d confirmed", host, port
                            )
                            successful_urls.append(url)
                        else:
                            logger.debug(
                                "✗ TCP connection to %s:%d failed (error: %d)",
                                host,
                                port,
                                result,
                            )
                    except socket.gaierror:
                        logger.debug("✗ Cannot resolve hostname: %s", host)
                    finally:
                        sock.close()

                except Exception as e:
                    logger.debug("Network diagnostic error for %s: %s", url, str(e))

            if not successful_urls:
                logger.info(
                    "⚠ No dashboard endpoints reachable. Bot will continue in offline mode."
                )

        except Exception:
            logger.debug("Network diagnostics failed")

    async def _connect_with_fallback(self) -> None:
        """Try to connect using primary URL and fallback URLs if needed."""
        urls_to_try = [
            self.dashboard_url,
            *self.fallback_urls[:5],
        ]  # Limit fallbacks to prevent long delays

        # Quick connectivity pre-check
        reachable_urls = []
        for url in urls_to_try:
            try:
                import urllib.parse

                parsed = urllib.parse.urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8000

                # Quick socket test
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                try:
                    ip = socket.gethostbyname(host)
                    if sock.connect_ex((ip, port)) == 0:
                        reachable_urls.append(url)
                except socket.gaierror:
                    pass
                finally:
                    sock.close()
            except Exception:
                pass

        # Try reachable URLs first
        urls_to_try = reachable_urls + [
            u for u in urls_to_try if u not in reachable_urls
        ]

        if not reachable_urls:
            logger.info(
                "No reachable dashboard URLs found. Enabling null publisher mode."
            )
            raise Exception("No reachable dashboard endpoints")

        for i, url in enumerate(urls_to_try):
            try:
                logger.info(
                    "Attempting WebSocket connection to %s (attempt %s/%s)",
                    url,
                    i + 1,
                    len(urls_to_try),
                )
                self._current_url = url
                self._url_index = i

                await self._connect_to_url(url)

                if self._connected:
                    logger.info("✓ Successfully connected to %s", url)
                    return

            except Exception as e:
                logger.debug("✗ Connection to %s failed: %s", url, str(e))

                if i < len(urls_to_try) - 1:
                    await asyncio.sleep(1)  # Brief delay between attempts
                else:
                    logger.info(
                        "All WebSocket URLs failed. Switching to null publisher mode."
                    )
                    raise Exception("All connection attempts failed") from None

    async def _connect_to_url(self, url: str) -> None:
        """Connect to a specific WebSocket URL."""
        try:
            logger.debug("Connecting to dashboard WebSocket at %s", url)
            self._connection_start_time = time.time()

            # Use shorter timeout for faster failure detection
            timeout = min(
                (
                    self.initial_connect_timeout
                    if self._retry_count == 0
                    else self.connection_timeout
                ),
                10.0,  # Cap at 10 seconds for faster fallback
            )

            # Enhanced connection parameters for better stability
            self._ws = await websockets.connect(
                url,
                open_timeout=timeout,  # Updated for websockets >= 15.0
                ping_interval=self.ping_interval,  # Configurable ping interval
                ping_timeout=self.ping_timeout,  # Configurable ping timeout
                close_timeout=5,  # Quick close timeout
                max_size=2**20,  # 1MB max message size
                compression=None,  # Disable compression to reduce CPU overhead
                # Additional headers for better compatibility
                additional_headers={
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
            logger.info(
                "Successfully connected to dashboard WebSocket in %.2fs",
                connection_time,
            )

        except (TimeoutError, OSError, InvalidURI) as e:
            self._connected = False
            self._consecutive_failures += 1
            logger.debug("Failed to connect to dashboard WebSocket: %s", str(e))
            raise
        except Exception as e:
            self._connected = False
            self._consecutive_failures += 1
            logger.debug(
                "Unexpected error connecting to dashboard WebSocket: %s", str(e)
            )
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect with enhanced exponential backoff."""
        # Don't reconnect in null publisher mode
        if self._null_publisher_mode:
            return

        if self._retry_count >= self.max_retries:
            logger.warning(
                "Max retries (%s) reached, switching to null publisher mode",
                self.max_retries,
            )
            self._enable_null_publisher_mode()
            return

        # Clean up existing connection
        await self._cleanup_connection()

        # Clear queues to prevent memory buildup during disconnection
        if self._retry_count > 3:
            await self._clear_queues(keep_priority=True)

        self._retry_count += 1

        # Enhanced exponential backoff with jitter to prevent thundering herd
        base_delay = self.retry_delay * (2 ** (self._retry_count - 1))
        jitter = base_delay * 0.1 * (0.5 - time.time() % 1)
        delay = min(base_delay + jitter, 60)  # Cap at 60 seconds

        # Additional delay for consecutive failures
        if self._consecutive_failures > 3:
            delay = min(delay * 1.5, 120)  # Extended delay for persistent issues

        logger.warning(
            "Reconnecting in %.1fs (attempt %d/%d, failures: %d)",
            delay,
            self._retry_count,
            self.max_retries,
            self._consecutive_failures,
        )

        await asyncio.sleep(delay)

        try:
            await self._connect_with_fallback()
        except Exception:
            logger.debug("Reconnection attempt %s failed", self._retry_count)
            # Don't bother with health checks in null publisher mode
            if self._retry_count > 2 and not self._null_publisher_mode:
                await self._check_endpoint_health()

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send message to dashboard with queue-based delivery and prioritization."""
        if not self._publish_enabled:
            return

        # In null publisher mode, just drop all messages silently
        if self._null_publisher_mode:
            return

        # Check if we're connected before queuing messages
        if not self._connected and self._consecutive_failures > 3:
            # Skip non-priority messages if we've been disconnected for a while
            message_type = message.get("type", "unknown")
            if message_type not in self._priority_message_types:
                return

        try:
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(UTC).isoformat()

            message_type = message.get("type", "unknown")

            # Update queue statistics
            current_queue_size = self._message_queue.qsize()
            priority_queue_size = self._priority_queue.qsize()

            # Implement backpressure - if queues are too full, start dropping non-critical messages early
            total_queue_usage = (current_queue_size + priority_queue_size) / (
                self.queue_size + self._priority_queue.maxsize
            )
            if (
                total_queue_usage > 0.8
                and message_type not in self._priority_message_types
            ):
                # Drop non-critical messages when queues are 80% full
                self._queue_stats["messages_dropped"] += 1
                return

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
                    except (TimeoutError, asyncio.QueueFull):
                        # Priority queue full, fall back to regular queue
                        logger.debug(
                            "Priority queue full, using regular queue for: %s",
                            message_type,
                        )
                    else:
                        logger.debug("Queued priority message: %s", message_type)
                        return

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
                    logger.warning(
                        "Message queue full (size: %s/%s), dropping %s message. "
                        "Stats: sent=%s, dropped=%s, queue_full_events=%s",
                        current_queue_size,
                        self.queue_size,
                        message_type,
                        self._queue_stats["messages_sent"],
                        self._queue_stats["messages_dropped"],
                        self._queue_stats["queue_full_events"],
                    )

                # For critical messages, try to make room by dropping older non-priority messages
                if is_priority and current_queue_size > self.queue_size * 0.9:
                    self._make_room_for_priority_message(message)

        except Exception:
            logger.exception("Error queuing WebSocket message")

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
                        logger.debug(
                            "Dropped %s message to make room for priority message",
                            msg_type,
                        )

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

        except Exception:
            logger.exception("Error making room for priority message")

    async def _queue_worker(self) -> None:
        """Background worker to process message queue with priority handling."""
        # Don't run worker in null publisher mode
        if self._null_publisher_mode:
            return

        backoff_delay = 0.1
        max_backoff = 5.0

        while self._publish_enabled and not self._null_publisher_mode:
            try:
                # Get next message from appropriate queue
                message = await self._get_next_message()
                if not message:
                    continue

                # Check connection and handle disconnection
                if not self._is_ready_to_send():
                    self._handle_message_when_disconnected(message)
                    # Apply exponential backoff when disconnected
                    await asyncio.sleep(min(backoff_delay, max_backoff))
                    backoff_delay = min(backoff_delay * 2, max_backoff)
                    continue

                # Reset backoff on successful connection
                backoff_delay = 0.1

                # Send the message
                await self._send_message_with_error_handling(message)

            except Exception:
                logger.exception("Error in queue worker")
                await asyncio.sleep(1)

    async def _get_next_message(self):
        """Get the next message from priority or regular queue."""
        # Check priority queue first
        try:
            message = self._priority_queue.get_nowait()
            logger.debug("Processing priority message")
            return message
        except asyncio.QueueEmpty:
            pass

        # No priority messages, check regular queue
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=5.0)
        except TimeoutError:
            return None

    def _is_ready_to_send(self) -> bool:
        """Check if WebSocket is ready to send messages."""
        return bool(self._connected and self._ws)

    def _handle_message_when_disconnected(self, message):
        """Handle message when WebSocket is disconnected."""
        if message and message.get("type") in self._priority_message_types:
            logger.debug(
                "Dropping priority message %s due to disconnection",
                message.get("type"),
            )

    async def _send_message_with_error_handling(self, message):
        """Send message with comprehensive error handling."""
        if not message:
            return

        try:
            # Send the message
            json_data = json.dumps(message, cls=TradingJSONEncoder)
            await self._ws.send(json_data)

            # Update statistics and logging
            self._update_send_statistics(message)

        except (ConnectionClosed, WebSocketException) as e:
            logger.warning("WebSocket connection lost during send: %s", e)
            self._connected = False
            await self._handle_connection_lost(message)

        except Exception:
            logger.exception("Error sending WebSocket message")

    def _update_send_statistics(self, message):
        """Update sending statistics and log health if needed."""
        self._queue_stats["messages_sent"] += 1
        message_type = message.get("type", "unknown")
        logger.debug("Sent WebSocket message: %s", message_type)

        # Log queue health every 100 messages
        if self._queue_stats["messages_sent"] % 100 == 0:
            self._log_queue_health()

    async def _handle_connection_lost(self, message):
        """Handle connection loss by re-queuing priority messages."""
        if message.get("type") in self._priority_message_types:
            try:
                self._priority_queue.put_nowait(message)
                logger.debug("Re-queued priority message after connection loss")
            except asyncio.QueueFull:
                logger.warning(
                    "Priority queue full, dropping priority message after connection loss"
                )
        else:
            logger.debug(
                "Dropping non-priority message %s after connection loss",
                message.get("type"),
            )

    def _log_queue_health(self) -> None:
        """Log queue health statistics."""
        regular_queue_size = self._message_queue.qsize()
        priority_queue_size = self._priority_queue.qsize()

        logger.info(
            "📊 WebSocket Queue Health: "
            "Regular=%s/%s, "
            "Priority=%s/%s, "
            "Sent=%s, "
            "Dropped=%s, "
            "MaxSeen=%s",
            regular_queue_size,
            self.queue_size,
            priority_queue_size,
            self._priority_queue.maxsize,
            self._queue_stats["messages_sent"],
            self._queue_stats["messages_dropped"],
            self._queue_stats["max_queue_size_seen"],
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
                        logger.warning(
                            "No pong received for %.1fs, connection may be stale",
                            time_since_pong,
                        )
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
                except Exception:
                    logger.warning("Health check failed")
                    self._connected = False
                    break

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in connection monitor")
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
                # Log cleanup errors but don't raise them
                logger.debug("Error closing WebSocket connection during cleanup")
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
                                logger.info(
                                    "Dashboard endpoint %s is reachable", base_url
                                )
                            else:
                                logger.warning(
                                    "Dashboard endpoint %s returned status %s",
                                    base_url,
                                    response.status,
                                )
                    except Exception:
                        logger.warning("Dashboard endpoint %s health check failed", url)

        except Exception:
            logger.warning("Dashboard endpoint health check failed")

    async def disconnect(self) -> None:
        """Disconnect WebSocket and cleanup resources - alias for close()."""
        await self.close()

    async def _clear_queues(self, keep_priority: bool = False) -> None:
        """Clear message queues to prevent memory buildup.

        Args:
            keep_priority: If True, keeps priority messages
        """
        cleared_regular = 0
        cleared_priority = 0

        # Clear regular queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
                cleared_regular += 1
            except asyncio.QueueEmpty:
                break

        # Clear priority queue if requested
        if not keep_priority:
            while not self._priority_queue.empty():
                try:
                    self._priority_queue.get_nowait()
                    cleared_priority += 1
                except asyncio.QueueEmpty:
                    break

        if cleared_regular > 0 or cleared_priority > 0:
            logger.info(
                "Cleared message queues: regular=%d, priority=%d",
                cleared_regular,
                cleared_priority,
            )

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

        # Clear all message queues
        await self._clear_queues(keep_priority=False)

        logger.info("WebSocket publisher closed")

    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected."""
        # In null publisher mode, always report as "connected" to prevent bot concerns
        if self._null_publisher_mode:
            return True
        return self._connected and self._ws is not None and not self._ws.closed
