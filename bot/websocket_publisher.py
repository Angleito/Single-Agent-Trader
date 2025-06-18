"""WebSocket publisher for sending real-time trading data to dashboard."""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException, InvalidURI

from .config import Settings
from .trading_types import MarketState, Position, TradeAction

logger = logging.getLogger(__name__)


class TradingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for trading data types."""
    
    def default(self, obj):
        """Convert non-serializable objects to JSON-compatible formats."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # datetime-like objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # pydantic models and other objects
            return obj.__dict__
        elif hasattr(obj, 'model_dump'):  # pydantic v2 models
            return obj.model_dump()
        return super().default(obj)


class WebSocketPublisher:
    """Publishes real-time trading data to dashboard via WebSocket."""

    def __init__(self, settings: Settings):
        """Initialize WebSocket publisher with settings."""
        self.settings = settings
        self.dashboard_url = getattr(
            settings.system, 'websocket_dashboard_url', 'ws://dashboard-backend:8000/ws'
        )
        self.publish_interval = getattr(
            settings.system, 'websocket_publish_interval', 1.0
        )
        self.max_retries = getattr(settings.system, 'websocket_max_retries', 5)
        self.retry_delay = getattr(settings.system, 'websocket_retry_delay', 5)
        self.connection_timeout = getattr(settings.system, 'websocket_timeout', 30)
        self.ping_interval = getattr(settings.system, 'websocket_ping_interval', 15)
        self.ping_timeout = getattr(settings.system, 'websocket_ping_timeout', 8)
        self.queue_size = getattr(settings.system, 'websocket_queue_size', 500)
        self.health_check_interval = getattr(settings.system, 'websocket_health_check_interval', 30)

        self._ws: WebSocketClientProtocol | None = None
        self._connected = False
        self._retry_count = 0
        self._publish_enabled = getattr(
            settings.system, 'enable_websocket_publishing', True
        )
        self._last_pong_time = None
        self._ping_task: asyncio.Task | None = None
        self._connection_monitor_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._queue_worker_task: asyncio.Task | None = None
        self._connection_start_time = None
        self._consecutive_failures = 0

        logger.info(f"WebSocketPublisher initialized - URL: {self.dashboard_url}")

    async def initialize(self) -> bool:
        """Initialize WebSocket connection to dashboard."""
        if not self._publish_enabled:
            logger.info("WebSocket publishing disabled in settings")
            return False

        try:
            await self._connect()
            if self._connected:
                logger.info("WebSocket publisher initialized successfully")
                # Start automatic reconnection monitoring
                asyncio.create_task(self._auto_reconnect_manager())
                return True
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket publisher: {e}")
            # Start auto-reconnect even if initial connection fails
            asyncio.create_task(self._auto_reconnect_manager())
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
                        logger.error(f"Auto-reconnect failed: {e}")
                
                # Check connection status every 10 seconds
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-reconnect manager: {e}")
                await asyncio.sleep(30)  # Wait longer on unexpected errors

    async def _connect(self) -> None:
        """Establish WebSocket connection to dashboard with improved parameters."""
        try:
            logger.info(f"Connecting to dashboard WebSocket at {self.dashboard_url}")
            self._connection_start_time = time.time()
            
            # Enhanced connection parameters for better stability
            self._ws = await websockets.connect(
                self.dashboard_url,
                timeout=self.connection_timeout,
                ping_interval=self.ping_interval,  # Configurable ping interval
                ping_timeout=self.ping_timeout,    # Configurable ping timeout
                close_timeout=5,   # Quick close timeout
                max_size=2**20,    # 1MB max message size
                compression=None,  # Disable compression to reduce CPU overhead
                # Additional headers for better compatibility
                extra_headers={
                    "User-Agent": "AI-Trading-Bot-WebSocket-Publisher/1.0",
                    "Accept": "*/*",
                    "Connection": "Upgrade"
                }
            )
            
            self._connected = True
            self._retry_count = 0
            self._consecutive_failures = 0
            self._last_pong_time = time.time()
            
            # Start monitoring tasks
            self._start_monitoring_tasks()
            
            logger.info(f"Successfully connected to dashboard WebSocket in {time.time() - self._connection_start_time:.2f}s")

        except (OSError, InvalidURI, asyncio.TimeoutError) as e:
            self._connected = False
            self._consecutive_failures += 1
            logger.error(f"Failed to connect to dashboard WebSocket: {e}")
            raise
        except Exception as e:
            self._connected = False
            self._consecutive_failures += 1
            logger.error(f"Unexpected error connecting to dashboard WebSocket: {e}")
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect with enhanced exponential backoff."""
        if self._retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) reached, giving up")
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
        
        logger.warning(f"Reconnecting in {delay:.1f}s (attempt {self._retry_count}/{self.max_retries}, failures: {self._consecutive_failures})")

        await asyncio.sleep(delay)

        try:
            await self._connect()
        except Exception as e:
            logger.error(f"Reconnection attempt {self._retry_count} failed: {e}")
            # If we've failed multiple times, check if URL is reachable
            if self._retry_count > 2:
                await self._check_endpoint_health()

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send message to dashboard with queue-based delivery."""
        if not self._publish_enabled:
            return

        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now(UTC).isoformat()

            # Queue message for delivery
            await asyncio.wait_for(self._message_queue.put(message), timeout=1.0)
            
        except asyncio.TimeoutError:
            logger.warning("Message queue full, dropping message")
        except Exception as e:
            logger.error(f"Error queuing WebSocket message: {e}")

    async def _queue_worker(self) -> None:
        """Background worker to process message queue."""
        while self._publish_enabled:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(self._message_queue.get(), timeout=5.0)
                
                if not self._connected or not self._ws:
                    # Drop message if not connected
                    continue

                try:
                    # Use custom encoder for JSON serialization
                    json_data = json.dumps(message, cls=TradingJSONEncoder)
                    await self._ws.send(json_data)
                    logger.debug(f"Sent WebSocket message: {message.get('type', 'unknown')}")
                    
                except (ConnectionClosed, WebSocketException) as e:
                    logger.warning(f"WebSocket connection lost during send: {e}")
                    self._connected = False
                    # Re-queue the message for retry
                    try:
                        self._message_queue.put_nowait(message)
                    except asyncio.QueueFull:
                        logger.warning("Queue full, dropping message after connection loss")
                    
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")

            except asyncio.TimeoutError:
                # No messages to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                await asyncio.sleep(1)

    async def publish_system_status(self, status: str, details: dict[str, Any] = None, health: bool = None, status_message_text: str = None, **kwargs) -> None:
        """Publish system status update."""
        status_message: dict[str, Any] = {
            'type': 'system_status',
            'status': status,
            'details': details or {},
            'bot_id': 'ai-trading-bot'
        }

        # Support legacy parameters
        if health is not None:
            status_message['health'] = health
        if status_message_text is not None:
            status_message['message'] = status_message_text

        # Add any additional kwargs
        status_message.update(kwargs)

        await self._send_message(status_message)

    async def publish_market_data(self, symbol: str, price: float, volume: float = 0, market_state: MarketState = None, timestamp: Any = None) -> None:
        """Publish market data update."""
        message = {
            'type': 'market_data',
            'symbol': symbol,
            'price': float(price),
            'volume': float(volume),
            'market_state': market_state,  # Let the encoder handle serialization
            'data_timestamp': timestamp  # Let the encoder handle timestamp conversion
        }
        await self._send_message(message)

    async def publish_indicator_data(self, symbol: str, indicators: Any) -> None:
        """Publish technical indicator data."""
        message = {
            'type': 'indicator_data',
            'symbol': symbol,
            'indicators': indicators
        }
        await self._send_message(message)

    async def publish_ai_decision(self, action: str = None, decision: str = None, confidence: float = 0, reasoning: str = "", context: dict[str, Any] = None) -> None:
        """Publish AI trading decision."""
        # Support both 'action' and 'decision' parameters for backward compatibility
        final_decision = action or decision or "HOLD"
        message = {
            'type': 'ai_decision',
            'decision': final_decision,
            'action': final_decision,  # Include both for compatibility
            'confidence': float(confidence),
            'reasoning': reasoning,
            'context': context or {}
        }
        await self._send_message(message)

    async def publish_trading_decision(self, trade_action: TradeAction, symbol: str, current_price: float, context: dict[str, Any] = None) -> None:
        """Publish trading decision."""
        message = {
            'type': 'trading_decision',
            'action': trade_action.action,
            'symbol': symbol,
            'size_pct': float(trade_action.size_pct),
            'price': float(current_price),
            'reasoning': trade_action.rationale or '',
            'take_profit_pct': float(trade_action.take_profit_pct) if trade_action.take_profit_pct else None,
            'stop_loss_pct': float(trade_action.stop_loss_pct) if trade_action.stop_loss_pct else None,
            'leverage': trade_action.leverage,
            'context': context or {}
        }
        await self._send_message(message)

    async def publish_trade_execution(self, trade_result: dict[str, Any]) -> None:
        """Publish trade execution result."""
        message = {
            'type': 'trade_execution',
            'result': trade_result
        }
        await self._send_message(message)

    async def publish_position_update(self, position: Position = None, positions: list[Position] = None) -> None:
        """Publish position update."""
        message = {
            'type': 'position_update',
            'position': position.model_dump() if position and hasattr(position, 'model_dump') else (position.__dict__ if position else None),
            'positions': [pos.model_dump() if hasattr(pos, 'model_dump') else pos.__dict__ for pos in (positions or [])]
        }
        await self._send_message(message)

    async def publish_performance_update(self, metrics: dict[str, Any]) -> None:
        """Publish performance metrics."""
        message = {
            'type': 'performance_update',
            'metrics': metrics
        }
        await self._send_message(message)

    async def publish_error(self, error_type: str, error_message: str, details: dict[str, Any] = None) -> None:
        """Publish error notification."""
        message = {
            'type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'details': details or {}
        }
        await self._send_message(message)

    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Start queue worker
        if self._queue_worker_task is None or self._queue_worker_task.done():
            self._queue_worker_task = asyncio.create_task(self._queue_worker())
        
        # Start connection monitor
        if self._connection_monitor_task is None or self._connection_monitor_task.done():
            self._connection_monitor_task = asyncio.create_task(self._connection_monitor())

    async def _connection_monitor(self) -> None:
        """Monitor connection health and trigger reconnections if needed."""
        while self._connected and self._publish_enabled:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if not self._connected or not self._ws:
                    break
                
                # Check if we've received a pong recently
                if self._last_pong_time:
                    time_since_pong = time.time() - self._last_pong_time
                    if time_since_pong > 60:  # 60 seconds without pong
                        logger.warning(f"No pong received for {time_since_pong:.1f}s, connection may be stale")
                        self._connected = False
                        break
                
                # Send a custom ping to test connection
                try:
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    self._last_pong_time = time.time()
                    logger.debug("Connection health check passed")
                except asyncio.TimeoutError:
                    logger.warning("Health check ping timeout, marking connection as unhealthy")
                    self._connected = False
                    break
                except Exception as e:
                    logger.warning(f"Health check failed: {e}")
                    self._connected = False
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(5)

    async def _cleanup_connection(self) -> None:
        """Clean up existing connection and tasks."""
        self._connected = False
        
        # Cancel monitoring tasks
        if self._connection_monitor_task and not self._connection_monitor_task.done():
            self._connection_monitor_task.cancel()
            try:
                await self._connection_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._ws = None

    async def _check_endpoint_health(self) -> None:
        """Check if the WebSocket endpoint is reachable."""
        try:
            # Try a simple HTTP request to the base URL
            import aiohttp
            base_url = self.dashboard_url.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '/health')
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(base_url) as response:
                    if response.status == 200:
                        logger.info("Dashboard endpoint is reachable")
                    else:
                        logger.warning(f"Dashboard endpoint returned status {response.status}")
        except Exception as e:
            logger.warning(f"Dashboard endpoint health check failed: {e}")

    async def close(self) -> None:
        """Close WebSocket connection and cleanup resources."""
        logger.info("Closing WebSocket publisher")
        self._publish_enabled = False
        
        # Cancel queue worker
        if self._queue_worker_task and not self._queue_worker_task.done():
            self._queue_worker_task.cancel()
            try:
                await self._queue_worker_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup connection
        await self._cleanup_connection()
        
        logger.info("WebSocket publisher closed")
