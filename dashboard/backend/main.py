#!/usr/bin/env python3
"""
FastAPI server for AI Trading Bot Dashboard

Provides WebSocket streaming of bot logs and REST API endpoints for monitoring
the AI trading bot running in Docker container.
"""

import asyncio
import json
import logging
import math
import os
import subprocess
import time
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Optional imports for fallback functionality
try:
    import psutil
except ImportError:
    psutil = None

import aiohttp
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llm_log_parser import AlertThresholds, create_llm_log_parser
from rate_limiter import rate_limit_middleware

# Import TradingView data feed and LLM log parser
from tradingview_feed import generate_sample_data, tradingview_feed


class DockerCommandError(Exception):
    """Raised when Docker commands are not available or fail."""


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Background tasks tracking
background_tasks: set[asyncio.Task] = set()

# Bluefin Service Configuration
BLUEFIN_SERVICE_URL = os.getenv("BLUEFIN_SERVICE_URL", "http://localhost:8080")
BLUEFIN_SERVICE_API_KEY = os.getenv("BLUEFIN_SERVICE_API_KEY", "")


class BluefinServiceClient:
    """HTTP client for communicating with the Bluefin SDK service."""

    def __init__(
        self,
        base_url: str = BLUEFIN_SERVICE_URL,
        api_key: str = BLUEFIN_SERVICE_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Content-Type"] = "application/json"

            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> dict:
        """Check if Bluefin service is healthy."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status}",
                    "message": await response.text(),
                }
        except Exception as e:
            logger.exception("Bluefin health check failed")
            return {
                "status": "unreachable",
                "error": str(e),
                "service_url": self.base_url,
            }

    async def get_account_info(self) -> dict:
        """Get account information from Bluefin service."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/account") as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Bluefin service error: {await response.text()}",
                )
        except aiohttp.ClientError as e:
            logger.exception("Failed to get Bluefin account info")
            raise HTTPException(
                status_code=503, detail=f"Bluefin service unavailable: {e!s}"
            ) from e

    async def get_positions(self) -> dict:
        """Get current positions from Bluefin service."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/positions") as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Bluefin service error: {await response.text()}",
                )
        except aiohttp.ClientError as e:
            logger.exception("Failed to get Bluefin positions")
            raise HTTPException(
                status_code=503, detail=f"Bluefin service unavailable: {e!s}"
            ) from e

    async def get_orders(self) -> dict:
        """Get current orders from Bluefin service."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/orders") as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Bluefin service error: {await response.text()}",
                )
        except aiohttp.ClientError as e:
            logger.exception("Failed to get Bluefin orders")
            raise HTTPException(
                status_code=503, detail=f"Bluefin service unavailable: {e!s}"
            ) from e

    async def get_market_ticker(self, symbol: str) -> dict:
        """Get market ticker data from Bluefin service."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/market/ticker", params={"symbol": symbol}
            ) as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Bluefin service error: {await response.text()}",
                )
        except aiohttp.ClientError as e:
            logger.exception("Failed to get Bluefin market ticker")
            raise HTTPException(
                status_code=503, detail=f"Bluefin service unavailable: {e!s}"
            ) from e

    async def place_order(self, order_data: dict) -> dict:
        """Place an order via Bluefin service."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/orders", json=order_data
            ) as response:
                if response.status in [200, 201]:
                    return await response.json()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Bluefin order error: {await response.text()}",
                )
        except aiohttp.ClientError as e:
            logger.exception("Failed to place Bluefin order")
            raise HTTPException(
                status_code=503, detail=f"Bluefin service unavailable: {e!s}"
            ) from e


# Global Bluefin service client
bluefin_client = BluefinServiceClient()


# Enhanced connection manager for WebSocket connections with message persistence
class ConnectionManager:
    """
    Enhanced WebSocket connection manager with message persistence and replay capabilities.

    Features:
    - Message categorization (trading, indicator, system, log)
    - Configurable replay buffer sizes per category
    - Message filtering and querying
    - Performance metrics tracking
    - Connection-specific message state
    """

    def __init__(self):
        self.active_connections: set[WebSocket] = set()

        # Enhanced message storage with categorization
        self.message_buffers = {
            "trading": [],  # Trading decisions and executions
            "indicator": [],  # Technical indicator data
            "system": [],  # System status and health
            "log": [],  # General log messages
            "ai": [],  # AI/LLM decision messages
        }

        # Buffer size limits per category
        self.buffer_limits = {
            "trading": 500,  # Keep more trading history
            "indicator": 1000,  # Keep recent indicator data
            "system": 200,  # System status messages
            "log": 1000,  # General logs
            "ai": 300,  # AI decision history
        }

        # Connection tracking for personalized replay
        self.connection_metadata = {}  # websocket -> {connected_at, last_message_id, categories}

        # Performance tracking
        self.stats = {
            "total_messages": 0,
            "messages_by_category": dict.fromkeys(self.message_buffers, 0),
            "connections_served": 0,
            "total_replay_messages": 0,
        }

    def _categorize_message(self, message: dict) -> str:
        """Categorize message based on type and content."""
        msg_type = message.get("type", "").lower()
        source = message.get("source", "").lower()

        # Define categories and their associated keywords
        category_keywords = {
            "trading": {
                "type": ["trade", "order", "position", "execution"],
                "source": ["trading", "exchange", "order"],
            },
            "ai": {"type": ["ai", "llm", "decision"], "source": ["llm", "agent"]},
            "indicator": {
                "type": ["indicator", "signal", "technical"],
                "source": ["indicator", "signal"],
            },
            "system": {
                "type": ["system", "health", "status", "error"],
                "source": ["system", "health"],
            },
        }

        # Check each category
        for category, keywords in category_keywords.items():
            # Check type keywords
            if any(keyword in msg_type for keyword in keywords["type"]):
                return category
            # Check source keywords (special handling for single keywords)
            if category == "ai":
                if "llm" in source or "agent" in source:
                    return category
            elif category == "system":
                if "system" in source or "health" in source:
                    return category
            elif any(keyword in source for keyword in keywords["source"]):
                return category

        # Default to log category
        return "log"

    async def connect(
        self,
        websocket: WebSocket,
        replay_categories: list[str] | None = None,
        connection_info: dict | None = None,
    ):
        """
        Accept WebSocket connection and send buffered messages by category.

        Args:
            websocket: WebSocket connection
            replay_categories: List of categories to replay (default: all)
            connection_info: Additional connection information for diagnostics
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        # Enhanced connection metadata with diagnostics
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now(UTC).isoformat(),
            "categories": replay_categories or list(self.message_buffers.keys()),
            "messages_sent": 0,
            "messages_received": 0,
            "last_activity": datetime.now(UTC).isoformat(),
            "connection_info": connection_info or {},
            "client_ip": (
                connection_info.get("client_ip", "unknown")
                if connection_info
                else "unknown"
            ),
            "origin": (
                connection_info.get("origin", "unknown")
                if connection_info
                else "unknown"
            ),
            "user_agent": (
                connection_info.get("user_agent", "unknown")
                if connection_info
                else "unknown"
            ),
            "is_healthy": True,
            "error_count": 0,
        }

        logger.info(
            "WebSocket connection established from %s (Origin: %s). "
            "Total connections: %s",
            self.connection_metadata[websocket]["client_ip"],
            self.connection_metadata[websocket]["origin"],
            len(self.active_connections),
        )

        # Send replay messages by category
        await self._send_replay_messages(websocket, replay_categories)

        # Update stats
        self.stats["connections_served"] += 1

    async def _send_replay_messages(
        self, websocket: WebSocket, categories: list[str] | None = None
    ):
        """Send buffered messages to new connection, organized by category."""
        categories = categories or list(self.message_buffers.keys())
        total_sent = 0

        # Send a connection status message first
        status_message = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "connection_status",
            "source": "dashboard-backend",
            "message": "Connection established - replaying recent messages",
            "categories_available": list(self.message_buffers.keys()),
            "categories_requested": categories,
        }

        try:
            await websocket.send_text(json.dumps(status_message))
            total_sent += 1
        except Exception:
            logger.exception("Error sending connection status")
            return

        # Send messages by category with limits
        replay_limits = {
            "trading": 50,  # Last 50 trading events
            "indicator": 20,  # Last 20 indicator updates
            "system": 10,  # Last 10 system messages
            "log": 30,  # Last 30 log entries
            "ai": 25,  # Last 25 AI decisions
        }

        for category in categories:
            if category in self.message_buffers:
                buffer = self.message_buffers[category]
                limit = replay_limits.get(category, 20)
                messages_to_send = buffer[-limit:] if buffer else []

                for message in messages_to_send:
                    try:
                        # Add replay marker
                        replay_message = message.copy()
                        replay_message["is_replay"] = True
                        replay_message["replay_category"] = category

                        await websocket.send_text(json.dumps(replay_message))
                        total_sent += 1
                    except Exception:
                        logger.exception(
                            "Error sending replay message from %s", category
                        )
                        break

        # Send replay completion message
        completion_message = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "replay_complete",
            "source": "dashboard-backend",
            "message": f"Replay complete - {total_sent} messages sent",
            "total_messages": total_sent,
            "categories": categories,
        }

        try:
            await websocket.send_text(json.dumps(completion_message))
        except Exception:
            logger.exception("Error sending replay completion")

        # Update connection metadata
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["messages_sent"] = total_sent

        # Update stats
        self.stats["total_replay_messages"] += total_sent

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection and cleanup metadata."""
        self.active_connections.discard(websocket)

        # Clean up connection metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        logger.info(
            "WebSocket connection closed. Total connections: %s",
            len(self.active_connections),
        )

    async def broadcast(self, message: dict):
        """
        Enhanced broadcast with message categorization and persistence.

        Args:
            message: Message dictionary to broadcast
        """
        if not self.active_connections and not message.get(
            "persist_when_no_connections", True
        ):
            return

        # Validate and enhance message structure
        try:
            if not isinstance(message, dict):
                logger.warning("Invalid message type for broadcast: %s", type(message))
                return

            # Ensure required fields
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(UTC).isoformat()

            if "type" not in message:
                message["type"] = "unknown"

            # Add message ID for tracking
            message["message_id"] = (
                f"{int(time.time() * 1000000)}"  # Microsecond timestamp
            )

            # Test JSON serialization
            json.dumps(message)

        except (TypeError, ValueError):
            logger.exception("Invalid message for broadcast")
            # Create a safe fallback message
            message = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "error",
                "message": "Invalid message format",
                "source": "dashboard-api",
                "message_id": f"{int(time.time() * 1000000)}",
            }

        # Categorize and store message
        category = self._categorize_message(message)
        buffer = self.message_buffers[category]
        limit = self.buffer_limits[category]

        # Add to appropriate buffer
        buffer.append(message)
        if len(buffer) > limit:
            buffer.pop(0)  # Remove oldest message

        # Update statistics
        self.stats["total_messages"] += 1
        self.stats["messages_by_category"][category] += 1

        # Broadcast to all active connections
        if self.active_connections:
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception:
                    logger.exception("Error broadcasting to connection")
                    disconnected.add(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.disconnect(conn)

    def get_messages_by_category(self, category: str, limit: int = 100) -> list[dict]:
        """Get recent messages from a specific category."""
        if category not in self.message_buffers:
            return []

        buffer = self.message_buffers[category]
        return buffer[-limit:] if buffer else []

    def get_message_stats(self) -> dict:
        """Get comprehensive message statistics."""
        return {
            **self.stats,
            "active_connections": len(self.active_connections),
            "buffer_sizes": {
                cat: len(buf) for cat, buf in self.message_buffers.items()
            },
            "buffer_limits": self.buffer_limits,
        }


# Global connection manager instance
manager = ConnectionManager()

# Global LLM log parser
llm_parser = create_llm_log_parser(
    log_file="/app/trading-logs/llm_completions.log",
    alert_thresholds=AlertThresholds(
        max_response_time_ms=30000,
        max_cost_per_hour=10.0,
        min_success_rate=0.90,
        max_consecutive_failures=3,
    ),
)


# Background task for log streaming
class LogStreamer:
    """Streams Docker container logs in background"""

    def __init__(self, container_name: str = "ai-trading-bot"):
        self.container_name = container_name
        self.process: subprocess.Popen | None = None
        self.running = False
        self.file_watchers = []
        self.use_file_based = (
            os.getenv("USE_FILE_BASED_LOGS", "false").lower() == "true"
        )
        # Task tracking for proper cleanup
        self._stream_logs_task: asyncio.Task | None = None
        self._file_watcher_tasks: list[asyncio.Task] = []

    async def start(self):
        """Start log streaming in background"""
        if self.running:
            return

        self.running = True
        logger.info("Starting log streamer for container: %s", self.container_name)

        if self.use_file_based:
            # Use file-based log streaming (more secure)
            await self._start_file_based_streaming()
        else:
            # Use Docker socket-based streaming (requires Docker socket access)
            await self._start_docker_streaming()

    async def _start_docker_streaming(self):
        """Start Docker socket-based log streaming"""
        try:
            # Check if docker command is available
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                raise DockerCommandError("Docker command not available")

            # Start docker logs process
            self.process = await asyncio.to_thread(
                subprocess.Popen,
                ["docker", "logs", "-f", "--tail", "100", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Stream logs in background
            self._stream_logs_task = asyncio.create_task(self._stream_logs())
            logger.info(
                "Started Docker-based log streaming for %s", self.container_name
            )

        except Exception:
            logger.exception("Failed to start Docker log streaming")
            logger.info("Falling back to file-based log streaming")
            await self._start_file_based_streaming()

    async def _start_file_based_streaming(self):
        """Start file-based log streaming (doesn't require Docker socket)"""
        try:
            # Watch trading bot log files
            log_paths = [
                "/app/trading-logs/trading.log",
                "/app/trading-logs/llm_completions.log",
                "/app/llm-logs/llm_completions.log",
            ]

            for log_path in log_paths:
                if Path(log_path).exists():
                    task = asyncio.create_task(self._watch_log_file(log_path))
                    self._file_watcher_tasks.append(task)
                    self.file_watchers.append(log_path)
                    logger.info("Started watching log file: %s", log_path)

            if not self.file_watchers:
                logger.warning("No log files found to watch")
            else:
                logger.info(
                    "Started file-based log streaming for %s files",
                    len(self.file_watchers),
                )

        except Exception:
            logger.exception("Failed to start file-based log streaming")
            self.running = False

    async def _watch_log_file(self, file_path: str):
        """Watch a log file for new lines"""
        try:
            # Get initial file size
            if not Path(file_path).exists():
                logger.warning("Log file not found: %s", file_path)
                return

            # Open file using asyncio.to_thread for the blocking open call
            f = await asyncio.to_thread(Path(file_path).open)

            try:
                # Go to end of file
                f.seek(0, 2)

                while self.running:
                    line = f.readline()
                    if line:
                        # Broadcast new log line
                        log_entry = {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "level": "INFO",
                            "message": line.strip(),
                            "source": f"file-{Path(file_path).name}",
                            "file_path": file_path,
                        }

                        # Extract log level if present
                        line_upper = line.upper()
                        for level in ["ERROR", "WARN", "INFO", "DEBUG"]:
                            if level in line_upper:
                                log_entry["level"] = level
                                break

                        await manager.broadcast(log_entry)
                    else:
                        # No new data, sleep briefly
                        await asyncio.sleep(0.5)
            finally:
                f.close()

        except Exception:
            logger.exception("Error watching log file %s", file_path)

    async def _stream_logs(self):
        """Stream logs from Docker container"""
        try:
            while self.running and self.process:
                line = self.process.stdout.readline()
                if not line:
                    break

                # Parse and broadcast log entry
                log_entry = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": "INFO",
                    "message": line.strip(),
                    "source": "docker-logs",
                }

                # Try to parse log level from message
                if any(
                    level in line.upper()
                    for level in ["ERROR", "WARN", "INFO", "DEBUG"]
                ):
                    for level in ["ERROR", "WARN", "INFO", "DEBUG"]:
                        if level in line.upper():
                            log_entry["level"] = level
                            break

                await manager.broadcast(log_entry)

        except Exception:
            logger.exception("Error in log streaming")
        finally:
            self.running = False
            if self.process:
                self.process.terminate()

    async def stop(self):
        """Stop log streaming and cleanup tasks"""
        self.running = False

        # Cancel stream logs task
        if self._stream_logs_task and not self._stream_logs_task.done():
            self._stream_logs_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stream_logs_task

        # Cancel file watcher tasks
        for task in self._file_watcher_tasks:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        self._file_watcher_tasks.clear()

        if self.process:
            self.process.terminate()
        self.file_watchers.clear()
        logger.info("Log streaming stopped")


# Global log streamer
log_streamer = LogStreamer()


# Track application state
class AppState:
    """Container for application state to avoid global variables."""

    def __init__(self):
        self.delayed_startup_task: asyncio.Task | None = None


app_state = AppState()


# Application lifecycle management
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application startup and shutdown"""

    # Startup
    logger.info("Starting FastAPI dashboard server...")

    # Try to start log streamer, but don't fail if target container doesn't exist
    try:
        # Use file-based log streaming for better reliability
        log_streamer.use_file_based = True

        # Start log streamer in background with delay to avoid startup blocking
        async def start_log_streaming_delayed():
            await asyncio.sleep(2)  # Wait 2 seconds before starting
            try:
                await log_streamer.start()
                logger.info("File-based log streaming started successfully")
            except Exception as e:
                logger.warning("Failed to start file-based log streaming: %s", e)

        app_state.delayed_startup_task = asyncio.create_task(
            start_log_streaming_delayed()
        )
        logger.info("Log streamer initialization scheduled (file-based)")
    except Exception as e:
        logger.warning(
            "Failed to start log streamer: %s. Continuing without log streaming.", e
        )

    # Initialize TradingView feed with sample data for demo
    generate_sample_data()
    logger.info("Initialized TradingView data feed with sample data")

    # Initialize LLM log parser
    try:
        # Parse existing logs
        counts = llm_parser.parse_log_file()
        logger.info("Parsed existing LLM logs: %s", counts)

        # Set up callback to broadcast LLM events to WebSocket clients
        def llm_event_callback(event_data):
            # Map event types to WebSocket message types
            ws_event_type = (
                "llm_decision"
                if event_data.get("event_type") == "trading_decision"
                else "llm_event"
            )

            # Extract key fields for easier frontend consumption
            formatted_event = {
                "timestamp": event_data.get("timestamp"),
                "type": ws_event_type,
                "event_type": event_data.get("event_type"),
                "source": "llm_parser",
            }

            # Add specific fields based on event type
            if event_data.get("event_type") == "trading_decision":
                formatted_event.update(
                    {
                        "action": event_data.get("action"),
                        "size_pct": event_data.get("size_pct"),
                        "rationale": event_data.get("rationale"),
                        "symbol": event_data.get("symbol"),
                        "current_price": event_data.get("current_price"),
                        "indicators": event_data.get("indicators", {}),
                        "session_id": event_data.get("session_id"),
                    }
                )
            elif event_data.get("event_type") == "performance_metrics":
                formatted_event.update(
                    {
                        "total_completions": event_data.get("total_completions"),
                        "avg_response_time_ms": event_data.get("avg_response_time_ms"),
                        "total_cost_estimate_usd": event_data.get(
                            "total_cost_estimate_usd"
                        ),
                    }
                )

            # Always include full data for detailed views
            formatted_event["data"] = event_data

            # Schedule the broadcast in a thread-safe way
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.create_task(manager.broadcast(formatted_event))
                    # Store task reference to prevent garbage collection
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                else:
                    # If no event loop is running, we'll skip this broadcast
                    logger.debug(
                        "No running event loop for LLM callback, skipping broadcast"
                    )
            except RuntimeError:
                # No event loop in current thread, skip broadcast
                logger.debug("No event loop in current thread for LLM callback")

        llm_parser.add_callback(llm_event_callback)

        # Start real-time monitoring
        llm_parser.start_real_time_monitoring(poll_interval=1.0)
        logger.info("Started LLM log real-time monitoring")

    except Exception as e:
        logger.warning(
            "Failed to initialize LLM log parser: %s. Continuing without LLM monitoring.",
            e,
        )

    yield

    # Shutdown
    logger.info("Shutting down FastAPI dashboard server...")

    # Cancel delayed startup task
    if app_state.delayed_startup_task and not app_state.delayed_startup_task.done():
        app_state.delayed_startup_task.cancel()
        with suppress(asyncio.CancelledError):
            await app_state.delayed_startup_task

    await log_streamer.stop()

    # Close Bluefin service client
    try:
        await bluefin_client.close()
        logger.info("Closed Bluefin service client")
    except Exception as e:
        logger.warning("Error closing Bluefin client: %s", e)

    # Stop LLM log monitoring
    try:
        llm_parser.stop_real_time_monitoring()
        logger.info("Stopped LLM log monitoring")
    except Exception as e:
        logger.warning("Error stopping LLM log monitoring: %s", e)


# Create FastAPI app
app = FastAPI(
    title="AI Trading Bot Dashboard",
    description="WebSocket and REST API for monitoring AI trading bot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
# Get allowed origins from environment or use comprehensive defaults
#
# Environment variables for CORS customization:
# - CORS_ORIGINS: Comma-separated list of allowed origins
# - FRONTEND_URL: Primary frontend URL for fallback scenarios
# - CORS_DEFAULT_ORIGIN: Default origin when no other fallback applies
# - CORS_ALLOW_CREDENTIALS: Enable/disable credentials (default: true)
# - ENVIRONMENT: When set to 'development', allows '*' origin for WebSocket upgrades
default_origins = "http://localhost:3000,http://127.0.0.1:3000,https://localhost:3000,https://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,https://localhost:3001,https://127.0.0.1:3001,http://localhost:8080,http://127.0.0.1:8080,https://localhost:8080,https://127.0.0.1:8080,http://localhost:80,http://127.0.0.1:80,https://localhost:443,https://127.0.0.1:443,http://localhost,https://localhost,http://dashboard-frontend:8080,http://dashboard-frontend-prod:8080,http://cursorprod-dashboard-frontend:8080,http://cursorprod_dashboard_frontend:8080,https://dashboard-frontend:8080,https://dashboard-frontend-prod:8080,http://dashboard-backend:8000,http://cursorprod-dashboard-backend:8000,http://cursorprod_dashboard_backend:8000,http://0.0.0.0:3000,http://0.0.0.0:3001,http://0.0.0.0:8080,http://0.0.0.0:80,ws://localhost:8000,ws://127.0.0.1:8000,ws://0.0.0.0:8000,wss://localhost:8000,wss://127.0.0.1:8000,wss://0.0.0.0:8000,http://localhost:8000,http://127.0.0.1:8000,http://0.0.0.0:8000,https://localhost:8000,https://127.0.0.1:8000,https://0.0.0.0:8000"

allowed_origins = os.getenv("CORS_ORIGINS", default_origins).split(",")
# Strip whitespace from origins
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Cache-Control",
        "Pragma",
    ],
    expose_headers=[
        "Content-Type",
        "Content-Length",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Headers",
    ],
)

# Log CORS configuration for debugging
logger.info("CORS configured with %s allowed origins:", len(allowed_origins))
for i, origin in enumerate(allowed_origins, 1):
    logger.info("  %s. %s", i, origin)
logger.info("CORS credentials allowed: %s", allow_credentials)


# Add rate limiting middleware
@app.middleware("http")
async def rate_limiting(request, call_next):
    """Apply rate limiting to all requests"""
    return await rate_limit_middleware(request, call_next)


# Add middleware for JSON response handling and security
@app.middleware("http")
async def add_json_headers(request, call_next):
    """Add proper JSON headers and security headers to all responses"""
    response = await call_next(request)

    # Add JSON content type for API responses
    if request.url.path.startswith("/api/") or request.url.path.startswith("/udf/"):
        response.headers["Content-Type"] = "application/json"

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Add CORS headers for WebSocket upgrades
    if request.headers.get("connection", "").lower() == "upgrade":
        origin = request.headers.get("origin")

        # Enhanced WebSocket CORS handling
        allow_origin = None
        if origin and (
            origin in allowed_origins
            or any(
                origin.startswith(prefix)
                for prefix in [
                    "http://localhost:",
                    "http://127.0.0.1:",
                    "ws://localhost:",
                    "ws://127.0.0.1:",
                ]
            )
            or any(
                origin.startswith(prefix)
                for prefix in [
                    "http://dashboard-frontend",
                    "https://dashboard-frontend",
                    "http://dashboard-backend",
                    "https://dashboard-backend",
                    "http://cursorprod-dashboard-frontend",
                    "https://cursorprod-dashboard-frontend",
                    "http://cursorprod_dashboard_frontend",
                    "https://cursorprod_dashboard_frontend",
                ]
            )
        ):
            allow_origin = origin

        # Set appropriate CORS headers
        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Credentials"] = str(
                allow_credentials
            ).lower()
        elif os.getenv("ENVIRONMENT", "development") == "development":
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            # Dynamic fallback based on request origin or environment
            fallback_origin = origin or os.getenv(
                "FRONTEND_URL", "http://localhost:3000"
            )
            response.headers["Access-Control-Allow-Origin"] = fallback_origin

        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, UPGRADE"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-Requested-With, Upgrade, Connection, Sec-WebSocket-Key, Sec-WebSocket-Version, Sec-WebSocket-Protocol"
        )

    return response


# OPTIONS handler for CORS preflight requests
@app.options("/{path:path}")
async def handle_options(_path: str, request: Request):
    """Handle CORS preflight requests"""
    origin = request.headers.get("origin")

    # Determine the appropriate origin to allow with dynamic fallback
    if origin and origin in allowed_origins:
        allow_origin = origin
    elif origin and any(
        origin.startswith(prefix)
        for prefix in [
            "http://localhost:",
            "http://127.0.0.1:",
            "https://localhost:",
            "https://127.0.0.1:",
        ]
    ):
        # Allow localhost/127.0.0.1 variations for development
        allow_origin = origin
    elif origin and any(
        origin.startswith(prefix)
        for prefix in [
            "http://dashboard-frontend",
            "https://dashboard-frontend",
            "http://dashboard-backend",
            "https://dashboard-backend",
            "http://cursorprod-dashboard-frontend",
            "https://cursorprod-dashboard-frontend",
            "http://cursorprod_dashboard_frontend",
            "https://cursorprod_dashboard_frontend",
        ]
    ):
        # Allow Docker container networking
        allow_origin = origin
    else:
        # Dynamic fallback based on environment variables or sensible defaults
        allow_origin = (
            origin
            or os.getenv("FRONTEND_URL")
            or os.getenv("CORS_DEFAULT_ORIGIN", "http://localhost:3000")
        )

    headers = {
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Origin, Accept, Accept-Language, Cache-Control, Pragma",
        "Access-Control-Max-Age": "86400",
    }

    # Add credentials header if credentials are allowed
    if allow_credentials:
        headers["Access-Control-Allow-Credentials"] = "true"

    return JSONResponse(
        content={"message": "OK"},
        status_code=200,
        headers=headers,
    )


# WebSocket endpoint for real-time log streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming with enhanced error handling"""
    # Enhanced connection logging with more details
    origin = websocket.headers.get("origin")
    user_agent = websocket.headers.get("user-agent", "Unknown")
    host = websocket.headers.get("host", "Unknown")

    logger.info(
        "WebSocket connection attempt - Origin: %s, Host: %s, User-Agent: %s",
        origin,
        host,
        user_agent,
    )

    # Enhanced CORS validation for WebSocket connections
    cors_validation_passed = False
    if origin:
        if origin in allowed_origins:
            logger.info("WebSocket origin '%s' is allowed by CORS policy", origin)
            cors_validation_passed = True
        # More permissive localhost checking for development
        elif any(
            origin.startswith(prefix)
            for prefix in [
                "http://localhost:",
                "http://127.0.0.1:",
                "ws://localhost:",
                "ws://127.0.0.1:",
            ]
        ):
            logger.info(
                "WebSocket origin '%s' allowed as localhost development connection",
                origin,
            )
            cors_validation_passed = True
        # Docker internal network origins
        elif origin.startswith(
            ("http://dashboard-frontend", "http://dashboard-backend")
        ):
            logger.info(
                "WebSocket origin '%s' allowed as container network connection", origin
            )
            cors_validation_passed = True
        else:
            logger.warning(
                "WebSocket origin '%s' is NOT in allowed origins list", origin
            )
            logger.warning(
                "Allowed origins: %s... (showing first 5)",
                ", ".join(allowed_origins[:5]),
            )

            # In development, still allow connection but log warning
            if os.getenv("ENVIRONMENT", "development") == "development":
                logger.warning(
                    "Allowing connection in development mode despite CORS mismatch"
                )
                cors_validation_passed = True
    else:
        logger.info(
            "WebSocket connection has no origin header (direct client connection or curl/test tool)"
        )
        cors_validation_passed = True  # Allow connections without origin header

    # Log connection details for debugging
    logger.info(
        "WebSocket connection details: Host=%s, User-Agent=%s, CORS=%s",
        host,
        user_agent[:50] if user_agent else "Unknown",
        "PASS" if cors_validation_passed else "FAIL",
    )

    try:
        # Accept connection with detailed logging
        # Prepare connection information for diagnostics
        connection_info = {
            "client_ip": websocket.client.host if websocket.client else "unknown",
            "origin": origin,
            "user_agent": user_agent,
            "host": host,
            "cors_validated": cors_validation_passed,
            "connection_time": datetime.now(UTC).isoformat(),
        }

        await manager.connect(websocket, connection_info=connection_info)
        logger.info("WebSocket connection established successfully from %s", origin)

        # Send initial connection confirmation with enhanced details
        welcome_message = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "connection_established",
            "message": "WebSocket connection successful",
            "connection_id": str(id(websocket)),  # Simple connection identifier
            "server_info": {
                "version": "1.0.0",
                "endpoints": ["/ws", "/api/ws"],
                "features": [
                    "real_time_logs",
                    "llm_monitoring",
                    "trading_data",
                    "ai_decisions",
                    "system_status",
                ],
                "active_connections": len(manager.active_connections),
                "cors_status": "validated" if cors_validation_passed else "warning",
            },
            "client_info": {
                "ip": connection_info["client_ip"],
                "origin": connection_info["origin"],
                "connected_at": connection_info["connection_time"],
            },
        }
        await websocket.send_text(json.dumps(welcome_message))

        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            logger.debug(
                "Received WebSocket message: %s%s",
                data[:100],
                "..." if len(data) > 100 else "",
            )

            # Parse and validate client message
            try:
                parsed_data = (
                    json.loads(data) if data.strip().startswith("{") else {"raw": data}
                )
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON from client: %s", e)
                parsed_data = {"raw": data, "error": "invalid_json"}

            # Handle specific message types
            message_type = parsed_data.get("type", "unknown")

            # Echo back or handle client commands with proper error handling
            try:
                if message_type == "ping":
                    response = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "type": "pong",
                        "message": "Server is alive",
                    }
                elif message_type == "subscribe":
                    # Handle subscription requests
                    response = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "type": "subscription_confirmed",
                        "subscriptions": parsed_data.get("channels", []),
                        "message": f"Subscribed to {len(parsed_data.get('channels', []))} channels",
                    }
                else:
                    # Default echo response
                    response = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "type": "echo",
                        "message": f"Server received: {message_type}",
                        "parsed": parsed_data,
                    }

                await websocket.send_text(json.dumps(response))

            except Exception as e:
                logger.exception("Failed to send WebSocket response")
                # Try to send a simple error message
                try:
                    error_response = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "type": "error",
                        "message": "Failed to process message",
                        "error_details": str(e),
                    }
                    await websocket.send_text(json.dumps(error_response))
                except Exception:
                    # If we can't send error response, break the connection
                    logger.exception("Cannot send error response, breaking connection")
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected gracefully from %s", origin)
        manager.disconnect(websocket)
    except Exception as e:
        logger.exception("WebSocket error from %s", origin)
        manager.disconnect(websocket)
        # Try to send error information back to client
        try:
            if websocket.client_state == websocket.CONNECTED:
                error_msg = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "connection_error",
                    "message": "WebSocket connection error occurred",
                    "error": str(e),
                }
                await websocket.send_text(json.dumps(error_msg))
        except Exception:
            logger.exception(
                "Failed to send error message to closed WebSocket connection"
            )


# REST API Endpoints


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "AI Trading Bot Dashboard",
        "version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "running",
    }


@app.get("/status")
async def get_status():
    """Get bot health check and status information"""
    try:
        # Check if Docker container is running with error handling
        container_status = "unknown"
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=ai-trading-bot",
                    "--format",
                    "{{.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                status_text = result.stdout.strip()
                if "Up" in status_text:
                    container_status = "running"
                elif "Exited" in status_text:
                    container_status = "stopped"
            else:
                container_status = "not_found"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning("Failed to check container status: %s", e)
            container_status = "unavailable"

        # Get basic system info with graceful fallback
        system_uptime = "unknown"
        try:
            uptime_result = await asyncio.to_thread(
                subprocess.run,
                ["uptime"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if uptime_result.returncode == 0:
                system_uptime = uptime_result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning("Failed to get system uptime: %s", e)
            # Provide a simple alternative if psutil is available
            if psutil:
                try:
                    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=UTC)
                    uptime_seconds = (datetime.now(UTC) - boot_time).total_seconds()
                    hours = int(uptime_seconds // 3600)
                    minutes = int((uptime_seconds % 3600) // 60)
                    system_uptime = f"up {hours}:{minutes:02d}"
                except Exception:
                    system_uptime = "unknown"

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "bot_status": container_status,
            "system_uptime": system_uptime,
            "websocket_connections": len(manager.active_connections),
            "log_buffer_size": len(manager.message_buffers.get("log", [])),
            "log_streamer_running": log_streamer.running,
        }

    except Exception as e:
        logger.exception("Error getting status")
        # Return a minimal but functional response instead of raising
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "bot_status": "error",
            "system_uptime": "unknown",
            "websocket_connections": len(manager.active_connections),
            "log_buffer_size": len(manager.message_buffers.get("log", [])),
            "log_streamer_running": log_streamer.running,
            "error": str(e),
        }


@app.get("/trading-data")
async def get_trading_data():
    """Get current trading information and metrics"""
    try:
        # Try to get container stats with graceful fallback
        cpu_usage = "N/A"
        memory_usage = "N/A"

        try:
            stats_result = await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "stats",
                    "ai-trading-bot",
                    "--no-stream",
                    "--format",
                    "table {{.CPUPerc}}\t{{.MemUsage}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            if stats_result.returncode == 0:
                lines = stats_result.stdout.strip().split("\n")
                if len(lines) > 1:  # Skip header
                    stats = lines[1].split("\t")
                    if len(stats) >= 2:
                        cpu_usage = stats[0]
                        memory_usage = stats[1]
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning("Failed to get container stats: %s", e)
            # Try to get system stats as fallback
            if psutil:
                try:
                    cpu_usage = f"{psutil.cpu_percent(interval=0.1):.1f}%"
                    memory = psutil.virtual_memory()
                    memory_usage = f"{memory.used / (1024**3):.1f}GiB / {memory.total / (1024**3):.1f}GiB"
                except Exception:
                    logger.exception("Failed to get system stats from psutil")

        # Determine trading mode from environment or config
        # In real implementation, this would come from the bot's config
        futures_enabled = os.getenv("TRADING__ENABLE_FUTURES", "true").lower() == "true"
        exchange_type = os.getenv("EXCHANGE__EXCHANGE_TYPE", "coinbase").lower()
        trading_mode = "futures" if futures_enabled else "spot"
        account_type = "CFM" if futures_enabled else "CBI"

        # Try to get real Bluefin data if using Bluefin exchange
        bluefin_data = {}
        if exchange_type == "bluefin":
            try:
                # Check Bluefin service health first
                health = await bluefin_client.health_check()
                if health.get("status") == "healthy":
                    # Get real account data
                    try:
                        account_info = await bluefin_client.get_account_info()
                        positions = await bluefin_client.get_positions()
                        orders = await bluefin_client.get_orders()

                        bluefin_data = {
                            "account_info": account_info,
                            "positions": positions.get("positions", []),
                            "orders": orders.get("orders", []),
                            "service_status": "connected",
                        }
                    except Exception as e:
                        logger.warning("Could not fetch Bluefin data: %s", e)
                        bluefin_data = {
                            "service_status": "data_fetch_failed",
                            "error": str(e),
                        }
                else:
                    bluefin_data = {
                        "service_status": "unhealthy",
                        "error": health.get("error", "Service unhealthy"),
                    }
            except Exception as e:
                logger.exception("Bluefin service error")
                bluefin_data = {"service_status": "unreachable", "error": str(e)}

        # Use real Bluefin data if available, otherwise use mock data
        if (
            exchange_type == "bluefin"
            and bluefin_data.get("service_status") == "connected"
        ):
            account_info = bluefin_data.get("account_info", {})
            positions = bluefin_data.get("positions", [])
            orders = bluefin_data.get("orders", [])

            # Extract real position data
            current_position = {}
            if positions:
                # Get the first position as current (in real app, might need to filter)
                pos = positions[0] if isinstance(positions, list) else positions
                current_position = {
                    "symbol": pos.get("symbol", "BTC-PERP"),
                    "side": pos.get("side", "flat"),
                    "size": str(pos.get("size", "0.0")),
                    "entry_price": str(pos.get("entry_price", "0.0")),
                    "current_price": str(pos.get("mark_price", "0.0")),
                    "unrealized_pnl": str(pos.get("unrealized_pnl", "0.0")),
                    "leverage": pos.get("leverage", 1),
                    "liquidation_price": str(pos.get("liquidation_price", "0.0")),
                    "margin_used": str(pos.get("margin", "0.0")),
                }
            else:
                current_position = {
                    "symbol": "BTC-PERP",
                    "side": "flat",
                    "size": "0.0",
                    "entry_price": "0.0",
                    "current_price": "0.0",
                    "unrealized_pnl": "0.0",
                    "leverage": 1,
                    "liquidation_price": "0.0",
                    "margin_used": "0.0",
                }

            # Extract account data
            account_data = {
                "balance": str(account_info.get("balance", "0.0")),
                "currency": "USDC",  # Bluefin uses USDC
                "available_balance": str(account_info.get("available_balance", "0.0")),
                "unrealized_pnl": str(account_info.get("unrealized_pnl", "0.0")),
                "margin_balance": str(account_info.get("margin_balance", "0.0")),
                "margin_ratio": f"{account_info.get('margin_ratio', 0):.1f}%",
                "available_margin": str(account_info.get("available_margin", "0.0")),
                "network": account_info.get("network", "mainnet"),
                "address": account_info.get("address", "unknown"),
            }

            # Recent trades from orders
            recent_trades = []
            for order in orders[:5]:  # Last 5 orders
                if order.get("status") == "filled":
                    recent_trades.append(
                        {
                            "timestamp": order.get(
                                "created_at", datetime.now(UTC).isoformat()
                            ),
                            "symbol": order.get("symbol", "BTC-PERP"),
                            "side": order.get("side", "buy"),
                            "quantity": str(order.get("size", "0.0")),
                            "price": str(order.get("price", "0.0")),
                            "status": "filled",
                            "trade_type": "futures",
                            "leverage": order.get("leverage", 1),
                            "order_id": order.get("id", "unknown"),
                        }
                    )

        else:
            # Mock trading data - fallback when Bluefin not available or using Coinbase
            symbol_suffix = "-PERP" if exchange_type == "bluefin" else "-USD"
            current_position = {
                "symbol": f"BTC{symbol_suffix}",
                "side": "long",
                "size": "0.05",
                "entry_price": "65000.00",
                "current_price": "65500.00",
                "unrealized_pnl": "25.00",
                "leverage": 5 if futures_enabled else None,
                "contracts": 1 if futures_enabled else None,
                "liquidation_price": "58500.00" if futures_enabled else None,
                "margin_used": "650.00" if futures_enabled else None,
            }

            account_currency = "USDC" if exchange_type == "bluefin" else "USD"
            account_data = {
                "balance": "10000.00",
                "currency": account_currency,
                "available_balance": "8500.00",
                "unrealized_pnl": "150.00",
                "margin_balance": "10000.00" if futures_enabled else None,
                "margin_ratio": "15.0%" if futures_enabled else None,
                "available_margin": "8500.00" if futures_enabled else None,
            }

            recent_trades = [
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "symbol": f"BTC{symbol_suffix}",
                    "side": "buy",
                    "quantity": "0.05",
                    "price": "65000.00",
                    "status": "filled",
                    "trade_type": trading_mode,
                    "leverage": 5 if futures_enabled else None,
                }
            ]

        # Mock trading data - in real implementation, this would come from the bot
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "trading_mode": trading_mode,
            "futures_enabled": futures_enabled,
            "exchange_type": exchange_type,
            "account_type": account_type,
            "leverage_available": futures_enabled,
            "account": account_data,
            "current_position": current_position,
            "recent_trades": recent_trades,
            "performance": {
                "total_trades": 42,
                "winning_trades": 28,
                "losing_trades": 14,
                "win_rate": "66.67%",
                "total_pnl": "1250.00",
            },
            "system": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "last_update": datetime.now(UTC).isoformat(),
            },
            "bluefin_data": bluefin_data if exchange_type == "bluefin" else None,
        }

    except Exception as e:
        logger.exception("Error getting trading data")
        # Return mock data even on error to keep dashboard functional
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "trading_mode": "spot",  # Default to spot mode on error
            "futures_enabled": False,
            "account_type": "CBI",
            "leverage_available": False,
            "account": {
                "balance": "0.00",
                "currency": "USD",
                "available_balance": "0.00",
                "unrealized_pnl": "0.00",
            },
            "current_position": {
                "symbol": "BTC-USD",
                "side": "flat",
                "size": "0.00",
                "entry_price": "0.00",
                "current_price": "0.00",
                "unrealized_pnl": "0.00",
            },
            "recent_trades": [],
            "performance": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.00%",
                "total_pnl": "0.00",
            },
            "system": {
                "cpu_usage": "N/A",
                "memory_usage": "N/A",
                "last_update": datetime.now(UTC).isoformat(),
            },
            "error": str(e),
        }


@app.get("/trading-mode")
async def get_trading_mode():
    """Get current trading mode configuration"""
    try:
        # Read from environment or config
        futures_enabled = os.getenv("TRADING__ENABLE_FUTURES", "true").lower() == "true"
        exchange_type = os.getenv("EXCHANGE__EXCHANGE_TYPE", "coinbase").lower()
        leverage = int(os.getenv("TRADING__LEVERAGE", "5"))
        max_futures_leverage = int(os.getenv("TRADING__MAX_FUTURES_LEVERAGE", "20"))

        # Get fee rates
        spot_maker_fee = float(os.getenv("TRADING__SPOT_MAKER_FEE_RATE", "0.006"))
        spot_taker_fee = float(os.getenv("TRADING__SPOT_TAKER_FEE_RATE", "0.012"))
        futures_fee = float(os.getenv("TRADING__FUTURES_FEE_RATE", "0.0015"))

        # Default to basic tier for spot trading
        current_volume = 0  # In real implementation, this would come from exchange API
        fee_tier = "Basic"

        # Exchange-specific configuration
        exchange_config = {}
        bluefin_health = None

        if exchange_type == "bluefin":
            # Get Bluefin service health and configuration
            try:
                bluefin_health = await bluefin_client.health_check()
                if bluefin_health.get("status") == "healthy":
                    # Get additional Bluefin-specific info if service is healthy
                    try:
                        account_info = await bluefin_client.get_account_info()
                        exchange_config = {
                            "service_url": bluefin_client.base_url,
                            "service_status": "connected",
                            "network": account_info.get("network", "mainnet"),
                            "account_address": account_info.get("address", "unknown"),
                        }
                    except Exception as e:
                        logger.warning("Could not get Bluefin account info: %s", e)
                        exchange_config = {
                            "service_url": bluefin_client.base_url,
                            "service_status": "service_healthy_but_no_account",
                            "error": str(e),
                        }
                else:
                    exchange_config = {
                        "service_url": bluefin_client.base_url,
                        "service_status": "unhealthy",
                        "error": bluefin_health.get("error", "Unknown error"),
                    }
            except Exception as e:
                logger.exception("Error checking Bluefin service")
                exchange_config = {
                    "service_url": bluefin_client.base_url,
                    "service_status": "unreachable",
                    "error": str(e),
                }

        # Bluefin-specific trading features
        if exchange_type == "bluefin":
            supported_symbols = {
                "spot": [],  # Bluefin is DEX focused on perpetuals
                "futures": ["BTC-PERP", "ETH-PERP", "SOL-PERP", "SUI-PERP"],
            }
        else:
            supported_symbols = {
                "spot": (
                    ["BTC-USD", "ETH-USD", "SOL-USD"] if not futures_enabled else []
                ),
                "futures": ["BTC-USD", "ETH-USD"] if futures_enabled else [],
            }

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "trading_mode": "futures" if futures_enabled else "spot",
            "futures_enabled": futures_enabled,
            "exchange_type": exchange_type,
            "account_type": "CFM" if futures_enabled else "CBI",
            "leverage_available": futures_enabled,
            "default_leverage": leverage if futures_enabled else None,
            "max_leverage": max_futures_leverage if futures_enabled else None,
            "features": {
                "margin_trading": futures_enabled,
                "stop_loss": True,
                "take_profit": True,
                "contracts": futures_enabled,
                "liquidation_price": futures_enabled,
                "dex_trading": exchange_type == "bluefin",
                "sui_blockchain": exchange_type == "bluefin",
            },
            "supported_symbols": supported_symbols,
            # Fee information
            "spot_maker_fee_rate": spot_maker_fee,
            "spot_taker_fee_rate": spot_taker_fee,
            "futures_fee_rate": futures_fee,
            "current_volume": current_volume,
            "fee_tier": fee_tier,
            # Exchange-specific configuration
            "exchange_config": exchange_config,
            "bluefin_health": bluefin_health,
            # Legacy names for compatibility
            "mode": "futures" if futures_enabled else "spot",
            "maker_fee_rate": spot_maker_fee if not futures_enabled else futures_fee,
            "taker_fee_rate": spot_taker_fee if not futures_enabled else futures_fee,
        }
    except Exception as e:
        logger.exception("Error getting trading mode")
        raise HTTPException(
            status_code=500, detail=f"Error getting trading mode: {e!s}"
        ) from e


@app.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent logs from buffer"""
    try:
        logs = (
            manager.message_buffers.get("log", [])[-limit:]
            if manager.message_buffers.get("log")
            else []
        )
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_logs": len(manager.message_buffers.get("log", [])),
            "returned_logs": len(logs),
            "logs": logs,
        }
    except Exception as e:
        logger.exception("Error getting logs")
        raise HTTPException(status_code=500, detail=f"Error getting logs: {e!s}") from e


# LLM Monitoring Endpoints


@app.get("/llm/status")
async def get_llm_status():
    """Get LLM completion monitoring status and summary"""
    try:
        from datetime import timedelta

        # Gracefully handle missing LLM parser or log files
        if not llm_parser:
            logger.warning("LLM parser not available")
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "monitoring_active": False,
                "log_file": "not_available",
                "total_parsed": {
                    "requests": 0,
                    "responses": 0,
                    "decisions": 0,
                    "alerts": 0,
                    "performance_metrics": 0,
                },
                "metrics": {
                    "1_hour": {},
                    "24_hours": {},
                    "all_time": {},
                },
                "active_alerts": 0,
                "recent_activity": {
                    "decisions_last_hour": 0,
                    "decision_distribution": {},
                    "last_decision": None,
                },
                "error": "LLM monitoring not available",
            }

        # Get aggregated metrics for different time windows with error handling
        try:
            metrics_1h = llm_parser.get_aggregated_metrics(timedelta(hours=1))
            metrics_24h = llm_parser.get_aggregated_metrics(timedelta(hours=24))
            metrics_all = llm_parser.get_aggregated_metrics()
        except Exception as e:
            logger.warning("Failed to get aggregated metrics: %s", e)
            metrics_1h = metrics_24h = metrics_all = {}

        # Calculate derived metrics from available data with safety checks
        try:
            (len(llm_parser.decisions) if hasattr(llm_parser, "decisions") else 0)
            recent_decisions = []
            if hasattr(llm_parser, "decisions") and llm_parser.decisions:
                recent_decisions = [
                    d
                    for d in llm_parser.decisions
                    if d.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                ]

            # Calculate decision distribution
            decision_distribution = {}
            if hasattr(llm_parser, "decisions") and llm_parser.decisions:
                for decision in llm_parser.decisions:
                    action = getattr(decision, "action", "unknown")
                    decision_distribution[action] = (
                        decision_distribution.get(action, 0) + 1
                    )
        except Exception as e:
            logger.warning("Failed to calculate decision metrics: %s", e)
            recent_decisions = []
            decision_distribution = {}

        # Get last decision safely
        last_decision = None
        try:
            if hasattr(llm_parser, "decisions") and llm_parser.decisions:
                last_decision = llm_parser.decisions[-1].to_dict()
        except Exception as e:
            logger.warning("Failed to get last decision: %s", e)

        # Get active alerts safely
        active_alerts_count = 0
        try:
            if hasattr(llm_parser, "alerts") and llm_parser.alerts:
                active_alerts_count = len(
                    [
                        a
                        for a in llm_parser.alerts
                        if a.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                    ]
                )
        except Exception as e:
            logger.warning("Failed to count active alerts: %s", e)

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "monitoring_active": True,
            "log_file": str(getattr(llm_parser, "log_file", "unknown")),
            "total_parsed": {
                "requests": len(getattr(llm_parser, "requests", [])),
                "responses": len(getattr(llm_parser, "responses", [])),
                "decisions": len(getattr(llm_parser, "decisions", [])),
                "alerts": len(getattr(llm_parser, "alerts", [])),
                "performance_metrics": len(getattr(llm_parser, "metrics", [])),
            },
            "metrics": {
                "1_hour": metrics_1h,
                "24_hours": metrics_24h,
                "all_time": metrics_all,
            },
            "active_alerts": active_alerts_count,
            "recent_activity": {
                "decisions_last_hour": len(recent_decisions),
                "decision_distribution": decision_distribution,
                "last_decision": last_decision,
            },
        }

    except Exception as e:
        logger.exception("Error getting LLM status")
        # Return a functional response instead of raising an exception
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "monitoring_active": False,
            "log_file": "error",
            "total_parsed": {
                "requests": 0,
                "responses": 0,
                "decisions": 0,
                "alerts": 0,
                "performance_metrics": 0,
            },
            "metrics": {
                "1_hour": {},
                "24_hours": {},
                "all_time": {},
            },
            "active_alerts": 0,
            "recent_activity": {
                "decisions_last_hour": 0,
                "decision_distribution": {},
                "last_decision": None,
            },
            "error": str(e),
        }


@app.get("/llm/metrics")
async def get_llm_metrics(
    time_window: str | None = Query(
        None, description="Time window: 1h, 24h, 7d, or all"
    ),
):
    """Get detailed LLM performance metrics"""
    try:
        from datetime import timedelta

        # Parse time window
        window_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "all": None,
        }

        time_delta = window_map.get(time_window, timedelta(hours=24))

        # Get metrics with error handling
        metrics = {}
        if llm_parser:
            try:
                metrics = llm_parser.get_aggregated_metrics(time_delta)
            except Exception as e:
                logger.warning("Failed to get aggregated metrics: %s", e)
                metrics = {
                    "total_requests": 0,
                    "total_responses": 0,
                    "success_rate": 0.0,
                    "avg_response_time_ms": 0.0,
                    "total_cost_usd": 0.0,
                    "error": str(e),
                }
        else:
            metrics = {
                "total_requests": 0,
                "total_responses": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "total_cost_usd": 0.0,
                "error": "LLM parser not available",
            }

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "time_window": time_window or "24h",
            "metrics": metrics,
        }

    except Exception as e:
        logger.exception("Error getting LLM metrics")
        # Return empty metrics instead of raising
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "time_window": time_window or "24h",
            "metrics": {
                "total_requests": 0,
                "total_responses": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "total_cost_usd": 0.0,
                "error": str(e),
            },
        }


@app.get("/llm/activity")
async def get_llm_activity(limit: int = Query(50, ge=1, le=500)):
    """Get recent LLM activity across all event types"""
    try:
        activity = []
        if llm_parser:
            try:
                activity = llm_parser.get_recent_activity(limit)
            except Exception as e:
                logger.warning("Failed to get recent activity: %s", e)
                # Return empty activity list instead of failing
                activity = []
        else:
            logger.warning("LLM parser not available for activity")

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "limit": limit,
            "total_events": len(activity),
            "activity": activity,
        }

    except Exception as e:
        logger.exception("Error getting LLM activity")
        # Return empty activity instead of raising
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "limit": limit,
            "total_events": 0,
            "activity": [],
            "error": str(e),
        }


@app.get("/llm/decisions")
async def get_llm_decisions(
    limit: int = Query(50, ge=1, le=500),
    action_filter: str | None = Query(
        None, description="Filter by action: LONG, SHORT, CLOSE, HOLD"
    ),
):
    """Get recent LLM trading decisions"""
    try:
        decisions = llm_parser.decisions

        # Apply action filter if specified
        if action_filter:
            decisions = [d for d in decisions if d.action == action_filter.upper()]

        # Get most recent decisions
        recent_decisions = decisions[-limit:] if len(decisions) > limit else decisions
        recent_decisions.reverse()  # Most recent first

        # Calculate statistics
        action_counts = {}
        for decision in decisions:
            action = decision.action
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_decisions": len(decisions),
            "returned_decisions": len(recent_decisions),
            "action_distribution": action_counts,
            "decisions": [d.to_dict() for d in recent_decisions],
        }

    except Exception as e:
        logger.exception("Error getting LLM decisions")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM decisions: {e!s}"
        ) from e


@app.get("/llm/alerts")
async def get_llm_alerts(
    active_only: bool = Query(False, description="Return only recent alerts"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get LLM monitoring alerts"""
    try:
        from datetime import timedelta

        alerts = llm_parser.alerts

        if active_only:
            # Only alerts from last hour
            cutoff = datetime.now(UTC) - timedelta(hours=1)
            alerts = [a for a in alerts if a.timestamp >= cutoff]

        # Apply limit
        alerts = alerts[-limit:] if len(alerts) > limit else alerts

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "active_only": active_only,
            "total_alerts": len(alerts),
            "alerts": [alert.to_dict() for alert in alerts],
        }

    except Exception as e:
        logger.exception("Error getting LLM alerts")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM alerts: {e!s}"
        ) from e


@app.get("/llm/sessions")
async def get_llm_sessions():
    """Get LLM session information and statistics"""
    try:
        from collections import defaultdict

        # Group by session ID
        sessions = defaultdict(
            lambda: {"requests": [], "responses": [], "decisions": []}
        )

        for req in llm_parser.requests:
            sessions[req.session_id]["requests"].append(req.to_dict())

        for resp in llm_parser.responses:
            sessions[resp.session_id]["responses"].append(resp.to_dict())

        for decision in llm_parser.decisions:
            sessions[decision.session_id]["decisions"].append(decision.to_dict())

        # Calculate session stats
        session_stats = []
        for session_id, data in sessions.items():
            requests = data["requests"]
            responses = data["responses"]
            decisions = data["decisions"]

            if requests:
                start_time = min(req["timestamp"] for req in requests)
                end_time = max(req["timestamp"] for req in requests)

                session_stats.append(
                    {
                        "session_id": session_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "total_requests": len(requests),
                        "total_responses": len(responses),
                        "total_decisions": len(decisions),
                        "success_rate": (
                            sum(1 for r in responses if r["success"]) / len(responses)
                            if responses
                            else 0
                        ),
                        "avg_response_time_ms": (
                            sum(r["response_time_ms"] for r in responses)
                            / len(responses)
                            if responses
                            else 0
                        ),
                    }
                )

        # Sort by start time (most recent first)
        session_stats.sort(key=lambda x: x["start_time"], reverse=True)

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_sessions": len(session_stats),
            "sessions": session_stats,
        }

    except Exception as e:
        logger.exception("Error getting LLM sessions")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM sessions: {e!s}"
        ) from e


@app.get("/llm/cost-analysis")
async def get_llm_cost_analysis():
    """Get detailed cost analysis and projections"""
    try:
        from collections import defaultdict

        # Calculate costs by time period
        costs_by_hour = defaultdict(float)
        costs_by_day = defaultdict(float)
        costs_by_model = defaultdict(float)

        for response in llm_parser.responses:
            if response.cost_estimate_usd > 0:
                hour_key = response.timestamp.strftime("%Y-%m-%d-%H")
                day_key = response.timestamp.strftime("%Y-%m-%d")

                costs_by_hour[hour_key] += response.cost_estimate_usd
                costs_by_day[day_key] += response.cost_estimate_usd

        # Get model info from requests
        for request in llm_parser.requests:
            # Find matching response
            matching_responses = [
                r for r in llm_parser.responses if r.request_id == request.request_id
            ]
            if matching_responses:
                cost = matching_responses[0].cost_estimate_usd
                costs_by_model[request.model] += cost

        # Calculate current hour and day costs
        now = datetime.now(UTC)
        current_hour_key = now.strftime("%Y-%m-%d-%H")
        current_day_key = now.strftime("%Y-%m-%d")

        # Project daily and monthly costs
        hourly_costs = list(costs_by_hour.values())[-24:]  # Last 24 hours
        avg_hourly_cost = sum(hourly_costs) / len(hourly_costs) if hourly_costs else 0

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "current_hour_cost": costs_by_hour.get(current_hour_key, 0),
            "current_day_cost": costs_by_day.get(current_day_key, 0),
            "total_cost": sum(r.cost_estimate_usd for r in llm_parser.responses),
            "avg_hourly_cost": avg_hourly_cost,
            "projected_daily_cost": avg_hourly_cost * 24,
            "projected_monthly_cost": avg_hourly_cost * 24 * 30,
            "costs_by_model": dict(costs_by_model),
            "hourly_breakdown": dict(
                list(costs_by_hour.items())[-24:]
            ),  # Last 24 hours
            "daily_breakdown": dict(list(costs_by_day.items())[-7:]),  # Last 7 days
        }

    except Exception as e:
        logger.exception("Error getting LLM cost analysis")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM cost analysis: {e!s}"
        ) from e


@app.get("/llm/export")
async def export_llm_data(
    export_format: str = Query("json", description="Export format: json"),
):
    """Export all LLM data for analysis"""
    try:
        if export_format.lower() != "json":
            raise HTTPException(
                status_code=400, detail="Only JSON format is currently supported"
            )

        exported_data = llm_parser.export_data(export_format)

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "format": export_format,
            "data": json.loads(exported_data),
        }

    except Exception as e:
        logger.exception("Error exporting LLM data")
        raise HTTPException(
            status_code=500, detail=f"Error exporting LLM data: {e!s}"
        ) from e


@app.post("/llm/alerts/configure")
async def configure_llm_alerts(thresholds: dict[str, Any]):
    """Configure LLM alert thresholds"""
    try:
        # Update alert thresholds
        if "max_response_time_ms" in thresholds:
            llm_parser.alert_thresholds.max_response_time_ms = thresholds[
                "max_response_time_ms"
            ]
        if "max_cost_per_hour" in thresholds:
            llm_parser.alert_thresholds.max_cost_per_hour = thresholds[
                "max_cost_per_hour"
            ]
        if "min_success_rate" in thresholds:
            llm_parser.alert_thresholds.min_success_rate = thresholds[
                "min_success_rate"
            ]
        if "max_consecutive_failures" in thresholds:
            llm_parser.alert_thresholds.max_consecutive_failures = thresholds[
                "max_consecutive_failures"
            ]

        return {
            "status": "success",
            "message": "Alert thresholds updated",
            "current_thresholds": {
                "max_response_time_ms": llm_parser.alert_thresholds.max_response_time_ms,
                "max_cost_per_hour": llm_parser.alert_thresholds.max_cost_per_hour,
                "min_success_rate": llm_parser.alert_thresholds.min_success_rate,
                "max_consecutive_failures": llm_parser.alert_thresholds.max_consecutive_failures,
            },
        }

    except Exception as e:
        logger.exception("Error configuring LLM alerts")
        raise HTTPException(
            status_code=500, detail=f"Error configuring alerts: {e!s}"
        ) from e


@app.post("/control/restart")
async def restart_bot():
    """Restart the trading bot container"""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "restart", "ai-trading-bot"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0:
            # Broadcast restart notification
            await manager.broadcast(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": "INFO",
                    "message": "Trading bot container restarted",
                    "source": "dashboard-api",
                }
            )

            return {"status": "success", "message": "Bot restarted successfully"}
        raise HTTPException(
            status_code=500, detail=f"Failed to restart bot: {result.stderr}"
        )

    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail="Timeout restarting bot") from e
    except Exception as e:
        logger.exception("Error restarting bot")
        raise HTTPException(
            status_code=500, detail=f"Error restarting bot: {e!s}"
        ) from e


# Enhanced Message Management and Bot Control APIs


@app.get("/api/messages/{category}")
async def get_messages_by_category(
    category: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """
    Get messages from a specific category with pagination.

    Categories: trading, indicator, system, log, ai
    """
    if category not in manager.message_buffers:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. Available: {list(manager.message_buffers.keys())}",
        )

    messages = manager.get_messages_by_category(category, limit + offset)
    paginated_messages = messages[offset : offset + limit] if messages else []

    return {
        "category": category,
        "messages": paginated_messages,
        "total_available": len(messages),
        "returned": len(paginated_messages),
        "offset": offset,
        "limit": limit,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/messages/stats")
async def get_message_statistics():
    """Get comprehensive message buffer statistics."""
    stats = manager.get_message_stats()
    return {
        **stats,
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_seconds": time.time() - (stats.get("start_time", time.time())),
    }


@app.post("/api/messages/broadcast")
async def broadcast_message(message: dict):
    """
    Broadcast a custom message to all connected WebSocket clients.

    Useful for testing and manual notifications.
    """
    try:
        # Validate message structure
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Message must be a JSON object")

        # Add source and validation
        enhanced_message = {
            **message,
            "source": "dashboard-api",
            "manual_broadcast": True,
            "api_timestamp": datetime.now(UTC).isoformat(),
        }

        await manager.broadcast(enhanced_message)

        return {
            "status": "success",
            "message": "Message broadcasted successfully",
            "active_connections": len(manager.active_connections),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Error broadcasting message")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Bot Control Command Queue System


class BotCommand:
    """Represents a command to be sent to the trading bot."""

    def __init__(
        self, command_type: str, parameters: dict | None = None, priority: int = 5
    ):
        self.id = f"cmd_{int(time.time() * 1000000)}"
        self.command_type = command_type
        self.parameters = parameters or {}
        self.priority = priority  # 1=highest, 10=lowest
        self.created_at = datetime.now(UTC).isoformat()
        self.status = "pending"  # pending, sent, acknowledged, failed
        self.attempts = 0
        self.max_attempts = 3


# Global command queue
bot_command_queue = []
command_history = []


@app.post("/api/bot/commands/emergency-stop")
async def emergency_stop_bot():
    """
    Send emergency stop command to trading bot.
    Highest priority command that should halt all trading immediately.
    """
    try:
        command = BotCommand("emergency_stop", priority=1)
        bot_command_queue.insert(0, command)  # Insert at front for highest priority

        # Broadcast emergency notification
        emergency_message = {
            "type": "emergency_stop",
            "command_id": command.id,
            "message": "Emergency stop command issued",
            "source": "dashboard-api",
            "priority": "critical",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await manager.broadcast(emergency_message)

        return {
            "status": "success",
            "command_id": command.id,
            "message": "Emergency stop command queued",
            "queue_position": 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Error issuing emergency stop")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/bot/commands/pause-trading")
async def pause_trading():
    """Pause trading operations without stopping the bot."""
    try:
        command = BotCommand("pause_trading", priority=2)
        bot_command_queue.append(command)

        await manager.broadcast(
            {
                "type": "trading_pause",
                "command_id": command.id,
                "message": "Trading pause command issued",
                "source": "dashboard-api",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "status": "success",
            "command_id": command.id,
            "message": "Trading pause command queued",
            "queue_size": len(bot_command_queue),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Error pausing trading")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/bot/commands/resume-trading")
async def resume_trading():
    """Resume trading operations."""
    try:
        command = BotCommand("resume_trading", priority=2)
        bot_command_queue.append(command)

        await manager.broadcast(
            {
                "type": "trading_resume",
                "command_id": command.id,
                "message": "Trading resume command issued",
                "source": "dashboard-api",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "status": "success",
            "command_id": command.id,
            "message": "Trading resume command queued",
            "queue_size": len(bot_command_queue),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Error resuming trading")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/bot/commands/update-risk-limits")
async def update_risk_limits(
    max_position_size: float = Query(None, ge=0.01, le=100.0),
    stop_loss_percentage: float = Query(None, ge=0.1, le=50.0),
    max_daily_loss: float = Query(None, ge=1.0, le=10000.0),
):
    """Update risk management parameters."""
    try:
        parameters = {}
        if max_position_size is not None:
            parameters["max_position_size"] = max_position_size
        if stop_loss_percentage is not None:
            parameters["stop_loss_percentage"] = stop_loss_percentage
        if max_daily_loss is not None:
            parameters["max_daily_loss"] = max_daily_loss

        if not parameters:
            raise HTTPException(status_code=400, detail="No risk parameters provided")

        command = BotCommand("update_risk_limits", parameters, priority=3)
        bot_command_queue.append(command)

        await manager.broadcast(
            {
                "type": "risk_limits_update",
                "command_id": command.id,
                "parameters": parameters,
                "message": "Risk limits update command issued",
                "source": "dashboard-api",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "status": "success",
            "command_id": command.id,
            "message": "Risk limits update command queued",
            "parameters": parameters,
            "queue_size": len(bot_command_queue),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Error updating risk limits")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/bot/commands/manual-trade")
async def manual_trade_command(
    action: str = Query(..., pattern="^(buy|sell|close)$"),
    symbol: str = Query(..., min_length=3, max_length=20),
    size_percentage: float = Query(..., ge=0.1, le=100.0),
    reason: str = Query(default="Manual trade from dashboard"),
):
    """
    Issue a manual trading command.

    WARNING: This will execute real trades if bot is in live mode!
    """
    try:
        # Get current exchange type to customize parameters
        exchange_type = os.getenv("EXCHANGE__EXCHANGE_TYPE", "coinbase").lower()

        parameters = {
            "action": action,
            "symbol": symbol,
            "size_percentage": size_percentage,
            "reason": reason,
            "manual_trade": True,
            "exchange_type": exchange_type,
        }

        # Add Bluefin-specific parameters if using Bluefin
        if exchange_type == "bluefin":
            # Check if Bluefin service is available
            try:
                health = await bluefin_client.health_check()
                if health.get("status") != "healthy":
                    raise HTTPException(
                        status_code=503,
                        detail=f"Bluefin service is not healthy: {health.get('error', 'Unknown error')}",
                    )
                parameters["bluefin_service_available"] = True
            except Exception as e:
                logger.exception("Bluefin service check failed")
                parameters["bluefin_service_available"] = False
                parameters["bluefin_error"] = str(e)

        command = BotCommand("manual_trade", parameters, priority=4)
        bot_command_queue.append(command)

        await manager.broadcast(
            {
                "type": "manual_trade_command",
                "command_id": command.id,
                "parameters": parameters,
                "message": f"Manual trade command issued: {action} {size_percentage}% {symbol} on {exchange_type}",
                "source": "dashboard-api",
                "warning": "This may execute real trades!",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "status": "success",
            "command_id": command.id,
            "message": f"Manual trade command queued: {action} {size_percentage}% {symbol} on {exchange_type}",
            "parameters": parameters,
            "queue_size": len(bot_command_queue),
            "warning": "This command may execute real trades if bot is in live mode!",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Error issuing manual trade")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/bot/commands/queue")
async def get_command_queue():
    """Get current command queue status."""
    return {
        "queue_size": len(bot_command_queue),
        "pending_commands": [
            {
                "id": cmd.id,
                "type": cmd.command_type,
                "priority": cmd.priority,
                "created_at": cmd.created_at,
                "status": cmd.status,
                "attempts": cmd.attempts,
            }
            for cmd in bot_command_queue
        ],
        "history_size": len(command_history),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/bot/commands/history")
async def get_command_history(limit: int = Query(default=50, ge=1, le=500)):
    """Get command execution history."""
    return {
        "history": command_history[-limit:],
        "total_executed": len(command_history),
        "returned": min(limit, len(command_history)),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.delete("/api/bot/commands/{command_id}")
async def cancel_command(command_id: str):
    """Cancel a pending command by ID."""
    try:
        # Find and remove command from queue
        for i, cmd in enumerate(bot_command_queue):
            if cmd.id == command_id:
                removed_cmd = bot_command_queue.pop(i)

                # Add to history as cancelled
                command_history.append(
                    {
                        "id": removed_cmd.id,
                        "type": removed_cmd.command_type,
                        "status": "cancelled",
                        "cancelled_at": datetime.now(UTC).isoformat(),
                        "created_at": removed_cmd.created_at,
                    }
                )

                await manager.broadcast(
                    {
                        "type": "command_cancelled",
                        "command_id": command_id,
                        "message": f"Command {command_id} cancelled",
                        "source": "dashboard-api",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

                return {
                    "status": "success",
                    "message": f"Command {command_id} cancelled successfully",
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        raise HTTPException(
            status_code=404, detail=f"Command {command_id} not found in queue"
        )

    except Exception as e:
        logger.exception("Error cancelling command")
        raise HTTPException(status_code=500, detail=str(e)) from e


# API Routes (prefixed with /api for consistency)
# These routes provide the same functionality as the root-level routes
# but with API prefix for better organization and frontend consistency


@app.get("/api/status")
async def get_api_status():
    """API endpoint for bot status"""
    return await get_status()


@app.get("/api/health")
async def get_api_health():
    """API endpoint for health check"""
    return await health_check()


@app.get("/api/trading-data")
async def get_api_trading_data():
    """API endpoint for trading data"""
    return await get_trading_data()


@app.get("/api/trading-mode")
async def get_api_trading_mode():
    """API endpoint for trading mode"""
    return await get_trading_mode()


@app.get("/api/logs")
async def get_api_logs(limit: int = 100):
    """API endpoint for logs"""
    return await get_logs(limit)


@app.get("/api/llm/status")
async def get_api_llm_status():
    """API endpoint for LLM status"""
    return await get_llm_status()


@app.get("/api/llm/metrics")
async def get_api_llm_metrics(
    time_window: str | None = Query(
        None, description="Time window: 1h, 24h, 7d, or all"
    ),
):
    """API endpoint for LLM metrics"""
    return await get_llm_metrics(time_window)


@app.get("/api/llm/activity")
async def get_api_llm_activity(limit: int = Query(50, ge=1, le=500)):
    """API endpoint for LLM activity"""
    return await get_llm_activity(limit)


@app.post("/api/control/restart")
async def get_api_restart_bot():
    """API endpoint for bot restart"""
    return await restart_bot()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
        "websocket": {
            "endpoint": "/ws",
            "active_connections": len(manager.active_connections),
            "ready": True,
        },
        "cors": {
            "enabled": True,
            "allowed_origins_count": len(allowed_origins),
            "credentials_allowed": allow_credentials,
        },
    }


@app.get("/ws/health")
async def websocket_health_check():
    """WebSocket-specific health check endpoint"""
    return {
        "status": "healthy",
        "websocket_ready": True,
        "endpoint": "/ws",
        "active_connections": len(manager.active_connections),
        "connection_manager_status": "operational",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/ws/test", include_in_schema=False)
async def websocket_test():
    """WebSocket connectivity test endpoint - disabled in production"""
    # Only allow in development environment
    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production":
        raise HTTPException(status_code=404, detail="Not found")

    return {
        "websocket_available": True,
        "endpoint": "/ws",
        "alternative_endpoint": "/api/ws",
        "connection_manager": {
            "active_connections": len(manager.active_connections),
            "log_buffer_size": len(manager.message_buffers.get("log", [])),
        },
        "protocols": ["ws", "wss"],
        "cors_enabled": True,
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": environment,
    }


@app.get("/connectivity/test", include_in_schema=False)
async def connectivity_test(request: Request):
    """Network connectivity diagnostics endpoint"""
    # Get request details
    client_host = request.client.host if request.client else "unknown"
    origin = request.headers.get("origin", "no-origin")
    user_agent = request.headers.get("user-agent", "unknown")

    # Check CORS status
    cors_status = "allowed" if origin in allowed_origins else "not-in-list"
    if origin.startswith(
        ("http://localhost:", "http://127.0.0.1:", "ws://localhost:", "ws://127.0.0.1:")
    ):
        cors_status = "localhost-allowed"
    elif origin.startswith(("http://dashboard-frontend", "http://dashboard-backend")):
        cors_status = "container-network-allowed"

    # Network information
    network_info = {
        "client_ip": client_host,
        "origin": origin,
        "user_agent": user_agent[:100] if user_agent else None,
        "host_header": request.headers.get("host"),
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
    }

    # CORS configuration
    cors_info = {
        "status": cors_status,
        "allowed_origins_count": len(allowed_origins),
        "credentials_allowed": allow_credentials,
        "sample_allowed_origins": allowed_origins[:10],  # Show first 10 for debugging
    }

    # WebSocket status
    websocket_info = {
        "endpoint": "/ws",
        "active_connections": len(manager.active_connections),
        "suggested_url": f"ws://{request.headers.get('host', 'localhost:8000')}/ws",
    }

    # Backend health
    backend_health = {
        "server_time": datetime.now(UTC).isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "docker_env": bool(os.getenv("DOCKER_ENV")),
        "container": bool(os.getenv("CONTAINER")),
    }

    return {
        "status": "ok",
        "message": "Backend connectivity test successful",
        "network": network_info,
        "cors": cors_info,
        "websocket": websocket_info,
        "backend": backend_health,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/ws/connections", include_in_schema=False)
async def websocket_connections():
    """Get current WebSocket connection status and diagnostics"""
    # Only allow in development environment
    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production":
        raise HTTPException(status_code=404, detail="Not found")

    connections_info = []
    for websocket, metadata in manager.connection_metadata.items():
        # Check if connection is still active
        is_active = websocket in manager.active_connections

        connection_data = {
            "connection_id": str(id(websocket)),
            "is_active": is_active,
            "connected_at": metadata.get("connected_at"),
            "last_activity": metadata.get("last_activity"),
            "messages_sent": metadata.get("messages_sent", 0),
            "messages_received": metadata.get("messages_received", 0),
            "client_ip": metadata.get("client_ip", "unknown"),
            "origin": metadata.get("origin", "unknown"),
            "user_agent": (
                metadata.get("user_agent", "unknown")[:100]
                if metadata.get("user_agent")
                else "unknown"
            ),
            "is_healthy": metadata.get("is_healthy", True),
            "error_count": metadata.get("error_count", 0),
            "categories": metadata.get("categories", []),
        }
        connections_info.append(connection_data)

    return {
        "active_connections": len(manager.active_connections),
        "total_connections_served": manager.stats["connections_served"],
        "total_messages": manager.stats["total_messages"],
        "messages_by_category": manager.stats["messages_by_category"],
        "buffer_status": {
            category: {
                "current_size": len(buffer),
                "max_size": manager.buffer_limits[category],
                "utilization": f"{len(buffer) / manager.buffer_limits[category] * 100:.1f}%",
            }
            for category, buffer in manager.message_buffers.items()
        },
        "connections": connections_info,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# TradingView UDF API Endpoints


@app.get("/udf/config")
async def udf_config():
    """TradingView UDF configuration with explicit type enforcement"""
    config = {
        "supports_search": True,
        "supports_group_request": False,
        "supports_marks": True,
        "supports_timescale_marks": True,
        "supports_time": True,
        "supports_streaming": True,
        "exchanges": [
            {
                "value": "COINBASE",
                "name": "Coinbase",
                "desc": "Coinbase Pro/Advanced Trade",
            }
        ],
        "symbols_types": [{"value": "crypto", "name": "Cryptocurrency"}],
        "supported_resolutions": [
            str(x) for x in ["1", "5", "15", "60", "240", "1D", "1W", "1M"]
        ],
        "currencies": [str(x) for x in ["USD", "EUR", "BTC", "ETH"]],
    }

    # Ensure all values are properly typed - no undefined or null values
    def clean_response(obj):
        if isinstance(obj, dict):
            return {k: clean_response(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [clean_response(item) for item in obj if item is not None]
        if isinstance(obj, str):
            return str(obj)
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, int | float):
            return (
                obj if not (isinstance(obj, float) and math.isnan(obj)) else 0
            )  # Handle NaN
        return obj

    return clean_response(config)


@app.get("/udf/symbols")
async def udf_symbols(symbol: str):
    """Get symbol information with explicit type enforcement"""
    symbol_info = tradingview_feed.get_symbol_info(symbol)
    if not symbol_info:
        raise HTTPException(status_code=404, detail="Symbol not found")

    # Ensure all symbol info values are properly typed
    def clean_symbol_info(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if v is not None:
                    if k in [
                        "name",
                        "description",
                        "type",
                        "session",
                        "timezone",
                        "ticker",
                        "exchange",
                        "data_status",
                        "currency_code",
                        "original_name",
                    ]:
                        cleaned[k] = str(v)
                    elif k in ["minmov", "pricescale", "volume_precision"]:
                        cleaned[k] = (
                            int(v)
                            if isinstance(v, int | float)
                            and not (isinstance(v, float) and math.isnan(v))
                            else 1
                        )
                    elif k in ["has_intraday", "has_daily", "has_weekly_and_monthly"]:
                        cleaned[k] = bool(v)
                    elif k in ["intraday_multipliers", "supported_resolutions"]:
                        cleaned[k] = (
                            [str(item) for item in v] if isinstance(v, list) else []
                        )
                    else:
                        cleaned[k] = v
            return cleaned
        return obj

    return clean_symbol_info(symbol_info)


@app.get("/udf/search")
async def udf_search(query: str, limit: int = Query(10, ge=1, le=100)):
    """Search for symbols"""
    symbols = tradingview_feed.get_symbols_list()

    # Filter symbols based on query
    filtered_symbols = [
        symbol
        for symbol in symbols
        if query.upper() in symbol["symbol"].upper()
        or query.upper() in symbol["description"].upper()
    ]

    # Limit results
    return filtered_symbols[:limit]


@app.get("/udf/history")
async def udf_history(
    symbol: str,
    resolution: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
    countback: int | None = None,
):
    """Get historical data with explicit type enforcement"""
    history_data = tradingview_feed.get_history(
        symbol, resolution, from_, to, countback
    )

    # Ensure all historical data values are properly typed
    def clean_history_data(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if v is not None:
                    if k == "s":
                        cleaned[k] = str(v)  # Status must be string
                    elif k in ["t", "o", "h", "l", "c", "v"]:
                        # Time and OHLCV arrays must be numeric
                        if isinstance(v, list):
                            if k == "t":
                                cleaned[k] = [
                                    (
                                        int(item)
                                        if isinstance(item, int | float)
                                        and not (
                                            isinstance(item, float) and math.isnan(item)
                                        )
                                        else 0
                                    )
                                    for item in v
                                ]
                            else:
                                cleaned[k] = [
                                    (
                                        float(item)
                                        if isinstance(item, int | float)
                                        and not (
                                            isinstance(item, float) and math.isnan(item)
                                        )
                                        else 0.0
                                    )
                                    for item in v
                                ]
                        else:
                            cleaned[k] = v
                    elif k == "errmsg":
                        cleaned[k] = str(v)
                    else:
                        cleaned[k] = v
            return cleaned
        return obj

    return clean_history_data(history_data)


@app.get("/udf/marks")
async def udf_marks(
    symbol: str,
    resolution: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
):
    """Get AI decision marks with explicit type enforcement"""
    marks_data = tradingview_feed.get_marks(symbol, from_, to, resolution)

    # Ensure all marks data values are properly typed
    def clean_marks_data(marks):
        if isinstance(marks, list):
            cleaned_marks = []
            for mark in marks:
                if isinstance(mark, dict):
                    cleaned_mark = {}
                    for k, v in mark.items():
                        if v is not None:
                            if k in ["id", "text", "label", "color", "labelFontColor"]:
                                cleaned_mark[k] = str(v)
                            elif k in ["time", "minSize"]:
                                cleaned_mark[k] = (
                                    int(v)
                                    if isinstance(v, int | float)
                                    and not (isinstance(v, float) and math.isnan(v))
                                    else 0
                                )
                            else:
                                cleaned_mark[k] = v
                    cleaned_marks.append(cleaned_mark)
            return cleaned_marks
        return marks

    return clean_marks_data(marks_data)


@app.get("/udf/timescale_marks")
async def udf_timescale_marks(
    symbol: str,
    resolution: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
):
    """Get timescale marks"""
    return tradingview_feed.get_timescale_marks(symbol, from_, to, resolution)


@app.get("/udf/time")
async def udf_time():
    """Get server time"""
    return int(time.time())


# Additional TradingView API endpoints with LLM integration


@app.get("/tradingview/symbols")
async def get_tradingview_symbols():
    """Get all available symbols for TradingView"""
    return {
        "symbols": tradingview_feed.get_symbols_list(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/tradingview/indicators/{symbol}")
async def get_indicator_data(
    symbol: str,
    indicator: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
):
    """Get technical indicator data"""
    data = tradingview_feed.get_indicator_values(symbol, indicator, from_, to)
    return {
        "symbol": symbol,
        "indicator": indicator,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/tradingview/realtime/{symbol}")
async def get_realtime_data(symbol: str, resolution: str = "1"):
    """Get real-time bar data"""
    bar = tradingview_feed.get_real_time_bar(symbol, resolution)
    if not bar:
        raise HTTPException(status_code=404, detail="No real-time data available")
    return bar


@app.post("/tradingview/update/{symbol}")
async def update_trading_data(symbol: str, data: dict[str, Any]):
    """Update trading data from bot (for real-time updates)"""
    try:
        # Process different types of updates
        if "price_data" in data:
            # Handle OHLCV data updates
            resolution = data.get("resolution", "1")
            if "new_bar" in data:
                tradingview_feed.create_new_bar(symbol, resolution, data["new_bar"])
            elif "bar_update" in data:
                tradingview_feed.update_real_time_bar(
                    symbol, resolution, data["bar_update"]
                )

        if "ai_decision" in data:
            # Handle AI decision updates
            from tradingview_feed import convert_bot_trade_to_ai_decision

            decision = convert_bot_trade_to_ai_decision(data["ai_decision"], symbol)
            if decision:
                tradingview_feed.add_ai_decision(symbol, decision)

        if "indicator" in data:
            # Handle technical indicator updates
            from tradingview_feed import convert_indicator_data_to_technical_indicator

            indicator_name = data["indicator"].get("name", "unknown")
            indicator = convert_indicator_data_to_technical_indicator(
                data["indicator"], indicator_name, symbol
            )
            if indicator:
                tradingview_feed.add_technical_indicator(symbol, indicator)

        # Broadcast update to WebSocket clients
        await manager.broadcast(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "tradingview_update",
                "symbol": symbol,
                "data": data,
            }
        )

        return {"status": "success", "message": "Data updated successfully"}

    except Exception as e:
        logger.exception("Error updating trading data")
        raise HTTPException(
            status_code=500, detail=f"Error updating data: {e!s}"
        ) from e


@app.get("/tradingview/summary")
async def get_tradingview_summary():
    """Get summary of all TradingView data"""
    return tradingview_feed.export_data_summary()


@app.post("/tradingview/llm-decision")
async def add_llm_decision_to_chart(decision_data: dict[str, Any]):
    """Add LLM trading decision to TradingView chart"""
    try:
        from tradingview_feed import convert_bot_trade_to_ai_decision

        symbol = decision_data.get("symbol", "BTC-USD")

        # Convert LLM decision to TradingView marker
        marker = convert_bot_trade_to_ai_decision(decision_data, symbol)

        if marker:
            tradingview_feed.add_ai_decision(symbol, marker)

            # Broadcast to WebSocket clients
            await manager.broadcast(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "tradingview_decision",
                    "symbol": symbol,
                    "decision": marker.__dict__,
                }
            )

            return {"status": "success", "message": "Decision added to chart"}
        raise HTTPException(status_code=400, detail="Invalid decision data")

    except Exception as e:
        logger.exception("Error adding LLM decision to chart")
        raise HTTPException(
            status_code=500, detail=f"Error adding decision: {e!s}"
        ) from e


# WebSocket monitoring endpoints
@app.get("/api/websocket/status")
async def websocket_status():
    """Get WebSocket connection status and statistics"""
    return {
        "active_connections": len(manager.active_connections),
        "buffer_size": len(manager.message_buffers.get("log", [])),
        "log_streamer_running": log_streamer.running if log_streamer else False,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# Bluefin-specific API endpoints
@app.get("/api/bluefin/health")
async def get_bluefin_health():
    """Get Bluefin service health status"""
    try:
        health = await bluefin_client.health_check()
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "service_url": bluefin_client.base_url,
            "health": health,
        }
    except Exception as e:
        logger.exception("Error checking Bluefin health")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/bluefin/account")
async def get_bluefin_account():
    """Get Bluefin account information"""
    try:
        account_info = await bluefin_client.get_account_info()
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "account": account_info,
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions from the client
    except Exception as e:
        logger.exception("Error getting Bluefin account")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/bluefin/positions")
async def get_bluefin_positions():
    """Get current Bluefin positions"""
    try:
        positions = await bluefin_client.get_positions()
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "positions": positions,
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions from the client
    except Exception as e:
        logger.exception("Error getting Bluefin positions")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/bluefin/orders")
async def get_bluefin_orders():
    """Get current Bluefin orders"""
    try:
        orders = await bluefin_client.get_orders()
        return {"timestamp": datetime.now(UTC).isoformat(), "orders": orders}
    except HTTPException:
        raise  # Re-raise HTTP exceptions from the client
    except Exception as e:
        logger.exception("Error getting Bluefin orders")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/bluefin/market/{symbol}")
async def get_bluefin_market_ticker(symbol: str):
    """Get Bluefin market ticker for specific symbol"""
    try:
        ticker = await bluefin_client.get_market_ticker(symbol)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "ticker": ticker,
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions from the client
    except Exception as e:
        logger.exception("Error getting Bluefin market ticker")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/bluefin/orders")
async def place_bluefin_order(order_data: dict):
    """Place an order via Bluefin service"""
    try:
        # Validate required fields
        required_fields = ["symbol", "side", "size", "order_type"]
        for field in required_fields:
            if field not in order_data:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        # Add source information
        order_data["source"] = "dashboard-api"
        order_data["timestamp"] = datetime.now(UTC).isoformat()

        result = await bluefin_client.place_order(order_data)

        # Broadcast order placement
        await manager.broadcast(
            {
                "type": "bluefin_order_placed",
                "order_data": order_data,
                "result": result,
                "source": "dashboard-api",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "success",
            "order": result,
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions from the client
    except Exception as e:
        logger.exception("Error placing Bluefin order")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


if __name__ == "__main__":
    import os

    # Disable reload in container environments for better stability
    is_container = os.getenv("DOCKER_ENV") or os.getenv("CONTAINER")
    enable_reload = not is_container

    # Configure host - use localhost by default, allow override for container deployments
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("DASHBOARD_PORT", "8000"))

    logger.info(
        "Starting uvicorn server with reload=%s on %s:%d", enable_reload, host, port
    )

    uvicorn.run(
        "main:app", host=host, port=port, reload=enable_reload, log_level="info"
    )
