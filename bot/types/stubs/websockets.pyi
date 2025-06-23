"""Type stubs for websockets library."""

import asyncio
import ssl
from collections.abc import AsyncIterator, Coroutine
from types import TracebackType
from typing import Any

__all__ = [
    "ConnectionClosed",
    "ConnectionClosedError",
    "ConnectionClosedOK",
    "InvalidHeader",
    "InvalidURI",
    "WebSocketClientProtocol",
    "WebSocketException",
    "connect",
    "serve",
]

class WebSocketException(Exception):
    """Base exception for websockets."""

class ConnectionClosed(WebSocketException):
    """Connection closed exception."""

    code: int
    reason: str

    def __init__(self, code: int, reason: str) -> None: ...

class ConnectionClosedOK(ConnectionClosed):
    """Connection closed cleanly."""

class ConnectionClosedError(ConnectionClosed):
    """Connection closed with an error."""

class InvalidURI(WebSocketException):
    """Invalid URI exception."""

    uri: str

    def __init__(self, uri: str, msg: str = "") -> None: ...

class InvalidHeader(WebSocketException):
    """Invalid header exception."""

    name: str
    value: Any

    def __init__(self, name: str, value: Any) -> None: ...

class WebSocketClientProtocol:
    """WebSocket client protocol."""

    closed: asyncio.Future[None]
    close_code: int | None
    close_reason: str | None

    async def send(self, message: str | bytes) -> None:
        """Send a message."""

    async def recv(self) -> str | bytes:
        """Receive a message."""

    async def ping(self, data: str | bytes | None = None) -> Coroutine[Any, Any, None]:
        """Send a ping."""

    async def pong(self, data: str | bytes = b"") -> None:
        """Send a pong."""

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection."""

    async def wait_closed(self) -> None:
        """Wait until the connection is closed."""

    async def __aenter__(self) -> WebSocketClientProtocol:
        """Async context manager entry."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        """Async iterator."""

    async def __anext__(self) -> str | bytes:
        """Get next message."""

async def connect(
    uri: str,
    *,
    create_protocol: Any | None = None,
    logger: Any | None = None,
    compression: str | None = None,
    origin: str | None = None,
    extensions: Any | None = None,
    subprotocols: Any | None = None,
    extra_headers: Any | None = None,
    user_agent_header: str | None = None,
    open_timeout: float | None = 10,
    ping_interval: float | None = 20,
    ping_timeout: float | None = 20,
    close_timeout: float | None = None,
    max_size: int | None = 2**20,
    max_queue: int | None = 2**5,
    read_limit: int = 2**16,
    write_limit: int = 2**16,
    ssl: bool | ssl.SSLContext | None = None,
    server_hostname: str | None = None,
    **kwargs: Any,
) -> WebSocketClientProtocol:
    """Connect to a WebSocket server."""

async def serve(
    handler: Any,
    host: str | None = None,
    port: int | None = None,
    *,
    create_protocol: Any | None = None,
    logger: Any | None = None,
    compression: str | None = None,
    origins: Any | None = None,
    extensions: Any | None = None,
    subprotocols: Any | None = None,
    extra_headers: Any | None = None,
    server_header: str | None = None,
    process_request: Any | None = None,
    select_subprotocol: Any | None = None,
    ping_interval: float | None = 20,
    ping_timeout: float | None = 20,
    close_timeout: float | None = None,
    max_size: int | None = 2**20,
    max_queue: int | None = 2**5,
    read_limit: int = 2**16,
    write_limit: int = 2**16,
    **kwargs: Any,
) -> Any:
    """Start a WebSocket server."""
