"""Type stubs for websockets library."""

from typing import Any, AsyncIterator, Coroutine, Optional, Type, Union
from types import TracebackType
import ssl
import asyncio

__all__ = [
    "WebSocketException",
    "ConnectionClosed",
    "ConnectionClosedOK",
    "ConnectionClosedError",
    "InvalidURI",
    "InvalidHeader",
    "WebSocketClientProtocol",
    "connect",
    "serve",
]

class WebSocketException(Exception):
    """Base exception for websockets."""
    ...

class ConnectionClosed(WebSocketException):
    """Connection closed exception."""
    code: int
    reason: str
    
    def __init__(self, code: int, reason: str) -> None: ...

class ConnectionClosedOK(ConnectionClosed):
    """Connection closed cleanly."""
    ...

class ConnectionClosedError(ConnectionClosed):
    """Connection closed with an error."""
    ...

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
    close_code: Optional[int]
    close_reason: Optional[str]
    
    async def send(self, message: Union[str, bytes]) -> None:
        """Send a message."""
        ...
    
    async def recv(self) -> Union[str, bytes]:
        """Receive a message."""
        ...
    
    async def ping(self, data: Optional[Union[str, bytes]] = None) -> Coroutine[Any, Any, None]:
        """Send a ping."""
        ...
    
    async def pong(self, data: Union[str, bytes] = b"") -> None:
        """Send a pong."""
        ...
    
    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection."""
        ...
    
    async def wait_closed(self) -> None:
        """Wait until the connection is closed."""
        ...
    
    async def __aenter__(self) -> "WebSocketClientProtocol":
        """Async context manager entry."""
        ...
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        ...
    
    def __aiter__(self) -> AsyncIterator[Union[str, bytes]]:
        """Async iterator."""
        ...
    
    async def __anext__(self) -> Union[str, bytes]:
        """Get next message."""
        ...

async def connect(
    uri: str,
    *,
    create_protocol: Optional[Any] = None,
    logger: Optional[Any] = None,
    compression: Optional[str] = None,
    origin: Optional[str] = None,
    extensions: Optional[Any] = None,
    subprotocols: Optional[Any] = None,
    extra_headers: Optional[Any] = None,
    user_agent_header: Optional[str] = None,
    open_timeout: Optional[float] = 10,
    ping_interval: Optional[float] = 20,
    ping_timeout: Optional[float] = 20,
    close_timeout: Optional[float] = None,
    max_size: Optional[int] = 2**20,
    max_queue: Optional[int] = 2**5,
    read_limit: int = 2**16,
    write_limit: int = 2**16,
    ssl: Optional[Union[bool, ssl.SSLContext]] = None,
    server_hostname: Optional[str] = None,
    **kwargs: Any,
) -> WebSocketClientProtocol:
    """Connect to a WebSocket server."""
    ...

async def serve(
    handler: Any,
    host: Optional[str] = None,
    port: Optional[int] = None,
    *,
    create_protocol: Optional[Any] = None,
    logger: Optional[Any] = None,
    compression: Optional[str] = None,
    origins: Optional[Any] = None,
    extensions: Optional[Any] = None,
    subprotocols: Optional[Any] = None,
    extra_headers: Optional[Any] = None,
    server_header: Optional[str] = None,
    process_request: Optional[Any] = None,
    select_subprotocol: Optional[Any] = None,
    ping_interval: Optional[float] = 20,
    ping_timeout: Optional[float] = 20,
    close_timeout: Optional[float] = None,
    max_size: Optional[int] = 2**20,
    max_queue: Optional[int] = 2**5,
    read_limit: int = 2**16,
    write_limit: int = 2**16,
    **kwargs: Any,
) -> Any:
    """Start a WebSocket server."""
    ...