"""Type stubs for aiohttp library."""

from typing import (
    Any, Dict, List, Optional, Union, Type, Callable, Awaitable,
    Mapping, MutableMapping, Iterable, AsyncIterator, overload
)
from types import TracebackType
import ssl
from multidict import CIMultiDict, MultiDict
from yarl import URL
import asyncio

__all__ = [
    "ClientSession",
    "ClientResponse",
    "ClientError",
    "ClientConnectionError",
    "ClientConnectorError",
    "ClientTimeout",
    "TCPConnector",
    "BaseConnector",
    "FormData",
    "ClientWebSocketResponse",
    "WSMsgType",
    "WSMessage",
]

class ClientTimeout:
    total: Optional[float]
    connect: Optional[float]
    sock_connect: Optional[float]
    sock_read: Optional[float]
    
    def __init__(
        self,
        *,
        total: Optional[float] = None,
        connect: Optional[float] = None,
        sock_connect: Optional[float] = None,
        sock_read: Optional[float] = None,
    ) -> None: ...

class ClientError(Exception):
    """Base exception for aiohttp client."""
    ...

class ClientConnectionError(ClientError):
    """Connection error."""
    ...

class ClientConnectorError(ClientConnectionError):
    """Connector error."""
    ...

class ServerDisconnectedError(ClientConnectionError):
    """Server disconnected."""
    ...

class ContentTypeError(ClientError):
    """Invalid content type."""
    ...

class ClientResponseError(ClientError):
    """Response error."""
    request_info: Any
    history: Any
    status: int
    message: str
    headers: Any

class WSMsgType:
    TEXT: int
    BINARY: int
    PING: int
    PONG: int
    CLOSE: int
    ERROR: int
    CLOSED: int

class WSMessage:
    type: WSMsgType
    data: Any
    extra: Any

class ClientWebSocketResponse:
    closed: bool
    close_code: Optional[int]
    
    async def send_str(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def send_json(self, data: Any, dumps: Callable[[Any], str] = ...) -> None: ...
    async def close(self, *, code: int = 1000, message: bytes = b"") -> bool: ...
    async def receive(self) -> WSMessage: ...
    async def receive_str(self) -> str: ...
    async def receive_bytes(self) -> bytes: ...
    async def receive_json(self, loads: Callable[[str], Any] = ...) -> Any: ...
    
    def __aiter__(self) -> AsyncIterator[WSMessage]: ...
    async def __anext__(self) -> WSMessage: ...

class ClientResponse:
    version: Any
    status: int
    reason: Optional[str]
    ok: bool
    method: str
    url: URL
    real_url: URL
    connection: Any
    content: Any
    cookies: Any
    headers: CIMultiDict[str]
    raw_headers: List[tuple[bytes, bytes]]
    links: List[Any]
    content_type: str
    charset: Optional[str]
    history: tuple[ClientResponse, ...]
    request_info: Any
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    
    async def text(self, encoding: Optional[str] = None) -> str: ...
    async def json(
        self,
        *,
        encoding: Optional[str] = None,
        loads: Callable[[str], Any] = ...,
        content_type: Optional[str] = "application/json",
    ) -> Any: ...
    async def read(self) -> bytes: ...
    def release(self) -> None: ...
    def raise_for_status(self) -> None: ...
    
    async def __aenter__(self) -> "ClientResponse": ...
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...

class BaseConnector:
    """Base connector class."""
    closed: bool
    
    async def close(self) -> None: ...
    
    async def __aenter__(self) -> "BaseConnector": ...
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...

class TCPConnector(BaseConnector):
    """TCP connector."""
    
    def __init__(
        self,
        *,
        ssl: Union[bool, ssl.SSLContext] = True,
        verify_ssl: bool = True,
        fingerprint: Optional[bytes] = None,
        use_dns_cache: bool = True,
        ttl_dns_cache: Optional[int] = 10,
        family: int = 0,
        ssl_context: Optional[ssl.SSLContext] = None,
        local_addr: Optional[tuple[str, int]] = None,
        resolver: Optional[Any] = None,
        keepalive_timeout: Optional[float] = None,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None: ...

class FormData:
    """Form data for multipart uploads."""
    
    def __init__(
        self,
        fields: Optional[
            Union[
                Mapping[str, Any],
                Iterable[tuple[str, Any]],
                Iterable[tuple[str, Any, Dict[str, Any]]],
            ]
        ] = None,
        quote_fields: bool = True,
        charset: Optional[str] = None,
    ) -> None: ...
    
    def add_field(
        self,
        name: str,
        value: Any,
        *,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
        content_transfer_encoding: Optional[str] = None,
    ) -> None: ...

class ClientSession:
    """Client session for making HTTP requests."""
    
    closed: bool
    connector: Optional[BaseConnector]
    cookie_jar: Any
    
    def __init__(
        self,
        *,
        base_url: Optional[Union[str, URL]] = None,
        connector: Optional[BaseConnector] = None,
        cookies: Optional[Any] = None,
        headers: Optional[Union[Dict[str, str], CIMultiDict[str]]] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        auth: Optional[Any] = None,
        json_serialize: Callable[[Any], str] = ...,
        version: Any = ...,
        cookie_jar: Optional[Any] = None,
        read_timeout: Optional[float] = None,
        conn_timeout: Optional[float] = None,
        timeout: Union[ClientTimeout, float] = ...,
        raise_for_status: bool = False,
        connector_owner: bool = True,
        auto_decompress: bool = True,
        read_bufsize: int = 2**16,
        trust_env: bool = False,
        trace_configs: Optional[List[Any]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None: ...
    
    @overload
    async def request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Union[Dict[str, str], CIMultiDict[str]]] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        auth: Optional[Any] = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        compress: Optional[str] = None,
        chunked: Optional[bool] = None,
        expect100: bool = False,
        raise_for_status: Optional[bool] = None,
        read_until_eof: bool = True,
        timeout: Union[ClientTimeout, float, None] = ...,
        verify_ssl: Optional[bool] = None,
        fingerprint: Optional[bytes] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        ssl: Optional[Union[bool, ssl.SSLContext]] = None,
        proxy: Optional[str] = None,
        proxy_auth: Optional[Any] = None,
        trace_request_ctx: Optional[Any] = None,
    ) -> ClientResponse: ...
    
    async def get(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    async def post(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    async def put(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    async def patch(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    async def delete(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    async def head(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    async def options(self, url: Union[str, URL], **kwargs: Any) -> ClientResponse: ...
    
    async def ws_connect(
        self,
        url: Union[str, URL],
        *,
        protocols: Iterable[str] = (),
        timeout: float = 10.0,
        receive_timeout: Optional[float] = None,
        auth: Optional[Any] = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: Optional[float] = None,
        origin: Optional[str] = None,
        headers: Optional[Union[Dict[str, str], CIMultiDict[str]]] = None,
        proxy: Optional[str] = None,
        proxy_auth: Optional[Any] = None,
        ssl: Optional[Union[bool, ssl.SSLContext]] = None,
        verify_ssl: Optional[bool] = None,
        fingerprint: Optional[bytes] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        compress: int = 0,
        max_msg_size: int = 4194304,
    ) -> ClientWebSocketResponse: ...
    
    async def close(self) -> None: ...
    
    async def __aenter__(self) -> "ClientSession": ...
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...