"""Type stubs for aiohttp library."""

import asyncio
import ssl
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from types import TracebackType
from typing import Any, overload

from multidict import CIMultiDict
from yarl import URL

__all__ = [
    "BaseConnector",
    "ClientConnectionError",
    "ClientConnectorError",
    "ClientError",
    "ClientResponse",
    "ClientSession",
    "ClientTimeout",
    "ClientWebSocketResponse",
    "FormData",
    "TCPConnector",
    "WSMessage",
    "WSMsgType",
]

class ClientTimeout:
    total: float | None
    connect: float | None
    sock_connect: float | None
    sock_read: float | None

    def __init__(
        self,
        *,
        total: float | None = None,
        connect: float | None = None,
        sock_connect: float | None = None,
        sock_read: float | None = None,
    ) -> None: ...

class ClientError(Exception):
    """Base exception for aiohttp client."""

class ClientConnectionError(ClientError):
    """Connection error."""

class ClientConnectorError(ClientConnectionError):
    """Connector error."""

class ServerDisconnectedError(ClientConnectionError):
    """Server disconnected."""

class ContentTypeError(ClientError):
    """Invalid content type."""

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
    close_code: int | None

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
    reason: str | None
    ok: bool
    method: str
    url: URL
    real_url: URL
    connection: Any
    content: Any
    cookies: Any
    headers: CIMultiDict[str]
    raw_headers: list[tuple[bytes, bytes]]
    links: list[Any]
    content_type: str
    charset: str | None
    history: tuple[ClientResponse, ...]
    request_info: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    async def text(self, encoding: str | None = None) -> str: ...
    async def json(
        self,
        *,
        encoding: str | None = None,
        loads: Callable[[str], Any] = ...,
        content_type: str | None = "application/json",
    ) -> Any: ...
    async def read(self) -> bytes: ...
    def release(self) -> None: ...
    def raise_for_status(self) -> None: ...
    async def __aenter__(self) -> ClientResponse: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

class BaseConnector:
    """Base connector class."""

    closed: bool

    async def close(self) -> None: ...
    async def __aenter__(self) -> BaseConnector: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

class TCPConnector(BaseConnector):
    """TCP connector."""

    def __init__(
        self,
        *,
        ssl: bool | ssl.SSLContext = True,
        verify_ssl: bool = True,
        fingerprint: bytes | None = None,
        use_dns_cache: bool = True,
        ttl_dns_cache: int | None = 10,
        family: int = 0,
        ssl_context: ssl.SSLContext | None = None,
        local_addr: tuple[str, int] | None = None,
        resolver: Any | None = None,
        keepalive_timeout: float | None = None,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None: ...

class FormData:
    """Form data for multipart uploads."""

    def __init__(
        self,
        fields: (
            Mapping[str, Any]
            | Iterable[tuple[str, Any]]
            | Iterable[tuple[str, Any, dict[str, Any]]]
            | None
        ) = None,
        quote_fields: bool = True,
        charset: str | None = None,
    ) -> None: ...
    def add_field(
        self,
        name: str,
        value: Any,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        content_transfer_encoding: str | None = None,
    ) -> None: ...

class ClientSession:
    """Client session for making HTTP requests."""

    closed: bool
    connector: BaseConnector | None
    cookie_jar: Any

    def __init__(
        self,
        *,
        base_url: str | URL | None = None,
        connector: BaseConnector | None = None,
        cookies: Any | None = None,
        headers: dict[str, str] | CIMultiDict[str] | None = None,
        skip_auto_headers: Iterable[str] | None = None,
        auth: Any | None = None,
        json_serialize: Callable[[Any], str] = ...,
        version: Any = ...,
        cookie_jar: Any | None = None,
        read_timeout: float | None = None,
        conn_timeout: float | None = None,
        timeout: ClientTimeout | float = ...,
        raise_for_status: bool = False,
        connector_owner: bool = True,
        auto_decompress: bool = True,
        read_bufsize: int = 2**16,
        trust_env: bool = False,
        trace_configs: list[Any] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None: ...
    @overload
    async def request(
        self,
        method: str,
        url: str | URL,
        *,
        params: Mapping[str, Any] | None = None,
        data: Any | None = None,
        json: Any | None = None,
        headers: dict[str, str] | CIMultiDict[str] | None = None,
        skip_auto_headers: Iterable[str] | None = None,
        auth: Any | None = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        compress: str | None = None,
        chunked: bool | None = None,
        expect100: bool = False,
        raise_for_status: bool | None = None,
        read_until_eof: bool = True,
        timeout: ClientTimeout | float | None = ...,
        verify_ssl: bool | None = None,
        fingerprint: bytes | None = None,
        ssl_context: ssl.SSLContext | None = None,
        ssl: bool | ssl.SSLContext | None = None,
        proxy: str | None = None,
        proxy_auth: Any | None = None,
        trace_request_ctx: Any | None = None,
    ) -> ClientResponse: ...
    async def get(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def post(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def put(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def patch(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def delete(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def head(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def options(self, url: str | URL, **kwargs: Any) -> ClientResponse: ...
    async def ws_connect(
        self,
        url: str | URL,
        *,
        protocols: Iterable[str] = (),
        timeout: float = 10.0,
        receive_timeout: float | None = None,
        auth: Any | None = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: float | None = None,
        origin: str | None = None,
        headers: dict[str, str] | CIMultiDict[str] | None = None,
        proxy: str | None = None,
        proxy_auth: Any | None = None,
        ssl: bool | ssl.SSLContext | None = None,
        verify_ssl: bool | None = None,
        fingerprint: bytes | None = None,
        ssl_context: ssl.SSLContext | None = None,
        compress: int = 0,
        max_msg_size: int = 4194304,
    ) -> ClientWebSocketResponse: ...
    async def close(self) -> None: ...
    async def __aenter__(self) -> ClientSession: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
