"""Type stubs for ccxt library (if used)."""

from typing import Any, Literal

__all__ = ["BaseError", "Exchange", "ExchangeError", "NetworkError", "exchanges"]

# Exceptions
class BaseError(Exception):
    """Base CCXT exception."""

class NetworkError(BaseError):
    """Network-related error."""

class ExchangeError(BaseError):
    """Exchange-specific error."""

class NotSupported(ExchangeError):
    """Method not supported by exchange."""

class InvalidOrder(ExchangeError):
    """Invalid order parameters."""

class InsufficientFunds(ExchangeError):
    """Insufficient funds."""

class OrderNotFound(ExchangeError):
    """Order not found."""

class Exchange:
    """Base exchange class."""

    id: str
    name: str
    countries: list[str]
    version: str
    rateLimit: int
    has: dict[str, bool]
    timeframes: dict[str, str]
    timeout: int
    apiKey: str
    secret: str
    password: str
    uid: str

    def __init__(self, config: dict[str, Any] | None = None) -> None: ...

    # Market Data
    def load_markets(self, reload: bool = False) -> dict[str, Any]: ...
    def fetch_markets(
        self, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...
    def fetch_ticker(
        self, symbol: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def fetch_tickers(
        self, symbols: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]: ...
    def fetch_order_book(
        self,
        symbol: str,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def fetch_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[float]]: ...

    # Trading
    def create_order(
        self,
        symbol: str,
        type: Literal["market", "limit", "stop", "stop_limit"],
        side: Literal["buy", "sell"],
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def cancel_order(
        self, id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def cancel_all_orders(
        self, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...
    def fetch_order(
        self, id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def fetch_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def fetch_closed_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def fetch_my_trades(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    # Account
    def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]: ...
    def fetch_positions(
        self, symbols: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...
    def fetch_position(
        self, symbol: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def set_leverage(
        self,
        leverage: int,
        symbol: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def set_position_mode(
        self,
        hedged: bool,
        symbol: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def set_margin_mode(
        self,
        marginMode: str,
        symbol: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    # Deposits/Withdrawals
    def fetch_deposits(
        self,
        code: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def fetch_withdrawals(
        self,
        code: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def withdraw(
        self,
        code: str,
        amount: float,
        address: str,
        tag: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    # Utilities
    def parse_ticker(
        self, ticker: dict[str, Any], market: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def parse_trade(
        self, trade: dict[str, Any], market: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def parse_order(
        self, order: dict[str, Any], market: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    def parse_ohlcv(
        self, ohlcv: list[Any], market: dict[str, Any] | None = None
    ) -> list[float]: ...
    def parse_balance(self, response: dict[str, Any]) -> dict[str, Any]: ...

    # Time
    def milliseconds(self) -> int: ...
    def seconds(self) -> float: ...
    def iso8601(self, timestamp: int | str | None = None) -> str: ...
    def parse8601(self, datetime_str: str) -> int | None: ...

    # Network
    def fetch(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str | dict[str, Any] | None = None,
    ) -> Any: ...

# Available exchanges
exchanges: list[str]

# Exchange classes
class binance(Exchange):
    """Binance exchange."""

class coinbase(Exchange):
    """Coinbase exchange."""

class coinbasepro(Exchange):
    """Coinbase Pro exchange."""

class kraken(Exchange):
    """Kraken exchange."""

class bitfinex(Exchange):
    """Bitfinex exchange."""

class huobi(Exchange):
    """Huobi exchange."""

class okx(Exchange):
    """OKX exchange."""

class bybit(Exchange):
    """Bybit exchange."""

class kucoin(Exchange):
    """KuCoin exchange."""
