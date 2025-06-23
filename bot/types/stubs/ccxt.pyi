"""Type stubs for ccxt library (if used)."""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime

__all__ = ["Exchange", "BaseError", "NetworkError", "ExchangeError", "exchanges"]

# Exceptions
class BaseError(Exception):
    """Base CCXT exception."""
    ...

class NetworkError(BaseError):
    """Network-related error."""
    ...

class ExchangeError(BaseError):
    """Exchange-specific error."""
    ...

class NotSupported(ExchangeError):
    """Method not supported by exchange."""
    ...

class InvalidOrder(ExchangeError):
    """Invalid order parameters."""
    ...

class InsufficientFunds(ExchangeError):
    """Insufficient funds."""
    ...

class OrderNotFound(ExchangeError):
    """Order not found."""
    ...

class Exchange:
    """Base exchange class."""
    
    id: str
    name: str
    countries: List[str]
    version: str
    rateLimit: int
    has: Dict[str, bool]
    timeframes: Dict[str, str]
    timeout: int
    apiKey: str
    secret: str
    password: str
    uid: str
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None: ...
    
    # Market Data
    def load_markets(self, reload: bool = False) -> Dict[str, Any]: ...
    def fetch_markets(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def fetch_ticker(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def fetch_tickers(self, symbols: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]: ...
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def fetch_trades(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[List[float]]: ...
    
    # Trading
    def create_order(
        self,
        symbol: str,
        type: Literal["market", "limit", "stop", "stop_limit"],
        side: Literal["buy", "sell"],
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...
    
    def cancel_order(self, id: str, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def cancel_all_orders(self, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def fetch_order(self, id: str, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def fetch_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    def fetch_open_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    def fetch_closed_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    def fetch_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    
    # Account
    def fetch_balance(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def fetch_position(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def set_leverage(self, leverage: int, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def set_position_mode(self, hedged: bool, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def set_margin_mode(self, marginMode: str, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    
    # Deposits/Withdrawals
    def fetch_deposits(
        self,
        code: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    def fetch_withdrawals(
        self,
        code: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    def withdraw(
        self,
        code: str,
        amount: float,
        address: str,
        tag: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...
    
    # Utilities
    def parse_ticker(self, ticker: Dict[str, Any], market: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def parse_trade(self, trade: Dict[str, Any], market: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def parse_order(self, order: Dict[str, Any], market: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def parse_ohlcv(self, ohlcv: List[Any], market: Optional[Dict[str, Any]] = None) -> List[float]: ...
    def parse_balance(self, response: Dict[str, Any]) -> Dict[str, Any]: ...
    
    # Time
    def milliseconds(self) -> int: ...
    def seconds(self) -> float: ...
    def iso8601(self, timestamp: Optional[Union[int, str]] = None) -> str: ...
    def parse8601(self, datetime_str: str) -> Optional[int]: ...
    
    # Network
    def fetch(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Any: ...

# Available exchanges
exchanges: List[str]

# Exchange classes
class binance(Exchange):
    """Binance exchange."""
    ...

class coinbase(Exchange):
    """Coinbase exchange."""
    ...

class coinbasepro(Exchange):
    """Coinbase Pro exchange."""
    ...

class kraken(Exchange):
    """Kraken exchange."""
    ...

class bitfinex(Exchange):
    """Bitfinex exchange."""
    ...

class huobi(Exchange):
    """Huobi exchange."""
    ...

class okx(Exchange):
    """OKX exchange."""
    ...

class bybit(Exchange):
    """Bybit exchange."""
    ...

class kucoin(Exchange):
    """KuCoin exchange."""
    ...