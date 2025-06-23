"""Type stubs for coinbase-advanced-py library."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

class RESTClient:
    """Coinbase Advanced Trade REST API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        key_file: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any
    ) -> None: ...
    
    def get_accounts(
        self,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_account(self, account_uuid: str) -> Dict[str, Any]: ...
    
    def create_order(
        self,
        client_order_id: str,
        product_id: str,
        side: str,
        order_configuration: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]: ...
    
    def get_order(self, order_id: str) -> Dict[str, Any]: ...
    
    def list_orders(
        self,
        product_id: Optional[str] = None,
        order_status: Optional[List[str]] = None,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_fills(
        self,
        order_id: Optional[str] = None,
        product_id: Optional[str] = None,
        limit: Optional[int] = None,
        start_sequence_timestamp: Optional[datetime] = None,
        end_sequence_timestamp: Optional[datetime] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_products(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        product_type: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_product(self, product_id: str) -> Dict[str, Any]: ...
    
    def get_product_candles(
        self,
        product_id: str,
        start: Union[str, int],
        end: Union[str, int],
        granularity: str,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_market_trades(
        self,
        product_id: str,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_best_bid_ask(
        self,
        product_ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_product_book(
        self,
        product_id: str,
        limit: Optional[int] = None,
        aggregation_price_increment: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def preview_order(
        self,
        product_id: str,
        side: str,
        order_configuration: Dict[str, Any],
        leverage: Optional[str] = None,
        margin_type: Optional[str] = None,
        retail_portfolio_id: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def allocate_portfolio(
        self,
        portfolio_uuid: str,
        symbol: str,
        amount: str,
        currency: str,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def get_futures_balance_summary(self) -> Dict[str, Any]: ...
    
    def list_futures_positions(self) -> Dict[str, Any]: ...
    
    def get_futures_position(self, product_id: str) -> Dict[str, Any]: ...
    
    def schedule_futures_sweep(self, usd_amount: str) -> Dict[str, Any]: ...
    
    def list_futures_sweeps(self) -> Dict[str, Any]: ...
    
    def cancel_pending_futures_sweep(self) -> Dict[str, Any]: ...

class WebSocketClient:
    """Coinbase Advanced Trade WebSocket client."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        on_message: Optional[Any] = None,
        on_open: Optional[Any] = None,
        on_close: Optional[Any] = None,
        on_error: Optional[Any] = None,
        **kwargs: Any
    ) -> None: ...
    
    def subscribe(
        self,
        product_ids: List[str],
        channels: List[str],
        **kwargs: Any
    ) -> None: ...
    
    def unsubscribe(
        self,
        product_ids: List[str],
        channels: List[str],
        **kwargs: Any
    ) -> None: ...
    
    def open(self) -> None: ...
    
    def close(self) -> None: ...
    
    def run_forever(self) -> None: ...