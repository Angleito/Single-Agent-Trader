"""Type stubs for coinbase-advanced-py library."""

from datetime import datetime
from typing import Any

class RESTClient:
    """Coinbase Advanced Trade REST API client."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        key_file: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> None: ...
    def get_accounts(
        self, limit: int | None = None, cursor: str | None = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    def get_account(self, account_uuid: str) -> dict[str, Any]: ...
    def create_order(
        self,
        client_order_id: str,
        product_id: str,
        side: str,
        order_configuration: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def cancel_orders(self, order_ids: list[str]) -> dict[str, Any]: ...
    def get_order(self, order_id: str) -> dict[str, Any]: ...
    def list_orders(
        self,
        product_id: str | None = None,
        order_status: list[str] | None = None,
        limit: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def get_fills(
        self,
        order_id: str | None = None,
        product_id: str | None = None,
        limit: int | None = None,
        start_sequence_timestamp: datetime | None = None,
        end_sequence_timestamp: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def get_products(
        self,
        limit: int | None = None,
        offset: int | None = None,
        product_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def get_product(self, product_id: str) -> dict[str, Any]: ...
    def get_product_candles(
        self,
        product_id: str,
        start: str | int,
        end: str | int,
        granularity: str,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def get_market_trades(
        self, product_id: str, limit: int | None = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    def get_best_bid_ask(
        self, product_ids: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    def get_product_book(
        self,
        product_id: str,
        limit: int | None = None,
        aggregation_price_increment: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def preview_order(
        self,
        product_id: str,
        side: str,
        order_configuration: dict[str, Any],
        leverage: str | None = None,
        margin_type: str | None = None,
        retail_portfolio_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def allocate_portfolio(
        self,
        portfolio_uuid: str,
        symbol: str,
        amount: str,
        currency: str,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def get_futures_balance_summary(self) -> dict[str, Any]: ...
    def list_futures_positions(self) -> dict[str, Any]: ...
    def get_futures_position(self, product_id: str) -> dict[str, Any]: ...
    def schedule_futures_sweep(self, usd_amount: str) -> dict[str, Any]: ...
    def list_futures_sweeps(self) -> dict[str, Any]: ...
    def cancel_pending_futures_sweep(self) -> dict[str, Any]: ...

class WebSocketClient:
    """Coinbase Advanced Trade WebSocket client."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        on_message: Any | None = None,
        on_open: Any | None = None,
        on_close: Any | None = None,
        on_error: Any | None = None,
        **kwargs: Any,
    ) -> None: ...
    def subscribe(
        self, product_ids: list[str], channels: list[str], **kwargs: Any
    ) -> None: ...
    def unsubscribe(
        self, product_ids: list[str], channels: list[str], **kwargs: Any
    ) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def run_forever(self) -> None: ...
