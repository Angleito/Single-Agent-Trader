"""
Bluefin exchange client for perpetual futures trading on Sui network.

This module provides integration with Bluefin DEX, a decentralized perpetual
futures exchange built on the Sui blockchain.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from bluefin_v2_client import BluefinClient, Networks, ORDER_SIDE, ORDER_TYPE
    from bluefin_v2_client.client import OrderSignatureRequest
    BLUEFIN_AVAILABLE = True
except ImportError:
    # Mock classes for when Bluefin SDK is not installed
    class BluefinClient:
        def __init__(self, **kwargs):
            pass
    
    class Networks:
        PRODUCTION = "PRODUCTION"
        TESTNET = "TESTNET"
    
    class ORDER_SIDE:
        BUY = "BUY"
        SELL = "SELL"
    
    class ORDER_TYPE:
        MARKET = "MARKET"
        LIMIT = "LIMIT"
    
    class OrderSignatureRequest:
        def __init__(self, **kwargs):
            pass
    
    BLUEFIN_AVAILABLE = False

from ..config import settings
from ..types import (
    AccountType,
    FuturesAccountInfo,
    MarginHealthStatus,
    MarginInfo,
    Order,
    OrderStatus,
    Position,
    TradeAction,
)
from .base import (
    BaseExchange,
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeInsufficientFundsError,
    ExchangeOrderError,
)

logger = logging.getLogger(__name__)


class BluefinRateLimiter:
    """Rate limiter for Bluefin API requests."""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make an API request."""
        async with self._lock:
            import time
            now = time.time()
            # Remove old requests outside the window
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.window_seconds - (now - oldest_request)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            self.requests.append(now)


class BluefinClient(BaseExchange):
    """
    Bluefin exchange client for perpetual futures trading.
    
    Provides methods for trading perpetual futures on Bluefin DEX,
    including order placement, position management, and market data.
    """
    
    def __init__(
        self,
        private_key: Optional[str] = None,
        network: str = "mainnet",
        dry_run: bool = True,
    ):
        """
        Initialize the Bluefin client.
        
        Args:
            private_key: Sui wallet private key
            network: Network to connect to ('mainnet' or 'testnet')
            dry_run: Whether to run in paper trading mode
        """
        super().__init__(dry_run)
        
        self.private_key = private_key or (
            settings.exchange.bluefin_private_key.get_secret_value() 
            if settings.exchange.bluefin_private_key else None
        )
        self.network = Networks.PRODUCTION if network == "mainnet" else Networks.TESTNET
        
        # Bluefin client
        self._client = None
        self._public_client = None
        
        # Rate limiting
        self._rate_limiter = BluefinRateLimiter(
            max_requests=settings.exchange.rate_limit_requests * 3,  # Bluefin has higher limits
            window_seconds=settings.exchange.rate_limit_window_seconds
        )
        
        # Cached data
        self._account_address = None
        self._leverage_settings = {}
        self._contract_info = {}
        
        logger.info(
            f"Initialized BluefinClient (network={network}, "
            f"dry_run={dry_run}, has_key={bool(self.private_key)})"
        )
    
    async def connect(self) -> bool:
        """
        Connect and authenticate with Bluefin.
        
        Returns:
            True if connection successful
        """
        try:
            if not BLUEFIN_AVAILABLE:
                logger.warning(
                    "bluefin-v2-client is not installed. "
                    "Install it with: pip install bluefin-v2-client"
                )
                if self.dry_run:
                    logger.info(
                        "PAPER TRADING MODE: Bluefin connection skipped (SDK missing). "
                        "All trades will be simulated."
                    )
                    self._connected = False
                    self._last_health_check = datetime.utcnow()
                    return True
                else:
                    raise ExchangeConnectionError(
                        "bluefin-v2-client is required for live trading"
                    )
            
            if not self.private_key and not self.dry_run:
                logger.error(
                    "Missing Bluefin private key. Please set BLUEFIN_PRIVATE_KEY "
                    "environment variable."
                )
                raise ExchangeAuthError("Missing Bluefin private key")
            
            # Initialize clients
            if self.private_key:
                # Authenticated client for trading
                self._client = BluefinClient(
                    True,  # on-chain mode
                    network=self.network,
                    private_key=self.private_key,
                )
                await self._init_client()
            
            # Public client for market data (always needed)
            self._public_client = BluefinClient(
                False,  # read-only mode
                network=self.network,
            )
            
            self._connected = True
            self._last_health_check = datetime.utcnow()
            
            logger.info(
                f"Connected to Bluefin {self.network} successfully"
            )
            
            # Get account info if authenticated
            if self._client:
                self._account_address = await self._get_account_address()
                logger.info(f"Bluefin account: {self._account_address}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Bluefin: {e}")
            raise ExchangeConnectionError(f"Connection failed: {e}") from e
    
    async def _init_client(self) -> None:
        """Initialize the Bluefin client."""
        try:
            # Initialize the client with required setup
            await self._rate_limiter.acquire()
            self._client.init()
            
            # Get and cache contract information
            await self._load_contract_info()
            
        except Exception as e:
            logger.error(f"Failed to initialize Bluefin client: {e}")
            raise
    
    async def _load_contract_info(self) -> None:
        """Load and cache perpetual contract information."""
        try:
            markets = await self._retry_request(self._public_client.get_exchange_info)
            
            for market in markets.get("symbols", []):
                symbol = market["symbol"]
                self._contract_info[symbol] = {
                    "base_asset": market["baseAsset"],
                    "quote_asset": market["quoteAsset"],
                    "min_quantity": Decimal(market["minQty"]),
                    "max_quantity": Decimal(market["maxQty"]),
                    "tick_size": Decimal(market["tickSize"]),
                    "min_notional": Decimal(market.get("minNotional", "10")),
                }
                
            logger.debug(f"Loaded {len(self._contract_info)} perpetual contracts")
            
        except Exception as e:
            logger.warning(f"Failed to load contract info: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Bluefin."""
        if self._client:
            try:
                # Clean up any resources
                logger.info("Disconnecting from Bluefin...")
            except Exception as e:
                logger.warning(f"Error during Bluefin disconnect: {e}")
        
        self._client = None
        self._public_client = None
        self._connected = False
        self._last_health_check = None
        logger.info("Disconnected from Bluefin")
    
    async def execute_trade_action(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Optional[Order]:
        """
        Execute a trade action on Bluefin.
        
        Args:
            trade_action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Order object if successful, None otherwise
        """
        if not self._connected:
            logger.error("Not connected to Bluefin")
            return None
        
        try:
            # Convert symbol to Bluefin format (e.g., ETH-USD -> ETH-PERP)
            bluefin_symbol = self._convert_symbol(symbol)
            
            if trade_action.action == "HOLD":
                logger.info("Action is HOLD - no trade executed")
                return None
            
            elif trade_action.action == "CLOSE":
                return await self._close_position(bluefin_symbol)
            
            elif trade_action.action in ["LONG", "SHORT"]:
                return await self._open_position(
                    trade_action, bluefin_symbol, current_price
                )
            
            else:
                logger.error(f"Unknown action: {trade_action.action}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to execute trade action: {e}")
            return None
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Bluefin perpetual format."""
        # Map common symbols to Bluefin perpetual contracts
        symbol_map = {
            "BTC-USD": "BTC-PERP",
            "ETH-USD": "ETH-PERP",
            "SOL-USD": "SOL-PERP",
            "SUI-USD": "SUI-PERP",
        }
        
        return symbol_map.get(symbol, symbol)
    
    async def _open_position(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Optional[Order]:
        """Open a new perpetual position."""
        try:
            # Get account balance
            account_balance = await self.get_account_balance()
            
            # Calculate position size
            position_value = account_balance * Decimal(str(trade_action.size_pct / 100))
            
            # Apply leverage
            leverage = trade_action.leverage or settings.trading.leverage
            notional_value = position_value * leverage
            
            # Calculate quantity
            quantity = notional_value / current_price
            
            # Round to contract specifications
            contract_info = self._contract_info.get(symbol, {})
            tick_size = contract_info.get("tick_size", Decimal("0.001"))
            quantity = self._round_to_tick(quantity, tick_size)
            
            # Validate quantity
            min_qty = contract_info.get("min_quantity", Decimal("0.001"))
            if quantity < min_qty:
                logger.warning(f"Quantity {quantity} below minimum {min_qty}")
                quantity = min_qty
            
            # Determine order side
            side = ORDER_SIDE.BUY if trade_action.action == "LONG" else ORDER_SIDE.SELL
            
            # Place market order
            order = await self.place_market_order(symbol, side, quantity)
            
            if order:
                # Set leverage for the position
                await self._set_leverage(symbol, leverage)
                
                # Place stop loss and take profit orders
                await self._place_stop_loss(order, trade_action, current_price)
                await self._place_take_profit(order, trade_action, current_price)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None
    
    async def _close_position(self, symbol: str) -> Optional[Order]:
        """Close existing position."""
        try:
            # Get current position
            positions = await self.get_positions(symbol)
            
            if not positions:
                logger.info("No position to close")
                return None
            
            position = positions[0]
            
            # Determine opposite side to close
            side = ORDER_SIDE.SELL if position.side == "LONG" else ORDER_SIDE.BUY
            
            # Place market order to close
            return await self.place_market_order(
                symbol, side, abs(position.size)
            )
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None
    
    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> Optional[Order]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            
        Returns:
            Order object if successful
        """
        if self.dry_run:
            logger.info(
                f"PAPER TRADING: Simulating {side} {quantity} {symbol} at market"
            )
            return Order(
                id=f"paper_bluefin_{int(datetime.utcnow().timestamp() * 1000)}",
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.utcnow(),
                filled_quantity=quantity,
            )
        
        try:
            # Get current market price for signature
            ticker = await self._retry_request(
                self._public_client.get_ticker, symbol=symbol
            )
            current_price = Decimal(ticker["lastPrice"])
            
            # Prepare order signature request
            signature_request = OrderSignatureRequest(
                symbol=symbol,
                price=float(current_price),
                quantity=float(quantity),
                side=side,
                orderType=ORDER_TYPE.MARKET,
            )
            
            # Sign and create order
            signed_order = self._client.create_signed_order(signature_request)
            
            logger.info(f"Placing {side} market order: {quantity} {symbol}")
            
            # Post order
            response = await self._retry_request(
                self._client.post_signed_order, signed_order
            )
            
            # Parse response
            if response.get("status") == "success":
                order_data = response.get("order", {})
                order = Order(
                    id=order_data.get("id", f"bluefin_{datetime.utcnow().timestamp()}"),
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                    status=OrderStatus.PENDING,
                    timestamp=datetime.utcnow(),
                    filled_quantity=Decimal("0"),
                )
                
                logger.info(f"Market order placed successfully: {order.id}")
                return order
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Order placement failed: {error_msg}")
                raise ExchangeOrderError(f"Order placement failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            if "insufficient" in str(e).lower():
                raise ExchangeInsufficientFundsError(f"Insufficient funds: {e}") from e
            else:
                raise ExchangeOrderError(f"Failed to place market order: {e}") from e
    
    async def place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> Optional[Order]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Order object if successful
        """
        if self.dry_run:
            logger.info(
                f"PAPER TRADING: Simulating {side} {quantity} {symbol} limit @ {price}"
            )
            return Order(
                id=f"paper_limit_{int(datetime.utcnow().timestamp() * 1000)}",
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                price=price,
                status=OrderStatus.OPEN,
                timestamp=datetime.utcnow(),
            )
        
        try:
            # Prepare order signature request
            signature_request = OrderSignatureRequest(
                symbol=symbol,
                price=float(price),
                quantity=float(quantity),
                side=side,
                orderType=ORDER_TYPE.LIMIT,
            )
            
            # Sign and create order
            signed_order = self._client.create_signed_order(signature_request)
            
            logger.info(f"Placing {side} limit order: {quantity} {symbol} @ {price}")
            
            # Post order
            response = await self._retry_request(
                self._client.post_signed_order, signed_order
            )
            
            # Parse response
            if response.get("status") == "success":
                order_data = response.get("order", {})
                order = Order(
                    id=order_data.get("id", f"bluefin_limit_{datetime.utcnow().timestamp()}"),
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.OPEN,
                    timestamp=datetime.utcnow(),
                    filled_quantity=Decimal("0"),
                )
                
                logger.info(f"Limit order placed successfully: {order.id}")
                return order
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Limit order placement failed: {error_msg}")
                raise ExchangeOrderError(f"Limit order placement failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise ExchangeOrderError(f"Failed to place limit order: {e}") from e
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get current positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of Position objects
        """
        if self.dry_run:
            return []
        
        try:
            # Get user positions
            positions_data = await self._retry_request(
                self._client.get_user_positions
            )
            
            positions = []
            for pos_data in positions_data:
                pos_symbol = pos_data.get("symbol")
                
                # Skip if symbol filter provided and doesn't match
                if symbol and pos_symbol != self._convert_symbol(symbol):
                    continue
                
                # Parse position data
                size = Decimal(str(pos_data.get("quantity", "0")))
                if size != 0:
                    side = "LONG" if size > 0 else "SHORT"
                    position = Position(
                        symbol=pos_symbol,
                        side=side,
                        size=abs(size),
                        entry_price=Decimal(str(pos_data.get("avgPrice", "0"))),
                        unrealized_pnl=Decimal(str(pos_data.get("unrealizedPnl", "0"))),
                        realized_pnl=Decimal(str(pos_data.get("realizedPnl", "0"))),
                        timestamp=datetime.utcnow(),
                        is_futures=True,
                        leverage=pos_data.get("leverage", 1),
                        margin_used=Decimal(str(pos_data.get("margin", "0"))),
                    )
                    positions.append(position)
            
            logger.debug(f"Retrieved {len(positions)} positions from Bluefin")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_balance(
        self, account_type: Optional[AccountType] = None
    ) -> Decimal:
        """
        Get account balance in USD.
        
        Args:
            account_type: Not used for Bluefin (only one account type)
            
        Returns:
            Account balance in USD
        """
        if self.dry_run:
            # Return mock balance for paper trading
            return Decimal("10000.00")
        
        try:
            # Get account data
            account_data = await self._retry_request(
                self._client.get_user_account_data
            )
            
            # Extract USDC balance (Bluefin uses USDC as margin)
            total_balance = Decimal(str(account_data.get("balance", "0")))
            
            logger.debug(f"Retrieved Bluefin balance: ${total_balance}")
            return total_balance
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return Decimal("0")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(f"PAPER TRADING: Simulating cancel order {order_id}")
            return True
        
        try:
            logger.info(f"Cancelling order: {order_id}")
            
            response = await self._retry_request(
                self._client.cancel_order, order_id=order_id
            )
            
            if response.get("status") == "success":
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Order {order_id} cancellation failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional trading symbol filter
            
        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(f"PAPER TRADING: Simulating cancel all orders")
            return True
        
        try:
            # Convert symbol if provided
            bluefin_symbol = self._convert_symbol(symbol) if symbol else None
            
            # Cancel all orders for symbol or all if no symbol
            if bluefin_symbol:
                response = await self._retry_request(
                    self._client.cancel_all_orders, symbol=bluefin_symbol
                )
            else:
                response = await self._retry_request(
                    self._client.cancel_all_orders
                )
            
            if response.get("status") == "success":
                cancelled_count = response.get("cancelledCount", 0)
                logger.info(f"Successfully cancelled {cancelled_count} orders")
                return True
            else:
                logger.warning("Failed to cancel all orders")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information."""
        return {
            "connected": self._connected,
            "exchange": "Bluefin",
            "network": self.network,
            "account_address": self._account_address,
            "has_private_key": bool(self.private_key),
            "dry_run": self.dry_run,
            "trading_mode": "PAPER TRADING" if self.dry_run else "LIVE TRADING",
            "last_health_check": (
                self._last_health_check.isoformat() if self._last_health_check else None
            ),
            "rate_limit_remaining": self._rate_limiter.max_requests - len(
                self._rate_limiter.requests
            ),
            "is_decentralized": True,
            "blockchain": "Sui",
        }
    
    # Bluefin-specific methods
    
    async def _get_account_address(self) -> Optional[str]:
        """Get the Sui wallet address."""
        try:
            if self._client:
                # Get account address from client
                return self._client.get_public_address()
        except Exception as e:
            logger.error(f"Failed to get account address: {e}")
        return None
    
    async def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            if self.dry_run:
                return True
            
            # Cache leverage setting
            self._leverage_settings[symbol] = leverage
            
            # Set leverage via API
            response = await self._retry_request(
                self._client.adjust_leverage,
                symbol=symbol,
                leverage=leverage
            )
            
            return response.get("status") == "success"
            
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    async def _place_stop_loss(
        self, base_order: Order, trade_action: TradeAction, current_price: Decimal
    ) -> Optional[Order]:
        """Place stop loss order."""
        # Calculate stop loss price
        sl_pct = trade_action.stop_loss_pct / 100
        
        if base_order.side == "BUY":  # Long position
            stop_price = current_price * (1 - Decimal(str(sl_pct)))
            side = ORDER_SIDE.SELL
        else:  # Short position
            stop_price = current_price * (1 + Decimal(str(sl_pct)))
            side = ORDER_SIDE.BUY
        
        # Round to tick size
        contract_info = self._contract_info.get(base_order.symbol, {})
        tick_size = contract_info.get("tick_size", Decimal("0.01"))
        stop_price = self._round_to_tick(stop_price, tick_size)
        
        # For Bluefin, we'll use a stop-limit order
        # Place the limit order slightly below/above the stop for slippage
        slippage = Decimal("0.001")  # 0.1% slippage
        if side == ORDER_SIDE.SELL:
            limit_price = stop_price * (1 - slippage)
        else:
            limit_price = stop_price * (1 + slippage)
        
        # Note: Bluefin may require specific stop order implementation
        # For now, using limit order as placeholder
        return await self.place_limit_order(
            base_order.symbol,
            side,
            base_order.quantity,
            limit_price
        )
    
    async def _place_take_profit(
        self, base_order: Order, trade_action: TradeAction, current_price: Decimal
    ) -> Optional[Order]:
        """Place take profit order."""
        # Calculate take profit price
        tp_pct = trade_action.take_profit_pct / 100
        
        if base_order.side == "BUY":  # Long position
            limit_price = current_price * (1 + Decimal(str(tp_pct)))
            side = ORDER_SIDE.SELL
        else:  # Short position
            limit_price = current_price * (1 - Decimal(str(tp_pct)))
            side = ORDER_SIDE.BUY
        
        # Round to tick size
        contract_info = self._contract_info.get(base_order.symbol, {})
        tick_size = contract_info.get("tick_size", Decimal("0.01"))
        limit_price = self._round_to_tick(limit_price, tick_size)
        
        return await self.place_limit_order(
            base_order.symbol,
            side,
            base_order.quantity,
            limit_price
        )
    
    def _round_to_tick(self, value: Decimal, tick_size: Decimal) -> Decimal:
        """Round a value to the nearest tick size."""
        return (value / tick_size).quantize(Decimal("1")) * tick_size
    
    async def _retry_request(self, func, *args, **kwargs):
        """Execute a request with retry logic."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                await self._rate_limiter.acquire()
                return func(*args, **kwargs)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise
    
    @property
    def supports_futures(self) -> bool:
        """Bluefin only supports perpetual futures."""
        return True
    
    @property
    def is_decentralized(self) -> bool:
        """Bluefin is a decentralized exchange."""
        return True
    
    # Override futures-specific methods since Bluefin only does perps
    async def get_futures_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """All positions on Bluefin are futures positions."""
        return await self.get_positions(symbol)
    
    async def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        leverage: Optional[int] = None,
        reduce_only: bool = False,
    ) -> Optional[Order]:
        """All orders on Bluefin are futures orders."""
        # Set leverage if specified
        if leverage:
            await self._set_leverage(symbol, leverage)
        
        return await self.place_market_order(symbol, side, quantity)