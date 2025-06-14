"""
Coinbase exchange client for trading operations.

This module provides a wrapper around the coinbase-advanced-py SDK
for executing trades, managing orders, and retrieving account information.
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

try:
    # Prefer the new official SDK import path first
    from coinbase.rest import RESTClient as _BaseClient

    # Use standard exceptions since SDK doesn't export specific ones
    class CoinbaseAPIException(Exception):
        pass

    class CoinbaseAuthenticationException(Exception):
        pass

    class CoinbaseConnectionException(Exception):
        pass

    class CoinbaseAdvancedTrader(_BaseClient):
        """Adapter class exposing legacy method names used in this codebase."""

        # Legacy CFM helper names used in the code
        def get_fcm_balance_summary(self, **kwargs):
            return self.get_futures_balance_summary(**kwargs)

        def get_fcm_positions(self, **kwargs):
            return self.list_futures_positions(**kwargs)

        # Override get_accounts to return a dict similar to legacy SDK
        def get_accounts(self, *args, **kwargs):  # type: ignore
            resp = super().get_accounts(*args, **kwargs)
            if isinstance(resp, dict):
                return resp
            # RESTClient returns an object with `.accounts` attribute
            try:
                accounts = resp.accounts  # type: ignore
            except AttributeError:
                return {"accounts": []}
            # Convert each account object to dict if possible
            accounts_list = []
            for acc in accounts:
                if isinstance(acc, dict):
                    accounts_list.append(acc)
                else:
                    accounts_list.append(
                        acc.to_dict() if hasattr(acc, "to_dict") else acc.__dict__
                    )
            return {"accounts": accounts_list}

    COINBASE_AVAILABLE = True

except ImportError:
    # Fallback to the deprecated coinbase-advanced-trader package if present
    try:
        from coinbase_advanced_trader.client import (
            CoinbaseAdvancedTrader,  # type: ignore
        )
        from coinbase_advanced_trader.exceptions import (  # type: ignore
            CoinbaseAPIException,
            CoinbaseAuthenticationException,
            CoinbaseConnectionException,
        )

        COINBASE_AVAILABLE = True
    except ImportError:  # pragma: no cover
        # Mock classes for when *no* Coinbase SDK is installed
        class CoinbaseAdvancedTrader:  # type: ignore
            def __init__(self, **kwargs):
                pass

        class CoinbaseAPIException(Exception):
            pass

        class CoinbaseConnectionException(Exception):
            pass

        class CoinbaseAuthenticationException(Exception):
            pass

        COINBASE_AVAILABLE = False

from ..config import settings
from ..types import (
    AccountType,
    CashTransferRequest,
    FuturesAccountInfo,
    MarginHealthStatus,
    MarginInfo,
    Order,
    OrderStatus,
    Position,
    TradeAction,
)
from .futures_utils import FuturesContractManager

logger = logging.getLogger(__name__)


class CoinbaseRateLimiter:
    """Rate limiter for Coinbase API requests."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make an API request."""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the window
            self.requests = [
                req_time
                for req_time in self.requests
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


class CoinbaseExchangeError(Exception):
    """Base exception for Coinbase exchange errors."""

    pass


class CoinbaseConnectionError(CoinbaseExchangeError):
    """Connection-related errors."""

    pass


class CoinbaseAuthError(CoinbaseExchangeError):
    """Authentication-related errors."""

    pass


class CoinbaseOrderError(CoinbaseExchangeError):
    """Order execution errors."""

    pass


class CoinbaseInsufficientFundsError(CoinbaseExchangeError):
    """Insufficient funds errors."""

    pass


class CoinbaseClient:
    """
    Coinbase exchange client for trading operations.

    Provides methods for placing orders, managing positions,
    and retrieving account/market data from Coinbase.
    Supports both spot trading (CBI) and futures trading (CFM).
    """

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
        cdp_api_key_name: str = None,
        cdp_private_key: str = None,
    ):
        """
        Initialize the Coinbase client.

        Args:
            api_key: Coinbase API key (legacy)
            api_secret: Coinbase API secret (legacy)
            passphrase: Coinbase passphrase (legacy)
            cdp_api_key_name: CDP API key name
            cdp_private_key: CDP private key (PEM format)
        """
        # Determine authentication method by checking provided credentials
        # First check for explicitly provided credentials
        provided_legacy = all([api_key, api_secret, passphrase])
        provided_cdp = all([cdp_api_key_name, cdp_private_key])

        # Get credentials from settings if not explicitly provided
        settings_legacy = all(
            [
                settings.exchange.cb_api_key
                and settings.exchange.cb_api_key.get_secret_value().strip(),
                settings.exchange.cb_api_secret
                and settings.exchange.cb_api_secret.get_secret_value().strip(),
                settings.exchange.cb_passphrase
                and settings.exchange.cb_passphrase.get_secret_value().strip(),
            ]
        )

        settings_cdp = all(
            [
                settings.exchange.cdp_api_key_name
                and settings.exchange.cdp_api_key_name.get_secret_value().strip(),
                settings.exchange.cdp_private_key
                and settings.exchange.cdp_private_key.get_secret_value().strip(),
            ]
        )

        # Determine which authentication method to use
        if provided_legacy or (settings_legacy and not provided_cdp):
            # Use legacy authentication
            self.auth_method = "legacy"
            self.api_key = api_key or settings.exchange.cb_api_key.get_secret_value()
            self.api_secret = (
                api_secret or settings.exchange.cb_api_secret.get_secret_value()
            )
            self.passphrase = (
                passphrase or settings.exchange.cb_passphrase.get_secret_value()
            )
            self.cdp_api_key_name = None
            self.cdp_private_key = None

        elif provided_cdp or settings_cdp:
            # Use CDP authentication
            self.auth_method = "cdp"
            self.api_key = None
            self.api_secret = None
            self.passphrase = None
            self.cdp_api_key_name = (
                cdp_api_key_name
                or settings.exchange.cdp_api_key_name.get_secret_value()
            )
            self.cdp_private_key = (
                cdp_private_key or settings.exchange.cdp_private_key.get_secret_value()
            )
        else:
            # No credentials provided - will work in dry run mode only
            self.auth_method = "none"
            self.api_key = None
            self.api_secret = None
            self.passphrase = None
            self.cdp_api_key_name = None
            self.cdp_private_key = None

        self.sandbox = settings.exchange.cb_sandbox

        # Client will be initialized when needed
        self._client = None
        self._authenticated = False
        self._last_health_check = None

        # Rate limiting
        self._rate_limiter = CoinbaseRateLimiter(
            max_requests=settings.exchange.rate_limit_requests,
            window_seconds=settings.exchange.rate_limit_window_seconds,
        )

        # Retry configuration
        self._max_retries = 3
        self._retry_delay = 1.0
        self._retry_backoff = 2.0

        # Futures trading configuration
        self.enable_futures = settings.trading.enable_futures
        self.futures_account_type = settings.trading.futures_account_type
        self.auto_cash_transfer = settings.trading.auto_cash_transfer
        self.max_futures_leverage = settings.trading.max_futures_leverage

        # Cached futures account info
        self._futures_account_info = None
        self._last_margin_check = None

        # Portfolio management
        self._portfolios = {}
        self._futures_portfolio_id = None
        self._default_portfolio_id = None
        
        # Futures contract management
        self._futures_contract_manager = None

        # Compose a concise init message to avoid line length issues
        trading_mode = "PAPER TRADING" if settings.system.dry_run else "LIVE TRADING"
        init_msg = (
            "Initialized CoinbaseClient (mode=%s auth=%s sandbox=%s futures=%s "
            "acct_type=%s)"
        )
        logger.info(
            init_msg,
            trading_mode,
            self.auth_method,
            self.sandbox,
            self.enable_futures,
            self.futures_account_type,
        )

        # Log detailed connection configuration
        logger.debug("CoinbaseClient Configuration Details:")
        logger.debug(f"  Trading mode: {trading_mode}")
        logger.debug(f"  Authentication method: {self.auth_method}")
        logger.debug(f"  Sandbox mode: {self.sandbox}")
        logger.debug(f"  Futures enabled: {self.enable_futures}")
        logger.debug(f"  Futures account type: {self.futures_account_type}")
        logger.debug(f"  Auto cash transfer: {self.auto_cash_transfer}")
        logger.debug(f"  Max futures leverage: {self.max_futures_leverage}")
        logger.debug(
            f"  Rate limit: {self._rate_limiter.max_requests} req/{self._rate_limiter.window_seconds}s"
        )
        logger.debug(f"  Max retries: {self._max_retries}")
        logger.debug(f"  Has credentials: {bool(self.auth_method != 'none')}")

        # Log warning if in live trading mode
        if not settings.system.dry_run:
            logger.warning(
                "⚠️  LIVE TRADING MODE ENABLED - Real money will be used! "
                "Use --dry-run flag for paper trading."
            )

    async def _load_portfolios(self) -> None:
        """Load and cache portfolio information."""
        try:
            response = await self._retry_request(self._client.get_portfolios)

            if hasattr(response, "portfolios"):
                portfolios = response.portfolios
            else:
                portfolios = response.get("portfolios", [])

            for portfolio in portfolios:
                portfolio_data = (
                    portfolio
                    if isinstance(portfolio, dict)
                    else {
                        "uuid": getattr(portfolio, "uuid", None),
                        "name": getattr(portfolio, "name", None),
                        "type": getattr(portfolio, "type", None),
                    }
                )

                self._portfolios[portfolio_data["uuid"]] = portfolio_data

                # Identify futures portfolio
                if (
                    "futures" in portfolio_data.get("name", "").lower()
                    or portfolio_data.get("type") == "FUTURES"
                ):
                    self._futures_portfolio_id = portfolio_data["uuid"]
                    logger.info(
                        f"Found futures portfolio: {portfolio_data['name']} ({portfolio_data['uuid']})"
                    )
                elif portfolio_data.get("type") == "DEFAULT":
                    self._default_portfolio_id = portfolio_data["uuid"]

        except Exception as e:
            logger.warning(f"Failed to load portfolios: {e}")
            # Use the hardcoded default if we can't load portfolios
            self._default_portfolio_id = "1f3ed8bf-a65c-5022-8258-87ce50c517f6"

    async def connect(self) -> bool:
        """
        Connect and authenticate with Coinbase.

        Returns:
            True if connection successful
        """
        try:
            if not COINBASE_AVAILABLE:
                logger.warning(
                    "coinbase-advanced-py is not installed. "
                    "Install it with: pip install coinbase-advanced-py"
                )
                # In dry-run / paper-trading mode we can operate without the SDK
                if settings.system.dry_run:
                    logger.info(
                        "PAPER TRADING MODE: Coinbase connection skipped (SDK missing). All trades will be simulated."
                    )
                    # Mark as not authenticated but allow engine startup.
                    self._client = None
                    self._authenticated = False
                    self._last_health_check = datetime.utcnow()
                    return True
                if not settings.system.dry_run:
                    raise CoinbaseConnectionError(
                        "coinbase-advanced-py is required for live trading"
                    )

            # Check credentials based on authentication method
            if self.auth_method == "legacy":
                if not all([self.api_key, self.api_secret, self.passphrase]):
                    logger.error(
                        "Missing legacy Coinbase credentials. Please set CB_API_KEY, "
                        "CB_API_SECRET, and CB_PASSPHRASE environment variables."
                    )
                    raise CoinbaseAuthError("Missing legacy API credentials")

                # Initialize with legacy credentials
                self._client = CoinbaseAdvancedTrader(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    passphrase=self.passphrase,
                    sandbox=self.sandbox,
                )

            elif self.auth_method == "cdp":
                if not all([self.cdp_api_key_name, self.cdp_private_key]):
                    logger.error(
                        "Missing CDP credentials. Please set CDP_API_KEY_NAME "
                        "and CDP_PRIVATE_KEY environment variables."
                    )
                    raise CoinbaseAuthError("Missing CDP API credentials")

                # Initialize RESTClient with CDP credentials directly
                # Based on official docs: api_key should be the CDP key name, api_secret should be the private key
                # Note: Modern coinbase-advanced-py SDK handles sandbox vs production based on the API key,
                # not through a separate sandbox parameter
                self._client = CoinbaseAdvancedTrader(
                    api_key=self.cdp_api_key_name, api_secret=self.cdp_private_key
                )

            else:
                logger.error(
                    "No valid authentication method configured. Please provide "
                    "either legacy (CB_API_KEY, CB_API_SECRET, CB_PASSPHRASE) or "
                    "CDP (CDP_API_KEY_NAME, CDP_PRIVATE_KEY) credentials."
                )
                raise CoinbaseAuthError("No authentication credentials provided")

            # Test connection with a simple API call
            await self._test_connection()

            self._authenticated = True
            self._last_health_check = datetime.utcnow()

            logger.info(
                f"Connected to Coinbase {'Sandbox' if self.sandbox else 'Live'} successfully"
            )

            # Load portfolio information
            await self._load_portfolios()

            # Log connection success details
            logger.debug("Connection Success Details:")
            logger.debug(
                f"  Environment: {'Sandbox' if self.sandbox else 'Production'}"
            )
            logger.debug(f"  Authentication method: {self.auth_method}")
            logger.debug(f"  Health check timestamp: {self._last_health_check}")
            logger.debug(f"  SDK available: {COINBASE_AVAILABLE}")

            # Log account access test
            try:
                accounts = await self._retry_request(self._client.get_accounts)
                account_count = len(accounts.get("accounts", []))
                logger.debug(
                    f"  Account access test: SUCCESS ({account_count} accounts found)"
                )
            except Exception as e:
                logger.warning(f"  Account access test: FAILED ({e})")

            # Log futures capabilities if enabled
            if self.enable_futures:
                try:
                    balance_response = await self._retry_request(
                        self._client.get_fcm_balance_summary
                    )
                    logger.debug("  Futures access test: SUCCESS")
                    logger.debug(
                        f"  CFM account ready: {hasattr(balance_response, 'balance_summary')}"
                    )
                except Exception as e:
                    logger.warning(f"  Futures access test: FAILED ({e})")

            # Load portfolios information
            await self._load_portfolios()
            
            # Initialize futures contract manager if futures are enabled
            if self.enable_futures:
                self._futures_contract_manager = FuturesContractManager(self)

            return True

        except CoinbaseAuthenticationException as e:
            logger.error(f"Coinbase authentication failed: {e}")
            raise CoinbaseAuthError(f"Authentication failed: {e}") from e
        except CoinbaseConnectionException as e:
            logger.error(f"Coinbase connection failed: {e}")
            raise CoinbaseConnectionError(f"Connection failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            logger.debug(f"Connection error traceback: {traceback.format_exc()}")
            raise CoinbaseConnectionError(f"Unexpected error: {e}") from e

    async def _test_connection(self) -> None:
        """Test the connection with a simple API call."""
        try:
            await self._rate_limiter.acquire()
            accounts = self._client.get_accounts()
            logger.debug(
                f"Connection test successful, found {len(accounts.get('accounts', []))} accounts"
            )
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Coinbase."""
        if self._client:
            # Cancel any pending orders if needed
            try:
                if self._authenticated:
                    logger.info("Cleaning up before disconnect...")
                    # Could add cleanup logic here
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

        self._client = None
        self._authenticated = False
        self._last_health_check = None
        logger.info("Disconnected from Coinbase")

    async def _health_check(self) -> bool:
        """Perform a health check on the connection."""
        if not self._authenticated:
            return False

        try:
            # Only perform health check if enough time has passed
            if (
                self._last_health_check
                and datetime.utcnow() - self._last_health_check
                < timedelta(seconds=settings.exchange.health_check_interval)
            ):
                return True

            await self._rate_limiter.acquire()
            self._client.get_accounts()
            self._last_health_check = datetime.utcnow()
            return True

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self._authenticated = False
            return False

    async def _retry_request(self, func, *args, **kwargs):
        """Execute a request with retry logic."""
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                # Check connection health before important operations
                if not await self._health_check():
                    logger.warning("Connection unhealthy, attempting to reconnect...")
                    if not await self.connect():
                        raise CoinbaseConnectionError("Failed to reconnect")

                await self._rate_limiter.acquire()
                return func(*args, **kwargs)

            except (CoinbaseConnectionException, CoinbaseAPIException) as e:
                last_exception = e

                if attempt < self._max_retries:
                    wait_time = self._retry_delay * (self._retry_backoff**attempt)
                    logger.warning(
                        "Request failed (attempt %s/%s): %s. Retry in %.2fs...",
                        attempt + 1,
                        self._max_retries + 1,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Request failed after {self._max_retries + 1} attempts: {e}"
                    )
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in request: {e}")
                raise

        if last_exception:
            raise last_exception

    async def _load_portfolios(self) -> None:
        """Load portfolio information and identify futures portfolio."""
        try:
            logger.debug("Loading portfolio information...")

            # Try to get portfolios using the REST API
            try:
                # First try the standard portfolios endpoint
                portfolios_response = await self._retry_request(
                    lambda: self._client.get("/api/v3/brokerage/portfolios")
                )

                if isinstance(portfolios_response, dict):
                    self._portfolios = portfolios_response.get("portfolios", [])
                else:
                    self._portfolios = []

            except Exception as e:
                logger.debug(f"Could not fetch portfolios via API: {e}")
                self._portfolios = []

            # If we have portfolios, try to identify the futures portfolio
            if self._portfolios:
                for portfolio in self._portfolios:
                    portfolio_name = portfolio.get("name", "").lower()
                    portfolio_type = portfolio.get("type", "").lower()
                    portfolio_id = portfolio.get("uuid") or portfolio.get("id")

                    # Check if this is the default portfolio
                    if portfolio.get("is_default"):
                        self._default_portfolio_id = portfolio_id
                        logger.debug(f"Found default portfolio: {portfolio_id}")

                    # Check if this is a futures portfolio
                    if (
                        "futures" in portfolio_name
                        or "cfm" in portfolio_name
                        or portfolio_type == "futures"
                        or portfolio_type == "cfm"
                    ):
                        self._futures_portfolio_id = portfolio_id
                        logger.debug(f"Found futures portfolio: {portfolio_id}")

                logger.info(f"Loaded {len(self._portfolios)} portfolios")
                if self._futures_portfolio_id:
                    logger.info(f"Futures portfolio ID: {self._futures_portfolio_id}")
            else:
                logger.debug("No portfolios found or portfolios API not available")

        except Exception as e:
            logger.warning(f"Failed to load portfolios: {e}")
            self._portfolios = []

    async def get_portfolios(self) -> list[dict[str, Any]]:
        """
        Get list of all portfolios.

        Returns:
            List of portfolio dictionaries
        """
        if self._portfolios is None:
            await self._load_portfolios()
        return self._portfolios or []

    def get_futures_portfolio_id(self) -> str | None:
        """
        Get the futures portfolio ID if available.

        Returns:
            Futures portfolio UUID or None
        """
        return self._futures_portfolio_id

    async def get_trading_symbol(self, base_symbol: str) -> str:
        """
        Get the appropriate trading symbol based on whether futures are enabled.
        
        Args:
            base_symbol: Base symbol like "ETH-USD"
            
        Returns:
            Actual trading symbol (spot or futures contract)
        """
        if not self.enable_futures:
            # Use spot symbol as-is
            return base_symbol
        
        # For Coinbase, ETH-USD and BTC-USD support futures trading directly
        # when the leverage parameter is included in orders
        if base_symbol in ['ETH-USD', 'BTC-USD', 'SOL-USD', 'DOGE-USD', 'LTC-USD', 'BCH-USD']:
            logger.info(f"{base_symbol} supports futures trading with leverage parameter")
            return base_symbol
        
        # For other symbols, try to find dated contracts
        base_currency = base_symbol.split('-')[0]
        
        # Try to get cached contract first
        if self._futures_contract_manager:
            cached_contract = self._futures_contract_manager.get_cached_contract()
            if cached_contract:
                return cached_contract
            
            # Fetch current active contract
            active_contract = await self._futures_contract_manager.get_active_futures_contract(base_currency)
            if active_contract:
                return active_contract
        
        # Fallback to original symbol
        logger.info(f"Using {base_symbol} for futures trading")
        return base_symbol

    async def execute_trade_action(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Execute a trade action on Coinbase.

        Args:
            trade_action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object if successful, None otherwise
        """
        if not self._authenticated:
            logger.error("Not authenticated with Coinbase")
            return None

        try:
            # Get the actual trading symbol (spot or futures contract)
            actual_symbol = await self.get_trading_symbol(symbol)
            logger.info(f"Using trading symbol: {actual_symbol} (requested: {symbol})")
            
            if trade_action.action == "HOLD":
                logger.info("Action is HOLD - no trade executed")
                return None

            elif trade_action.action == "CLOSE":
                return await self._close_position(actual_symbol)

            elif trade_action.action in ["LONG", "SHORT"]:
                return await self._open_position(trade_action, actual_symbol, current_price)

            else:
                logger.error(f"Unknown action: {trade_action.action}")
                return None

        except Exception as e:
            logger.error(f"Failed to execute trade action: {e}")
            return None

    async def _open_position(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Open a new position.

        Args:
            trade_action: Trade action with position details
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object if successful
        """
        try:
            # Check if this is a futures position
            if self.enable_futures:
                return await self._open_futures_position(
                    trade_action, symbol, current_price
                )
            else:
                return await self._open_spot_position(
                    trade_action, symbol, current_price
                )
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None

    async def _open_futures_position(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Open a futures position with proper margin calculations.

        Args:
            trade_action: Trade action with position details
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object if successful
        """
        try:
            # Get futures account info and check margin health
            futures_account = await self.get_futures_account_info()
            if not futures_account:
                logger.error("Cannot open futures position - account info unavailable")
                return None

            # Check margin health before opening position
            if (
                futures_account.margin_info.health_status
                == MarginHealthStatus.LIQUIDATION_RISK
            ):
                logger.warning("Cannot open position - liquidation risk detected")
                return None

            # Calculate position size based on available margin
            available_margin = futures_account.margin_info.available_margin
            
            # For safety, use only 80% of available margin to avoid insufficient funds errors
            usable_margin = available_margin * Decimal("0.8")
            
            position_value = min(
                usable_margin * Decimal(str(trade_action.size_pct / 100)),
                futures_account.max_position_size,
            )

            # Use leverage from trade action or default
            leverage = trade_action.leverage or settings.trading.leverage
            leverage = min(leverage, self.max_futures_leverage)

            # Calculate quantity for futures
            # On Coinbase, 1 ETH futures contract = 0.1 ETH (nano-sized)
            CONTRACT_SIZE = Decimal("0.1")  # 0.1 ETH per contract

            # Check if we're using fixed contract size from config
            if (
                hasattr(settings.trading, "fixed_contract_size")
                and settings.trading.fixed_contract_size
            ):
                # Use fixed number of contracts
                num_contracts = int(settings.trading.fixed_contract_size)
                quantity = CONTRACT_SIZE * num_contracts
            else:
                # Calculate based on position value
                notional_value = position_value * leverage
                quantity_in_eth = notional_value / current_price
                # Convert to number of contracts and round down
                num_contracts = int(quantity_in_eth / CONTRACT_SIZE)
                num_contracts = max(1, num_contracts)  # Minimum 1 contract
                quantity = CONTRACT_SIZE * num_contracts

            logger.info(f"Futures position: {num_contracts} contracts = {quantity} ETH")

            # Calculate actual notional value based on contracts
            actual_notional_value = quantity * current_price

            # Check if we need to transfer cash for margin
            margin_required = actual_notional_value / leverage
            
            # Check CFM balance and transfer if needed
            if futures_account.futures_balance < margin_required:
                transfer_amount = margin_required - futures_account.futures_balance + Decimal("10")  # Add buffer
                logger.info(
                    f"CFM balance ${futures_account.futures_balance} < margin required ${margin_required}. "
                    f"Transferring ${transfer_amount} from CBI to CFM..."
                )
                
                transfer_success = await self.transfer_cash_to_futures(
                    amount=transfer_amount, 
                    reason="AUTO_REBALANCE"
                )
                
                if not transfer_success:
                    logger.error("Failed to transfer funds to CFM. Cannot open futures position.")
                    return None
                    
                # Wait a bit for transfer to settle
                await asyncio.sleep(2)
                
                # Re-check futures account info
                futures_account = await self.get_futures_account_info()
                if not futures_account:
                    logger.error("Cannot verify transfer - account info unavailable")
                    return None
            
            logger.info(
                f"Margin required: ${margin_required}, Available margin: ${available_margin}"
            )

            # Determine order side
            side = "BUY" if trade_action.action == "LONG" else "SELL"

            # Place futures market order
            order = await self.place_futures_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                leverage=leverage,
                reduce_only=(
                    trade_action.reduce_only
                    if hasattr(trade_action, "reduce_only")
                    else False
                ),
            )

            if order:
                # Place stop loss and take profit orders
                await self._place_stop_loss(order, trade_action, current_price)
                await self._place_take_profit(order, trade_action, current_price)

            return order

        except Exception as e:
            logger.error(f"Failed to open futures position: {e}")
            return None

    async def _open_spot_position(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Open a spot position (original logic).

        Args:
            trade_action: Trade action with position details
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object if successful
        """
        # Calculate position size in base currency
        account_balance = await self.get_account_balance(AccountType.CBI)
        position_value = account_balance * Decimal(str(trade_action.size_pct / 100))

        # Calculate quantity based on leverage
        leverage = settings.trading.leverage
        quantity = (position_value * leverage) / current_price

        # Determine order side
        side = "BUY" if trade_action.action == "LONG" else "SELL"

        # Place market order
        order = await self.place_market_order(
            symbol=symbol, side=side, quantity=quantity
        )

        if order:
            # Place stop loss and take profit orders
            await self._place_stop_loss(order, trade_action, current_price)
            await self._place_take_profit(order, trade_action, current_price)

        return order

    async def _close_position(self, symbol: str) -> Order | None:
        """
        Close existing position.

        Args:
            symbol: Trading symbol

        Returns:
            Order object if successful
        """
        # Get current position
        position = await self._get_position(symbol)

        if not position or position.side == "FLAT":
            logger.info("No position to close")
            return None

        # Determine opposite side
        side = "SELL" if position.side == "LONG" else "BUY"

        # Place market order to close
        return await self.place_market_order(
            symbol=symbol, side=side, quantity=abs(position.size)
        )

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> Order | None:
        """
        Place a market order.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in base currency

        Returns:
            Order object if successful
        """
        if settings.system.dry_run:
            # Simulate order in dry-run mode
            logger.info(
                f"PAPER TRADING: Simulating {side} {quantity} {symbol} at market"
            )
            return Order(
                id=f"paper_{int(datetime.utcnow().timestamp() * 1000)}",
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.utcnow(),
                filled_quantity=quantity,
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise CoinbaseOrderError("Quantity must be positive")

            if side not in ["BUY", "SELL"]:
                raise CoinbaseOrderError(f"Invalid side: {side}")

            # Generate a unique client order ID
            import uuid

            client_order_id = f"ai-bot-{uuid.uuid4().hex[:8]}"

            # Round quantity to appropriate precision (6 decimal places for ETH)
            rounded_quantity = round(float(quantity), 6)

            # Place the order using Coinbase Advanced Trader
            order_data = {
                "product_id": symbol,
                "side": side,
                "order_configuration": {
                    "market_market_ioc": {"base_size": str(rounded_quantity)}
                },
            }

            # Add default portfolio ID if available and not futures
            if self._default_portfolio_id and not self.enable_futures:
                order_data["retail_portfolio_id"] = self._default_portfolio_id
                logger.debug(
                    f"Using default portfolio ID: {self._default_portfolio_id}"
                )

            logger.info(f"Placing {side} market order: {quantity} {symbol}")
            result = await self._retry_request(
                self._client.create_order, client_order_id, **order_data
            )

            # Parse the response - handle CDP API response object
            try:
                # CDP API returns a response object with success/success_response
                if hasattr(result, "success") and result.success:
                    # Handle the new SDK response format
                    resp = result.success_response
                    if isinstance(resp, dict):
                        order_id = resp.get('order_id')
                    else:
                        order_id = resp.order_id if hasattr(resp, 'order_id') else None

                    order = Order(
                        id=order_id
                        or f"cb_{int(datetime.utcnow().timestamp() * 1000)}",
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.utcnow(),
                        filled_quantity=Decimal("0"),
                    )

                    logger.info(f"Market order placed successfully: {order_id}")
                    return order
                elif hasattr(result, "order_id"):
                    # Fallback for old format
                    order_id = result.order_id
                else:
                    # Try dictionary access for backward compatibility
                    if isinstance(result, dict) and result.get("success"):
                        order_info = result.get("order", {})
                        order_id = order_info.get("order_id")

                        order = Order(
                            id=order_id
                            or f"cb_{int(datetime.utcnow().timestamp() * 1000)}",
                            symbol=symbol,
                            side=side,
                            type="MARKET",
                            quantity=quantity,
                            status=OrderStatus.PENDING,
                            timestamp=datetime.utcnow(),
                            filled_quantity=Decimal("0"),
                        )

                        logger.info(f"Market order placed successfully: {order_id}")
                        return order
                    else:
                        error_msg = str(result) if result else "Unknown error"
                        logger.error(f"Order placement failed: {error_msg}")
                        raise CoinbaseOrderError(f"Order placement failed: {error_msg}")
            except AttributeError as e:
                logger.error(f"Error parsing order response: {e}")
                logger.error(f"Response type: {type(result)}, Response: {result}")
                raise CoinbaseOrderError(f"Failed to parse order response: {e}") from e

        except CoinbaseAPIException as e:
            if "insufficient funds" in str(e).lower():
                raise CoinbaseInsufficientFundsError(f"Insufficient funds: {e}") from e
            else:
                raise CoinbaseOrderError(f"API error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            logger.debug(f"Market order error traceback: {traceback.format_exc()}")
            raise CoinbaseOrderError(f"Failed to place market order: {e}") from e

    async def place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> Order | None:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in base currency
            price: Limit price

        Returns:
            Order object if successful
        """
        if settings.system.dry_run:
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
            # Validate inputs
            if quantity <= 0:
                raise CoinbaseOrderError("Quantity must be positive")

            if price <= 0:
                raise CoinbaseOrderError("Price must be positive")

            if side not in ["BUY", "SELL"]:
                raise CoinbaseOrderError(f"Invalid side: {side}")

            # Place the limit order
            order_data = {
                "product_id": symbol,
                "side": side,
                "order_configuration": {
                    "limit_limit_gtc": {
                        "base_size": str(quantity),
                        "limit_price": str(price),
                    }
                },
            }

            logger.info(f"Placing {side} limit order: {quantity} {symbol} @ {price}")
            result = await self._retry_request(self._client.create_order, **order_data)

            # Parse the response
            if result.get("success"):
                order_info = result.get("order", {})
                order_id = order_info.get("order_id")

                order = Order(
                    id=order_id
                    or f"cb_limit_{int(datetime.utcnow().timestamp() * 1000)}",
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.OPEN,
                    timestamp=datetime.utcnow(),
                    filled_quantity=Decimal("0"),
                )

                logger.info(f"Limit order placed successfully: {order_id}")
                return order
            else:
                error_msg = result.get("error_response", {}).get(
                    "message", "Unknown error"
                )
                logger.error(f"Limit order placement failed: {error_msg}")
                raise CoinbaseOrderError(f"Limit order placement failed: {error_msg}")

        except CoinbaseAPIException as e:
            if "insufficient funds" in str(e).lower():
                raise CoinbaseInsufficientFundsError(f"Insufficient funds: {e}") from e
            else:
                raise CoinbaseOrderError(f"API error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            logger.debug(f"Limit order error traceback: {traceback.format_exc()}")
            raise CoinbaseOrderError(f"Failed to place limit order: {e}") from e

    async def _place_stop_loss(
        self, base_order: Order, trade_action: TradeAction, current_price: Decimal
    ) -> Order | None:
        """
        Place stop loss order.

        Args:
            base_order: Base order for the position
            trade_action: Trade action with stop loss percentage
            current_price: Current market price

        Returns:
            Stop loss order if successful
        """
        # Calculate stop loss price
        sl_pct = trade_action.stop_loss_pct / 100

        if base_order.side == "BUY":  # Long position
            stop_price = current_price * (1 - Decimal(str(sl_pct)))
            side = "SELL"
        else:  # Short position
            stop_price = current_price * (1 + Decimal(str(sl_pct)))
            side = "BUY"

        return await self._place_stop_order(
            symbol=base_order.symbol,
            side=side,
            quantity=base_order.quantity,
            stop_price=stop_price,
        )

    async def _place_take_profit(
        self, base_order: Order, trade_action: TradeAction, current_price: Decimal
    ) -> Order | None:
        """
        Place take profit order.

        Args:
            base_order: Base order for the position
            trade_action: Trade action with take profit percentage
            current_price: Current market price

        Returns:
            Take profit order if successful
        """
        # Calculate take profit price
        tp_pct = trade_action.take_profit_pct / 100

        if base_order.side == "BUY":  # Long position
            limit_price = current_price * (1 + Decimal(str(tp_pct)))
            side = "SELL"
        else:  # Short position
            limit_price = current_price * (1 - Decimal(str(tp_pct)))
            side = "BUY"

        return await self.place_limit_order(
            symbol=base_order.symbol,
            side=side,
            quantity=base_order.quantity,
            price=limit_price,
        )

    async def _place_stop_order(
        self, symbol: str, side: str, quantity: Decimal, stop_price: Decimal
    ) -> Order | None:
        """
        Place a stop order (implemented as stop-limit).

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            stop_price: Stop trigger price

        Returns:
            Order object if successful
        """
        if settings.system.dry_run:
            logger.info(
                f"PAPER TRADING: Simulating {side} {quantity} {symbol} stop @ {stop_price}"
            )
            return Order(
                id=f"paper_stop_{int(datetime.utcnow().timestamp() * 1000)}",
                symbol=symbol,
                side=side,
                type="STOP",
                quantity=quantity,
                stop_price=stop_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.utcnow(),
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise CoinbaseOrderError("Quantity must be positive")

            if stop_price <= 0:
                raise CoinbaseOrderError("Stop price must be positive")

            if side not in ["BUY", "SELL"]:
                raise CoinbaseOrderError(f"Invalid side: {side}")

            # Calculate limit price with small buffer
            slippage_pct = settings.trading.slippage_tolerance_pct / 100
            if side == "BUY":
                limit_price = stop_price * (1 + Decimal(str(slippage_pct)))
            else:
                limit_price = stop_price * (1 - Decimal(str(slippage_pct)))

            # Place stop-limit order
            order_data = {
                "product_id": symbol,
                "side": side,
                "order_configuration": {
                    "stop_limit_stop_limit_gtc": {
                        "base_size": str(quantity),
                        "limit_price": str(limit_price),
                        "stop_price": str(stop_price),
                        "stop_direction": (
                            "STOP_DIRECTION_STOP_DOWN"
                            if side == "SELL"
                            else "STOP_DIRECTION_STOP_UP"
                        ),
                    }
                },
            }

            logger.info(
                f"Placing {side} stop order: {quantity} {symbol} stop @ {stop_price}, limit @ {limit_price}"
            )
            result = await self._retry_request(self._client.create_order, **order_data)

            # Parse the response
            if result.get("success"):
                order_info = result.get("order", {})
                order_id = order_info.get("order_id")

                order = Order(
                    id=order_id
                    or f"cb_stop_{int(datetime.utcnow().timestamp() * 1000)}",
                    symbol=symbol,
                    side=side,
                    type="STOP_LIMIT",
                    quantity=quantity,
                    price=limit_price,
                    stop_price=stop_price,
                    status=OrderStatus.OPEN,
                    timestamp=datetime.utcnow(),
                    filled_quantity=Decimal("0"),
                )

                logger.info(f"Stop order placed successfully: {order_id}")
                return order
            else:
                error_msg = result.get("error_response", {}).get(
                    "message", "Unknown error"
                )
                logger.error(f"Stop order placement failed: {error_msg}")
                raise CoinbaseOrderError(f"Stop order placement failed: {error_msg}")

        except CoinbaseAPIException as e:
            if "insufficient funds" in str(e).lower():
                raise CoinbaseInsufficientFundsError(f"Insufficient funds: {e}") from e
            else:
                raise CoinbaseOrderError(f"API error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to place stop order: {e}")
            logger.debug(f"Stop order error traceback: {traceback.format_exc()}")
            raise CoinbaseOrderError(f"Failed to place stop order: {e}") from e

    async def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        leverage: int = None,
        reduce_only: bool = False,
    ) -> Order | None:
        """
        Place a futures market order with leverage.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in base currency
            leverage: Leverage multiplier for futures
            reduce_only: True if this order should only reduce existing position

        Returns:
            Order object if successful
        """
        if settings.system.dry_run:
            # Simulate futures order in dry-run mode
            logger.info(
                f"PAPER TRADING FUTURES: Simulating {side} {quantity} {symbol} at market (leverage: {leverage}x)"
            )
            return Order(
                id=f"paper_futures_{int(datetime.utcnow().timestamp() * 1000)}",
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.utcnow(),
                filled_quantity=quantity,
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise CoinbaseOrderError("Quantity must be positive")

            if side not in ["BUY", "SELL"]:
                raise CoinbaseOrderError(f"Invalid side: {side}")

            leverage = leverage or self.max_futures_leverage

            # Generate a unique client order ID
            import uuid

            client_order_id = f"ai-bot-{uuid.uuid4().hex[:8]}"

            # Round quantity to appropriate precision (6 decimal places for ETH)
            rounded_quantity = round(float(quantity), 6)

            # Place the futures order using Coinbase Advanced Trader
            order_data = {
                "product_id": symbol,
                "side": side,
                "order_configuration": {
                    "market_market_ioc": {"base_size": str(rounded_quantity)}
                },
                "leverage": str(leverage),  # This makes it a futures order
            }

            # Add reduce_only only if True
            if reduce_only:
                order_data["reduce_only"] = reduce_only
                
            # Don't add portfolio ID for futures - Coinbase handles the routing automatically

            logger.info(
                f"Placing FUTURES {side} market order: {quantity} {symbol} (leverage: {leverage}x)"
            )
            logger.debug(f"Order data: {order_data}")
            result = await self._retry_request(
                self._client.create_order, client_order_id, **order_data
            )

            # Parse the response - handle CDP API response object
            try:
                # CDP API returns a response object with success/success_response
                if hasattr(result, "success") and result.success:
                    # Handle the new SDK response format
                    resp = result.success_response
                    if isinstance(resp, dict):
                        order_id = resp.get('order_id')
                    else:
                        order_id = resp.order_id if hasattr(resp, 'order_id') else None

                    order = Order(
                        id=order_id
                        or f"cb_futures_{int(datetime.utcnow().timestamp() * 1000)}",
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.utcnow(),
                        filled_quantity=Decimal("0"),
                    )

                    logger.info(f"Futures market order placed successfully: {order_id}")
                    return order
                elif hasattr(result, "order_id"):
                    # Fallback for old format
                    order_id = result.order_id
                else:
                    # Try dictionary access for backward compatibility
                    if isinstance(result, dict) and result.get("success"):
                        order_info = result.get("order", {})
                        order_id = order_info.get("order_id")

                        order = Order(
                            id=order_id
                            or f"cb_futures_{int(datetime.utcnow().timestamp() * 1000)}",
                            symbol=symbol,
                            side=side,
                            type="MARKET",
                            quantity=quantity,
                            status=OrderStatus.PENDING,
                            timestamp=datetime.utcnow(),
                            filled_quantity=Decimal("0"),
                        )

                        logger.info(
                            f"Futures market order placed successfully: {order_id}"
                        )
                        return order
                    else:
                        error_msg = str(result) if result else "Unknown error"
                        logger.error(f"Futures order placement failed: {error_msg}")
                        raise CoinbaseOrderError(
                            f"Futures order placement failed: {error_msg}"
                        )
            except AttributeError as e:
                logger.error(f"Error parsing order response: {e}")
                logger.error(f"Response type: {type(result)}, Response: {result}")
                raise CoinbaseOrderError(f"Failed to parse order response: {e}") from e

        except CoinbaseAPIException as e:
            if (
                "insufficient funds" in str(e).lower()
                or "insufficient margin" in str(e).lower()
            ):
                raise CoinbaseInsufficientFundsError(
                    f"Insufficient margin for futures order: {e}"
                ) from e
            else:
                raise CoinbaseOrderError(f"Futures API error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to place futures market order: {e}")
            logger.debug(
                f"Futures market order error traceback: {traceback.format_exc()}"
            )
            raise CoinbaseOrderError(
                f"Failed to place futures market order: {e}"
            ) from e

    async def get_account_balance(
        self, account_type: AccountType | None = None
    ) -> Decimal:
        """
        Get account balance in USD.

        Args:
            account_type: Specific account type (CFM/CBI) or None for total

        Returns:
            Account balance in USD
        """
        if settings.system.dry_run:
            # Return mock balance for paper trading
            mock_balance = Decimal("10000.00")
            logger.debug(f"PAPER TRADING: Returning mock balance ${mock_balance}")
            return mock_balance

        try:
            if self.enable_futures and account_type == AccountType.CFM:
                return await self.get_futures_balance()
            elif account_type == AccountType.CBI:
                return await self.get_spot_balance()
            else:
                # Get total balance across all accounts
                spot_balance = await self.get_spot_balance()
                if self.enable_futures:
                    futures_balance = await self.get_futures_balance()
                    return spot_balance + futures_balance
                return spot_balance

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise CoinbaseExchangeError(f"Failed to get account balance: {e}") from e

    async def get_spot_balance(self) -> Decimal:
        """
        Get spot account (CBI) balance in USD.

        Returns:
            Spot account balance in USD
        """
        try:
            accounts_data = await self._retry_request(self._client.get_accounts)

            # Find USD balance in spot accounts
            total_balance = Decimal("0")
            for account in accounts_data.get("accounts", []):
                if (
                    account.get("currency") == "USD"
                    and account.get("type") != "futures"
                ):
                    available_balance = account.get("available_balance", {}).get(
                        "value", "0"
                    )
                    total_balance += Decimal(str(available_balance))

            logger.debug(f"Retrieved spot USD balance: {total_balance}")
            return total_balance

        except Exception as e:
            logger.error(f"Failed to get spot balance: {e}")
            raise CoinbaseExchangeError(f"Failed to get spot balance: {e}") from e

    async def get_futures_balance(self) -> Decimal:
        """
        Get futures account (CFM) balance in USD.

        Returns:
            Futures account balance in USD
        """
        try:
            # Get FCM balance summary
            balance_response = await self._retry_request(
                self._client.get_fcm_balance_summary
            )

            # Handle response object format
            if hasattr(balance_response, "balance_summary"):
                balance_data = balance_response.balance_summary
                # Get CFM USD balance from the response - use direct attribute access
                cfm_balance = balance_data.cfm_usd_balance.get("value", "0")
                futures_balance = Decimal(str(cfm_balance))
            else:
                # Fallback for dict format
                cash_balance = balance_response.get("cash_balance", {}).get(
                    "value", "0"
                )
                futures_balance = Decimal(str(cash_balance))

            logger.debug(f"Retrieved futures USD balance: {futures_balance}")
            return futures_balance

        except Exception as e:
            logger.error(f"Failed to get futures balance: {e}")
            # Fallback to regular accounts API
            return await self._get_futures_balance_fallback()

    async def _get_futures_balance_fallback(self) -> Decimal:
        """
        Fallback method to get futures balance using regular accounts API.

        Returns:
            Futures account balance in USD
        """
        try:
            accounts_data = await self._retry_request(self._client.get_accounts)

            total_balance = Decimal("0")
            for account in accounts_data.get("accounts", []):
                if (
                    account.get("currency") == "USD"
                    and account.get("type") == "futures"
                ):
                    available_balance = account.get("available_balance", {}).get(
                        "value", "0"
                    )
                    total_balance += Decimal(str(available_balance))

            return total_balance

        except Exception as e:
            logger.error(f"Fallback futures balance failed: {e}")
            return Decimal("0")

    async def get_futures_account_info(
        self, refresh: bool = False
    ) -> FuturesAccountInfo | None:
        """
        Get comprehensive futures account information.

        Args:
            refresh: Force refresh of cached data

        Returns:
            FuturesAccountInfo object or None if not available
        """
        if not self.enable_futures:
            return None

        if self._futures_account_info and not refresh:
            return self._futures_account_info

        try:
            # Get FCM balance summary
            balance_response = await self._retry_request(
                self._client.get_fcm_balance_summary
            )

            # Get margin info
            margin_info = await self.get_margin_info()

            # Parse account information based on response format
            if hasattr(balance_response, "balance_summary"):
                balance_data = balance_response.balance_summary
                cash_balance = Decimal(
                    str(balance_data.cbi_usd_balance.get("value", "0"))
                )
                futures_balance = Decimal(
                    str(balance_data.cfm_usd_balance.get("value", "0"))
                )
                total_balance = Decimal(
                    str(balance_data.total_usd_balance.get("value", "0"))
                )
                account_id = "cfm_account"  # Response doesn't include account_id
            else:
                # Fallback for dict format
                cash_balance = Decimal(
                    str(balance_response.get("cash_balance", {}).get("value", "0"))
                )
                futures_balance = cash_balance
                total_balance = cash_balance + futures_balance
                account_id = balance_response.get("account_id", "unknown")

            account_info = FuturesAccountInfo(
                account_type=AccountType.CFM,
                account_id=account_id,
                currency="USD",
                cash_balance=cash_balance,
                futures_balance=futures_balance,
                total_balance=total_balance,
                margin_info=margin_info,
                auto_cash_transfer_enabled=self.auto_cash_transfer,
                min_cash_transfer_amount=Decimal("100"),
                max_cash_transfer_amount=Decimal("10000"),
                max_leverage=self.max_futures_leverage,
                max_position_size=total_balance
                * Decimal(str(settings.trading.max_size_pct / 100)),
                current_positions_count=len(await self.get_futures_positions()),
                timestamp=datetime.utcnow(),
            )

            self._futures_account_info = account_info
            return account_info

        except Exception as e:
            logger.error(f"Failed to get futures account info: {e}")
            return None

    async def get_margin_info(self) -> MarginInfo:
        """
        Get futures margin information and health status.

        Returns:
            MarginInfo object with current margin status
        """
        try:
            # Get margin data from FCM balance summary
            balance_response = await self._retry_request(
                self._client.get_fcm_balance_summary
            )

            # Parse margin information based on response format
            if hasattr(balance_response, "balance_summary"):
                balance_data = balance_response.balance_summary
                cash_balance = Decimal(
                    str(balance_data.total_usd_balance.get("value", "0"))
                )
                initial_margin = Decimal(
                    str(balance_data.initial_margin.get("value", "0"))
                )
                # Use futures_buying_power if available, otherwise available_margin
                if hasattr(balance_data, "futures_buying_power"):
                    available_margin = Decimal(
                        str(balance_data.futures_buying_power.get("value", "0"))
                    )
                else:
                    available_margin = Decimal(
                        str(balance_data.available_margin.get("value", "0"))
                    )
                used_margin = cash_balance - available_margin
                liquidation_threshold = Decimal(
                    str(balance_data.liquidation_threshold.get("value", "0"))
                )
            else:
                # Fallback for dict format
                cash_balance = Decimal(
                    str(balance_response.get("cash_balance", {}).get("value", "0"))
                )
                used_margin = Decimal(
                    str(balance_response.get("used_margin", {}).get("value", "0"))
                )
                available_margin = cash_balance - used_margin
                initial_margin = used_margin * Decimal("1.0")  # 100% initial
                liquidation_threshold = cash_balance * Decimal("0.9")

            # Calculate margin requirements
            maintenance_margin = used_margin * Decimal("0.5")  # 50% maintenance

            # Calculate margin ratio and health
            margin_ratio = (
                float(used_margin / cash_balance) if cash_balance > 0 else 0.0
            )

            # Determine health status
            if margin_ratio < 0.5:
                health_status = MarginHealthStatus.HEALTHY
            elif margin_ratio < 0.75:
                health_status = MarginHealthStatus.WARNING
            elif margin_ratio < 0.9:
                health_status = MarginHealthStatus.CRITICAL
            else:
                health_status = MarginHealthStatus.LIQUIDATION_RISK

            # Calculate intraday vs overnight requirements
            intraday_multiplier = settings.trading.intraday_margin_multiplier
            overnight_multiplier = settings.trading.overnight_margin_multiplier

            margin_info = MarginInfo(
                total_margin=cash_balance,
                available_margin=available_margin,
                used_margin=used_margin,
                maintenance_margin=maintenance_margin,
                initial_margin=initial_margin,
                margin_ratio=margin_ratio,
                health_status=health_status,
                liquidation_threshold=liquidation_threshold,
                intraday_margin_requirement=used_margin
                * Decimal(str(intraday_multiplier)),
                overnight_margin_requirement=used_margin
                * Decimal(str(overnight_multiplier)),
                is_overnight_position=False,  # Would need to check position timing
            )

            self._last_margin_check = datetime.utcnow()
            return margin_info

        except Exception as e:
            logger.error(f"Failed to get margin info: {e}")
            # Return default margin info
            return MarginInfo(
                total_margin=Decimal("0"),
                available_margin=Decimal("0"),
                used_margin=Decimal("0"),
                maintenance_margin=Decimal("0"),
                initial_margin=Decimal("0"),
                margin_ratio=0.0,
                health_status=MarginHealthStatus.HEALTHY,
                liquidation_threshold=Decimal("0"),
                intraday_margin_requirement=Decimal("0"),
                overnight_margin_requirement=Decimal("0"),
            )

    async def transfer_cash_to_futures(
        self, amount: Decimal, reason: str = "MANUAL"
    ) -> bool:
        """
        Transfer cash from spot to futures account for margin.

        Args:
            amount: Amount to transfer in USD
            reason: Reason for transfer

        Returns:
            True if transfer successful
        """
        if not self.auto_cash_transfer:
            logger.warning("Auto cash transfer is disabled")
            return False

        try:
            CashTransferRequest(
                from_account=AccountType.CBI,
                to_account=AccountType.CFM,
                amount=amount,
                currency="USD",
                reason=reason,
            )

            # Execute transfer via Coinbase API
            logger.info(f"Transferring ${amount} from spot to futures for {reason}")

            if settings.system.dry_run:
                logger.info(f"PAPER TRADING: Simulating transfer ${amount} CBI -> CFM")
                return True

            # Use futures sweep API to transfer funds
            try:
                # Cancel any pending sweeps first
                try:
                    await self._retry_request(self._client.cancel_pending_futures_sweep)
                    logger.debug("Cancelled pending futures sweep")
                except Exception:
                    pass  # Ignore if no pending sweep
                
                # Schedule a new sweep
                result = await self._retry_request(
                    self._client.schedule_futures_sweep,
                    usd_amount=str(amount)
                )
                
                logger.info(f"Successfully scheduled futures sweep for ${amount}")
                
                # Wait a bit for the sweep to process
                await asyncio.sleep(3)
                
                # Check if transfer completed
                balance_resp = await self._retry_request(self._client.get_futures_balance_summary)
                if hasattr(balance_resp, 'balance_summary'):
                    cfm_balance = Decimal(balance_resp.balance_summary.cfm_usd_balance['value'])
                    if cfm_balance > 0:
                        logger.info(f"Transfer completed. CFM balance: ${cfm_balance}")
                        return True
                
                logger.warning("Transfer scheduled but not yet completed")
                return True  # Return true since sweep was scheduled
                
            except Exception as e:
                logger.error(f"Failed to schedule futures sweep: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to transfer cash to futures: {e}")
            return False

    async def get_futures_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current futures positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of futures Position objects
        """
        if not self.enable_futures:
            return []

        try:
            # Get positions from FCM account
            positions_response = await self._retry_request(
                self._client.get_fcm_positions
            )

            # Handle response object format
            if hasattr(positions_response, "positions"):
                positions_list = positions_response.positions
            else:
                # Fallback for dict format
                positions_list = positions_response.get("positions", [])

            positions = []
            for pos_data in positions_list:
                product_id = pos_data.get("product_id")

                # Skip if symbol filter provided and doesn't match
                if symbol and product_id != symbol:
                    continue

                size = Decimal(str(pos_data.get("size", "0")))
                if abs(size) > 0:
                    side = "LONG" if size > 0 else "SHORT"

                    position = Position(
                        symbol=product_id,
                        side=side,
                        size=abs(size),
                        entry_price=Decimal(str(pos_data.get("avg_entry_price", "0"))),
                        unrealized_pnl=Decimal(
                            str(pos_data.get("unrealized_pnl", "0"))
                        ),
                        realized_pnl=Decimal(str(pos_data.get("realized_pnl", "0"))),
                        timestamp=datetime.utcnow(),
                        is_futures=True,
                        leverage=pos_data.get("leverage", self.max_futures_leverage),
                        margin_used=Decimal(str(pos_data.get("margin_used", "0"))),
                        liquidation_price=Decimal(
                            str(pos_data.get("liquidation_price", "0"))
                        ),
                        margin_health=MarginHealthStatus.HEALTHY,  # Would calculate based on margin
                    )
                    positions.append(position)

            logger.debug(f"Retrieved {len(positions)} futures positions")
            return positions

        except Exception as e:
            logger.error(f"Failed to get futures positions: {e}")
            # Fallback to regular positions API
            return await self.get_positions(symbol)

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """
        if settings.system.dry_run:
            # Return empty positions for dry run
            return []

        try:
            accounts_data = await self._retry_request(self._client.get_accounts)
            positions = []

            for account in accounts_data.get("accounts", []):
                currency = account.get("currency")
                if currency == "USD":
                    continue  # Skip USD balance

                # Skip if symbol filter provided and doesn't match
                if symbol and not currency.startswith(symbol.split("-")[0]):
                    continue

                balance_data = account.get("available_balance", {})
                balance = Decimal(str(balance_data.get("value", "0")))

                if balance > 0:
                    position = Position(
                        symbol=f"{currency}-USD",
                        side="LONG" if balance > 0 else "SHORT",
                        size=abs(balance),
                        entry_price=None,  # Would need additional API call to get this
                        unrealized_pnl=Decimal(
                            "0"
                        ),  # Would need market data to calculate
                        realized_pnl=Decimal("0"),
                        timestamp=datetime.utcnow(),
                    )
                    positions.append(position)

            logger.debug(f"Retrieved {len(positions)} positions")
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise CoinbaseExchangeError(f"Failed to get positions: {e}") from e

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        if settings.system.dry_run:
            logger.info(f"PAPER TRADING: Simulating cancel order {order_id}")
            return True

        try:
            logger.info(f"Cancelling order: {order_id}")
            result = await self._retry_request(
                self._client.cancel_orders, order_ids=[order_id]
            )

            # Check if cancellation was successful
            cancelled_orders = result.get("results", [])
            for cancelled in cancelled_orders:
                if cancelled.get("order_id") == order_id and cancelled.get("success"):
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True

            logger.warning(f"Order {order_id} cancellation may have failed")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str | None = None) -> bool:
        """
        Cancel all open orders, optionally filtered by symbol.

        Args:
            symbol: Optional trading symbol filter

        Returns:
            True if successful
        """
        if settings.system.dry_run:
            logger.info(
                f"PAPER TRADING: Simulating cancel all orders{' for ' + symbol if symbol else ''}"
            )
            return True

        try:
            # Get all open orders
            orders_data = await self._retry_request(
                self._client.list_orders, order_status="OPEN"
            )

            orders_to_cancel = []
            for order in orders_data.get("orders", []):
                order_symbol = order.get("product_id")
                order_id = order.get("order_id")

                # Filter by symbol if provided
                if symbol and order_symbol != symbol:
                    continue

                if order_id:
                    orders_to_cancel.append(order_id)

            if not orders_to_cancel:
                logger.info(
                    f"No open orders to cancel{' for ' + symbol if symbol else ''}"
                )
                return True

            logger.info(
                f"Cancelling {len(orders_to_cancel)} orders{' for ' + symbol if symbol else ''}"
            )
            result = await self._retry_request(
                self._client.cancel_orders, order_ids=orders_to_cancel
            )

            # Count successful cancellations
            successful_cancellations = 0
            for cancelled in result.get("results", []):
                if cancelled.get("success"):
                    successful_cancellations += 1

            logger.info(
                f"Successfully cancelled {successful_cancellations}/{len(orders_to_cancel)} orders"
            )
            return successful_cancellations == len(orders_to_cancel)

        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        """
        Get status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatus or None
        """
        if settings.system.dry_run:
            # Return mock status for dry run
            return OrderStatus.FILLED

        try:
            result = await self._retry_request(
                self._client.get_order, order_id=order_id
            )

            order_data = result.get("order", {})
            status_str = order_data.get("status", "").upper()

            # Map Coinbase status to our OrderStatus enum
            status_mapping = {
                "PENDING": OrderStatus.PENDING,
                "OPEN": OrderStatus.OPEN,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "EXPIRED": OrderStatus.CANCELLED,
                "FAILED": OrderStatus.FAILED,
                "UNKNOWN": OrderStatus.PENDING,
            }

            status = status_mapping.get(status_str, OrderStatus.PENDING)
            logger.debug(f"Order {order_id} status: {status_str} -> {status}")

            return status

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    def is_connected(self) -> bool:
        """
        Check if client is connected and authenticated.

        Returns:
            True if connected
        """
        return self._authenticated

    def get_connection_status(self) -> dict[str, Any]:
        """
        Get connection status information.

        Returns:
            Dictionary with connection details
        """
        # Check credentials based on auth method
        has_credentials = False
        if self.auth_method == "legacy":
            has_credentials = bool(self.api_key and self.api_secret and self.passphrase)
        elif self.auth_method == "cdp":
            has_credentials = bool(self.cdp_api_key_name and self.cdp_private_key)

        status = {
            "connected": self._authenticated,
            "sandbox": self.sandbox,
            "auth_method": self.auth_method,
            "has_credentials": has_credentials,
            "dry_run": settings.system.dry_run,
            "trading_mode": (
                "PAPER TRADING" if settings.system.dry_run else "LIVE TRADING"
            ),
            "last_health_check": (
                self._last_health_check.isoformat() if self._last_health_check else None
            ),
            "rate_limit_remaining": self._rate_limiter.max_requests
            - len(self._rate_limiter.requests),
            # Futures-specific status
            "futures_enabled": self.enable_futures,
            "futures_account_type": self.futures_account_type,
            "auto_cash_transfer": self.auto_cash_transfer,
            "max_futures_leverage": self.max_futures_leverage,
            "last_margin_check": (
                self._last_margin_check.isoformat() if self._last_margin_check else None
            ),
        }

        # Add futures account info if available
        if self._futures_account_info:
            status["futures_account_status"] = {
                "account_id": self._futures_account_info.account_id,
                "total_balance": float(self._futures_account_info.total_balance),
                "margin_health": self._futures_account_info.margin_info.health_status.value,
                "margin_ratio": self._futures_account_info.margin_info.margin_ratio,
                "current_positions": self._futures_account_info.current_positions_count,
                "auto_transfer_enabled": self._futures_account_info.auto_cash_transfer_enabled,
            }

        return status

    # Legacy method aliases for backward compatibility
    async def _get_account_balance(self) -> Decimal:
        """Legacy method for backward compatibility."""
        return await self.get_account_balance()

    async def _get_position(self, symbol: str) -> Position | None:
        """Get current position for symbol (legacy method)."""
        positions = await self.get_positions(symbol)
        return (
            positions[0]
            if positions
            else Position(
                symbol=symbol,
                side="FLAT",
                size=Decimal("0"),
                timestamp=datetime.utcnow(),
            )
        )

    async def _place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> Order | None:
        """Legacy method for backward compatibility."""
        return await self.place_market_order(symbol, side, quantity)

    async def _place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> Order | None:
        """Legacy method for backward compatibility."""
        return await self.place_limit_order(symbol, side, quantity, price)
