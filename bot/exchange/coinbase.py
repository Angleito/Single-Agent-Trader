"""
Coinbase exchange client for trading operations.

This module provides a wrapper around the coinbase-advanced-py SDK
for executing trades, managing orders, and retrieving account information.
"""

import asyncio
import decimal
import logging
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, cast

try:
    # Prefer the new official SDK import path first
    from coinbase.rest import RESTClient as _BaseClient

    # Use standard exceptions since SDK doesn't export specific ones
    class CoinbaseAPIError(Exception):
        pass

    class CoinbaseAuthenticationError(Exception):
        pass

    class CoinbaseConnectionError(Exception):
        pass

    class CoinbaseAdvancedTrader(_BaseClient):
        """Adapter class exposing legacy method names used in this codebase."""

        # Legacy CFM helper names used in the code
        def get_fcm_balance_summary(self, **kwargs):
            return self.get_futures_balance_summary(**kwargs)

        def get_fcm_positions(self, **kwargs):
            return self.list_futures_positions(**kwargs)

        # Override get_accounts to return a dict similar to legacy SDK
        def get_accounts(self, *args, **kwargs):  # type: ignore[override]
            resp = super().get_accounts(*args, **kwargs)
            if isinstance(resp, dict):
                return resp
            # RESTClient returns an object with `.accounts` attribute
            try:
                accounts = resp.accounts  # type: ignore[attr-defined]
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
            CoinbaseAdvancedTrader as _FallbackTrader,  # type: ignore[import-untyped]
        )
        from coinbase_advanced_trader.exceptions import (  # type: ignore[import-untyped]
            CoinbaseAPIException as _FallbackAPIError,
        )
        from coinbase_advanced_trader.exceptions import (
            CoinbaseAuthenticationException as _FallbackAuthError,
        )
        from coinbase_advanced_trader.exceptions import (
            CoinbaseConnectionException as _FallbackConnectionError,
        )

        # Use fallback imports
        CoinbaseAdvancedTrader = _FallbackTrader  # type: ignore[misc]
        CoinbaseAPIError = _FallbackAPIError  # type: ignore[misc]
        CoinbaseAuthenticationError = _FallbackAuthError  # type: ignore[misc]
        CoinbaseConnectionError = _FallbackConnectionError  # type: ignore[misc]

        COINBASE_AVAILABLE = True
    except ImportError:  # pragma: no cover
        # Mock classes for when *no* Coinbase SDK is installed
        class CoinbaseAdvancedTrader:  # type: ignore[misc]
            def __init__(self, **kwargs):
                pass

        class CoinbaseAPIError(Exception):  # type: ignore[no-redef]
            pass

        class CoinbaseConnectionError(Exception):  # type: ignore[no-redef]
            pass

        class CoinbaseAuthenticationError(Exception):  # type: ignore[no-redef]
            pass

        COINBASE_AVAILABLE = False

from bot.config import settings
from bot.fp.types import (
    AccountType,
    MarginHealthStatus,
    MarginInfo,
    Order,
    OrderStatus,
    Position,
    TradeAction,
)

# Backward compatibility imports for types not yet in functional system
from bot.trading_types import (
    CashTransferRequest,
    FuturesAccountInfo,
)

from .base import (
    BalanceRetrievalError,
    BalanceServiceUnavailableError,
    BalanceTimeoutError,
    BalanceValidationError,
    BaseExchange,
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeInsufficientFundsError,
    ExchangeOrderError,
)
from .futures_utils import FuturesContractManager

# Functional programming imports
try:
    from bot.fp.adapters.coinbase_adapter import CoinbaseExchangeAdapter
    from bot.fp.adapters.exchange_adapter import register_exchange_adapter

    FP_ADAPTERS_AVAILABLE = True
except ImportError:
    # Fallback when FP adapters are not available
    FP_ADAPTERS_AVAILABLE = False
    CoinbaseExchangeAdapter = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class CoinbaseResponseValidator:
    """
    Validator for Coinbase API response schemas to prevent data corruption.

    Validates API responses for required fields, data types, and consistency
    before processing to prevent corruption-based trading losses.
    """

    def __init__(
        self, failure_callback: Callable[[str, dict[str, Any]], None] | None = None
    ):
        """
        Initialize the response validator.

        Args:
            failure_callback: Optional callback function for validation failures
        """
        self.response_count = 0
        self.validation_failures = 0
        self.failure_callback = failure_callback

    def validate_account_response(self, response: dict[str, Any]) -> bool:
        """
        Validate account response structure.

        Args:
            response: Account API response

        Returns:
            True if valid, False otherwise
        """
        try:
            self.response_count += 1

            # Check if response is dict
            if not isinstance(response, dict):
                logger.warning("Account response is not a dictionary")
                return False

            # Check for accounts array
            if "accounts" not in response:
                logger.warning("Account response missing 'accounts' field")
                return False

            accounts = response["accounts"]
            if not isinstance(accounts, list):
                logger.warning("Account 'accounts' field is not a list")
                return False

            # Validate individual accounts
            return all(self._validate_account_data(account) for account in accounts)

        except Exception:
            logger.exception("Error validating account response")
            return False

    def validate_balance_response(self, response: dict[str, Any] | Any) -> bool:
        """
        Validate balance response structure.

        Args:
            response: Balance API response (dict or object)

        Returns:
            True if valid, False otherwise
        """
        try:
            self.response_count += 1

            # Handle different response formats
            if hasattr(response, "balance_summary"):
                return self._validate_object_balance_response(response)
            if isinstance(response, dict):
                return self._validate_dict_balance_response(response)
        except Exception:
            logger.exception("Error validating balance response")
            return False
        else:
            logger.warning(
                "Balance response is neither dict nor object with balance_summary"
            )
            return False

    def _validate_object_balance_response(self, response: Any) -> bool:
        """Validate object format balance response from CDP SDK."""
        balance_summary = response.balance_summary
        if not hasattr(balance_summary, "cfm_usd_balance"):
            logger.warning("Balance response missing cfm_usd_balance")
            return False

        balance_data = balance_summary.cfm_usd_balance
        if not hasattr(balance_data, "value"):
            logger.warning("Balance data missing 'value' field")
            return False

        return self._validate_balance_value(balance_data.value)

    def _validate_dict_balance_response(self, response: dict[str, Any]) -> bool:
        """Validate dict format balance response."""
        if "balance" not in response:
            logger.warning("Balance response missing 'balance' field")
            return False

        return self._validate_balance_value(response["balance"])

    def _validate_balance_value(
        self, balance_value: str | int | float | Decimal
    ) -> bool:
        """Validate balance value format and range."""
        try:
            balance_decimal = Decimal(str(balance_value))
            if balance_decimal < 0:
                logger.warning("Invalid negative balance: %s", balance_decimal)
                return False
        except (ValueError, TypeError) as e:
            logger.warning("Invalid balance value format: %s", e)
            return False
        else:
            return True

    def validate_order_response(self, response: dict[str, Any] | Any) -> bool:
        """
        Validate order response structure.

        Args:
            response: Order API response

        Returns:
            True if valid, False otherwise
        """
        try:
            self.response_count += 1

            # Handle both dict and object responses
            order_data = None
            if hasattr(response, "order"):
                order_data = response.order
            elif isinstance(response, dict):
                order_data = response
            else:
                logger.warning("Order response format not recognized")
                return False

            # Validate required order fields
            required_fields = ["order_id", "status", "side"]
            for field in required_fields:
                if hasattr(order_data, field):
                    continue  # Object format
                if isinstance(order_data, dict) and field in order_data:
                    continue  # Dict format
                logger.warning("Order response missing required field: %s", field)
                return False

            # Validate order status
            if hasattr(order_data, "status"):
                status = order_data.status
            else:
                status = order_data.get("status")

            valid_statuses = ["OPEN", "FILLED", "CANCELLED", "PENDING", "REJECTED"]
            if status not in valid_statuses:
                logger.warning("Invalid order status: %s", status)
                return False

            # Validate side
            if hasattr(order_data, "side"):
                side = order_data.side
            else:
                side = order_data.get("side")

            if side not in ["BUY", "SELL"]:
                logger.warning("Invalid order side: %s", side)
                return False

        except Exception:
            logger.exception("Error validating order response")
            return False
        else:
            return True

    def validate_position_response(self, response: dict[str, Any] | object) -> bool:
        """
        Validate positions response structure.

        Args:
            response: Positions API response

        Returns:
            True if valid, False otherwise
        """
        try:
            self.response_count += 1

            # Handle different response formats
            positions_list = None
            if hasattr(response, "positions"):
                positions_list = response.positions
            elif isinstance(response, dict) and "positions" in response:
                positions_list = response["positions"]
            elif isinstance(response, list):
                positions_list = response
            else:
                logger.warning("Positions response format not recognized")
                return False

            if not isinstance(positions_list, list):
                logger.warning("Positions data is not a list")
                return False

            # Validate individual positions
            for position in positions_list:
                if not self._validate_position_data(position):
                    return False

        except Exception:
            logger.exception("Error validating position response")
            return False
        else:
            return True

    def _validate_account_data(self, account: dict[str, Any]) -> bool:
        """
        Validate individual account data.

        Args:
            account: Account data object

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for required fields
            required_fields = ["uuid", "name", "currency"]
            for field in required_fields:
                if field not in account:
                    logger.warning("Account missing required field: %s", field)
                    return False

            # Validate UUID format (basic check)
            uuid_str = account["uuid"]
            if not isinstance(uuid_str, str) or len(uuid_str) != 36:
                logger.warning("Invalid account UUID format: %s", uuid_str)
                return False

            # Validate currency
            currency = account["currency"]
            if not isinstance(currency, str) or len(currency) < 2:
                logger.warning("Invalid currency format: %s", currency)
                return False

            # Validate balance if present
            if "balance" in account:
                try:
                    balance = Decimal(str(account["balance"]))
                    if balance < 0:
                        logger.warning("Invalid negative account balance: %s", balance)
                        return False
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid account balance format: %s", e)
                    return False

        except Exception:
            logger.exception("Error validating account data")
            return False
        else:
            return True

    def _validate_position_data(self, position: dict[str, Any] | object) -> bool:
        """
        Validate individual position data.

        Args:
            position: Position data object

        Returns:
            True if valid, False otherwise
        """
        try:
            # Handle both dict and object formats
            if hasattr(position, "product_id"):
                # Object format
                product_id = position.product_id
                side = getattr(position, "side", None)
                size = getattr(position, "number_of_contracts", 0)
            else:
                # Dict format
                product_id = position.get("product_id")  # type: ignore[attr-defined]
                side = position.get("side")  # type: ignore[attr-defined]
                size = position.get("number_of_contracts", 0)  # type: ignore[attr-defined]

            # Validate product ID
            if not product_id or not isinstance(product_id, str):
                logger.warning("Invalid position product_id: %s", product_id)
                return False

            # Validate side if present
            if side and side not in ["LONG", "SHORT"]:
                logger.warning("Invalid position side: %s", side)
                return False

            # Validate size
            try:
                size_decimal = Decimal(str(size))
                if size_decimal < 0:
                    logger.warning("Invalid negative position size: %s", size_decimal)
                    return False
            except (ValueError, TypeError) as e:
                logger.warning("Invalid position size format: %s", e)
                return False
            else:
                return True

        except Exception:
            logger.exception("Error validating position data")
            return False

    def get_validation_stats(self) -> dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation metrics
        """
        success_rate = (
            (
                (self.response_count - self.validation_failures)
                / self.response_count
                * 100
            )
            if self.response_count > 0
            else 100.0
        )

        return {
            "total_responses": self.response_count,
            "validation_failures": self.validation_failures,
            "success_rate_pct": success_rate,
        }

    def _validate_exchange_response(self, response: dict[str, Any] | object) -> bool:
        """
        Validate generic exchange response for data integrity.

        Args:
            response: Exchange API response

        Returns:
            True if valid, False otherwise
        """
        try:
            if not response:
                logger.warning("Empty exchange response received")
                return False

            # Basic type validation
            if not isinstance(response, dict | object):
                logger.warning("Invalid response type: %s", type(response))
                return False

            # Check for error indicators in response
            if isinstance(response, dict):
                if "error" in response or "errors" in response:
                    logger.warning(
                        "Error in response: %s",
                        response.get("error", response.get("errors")),
                    )
                    return False

                # Check for success indicators
                if "success" in response and not response["success"]:
                    logger.warning("Response indicates failure")
                    return False

            # Check object attributes
            elif hasattr(response, "success") and not response.success:
                logger.warning("Response object indicates failure")
                return False

        except Exception:
            logger.exception("Error validating exchange response")
            return False
        else:
            return True

    def validate_order_creation_response(self, response: dict | object) -> bool:
        """
        Validate order creation response.

        Args:
            response: Order creation API response

        Returns:
            True if valid, False otherwise
        """
        try:
            self.response_count += 1

            # Basic response validation
            if not self._validate_exchange_response(response):
                self.validation_failures += 1
                if self.failure_callback:
                    self.failure_callback(
                        "order_response_validation",
                        "Order creation response validation failed",
                        "medium",
                    )
                return False

            # Order-specific validation
            order_data = None
            if hasattr(response, "success_response"):
                order_data = response.success_response
            elif hasattr(response, "order_id") or isinstance(response, dict):
                order_data = response

            if not order_data:
                logger.warning("Order creation response missing order data")
                self.validation_failures += 1
                return False

            # Check for order ID
            order_id = None
            if isinstance(order_data, dict):
                order_id = order_data.get("order_id")
            elif hasattr(order_data, "order_id"):
                order_id = order_data.order_id

            if not order_id:
                logger.warning("Order creation response missing order_id")
                self.validation_failures += 1
                if self.failure_callback:
                    self.failure_callback(
                        "missing_order_id",
                        "Order creation response missing order_id",
                        "high",
                    )
                return False

        except Exception:
            logger.exception("Error validating order creation response")
            self.validation_failures += 1
            return False
        else:
            return True

    def validate_order_cancellation_response(self, response: dict | object) -> bool:
        """
        Validate order cancellation response.

        Args:
            response: Order cancellation API response

        Returns:
            True if valid, False otherwise
        """
        try:
            self.response_count += 1

            # Basic response validation
            if not self._validate_exchange_response(response):
                self.validation_failures += 1
                return False

            # Check for success indicators
            success_confirmed = False
            if isinstance(response, dict):
                success_confirmed = (
                    response.get("success", False)
                    or "cancelled" in str(response).lower()
                )
            elif hasattr(response, "success"):
                success_confirmed = response.success

            if not success_confirmed:
                logger.warning("Order cancellation response does not confirm success")
                self.validation_failures += 1
                return False

        except Exception:
            logger.exception("Error validating order cancellation response")
            self.validation_failures += 1
            return False
        else:
            return True


class CoinbaseRateLimiter:
    """Rate limiter for Coinbase API requests."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []
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
                    logger.debug("Rate limit reached, waiting %.2fs", wait_time)
                    await asyncio.sleep(wait_time)
                    return await self.acquire()

            self.requests.append(now)
            return None


class CoinbaseClient(BaseExchange):
    """
    Coinbase exchange client for trading operations.

    Provides methods for placing orders, managing positions,
    and retrieving account/market data from Coinbase.
    Supports both spot trading (CBI) and futures trading (CFM).
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        passphrase: str | None = None,
        cdp_api_key_name: str | None = None,
        cdp_private_key: str | None = None,
        dry_run: bool | None = None,
    ):
        """
        Initialize the Coinbase client.

        Args:
            api_key: Coinbase API key (legacy)
            api_secret: Coinbase API secret (legacy)
            passphrase: Coinbase passphrase (legacy)
            cdp_api_key_name: CDP API key name
            cdp_private_key: CDP private key (PEM format)
            dry_run: Override dry run setting
        """
        # Initialize parent class
        if dry_run is None:
            dry_run = self.dry_run
        super().__init__(dry_run)
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
            self.api_key = api_key or (
                settings.exchange.cb_api_key.get_secret_value()
                if settings.exchange.cb_api_key
                else None
            )
            self.api_secret = api_secret or (
                settings.exchange.cb_api_secret.get_secret_value()
                if settings.exchange.cb_api_secret
                else None
            )
            self.passphrase = passphrase or (
                settings.exchange.cb_passphrase.get_secret_value()
                if settings.exchange.cb_passphrase
                else None
            )
            self.cdp_api_key_name = None
            self.cdp_private_key = None

        elif provided_cdp or settings_cdp:
            # Use CDP authentication
            self.auth_method = "cdp"
            self.api_key = None
            self.api_secret = None
            self.passphrase = None
            self.cdp_api_key_name = cdp_api_key_name or (
                settings.exchange.cdp_api_key_name.get_secret_value()
                if settings.exchange.cdp_api_key_name
                else None
            )
            self.cdp_private_key = cdp_private_key or (
                settings.exchange.cdp_private_key.get_secret_value()
                if settings.exchange.cdp_private_key
                else None
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
        self._client: Any | None = None
        self._connected = False  # Use _connected from parent class
        self._last_health_check: datetime | None = None

        # Rate limiting
        self._rate_limiter = CoinbaseRateLimiter(
            max_requests=settings.exchange.rate_limit_requests,
            window_seconds=settings.exchange.rate_limit_window_seconds,
        )

        # Response validation
        self._response_validator = CoinbaseResponseValidator()

        # Retry configuration
        self._max_retries = 3
        self._retry_delay = 1.0
        self._retry_backoff = 2.0

        # Futures trading configuration
        self._enable_futures = settings.trading.enable_futures
        self.futures_account_type = settings.trading.futures_account_type
        self.auto_cash_transfer = settings.trading.auto_cash_transfer
        self.max_futures_leverage = settings.trading.max_futures_leverage

        # Cached futures account info
        self._futures_account_info: Any | None = None
        self._last_margin_check: datetime | None = None

        # Portfolio management
        self._portfolios: dict[str, Any] = {}
        self._futures_portfolio_id: str | None = None
        self._default_portfolio_id: str | None = None

        # Futures contract management
        self._futures_contract_manager: Any | None = None

        # Volume tracking for fee tiers
        self._monthly_volume = Decimal(0)
        self._last_volume_check: datetime | None = None
        self._volume_check_interval = timedelta(hours=1)

        # Compose a concise init message to avoid line length issues
        trading_mode = "PAPER TRADING" if self.dry_run else "LIVE TRADING"
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
        logger.debug("  Trading mode: %s", trading_mode)
        logger.debug("  Authentication method: %s", self.auth_method)
        logger.debug("  Sandbox mode: %s", self.sandbox)
        logger.debug("  Futures enabled: %s", self.enable_futures)
        logger.debug("  Futures account type: %s", self.futures_account_type)
        logger.debug("  Auto cash transfer: %s", self.auto_cash_transfer)
        logger.debug("  Max futures leverage: %s", self.max_futures_leverage)
        logger.debug(
            "  Rate limit: %s req/%ss",
            self._rate_limiter.max_requests,
            self._rate_limiter.window_seconds,
        )
        logger.debug("  Max retries: %s", self._max_retries)
        logger.debug("  Has credentials: %s", bool(self.auth_method != "none"))

        # Initialize functional adapter for side-effect-free operations
        self._fp_adapter: CoinbaseExchangeAdapter | None = None
        if FP_ADAPTERS_AVAILABLE:
            try:
                self._fp_adapter = CoinbaseExchangeAdapter(self)
                # Register with unified adapter system
                register_exchange_adapter("coinbase", self._fp_adapter)
                logger.debug("✅ Functional adapter initialized for Coinbase")
            except Exception as e:
                logger.warning("Failed to initialize functional adapter: %s", e)
                self._fp_adapter = None

        # Log warning if in live trading mode
        if not self.dry_run:
            logger.warning(
                "⚠️  LIVE TRADING MODE ENABLED - Real money will be used! "
                "Use --dry-run flag for paper trading."
            )

    def set_failure_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """
        Set failure callback for validation errors.

        Args:
            callback: Function to call on validation failures
        """
        self._response_validator.failure_callback = callback

    def _initialize_paper_trading(self) -> bool:
        """Initialize paper trading mode."""
        logger.info(
            "PAPER TRADING MODE: Skipping Coinbase authentication. All trades will be simulated."
        )
        self._client = None
        self._connected = True
        self._last_health_check = datetime.now(UTC)

        # Initialize mock portfolios for paper trading
        self._portfolios = {}
        self._default_portfolio_id = "paper-trading-portfolio"
        if self.enable_futures:
            self._futures_portfolio_id = "paper-trading-futures-portfolio"

        # Initialize futures contract manager if futures are enabled
        if self.enable_futures:
            self._futures_contract_manager = FuturesContractManager(self)

        logger.info("Paper trading mode initialized successfully")
        return True

    def _validate_coinbase_sdk(self) -> None:
        """Validate that Coinbase SDK is available."""
        if not COINBASE_AVAILABLE:
            logger.error(
                "coinbase-advanced-py is not installed. "
                "Install it with: pip install coinbase-advanced-py"
            )
            raise ExchangeConnectionError(
                "coinbase-advanced-py is required for live trading"
            )

    def _initialize_legacy_auth(self) -> None:
        """Initialize client with legacy authentication."""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            logger.error(
                "Missing legacy Coinbase credentials. Please set CB_API_KEY, "
                "CB_API_SECRET, and CB_PASSPHRASE environment variables."
            )
            raise ExchangeAuthError("Missing legacy API credentials")

        self._client = CoinbaseAdvancedTrader(
            api_key=self.api_key,
            api_secret=self.api_secret,
            passphrase=self.passphrase,
            sandbox=self.sandbox,
        )

    def _initialize_cdp_auth(self) -> None:
        """Initialize client with CDP authentication."""
        if not all([self.cdp_api_key_name, self.cdp_private_key]):
            logger.error(
                "Missing CDP credentials. Please set CDP_API_KEY_NAME "
                "and CDP_PRIVATE_KEY environment variables."
            )
            raise ExchangeAuthError("Missing CDP API credentials")

        self._client = CoinbaseAdvancedTrader(
            api_key=self.cdp_api_key_name, api_secret=self.cdp_private_key
        )

    def _initialize_authentication(self) -> None:
        """Initialize client based on authentication method."""
        if self.auth_method == "legacy":
            self._initialize_legacy_auth()
        elif self.auth_method == "cdp":
            self._initialize_cdp_auth()
        else:
            logger.error(
                "No valid authentication method configured. Please provide "
                "either legacy (CB_API_KEY, CB_API_SECRET, CB_PASSPHRASE) or "
                "CDP (CDP_API_KEY_NAME, CDP_PRIVATE_KEY) credentials."
            )
            raise ExchangeAuthError("No authentication credentials provided")

    async def _finalize_connection(self) -> None:
        """Finalize connection setup."""
        self._connected = True
        self._last_health_check = datetime.now(UTC)

        logger.info(
            "Connected to Coinbase %s successfully",
            "Sandbox" if self.sandbox else "Live",
        )

        # Load portfolio information
        await self._load_portfolios()

        # Log connection success details
        logger.debug("Connection Success Details:")
        logger.debug("  Environment: %s", "Sandbox" if self.sandbox else "Production")
        logger.debug("  Authentication method: %s", self.auth_method)

    async def connect(self) -> bool:
        """
        Connect and authenticate with Coinbase.

        Returns:
            True if connection successful
        """
        try:
            # In paper trading mode, skip all authentication
            if self.dry_run:
                return self._initialize_paper_trading()

            # Live trading mode requires SDK and credentials
            self._validate_coinbase_sdk()
            self._initialize_authentication()

            # Test connection with a simple API call
            await self._test_connection()

            await self._finalize_connection()
        except CoinbaseAuthenticationError as e:
            logger.exception("Coinbase authentication failed")
            raise ExchangeAuthError(f"Authentication failed: {e}") from e
        except CoinbaseConnectionError as e:
            logger.exception("Coinbase connection failed")
            raise ExchangeConnectionError(f"Connection failed: {e}") from e
        except Exception as e:
            logger.exception("Failed to connect to Coinbase")
            logger.debug("Connection error traceback: %s", traceback.format_exc())
            raise ExchangeConnectionError(f"Unexpected error: {e}") from e
        else:
            return True

    async def _test_connection(self) -> None:
        """Test the connection with a simple API call."""
        try:
            await self._rate_limiter.acquire()
            if self._client is None:
                raise ExchangeConnectionError("Client not initialized")
            accounts = self._client.get_accounts()
            logger.debug(
                "Connection test successful, found %s accounts",
                len(accounts.get("accounts", [])),
            )
        except Exception as e:
            logger.exception("Connection test failed")
            raise ExchangeConnectionError(f"Connection test failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Coinbase."""
        if self._client:
            # Cancel any pending orders if needed
            try:
                if self._connected:
                    logger.info("Cleaning up before disconnect...")
                    # Could add cleanup logic here
            except Exception as e:
                logger.warning("Error during cleanup: %s", e)

        self._client = None
        self._connected = False
        self._last_health_check = None
        logger.info("Disconnected from Coinbase")

    async def _health_check(self) -> bool:
        """Perform a health check on the connection."""
        if not self._connected:
            return False

        # In paper trading mode, always return healthy
        if self.dry_run:
            self._last_health_check = datetime.now(UTC)
            return True

        try:
            # Only perform health check if enough time has passed
            if self._last_health_check and datetime.now(
                UTC
            ) - self._last_health_check < timedelta(
                seconds=settings.exchange.health_check_interval
            ):
                return True

            await self._rate_limiter.acquire()
            if self._client is None:
                return False
            self._client.get_accounts()
            self._last_health_check = datetime.now(UTC)
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            self._connected = False
            return False
        else:
            return True

    async def _retry_request(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a request with retry logic."""
        # In paper trading mode, don't make any real API calls
        if self.dry_run:
            logger.debug("PAPER TRADING: Skipping API call to %s", func.__name__)
            # Return mock responses for common methods
            if "get_accounts" in func.__name__:
                return {"accounts": []}
            if "get_portfolios" in func.__name__:
                return {"portfolios": []}
            if "get_fcm_balance_summary" in func.__name__:
                return {
                    "balance_summary": {
                        "cbi_usd_balance": {"value": "10000.00"},
                        "cfm_usd_balance": {"value": "10000.00"},
                        "total_usd_balance": {"value": "20000.00"},
                        "initial_margin": {"value": "0"},
                        "available_margin": {"value": "10000.00"},
                        "liquidation_threshold": {"value": "0"},
                        "futures_buying_power": {"value": "10000.00"},
                    }
                }
            if "list_futures_positions" in func.__name__:
                return {"positions": []}
            if "get_products" in func.__name__:
                # Mock futures contracts for paper trading
                return {
                    "products": [
                        {
                            "product_id": "ET-27JUN25-CDE",
                            "base_currency": "ETH",
                            "quote_currency": "USD",
                            "trading_disabled": False,
                            "display_name": "ETH Futures - Jun 2025",
                        },
                        {
                            "product_id": "BT-27JUN25-CDE",
                            "base_currency": "BTC",
                            "quote_currency": "USD",
                            "trading_disabled": False,
                            "display_name": "BTC Futures - Jun 2025",
                        },
                    ]
                }
            return {}

        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                # Check connection health before important operations
                if not await self._health_check():
                    logger.warning("Connection unhealthy, attempting to reconnect...")
                    if not await self.connect():
                        raise ExchangeConnectionError("Failed to reconnect")

                await self._rate_limiter.acquire()
                return func(*args, **kwargs)

            except (CoinbaseConnectionError, CoinbaseAPIError) as e:
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
                    logger.exception(
                        "Request failed after %s attempts", self._max_retries + 1
                    )
                    # Let the last_exception be raised below rather than bare raise
                    break

            except Exception as e:
                logger.exception("Unexpected error in request")
                # Store as last exception to be raised below
                last_exception = e
                break

        if last_exception:
            raise last_exception
        return None

    async def _load_portfolios(self) -> None:
        """Load portfolio information and identify futures portfolio."""
        try:
            logger.debug("Loading portfolio information...")

            # Try to get portfolios using the REST API
            try:
                if self._client is None:
                    logger.error("Client not initialized")
                    return

                # First try the standard portfolios endpoint
                portfolios_response = await self._retry_request(
                    lambda: self._client.get("/api/v3/brokerage/portfolios")
                )

                if isinstance(portfolios_response, dict):
                    portfolios_list = portfolios_response.get("portfolios", [])
                    # Convert list to dict indexed by portfolio ID
                    self._portfolios = {
                        portfolio.get("id", f"portfolio_{i}"): portfolio
                        for i, portfolio in enumerate(portfolios_list)
                        if isinstance(portfolio, dict)
                    }
                else:
                    self._portfolios = {}

            except Exception as e:
                logger.debug("Could not fetch portfolios via API: %s", e)
                self._portfolios = {}

            # If we have portfolios, try to identify the futures portfolio
            if self._portfolios:
                for portfolio in self._portfolios.values():
                    portfolio_name = portfolio.get("name", "").lower()
                    portfolio_type = portfolio.get("type", "").lower()
                    portfolio_id = portfolio.get("uuid") or portfolio.get("id")

                    # Check if this is the default portfolio
                    if portfolio.get("is_default"):
                        self._default_portfolio_id = portfolio_id
                        logger.debug("Found default portfolio: %s", portfolio_id)

                    # Check if this is a futures portfolio
                    if (
                        "futures" in portfolio_name
                        or "cfm" in portfolio_name
                        or portfolio_type == "futures"
                        or portfolio_type == "cfm"
                    ):
                        self._futures_portfolio_id = portfolio_id
                        logger.debug("Found futures portfolio: %s", portfolio_id)

                logger.info("Loaded %d portfolios", len(self._portfolios))
                if self._futures_portfolio_id:
                    logger.info("Futures portfolio ID: %s", self._futures_portfolio_id)
            else:
                logger.debug("No portfolios found or portfolios API not available")

        except Exception as e:
            logger.warning("Failed to load portfolios: %s", e)
            self._portfolios = {}

    async def get_portfolios(self) -> list[dict[str, str | float | bool]]:
        """
        Get list of all portfolios.

        Returns:
            List of portfolio dictionaries
        """
        if not self._portfolios:
            await self._load_portfolios()
        return list(self._portfolios.values())

    def _normalize_balance(self, amount: Decimal) -> Decimal:
        """
        Normalize balance to 2 decimal places for USD currency.
        This prevents floating-point precision errors and excessive decimal places.

        Args:
            amount: USD amount to normalize

        Returns:
            Normalized amount with 2 decimal places

        Raises:
            ValueError: If amount is invalid (NaN, infinite)
        """
        if amount is None:
            return Decimal("0.00")

        # Validate that the amount is a proper decimal
        if not isinstance(amount, Decimal):
            try:
                amount = Decimal(str(amount))
            except (ValueError, TypeError) as e:
                logger.exception(
                    "Invalid balance amount (type: %s)", type(amount).__name__
                )
                raise ValueError(f"Cannot convert to Decimal: {amount}") from e

        # Check for invalid values
        if amount.is_nan():
            logger.error("Balance amount is NaN")
            raise ValueError("Balance amount cannot be NaN")

        if amount.is_infinite():
            logger.error("Balance amount is infinite")
            raise ValueError("Balance amount cannot be infinite")

        # Quantize to 2 decimal places using banker's rounding
        from decimal import ROUND_HALF_EVEN

        return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

    def _normalize_crypto_amount(self, amount: Decimal) -> Decimal:
        """
        Normalize crypto amounts to 8 decimal places for precision.

        Args:
            amount: Crypto amount to normalize

        Returns:
            Normalized amount with 8 decimal places

        Raises:
            ValueError: If amount is invalid (NaN, infinite)
        """
        if amount is None:
            return Decimal("0.00000000")

        # Validate that the amount is a proper decimal
        if not isinstance(amount, Decimal):
            try:
                amount = Decimal(str(amount))
            except (ValueError, TypeError) as e:
                logger.exception(
                    "Invalid crypto amount (type: %s)", type(amount).__name__
                )
                raise ValueError(f"Cannot convert to Decimal: {amount}") from e

        # Check for invalid values
        if amount.is_nan():
            logger.error("Crypto amount is NaN")
            raise ValueError("Crypto amount cannot be NaN")

        if amount.is_infinite():
            logger.error("Crypto amount is infinite")
            raise ValueError("Crypto amount cannot be infinite")

        from decimal import ROUND_HALF_EVEN

        return amount.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)

    def get_futures_portfolio_id(self) -> str | None:
        """
        Get the futures portfolio ID if available.

        Returns:
            Futures portfolio UUID or None
        """
        return self._futures_portfolio_id

    async def get_trading_symbol(self, symbol: str) -> str:
        """
        Get the appropriate trading symbol based on whether futures are enabled.

        Args:
            symbol: Base symbol like "ETH-USD"

        Returns:
            Actual trading symbol (spot or futures contract)
        """
        if not self.enable_futures:
            # Use spot symbol as-is
            return symbol

        # Map spot symbols to actual futures contracts
        # Coinbase uses abbreviated symbols with date suffixes
        futures_mappings = {
            "ETH-USD": "ET-27JUN25-CDE",
            "BTC-USD": "BT-27JUN25-CDE",  # Assuming similar pattern
        }

        if symbol in futures_mappings:
            futures_symbol = futures_mappings[symbol]
            logger.info("Mapped %s to futures contract %s", symbol, futures_symbol)
            return futures_symbol

        # For other symbols, try to find dated contracts
        base_currency = symbol.split("-")[0]

        # Try to get cached contract first
        if self._futures_contract_manager:
            cached_contract = self._futures_contract_manager.get_cached_contract()
            if cached_contract:
                return cached_contract

            # Fetch current active contract
            active_contract = (
                await self._futures_contract_manager.get_active_futures_contract(
                    base_currency
                )
            )
            if active_contract:
                return active_contract

        # Fallback to original symbol
        logger.info("Using %s for futures trading", symbol)
        return symbol

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
        if not self._connected:
            logger.error("Not authenticated with Coinbase")
            return None

        try:
            # Get the actual trading symbol (spot or futures contract)
            actual_symbol = await self.get_trading_symbol(symbol)
            logger.info(
                "Using trading symbol: %s (requested: %s)", actual_symbol, symbol
            )

            if trade_action.action == "HOLD":
                logger.info("Action is HOLD - no trade executed")
                return None

            if trade_action.action == "CLOSE":
                return await self._close_position(actual_symbol)

            if trade_action.action in ["LONG", "SHORT"]:
                return await self._open_position(
                    trade_action, actual_symbol, current_price
                )

            logger.error("Unknown action: %s", trade_action.action)
            return None

        except Exception:
            logger.exception("Failed to execute trade action")
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
            return await self._open_spot_position(trade_action, symbol, current_price)
        except Exception:
            logger.exception("Failed to open position")
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

            # Use leverage from trade action or default
            leverage = trade_action.leverage or settings.trading.leverage
            leverage = min(leverage, self.max_futures_leverage)

            # Calculate quantity for futures - ALWAYS USE 1 CONTRACT
            # Check if this is an actual futures contract (ends with -CDE)
            if symbol.endswith("-CDE"):
                # Real futures contracts use CONTRACT COUNT, not ETH amount
                # Each contract = 0.1 ETH (nano futures)
                # FIXED: Always trade exactly 1 contract
                num_contracts = 1
                quantity = Decimal(
                    1
                )  # For CDE contracts, quantity is the number of contracts
                logger.info("Futures contract %s: FIXED 1 contract (0.1 ETH)", symbol)
            else:
                # Spot with leverage: use nano contract sizing
                contract_size = Decimal("0.1")  # 0.1 ETH per contract

                # FIXED: Always trade exactly 1 contract
                num_contracts = 1
                quantity = contract_size * num_contracts  # 0.1 ETH

                logger.info("Futures position: FIXED 1 contract = %s ETH", quantity)

            # Calculate actual notional value based on contracts
            actual_notional_value = quantity * current_price

            # Check if we need to transfer cash for margin
            margin_required = actual_notional_value / leverage

            # Check CFM balance and transfer if needed
            if futures_account.futures_balance < margin_required:
                transfer_amount = (
                    margin_required - futures_account.futures_balance + Decimal(10)
                )  # Add buffer
                logger.info(
                    "CFM balance $%.2f < margin required $%.2f. "
                    "Transferring $%.2f from CBI to CFM...",
                    futures_account.futures_balance,
                    margin_required,
                    transfer_amount,
                )

                transfer_success = await self.transfer_cash_to_futures(
                    amount=transfer_amount, reason="AUTO_REBALANCE"
                )

                if not transfer_success:
                    logger.error(
                        "Failed to transfer funds to CFM. Cannot open futures position."
                    )
                    return None

                # Wait a bit for transfer to settle
                await asyncio.sleep(2)

                # Re-check futures account info
                futures_account = await self.get_futures_account_info()
                if not futures_account:
                    logger.error("Cannot verify transfer - account info unavailable")
                    return None

            logger.info(
                "Margin required: $%s, Available margin: $%s",
                margin_required,
                available_margin,
            )

            # Determine order side
            side = cast(
                "Literal['BUY', 'SELL']",
                "BUY" if trade_action.action == "LONG" else "SELL",
            )

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
                # Place stop loss and take profit orders - CRITICAL for risk management
                logger.info(
                    "Placing protective orders: SL=%s%, TP=%s%",
                    trade_action.stop_loss_pct,
                    trade_action.take_profit_pct,
                )

                try:
                    stop_loss_order = await self._place_stop_loss(
                        order, trade_action, current_price
                    )
                    if stop_loss_order:
                        logger.info(
                            "✅ Stop loss order placed successfully: %s",
                            stop_loss_order.id,
                        )
                    else:
                        logger.error(
                            "❌ CRITICAL: Stop loss order failed for position %s",
                            order.id,
                        )
                        # Consider canceling the main order if stop loss fails

                    take_profit_order = await self._place_take_profit(
                        order, trade_action, current_price
                    )
                    if take_profit_order:
                        logger.info(
                            "✅ Take profit order placed successfully: %s",
                            take_profit_order.id,
                        )
                    else:
                        logger.warning(
                            "⚠️ Take profit order failed for position %s", order.id
                        )

                except Exception:
                    logger.exception("❌ CRITICAL ERROR placing protective orders")
                    # Continue with trade but log the critical failure

        except Exception:
            logger.exception("Failed to open futures position")
            return None
        else:
            return order

    async def _open_spot_position(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Open a spot position.

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

        # For spot trading, leverage is always 1
        quantity = position_value / current_price

        # Log fee information
        fee_rates = self.get_current_fee_rates()
        logger.info(
            "Spot order fees - Maker: %.4f%%, Taker: %.4f%%, Volume: $%.2f",
            fee_rates["maker"] * 100,
            fee_rates["taker"] * 100,
            fee_rates["volume"],
        )

        # Determine order side (BUY/SELL for spot)
        side = cast(
            "Literal['BUY', 'SELL']", "BUY" if trade_action.action == "LONG" else "SELL"
        )

        # Place market order
        order = await self.place_market_order(
            symbol=symbol, side=side, quantity=quantity
        )

        if order:
            # Place stop loss and take profit orders - CRITICAL for risk management
            logger.info(
                "Placing protective orders: SL=%s%, TP=%s%",
                trade_action.stop_loss_pct,
                trade_action.take_profit_pct,
            )

            try:
                stop_loss_order = await self._place_stop_loss(
                    order, trade_action, current_price
                )
                if stop_loss_order:
                    logger.info(
                        "✅ Stop loss order placed successfully: %s", stop_loss_order.id
                    )
                else:
                    logger.error(
                        "❌ CRITICAL: Stop loss order failed for position %s", order.id
                    )
                    # Consider canceling the main order if stop loss fails

                take_profit_order = await self._place_take_profit(
                    order, trade_action, current_price
                )
                if take_profit_order:
                    logger.info(
                        "✅ Take profit order placed successfully: %s",
                        take_profit_order.id,
                    )
                else:
                    logger.warning(
                        "⚠️ Take profit order failed for position %s", order.id
                    )

            except Exception:
                logger.exception("❌ CRITICAL ERROR placing protective orders")
                # Continue with trade but log the critical failure

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
        side = cast(
            "Literal['BUY', 'SELL']", "SELL" if position.side == "LONG" else "BUY"
        )

        # Place market order to close
        return await self.place_market_order(
            symbol=symbol, side=side, quantity=abs(position.size)
        )

    async def place_market_order(
        self, symbol: str, side: Literal["BUY", "SELL"], quantity: Decimal
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
        if self.dry_run:
            # Simulate order in dry-run mode
            logger.info(
                "PAPER TRADING: Simulating %s %s %s at market", side, quantity, symbol
            )
            return Order(
                id=f"paper_{int(datetime.now(UTC).timestamp() * 1000)}",
                symbol=symbol,
                side=cast("Literal['BUY', 'SELL']", side),
                type="MARKET",
                quantity=quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(UTC),
                filled_quantity=quantity,
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise ExchangeOrderError("Quantity must be positive")

            if side not in ["BUY", "SELL"]:
                raise ExchangeOrderError(f"Invalid side: {side}")

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
                    "Using default portfolio ID: %s", self._default_portfolio_id
                )

            if self._client is None:
                logger.error("Client not initialized")
                return None

            logger.info("Placing %s market order: %s %s", side, quantity, symbol)
            result = await self._retry_request(
                self._client.create_order, client_order_id, **order_data
            )

            # Parse the response - handle CDP API response object
            try:
                # Validate response before processing
                if not self._response_validator.validate_order_creation_response(
                    result
                ):
                    logger.error("Order creation response validation failed")
                    raise ExchangeOrderError("Invalid response from exchange")

                # CDP API returns a response object with success/success_response
                if hasattr(result, "success") and result.success:
                    # Handle the new SDK response format
                    resp = result.success_response
                    if isinstance(resp, dict):
                        order_id = resp.get("order_id")
                    else:
                        order_id = resp.order_id if hasattr(resp, "order_id") else None

                    order = Order(
                        id=order_id
                        or f"cb_{int(datetime.now(UTC).timestamp() * 1000)}",
                        symbol=symbol,
                        side=cast("Literal['BUY', 'SELL']", side),
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(UTC),
                        filled_quantity=Decimal(0),
                    )

                    logger.info("Market order placed successfully: %s", order_id)
                    return order
                if hasattr(result, "order_id"):
                    # Fallback for old format
                    order_id = result.order_id

                    order = Order(
                        id=order_id
                        or f"cb_{int(datetime.now(UTC).timestamp() * 1000)}",
                        symbol=symbol,
                        side=cast("Literal['BUY', 'SELL']", side),
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(UTC),
                        filled_quantity=Decimal(0),
                    )

                    logger.info("Market order placed successfully: %s", order_id)
                    return order
                if isinstance(result, dict) and result.get("success"):
                    order_info = result.get("order", {})
                    order_id = order_info.get("order_id")

                    order = Order(
                        id=order_id
                        or f"cb_{int(datetime.now(UTC).timestamp() * 1000)}",
                        symbol=symbol,
                        side=cast("Literal['BUY', 'SELL']", side),
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(UTC),
                        filled_quantity=Decimal(0),
                    )

                    logger.info("Market order placed successfully: %s", order_id)
                    return order
                error_msg = str(result) if result else "Unknown error"
                logger.error("Order placement failed: %s", error_msg)
                raise ExchangeOrderError(f"Order placement failed: {error_msg}")
            except AttributeError as e:
                logger.exception("Error parsing order response")
                logger.exception(
                    "Response type: %s, Response: %s",
                    type(result).__name__,
                    "[response data]",
                )
                raise ExchangeOrderError(f"Failed to parse order response: {e}") from e

        except CoinbaseAPIError as e:
            if "insufficient funds" in str(e).lower():
                raise ExchangeInsufficientFundsError(f"Insufficient funds: {e}") from e
            raise ExchangeOrderError(f"API error: {e}") from e
        except Exception as e:
            logger.exception("Failed to place market order")
            logger.debug("Market order error traceback: %s", traceback.format_exc())
            raise ExchangeOrderError(f"Failed to place market order: {e}") from e

    async def place_limit_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        price: Decimal,
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
        if self.dry_run:
            logger.info(
                "PAPER TRADING: Simulating %s %s %s limit @ %s",
                side,
                quantity,
                symbol,
                price,
            )
            return Order(
                id=f"paper_limit_{int(datetime.now(UTC).timestamp() * 1000)}",
                symbol=symbol,
                side=cast("Literal['BUY', 'SELL']", side),
                type="LIMIT",
                quantity=quantity,
                price=price,
                status=OrderStatus.OPEN,
                timestamp=datetime.now(UTC),
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise ExchangeOrderError("Quantity must be positive")

            if price <= 0:
                raise ExchangeOrderError("Price must be positive")

            if side not in ["BUY", "SELL"]:
                raise ExchangeOrderError(f"Invalid side: {side}")

            # Generate a unique client order ID
            import uuid

            client_order_id = f"ai-bot-{uuid.uuid4().hex[:8]}"

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

            if self._client is None:
                logger.error("Client not initialized")
                return None

            logger.info(
                "Placing %s limit order: %s %s @ %s", side, quantity, symbol, price
            )
            result = await self._retry_request(
                self._client.create_order, client_order_id, **order_data
            )

            # Parse the response
            if result.get("success"):
                order_info = result.get("order", {})
                order_id = order_info.get("order_id")

                order = Order(
                    id=order_id
                    or f"cb_limit_{int(datetime.now(UTC).timestamp() * 1000)}",
                    symbol=symbol,
                    side=cast("Literal['BUY', 'SELL']", side),
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.OPEN,
                    timestamp=datetime.now(UTC),
                    filled_quantity=Decimal(0),
                )

                logger.info("Limit order placed successfully: %s", order_id)
                return order
            error_msg = result.get("error_response", {}).get("message", "Unknown error")
            logger.error("Limit order placement failed: %s", error_msg)
            raise ExchangeOrderError(f"Limit order placement failed: {error_msg}")

        except CoinbaseAPIError as e:
            if "insufficient funds" in str(e).lower():
                raise ExchangeInsufficientFundsError(f"Insufficient funds: {e}") from e
            raise ExchangeOrderError(f"API error: {e}") from e
        except Exception as e:
            logger.exception("Failed to place limit order")
            logger.debug("Limit order error traceback: %s", traceback.format_exc())
            raise ExchangeOrderError(f"Failed to place limit order: {e}") from e

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
            side = cast("Literal['BUY', 'SELL']", "SELL")
        else:  # Short position
            stop_price = current_price * (1 + Decimal(str(sl_pct)))
            side = cast("Literal['BUY', 'SELL']", "BUY")

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
            side = cast("Literal['BUY', 'SELL']", "SELL")
        else:  # Short position
            limit_price = current_price * (1 - Decimal(str(tp_pct)))
            side = cast("Literal['BUY', 'SELL']", "BUY")

        return await self.place_limit_order(
            symbol=base_order.symbol,
            side=side,
            quantity=base_order.quantity,
            price=limit_price,
        )

    async def _place_stop_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        stop_price: Decimal,
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
        if self.dry_run:
            logger.info(
                "PAPER TRADING: Simulating %s %s %s stop @ %s",
                side,
                quantity,
                symbol,
                stop_price,
            )
            return Order(
                id=f"paper_stop_{int(datetime.now(UTC).timestamp() * 1000)}",
                symbol=symbol,
                side=cast("Literal['BUY', 'SELL']", side),
                type="STOP",
                quantity=quantity,
                stop_price=stop_price,
                status=OrderStatus.OPEN,
                timestamp=datetime.now(UTC),
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise ExchangeOrderError("Quantity must be positive")

            if stop_price <= 0:
                raise ExchangeOrderError("Stop price must be positive")

            if side not in ["BUY", "SELL"]:
                raise ExchangeOrderError(f"Invalid side: {side}")

            # Calculate limit price with small buffer
            slippage_pct = settings.trading.slippage_tolerance_pct / 100
            if side == "BUY":
                limit_price = stop_price * (1 + Decimal(str(slippage_pct)))
            else:
                limit_price = stop_price * (1 - Decimal(str(slippage_pct)))

            # Round prices to 2 decimal places for Coinbase futures
            stop_price_float = round(float(stop_price), 2)
            limit_price_float = round(float(limit_price), 2)

            # Generate a unique client order ID
            import uuid

            client_order_id = f"ai-bot-{uuid.uuid4().hex[:8]}"

            # Place stop-limit order
            order_data = {
                "product_id": symbol,
                "side": side,
                "order_configuration": {
                    "stop_limit_stop_limit_gtc": {
                        "base_size": str(quantity),
                        "limit_price": str(limit_price_float),
                        "stop_price": str(stop_price_float),
                        "stop_direction": (
                            "STOP_DIRECTION_STOP_DOWN"
                            if side == "SELL"
                            else "STOP_DIRECTION_STOP_UP"
                        ),
                    }
                },
            }

            if self._client is None:
                logger.error("Client not initialized")
                return None

            logger.info(
                "Placing %s stop order: %s %s stop @ %s, limit @ %s",
                side,
                quantity,
                symbol,
                stop_price_float,
                limit_price_float,
            )
            result = await self._retry_request(
                self._client.create_order, client_order_id, **order_data
            )

            # Parse the response
            if result.get("success"):
                order_info = result.get("order", {})
                order_id = order_info.get("order_id")

                order = Order(
                    id=order_id
                    or f"cb_stop_{int(datetime.now(UTC).timestamp() * 1000)}",
                    symbol=symbol,
                    side=cast("Literal['BUY', 'SELL']", side),
                    type="STOP_LIMIT",
                    quantity=quantity,
                    price=Decimal(str(limit_price_float)),
                    stop_price=Decimal(str(stop_price_float)),
                    status=OrderStatus.OPEN,
                    timestamp=datetime.now(UTC),
                    filled_quantity=Decimal(0),
                )

                logger.info("Stop order placed successfully: %s", order_id)
                return order
            error_msg = result.get("error_response", {}).get("message", "Unknown error")
            logger.error("Stop order placement failed: %s", error_msg)
            raise ExchangeOrderError(f"Stop order placement failed: {error_msg}")

        except CoinbaseAPIError as e:
            if "insufficient funds" in str(e).lower():
                raise ExchangeInsufficientFundsError(f"Insufficient funds: {e}") from e
            raise ExchangeOrderError(f"API error: {e}") from e
        except Exception as e:
            logger.exception("Failed to place stop order")
            logger.debug("Stop order error traceback: %s", traceback.format_exc())
            raise ExchangeOrderError(f"Failed to place stop order: {e}") from e

    async def place_futures_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        leverage: int | None = None,
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
        if self.dry_run:
            # Simulate futures order in dry-run mode
            logger.info(
                "PAPER TRADING FUTURES: Simulating %s %s %s at market (leverage: %sx)",
                side,
                quantity,
                symbol,
                leverage,
            )
            return Order(
                id=f"paper_futures_{int(datetime.now(UTC).timestamp() * 1000)}",
                symbol=symbol,
                side=cast("Literal['BUY', 'SELL']", side),
                type="MARKET",
                quantity=quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(UTC),
                filled_quantity=quantity,
            )

        try:
            # Validate inputs
            if quantity <= 0:
                raise ExchangeOrderError("Quantity must be positive")

            if side not in ["BUY", "SELL"]:
                raise ExchangeOrderError(f"Invalid side: {side}")

            leverage = leverage or self.max_futures_leverage

            # Generate a unique client order ID
            import uuid

            client_order_id = f"ai-bot-{uuid.uuid4().hex[:8]}"

            # Round quantity appropriately
            if symbol.endswith("-CDE"):
                # For futures contracts, quantity must be whole number (contracts)
                rounded_quantity_int = int(quantity)
                rounded_quantity = str(rounded_quantity_int)
            else:
                # For spot/leverage, use decimal precision
                rounded_quantity = str(round(float(quantity), 6))

            # Place the futures order using Coinbase Advanced Trader
            order_data: dict[str, Any] = {
                "product_id": symbol,
                "side": side,
                "order_configuration": {
                    "market_market_ioc": {"base_size": str(rounded_quantity)}
                },
            }

            # Only add leverage for spot symbols, not for actual futures contracts
            if not symbol.endswith("-CDE"):
                order_data["leverage"] = str(leverage)

            # Add reduce_only only if True
            if reduce_only:
                order_data["reduce_only"] = True

            # Don't add portfolio ID for futures - Coinbase handles the routing automatically

            if self._client is None:
                logger.error("Client not initialized")
                return None

            logger.info(
                "Placing FUTURES %s market order: %s %s (leverage: %sx)",
                side,
                quantity,
                symbol,
                leverage,
            )
            logger.debug("Order data: %s", order_data)
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
                        order_id = resp.get("order_id")
                    else:
                        order_id = resp.order_id if hasattr(resp, "order_id") else None

                    order = Order(
                        id=order_id
                        or f"cb_futures_{int(datetime.now(UTC).timestamp() * 1000)}",
                        symbol=symbol,
                        side=cast("Literal['BUY', 'SELL']", side),
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(UTC),
                        filled_quantity=Decimal(0),
                    )

                    logger.info(
                        "Futures market order placed successfully: %s", order_id
                    )
                    return order
                if hasattr(result, "order_id"):
                    # Fallback for old format
                    order_id = result.order_id

                    order = Order(
                        id=order_id
                        or f"cb_futures_{int(datetime.now(UTC).timestamp() * 1000)}",
                        symbol=symbol,
                        side=cast("Literal['BUY', 'SELL']", side),
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(UTC),
                        filled_quantity=Decimal(0),
                    )

                    logger.info(
                        "Futures market order placed successfully: %s", order_id
                    )
                    return order
                if isinstance(result, dict) and result.get("success"):
                    order_info = result.get("order", {})
                    order_id = order_info.get("order_id")

                    order = Order(
                        id=order_id
                        or f"cb_futures_{int(datetime.now(UTC).timestamp() * 1000)}",
                        symbol=symbol,
                        side=cast("Literal['BUY', 'SELL']", side),
                        type="MARKET",
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(UTC),
                        filled_quantity=Decimal(0),
                    )

                    logger.info(
                        "Futures market order placed successfully: %s", order_id
                    )
                    return order
                error_msg = str(result) if result else "Unknown error"
                logger.error("Futures order placement failed: %s", error_msg)
                raise ExchangeOrderError(f"Futures order placement failed: {error_msg}")
            except AttributeError as e:
                logger.exception("Error parsing order response")
                logger.exception(
                    "Response type: %s, Response: %s",
                    type(result).__name__,
                    "[response data]",
                )
                raise ExchangeOrderError(f"Failed to parse order response: {e}") from e

        except CoinbaseAPIError as e:
            if (
                "insufficient funds" in str(e).lower()
                or "insufficient margin" in str(e).lower()
            ):
                raise ExchangeInsufficientFundsError(
                    f"Insufficient margin for futures order: {e}"
                ) from e
            raise ExchangeOrderError(f"Futures API error: {e}") from e
        except Exception as e:
            logger.exception("Failed to place futures market order")
            logger.debug(
                "Futures market order error traceback: %s", traceback.format_exc()
            )
            raise ExchangeOrderError(
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

        Raises:
            BalanceServiceUnavailableError: When Coinbase API is unavailable
            BalanceTimeoutError: When request times out
            BalanceValidationError: When balance data is invalid
            BalanceRetrievalError: For other balance retrieval issues
        """
        if self.dry_run:
            # Return mock balance for paper trading with proper normalization
            mock_balance = self._normalize_balance(Decimal("10000.00"))
            logger.debug("PAPER TRADING: Returning mock balance $%s", mock_balance)
            return mock_balance

        try:
            if self.enable_futures and account_type == AccountType.CFM:
                futures_balance = await self.get_futures_balance()
                return self._normalize_balance(futures_balance)
            if account_type == AccountType.CBI:
                spot_balance = await self.get_spot_balance()
                return self._normalize_balance(spot_balance)
            # Get total balance across all accounts
            spot_balance = await self.get_spot_balance()
            if self.enable_futures:
                futures_balance = await self.get_futures_balance()
                total_balance = spot_balance + futures_balance
                return self._normalize_balance(total_balance)
            return self._normalize_balance(spot_balance)

        except (
            BalanceRetrievalError,
            BalanceServiceUnavailableError,
            BalanceTimeoutError,
            BalanceValidationError,
        ):
            # Re-raise specific balance exceptions with context
            logger.exception("Balance operation failed")
            raise
        except Exception as e:
            logger.exception("Failed to get account balance")
            # Wrap generic exceptions in BalanceRetrievalError
            raise BalanceRetrievalError(
                f"Failed to retrieve account balance: {e}",
                account_type=str(account_type) if account_type else None,
                balance_context={
                    "exchange": "coinbase",
                    "enable_futures": self.enable_futures,
                    "dry_run": self.dry_run,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def get_spot_balance(self) -> Decimal:
        """
        Get spot account (CBI) balance in USD.

        Returns:
            Spot account balance in USD

        Raises:
            BalanceServiceUnavailableError: When client not initialized or service unavailable
            BalanceValidationError: When balance data is invalid
            BalanceRetrievalError: For other balance retrieval issues
        """
        try:
            if self._client is None:
                raise BalanceServiceUnavailableError(
                    "Coinbase client not initialized",
                    service_name="coinbase_client",
                )

            # Use retry request which handles API-level errors
            accounts_data = await self._retry_request(self._client.get_accounts)

            # Validate response structure
            if not isinstance(accounts_data, dict) or "accounts" not in accounts_data:
                raise BalanceValidationError(
                    "Invalid accounts data structure from Coinbase API",
                    invalid_value=type(accounts_data).__name__,
                    validation_rule="accounts_data_structure",
                    account_type="CBI",
                )

            # Find USD balance in spot accounts
            total_balance = Decimal(0)
            for account in accounts_data.get("accounts", []):
                if (
                    account.get("currency") == "USD"
                    and account.get("type") != "futures"
                ):
                    balance_data = account.get("available_balance", {})
                    balance_value = balance_data.get("value", "0")

                    # Validate balance value format
                    try:
                        account_balance = Decimal(str(balance_value))
                        total_balance += account_balance
                    except (
                        ValueError,
                        TypeError,
                        decimal.InvalidOperation,
                    ) as decimal_err:
                        raise BalanceValidationError(
                            f"Invalid balance value format in spot account: {balance_value}",
                            invalid_value=balance_value,
                            validation_rule="decimal_conversion",
                            account_type="CBI",
                        ) from decimal_err

            # Validate final balance
            if total_balance < Decimal(0):
                logger.warning("Retrieved negative spot balance: $%s", total_balance)

            logger.debug("Retrieved spot USD balance: %s", total_balance)
        except (
            BalanceRetrievalError,
            BalanceServiceUnavailableError,
            BalanceValidationError,
        ):
            # Re-raise specific balance exceptions with context
            logger.exception("Spot balance operation failed")
            raise
        except Exception as e:
            logger.exception("Failed to get spot balance")
            # Check for specific API error types
            if hasattr(e, "status_code") and e.status_code == 503:
                raise BalanceServiceUnavailableError(
                    f"Coinbase API service unavailable: {e}",
                    service_name="coinbase_api",
                    status_code=e.status_code,
                ) from e

            raise BalanceRetrievalError(
                f"Failed to retrieve spot balance: {e}",
                account_type="CBI",
                balance_context={
                    "exchange": "coinbase",
                    "account_type": "spot",
                    "error_type": type(e).__name__,
                },
            ) from e
        else:
            return total_balance

    async def get_futures_balance(self) -> Decimal:
        """
        Get futures account (CFM) balance in USD.

        Returns:
            Futures account balance in USD

        Raises:
            BalanceServiceUnavailableError: When client not initialized or service unavailable
            BalanceValidationError: When balance data is invalid
            BalanceRetrievalError: For other balance retrieval issues
        """
        try:
            if self._client is None:
                raise BalanceServiceUnavailableError(
                    "Coinbase client not initialized",
                    service_name="coinbase_client",
                )

            # Get FCM balance summary with retry mechanism
            balance_response = await self._retry_request(
                self._client.get_fcm_balance_summary
            )

            # Validate response structure and extract balance
            futures_balance = None

            # Handle response object format
            if hasattr(balance_response, "balance_summary"):
                balance_data = balance_response.balance_summary
                if hasattr(balance_data, "cfm_usd_balance"):
                    balance_value = balance_data.cfm_usd_balance.get("value", "0")
                    try:
                        futures_balance = Decimal(str(balance_value))
                    except (
                        ValueError,
                        TypeError,
                        decimal.InvalidOperation,
                    ) as decimal_err:
                        raise BalanceValidationError(
                            f"Invalid CFM balance value format: {balance_value}",
                            invalid_value=balance_value,
                            validation_rule="decimal_conversion",
                            account_type="CFM",
                        ) from decimal_err
                else:
                    raise BalanceValidationError(
                        "Missing cfm_usd_balance field in balance summary",
                        invalid_value="missing_field",
                        validation_rule="response_structure",
                        account_type="CFM",
                    )
            elif isinstance(balance_response, dict):
                # Fallback for dict format
                cash_balance_data = balance_response.get("cash_balance", {})
                balance_value = cash_balance_data.get("value", "0")
                try:
                    futures_balance = Decimal(str(balance_value))
                except (ValueError, TypeError, decimal.InvalidOperation) as decimal_err:
                    raise BalanceValidationError(
                        f"Invalid cash balance value format: {balance_value}",
                        invalid_value=balance_value,
                        validation_rule="decimal_conversion",
                        account_type="CFM",
                    ) from decimal_err
            else:
                raise BalanceValidationError(
                    f"Invalid balance response format: {type(balance_response)}",
                    invalid_value=type(balance_response).__name__,
                    validation_rule="response_format",
                    account_type="CFM",
                )

            if futures_balance is None:
                raise BalanceValidationError(
                    "Unable to extract balance from response",
                    invalid_value="null_balance",
                    validation_rule="balance_extraction",
                    account_type="CFM",
                )

            # Validate balance value
            if futures_balance < Decimal(0):
                logger.warning(
                    "Retrieved negative futures balance: $%s", futures_balance
                )

            logger.debug("Retrieved futures USD balance: %s", futures_balance)
        except (
            BalanceRetrievalError,
            BalanceServiceUnavailableError,
            BalanceValidationError,
        ):
            # Re-raise specific balance exceptions with context
            logger.exception("Futures balance operation failed")
            raise
        except Exception as e:
            logger.exception("Failed to get futures balance")
            # Try fallback method before raising final exception
            try:
                return await self._get_futures_balance_fallback()
            except Exception as fallback_error:
                logger.exception("Fallback futures balance also failed")
                raise BalanceRetrievalError(
                    f"Failed to retrieve futures balance (primary and fallback): {e}",
                    account_type="CFM",
                    balance_context={
                        "exchange": "coinbase",
                        "account_type": "futures",
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error),
                        "error_type": type(e).__name__,
                    },
                ) from e
        else:
            return futures_balance

    async def _get_futures_balance_fallback(self) -> Decimal:
        """
        Fallback method to get futures balance using regular accounts API.

        Returns:
            Futures account balance in USD

        Raises:
            BalanceServiceUnavailableError: When client not initialized
            BalanceValidationError: When balance data is invalid
            BalanceRetrievalError: For other balance retrieval issues
        """
        try:
            if self._client is None:
                raise BalanceServiceUnavailableError(
                    "Coinbase client not initialized for fallback balance retrieval",
                    service_name="coinbase_client",
                )

            accounts_data = await self._retry_request(self._client.get_accounts)

            # Validate response structure
            if not isinstance(accounts_data, dict) or "accounts" not in accounts_data:
                raise BalanceValidationError(
                    "Invalid accounts data structure from Coinbase fallback API",
                    invalid_value=type(accounts_data).__name__,
                    validation_rule="accounts_data_structure",
                    account_type="CFM",
                )

            total_balance = Decimal(0)
            for account in accounts_data.get("accounts", []):
                if (
                    account.get("currency") == "USD"
                    and account.get("type") == "futures"
                ):
                    balance_data = account.get("available_balance", {})
                    balance_value = balance_data.get("value", "0")

                    # Validate balance value format
                    try:
                        account_balance = Decimal(str(balance_value))
                        total_balance += account_balance
                    except (
                        ValueError,
                        TypeError,
                        decimal.InvalidOperation,
                    ) as decimal_err:
                        raise BalanceValidationError(
                            f"Invalid balance value format in futures account (fallback): {balance_value}",
                            invalid_value=balance_value,
                            validation_rule="decimal_conversion",
                            account_type="CFM",
                        ) from decimal_err

        except (
            BalanceServiceUnavailableError,
            BalanceValidationError,
        ):
            # Re-raise specific balance exceptions with context
            logger.exception("Fallback futures balance operation failed")
            raise
        except Exception as e:
            logger.exception("Fallback futures balance failed")
            raise BalanceRetrievalError(
                f"Fallback futures balance retrieval failed: {e}",
                account_type="CFM",
                balance_context={
                    "exchange": "coinbase",
                    "account_type": "futures_fallback",
                    "error_type": type(e).__name__,
                },
            ) from e
        else:
            return total_balance

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
            if self._client is None:
                return None

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
                min_cash_transfer_amount=Decimal(100),
                max_cash_transfer_amount=Decimal(10000),
                max_leverage=self.max_futures_leverage,
                max_position_size=total_balance
                * Decimal(str(settings.trading.max_size_pct / 100)),
                current_positions_count=len(await self.get_futures_positions()),
                timestamp=datetime.now(UTC),
            )

            self._futures_account_info = account_info
        except Exception:
            logger.exception("Failed to get futures account info")
            return None
        else:
            return account_info

    async def get_margin_info(self) -> MarginInfo:
        """
        Get futures margin information and health status.

        Returns:
            MarginInfo object with current margin status
        """
        try:
            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

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

            self._last_margin_check = datetime.now(UTC)
        except Exception:
            logger.exception("Failed to get margin info")
            # Return default margin info
            return MarginInfo(
                total_margin=Decimal(0),
                available_margin=Decimal(0),
                used_margin=Decimal(0),
                maintenance_margin=Decimal(0),
                initial_margin=Decimal(0),
                margin_ratio=0.0,
                health_status=MarginHealthStatus.HEALTHY,
                liquidation_threshold=Decimal(0),
                intraday_margin_requirement=Decimal(0),
                overnight_margin_requirement=Decimal(0),
            )
        else:
            return margin_info

    async def transfer_cash_to_futures(
        self,
        amount: Decimal,
        reason: Literal["MARGIN_CALL", "MANUAL", "AUTO_REBALANCE"] = "MANUAL",
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
            logger.info("Transferring $%s from spot to futures for %s", amount, reason)

            if self.dry_run:
                logger.info("PAPER TRADING: Simulating transfer $%s CBI -> CFM", amount)
                return True

            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

            # Use futures sweep API to transfer funds
            try:
                # Cancel any pending sweeps first
                try:
                    await self._retry_request(self._client.cancel_pending_futures_sweep)
                    logger.debug("Cancelled pending futures sweep")
                except Exception as e:
                    # Log but ignore if no pending sweep exists
                    logger.debug("No pending futures sweep to cancel: %s", e)

                # Schedule a new sweep
                await self._retry_request(
                    self._client.schedule_futures_sweep, usd_amount=str(amount)
                )

                logger.info("Successfully scheduled futures sweep for $%s", amount)

                # Wait a bit for the sweep to process
                await asyncio.sleep(3)

                # Check if transfer completed
                balance_resp = await self._retry_request(
                    self._client.get_futures_balance_summary
                )
                if hasattr(balance_resp, "balance_summary"):
                    cfm_balance = Decimal(
                        balance_resp.balance_summary.cfm_usd_balance["value"]
                    )
                    if cfm_balance > 0:
                        logger.info("Transfer completed. CFM balance: $%s", cfm_balance)
                        return True

                logger.warning("Transfer scheduled but not yet completed")
            except Exception:
                logger.exception("Failed to schedule futures sweep")
                return False
            else:
                return True  # Return true since sweep was scheduled

        except Exception:
            logger.exception("Failed to transfer cash to futures")
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
            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

            # Get positions from FCM account
            positions_response = await self._retry_request(
                self._client.list_futures_positions
            )

            # Handle response object format
            if hasattr(positions_response, "positions"):
                positions_list = positions_response.positions
            else:
                # Fallback for dict format
                positions_list = positions_response.get("positions", [])

            positions = []
            for pos_data in positions_list:
                # Handle object attributes
                if hasattr(pos_data, "product_id"):
                    product_id = pos_data.product_id
                    num_contracts = int(pos_data.number_of_contracts)
                    side = pos_data.side
                    avg_entry_price = Decimal(str(pos_data.avg_entry_price))
                    unrealized_pnl = Decimal(
                        str(getattr(pos_data, "unrealized_pnl", "0"))
                    )
                else:
                    # Handle dict format
                    product_id = pos_data.get("product_id")
                    num_contracts = int(pos_data.get("number_of_contracts", 0))
                    side = pos_data.get("side", "")
                    avg_entry_price = Decimal(str(pos_data.get("avg_entry_price", "0")))
                    unrealized_pnl = Decimal(str(pos_data.get("unrealized_pnl", "0")))

                # Skip if symbol filter provided and doesn't match
                if symbol and product_id != symbol:
                    continue

                if num_contracts > 0:
                    # Convert contracts to ETH size (1 contract = 0.1 ETH for nano futures)
                    contract_size = Decimal("0.1")
                    size = num_contracts * contract_size

                    position = Position(
                        symbol=product_id,
                        side=side,
                        size=size,
                        entry_price=avg_entry_price,
                        unrealized_pnl=unrealized_pnl,
                        realized_pnl=Decimal(0),  # Not provided in API
                        timestamp=datetime.now(UTC),
                        is_futures=True,
                        leverage=self.max_futures_leverage,  # Actual leverage not in response
                        margin_used=Decimal(0),  # Calculate from position if needed
                        liquidation_price=Decimal(0),  # Not provided directly
                        margin_health=MarginHealthStatus.HEALTHY,  # Would calculate based on margin
                    )
                    positions.append(position)

            logger.debug("Retrieved %d futures positions", len(positions))
        except Exception:
            logger.exception("Failed to get futures positions")
            # Fallback to regular positions API
            return await self.get_positions(symbol)
        else:
            return positions

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """
        if self.dry_run:
            # Return empty positions for dry run
            return []

        try:
            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

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
                            0
                        ),  # Would need market data to calculate
                        realized_pnl=Decimal(0),
                        timestamp=datetime.now(UTC),
                    )
                    positions.append(position)

            logger.debug("Retrieved %d positions", len(positions))
        except Exception as e:
            logger.exception("Failed to get positions")
            raise ExchangeConnectionError(f"Failed to get positions: {e}") from e
        else:
            return positions

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info("PAPER TRADING: Simulating cancel order %s", order_id)
            return True

        try:
            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

            logger.info("Cancelling order: %s", order_id)
            result = await self._retry_request(
                self._client.cancel_orders, order_ids=[order_id]
            )

            # Validate response before processing
            if not self._response_validator.validate_order_cancellation_response(
                result
            ):
                logger.error("Order cancellation response validation failed")
                return False

            # Check if cancellation was successful
            cancelled_orders = result.get("results", [])
            for cancelled in cancelled_orders:
                if cancelled.get("order_id") == order_id and cancelled.get("success"):
                    logger.info("Order %s cancelled successfully", order_id)
                    return True

            logger.warning("Order %s cancellation may have failed", order_id)
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False
        else:
            return False

    async def cancel_all_orders(
        self, symbol: str | None = None, _status: str | None = None
    ) -> bool:
        """
        Cancel all open orders, optionally filtered by symbol.

        Args:
            symbol: Optional trading symbol filter
            status: Optional order status filter (not used for Coinbase)

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(
                "PAPER TRADING: Simulating cancel all orders%s",
                " for " + symbol if symbol else "",
            )
            return True

        try:
            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

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
                    "No open orders to cancel%s", " for " + symbol if symbol else ""
                )
                return True

            logger.info(
                "Cancelling %s orders%s",
                len(orders_to_cancel),
                " for " + symbol if symbol else "",
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
                "Successfully cancelled %s/%s orders",
                successful_cancellations,
                len(orders_to_cancel),
            )
            return successful_cancellations == len(orders_to_cancel)

        except Exception:
            logger.exception("Failed to cancel orders")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        """
        Get status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatus or None
        """
        if self.dry_run:
            # Return mock status for dry run
            return OrderStatus.FILLED

        try:
            # Check if client is available
            if self._client is None:
                raise ExchangeConnectionError("Exchange client not initialized")

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
            logger.debug("Order %s status: %s -> %s", order_id, status_str, status)

        except Exception:
            logger.exception("Failed to get order status for %s", order_id)
            return None
        else:
            return status

    def is_connected(self) -> bool:
        """
        Check if client is connected and authenticated.

        Returns:
            True if connected
        """
        return self._connected

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
            "connected": self._connected,
            "sandbox": self.sandbox,
            "auth_method": self.auth_method,
            "has_credentials": has_credentials,
            "dry_run": self.dry_run,
            "trading_mode": ("PAPER TRADING" if self.dry_run else "LIVE TRADING"),
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
                size=Decimal(0),
                timestamp=datetime.now(UTC),
            )
        )

    async def _place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> Order | None:
        """Legacy method for backward compatibility."""
        return await self.place_market_order(
            symbol, cast("Literal['BUY', 'SELL']", side), quantity
        )

    async def _place_limit_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> Order | None:
        """Legacy method for backward compatibility."""
        return await self.place_limit_order(
            symbol, cast("Literal['BUY', 'SELL']", side), quantity, price
        )

    @property
    def supports_futures(self) -> bool:
        """Check if exchange supports futures trading."""
        return self.enable_futures

    @property
    def is_decentralized(self) -> bool:
        """Coinbase is a centralized exchange."""
        return False

    async def get_monthly_volume(self, force_refresh: bool = False) -> Decimal:
        """
        Get 30-day trading volume for fee tier calculation.

        Args:
            force_refresh: Force refresh of volume data

        Returns:
            Monthly trading volume in USD
        """
        if self.dry_run:
            # Return mock volume for paper trading
            return Decimal(0)  # Basic tier

        # Check if we need to refresh volume
        now = datetime.now(UTC)
        if (
            not force_refresh
            and self._last_volume_check
            and now - self._last_volume_check < self._volume_check_interval
        ):
            return self._monthly_volume

        try:
            # Calculate date range for 30 days
            end_date = now
            start_date = end_date - timedelta(days=30)

            # Get fills/trades for the period
            # Note: This is a simplified implementation. The actual Coinbase API
            # may require pagination for large trade histories
            volume = Decimal(0)

            try:
                # Check if client is available
                if self._client is None:
                    raise ExchangeConnectionError("Exchange client not initialized")

                # Try to get fills from the API
                fills_response = await self._retry_request(
                    self._client.get_fills,
                    start_sequence_timestamp=start_date.isoformat(),
                    end_sequence_timestamp=end_date.isoformat(),
                )

                if isinstance(fills_response, dict) and "fills" in fills_response:
                    for fill in fills_response["fills"]:
                        # Calculate volume from each fill
                        if fill.get("liquidity_indicator") in ["MAKER", "TAKER"]:
                            size = Decimal(str(fill.get("size", "0")))
                            price = Decimal(str(fill.get("price", "0")))
                            volume += size * price

            except Exception as e:
                logger.warning("Failed to get fills for volume calculation: %s", e)
                # Fallback to basic tier
                volume = Decimal(0)

            self._monthly_volume = volume
            self._last_volume_check = now

            # Update fee calculator with current volume
            from bot.fee_calculator import fee_calculator

            fee_calculator.update_volume_tier(float(volume))

            # Determine fee tier
            tier = "Basic"
            if volume >= 250000000:
                tier = "VIP Ultra"
            elif volume >= 75000000:
                tier = "VIP"
            elif volume >= 15000000:
                tier = "Advanced"
            elif volume >= 1000000:
                tier = "Pro"
            elif volume >= 100000:
                tier = "Plus"
            elif volume >= 50000:
                tier = "Standard"
            elif volume >= 10000:
                tier = "Active"

            logger.info(
                "Updated monthly trading volume: $%,.2f (Tier: %s)", volume, tier
            )
        except Exception:
            logger.exception("Failed to get monthly volume")
            return self._monthly_volume
        else:
            return volume

    def get_current_fee_rates(self) -> dict[str, float]:
        """
        Get current fee rates based on volume tier.

        Returns:
            Dictionary with maker and taker fee rates
        """
        from bot.fee_calculator import fee_calculator

        return {
            "maker": fee_calculator.maker_fee_rate,
            "taker": fee_calculator.taker_fee_rate,
            "futures": fee_calculator.futures_fee_rate,
            "volume": float(self._monthly_volume),
            "tier": fee_calculator.current_tier,
        }

    @property
    def enable_futures(self) -> bool:
        """Check if futures trading is enabled for this exchange instance."""
        return self._enable_futures

    # Functional Programming Interface

    def get_functional_adapter(self) -> "CoinbaseExchangeAdapter | None":
        """
        Get the functional adapter for side-effect-free operations.

        Returns:
            CoinbaseExchangeAdapter instance or None if not available
        """
        return self._fp_adapter

    def supports_functional_operations(self) -> bool:
        """
        Check if functional programming operations are supported.

        Returns:
            True if functional adapter is available
        """
        return self._fp_adapter is not None

    async def place_order_functional(self, order: Order) -> dict[str, Any] | None:
        """
        Place order using functional adapter (demonstration method).

        Args:
            order: Order to place

        Returns:
            Order result or None if functional adapter not available
        """
        if not self._fp_adapter:
            logger.warning(
                "Functional adapter not available, falling back to imperative method"
            )
            return None

        # This demonstrates how the functional adapter can be used
        # In practice, you'd convert the Order to the FP type first
        try:
            from bot.fp.adapters.type_converters import current_order_to_fp_order

            fp_order = current_order_to_fp_order(order)
            result = self._fp_adapter.place_order_impl(fp_order)

            # Execute the IOEither and handle the result
            either_result = result.run()
            if either_result.is_right():
                return either_result.value
            logger.error("Functional order placement failed: %s", either_result.value)
            return None
        except Exception as e:
            logger.error("Error in functional order placement: %s", e)
            return None
