"""
Bluefin exchange client for perpetual futures trading on Sui network.

This module provides integration with Bluefin DEX, a decentralized perpetual
futures exchange built on the Sui blockchain.
"""

import asyncio
import decimal
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, NoReturn, cast

from bot.config import settings
from bot.trading_types import AccountType, Order, OrderStatus, Position, TradeAction
from bot.utils.symbol_utils import (
    get_testnet_symbol_fallback,
    is_bluefin_symbol_supported,
    to_bluefin_perp,
    validate_symbol,
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
from .bluefin_fee_calculator import BluefinFeeCalculator, BluefinFees

# Import order signature system
try:
    from .bluefin_order_signature import (
        BluefinOrderSignatureManager,
        OrderSignatureError,
    )

    ORDER_SIGNATURE_AVAILABLE = True
except ImportError:
    # Fallback if signature system is not available
    class BluefinOrderSignatureManager:  # type: ignore[misc]
        def __init__(self, private_key_hex: str):
            pass

        def sign_market_order(self, *args, **kwargs):
            return {}

        def sign_limit_order(self, *args, **kwargs):
            return {}

        def sign_stop_order(self, *args, **kwargs):
            return {}

    class OrderSignatureError(Exception):  # type: ignore[misc]
        """Fallback exception if signature system not available."""

    ORDER_SIGNATURE_AVAILABLE = False

# Use the service client instead of direct SDK
try:
    from .bluefin_client import BluefinServiceClient, BluefinServiceConnectionError
except ImportError:
    # Fallback if BluefinServiceClient is not available
    class BluefinServiceClient:  # type: ignore[misc]
        def __init__(self):
            pass

    class BluefinServiceConnectionError(Exception):  # type: ignore[misc]
        """Fallback exception if client not available."""


# Functional programming imports
try:
    from bot.fp.adapters.bluefin_adapter import BluefinExchangeAdapter
    from bot.fp.adapters.exchange_adapter import register_exchange_adapter

    FP_ADAPTERS_AVAILABLE = True
except ImportError:
    # Fallback when FP adapters are not available
    FP_ADAPTERS_AVAILABLE = False
    BluefinExchangeAdapter = None  # type: ignore[assignment,misc]


# Import monitoring components
try:
    from bot.monitoring.balance_alerts import get_balance_alert_manager
    from bot.monitoring.balance_metrics import (
        get_balance_metrics_collector,
        record_operation_complete,
        record_operation_start,
        record_timeout,
    )

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


BLUEFIN_AVAILABLE = True  # Always available via service


# Mock classes for compatibility
class OrderSide:
    BUY = "BUY"
    SELL = "SELL"


class OrderType:
    MARKET = "MARKET"
    LIMIT = "LIMIT"


logger = logging.getLogger(__name__)


class BluefinRateLimiter:
    """Rate limiter for Bluefin API requests."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
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
                    logger.debug("Rate limit reached, waiting %ss", f"{wait_time:.2f}")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()

            self.requests.append(now)
            return None


class BluefinClient(BaseExchange):
    """
    Bluefin exchange client for perpetual futures trading.

    Provides methods for trading perpetual futures on Bluefin DEX,
    including order placement, position management, and market data.
    """

    def __init__(
        self,
        private_key: str | None = None,
        network: str = "mainnet",
        service_url: str | None = None,
        dry_run: bool = True,
    ):
        """
        Initialize the Bluefin client.

        Args:
            private_key: Sui wallet private key
            network: Network to connect to ('mainnet' or 'testnet')
            service_url: URL of the Bluefin service container
            dry_run: Whether to run in paper trading mode
        """
        # Enhanced live trading mode detection with environment variable overrides
        import os

        actual_dry_run = dry_run

        # Check multiple sources for live trading mode
        env_dry_run = os.getenv("SYSTEM__DRY_RUN", "").lower()
        force_live_mode = os.getenv("BLUEFIN_FORCE_LIVE_MODE", "").lower() == "true"

        if env_dry_run == "false" or force_live_mode:
            actual_dry_run = False
            logger.warning(
                "🚨 LIVE TRADING MODE ENABLED - Trading with REAL MONEY on Bluefin DEX"
            )
            logger.warning("🚨 Trading SUI-PERP with configured credentials")
        elif hasattr(settings.system, "dry_run") and not settings.system.dry_run:
            actual_dry_run = False
            logger.warning(
                "🚨 LIVE TRADING MODE ENABLED via settings - Trading with REAL MONEY"
            )
        else:
            logger.info("📊 Paper Trading Mode - Safe simulation mode enabled")

        super().__init__(actual_dry_run)

        # Extract private key string value, handling SecretStr if needed
        if private_key:
            # If passed directly as a parameter, use it
            self.private_key: str | None = private_key
        elif settings.exchange.bluefin_private_key:
            # Extract from SecretStr settings
            self.private_key = settings.exchange.bluefin_private_key.get_secret_value()
        else:
            self.private_key = None

        # Validate private key for live trading
        if not actual_dry_run and not self.private_key:
            raise ExchangeAuthError(
                "Private key required for live trading. Set EXCHANGE__BLUEFIN_PRIVATE_KEY "
                "environment variable with your Sui wallet private key."
            )
        # Store the network name for display
        self.network_name = network

        # Use service client instead of direct SDK
        # Connect to the Bluefin SDK service container
        # Use provided service_url or fall back to environment variable
        if service_url is None:
            service_url = os.getenv(
                "BLUEFIN_SERVICE_URL", "http://bluefin-service:8080"
            )
        api_key = os.getenv("BLUEFIN_SERVICE_API_KEY")
        self._service_client = BluefinServiceClient(service_url, api_key)

        # Rate limiting
        self._rate_limiter = BluefinRateLimiter(
            max_requests=settings.exchange.rate_limit_requests
            * 3,  # Bluefin has higher limits
            window_seconds=settings.exchange.rate_limit_window_seconds,
        )

        # Cached data
        self._account_address: str | None = None
        self._leverage_settings: dict[str, Any] = {}
        self._contract_info: dict[str, Any] = {}

        # Market making state tracking
        self._active_orders: dict[str, list[Order]] = {}  # symbol -> list of orders
        self._order_callbacks: dict[str, callable] = {}  # order_id -> callback
        self._order_book_cache: dict[str, dict] = {}  # symbol -> order book data

        # Initialize fee calculator for market making
        self._fee_calculator = BluefinFeeCalculator()

        # Initialize order signature manager for live trading
        self._signature_manager: BluefinOrderSignatureManager | None = None
        if not actual_dry_run and self.private_key and ORDER_SIGNATURE_AVAILABLE:
            try:
                self._signature_manager = BluefinOrderSignatureManager(self.private_key)
                logger.info("✅ Order signature system initialized for live trading")
            except OrderSignatureError as e:
                logger.exception(
                    "❌ Failed to initialize order signature system: %s", e
                )
                raise ExchangeAuthError(
                    f"Order signature initialization failed: {e}"
                ) from e
        elif not actual_dry_run and not ORDER_SIGNATURE_AVAILABLE:
            logger.warning(
                "⚠️ Order signature system not available - install cryptography package"
            )
        else:
            logger.info("📊 Paper trading mode - order signatures not required")

        # Initialize monitoring
        self.monitoring_enabled = MONITORING_AVAILABLE
        if self.monitoring_enabled:
            try:
                self.metrics_collector = get_balance_metrics_collector()
                self.alert_manager = get_balance_alert_manager()
                logger.info("✅ Bluefin exchange monitoring enabled")
            except Exception as e:
                logger.warning("Failed to initialize Bluefin monitoring: %s", e)
                self.monitoring_enabled = False

        # Initialize functional adapter for side-effect-free operations
        self._fp_adapter: BluefinExchangeAdapter | None = None
        if FP_ADAPTERS_AVAILABLE:
            try:
                self._fp_adapter = BluefinExchangeAdapter(self)
                # Register with unified adapter system
                register_exchange_adapter("bluefin", self._fp_adapter)
                logger.debug("✅ Functional adapter initialized for Bluefin")
            except Exception as e:
                logger.warning("Failed to initialize functional adapter: %s", e)
                self._fp_adapter = None

        logger.info(
            "Initialized BluefinClient (network=%s, service_url=%s, dry_run=%s)",
            network,
            service_url,
            dry_run,
        )

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
                    "Invalid balance amount: %s (type: %s)", amount, type(amount)
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
                    "Invalid crypto amount: %s (type: %s)", amount, type(amount)
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

    async def validate_symbol_exists(self, symbol: str) -> bool:
        """
        Validate that a symbol exists on Bluefin before making API calls.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol exists, False otherwise
        """
        try:
            # Convert to Bluefin format first
            bluefin_symbol = self._convert_symbol(symbol)

            # Check against known supported symbols
            if not is_bluefin_symbol_supported(bluefin_symbol, self.network_name):
                logger.error(
                    "Symbol %s not supported on %s", bluefin_symbol, self.network_name
                )
                return False

            # For additional validation, we could make an API call to get ticker data
            # but we'll skip this for now to avoid unnecessary API calls
            if self.dry_run:
                # In paper trading, accept all converted symbols
                logger.debug("Paper trading: accepting symbol %s", bluefin_symbol)
                return True
            # For live trading, we rely on the supported symbols list
            # In the future, we could add a ticker API call here for real-time validation
            logger.info("Symbol %s validated for %s", bluefin_symbol, self.network_name)

        except Exception:
            logger.exception("Symbol validation failed for %s", symbol)
            return False
        else:
            return True

    async def connect(self) -> bool:
        """
        Connect and authenticate with Bluefin via service.

        Returns:
            True if connection successful (or gracefully degraded)
        """
        try:
            # Connect to Bluefin service
            connected = False
            service_error_msg = None

            if (
                self._service_client
                and hasattr(self._service_client, "connect")
                and callable(self._service_client.connect)
            ):
                try:
                    logger.info(
                        "Attempting to connect to Bluefin service...",
                        extra={
                            "network": self.network_name,
                            "dry_run": self.dry_run,
                        },
                    )
                    connected = await self._service_client.connect()
                except BluefinServiceConnectionError as e:
                    service_error_msg = str(e)
                    logger.warning(
                        "Bluefin service connection failed (this is optional): %s",
                        service_error_msg,
                        extra={
                            "error_type": "connection_error",
                            "can_continue": True,
                            "dry_run": self.dry_run,
                        },
                    )
                    connected = False
                except Exception as e:
                    service_error_msg = str(e)
                    logger.warning(
                        "Unexpected error connecting to Bluefin service: %s",
                        service_error_msg,
                        extra={
                            "error_type": type(e).__name__,
                            "can_continue": True,
                            "dry_run": self.dry_run,
                        },
                    )
                    connected = False
            else:
                logger.debug("Service client not available or missing connect method")
                service_error_msg = "Service client not initialized"

            if connected:
                logger.info(
                    "✅ Successfully connected to Bluefin DEX via service",
                    extra={"network": self.network_name, "service_available": True},
                )
                self._connected = True
                self._last_health_check = datetime.now(UTC)
                await self._init_client()
                return True

            # Handle graceful degradation
            if self.dry_run:
                logger.info(
                    "📋 PAPER TRADING MODE: Continuing without Bluefin service.\n"
                    "   • All trades will be simulated locally\n"
                    "   • Market data will be fetched if available\n"
                    "   • No real positions or orders will be created\n"
                    "   Service issue: %s",
                    service_error_msg or "Service unavailable",
                    extra={"mode": "paper_trading", "service_required": False},
                )
                self._connected = True  # Allow dry run to continue
                self._last_health_check = datetime.now(UTC)
                await self._init_client()
                return True

            # Live mode warning
            logger.warning(
                "⚠️  LIVE TRADING MODE - Bluefin service unavailable!\n"
                "   • The Bluefin service container appears to be down\n"
                "   • Position queries and order execution WILL NOT WORK\n"
                "   • Consider switching to paper trading mode (DRY_RUN=true)\n"
                "   Service issue: %s\n"
                "   To fix: Ensure bluefin-service container is running",
                service_error_msg or "Service unavailable",
                extra={
                    "mode": "live_trading",
                    "service_required": True,
                    "recommendation": "switch_to_paper_trading",
                },
            )

            # In live mode, still allow initialization but with clear warnings
            self._connected = True
            self._last_health_check = datetime.now(UTC)
            await self._init_client()
            return True

        except Exception as e:
            logger.exception(
                "Unexpected error during Bluefin connection setup",
                extra={"error_type": type(e).__name__, "dry_run": self.dry_run},
            )
            # Only raise in live mode if it's a critical error
            if not self.dry_run and "auth" in str(e).lower():
                raise ExchangeConnectionError(f"Authentication failed: {e}") from e

            # Otherwise allow graceful degradation
            self._connected = True
            self._last_health_check = datetime.now(UTC)
            return True

    async def _init_client(self) -> None:
        """Initialize the Bluefin client."""
        try:
            # Verify service client is available
            if not self._service_client:
                logger.warning("Service client not available - using fallback mode")
                return

            # Get and cache contract information
            await self._load_contract_info()

        except Exception:
            logger.exception("Failed to initialize Bluefin client")
            if not self.dry_run:
                logger.exception("Bluefin client initialization failed in live mode")
                raise
            logger.warning(
                "Continuing in paper trading mode despite initialization failure"
            )

    async def _load_contract_info(self) -> None:
        """Load and cache perpetual contract information."""
        try:
            # Check if service client is available and has the expected methods
            if (
                self._service_client
                and hasattr(self._service_client, "get_account_data")
                and callable(self._service_client.get_account_data)
            ):
                try:
                    # Use service client for market info
                    await self._service_client.get_account_data()
                    logger.debug(
                        "Successfully retrieved market data from service client"
                    )
                except Exception as service_error:
                    logger.warning("Service client call failed: %s", service_error)
            else:
                logger.debug(
                    "Service client not available or missing methods - using fallback configuration"
                )

            # Set up contract info based on network and supported symbols
            if self.network_name.lower() in ["testnet", "staging", "sui_staging"]:
                from bot.utils.symbol_utils import BLUEFIN_TESTNET_SYMBOLS

                symbols_info = BLUEFIN_TESTNET_SYMBOLS
            else:
                from bot.utils.symbol_utils import BLUEFIN_MAINNET_SYMBOLS

                symbols_info = BLUEFIN_MAINNET_SYMBOLS

            for symbol, info in symbols_info.items():
                self._contract_info[symbol] = {
                    "base_asset": symbol.split("-")[0],
                    "quote_asset": "USDC",
                    "min_quantity": Decimal(str(info.get("min_trade_size", 0.001))),
                    "max_quantity": Decimal(str(info.get("max_trade_size", 1000000))),
                    "tick_size": Decimal(str(info.get("step_size", 0.001))),
                    "min_notional": Decimal(10),
                }

            logger.debug("Loaded %d perpetual contracts", len(self._contract_info))

        except Exception as e:
            logger.warning("Failed to load contract info: %s", e)
            # Set up basic contract info even if service fails
            try:
                if self.network_name.lower() in ["testnet", "staging", "sui_staging"]:
                    from bot.utils.symbol_utils import BLUEFIN_TESTNET_SYMBOLS

                    symbols_info = BLUEFIN_TESTNET_SYMBOLS
                else:
                    from bot.utils.symbol_utils import BLUEFIN_MAINNET_SYMBOLS

                    symbols_info = BLUEFIN_MAINNET_SYMBOLS

                for symbol, info in symbols_info.items():
                    self._contract_info[symbol] = {
                        "base_asset": symbol.split("-")[0],
                        "quote_asset": "USDC",
                        "min_quantity": Decimal(str(info.get("min_trade_size", 0.001))),
                        "max_quantity": Decimal(
                            str(info.get("max_trade_size", 1000000))
                        ),
                        "tick_size": Decimal(str(info.get("step_size", 0.001))),
                        "min_notional": Decimal(10),
                    }
            except ImportError:
                # Ultimate fallback if imports fail
                common_symbols = [
                    "BTC-PERP",
                    "ETH-PERP",
                    "SOL-PERP",
                ]  # Skip SUI-PERP for testnet safety
                for symbol in common_symbols:
                    self._contract_info[symbol] = {
                        "base_asset": symbol.split("-")[0],
                        "quote_asset": "USDC",
                        "min_quantity": Decimal("0.001"),
                        "max_quantity": Decimal(1000000),
                        "tick_size": Decimal("0.01"),
                        "min_notional": Decimal(10),
                    }
            logger.info(
                "Using fallback contract info for %s symbols", len(self._contract_info)
            )

    async def disconnect(self) -> None:
        """Disconnect from Bluefin."""
        try:
            # Clean up service client
            await self._service_client.disconnect()
            logger.info("Disconnecting from Bluefin...")
        except Exception as e:
            logger.warning("Error during Bluefin disconnect: %s", e)

        self._connected = False
        self._last_health_check = None
        logger.info("Disconnected from Bluefin")

    async def execute_trade_action(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
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
            # Validate symbol before proceeding
            if not await self.validate_symbol_exists(symbol):
                logger.error(
                    "Symbol validation failed for %s, cannot execute trade", symbol
                )
                return None

            # Convert symbol to Bluefin format (e.g., ETH-USD -> ETH-PERP)
            bluefin_symbol = self._convert_symbol(symbol)

            if trade_action.action == "HOLD":
                logger.info("Action is HOLD - no trade executed")
                return None
            if trade_action.action == "CLOSE":
                return await self._close_position(bluefin_symbol)
            if trade_action.action in ["LONG", "SHORT"]:
                return await self._open_position(
                    trade_action, bluefin_symbol, current_price
                )
            logger.error("Unknown action: %s", trade_action.action)

        except ValueError:
            logger.exception("Symbol validation error")
            return None
        except Exception:
            logger.exception("Failed to execute trade action")
            return None
        else:
            return None

    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Bluefin perpetual format using symbol utilities."""

        def _raise_unsupported_symbol_error(unsupported_symbol: str) -> None:
            raise ValueError(f"Unsupported symbol: {unsupported_symbol}")

        try:
            # Validate the symbol first
            if not validate_symbol(symbol):
                logger.warning(
                    "Invalid symbol format: %s, attempting conversion anyway", symbol
                )

            # Convert to Bluefin perpetual format
            bluefin_symbol = to_bluefin_perp(symbol)

            # Check if symbol is supported on current network
            if not is_bluefin_symbol_supported(bluefin_symbol, self.network_name):
                if self.network_name.lower() in ["testnet", "staging", "sui_staging"]:
                    # Get testnet fallback
                    fallback_symbol = get_testnet_symbol_fallback(bluefin_symbol)
                    logger.warning(
                        "Symbol %s not available on %s, using fallback: %s",
                        bluefin_symbol,
                        self.network_name,
                        fallback_symbol,
                    )
                    bluefin_symbol = fallback_symbol
                else:
                    logger.error(
                        "Symbol %s not supported on %s",
                        bluefin_symbol,
                        self.network_name,
                    )
                    _raise_unsupported_symbol_error(bluefin_symbol)

            logger.debug(
                "Converted symbol %s to Bluefin format: %s", symbol, bluefin_symbol
            )

        except Exception:
            logger.exception("Failed to convert symbol %s", symbol)
            # Fallback to original behavior for backward compatibility
            symbol_map = {
                "BTC-USD": "BTC-PERP",
                "ETH-USD": "ETH-PERP",
                "SOL-USD": "SOL-PERP",
                "SUI-USD": "SUI-PERP",
            }
            fallback_symbol = symbol_map.get(symbol, symbol)

            # Final check - if even fallback is not supported, use BTC-PERP
            if not is_bluefin_symbol_supported(
                fallback_symbol, self.network_name
            ) and self.network_name.lower() in ["testnet", "staging", "sui_staging"]:
                fallback_symbol = "BTC-PERP"  # Safe fallback for testnet
                logger.warning(
                    "All conversions failed, using safe testnet fallback: %s",
                    fallback_symbol,
                )

            logger.warning(
                "Using fallback conversion: %s -> %s", symbol, fallback_symbol
            )
            return fallback_symbol
        else:
            return bluefin_symbol

    async def _open_position(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """Open a new perpetual position."""
        try:
            # Get account balance
            account_balance = await self.get_account_balance()
            logger.info("Account balance: $%s", account_balance)

            # If balance is too low, log error and return None
            if account_balance <= Decimal(10):
                logger.error(
                    "Account balance $%s is too low for trading (minimum $10)",
                    account_balance,
                )
                return None

            # Calculate position size
            position_value = account_balance * Decimal(str(trade_action.size_pct / 100))
            logger.info(
                "Position value ($%s * %s%): $%s",
                account_balance,
                trade_action.size_pct,
                position_value,
            )

            # Apply leverage
            leverage = trade_action.leverage or settings.trading.leverage
            notional_value = position_value * leverage
            logger.info(
                "Notional value ($%s * %sx leverage): $%s",
                position_value,
                leverage,
                notional_value,
            )

            # Calculate quantity
            quantity = notional_value / current_price
            logger.info(
                "Calculated quantity ($%s / $%s): %s",
                notional_value,
                current_price,
                quantity,
            )

            # Round to contract specifications
            contract_info = self._contract_info.get(symbol, {})
            tick_size = contract_info.get("tick_size", Decimal("0.001"))
            quantity = self._round_to_tick(quantity, tick_size)
            logger.info("Rounded quantity to tick size %s: %s", tick_size, quantity)

            # Validate quantity
            min_qty = contract_info.get("min_quantity", Decimal("0.001"))
            if quantity < min_qty:
                logger.warning(
                    "Quantity %s below minimum %s, adjusting to minimum",
                    quantity,
                    min_qty,
                )
                quantity = min_qty

            # Final validation - ensure quantity is not zero
            if quantity <= Decimal(0):
                logger.error(
                    "Final quantity %s is zero or negative, cannot place order",
                    quantity,
                )
                return None

            # Determine order side
            side = OrderSide.BUY if trade_action.action == "LONG" else OrderSide.SELL

            # Final logging before order placement
            logger.info(
                "Placing %s order for %s %s at market price $%s",
                side,
                quantity,
                symbol,
                current_price,
            )

            # Place market order
            order = await self.place_market_order(
                symbol, cast("Literal['BUY', 'SELL']", side), quantity
            )

            if order:
                # Set leverage for the position
                await self._set_leverage(symbol, leverage)

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
            logger.exception("Failed to open position")
            return None
        else:
            return order

    async def _close_position(self, symbol: str) -> Order | None:
        """Close existing position."""
        try:
            # Get current position
            positions = await self.get_positions(symbol)

            if not positions:
                logger.info("No position to close")
                return None

            position = positions[0]

            # Determine opposite side to close
            side = OrderSide.SELL if position.side == "LONG" else OrderSide.BUY

            # Place market order to close
            return await self.place_market_order(
                symbol, cast("Literal['BUY', 'SELL']", side), abs(position.size)
            )

        except Exception:
            logger.exception("Failed to close position")
            return None

    async def place_market_order(
        self, symbol: str, side: Literal["BUY", "SELL"], quantity: Decimal
    ) -> Order | None:
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
                "PAPER TRADING: Simulating %s %s %s at market", side, quantity, symbol
            )
            return Order(
                id=f"paper_bluefin_{int(datetime.now(UTC).timestamp() * 1000)}",
                symbol=symbol,
                side=cast("Literal['BUY', 'SELL']", side),
                type="MARKET",
                quantity=quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(UTC),
                filled_quantity=quantity,
            )

        def _raise_order_placement_error(error_msg: str) -> NoReturn:
            raise ExchangeOrderError(f"Order placement failed: {error_msg}")

        def _raise_insufficient_funds_error(e: Exception) -> NoReturn:
            raise ExchangeInsufficientFundsError(f"Insufficient funds: {e}") from e

        def _raise_market_order_error(e: Exception) -> NoReturn:
            raise ExchangeOrderError(f"Failed to place market order: {e}") from e

        try:
            # Get current market price from service
            ticker = await self._service_client.get_market_ticker(symbol)
            current_price = Decimal(str(ticker["price"]))

            # Calculate fees for the order
            notional_value = current_price * quantity
            fees = self.calculate_order_fees(
                notional_value, is_maker=False
            )  # Market orders are taker

            logger.info(
                "Market order fees: $%.6f taker fee for $%.2f notional",
                fees.taker_fee,
                notional_value,
            )

            # Sign the order if in live trading mode
            if self._signature_manager:
                try:
                    # Sign the market order
                    signed_order_data = self._signature_manager.sign_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        estimated_fee=fees.taker_fee,
                    )

                    # Add current price for market orders
                    signed_order_data["price"] = float(current_price)

                    logger.info(
                        "✍️ Market order signed - Hash: %s, Public Key: %s",
                        signed_order_data.get("orderHash", "")[:16] + "...",
                        signed_order_data.get("publicKey", "")[:16] + "...",
                    )
                    order_data = signed_order_data

                except OrderSignatureError as e:
                    logger.exception("❌ Failed to sign market order: %s", e)
                    _raise_order_placement_error(f"Order signature failed: {e}")
            else:
                # Prepare unsigned order data for paper trading
                order_data = {
                    "symbol": symbol,
                    "price": float(current_price),
                    "quantity": float(quantity),
                    "side": side,
                    "orderType": "MARKET",
                    "estimated_fee": float(fees.taker_fee),
                }

            logger.info(
                "Placing %s market order: %s %s (est. fee: $%.6f) %s",
                side,
                quantity,
                symbol,
                fees.taker_fee,
                "🔐 [SIGNED]" if self._signature_manager else "📊 [UNSIGNED]",
            )

            # Post order via service
            response = await self._service_client.place_order(order_data)

            # Parse response
            if response.get("status") == "success":
                order_data = response.get("order", {})
                order = Order(
                    id=order_data.get("id", f"bluefin_{datetime.now(UTC).timestamp()}"),
                    symbol=symbol,
                    side=cast("Literal['BUY', 'SELL']", side),
                    type="MARKET",
                    quantity=quantity,
                    status=OrderStatus.PENDING,
                    timestamp=datetime.now(UTC),
                    filled_quantity=Decimal(0),
                )

                logger.info("Market order placed successfully: %s", order.id)
                return order
            error_msg = response.get("message", "Unknown error")
            logger.error("Order placement failed: %s", error_msg)
            _raise_order_placement_error(error_msg)

        except Exception as e:
            logger.exception("Failed to place market order")
            if "insufficient" in str(e).lower():
                _raise_insufficient_funds_error(e)
            _raise_market_order_error(e)

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
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
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

        def _raise_limit_order_placement_error(error_msg: str) -> NoReturn:
            raise ExchangeOrderError(f"Limit order placement failed: {error_msg}")

        def _raise_limit_order_error(e: Exception) -> NoReturn:
            raise ExchangeOrderError(f"Failed to place limit order: {e}") from e

        try:
            # Calculate fees for the order
            notional_value = price * quantity
            fees = self.calculate_order_fees(
                notional_value, is_maker=True
            )  # Limit orders are maker

            logger.debug(
                "Limit order fees: $%.6f maker fee for $%.2f notional",
                fees.maker_fee,
                notional_value,
            )

            # Sign the order if in live trading mode
            if self._signature_manager:
                try:
                    # Sign the limit order
                    signed_order_data = self._signature_manager.sign_limit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        estimated_fee=fees.maker_fee,
                    )

                    logger.info(
                        "✍️ Limit order signed - Hash: %s, Public Key: %s",
                        signed_order_data.get("orderHash", "")[:16] + "...",
                        signed_order_data.get("publicKey", "")[:16] + "...",
                    )
                    order_data = signed_order_data

                except OrderSignatureError as e:
                    logger.exception("❌ Failed to sign limit order: %s", e)
                    _raise_limit_order_placement_error(f"Order signature failed: {e}")
            else:
                # Prepare unsigned order data for paper trading
                order_data = {
                    "symbol": symbol,
                    "price": float(price),
                    "quantity": float(quantity),
                    "side": side,
                    "orderType": "LIMIT",
                    "estimated_fee": float(fees.maker_fee),
                }

            logger.info(
                "Placing %s limit order: %s %s @ %s (est. fee: $%.6f) %s",
                side,
                quantity,
                symbol,
                price,
                fees.maker_fee,
                "🔐 [SIGNED]" if self._signature_manager else "📊 [UNSIGNED]",
            )

            # Post order via service
            response = await self._service_client.place_order(order_data)

            # Parse response
            if response.get("status") == "success":
                order_data = response.get("order", {})
                order = Order(
                    id=order_data.get(
                        "id", f"bluefin_limit_{datetime.now(UTC).timestamp()}"
                    ),
                    symbol=symbol,
                    side=cast("Literal['BUY', 'SELL']", side),
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.OPEN,
                    timestamp=datetime.now(UTC),
                    filled_quantity=Decimal(0),
                )

                logger.info("Limit order placed successfully: %s", order.id)
                return order
            error_msg = response.get("message", "Unknown error")
            logger.error("Limit order placement failed: %s", error_msg)
            _raise_limit_order_placement_error(error_msg)

        except Exception as e:
            logger.exception("Failed to place limit order")
            _raise_limit_order_error(e)

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
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
            # Get user positions from service
            positions_data = await self._service_client.get_user_positions()

            positions = []
            for pos_data in positions_data:
                pos_symbol = pos_data.get("symbol")
                if not pos_symbol:
                    continue

                # Skip if symbol filter provided and doesn't match
                if symbol and pos_symbol != self._convert_symbol(symbol):
                    continue

                # Parse position data
                size = Decimal(str(pos_data.get("quantity", "0")))
                if size != 0:
                    # Use the side field from the service if available
                    side = pos_data.get("side", "LONG" if size > 0 else "SHORT")
                    position = Position(
                        symbol=pos_symbol,
                        side=cast("Literal['LONG', 'SHORT', 'FLAT']", side),
                        size=abs(size),
                        entry_price=Decimal(str(pos_data.get("avgPrice", "0"))),
                        unrealized_pnl=Decimal(str(pos_data.get("unrealizedPnl", "0"))),
                        realized_pnl=Decimal(str(pos_data.get("realizedPnl", "0"))),
                        timestamp=datetime.now(UTC),
                        is_futures=True,
                        leverage=pos_data.get("leverage", 1),
                        margin_used=Decimal(str(pos_data.get("margin", "0"))),
                    )
                    positions.append(position)

            logger.debug("Retrieved %d positions from Bluefin", len(positions))

        except Exception:
            logger.exception("Failed to get positions")
            return []
        else:
            return positions

    def _record_balance_operation(
        self,
        operation: str,
        balance_before: float | None = None,
        balance_after: float | None = None,
        success: bool = True,
        error_type: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a balance operation in the monitoring system."""
        try:
            if self.monitoring_enabled:
                correlation_id = f"bluefin_{operation}_{int(time.time() * 1000)}"

                start_time = record_operation_start(
                    operation=operation,
                    component="bluefin_exchange",
                    correlation_id=correlation_id,
                    metadata=metadata or {},
                )

                record_operation_complete(
                    operation=operation,
                    component="bluefin_exchange",
                    start_time=start_time,
                    success=success,
                    balance_before=balance_before,
                    balance_after=balance_after,
                    error_type=error_type,
                    correlation_id=correlation_id,
                    metadata=metadata,
                )
        except Exception as e:
            logger.debug("Failed to record balance operation %s: %s", operation, e)

    def _handle_dry_run_balance(self, account_type: AccountType | None) -> Decimal:
        """Handle balance retrieval for dry run mode."""
        mock_balance = self._normalize_balance(Decimal("10000.00"))

        self._record_balance_operation(
            operation="get_balance_mock",
            balance_before=None,
            balance_after=float(mock_balance),
            success=True,
            metadata={
                "account_type": str(account_type) if account_type else None,
                "dry_run": True,
                "network": self.network_name,
            },
        )

        return mock_balance

    def _start_balance_monitoring(
        self, account_type: AccountType | None
    ) -> tuple[str | None, float | None]:
        """Start balance operation monitoring."""
        if not self.monitoring_enabled:
            return None, None

        try:
            correlation_id = f"bluefin_balance_{int(time.time() * 1000)}"
            start_time = record_operation_start(
                operation="get_balance",
                component="bluefin_exchange",
                correlation_id=correlation_id,
                metadata={
                    "account_type": str(account_type) if account_type else None,
                    "network": self.network_name,
                },
            )
        except Exception as e:
            logger.debug("Failed to record balance operation start: %s", e)
            return None, None
        else:
            return correlation_id, start_time

    async def _fetch_account_data(self, account_type: AccountType | None) -> dict:
        """Fetch account data from service with error handling."""
        if not self._service_client:
            raise BalanceServiceUnavailableError(
                "Bluefin service client not initialized",
                service_name="bluefin_service",
            )

        try:
            return await asyncio.wait_for(
                self._service_client.get_account_data(),
                timeout=30.0,
            )
        except TimeoutError as timeout_err:
            raise BalanceTimeoutError(
                "Balance request timed out after 30 seconds",
                timeout_duration=30.0,
                endpoint="get_account_data",
                account_type=str(account_type) if account_type else None,
            ) from timeout_err
        except Exception as service_err:
            if (
                "connection" in str(service_err).lower()
                or "unavailable" in str(service_err).lower()
            ):
                raise BalanceServiceUnavailableError(
                    f"Bluefin service connection failed: {service_err}",
                    service_name="bluefin_service",
                ) from service_err
            raise

    def _validate_and_extract_balance(
        self, account_data: dict, account_type: AccountType | None
    ) -> Decimal:
        """Validate account data and extract balance."""
        if not isinstance(account_data, dict):
            raise BalanceValidationError(
                f"Invalid account data format: expected dict, got {type(account_data)}",
                invalid_value=type(account_data).__name__,
                validation_rule="account_data_dict_format",
                account_type=str(account_type) if account_type else None,
            )

        balance_value = account_data.get("balance", "0")

        try:
            # Import price conversion utility
            from bot.utils.price_conversion import (
                convert_from_18_decimal,
                is_likely_18_decimal,
            )

            # OPTIMIZATION: Try precision manager first for better performance
            try:
                from bot.utils.precision_manager import convert_price_optimized

                # Use optimized conversion for better precision handling
                raw_balance = convert_price_optimized(
                    value=balance_value, symbol="USDC", field_name="balance"
                )
            except ImportError:
                # Fallback to legacy conversion
                if is_likely_18_decimal(balance_value):
                    raw_balance = convert_from_18_decimal(
                        balance_value, "USDC", "balance"
                    )
                    logger.info(
                        "Converted astronomical balance from 18-decimal: %s -> %s",
                        balance_value,
                        raw_balance,
                    )
                else:
                    raw_balance = Decimal(str(balance_value))

            # Log successful conversion if precision manager was used
            if "convert_price_optimized" in locals():
                logger.debug(
                    "Used optimized precision conversion for balance: %s -> %s",
                    balance_value,
                    raw_balance,
                )

        except (ValueError, TypeError, decimal.InvalidOperation) as decimal_err:
            raise BalanceValidationError(
                f"Invalid balance value format: {balance_value}",
                invalid_value=balance_value,
                validation_rule="decimal_conversion",
                account_type=str(account_type) if account_type else None,
            ) from decimal_err

        return self._normalize_balance(raw_balance)

    def _log_balance_info(
        self, total_balance: Decimal, raw_balance: Decimal, account_data: dict
    ) -> None:
        """Log balance information and warnings."""
        if total_balance < Decimal(0):
            logger.warning("Retrieved negative balance: $%s", total_balance)

        logger.info("Retrieved Bluefin balance from service: $%s", total_balance)
        logger.debug("Full account data: %s", account_data)
        logger.debug("Balance normalization: %s -> %s", raw_balance, total_balance)

        if total_balance <= Decimal(0) and not self.dry_run:
            logger.warning(
                "Zero or negative balance in live mode - this will prevent trading"
            )
            logger.warning(
                "Please ensure your Bluefin account has USDC balance. Current: $%s",
                total_balance,
            )

    def _record_balance_success(
        self,
        correlation_id: str | None,
        start_time: float | None,
        total_balance: Decimal,
        raw_balance: Decimal,
        account_type: AccountType | None,
    ) -> None:
        """Record successful balance operation."""
        if not self.monitoring_enabled or start_time is None:
            return

        try:
            record_operation_complete(
                operation="get_balance",
                component="bluefin_exchange",
                start_time=start_time,
                success=True,
                balance_before=None,
                balance_after=float(total_balance),
                correlation_id=correlation_id,
                metadata={
                    "account_type": str(account_type) if account_type else None,
                    "network": self.network_name,
                    "raw_balance": str(raw_balance),
                    "normalized_balance": str(total_balance),
                    "is_negative": total_balance < Decimal(0),
                },
            )
        except Exception as e:
            logger.debug("Failed to record successful balance operation: %s", e)

    def _record_balance_error(
        self,
        correlation_id: str | None,
        start_time: float | None,
        balance_error: Exception,
        account_type: AccountType | None,
    ) -> None:
        """Record balance operation error."""
        if not self.monitoring_enabled or start_time is None:
            return

        try:
            error_type = type(balance_error).__name__
            if hasattr(balance_error, "timeout_duration"):
                record_timeout(
                    operation="get_balance",
                    component="bluefin_exchange",
                    timeout_duration_ms=getattr(balance_error, "timeout_duration", 30)
                    * 1000,
                    correlation_id=correlation_id,
                )
            else:
                record_operation_complete(
                    operation="get_balance",
                    component="bluefin_exchange",
                    start_time=start_time,
                    success=False,
                    error_type=error_type,
                    correlation_id=correlation_id,
                    metadata={
                        "account_type": str(account_type) if account_type else None,
                        "network": self.network_name,
                        "error_message": str(balance_error),
                    },
                )
        except Exception as e:
            logger.debug("Failed to record balance error: %s", e)

    def _record_unexpected_error(
        self,
        correlation_id: str | None,
        start_time: float | None,
        error: Exception,
        account_type: AccountType | None,
    ) -> None:
        """Record unexpected error in balance operation."""
        if not self.monitoring_enabled or start_time is None:
            return

        try:
            record_operation_complete(
                operation="get_balance",
                component="bluefin_exchange",
                start_time=start_time,
                success=False,
                error_type="unexpected_error",
                correlation_id=correlation_id,
                metadata={
                    "account_type": str(account_type) if account_type else None,
                    "network": self.network_name,
                    "error_message": str(error),
                    "error_type": type(error).__name__,
                },
            )
        except Exception as monitoring_error:
            logger.debug("Failed to record unexpected error: %s", monitoring_error)

    async def get_account_balance(
        self, account_type: AccountType | None = None
    ) -> Decimal:
        """
        Get account balance in USD.

        Args:
            account_type: Not used for Bluefin (only one account type)

        Returns:
            Account balance in USD

        Raises:
            BalanceServiceUnavailableError: When service is unavailable
            BalanceTimeoutError: When request times out
            BalanceValidationError: When balance data is invalid
            BalanceRetrievalError: For other balance retrieval issues
        """
        if self.dry_run:
            return self._handle_dry_run_balance(account_type)

        correlation_id, start_time = self._start_balance_monitoring(account_type)

        try:
            account_data = await self._fetch_account_data(account_type)
            total_balance = self._validate_and_extract_balance(
                account_data, account_type
            )

            # Get raw balance for logging (extract again for clarity)
            raw_balance = Decimal(str(account_data.get("balance", "0")))

            self._log_balance_info(total_balance, raw_balance, account_data)
            self._record_balance_success(
                correlation_id, start_time, total_balance, raw_balance, account_type
            )
        except (
            BalanceRetrievalError,
            BalanceServiceUnavailableError,
            BalanceTimeoutError,
            BalanceValidationError,
        ) as balance_error:
            self._record_balance_error(
                correlation_id, start_time, balance_error, account_type
            )
            logger.exception("Balance operation failed")
            raise
        except Exception as e:
            self._record_unexpected_error(correlation_id, start_time, e, account_type)
            logger.exception("Unexpected error getting account balance")
            raise BalanceRetrievalError(
                f"Failed to retrieve account balance: {e}",
                account_type=str(account_type) if account_type else None,
                balance_context={
                    "exchange": "bluefin",
                    "network": self.network_name,
                    "dry_run": self.dry_run,
                    "error_type": type(e).__name__,
                },
            ) from e
        else:
            return total_balance

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
            logger.info(
                "Cancelling order: %s %s",
                order_id,
                "🔐 [SIGNED]" if self._signature_manager else "📊 [UNSIGNED]",
            )

            # Cancel order via service
            success = await self._service_client.cancel_order(order_id)

            if success:
                logger.info("Order %s cancelled successfully", order_id)
                return True
            logger.warning("Order %s cancellation failed", order_id)

        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False
        else:
            return False

    async def cancel_all_orders(
        self, symbol: str | None = None, _status: str | None = None
    ) -> bool:
        """
        Cancel all open orders.

        Args:
            symbol: Optional trading symbol filter
            status: Optional order status filter (for Bluefin SDK compatibility)

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info("PAPER TRADING: Simulating cancel all orders")
            return True

        try:
            # Convert symbol if provided
            self._convert_symbol(symbol) if symbol else None

            # Since BluefinServiceClient doesn't exist, simulate the operation for now
            # In paper trading mode, this is just a simulation anyway
            if self.dry_run:
                logger.info("PAPER TRADING: Simulated cancelling all orders")
                return True
            # For live trading, we would need to implement actual Bluefin API calls
            # For now, log that this needs implementation
            logger.warning(
                "Cancel all orders not yet implemented for live Bluefin trading"
            )

        except Exception:
            logger.exception("Failed to cancel all orders")
            return False
        else:
            return True  # Return True to avoid errors during shutdown

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def get_connection_status(self) -> dict[str, Any]:
        """Get connection status information."""
        trading_mode = "PAPER TRADING" if self.dry_run else "LIVE TRADING"
        system_dry_run = getattr(settings.system, "dry_run", True)

        return {
            "connected": self._connected,
            "exchange": "Bluefin",
            "network": self.network_name,
            "account_address": self._account_address,
            "has_credentials": bool(self.private_key),
            "dry_run": self.dry_run,
            "system_dry_run": system_dry_run,
            "trading_mode": trading_mode,
            "trading_mode_override": not system_dry_run and self.dry_run,
            "last_health_check": (
                self._last_health_check.isoformat() if self._last_health_check else None
            ),
            "rate_limit_remaining": self._rate_limiter.max_requests
            - len(self._rate_limiter.requests),
            "is_decentralized": True,
            "blockchain": "Sui",
            "bluefin_sdk_available": BLUEFIN_AVAILABLE,
        }

    # Bluefin-specific methods

    async def _get_account_address(self) -> str | None:
        """Get the Sui wallet address."""
        try:
            # Get account info from service
            account_data = await self._service_client.get_account_data()
            return account_data.get("address")
        except Exception:
            logger.exception("Failed to get account address")
        return None

    async def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            if self.dry_run:
                return True

            # Cache leverage setting
            self._leverage_settings[symbol] = leverage

            # Set leverage via service (would need to implement in service)
            # For now, simulate success
            response = {"status": "success"}

            return response.get("status") == "success"

        except Exception:
            logger.exception("Failed to set leverage")
            return False

    async def _place_stop_loss(
        self, base_order: Order, trade_action: TradeAction, current_price: Decimal
    ) -> Order | None:
        """Place stop loss order."""
        # Calculate stop loss price
        sl_pct = trade_action.stop_loss_pct / 100

        if base_order.side == "BUY":  # Long position
            stop_price = current_price * (1 - Decimal(str(sl_pct)))
            side = OrderSide.SELL
        else:  # Short position
            stop_price = current_price * (1 + Decimal(str(sl_pct)))
            side = OrderSide.BUY

        # Round to tick size
        contract_info = self._contract_info.get(base_order.symbol, {})
        tick_size = contract_info.get("tick_size", Decimal("0.01"))
        stop_price = self._round_to_tick(stop_price, tick_size)

        # For Bluefin, we'll use a stop-limit order
        # Place the limit order slightly below/above the stop for slippage
        slippage = Decimal("0.001")  # 0.1% slippage
        if side == OrderSide.SELL:
            limit_price = stop_price * (1 - slippage)
        else:
            limit_price = stop_price * (1 + slippage)

        # Note: Bluefin may require specific stop order implementation
        # For now, using limit order as placeholder
        return await self.place_limit_order(
            base_order.symbol,
            cast("Literal['BUY', 'SELL']", side),
            base_order.quantity,
            limit_price,
        )

    async def _place_take_profit(
        self, base_order: Order, trade_action: TradeAction, current_price: Decimal
    ) -> Order | None:
        """Place take profit order."""
        # Calculate take profit price
        tp_pct = trade_action.take_profit_pct / 100

        if base_order.side == "BUY":  # Long position
            limit_price = current_price * (1 + Decimal(str(tp_pct)))
            side = OrderSide.SELL
        else:  # Short position
            limit_price = current_price * (1 - Decimal(str(tp_pct)))
            side = OrderSide.BUY

        # Round to tick size
        contract_info = self._contract_info.get(base_order.symbol, {})
        tick_size = contract_info.get("tick_size", Decimal("0.01"))
        limit_price = self._round_to_tick(limit_price, tick_size)

        return await self.place_limit_order(
            base_order.symbol,
            cast("Literal['BUY', 'SELL']", side),
            base_order.quantity,
            limit_price,
        )

    def _round_to_tick(self, value: Decimal, tick_size: Decimal) -> Decimal:
        """Round a value to the nearest tick size."""
        return (value / tick_size).quantize(Decimal(1)) * tick_size

    async def _retry_request(self, func, *args, **kwargs):
        """Execute a request with retry logic."""
        # Apply rate limiting for service calls
        await self._rate_limiter.acquire()
        return await func(*args, **kwargs)

    async def get_historical_candles(
        self, symbol: str, interval: str = "5m", limit: int = 500
    ) -> list:
        """
        Fetch historical candlestick data from Bluefin.

        Args:
            symbol: Trading symbol (e.g., 'ETH-PERP', 'BTC-PERP')
            interval: Timeframe (e.g., '1m', '5m', '15m', '1h')
            limit: Number of candles to fetch (default: 500)

        Returns:
            List of OHLCV data dictionaries
        """
        logger.info(
            "Fetching real historical data for %s (%s candles, %s interval) from Bluefin",
            symbol,
            limit,
            interval,
        )

        try:
            # Convert standard symbol to Bluefin format
            bluefin_symbol = self._convert_symbol(symbol)

            logger.info(
                "Fetching %s historical candles for %s (%s interval) from Bluefin service",
                limit,
                bluefin_symbol,
                interval,
            )

            # Use service client to get candlestick data
            params = {"symbol": bluefin_symbol, "interval": interval, "limit": limit}

            candles = await self._service_client.get_candlestick_data(params)

            if candles:
                logger.info(
                    "Successfully fetched %s candles from Bluefin service for %s",
                    len(candles),
                    bluefin_symbol,
                )
                # Convert service response to expected format
                formatted_candles = []
                for candle in candles:
                    if len(candle) >= 6:
                        formatted_candles.append(
                            {
                                "timestamp": candle[0],
                                "open": candle[1],
                                "high": candle[2],
                                "low": candle[3],
                                "close": candle[4],
                                "volume": candle[5],
                            }
                        )

                return formatted_candles

        except Exception:
            logger.exception("Failed to fetch historical candles from Bluefin")
            return []
        else:
            logger.warning("No candlestick data returned for %s", bluefin_symbol)
            return []

    @property
    def supports_futures(self) -> bool:
        """Bluefin only supports perpetual futures."""
        return True

    @property
    def is_decentralized(self) -> bool:
        """Bluefin is a decentralized exchange."""
        return True

    # Override futures-specific methods since Bluefin only does perps
    async def get_futures_positions(self, symbol: str | None = None) -> list[Position]:
        """All positions on Bluefin are futures positions."""
        return await self.get_positions(symbol)

    async def place_futures_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        leverage: int | None = None,
        _reduce_only: bool = False,
    ) -> Order | None:
        """All orders on Bluefin are futures orders."""
        # Set leverage if specified
        if leverage:
            await self._set_leverage(symbol, leverage)

        return await self.place_market_order(symbol, side, quantity)

    @property
    def enable_futures(self) -> bool:
        """Enable futures trading. Bluefin only does futures, so always returns True."""
        return True

    async def get_trading_symbol(self, symbol: str) -> str:
        """Get the actual trading symbol for the given symbol (convert to PERP format)."""
        return self._convert_symbol(symbol)

    # Market Making Specific Methods

    async def place_ladder_limit_orders(
        self, order_levels: list[dict], symbol: str
    ) -> list[Order]:
        """
        Place multiple limit orders at different price levels (ladder strategy).

        Args:
            order_levels: List of dicts with 'side', 'price', 'quantity', 'level' keys
            symbol: Trading symbol

        Returns:
            List of successfully placed orders
        """
        if not self._connected:
            logger.error("Not connected to Bluefin")
            return []

        placed_orders = []
        bluefin_symbol = self._convert_symbol(symbol)

        logger.info(
            "Placing ladder orders for %s: %d levels", bluefin_symbol, len(order_levels)
        )

        # Group orders by side for batch processing
        buy_orders = [ol for ol in order_levels if ol["side"] == "BUY"]
        sell_orders = [ol for ol in order_levels if ol["side"] == "SELL"]

        # Process buy orders
        for order_level in buy_orders:
            try:
                order = await self.place_limit_order(
                    symbol=bluefin_symbol,
                    side=cast("Literal['BUY', 'SELL']", order_level["side"]),
                    quantity=Decimal(str(order_level["quantity"])),
                    price=Decimal(str(order_level["price"])),
                )

                if order:
                    placed_orders.append(order)
                    # Track order for the symbol
                    if bluefin_symbol not in self._active_orders:
                        self._active_orders[bluefin_symbol] = []
                    self._active_orders[bluefin_symbol].append(order)

                    logger.debug(
                        "Placed %s limit order: %s %s @ %s (Level %d)",
                        order_level["side"],
                        order_level["quantity"],
                        bluefin_symbol,
                        order_level["price"],
                        int(order_level.get("level", 0)),
                    )
                else:
                    logger.warning(
                        "Failed to place %s order at level %d",
                        order_level["side"],
                        int(order_level.get("level", 0)),
                    )

                # Small delay between orders to avoid rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(
                    "Error placing %s order at level %d: %s",
                    order_level["side"],
                    int(order_level.get("level", 0)),
                    e,
                )
                continue

        # Process sell orders
        for order_level in sell_orders:
            try:
                order = await self.place_limit_order(
                    symbol=bluefin_symbol,
                    side=cast("Literal['BUY', 'SELL']", order_level["side"]),
                    quantity=Decimal(str(order_level["quantity"])),
                    price=Decimal(str(order_level["price"])),
                )

                if order:
                    placed_orders.append(order)
                    # Track order for the symbol
                    if bluefin_symbol not in self._active_orders:
                        self._active_orders[bluefin_symbol] = []
                    self._active_orders[bluefin_symbol].append(order)

                    logger.debug(
                        "Placed %s limit order: %s %s @ %s (Level %d)",
                        order_level["side"],
                        order_level["quantity"],
                        bluefin_symbol,
                        order_level["price"],
                        int(order_level.get("level", 0)),
                    )
                else:
                    logger.warning(
                        "Failed to place %s order at level %d",
                        order_level["side"],
                        int(order_level.get("level", 0)),
                    )

                # Small delay between orders to avoid rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(
                    "Error placing %s order at level %d: %s",
                    order_level["side"],
                    int(order_level.get("level", 0)),
                    e,
                )
                continue

        logger.info(
            "Successfully placed %d/%d ladder orders for %s",
            len(placed_orders),
            len(order_levels),
            bluefin_symbol,
        )

        return placed_orders

    async def cancel_orders_by_symbol(self, symbol: str) -> bool:
        """
        Cancel all open orders for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if successful
        """
        bluefin_symbol = self._convert_symbol(symbol)

        if self.dry_run:
            logger.info(
                "PAPER TRADING: Simulating cancel orders for %s", bluefin_symbol
            )
            # Clear tracking
            self._active_orders.pop(bluefin_symbol, [])
            return True

        logger.info("Cancelling all orders for %s", bluefin_symbol)

        try:
            # Get active orders for the symbol
            symbol_orders = self._active_orders.get(bluefin_symbol, [])

            if not symbol_orders:
                logger.debug("No active orders found for %s", bluefin_symbol)
                return True

            # Cancel each order
            cancelled_count = 0
            for order in symbol_orders:
                try:
                    if await self.cancel_order(order.id):
                        cancelled_count += 1
                    # Small delay between cancellations
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.warning("Failed to cancel order %s: %s", order.id, e)
                    continue

            # Clear tracking for the symbol
            self._active_orders.pop(bluefin_symbol, [])

            logger.info(
                "Cancelled %d/%d orders for %s",
                cancelled_count,
                len(symbol_orders),
                bluefin_symbol,
            )

            return cancelled_count > 0

        except Exception:
            logger.exception("Failed to cancel orders for %s", bluefin_symbol)
            return False

    async def get_current_spread(self, symbol: str) -> dict[str, Decimal]:
        """
        Get current bid/ask spread data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with bid, ask, spread, and spread_pct data
        """
        bluefin_symbol = self._convert_symbol(symbol)

        try:
            # Get order book from service
            if self._service_client and hasattr(self._service_client, "get_order_book"):
                order_book = await self._service_client.get_order_book(bluefin_symbol)

                if order_book and "bids" in order_book and "asks" in order_book:
                    bids = order_book["bids"]
                    asks = order_book["asks"]

                    if bids and asks:
                        best_bid = Decimal(str(bids[0][0]))  # [price, size]
                        best_ask = Decimal(str(asks[0][0]))

                        spread = best_ask - best_bid
                        mid_price = (best_bid + best_ask) / 2
                        spread_pct = (
                            (spread / mid_price) * 100 if mid_price > 0 else Decimal(0)
                        )

                        return {
                            "bid": best_bid,
                            "ask": best_ask,
                            "spread": spread,
                            "spread_pct": spread_pct,
                            "mid_price": mid_price,
                        }

            # Fallback: use ticker data
            ticker = await self._service_client.get_market_ticker(bluefin_symbol)
            current_price = Decimal(str(ticker.get("price", "0")))

            # Estimate spread (0.05% for paper trading)
            estimated_spread_pct = Decimal("0.05")
            estimated_spread = current_price * (estimated_spread_pct / 100)

            bid = current_price - (estimated_spread / 2)
            ask = current_price + (estimated_spread / 2)

            logger.debug(
                "Using estimated spread for %s: %.4f%% ($%.6f)",
                bluefin_symbol,
                estimated_spread_pct,
                estimated_spread,
            )

            return {
                "bid": bid,
                "ask": ask,
                "spread": estimated_spread,
                "spread_pct": estimated_spread_pct,
                "mid_price": current_price,
            }

        except Exception:
            logger.exception("Failed to get spread for %s", bluefin_symbol)
            # Return zero values on error
            return {
                "bid": Decimal(0),
                "ask": Decimal(0),
                "spread": Decimal(0),
                "spread_pct": Decimal(0),
                "mid_price": Decimal(0),
            }

    async def get_order_book_depth(self, symbol: str, levels: int = 5) -> dict:
        """
        Get order book depth data for market making decisions.

        Args:
            symbol: Trading symbol
            levels: Number of levels to retrieve (default: 5)

        Returns:
            Dictionary with bids, asks, and depth analysis
        """
        bluefin_symbol = self._convert_symbol(symbol)

        try:
            # Try to get order book from service
            if self._service_client and hasattr(self._service_client, "get_order_book"):
                order_book = await self._service_client.get_order_book(
                    bluefin_symbol, depth=levels
                )

                if order_book:
                    bids = order_book.get("bids", [])[:levels]
                    asks = order_book.get("asks", [])[:levels]

                    # Calculate depth metrics
                    bid_volume = sum(Decimal(str(bid[1])) for bid in bids)
                    ask_volume = sum(Decimal(str(ask[1])) for ask in asks)
                    total_volume = bid_volume + ask_volume

                    # Calculate weighted average prices
                    if bids:
                        weighted_bid = (
                            sum(
                                Decimal(str(bid[0])) * Decimal(str(bid[1]))
                                for bid in bids
                            )
                            / bid_volume
                            if bid_volume > 0
                            else Decimal(0)
                        )
                    else:
                        weighted_bid = Decimal(0)

                    if asks:
                        weighted_ask = (
                            sum(
                                Decimal(str(ask[0])) * Decimal(str(ask[1]))
                                for ask in asks
                            )
                            / ask_volume
                            if ask_volume > 0
                            else Decimal(0)
                        )
                    else:
                        weighted_ask = Decimal(0)

                    # Cache the data
                    self._order_book_cache[bluefin_symbol] = {
                        "timestamp": datetime.now(UTC),
                        "bids": bids,
                        "asks": asks,
                        "bid_volume": bid_volume,
                        "ask_volume": ask_volume,
                        "total_volume": total_volume,
                        "weighted_bid": weighted_bid,
                        "weighted_ask": weighted_ask,
                        "imbalance": (
                            (bid_volume - ask_volume) / total_volume
                            if total_volume > 0
                            else Decimal(0)
                        ),
                    }

                    return self._order_book_cache[bluefin_symbol]

            # Fallback: create mock order book for paper trading
            ticker = await self._service_client.get_market_ticker(bluefin_symbol)
            current_price = Decimal(str(ticker.get("price", "100")))

            # Generate mock order book around current price
            mock_bids = []
            mock_asks = []

            for i in range(levels):
                bid_price = current_price * (
                    1 - Decimal(str((i + 1) * 0.001))
                )  # 0.1% increments
                ask_price = current_price * (1 + Decimal(str((i + 1) * 0.001)))

                # Mock volume decreasing with distance
                volume = Decimal(str(100 / (i + 1)))

                mock_bids.append([float(bid_price), float(volume)])
                mock_asks.append([float(ask_price), float(volume)])

            mock_data = {
                "timestamp": datetime.now(UTC),
                "bids": mock_bids,
                "asks": mock_asks,
                "bid_volume": Decimal(250),  # Mock total volumes
                "ask_volume": Decimal(250),
                "total_volume": Decimal(500),
                "weighted_bid": current_price * Decimal("0.9995"),
                "weighted_ask": current_price * Decimal("1.0005"),
                "imbalance": Decimal(0),  # Balanced
            }

            self._order_book_cache[bluefin_symbol] = mock_data
            return mock_data

        except Exception:
            logger.exception("Failed to get order book depth for %s", bluefin_symbol)
            # Return empty order book on error
            return {
                "timestamp": datetime.now(UTC),
                "bids": [],
                "asks": [],
                "bid_volume": Decimal(0),
                "ask_volume": Decimal(0),
                "total_volume": Decimal(0),
                "weighted_bid": Decimal(0),
                "weighted_ask": Decimal(0),
                "imbalance": Decimal(0),
            }

    async def estimate_market_impact(self, symbol: str, quantity: Decimal) -> dict:
        """
        Estimate market impact of a trade on the order book.

        Args:
            symbol: Trading symbol
            quantity: Trade quantity (positive for buy, negative for sell)

        Returns:
            Dictionary with impact analysis
        """
        bluefin_symbol = self._convert_symbol(symbol)

        try:
            # Get order book depth
            order_book = await self.get_order_book_depth(bluefin_symbol, levels=10)

            if (
                not order_book
                or not order_book.get("bids")
                or not order_book.get("asks")
            ):
                logger.warning("No order book data for market impact estimation")
                return {
                    "estimated_price": Decimal(0),
                    "price_impact_pct": Decimal(0),
                    "slippage_cost": Decimal(0),
                    "liquidity_consumed_pct": Decimal(0),
                }

            is_buy = quantity > 0
            abs_quantity = abs(quantity)

            if is_buy:
                # Buying consumes ask liquidity
                levels = order_book["asks"]
                best_price = Decimal(str(levels[0][0])) if levels else Decimal(0)
            else:
                # Selling consumes bid liquidity
                levels = order_book["bids"]
                best_price = Decimal(str(levels[0][0])) if levels else Decimal(0)

            if not levels or best_price <= 0:
                return {
                    "estimated_price": Decimal(0),
                    "price_impact_pct": Decimal(0),
                    "slippage_cost": Decimal(0),
                    "liquidity_consumed_pct": Decimal(0),
                }

            # Calculate weighted average execution price
            remaining_quantity = abs_quantity
            total_cost = Decimal(0)
            total_quantity_filled = Decimal(0)
            levels_consumed = 0

            for level_price, level_size in levels:
                if remaining_quantity <= 0:
                    break

                level_price = Decimal(str(level_price))
                level_size = Decimal(str(level_size))

                quantity_from_level = min(remaining_quantity, level_size)
                total_cost += quantity_from_level * level_price
                total_quantity_filled += quantity_from_level
                remaining_quantity -= quantity_from_level
                levels_consumed += 1

            if total_quantity_filled <= 0:
                return {
                    "estimated_price": best_price,
                    "price_impact_pct": Decimal(0),
                    "slippage_cost": Decimal(0),
                    "liquidity_consumed_pct": Decimal(0),
                }

            # Calculate metrics
            weighted_avg_price = total_cost / total_quantity_filled
            price_impact_pct = (
                abs((weighted_avg_price - best_price) / best_price * 100)
                if best_price > 0
                else Decimal(0)
            )
            slippage_cost = abs(
                (weighted_avg_price - best_price) * total_quantity_filled
            )

            # Calculate liquidity consumption percentage
            side_volume = (
                order_book["ask_volume"] if is_buy else order_book["bid_volume"]
            )
            liquidity_consumed_pct = (
                (total_quantity_filled / side_volume * 100)
                if side_volume > 0
                else Decimal(100)
            )

            impact_analysis = {
                "estimated_price": weighted_avg_price,
                "best_price": best_price,
                "price_impact_pct": price_impact_pct,
                "slippage_cost": slippage_cost,
                "liquidity_consumed_pct": liquidity_consumed_pct,
                "levels_consumed": levels_consumed,
                "quantity_filled": total_quantity_filled,
                "quantity_remaining": remaining_quantity,
            }

            logger.debug(
                "Market impact for %s %s: %.4f%% price impact, $%.6f slippage",
                quantity,
                bluefin_symbol,
                price_impact_pct,
                slippage_cost,
            )

            return impact_analysis

        except Exception:
            logger.exception("Failed to estimate market impact for %s", bluefin_symbol)
            return {
                "estimated_price": Decimal(0),
                "price_impact_pct": Decimal(0),
                "slippage_cost": Decimal(0),
                "liquidity_consumed_pct": Decimal(0),
            }

    def register_order_callback(self, order_id: str, callback: callable) -> None:
        """
        Register a callback for order state changes.

        Args:
            order_id: Order ID to monitor
            callback: Function to call on order state change
        """
        self._order_callbacks[order_id] = callback
        logger.debug("Registered callback for order %s", order_id)

    def unregister_order_callback(self, order_id: str) -> None:
        """
        Unregister an order callback.

        Args:
            order_id: Order ID to stop monitoring
        """
        self._order_callbacks.pop(order_id, None)
        logger.debug("Unregistered callback for order %s", order_id)

    async def _notify_order_callback(
        self, order_id: str, order_status: str, order_data: dict | None = None
    ) -> None:
        """
        Notify registered callback about order state change.

        Args:
            order_id: Order ID that changed
            order_status: New order status
            order_data: Additional order data
        """
        callback = self._order_callbacks.get(order_id)
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order_id, order_status, order_data)
                else:
                    callback(order_id, order_status, order_data)
            except Exception:
                logger.exception("Error in order callback for %s", order_id)

    def get_fee_calculator(self) -> BluefinFeeCalculator:
        """
        Get the BluefinFeeCalculator instance for fee calculations.

        Returns:
            BluefinFeeCalculator instance
        """
        return self._fee_calculator

    def calculate_order_fees(
        self, notional_value: Decimal, is_maker: bool = True
    ) -> BluefinFees:
        """
        Calculate fees for an order.

        Args:
            notional_value: Notional value of the order
            is_maker: Whether this is a maker order (limit) or taker order (market)

        Returns:
            BluefinFees object with fee breakdown
        """
        return self._fee_calculator.calculate_fee_breakdown(
            notional_value=notional_value, use_limit_orders=is_maker
        )

    async def batch_cancel_orders(self, order_ids: list[str]) -> list[bool]:
        """
        Cancel multiple orders in batch for efficiency.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            List of success status for each order
        """
        if self.dry_run:
            logger.info(
                "PAPER TRADING: Simulating batch cancel of %d orders", len(order_ids)
            )
            return [True] * len(order_ids)

        results = []

        # Process cancellations with small delays to avoid rate limiting
        for order_id in order_ids:
            try:
                success = await self.cancel_order(order_id)
                results.append(success)

                # Small delay between cancellations
                if order_id != order_ids[-1]:  # Don't delay after the last one
                    await asyncio.sleep(0.05)

            except Exception as e:
                logger.warning("Failed to cancel order %s in batch: %s", order_id, e)
                results.append(False)

        success_count = sum(results)
        logger.info("Batch cancelled %d/%d orders", success_count, len(order_ids))

        return results

    async def replace_order(
        self, old_order_id: str, new_price: Decimal, new_quantity: Decimal | None = None
    ) -> Order | None:
        """
        Replace an existing order with new parameters (cancel + place new).

        Args:
            old_order_id: ID of order to replace
            new_price: New price for the order
            new_quantity: New quantity (optional, uses original if None)

        Returns:
            New order if successful, None otherwise
        """
        try:
            # Find the original order in our tracking
            original_order = None
            symbol = None

            for sym, orders in self._active_orders.items():
                for order in orders:
                    if order.id == old_order_id:
                        original_order = order
                        symbol = sym
                        break
                if original_order:
                    break

            if not original_order:
                logger.warning(
                    "Original order %s not found for replacement", old_order_id
                )
                return None

            # Use original quantity if not specified
            quantity = new_quantity or original_order.quantity

            logger.debug(
                "Replacing order %s: %s %s @ %s -> @ %s",
                old_order_id,
                original_order.side,
                quantity,
                original_order.price,
                new_price,
            )

            # Cancel the old order first
            cancel_success = await self.cancel_order(old_order_id)
            if not cancel_success:
                logger.warning(
                    "Failed to cancel order %s for replacement", old_order_id
                )
                return None

            # Remove from tracking
            if symbol and symbol in self._active_orders:
                self._active_orders[symbol] = [
                    o for o in self._active_orders[symbol] if o.id != old_order_id
                ]

            # Place new order
            new_order = await self.place_limit_order(
                symbol=symbol,
                side=original_order.side,
                quantity=quantity,
                price=new_price,
            )

            if new_order and symbol:
                # Add to tracking
                if symbol not in self._active_orders:
                    self._active_orders[symbol] = []
                self._active_orders[symbol].append(new_order)

                logger.info(
                    "Successfully replaced order %s with %s", old_order_id, new_order.id
                )

            return new_order

        except Exception:
            logger.exception("Failed to replace order %s", old_order_id)
            return None

    def get_active_orders_count(self, symbol: str | None = None) -> int:
        """
        Get count of active orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of active orders
        """
        if symbol:
            bluefin_symbol = self._convert_symbol(symbol)
            return len(self._active_orders.get(bluefin_symbol, []))
        return sum(len(orders) for orders in self._active_orders.values())

    def get_active_orders_for_symbol(self, symbol: str) -> list[Order]:
        """
        Get all active orders for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of active orders
        """
        bluefin_symbol = self._convert_symbol(symbol)
        return self._active_orders.get(bluefin_symbol, [])

    def clear_order_tracking(self, symbol: str | None = None) -> None:
        """
        Clear order tracking data.

        Args:
            symbol: Optional symbol to clear (clears all if None)
        """
        if symbol:
            bluefin_symbol = self._convert_symbol(symbol)
            self._active_orders.pop(bluefin_symbol, [])
            logger.debug("Cleared order tracking for %s", bluefin_symbol)
        else:
            self._active_orders.clear()
            logger.debug("Cleared all order tracking")

    async def start_order_book_monitoring(
        self, symbol: str, callback: Callable | None = None
    ) -> None:
        """
        Start real-time order book monitoring for a symbol.

        Args:
            symbol: Trading symbol to monitor
            callback: Optional callback for order book updates
        """
        # This would require WebSocket implementation in the service client
        # For now, we'll just log that monitoring would start
        bluefin_symbol = self._convert_symbol(symbol)
        logger.info(
            "Order book monitoring requested for %s (WebSocket implementation required)",
            bluefin_symbol,
        )

        # In a full implementation, this would:
        # 1. Subscribe to order book updates via WebSocket
        # 2. Call the callback with updated order book data
        # 3. Update the order book cache

    async def stop_order_book_monitoring(self, symbol: str) -> None:
        """
        Stop order book monitoring for a symbol.

        Args:
            symbol: Trading symbol to stop monitoring
        """
        bluefin_symbol = self._convert_symbol(symbol)
        logger.info("Stopping order book monitoring for %s", bluefin_symbol)

    # Market Making Optimization Methods

    async def batch_place_orders(self, orders: list[dict]) -> list[Order]:
        """
        Place multiple orders in batch for better efficiency.

        Args:
            orders: List of order dicts with symbol, side, price, quantity

        Returns:
            List of successfully placed orders
        """
        placed_orders = []

        logger.info("Batch placing %d orders", len(orders))

        # Group orders by symbol for better tracking
        orders_by_symbol = {}
        for order in orders:
            symbol = order["symbol"]
            if symbol not in orders_by_symbol:
                orders_by_symbol[symbol] = []
            orders_by_symbol[symbol].append(order)

        # Process orders with minimal delay
        for symbol, symbol_orders in orders_by_symbol.items():
            for order in symbol_orders:
                try:
                    placed_order = await self.place_limit_order(
                        symbol=order["symbol"],
                        side=cast("Literal['BUY', 'SELL']", order["side"]),
                        quantity=Decimal(str(order["quantity"])),
                        price=Decimal(str(order["price"])),
                    )

                    if placed_order:
                        placed_orders.append(placed_order)

                        # Track the order
                        bluefin_symbol = self._convert_symbol(order["symbol"])
                        if bluefin_symbol not in self._active_orders:
                            self._active_orders[bluefin_symbol] = []
                        self._active_orders[bluefin_symbol].append(placed_order)

                    # Minimal delay for rate limiting
                    await asyncio.sleep(0.02)

                except Exception as e:
                    logger.warning("Failed to place order in batch: %s", e)
                    continue

        logger.info(
            "Successfully placed %d/%d orders in batch", len(placed_orders), len(orders)
        )

        return placed_orders

    async def optimize_order_placement(
        self, order_levels: list[dict], symbol: str
    ) -> list[Order]:
        """
        Optimized order placement with intelligent sequencing and error recovery.

        Args:
            order_levels: List of order level dicts
            symbol: Trading symbol

        Returns:
            List of successfully placed orders
        """
        bluefin_symbol = self._convert_symbol(symbol)

        # Sort orders by distance from mid-price for optimal placement sequence
        try:
            spread_data = await self.get_current_spread(symbol)
            mid_price = spread_data.get("mid_price", Decimal(0))

            if mid_price > 0:
                # Sort by distance from mid price (closest first)
                order_levels.sort(
                    key=lambda ol: abs(Decimal(str(ol["price"])) - mid_price)
                )
                logger.debug(
                    "Optimized order placement sequence by distance from mid-price"
                )

        except Exception:
            logger.warning("Could not optimize order sequence, using original order")

        # Use batch placement with optimized error handling
        return await self.batch_place_orders(
            [
                {
                    "symbol": bluefin_symbol,
                    "side": ol["side"],
                    "price": ol["price"],
                    "quantity": ol["quantity"],
                }
                for ol in order_levels
            ]
        )

    async def fast_order_update(self, order_id: str, new_price: Decimal) -> bool:
        """
        Fast order price update using replace optimization.

        Args:
            order_id: Order to update
            new_price: New price

        Returns:
            True if successful
        """
        try:
            # Use the replace_order method for atomic update
            new_order = await self.replace_order(order_id, new_price)
            return new_order is not None

        except Exception:
            logger.exception("Fast order update failed for %s", order_id)
            return False

    def get_order_latency_stats(self) -> dict:
        """
        Get order placement latency statistics for performance monitoring.

        Returns:
            Dictionary with latency metrics
        """
        # This would require actual latency tracking in a full implementation
        # For now, return mock stats
        return {
            "avg_placement_latency_ms": 50.0,
            "avg_cancellation_latency_ms": 30.0,
            "success_rate_pct": 98.5,
            "rate_limit_hits": 0,
            "connection_status": "healthy",
        }

    def enable_high_frequency_mode(self) -> None:
        """
        Enable optimizations for high-frequency market making.
        """
        # Reduce rate limiter constraints for market making
        self._rate_limiter.max_requests = min(
            self._rate_limiter.max_requests * 2,
            100,  # Cap at 100 requests per window
        )

        logger.info(
            "Enabled high-frequency mode: %d requests per %ds",
            self._rate_limiter.max_requests,
            self._rate_limiter.window_seconds,
        )

    def disable_high_frequency_mode(self) -> None:
        """
        Disable high-frequency optimizations and return to normal limits.
        """
        # Reset to original rate limits
        self._rate_limiter.max_requests = settings.exchange.rate_limit_requests * 3

        logger.info(
            "Disabled high-frequency mode: %d requests per %ds",
            self._rate_limiter.max_requests,
            self._rate_limiter.window_seconds,
        )

    async def preload_market_data(self, symbols: list[str]) -> None:
        """
        Preload market data for multiple symbols to reduce latency.

        Args:
            symbols: List of symbols to preload
        """
        logger.info("Preloading market data for %d symbols", len(symbols))

        # Preload order books and spreads concurrently
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_order_book_depth(symbol, levels=5))
            tasks.append(self.get_current_spread(symbol))

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Market data preloaded successfully")
        except Exception:
            logger.warning("Some market data preloading failed")

    def get_market_making_metrics(self) -> dict:
        """
        Get comprehensive metrics for market making performance.

        Returns:
            Dictionary with market making metrics
        """
        total_active_orders = sum(
            len(orders) for orders in self._active_orders.values()
        )

        return {
            "total_active_orders": total_active_orders,
            "symbols_with_orders": len(self._active_orders),
            "order_callbacks_registered": len(self._order_callbacks),
            "cached_order_books": len(self._order_book_cache),
            "fee_calculator_available": self._fee_calculator is not None,
            "high_frequency_mode": self._rate_limiter.max_requests
            > settings.exchange.rate_limit_requests * 3,
            "connection_health": self.is_connected(),
            "dry_run_mode": self.dry_run,
        }

    # Functional Programming Interface

    def get_functional_adapter(self) -> "BluefinExchangeAdapter | None":
        """
        Get the functional adapter for side-effect-free operations.

        Returns:
            BluefinExchangeAdapter instance or None if not available
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
            logger.exception("Error in functional order placement: %s", e)
            return None
