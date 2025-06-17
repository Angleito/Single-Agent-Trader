"""Main CLI entry point for the AI Trading Bot."""

# ruff: noqa: E402
# CRITICAL: Initialize comprehensive warning suppression before ANY imports
# This must be the very first code that runs to catch import-time warnings
import os
import sys
import warnings

# Clear any existing warning registry and set up fresh
warnings.resetwarnings()
try:
    current_module = sys.modules[__name__]
    if not hasattr(current_module, "__warningregistry__"):
        current_module.__warningregistry__ = {}
except (AttributeError, TypeError):
    # Some modules don't support setting attributes
    # This is fine, warnings will still be filtered
    pass

# Set environment variable to suppress warnings at the Python level
os.environ["PYTHONWARNINGS"] = (
    "ignore::UserWarning,ignore::DeprecationWarning,ignore::SyntaxWarning"
)

# Nuclear option: ignore ALL warnings temporarily during imports
warnings.filterwarnings("ignore")

# Then apply specific comprehensive filters
message_patterns = [
    r".*pkg_resources.*",
    r".*deprecated.*",
    r".*slated.*removal.*",
    r".*escape sequence.*",
    r".*setup\.py.*",
    r".*distutils.*",
    r".*importlib.*",
    r".*setuptools.*",
    r".*pandas_ta.*",
]

warning_categories = [
    UserWarning,
    DeprecationWarning,
    FutureWarning,
    SyntaxWarning,
    ImportWarning,
    RuntimeWarning,
]

# Apply comprehensive message-based filters
for pattern in message_patterns:
    for category in warning_categories:
        warnings.filterwarnings("ignore", message=pattern, category=category)

# Apply module-based filters for problematic modules
problematic_modules = [
    "pkg_resources",
    "pandas_ta",
    "setuptools",
    "distutils",
    "importlib_metadata",
    "_distutils_hack",
]
for module_name in problematic_modules:
    for category in warning_categories:
        warnings.filterwarnings("ignore", category=category, module=module_name)
        warnings.filterwarnings("ignore", category=category, module=f"{module_name}.*")
        warnings.filterwarnings(
            "ignore", category=category, module=f".*{module_name}.*"
        )

# Global catch-all filters - be very aggressive
warnings.filterwarnings("ignore", message=r".*pkg_resources.*")
warnings.filterwarnings("ignore", module=r".*pkg_resources.*")
warnings.filterwarnings("ignore", module=r".*pandas_ta.*")

# Use simplefilter to ignore by category globally
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", SyntaxWarning)

# Standard library imports
import asyncio
import logging
import signal
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party imports
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# from .websocket_publisher import WebSocketPublisher  # TODO: Implement WebSocketPublisher
from .command_consumer import CommandConsumer

# Local imports
from .config import Settings, create_settings
from .data.dominance import DominanceCandleBuilder, DominanceDataProvider
from .data.market import MarketDataProvider

if TYPE_CHECKING:
    from .data.bluefin_market import BluefinMarketDataProvider

    # Union type for all possible market data providers
    MarketDataProviderType = MarketDataProvider | BluefinMarketDataProvider | None
else:
    MarketDataProviderType = MarketDataProvider | None
from .exchange.factory import ExchangeFactory
from .indicators.vumanchu import VuManChuIndicators
from .learning.experience_manager import ExperienceManager
from .mcp.memory_server import MCPMemoryServer
from .mcp.omnisearch_client import OmniSearchClient
from .paper_trading import PaperTradingAccount
from .position_manager import PositionManager
from .risk import RiskManager
from .strategy.llm_agent import LLMAgent
from .strategy.memory_enhanced_agent import MemoryEnhancedLLMAgent
from .types import IndicatorData, MarketState, Position, TradeAction
from .utils import setup_warnings_suppression
from .validator import TradeValidator

console = Console()


class TradingEngine:
    """
    Main trading engine that orchestrates all components.

    Manages the complete trading loop including:
    - Market data ingestion
    - Technical indicator calculation
    - LLM-based decision making
    - Risk management validation
    - Trade execution
    - Position tracking and monitoring
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval: str = "15s",
        config_file: str | None = None,
        dry_run: bool = True,
    ):
        """
        Initialize the trading engine.

        Args:
            symbol: Trading symbol
            interval: Candle interval
            config_file: Optional configuration file path
            dry_run: Whether to run in dry-run mode
        """
        print("DEBUG: TradingEngine.__init__ started")
        self.symbol = symbol
        self.interval = interval
        self.dry_run = dry_run
        self._running = False
        self._shutdown_requested = False
        self._memory_available = False  # Initialize early to prevent AttributeError

        print("DEBUG: About to load configuration")
        # Load configuration
        self.settings = self._load_configuration(config_file, dry_run)
        print("DEBUG: Configuration loaded")

        # Setup logging
        print("DEBUG: About to setup logging")
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        print("DEBUG: Logging setup complete")

        # Initialize paper trading if in dry run mode
        print("DEBUG: About to initialize paper trading")
        self.paper_account = None
        if self.dry_run:
            print("DEBUG: Accessing paper_trading settings...")
            balance = self.settings.paper_trading.starting_balance
            print(f"DEBUG: Starting balance: {balance}")
            print("DEBUG: Creating PaperTradingAccount...")
            self.paper_account = PaperTradingAccount(starting_balance=balance)
            print("DEBUG: PaperTradingAccount created successfully")

        # Initialize components (market data will be initialized after exchange connection)
        self.market_data: MarketDataProviderType = None
        self.logger.debug("About to initialize VuManChu indicators...")
        print("DEBUG: About to create VuManChuIndicators()")
        self.indicator_calc = VuManChuIndicators()
        print("DEBUG: VuManChuIndicators created successfully")
        print("DEBUG: About to log VuManChu success...")
        self.logger.debug("VuManChu indicators initialized successfully")
        print("DEBUG: Logged VuManChu success")
        print("DEBUG: About to set actual_trading_symbol")
        self.actual_trading_symbol = symbol  # Will be updated if futures are enabled
        print("DEBUG: Set actual_trading_symbol")

        # Initialize MCP memory components if enabled
        print("DEBUG: About to initialize MCP memory components")
        self.logger.debug("About to initialize MCP memory components...")
        self.memory_server = None
        self.experience_manager = None
        self._memory_available = False
        self.logger.debug("MCP memory components initialized")

        # Initialize OmniSearch client if enabled
        self.logger.debug("About to initialize OmniSearch client...")
        self.omnisearch_client = None
        if self.settings.omnisearch.enabled:
            self.logger.info("OmniSearch integration enabled, initializing client...")
            try:
                self.omnisearch_client = OmniSearchClient(
                    server_url=self.settings.omnisearch.server_url,
                    api_key=(
                        self.settings.omnisearch.api_key.get_secret_value()
                        if self.settings.omnisearch.api_key
                        else None
                    ),
                    enable_cache=True,
                    cache_ttl=self.settings.omnisearch.cache_ttl_seconds,
                    rate_limit_requests=self.settings.omnisearch.rate_limit_requests_per_minute,
                    rate_limit_window=60,
                )
                self.logger.info("Successfully initialized OmniSearch client")
            except Exception as e:
                self.logger.error(f"Failed to initialize OmniSearch client: {e}")
                self.logger.warning("Continuing without OmniSearch integration")
                self.omnisearch_client = None

        # Initialize WebSocket publisher for real-time dashboard integration
        self.websocket_publisher = None
        if self.settings.system.enable_websocket_publishing:
            self.logger.info("WebSocket publishing enabled, initializing publisher...")
            try:
                # TODO: Implement WebSocketPublisher class
                # self.websocket_publisher = WebSocketPublisher(self.settings)
                self.logger.warning(
                    "WebSocketPublisher not yet implemented, skipping initialization"
                )
                self.websocket_publisher = None
            except Exception as e:
                self.logger.error(f"Failed to initialize WebSocket publisher: {e}")
                self.logger.warning("Continuing without WebSocket publishing")
                self.websocket_publisher = None

        # Initialize Command Consumer for bidirectional dashboard control
        self.command_consumer = None
        if self.settings.system.enable_websocket_publishing:  # Use same setting for now
            self.logger.info(
                "Dashboard control enabled, initializing command consumer..."
            )
            try:
                self.command_consumer = CommandConsumer()
                self._register_command_callbacks()
                self.logger.info("Successfully initialized command consumer")
            except Exception as e:
                self.logger.error(f"Failed to initialize command consumer: {e}")
                self.logger.warning("Continuing without dashboard control")
                self.command_consumer = None

        # Initialize LLM agent (will be either LLMAgent or MemoryEnhancedLLMAgent)
        self.llm_agent: LLMAgent | MemoryEnhancedLLMAgent

        if self.settings.mcp.enabled:
            self.logger.info("MCP memory system enabled, initializing components...")
            try:
                self.memory_server = MCPMemoryServer(
                    server_url=self.settings.mcp.server_url,
                    api_key=(
                        self.settings.mcp.memory_api_key.get_secret_value()
                        if self.settings.mcp.memory_api_key
                        else None
                    ),
                )
                self.experience_manager = ExperienceManager(self.memory_server)

                # Use memory-enhanced agent
                self.llm_agent = MemoryEnhancedLLMAgent(
                    model_provider=self.settings.llm.provider,
                    model_name=self.settings.llm.model_name,
                    memory_server=self.memory_server,
                    omnisearch_client=self.omnisearch_client,
                )
                self.logger.info("Successfully initialized memory-enhanced agent")
                self._memory_available = True
            except Exception as e:
                self.logger.error(f"Failed to initialize MCP components: {e}")
                self.logger.warning("Falling back to standard LLM agent")
                self.llm_agent = LLMAgent(
                    model_provider=self.settings.llm.provider,
                    model_name=self.settings.llm.model_name,
                    omnisearch_client=self.omnisearch_client,
                )
        else:
            # Standard LLM agent without memory
            self.llm_agent = LLMAgent(
                model_provider=self.settings.llm.provider,
                model_name=self.settings.llm.model_name,
                omnisearch_client=self.omnisearch_client,
            )

        self.validator = TradeValidator()
        self.position_manager = PositionManager(
            paper_trading_account=self.paper_account,
            use_fifo=self.settings.trading.use_fifo_accounting,
        )
        self.risk_manager = RiskManager(position_manager=self.position_manager)
        self.exchange_client = ExchangeFactory.create_exchange(
            exchange_type=self.settings.exchange.exchange_type,
            dry_run=self.dry_run,
        )

        # Connect exchange validation failures to circuit breaker
        if hasattr(self.exchange_client, "set_failure_callback"):
            self.exchange_client.set_failure_callback(
                self.risk_manager.circuit_breaker.record_failure
            )

        # Initialize dominance data provider if enabled
        self.dominance_provider = None
        if self.settings.dominance.enable_dominance_data:
            self.dominance_provider = DominanceDataProvider(
                data_source=self.settings.dominance.data_source,
                api_key=(
                    self.settings.dominance.api_key.get_secret_value()
                    if self.settings.dominance.api_key
                    else None
                ),
                update_interval=self.settings.dominance.update_interval,
            )

        # Position tracking
        self.current_position = Position(
            symbol=symbol,
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC),
        )

        # Performance tracking
        self.trade_count = 0
        self.successful_trades = 0
        self.total_pnl = Decimal("0")
        self.start_time = datetime.now(UTC)

        # Trading interval control
        self.last_trade_time: datetime | None = None
        self.last_candle_analysis_time: datetime | None = (
            None  # Track when we last analyzed a candle
        )
        self.trading_enabled = False  # Will be enabled after data validation
        self.data_validation_complete = False

        self.logger.info(f"Initialized TradingEngine for {symbol} at {interval}")

    @property
    def exchange(self) -> Any:
        """Backward compatibility property - maps to exchange_client."""
        return self.exchange_client

    def _register_command_callbacks(self) -> None:
        """Register callback functions for dashboard commands."""
        if not self.command_consumer:
            return

        # Register emergency stop callback
        self.command_consumer.register_callback(
            "emergency_stop", self._handle_emergency_stop
        )

        # Register pause/resume trading callbacks
        self.command_consumer.register_callback(
            "pause_trading", self._handle_pause_trading
        )
        self.command_consumer.register_callback(
            "resume_trading", self._handle_resume_trading
        )

        # Register risk limit update callback
        self.command_consumer.register_callback(
            "update_risk_limits", self._handle_update_risk_limits
        )

        # Register manual trade callback
        self.command_consumer.register_callback(
            "manual_trade", self._handle_manual_trade
        )

        self.logger.info("Registered all command callbacks for dashboard integration")

    async def _handle_emergency_stop(self) -> None:
        """Handle emergency stop command from dashboard."""
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED FROM DASHBOARD")
        self._shutdown_requested = True
        self.trading_enabled = False

        # Close all positions if possible
        try:
            if self.current_position.side != "FLAT":
                self.logger.info("Emergency stop: Closing all positions")
                await self._close_all_positions()
        except Exception as e:
            self.logger.error(f"Error closing positions during emergency stop: {e}")

        # Publish emergency stop status
        if self.websocket_publisher:
            await self.websocket_publisher.publish_system_status(
                status="emergency_stopped",
                health=False,
                message="Emergency stop activated from dashboard",
            )

    async def _handle_pause_trading(self) -> None:
        """Handle pause trading command from dashboard."""
        self.logger.warning("📍 Trading paused from dashboard")
        self.trading_enabled = False

        if self.websocket_publisher:
            await self.websocket_publisher.publish_system_status(
                status="trading_paused",
                health=True,
                message="Trading paused from dashboard",
            )

    async def _handle_resume_trading(self) -> None:
        """Handle resume trading command from dashboard."""
        self.logger.info("▶️ Trading resumed from dashboard")
        self.trading_enabled = True
        self._shutdown_requested = False

        if self.websocket_publisher:
            await self.websocket_publisher.publish_system_status(
                status="trading_active",
                health=True,
                message="Trading resumed from dashboard",
            )

    async def _handle_update_risk_limits(self, parameters: dict) -> None:
        """Handle risk limit update command from dashboard."""
        self.logger.info(f"Updating risk limits from dashboard: {parameters}")

        try:
            # Update risk manager settings
            if "max_position_size" in parameters:
                self.risk_manager.max_position_size = parameters["max_position_size"]

            if "stop_loss_percentage" in parameters:
                self.risk_manager.stop_loss_percentage = (
                    parameters["stop_loss_percentage"] / 100.0
                )

            if "max_daily_loss" in parameters:
                self.risk_manager.max_daily_loss = parameters["max_daily_loss"]

            self.logger.info("Risk limits updated successfully")

            if self.websocket_publisher:
                await self.websocket_publisher.publish_system_status(
                    status="risk_limits_updated",
                    health=True,
                    message=f"Risk limits updated: {parameters}",
                )

        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")
            raise

    async def _handle_manual_trade(self, parameters: dict) -> bool:
        """Handle manual trade command from dashboard."""
        self.logger.info(f"Executing manual trade from dashboard: {parameters}")

        try:
            # Create trade action from parameters
            trade_action = TradeAction(
                action=parameters["action"].upper(),
                size_pct=int(parameters["size_percentage"]),
                take_profit_pct=2.0,  # Default take profit
                stop_loss_pct=1.5,  # Default stop loss
                rationale=parameters.get("reason", "Manual trade from dashboard"),
            )

            # Execute the trade
            success = await self._execute_trade_action(trade_action)

            if success:
                self.logger.info("Manual trade executed successfully")
                if self.websocket_publisher:
                    await self.websocket_publisher.publish_trade_execution(
                        order_id=f"manual_{int(time.time())}",
                        symbol=trade_action.symbol,
                        side=trade_action.action,
                        size=str(trade_action.size_pct),
                        price="market",
                        status="filled",
                        manual=True,
                    )
            else:
                self.logger.warning("Manual trade execution failed")

            return success

        except Exception as e:
            self.logger.error(f"Error executing manual trade: {e}")
            return False

    async def _close_all_positions(self) -> None:
        """Close all open positions (used for emergency stop)."""
        try:
            if self.current_position.side != "FLAT":
                close_action = TradeAction(
                    action="CLOSE",
                    size_pct=0,  # Close actions use size_pct=0
                    take_profit_pct=0.0,
                    stop_loss_pct=0.0,
                    rationale="Emergency position closure",
                )
                await self._execute_trade_action(close_action)
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

    def _ensure_market_data_available(self) -> bool:
        """
        Ensure market data provider is initialized and connected.

        Returns:
            True if market data is available and connected, False otherwise
        """
        if self.market_data is None:
            self.logger.error("Market data provider not initialized")
            return False

        if not self.market_data.is_connected():
            self.logger.warning("Market data provider not connected")
            return False

        return True

    def _get_market_data_provider(self):
        """
        Get the market data provider with proper type safety.

        Returns:
            The market data provider instance

        Raises:
            RuntimeError: If market data provider is not initialized
        """
        if self.market_data is None:
            raise RuntimeError("Market data provider not initialized")
        return self.market_data

    def _can_trade_now(self) -> bool:
        """
        Check if trading is currently allowed based on data availability and timing.
        For high-frequency scalping, we trade on every new candle completion (1s-15s).

        Returns:
            True if trading is allowed, False otherwise
        """
        # Check if trading is enabled (based on data validation)
        if not self.trading_enabled:
            return False

        # Check if data validation is complete
        if not self.data_validation_complete:
            return False

        # Check if market data is available
        if not self._ensure_market_data_available():
            return False

        # Get latest candle to check if a new one has completed
        # Handle both sync and async providers
        try:
            import inspect

            latest_data = []
            if self.market_data is not None and hasattr(
                self.market_data, "get_latest_ohlcv"
            ):
                method = self.market_data.get_latest_ohlcv
                if inspect.iscoroutinefunction(method):
                    # Can't await in non-async method, defer to caller
                    self.logger.debug(
                        "Async method detected, skipping fresh data check"
                    )
                    return False
                else:
                    latest_data = method(limit=1)
            else:
                latest_data = []
        except Exception as e:
            self.logger.warning(f"Error checking latest data: {e}")
            return False

        if not latest_data:
            self.logger.debug("📊 No market data available")
            return False

        latest_candle = latest_data[-1]

        # Check if this is a new candle we haven't analyzed yet
        if self.last_candle_analysis_time is not None:
            # Ensure both timestamps have timezone info for comparison
            candle_time = latest_candle.timestamp
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=UTC)

            last_analysis_time = self.last_candle_analysis_time
            if last_analysis_time.tzinfo is None:
                last_analysis_time = last_analysis_time.replace(tzinfo=UTC)

            if candle_time <= last_analysis_time:
                # This is the same candle we already analyzed
                return False

        # For high-frequency scalping: analyze on every new candle completion
        # A 15-second candle completes every 15 seconds (e.g., 10:00:00, 10:00:15, 10:00:30)
        current_time = datetime.now(UTC)
        candle_interval_seconds = self._get_interval_seconds(self.interval)

        # Check if enough time has passed since last analysis (at least candle interval)
        if self.last_candle_analysis_time is not None:
            # Ensure both timestamps have timezone info for comparison
            last_analysis_time = self.last_candle_analysis_time
            if last_analysis_time.tzinfo is None:
                last_analysis_time = last_analysis_time.replace(tzinfo=UTC)

            time_since_last_analysis = (
                current_time - last_analysis_time
            ).total_seconds()
            if time_since_last_analysis < candle_interval_seconds:
                return False

        # Check minimum interval between actual trades (can be different from analysis)
        if self.last_trade_time is not None:
            min_interval = self.settings.trading.min_trading_interval_seconds

            # Ensure both timestamps have timezone info for comparison
            last_trade_time = self.last_trade_time
            if last_trade_time.tzinfo is None:
                last_trade_time = last_trade_time.replace(tzinfo=UTC)

            time_since_last_trade = (current_time - last_trade_time).total_seconds()

            if time_since_last_trade < min_interval:
                self.logger.debug(
                    f"⏱️ Waiting for trade interval: {time_since_last_trade:.1f}s / {min_interval}s"
                )
                return False

        return True

    def _load_configuration(self, config_file: str | None, dry_run: bool) -> Settings:
        """Load and validate configuration."""
        if config_file:
            settings = Settings.load_from_file(config_file)
        else:
            settings = create_settings()

        # Override dry_run mode if specified
        if dry_run != settings.system.dry_run:
            system_settings = settings.system.model_copy(update={"dry_run": dry_run})
            settings = settings.model_copy(update={"system": system_settings})

        # Validate configuration for trading
        warnings = settings.validate_trading_environment()
        if warnings:
            console.print("[yellow]Configuration warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        return settings

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.settings.system.log_level)

        # Create logs directory if needed
        if self.settings.system.log_file_path:
            self.settings.system.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        handlers: list[logging.Handler] = []
        if self.settings.system.log_to_console:
            handlers.append(logging.StreamHandler())
        if self.settings.system.log_to_file:
            handlers.append(
                logging.FileHandler(str(self.settings.system.log_file_path))
            )

        logging.basicConfig(
            level=log_level,
            format=self.settings.system.log_format,
            handlers=handlers,
        )

        # Remove None handlers
        root_logger = logging.getLogger()
        root_logger.handlers = [h for h in root_logger.handlers if h is not None]

    async def run(self) -> None:
        """
        Main trading loop entry point.

        Orchestrates the complete trading process with error handling
        and graceful shutdown capabilities.
        """
        self.logger.info("Starting trading engine...")

        # Track if we've already started shutdown to prevent double shutdown
        shutdown_called = False

        try:
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Initialize all components
            await self._initialize_components()

            # Display startup summary
            self._display_startup_summary()

            # Log initial data status for debugging
            if self.market_data is not None:
                data_status = self.market_data.get_data_status()
                self.logger.info(f"Initial market data status: {data_status}")
            else:
                self.logger.warning("Market data not initialized during startup")

            # Start main trading loop
            await self._main_trading_loop()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down gracefully...")
            console.print(
                "\n[yellow]Received interrupt signal, shutting down...[/yellow]"
            )
        except Exception as e:
            self.logger.error(f"Critical error in trading engine: {e}")
            console.print(f"[red]Critical error: {e}[/red]")
            raise
        finally:
            # Ensure shutdown is called exactly once
            if not shutdown_called:
                shutdown_called = True
                try:
                    await self._shutdown()
                except Exception as e:
                    self.logger.error(f"Error in shutdown: {e}")
                    # Force cleanup of dominance provider session as last resort
                    if hasattr(self, "dominance_provider") and self.dominance_provider:
                        if hasattr(self.dominance_provider, "_session"):
                            self.dominance_provider._session = None

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            self.logger.info(f"Received signal {signum}, requesting shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _initialize_components(self) -> None:
        """Initialize all trading components."""
        console.print("[cyan]Initializing trading components...[/cyan]")

        # Initialize exchange client first to determine the correct trading symbol
        console.print("  • Connecting to exchange...")
        connected = await self.exchange_client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to exchange")

        # Get the actual trading symbol (futures contract if enabled)
        if self.exchange_client.enable_futures:
            console.print("  • Determining active futures contract...")
            self.actual_trading_symbol = await self.exchange_client.get_trading_symbol(
                self.symbol
            )
            console.print(
                f"    Using futures contract: [green]{self.actual_trading_symbol}[/green]"
            )
        else:
            self.actual_trading_symbol = self.symbol
            console.print(
                f"    Using spot symbol: [green]{self.actual_trading_symbol}[/green]"
            )

        # Initialize market data provider based on exchange type and trading mode
        console.print("  • Connecting to market data feed...")

        # Check if this is high-frequency trading (interval < 1 minute)
        interval_seconds = self._get_interval_seconds(self.interval)
        is_high_frequency = interval_seconds <= 60  # 1 minute or less

        if self.settings.exchange.exchange_type == "bluefin":
            # Always try BluefinMarketDataProvider first for Bluefin exchange
            try:
                from .data.bluefin_market import BluefinMarketDataProvider

                self.logger.info(
                    f"Using Bluefin native market data provider for {self.symbol}"
                )
                self.market_data = BluefinMarketDataProvider(self.symbol, self.interval)
                console.print("    Using Bluefin DEX native market data")
            except ImportError as e:
                self.logger.warning(f"BluefinMarketDataProvider not available: {e}")
                # Try real-time provider for high-frequency trading
                if is_high_frequency:
                    try:
                        from .data.realtime_market import RealtimeMarketDataProvider

                        self.logger.info(
                            f"Using real-time WebSocket market data provider for HF trading: {self.symbol}"
                        )
                        # Convert interval to seconds for real-time provider
                        realtime_intervals = [interval_seconds]
                        if interval_seconds > 1:
                            realtime_intervals.append(
                                1
                            )  # Always include 1-second candles for scalping
                        if 5 not in realtime_intervals and interval_seconds != 5:
                            realtime_intervals.append(5)  # Include 5-second candles
                        self.market_data = RealtimeMarketDataProvider(
                            self.symbol, realtime_intervals
                        )
                        console.print(
                            f"    Using real-time WebSocket data for HF trading ({realtime_intervals}s intervals)"
                        )
                    except ImportError:
                        self.logger.warning(
                            "RealtimeMarketDataProvider not available, falling back to standard provider"
                        )
                        self.market_data = MarketDataProvider(
                            self.symbol, self.interval
                        )
                        console.print(
                            "    Using standard market data provider (real-time module not available)"
                        )
                else:
                    # Fallback to standard market data provider
                    self.logger.info(
                        f"Falling back to standard MarketDataProvider for {self.symbol}"
                    )
                    self.market_data = MarketDataProvider(self.symbol, self.interval)
                    console.print("    Using fallback market data provider for Bluefin")
        else:
            # Use Coinbase market data for Coinbase trading
            market_data_symbol = self.actual_trading_symbol
            self.logger.info(f"Using Coinbase market data for {market_data_symbol}")
            self.market_data = MarketDataProvider(market_data_symbol, self.interval)
            console.print(f"    Using Coinbase market data for {market_data_symbol}")

        # Ensure market_data is properly initialized before connecting
        if self.market_data is None:
            raise RuntimeError("Market data provider was not properly initialized")
        await self.market_data.connect()

        # Verify LLM agent is available
        console.print("  • Verifying LLM agent...")
        if not self.llm_agent.is_available():
            console.print(
                "[yellow]    Warning: LLM not available, using fallback logic[/yellow]"
            )

        # Connect to OmniSearch if available
        if self.omnisearch_client:
            console.print("  • Connecting to OmniSearch service...")
            try:
                connected = await self.omnisearch_client.connect()
                if connected:
                    console.print("    [green]✓ OmniSearch connected[/green]")
                else:
                    console.print(
                        "    [yellow]⚠ OmniSearch service unavailable[/yellow]"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to connect to OmniSearch: {e}")
                console.print("    [yellow]⚠ OmniSearch connection failed[/yellow]")

        # Load initial market data
        console.print("  • Loading initial market data...")
        await self._wait_for_initial_data()

        # Initialize MCP experience manager if enabled
        if self.experience_manager:
            console.print("  • Starting experience tracking...")
            try:
                await self.experience_manager.start()
                console.print("    [green]✓ Experience tracking started[/green]")

                # Log memory system status and pattern statistics
                if (
                    self.memory_server
                    and hasattr(self.memory_server, "_connected")
                    and self.memory_server._connected
                ):
                    self._memory_available = True
                    memory_count = len(self.memory_server.memory_cache)
                    console.print(
                        f"    [cyan]📊 {memory_count} stored experiences loaded[/cyan]"
                    )

                    # Log pattern performance if we have enough data
                    if memory_count >= 10:
                        try:
                            pattern_stats = (
                                await self.memory_server.get_pattern_statistics()
                            )
                            if pattern_stats:
                                self.logger.info("=== Pattern Performance Summary ===")
                                sorted_patterns = sorted(
                                    pattern_stats.items(),
                                    key=lambda x: x[1]["success_rate"] * x[1]["count"],
                                    reverse=True,
                                )[:5]
                                for pattern, stats in sorted_patterns:
                                    if (
                                        stats["count"] >= 3
                                    ):  # Only show patterns with enough samples
                                        self.logger.info(
                                            f"  {pattern}: {stats['success_rate']:.1%} win rate "
                                            f"({stats['count']} trades, avg PnL=${stats['avg_pnl']:.2f})"
                                        )
                        except Exception as e:
                            self.logger.debug(
                                f"Could not retrieve pattern statistics: {e}"
                            )

            except Exception as e:
                self.logger.warning(f"Failed to start experience manager: {e}")
                console.print("    [yellow]⚠ Experience tracking unavailable[/yellow]")

        # Initialize dominance data provider
        if self.dominance_provider:
            console.print("  • Connecting to stablecoin dominance data...")
            try:
                await self.dominance_provider.connect()
                console.print("    [green]✓ Dominance data connected[/green]")
            except Exception as e:
                self.logger.warning(f"Failed to connect dominance data: {e}")
                console.print("    [yellow]⚠ Dominance data unavailable[/yellow]")

        # Check for existing positions on exchange
        console.print("  • Checking for existing positions...")
        await self._reconcile_positions()

        # Initialize WebSocket publisher if enabled
        if self.websocket_publisher:
            console.print("  • Connecting to dashboard WebSocket...")
            try:
                connected = await self.websocket_publisher.initialize()
                if connected:
                    console.print("    [green]✓ Dashboard WebSocket connected[/green]")
                    await self.websocket_publisher.publish_system_status(
                        status="initialized", health=True
                    )
                else:
                    console.print(
                        "    [yellow]⚠ Dashboard WebSocket unavailable[/yellow]"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to connect to dashboard WebSocket: {e}")
                console.print(
                    "    [yellow]⚠ Dashboard WebSocket connection failed[/yellow]"
                )

        # Initialize Command Consumer if enabled
        if self.command_consumer:
            console.print("  • Starting dashboard command consumer...")
            try:
                await self.command_consumer.initialize()
                # Start polling in background
                asyncio.create_task(self.command_consumer.start_polling())
                console.print("    [green]✓ Dashboard command consumer started[/green]")
            except Exception as e:
                self.logger.warning(f"Failed to start command consumer: {e}")
                console.print(
                    "    [yellow]⚠ Dashboard command consumer unavailable[/yellow]"
                )

        console.print("[green]✓ All components initialized successfully[/green]")

    async def _wait_for_initial_data(self) -> None:
        """Wait for sufficient market data to begin trading."""
        max_wait_time = 180  # 3 minutes to allow for initial data collection
        wait_start = datetime.now(UTC)
        historical_data_loaded = False
        websocket_data_received = False
        min_candles_required = self.settings.trading.min_candles_for_trading
        require_24h_data = self.settings.trading.require_24h_data_before_trading

        while True:
            elapsed_time = (datetime.now(UTC) - wait_start).total_seconds()
            if elapsed_time > max_wait_time:
                # Get detailed status before failing
                if self.market_data is not None:
                    status = self.market_data.get_data_status()
                    self.logger.error(
                        f"Timeout waiting for initial market data. Status: {status}"
                    )
                else:
                    self.logger.error(
                        "Timeout waiting for initial market data. Market data provider not initialized"
                    )
                raise RuntimeError(
                    f"Timeout waiting for initial market data after {max_wait_time}s"
                )

            # Check for historical data - handle both sync and async providers
            import inspect

            data = []
            if self.market_data is not None and hasattr(
                self.market_data, "get_latest_ohlcv"
            ):
                method = self.market_data.get_latest_ohlcv
                try:
                    if inspect.iscoroutinefunction(method):
                        data = await method(limit=500)  # Get more data for 24h check
                    else:
                        data = method(limit=500)  # Get more data for 24h check

                    # Safety check - ensure data is not a coroutine or Task
                    if inspect.iscoroutine(data):
                        self.logger.warning("Detected coroutine data, awaiting...")
                        data = await data
                    elif isinstance(data, asyncio.Task):
                        self.logger.warning("Detected asyncio.Task data, awaiting...")
                        data = await data
                    elif hasattr(data, "__await__"):
                        self.logger.warning("Detected awaitable object, awaiting...")
                        data = await data  # type: ignore[misc]

                except Exception as e:
                    self.logger.warning(f"Error getting market data: {e}")
                    data = []

            # Ensure data is a list/sequence before using len()
            if isinstance(data, asyncio.Task):
                self.logger.error(
                    f"Data is still an asyncio.Task: {data}. This should have been awaited."
                )
                data = []
            elif inspect.iscoroutine(data):
                self.logger.error(
                    f"Data is still a coroutine: {data}. This should have been awaited."
                )
                data = []
            elif not isinstance(data, list | tuple):
                self.logger.warning(
                    f"Unexpected data type: {type(data)}, converting to list"
                )
                data = list(data) if data else []

            if not historical_data_loaded:
                # Calculate how many candles represent 24 hours
                interval_seconds = self._get_interval_seconds(self.interval)
                candles_per_24h = (24 * 60 * 60) // interval_seconds

                if require_24h_data:
                    # For scalping, we don't need 24h of data - just enough for indicators
                    if len(data) >= min_candles_required:
                        hours_available = (len(data) * interval_seconds) / 3600
                        self.logger.info(
                            f"✅ Loaded {len(data)} historical candles ({hours_available:.1f} hours at {self.interval} intervals) for scalping analysis"
                        )
                        historical_data_loaded = True
                        self.trading_enabled = True
                    elif len(data) >= 50:
                        hours_available = (len(data) * interval_seconds) / 3600
                        self.logger.warning(
                            f"⚠️ Limited data available ({hours_available:.2f} hours, {len(data)} candles). "
                            f"Starting with reduced data for scalping..."
                        )
                        historical_data_loaded = True
                        self.trading_enabled = (
                            True  # Enable for scalping with limited data
                        )
                else:
                    # Just require minimum candles
                    if len(data) >= min_candles_required:
                        self.logger.info(
                            f"✅ Loaded {len(data)} historical candles (minimum {min_candles_required}) for analysis"
                        )
                        historical_data_loaded = True
                        self.trading_enabled = True
                    elif len(data) >= 50:
                        # Fallback with warning
                        self.logger.warning(
                            f"⚠️ Limited historical data available: {len(data)} candles. "
                            f"Indicators may be unreliable until more data is accumulated."
                        )
                        historical_data_loaded = True
                        self.trading_enabled = (
                            False  # Don't enable trading with limited data
                        )

            # Check for WebSocket data
            if (
                self.market_data is not None
                and self.market_data.has_websocket_data()
                and not websocket_data_received
            ):
                self.logger.info("📡 WebSocket is receiving real-time market data")
                websocket_data_received = True

            # We're ready when we have sufficient historical data and either:
            # 1. WebSocket is receiving data, OR
            # 2. We've waited at least 10 seconds for WebSocket (scalping needs to start fast)
            if historical_data_loaded:
                if websocket_data_received:
                    self.logger.info(
                        "🚀 Both historical and real-time data available, ready for scalping"
                    )
                    self.data_validation_complete = True
                    break
                elif elapsed_time > 10:
                    # After 10 seconds, proceed if we have historical data even without WebSocket
                    self.logger.warning(
                        "⚠️ Proceeding with historical data only for scalping. WebSocket data not yet received "
                        "(market may be closed or starting fast for scalping)"
                    )
                    self.data_validation_complete = True
                    break

            # Log progress every 10 seconds
            if int(elapsed_time) % 10 == 0 and elapsed_time > 0:
                try:
                    from .data.realtime_market import RealtimeMarketDataProvider

                    if (
                        RealtimeMarketDataProvider
                        and self.market_data is not None
                        and isinstance(self.market_data, RealtimeMarketDataProvider)
                    ):
                        # Real-time provider status
                        status = self.market_data.get_status()
                        tick_rate = status.get("tick_rate_per_second", 0)
                        current_price = status.get("current_price", "N/A")

                        self.logger.info(
                            f"⏳ Waiting for real-time data... Elapsed: {int(elapsed_time)}s\n"
                            f"   📊 Current candles: {len(data)} available\n"
                            f"   🌐 WebSocket connected: {status.get('websocket_connected', False)}\n"
                            f"   📈 Tick rate: {tick_rate:.1f} ticks/sec\n"
                            f"   💰 Current price: ${current_price}\n"
                            f"   ⚡ Trading enabled: {self.trading_enabled}"
                        )
                    else:
                        # Standard provider status
                        if self.market_data is not None:
                            status = self.market_data.get_data_status()
                        else:
                            status = {
                                "connected": False,
                                "websocket_data_received": False,
                            }
                        interval_seconds = self._get_interval_seconds(self.interval)
                        candles_per_24h = (24 * 60 * 60) // interval_seconds
                        hours_available = (
                            (len(data) * interval_seconds) / 3600 if data else 0
                        )

                        self.logger.info(
                            f"⏳ Waiting for data... Elapsed: {int(elapsed_time)}s\n"
                            f"   📊 Historical: {len(data)}/{candles_per_24h} candles ({hours_available:.1f}/24 hours)\n"
                            f"   🌐 WebSocket connected: {status.get('websocket_connected', False)}\n"
                            f"   📈 WebSocket data: {status.get('websocket_data_received', False)}\n"
                            f"   💰 Latest price: ${status.get('latest_price', 'N/A')}\n"
                            f"   ⚡ Trading enabled: {self.trading_enabled}"
                        )
                except ImportError:
                    # RealtimeMarketDataProvider not available, just log basic info
                    self.logger.info(
                        f"⏳ Waiting for data... Elapsed: {int(elapsed_time)}s\n"
                        f"   📊 Current candles: {len(data)} available\n"
                        f"   ⚡ Trading enabled: {self.trading_enabled}"
                    )

            await asyncio.sleep(1)

    def _display_startup_summary(self) -> None:
        """Display trading engine startup summary."""
        table = Table(title="Trading Engine Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        # Market data status
        try:
            from .data.realtime_market import RealtimeMarketDataProvider

            if (
                RealtimeMarketDataProvider
                and self.market_data is not None
                and isinstance(self.market_data, RealtimeMarketDataProvider)
            ):
                # Real-time provider status
                data_status = self.market_data.get_status()
                ws_status = (
                    "✓ Real-time data"
                    if data_status.get("websocket_connected", False)
                    else "⚠ Waiting for data"
                )

                # Show tick rate and current candles
                tick_rate = data_status.get("tick_rate_per_second", 0)
                tick_info = f"{tick_rate:.1f} ticks/sec" if tick_rate else "No ticks"
                details = f"{tick_info}, WebSocket: {ws_status}"

                table.add_row(
                    "Market Data (RT)",
                    "✓ Connected" if data_status["connected"] else "✗ Disconnected",
                    details,
                )
        except ImportError:
            # RealtimeMarketDataProvider not available
            RealtimeMarketDataProvider = None

        if not RealtimeMarketDataProvider or not isinstance(
            self.market_data, RealtimeMarketDataProvider
        ):
            # Standard provider status
            if self.market_data is None:
                data_status = {"websocket_data_received": False}
            else:
                data_status = self.market_data.get_data_status()
            ws_status = (
                "✓ Receiving data"
                if data_status.get("websocket_data_received", False)
                else "⚠ Waiting for data"
            )
            table.add_row(
                "Market Data",
                "✓ Connected" if data_status["connected"] else "✗ Disconnected",
                f"{data_status['cached_candles']} candles, WebSocket: {ws_status}",
            )

        # Exchange status
        exchange_status = self.exchange_client.get_connection_status()
        exchange_name = exchange_status.get(
            "exchange", self.exchange_client.exchange_name
        )
        exchange_details = []

        if exchange_status.get("is_decentralized", False):
            exchange_details.append(
                f"{exchange_status.get('network', 'mainnet')} network"
            )
            if exchange_status.get("blockchain"):
                exchange_details.append(f"({exchange_status['blockchain']})")
        else:
            exchange_details.append(
                f"{'Sandbox' if exchange_status.get('sandbox', False) else 'Live'} mode"
            )

        table.add_row(
            f"Exchange ({exchange_name})",
            "✓ Connected" if exchange_status["connected"] else "✗ Disconnected",
            " ".join(exchange_details),
        )

        # LLM status
        llm_status = self.llm_agent.get_status()
        table.add_row(
            "LLM Agent",
            "✓ Available" if llm_status["llm_available"] else "⚠ Fallback",
            f"{llm_status['model_provider']}:{llm_status['model_name']}",
        )

        # OmniSearch status
        omnisearch_status = (
            "✓ Enabled" if llm_status.get("omnisearch_enabled", False) else "✗ Disabled"
        )
        omnisearch_details = (
            "Market intelligence active"
            if llm_status.get("omnisearch_enabled", False)
            else "Standard analysis only"
        )
        table.add_row(
            "OmniSearch",
            omnisearch_status,
            omnisearch_details,
        )

        # Risk manager
        table.add_row(
            "Risk Manager",
            "✓ Active",
            f"Max size: {self.settings.trading.max_size_pct}%",
        )

        console.print(table)
        console.print()

    async def _main_trading_loop(self) -> None:
        """
        Main trading loop that runs continuously.

        Processes market data, calculates indicators, gets LLM decisions,
        validates trades, applies risk management, and executes orders.
        """
        self.logger.info("Starting main trading loop...")
        self._running = True

        loop_count = 0
        last_status_log = datetime.now(UTC)

        while self._running and not self._shutdown_requested:
            try:
                loop_start = datetime.now(UTC)
                loop_count += 1

                # Check if we have fresh market data
                if self.market_data is None:
                    self.logger.error(
                        "Market data provider not initialized, cannot continue trading"
                    )
                    break

                if not self.market_data.is_connected():
                    self.logger.warning(
                        "Market data connection lost, checking reconnection status..."
                    )
                    # Check if WebSocket handler is already reconnecting
                    data_status = self.market_data.get_data_status()
                    if data_status.get("reconnect_attempts", 0) > 0:
                        self.logger.info(
                            "WebSocket handler is already attempting reconnection, waiting..."
                        )
                        await asyncio.sleep(5)  # Wait for WebSocket reconnection
                        continue
                    else:
                        # Only attempt manual reconnection if WebSocket handler isn't trying
                        self.logger.warning("Attempting manual reconnection...")
                        if self.market_data is not None:
                            await self.market_data.connect()
                        continue

                # Get latest market data - handle different provider types
                import inspect

                try:
                    from .data.realtime_market import RealtimeMarketDataProvider

                    is_realtime_provider = RealtimeMarketDataProvider and isinstance(
                        self.market_data, RealtimeMarketDataProvider
                    )
                except ImportError:
                    RealtimeMarketDataProvider = None
                    is_realtime_provider = False

                if is_realtime_provider and self.market_data is not None:
                    # Real-time provider - get candles for the current trading interval
                    interval_seconds = self._get_interval_seconds(self.interval)

                    # Type check: ensure we have the right provider type
                    if hasattr(self.market_data, "get_candle_history"):
                        latest_data = self.market_data.get_candle_history(
                            interval_seconds, limit=200
                        )

                        # If we don't have enough historical data, try to get completed candles
                        if len(latest_data) < 50 and hasattr(
                            self.market_data, "tick_aggregator"
                        ):
                            # Force completion of current candles and retry
                            self.market_data.tick_aggregator.force_complete_candles(
                                self.symbol
                            )
                            latest_data = self.market_data.get_candle_history(
                                interval_seconds, limit=200
                            )
                    else:
                        # Fallback to standard provider method
                        latest_data = []

                elif self.market_data is not None and hasattr(
                    self.market_data, "get_latest_ohlcv"
                ):
                    method = self.market_data.get_latest_ohlcv
                    try:
                        if inspect.iscoroutinefunction(method):
                            latest_data = await method(limit=200)
                        else:
                            latest_data = method(limit=200)

                        # Safety check - ensure data is not a coroutine or Task
                        if inspect.iscoroutine(latest_data):
                            self.logger.warning(
                                "Detected coroutine data in main loop, awaiting..."
                            )
                            latest_data = await latest_data
                        elif isinstance(latest_data, asyncio.Task):
                            self.logger.warning(
                                "Detected asyncio.Task data in main loop, awaiting..."
                            )
                            latest_data = await latest_data
                        elif hasattr(latest_data, "__await__"):
                            self.logger.warning(
                                "Detected awaitable object in main loop, awaiting..."
                            )
                            latest_data = await latest_data
                    except Exception as e:
                        self.logger.warning(
                            f"Error getting market data in main loop: {e}"
                        )
                        latest_data = []
                else:
                    latest_data = []

                # Ensure latest_data is a list/sequence before using it
                if isinstance(latest_data, asyncio.Task):
                    self.logger.error(
                        f"latest_data is still an asyncio.Task: {latest_data}. This should have been awaited."
                    )
                    latest_data = []
                elif inspect.iscoroutine(latest_data):
                    self.logger.error(
                        f"latest_data is still a coroutine: {latest_data}. This should have been awaited."
                    )
                    latest_data = []
                elif not isinstance(latest_data, list | tuple):
                    self.logger.warning(
                        f"Unexpected latest_data type: {type(latest_data)}, converting to list"
                    )
                    latest_data = list(latest_data) if latest_data else []

                if not latest_data:
                    self.logger.warning("No market data available, waiting...")
                    await asyncio.sleep(5)
                    continue

                current_price = latest_data[-1].close

                # Publish market data to dashboard
                if self.websocket_publisher:
                    await self.websocket_publisher.publish_market_data(
                        symbol=self.actual_trading_symbol,
                        price=float(current_price),
                        timestamp=latest_data[-1].timestamp,
                    )

                # Calculate technical indicators - handle different provider types
                if self.market_data is None:
                    self.logger.error(
                        "Market data provider not available for indicator calculation"
                    )
                    continue

                # At this point, market_data is guaranteed to be non-None
                market_data_provider = self.market_data

                # All providers use standard to_dataframe signature
                df = market_data_provider.to_dataframe(limit=200)

                # Initialize dominance candles
                dominance_candles = None

                # Generate dominance candlesticks for technical analysis if not already done
                if dominance_candles is None and self.dominance_provider:
                    dominance_history = self.dominance_provider.get_dominance_history(
                        hours=2
                    )
                    if (
                        len(dominance_history) >= 6
                    ):  # Need at least 6 snapshots for 3-minute candles
                        try:
                            candle_builder = DominanceCandleBuilder(dominance_history)
                            dominance_candles = candle_builder.build_candles(
                                interval="3T"
                            )
                            # Keep only the last 20 candles for analysis
                            dominance_candles = (
                                dominance_candles[-20:]
                                if len(dominance_candles) > 20
                                else dominance_candles
                            )
                            self.logger.debug(
                                f"Generated {len(dominance_candles)} dominance candles for VuManChu analysis"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to build dominance candles for VuManChu: {e}"
                            )
                            dominance_candles = (
                                None  # Ensure variable is properly reset on error
                            )

                # Validate data sufficiency before indicator calculation
                if len(df) < 100:
                    self.logger.warning(
                        f"Insufficient data for reliable indicators: {len(df)} candles. "
                        f"Using fallback values until more data is available."
                    )
                    indicator_state = self._get_fallback_indicator_state()
                else:
                    # Calculate indicators with dominance candle support - add error boundary
                    try:
                        df_with_indicators = self.indicator_calc.calculate_all(
                            df, dominance_candles=dominance_candles
                        )
                        indicator_state = self.indicator_calc.get_latest_state(
                            df_with_indicators
                        )

                        # Publish indicator data to dashboard
                        if self.websocket_publisher:
                            await self.websocket_publisher.publish_indicator_data(
                                symbol=self.actual_trading_symbol,
                                indicators={
                                    "cipher_a": indicator_state.get("cipher_a", {}),
                                    "cipher_b": indicator_state.get("cipher_b", {}),
                                    "wave_trend_1": indicator_state.get("wave_trend_1"),
                                    "wave_trend_2": indicator_state.get("wave_trend_2"),
                                    "rsi": indicator_state.get("rsi"),
                                    "stoch_rsi": indicator_state.get("stoch_rsi"),
                                    "schaff_trend": indicator_state.get("schaff_trend"),
                                    "rsimfi": indicator_state.get("rsimfi"),
                                },
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Indicator calculation failed: {e}, using fallback values"
                        )
                        # Use fallback indicator state
                        indicator_state = self._get_fallback_indicator_state()

                        # Publish fallback indicator data to dashboard
                        if self.websocket_publisher:
                            await self.websocket_publisher.publish_indicator_data(
                                symbol=self.actual_trading_symbol,
                                indicators=indicator_state,
                            )

                # Prepare indicator data
                indicator_dict = {
                    "timestamp": datetime.now(UTC),
                    "cipher_a_dot": indicator_state.get("cipher_a", {}).get(
                        "trend_dot"
                    ),
                    "cipher_b_wave": indicator_state.get("cipher_b", {}).get("wave"),
                    "cipher_b_money_flow": indicator_state.get("cipher_b", {}).get(
                        "money_flow"
                    ),
                    "rsi": indicator_state.get("cipher_a", {}).get("rsi"),
                    "ema_fast": indicator_state.get("cipher_a", {}).get("ema_fast"),
                    "ema_slow": indicator_state.get("cipher_a", {}).get("ema_slow"),
                    "vwap": indicator_state.get("cipher_b", {}).get("vwap"),
                }

                # Add VuManChu dominance analysis if available
                dominance_analysis = indicator_state.get("dominance_analysis", {})
                if dominance_analysis:
                    # Add key dominance indicators to the main indicator dict
                    indicator_dict.update(
                        {
                            "dominance_cipher_a_signal": dominance_analysis.get(
                                "cipher_a_signal"
                            ),
                            "dominance_cipher_b_signal": dominance_analysis.get(
                                "cipher_b_signal"
                            ),
                            "dominance_sentiment": dominance_analysis.get("sentiment"),
                            "dominance_price_divergence": dominance_analysis.get(
                                "price_divergence"
                            ),
                            "dominance_trend": dominance_analysis.get("trend"),
                            "dominance_wt1": dominance_analysis.get("wt1"),
                            "dominance_wt2": dominance_analysis.get("wt2"),
                        }
                    )
                    self.logger.debug(
                        f"Added dominance analysis indicators: {list(dominance_analysis.keys())}"
                    )

                # Add dominance data to indicators if available - add error boundary
                dominance_obj = None
                if self.dominance_provider:
                    try:
                        dominance_data = self.dominance_provider.get_latest_dominance()
                        if dominance_data:
                            # Add dominance metrics to indicator dict - validate values
                            dominance_metrics = {
                                "usdt_dominance": (
                                    dominance_data.usdt_dominance
                                    if dominance_data.usdt_dominance is not None
                                    else 0.0
                                ),
                                "usdc_dominance": (
                                    dominance_data.usdc_dominance
                                    if dominance_data.usdc_dominance is not None
                                    else 0.0
                                ),
                                "stablecoin_dominance": (
                                    dominance_data.stablecoin_dominance
                                    if dominance_data.stablecoin_dominance is not None
                                    else 0.0
                                ),
                                "dominance_trend": (
                                    dominance_data.dominance_24h_change
                                    if dominance_data.dominance_24h_change is not None
                                    else 0.0
                                ),
                                "dominance_rsi": (
                                    dominance_data.dominance_rsi
                                    if dominance_data.dominance_rsi is not None
                                    else 50.0
                                ),
                                "stablecoin_velocity": (
                                    dominance_data.stablecoin_velocity
                                    if dominance_data.stablecoin_velocity is not None
                                    else 1.0
                                ),
                            }
                            indicator_dict.update(dominance_metrics)

                            # Get market sentiment based on dominance
                            try:
                                sentiment_analysis = (
                                    self.dominance_provider.get_market_sentiment()
                                )
                                indicator_dict["market_sentiment"] = (
                                    sentiment_analysis.get("sentiment", "NEUTRAL")
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to get market sentiment: {e}"
                                )
                                indicator_dict["market_sentiment"] = "NEUTRAL"

                            # Store dominance object for MarketState
                            dominance_obj = dominance_data
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to process dominance data: {e}, using default values"
                        )
                        # Add default dominance values
                        indicator_dict.update(
                            {
                                "usdt_dominance": 0.0,
                                "usdc_dominance": 0.0,
                                "stablecoin_dominance": 0.0,
                                "dominance_trend": 0.0,
                                "dominance_rsi": 50.0,
                                "stablecoin_velocity": 1.0,
                                "market_sentiment": "NEUTRAL",
                            }
                        )

                # Calculate how many candles represent 24 hours based on interval
                interval_seconds = self._get_interval_seconds(self.interval)
                candles_per_24h = min(
                    (24 * 60 * 60) // interval_seconds, len(latest_data)
                )

                # Get the last 24 hours of data (or all available if less)
                historical_data = latest_data[-candles_per_24h:]

                # Create market state for LLM analysis with enhanced historical context
                market_state = MarketState(
                    symbol=self.symbol,
                    interval=self.interval,
                    timestamp=datetime.now(UTC),
                    current_price=current_price,
                    ohlcv_data=historical_data,  # Full 24h history
                    indicators=IndicatorData(**indicator_dict),
                    current_position=self.current_position,
                    dominance_data=dominance_obj,
                    dominance_candles=dominance_candles,
                )

                # Check if trading is enabled and enough time has passed since last trade
                if not self._can_trade_now():
                    await asyncio.sleep(1)
                    continue

                # Mark that we're analyzing this candle (momentum trading: analyze on candle close)
                latest_candle = latest_data[-1]
                # Ensure the timestamp has timezone info
                candle_timestamp = latest_candle.timestamp
                if candle_timestamp.tzinfo is None:
                    candle_timestamp = candle_timestamp.replace(tzinfo=UTC)
                self.last_candle_analysis_time = candle_timestamp

                self.logger.info(
                    f"⚡ Scalping analysis: {self.interval} candle at {latest_candle.timestamp.strftime('%H:%M:%S.%f')[:-3]} - Price: ${current_price}"
                )

                # Get LLM trading decision
                self.logger.debug(
                    f"🤔 Requesting trading decision from "
                    f"{'Memory-Enhanced' if self._memory_available else 'Standard'} LLM Agent"
                )
                trade_action = await self.llm_agent.analyze_market(market_state)

                # Publish LLM decision to dashboard
                if self.websocket_publisher:
                    await self.websocket_publisher.publish_ai_decision(
                        action=trade_action.action,
                        reasoning=trade_action.rationale,
                        confidence=trade_action.size_pct
                        / 100.0,  # Convert percentage to decimal
                    )

                    # Also publish detailed trading decision
                    await self.websocket_publisher.publish_trading_decision(
                        request_id=f"trade_{int(datetime.now(UTC).timestamp())}",
                        action=trade_action.action,
                        confidence=trade_action.size_pct / 100.0,
                        reasoning=trade_action.rationale,
                        price=float(current_price),
                        quantity=trade_action.size_pct / 100.0,
                        leverage=self.settings.trading.leverage,
                        indicators={
                            "cipher_a": indicator_state.get("cipher_a", {}),
                            "cipher_b": indicator_state.get("cipher_b", {}),
                            "wave_trend_1": indicator_state.get("wave_trend_1"),
                            "wave_trend_2": indicator_state.get("wave_trend_2"),
                            "rsi": indicator_state.get("rsi"),
                            "stoch_rsi": indicator_state.get("stoch_rsi"),
                        },
                        risk_analysis={
                            "current_price": float(current_price),
                            "position_size": trade_action.size_pct / 100.0,
                            "leverage": self.settings.trading.leverage,
                        },
                    )

                # Record trading decision in memory if MCP is enabled
                experience_id = None
                if self.experience_manager and trade_action.action != "HOLD":
                    try:
                        experience_id = (
                            await self.experience_manager.record_trading_decision(
                                market_state, trade_action
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to record trading decision: {e}")

                # LLM has final say - if it says LONG/SHORT, execute immediately
                if trade_action.action in ["LONG", "SHORT"]:
                    # Validate the trade action for basic structure only
                    validated_action = self.validator.validate(trade_action)

                    self.logger.info(
                        f"Loop {loop_count}: Price=${current_price} | "
                        f"LLM={trade_action.action} | "
                        f"Action={validated_action.action} ({validated_action.size_pct}%) | "
                        f"Risk=LLM_OVERRIDE - AI has final say"
                    )

                    # Execute LLM decision immediately without risk management filtering
                    await self._execute_trade(
                        validated_action, current_price, market_state, experience_id
                    )
                    final_action = validated_action

                    # Update last trade time for interval control
                    self.last_trade_time = datetime.now(UTC)

                else:
                    # For HOLD or CLOSE actions, apply normal risk management
                    validated_action = self.validator.validate(trade_action)

                    # Apply risk management for non-directional trades
                    risk_approved, final_action, risk_reason = (
                        self.risk_manager.evaluate_risk(
                            validated_action, self.current_position, current_price
                        )
                    )

                    self.logger.info(
                        f"Loop {loop_count}: Price=${current_price} | "
                        f"LLM={trade_action.action} | "
                        f"Action={final_action.action} ({final_action.size_pct}%) | "
                        f"Risk={risk_reason}"
                    )

                    # Execute trade if approved
                    if risk_approved and final_action.action != "HOLD":
                        await self._execute_trade(
                            final_action, current_price, market_state, experience_id
                        )

                        # Update last trade time for interval control
                        self.last_trade_time = datetime.now(UTC)

                # Update position tracking and risk metrics
                await self._update_position_tracking(current_price)

                # Display periodic status updates
                if loop_count % 10 == 0:  # Every 10 loops
                    self._display_status_update(loop_count, current_price, final_action)

                # Log heartbeat to confirm loop is running
                if loop_count % 5 == 0:
                    self.logger.debug(
                        f"Trading loop heartbeat - iteration {loop_count}"
                    )

                # Log pattern statistics every 100 loops if memory is enabled
                if (
                    loop_count % 100 == 0
                    and self.memory_server
                    and self._memory_available
                ):
                    try:
                        pattern_stats = (
                            await self.memory_server.get_pattern_statistics()
                        )
                        if pattern_stats:
                            self.logger.info(
                                "📊 === MCP Pattern Performance Update ==="
                            )
                            sorted_patterns = sorted(
                                pattern_stats.items(),
                                key=lambda x: x[1]["success_rate"] * x[1]["count"],
                                reverse=True,
                            )[:5]
                            for pattern, stats in sorted_patterns:
                                if (
                                    stats["count"] >= 2
                                ):  # Show patterns with at least 2 samples
                                    self.logger.info(
                                        f"  📈 {pattern}: {stats['success_rate']:.1%} win rate | "
                                        f"{stats['count']} trades | Avg PnL: ${stats['avg_pnl']:.2f}"
                                    )
                    except Exception as e:
                        self.logger.debug(f"Could not retrieve pattern statistics: {e}")

                # Periodic status logging (every 2 minutes)
                if (datetime.now(UTC) - last_status_log).total_seconds() > 120:
                    if self.market_data is not None:
                        data_status = self.market_data.get_data_status()
                        self.logger.info(
                            f"🔄 Trading Status: Loop #{loop_count} | "
                            f"WebSocket: {'✓' if data_status.get('websocket_connected') else '✗'} | "
                            f"Latest Price: ${data_status.get('latest_price', 'N/A')} | "
                            f"OmniSearch: {'✓ Active' if hasattr(self, 'omnisearch_client') else '✗ Disabled'}"
                        )
                    else:
                        self.logger.info(
                            f"🔄 Trading Status: Loop #{loop_count} | "
                            f"Market Data: ✗ Not Initialized | "
                            f"OmniSearch: {'✓ Active' if hasattr(self, 'omnisearch_client') else '✗ Disabled'}"
                        )
                    last_status_log = datetime.now(UTC)

                # Calculate sleep time to maintain update frequency
                loop_duration = (datetime.now(UTC) - loop_start).total_seconds()
                sleep_time = max(
                    0, self.settings.system.update_frequency_seconds - loop_duration
                )

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                console.print(f"[red]Loop error: {e}[/red]")

                # Implement exponential backoff for error recovery
                error_sleep = min(30, 1 * (loop_count % 5 + 1))
                self.logger.info(f"Waiting {error_sleep}s before retry...")
                await asyncio.sleep(error_sleep)
                continue

        self.logger.info("Trading loop stopped")

    async def _execute_trade_action(self, trade_action: TradeAction) -> bool:
        """
        Execute a trade action with current market data (simplified wrapper).

        This method is used for manual trades and emergency actions where
        we only have the trade action and need to fetch current market data.

        Args:
            trade_action: Trade action to execute

        Returns:
            bool: True if trade was executed successfully, False otherwise
        """
        try:
            # Get current price from market data
            if not self.market_data:
                self.logger.error("Market data not available for trade execution")
                return False
            # Both MarketDataProvider and BluefinMarketDataProvider have get_current_price method
            current_price = await self.market_data.get_current_price()  # type: ignore[union-attr]
            if not current_price:
                self.logger.error("Current price not available for trade execution")
                return False

            # Execute trade with current price and no experience tracking
            await self._execute_trade(trade_action, current_price)
            return True

        except Exception as e:
            self.logger.error(f"Error executing trade action: {e}")
            return False

    async def _execute_trade(
        self,
        trade_action: TradeAction,
        current_price: Decimal,
        market_state: MarketState | None = None,
        experience_id: str | None = None,
    ):
        """
        Execute a validated trade action.

        Args:
            trade_action: Validated trade action to execute
            current_price: Current market price
            market_state: Current market state (for experience tracking)
            experience_id: Experience ID if trade decision was recorded
        """
        try:
            self.logger.info(
                f"📦 Executing trade: {trade_action.action} {trade_action.size_pct}% | "
                f"Experience ID: {experience_id[:8] if experience_id else 'None'}..."
            )
            self.logger.debug(
                f"Trade execution started at {datetime.now(UTC).isoformat()}"
            )

            # Check if we already have an open position and the action is LONG or SHORT
            if self.current_position.side != "FLAT" and trade_action.action in [
                "LONG",
                "SHORT",
            ]:
                self.logger.warning(
                    f"Cannot open new {trade_action.action} position - already have "
                    f"{self.current_position.side} position with size {self.current_position.size}"
                )
                console.print(
                    f"[yellow]⚠ Trade rejected: Already have open {self.current_position.side} position[/yellow]"
                )
                return

            # Execute trade based on mode (paper trading vs live)
            if self.dry_run and self.paper_account:
                # Paper trading execution
                order = self.paper_account.execute_trade_action(
                    trade_action, self.symbol, current_price
                )
            else:
                # Live trading execution
                order = await self.exchange_client.execute_trade_action(
                    trade_action, self.symbol, current_price
                )

            if order:
                self.trade_count += 1

                # Link order to experience for tracking
                if self.experience_manager and experience_id:
                    try:
                        self.experience_manager.link_order_to_experience(
                            order.id, experience_id
                        )
                        self.logger.debug(
                            f"MCP Integration: Linked order {order.id} to experience {experience_id[:8]}..."
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to link order to experience: {e}")

                if order.status in ["FILLED", "PENDING"]:
                    self.successful_trades += 1

                    # Publish trade execution to dashboard
                    if self.websocket_publisher:
                        await self.websocket_publisher.publish_trade_execution(
                            order_id=order.id,
                            symbol=self.actual_trading_symbol,
                            side=trade_action.action,
                            quantity=float(order.quantity),
                            price=float(order.price),
                            status=order.status,
                            trade_action={
                                "action": trade_action.action,
                                "size_pct": trade_action.size_pct,
                                "rationale": trade_action.rationale,
                                "experience_id": experience_id,
                            },
                        )

                    # Update position manager
                    if hasattr(order, "filled_quantity") and order.filled_quantity > 0:
                        # Store previous position before update
                        previous_position = self.current_position

                        updated_position = (
                            self.position_manager.update_position_from_order(
                                order, order.price or current_price
                            )
                        )
                        self.current_position = updated_position

                        # Check if this order closed a position
                        if (
                            previous_position.side != "FLAT"
                            and updated_position.side == "FLAT"
                            and self.experience_manager
                            and market_state
                        ):
                            # Trade was closed, complete the experience
                            try:
                                # Add timeout to prevent blocking
                                await asyncio.wait_for(
                                    self.experience_manager.complete_trade(
                                        order,
                                        order.price or current_price,
                                        market_state,
                                    ),
                                    timeout=5.0,  # 5 second timeout
                                )
                                self.logger.info(
                                    "✅ MCP Integration: Completed trade tracking for closed position"
                                )
                            except TimeoutError:
                                self.logger.warning(
                                    "Trade tracking completion timed out after 5 seconds"
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to complete trade tracking: {e}"
                                )

                        # Start tracking new trades (LONG/SHORT entry)
                        elif (
                            previous_position.side == "FLAT"
                            and updated_position.side != "FLAT"
                            and self.experience_manager
                            and market_state
                        ):
                            try:
                                trade_id = self.experience_manager.start_tracking_trade(
                                    order, trade_action, market_state
                                )
                                if trade_id:
                                    self.logger.info(
                                        f"📡 MCP Integration: Started tracking new trade - ID: {trade_id}"
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to start trade tracking: {e}"
                                )

                    console.print(
                        f"[green]✓ Trade executed:[/green] {trade_action.action} "
                        f"{trade_action.size_pct}% @ ${current_price}"
                    )

                    # Log paper trading account status if in dry run
                    if self.dry_run and self.paper_account:
                        account_status = self.paper_account.get_account_status()
                        self.logger.info(
                            f"Paper account: ${account_status['equity']:,.2f} equity, "
                            f"P&L: ${account_status['total_pnl']:,.2f} "
                            f"({account_status['roi_percent']:.2f}%)"
                        )

                    self.logger.debug(
                        f"Trade execution completed at {datetime.now(UTC).isoformat()}"
                    )
                else:
                    console.print(f"[yellow]⚠ Trade failed:[/yellow] {order.status}")
            else:
                console.print("[red]✗ Trade execution failed[/red]")

        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            console.print(f"[red]Trade execution error: {e}[/red]")

    def _get_interval_seconds(self, interval: str) -> int:
        """
        Convert interval string to seconds.

        Args:
            interval: Interval string (e.g., '1s', '15s', '1m', '5m', '1h', '1d')

        Returns:
            Number of seconds in the interval
        """
        interval_map = {
            "1s": 1,
            "5s": 5,
            "15s": 15,
            "30s": 30,
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
        }
        return interval_map.get(
            interval.lower(), 15
        )  # Default to 15 seconds for scalping

    def _get_interval_minutes(self, interval: str) -> float:
        """
        Convert interval string to minutes.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h', '1d')

        Returns:
            Number of minutes in the interval
        """
        return self._get_interval_seconds(interval) / 60

    async def _update_position_tracking(self, current_price: Decimal) -> None:
        """
        Update position tracking and P&L calculations.

        Args:
            current_price: Current market price
        """
        if self.current_position.side != "FLAT" and self.current_position.entry_price:
            # Calculate unrealized P&L
            if self.current_position.side == "LONG":
                pnl = (
                    current_price - self.current_position.entry_price
                ) * self.current_position.size
            else:  # SHORT
                pnl = (
                    self.current_position.entry_price - current_price
                ) * self.current_position.size

            self.current_position.unrealized_pnl = pnl

            # Publish position update to dashboard
            if self.websocket_publisher:
                await self.websocket_publisher.publish_position_update(
                    symbol=self.actual_trading_symbol,
                    side=self.current_position.side.lower(),
                    size=float(self.current_position.size),
                    entry_price=float(self.current_position.entry_price),
                    current_price=float(current_price),
                    unrealized_pnl=float(pnl),
                )

            # Update risk manager with P&L
            self.risk_manager.update_daily_pnl(Decimal("0"), pnl)

            # Update experience manager with trade progress
            if self.experience_manager:
                try:
                    await self.experience_manager.update_trade_progress(
                        self.current_position, current_price
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to update trade progress: {e}")
        else:
            # Publish flat position to dashboard
            if self.websocket_publisher:
                await self.websocket_publisher.publish_position_update(
                    symbol=self.actual_trading_symbol,
                    side="flat",
                    size=0.0,
                    entry_price=0.0,
                    current_price=float(current_price),
                    unrealized_pnl=0.0,
                )

            # This code was moved to the main trading loop

    def _display_status_update(
        self, loop_count: int, current_price: Decimal, last_action: TradeAction
    ):
        """Display periodic status updates."""
        uptime = datetime.now(UTC) - self.start_time
        success_rate = (self.successful_trades / max(self.trade_count, 1)) * 100

        status_table = Table(title=f"Trading Status - Loop {loop_count}")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")

        status_table.add_row("Current Price", f"${current_price:,.2f}")
        status_table.add_row(
            "Position", f"{self.current_position.side} {self.current_position.size}"
        )
        status_table.add_row(
            "Unrealized P&L", f"${self.current_position.unrealized_pnl:,.2f}"
        )
        status_table.add_row("Total Trades", str(self.trade_count))
        status_table.add_row("Success Rate", f"{success_rate:.1f}%")
        status_table.add_row(
            "Last Action", f"{last_action.action} ({last_action.rationale})"
        )
        status_table.add_row("Uptime", str(uptime).split(".")[0])

        # Add dominance data if available
        if self.dominance_provider:
            dominance_data = self.dominance_provider.get_latest_dominance()
            if dominance_data:
                status_table.add_row(
                    "Stablecoin Dominance",
                    f"{dominance_data.stablecoin_dominance:.2f}% ({dominance_data.dominance_24h_change:+.2f}%)",
                )
                sentiment = self.dominance_provider.get_market_sentiment()
                status_table.add_row(
                    "Market Sentiment", sentiment.get("sentiment", "UNKNOWN")
                )

        # Add paper trading specific metrics
        if self.dry_run and self.paper_account:
            account_status = self.paper_account.get_account_status()
            status_table.add_row(
                "Paper Balance", f"${account_status['current_balance']:,.2f}"
            )
            status_table.add_row("Paper Equity", f"${account_status['equity']:,.2f}")
            status_table.add_row("Total P&L", f"${account_status['total_pnl']:,.2f}")
            status_table.add_row("ROI", f"{account_status['roi_percent']:,.2f}%")
            status_table.add_row(
                "Max Drawdown", f"{account_status['max_drawdown']:.2f}%"
            )
            status_table.add_row(
                "Open Positions", str(account_status["open_positions"])
            )

        console.print(status_table)
        console.print()

        # Generate and display daily report every 100 loops or at end of day
        if (
            self.dry_run
            and self.paper_account
            and (loop_count % 100 == 0 or self._is_end_of_trading_day())
        ):
            try:
                daily_report = self.position_manager.generate_daily_report()
                if daily_report and "No trading data" not in daily_report:
                    console.print(
                        Panel(
                            daily_report,
                            title="📊 Daily Performance Report",
                            style="green",
                        )
                    )
            except Exception as e:
                self.logger.warning(f"Could not generate daily report: {e}")

    def _is_end_of_trading_day(self) -> bool:
        """Check if it's near the end of trading day for reporting."""
        current_time = datetime.now(UTC)
        return current_time.hour == 23 and current_time.minute >= 50

    async def _shutdown(self) -> None:
        """
        Graceful shutdown procedure.

        Cancels open orders, closes connections, and saves final state.
        """
        self.logger.info("Initiating graceful shutdown...")
        console.print("[yellow]Shutting down trading engine...[/yellow]")

        # Set running to False first to stop any loops
        self._running = False

        # Create a list of cleanup tasks to run concurrently with timeout
        cleanup_tasks = []

        try:
            # Cancel all open orders
            if (
                hasattr(self, "exchange_client")
                and self.exchange_client is not None
                and self.exchange_client.is_connected()
            ):
                console.print("  • Cancelling open orders...")
                task = asyncio.create_task(
                    self.exchange_client.cancel_all_orders(self.symbol)
                )
                cleanup_tasks.append(task)

            # Close market data connection
            if hasattr(self, "market_data") and self.market_data is not None:
                console.print("  • Disconnecting from market data...")
                task = asyncio.create_task(self.market_data.disconnect())  # type: ignore[arg-type]
                cleanup_tasks.append(task)

            # Close exchange connection
            if hasattr(self, "exchange_client") and self.exchange_client is not None:
                console.print("  • Disconnecting from exchange...")
                task = asyncio.create_task(self.exchange_client.disconnect())  # type: ignore[arg-type]
                cleanup_tasks.append(task)

            # Close OmniSearch connection
            if (
                hasattr(self, "omnisearch_client")
                and self.omnisearch_client is not None
            ):
                console.print("  • Disconnecting from OmniSearch...")
                task = asyncio.create_task(self.omnisearch_client.disconnect())  # type: ignore[arg-type]
                cleanup_tasks.append(task)

            # Close WebSocket publisher connection
            if (
                hasattr(self, "websocket_publisher")
                and self.websocket_publisher is not None
            ):
                console.print("  • Disconnecting from dashboard WebSocket...")
                cleanup_tasks.append(
                    asyncio.create_task(self.websocket_publisher.close())
                )

            # Stop command consumer
            if hasattr(self, "command_consumer") and self.command_consumer is not None:
                console.print("  • Stopping dashboard command consumer...")
                task1 = asyncio.create_task(self.command_consumer.stop_polling())
                task2 = asyncio.create_task(self.command_consumer.close())
                cleanup_tasks.append(task1)
                cleanup_tasks.append(task2)

            # Stop experience manager if enabled
            if (
                hasattr(self, "experience_manager")
                and self.experience_manager is not None
            ):
                console.print("  • Stopping experience tracking...")
                task = asyncio.create_task(self.experience_manager.stop())  # type: ignore[arg-type]
                cleanup_tasks.append(task)

            # Close dominance data connection - CRITICAL for async session cleanup
            if (
                hasattr(self, "dominance_provider")
                and self.dominance_provider is not None
            ):
                console.print("  • Disconnecting from dominance data...")
                task = asyncio.create_task(self.dominance_provider.disconnect())  # type: ignore[arg-type]
                cleanup_tasks.append(task)

            # Wait for all cleanup tasks with a timeout
            if cleanup_tasks:
                done, pending = await asyncio.wait(
                    cleanup_tasks, timeout=5.0, return_when=asyncio.ALL_COMPLETED
                )

                # Cancel any tasks that didn't complete in time
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Display final summary
            self._display_final_summary()

            console.print("[green]✓ Shutdown complete[/green]")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            console.print(f"[red]Shutdown error: {e}[/red]")
        finally:
            # Final cleanup - ensure all async sessions are closed
            # This is a last resort cleanup
            if hasattr(self, "dominance_provider") and self.dominance_provider:
                if (
                    hasattr(self.dominance_provider, "_session")
                    and self.dominance_provider._session
                ):
                    if not self.dominance_provider._session.closed:
                        try:
                            # Force close without await since we might be in cleanup
                            if (
                                hasattr(self.dominance_provider._session, "_connector")
                                and self.dominance_provider._session._connector
                                is not None
                            ):
                                self.dominance_provider._session._connector.close()
                        except Exception:
                            pass

    async def _reconcile_positions(self) -> None:
        """
        Reconcile local position state with actual exchange positions.

        This method checks for existing positions on the exchange and updates
        the local position state to match, preventing conflicts when the bot
        restarts with open positions.
        """
        try:
            # Get current positions from exchange
            if self.exchange_client.enable_futures:
                # For futures trading, check CFM positions
                positions = await self.exchange_client.get_futures_positions(
                    self.actual_trading_symbol
                )
            else:
                # For spot trading, check regular positions
                positions = await self.exchange_client.get_positions(
                    self.actual_trading_symbol
                )

            if not positions:
                self.logger.info("No existing positions found on exchange")
                console.print("    [green]✓ No existing positions detected[/green]")
                return

            # Process the first position found for our symbol
            for position in positions:
                # Handle both dict and object formats
                if hasattr(position, "get"):
                    # Dictionary format
                    symbol = position.get("symbol") or position.get("product_id")
                    size = Decimal(str(position.get("size", 0)))
                    side = position.get("side", "FLAT")
                    entry_price = Decimal(str(position.get("entry_price", 0)))
                    unrealized_pnl = Decimal(str(position.get("unrealized_pnl", 0)))
                else:
                    # Object format
                    symbol = getattr(position, "symbol", None) or getattr(
                        position, "product_id", None
                    )
                    size = Decimal(str(getattr(position, "size", 0)))
                    side = getattr(position, "side", "FLAT")
                    entry_price = Decimal(str(getattr(position, "entry_price", 0)))
                    unrealized_pnl = Decimal(
                        str(getattr(position, "unrealized_pnl", 0))
                    )

                if symbol == self.actual_trading_symbol:
                    # Convert position data to our format
                    if size > 0:
                        if side.upper() in ["LONG", "BUY"]:
                            position_side: Literal["LONG", "SHORT", "FLAT"] = "LONG"
                        elif side.upper() in ["SHORT", "SELL"]:
                            position_side = "SHORT"
                        else:
                            position_side = "LONG" if size > 0 else "SHORT"

                        # Update current position
                        self.current_position = Position(
                            symbol=self.actual_trading_symbol,
                            side=position_side,
                            size=abs(size),
                            timestamp=datetime.now(UTC),
                            entry_price=entry_price,
                            unrealized_pnl=unrealized_pnl,
                        )

                        # Update position manager state
                        self.position_manager.update_position_from_exchange(
                            symbol=self.actual_trading_symbol,
                            side=position_side,
                            size=abs(size),
                            entry_price=self.current_position.entry_price
                            or Decimal("0"),
                        )

                        self.logger.info(
                            f"Reconciled position: {position_side} {size} {self.actual_trading_symbol} "
                            f"at ${self.current_position.entry_price}"
                        )
                        console.print(
                            f"    [yellow]⚠ Found existing {position_side} position: "
                            f"{size} {self.actual_trading_symbol} at ${self.current_position.entry_price}[/yellow]"
                        )
                        return

            # If we get here, no matching position was found
            self.logger.info("No matching positions found for trading symbol")
            console.print("    [green]✓ No existing positions detected[/green]")

        except Exception as e:
            self.logger.error(f"Failed to reconcile positions: {e}")
            console.print(f"    [red]✗ Position reconciliation failed: {e}[/red]")
            # Continue with FLAT position assumption on error

    def _display_final_summary(self) -> None:
        """Display final trading session summary."""
        total_runtime = datetime.now(UTC) - self.start_time
        success_rate = (self.successful_trades / max(self.trade_count, 1)) * 100

        summary_table = Table(title="Trading Session Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Total Runtime", str(total_runtime).split(".")[0])
        summary_table.add_row("Total Trades", str(self.trade_count))
        summary_table.add_row("Successful Trades", str(self.successful_trades))
        summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
        summary_table.add_row(
            "Final Position",
            f"{self.current_position.side} {self.current_position.size}",
        )
        summary_table.add_row(
            "Final P&L", f"${self.current_position.unrealized_pnl:,.2f}"
        )

        # Add comprehensive paper trading summary
        if self.dry_run and self.paper_account:
            console.print()
            console.print("[bold cyan]📈 Paper Trading Performance Summary[/bold cyan]")
            console.print("=" * 60)

            # Update and get performance metrics
            self.paper_account.update_daily_performance()
            performance = self.position_manager.get_paper_trading_performance(days=30)

            if "error" not in performance:
                perf_table = Table(title="Paper Trading Results")
                perf_table.add_column("Metric", style="cyan")
                perf_table.add_column("Value", style="white")

                perf_table.add_row(
                    "Starting Balance",
                    f"${performance.get('starting_balance', 0):,.2f}",
                )
                perf_table.add_row(
                    "Final Equity", f"${performance.get('current_equity', 0):,.2f}"
                )
                perf_table.add_row(
                    "Total Return", f"${performance.get('net_pnl', 0):,.2f}"
                )
                perf_table.add_row("ROI", f"{performance.get('roi_percent', 0):.2f}%")
                perf_table.add_row(
                    "Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%"
                )
                perf_table.add_row(
                    "Total Trades", str(performance.get("total_trades", 0))
                )
                perf_table.add_row(
                    "Win Rate", f"{performance.get('overall_win_rate', 0):.1f}%"
                )
                perf_table.add_row(
                    "Avg Daily P&L", f"${performance.get('avg_daily_pnl', 0):,.2f}"
                )
                perf_table.add_row(
                    "Sharpe Ratio", f"{performance.get('sharp_ratio', 0):.2f}"
                )
                perf_table.add_row(
                    "Fees Paid", f"${performance.get('total_fees_paid', 0):,.2f}"
                )

                console.print(perf_table)

                # Export trade history
                try:
                    trade_history = self.position_manager.export_trade_history(
                        days=30, format="json"
                    )
                    history_file = Path("data/paper_trading/session_trades.json")
                    history_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(history_file, "w") as f:
                        f.write(trade_history)
                    console.print(
                        f"[green]✓ Trade history exported to {history_file}[/green]"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not export trade history: {e}")

            # Display final daily report
            try:
                final_report = self.position_manager.generate_daily_report()
                if final_report and "No trading data" not in final_report:
                    console.print()
                    console.print(
                        Panel(final_report, title="📊 Final Daily Report", style="blue")
                    )
            except Exception as e:
                self.logger.warning(f"Could not generate final daily report: {e}")

        console.print(summary_table)

    def _apply_cipher_b_filter(
        self, trade_action: TradeAction, market_state: MarketState
    ) -> TradeAction:
        """
        Apply Cipher B signal filtering to the LLM trading decision.

        This method implements filtering logic to only allow trades when Cipher B signals
        are aligned with the trading direction. Acts as a confirmation layer.

        Args:
            trade_action: Original trade action from LLM
            market_state: Current market state with indicators

        Returns:
            Filtered trade action (may be converted to HOLD if signals don't align)
        """
        try:
            # Check if Cipher B filtering is enabled
            if not self.settings.data.enable_cipher_b_filter:
                self.logger.debug(
                    "Cipher B filter: Disabled in configuration, allowing original action"
                )
                return trade_action

            # Get Cipher B indicator values
            cipher_b_wave = market_state.indicators.cipher_b_wave
            cipher_b_money_flow = market_state.indicators.cipher_b_money_flow

            # Skip filtering for HOLD and CLOSE actions
            if trade_action.action in ["HOLD", "CLOSE"]:
                self.logger.debug(
                    "Cipher B filter: Allowing HOLD/CLOSE action without filtering"
                )
                return trade_action

            # Skip filtering if Cipher B indicators are not available
            if cipher_b_wave is None or cipher_b_money_flow is None:
                self.logger.warning(
                    "Cipher B filter: Indicators not available, allowing original action"
                )
                return trade_action

            # Get Cipher B signal thresholds from configuration
            wave_bullish_threshold = self.settings.data.cipher_b_wave_bullish_threshold
            wave_bearish_threshold = self.settings.data.cipher_b_wave_bearish_threshold
            money_flow_bullish_threshold = (
                self.settings.data.cipher_b_money_flow_bullish_threshold
            )
            money_flow_bearish_threshold = (
                self.settings.data.cipher_b_money_flow_bearish_threshold
            )

            # Determine Cipher B signals
            wave_bullish = cipher_b_wave > wave_bullish_threshold
            wave_bearish = cipher_b_wave < wave_bearish_threshold
            money_flow_bullish = cipher_b_money_flow > money_flow_bullish_threshold
            money_flow_bearish = cipher_b_money_flow < money_flow_bearish_threshold

            # Check signal alignment for LONG trades
            if trade_action.action == "LONG":
                # Require both wave and money flow to be bullish
                if wave_bullish and money_flow_bullish:
                    self.logger.info(
                        f"Cipher B filter: LONG signal CONFIRMED - "
                        f"Wave: {cipher_b_wave:.2f} (bullish), "
                        f"Money Flow: {cipher_b_money_flow:.2f} (bullish)"
                    )
                    return trade_action
                else:
                    self.logger.info(
                        f"Cipher B filter: LONG signal FILTERED OUT - "
                        f"Wave: {cipher_b_wave:.2f} ({'bullish' if wave_bullish else 'bearish'}), "
                        f"Money Flow: {cipher_b_money_flow:.2f} ({'bullish' if money_flow_bullish else 'bearish'})"
                    )
                    # Convert to HOLD with explanation
                    return TradeAction(
                        action="HOLD",
                        size_pct=0,
                        take_profit_pct=1.0,
                        stop_loss_pct=1.0,
                        leverage=trade_action.leverage,
                        reduce_only=False,
                        rationale=f"Cipher B filter: LONG rejected - Wave:{cipher_b_wave:.2f}, MF:{cipher_b_money_flow:.2f}",
                    )

            # Check signal alignment for SHORT trades
            elif trade_action.action == "SHORT":
                # Require both wave and money flow to be bearish
                if wave_bearish and money_flow_bearish:
                    self.logger.info(
                        f"Cipher B filter: SHORT signal CONFIRMED - "
                        f"Wave: {cipher_b_wave:.2f} (bearish), "
                        f"Money Flow: {cipher_b_money_flow:.2f} (bearish)"
                    )
                    return trade_action
                else:
                    self.logger.info(
                        f"Cipher B filter: SHORT signal FILTERED OUT - "
                        f"Wave: {cipher_b_wave:.2f} ({'bearish' if wave_bearish else 'bullish'}), "
                        f"Money Flow: {cipher_b_money_flow:.2f} ({'bearish' if money_flow_bearish else 'bullish'})"
                    )
                    # Convert to HOLD with explanation
                    return TradeAction(
                        action="HOLD",
                        size_pct=0,
                        take_profit_pct=1.0,
                        stop_loss_pct=1.0,
                        leverage=trade_action.leverage,
                        reduce_only=False,
                        rationale=f"Cipher B filter: SHORT rejected - Wave:{cipher_b_wave:.2f}, MF:{cipher_b_money_flow:.2f}",
                    )

            # Default fallback - should not reach here
            self.logger.warning(
                f"Cipher B filter: Unexpected action '{trade_action.action}', allowing original"
            )
            return trade_action

        except Exception as e:
            self.logger.error(f"Error in Cipher B filtering: {e}")
            # On error, allow the original trade action to prevent system failure
            return trade_action

    def _get_fallback_indicator_state(self) -> dict[str, Any]:
        """
        Get fallback indicator state when calculation fails.

        Returns:
            Dictionary with safe default values for all indicators
        """
        return {
            "cipher_a": {
                "trend_dot": 0.0,
                "rsi": 50.0,
                "ema_fast": 0.0,
                "ema_slow": 0.0,
                "signal": 0,
                "confidence": 0.0,
            },
            "cipher_b": {
                "wave": 0.0,
                "money_flow": 50.0,
                "vwap": 0.0,
                "signal": 0,
                "confidence": 0.0,
            },
            "dominance_analysis": {
                "cipher_a_signal": 0,
                "cipher_b_signal": 0,
                "sentiment": "NEUTRAL",
                "price_divergence": "NONE",
                "trend": "SIDEWAYS",
                "wt1": 0.0,
                "wt2": 0.0,
            },
        }


@click.group()
@click.version_option(version="0.1.0", prog_name="ai-trading-bot")
def cli() -> None:
    """AI Trading Bot - LangChain-powered crypto futures trading."""
    # Set up comprehensive warning suppression for third-party libraries
    setup_warnings_suppression()
    pass


@cli.command()
@click.option(
    "--dry-run/--no-dry-run", default=True, help="Run in dry-run mode (default)"
)
@click.option("--symbol", default="BTC-USD", help="Trading symbol")
@click.option(
    "--interval", default="15s", help="Candle interval (scalping: 1s/5s/15s/30s)"
)
@click.option("--config", default=None, help="Configuration file path")
@click.option("--force", is_flag=True, help="Skip confirmation prompt for live trading")
def live(
    dry_run: bool, symbol: str, interval: str, config: str | None, force: bool
) -> None:
    """Start live trading bot."""
    if dry_run:
        console.print(
            Panel(
                f"🚀 Starting AI Trading Bot in DRY-RUN mode\n"
                f"Symbol: {symbol}\n"
                f"Interval: {interval}\n"
                f"Mode: Paper Trading (No real orders)",
                title="AI Trading Bot",
                style="cyan",
            )
        )
    else:
        console.print(
            Panel(
                f"⚠️  Starting AI Trading Bot in LIVE mode\n"
                f"Symbol: {symbol}\n"
                f"Interval: {interval}\n"
                f"Mode: Real Trading (Real money at risk!)",
                title="AI Trading Bot",
                style="red",
            )
        )

        # Confirmation for live trading (skip if --force flag is used)
        if not force and not click.confirm(
            "Are you sure you want to trade with real money?"
        ):
            console.print("Cancelled live trading.")
            sys.exit(0)

    try:
        # Start the trading engine
        engine = TradingEngine(
            symbol=symbol, interval=interval, config_file=config, dry_run=dry_run
        )
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        console.print("\n👋 Bot stopped by user")
    except Exception as e:
        import traceback

        console.print(f"❌ Error: {e}", style="red")
        console.print(f"Full traceback:\n{traceback.format_exc()}", style="yellow")
        sys.exit(1)


@cli.command()
@click.option(
    "--from", "start_date", default="2024-01-01", help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "end_date", default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--symbol", default="BTC-USD", help="Trading symbol")
@click.option("--initial-balance", default=10000.0, help="Initial balance for backtest")
def backtest(
    start_date: str, end_date: str, symbol: str, initial_balance: float
) -> None:
    """Run strategy backtest on historical data."""
    console.print(
        Panel(
            f"📊 Starting Backtest\n"
            f"Symbol: {symbol}\n"
            f"Period: {start_date} to {end_date}\n"
            f"Initial Balance: ${initial_balance:,.2f}",
            title="Backtest",
            style="green",
        )
    )

    # This will be implemented later with the actual backtesting logic
    console.print("🔄 Backtesting engine not yet implemented. Coming soon!")


@cli.command()
@click.option("--days", default=7, help="Number of days to analyze")
def performance(days: int) -> None:
    """Show paper trading performance report."""
    try:
        # Initialize position manager with paper trading
        paper_account = PaperTradingAccount()
        position_manager = PositionManager(paper_trading_account=paper_account)

        # Get performance data
        performance = position_manager.get_paper_trading_performance(days=days)

        if "error" in performance:
            console.print(f"[red]Error: {performance['error']}[/red]")
            return

        # Display performance summary
        console.print(
            f"[bold cyan]📈 Paper Trading Performance ({days} days)[/bold cyan]"
        )
        console.print("=" * 60)

        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="white")

        perf_table.add_row(
            "Starting Balance", f"${performance.get('starting_balance', 0):,.2f}"
        )
        perf_table.add_row(
            "Current Equity", f"${performance.get('current_equity', 0):,.2f}"
        )
        perf_table.add_row("Total Return", f"${performance.get('net_pnl', 0):,.2f}")
        perf_table.add_row("ROI", f"{performance.get('roi_percent', 0):.2f}%")
        perf_table.add_row("Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%")
        perf_table.add_row("Total Trades", str(performance.get("total_trades", 0)))
        perf_table.add_row("Win Rate", f"{performance.get('overall_win_rate', 0):.1f}%")
        perf_table.add_row(
            "Avg Daily P&L", f"${performance.get('avg_daily_pnl', 0):,.2f}"
        )
        perf_table.add_row("Sharpe Ratio", f"{performance.get('sharp_ratio', 0):.2f}")

        console.print(perf_table)

        # Show recent trades
        recent_trades = performance.get("recent_trades", [])
        if recent_trades:
            console.print()
            console.print("[bold cyan]📋 Recent Trades[/bold cyan]")

            trades_table = Table()
            trades_table.add_column("ID", style="dim")
            trades_table.add_column("Symbol", style="cyan")
            trades_table.add_column("Side", style="magenta")
            trades_table.add_column("Entry", style="green")
            trades_table.add_column("Exit", style="red")
            trades_table.add_column("P&L", style="white")
            trades_table.add_column("Duration", style="dim")

            for trade in recent_trades[-10:]:  # Last 10 trades
                pnl = trade.get("realized_pnl", 0)
                pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "white"

                trades_table.add_row(
                    trade.get("id", "N/A")[:8],
                    trade.get("symbol", "N/A"),
                    trade.get("side", "N/A"),
                    f"${trade.get('entry_price', 0):,.2f}",
                    (
                        f"${trade.get('exit_price', 0):,.2f}"
                        if trade.get("exit_price")
                        else "Open"
                    ),
                    f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]",
                    (
                        f"{trade.get('duration_hours', 0):.1f}h"
                        if trade.get("duration_hours")
                        else "N/A"
                    ),
                )

            console.print(trades_table)

    except Exception as e:
        console.print(f"[red]Error displaying performance: {e}[/red]")


@cli.command()
@click.option("--balance", default=10000.0, help="New starting balance")
@click.option("--confirm", is_flag=True, help="Confirm reset without prompt")
def reset_paper(balance: float, confirm: bool) -> None:
    """Reset paper trading account."""
    if not confirm:
        if not click.confirm(
            f"Are you sure you want to reset the paper trading account to ${balance:,.2f}?"
        ):
            console.print("Reset cancelled.")
            return

    try:
        paper_account = PaperTradingAccount()
        paper_account.reset_account(Decimal(str(balance)))
        console.print(
            f"[green]✅ Paper trading account reset to ${balance:,.2f}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error resetting account: {e}[/red]")


@cli.command()
@click.option("--days", default=30, help="Number of days to export")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="Export format",
)
@click.option("--output", default=None, help="Output file path")
def export_trades(days: int, export_format: str, output: str | None) -> None:
    """Export paper trading history."""
    try:
        paper_account = PaperTradingAccount()
        position_manager = PositionManager(paper_trading_account=paper_account)

        trade_history = position_manager.export_trade_history(
            days=days, format=export_format
        )

        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"paper_trades_{timestamp}.{export_format}"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(trade_history)

        console.print(f"[green]✅ Trade history exported to {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error exporting trades: {e}[/red]")


@cli.command()
@click.option("--date", default=None, help="Date for report (YYYY-MM-DD)")
def daily_report(date: str | None) -> None:
    """Show daily trading report."""
    try:
        paper_account = PaperTradingAccount()
        position_manager = PositionManager(paper_trading_account=paper_account)

        report = position_manager.generate_daily_report(date)
        console.print(Panel(report, title="📊 Daily Trading Report", style="cyan"))

    except Exception as e:
        console.print(f"[red]Error generating daily report: {e}[/red]")


@cli.command()
def init() -> None:
    """Initialize project configuration."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        console.print("✅ .env file already exists")
        return

    if env_example.exists():
        env_example.rename(env_file)
        console.print("✅ Created .env file from .env.example")
        console.print("🔧 Please edit .env with your API keys and configuration")
    else:
        console.print("❌ .env.example file not found", style="red")


if __name__ == "__main__":
    cli()
