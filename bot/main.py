"""Main CLI entry point for the AI Trading Bot."""

# ruff: noqa: E402
# CRITICAL: Initialize comprehensive warning suppression before ANY imports
# This must be the very first code that runs to catch import-time warnings
import os
import sys
import warnings

# Clear any existing warning registry and set up fresh
warnings.resetwarnings()
if not hasattr(sys.modules[__name__], "__warningregistry__"):
    sys.modules[__name__].__warningregistry__ = {}

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
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Third-party imports
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Local imports
from .config import Settings, create_settings
from .data.dominance import DominanceCandleBuilder, DominanceDataProvider
from .data.market import MarketDataProvider
from .exchange.coinbase import CoinbaseClient
from .indicators.vumanchu import VuManChuIndicators
from .learning.experience_manager import ExperienceManager
from .mcp.memory_server import MCPMemoryServer
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
        interval: str = "1m",
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
        self.symbol = symbol
        self.interval = interval
        self.dry_run = dry_run
        self._running = False
        self._shutdown_requested = False

        # Load configuration
        self.settings = self._load_configuration(config_file, dry_run)

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize paper trading if in dry run mode
        self.paper_account = None
        if self.dry_run:
            self.paper_account = PaperTradingAccount(
                starting_balance=self.settings.paper_trading.starting_balance
            )

        # Initialize components
        self.market_data = MarketDataProvider(symbol, interval)
        self.indicator_calc = VuManChuIndicators()

        # Initialize MCP memory components if enabled
        self.memory_server = None
        self.experience_manager = None

        if self.settings.mcp.enabled:
            self.logger.info("MCP memory system enabled, initializing components...")
            try:
                self.memory_server = MCPMemoryServer(
                    server_url=self.settings.mcp.server_url,
                    api_key=self.settings.mcp.api_key,
                )
                self.experience_manager = ExperienceManager(self.memory_server)

                # Use memory-enhanced agent
                self.llm_agent = MemoryEnhancedLLMAgent(
                    model_provider=self.settings.llm.provider,
                    model_name=self.settings.llm.model_name,
                    memory_server=self.memory_server,
                )
                self.logger.info("Successfully initialized memory-enhanced agent")
            except Exception as e:
                self.logger.error(f"Failed to initialize MCP components: {e}")
                self.logger.warning("Falling back to standard LLM agent")
                self.llm_agent = LLMAgent(
                    model_provider=self.settings.llm.provider,
                    model_name=self.settings.llm.model_name,
                )
        else:
            # Standard LLM agent without memory
            self.llm_agent = LLMAgent(
                model_provider=self.settings.llm.provider,
                model_name=self.settings.llm.model_name,
            )

        self.validator = TradeValidator()
        self.position_manager = PositionManager(
            paper_trading_account=self.paper_account,
            use_fifo=self.settings.trading.use_fifo_accounting,
        )
        self.risk_manager = RiskManager(position_manager=self.position_manager)
        self.exchange_client = CoinbaseClient()

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

        self.logger.info(f"Initialized TradingEngine for {symbol} at {interval}")

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

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.settings.system.log_level)

        # Create logs directory if needed
        if self.settings.system.log_file_path:
            self.settings.system.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        handlers = []
        if self.settings.system.log_to_console:
            handlers.append(logging.StreamHandler())
        if self.settings.system.log_to_file:
            handlers.append(logging.FileHandler(self.settings.system.log_file_path))

        logging.basicConfig(
            level=log_level,
            format=self.settings.system.log_format,
            handlers=handlers,
        )

        # Remove None handlers
        root_logger = logging.getLogger()
        root_logger.handlers = [h for h in root_logger.handlers if h is not None]

    async def run(self):
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
            data_status = self.market_data.get_data_status()
            self.logger.info(f"Initial market data status: {data_status}")

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

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, requesting shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _initialize_components(self):
        """Initialize all trading components."""
        console.print("[cyan]Initializing trading components...[/cyan]")

        # Initialize market data provider
        console.print("  • Connecting to market data feed...")
        await self.market_data.connect()

        # Initialize exchange client
        console.print("  • Connecting to exchange...")
        connected = await self.exchange_client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to exchange")

        # Verify LLM agent is available
        console.print("  • Verifying LLM agent...")
        if not self.llm_agent.is_available():
            console.print(
                "[yellow]    Warning: LLM not available, using fallback logic[/yellow]"
            )

        # Load initial market data
        console.print("  • Loading initial market data...")
        await self._wait_for_initial_data()

        # Initialize MCP experience manager if enabled
        if self.experience_manager:
            console.print("  • Starting experience tracking...")
            try:
                await self.experience_manager.start()
                console.print("    [green]✓ Experience tracking started[/green]")
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

        console.print("[green]✓ All components initialized successfully[/green]")

    async def _wait_for_initial_data(self):
        """Wait for sufficient market data to begin trading."""
        max_wait_time = 60  # seconds
        wait_start = datetime.now(UTC)
        historical_data_loaded = False
        websocket_data_received = False

        while True:
            elapsed_time = (datetime.now(UTC) - wait_start).total_seconds()
            if elapsed_time > max_wait_time:
                # Get detailed status before failing
                status = self.market_data.get_data_status()
                self.logger.error(
                    f"Timeout waiting for initial market data. Status: {status}"
                )
                raise RuntimeError(
                    f"Timeout waiting for initial market data after {max_wait_time}s"
                )

            # Check for historical data
            data = self.market_data.get_latest_ohlcv(
                limit=100
            )  # Check for indicator minimum
            if len(data) >= 100 and not historical_data_loaded:
                self.logger.info(f"Loaded {len(data)} historical candles for analysis")
                historical_data_loaded = True
            elif len(data) >= 50 and not historical_data_loaded:
                # Fallback: proceed with limited data but warn about potential issues
                self.logger.warning(
                    f"Limited historical data available: {len(data)} candles. "
                    f"Indicators may be unreliable until more data is accumulated."
                )
                historical_data_loaded = True

            # Check for WebSocket data
            if self.market_data.has_websocket_data() and not websocket_data_received:
                self.logger.info("WebSocket is receiving real-time market data")
                websocket_data_received = True

            # We're ready when we have historical data and either:
            # 1. WebSocket is receiving data, OR
            # 2. We've waited at least 10 seconds for WebSocket (market might be closed/inactive)
            if historical_data_loaded:
                if websocket_data_received:
                    self.logger.info(
                        "Both historical and real-time data available, ready to trade"
                    )
                    break
                elif elapsed_time > 10:
                    # After 10 seconds, proceed if we have historical data even without WebSocket
                    self.logger.warning(
                        "Proceeding with historical data only. WebSocket data not yet received "
                        "(market may be closed or inactive)"
                    )
                    break

            # Log progress every 5 seconds
            if int(elapsed_time) % 5 == 0 and elapsed_time > 0:
                status = self.market_data.get_data_status()
                self.logger.info(
                    f"Waiting for data... Elapsed: {int(elapsed_time)}s, "
                    f"Historical: {len(data)} candles, "
                    f"WebSocket connected: {status.get('websocket_connected', False)}, "
                    f"WebSocket data: {status.get('websocket_data_received', False)}, "
                    f"Latest price: ${status.get('latest_price', 'N/A')}"
                )

            await asyncio.sleep(1)

    def _display_startup_summary(self):
        """Display trading engine startup summary."""
        table = Table(title="Trading Engine Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        # Market data status
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
        table.add_row(
            "Exchange",
            "✓ Connected" if exchange_status["connected"] else "✗ Disconnected",
            f"{'Sandbox' if exchange_status['sandbox'] else 'Live'} mode",
        )

        # LLM status
        llm_status = self.llm_agent.get_status()
        table.add_row(
            "LLM Agent",
            "✓ Available" if llm_status["llm_available"] else "⚠ Fallback",
            f"{llm_status['model_provider']}:{llm_status['model_name']}",
        )

        # Risk manager
        table.add_row(
            "Risk Manager",
            "✓ Active",
            f"Max size: {self.settings.trading.max_size_pct}%",
        )

        console.print(table)
        console.print()

    async def _main_trading_loop(self):
        """
        Main trading loop that runs continuously.

        Processes market data, calculates indicators, gets LLM decisions,
        validates trades, applies risk management, and executes orders.
        """
        self.logger.info("Starting main trading loop...")
        self._running = True

        loop_count = 0

        while self._running and not self._shutdown_requested:
            try:
                loop_start = datetime.now(UTC)
                loop_count += 1

                # Check if we have fresh market data
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
                        await self.market_data.connect()
                        continue

                # Get latest market data
                latest_data = self.market_data.get_latest_ohlcv(limit=200)
                if not latest_data:
                    self.logger.warning("No market data available, waiting...")
                    await asyncio.sleep(5)
                    continue

                current_price = latest_data[-1].close

                # Calculate technical indicators
                df = self.market_data.to_dataframe(limit=200)

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
                    except Exception as e:
                        self.logger.warning(
                            f"Indicator calculation failed: {e}, using fallback values"
                        )
                        # Use fallback indicator state
                        indicator_state = self._get_fallback_indicator_state()

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
                interval_minutes = self._get_interval_minutes(self.interval)
                candles_per_24h = min((24 * 60) // interval_minutes, len(latest_data))

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

                # Get LLM trading decision
                trade_action = await self.llm_agent.analyze_market(market_state)

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

                # Update position tracking and risk metrics
                await self._update_position_tracking(current_price)

                # Display periodic status updates
                if loop_count % 10 == 0:  # Every 10 loops
                    self._display_status_update(loop_count, current_price, final_action)

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
                f"Executing trade: {trade_action.action} {trade_action.size_pct}%"
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
                    except Exception as e:
                        self.logger.warning(f"Failed to link order to experience: {e}")

                if order.status in ["FILLED", "PENDING"]:
                    self.successful_trades += 1

                    # Update position manager
                    if hasattr(order, "filled_quantity") and order.filled_quantity > 0:
                        # Store previous position before update
                        previous_position = self.current_position

                        updated_position = (
                            self.position_manager.update_position_from_order(
                                order, order.price
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
                                await self.experience_manager.complete_trade(
                                    order, order.price, market_state
                                )
                                self.logger.info(
                                    "Completed trade tracking for closed position"
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
                                        f"Started tracking trade: {trade_id}"
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
                else:
                    console.print(f"[yellow]⚠ Trade failed:[/yellow] {order.status}")
            else:
                console.print("[red]✗ Trade execution failed[/red]")

        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            console.print(f"[red]Trade execution error: {e}[/red]")

    def _get_interval_minutes(self, interval: str) -> int:
        """
        Convert interval string to minutes.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h', '1d')

        Returns:
            Number of minutes in the interval
        """
        interval_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
        }
        return interval_map.get(interval.lower(), 5)  # Default to 5 minutes

    async def _update_position_tracking(self, current_price: Decimal):
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

    async def _shutdown(self):
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
            if hasattr(self, "exchange_client") and self.exchange_client.is_connected():
                console.print("  • Cancelling open orders...")
                cleanup_tasks.append(
                    asyncio.create_task(
                        self.exchange_client.cancel_all_orders(self.symbol)
                    )
                )

            # Close market data connection
            if hasattr(self, "market_data"):
                console.print("  • Disconnecting from market data...")
                cleanup_tasks.append(asyncio.create_task(self.market_data.disconnect()))

            # Close exchange connection
            if hasattr(self, "exchange_client"):
                console.print("  • Disconnecting from exchange...")
                cleanup_tasks.append(
                    asyncio.create_task(self.exchange_client.disconnect())
                )

            # Stop experience manager if enabled
            if hasattr(self, "experience_manager") and self.experience_manager:
                console.print("  • Stopping experience tracking...")
                cleanup_tasks.append(
                    asyncio.create_task(self.experience_manager.stop())
                )

            # Close dominance data connection - CRITICAL for async session cleanup
            if hasattr(self, "dominance_provider") and self.dominance_provider:
                console.print("  • Disconnecting from dominance data...")
                cleanup_tasks.append(
                    asyncio.create_task(self.dominance_provider.disconnect())
                )

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
                            self.dominance_provider._session._connector.close()
                        except Exception:
                            pass

    def _display_final_summary(self):
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
@click.option("--interval", default="1m", help="Candle interval")
@click.option("--config", default=None, help="Configuration file path")
def live(dry_run: bool, symbol: str, interval: str, config: str | None) -> None:
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

        # Confirmation for live trading
        if not click.confirm("Are you sure you want to trade with real money?"):
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
        console.print(f"❌ Error: {e}", style="red")
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
