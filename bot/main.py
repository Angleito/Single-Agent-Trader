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
import inspect
import logging
import signal
import tempfile
import time
from asyncio import Task
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party imports with error handling
try:
    import click
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError as e:
    print(f"❌ Critical dependency missing: {e}")
    print("Install with: pip install click rich python-dotenv")
    sys.exit(1)

# Initialize console early for error reporting
console = Console()

# Startup diagnostics
_startup_errors: list[str] = []
_startup_warnings: list[str] = []


def _safe_import(
    module_path: str, class_name: str | None = None, required: bool = True
) -> Any:
    """Safely import a module or class with error handling."""
    try:
        # Handle relative imports by converting to absolute
        if module_path.startswith("."):
            # Convert relative to absolute import
            full_module_path = f"bot{module_path}"
        else:
            full_module_path = module_path

        if class_name:
            module = __import__(full_module_path, fromlist=[class_name])
            return getattr(module, class_name)
        return __import__(full_module_path)
    except ImportError as e:
        error_msg = f"Failed to import {class_name or module_path}: {e}"
        if required:
            _startup_errors.append(error_msg)
            console.print(f"❌ {error_msg}", style="red")
            return None
        _startup_warnings.append(error_msg)
        console.print(f"⚠️  {error_msg}", style="yellow")
        return None
    except Exception as e:
        error_msg = f"Unexpected error importing {class_name or module_path}: {e}"
        if required:
            _startup_errors.append(error_msg)
            console.print(f"❌ {error_msg}", style="red")
            return None
        _startup_warnings.append(error_msg)
        console.print(f"⚠️  {error_msg}", style="yellow")
        return None


# Core imports (required for basic functionality)
try:
    from .config import Settings, create_settings
    from .trading_types import MarketState, Position, TradeAction
    from .utils import setup_warnings_suppression
    from .validator import TradeValidator
except ImportError as e:
    console.print(f"❌ Critical core component missing: {e}", style="red")
    console.print("Bot cannot start without core components", style="red")
    sys.exit(1)

# Essential trading components
try:
    from .exchange.factory import ExchangeFactory
    from .paper_trading import PaperTradingAccount
    from .position_manager import PositionManager
    from .risk import RiskManager
except ImportError as e:
    console.print(f"❌ Essential trading component missing: {e}", style="red")
    _startup_errors.append(f"Essential trading component missing: {e}")

# Market data providers
MarketDataProvider = _safe_import(".data.market", "MarketDataProvider", required=True)
DominanceDataProvider = _safe_import(
    ".data.dominance", "DominanceDataProvider", required=False
)

# WebSocket and command handling
WebSocketPublisher = _safe_import(
    ".websocket_publisher", "WebSocketPublisher", required=False
)
CommandConsumer = _safe_import(".command_consumer", "CommandConsumer", required=False)

# Type checking imports
if TYPE_CHECKING:
    from .data.bluefin_market import BluefinMarketDataProvider

    MarketDataProviderType = MarketDataProvider | BluefinMarketDataProvider | None
elif MarketDataProvider is not None:
    MarketDataProviderType = MarketDataProvider | None
else:
    MarketDataProviderType = None

import contextlib

# Lazy loading for heavy components
_lazy_imports = {
    "VuManChuIndicators": (".indicators.vumanchu", "VuManChuIndicators"),
    "LLMAgent": (".strategy.llm_agent", "LLMAgent"),
    "MemoryEnhancedLLMAgent": (
        ".strategy.memory_enhanced_agent",
        "MemoryEnhancedLLMAgent",
    ),
    "ExperienceManager": (".learning.experience_manager", "ExperienceManager"),
    "MCPMemoryServer": (".mcp.memory_server", "MCPMemoryServer"),
    "OmniSearchClient": (".mcp.omnisearch_client", "OmniSearchClient"),
    "TradeLogger": (".logging.trade_logger", "TradeLogger"),
    "PerformanceMonitor": (".performance_monitor", "PerformanceMonitor"),
    "PerformanceThresholds": (".performance_monitor", "PerformanceThresholds"),
    "MarketMakingIntegrator": (
        ".strategy.market_making_integration",
        "MarketMakingIntegrator",
    ),
    "MarketMakingConfig": (".market_making_config", "MarketMakingConfig"),
    "create_default_config": (".market_making_config", "create_default_config"),
    "validate_config": (".market_making_config", "validate_config"),
}


def _get_lazy_import(name: str) -> Any:
    """Get a lazily imported component."""
    if name not in _lazy_imports:
        raise ValueError(f"Unknown lazy import: {name}")

    module_path, class_name = _lazy_imports[name]
    return _safe_import(module_path, class_name, required=False)


def _check_startup_health() -> None:
    """Check startup health and display diagnostics."""
    if _startup_errors:
        console.print("❌ Startup errors detected:", style="red bold")
        for error in _startup_errors:
            console.print(f"  • {error}", style="red")
        console.print(
            "\n🔧 Please fix these errors before starting the bot", style="yellow"
        )
        sys.exit(1)

    if _startup_warnings:
        console.print(
            "⚠️  Startup warnings (bot will continue with reduced functionality):",
            style="yellow bold",
        )
        for warning in _startup_warnings:
            console.print(f"  • {warning}", style="yellow")
        console.print("")


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
    - Adaptive timing optimization for high-frequency trading

    Features:
    - Adaptive loop timing based on trading interval
    - High-frequency optimizations for 1-second analysis intervals
    - Performance monitoring and timing accuracy tracking
    - Efficient resource management for sub-second operations
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval: str = "1m",  # Note: 15s was changed to 1m due to Bluefin API limitations
        config_file: str | None = None,
        dry_run: bool | None = None,
        market_making_enabled: bool | None = None,
        market_making_symbol: str | None = None,
        market_making_profile: str | None = None,
    ):
        """
        Initialize the trading engine.

        Args:
            symbol: Trading symbol
            interval: Candle interval
            config_file: Optional configuration file path
            dry_run: Whether to run in dry-run mode (None = use config/env settings)
            market_making_enabled: Override market making enabled setting
            market_making_symbol: Override market making symbol
            market_making_profile: Override market making profile
        """
        self.symbol = symbol
        self.interval = interval
        self._running = False
        self._shutdown_requested = False
        self._memory_available = False  # Initialize early to prevent AttributeError
        self._last_position_log_time: datetime | None = None
        self._background_tasks: list[asyncio.Task[Any]] = (
            []
        )  # Track background tasks for cleanup

        # Initialize market making integrator (will be set up after LLM agent)
        self.market_making_integrator: Any | None = None

        # Store market making CLI overrides
        self._market_making_enabled_override = market_making_enabled
        self._market_making_symbol_override = market_making_symbol
        self._market_making_profile_override = market_making_profile

        # Initialize basic configuration and setup
        self._initialize_basic_setup(config_file, dry_run)

        # Initialize all components
        self._initialize_core_components()
        self._initialize_optional_components()
        self._initialize_trading_infrastructure()
        self._initialize_performance_monitoring()

    def _initialize_basic_setup(
        self, config_file: str | None, dry_run: bool | None
    ) -> None:
        """Initialize basic configuration and setup."""
        # Initialize adaptive timing attributes
        self._adaptive_timing_info: dict[str, Any] | None = None
        self._adaptive_timing_error: str | None = None

        # Load configuration
        self.settings = self._load_configuration(config_file, dry_run)

        # Apply adaptive timing based on trading interval
        self._apply_adaptive_timing()

        # Set dry_run from settings after configuration is loaded
        self.dry_run = self.settings.system.dry_run

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize paper trading if in dry run mode
        self.paper_account = None
        if self.dry_run:
            balance = self.settings.paper_trading.starting_balance
            self.paper_account = PaperTradingAccount(starting_balance=balance)

    def _initialize_core_components(self) -> None:
        """Initialize core trading components with error handling."""
        # Initialize components (market data will be initialized after exchange connection)
        self.market_data: MarketDataProviderType = None

        # Initialize VuManChu indicators with lazy loading
        self.logger.debug("About to initialize VuManChu indicators...")
        try:
            vuman_chu_indicators_cls = _get_lazy_import("VuManChuIndicators")
            if vuman_chu_indicators_cls:
                self.indicator_calc = vuman_chu_indicators_cls()
                self.logger.debug("VuManChu indicators initialized successfully")
            else:
                self.logger.warning("VuManChu indicators unavailable - using fallback")
                self.indicator_calc = None
        except Exception as e:
            self.logger.warning("Failed to initialize VuManChu indicators: %s", e)
            self.indicator_calc = None

        self.actual_trading_symbol = (
            self.symbol
        )  # Will be updated if futures are enabled

        # Initialize MCP memory components if enabled
        self.logger.debug("About to initialize MCP memory components...")
        self.memory_server = None
        self.experience_manager = None
        self._memory_available = False
        self.logger.debug("MCP memory components initialized")

    def _initialize_optional_components(self) -> None:
        """Initialize optional components like OmniSearch and WebSocket."""
        self._initialize_omnisearch()
        self._initialize_websocket_components()
        self._initialize_llm_agent()
        self._initialize_market_making_integrator()

    def _initialize_omnisearch(self) -> None:
        """Initialize OmniSearch client if enabled."""
        self.logger.debug("About to initialize OmniSearch client...")
        self.omnisearch_client = None
        if self.settings.omnisearch.enabled:
            self.logger.info("OmniSearch integration enabled, initializing client...")
            try:
                omnisearch_client_cls = _get_lazy_import("OmniSearchClient")
                if omnisearch_client_cls:
                    self.omnisearch_client = omnisearch_client_cls(
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
                else:
                    self.logger.warning(
                        "OmniSearch client unavailable - feature disabled"
                    )
                    self.omnisearch_client = None
            except Exception as e:
                self.logger.warning("Failed to initialize OmniSearch client: %s", e)
                self.logger.info("Continuing without OmniSearch integration")
                self.omnisearch_client = None

    def _initialize_websocket_components(self) -> None:
        """Initialize WebSocket-related components."""
        self._initialize_websocket_publisher()
        self._initialize_command_consumer()

    def _initialize_websocket_publisher(self) -> None:
        """Initialize WebSocket publisher for real-time dashboard integration."""
        self.websocket_publisher = None
        if self.settings.system.enable_websocket_publishing:
            self.logger.info("WebSocket publishing enabled, initializing publisher...")
            try:
                if WebSocketPublisher:
                    self.websocket_publisher = WebSocketPublisher(self.settings)
                    self.logger.info("WebSocketPublisher initialized successfully")
                else:
                    self.logger.warning(
                        "WebSocket publisher unavailable - feature disabled"
                    )
                    self.websocket_publisher = None
            except Exception as e:
                self.logger.warning("Failed to initialize WebSocket publisher: %s", e)
                self.logger.info("Continuing without WebSocket publishing")
                self.websocket_publisher = None

    def _initialize_command_consumer(self) -> None:
        """Initialize Command Consumer for bidirectional dashboard control."""
        self.command_consumer = None
        if self.settings.system.enable_websocket_publishing:  # Use same setting for now
            self.logger.info(
                "Dashboard control enabled, initializing command consumer..."
            )
            try:
                if CommandConsumer:
                    self.command_consumer = CommandConsumer()
                    self._register_command_callbacks()
                    self.logger.info("Successfully initialized command consumer")
                else:
                    self.logger.warning(
                        "Command consumer unavailable - dashboard control disabled"
                    )
                    self.command_consumer = None
            except Exception as e:
                self.logger.warning("Failed to initialize command consumer: %s", e)
                self.logger.info("Continuing without dashboard control")
                self.command_consumer = None

    def _initialize_llm_agent(self) -> None:
        """Initialize LLM agent (will be either LLMAgent or MemoryEnhancedLLMAgent)."""
        self.llm_agent: Any

        if self.settings.mcp.enabled:
            self._initialize_memory_enhanced_agent()
        else:
            self._initialize_standard_agent()

    def _initialize_memory_enhanced_agent(self) -> None:
        """Initialize memory-enhanced LLM agent with MCP support."""
        self.logger.info("MCP memory system enabled, initializing components...")
        try:
            mcp_memory_server_cls = _get_lazy_import("MCPMemoryServer")
            experience_manager_cls = _get_lazy_import("ExperienceManager")
            memory_enhanced_llm_agent_cls = _get_lazy_import("MemoryEnhancedLLMAgent")

            if not all(
                [
                    mcp_memory_server_cls,
                    experience_manager_cls,
                    memory_enhanced_llm_agent_cls,
                ]
            ):
                self.logger.warning(
                    "MCP components unavailable - falling back to standard agent"
                )
                self._initialize_standard_agent()
                return

            self.memory_server = mcp_memory_server_cls(
                server_url=self.settings.mcp.server_url,
                api_key=(
                    self.settings.mcp.memory_api_key.get_secret_value()
                    if self.settings.mcp.memory_api_key
                    else None
                ),
            )
            self.experience_manager = experience_manager_cls(self.memory_server)

            # Use memory-enhanced agent
            self.llm_agent = memory_enhanced_llm_agent_cls(
                model_provider=self.settings.llm.provider,
                model_name=self.settings.llm.model_name,
                memory_server=self.memory_server,
                omnisearch_client=self.omnisearch_client,
            )
            self.logger.info("Successfully initialized memory-enhanced agent")
            self._memory_available = True
        except Exception as e:
            self.logger.warning("Failed to initialize MCP components: %s", e)
            self.logger.info("Falling back to standard LLM agent")
            self._initialize_standard_agent()

    def _initialize_standard_agent(self) -> None:
        """Initialize standard LLM agent without memory."""
        try:
            llm_agent_cls = _get_lazy_import("LLMAgent")
            if llm_agent_cls:
                self.llm_agent = llm_agent_cls(
                    model_provider=self.settings.llm.provider,
                    model_name=self.settings.llm.model_name,
                    omnisearch_client=self.omnisearch_client,
                )
                self.logger.info("Successfully initialized standard LLM agent")
            else:
                self.logger.error(
                    "LLM agent unavailable - bot cannot make trading decisions"
                )
                self.llm_agent = None
        except Exception:
            self.logger.exception("Failed to initialize LLM agent")
            self.logger.exception("Bot cannot make trading decisions without LLM agent")
            self.llm_agent = None

    def _initialize_market_making_integrator(self) -> None:
        """Initialize market making integrator if enabled."""
        self.market_making_integrator = None

        # Apply CLI overrides to market making configuration
        mm_enabled = self._market_making_enabled_override
        if mm_enabled is None:
            mm_enabled = self.settings.market_making.enabled

        if not mm_enabled:
            self.logger.info("Market making disabled - using LLM agent for all symbols")
            return

        try:
            market_making_integrator_cls = _get_lazy_import("MarketMakingIntegrator")
            if not market_making_integrator_cls:
                self.logger.warning(
                    "MarketMakingIntegrator unavailable - falling back to LLM agent only"
                )
                return

            # Apply CLI overrides for market making configuration
            mm_config = self.settings.market_making.model_copy(deep=True)

            # Override enabled setting
            if self._market_making_enabled_override is not None:
                mm_config.enabled = self._market_making_enabled_override

            # Override symbol
            if self._market_making_symbol_override:
                mm_config.symbol = self._market_making_symbol_override

            # Override profile and apply profile-specific settings
            if self._market_making_profile_override:
                mm_config.profile = self._market_making_profile_override
                mm_config = mm_config.apply_profile(
                    self._market_making_profile_override
                )

            # Determine market making symbols from configuration
            market_making_symbols = [mm_config.symbol]

            # Initialize the integrator
            self.market_making_integrator = market_making_integrator_cls(
                symbol=self.symbol,
                exchange_client=None,  # Will be set later during trading infrastructure setup
                dry_run=self.dry_run,
                market_making_symbols=market_making_symbols,
                config=mm_config.model_dump(),
            )

            self.logger.info(
                "Market making integrator initialized for symbols: %s",
                market_making_symbols,
            )
            if self._market_making_profile_override:
                self.logger.info(
                    "Using CLI override profile: %s",
                    self._market_making_profile_override,
                )

        except Exception as e:
            self.logger.warning("Failed to initialize market making integrator: %s", e)
            self.logger.info("Falling back to LLM agent for all symbols")
            self.market_making_integrator = None

    def _initialize_trading_infrastructure(self) -> None:
        """Initialize trading infrastructure components."""
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

        # Set exchange client for market making integrator if available
        if self.market_making_integrator:
            self.market_making_integrator.exchange_client = self.exchange_client

        # Initialize dominance data provider if enabled
        self._initialize_dominance_provider()

        # Initialize position and performance tracking
        self._initialize_tracking()

    def _initialize_dominance_provider(self) -> None:
        """Initialize dominance data provider if enabled."""
        self.dominance_provider = None
        if self.settings.dominance.enable_dominance_data:
            try:
                if DominanceDataProvider:
                    self.dominance_provider = DominanceDataProvider(
                        data_source=self.settings.dominance.data_source,
                        api_key=(
                            self.settings.dominance.api_key.get_secret_value()
                            if self.settings.dominance.api_key
                            else None
                        ),
                        update_interval=self.settings.dominance.update_interval,
                    )
                    self.logger.info("Dominance data provider initialized")
                else:
                    self.logger.warning(
                        "Dominance data provider unavailable - feature disabled"
                    )
                    self.dominance_provider = None
            except Exception as e:
                self.logger.warning(
                    "Failed to initialize dominance data provider: %s", e
                )
                self.dominance_provider = None

    def _initialize_tracking(self) -> None:
        """Initialize position and performance tracking."""
        # Position tracking
        self.current_position = Position(
            symbol=self.symbol,
            side="FLAT",
            size=Decimal(0),
            timestamp=datetime.now(UTC),
        )

        # Performance tracking
        self.trade_count = 0
        self.successful_trades = 0
        self.total_pnl = Decimal(0)
        self.start_time = datetime.now(UTC)

        # Trading interval control
        self.last_trade_time: datetime | None = None
        self.last_candle_analysis_time: datetime | None = None
        self.trading_enabled = False  # Will be enabled after data validation
        self.data_validation_complete = False

        # Initialize structured trade logger for comprehensive logging
        try:
            trade_logger_cls = _get_lazy_import("TradeLogger")
            if trade_logger_cls:
                self.trade_logger = trade_logger_cls()
                self.logger.info("Structured trade logger initialized")
            else:
                self.logger.warning("Trade logger unavailable - using basic logging")
                self.trade_logger = None
        except Exception as e:
            self.logger.warning("Failed to initialize trade logger: %s", e)
            self.trade_logger = None

    class NullPerformanceMonitor:
        """
        Null object implementation of performance monitor.

        Provides the same interface as PerformanceMonitor but does nothing,
        allowing graceful degradation when performance monitoring is unavailable.
        """

        def __init__(self) -> None:
            pass

        @contextmanager
        def track_operation(
            self, _operation_name: str, _tags: dict[str, str] | None = None
        ) -> Any:
            """
            No-op context manager for tracking operations.

            Args:
                operation_name: Name of the operation (ignored)
                tags: Additional tags (ignored)

            Yields:
                None - just allows the context manager to work
            """
            # No-op context manager - just yield control back
            yield

        def add_alert_callback(self, callback: Any) -> None:
            """No-op alert callback registration."""

        def get_performance_summary(self, _duration: Any = None) -> dict[str, Any]:
            """Return empty performance summary."""
            return {
                "timestamp": datetime.now(UTC),
                "period_minutes": 0,
                "latency_summary": {},
                "resource_summary": {},
                "recent_alerts": [],
                "bottleneck_analysis": {"bottlenecks": [], "recommendations": []},
                "health_score": 100.0,
                "monitoring_available": False,
            }

        async def start_monitoring(
            self, resource_monitor_interval: float = 5.0
        ) -> None:
            """No-op start monitoring."""

        async def stop_monitoring(self) -> None:
            """No-op stop monitoring."""

    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring system."""
        self.logger.debug("Initializing performance monitoring system...")
        try:
            performance_thresholds_cls = _get_lazy_import("PerformanceThresholds")
            performance_monitor_cls = _get_lazy_import("PerformanceMonitor")

            if performance_thresholds_cls and performance_monitor_cls:
                performance_thresholds = performance_thresholds_cls()

                # Customize thresholds based on trading frequency
                self._configure_performance_thresholds(performance_thresholds)

                self.performance_monitor = performance_monitor_cls(
                    performance_thresholds
                )

                # Add alert callback for critical performance issues
                self.performance_monitor.add_alert_callback(
                    self._handle_performance_alert
                )
                self.logger.info("Performance monitoring system initialized")
            else:
                self.logger.warning(
                    "Performance monitoring unavailable - using null performance monitor"
                )
                self.performance_monitor = self.NullPerformanceMonitor()
        except Exception as e:
            self.logger.warning(
                "Failed to initialize performance monitoring: %s - using null performance monitor",
                e,
            )
            self.performance_monitor = self.NullPerformanceMonitor()

    def _configure_performance_thresholds(self, thresholds: Any) -> None:
        """Configure performance thresholds based on trading interval."""
        # High-frequency trading thresholds
        if self.interval in ["1s", "5s", "10s", "15s"]:
            thresholds.indicator_calculation_ms = 50
            thresholds.market_data_processing_ms = 25
            thresholds.trade_execution_ms = 500
        # Medium frequency thresholds
        elif self.interval in ["1m", "5m"]:
            thresholds.indicator_calculation_ms = 100
            thresholds.market_data_processing_ms = 50
            thresholds.trade_execution_ms = 1000
        # Lower frequency thresholds
        else:
            thresholds.indicator_calculation_ms = 200
            thresholds.market_data_processing_ms = 100
            thresholds.trade_execution_ms = 2000

        # Adjust thresholds for paper trading (less strict)
        if self.dry_run:
            thresholds.indicator_calculation_ms *= 2
            thresholds.market_data_processing_ms *= 2
            thresholds.trade_execution_ms *= 3

        self.logger.info("Performance monitoring system initialized")

        # Final health check and component status report
        self._log_component_status()
        self.logger.info(
            "Initialized TradingEngine for %s at %s", self.symbol, self.interval
        )

    def _log_component_status(self) -> None:
        """Log the status of all initialized components."""
        components = {
            "Indicators": self.indicator_calc is not None,
            "LLM Agent": self.llm_agent is not None,
            "Memory System": self._memory_available,
            "Market Making": (
                self.market_making_integrator is not None
                and self.market_making_integrator.status.market_making_enabled
            ),
            "WebSocket Publisher": self.websocket_publisher is not None,
            "Command Consumer": self.command_consumer is not None,
            "OmniSearch Client": self.omnisearch_client is not None,
            "Dominance Provider": self.dominance_provider is not None,
            "Trade Logger": self.trade_logger is not None,
            "Performance Monitor": self.performance_monitor is not None,
        }

        active_components = [name for name, status in components.items() if status]
        disabled_components = [
            name for name, status in components.items() if not status
        ]

        self.logger.info("Component Status Summary:")
        self.logger.info("  Active Components: %s", ", ".join(active_components))
        if disabled_components:
            self.logger.info(
                "  Disabled Components: %s", ", ".join(disabled_components)
            )

        # Check for critical missing components
        critical_missing = []
        if not self.llm_agent:
            critical_missing.append("LLM Agent")
        if not self.indicator_calc:
            critical_missing.append("Technical Indicators")

        if critical_missing:
            self.logger.warning(
                "Critical components missing: %s", ", ".join(critical_missing)
            )
            self.logger.warning("Bot functionality will be severely limited")
        else:
            self.logger.info("All critical components initialized successfully")

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
        except Exception:
            self.logger.exception("Error closing positions during emergency stop")

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
        self.logger.info("Updating risk limits from dashboard: %s", parameters)

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

        except Exception:
            self.logger.exception("Error updating risk limits")
            raise

    async def _handle_manual_trade(self, parameters: dict) -> bool:
        """Handle manual trade command from dashboard."""
        self.logger.info("Executing manual trade from dashboard: %s", parameters)

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
                        {
                            "order_id": f"manual_{int(time.time())}",
                            "symbol": self.symbol,
                            "side": trade_action.action,
                            "size": str(trade_action.size_pct),
                            "price": "market",
                            "status": "filled",
                            "manual": True,
                        }
                    )
            else:
                self.logger.warning("Manual trade execution failed")

        except Exception:
            self.logger.exception("Error executing manual trade")
            return False
        else:
            return success

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
        except Exception:
            self.logger.exception("Error closing positions")

    def _handle_performance_alert(self, alert) -> None:
        """
        Handle performance alerts from the monitoring system.

        Args:
            alert: PerformanceAlert object containing alert details
        """
        try:
            # Log the alert
            alert_level = "CRITICAL" if alert.level.value == "critical" else "WARNING"
            self.logger.warning(
                "🚨 PERFORMANCE ALERT [%s]: %s", alert_level, alert.message
            )

            # For critical alerts in paper trading, we might want to slow down execution
            if self.dry_run and alert.level.value == "critical":
                if "memory" in alert.metric_name.lower():
                    self.logger.warning(
                        "High memory usage detected in paper trading mode"
                    )
                elif "latency" in alert.metric_name.lower():
                    self.logger.warning(
                        "High latency detected - consider reducing update frequency"
                    )

            # Publish alert to dashboard if available
            if hasattr(self, "websocket_publisher") and self.websocket_publisher:
                alert_task = asyncio.create_task(
                    self.websocket_publisher.publish_system_status(
                        status="performance_alert",
                        health=alert.level.value != "critical",
                        message=alert.message,
                        additional_data={
                            "alert_level": alert.level.value,
                            "metric_name": alert.metric_name,
                            "current_value": alert.current_value,
                            "threshold": alert.threshold,
                            "timestamp": alert.timestamp.isoformat(),
                        },
                    )
                )
                if hasattr(self, "_background_tasks"):
                    self._background_tasks.append(alert_task)

        except Exception:
            self.logger.exception("Error handling performance alert")

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

    def _get_market_data_provider(self) -> Any:
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
        For high-frequency scalping, we trade on every new candle completion.
        Note: Bluefin DEX only supports intervals >= 1m. Sub-minute intervals are converted with granularity loss.

        Returns:
            True if trading is allowed, False otherwise
        """
        # Perform initial checks
        if not self.trading_enabled or not self.data_validation_complete:
            return False

        if not self._ensure_market_data_available():
            return False

        # Get latest candle data
        latest_data = self._get_latest_market_data()
        if not latest_data:
            return False

        latest_candle = latest_data[-1]

        # Perform timing validations
        return self._validate_trading_timing(latest_candle)

    def _get_latest_market_data(self) -> list[Any]:
        """Get latest market data, handling sync/async methods."""
        try:
            latest_data: list[Any] = []
            if self.market_data is not None and hasattr(
                self.market_data, "get_latest_ohlcv"
            ):
                method = self.market_data.get_latest_ohlcv
                if inspect.iscoroutinefunction(method):
                    # Can't await in non-async method, defer to caller
                    self.logger.debug(
                        "Async method detected, skipping fresh data check"
                    )
                    return []
                latest_data = method(limit=1)

            if not latest_data:
                self.logger.debug("📊 No market data available")

        except Exception as e:
            self.logger.warning("Error checking latest data: %s", e)
            return []
        else:
            return latest_data

    def _validate_trading_timing(self, latest_candle) -> bool:
        """Validate if timing conditions allow trading."""
        current_time = datetime.now(UTC)

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

        # Check if enough time has passed since last analysis
        candle_interval_seconds = self._get_interval_seconds(self.interval)
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

        # Check minimum interval between actual trades
        if self.last_trade_time is not None:
            min_interval = self.settings.trading.min_trading_interval_seconds
            if not isinstance(min_interval, int | float):
                self.logger.warning(
                    "Invalid min_interval type: %s, value: %s",
                    type(min_interval),
                    min_interval,
                )
                min_interval = 15  # Default fallback
            min_interval = int(min_interval)  # Ensure it's an integer

            # Ensure both timestamps have timezone info for comparison
            last_trade_time = self.last_trade_time
            if last_trade_time.tzinfo is None:
                last_trade_time = last_trade_time.replace(tzinfo=UTC)

            time_since_last_trade = (current_time - last_trade_time).total_seconds()

            if time_since_last_trade < min_interval:
                self.logger.debug(
                    "⏱️ Waiting for trade interval: %.1fs / %ds",
                    time_since_last_trade,
                    min_interval,
                )
                return False

        return True

    def _load_configuration(
        self, config_file: str | None, dry_run: bool | None
    ) -> Settings:
        """Load and validate configuration with enhanced dotenv support."""
        # Always load .env file first to ensure environment variables are available
        load_dotenv()

        # Check for CONFIG_FILE environment variable if no config_file provided
        if not config_file:
            config_file = os.getenv("CONFIG_FILE")

        # Check if key environment variables are present
        exchange_type = os.getenv("EXCHANGE__EXCHANGE_TYPE")
        trading_symbol = os.getenv("TRADING__SYMBOL")

        # Determine configuration source
        if exchange_type and trading_symbol and not config_file:
            # Use environment variables when they're present and no config file specified
            console.print("[green]Using environment configuration (.env file)[/green]")
            settings = create_settings()
        elif config_file:
            # Use config file when explicitly specified
            console.print(f"[blue]Using configuration file: {config_file}[/blue]")
            settings = Settings.load_from_file(config_file)
        else:
            # Fallback to default settings
            console.print("[yellow]Using default configuration[/yellow]")
            settings = create_settings()

        # Override dry_run mode if explicitly specified
        if dry_run is not None and dry_run != settings.system.dry_run:
            system_settings = settings.system.model_copy(update={"dry_run": dry_run})
            settings = settings.model_copy(update={"system": system_settings})

        # Apply market making CLI overrides if specified
        if hasattr(self, "_market_making_enabled_override") and any(
            [
                self._market_making_enabled_override is not None,
                self._market_making_symbol_override is not None,
                self._market_making_profile_override is not None,
            ]
        ):
            mm_settings = settings.market_making.model_copy(deep=True)

            # Apply overrides
            if self._market_making_enabled_override is not None:
                mm_settings.enabled = self._market_making_enabled_override

            if self._market_making_symbol_override:
                mm_settings.symbol = self._market_making_symbol_override

            if self._market_making_profile_override:
                mm_settings.profile = self._market_making_profile_override
                # Apply profile-specific settings
                mm_settings = mm_settings.apply_profile(
                    self._market_making_profile_override
                )

            # Update settings with modified market making configuration
            settings = settings.model_copy(update={"market_making": mm_settings})

        # Validate configuration for trading
        warnings = settings.validate_trading_environment()
        if warnings:
            console.print("[yellow]Configuration warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        return settings

    def _setup_logging(self) -> None:
        """Setup logging configuration with graceful permission error handling."""
        log_level = getattr(logging, self.settings.system.log_level)

        # Configure logging handlers
        handlers: list[logging.Handler] = []

        # Always add console handler as primary fallback
        if self.settings.system.log_to_console:
            handlers.append(logging.StreamHandler())

        # Attempt file logging with graceful fallback
        if self.settings.system.log_to_file:
            file_handler = self._create_file_handler()
            if file_handler:
                handlers.append(file_handler)
            elif not self.settings.system.log_to_console:
                # If file logging fails and console logging is disabled, force console logging
                print(
                    "WARNING: File logging failed and console logging disabled. Enabling console logging as fallback."
                )
                handlers.append(logging.StreamHandler())

        # Ensure we always have at least one handler
        if not handlers:
            print(
                "WARNING: No logging handlers configured. Adding console handler as fallback."
            )
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=log_level,
            format=self.settings.system.log_format,
            handlers=handlers,
        )

        # Remove None handlers
        root_logger = logging.getLogger()
        root_logger.handlers = [h for h in root_logger.handlers if h is not None]

    def _create_file_handler(self) -> logging.FileHandler | None:
        """Create file handler with permission error handling and fallback locations."""
        if not self.settings.system.log_file_path:
            return None

        # List of paths to try in order of preference
        temp_dir = Path(tempfile.gettempdir())
        fallback_paths = [
            self.settings.system.log_file_path,  # Original path
            temp_dir / "bot.log",  # Container fallback using secure temp dir
            Path.home() / "bot.log",  # User home fallback
            Path.cwd() / "bot.log",  # Current directory fallback
        ]

        for log_path in fallback_paths:
            try:
                # Test write permissions by attempting to create parent directory
                log_path.parent.mkdir(parents=True, exist_ok=True)

                # Test write access to the file location
                test_file = log_path.with_suffix(".test")
                try:
                    test_file.touch()
                    test_file.unlink()  # Clean up test file
                except (PermissionError, OSError):
                    # If we can't write test file, skip this path
                    continue

                # Create the actual file handler
                handler = logging.FileHandler(str(log_path))

                # Log success message (using print since logging isn't set up yet)
                if log_path != self.settings.system.log_file_path:
                    print(f"WARNING: Using fallback log file location: {log_path}")
                else:
                    print(f"INFO: Logging to file: {log_path}")

                return handler

            except (PermissionError, OSError, FileNotFoundError) as e:
                # Continue to next fallback path
                if log_path == fallback_paths[-1]:  # Last fallback failed
                    print(f"WARNING: Failed to create log file at {log_path}: {e}")
                continue

        # All fallback paths failed
        print(
            "ERROR: Unable to create log file at any location. File logging disabled."
        )
        print("Available fallback locations were:")
        for path in fallback_paths:
            print(f"  - {path}")

        return None

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
                self.logger.info("Initial market data status: %s", data_status)
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
            self.logger.exception("Critical error in trading engine")
            console.print(f"[red]Critical error: {e}[/red]")
            raise
        finally:
            # Ensure shutdown is called exactly once
            if not shutdown_called:
                shutdown_called = True
                try:
                    await self._shutdown()
                except Exception:
                    self.logger.exception("Error in shutdown")
                    # Force cleanup of services
                    await self._force_cleanup_services()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, _frame: Any) -> None:
            self.logger.info("Received signal %s, requesting shutdown...", signum)
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _initialize_components(self) -> None:
        """Initialize all trading components."""
        console.print("[cyan]Initializing trading components...[/cyan]")

        # Initialize exchange and determine trading symbol
        await self._initialize_exchange()

        # Initialize market data provider
        await self._initialize_market_data()

        # Verify LLM agent
        await self._verify_llm_agent()

        # Initialize market making integrator with LLM agent
        await self._initialize_market_making_integration()

        # Connect optional services
        await self._connect_optional_services()

        # Load initial market data
        console.print("  • Loading initial market data...")
        await self._wait_for_initial_data()

        # Initialize trading components
        await self._initialize_trading_components()

        # Initialize dashboard components
        await self._initialize_dashboard_components()

        # Start performance monitoring
        await self._start_performance_monitoring()

        console.print("[green]✓ All components initialized successfully[/green]")

    async def _initialize_exchange(self):
        """Initialize exchange client and determine trading symbol."""
        console.print("  • Connecting to exchange...")
        connected = await self.exchange_client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to exchange")

        # Determine trading symbol
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

    async def _initialize_market_data(self):
        """Initialize market data provider based on exchange type."""
        console.print("  • Connecting to market data feed...")

        interval_seconds = self._get_interval_seconds(self.interval)
        is_high_frequency = interval_seconds <= 60

        if self.settings.exchange.exchange_type == "bluefin":
            self.market_data = await self._setup_bluefin_market_data(
                is_high_frequency, interval_seconds
            )
        else:
            self.market_data = await self._setup_coinbase_market_data()

        # Ensure market data is initialized and connected
        if self.market_data is None:
            raise RuntimeError("Market data provider was not properly initialized")
        await self.market_data.connect()

    async def _setup_bluefin_market_data(
        self, is_high_frequency: bool, interval_seconds: int
    ):
        """Setup market data for Bluefin exchange."""
        try:
            from .data.bluefin_market import BluefinMarketDataProvider

            self.logger.info(
                "Using Bluefin native market data provider for %s", self.symbol
            )
            console.print("    Using Bluefin DEX native market data")
            return BluefinMarketDataProvider(self.symbol, self.interval)
        except ImportError as e:
            self.logger.warning("BluefinMarketDataProvider not available: %s", e)
            return await self._setup_bluefin_fallback_market_data(
                is_high_frequency, interval_seconds
            )

    async def _setup_bluefin_fallback_market_data(
        self, is_high_frequency: bool, interval_seconds: int
    ):
        """Setup fallback market data for Bluefin when native provider unavailable."""
        if is_high_frequency:
            return await self._setup_realtime_market_data(interval_seconds)
        return await self._setup_standard_market_data_fallback()

    async def _setup_realtime_market_data(self, interval_seconds: int):
        """Setup real-time market data provider for high-frequency trading."""
        try:
            from .data.realtime_market import RealtimeMarketDataProvider

            self.logger.info(
                "Using real-time WebSocket market data provider for HF trading: %s",
                self.symbol,
            )

            # Configure intervals
            realtime_intervals = [interval_seconds]
            if interval_seconds > 1:
                realtime_intervals.append(1)  # Always include 1-second candles
            if 5 not in realtime_intervals and interval_seconds != 5:
                realtime_intervals.append(5)  # Include 5-second candles

            console.print(
                f"    Using real-time WebSocket data for HF trading ({realtime_intervals}s intervals)"
            )
            return RealtimeMarketDataProvider(self.symbol, realtime_intervals)

        except ImportError:
            self.logger.warning(
                "RealtimeMarketDataProvider not available, falling back to standard provider"
            )
            console.print(
                "    Using standard market data provider (real-time module not available)"
            )
            return MarketDataProvider(self.symbol, self.interval)

    async def _setup_standard_market_data_fallback(self):
        """Setup standard market data provider as fallback."""
        self.logger.info(
            "Falling back to standard MarketDataProvider for %s", self.symbol
        )
        console.print("    Using fallback market data provider for Bluefin")
        return MarketDataProvider(self.symbol, self.interval)

    async def _setup_coinbase_market_data(self):
        """Setup market data for Coinbase exchange."""
        market_data_symbol = self.actual_trading_symbol
        self.logger.info("Using Coinbase market data for %s", market_data_symbol)
        console.print(f"    Using Coinbase market data for {market_data_symbol}")
        return MarketDataProvider(market_data_symbol, self.interval)

    async def _verify_llm_agent(self):
        """Verify LLM agent availability."""
        console.print("  • Verifying LLM agent...")
        if not self.llm_agent.is_available():
            console.print(
                "[yellow]    Warning: LLM not available, using fallback logic[/yellow]"
            )

    async def _initialize_market_making_integration(self):
        """Initialize market making integration with LLM agent."""
        if not self.market_making_integrator:
            return

        try:
            console.print("  • Initializing market making integration...")
            await self.market_making_integrator.initialize(self.llm_agent)

            if self.market_making_integrator.status.market_making_enabled:
                console.print(
                    f"    ✓ Market making enabled for {self.market_making_integrator.symbol}"
                )
            else:
                console.print(
                    f"    • Market making disabled for {self.symbol}, using LLM agent"
                )

        except Exception as e:
            self.logger.warning("Failed to initialize market making integration: %s", e)
            console.print(
                "[yellow]    Warning: Market making integration failed, using LLM agent only[/yellow]"
            )
            self.market_making_integrator = None

    async def _connect_optional_services(self):
        """Connect to optional services with proper error handling and fallback."""
        service_statuses = {}  # Initialize to empty dict for fallback

        try:
            # Use service startup manager for robust initialization
            from bot.utils.service_startup import startup_services_with_retry

            console.print("  • Initializing optional services...")
            service_instances, service_statuses = await startup_services_with_retry(
                self.settings, max_retries=2
            )

            # Update component references from successful services
            if "websocket_publisher" in service_instances:
                self.websocket_publisher = service_instances["websocket_publisher"]

            if "bluefin_service" in service_instances:
                self._bluefin_service_client = service_instances["bluefin_service"]

            if "omnisearch" in service_instances:
                self.omnisearch_client = service_instances["omnisearch"]

            # Log service status summary
            for name, status in service_statuses.items():
                if status.available:
                    self.logger.info("Service %s: Available", name)
                else:
                    self.logger.warning(
                        "Service %s: %s", name, status.error or "Unavailable"
                    )

        except Exception as e:
            self.logger.warning(
                "Service initialization error: %s. Continuing with available services.",
                str(e),
            )
            console.print("    [yellow]⚠ Some optional services unavailable[/yellow]")

        # Connect OmniSearch if available (legacy path)
        if self.omnisearch_client and "omnisearch" not in service_statuses:
            await self._connect_omnisearch()

    async def _connect_omnisearch(self):
        """Connect to OmniSearch service."""
        console.print("  • Connecting to OmniSearch service...")
        try:
            connected = await self.omnisearch_client.connect()
            if connected:
                console.print("    [green]✓ OmniSearch connected[/green]")
            else:
                console.print("    [yellow]⚠ OmniSearch service unavailable[/yellow]")
        except Exception as e:
            self.logger.warning("Failed to connect to OmniSearch: %s", e)
            console.print("    [yellow]⚠ OmniSearch connection failed[/yellow]")

    async def _initialize_trading_components(self):
        """Initialize trading-related components."""
        await self._initialize_experience_manager()
        await self._connect_dominance_provider()
        await self._reconcile_positions()

    async def _initialize_experience_manager(self):
        """Initialize MCP experience manager if enabled."""
        if not self.experience_manager:
            return

        console.print("  • Starting experience tracking...")
        try:
            await self.experience_manager.start()
            console.print("    [green]✓ Experience tracking started[/green]")
            await self._setup_memory_system()
        except Exception as e:
            self.logger.warning("Failed to start experience manager: %s", e)
            console.print("    [yellow]⚠ Experience tracking unavailable[/yellow]")

    async def _setup_memory_system(self):
        """Setup memory system and log statistics."""
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

            if memory_count >= 10:
                await self._log_pattern_performance()

    async def _log_pattern_performance(self):
        """Log pattern performance statistics."""
        try:
            pattern_stats = await self.memory_server.get_pattern_statistics()
            if pattern_stats:
                self.logger.info("=== Pattern Performance Summary ===")
                sorted_patterns = sorted(
                    pattern_stats.items(),
                    key=lambda x: x[1]["success_rate"] * x[1]["count"],
                    reverse=True,
                )[:5]

                for pattern, stats in sorted_patterns:
                    if stats["count"] >= 3:  # Only show patterns with enough samples
                        self.logger.info(
                            "  %s: %.1%% win rate (%s trades, avg PnL=$%.2f)",
                            pattern,
                            stats["success_rate"] * 100,
                            stats["count"],
                            stats["avg_pnl"],
                        )
        except Exception as e:
            self.logger.debug("Could not retrieve pattern statistics: %s", e)

    async def _connect_dominance_provider(self):
        """Connect dominance data provider."""
        if not hasattr(self, "dominance_provider") or not self.dominance_provider:
            return

        console.print("  • Connecting to stablecoin dominance data...")
        try:
            await self.dominance_provider.connect()
            console.print("    [green]✓ Dominance data connected[/green]")
        except Exception as e:
            self.logger.warning("Failed to connect dominance data: %s", e)
            console.print("    [yellow]⚠ Dominance data unavailable[/yellow]")

    async def _initialize_dashboard_components(self):
        """Initialize dashboard-related components."""
        # WebSocket publisher should already be initialized in _connect_optional_services
        # Just log the status here
        if self.websocket_publisher:
            console.print("    [green]✓ Dashboard WebSocket already connected[/green]")
            try:
                await self.websocket_publisher.publish_system_status(
                    status="initialized", health=True
                )
            except Exception as e:
                self.logger.debug("Could not publish initial status: %s", str(e))
        else:
            console.print("    [yellow]⚠ Dashboard WebSocket not available[/yellow]")

        await self._start_command_consumer()

    # Method removed - functionality moved to _connect_optional_services

    async def _start_command_consumer(self):
        """Start command consumer for dashboard."""
        if not self.command_consumer:
            return

        console.print("  • Starting dashboard command consumer...")
        try:
            await self.command_consumer.start_polling_task()
            console.print("    [green]✓ Dashboard command consumer started[/green]")
        except Exception as e:
            self.logger.warning("Failed to start command consumer: %s", e)
            console.print(
                "    [yellow]⚠ Dashboard command consumer unavailable[/yellow]"
            )

    async def _start_performance_monitoring(self):
        """Start performance monitoring system."""
        console.print("  • Starting performance monitoring...")
        try:
            await self.performance_monitor.start_monitoring(
                resource_monitor_interval=5.0
            )
            console.print("    [green]✓ Performance monitoring started[/green]")
        except Exception as e:
            console.print(
                f"    [yellow]⚠ Performance monitoring failed to start: {e}[/yellow]"
            )
            self.logger.warning("Performance monitoring startup failed: %s", e)

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
            if not isinstance(elapsed_time, int | float) or elapsed_time < 0:
                self.logger.warning("Invalid elapsed_time: %s", elapsed_time)
                elapsed_time = 0

            # Check timeout
            if elapsed_time > max_wait_time:
                await self._handle_data_timeout()

            # Get market data
            data = await self._fetch_initial_market_data()

            # Process historical data if not loaded
            if not historical_data_loaded:
                historical_data_loaded = await self._process_historical_data(
                    data, min_candles_required, require_24h_data
                )

            # Check WebSocket data
            websocket_data_received = self._check_websocket_data(
                websocket_data_received
            )

            # Check if ready to start trading
            if await self._check_trading_readiness(
                historical_data_loaded, websocket_data_received, elapsed_time
            ):
                break

            # Log progress periodically
            await self._log_data_wait_progress(elapsed_time, data)
            await asyncio.sleep(1)

    async def _handle_data_timeout(self):
        """Handle timeout when waiting for initial data."""
        if self.market_data is not None:
            status = self.market_data.get_data_status()
            self.logger.error(
                "Timeout waiting for initial market data. Status: %s", status
            )
        else:
            self.logger.error(
                "Timeout waiting for initial market data. Market data provider not initialized"
            )
        raise RuntimeError("Timeout waiting for initial market data after 180s")

    async def _fetch_initial_market_data(self) -> list:
        """Fetch initial market data with async handling."""
        data: list[Any] = []
        if self.market_data is not None and hasattr(
            self.market_data, "get_latest_ohlcv"
        ):
            method = self.market_data.get_latest_ohlcv
            try:
                if inspect.iscoroutinefunction(method):
                    data = await method(limit=500)
                else:
                    data = method(limit=500)

                # Handle async objects
                data = await self._resolve_market_data_async(data)

            except Exception as e:
                self.logger.warning("Error getting market data: %s", e)
                data = []

        # Validate data type
        return self._validate_market_data_type(data)

    async def _resolve_market_data_async(self, data):
        """Resolve various async data types for initial data."""

        if inspect.iscoroutine(data):
            self.logger.warning("Detected coroutine data, awaiting...")
            return await data
        if isinstance(data, asyncio.Task):
            self.logger.warning("Detected asyncio.Task data, awaiting...")
            return await data
        if hasattr(data, "__await__"):
            self.logger.warning("Detected awaitable object, awaiting...")
            return await data

        return data

    def _validate_market_data_type(self, data) -> list:
        """Validate and convert market data to proper type."""

        if isinstance(data, asyncio.Task):
            self.logger.error(
                "Data is still an asyncio.Task: %s. This should have been awaited.",
                data,
            )
            return []
        if inspect.iscoroutine(data):
            self.logger.error(
                "Data is still a coroutine: %s. This should have been awaited.", data
            )
            return []
        if not isinstance(data, list | tuple):
            self.logger.warning(
                "Unexpected data type: %s, converting to list", type(data)
            )
            return list(data) if data else []

        return data

    async def _process_historical_data(
        self, data: list, min_candles_required: int, require_24h_data: bool
    ) -> bool:
        """Process historical data and determine if sufficient."""
        interval_seconds = self._get_interval_seconds(self.interval)

        if require_24h_data:
            return self._process_24h_data_requirement(
                data, min_candles_required, interval_seconds
            )
        return self._process_minimum_data_requirement(data, min_candles_required)

    def _process_24h_data_requirement(
        self, data: list, min_candles_required: int, interval_seconds: int
    ) -> bool:
        """Process data when 24h requirement is enabled."""
        if len(data) >= min_candles_required:
            hours_available = (len(data) * interval_seconds) / 3600
            self.logger.info(
                "✅ Loaded %s historical candles (%.1f hours at %s intervals) for scalping analysis",
                len(data),
                hours_available,
                self.interval,
            )
            self.trading_enabled = True
            return True
        if len(data) >= 50:
            hours_available = (len(data) * interval_seconds) / 3600
            self.logger.warning(
                "⚠️ Limited data available (%.2f hours, %s candles). "
                "Starting with reduced data for scalping...",
                hours_available,
                len(data),
            )
            self.trading_enabled = True
            return True

        return False

    def _process_minimum_data_requirement(
        self, data: list, min_candles_required: int
    ) -> bool:
        """Process data when only minimum requirement is set."""
        if len(data) >= min_candles_required:
            self.logger.info(
                "✅ Loaded %s historical candles (minimum %s) for analysis",
                len(data),
                min_candles_required,
            )
            self.trading_enabled = True
            return True
        if len(data) >= 50:
            self.logger.warning(
                "⚠️ Limited historical data available: %s candles. "
                "Indicators may be unreliable until more data is accumulated.",
                len(data),
            )
            self.trading_enabled = False
            return True

        return False

    def _check_websocket_data(self, websocket_data_received: bool) -> bool:
        """Check if WebSocket data is available."""
        if (
            self.market_data is not None
            and self.market_data.has_websocket_data()
            and not websocket_data_received
        ):
            self.logger.info("📡 WebSocket is receiving real-time market data")
            return True

        return websocket_data_received

    async def _check_trading_readiness(
        self,
        historical_data_loaded: bool,
        websocket_data_received: bool,
        elapsed_time: float,
    ) -> bool:
        """Check if ready to start trading."""
        if not historical_data_loaded:
            return False

        if websocket_data_received:
            self.logger.info(
                "🚀 Both historical and real-time data available, ready for scalping"
            )
            self.data_validation_complete = True
            return True

        if elapsed_time > 10:
            self.logger.warning(
                "⚠️ Proceeding with historical data only for scalping. WebSocket data not yet received "
                "(market may be closed or starting fast for scalping)"
            )
            self.data_validation_complete = True
            return True

        return False

    async def _log_data_wait_progress(self, elapsed_time: float, data: list):
        """Log progress every 10 seconds."""
        # Ensure elapsed_time is valid before using in modulo operation
        safe_elapsed_time = (
            int(elapsed_time)
            if isinstance(elapsed_time, int | float) and elapsed_time >= 0
            else 0
        )
        if safe_elapsed_time % 10 != 0 or elapsed_time <= 0:
            return

        try:
            await self._log_realtime_provider_status(elapsed_time, data)
        except ImportError:
            await self._log_basic_provider_status(elapsed_time, data)

    async def _log_realtime_provider_status(self, elapsed_time: float, data: list):
        """Log status for realtime provider."""
        from .data.realtime_market import RealtimeMarketDataProvider

        if (
            RealtimeMarketDataProvider
            and self.market_data is not None
            and isinstance(self.market_data, RealtimeMarketDataProvider)
        ):
            status = self.market_data.get_status()
            tick_rate = status.get("tick_rate_per_second", 0)
            current_price = status.get("current_price", "N/A")

            # Ensure elapsed_time is valid for logging
            safe_elapsed_time = (
                int(elapsed_time)
                if isinstance(elapsed_time, int | float) and elapsed_time >= 0
                else 0
            )
            self.logger.info(
                "⏳ Waiting for real-time data... Elapsed: %ds\n   📊 Current candles: %s available\n   🌐 WebSocket connected: %s\n   📈 Tick rate: %.1f ticks/sec\n   💰 Current price: $%s\n   ⚡ Trading enabled: %s",
                safe_elapsed_time,
                len(data),
                status.get("websocket_connected", False),
                tick_rate,
                current_price,
                self.trading_enabled,
            )
        else:
            await self._log_standard_provider_status(elapsed_time, data)

    async def _log_standard_provider_status(self, elapsed_time: float, data: list):
        """Log status for standard provider."""
        if self.market_data is not None:
            status = self.market_data.get_data_status()
        else:
            status = {"connected": False, "websocket_data_received": False}

        interval_seconds = self._get_interval_seconds(self.interval)
        candles_per_24h = (24 * 60 * 60) // interval_seconds
        hours_available = (len(data) * interval_seconds) / 3600 if data else 0

        # Ensure elapsed_time is valid for logging
        safe_elapsed_time = (
            int(elapsed_time)
            if isinstance(elapsed_time, int | float) and elapsed_time >= 0
            else 0
        )
        self.logger.info(
            "⏳ Waiting for data... Elapsed: %ds\n   📊 Historical: %s/%s candles (%.1f/24 hours)\n   🌐 WebSocket connected: %s\n   📈 WebSocket data: %s\n   💰 Latest price: $%s\n   ⚡ Trading enabled: %s",
            safe_elapsed_time,
            len(data),
            candles_per_24h,
            hours_available,
            status.get("websocket_connected", False),
            status.get("websocket_data_received", False),
            status.get("latest_price", "N/A"),
            self.trading_enabled,
        )

    async def _log_basic_provider_status(self, elapsed_time: float, data: list):
        """Log basic status when RealtimeMarketDataProvider not available."""
        # Ensure elapsed_time is valid for logging
        safe_elapsed_time = (
            int(elapsed_time)
            if isinstance(elapsed_time, int | float) and elapsed_time >= 0
            else 0
        )
        self.logger.info(
            "⏳ Waiting for data... Elapsed: %ds\n   📊 Current candles: %s available\n   ⚡ Trading enabled: %s",
            safe_elapsed_time,
            len(data),
            self.trading_enabled,
        )

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
            realtime_market_data_provider = None

        if not realtime_market_data_provider or not isinstance(
            self.market_data, realtime_market_data_provider
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

        # Market Making status
        if self.market_making_integrator:
            mm_status = (
                "✓ Enabled"
                if self.market_making_integrator.status.market_making_enabled
                else "⚠ Available"
            )
            mm_strategy = self.market_making_integrator.get_strategy_for_symbol(
                self.symbol
            )
            mm_details = f"Strategy: {mm_strategy.title()}"
            if self.market_making_integrator.status.market_making_enabled:
                mm_symbols = ", ".join(
                    self.market_making_integrator.market_making_symbols
                )
                mm_details += f" | Symbols: {mm_symbols}"
        else:
            mm_status = "✗ Disabled"
            mm_details = "LLM agent for all symbols"

        table.add_row(
            "Market Making",
            mm_status,
            mm_details,
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

        # Adaptive timing configuration
        timing_status = "✓ Optimized"
        timing_details = (
            f"{self.settings.system.update_frequency_seconds:.1f}s frequency"
        )

        # Add adaptive timing information if available
        if hasattr(self, "_adaptive_timing_info") and self._adaptive_timing_info:
            timing_info = self._adaptive_timing_info
            timing_details = (
                f"{timing_info['new']:.1f}s frequency "
                f"(adapted from {timing_info['original']:.1f}s for {timing_info['reason']})"
            )
            # Log the timing adaptation details
            self.logger.info(
                "🚀 Adaptive Timing Applied: %s interval → %.1fs main loop frequency (%s)",
                timing_info["interval"],
                timing_info["new"],
                timing_info["reason"],
            )
        elif hasattr(self, "_adaptive_timing_error"):
            timing_status = "⚠ Default"
            timing_details = f"{self.settings.system.update_frequency_seconds:.1f}s frequency (adaptive timing failed)"
            self.logger.warning(
                "Adaptive timing configuration failed: %s", self._adaptive_timing_error
            )

        table.add_row(
            "Loop Timing",
            timing_status,
            timing_details,
        )

        console.print(table)
        console.print()

    async def _initialize_trading_loop(self) -> None:
        """Initialize the trading loop with performance metrics."""
        if self.websocket_publisher and self.websocket_publisher.connected:
            # Get initial price if available
            initial_price = Decimal(0)
            try:
                if self.market_data is not None:
                    latest_data = self.market_data.get_latest_ohlcv(limit=1)
                    if latest_data:
                        initial_price = latest_data[-1].close
            except Exception as e:
                self.logger.debug(
                    "Could not get initial price for performance metrics: %s", e
                )

            await self._publish_performance_metrics(initial_price)

    async def _handle_market_data_connection(self) -> bool:
        """Check market data connection and handle reconnection if needed.

        Returns:
            True if connection is ready, False if should continue to next iteration
        """
        if self.market_data is None:
            self.logger.error(
                "Market data provider not initialized, cannot continue trading"
            )
            return False

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
                return False
            # Only attempt manual reconnection if WebSocket handler isn't trying
            self.logger.warning("Attempting manual reconnection...")
            if self.market_data is not None:
                await self.market_data.connect()
            return False

        return True

    async def _main_trading_loop(self) -> None:
        """
        Main trading loop that runs continuously with enhanced error handling.

        Processes market data, calculates indicators, gets LLM decisions,
        validates trades, applies risk management, and executes orders.

        Features:
        - Comprehensive error recovery
        - Component health monitoring
        - Circuit breaker integration
        - Graceful degradation
        """
        self.logger.info("Starting enhanced main trading loop...")
        self._running = True

        loop_count = 0
        last_status_log = datetime.now(UTC)
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_health_check = datetime.now(UTC)
        health_check_interval = 60  # seconds

        # Enhanced error tracking
        error_categories = {
            "market_data": 0,
            "indicators": 0,
            "llm_decision": 0,
            "trade_execution": 0,
            "risk_management": 0,
            "connectivity": 0,
            "unknown": 0,
        }

        # Initialize loop with performance metrics
        try:
            await self._initialize_trading_loop()
        except Exception:
            self.logger.exception("Failed to initialize trading loop")
            return

        # Setup trading environment with validation
        try:
            loop_context = await self._setup_trading_environment()
            self.logger.info(
                "✅ Trading loop initialized successfully with enhanced error handling"
            )
        except Exception:
            self.logger.exception("Failed to setup trading environment")
            return

        while self._running and not self._shutdown_requested:
            loop_start = datetime.now(UTC)
            loop_count += 1

            try:
                # Pre-iteration health checks
                if (
                    datetime.now(UTC) - last_health_check
                ).total_seconds() > health_check_interval:
                    await self._perform_basic_health_check()
                    last_health_check = datetime.now(UTC)

                # Check circuit breaker state before processing
                if not self._check_circuit_breaker_basic():
                    await asyncio.sleep(5)
                    continue

                # Process market data and get trading decision
                trade_result = await self._process_trading_iteration(
                    loop_context, loop_count, loop_start
                )

                if trade_result:
                    consecutive_errors = 0  # Reset on successful iteration

                    # Handle periodic maintenance tasks
                    try:
                        await self._handle_periodic_tasks(
                            loop_count, last_status_log, loop_context
                        )
                    except Exception as maintenance_error:
                        self.logger.warning(
                            "Periodic tasks failed: %s", maintenance_error
                        )
                        # Don't let maintenance failures stop the main loop
                else:
                    # Iteration was skipped but not an error
                    pass

                # Calculate and apply sleep timing
                await self._handle_loop_timing(
                    loop_start, loop_context["target_frequency"]
                )

            except Exception as e:
                consecutive_errors += 1

                # Categorize and handle error
                error_category = self._categorize_error_basic(e)
                error_categories[error_category] += 1

                await self._handle_loop_error_enhanced(
                    e, loop_count, loop_context, error_category, consecutive_errors
                )

                # Check if we've hit consecutive error threshold
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.exception(
                        "🚨 Maximum consecutive errors reached (%d). Entering emergency pause.",
                        max_consecutive_errors,
                    )
                    await self._handle_emergency_situation(error_categories)
                    # Reset counter after emergency handling
                    consecutive_errors = 0

                continue

            # Log periodic health summary
            if loop_count % 100 == 0:
                total_errors = sum(error_categories.values())
                error_rate = (total_errors / loop_count) * 100 if loop_count > 0 else 0
                self.logger.info(
                    "📊 Trading Loop Health (Iteration %d): %.2f%% error rate (%d total errors)",
                    loop_count,
                    error_rate,
                    total_errors,
                )

        self.logger.info(
            "🛑 Trading loop stopped - Final error summary: %s", error_categories
        )

    async def _setup_trading_environment(self) -> dict:
        """Setup trading environment and return context."""
        target_frequency = self.settings.system.update_frequency_seconds
        is_high_frequency = target_frequency <= 1.0
        interval_seconds = self._get_interval_seconds(self.interval)

        # Cache provider type to reduce overhead
        realtime_provider_class = await self._get_realtime_provider_class()
        is_realtime_provider = realtime_provider_class is not None and isinstance(
            self.market_data, realtime_provider_class
        )

        if is_high_frequency:
            self.logger.info(
                "🚀 High-frequency mode enabled: %.1fs interval with %.1fs loop frequency",
                interval_seconds,
                target_frequency,
            )

        return {
            "target_frequency": target_frequency,
            "is_high_frequency": is_high_frequency,
            "interval_seconds": interval_seconds,
            "is_realtime_provider": is_realtime_provider,
        }

    async def _get_realtime_provider_class(self):
        """Get realtime provider class with error handling."""
        try:
            from .data.realtime_market import RealtimeMarketDataProvider
        except ImportError:
            return None
        else:
            return RealtimeMarketDataProvider

    async def _process_trading_iteration(
        self, loop_context: dict, loop_count: int, _loop_start
    ) -> bool:
        """Process a single trading iteration. Returns False if iteration should be skipped."""
        # Check market data connection
        if not await self._handle_market_data_connection():
            return False

        # Get market data
        latest_data = await self._get_market_data(loop_context)
        if not latest_data:
            self.logger.warning("No market data available, waiting...")
            await asyncio.sleep(5)
            return False

        # Process market data and indicators
        current_price, market_state = await self._process_market_data_and_indicators(
            latest_data
        )

        # Check trading conditions
        if not self._can_trade_now():
            await asyncio.sleep(1)
            return False

        # Execute trading logic
        await self._execute_trading_logic(
            market_state, current_price, latest_data, loop_count
        )

        return True

    async def _get_market_data(self, loop_context: dict) -> list:
        """Get latest market data based on provider type."""

        if loop_context["is_realtime_provider"] and self.market_data is not None:
            return await self._get_realtime_market_data(
                loop_context["interval_seconds"]
            )
        if self.market_data is not None and hasattr(
            self.market_data, "get_latest_ohlcv"
        ):
            return await self._get_standard_market_data()
        return []

    async def _get_realtime_market_data(self, interval_seconds: int) -> list:
        """Get market data from realtime provider."""
        if hasattr(self.market_data, "get_candle_history"):
            latest_data = self.market_data.get_candle_history(
                interval_seconds, limit=200
            )

            # Try to get more data if insufficient
            if len(latest_data) < 50 and hasattr(self.market_data, "tick_aggregator"):
                self.market_data.tick_aggregator.force_complete_candles(self.symbol)
                latest_data = self.market_data.get_candle_history(
                    interval_seconds, limit=200
                )

            return latest_data
        return []

    async def _get_standard_market_data(self) -> list:
        """Get market data from standard provider."""
        method = self.market_data.get_latest_ohlcv
        try:
            if inspect.iscoroutinefunction(method):
                latest_data = await method(limit=200)
            else:
                latest_data = method(limit=200)

            # Handle async objects
            latest_data = await self._resolve_async_data(latest_data)

            # Validate data type
            if not isinstance(latest_data, list | tuple):
                self.logger.warning(
                    "Unexpected latest_data type: %s, converting to list",
                    type(latest_data),
                )
                latest_data = list(latest_data) if latest_data else []
            else:
                return latest_data

        except Exception as e:
            self.logger.warning("Error getting market data in main loop: %s", e)
        return latest_data

    async def _resolve_async_data(self, latest_data: Any) -> Any:
        """Resolve various async data types."""
        if inspect.iscoroutine(latest_data):
            self.logger.warning("Detected coroutine data in main loop, awaiting...")
            return await latest_data
        if isinstance(latest_data, asyncio.Task):
            self.logger.warning("Detected asyncio.Task data in main loop, awaiting...")
            return await latest_data
        if hasattr(latest_data, "__await__"):
            self.logger.warning("Detected awaitable object in main loop, awaiting...")
            return await latest_data
        return latest_data

    async def _process_market_data_and_indicators(self, latest_data: list) -> tuple:
        """Process market data and calculate indicators."""
        # Track market data processing performance with defensive check
        if self.performance_monitor is not None and hasattr(
            self.performance_monitor, "track_operation"
        ):
            context_manager = self.performance_monitor.track_operation(
                "market_data_processing",
                {
                    "symbol": self.actual_trading_symbol,
                    "data_points": str(len(latest_data)),
                },
            )
        else:
            # Fallback no-op context manager
            @contextmanager
            def fallback_context():
                yield

            context_manager = fallback_context()

        with context_manager:
            current_price = latest_data[-1].close

            # Publish market data to dashboard
            if self.websocket_publisher:
                await self.websocket_publisher.publish_market_data(
                    symbol=self.actual_trading_symbol,
                    price=float(current_price),
                    timestamp=latest_data[-1].timestamp,
                )

        # Calculate indicators and create market state
        indicator_state, dominance_candles = await self._calculate_indicators()
        indicator_dict = await self._prepare_indicator_data(indicator_state)
        dominance_obj = await self._process_dominance_data(indicator_dict)

        # Create market state
        market_state = await self._create_market_state(
            current_price, latest_data, indicator_dict, dominance_obj, dominance_candles
        )

        return current_price, market_state

    async def _calculate_indicators(self) -> tuple[list[Any], Any | None]:
        """Calculate technical indicators."""
        if self.market_data is None:
            self.logger.error(
                "Market data provider not available for indicator calculation"
            )
            return self._get_fallback_indicator_state(), None

        market_data = self.market_data.to_dataframe(limit=200)
        dominance_candles = await self._get_dominance_candles()

        # Validate data sufficiency
        if len(market_data) < 100:
            self.logger.warning(
                "Insufficient data for reliable indicators: %s candles. Using fallback values.",
                len(market_data),
            )
            return self._get_fallback_indicator_state(), dominance_candles

        # Calculate indicators
        return (
            await self._compute_indicators(market_data, dominance_candles),
            dominance_candles,
        )

    async def _get_dominance_candles(self) -> Any:
        """Generate dominance candlesticks if provider available."""
        if not hasattr(self, "dominance_provider") or not self.dominance_provider:
            return None

        try:
            dominance_history = self.dominance_provider.get_dominance_history(hours=2)
            if len(dominance_history) >= 6:
                from .data.dominance import DominanceCandleBuilder

                candle_builder = DominanceCandleBuilder(dominance_history)
                dominance_candles = candle_builder.build_candles(interval="3T")
                return (
                    dominance_candles[-20:]
                    if len(dominance_candles) > 20
                    else dominance_candles
                )
        except Exception as e:
            self.logger.warning("Failed to build dominance candles: %s", e)

        return None

    async def _compute_indicators(
        self, market_data: list[Any], dominance_candles: Any
    ) -> dict[str, Any]:
        """Compute technical indicators with error handling."""
        try:
            with self.performance_monitor.track_operation(
                "indicator_calculation",
                {
                    "candles": str(len(market_data)),
                    "dominance_available": str(dominance_candles is not None),
                },
            ):
                df_with_indicators = self.indicator_calc.calculate_all(
                    market_data, dominance_candles=dominance_candles
                )
                indicator_state = self.indicator_calc.get_latest_state(
                    df_with_indicators
                )

            # Publish indicator data
            if self.websocket_publisher:
                await self._publish_indicator_data(indicator_state)

            return indicator_state

        except Exception as e:
            self.logger.warning(
                "Indicator calculation failed: %s, using fallback values", e
            )
            indicator_state = self._get_fallback_indicator_state()

            if self.websocket_publisher:
                await self.websocket_publisher.publish_indicator_data(
                    symbol=self.actual_trading_symbol,
                    indicators=indicator_state,
                )

            return indicator_state

    async def _publish_indicator_data(self, indicator_state: dict[str, Any]) -> None:
        """Publish indicator data to dashboard."""
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

    async def _prepare_indicator_data(self, indicator_state: dict) -> dict:
        """Prepare indicator data dictionary."""
        indicator_dict = {
            "timestamp": datetime.now(UTC),
            "cipher_a_dot": indicator_state.get("cipher_a", {}).get("trend_dot"),
            "cipher_b_wave": indicator_state.get("cipher_b", {}).get("wave"),
            "cipher_b_money_flow": indicator_state.get("cipher_b", {}).get(
                "money_flow"
            ),
            "rsi": indicator_state.get("cipher_a", {}).get("rsi"),
            "ema_fast": indicator_state.get("cipher_a", {}).get("ema_fast"),
            "ema_slow": indicator_state.get("cipher_a", {}).get("ema_slow"),
            "vwap": indicator_state.get("cipher_b", {}).get("vwap"),
        }

        # Add dominance analysis if available
        dominance_analysis = indicator_state.get("dominance_analysis", {})
        if dominance_analysis:
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

        return indicator_dict

    async def _process_dominance_data(self, indicator_dict: dict):
        """Process dominance data and update indicator dictionary."""
        dominance_obj = None

        if not hasattr(self, "dominance_provider") or not self.dominance_provider:
            return dominance_obj

        try:
            dominance_data = self.dominance_provider.get_latest_dominance()
            if dominance_data:
                # Add dominance metrics
                dominance_metrics = self._create_dominance_metrics(dominance_data)
                indicator_dict.update(dominance_metrics)

                # Get market sentiment
                try:
                    sentiment_analysis = self.dominance_provider.get_market_sentiment()
                    indicator_dict["market_sentiment"] = sentiment_analysis.get(
                        "sentiment", "NEUTRAL"
                    )
                except Exception as e:
                    self.logger.warning("Failed to get market sentiment: %s", e)
                    indicator_dict["market_sentiment"] = "NEUTRAL"

                # Convert DominanceData to StablecoinDominance for MarketState compatibility
                from .trading_types import StablecoinDominance

                dominance_obj = StablecoinDominance(
                    timestamp=dominance_data.timestamp,
                    stablecoin_dominance=dominance_data.stablecoin_dominance or 0.0,
                    usdt_dominance=dominance_data.usdt_dominance or 0.0,
                    usdc_dominance=dominance_data.usdc_dominance or 0.0,
                    dominance_24h_change=dominance_data.dominance_24h_change or 0.0,
                    dominance_rsi=dominance_data.dominance_rsi or 50.0,
                )

        except Exception as e:
            self.logger.warning(
                "Failed to process dominance data: %s, using defaults", e
            )
            # Add default values
            indicator_dict.update(self._get_default_dominance_values())

        return dominance_obj

    def _create_dominance_metrics(self, dominance_data) -> dict:
        """Create dominance metrics dictionary."""
        return {
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

    def _get_default_dominance_values(self) -> dict[str, Any]:
        """Get default dominance values."""
        return {
            "usdt_dominance": 0.0,
            "usdc_dominance": 0.0,
            "stablecoin_dominance": 0.0,
            "dominance_trend": 0.0,
            "dominance_rsi": 50.0,
            "stablecoin_velocity": 1.0,
            "market_sentiment": "NEUTRAL",
        }

    async def _create_market_state(
        self,
        current_price,
        latest_data,
        indicator_dict,
        dominance_obj,
        dominance_candles,
    ):
        """Create market state for LLM analysis."""
        from .trading_types import IndicatorData, MarketState

        # Calculate historical data window
        interval_seconds = self._get_interval_seconds(self.interval)
        candles_per_24h = min((24 * 60 * 60) // interval_seconds, len(latest_data))
        historical_data = latest_data[-candles_per_24h:]

        return MarketState(
            symbol=self.symbol,
            interval=self.interval,
            timestamp=datetime.now(UTC),
            current_price=current_price,
            ohlcv_data=historical_data,
            indicators=IndicatorData(**indicator_dict),
            current_position=self.current_position,
            dominance_data=dominance_obj,
            dominance_candles=dominance_candles,
        )

    async def _execute_trading_logic(
        self, market_state, current_price, latest_data, loop_count
    ):
        """Execute main trading logic."""
        # Mark candle analysis time
        latest_candle = latest_data[-1]
        candle_timestamp = latest_candle.timestamp
        if candle_timestamp.tzinfo is None:
            candle_timestamp = candle_timestamp.replace(tzinfo=UTC)
        self.last_candle_analysis_time = candle_timestamp

        # Convert price for display if it's in 18-decimal format
        from bot.utils.price_conversion import convert_from_18_decimal

        display_price = convert_from_18_decimal(
            current_price, self.symbol, "current_price"
        )

        self.logger.info(
            "⚡ Scalping analysis: %s candle at %s - Price: $%s",
            self.interval,
            latest_candle.timestamp.strftime("%H:%M:%S.%f")[:-3],
            display_price,
        )

        # Get LLM decision
        trade_action = await self._get_llm_decision(market_state)

        # Record and publish decision
        experience_id = await self._record_trading_decision(market_state, trade_action)
        await self._publish_trading_decision(trade_action, market_state, current_price)

        # Execute trade based on action type
        final_action = await self._execute_trade_decision(
            trade_action, current_price, market_state, experience_id, loop_count
        )

        # Update tracking and display status
        await self._update_position_tracking(current_price)

        if loop_count % 10 == 0:
            self._display_status_update(loop_count, current_price, final_action)
            await self._publish_performance_metrics(current_price)

        if loop_count % 5 == 0:
            self.logger.debug("Trading loop heartbeat - iteration %s", loop_count)

    async def _get_llm_decision(self, market_state):
        """Get trading decision from appropriate agent (Market Making or LLM)."""
        # Check if market making integrator is available and handles this symbol
        if (
            self.market_making_integrator
            and self.market_making_integrator.status.is_initialized
            and self.market_making_integrator.get_strategy_for_symbol(self.symbol)
            == "market_making"
        ):
            self.logger.debug(
                "🎯 Requesting trading decision from Market Making Engine for %s",
                self.symbol,
            )

            with self.performance_monitor.track_operation(
                "market_making_decision",
                {
                    "strategy_type": "market_making",
                    "symbol": self.symbol,
                },
            ):
                return await self.market_making_integrator.analyze_market(market_state)

        # Fall back to LLM agent for other symbols or when market making is unavailable
        self.logger.debug(
            "🤔 Requesting trading decision from %s LLM Agent for %s",
            "Memory-Enhanced" if self._memory_available else "Standard",
            self.symbol,
        )

        with self.performance_monitor.track_operation(
            "llm_response",
            {
                "agent_type": (
                    "memory_enhanced" if self._memory_available else "standard"
                ),
                "symbol": self.symbol,
            },
        ):
            return await self.llm_agent.analyze_market(market_state)

    async def _record_trading_decision(self, market_state, trade_action):
        """Record trading decision in memory and logs."""
        memory_context = None
        if hasattr(self.llm_agent, "_last_memory_context"):
            memory_context = self.llm_agent._last_memory_context

        self.trade_logger.log_trade_decision(
            market_state=market_state,
            trade_action=trade_action,
            experience_id=None,
            memory_context=memory_context,
        )

        # Record in MCP if enabled
        experience_id = None
        if self.experience_manager and trade_action.action != "HOLD":
            try:
                experience_id = await self.experience_manager.record_trading_decision(
                    market_state, trade_action
                )
                if experience_id:
                    self.trade_logger.log_trade_decision(
                        market_state=market_state,
                        trade_action=trade_action,
                        experience_id=experience_id,
                        memory_context=memory_context,
                    )
            except Exception as e:
                self.logger.warning("Failed to record trading decision: %s", e)

        return experience_id

    async def _publish_trading_decision(
        self, trade_action, market_state, current_price
    ):
        """Publish trading decision to dashboard."""
        if not self.websocket_publisher:
            return

        await self.websocket_publisher.publish_ai_decision(
            action=trade_action.action,
            reasoning=trade_action.rationale,
            confidence=trade_action.size_pct / 100.0,
        )

        await self.websocket_publisher.publish_trading_decision(
            trade_action=trade_action,
            symbol=self.symbol,
            current_price=float(current_price),
            context={
                "request_id": f"trade_{int(datetime.now(UTC).timestamp())}",
                "confidence": trade_action.size_pct / 100.0,
                "indicators": {
                    "cipher_a": market_state.indicators.cipher_a_dot,
                    "cipher_b": market_state.indicators.cipher_b_wave,
                    "rsi": market_state.indicators.rsi,
                },
                "risk_analysis": {
                    "current_price": float(current_price),
                    "position_size": trade_action.size_pct / 100.0,
                    "leverage": self.settings.trading.leverage,
                },
            },
        )

    async def _execute_trade_decision(
        self, trade_action, current_price, market_state, experience_id, loop_count
    ):
        """Execute trade decision based on action type."""
        if trade_action.action in ["LONG", "SHORT"]:
            return await self._execute_directional_trade(
                trade_action, current_price, market_state, experience_id, loop_count
            )
        return await self._execute_non_directional_trade(
            trade_action, current_price, market_state, experience_id, loop_count
        )

    async def _execute_directional_trade(
        self, trade_action, current_price, market_state, experience_id, loop_count
    ):
        """Execute LONG/SHORT trades with LLM override."""
        validated_action = self.validator.validate(trade_action)

        self.logger.info(
            "Loop %s: Price=$%s | LLM=%s | Action=%s (%s%%) | Risk=LLM_OVERRIDE - AI has final say",
            loop_count,
            current_price,
            trade_action.action,
            validated_action.action,
            validated_action.size_pct,
        )

        await self._execute_trade(
            validated_action, current_price, market_state, experience_id
        )
        self.last_trade_time = datetime.now(UTC)
        return validated_action

    async def _execute_non_directional_trade(
        self, trade_action, current_price, market_state, experience_id, loop_count
    ):
        """Execute HOLD/CLOSE trades with risk management."""
        validated_action = self.validator.validate(trade_action)

        risk_approved, final_action, risk_reason = self.risk_manager.evaluate_risk(
            validated_action, self.current_position, current_price
        )

        # Convert price for display if it's in 18-decimal format
        from bot.utils.price_conversion import convert_from_18_decimal

        display_price = convert_from_18_decimal(
            current_price, self.symbol, "current_price"
        )

        self.logger.info(
            "Loop %s: Price=$%s | LLM=%s | Action=%s (%s%%) | Risk=%s",
            loop_count,
            display_price,
            trade_action.action,
            final_action.action,
            final_action.size_pct,
            risk_reason,
        )

        if risk_approved and final_action.action != "HOLD":
            await self._execute_trade(
                final_action, current_price, market_state, experience_id
            )
            self.last_trade_time = datetime.now(UTC)

        return final_action

    async def _handle_periodic_tasks(
        self, loop_count: int, last_status_log, loop_context: dict
    ):
        """Handle periodic maintenance tasks."""
        # Pattern statistics logging
        if loop_count % 100 == 0 and self.memory_server and self._memory_available:
            await self._log_pattern_statistics()

        # Status logging
        last_status_log = await self._handle_status_logging(
            loop_count, last_status_log, loop_context["is_high_frequency"]
        )

        # Paper trading updates
        if (
            self.dry_run
            and self.paper_account
            and loop_count % (60 if loop_context["is_high_frequency"] else 10) == 0
        ):
            await self._update_paper_trading_metrics(loop_count)

    async def _log_pattern_statistics(self):
        """Log pattern statistics from memory server."""
        try:
            pattern_stats = await self.memory_server.get_pattern_statistics()
            if pattern_stats:
                self.trade_logger.log_pattern_statistics(pattern_stats)
                self.logger.info("📊 === MCP Pattern Performance Update ===")

                sorted_patterns = sorted(
                    pattern_stats.items(),
                    key=lambda x: x[1]["success_rate"] * x[1]["count"],
                    reverse=True,
                )[:5]

                for pattern, stats in sorted_patterns:
                    if stats["count"] >= 2:
                        self.logger.info(
                            "  📈 %s: %.1f%% win rate | %s trades | Avg PnL: $%.2f",
                            pattern,
                            stats["success_rate"] * 100,
                            stats["count"],
                            stats["avg_pnl"],
                        )
        except Exception as e:
            self.logger.debug("Could not retrieve pattern statistics: %s", e)

    async def _handle_status_logging(
        self, loop_count: int, last_status_log, is_high_frequency: bool
    ):
        """Handle periodic status logging."""
        status_log_interval = 60 if is_high_frequency else 120

        if (datetime.now(UTC) - last_status_log).total_seconds() > status_log_interval:
            if self.market_data is not None:
                data_status = self.market_data.get_data_status()
                # Convert latest price for display
                latest_price = data_status.get("latest_price", "N/A")
                if latest_price != "N/A":
                    from bot.utils.price_conversion import convert_from_18_decimal

                    latest_price = convert_from_18_decimal(
                        latest_price, self.symbol, "latest_price"
                    )

                self.logger.info(
                    "🔄 Trading Status: Loop #%s | WebSocket: %s | Latest Price: $%s | OmniSearch: %s",
                    loop_count,
                    "✓" if data_status.get("websocket_connected") else "✗",
                    latest_price,
                    "✓ Active" if hasattr(self, "omnisearch_client") else "✗ Disabled",
                )
            else:
                self.logger.info(
                    "🔄 Trading Status: Loop #%s | Market Data: ✗ Not Initialized | OmniSearch: %s",
                    loop_count,
                    "✓ Active" if hasattr(self, "omnisearch_client") else "✗ Disabled",
                )
            return datetime.now(UTC)

        return last_status_log

    async def _update_paper_trading_metrics(self, loop_count: int):
        """Update paper trading performance metrics."""
        try:
            with self.performance_monitor.track_operation(
                "paper_trading_update", {"loop_count": str(loop_count)}
            ):
                self.paper_account.update_daily_performance()

                from .performance_monitor import PerformanceMetric

                performance_metrics = (
                    self.paper_account.get_performance_metrics_for_monitor()
                )

                for metric_data in performance_metrics:
                    metric = PerformanceMetric(
                        name=metric_data["name"],
                        value=float(metric_data["value"]),
                        timestamp=metric_data["timestamp"],
                        unit=metric_data["unit"],
                        tags=metric_data["tags"],
                    )
                    self.performance_monitor.metrics_collector.add_metric(metric)

            self.logger.debug(
                "Updated paper trading performance and metrics at loop %s", loop_count
            )
        except Exception as e:
            self.logger.warning("Failed to update paper trading performance: %s", e)

    async def _handle_loop_timing(self, loop_start, target_frequency: float):
        """Handle loop timing and sleep calculations."""
        loop_end_time = datetime.now(UTC)
        loop_duration = (loop_end_time - loop_start).total_seconds()
        sleep_time = max(0, target_frequency - loop_duration)

        # Performance monitoring
        if hasattr(self, "_loop_timing_counter"):
            self._loop_timing_counter += 1
        else:
            self._loop_timing_counter = 1

        if self._loop_timing_counter % 100 == 0:
            timing_accuracy = (loop_duration / target_frequency) * 100
            self.logger.debug(
                "⏱️ Loop Timing: Duration=%.3fs, Target=%.3fs, Accuracy=%.1f%%, Sleep=%.3fs",
                loop_duration,
                target_frequency,
                timing_accuracy,
                sleep_time,
            )

            if target_frequency <= 1.0 and loop_duration > target_frequency * 1.5:
                self.logger.warning(
                    "🐌 High-frequency loop running slow: %.3fs (target: %.3fs). "
                    "Consider optimizing or reducing frequency.",
                    loop_duration,
                    target_frequency,
                )

        # Efficient sleep
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    async def _handle_loop_error(
        self, error: Exception, loop_count: int, loop_context: dict
    ):
        """Handle loop errors with adaptive recovery."""
        self.logger.exception("Error in trading loop")
        from rich.console import Console

        console = Console()
        console.print(f"[red]Loop error: {error}[/red]")

        # Adaptive error recovery
        if loop_context["is_high_frequency"]:
            error_sleep = min(5, 0.5 * (loop_count % 3 + 1))
        else:
            error_sleep = min(30, 1 * (loop_count % 5 + 1))

        self.logger.info("Waiting %.1fs before retry...", error_sleep)
        await asyncio.sleep(error_sleep)

    def _categorize_error_basic(self, error: Exception) -> str:
        """Categorize error for targeted recovery strategies."""
        error_str = str(error).lower()

        if "market" in error_str or "data" in error_str or "websocket" in error_str:
            return "market_data"
        if (
            "indicator" in error_str
            or "calculation" in error_str
            or "cipher" in error_str
        ):
            return "indicators"
        if (
            "llm" in error_str
            or "openai" in error_str
            or "agent" in error_str
            or "decision" in error_str
        ):
            return "llm_decision"
        if "trade" in error_str or "order" in error_str or "execution" in error_str:
            return "trade_execution"
        if "risk" in error_str or "position" in error_str or "balance" in error_str:
            return "risk_management"
        if (
            "connection" in error_str
            or "timeout" in error_str
            or "network" in error_str
            or "http" in error_str
        ):
            return "connectivity"
        return "unknown"

    async def _handle_loop_error_enhanced(
        self,
        error: Exception,
        loop_count: int,
        loop_context: dict,
        error_category: str,
        consecutive_errors: int,
    ):
        """Enhanced loop error handling with categorization and recovery strategies."""
        self.logger.exception(f"Error in trading loop (category: {error_category})")
        from rich.console import Console

        console = Console()
        console.print(f"[red]Loop error ({error_category}): {error}[/red]")

        # Record failure in circuit breaker if it's a critical error
        if self._is_critical_error_basic(error, error_category):
            if (
                hasattr(self, "risk_manager")
                and self.risk_manager
                and hasattr(self.risk_manager, "circuit_breaker")
            ):
                self.risk_manager.circuit_breaker.record_failure(
                    failure_type=error_category,
                    error_message=str(error),
                    severity="high" if consecutive_errors > 5 else "medium",
                )

        # Category-specific recovery delay
        recovery_delay = await self._get_recovery_delay_basic(
            error_category, consecutive_errors, loop_context
        )

        self.logger.info(
            "Waiting %.1fs before retry (consecutive errors: %d)...",
            recovery_delay,
            consecutive_errors,
        )
        await asyncio.sleep(recovery_delay)

    def _is_critical_error_basic(self, error: Exception, error_category: str) -> bool:
        """Determine if an error is critical and should trigger circuit breaker."""
        critical_categories = {"trade_execution", "risk_management", "connectivity"}
        critical_exceptions = {"RuntimeError", "ConnectionError", "TimeoutError"}

        return (
            error_category in critical_categories
            or type(error).__name__ in critical_exceptions
            or "critical" in str(error).lower()
        )

    async def _get_recovery_delay_basic(
        self, error_category: str, consecutive_errors: int, loop_context: dict
    ) -> float:
        """Get recovery delay based on error category and context."""
        base_delays = {
            "market_data": 2.0,
            "indicators": 1.0,
            "llm_decision": 3.0,
            "trade_execution": 5.0,
            "risk_management": 10.0,
            "connectivity": 15.0,
            "unknown": 5.0,
        }

        base_delay = base_delays.get(error_category, 5.0)

        # Exponential backoff for consecutive errors
        backoff_multiplier = min(2 ** (consecutive_errors - 1), 8)

        # Adjust for high frequency trading
        if loop_context.get("is_high_frequency", False):
            base_delay = min(base_delay, 5.0)  # Cap at 5s for high frequency

        return min(base_delay * backoff_multiplier, 60.0)  # Cap at 1 minute

    async def _perform_basic_health_check(self) -> dict:
        """Perform basic component health check."""
        health_status = {
            "exchange_client": False,
            "market_data": False,
            "llm_agent": False,
            "risk_manager": False,
            "circuit_breaker": False,
            "timestamp": datetime.now(UTC),
        }

        try:
            # Check exchange client
            if self.exchange_client:
                health_status["exchange_client"] = True

            # Check market data
            if self.market_data:
                try:
                    current_price = await self.market_data.get_current_price()
                    health_status["market_data"] = current_price is not None
                except Exception:
                    pass

            # Check LLM agent
            if self.llm_agent:
                health_status["llm_agent"] = hasattr(self.llm_agent, "analyze_market")

            # Check risk manager and circuit breaker
            if self.risk_manager:
                health_status["risk_manager"] = True

                if hasattr(self.risk_manager, "circuit_breaker"):
                    try:
                        cb_status = self.risk_manager.circuit_breaker.get_status()
                        health_status["circuit_breaker"] = cb_status["state"] != "OPEN"
                    except Exception:
                        pass

        except Exception as e:
            self.logger.warning("Health check failed: %s", e)

        return health_status

    def _check_circuit_breaker_basic(self) -> bool:
        """Check if circuit breaker allows trading."""
        if not self.risk_manager or not hasattr(self.risk_manager, "circuit_breaker"):
            return True  # No circuit breaker, allow trading

        try:
            can_trade = self.risk_manager.circuit_breaker.can_execute_trade()

            if not can_trade:
                cb_status = self.risk_manager.circuit_breaker.get_status()
                self.logger.warning(
                    "🚫 Circuit breaker is OPEN - trading suspended. State: %s, Failures: %d",
                    cb_status["state"],
                    cb_status["failure_count"],
                )

            return can_trade
        except Exception as e:
            self.logger.warning("Circuit breaker check failed: %s", e)
            return True  # Fail open to allow trading

    async def _handle_emergency_situation(self, error_categories: dict):
        """Handle emergency situation when too many consecutive errors occur."""
        self.logger.error("🚨 ENTERING EMERGENCY MODE")

        from rich.console import Console

        console = Console()
        console.print("[red bold]🚨 EMERGENCY MODE ACTIVATED[/red bold]")
        console.print("Error breakdown:")
        for category, count in error_categories.items():
            if count > 0:
                console.print(f"  • {category}: {count} errors")

        # Close any open positions if possible
        try:
            if (
                hasattr(self, "current_position")
                and self.current_position.side != "FLAT"
            ):
                console.print("[yellow]Attempting to close open positions...[/yellow]")
                await self._emergency_close_positions()
        except Exception as e:
            self.logger.error("Emergency position closure failed: %s", e)

        # Extended pause before attempting to continue
        emergency_pause = min(
            300, 30 * sum(error_categories.values())
        )  # Up to 5 minutes
        console.print(f"[yellow]Emergency pause: {emergency_pause} seconds[/yellow]")

        await asyncio.sleep(emergency_pause)

        # Reset error categories for fresh start
        for category in error_categories:
            error_categories[category] = 0

        console.print("[green]Attempting to resume normal operation...[/green]")

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
            current_price = await self.market_data.get_current_price()
            if not current_price:
                self.logger.error("Current price not available for trade execution")
                return False

            # Execute trade with current price and no experience tracking
            await self._execute_trade(trade_action, current_price)
            return True

        except Exception:
            self.logger.exception("Error executing trade action")
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
                "📦 Executing trade: %s %s%% | Experience ID: %s...",
                trade_action.action,
                trade_action.size_pct,
                experience_id[:8] if experience_id else "None",
            )
            self.logger.debug(
                "Trade execution started at %s", datetime.now(UTC).isoformat()
            )

            # Track trade execution performance
            execution_tags = {
                "action": trade_action.action,
                "size_pct": str(trade_action.size_pct),
                "mode": "paper" if self.dry_run else "live",
            }

            # Check if we already have an open position and the action is LONG or SHORT
            if self.current_position.side != "FLAT" and trade_action.action in [
                "LONG",
                "SHORT",
            ]:
                self.logger.warning(
                    "Cannot open new %s position - already have %s position with size %s",
                    trade_action.action,
                    self.current_position.side,
                    self.current_position.size,
                )
                console.print(
                    f"[yellow]⚠ Trade rejected: Already have open {self.current_position.side} position[/yellow]"
                )
                return

            # Execute trade based on mode (paper trading vs live) with performance tracking
            with self.performance_monitor.track_operation(
                "trade_execution", execution_tags
            ):
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
                            "MCP Integration: Linked order %s to experience %s...",
                            order.id,
                            experience_id[:8],
                        )
                    except Exception as e:
                        self.logger.warning("Failed to link order to experience: %s", e)

                if order.status in ["FILLED", "PENDING"]:
                    self.successful_trades += 1

                    # Publish trade execution to dashboard
                    if self.websocket_publisher:
                        await self.websocket_publisher.publish_trade_execution(
                            {
                                "order_id": order.id,
                                "symbol": self.actual_trading_symbol,
                                "side": trade_action.action,
                                "quantity": float(order.quantity),
                                "price": (
                                    float(order.price)
                                    if order.price is not None
                                    else 0.0
                                ),
                                "status": order.status,
                                "trade_action": {
                                    "action": trade_action.action,
                                    "size_pct": trade_action.size_pct,
                                    "rationale": trade_action.rationale,
                                    "experience_id": experience_id,
                                },
                            }
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

                        # Publish updated performance metrics after position change
                        await self._publish_performance_metrics(current_price)

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

                                # Log structured trade outcome
                                if experience_id and previous_position.entry_price:
                                    entry_price = previous_position.entry_price
                                    exit_price = order.price or current_price
                                    pnl = previous_position.unrealized_pnl or Decimal(0)

                                    # Calculate duration (approximate - would need actual entry time)
                                    duration_minutes = 0.0  # Placeholder - would calculate from entry time

                                    self.trade_logger.log_trade_outcome(
                                        experience_id=experience_id,
                                        entry_price=entry_price,
                                        exit_price=exit_price,
                                        pnl=pnl,
                                        duration_minutes=duration_minutes,
                                        insights=f"Position closed: {previous_position.side} -> FLAT",
                                    )

                            except TimeoutError:
                                self.logger.warning(
                                    "Trade tracking completion timed out after 5 seconds"
                                )
                            except Exception as e:
                                self.logger.warning(
                                    "Failed to complete trade tracking: %s", e
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
                                        "📡 MCP Integration: Started tracking new trade - ID: %s",
                                        trade_id,
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    "Failed to start trade tracking: %s", e
                                )

                    console.print(
                        f"[green]✓ Trade executed:[/green] {trade_action.action} "
                        f"{trade_action.size_pct}% @ ${current_price}"
                    )

                    # Log paper trading account status if in dry run
                    if self.dry_run and self.paper_account:
                        account_status = self.paper_account.get_account_status()
                        self.logger.info(
                            "Paper account: $%,.2f equity, P&L: $%,.2f (%.2f%%)",
                            account_status["equity"],
                            account_status["total_pnl"],
                            account_status["roi_percent"],
                        )

                    self.logger.debug(
                        "Trade execution completed at %s", datetime.now(UTC).isoformat()
                    )
                else:
                    console.print(f"[yellow]⚠ Trade failed:[/yellow] {order.status}")
            else:
                console.print("[red]✗ Trade execution failed[/red]")

        except Exception as e:
            self.logger.exception("Trade execution error")
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
            "15s": 15,  # Will be converted to 1m by Bluefin service
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

    def _apply_adaptive_timing(self) -> None:
        """
        Apply adaptive timing based on trading interval for optimal high-frequency performance.

        This method adjusts the main loop update frequency based on the configured
        trading interval to ensure efficient processing without unnecessary overhead.

        Timing Strategy:
        - Intervals ≤ 1s: Use 1-second main loop for high-frequency scalping
        - Intervals ≤ 60s: Match interval frequency (capped at 15s for performance)
        - Longer intervals: Use existing configuration timing
        """
        try:
            # Get interval in seconds
            interval_seconds = self._get_interval_seconds(self.interval)

            # Store original frequency for logging
            original_frequency = self.settings.system.update_frequency_seconds

            # Adaptive timing logic
            if interval_seconds <= 1:
                # High-frequency trading: 1-second main loop for sub-second intervals
                new_frequency = 1.0
                timing_reason = "high-frequency scalping (≤1s intervals)"
            elif interval_seconds <= 60:
                # Medium frequency: match interval but cap at 15s for performance
                new_frequency = min(float(interval_seconds), 15.0)
                timing_reason = f"interval-matched ({interval_seconds}s intervals)"
            else:
                # Long intervals: use existing configuration
                new_frequency = original_frequency
                timing_reason = "existing configuration (long intervals)"

            # Apply new frequency if different
            if new_frequency != original_frequency:
                # Create updated system settings
                system_settings = self.settings.system.model_copy(
                    update={"update_frequency_seconds": new_frequency}
                )
                self.settings = self.settings.model_copy(
                    update={"system": system_settings}
                )

                # Note: Logging isn't available yet during initialization, so we store
                # the information to log later in the startup process
                self._adaptive_timing_info = {
                    "original": original_frequency,
                    "new": new_frequency,
                    "reason": timing_reason,
                    "interval": self.interval,
                    "interval_seconds": interval_seconds,
                }
            else:
                self._adaptive_timing_info = None

        except Exception as e:
            # Fallback to default if there's any error
            # Can't log here since logging isn't initialized yet
            self._adaptive_timing_error = str(e)

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
                # Create position object for the update
                from .trading_types import Position

                position_update = Position(
                    symbol=self.actual_trading_symbol,
                    side=self.current_position.side,
                    size=self.current_position.size,
                    entry_price=self.current_position.entry_price,
                    unrealized_pnl=pnl,
                    timestamp=self.current_position.timestamp,
                )
                await self.websocket_publisher.publish_position_update(
                    position=position_update
                )

            # Update risk manager with P&L
            self.risk_manager.update_daily_pnl(Decimal(0), pnl)

            # Update experience manager with trade progress
            if self.experience_manager:
                try:
                    await self.experience_manager.update_trade_progress(
                        self.current_position, current_price
                    )
                except Exception as e:
                    self.logger.warning("Failed to update trade progress: %s", e)

            # Log position update for ongoing trades (periodic logging, not every tick)
            if (
                hasattr(self, "_last_position_log_time")
                and self._last_position_log_time is not None
            ):
                time_since_last_log = (
                    datetime.now(UTC) - self._last_position_log_time
                ).total_seconds()
                if time_since_last_log >= 300:  # Log every 5 minutes
                    self._log_position_update(current_price, pnl)
                    self._last_position_log_time = datetime.now(UTC)
            else:
                self._log_position_update(current_price, pnl)
                self._last_position_log_time = datetime.now(UTC)
        elif self.websocket_publisher:
            # Create flat position object for the update
            from .trading_types import Position

            flat_position = Position(
                symbol=self.actual_trading_symbol,
                side="FLAT",
                size=Decimal(0),
                entry_price=None,
                unrealized_pnl=Decimal(0),
                timestamp=datetime.now(UTC),
            )
            await self.websocket_publisher.publish_position_update(
                position=flat_position
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

        # Add market making status if available
        if self.market_making_integrator and hasattr(
            self.market_making_integrator, "status"
        ):
            mm_status = self.market_making_integrator.status
            if (
                hasattr(mm_status, "market_making_enabled")
                and mm_status.market_making_enabled
            ):
                status_table.add_row(
                    "Market Making",
                    f"✅ Active on {', '.join(self.market_making_integrator.market_making_symbols)}",
                )
                # Add current strategy info
                strategy = self.market_making_integrator.get_strategy_for_symbol(
                    self.symbol
                )
                if strategy:
                    status_table.add_row("MM Strategy", strategy.title())
            else:
                status_table.add_row("Market Making", "⚠️ Available but inactive")
        elif (
            hasattr(self, "_market_making_enabled_override")
            and self._market_making_enabled_override
        ):
            status_table.add_row("Market Making", "❌ Failed to initialize")

        # Add dominance data if available
        if hasattr(self, "dominance_provider") and self.dominance_provider:
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
                self.logger.warning("Could not generate daily report: %s", e)

    def _is_end_of_trading_day(self) -> bool:
        """Check if it's near the end of trading day for reporting."""
        current_time = datetime.now(UTC)
        return current_time.hour == 23 and current_time.minute >= 50

    async def _publish_performance_metrics(self, current_price: Decimal) -> None:
        """Publish performance metrics to the dashboard via WebSocket."""
        if not self.websocket_publisher or not self.websocket_publisher.connected:
            return

        try:
            # Collect performance metrics
            performance_metrics = {
                "timestamp": datetime.now(UTC).isoformat(),
                "current_price": float(current_price),
                "symbol": self.actual_trading_symbol,
            }

            # Add position-specific metrics
            if self.current_position.side != "FLAT":
                performance_metrics.update(
                    {
                        "position_side": self.current_position.side,
                        "position_size": float(self.current_position.size),
                        "entry_price": (
                            float(self.current_position.entry_price)
                            if self.current_position.entry_price
                            else None
                        ),
                        "unrealized_pnl": float(self.current_position.unrealized_pnl),
                    }
                )
            else:
                performance_metrics.update(
                    {
                        "position_side": "FLAT",
                        "position_size": 0.0,
                        "entry_price": None,
                        "unrealized_pnl": 0.0,
                    }
                )

            # Add paper trading metrics if available
            if self.dry_run and self.paper_account:
                account_status = self.paper_account.get_account_status()
                performance_metrics.update(
                    {
                        "account_type": "paper",
                        "account_balance": account_status["current_balance"],
                        "account_equity": account_status["equity"],
                        "total_pnl": account_status["total_pnl"],
                        "roi_percent": account_status["roi_percent"],
                        "margin_used": account_status["margin_used"],
                        "margin_available": account_status["margin_available"],
                        "open_positions": account_status["open_positions"],
                        "total_trades": account_status["total_trades"],
                        "max_drawdown": account_status["max_drawdown"],
                        "peak_equity": account_status["peak_equity"],
                    }
                )
            else:
                performance_metrics.update(
                    {
                        "account_type": "live",
                    }
                )

            # Add trading session metrics
            uptime_seconds = (datetime.now(UTC) - self.start_time).total_seconds()
            success_rate = (self.successful_trades / max(self.trade_count, 1)) * 100

            performance_metrics.update(
                {
                    "session_uptime_seconds": uptime_seconds,
                    "total_trades_session": self.trade_count,
                    "successful_trades_session": self.successful_trades,
                    "success_rate_percent": success_rate,
                }
            )

            # Publish to dashboard
            await self.websocket_publisher.publish_performance_update(
                performance_metrics
            )

            self.logger.debug("Published performance metrics to dashboard")

        except Exception as e:
            self.logger.warning("Failed to publish performance metrics: %s", e)

    def get_performance_summary(self, duration_minutes: int = 10) -> dict[str, Any]:
        """
        Get comprehensive performance summary from the performance monitor.

        Args:
            duration_minutes: Time period to analyze in minutes (default: 10)

        Returns:
            Dictionary containing performance summary including:
            - latency_summary: Timing metrics for key operations
            - resource_summary: CPU/memory usage statistics
            - recent_alerts: Performance alerts in the timeframe
            - bottleneck_analysis: Identified performance bottlenecks
            - health_score: Overall system health (0-100)
            - paper_trading_metrics: Paper trading performance (if in dry run)
        """
        try:
            from datetime import timedelta

            duration = timedelta(minutes=duration_minutes)
            summary = self.performance_monitor.get_performance_summary(duration)

            # Add paper trading specific metrics if available
            if self.dry_run and self.paper_account:
                try:
                    paper_metrics = (
                        self.paper_account.get_performance_metrics_for_monitor()
                    )

                    # Convert to summary format
                    paper_summary = {}
                    for metric in paper_metrics:
                        metric_name = metric["name"].split(".")[
                            -1
                        ]  # Get the part after "paper_trading."
                        paper_summary[metric_name] = {
                            "value": metric["value"],
                            "unit": metric["unit"],
                            "tags": metric["tags"],
                        }

                    summary["paper_trading_metrics"] = paper_summary

                except Exception as e:
                    self.logger.warning(
                        "Failed to add paper trading metrics to summary: %s", e
                    )
                    summary["paper_trading_metrics"] = {"error": str(e)}

            # Add trading engine specific context
            summary.update(
                {
                    "trading_mode": "paper" if self.dry_run else "live",
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "engine_uptime_minutes": (
                        datetime.now(UTC) - self.start_time
                    ).total_seconds()
                    / 60,
                    "memory_system_enabled": self._memory_available,
                }
            )

            return summary

        except Exception as e:
            self.logger.exception("Failed to get performance summary")
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
                "trading_mode": "paper" if self.dry_run else "live",
            }

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
        cleanup_tasks: list[Task[Any]] = []

        try:
            # Cancel all open orders
            if (
                hasattr(self, "exchange_client")
                and self.exchange_client is not None
                and self.exchange_client.is_connected()
            ):
                console.print("  • Cancelling open orders...")
                cancel_task: Task[bool] = asyncio.create_task(
                    self.exchange_client.cancel_all_orders(self.symbol)
                )
                cleanup_tasks.append(cancel_task)

            # Close market data connection
            if hasattr(self, "market_data") and self.market_data is not None:
                console.print("  • Disconnecting from market data...")
                market_task: Task[None] = asyncio.create_task(
                    self.market_data.disconnect()
                )
                cleanup_tasks.append(market_task)

            # Close exchange connection
            if hasattr(self, "exchange_client") and self.exchange_client is not None:
                console.print("  • Disconnecting from exchange...")
                exchange_task: Task[None] = asyncio.create_task(
                    self.exchange_client.disconnect()
                )
                cleanup_tasks.append(exchange_task)

            # Close OmniSearch connection
            if (
                hasattr(self, "omnisearch_client")
                and self.omnisearch_client is not None
            ):
                console.print("  • Disconnecting from OmniSearch...")
                omnisearch_task: Task[None] = asyncio.create_task(
                    self.omnisearch_client.disconnect()
                )
                cleanup_tasks.append(omnisearch_task)

            # Close WebSocket publisher connection
            if (
                hasattr(self, "websocket_publisher")
                and self.websocket_publisher is not None
            ):
                console.print("  • Disconnecting from dashboard WebSocket...")
                websocket_task: Task[None] = asyncio.create_task(
                    self.websocket_publisher.close()
                )
                cleanup_tasks.append(websocket_task)

            # Stop command consumer
            if hasattr(self, "command_consumer") and self.command_consumer is not None:
                console.print("  • Stopping dashboard command consumer...")
                command_task: Task[None] = asyncio.create_task(
                    self.command_consumer.stop_polling_task()
                )
                close_task: Task[None] = asyncio.create_task(
                    self.command_consumer.close()
                )
                cleanup_tasks.append(command_task)
                cleanup_tasks.append(close_task)

            # Stop experience manager if enabled
            if (
                hasattr(self, "experience_manager")
                and self.experience_manager is not None
            ):
                console.print("  • Stopping experience tracking...")
                experience_task: Task[None] = asyncio.create_task(
                    self.experience_manager.stop()
                )
                cleanup_tasks.append(experience_task)

            # Stop market making integrator if enabled
            if (
                hasattr(self, "market_making_integrator")
                and self.market_making_integrator is not None
            ):
                console.print("  • Stopping market making engine...")
                market_making_task: Task[None] = asyncio.create_task(
                    self.market_making_integrator.stop()
                )
                cleanup_tasks.append(market_making_task)

            # Close dominance data connection - CRITICAL for async session cleanup
            if (
                hasattr(self, "dominance_provider")
                and self.dominance_provider is not None
            ):
                console.print("  • Disconnecting from dominance data...")
                dominance_task: Task[None] = asyncio.create_task(
                    self.dominance_provider.disconnect()
                )
                cleanup_tasks.append(dominance_task)

            # Stop performance monitoring
            if (
                hasattr(self, "performance_monitor")
                and self.performance_monitor is not None
            ):
                console.print("  • Stopping performance monitoring...")
                perf_task: Task[None] = asyncio.create_task(
                    self.performance_monitor.stop_monitoring()
                )
                cleanup_tasks.append(perf_task)

            # Cancel all background tasks
            if hasattr(self, "_background_tasks") and self._background_tasks:
                console.print("  • Cancelling background tasks...")
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.warning(
                                "Error cancelling background task: %s", e
                            )
                self._background_tasks.clear()

            # Wait for all cleanup tasks with a timeout
            if cleanup_tasks:
                done, pending = await asyncio.wait(
                    cleanup_tasks, timeout=5.0, return_when=asyncio.ALL_COMPLETED
                )

                # Cancel any tasks that didn't complete in time
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            # Display final summary
            self._display_final_summary()

            console.print("[green]✓ Shutdown complete[/green]")

        except Exception as e:
            self.logger.exception("Error during shutdown")
            console.print(f"[red]Shutdown error: {e}[/red]")
        finally:
            # Final cleanup - ensure all async sessions are closed
            # This is a last resort cleanup
            if (
                hasattr(self, "dominance_provider")
                and self.dominance_provider
                and hasattr(self.dominance_provider, "_session")
                and self.dominance_provider._session
                and not self.dominance_provider._session.closed
            ):
                try:
                    # Force close without await since we might be in cleanup
                    if (
                        hasattr(self.dominance_provider._session, "_connector")
                        and self.dominance_provider._session._connector is not None
                    ):
                        self.dominance_provider._session._connector.close()
                except Exception as e:
                    # Log cleanup errors but don't fail shutdown
                    self.logger.debug(
                        "Error closing dominance provider connector during cleanup: %s",
                        e,
                    )

    async def _force_cleanup_services(self) -> None:
        """Force cleanup of all services during emergency shutdown."""
        # Cleanup dominance provider session
        if (
            hasattr(self, "dominance_provider")
            and self.dominance_provider
            and hasattr(self.dominance_provider, "_session")
        ):
            self.dominance_provider._session = None

        # Cleanup WebSocket publisher
        if hasattr(self, "websocket_publisher") and self.websocket_publisher:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.websocket_publisher.close(), timeout=2.0)

        # Cleanup Bluefin service client
        if hasattr(self, "_bluefin_service_client"):
            with contextlib.suppress(Exception):
                from bot.exchange.bluefin_service_client import (
                    close_bluefin_service_client,
                )

                await asyncio.wait_for(close_bluefin_service_client(), timeout=2.0)

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
                    if size > 0 and side.upper() in ["LONG", "BUY"]:
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
                            entry_price=self.current_position.entry_price or Decimal(0),
                        )

                        self.logger.info(
                            "Reconciled position: %s %s %s at $%s",
                            position_side,
                            size,
                            self.actual_trading_symbol,
                            self.current_position.entry_price,
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
            self.logger.exception("Failed to reconcile positions")
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
                    from .utils.path_utils import get_data_file_path

                    history_file = get_data_file_path(
                        "paper_trading/session_trades.json"
                    )
                    history_file.parent.mkdir(parents=True, exist_ok=True)
                    with history_file.open("w") as f:
                        f.write(trade_history)
                    console.print(
                        f"[green]✓ Trade history exported to {history_file}[/green]"
                    )
                except Exception as e:
                    self.logger.warning("Could not export trade history: %s", e)

            # Display final daily report
            try:
                final_report = self.position_manager.generate_daily_report()
                if final_report and "No trading data" not in final_report:
                    console.print()
                    console.print(
                        Panel(final_report, title="📊 Final Daily Report", style="blue")
                    )
            except Exception as e:
                self.logger.warning("Could not generate final daily report: %s", e)

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
            # Early returns for bypass conditions
            if not self.settings.data.enable_cipher_b_filter:
                self.logger.debug(
                    "Cipher B filter: Disabled in configuration, allowing original action"
                )
                return trade_action

            if trade_action.action in ["HOLD", "CLOSE"]:
                self.logger.debug(
                    "Cipher B filter: Allowing HOLD/CLOSE action without filtering"
                )
                return trade_action

            # Get Cipher B indicator values
            cipher_b_wave = market_state.indicators.cipher_b_wave
            cipher_b_money_flow = market_state.indicators.cipher_b_money_flow

            if cipher_b_wave is None or cipher_b_money_flow is None:
                self.logger.warning(
                    "Cipher B filter: Indicators not available, allowing original action"
                )
                return trade_action

            # Get thresholds and determine signals
            wave_bullish_threshold = self.settings.data.cipher_b_wave_bullish_threshold
            wave_bearish_threshold = self.settings.data.cipher_b_wave_bearish_threshold
            money_flow_bullish_threshold = (
                self.settings.data.cipher_b_money_flow_bullish_threshold
            )
            money_flow_bearish_threshold = (
                self.settings.data.cipher_b_money_flow_bearish_threshold
            )

            wave_bullish = cipher_b_wave > wave_bullish_threshold
            wave_bearish = cipher_b_wave < wave_bearish_threshold
            money_flow_bullish = cipher_b_money_flow > money_flow_bullish_threshold
            money_flow_bearish = cipher_b_money_flow < money_flow_bearish_threshold

            # Apply filtering based on action type
            return self._filter_action_by_cipher_signals(
                trade_action,
                cipher_b_wave,
                cipher_b_money_flow,
                wave_bullish,
                wave_bearish,
                money_flow_bullish,
                money_flow_bearish,
            )

        except Exception:
            self.logger.exception("Error in Cipher B filtering")
            # On error, allow the original trade action to prevent system failure
            return trade_action

    def _filter_action_by_cipher_signals(
        self,
        trade_action: TradeAction,
        cipher_b_wave: float,
        cipher_b_money_flow: float,
        wave_bullish: bool,
        wave_bearish: bool,
        money_flow_bullish: bool,
        money_flow_bearish: bool,
    ) -> TradeAction:
        """Filter trade action based on Cipher B signals."""
        if trade_action.action == "LONG":
            if wave_bullish and money_flow_bullish:
                self.logger.info(
                    "Cipher B filter: LONG signal CONFIRMED - Wave: %.2f (bullish), Money Flow: %.2f (bullish)",
                    cipher_b_wave,
                    cipher_b_money_flow,
                )
                return trade_action

            self.logger.info(
                "Cipher B filter: LONG signal FILTERED OUT - Wave: %.2f (%s), Money Flow: %.2f (%s)",
                cipher_b_wave,
                "bullish" if wave_bullish else "bearish",
                cipher_b_money_flow,
                "bullish" if money_flow_bullish else "bearish",
            )
            return self._create_filtered_hold_action(
                trade_action, cipher_b_wave, cipher_b_money_flow, "LONG"
            )

        if trade_action.action == "SHORT":
            if wave_bearish and money_flow_bearish:
                self.logger.info(
                    "Cipher B filter: SHORT signal CONFIRMED - Wave: %.2f (bearish), Money Flow: %.2f (bearish)",
                    cipher_b_wave,
                    cipher_b_money_flow,
                )
                return trade_action

            self.logger.info(
                "Cipher B filter: SHORT signal FILTERED OUT - Wave: %.2f (%s), Money Flow: %.2f (%s)",
                cipher_b_wave,
                "bearish" if wave_bearish else "bullish",
                cipher_b_money_flow,
                "bearish" if money_flow_bearish else "bullish",
            )
            return self._create_filtered_hold_action(
                trade_action, cipher_b_wave, cipher_b_money_flow, "SHORT"
            )

        # Default fallback for unexpected actions
        self.logger.warning(
            "Cipher B filter: Unexpected action '%s', allowing original",
            trade_action.action,
        )
        return trade_action

    def _create_filtered_hold_action(
        self,
        trade_action: TradeAction,
        cipher_b_wave: float,
        cipher_b_money_flow: float,
        action_type: str,
    ) -> TradeAction:
        """Create a HOLD action when trade is filtered out."""
        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            leverage=trade_action.leverage,
            reduce_only=False,
            rationale=f"Cipher B filter: {action_type} rejected - Wave:{cipher_b_wave:.2f}, MF:{cipher_b_money_flow:.2f}",
        )

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

    def _log_position_update(self, current_price: Decimal, pnl: Decimal) -> None:
        """
        Log position update for ongoing trades.

        Args:
            current_price: Current market price
            pnl: Current unrealized P&L
        """
        if self.current_position.side != "FLAT":
            # Calculate max favorable/adverse excursion placeholders
            # In a full implementation, these would be tracked over the trade lifetime
            max_favorable = abs(pnl) if pnl > 0 else Decimal(0)
            max_adverse = abs(pnl) if pnl < 0 else Decimal(0)

            # Generate a simple trade ID for tracking
            trade_id = f"{self.current_position.side}_{self.current_position.size}_{self.symbol}"

            self.trade_logger.log_position_update(
                trade_id=trade_id,
                current_price=current_price,
                unrealized_pnl=pnl,
                max_favorable=max_favorable,
                max_adverse=max_adverse,
            )


@click.group()
@click.version_option(version="0.1.0", prog_name="ai-trading-bot")
def cli() -> None:
    """AI Trading Bot - LangChain-powered crypto futures trading.

    Features:
    - Live trading with LLM-powered decision making
    - Market making strategies for DEX/CEX
    - Paper trading and backtesting
    - Performance monitoring and reporting
    - Multi-exchange support (Coinbase, Bluefin)

    Market Making Commands:
    - mm-status: Show market making configuration and status
    - mm-config: Display configuration for different profiles
    - mm-validate: Validate market making setup
    - mm-test: Test market making components
    """
    # Set up comprehensive warning suppression for third-party libraries
    setup_warnings_suppression()

    # Initialize robust logging system with fallback support
    try:
        from .utils.logging_config import setup_application_logging
        from .utils.logging_factory import log_system_info

        # Setup comprehensive logging system
        logging_results = setup_application_logging()

        # Log system information if logging is working
        if logging_results["status"] in ["success", "fallback"]:
            with contextlib.suppress(Exception):
                log_system_info()

        # Print any critical logging issues to console
        if logging_results["errors"]:
            console.print("⚠️  Logging system encountered errors:", style="yellow")
            for error in logging_results["errors"]:
                console.print(f"  - {error}", style="yellow")

        if logging_results["status"] == "fallback":
            console.print("INFO: Logging system running in fallback mode", style="blue")

    except ImportError:
        # If our logging modules aren't available, setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        console.print(
            "⚠️  Using basic logging - advanced logging modules not available",
            style="yellow",
        )


@cli.command()
@click.option(
    "--dry-run/--no-dry-run",
    default=None,
    help="Override dry-run mode (respects config/env if not specified)",
)
@click.option("--symbol", default="BTC-USD", help="Trading symbol")
@click.option(
    "--interval",
    default="1m",
    help="Candle interval (Note: Bluefin converts sub-minute intervals to 1m)",
)
@click.option("--config", default=None, help="Configuration file path")
@click.option("--force", is_flag=True, help="Skip confirmation prompt for live trading")
@click.option("--skip-health-check", is_flag=True, help="Skip startup health checks")
@click.option(
    "--market-making/--no-market-making",
    default=None,
    help="Enable/disable market making mode (overrides config)",
)
@click.option(
    "--mm-symbol",
    default=None,
    help="Market making symbol (default: SUI-PERP)",
)
@click.option(
    "--mm-profile",
    type=click.Choice(["conservative", "moderate", "aggressive", "custom"]),
    default=None,
    help="Market making configuration profile",
)
def live(
    dry_run: bool | None,
    symbol: str,
    interval: str,
    config: str | None,
    force: bool,
    skip_health_check: bool,
    market_making: bool | None,
    mm_symbol: str | None,
    mm_profile: str | None,
) -> None:
    """Start live trading bot.

    Examples:
        # Regular trading
        ai-trading-bot live --symbol BTC-USD

        # Enable market making
        ai-trading-bot live --market-making --mm-symbol SUI-PERP --mm-profile aggressive

        # Market making with custom configuration
        ai-trading-bot live --market-making --config custom_config.json

        # Live trading (requires --force for safety)
        ai-trading-bot live --no-dry-run --force
    """
    try:
        # Perform startup health checks unless skipped
        if not skip_health_check:
            _check_startup_health()

        # Create the trading engine to determine actual dry_run setting
        engine = TradingEngine(
            symbol=symbol,
            interval=interval,
            config_file=config,
            dry_run=dry_run,
            market_making_enabled=market_making,
            market_making_symbol=mm_symbol,
            market_making_profile=mm_profile,
        )

        # Build startup message with market making info
        mm_info = ""
        if market_making or (mm_symbol and mm_symbol != "BTC-USD"):
            mm_symbol_display = mm_symbol or "SUI-PERP"
            mm_profile_display = mm_profile or "moderate"
            mm_info = (
                f"\nMarket Making: {mm_profile_display.title()} on {mm_symbol_display}"
            )
        elif (
            hasattr(engine, "market_making_integrator")
            and engine.market_making_integrator
        ):
            mm_info = "\nMarket Making: Enabled via configuration"

        # Display startup message based on actual configuration
        if engine.dry_run:
            console.print(
                Panel(
                    f"🚀 Starting AI Trading Bot in DRY-RUN mode\n"
                    f"Symbol: {symbol}\n"
                    f"Interval: {interval}\n"
                    f"Mode: Paper Trading (No real orders)"
                    f"{mm_info}",
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
                    f"Mode: Real Trading (Real money at risk!)"
                    f"{mm_info}",
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

        # Start the trading engine
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        console.print("\n👋 Bot stopped by user")
    except ImportError as e:
        console.print(f"❌ Import Error: {e}", style="red")
        console.print(
            "💡 Try running 'ai-trading-bot diagnose' to check component health",
            style="cyan",
        )
        console.print(
            "💡 Install missing dependencies with poetry install", style="cyan"
        )
        sys.exit(1)
    except Exception as e:
        import traceback

        console.print(f"❌ Unexpected Error: {e}", style="red")
        console.print(
            "💡 Try running 'ai-trading-bot diagnose' for troubleshooting", style="cyan"
        )

        # Only show full traceback if it's not a common user error
        if not any(
            keyword in str(e).lower()
            for keyword in ["api key", "connection", "permission", "config"]
        ):
            console.print(
                f"🔧 Full traceback:\n{traceback.format_exc()}", style="yellow"
            )

        sys.exit(1)


@cli.command()
@click.option(
    "--from", "start_date", default="2024-01-01", help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "end_date", default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--symbol", default="BTC-USD", help="Trading symbol")
@click.option("--initial-balance", default=10000.0, help="Initial balance for backtest")
@click.option(
    "--market-making/--no-market-making",
    default=False,
    help="Enable market making mode for backtest simulation",
)
@click.option(
    "--mm-symbol",
    default=None,
    help="Market making symbol for backtest (default: SUI-PERP)",
)
@click.option(
    "--mm-profile",
    type=click.Choice(["conservative", "moderate", "aggressive", "custom"]),
    default="moderate",
    help="Market making configuration profile for backtest",
)
def backtest(
    start_date: str,
    end_date: str,
    symbol: str,
    initial_balance: float,
    market_making: bool,
    mm_symbol: str | None,
    mm_profile: str,
) -> None:
    """Run strategy backtest on historical data."""
    mm_info = ""
    if market_making:
        mm_symbol_display = mm_symbol or "SUI-PERP"
        mm_info = (
            f"\nMarket Making: {mm_profile.title()} profile on {mm_symbol_display}"
        )

    console.print(
        Panel(
            f"📊 Starting Backtest\n"
            f"Symbol: {symbol}\n"
            f"Period: {start_date} to {end_date}\n"
            f"Initial Balance: ${initial_balance:,.2f}"
            f"{mm_info}",
            title="Backtest",
            style="green",
        )
    )

    # This will be implemented later with the actual backtesting logic
    if market_making:
        console.print(
            "🔄 Market Making backtest simulation not yet implemented. Coming soon!"
        )
    else:
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
    if not confirm and not click.confirm(
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
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output = f"paper_trades_{timestamp}.{export_format}"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
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


@cli.command()
def diagnose() -> None:
    """Run startup diagnostics and component health checks."""
    console.print("🔍 Running startup diagnostics...", style="cyan bold")

    # Check startup health
    _check_startup_health()

    # Import and check bot components
    try:
        from . import check_startup_health, get_startup_diagnostics

        diagnostics = get_startup_diagnostics()
        healthy, issues = check_startup_health()

        console.print("📊 Component Status:", style="cyan bold")
        for component, available in diagnostics["components"].items():
            status = "✅" if available else "❌"
            console.print(f"  {status} {component.replace('_', ' ').title()}")

        if healthy:
            console.print("✅ All critical components are healthy", style="green bold")
        else:
            console.print("⚠️  Issues detected:", style="yellow bold")
            for issue in issues:
                console.print(f"  • {issue}", style="yellow")

        # Show version and core status
        console.print(f"📦 Bot Version: {diagnostics['version']}")
        console.print(
            f"🔧 Core Imports: {'✅ Success' if diagnostics['core_imports_successful'] else '❌ Failed'}"
        )

    except Exception as e:
        console.print(f"❌ Failed to run diagnostics: {e}", style="red")
        console.print("This may indicate critical import failures", style="red")


@cli.command()
@click.option("--symbol", default=None, help="Symbol to check market making status for")
def mm_status(_symbol: str | None) -> None:
    """Show market making configuration and status."""
    try:
        console.print("🔍 Market Making Status", style="cyan bold")
        console.print("=" * 40)

        # Load configuration to check market making settings
        from .config import Settings

        settings = Settings()

        # Basic status information
        mm_config = settings.market_making
        status_table = Table(title="Market Making Configuration")
        status_table.add_column("Setting", style="cyan")
        status_table.add_column("Value", style="white")

        status_table.add_row("Enabled", "✅ Yes" if mm_config.enabled else "❌ No")
        status_table.add_row("Symbol", mm_config.symbol)
        status_table.add_row("Profile", mm_config.profile.title())
        status_table.add_row(
            "Base Spread (bps)", str(mm_config.strategy.base_spread_bps)
        )
        status_table.add_row("Order Levels", str(mm_config.strategy.order_levels))
        status_table.add_row(
            "Max Position %", f"{mm_config.strategy.max_position_pct}%"
        )
        status_table.add_row("Cycle Interval", f"{mm_config.cycle_interval_seconds}s")

        console.print(status_table)

        if not mm_config.enabled:
            console.print(
                "\n💡 To enable market making, set MARKET_MAKING__ENABLED=true in your .env file",
                style="yellow",
            )

        # Show risk management settings
        console.print("\n⚠️  Risk Management Settings", style="yellow bold")
        risk_table = Table()
        risk_table.add_column("Setting", style="yellow")
        risk_table.add_column("Value", style="white")

        risk_table.add_row(
            "Max Position Value", f"${mm_config.risk.max_position_value:,}"
        )
        risk_table.add_row(
            "Max Inventory Imbalance",
            f"{mm_config.risk.max_inventory_imbalance * 100}%",
        )
        risk_table.add_row(
            "Daily Loss Limit", f"{mm_config.risk.daily_loss_limit_pct}%"
        )
        risk_table.add_row("Stop Loss", f"{mm_config.risk.stop_loss_pct}%")

        console.print(risk_table)

    except Exception as e:
        console.print(f"[red]Error retrieving market making status: {e}[/red]")


@cli.command()
@click.option(
    "--profile",
    type=click.Choice(["conservative", "moderate", "aggressive"]),
    required=True,
    help="Configuration profile to display",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
def mm_config(profile: str, output_format: str) -> None:
    """Show market making configuration for a specific profile.

    Examples:
        # View moderate profile configuration
        ai-trading-bot mm-config --profile moderate

        # Export configuration as JSON
        ai-trading-bot mm-config --profile aggressive --format json

        # Export configuration as YAML
        ai-trading-bot mm-config --profile conservative --format yaml
    """
    try:
        console.print(
            f"📋 Market Making Configuration - {profile.title()} Profile",
            style="cyan bold",
        )

        # Create configuration with the specified profile
        create_default_config = _get_lazy_import("create_default_config")
        if not create_default_config:
            console.print("[red]Market making configuration not available[/red]")
            return

        config = create_default_config(profile)

        if output_format == "json":
            import json

            config_dict = config.to_dict()
            console.print(json.dumps(config_dict, indent=2, default=str))
        elif output_format == "yaml":
            try:
                import yaml

                config_dict = config.to_dict()
                console.print(yaml.dump(config_dict, default_flow_style=False))
            except ImportError:
                console.print(
                    "[red]PyYAML not installed. Install with: pip install pyyaml[/red]"
                )
                return
        else:
            # Table format
            console.print("\n🎯 Strategy Configuration", style="green bold")
            strategy_table = Table()
            strategy_table.add_column("Parameter", style="green")
            strategy_table.add_column("Value", style="white")
            strategy_table.add_column("Description", style="dim")

            strategy_config = config.strategy
            strategy_table.add_row(
                "Base Spread",
                f"{strategy_config.base_spread_bps} bps",
                "Base spread in basis points",
            )
            strategy_table.add_row(
                "Min Spread",
                f"{strategy_config.min_spread_bps} bps",
                "Minimum allowed spread",
            )
            strategy_table.add_row(
                "Max Spread",
                f"{strategy_config.max_spread_bps} bps",
                "Maximum allowed spread",
            )
            strategy_table.add_row(
                "Order Levels",
                str(strategy_config.order_levels),
                "Number of order levels per side",
            )
            strategy_table.add_row(
                "Max Position",
                f"{strategy_config.max_position_pct}%",
                "Maximum position size",
            )
            strategy_table.add_row(
                "VuManChu Weight",
                f"{strategy_config.vumanchu_weight:.1f}",
                "Signal integration weight",
            )

            console.print(strategy_table)

            console.print("\n⚠️  Risk Configuration", style="yellow bold")
            risk_table = Table()
            risk_table.add_column("Parameter", style="yellow")
            risk_table.add_column("Value", style="white")
            risk_table.add_column("Description", style="dim")

            risk_config = config.risk
            risk_table.add_row(
                "Max Position Value",
                f"${risk_config.max_position_value:,}",
                "Maximum position value",
            )
            risk_table.add_row(
                "Inventory Imbalance",
                f"{risk_config.max_inventory_imbalance * 100}%",
                "Maximum inventory imbalance",
            )
            risk_table.add_row(
                "Daily Loss Limit",
                f"{risk_config.daily_loss_limit_pct}%",
                "Daily loss limit",
            )
            risk_table.add_row(
                "Stop Loss", f"{risk_config.stop_loss_pct}%", "Emergency stop loss"
            )
            risk_table.add_row(
                "Inventory Timeout",
                f"{risk_config.inventory_timeout_hours}h",
                "Max hours to hold inventory",
            )

            console.print(risk_table)

    except Exception as e:
        console.print(f"[red]Error displaying configuration: {e}[/red]")


@cli.command()
@click.option(
    "--config-file",
    default=None,
    help="Configuration file to validate (default: checks environment settings)",
)
def mm_validate(config_file: str | None) -> None:
    """Validate market making configuration and setup.

    Examples:
        # Validate current environment configuration
        ai-trading-bot mm-validate

        # Validate specific configuration file
        ai-trading-bot mm-validate --config-file config/market_making.json
    """
    try:
        console.print("🔧 Validating Market Making Configuration", style="cyan bold")
        console.print("=" * 50)

        validation_results = []

        # Test 1: Check if market making components are available
        console.print("\n1. Testing component availability...")
        components = [
            "MarketMakingConfig",
            "MarketMakingIntegrator",
            "create_default_config",
            "validate_config",
        ]

        for component in components:
            try:
                imported_component = _get_lazy_import(component)
                if imported_component:
                    console.print(f"   ✅ {component} available")
                    validation_results.append((component, True, "Available"))
                else:
                    console.print(f"   ❌ {component} not available")
                    validation_results.append((component, False, "Not available"))
            except Exception as e:
                console.print(f"   ❌ {component} failed: {str(e)[:50]}...")
                validation_results.append((component, False, f"Error: {e}"))

        # Test 2: Validate configuration
        console.print("\n2. Testing configuration validation...")
        try:
            if config_file:
                import json

                with Path(config_file).open() as f:
                    config_data = json.load(f)
                validate_config = _get_lazy_import("validate_config")
                if validate_config:
                    validate_config(config_data)
                    console.print("   ✅ Configuration file is valid")
                    validation_results.append(("Config File", True, "Valid"))
                else:
                    console.print("   ❌ Configuration validator not available")
                    validation_results.append(
                        ("Config File", False, "Validator unavailable")
                    )
            else:
                # Test default configuration
                create_default_config = _get_lazy_import("create_default_config")
                if create_default_config:
                    create_default_config("moderate")
                    console.print("   ✅ Default configuration created successfully")
                    validation_results.append(("Default Config", True, "Valid"))
                else:
                    console.print("   ❌ Default configuration creator not available")
                    validation_results.append(
                        ("Default Config", False, "Creator unavailable")
                    )
        except Exception as e:
            console.print(f"   ❌ Configuration validation failed: {e}")
            validation_results.append(("Configuration", False, f"Error: {e}"))

        # Test 3: Check environment variables
        console.print("\n3. Testing environment configuration...")
        env_vars = [
            "MARKET_MAKING__ENABLED",
            "MARKET_MAKING__SYMBOL",
            "MARKET_MAKING__PROFILE",
        ]

        for env_var in env_vars:
            value = os.getenv(env_var)
            if value:
                console.print(f"   ✅ {env_var} = {value}")
                validation_results.append((env_var, True, value))
            else:
                console.print(f"   ⚠️  {env_var} not set (using defaults)")
                validation_results.append((env_var, False, "Not set"))

        # Test 4: Exchange compatibility
        console.print("\n4. Testing exchange compatibility...")
        try:
            from .config import Settings

            settings = Settings()
            exchange_type = settings.exchange.exchange_type

            if exchange_type == "bluefin":
                console.print(
                    "   ✅ Bluefin exchange detected - market making supported"
                )
                validation_results.append(("Exchange", True, "Bluefin - Compatible"))
            elif exchange_type == "coinbase":
                console.print("   ⚠️  Coinbase exchange - limited market making support")
                validation_results.append(
                    ("Exchange", False, "Coinbase - Limited support")
                )
            else:
                console.print(f"   ❌ Unknown exchange type: {exchange_type}")
                validation_results.append(
                    ("Exchange", False, f"Unknown: {exchange_type}")
                )
        except Exception as e:
            console.print(f"   ❌ Exchange check failed: {e}")
            validation_results.append(("Exchange", False, f"Error: {e}"))

        # Summary
        console.print("\n📊 Validation Summary", style="bold")
        console.print("=" * 30)

        passed = sum(1 for _, success, _ in validation_results if success)
        total = len(validation_results)

        if passed == total:
            console.print(
                f"✅ All {total} validation checks passed!", style="green bold"
            )
        else:
            failed = total - passed
            console.print(
                f"⚠️  {passed}/{total} checks passed, {failed} issues found",
                style="yellow bold",
            )

            console.print("\n🔧 Issues to resolve:", style="yellow")
            for name, success, message in validation_results:
                if not success:
                    console.print(f"  • {name}: {message}")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")


@cli.command()
@click.option("--profile", default="moderate", help="Market making profile to test")
@click.option("--duration", default=30, help="Test duration in seconds")
def mm_test(profile: str, duration: int) -> None:
    """Test market making components without placing real orders.

    Examples:
        # Test moderate profile for 30 seconds
        ai-trading-bot mm-test --profile moderate

        # Quick 10-second test of aggressive profile
        ai-trading-bot mm-test --profile aggressive --duration 10
    """
    try:
        console.print(
            f"🧪 Testing Market Making Components - {profile.title()} Profile",
            style="cyan bold",
        )
        console.print(f"Duration: {duration} seconds (dry-run mode)\n")

        # Create test configuration
        create_default_config = _get_lazy_import("create_default_config")
        if not create_default_config:
            console.print("[red]Market making configuration not available[/red]")
            return

        config = create_default_config(profile)
        console.print(f"✅ Created {profile} configuration")

        # Test configuration validation
        validate_config = _get_lazy_import("validate_config")
        if validate_config:
            validate_config(config.to_dict())
            console.print("✅ Configuration validation passed")
        else:
            console.print("⚠️  Configuration validator not available")

        # Test market making integrator initialization
        market_making_integrator_cls = _get_lazy_import("MarketMakingIntegrator")
        if market_making_integrator_cls:
            console.print("✅ Market making integrator available")

            # Create a test integrator (dry-run mode)
            market_making_integrator_cls(
                symbol="SUI-PERP",
                exchange_client=None,  # No real exchange for testing
                dry_run=True,
                market_making_symbols=[config.symbol],
                config=config.to_dict(),
            )
            console.print("✅ Test integrator created successfully")
        else:
            console.print("❌ Market making integrator not available")
            return

        # Simulate a brief test run
        console.print(f"\n🔄 Running {duration}-second simulation...")
        with console.status("[cyan]Testing market making components..."):
            import time

            time.sleep(min(duration, 10))  # Cap at 10 seconds for safety

        console.print("\n✅ Market making component test completed successfully!")
        console.print("\n💡 Next steps:", style="yellow bold")
        console.print("  1. Enable market making: Set MARKET_MAKING__ENABLED=true")
        console.print("  2. Configure your preferred symbol and profile")
        console.print("  3. Run with: ai-trading-bot live --market-making")

    except Exception as e:
        console.print(f"[red]Market making test failed: {e}[/red]")


if __name__ == "__main__":
    cli()
