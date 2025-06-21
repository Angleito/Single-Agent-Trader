"""
Market Making Integration Module.

This module provides the integration bridge between the market making engine and the main
trading bot, allowing seamless replacement of the LLM agent for specific symbols while
maintaining all existing bot functionality.

Key Features:
- Symbol-specific strategy replacement (SUI-PERP uses market making, others use LLM)
- Seamless integration with existing bot architecture
- Maintains compatibility with paper trading mode
- Comprehensive health monitoring and error recovery
- Thread-safe operation for real-time trading
- Proper startup/shutdown sequences
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from bot.config import settings
from bot.exchange.bluefin import BluefinClient
from bot.exchange.factory import ExchangeFactory
from bot.trading_types import MarketState, TradeAction


# Use lazy import for market making engine to avoid decorator issues
def _get_market_making_engine_factory():
    """Lazy import for MarketMakingEngineFactory."""
    try:
        from .market_making_engine import MarketMakingEngineFactory

        return MarketMakingEngineFactory
    except ImportError as e:
        logger.exception(f"Failed to import MarketMakingEngineFactory: {e}")
        return None


logger = logging.getLogger(__name__)


class MarketMakingIntegrationStatus:
    """Status tracking for market making integration."""

    def __init__(self):
        self.is_initialized = False
        self.is_running = False
        self.market_making_enabled = False
        self.symbol_strategy_map: dict[str, str] = {}
        self.initialization_time: datetime | None = None
        self.last_health_check = datetime.now(UTC)
        self.error_count = 0
        self.last_error_time: datetime | None = None
        self.last_error_message: str | None = None
        self.engine_status: dict[str, Any] | None = None


class MarketMakingIntegrator:
    """
    Integration bridge between market making engine and main trading bot.

    This class provides a seamless way to replace the LLM agent with market making
    for specific symbols while maintaining all existing bot functionality.

    Features:
    - Symbol-specific strategy replacement
    - Maintains LLM agent for non-market-making symbols
    - Health monitoring and error recovery
    - Paper trading mode compatibility
    - Thread-safe operations
    """

    def __init__(
        self,
        symbol: str,
        exchange_client: Any | None = None,
        dry_run: bool = True,
        market_making_symbols: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the market making integrator.

        Args:
            symbol: Primary trading symbol
            exchange_client: Exchange client instance
            dry_run: Whether to run in paper trading mode
            market_making_symbols: List of symbols to use market making for
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.exchange_client = exchange_client
        self.dry_run = dry_run
        self.config = config or {}

        # Default market making symbols (can be overridden)
        self.market_making_symbols = market_making_symbols or ["SUI-PERP"]

        # Initialize status tracking
        self.status = MarketMakingIntegrationStatus()

        # Initialize components
        self.market_making_engine: Any | None = None  # MarketMakingEngine
        self.llm_agent: Any | None = None  # Will be set during initialization

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Shutdown tracking
        self._shutdown_requested = False

        logger.info(
            f"MarketMakingIntegrator initialized for symbol {symbol} "
            f"(dry_run={dry_run}, market_making_symbols={self.market_making_symbols})"
        )

    async def initialize(self, llm_agent: Any) -> None:
        """
        Initialize the market making integration.

        Args:
            llm_agent: The LLM agent instance from the main bot
        """
        async with self._lock:
            try:
                logger.info("Initializing market making integration...")

                # Store the LLM agent for non-market-making symbols
                self.llm_agent = llm_agent

                # Determine if market making should be enabled for any symbols
                should_enable_market_making = len(self.market_making_symbols) > 0

                if should_enable_market_making:
                    await self._initialize_market_making_engine()
                    self.status.market_making_enabled = True
                    logger.info(
                        f"Market making enabled for symbols {self.market_making_symbols}"
                    )
                else:
                    logger.info(
                        "Market making disabled, no symbols configured for market making"
                    )

                # Update status mapping
                for symbol in self.market_making_symbols:
                    self.status.symbol_strategy_map[symbol] = "market_making"

                # Mark as initialized
                self.status.is_initialized = True
                self.status.initialization_time = datetime.now(UTC)

                logger.info("Market making integration initialized successfully")

            except Exception as e:
                logger.exception(f"Failed to initialize market making integration: {e}")
                self.status.error_count += 1
                self.status.last_error_time = datetime.now(UTC)
                self.status.last_error_message = str(e)
                raise

    async def _initialize_market_making_engine(self) -> None:
        """Initialize the market making engine."""
        try:
            # Ensure we have a Bluefin client for market making
            if not isinstance(self.exchange_client, BluefinClient):
                # Check if it's a mock client for testing
                if (
                    hasattr(self.exchange_client, "client_type")
                    and self.exchange_client.client_type == "bluefin"
                ):
                    # This is a mock Bluefin client, allow it for testing
                    logger.debug("Using mock Bluefin client for testing")
                elif self.exchange_client is None:
                    # Create a new Bluefin client
                    logger.info("Creating Bluefin client for market making...")
                    factory = ExchangeFactory()
                    self.exchange_client = factory.create_exchange(
                        exchange_type="bluefin", dry_run=self.dry_run
                    )
                else:
                    raise ValueError(
                        f"Market making requires Bluefin exchange, got {type(self.exchange_client)}"
                    )

            # Create market making engine
            MarketMakingEngineFactory = _get_market_making_engine_factory()
            if not MarketMakingEngineFactory:
                raise ImportError("MarketMakingEngineFactory not available")

            self.market_making_engine = MarketMakingEngineFactory.create_engine(
                exchange_client=self.exchange_client,
                symbol=self.symbol,
                config=self.config,
            )

            # Initialize the engine
            await self.market_making_engine.initialize()

            logger.info("Market making engine initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize market making engine: {e}")
            raise

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market state and generate trading decision.

        This method routes the analysis to either the market making engine or
        the LLM agent based on the symbol configuration.

        Args:
            market_state: Current market state

        Returns:
            TradeAction with decision and parameters
        """
        if not self.status.is_initialized:
            raise RuntimeError("MarketMakingIntegrator not initialized")

        symbol = getattr(market_state, "symbol", self.symbol)

        try:
            # Route to appropriate strategy based on symbol
            if symbol in self.market_making_symbols and self.market_making_engine:
                # Use market making engine
                action = await self._analyze_with_market_making(market_state)
                logger.debug(f"Market making decision for {symbol}: {action.action}")
                return action
            # Use LLM agent
            if not self.llm_agent:
                raise RuntimeError("LLM agent not available")
            action = await self.llm_agent.analyze_market(market_state)
            logger.debug(f"LLM decision for {symbol}: {action.action}")
            return action

        except Exception as e:
            logger.exception(f"Failed to analyze market for {symbol}: {e}")
            self.status.error_count += 1
            self.status.last_error_time = datetime.now(UTC)
            self.status.last_error_message = str(e)

            # Return safe HOLD action on error
            return TradeAction(
                action="HOLD",
                size_pct=0.0,
                take_profit_pct=0.0,
                stop_loss_pct=0.0,
                rationale=f"Error in analysis: {e!s}",
            )

    async def _analyze_with_market_making(
        self, market_state: MarketState
    ) -> TradeAction:
        """
        Analyze market state using the market making engine.

        Args:
            market_state: Current market state

        Returns:
            TradeAction from market making engine
        """
        if not self.market_making_engine:
            raise RuntimeError("Market making engine not initialized")

        # Use the market making engine's compatibility method
        return await self.market_making_engine.analyze_market_and_decide(market_state)

    async def start(self) -> None:
        """Start the market making integration."""
        async with self._lock:
            if self.status.is_running:
                logger.warning("Market making integration already running")
                return

            try:
                logger.info("Starting market making integration...")

                if self.market_making_engine:
                    await self.market_making_engine.start()

                self.status.is_running = True
                self._shutdown_requested = False

                logger.info("Market making integration started successfully")

            except Exception as e:
                logger.exception(f"Failed to start market making integration: {e}")
                self.status.error_count += 1
                self.status.last_error_time = datetime.now(UTC)
                self.status.last_error_message = str(e)
                raise

    async def stop(self) -> None:
        """Stop the market making integration."""
        async with self._lock:
            if not self.status.is_running:
                logger.info("Market making integration already stopped")
                return

            try:
                logger.info("Stopping market making integration...")

                self._shutdown_requested = True

                if self.market_making_engine:
                    await self.market_making_engine.stop()

                self.status.is_running = False

                logger.info("Market making integration stopped successfully")

            except Exception as e:
                logger.exception(
                    f"Error during market making integration shutdown: {e}"
                )
                self.status.error_count += 1
                self.status.last_error_time = datetime.now(UTC)
                self.status.last_error_message = str(e)
                # Don't re-raise during shutdown

    async def get_status(self) -> dict[str, Any]:
        """
        Get comprehensive status information.

        Returns:
            Dictionary containing status information
        """
        # Update health check time
        self.status.last_health_check = datetime.now(UTC)

        # Get engine status if available
        engine_status = None
        if self.market_making_engine:
            try:
                engine_status = await self.market_making_engine.get_status()
            except Exception as e:
                logger.debug(f"Failed to get engine status: {e}")

        self.status.engine_status = engine_status

        return {
            "is_initialized": self.status.is_initialized,
            "is_running": self.status.is_running,
            "market_making_enabled": self.status.market_making_enabled,
            "symbol": self.symbol,
            "market_making_symbols": self.market_making_symbols,
            "symbol_strategy_map": self.status.symbol_strategy_map,
            "initialization_time": self.status.initialization_time,
            "last_health_check": self.status.last_health_check,
            "error_count": self.status.error_count,
            "last_error_time": self.status.last_error_time,
            "last_error_message": self.status.last_error_message,
            "engine_status": engine_status,
            "dry_run": self.dry_run,
            "shutdown_requested": self._shutdown_requested,
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status
        """
        try:
            status = await self.get_status()

            # Check if everything is healthy
            is_healthy = (
                status["is_initialized"]
                and (not status["market_making_enabled"] or status["is_running"])
                and status["error_count"] == 0
            )

            # Check engine health if available
            engine_healthy = True
            if self.market_making_engine and status["market_making_enabled"]:
                try:
                    engine_status = await self.market_making_engine.get_status()
                    engine_healthy = not engine_status.get("emergency_stop", False)
                except Exception:
                    engine_healthy = False

            return {
                "healthy": is_healthy and engine_healthy,
                "status": status,
                "checks": {
                    "initialized": status["is_initialized"],
                    "running": (
                        status["is_running"]
                        if status["market_making_enabled"]
                        else True
                    ),
                    "no_errors": status["error_count"] == 0,
                    "engine_healthy": engine_healthy,
                },
            }

        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "checks": {
                    "initialized": False,
                    "running": False,
                    "no_errors": False,
                    "engine_healthy": False,
                },
            }

    def is_available(self) -> bool:
        """
        Check if the integrator is available for trading decisions.

        Returns:
            True if available, False otherwise
        """
        return self.status.is_initialized and (
            self.llm_agent is not None or self.market_making_engine is not None
        )

    def get_strategy_for_symbol(self, symbol: str) -> str:
        """
        Get the strategy type for a given symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Strategy type ("market_making" or "llm")
        """
        if symbol in self.market_making_symbols:
            return "market_making"
        return "llm"

    @asynccontextmanager
    async def managed_lifecycle(self):
        """
        Context manager for managing the complete lifecycle.

        Usage:
            async with integrator.managed_lifecycle():
                # Use integrator for trading
                pass
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    async def emergency_stop(self) -> None:
        """
        Emergency stop for all market making activities.

        This method immediately stops all market making operations
        and cancels any pending orders.
        """
        logger.warning("Emergency stop requested for market making integration")

        if self.market_making_engine:
            try:
                await self.market_making_engine.emergency_stop()
            except Exception as e:
                logger.exception(f"Error during emergency stop: {e}")

        await self.stop()

    def __repr__(self) -> str:
        """String representation of the integrator."""
        return (
            f"MarketMakingIntegrator(symbol={self.symbol}, "
            f"market_making_enabled={self.status.market_making_enabled}, "
            f"initialized={self.status.is_initialized}, "
            f"running={self.status.is_running})"
        )


class MarketMakingIntegratorFactory:
    """Factory for creating market making integrators with proper configuration."""

    @staticmethod
    def create_integrator(
        symbol: str,
        exchange_client: Any | None = None,
        dry_run: bool = True,
        config: dict[str, Any] | None = None,
    ) -> MarketMakingIntegrator:
        """
        Create a market making integrator with configuration from settings.

        Args:
            symbol: Trading symbol
            exchange_client: Exchange client instance
            dry_run: Whether to run in paper trading mode
            config: Optional configuration override

        Returns:
            Configured MarketMakingIntegrator instance
        """
        # Get market making symbols from settings or use default
        market_making_symbols = ["SUI-PERP"]  # Default to SUI-PERP

        # Override with config if provided
        if config and "market_making_symbols" in config:
            market_making_symbols = config["market_making_symbols"]

        # Create integrator configuration
        integrator_config = {
            "market_making_symbols": market_making_symbols,
            **(config or {}),
        }

        return MarketMakingIntegrator(
            symbol=symbol,
            exchange_client=exchange_client,
            dry_run=dry_run,
            market_making_symbols=market_making_symbols,
            config=integrator_config,
        )

    @staticmethod
    def create_from_settings(
        symbol: str,
        exchange_client: Any | None = None,
    ) -> MarketMakingIntegrator:
        """
        Create a market making integrator using global settings.

        Args:
            symbol: Trading symbol
            exchange_client: Exchange client instance

        Returns:
            Configured MarketMakingIntegrator instance
        """
        return MarketMakingIntegrator(
            symbol=symbol,
            exchange_client=exchange_client,
            dry_run=settings.system.dry_run,
            market_making_symbols=["SUI-PERP"],  # Default
            config=None,
        )
