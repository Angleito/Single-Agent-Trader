"""
Market Making Main Engine Implementation.

This module implements the main orchestration engine that coordinates all market making
components to provide systematic liquidity provision on the Bluefin DEX.

Key Features:
- Coordinates all market making components (strategy, orders, inventory, monitoring)
- Integrates with existing bot architecture and exchange interfaces
- Provides main execution loop for market making operations
- Includes emergency controls and comprehensive risk management
- Thread-safe operation for real-time trading
- Replaces LLM agent for SUI-PERP symbol specifically
"""

import logging
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from bot.exchange.bluefin import BluefinClient
from bot.exchange.bluefin_fee_calculator import BluefinFeeCalculator
from bot.indicators.vumanchu import VuManChuIndicators
from bot.trading_types import MarketState, TradeAction

from .inventory_manager import InventoryManager, RebalancingAction
from .market_making_order_manager import MarketMakingOrderManager
from .market_making_performance_monitor import MarketMakingPerformanceMonitor
from .market_making_strategy import DirectionalBias, MarketMakingStrategy, OrderLevel
from .spread_calculator import DynamicSpreadCalculator

logger = logging.getLogger(__name__)


class MarketMakingState:
    """Current state of the market making engine."""

    def __init__(self):
        self.is_running = False
        self.last_cycle_time = datetime.now(UTC)
        self.cycle_count = 0
        self.total_runtime = timedelta()
        self.emergency_stop = False
        self.error_count = 0
        self.last_error_time: datetime | None = None
        self.last_market_state: MarketState | None = None
        self.last_directional_bias: DirectionalBias | None = None
        self.active_orders: list[Any] = []


class MarketMakingEngine:
    """
    Main Market Making Engine.

    This engine coordinates all market making components to provide systematic
    liquidity provision while managing inventory risk and optimizing profitability.

    The engine replaces the LLM agent for SUI-PERP symbol specifically, while
    maintaining compatibility with the existing bot architecture.

    Components coordinated:
    - MarketMakingStrategy: Calculates optimal spreads and order levels
    - MarketMakingOrderManager: Manages order placement and fills
    - InventoryManager: Tracks and rebalances inventory positions
    - MarketMakingPerformanceMonitor: Monitors performance and risk metrics
    - DynamicSpreadCalculator: Calculates optimal spreads based on market conditions
    - BluefinFeeCalculator: Calculates fees for profitability analysis
    """

    def __init__(
        self,
        exchange_client: BluefinClient,
        symbol: str = "SUI-PERP",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Market Making Engine.

        Args:
            exchange_client: Bluefin exchange client
            symbol: Trading symbol (default: SUI-PERP)
            config: Optional configuration overrides
        """
        self.exchange_client = exchange_client
        self.symbol = symbol
        self.config = config or {}

        # Initialize state
        self.state = MarketMakingState()
        self._lock = threading.RLock()

        # Initialize components
        self._initialize_components()

        # Configuration parameters
        self.cycle_interval = self.config.get("cycle_interval", 1.0)  # seconds
        self.max_errors_per_hour = self.config.get("max_errors_per_hour", 50)
        self.emergency_stop_threshold = self.config.get("emergency_stop_threshold", 10)
        self.max_position_value = Decimal(
            str(self.config.get("max_position_value", 10000))
        )

        logger.info(
            "Initialized MarketMakingEngine for %s with cycle interval %.2fs",
            self.symbol,
            self.cycle_interval,
        )

    def _initialize_components(self):
        """Initialize all market making components."""
        try:
            # Initialize fee calculator
            self.fee_calculator = BluefinFeeCalculator()

            # Initialize core strategy
            self.strategy = MarketMakingStrategy(
                fee_calculator=self.fee_calculator,
                exchange_client=self.exchange_client,
                config=self.config.get("strategy", {}),
            )

            # Initialize order manager
            self.order_manager = MarketMakingOrderManager(
                exchange_client=self.exchange_client,
                symbol=self.symbol,
                max_levels=self.config.get("order_manager", {}).get("max_levels", 5),
                price_move_threshold=Decimal(
                    str(
                        self.config.get("order_manager", {}).get(
                            "price_move_threshold", "0.001"
                        )
                    )
                ),
                max_position_value=Decimal(
                    str(
                        self.config.get("order_manager", {}).get(
                            "max_position_value", "10000"
                        )
                    )
                ),
                emergency_stop_loss=Decimal(
                    str(
                        self.config.get("order_manager", {}).get(
                            "emergency_stop_loss", "0.02"
                        )
                    )
                ),
            )

            # Initialize inventory manager
            self.inventory_manager = InventoryManager(
                symbol=self.symbol,
                max_position_pct=self.config.get("inventory", {}).get(
                    "max_position_pct", 10.0
                ),
                rebalancing_threshold=self.config.get("inventory", {}).get(
                    "rebalancing_threshold", 5.0
                ),
                emergency_threshold=self.config.get("inventory", {}).get(
                    "emergency_threshold", 15.0
                ),
                inventory_timeout_hours=self.config.get("inventory", {}).get(
                    "inventory_timeout_hours", 4.0
                ),
            )

            # Initialize performance monitor
            self.performance_monitor = MarketMakingPerformanceMonitor(
                fee_calculator=self.fee_calculator,
                symbol=self.symbol,
                max_history_size=self.config.get("performance", {}).get(
                    "max_history_size", 10000
                ),
            )

            # Initialize dynamic spread calculator
            self.spread_calculator = DynamicSpreadCalculator(
                fee_calculator=self.fee_calculator,
                config=self.config.get("spread_calculator", {}),
            )

            # Initialize VuManChu indicators for directional bias
            self.indicators = VuManChuIndicators()

            logger.info("Successfully initialized all market making components")

        except Exception as e:
            logger.exception("Failed to initialize market making components: %s", e)
            raise

    async def analyze_market_state(self, market_state: MarketState) -> DirectionalBias:
        """
        Analyze current market state and calculate directional bias.

        Args:
            market_state: Current market state with OHLCV data and indicators

        Returns:
            DirectionalBias with direction, strength, and confidence
        """
        try:
            # Calculate VuManChu indicators
            indicators = self.indicators.calculate(market_state.ohlcv_data)

            # Get directional bias from strategy
            bias = self.strategy.calculate_directional_bias(
                market_state=market_state, indicators=indicators
            )

            # Store for monitoring
            self.state.last_directional_bias = bias
            self.state.last_market_state = market_state

            logger.debug(
                "Market analysis complete - Direction: %s, Strength: %.2f, Confidence: %.2f",
                bias.direction,
                bias.strength,
                bias.confidence,
            )

            return bias

        except Exception as e:
            logger.exception("Error analyzing market state: %s", e)
            # Return neutral bias on error
            return DirectionalBias(
                direction="neutral", strength=0.0, confidence=0.0, signals={}
            )

    async def calculate_optimal_actions(
        self, market_state: MarketState, directional_bias: DirectionalBias
    ) -> tuple[list[OrderLevel], RebalancingAction | None]:
        """
        Calculate optimal order levels and rebalancing actions.

        Args:
            market_state: Current market state
            directional_bias: Calculated directional bias

        Returns:
            Tuple of (order_levels, rebalancing_action)
        """
        try:
            # Get current inventory metrics
            inventory_metrics = self.inventory_manager.get_inventory_metrics()

            # Calculate dynamic spread
            spread_info = self.spread_calculator.calculate_optimal_spread(
                market_state=market_state,
                directional_bias=directional_bias,
                inventory_metrics=inventory_metrics,
            )

            # Generate order levels
            order_levels = self.strategy.generate_order_levels(
                market_state=market_state,
                directional_bias=directional_bias,
                spread_calculation=spread_info,
                inventory_metrics=inventory_metrics,
            )

            # Check if rebalancing is needed
            rebalancing_action = None
            if inventory_metrics.risk_score > inventory_metrics.rebalancing_threshold:
                rebalancing_action = self.inventory_manager.suggest_rebalancing_action(
                    current_price=market_state.current_price,
                    volatility=(
                        market_state.volatility
                        if hasattr(market_state, "volatility")
                        else 0.02
                    ),
                    market_trend=directional_bias.direction,
                )

            logger.debug(
                "Calculated %d order levels and %s rebalancing action",
                len(order_levels),
                "required" if rebalancing_action else "no",
            )

            return order_levels, rebalancing_action

        except Exception as e:
            logger.exception("Error calculating optimal actions: %s", e)
            return [], None

    async def execute_market_making_cycle(
        self,
        order_levels: list[OrderLevel],
        rebalancing_action: RebalancingAction | None,
    ) -> bool:
        """
        Execute a complete market making cycle.

        Args:
            order_levels: Calculated optimal order levels
            rebalancing_action: Optional rebalancing action

        Returns:
            True if cycle completed successfully
        """
        try:
            # Execute rebalancing if needed
            if rebalancing_action and rebalancing_action.action_type != "HOLD":
                success = await self._execute_rebalancing(rebalancing_action)
                if not success:
                    logger.warning(
                        "Rebalancing failed, proceeding with order management"
                    )

            # Update orders based on new levels
            await self.order_manager.place_ladder_orders(
                levels=order_levels,
                symbol=self.symbol,
                current_price=(
                    self.state.last_market_state.current_price
                    if self.state.last_market_state
                    else Decimal("1.0")
                ),
            )

            # Update performance tracking
            inventory_metrics = self.inventory_manager.get_inventory_metrics()
            self.performance_monitor.record_inventory_snapshot(inventory_metrics)

            # Check for emergency conditions
            if await self._check_emergency_conditions():
                logger.warning(
                    "Emergency conditions detected, triggering emergency stop"
                )
                await self.emergency_stop_trading()
                return False

            logger.debug("Market making cycle completed successfully")
            return True

        except Exception as e:
            logger.exception("Error executing market making cycle: %s", e)
            self.state.error_count += 1
            self.state.last_error_time = datetime.now(UTC)
            return False

    async def monitor_and_adjust(self) -> None:
        """
        Monitor positions and performance, make real-time adjustments.
        """
        try:
            # Update inventory tracking - inventory manager handles this internally

            # Check for filled orders and update tracking - handled by order manager monitoring

            # Monitor performance metrics
            performance_metrics = self.performance_monitor.get_performance_metrics()

            # Log performance summary
            if self.state.cycle_count % 60 == 0:  # Every minute
                await self._log_performance_summary(performance_metrics)

            # Check for risk threshold breaches
            inventory_metrics = self.inventory_manager.get_inventory_metrics()
            if inventory_metrics.risk_score > 0.8:  # High risk threshold
                logger.warning(
                    "High inventory risk detected: %.2f (position: %s)",
                    inventory_metrics.risk_score,
                    inventory_metrics.net_position,
                )

                # Consider emergency rebalancing
                if inventory_metrics.risk_score > 0.9:
                    await self._emergency_rebalance()

        except Exception as e:
            logger.exception("Error in monitoring and adjustment: %s", e)

    async def _execute_rebalancing(self, action: RebalancingAction) -> bool:
        """Execute inventory rebalancing action."""
        try:
            logger.info(
                "Executing rebalancing: %s %s (urgency: %s)",
                action.action_type,
                action.quantity,
                action.urgency,
            )

            # Create trade action for rebalancing
            TradeAction(
                action="LONG" if action.action_type == "BUY" else "SHORT",
                size_pct=float(
                    min(action.quantity / self.max_position_value * 100, 25)
                ),
                take_profit_pct=2.0,  # Conservative for rebalancing
                stop_loss_pct=1.0,
                rationale=f"Inventory rebalancing: {action.reason}",
            )

            # Execute rebalancing trade through inventory manager
            success = self.inventory_manager.execute_rebalancing_trade(
                action=action,
                current_price=(
                    action.target_price or self.state.last_market_state.current_price
                    if self.state.last_market_state
                    else Decimal("1.0")
                ),
            )

            if success:
                logger.info("Rebalancing trade executed successfully")
            else:
                logger.error("Rebalancing trade failed")

            return success

        except Exception as e:
            logger.exception("Error executing rebalancing: %s", e)
            return False

    async def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions that require immediate action."""
        try:
            # Check error rate
            if self.state.error_count >= self.emergency_stop_threshold and (
                self.state.last_error_time
                and (datetime.now(UTC) - self.state.last_error_time).total_seconds()
                < 3600
            ):
                logger.error("Emergency stop triggered: too many errors")
                return True

            # Check position size
            inventory_metrics = self.inventory_manager.get_inventory_metrics()
            if abs(inventory_metrics.position_value) > self.max_position_value:
                logger.error("Emergency stop triggered: position too large")
                return True

            # Check performance metrics
            performance_metrics = self.performance_monitor.get_performance_metrics()
            if (
                hasattr(performance_metrics, "daily_pnl")
                and performance_metrics["daily_pnl"]
                < -float(self.max_position_value) * 0.1
            ):
                logger.error("Emergency stop triggered: large daily loss")
                return True

            return False

        except Exception as e:
            logger.exception("Error checking emergency conditions: %s", e)
            return True  # Err on the side of caution

    async def _emergency_rebalance(self) -> None:
        """Execute emergency inventory rebalancing."""
        try:
            logger.warning("Executing emergency rebalancing")

            inventory_metrics = self.inventory_manager.get_inventory_metrics()

            # Calculate emergency rebalancing action
            action = RebalancingAction(
                action_type="SELL" if inventory_metrics.net_position > 0 else "BUY",
                quantity=abs(inventory_metrics.net_position)
                * Decimal("0.5"),  # Reduce by 50%
                urgency="EMERGENCY",
                reason="Emergency risk management rebalancing",
                confidence=1.0,
            )

            await self._execute_rebalancing(action)

        except Exception as e:
            logger.exception("Error in emergency rebalancing: %s", e)

    async def _log_performance_summary(self, performance_metrics: Any) -> None:
        """Log performance summary for monitoring."""
        try:
            inventory_metrics = self.inventory_manager.get_inventory_metrics()

            logger.info(
                "Performance Summary - Cycles: %d, Runtime: %s, "
                "Position: %s, Risk Score: %.2f, Errors: %d",
                self.state.cycle_count,
                str(self.state.total_runtime).split(".")[0],  # Remove microseconds
                inventory_metrics.net_position,
                inventory_metrics.risk_score,
                self.state.error_count,
            )

        except Exception as e:
            logger.exception("Error logging performance summary: %s", e)

    async def emergency_stop_trading(self) -> None:
        """
        Emergency stop all trading operations.

        This method should be called in emergency situations to immediately
        stop all trading activities and cancel open orders.
        """
        try:
            logger.critical("EMERGENCY STOP activated for market making engine")

            with self._lock:
                self.state.emergency_stop = True
                self.state.is_running = False

            # Cancel all open orders
            await self.order_manager.cancel_all_orders(self.symbol)

            # Log emergency stop details
            inventory_metrics = self.inventory_manager.get_inventory_metrics()
            logger.critical(
                "Emergency stop completed - Position: %s, Risk Score: %.2f",
                inventory_metrics.net_position,
                inventory_metrics.risk_score,
            )

        except Exception as e:
            logger.exception("Error during emergency stop: %s", e)

    async def initialize(self) -> None:
        """Initialize the market making engine (alias for start)."""
        await self.start()

    async def start(self) -> None:
        """Start the market making engine."""
        try:
            logger.info("Starting market making engine for %s", self.symbol)

            with self._lock:
                if self.state.is_running:
                    logger.warning("Market making engine already running")
                    return

                self.state.is_running = True
                self.state.emergency_stop = False
                self.state.cycle_count = 0
                self.state.error_count = 0

            # Initialize components
            await self.order_manager.start()
            # Inventory manager doesn't have async start method
            await self.performance_monitor.start_monitoring()

            logger.info("Market making engine started successfully")

        except Exception as e:
            logger.exception("Error starting market making engine: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the market making engine gracefully."""
        try:
            logger.info("Stopping market making engine")

            with self._lock:
                self.state.is_running = False

            # Cancel all orders
            await self.order_manager.cancel_all_orders(self.symbol)

            # Stop components
            await self.order_manager.stop()
            # Inventory manager doesn't have async stop method
            await self.performance_monitor.stop_monitoring()

            logger.info("Market making engine stopped successfully")

        except Exception as e:
            logger.exception("Error stopping market making engine: %s", e)

    @asynccontextmanager
    async def run_context(self):
        """Context manager for running the market making engine."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    async def run_single_cycle(self, market_state: MarketState) -> bool:
        """
        Run a single market making cycle.

        This method integrates with the existing bot architecture by providing
        a single cycle execution that can be called from the main trading loop.

        Args:
            market_state: Current market state

        Returns:
            True if cycle completed successfully
        """
        try:
            if self.state.emergency_stop:
                logger.warning("Market making engine in emergency stop mode")
                return False

            cycle_start_time = datetime.now(UTC)

            # Step 1: Analyze market state
            directional_bias = await self.analyze_market_state(market_state)

            # Step 2: Calculate optimal actions
            order_levels, rebalancing_action = await self.calculate_optimal_actions(
                market_state, directional_bias
            )

            # Step 3: Execute market making cycle
            success = await self.execute_market_making_cycle(
                order_levels, rebalancing_action
            )

            # Step 4: Monitor and adjust
            await self.monitor_and_adjust()

            # Update cycle tracking
            cycle_end_time = datetime.now(UTC)
            cycle_duration = cycle_end_time - cycle_start_time

            with self._lock:
                self.state.cycle_count += 1
                self.state.last_cycle_time = cycle_end_time
                self.state.total_runtime += cycle_duration

            if success:
                logger.debug(
                    "Market making cycle %d completed in %.2fs",
                    self.state.cycle_count,
                    cycle_duration.total_seconds(),
                )

            return success

        except Exception as e:
            logger.exception("Error in market making cycle: %s", e)
            self.state.error_count += 1
            self.state.last_error_time = datetime.now(UTC)
            return False

    def is_running(self) -> bool:
        """Check if the market making engine is running."""
        with self._lock:
            return self.state.is_running and not self.state.emergency_stop

    def get_status(self) -> dict[str, Any]:
        """Get current status of the market making engine."""
        with self._lock:
            return {
                "is_running": self.state.is_running,
                "emergency_stop": self.state.emergency_stop,
                "cycle_count": self.state.cycle_count,
                "error_count": self.state.error_count,
                "last_cycle_time": (
                    self.state.last_cycle_time.isoformat()
                    if self.state.last_cycle_time
                    else None
                ),
                "total_runtime": str(self.state.total_runtime),
                "symbol": self.symbol,
                "active_orders": len(self.state.active_orders),
            }

    # LLM Agent compatibility methods for integration with existing bot

    async def analyze_market_and_decide(self, market_state: MarketState) -> TradeAction:
        """
        Compatibility method for LLM Agent replacement.

        This method provides the same interface as the LLM Agent's analyze_market_and_decide
        method, allowing the market making engine to be used as a drop-in replacement.

        Args:
            market_state: Current market state

        Returns:
            TradeAction (HOLD for market making mode)
        """
        try:
            # Run market making cycle
            success = await self.run_single_cycle(market_state)

            # Market making engine manages its own orders, so return HOLD
            # to prevent the main bot from executing additional trades
            return TradeAction(
                action="HOLD",
                size_pct=0.0,
                take_profit_pct=0.0,
                stop_loss_pct=0.0,
                rationale=(
                    f"Market making engine active - "
                    f"Cycle {'successful' if success else 'failed'}, "
                    f"Orders: {len(self.state.active_orders)}"
                ),
            )

        except Exception as e:
            logger.exception("Error in market making analysis: %s", e)
            return TradeAction(
                action="HOLD",
                size_pct=0.0,
                take_profit_pct=0.0,
                stop_loss_pct=0.0,
                rationale=f"Market making engine error: {str(e)[:100]}",
            )


class MarketMakingEngineFactory:
    """Factory for creating market making engines with proper configuration."""

    @staticmethod
    def create_engine(
        exchange_client: BluefinClient,
        symbol: str = "SUI-PERP",
        config: dict[str, Any] | None = None,
    ) -> MarketMakingEngine:
        """
        Create a market making engine with default configuration.

        Args:
            exchange_client: Bluefin exchange client
            symbol: Trading symbol
            config: Optional configuration overrides

        Returns:
            Configured MarketMakingEngine instance
        """
        # Default configuration
        default_config = {
            "cycle_interval": 1.0,
            "max_errors_per_hour": 50,
            "emergency_stop_threshold": 10,
            "max_position_value": 10000,
            "strategy": {
                "base_spread_bps": 10,
                "max_spread_bps": 50,
                "min_spread_bps": 5,
                "order_levels": 3,
                "max_position_pct": 25,
                "bias_adjustment_factor": 0.3,
            },
            "order_manager": {
                "max_orders_per_side": 5,
                "order_refresh_threshold": 0.5,
                "min_order_size": 10,
            },
            "inventory": {
                "max_position_limit": 1000,
                "rebalancing_threshold": 0.7,
                "target_inventory": 0.0,
            },
            "performance": {
                "tracking_window_hours": 24,
                "alert_thresholds": {"max_drawdown": 0.05, "min_profit_margin": 0.001},
            },
            "spread_calculator": {
                "volatility_lookback": 20,
                "liquidity_factor": 0.1,
                "market_impact_factor": 0.05,
            },
        }

        # Merge with provided config
        final_config = default_config.copy()
        if config:
            final_config.update(config)

        return MarketMakingEngine(
            exchange_client=exchange_client, symbol=symbol, config=final_config
        )
