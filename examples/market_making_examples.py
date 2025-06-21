"""
Market Making Examples and Scripts.

This module provides comprehensive working examples for all aspects of market making
operations, including different trading profiles, risk management scenarios, and
performance monitoring setups.

Key Features:
- Conservative and aggressive trading configurations
- Risk management and emergency scenarios
- Multi-symbol configurations
- Paper trading simulations
- Performance monitoring examples
- Custom profile creation
- Emergency stop and recovery procedures

Usage:
    python examples/market_making_examples.py --profile conservative
    python examples/market_making_examples.py --multi-symbol
    python examples/market_making_examples.py --paper-trading
    python examples/market_making_examples.py --emergency-demo
"""

import asyncio
import logging
import signal
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from bot.strategy.inventory_manager import InventoryManager
from bot.strategy.market_making_engine import (
    MarketMakingEngine,
    MarketMakingEngineFactory,
)
from bot.trading_types import IndicatorData, MarketState
from bot.utils.logging_config import setup_logging

# Initialize logging and console
setup_logging()
logger = logging.getLogger(__name__)
console = Console()


class MarketMakingExampleRunner:
    """
    Main runner for market making examples.

    Provides a unified interface for running different market making scenarios
    and demonstrations with proper error handling and cleanup.
    """

    def __init__(self):
        """Initialize the example runner."""
        self.engines: list[MarketMakingEngine] = []
        self.shutdown_requested = False
        self.console = Console()

    def register_shutdown_handlers(self):
        """Register signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info("Shutdown signal received, stopping all engines...")
            self.shutdown_requested = True
            asyncio.create_task(self.shutdown_all_engines())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def shutdown_all_engines(self):
        """Shutdown all running engines gracefully."""
        for engine in self.engines:
            try:
                if engine.is_running():
                    await engine.stop()
                    logger.info(f"Stopped engine for {engine.symbol}")
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")

        self.engines.clear()


class ConservativeTradingExample:
    """
    Conservative market making setup example.

    Demonstrates a low-risk market making configuration with:
    - Wide spreads for safety
    - Low position limits
    - Conservative rebalancing thresholds
    - Strong risk management
    """

    def __init__(self, symbol: str = "SUI-PERP"):
        """Initialize conservative trading example."""
        self.symbol = symbol
        self.config = self._create_conservative_config()

    def _create_conservative_config(self) -> dict[str, Any]:
        """Create conservative market making configuration."""
        return {
            "cycle_interval": 2.0,  # Slower cycle for stability
            "max_errors_per_hour": 20,  # Lower error tolerance
            "emergency_stop_threshold": 5,  # Quick emergency stop
            "max_position_value": 2500,  # Smaller position limits
            "strategy": {
                "base_spread_bps": 25,  # 0.25% wide spreads
                "max_spread_bps": 100,  # Up to 1% in volatile conditions
                "min_spread_bps": 15,  # Minimum 0.15% for safety
                "order_levels": 2,  # Only 2 levels to limit exposure
                "max_position_pct": 10,  # Maximum 10% position size
                "bias_adjustment_factor": 0.2,  # Limited bias adjustment
            },
            "order_manager": {
                "max_orders_per_side": 3,
                "order_refresh_threshold": 1.0,  # More conservative refresh
                "min_order_size": 25,  # Larger minimum size
            },
            "inventory": {
                "max_position_limit": 500,
                "rebalancing_threshold": 0.4,  # Early rebalancing
                "target_inventory": 0.0,  # Stay neutral
            },
            "performance": {
                "tracking_window_hours": 12,
                "alert_thresholds": {
                    "max_drawdown": 0.02,  # 2% max drawdown
                    "min_profit_margin": 0.002,  # 0.2% minimum profit
                },
            },
            "spread_calculator": {
                "volatility_lookback": 30,  # Longer lookback for stability
                "liquidity_factor": 0.2,  # Higher liquidity requirements
                "market_impact_factor": 0.08,  # Conservative impact estimates
            },
        }

    async def run_example(self, duration_minutes: int = 30) -> None:
        """Run conservative trading example."""
        console.print(
            Panel.fit(
                "[bold green]Conservative Market Making Example[/bold green]\n\n"
                "This example demonstrates a low-risk market making setup with:\n"
                "• Wide spreads (0.25% base)\n"
                "• Small position limits (10% max)\n"
                "• Early rebalancing (40% threshold)\n"
                "• Conservative risk management\n"
                f"• Duration: {duration_minutes} minutes",
                title="🛡️ Conservative Setup",
            )
        )

        try:
            # Create mock exchange client for demonstration
            exchange_client = self._create_mock_exchange_client()

            # Create market making engine
            engine = MarketMakingEngineFactory.create_engine(
                exchange_client=exchange_client, symbol=self.symbol, config=self.config
            )

            # Start engine
            await engine.start()

            # Run simulation
            start_time = datetime.now(UTC)
            end_time = start_time + timedelta(minutes=duration_minutes)

            with Live(self._create_status_table(), refresh_per_second=2) as live:
                while datetime.now(UTC) < end_time:
                    # Generate conservative market state
                    market_state = self._generate_conservative_market_state()

                    # Run market making cycle
                    success = await engine.run_single_cycle(market_state)

                    # Update display
                    live.update(self._create_status_table(engine, market_state))

                    # Conservative sleep interval
                    await asyncio.sleep(2.0)

            # Stop engine
            await engine.stop()

            # Display results
            self._display_conservative_results(engine)

        except Exception as e:
            logger.error(f"Error in conservative trading example: {e}")
            console.print(f"[red]Error: {e}[/red]")

    def _create_mock_exchange_client(self) -> Any:
        """Create a mock exchange client for demonstration."""

        class MockExchangeClient:
            def __init__(self):
                self.orders = {}
                self.positions = {}

            async def place_order(self, *args, **kwargs):
                return {"order_id": f"mock_{len(self.orders)}", "status": "PENDING"}

            async def cancel_order(self, *args, **kwargs):
                return {"status": "CANCELLED"}

            async def get_position(self, symbol):
                return {"symbol": symbol, "size": Decimal(0), "side": "NONE"}

        return MockExchangeClient()

    def _generate_conservative_market_state(self) -> MarketState:
        """Generate a conservative market state for demonstration."""
        import random

        current_price = Decimal("3.45") + Decimal(str(random.uniform(-0.02, 0.02)))

        # Create mock OHLCV data
        ohlcv_data = []
        for i in range(50):
            price = current_price + Decimal(str(random.uniform(-0.01, 0.01)))
            ohlcv_data.append(
                {
                    "timestamp": datetime.now(UTC) - timedelta(minutes=i),
                    "open": price,
                    "high": price * Decimal("1.002"),
                    "low": price * Decimal("0.998"),
                    "close": price,
                    "volume": Decimal(str(random.uniform(1000, 5000))),
                }
            )

        # Create mock indicators
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=random.uniform(-0.3, 0.3),  # Conservative range
            cipher_b_wave=random.uniform(-30, 30),  # Limited volatility
            cipher_b_money_flow=random.uniform(45, 55),  # Near neutral
            rsi=random.uniform(40, 60),  # Avoid extremes
            ema_fast=float(current_price * Decimal("1.001")),
            ema_slow=float(current_price * Decimal("0.999")),
        )

        return MarketState(
            symbol=self.symbol,
            current_price=current_price,
            ohlcv_data=ohlcv_data,
            indicators=indicators,
            timestamp=datetime.now(UTC),
        )

    def _create_status_table(
        self,
        engine: MarketMakingEngine | None = None,
        market_state: MarketState | None = None,
    ) -> Table:
        """Create status display table."""
        table = Table(title="Conservative Market Making Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        if engine and market_state:
            status = engine.get_status()
            table.add_row(
                "Engine Status",
                "Running" if status["is_running"] else "Stopped",
                "✅" if status["is_running"] else "❌",
            )
            table.add_row("Cycle Count", str(status["cycle_count"]), "📊")
            table.add_row("Current Price", f"${market_state.current_price:.4f}", "💲")
            table.add_row(
                "Error Count",
                str(status["error_count"]),
                "✅" if status["error_count"] < 3 else "⚠️",
            )
            table.add_row("Spread Config", "25 bps (Conservative)", "🛡️")
            table.add_row("Position Limit", "10% max", "🔒")
        else:
            table.add_row("Status", "Initializing...", "⏳")

        return table

    def _display_conservative_results(self, engine: MarketMakingEngine) -> None:
        """Display conservative trading results."""
        status = engine.get_status()

        console.print(
            Panel(
                f"[bold green]Conservative Trading Results[/bold green]\n\n"
                f"• Total Cycles: {status['cycle_count']}\n"
                f"• Runtime: {status['total_runtime']}\n"
                f"• Errors: {status['error_count']}\n"
                f"• Configuration: Wide spreads, low risk\n"
                f"• Outcome: {'✅ Stable operation' if status['error_count'] < 5 else '⚠️ Some issues detected'}",
                title="🛡️ Conservative Results",
            )
        )


class AggressiveHighFrequencyExample:
    """
    Aggressive high-frequency market making setup example.

    Demonstrates a high-performance market making configuration with:
    - Tight spreads for maximum capture
    - Multiple order levels
    - Fast cycle times
    - Higher position limits
    - Dynamic spread adjustments
    """

    def __init__(self, symbol: str = "SUI-PERP"):
        """Initialize aggressive trading example."""
        self.symbol = symbol
        self.config = self._create_aggressive_config()

    def _create_aggressive_config(self) -> dict[str, Any]:
        """Create aggressive market making configuration."""
        return {
            "cycle_interval": 0.2,  # Very fast cycles (200ms)
            "max_errors_per_hour": 100,  # Higher error tolerance
            "emergency_stop_threshold": 20,  # More lenient emergency stop
            "max_position_value": 25000,  # Larger position limits
            "strategy": {
                "base_spread_bps": 3,  # Very tight 0.03% spreads
                "max_spread_bps": 20,  # Quick adaptation to conditions
                "min_spread_bps": 1,  # Extremely tight minimum
                "order_levels": 5,  # Many levels for depth
                "max_position_pct": 40,  # Large position tolerance
                "bias_adjustment_factor": 0.8,  # Strong bias response
            },
            "order_manager": {
                "max_orders_per_side": 8,  # Many orders
                "order_refresh_threshold": 0.1,  # Frequent refresh
                "min_order_size": 5,  # Small minimum for flexibility
            },
            "inventory": {
                "max_position_limit": 2000,
                "rebalancing_threshold": 0.8,  # Late rebalancing
                "target_inventory": 0.0,
            },
            "performance": {
                "tracking_window_hours": 6,  # Shorter window for HFT
                "alert_thresholds": {
                    "max_drawdown": 0.08,  # Higher drawdown tolerance
                    "min_profit_margin": 0.0005,  # Very tight margins
                },
            },
            "spread_calculator": {
                "volatility_lookback": 10,  # Short lookback for responsiveness
                "liquidity_factor": 0.05,  # Lower liquidity requirements
                "market_impact_factor": 0.02,  # Aggressive impact estimates
            },
        }

    async def run_example(self, duration_minutes: int = 15) -> None:
        """Run aggressive high-frequency example."""
        console.print(
            Panel.fit(
                "[bold red]Aggressive High-Frequency Market Making[/bold red]\n\n"
                "This example demonstrates high-performance trading with:\n"
                "• Ultra-tight spreads (0.03% base)\n"
                "• Fast cycle times (200ms)\n"
                "• Multiple order levels (5 levels)\n"
                "• Large position limits (40% max)\n"
                "• Dynamic spread adjustments\n"
                f"• Duration: {duration_minutes} minutes",
                title="⚡ Aggressive HFT Setup",
            )
        )

        try:
            # Create mock high-performance exchange client
            exchange_client = self._create_hft_mock_client()

            # Create market making engine
            engine = MarketMakingEngineFactory.create_engine(
                exchange_client=exchange_client, symbol=self.symbol, config=self.config
            )

            # Start engine
            await engine.start()

            # Run high-frequency simulation
            start_time = datetime.now(UTC)
            end_time = start_time + timedelta(minutes=duration_minutes)
            cycle_count = 0

            with Live(self._create_hft_dashboard(), refresh_per_second=10) as live:
                while datetime.now(UTC) < end_time:
                    # Generate aggressive market state with high volatility
                    market_state = self._generate_hft_market_state()

                    # Run fast market making cycle
                    success = await engine.run_single_cycle(market_state)
                    cycle_count += 1

                    # Update high-frequency display
                    live.update(
                        self._create_hft_dashboard(engine, market_state, cycle_count)
                    )

                    # Fast cycle time
                    await asyncio.sleep(0.2)

            # Stop engine
            await engine.stop()

            # Display HFT results
            self._display_hft_results(engine, cycle_count)

        except Exception as e:
            logger.error(f"Error in aggressive HFT example: {e}")
            console.print(f"[red]Error: {e}[/red]")

    def _create_hft_mock_client(self) -> Any:
        """Create a high-performance mock exchange client."""

        class HighFrequencyMockClient:
            def __init__(self):
                self.orders = {}
                self.fills = 0
                self.latency_ms = 2  # Simulate low latency

            async def place_order(self, *args, **kwargs):
                # Simulate fast order placement
                await asyncio.sleep(0.002)  # 2ms latency
                order_id = f"hft_{len(self.orders)}_{datetime.now(UTC).microsecond}"
                return {
                    "order_id": order_id,
                    "status": "PENDING",
                    "latency_ms": self.latency_ms,
                }

            async def cancel_order(self, *args, **kwargs):
                await asyncio.sleep(0.001)  # 1ms cancellation
                return {"status": "CANCELLED", "latency_ms": 1}

            async def get_position(self, symbol):
                return {
                    "symbol": symbol,
                    "size": Decimal(str(self.fills * 0.1)),
                    "side": "LONG" if self.fills > 0 else "NONE",
                }

        return HighFrequencyMockClient()

    def _generate_hft_market_state(self) -> MarketState:
        """Generate high-frequency market state with volatility."""
        import random

        # More volatile price movements for HFT
        base_price = Decimal("3.45")
        volatility = random.uniform(0.0005, 0.003)  # 0.05% to 0.3% movements
        current_price = base_price + Decimal(
            str(random.uniform(-volatility, volatility))
        )

        # Create high-frequency OHLCV data
        ohlcv_data = []
        for i in range(200):  # More data points
            price = current_price + Decimal(str(random.gauss(0, volatility / 2)))
            ohlcv_data.append(
                {
                    "timestamp": datetime.now(UTC) - timedelta(seconds=i),
                    "open": price,
                    "high": price * Decimal("1.0002"),
                    "low": price * Decimal("0.9998"),
                    "close": price,
                    "volume": Decimal(str(random.uniform(100, 1000))),
                }
            )

        # More dynamic indicators for HFT
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=random.uniform(-1.0, 1.0),  # Full range
            cipher_b_wave=random.uniform(-100, 100),  # High volatility
            cipher_b_money_flow=random.uniform(10, 90),  # Wide range
            rsi=random.uniform(25, 75),  # Active RSI
            ema_fast=float(
                current_price * Decimal(str(1 + random.uniform(-0.002, 0.002)))
            ),
            ema_slow=float(
                current_price * Decimal(str(1 + random.uniform(-0.001, 0.001)))
            ),
        )

        return MarketState(
            symbol=self.symbol,
            current_price=current_price,
            ohlcv_data=ohlcv_data,
            indicators=indicators,
            timestamp=datetime.now(UTC),
        )

    def _create_hft_dashboard(
        self,
        engine: MarketMakingEngine | None = None,
        market_state: MarketState | None = None,
        cycle_count: int = 0,
    ) -> Table:
        """Create high-frequency trading dashboard."""
        import random

        table = Table(title="🚀 High-Frequency Market Making Dashboard")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="magenta", width=15)
        table.add_column("Performance", style="green", width=15)
        table.add_column("Status", style="yellow", width=10)

        if engine and market_state:
            status = engine.get_status()
            cycles_per_second = cycle_count / max(
                1,
                (
                    datetime.now(UTC) - datetime.now(UTC).replace(second=0)
                ).total_seconds(),
            )

            table.add_row(
                "Cycle Frequency",
                f"{cycles_per_second:.1f}/sec",
                "🚀" if cycles_per_second > 3 else "⚡",
                "FAST",
            )
            table.add_row(
                "Total Cycles", str(cycle_count), f"{cycle_count // 100}x100", "📊"
            )
            table.add_row(
                "Price",
                f"${market_state.current_price:.6f}",
                f"Δ{random.uniform(-0.001, 0.001):.4f}",
                "💲",
            )
            table.add_row("Spread Config", "3 bps (Tight)", "Ultra-narrow", "⚡")
            table.add_row("Order Levels", "5 levels", "Deep book", "📈")
            table.add_row("Position Limit", "40% max", "Aggressive", "🎯")
            table.add_row(
                "Errors",
                str(status["error_count"]),
                "Stable" if status["error_count"] < 10 else "Monitor",
                "✅" if status["error_count"] < 10 else "⚠️",
            )
        else:
            table.add_row("Status", "Initializing HFT...", "Warming up", "⏳")

        return table

    def _display_hft_results(
        self, engine: MarketMakingEngine, cycle_count: int
    ) -> None:
        """Display high-frequency trading results."""
        status = engine.get_status()
        cycles_per_minute = cycle_count / max(
            1,
            (
                int(status["total_runtime"].split(":")[1])
                if ":" in str(status["total_runtime"])
                else 1
            ),
        )

        console.print(
            Panel(
                f"[bold red]High-Frequency Trading Results[/bold red]\n\n"
                f"• Total Cycles: {cycle_count:,}\n"
                f"• Cycles/Minute: {cycles_per_minute:.0f}\n"
                f"• Runtime: {status['total_runtime']}\n"
                f"• Errors: {status['error_count']}\n"
                f"• Configuration: Ultra-tight spreads, high frequency\n"
                f"• Performance: {'🚀 Excellent throughput' if cycles_per_minute > 200 else '⚡ Good performance'}\n"
                f"• Stability: {'✅ Stable' if status['error_count'] < 20 else '⚠️ Some issues'}",
                title="⚡ HFT Results",
            )
        )


class RiskManagementExample:
    """
    Risk management configuration examples.

    Demonstrates various risk management scenarios including:
    - Position limits and controls
    - Emergency stop procedures
    - Inventory rebalancing
    - Loss limits and drawdown controls
    - Volatility-based adjustments
    """

    async def run_risk_scenarios(self) -> None:
        """Run various risk management scenarios."""
        console.print(
            Panel.fit(
                "[bold yellow]Risk Management Examples[/bold yellow]\n\n"
                "Demonstrating various risk scenarios:\n"
                "1. Position limit breaches\n"
                "2. Emergency stop triggers\n"
                "3. Inventory rebalancing\n"
                "4. Volatility adjustments\n"
                "5. Recovery procedures",
                title="⚠️ Risk Management",
            )
        )

        await self._demo_position_limits()
        await self._demo_emergency_stop()
        await self._demo_inventory_rebalancing()
        await self._demo_volatility_adjustment()
        await self._demo_recovery_procedures()

    async def _demo_position_limits(self) -> None:
        """Demonstrate position limit controls."""
        console.print("\n[bold cyan]1. Position Limit Controls[/bold cyan]")

        # Create configuration with strict position limits
        config = {
            "max_position_value": 1000,  # Low limit for demonstration
            "strategy": {
                "max_position_pct": 5,  # Very low position size
            },
            "inventory": {
                "max_position_limit": 100,
                "rebalancing_threshold": 0.3,  # Early rebalancing
            },
        }

        console.print("✅ Configuration: Max position $1,000, 5% max size")
        console.print("✅ Early rebalancing at 30% threshold")
        console.print("✅ Position monitoring active")

        # Simulate position limit breach
        console.print("\n🚨 Simulating position limit breach...")
        await asyncio.sleep(1)
        console.print("⚠️ Position value: $1,200 (120% of limit)")
        console.print("🛑 Action: Emergency rebalancing triggered")
        console.print("✅ Position reduced to $800 (80% of limit)")

    async def _demo_emergency_stop(self) -> None:
        """Demonstrate emergency stop procedures."""
        console.print("\n[bold red]2. Emergency Stop Procedures[/bold red]")

        console.print("🚨 Emergency conditions detected:")
        console.print("   • Error rate: 15 errors in 10 minutes")
        console.print("   • Position size: 150% of limit")
        console.print("   • Daily loss: -$500 (10% of capital)")

        await asyncio.sleep(1)
        console.print("\n🛑 EMERGENCY STOP ACTIVATED")
        console.print("✅ All orders cancelled")
        console.print("✅ Engine stopped")
        console.print("✅ Positions logged for manual review")
        console.print("✅ Risk manager notified")

    async def _demo_inventory_rebalancing(self) -> None:
        """Demonstrate inventory rebalancing."""
        console.print("\n[bold green]3. Inventory Rebalancing[/bold green]")

        # Create inventory manager for demonstration
        mock_client = self._create_mock_client()
        inventory_manager = InventoryManager(
            symbol="SUI-PERP",
            exchange_client=mock_client,
            config={"rebalancing_threshold": 0.5},
        )

        # Simulate inventory imbalance
        console.print("📊 Current inventory status:")
        console.print("   • Net position: +150 SUI (75% imbalance)")
        console.print("   • Risk score: 0.85 (HIGH)")
        console.print("   • Threshold: 0.50 (EXCEEDED)")

        await asyncio.sleep(1)
        console.print("\n🔄 Rebalancing action required:")
        console.print("✅ Action: SELL 75 SUI (reduce to 37.5% imbalance)")
        console.print("✅ Priority: HIGH (immediate execution)")
        console.print("✅ Expected risk reduction: 0.85 → 0.42")

    async def _demo_volatility_adjustment(self) -> None:
        """Demonstrate volatility-based adjustments."""
        console.print("\n[bold blue]4. Volatility Adjustments[/bold blue]")

        # Simulate different volatility scenarios
        scenarios = [
            ("Low volatility", 0.005, "Tighten spreads", "3 bps → 2 bps"),
            ("Normal volatility", 0.02, "Maintain spreads", "10 bps (no change)"),
            ("High volatility", 0.08, "Widen spreads", "10 bps → 25 bps"),
            ("Extreme volatility", 0.15, "Emergency mode", "Stop trading"),
        ]

        for scenario, vol, action, detail in scenarios:
            console.print(f"\n📈 {scenario}: {vol:.1%} volatility")
            console.print(f"   Action: {action}")
            console.print(f"   Detail: {detail}")
            await asyncio.sleep(0.5)

    async def _demo_recovery_procedures(self) -> None:
        """Demonstrate recovery procedures."""
        console.print("\n[bold magenta]5. Recovery Procedures[/bold magenta]")

        console.print("🔧 Post-emergency recovery checklist:")
        await asyncio.sleep(0.5)
        console.print("✅ 1. Verify system health")
        await asyncio.sleep(0.5)
        console.print("✅ 2. Check position accuracy")
        await asyncio.sleep(0.5)
        console.print("✅ 3. Validate configuration")
        await asyncio.sleep(0.5)
        console.print("✅ 4. Test order placement")
        await asyncio.sleep(0.5)
        console.print("✅ 5. Gradual restart with conservative settings")
        await asyncio.sleep(0.5)
        console.print("✅ 6. Monitor for 15 minutes before full operation")

        console.print("\n🟢 Recovery complete - System ready for normal operation")

    def _create_mock_client(self):
        """Create a mock client for demonstrations."""

        class MockClient:
            async def get_position(self, symbol):
                return {"symbol": symbol, "size": Decimal(150), "side": "LONG"}

        return MockClient()


class MultiSymbolExample:
    """
    Multi-symbol market making configuration example.

    Demonstrates running market making across multiple symbols with:
    - Different configurations per symbol
    - Cross-symbol risk management
    - Portfolio-level monitoring
    - Symbol-specific performance tracking
    """

    def __init__(self):
        """Initialize multi-symbol example."""
        self.symbols = ["SUI-PERP", "BTC-PERP", "ETH-PERP"]
        self.engines = {}
        self.symbol_configs = self._create_symbol_configs()

    def _create_symbol_configs(self) -> dict[str, dict[str, Any]]:
        """Create different configurations for each symbol."""
        return {
            "SUI-PERP": {
                "strategy": {
                    "base_spread_bps": 8,  # Tight spreads for high volume
                    "order_levels": 4,
                    "max_position_pct": 30,
                },
                "max_position_value": 15000,
                "description": "High-volume, tight spreads",
            },
            "BTC-PERP": {
                "strategy": {
                    "base_spread_bps": 5,  # Very tight for BTC
                    "order_levels": 5,
                    "max_position_pct": 25,
                },
                "max_position_value": 50000,  # Larger for BTC
                "description": "Premium pair, maximum liquidity",
            },
            "ETH-PERP": {
                "strategy": {
                    "base_spread_bps": 6,  # Moderate spreads
                    "order_levels": 3,
                    "max_position_pct": 20,
                },
                "max_position_value": 25000,
                "description": "Balanced approach",
            },
        }

    async def run_multi_symbol_example(self, duration_minutes: int = 20) -> None:
        """Run multi-symbol market making example."""
        console.print(
            Panel.fit(
                "[bold purple]Multi-Symbol Market Making[/bold purple]\n\n"
                "Running market making across multiple symbols:\n"
                f"• Symbols: {', '.join(self.symbols)}\n"
                "• Different configurations per symbol\n"
                "• Portfolio-level risk management\n"
                "• Cross-symbol monitoring\n"
                f"• Duration: {duration_minutes} minutes",
                title="🎯 Multi-Symbol Setup",
            )
        )

        try:
            # Initialize engines for each symbol
            for symbol in self.symbols:
                engine = await self._create_symbol_engine(symbol)
                self.engines[symbol] = engine
                await engine.start()
                console.print(f"✅ Started engine for {symbol}")

            # Run multi-symbol simulation
            start_time = datetime.now(UTC)
            end_time = start_time + timedelta(minutes=duration_minutes)

            with Live(
                self._create_multi_symbol_dashboard(), refresh_per_second=2
            ) as live:
                while datetime.now(UTC) < end_time:
                    # Run all engines concurrently
                    tasks = []
                    for symbol, engine in self.engines.items():
                        market_state = self._generate_symbol_market_state(symbol)
                        tasks.append(engine.run_single_cycle(market_state))

                    # Execute all cycles concurrently
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Update dashboard
                    live.update(self._create_multi_symbol_dashboard())

                    await asyncio.sleep(1.0)

            # Stop all engines
            for symbol, engine in self.engines.items():
                await engine.stop()
                console.print(f"✅ Stopped engine for {symbol}")

            # Display multi-symbol results
            self._display_multi_symbol_results()

        except Exception as e:
            logger.error(f"Error in multi-symbol example: {e}")
            console.print(f"[red]Error: {e}[/red]")

    async def _create_symbol_engine(self, symbol: str) -> MarketMakingEngine:
        """Create market making engine for a specific symbol."""
        config = self.symbol_configs[symbol].copy()
        config.update(
            {
                "cycle_interval": 1.0,
                "max_errors_per_hour": 30,
                "emergency_stop_threshold": 8,
            }
        )

        mock_client = self._create_symbol_mock_client(symbol)

        return MarketMakingEngineFactory.create_engine(
            exchange_client=mock_client, symbol=symbol, config=config
        )

    def _create_symbol_mock_client(self, symbol: str):
        """Create symbol-specific mock client."""

        class SymbolMockClient:
            def __init__(self, symbol):
                self.symbol = symbol
                self.orders = {}

            async def place_order(self, *args, **kwargs):
                return {"order_id": f"{symbol}_{len(self.orders)}", "status": "PENDING"}

            async def cancel_order(self, *args, **kwargs):
                return {"status": "CANCELLED"}

            async def get_position(self, symbol):
                return {"symbol": symbol, "size": Decimal(0), "side": "NONE"}

        return SymbolMockClient(symbol)

    def _generate_symbol_market_state(self, symbol: str) -> MarketState:
        """Generate market state for specific symbol."""
        import random

        # Different base prices for different symbols
        base_prices = {
            "SUI-PERP": Decimal("3.45"),
            "BTC-PERP": Decimal("67500.00"),
            "ETH-PERP": Decimal("3800.00"),
        }

        base_price = base_prices.get(symbol, Decimal("100.00"))
        current_price = base_price + base_price * Decimal(
            str(random.uniform(-0.01, 0.01))
        )

        # Create mock OHLCV data
        ohlcv_data = []
        for i in range(30):
            price = current_price + current_price * Decimal(
                str(random.uniform(-0.005, 0.005))
            )
            ohlcv_data.append(
                {
                    "timestamp": datetime.now(UTC) - timedelta(minutes=i),
                    "open": price,
                    "high": price * Decimal("1.001"),
                    "low": price * Decimal("0.999"),
                    "close": price,
                    "volume": Decimal(str(random.uniform(500, 2000))),
                }
            )

        # Symbol-specific indicators
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=random.uniform(-0.5, 0.5),
            cipher_b_wave=random.uniform(-50, 50),
            cipher_b_money_flow=random.uniform(35, 65),
            rsi=random.uniform(35, 65),
            ema_fast=float(current_price * Decimal("1.0005")),
            ema_slow=float(current_price * Decimal("0.9995")),
        )

        return MarketState(
            symbol=symbol,
            current_price=current_price,
            ohlcv_data=ohlcv_data,
            indicators=indicators,
            timestamp=datetime.now(UTC),
        )

    def _create_multi_symbol_dashboard(self) -> Table:
        """Create multi-symbol dashboard."""
        table = Table(title="🎯 Multi-Symbol Market Making Dashboard")
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Status", style="green", width=10)
        table.add_column("Cycles", style="magenta", width=8)
        table.add_column("Config", style="yellow", width=20)
        table.add_column("Performance", style="blue", width=15)

        for symbol in self.symbols:
            if symbol in self.engines:
                engine = self.engines[symbol]
                status = engine.get_status()
                config = self.symbol_configs[symbol]

                table.add_row(
                    symbol,
                    "🟢 Active" if status["is_running"] else "🔴 Stopped",
                    str(status["cycle_count"]),
                    config["description"],
                    f"Errors: {status['error_count']}",
                )
            else:
                table.add_row(symbol, "⏳ Init", "0", "Configuring...", "Starting...")

        return table

    def _display_multi_symbol_results(self) -> None:
        """Display multi-symbol results."""
        console.print("\n" + "=" * 60)
        console.print("[bold purple]Multi-Symbol Results Summary[/bold purple]")
        console.print("=" * 60)

        total_cycles = 0
        total_errors = 0

        for symbol, engine in self.engines.items():
            status = engine.get_status()
            config = self.symbol_configs[symbol]

            console.print(f"\n[cyan]{symbol}[/cyan]:")
            console.print(f"  • Configuration: {config['description']}")
            console.print(f"  • Cycles: {status['cycle_count']}")
            console.print(f"  • Errors: {status['error_count']}")
            console.print(
                f"  • Status: {'✅ Success' if status['error_count'] < 5 else '⚠️ Issues'}"
            )

            total_cycles += status["cycle_count"]
            total_errors += status["error_count"]

        console.print("\n[bold green]Portfolio Summary:[/bold green]")
        console.print(f"  • Total Cycles: {total_cycles:,}")
        console.print(f"  • Total Errors: {total_errors}")
        console.print(
            f"  • Success Rate: {((total_cycles - total_errors) / max(total_cycles, 1)) * 100:.1f}%"
        )


class PaperTradingSimulation:
    """
    Paper trading simulation examples.

    Demonstrates market making in paper trading mode with:
    - Safe testing environment
    - Real market data simulation
    - Performance tracking without risk
    - Configuration validation
    - Strategy testing
    """

    async def run_paper_trading_demo(self, duration_minutes: int = 25) -> None:
        """Run comprehensive paper trading demonstration."""
        console.print(
            Panel.fit(
                "[bold blue]Paper Trading Simulation[/bold blue]\n\n"
                "Safe testing environment demonstrating:\n"
                "• Real market data simulation\n"
                "• Full strategy execution (no real trades)\n"
                "• Performance tracking and metrics\n"
                "• Risk management validation\n"
                "• Configuration testing\n"
                f"• Duration: {duration_minutes} minutes",
                title="📊 Paper Trading Mode",
            )
        )

        # Create paper trading configuration
        config = self._create_paper_trading_config()

        # Display configuration
        self._display_paper_config(config)

        # Run simulation
        await self._run_paper_simulation(config, duration_minutes)

    def _create_paper_trading_config(self) -> dict[str, Any]:
        """Create paper trading configuration."""
        return {
            "paper_trading": True,
            "cycle_interval": 1.0,
            "max_errors_per_hour": 25,
            "emergency_stop_threshold": 8,
            "max_position_value": 10000,
            "strategy": {
                "base_spread_bps": 12,
                "max_spread_bps": 40,
                "min_spread_bps": 6,
                "order_levels": 3,
                "max_position_pct": 20,
                "bias_adjustment_factor": 0.4,
            },
            "order_manager": {
                "max_orders_per_side": 4,
                "order_refresh_threshold": 0.5,
                "min_order_size": 15,
                "simulate_fills": True,
                "fill_probability": 0.3,  # 30% chance of fills
            },
            "inventory": {
                "max_position_limit": 800,
                "rebalancing_threshold": 0.6,
                "target_inventory": 0.0,
                "track_paper_positions": True,
            },
            "performance": {
                "paper_mode": True,
                "track_hypothetical_pnl": True,
                "simulate_fees": True,
                "simulate_slippage": True,
            },
        }

    def _display_paper_config(self, config: dict[str, Any]) -> None:
        """Display paper trading configuration."""
        console.print("\n[bold yellow]Paper Trading Configuration[/bold yellow]")

        config_table = Table()
        config_table.add_column("Category", style="cyan")
        config_table.add_column("Setting", style="magenta")
        config_table.add_column("Value", style="green")

        config_table.add_row("Mode", "Paper Trading", "✅ SAFE")
        config_table.add_row(
            "Spread", "Base Spread", f"{config['strategy']['base_spread_bps']} bps"
        )
        config_table.add_row(
            "Position", "Max Position", f"{config['strategy']['max_position_pct']}%"
        )
        config_table.add_row(
            "Orders",
            "Max Orders/Side",
            str(config["order_manager"]["max_orders_per_side"]),
        )
        config_table.add_row(
            "Simulation",
            "Fill Probability",
            f"{config['order_manager']['fill_probability'] * 100:.0f}%",
        )
        config_table.add_row("Tracking", "P&L Simulation", "✅ Enabled")

        console.print(config_table)

    async def _run_paper_simulation(
        self, config: dict[str, Any], duration_minutes: int
    ) -> None:
        """Run paper trading simulation."""
        # Create paper trading engine
        mock_client = self._create_paper_trading_client()
        engine = MarketMakingEngineFactory.create_engine(
            exchange_client=mock_client, symbol="SUI-PERP", config=config
        )

        # Initialize paper trading tracker
        paper_tracker = PaperTradingTracker()

        try:
            await engine.start()

            start_time = datetime.now(UTC)
            end_time = start_time + timedelta(minutes=duration_minutes)

            with Live(
                self._create_paper_dashboard(paper_tracker), refresh_per_second=1
            ) as live:
                while datetime.now(UTC) < end_time:
                    # Generate realistic market state
                    market_state = self._generate_realistic_market_state()

                    # Run paper trading cycle
                    success = await engine.run_single_cycle(market_state)

                    # Update paper trading tracker
                    paper_tracker.update(market_state, success)

                    # Simulate order fills
                    await self._simulate_paper_fills(paper_tracker, market_state)

                    # Update display
                    live.update(self._create_paper_dashboard(paper_tracker, engine))

                    await asyncio.sleep(1.0)

            await engine.stop()

            # Display final paper trading results
            self._display_paper_results(paper_tracker, engine)

        except Exception as e:
            logger.error(f"Error in paper trading simulation: {e}")
            console.print(f"[red]Paper Trading Error: {e}[/red]")

    def _create_paper_trading_client(self):
        """Create paper trading mock client."""

        class PaperTradingClient:
            def __init__(self):
                self.paper_orders = {}
                self.paper_position = Decimal(0)
                self.order_counter = 0

            async def place_order(self, symbol, side, type, quantity, price, **kwargs):
                self.order_counter += 1
                order_id = f"paper_{self.order_counter}"
                order = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "price": price,
                    "status": "PENDING",
                    "timestamp": datetime.now(UTC),
                }
                self.paper_orders[order_id] = order
                return order

            async def cancel_order(self, order_id):
                if order_id in self.paper_orders:
                    self.paper_orders[order_id]["status"] = "CANCELLED"
                    return {"status": "CANCELLED"}
                return {"status": "NOT_FOUND"}

            async def get_position(self, symbol):
                return {
                    "symbol": symbol,
                    "size": self.paper_position,
                    "side": (
                        "LONG"
                        if self.paper_position > 0
                        else "SHORT" if self.paper_position < 0 else "NONE"
                    ),
                }

            def simulate_fill(self, order_id, fill_quantity):
                """Simulate order fill."""
                if order_id in self.paper_orders:
                    order = self.paper_orders[order_id]
                    if order["side"] == "BUY":
                        self.paper_position += fill_quantity
                    else:
                        self.paper_position -= fill_quantity
                    order["status"] = "FILLED"
                    return True
                return False

        return PaperTradingClient()

    def _generate_realistic_market_state(self) -> MarketState:
        """Generate realistic market state for paper trading."""
        import random

        # More realistic price movements
        if not hasattr(self, "_last_price"):
            self._last_price = Decimal("3.45")

        # Random walk with realistic volatility
        change_pct = random.gauss(0, 0.002)  # 0.2% standard deviation
        self._last_price = max(
            Decimal("1.00"), self._last_price * (Decimal(1) + Decimal(str(change_pct)))
        )

        # Create realistic OHLCV data
        ohlcv_data = []
        for i in range(60):  # 1 hour of data
            price = self._last_price + self._last_price * Decimal(
                str(random.gauss(0, 0.001))
            )
            ohlcv_data.append(
                {
                    "timestamp": datetime.now(UTC) - timedelta(minutes=i),
                    "open": price,
                    "high": price * Decimal("1.0015"),
                    "low": price * Decimal("0.9985"),
                    "close": price,
                    "volume": Decimal(str(random.uniform(1000, 5000))),
                }
            )

        # Realistic indicators
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=random.gauss(0, 0.3),
            cipher_b_wave=random.gauss(0, 25),
            cipher_b_money_flow=random.gauss(50, 15),
            rsi=random.gauss(50, 12),
            ema_fast=float(self._last_price * Decimal("1.0002")),
            ema_slow=float(self._last_price * Decimal("0.9998")),
        )

        return MarketState(
            symbol="SUI-PERP",
            current_price=self._last_price,
            ohlcv_data=ohlcv_data,
            indicators=indicators,
            timestamp=datetime.now(UTC),
        )

    async def _simulate_paper_fills(self, tracker, market_state):
        """Simulate paper trading fills."""
        import random

        # Simulate some fills based on market conditions
        if random.random() < 0.15:  # 15% chance of fill each cycle
            side = random.choice(["BUY", "SELL"])
            quantity = Decimal(str(random.uniform(5, 50)))
            price = market_state.current_price

            # Add to tracker
            fill_info = {
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now(UTC),
            }
            tracker.add_fill(fill_info)

    def _create_paper_dashboard(self, tracker, engine=None):
        """Create paper trading dashboard."""
        table = Table(title="📊 Paper Trading Dashboard")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="magenta", width=15)
        table.add_column("Status", style="green", width=15)

        table.add_row("Trading Mode", "PAPER", "✅ SAFE")
        table.add_row(
            "Current Price",
            (
                f"${tracker.current_price:.4f}"
                if hasattr(tracker, "current_price")
                else "$0.0000"
            ),
            "💲",
        )
        table.add_row(
            "Paper Position",
            (
                f"{tracker.paper_position:.2f} SUI"
                if hasattr(tracker, "paper_position")
                else "0.00 SUI"
            ),
            "📊",
        )
        table.add_row(
            "Hypothetical P&L",
            (
                f"${tracker.hypothetical_pnl:.2f}"
                if hasattr(tracker, "hypothetical_pnl")
                else "$0.00"
            ),
            "🟢" if getattr(tracker, "hypothetical_pnl", 0) >= 0 else "🔴",
        )
        table.add_row(
            "Total Fills",
            str(tracker.total_fills) if hasattr(tracker, "total_fills") else "0",
            "📈",
        )
        table.add_row("Strategy", "Market Making", "🎯")

        if engine:
            status = engine.get_status()
            table.add_row("Engine Cycles", str(status["cycle_count"]), "⚡")
            table.add_row(
                "Errors",
                str(status["error_count"]),
                "✅" if status["error_count"] < 5 else "⚠️",
            )

        return table

    def _display_paper_results(self, tracker, engine):
        """Display paper trading simulation results."""
        status = engine.get_status()

        console.print(
            Panel(
                f"[bold blue]Paper Trading Simulation Results[/bold blue]\n\n"
                f"• Mode: Paper Trading (100% safe)\n"
                f"• Duration: {status['total_runtime']}\n"
                f"• Total Cycles: {status['cycle_count']}\n"
                f"• Simulated Fills: {getattr(tracker, 'total_fills', 0)}\n"
                f"• Hypothetical P&L: ${getattr(tracker, 'hypothetical_pnl', 0):.2f}\n"
                f"• Final Position: {getattr(tracker, 'paper_position', 0):.2f} SUI\n"
                f"• Strategy Validation: {'✅ Passed' if status['error_count'] < 5 else '⚠️ Issues detected'}\n"
                f"• Ready for Live Trading: {'✅ Yes' if status['error_count'] < 3 else '❌ Needs review'}",
                title="📊 Paper Trading Results",
            )
        )


class PaperTradingTracker:
    """Helper class to track paper trading simulation."""

    def __init__(self):
        self.paper_position = Decimal(0)
        self.hypothetical_pnl = Decimal(0)
        self.total_fills = 0
        self.current_price = Decimal("3.45")
        self.fills = []

    def update(self, market_state, success):
        """Update tracker with market state."""
        self.current_price = market_state.current_price

    def add_fill(self, fill_info):
        """Add a simulated fill."""
        self.fills.append(fill_info)
        self.total_fills += 1

        if fill_info["side"] == "BUY":
            self.paper_position += fill_info["quantity"]
        else:
            self.paper_position -= fill_info["quantity"]

        # Simple P&L calculation (would be more complex in reality)
        self.hypothetical_pnl += (
            Decimal(str(0.1)) * fill_info["quantity"]
        )  # Assume small profit per fill


class EmergencyRecoveryExample:
    """
    Emergency stop and recovery examples.

    Demonstrates emergency scenarios and recovery procedures:
    - Emergency stop triggers
    - System recovery procedures
    - Data validation and cleanup
    - Gradual restart protocols
    - Health monitoring
    """

    async def run_emergency_scenarios(self) -> None:
        """Run emergency and recovery scenarios."""
        console.print(
            Panel.fit(
                "[bold red]Emergency Stop & Recovery Examples[/bold red]\n\n"
                "Demonstrating emergency procedures:\n"
                "1. Emergency stop triggers\n"
                "2. System shutdown procedures\n"
                "3. Recovery validation\n"
                "4. Gradual restart protocols\n"
                "5. Health monitoring",
                title="🚨 Emergency Procedures",
            )
        )

        await self._demo_emergency_triggers()
        await self._demo_shutdown_procedures()
        await self._demo_system_recovery()
        await self._demo_health_monitoring()

    async def _demo_emergency_triggers(self) -> None:
        """Demonstrate emergency stop triggers."""
        console.print("\n[bold red]🚨 Emergency Stop Triggers[/bold red]")

        triggers = [
            ("High Error Rate", "20 errors in 5 minutes", "System instability"),
            ("Position Breach", "Position 200% of limit", "Risk management"),
            ("Large Loss", "Daily loss exceeds -10%", "Capital protection"),
            ("Market Crash", "Price drops >15% in 1 minute", "Extreme volatility"),
            ("Network Issues", "Connection failures >30 seconds", "Technical problems"),
        ]

        for trigger, condition, reason in triggers:
            console.print(f"\n🔴 {trigger}:")
            console.print(f"   Condition: {condition}")
            console.print(f"   Reason: {reason}")
            console.print("   Action: EMERGENCY STOP")
            await asyncio.sleep(0.8)

    async def _demo_shutdown_procedures(self) -> None:
        """Demonstrate shutdown procedures."""
        console.print("\n[bold yellow]🛑 Emergency Shutdown Procedures[/bold yellow]")

        steps = [
            "Stop accepting new orders",
            "Cancel all pending orders",
            "Log current positions",
            "Save system state",
            "Notify risk management",
            "Generate emergency report",
            "Set system to safe mode",
        ]

        for i, step in enumerate(steps, 1):
            console.print(f"Step {i}: {step}")
            await asyncio.sleep(0.5)
            console.print("   ✅ Completed")
            await asyncio.sleep(0.3)

        console.print("\n🟢 Emergency shutdown completed successfully")

    async def _demo_system_recovery(self) -> None:
        """Demonstrate system recovery procedures."""
        console.print("\n[bold green]🔧 System Recovery Procedures[/bold green]")

        recovery_steps = [
            ("System Health Check", "Verify all components operational"),
            ("Position Validation", "Confirm position accuracy"),
            ("Configuration Review", "Validate all settings"),
            ("Network Connectivity", "Test exchange connections"),
            ("Order Placement Test", "Small test orders"),
            ("Risk Management", "Verify risk controls active"),
            ("Monitoring Setup", "Enable enhanced monitoring"),
            ("Gradual Restart", "Slowly increase activity"),
        ]

        for step, description in recovery_steps:
            console.print(f"\n🔍 {step}:")
            console.print(f"   {description}")
            await asyncio.sleep(0.5)

            # Simulate check
            if step == "Order Placement Test":
                console.print("   📝 Placing test order...")
                await asyncio.sleep(1)
                console.print("   ✅ Test order successful")
            else:
                console.print("   ✅ Check passed")

            await asyncio.sleep(0.3)

        console.print("\n🟢 System recovery completed - Ready for normal operation")

    async def _demo_health_monitoring(self) -> None:
        """Demonstrate enhanced health monitoring."""
        console.print("\n[bold blue]📊 Enhanced Health Monitoring[/bold blue]")

        # Create health monitoring dashboard
        health_table = Table(title="System Health Dashboard")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Metrics", style="magenta")
        health_table.add_column("Alert Level", style="yellow")

        components = [
            ("Market Making Engine", "✅ Operational", "Cycles: 1,234", "🟢 Normal"),
            ("Order Manager", "✅ Active", "Orders: 12 active", "🟢 Normal"),
            ("Risk Manager", "✅ Monitoring", "Risk Score: 0.3", "🟢 Low"),
            ("Network Connection", "✅ Stable", "Latency: 15ms", "🟢 Excellent"),
            ("Position Manager", "✅ Tracking", "Position: +45 SUI", "🟡 Monitor"),
            ("Performance Monitor", "✅ Recording", "P&L: +$23.45", "🟢 Positive"),
        ]

        for component, status, metrics, alert in components:
            health_table.add_row(component, status, metrics, alert)

        console.print(health_table)

        console.print("\n🔄 Continuous monitoring active:")
        console.print("   • Health checks every 30 seconds")
        console.print("   • Performance metrics every 5 minutes")
        console.print("   • Risk assessment every 2 minutes")
        console.print("   • Alert system: ACTIVE")


# CLI Interface
@click.command()
@click.option(
    "--profile",
    type=click.Choice(["conservative", "aggressive", "paper"]),
    default="conservative",
    help="Trading profile to demonstrate",
)
@click.option("--multi-symbol", is_flag=True, help="Run multi-symbol example")
@click.option("--paper-trading", is_flag=True, help="Run paper trading simulation")
@click.option("--risk-management", is_flag=True, help="Demo risk management scenarios")
@click.option("--emergency-demo", is_flag=True, help="Demo emergency procedures")
@click.option("--duration", default=15, help="Duration in minutes for examples")
def main(
    profile, multi_symbol, paper_trading, risk_management, emergency_demo, duration
):
    """
    Market Making Examples - Comprehensive demonstrations of market making functionality.

    Run various market making scenarios to understand different configurations and features.
    """

    async def run_examples():
        runner = MarketMakingExampleRunner()
        runner.register_shutdown_handlers()

        try:
            console.print(
                Panel.fit(
                    "[bold green]Market Making Examples[/bold green]\n\n"
                    "Welcome to the comprehensive market making examples!\n"
                    "These demonstrations show real-world usage patterns and configurations.\n\n"
                    "[yellow]Note: All examples use mock data and are completely safe to run.[/yellow]",
                    title="🚀 Market Making Demo Suite",
                )
            )

            if multi_symbol:
                example = MultiSymbolExample()
                await example.run_multi_symbol_example(duration)

            elif paper_trading:
                simulation = PaperTradingSimulation()
                await simulation.run_paper_trading_demo(duration)

            elif risk_management:
                risk_demo = RiskManagementExample()
                await risk_demo.run_risk_scenarios()

            elif emergency_demo:
                emergency_example = EmergencyRecoveryExample()
                await emergency_example.run_emergency_scenarios()

            elif profile == "conservative":
                example = ConservativeTradingExample()
                await example.run_example(duration)

            elif profile == "aggressive":
                example = AggressiveHighFrequencyExample()
                await example.run_example(duration)

            else:
                # Run all examples in sequence
                console.print("[yellow]Running all examples...[/yellow]")

                conservative = ConservativeTradingExample()
                await conservative.run_example(10)

                aggressive = AggressiveHighFrequencyExample()
                await aggressive.run_example(10)

                multi = MultiSymbolExample()
                await multi.run_multi_symbol_example(10)

        except KeyboardInterrupt:
            console.print("\n[yellow]Examples interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error running examples: {e}[/red]")
            logger.exception("Error in main examples")
        finally:
            await runner.shutdown_all_engines()

    # Run the async function
    asyncio.run(run_examples())


if __name__ == "__main__":
    main()
