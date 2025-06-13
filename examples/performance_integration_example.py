#!/usr/bin/env python3
"""
Example of integrating performance monitoring into the trading bot.

This script demonstrates how to use the performance monitoring system
with the main trading components to track performance in real-time.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from bot.indicators.vumanchu import VuManChuIndicators

# Import bot components
from bot.main import TradingEngine
from bot.performance_monitor import (
    PerformanceMonitor,
    track,
    track_async,
    track_sync,
)
from bot.strategy.llm_agent import LLMAgent
from bot.types import IndicatorData, MarketState, Position

logger = logging.getLogger(__name__)


class MonitoredTradingEngine(TradingEngine):
    """
    Trading engine with integrated performance monitoring.

    This extends the main trading engine to include comprehensive
    performance tracking and alerting.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with performance monitoring."""
        super().__init__(*args, **kwargs)

        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Set up alert callback
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)

        # Override components with monitored versions
        self._wrap_components_with_monitoring()

    def _wrap_components_with_monitoring(self):
        """Wrap trading components with performance monitoring."""
        # Wrap indicator calculator
        original_calculate_all = self.indicator_calc.calculate_all

        @track_sync("indicator_calculation")
        def monitored_calculate_all(df):
            return original_calculate_all(df)

        self.indicator_calc.calculate_all = monitored_calculate_all

        # Wrap LLM agent
        original_analyze_market = self.llm_agent.analyze_market

        @track_async("llm_response")
        async def monitored_analyze_market(market_state):
            return await original_analyze_market(market_state)

        self.llm_agent.analyze_market = monitored_analyze_market

        # Wrap trade execution
        original_execute_trade = self._execute_trade

        async def monitored_execute_trade(trade_action, current_price):
            with track("trade_execution"):
                return await original_execute_trade(trade_action, current_price)

        self._execute_trade = monitored_execute_trade

    def _handle_performance_alert(self, alert):
        """Handle performance alerts."""
        logger.warning(f"Performance Alert: {alert.message}")

        # You could integrate with external alerting systems here
        # For example: send to Slack, email, monitoring dashboard, etc.

        if alert.level.value == "critical":
            logger.critical(f"CRITICAL PERFORMANCE ISSUE: {alert.message}")
            # Could trigger emergency procedures here

    async def run(self):
        """Run the trading engine with performance monitoring."""
        # Start performance monitoring
        await self.performance_monitor.start_monitoring()

        try:
            # Run the main trading loop
            await super().run()
        finally:
            # Stop performance monitoring and get final report
            await self.performance_monitor.stop_monitoring()
            await self._generate_performance_report()

    async def _generate_performance_report(self):
        """Generate final performance report."""
        logger.info("Generating performance report...")

        # Get performance summary for the last session
        summary = self.performance_monitor.get_performance_summary(
            duration=timedelta(hours=1)
        )

        print(f"\n{'='*60}")
        print("TRADING SESSION PERFORMANCE REPORT")
        print(f"{'='*60}")
        print(f"Session Duration: {summary['period_minutes']:.1f} minutes")
        print(f"Health Score: {summary['health_score']:.1f}/100")
        print(f"{'='*60}")

        # Latency summary
        if summary["latency_summary"]:
            print("\nLATENCY METRICS:")
            for metric, stats in summary["latency_summary"].items():
                operation = metric.replace("latency.", "").replace("_", " ").title()
                print(f"  {operation}:")
                print(f"    Mean: {stats['mean']:.1f} ms")
                print(f"    P95:  {stats['p95']:.1f} ms")
                print(f"    P99:  {stats['p99']:.1f} ms")

        # Resource summary
        if summary["resource_summary"]:
            print("\nRESOURCE USAGE:")
            for metric, stats in summary["resource_summary"].items():
                resource = metric.replace("resource.", "").replace("_", " ").title()
                unit = "MB" if "memory" in metric else "%"
                print(f"  {resource}:")
                print(f"    Mean: {stats['mean']:.1f} {unit}")
                print(f"    Max:  {stats['max']:.1f} {unit}")

        # Alerts summary
        if summary["recent_alerts"]:
            print(f"\nALERTS ({len(summary['recent_alerts'])}):")
            for alert in summary["recent_alerts"][-5:]:  # Show last 5
                timestamp = alert.get("timestamp", "Unknown")
                level = alert.get("level", "unknown").upper()
                message = alert.get("message", "No message")
                print(f"  [{level}] {timestamp}: {message}")

        # Bottlenecks
        bottlenecks = summary.get("bottleneck_analysis", {}).get("bottlenecks", [])
        if bottlenecks:
            print(f"\nIDENTIFIED BOTTLENECKS ({len(bottlenecks)}):")
            for bottleneck in bottlenecks:
                print(f"  {bottleneck['type']}: {bottleneck['metric']}")

        # Recommendations
        recommendations = summary.get("bottleneck_analysis", {}).get(
            "recommendations", []
        )
        if recommendations:
            print("\nOPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n{'='*60}")


class PerformanceTestRunner:
    """Runs performance tests with real trading components."""

    def __init__(self):
        """Initialize the test runner."""
        self.performance_monitor = PerformanceMonitor()

        # Create test components
        self.indicator_calc = VuManChuIndicators()
        self.llm_agent = LLMAgent()

        # Test data
        self.test_market_state = self._create_test_market_state()

    def _create_test_market_state(self) -> MarketState:
        """Create a test market state."""
        test_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.utcnow(),
        )

        test_indicators = IndicatorData(
            cipher_a_dot=1.0,
            cipher_b_wave=0.5,
            cipher_b_money_flow=55.0,
            rsi=45.0,
            ema_fast=50000.0,
            ema_slow=49900.0,
        )

        return MarketState(
            symbol="BTC-USD",
            interval="1m",
            timestamp=datetime.utcnow(),
            current_price=Decimal("50000"),
            ohlcv_data=[],
            indicators=test_indicators,
            current_position=test_position,
        )

    async def run_performance_test(self, duration_minutes: int = 5):
        """Run a performance test for specified duration."""
        print(f"🚀 Starting {duration_minutes}-minute performance test...")

        # Start monitoring
        await self.performance_monitor.start_monitoring()

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        operation_count = 0

        try:
            while datetime.utcnow() < end_time:
                # Simulate trading operations

                # 1. Indicator calculation
                with track("indicator_calculation_test"):
                    # Simulate indicator calculation with synthetic data
                    import numpy as np
                    import pandas as pd

                    # Generate synthetic OHLCV data
                    data = {
                        "open": np.random.uniform(49000, 51000, 100),
                        "high": np.random.uniform(49500, 51500, 100),
                        "low": np.random.uniform(48500, 50500, 100),
                        "close": np.random.uniform(49000, 51000, 100),
                        "volume": np.random.uniform(1000, 10000, 100),
                    }
                    df = pd.DataFrame(data)
                    result = self.indicator_calc.calculate_all(df)

                # 2. LLM decision (occasionally)
                if operation_count % 10 == 0:  # Every 10th operation
                    with track("llm_decision_test"):
                        try:
                            decision = await self.llm_agent.analyze_market(
                                self.test_market_state
                            )
                        except Exception as e:
                            logger.warning(f"LLM test failed: {e}")

                # 3. Market data processing simulation
                with track("market_data_processing_test"):
                    # Simulate data processing
                    await asyncio.sleep(0.001)  # Simulate processing time

                operation_count += 1

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

        finally:
            # Stop monitoring and generate report
            await self.performance_monitor.stop_monitoring()

            print(f"✅ Performance test completed after {operation_count} operations")

            # Get performance summary
            summary = self.performance_monitor.get_performance_summary()

            print("\n📊 PERFORMANCE TEST RESULTS")
            print(f"Duration: {duration_minutes} minutes")
            print(f"Operations: {operation_count}")
            print(f"Health Score: {summary['health_score']:.1f}/100")

            # Show key metrics
            if summary["latency_summary"]:
                print("\n⏱️  LATENCY METRICS:")
                for metric, stats in summary["latency_summary"].items():
                    print(
                        f"  {metric}: {stats['mean']:.1f}ms (P95: {stats['p95']:.1f}ms)"
                    )

            if summary["resource_summary"]:
                print("\n💾 RESOURCE METRICS:")
                for metric, stats in summary["resource_summary"].items():
                    unit = "MB" if "memory" in metric else "%"
                    print(
                        f"  {metric}: {stats['mean']:.1f}{unit} (Max: {stats['max']:.1f}{unit})"
                    )


async def main():
    """Main demonstration function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🎯 AI Trading Bot Performance Integration Example")
    print("=" * 60)

    # Option 1: Run performance test
    print("\n1. Running standalone performance test...")
    test_runner = PerformanceTestRunner()
    await test_runner.run_performance_test(duration_minutes=2)

    print("\n" + "=" * 60)
    print("2. Example: How to integrate with TradingEngine")
    print("   (This would normally run the full trading bot)")

    # Option 2: Show how to use with TradingEngine (commented out for demo)
    # print("\n2. Running monitored trading engine...")
    # engine = MonitoredTradingEngine(
    #     symbol="BTC-USD",
    #     interval="1m",
    #     dry_run=True
    # )
    # await engine.run()

    print("\n✅ Performance integration example completed!")
    print("\nTo use performance monitoring in your trading bot:")
    print("1. Import: from bot.performance_monitor import track, track_async")
    print("2. Decorate functions: @track_async('operation_name')")
    print("3. Use context manager: with track('operation_name'):")
    print("4. Start monitoring: await performance_monitor.start_monitoring()")
    print("5. Get reports: performance_monitor.get_performance_summary()")


if __name__ == "__main__":
    asyncio.run(main())
