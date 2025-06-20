#!/usr/bin/env python3
"""
Example script demonstrating comprehensive health monitoring for Bluefin services.

This script shows how to set up and use all the monitoring components together
to create a robust health monitoring system for the Bluefin trading ecosystem.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.exchange.bluefin_client import BluefinServiceClient
from bot.monitoring import (
    BluefinDiagnosticTools,
    BluefinHealthMonitor,
    MonitoringDashboard,
    PerformanceMetricsCollector,
    ServiceDiscovery,
)
from bot.monitoring.auto_recovery import AutoRecoveryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComprehensiveMonitoringSystem:
    """
    Complete monitoring system for Bluefin services.

    This class demonstrates how to integrate all monitoring components
    to create a comprehensive health monitoring and automatic recovery system.
    """

    def __init__(self, bluefin_service_url: str = "http://bluefin-service:8080"):
        self.bluefin_service_url = bluefin_service_url

        # Initialize monitoring components
        self.health_monitor = BluefinHealthMonitor(bluefin_service_url)
        self.service_discovery = ServiceDiscovery()
        self.performance_collector = PerformanceMetricsCollector(bluefin_service_url)
        self.diagnostic_tools = BluefinDiagnosticTools(bluefin_service_url)
        self.monitoring_dashboard = MonitoringDashboard(
            port=9090, bluefin_service_url=bluefin_service_url
        )

        # Initialize auto recovery (depends on other components)
        self.auto_recovery = AutoRecoveryEngine(
            self.health_monitor, self.service_discovery
        )

        # Initialize Bluefin client for testing
        self.bluefin_client = BluefinServiceClient(bluefin_service_url)

        # State tracking
        self.is_running = False

    async def start(self) -> None:
        """Start all monitoring components."""
        logger.info("🚀 Starting comprehensive Bluefin monitoring system")

        try:
            # Start core monitoring components
            logger.info("Starting health monitor...")
            await self.health_monitor.start_monitoring()

            logger.info("Starting service discovery...")
            await self.service_discovery.start_discovery()

            logger.info("Starting performance metrics collection...")
            await self.performance_collector.start_collection()

            logger.info("Starting auto recovery engine...")
            await self.auto_recovery.start()

            logger.info("Starting monitoring dashboard...")
            await self.monitoring_dashboard.start()

            self.is_running = True

            logger.info("✅ All monitoring components started successfully!")
            logger.info("📊 Dashboard available at: http://localhost:9090")

            # Run initial diagnostics
            await self._run_initial_diagnostics()

            # Demonstrate monitoring capabilities
            await self._demonstrate_monitoring()

        except Exception as e:
            logger.exception("❌ Failed to start monitoring system: %s", e)
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop all monitoring components gracefully."""
        logger.info("🛑 Stopping comprehensive monitoring system")

        self.is_running = False

        # Stop components in reverse order
        try:
            await self.monitoring_dashboard.stop()
            await self.auto_recovery.stop()
            await self.performance_collector.stop_collection()
            await self.service_discovery.stop_discovery()
            await self.health_monitor.stop_monitoring()

            logger.info("✅ All monitoring components stopped successfully")
        except Exception as e:
            logger.exception("❌ Error stopping monitoring system: %s", e)

    async def _run_initial_diagnostics(self) -> None:
        """Run initial diagnostic checks."""
        logger.info("🔧 Running initial diagnostic checks...")

        try:
            # Run quick diagnostics
            report = await self.diagnostic_tools.run_quick_diagnostic()

            logger.info("📋 Diagnostic Report ID: %s", report.report_id)
            logger.info("🎯 Overall Status: %s", report.overall_status)

            # Log key findings
            if report.results:
                passed_checks = sum(1 for r in report.results if r.status == "pass")
                total_checks = len(report.results)
                logger.info("✅ Checks Passed: %s/%s", passed_checks, total_checks)

                # Log any failures
                failed_checks = [r for r in report.results if r.status == "fail"]
                if failed_checks:
                    logger.warning("⚠️ Failed Checks:")
                    for check in failed_checks:
                        logger.warning("  - %s: %s", check.check_name, check.message)

        except Exception as e:
            logger.exception("❌ Initial diagnostics failed: %s", e)

    async def _demonstrate_monitoring(self) -> None:
        """Demonstrate monitoring capabilities."""
        logger.info("🎯 Demonstrating monitoring capabilities...")

        try:
            # Test connectivity monitoring
            logger.info("Testing connectivity monitoring...")
            connectivity_status = await self.bluefin_client.get_connectivity_status()
            logger.info("📡 Connection Status: %s", connectivity_status['connection']['connected'])
            logger.info("📊 Success Rate: %s%", connectivity_status['metrics']['success_rate']:.1f)

            # Test comprehensive connectivity test
            logger.info("Running comprehensive connectivity test...")
            connectivity_test = (
                await self.bluefin_client.run_comprehensive_connectivity_test()
            )
            logger.info("🧪 Connectivity Test: %s", connectivity_test['overall_status'])

            # Show service discovery results
            logger.info("Checking service discovery...")
            discovery_status = self.service_discovery.get_discovery_status()
            logger.info("🔍 Services Discovered: %s", discovery_status['total_services'])
            logger.info("💚 Healthy Services: %s", discovery_status['healthy_services'])

            # Show performance metrics
            logger.info("Checking performance metrics...")
            metrics_summary = self.performance_collector.get_metrics_summary()
            if metrics_summary["collection_status"]["is_collecting"]:
                logger.info("📈 Performance metrics are being collected")
                logger.info("📊 Total Metrics: %s", metrics_summary['metrics_count'])

            # Show auto recovery status
            logger.info("Checking auto recovery...")
            recovery_status = self.auto_recovery.get_recovery_status()
            logger.info("🔄 Recovery Rules: %s", recovery_status['total_rules'])
            logger.info("✅ Enabled Rules: %s", recovery_status['enabled_rules'])

        except Exception as e:
            logger.exception("❌ Error demonstrating monitoring: %s", e)

    async def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        try:
            # Collect status from all components
            health_status = self.health_monitor.get_comprehensive_status()
            discovery_status = self.service_discovery.get_discovery_status()
            metrics_summary = self.performance_collector.get_metrics_summary()
            recovery_status = self.auto_recovery.get_recovery_status()

            return {
                "monitoring_active": self.is_running,
                "bluefin_service_url": self.bluefin_service_url,
                "health_monitor": {
                    "status": health_status.get("overall_health", "unknown"),
                    "monitoring_active": health_status.get("monitoring_active", False),
                    "uptime_seconds": health_status.get("uptime_seconds", 0),
                },
                "service_discovery": {
                    "active": discovery_status.get("discovery_active", False),
                    "total_services": discovery_status.get("total_services", 0),
                    "healthy_services": discovery_status.get("healthy_services", 0),
                },
                "performance_metrics": {
                    "collecting": metrics_summary.get("collection_status", {}).get(
                        "is_collecting", False
                    ),
                    "metrics_count": metrics_summary.get("metrics_count", 0),
                    "active_alerts": metrics_summary.get("active_alerts", 0),
                },
                "auto_recovery": {
                    "active": recovery_status.get("recovery_active", False),
                    "total_rules": recovery_status.get("total_rules", 0),
                    "recent_recoveries": recovery_status.get("recent_recoveries", 0),
                },
                "dashboard": {"url": "http://localhost:9090", "active": True},
            }
        except Exception as e:
            logger.exception("Error getting system status: %s", e)
            return {"error": str(e)}

    async def run_diagnostic_suite(self) -> dict:
        """Run comprehensive diagnostic suite."""
        logger.info("🔧 Running comprehensive diagnostic suite...")

        try:
            # Run comprehensive diagnostics
            report = await self.diagnostic_tools.run_comprehensive_diagnostics(
                include_performance_tests=True,
                include_stress_tests=False,  # Skip stress tests for this example
            )

            logger.info("📋 Comprehensive Diagnostic Report: %s", report.overall_status)

            # Log summary
            if hasattr(report, "summary"):
                summary = report.summary
                logger.info("✅ Success Rate: %s%", summary.get('success_rate', 0):.1f)
                logger.info("⏱️  Total Duration: %sms", summary.get('total_duration_ms', 0):.1f)

                # Log recommendations
                if summary.get("recommendations"):
                    logger.info("💡 Recommendations:")
                    for rec in summary["recommendations"][:3]:  # Show top 3
                        logger.info("  - %s", rec)

            return report.__dict__

        except Exception as e:
            logger.exception("❌ Diagnostic suite failed: %s", e)
            return {"error": str(e)}


async def main():
    """Main function demonstrating the monitoring system."""
    # Configuration
    BLUEFIN_SERVICE_URL = "http://localhost:8080"  # Adjust as needed

    # Create monitoring system
    monitoring_system = ComprehensiveMonitoringSystem(BLUEFIN_SERVICE_URL)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received signal %s, initiating shutdown...", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the monitoring system
        await monitoring_system.start()

        # Run for a demonstration period
        logger.info("🔄 Monitoring system is running...")
        logger.info("📊 Open http://localhost:9090 to view the dashboard")
        logger.info("🛑 Press Ctrl+C to stop the monitoring system")

        # Monitor and report status periodically
        status_report_interval = 60  # seconds
        diagnostic_interval = 300  # seconds (5 minutes)

        last_status_report = 0
        last_diagnostic = 0

        while monitoring_system.is_running:
            current_time = asyncio.get_event_loop().time()

            # Status report
            if current_time - last_status_report >= status_report_interval:
                system_status = await monitoring_system.get_system_status()
                logger.info("📊 System Status Update:")
                logger.info("  🏥 Health: %s", system_status['health_monitor']['status'])
                logger.info("  🔍 Services: %s/%s", system_status['service_discovery']['healthy_services'], system_status['service_discovery']['total_services'])
                logger.info("  📈 Metrics: %s collected", system_status['performance_metrics']['metrics_count'])
                logger.info("  🔄 Recoveries: %s recent", system_status['auto_recovery']['recent_recoveries'])

                last_status_report = current_time

            # Periodic diagnostics
            if current_time - last_diagnostic >= diagnostic_interval:
                logger.info("🔧 Running periodic diagnostics...")
                diagnostic_result = await monitoring_system.run_diagnostic_suite()

                if "error" not in diagnostic_result:
                    logger.info("✅ Periodic diagnostics completed successfully")
                else:
                    logger.warning("⚠️ Periodic diagnostics had issues: %s", diagnostic_result['error'])

                last_diagnostic = current_time

            # Short sleep to prevent busy waiting
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.exception("❌ Unexpected error: %s", e)
    finally:
        # Cleanup
        await monitoring_system.stop()
        logger.info("👋 Monitoring system shutdown complete")


if __name__ == "__main__":
    # Example of how to run the monitoring system
    print("🔵 Bluefin Services Comprehensive Monitoring System")
    print("=" * 50)
    print()
    print("This example demonstrates:")
    print("✅ Health monitoring for all Bluefin services")
    print("✅ Automatic service discovery")
    print("✅ Performance metrics collection and alerting")
    print("✅ Comprehensive diagnostic tools")
    print("✅ Real-time monitoring dashboard")
    print("✅ Automatic service recovery")
    print()
    print("🚀 Starting monitoring system...")
    print()

    # Run the monitoring system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
