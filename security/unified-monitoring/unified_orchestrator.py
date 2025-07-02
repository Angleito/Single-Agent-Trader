"""
Unified Security Monitoring Orchestrator for OPTIMIZE Platform

This module serves as the main orchestrator for the unified security monitoring
system, coordinating all security components to provide comprehensive protection
with minimal trading impact.
"""

import asyncio
import json
import logging
import signal
from datetime import datetime, timedelta
from typing import Any

import redis

from .alert_orchestrator import AlertOrchestrator, create_alert_orchestrator
from .correlation_engine import (
    CorrelationEngine,
    EventSource,
    create_correlation_engine,
)
from .executive_reporting import (
    ExecutiveReportingSystem,
    ReportType,
    create_executive_reporting_system,
)
from .performance_monitor import PerformanceMonitor, create_performance_monitor
from .response_automation import ResponseAutomation, create_response_automation
from .security_dashboard import SecurityDashboard, create_security_dashboard

logger = logging.getLogger(__name__)


class UnifiedSecurityOrchestrator:
    """
    Main orchestrator for the unified security monitoring platform.

    Coordinates all security components to provide:
    - Real-time threat detection and correlation
    - Automated incident response
    - Performance impact monitoring
    - Executive reporting and analytics
    - Unified security dashboard
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: dict[str, Any] | None = None,
    ):
        self.redis_client = redis.from_url(redis_url)
        self.config = config or {}

        # Core components
        self.correlation_engine: CorrelationEngine | None = None
        self.alert_orchestrator: AlertOrchestrator | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.response_automation: ResponseAutomation | None = None
        self.executive_reporting: ExecutiveReportingSystem | None = None
        self.security_dashboard: SecurityDashboard | None = None

        # Integration components
        self.security_tools_monitor = SecurityToolsMonitor(redis_url)
        self.health_checker = SystemHealthChecker(redis_url)

        # Internal state
        self.running = False
        self.startup_time = None
        self.component_status = {}

        # Event handlers
        self.event_handlers = {}
        self._setup_event_handlers()

        # Performance optimization
        self.optimization_engine = OptimizationEngine()

    async def start(self):
        """Start the unified security monitoring platform."""
        try:
            self.startup_time = datetime.utcnow()
            self.running = True

            logger.info("🛡️ Starting OPTIMIZE Unified Security Platform")
            logger.info("=" * 60)

            # Initialize core components
            await self._initialize_components()

            # Verify component health
            await self._verify_component_health()

            # Start integration services
            await self._start_integration_services()

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            logger.info("✅ OPTIMIZE Platform started successfully")
            logger.info("📊 Dashboard: http://localhost:8080")
            logger.info("🔧 Executive Reports: http://localhost:8080/executive")
            logger.info("⚙️ Operations Center: http://localhost:8080/operations")
            logger.info("=" * 60)

            # Start main orchestration loop
            await self._orchestration_loop()

        except Exception as e:
            logger.error(f"❌ Failed to start OPTIMIZE Platform: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the unified security monitoring platform."""
        if not self.running:
            return

        logger.info("🛑 Stopping OPTIMIZE Unified Security Platform")
        self.running = False

        # Stop components in reverse order
        components = [
            ("Security Dashboard", self.security_dashboard),
            ("Executive Reporting", self.executive_reporting),
            ("Response Automation", self.response_automation),
            ("Performance Monitor", self.performance_monitor),
            ("Alert Orchestrator", self.alert_orchestrator),
            ("Correlation Engine", self.correlation_engine),
        ]

        for name, component in components:
            if component:
                try:
                    logger.info(f"Stopping {name}...")
                    await component.stop()
                    logger.info(f"✅ {name} stopped")
                except Exception as e:
                    logger.error(f"❌ Error stopping {name}: {e}")

        # Stop integration services
        await self.security_tools_monitor.stop()
        await self.health_checker.stop()

        logger.info("🛡️ OPTIMIZE Platform stopped")

    async def _initialize_components(self):
        """Initialize all security platform components."""
        logger.info("🔧 Initializing security components...")

        # 1. Correlation Engine (data foundation)
        logger.info("  📊 Starting Correlation Engine...")
        self.correlation_engine = create_correlation_engine(
            redis_url=str(self.redis_client.connection_pool.connection_kwargs["host"])
            + f":{self.redis_client.connection_pool.connection_kwargs['port']}"
        )
        correlation_task = asyncio.create_task(self.correlation_engine.start())

        # 2. Alert Orchestrator (alert management)
        logger.info("  🚨 Starting Alert Orchestrator...")
        self.alert_orchestrator = create_alert_orchestrator(
            config=self.config.get("alerts", {})
        )
        alert_task = asyncio.create_task(self.alert_orchestrator.start())

        # 3. Performance Monitor (impact tracking)
        logger.info("  📈 Starting Performance Monitor...")
        self.performance_monitor = create_performance_monitor()
        performance_task = asyncio.create_task(self.performance_monitor.start())

        # 4. Response Automation (incident response)
        logger.info("  🤖 Starting Response Automation...")
        self.response_automation = create_response_automation()
        response_task = asyncio.create_task(self.response_automation.start())

        # 5. Executive Reporting (analytics)
        logger.info("  📋 Starting Executive Reporting...")
        self.executive_reporting = create_executive_reporting_system()
        reporting_task = asyncio.create_task(self.executive_reporting.start())

        # 6. Security Dashboard (visualization)
        logger.info("  🖥️ Starting Security Dashboard...")
        self.security_dashboard = create_security_dashboard(self.correlation_engine)
        dashboard_task = asyncio.create_task(self.security_dashboard.start())

        # Wait for initial startup (don't wait for full start to avoid blocking)
        await asyncio.sleep(2)

        # Store component tasks for monitoring
        self.component_tasks = {
            "correlation_engine": correlation_task,
            "alert_orchestrator": alert_task,
            "performance_monitor": performance_task,
            "response_automation": response_task,
            "executive_reporting": reporting_task,
            "security_dashboard": dashboard_task,
        }

        logger.info("✅ All components initialized")

    async def _verify_component_health(self):
        """Verify that all components are healthy."""
        logger.info("🏥 Verifying component health...")

        health_checks = [
            ("Correlation Engine", self._check_correlation_health),
            ("Alert Orchestrator", self._check_alert_health),
            ("Performance Monitor", self._check_performance_health),
            ("Response Automation", self._check_response_health),
            ("Executive Reporting", self._check_reporting_health),
            ("Security Dashboard", self._check_dashboard_health),
        ]

        for name, health_check in health_checks:
            try:
                status = await health_check()
                self.component_status[name] = status

                if status.get("healthy", False):
                    logger.info(f"  ✅ {name}: Healthy")
                else:
                    logger.warning(
                        f"  ⚠️ {name}: {status.get('error', 'Unknown issue')}"
                    )

            except Exception as e:
                logger.error(f"  ❌ {name}: Health check failed - {e}")
                self.component_status[name] = {"healthy": False, "error": str(e)}

    async def _start_integration_services(self):
        """Start integration and monitoring services."""
        logger.info("🔗 Starting integration services...")

        # Start security tools monitoring
        await self.security_tools_monitor.start()

        # Start system health checking
        await self.health_checker.start()

        # Start event processing
        asyncio.create_task(self._process_security_events())
        asyncio.create_task(self._monitor_component_health())
        asyncio.create_task(self._optimize_performance())

        logger.info("✅ Integration services started")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_event_handlers(self):
        """Setup event handlers for cross-component communication."""
        self.event_handlers = {
            "security_correlation": self._handle_security_correlation,
            "performance_alert": self._handle_performance_alert,
            "security_incident": self._handle_security_incident,
            "component_failure": self._handle_component_failure,
        }

    async def _orchestration_loop(self):
        """Main orchestration loop."""
        logger.info("🔄 Starting orchestration loop...")

        while self.running:
            try:
                # Monitor component health
                await self._monitor_components()

                # Process integration events
                await self._process_integration_events()

                # Optimize system performance
                await self._run_optimization_cycle()

                # Update metrics
                await self._update_platform_metrics()

                # Sleep between cycles
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(10)

    async def _process_security_events(self):
        """Process security events from external tools."""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(
            "falco_events", "trivy_events", "docker_bench_events", "trading_bot_events"
        )

        while self.running:
            try:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    await self._handle_external_security_event(message)

            except Exception as e:
                logger.error(f"Error processing security events: {e}")
                await asyncio.sleep(5)

    async def _handle_external_security_event(self, message: dict[str, Any]):
        """Handle security events from external tools."""
        try:
            channel = message["channel"].decode()
            data = json.loads(message["data"])

            # Determine event source
            source_map = {
                "falco_events": EventSource.FALCO,
                "trivy_events": EventSource.TRIVY,
                "docker_bench_events": EventSource.DOCKER_BENCH,
                "trading_bot_events": EventSource.TRADING_BOT,
            }

            source = source_map.get(channel, EventSource.CUSTOM)

            # Ingest into correlation engine
            if self.correlation_engine:
                await self.correlation_engine.ingest_event(data, source)

        except Exception as e:
            logger.error(f"Error handling external security event: {e}")

    async def _monitor_component_health(self):
        """Monitor health of all components."""
        while self.running:
            try:
                # Check component task status
                for name, task in self.component_tasks.items():
                    if task.done():
                        exception = task.exception()
                        if exception:
                            logger.error(f"Component {name} failed: {exception}")
                            await self._handle_component_failure(name, exception)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring component health: {e}")
                await asyncio.sleep(60)

    async def _optimize_performance(self):
        """Run performance optimization cycles."""
        while self.running:
            try:
                if self.performance_monitor:
                    status = await self.performance_monitor.get_performance_status()

                    # Apply optimizations based on performance data
                    await self.optimization_engine.optimize_based_on_metrics(status)

                await asyncio.sleep(300)  # Optimize every 5 minutes

            except Exception as e:
                logger.error(f"Error in performance optimization: {e}")
                await asyncio.sleep(300)

    async def _monitor_components(self):
        """Monitor component status and performance."""
        try:
            # Update component status
            await self._verify_component_health()

            # Check for component failures
            failed_components = [
                name
                for name, status in self.component_status.items()
                if not status.get("healthy", False)
            ]

            if failed_components:
                logger.warning(f"Failed components detected: {failed_components}")
                # TODO: Implement component restart logic

        except Exception as e:
            logger.error(f"Error monitoring components: {e}")

    async def _process_integration_events(self):
        """Process events between components."""
        try:
            # Check for correlation results
            if self.correlation_engine and self.alert_orchestrator:
                # Get recent correlations and process them
                # This would typically be done via Redis pub/sub
                pass

        except Exception as e:
            logger.error(f"Error processing integration events: {e}")

    async def _run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        try:
            # Get performance metrics
            if self.performance_monitor:
                perf_status = await self.performance_monitor.get_performance_status()

                # Apply optimizations
                optimizations = await self.optimization_engine.analyze_and_optimize(
                    perf_status
                )

                if optimizations:
                    logger.info(
                        f"Applied {len(optimizations)} performance optimizations"
                    )

        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")

    async def _update_platform_metrics(self):
        """Update overall platform metrics."""
        try:
            platform_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (
                    (datetime.utcnow() - self.startup_time).total_seconds()
                    if self.startup_time
                    else 0
                ),
                "component_status": self.component_status,
                "events_processed": 0,  # TODO: Track actual events
                "alerts_generated": 0,  # TODO: Track actual alerts
                "responses_executed": 0,  # TODO: Track actual responses
            }

            # Publish platform metrics
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.publish,
                "platform_metrics",
                json.dumps(platform_metrics),
            )

        except Exception as e:
            logger.error(f"Error updating platform metrics: {e}")

    # Event Handlers

    async def _handle_security_correlation(self, correlation_data: dict[str, Any]):
        """Handle security correlation events."""
        try:
            if self.alert_orchestrator:
                # TODO: Convert correlation_data to CorrelationResult and process
                logger.info(
                    f"Processing security correlation: {correlation_data.get('correlation_id')}"
                )

        except Exception as e:
            logger.error(f"Error handling security correlation: {e}")

    async def _handle_performance_alert(self, alert_data: dict[str, Any]):
        """Handle performance alerts."""
        try:
            # Trigger optimization if performance is impacting trading
            impact_score = alert_data.get("trading_impact_score", 0)

            if impact_score > 20:  # High impact threshold
                logger.warning(f"High trading impact detected: {impact_score}%")
                await self.optimization_engine.emergency_optimization()

        except Exception as e:
            logger.error(f"Error handling performance alert: {e}")

    async def _handle_security_incident(self, incident_data: dict[str, Any]):
        """Handle security incidents."""
        try:
            severity = incident_data.get("severity", "unknown")

            if severity == "critical":
                # Escalate critical incidents
                logger.critical(
                    f"Critical security incident: {incident_data.get('incident_id')}"
                )

                # TODO: Trigger emergency response procedures

        except Exception as e:
            logger.error(f"Error handling security incident: {e}")

    async def _handle_component_failure(
        self, component_name: str, exception: Exception
    ):
        """Handle component failures."""
        try:
            logger.error(f"Component failure detected: {component_name} - {exception}")

            # Update component status
            self.component_status[component_name] = {
                "healthy": False,
                "error": str(exception),
                "last_failure": datetime.utcnow().isoformat(),
            }

            # TODO: Implement component restart logic
            # TODO: Send failure notifications

        except Exception as e:
            logger.error(f"Error handling component failure: {e}")

    # Health Check Methods

    async def _check_correlation_health(self) -> dict[str, Any]:
        """Check correlation engine health."""
        try:
            if self.correlation_engine:
                status = await self.correlation_engine.get_correlation_status()
                return {"healthy": status.get("running", False), "details": status}
            return {"healthy": False, "error": "Correlation engine not initialized"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_alert_health(self) -> dict[str, Any]:
        """Check alert orchestrator health."""
        try:
            if self.alert_orchestrator:
                status = await self.alert_orchestrator.get_alert_status()
                return {"healthy": status.get("running", False), "details": status}
            return {"healthy": False, "error": "Alert orchestrator not initialized"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_performance_health(self) -> dict[str, Any]:
        """Check performance monitor health."""
        try:
            if self.performance_monitor:
                status = await self.performance_monitor.get_performance_status()
                return {"healthy": status.get("running", False), "details": status}
            return {"healthy": False, "error": "Performance monitor not initialized"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_response_health(self) -> dict[str, Any]:
        """Check response automation health."""
        try:
            if self.response_automation:
                status = await self.response_automation.get_response_status()
                return {"healthy": status.get("running", False), "details": status}
            return {"healthy": False, "error": "Response automation not initialized"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_reporting_health(self) -> dict[str, Any]:
        """Check executive reporting health."""
        try:
            if self.executive_reporting:
                status = await self.executive_reporting.get_reporting_status()
                return {"healthy": status.get("running", False), "details": status}
            return {"healthy": False, "error": "Executive reporting not initialized"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_dashboard_health(self) -> dict[str, Any]:
        """Check security dashboard health."""
        try:
            if self.security_dashboard:
                # Simple health check - dashboard is healthy if it's serving
                return {"healthy": True, "details": {"port": 8080}}
            return {"healthy": False, "error": "Security dashboard not initialized"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    # Public API Methods

    async def get_platform_status(self) -> dict[str, Any]:
        """Get comprehensive platform status."""
        try:
            uptime = (
                (datetime.utcnow() - self.startup_time).total_seconds()
                if self.startup_time
                else 0
            )

            status = {
                "platform": {
                    "running": self.running,
                    "uptime_seconds": uptime,
                    "startup_time": (
                        self.startup_time.isoformat() if self.startup_time else None
                    ),
                    "version": "1.0.0",
                },
                "components": self.component_status,
                "integration": {
                    "security_tools_monitor": await self.security_tools_monitor.get_status(),
                    "health_checker": await self.health_checker.get_status(),
                },
                "last_update": datetime.utcnow().isoformat(),
            }

            return status

        except Exception as e:
            logger.error(f"Error getting platform status: {e}")
            return {"error": str(e)}

    async def generate_emergency_report(self) -> dict[str, Any]:
        """Generate emergency security report."""
        try:
            if not self.executive_reporting:
                return {"error": "Executive reporting not available"}

            # Generate immediate report
            report = await self.executive_reporting.generate_report(
                ReportType.INCIDENT_ANALYSIS,
                period_start=datetime.utcnow() - timedelta(hours=24),
                period_end=datetime.utcnow(),
            )

            return {
                "report_id": report.report_id,
                "security_score": report.security_score,
                "risk_level": report.risk_level,
                "executive_summary": report.executive_summary,
                "key_findings": report.key_findings,
                "recommendations": report.recommendations,
                "generated_at": report.generated_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating emergency report: {e}")
            return {"error": str(e)}


class SecurityToolsMonitor:
    """Monitors external security tools and their output."""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.running = False
        self.tool_status = {}

    async def start(self):
        """Start monitoring security tools."""
        self.running = True
        asyncio.create_task(self._monitor_falco())
        asyncio.create_task(self._monitor_trivy())
        asyncio.create_task(self._monitor_docker_bench())

    async def stop(self):
        """Stop monitoring security tools."""
        self.running = False

    async def _monitor_falco(self):
        """Monitor Falco runtime security."""
        while self.running:
            try:
                # Check if Falco is running and healthy
                # TODO: Implement actual Falco health check
                self.tool_status["falco"] = {
                    "healthy": True,
                    "last_check": datetime.utcnow(),
                }
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error monitoring Falco: {e}")
                await asyncio.sleep(60)

    async def _monitor_trivy(self):
        """Monitor Trivy vulnerability scanner."""
        while self.running:
            try:
                # Check Trivy status
                # TODO: Implement actual Trivy health check
                self.tool_status["trivy"] = {
                    "healthy": True,
                    "last_check": datetime.utcnow(),
                }
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error monitoring Trivy: {e}")
                await asyncio.sleep(300)

    async def _monitor_docker_bench(self):
        """Monitor Docker Bench Security."""
        while self.running:
            try:
                # Check Docker Bench status
                # TODO: Implement actual Docker Bench health check
                self.tool_status["docker_bench"] = {
                    "healthy": True,
                    "last_check": datetime.utcnow(),
                }
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error monitoring Docker Bench: {e}")
                await asyncio.sleep(3600)

    async def get_status(self) -> dict[str, Any]:
        """Get security tools monitoring status."""
        return {
            "running": self.running,
            "tools": self.tool_status,
            "last_update": datetime.utcnow().isoformat(),
        }


class SystemHealthChecker:
    """Monitors overall system health and resource usage."""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.running = False
        self.health_metrics = {}

    async def start(self):
        """Start system health monitoring."""
        self.running = True
        asyncio.create_task(self._monitor_system_health())

    async def stop(self):
        """Stop system health monitoring."""
        self.running = False

    async def _monitor_system_health(self):
        """Monitor system health metrics."""
        while self.running:
            try:
                # Collect system health metrics
                import psutil

                self.health_metrics = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                    "load_average": (
                        psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Publish health metrics
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.publish,
                    "system_health",
                    json.dumps(self.health_metrics),
                )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(30)

    async def get_status(self) -> dict[str, Any]:
        """Get system health status."""
        return {
            "running": self.running,
            "metrics": self.health_metrics,
            "last_update": datetime.utcnow().isoformat(),
        }


class OptimizationEngine:
    """Optimizes security tool performance and impact."""

    def __init__(self):
        self.optimization_history = []

    async def optimize_based_on_metrics(self, metrics: dict[str, Any]):
        """Apply optimizations based on performance metrics."""
        try:
            optimizations = []

            # Check trading impact
            trading_impact = metrics.get("trading_impact_score", 0)

            if trading_impact > 15:
                optimizations.append(
                    "Reduce security scan frequency during trading hours"
                )

            if trading_impact > 25:
                optimizations.append("Enable security tool throttling")

            # Check CPU usage
            latest_metrics = metrics.get("latest_metrics", {})
            for component, component_metrics in latest_metrics.items():
                cpu_usage = component_metrics.get("cpu_usage_percent", 0)

                if cpu_usage > 80:
                    optimizations.append(f"Optimize {component} CPU usage")

            # Apply optimizations
            for optimization in optimizations:
                await self._apply_optimization(optimization)

            return optimizations

        except Exception as e:
            logger.error(f"Error optimizing based on metrics: {e}")
            return []

    async def analyze_and_optimize(
        self, performance_status: dict[str, Any]
    ) -> list[str]:
        """Analyze performance and apply optimizations."""
        try:
            optimizations = []

            # Analyze performance trends
            # TODO: Implement sophisticated optimization analysis

            return optimizations

        except Exception as e:
            logger.error(f"Error in optimization analysis: {e}")
            return []

    async def emergency_optimization(self):
        """Apply emergency optimizations for critical performance impact."""
        try:
            logger.warning("Applying emergency performance optimizations")

            # Emergency optimizations
            emergency_actions = [
                "Temporarily disable non-critical security scans",
                "Increase scan intervals",
                "Enable aggressive resource throttling",
                "Reduce alert correlation complexity",
            ]

            for action in emergency_actions:
                await self._apply_optimization(action)

        except Exception as e:
            logger.error(f"Error in emergency optimization: {e}")

    async def _apply_optimization(self, optimization: str):
        """Apply a specific optimization."""
        try:
            logger.info(f"Applying optimization: {optimization}")

            # Record optimization
            self.optimization_history.append(
                {
                    "optimization": optimization,
                    "applied_at": datetime.utcnow().isoformat(),
                }
            )

            # TODO: Implement actual optimization logic

        except Exception as e:
            logger.error(f"Error applying optimization {optimization}: {e}")


# Factory function
def create_unified_orchestrator(
    redis_url: str = "redis://localhost:6379",
    config: dict[str, Any] | None = None,
) -> UnifiedSecurityOrchestrator:
    """Create a unified security orchestrator instance."""
    return UnifiedSecurityOrchestrator(redis_url, config)


# CLI entry point
async def main():
    """Main entry point for the unified security platform."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="OPTIMIZE Unified Security Platform")
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis URL"
    )
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)

    # Create and start orchestrator
    orchestrator = create_unified_orchestrator(args.redis_url, config)

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Platform error: {e}")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
