"""
LLM Performance Monitoring and Optimization System.

This module provides comprehensive performance monitoring for the LLM trading agent,
tracking response times, cache efficiency, and decision quality to achieve
sub-2 second response times with 80% latency reduction.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import mean, median
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for LLM operations."""

    timestamp: datetime
    response_time_ms: float
    prompt_size_chars: int
    cache_hit: bool
    decision_action: str
    optimization_level: str = "standard"
    error_occurred: bool = False


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    fastest_response_ms: float = float("inf")
    slowest_response_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    target_achieved: bool = False
    performance_improvement_pct: float = 0.0
    recent_metrics: list[PerformanceMetrics] = field(default_factory=list)


class LLMPerformanceMonitor:
    """
    Advanced performance monitoring system for LLM operations.

    Tracks:
    - Response times with percentile analysis
    - Cache efficiency and hit rates
    - Prompt optimization effectiveness
    - Decision quality correlation
    - Performance trends over time
    """

    def __init__(
        self,
        target_response_time_ms: float = 2000.0,  # 2 seconds target
        baseline_response_time_ms: float = 5000.0,  # 5 seconds baseline
        max_metrics_history: int = 1000,
    ):
        """
        Initialize the performance monitor.

        Args:
            target_response_time_ms: Target response time in milliseconds
            baseline_response_time_ms: Baseline response time for improvement calculation
            max_metrics_history: Maximum number of metrics to keep in memory
        """
        self.target_response_time_ms = target_response_time_ms
        self.baseline_response_time_ms = baseline_response_time_ms
        self.max_metrics_history = max_metrics_history

        # Metrics storage
        self.metrics_history: list[PerformanceMetrics] = []
        self.hourly_stats: dict[str, PerformanceStats] = {}

        # Performance tracking
        self.monitoring_start_time = datetime.now(UTC)
        self.last_stats_calculation = datetime.now(UTC)

        # Alerting thresholds
        self.slow_response_threshold_ms = target_response_time_ms * 1.5  # 3 seconds
        self.critical_response_threshold_ms = target_response_time_ms * 2.5  # 5 seconds

        logger.info(
            "🚀 LLM Performance Monitor initialized: Target=%sms, Baseline=%sms",
            target_response_time_ms,
            baseline_response_time_ms,
        )

    def record_request(
        self,
        response_time_ms: float,
        prompt_size_chars: int,
        cache_hit: bool,
        decision_action: str,
        optimization_level: str = "standard",
        error_occurred: bool = False,
    ):
        """
        Record a new LLM request for performance analysis.

        Args:
            response_time_ms: Response time in milliseconds
            prompt_size_chars: Size of the prompt in characters
            cache_hit: Whether this was a cache hit
            decision_action: The trading action decided
            optimization_level: Level of optimization applied
            error_occurred: Whether an error occurred during processing
        """
        metric = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            response_time_ms=response_time_ms,
            prompt_size_chars=prompt_size_chars,
            cache_hit=cache_hit,
            decision_action=decision_action,
            optimization_level=optimization_level,
            error_occurred=error_occurred,
        )

        self.metrics_history.append(metric)

        # Limit history size
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history = self.metrics_history[-self.max_metrics_history :]

        # Log performance alerts
        self._check_performance_alerts(metric)

        # Log achievement of target
        if response_time_ms <= self.target_response_time_ms and not cache_hit:
            logger.info(
                "🎯 TARGET ACHIEVED: %.1fms (target: %sms) Action: %s",
                response_time_ms,
                self.target_response_time_ms,
                decision_action,
            )

    def _check_performance_alerts(self, metric: PerformanceMetrics):
        """
        Check for performance alerts and log warnings.

        Args:
            metric: Performance metric to check
        """
        if metric.error_occurred:
            logger.error(
                "🚨 LLM Error: Request failed after %.1fms", metric.response_time_ms
            )
            return

        if metric.response_time_ms >= self.critical_response_threshold_ms:
            logger.error(
                "🚨 CRITICAL SLOW: %.1fms (threshold: %sms) Cache: %s",
                metric.response_time_ms,
                self.critical_response_threshold_ms,
                "HIT" if metric.cache_hit else "MISS",
            )
        elif metric.response_time_ms >= self.slow_response_threshold_ms:
            logger.warning(
                "⚠️ SLOW RESPONSE: %.1fms (threshold: %sms) Cache: %s",
                metric.response_time_ms,
                self.slow_response_threshold_ms,
                "HIT" if metric.cache_hit else "MISS",
            )
        elif (
            metric.response_time_ms <= self.target_response_time_ms
            and not metric.cache_hit
        ):
            logger.info(
                "⚡ FAST: %.1fms (target: %sms) FRESH",
                metric.response_time_ms,
                self.target_response_time_ms,
            )

    def get_current_stats(self) -> PerformanceStats:
        """
        Get current performance statistics.

        Returns:
            Current performance statistics
        """
        if not self.metrics_history:
            return PerformanceStats()

        # Filter recent metrics (last hour)
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history if m.timestamp >= one_hour_ago
        ]

        if not recent_metrics:
            recent_metrics = self.metrics_history[-10:]  # Last 10 if no recent data

        # Calculate statistics
        response_times = [m.response_time_ms for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        errors = sum(1 for m in recent_metrics if m.error_occurred)

        avg_response_time = mean(response_times)
        median_response_time = median(response_times)

        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_response_time = sorted_times[p95_index] if sorted_times else 0.0

        # Calculate performance improvement
        improvement_pct = (
            (self.baseline_response_time_ms - avg_response_time)
            / self.baseline_response_time_ms
            * 100
        )

        return PerformanceStats(
            total_requests=len(recent_metrics),
            cache_hits=cache_hits,
            cache_misses=len(recent_metrics) - cache_hits,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            fastest_response_ms=min(response_times),
            slowest_response_ms=max(response_times),
            error_rate=errors / len(recent_metrics) * 100,
            cache_hit_rate=cache_hits / len(recent_metrics) * 100,
            target_achieved=avg_response_time <= self.target_response_time_ms,
            performance_improvement_pct=improvement_pct,
            recent_metrics=recent_metrics,
        )

    def get_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.

        Returns:
            Formatted performance report string
        """
        stats = self.get_current_stats()
        uptime = datetime.now(UTC) - self.monitoring_start_time

        # Performance status
        if stats.target_achieved:
            status_emoji = "🎯"
            status_text = "TARGET ACHIEVED"
        elif stats.avg_response_time_ms <= self.target_response_time_ms * 1.2:
            status_emoji = "🟡"
            status_text = "NEAR TARGET"
        else:
            status_emoji = "🔴"
            status_text = "NEEDS OPTIMIZATION"

        return f"""
🚀 LLM PERFORMANCE OPTIMIZER REPORT
{"=" * 45}

{status_emoji} STATUS: {status_text}
⏱️  Uptime: {uptime.total_seconds() / 3600:.1f} hours
📊 Total Requests: {stats.total_requests}

⚡ RESPONSE TIMES:
   Average:     {stats.avg_response_time_ms:.1f}ms
   Median:      {stats.median_response_time_ms:.1f}ms
   95th %ile:   {stats.p95_response_time_ms:.1f}ms
   Fastest:     {stats.fastest_response_ms:.1f}ms
   Slowest:     {stats.slowest_response_ms:.1f}ms
   Target:      {self.target_response_time_ms:.1f}ms

🎯 OPTIMIZATION RESULTS:
   Performance Improvement: {stats.performance_improvement_pct:+.1f}%
   Target Achievement:      {"✅ YES" if stats.target_achieved else "❌ NO"}
   Error Rate:             {stats.error_rate:.1f}%

💾 CACHE EFFICIENCY:
   Hit Rate:    {stats.cache_hit_rate:.1f}%
   Cache Hits:  {stats.cache_hits}
   Cache Miss:  {stats.cache_misses}

📈 TREND ANALYSIS:
{self._get_trend_analysis()}

🔧 RECOMMENDATIONS:
{self._get_optimization_recommendations(stats)}
"""

    def _get_trend_analysis(self) -> str:
        """Get trend analysis from recent metrics."""
        if len(self.metrics_history) < 10:
            return "   Insufficient data for trend analysis"

        # Compare last 10 vs previous 10
        recent_10 = self.metrics_history[-10:]
        previous_10 = (
            self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else []
        )

        if not previous_10:
            return "   Insufficient historical data for trend analysis"

        recent_avg = mean([m.response_time_ms for m in recent_10])
        previous_avg = mean([m.response_time_ms for m in previous_10])

        trend_change = ((recent_avg - previous_avg) / previous_avg) * 100

        if trend_change < -5:
            trend_emoji = "📈"
            trend_text = f"IMPROVING ({trend_change:+.1f}%)"
        elif trend_change > 5:
            trend_emoji = "📉"
            trend_text = f"DEGRADING ({trend_change:+.1f}%)"
        else:
            trend_emoji = "➡️"
            trend_text = f"STABLE ({trend_change:+.1f}%)"

        return f"   {trend_emoji} {trend_text}"

    def _get_optimization_recommendations(self, stats: PerformanceStats) -> str:
        """Get optimization recommendations based on current performance."""
        recommendations = []

        if stats.cache_hit_rate < 70:
            recommendations.append(
                "   💾 Increase cache TTL or improve cache key similarity"
            )

        if stats.avg_response_time_ms > self.target_response_time_ms:
            recommendations.append("   ⚡ Enable aggressive prompt optimization")

        if stats.error_rate > 5:
            recommendations.append("   🛠️ Investigate and fix error sources")

        if stats.p95_response_time_ms > self.target_response_time_ms * 2:
            recommendations.append("   🚨 Add timeout handling for slow requests")

        if not recommendations:
            recommendations.append("   ✅ Performance is optimal!")

        return "\n".join(recommendations)

    async def start_monitoring(self, report_interval_minutes: int = 15):
        """
        Start continuous performance monitoring with periodic reports.

        Args:
            report_interval_minutes: Interval between performance reports
        """
        logger.info(
            "🚀 Starting continuous performance monitoring (reports every %smin)",
            report_interval_minutes,
        )

        while True:
            try:
                await asyncio.sleep(report_interval_minutes * 60)

                stats = self.get_current_stats()

                # Log performance summary
                logger.info(
                    "📊 Performance Summary: Avg=%.1fms, Cache=%.1f%%, Target=%s",
                    stats.avg_response_time_ms,
                    stats.cache_hit_rate,
                    "✅" if stats.target_achieved else "❌",
                )

                # Generate full report if requested or if performance is poor
                if not stats.target_achieved or stats.error_rate > 10:
                    logger.info(self.get_performance_report())

            except asyncio.CancelledError:
                logger.info("Performance monitoring stopped")
                break
            except Exception:
                logger.exception("Error in performance monitoring")

    def get_cache_effectiveness_analysis(self) -> dict[str, Any]:
        """
        Analyze cache effectiveness and provide recommendations.

        Returns:
            Dictionary with cache analysis
        """
        if not self.metrics_history:
            return {"status": "insufficient_data"}

        recent_metrics = self.metrics_history[-100:]  # Last 100 requests

        cache_hits = [m for m in recent_metrics if m.cache_hit]
        cache_misses = [m for m in recent_metrics if not m.cache_hit]

        cache_hit_times = [m.response_time_ms for m in cache_hits]
        cache_miss_times = [m.response_time_ms for m in cache_misses]

        analysis: dict[str, Any] = {
            "cache_hit_rate": len(cache_hits) / len(recent_metrics) * 100,
            "avg_cache_hit_time_ms": mean(cache_hit_times) if cache_hit_times else 0,
            "avg_cache_miss_time_ms": mean(cache_miss_times) if cache_miss_times else 0,
            "cache_speed_improvement": 0,
            "recommendations": [],
        }

        if cache_hit_times and cache_miss_times:
            hit_avg = mean(cache_hit_times)
            miss_avg = mean(cache_miss_times)
            analysis["cache_speed_improvement"] = (
                (miss_avg - hit_avg) / miss_avg
            ) * 100

        # Generate recommendations
        recommendations: list[str] = analysis["recommendations"]  # type: ignore[assignment]
        if analysis["cache_hit_rate"] < 50:
            recommendations.append("Increase cache TTL")
        if analysis["cache_hit_rate"] < 30:
            recommendations.append("Improve cache key similarity algorithm")
        if analysis["cache_speed_improvement"] < 70:
            recommendations.append("Optimize cache lookup performance")

        return analysis


# Global performance monitor instance
_global_monitor: LLMPerformanceMonitor | None = None


def get_performance_monitor() -> LLMPerformanceMonitor:
    """
    Get or create the global performance monitor instance.

    Returns:
        Global performance monitor instance
    """
    global _global_monitor  # pylint: disable=global-statement # noqa: PLW0603

    if _global_monitor is None:
        _global_monitor = LLMPerformanceMonitor()

    return _global_monitor


def record_llm_performance(
    response_time_ms: float,
    prompt_size_chars: int,
    cache_hit: bool,
    decision_action: str,
    optimization_level: str = "standard",
    error_occurred: bool = False,
):
    """
    Convenience function to record LLM performance metrics.

    Args:
        response_time_ms: Response time in milliseconds
        prompt_size_chars: Size of the prompt in characters
        cache_hit: Whether this was a cache hit
        decision_action: The trading action decided
        optimization_level: Level of optimization applied
        error_occurred: Whether an error occurred during processing
    """
    monitor = get_performance_monitor()
    monitor.record_request(
        response_time_ms=response_time_ms,
        prompt_size_chars=prompt_size_chars,
        cache_hit=cache_hit,
        decision_action=decision_action,
        optimization_level=optimization_level,
        error_occurred=error_occurred,
    )
