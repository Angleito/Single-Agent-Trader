"""
Performance Monitoring Scripts for Market Making.

This module provides comprehensive performance monitoring examples and utilities
for tracking market making operations in real-time.

Features:
- Real-time P&L tracking
- Fill rate monitoring
- Signal effectiveness analysis
- Risk metric dashboard
- Performance alerts system
- Data export and reporting
- Custom metric calculations
"""

import asyncio
import json
import logging
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


class RealTimePerformanceMonitor:
    """
    Real-time performance monitoring system.

    Provides live monitoring of market making performance with:
    - Real-time P&L updates
    - Fill rate tracking
    - Signal effectiveness monitoring
    - Risk metrics dashboard
    - Performance alerts
    """

    def __init__(self, symbol: str = "SUI-PERP"):
        """Initialize the real-time monitor."""
        self.symbol = symbol
        self.metrics_history = []
        self.alerts_history = []
        self.start_time = datetime.now(UTC)

        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = Decimal(0)
        self.total_fees = Decimal(0)
        self.max_drawdown = Decimal(0)
        self.peak_pnl = Decimal(0)

        # Signal tracking
        self.signal_count = 0
        self.successful_signals = 0
        self.signal_pnl = Decimal(0)

        # Fill tracking
        self.orders_placed = 0
        self.orders_filled = 0
        self.fill_times = []

        console.print(
            Panel.fit(
                f"[bold green]Real-Time Performance Monitor[/bold green]\n\n"
                f"Symbol: {symbol}\n"
                f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                f"Monitoring: P&L, Fills, Signals, Risk Metrics",
                title="📊 Performance Monitor",
            )
        )

    async def start_monitoring(self, duration_minutes: int = 30) -> None:
        """Start real-time monitoring for specified duration."""
        end_time = self.start_time + timedelta(minutes=duration_minutes)

        with Live(self._create_dashboard(), refresh_per_second=2) as live:
            while datetime.now(UTC) < end_time:
                # Simulate market making activity
                await self._simulate_trading_activity()

                # Update metrics
                self._update_metrics()

                # Check for alerts
                alerts = self._check_performance_alerts()
                if alerts:
                    self._handle_alerts(alerts)

                # Update dashboard
                live.update(self._create_dashboard())

                await asyncio.sleep(0.5)

        # Generate final report
        self._generate_final_report()

    async def _simulate_trading_activity(self) -> None:
        """Simulate trading activity for demonstration."""
        import random

        # Simulate order placement and fills
        if random.random() < 0.3:  # 30% chance of new order
            self.orders_placed += 1

            # Simulate fill
            if random.random() < 0.6:  # 60% fill rate
                self.orders_filled += 1
                fill_time = random.uniform(0.1, 5.0)  # 0.1 to 5 seconds
                self.fill_times.append(fill_time)

                # Simulate P&L
                trade_pnl = Decimal(
                    str(random.uniform(-10, 25))
                )  # Slight positive bias
                self.total_pnl += trade_pnl
                self.total_trades += 1

                if trade_pnl > 0:
                    self.successful_trades += 1

                # Track fees
                fee = Decimal(str(random.uniform(0.5, 2.0)))
                self.total_fees += fee

                # Update drawdown tracking
                self.peak_pnl = max(self.peak_pnl, self.total_pnl)

                current_drawdown = self.peak_pnl - self.total_pnl
                self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Simulate signal generation
        if random.random() < 0.1:  # 10% chance of new signal
            self.signal_count += 1

            # Simulate signal outcome
            if random.random() < 0.65:  # 65% success rate
                self.successful_signals += 1
                signal_pnl = Decimal(str(random.uniform(5, 20)))
                self.signal_pnl += signal_pnl

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        current_time = datetime.now(UTC)
        runtime_minutes = (current_time - self.start_time).total_seconds() / 60

        metrics = {
            "timestamp": current_time,
            "runtime_minutes": runtime_minutes,
            "total_pnl": float(self.total_pnl),
            "win_rate": self.successful_trades / max(self.total_trades, 1),
            "fill_rate": self.orders_filled / max(self.orders_placed, 1),
            "signal_success_rate": self.successful_signals / max(self.signal_count, 1),
            "avg_fill_time": sum(self.fill_times) / max(len(self.fill_times), 1),
            "max_drawdown": float(self.max_drawdown),
            "fee_percentage": float(
                self.total_fees / max(abs(self.total_pnl), Decimal(1)) * 100
            ),
            "trades_per_minute": self.total_trades / max(runtime_minutes, 1),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
        }

        self.metrics_history.append(metrics)

        # Keep only last 100 metrics to prevent memory growth
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate approximate Sharpe ratio."""
        if len(self.metrics_history) < 10:
            return 0.0

        # Use recent P&L changes as returns
        recent_pnl = [m["total_pnl"] for m in self.metrics_history[-10:]]
        if len(recent_pnl) < 2:
            return 0.0

        returns = [recent_pnl[i] - recent_pnl[i - 1] for i in range(1, len(recent_pnl))]

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        if len(returns) == 1:
            return 0.0

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001

        return mean_return / std_dev

    def _check_performance_alerts(self) -> list[dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []

        # Check win rate
        win_rate = self.successful_trades / max(self.total_trades, 1)
        if win_rate < 0.4 and self.total_trades > 10:
            alerts.append(
                {
                    "type": "WIN_RATE_LOW",
                    "message": f"Win rate below 40%: {win_rate:.1%}",
                    "severity": "WARNING",
                    "value": win_rate,
                }
            )

        # Check fill rate
        fill_rate = self.orders_filled / max(self.orders_placed, 1)
        if fill_rate < 0.3 and self.orders_placed > 20:
            alerts.append(
                {
                    "type": "FILL_RATE_LOW",
                    "message": f"Fill rate below 30%: {fill_rate:.1%}",
                    "severity": "WARNING",
                    "value": fill_rate,
                }
            )

        # Check drawdown
        if self.max_drawdown > 100:  # $100 drawdown
            alerts.append(
                {
                    "type": "HIGH_DRAWDOWN",
                    "message": f"High drawdown: ${self.max_drawdown:.2f}",
                    "severity": "CRITICAL",
                    "value": float(self.max_drawdown),
                }
            )

        # Check fee percentage
        fee_pct = float(self.total_fees / max(abs(self.total_pnl), Decimal(1)) * 100)
        if fee_pct > 50 and abs(self.total_pnl) > 10:
            alerts.append(
                {
                    "type": "HIGH_FEES",
                    "message": f"High fee percentage: {fee_pct:.1f}%",
                    "severity": "WARNING",
                    "value": fee_pct,
                }
            )

        return alerts

    def _handle_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """Handle performance alerts."""
        for alert in alerts:
            self.alerts_history.append({**alert, "timestamp": datetime.now(UTC)})

            # Log critical alerts
            if alert["severity"] == "CRITICAL":
                logger.warning("CRITICAL ALERT: %s", alert["message"])

    def _create_dashboard(self) -> Table:
        """Create real-time performance dashboard."""
        # Main metrics table
        table = Table(title=f"🚀 Real-Time Performance Dashboard - {self.symbol}")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="magenta", width=15)
        table.add_column("Status", style="green", width=10)
        table.add_column("Trend", style="yellow", width=10)

        # Runtime
        runtime = datetime.now(UTC) - self.start_time
        table.add_row("Runtime", str(runtime).split(".")[0], "🕐", "▶️")

        # P&L metrics
        pnl_status = "🟢" if self.total_pnl >= 0 else "🔴"
        pnl_trend = (
            "📈"
            if len(self.metrics_history) > 1
            and self.metrics_history[-1]["total_pnl"]
            > self.metrics_history[-2]["total_pnl"]
            else "📉"
        )
        table.add_row("Total P&L", f"${self.total_pnl:.2f}", pnl_status, pnl_trend)

        # Trading metrics
        win_rate = self.successful_trades / max(self.total_trades, 1)
        win_status = "🟢" if win_rate >= 0.5 else "🟡" if win_rate >= 0.4 else "🔴"
        table.add_row("Win Rate", f"{win_rate:.1%}", win_status, "📊")

        fill_rate = self.orders_filled / max(self.orders_placed, 1)
        fill_status = "🟢" if fill_rate >= 0.5 else "🟡" if fill_rate >= 0.3 else "🔴"
        table.add_row("Fill Rate", f"{fill_rate:.1%}", fill_status, "🎯")

        # Signal performance
        signal_rate = self.successful_signals / max(self.signal_count, 1)
        signal_status = (
            "🟢" if signal_rate >= 0.6 else "🟡" if signal_rate >= 0.5 else "🔴"
        )
        table.add_row("Signal Success", f"{signal_rate:.1%}", signal_status, "🧠")

        # Risk metrics
        drawdown_status = (
            "🟢"
            if self.max_drawdown < 50
            else "🟡" if self.max_drawdown < 100 else "🔴"
        )
        table.add_row("Max Drawdown", f"${self.max_drawdown:.2f}", drawdown_status, "⚠️")

        # Fee analysis
        fee_pct = float(self.total_fees / max(abs(self.total_pnl), Decimal(1)) * 100)
        fee_status = "🟢" if fee_pct < 30 else "🟡" if fee_pct < 50 else "🔴"
        table.add_row("Fee Percentage", f"{fee_pct:.1f}%", fee_status, "💰")

        # Performance metrics
        avg_fill_time = sum(self.fill_times) / max(len(self.fill_times), 1)
        speed_status = (
            "🟢" if avg_fill_time < 2 else "🟡" if avg_fill_time < 5 else "🔴"
        )
        table.add_row("Avg Fill Time", f"{avg_fill_time:.2f}s", speed_status, "⚡")

        sharpe_ratio = self._calculate_sharpe_ratio()
        sharpe_status = "🟢" if sharpe_ratio > 1 else "🟡" if sharpe_ratio > 0 else "🔴"
        table.add_row("Sharpe Ratio", f"{sharpe_ratio:.2f}", sharpe_status, "📈")

        # Recent alerts
        recent_alerts = [
            a
            for a in self.alerts_history
            if (datetime.now(UTC) - a["timestamp"]).total_seconds() < 60
        ]
        alert_status = (
            "🔴"
            if any(a["severity"] == "CRITICAL" for a in recent_alerts)
            else "🟡" if recent_alerts else "🟢"
        )
        table.add_row("Recent Alerts", str(len(recent_alerts)), alert_status, "🚨")

        return table

    def _generate_final_report(self) -> None:
        """Generate final performance report."""
        console.print("\n" + "=" * 80)
        console.print("[bold green]FINAL PERFORMANCE REPORT[/bold green]")
        console.print("=" * 80)

        runtime = datetime.now(UTC) - self.start_time

        # Summary metrics
        summary_table = Table(title="Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        summary_table.add_column("Rating", style="green")

        summary_table.add_row("Total Runtime", str(runtime).split(".")[0], "ℹ️")
        summary_table.add_row(
            "Total P&L", f"${self.total_pnl:.2f}", "🟢" if self.total_pnl > 0 else "🔴"
        )
        summary_table.add_row("Total Trades", str(self.total_trades), "📊")
        summary_table.add_row(
            "Win Rate",
            f"{self.successful_trades / max(self.total_trades, 1):.1%}",
            "📈",
        )
        summary_table.add_row(
            "Fill Rate", f"{self.orders_filled / max(self.orders_placed, 1):.1%}", "🎯"
        )
        summary_table.add_row(
            "Signal Success",
            f"{self.successful_signals / max(self.signal_count, 1):.1%}",
            "🧠",
        )
        summary_table.add_row("Max Drawdown", f"${self.max_drawdown:.2f}", "⚠️")
        summary_table.add_row("Total Fees", f"${self.total_fees:.2f}", "💰")

        console.print(summary_table)

        # Export data
        self._export_performance_data()

    def _export_performance_data(self) -> None:
        """Export performance data to files."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Create exports directory
        exports_dir = Path("data/performance_exports")
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Export metrics history
        metrics_file = exports_dir / f"performance_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

        # Export alerts history
        alerts_file = exports_dir / f"performance_alerts_{timestamp}.json"
        with open(alerts_file, "w") as f:
            json.dump(self.alerts_history, f, indent=2, default=str)

        # Export summary
        summary = {
            "symbol": self.symbol,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now(UTC).isoformat(),
            "total_pnl": float(self.total_pnl),
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "signal_count": self.signal_count,
            "successful_signals": self.successful_signals,
            "max_drawdown": float(self.max_drawdown),
            "total_fees": float(self.total_fees),
        }

        summary_file = exports_dir / f"performance_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        console.print("\n✅ Performance data exported:")
        console.print(f"   • Metrics: {metrics_file}")
        console.print(f"   • Alerts: {alerts_file}")
        console.print(f"   • Summary: {summary_file}")


class PerformanceAnalyzer:
    """
    Historical performance analysis utility.

    Analyzes historical market making performance data to identify:
    - Performance patterns
    - Optimization opportunities
    - Risk factors
    - Strategy effectiveness
    """

    def __init__(self):
        """Initialize the performance analyzer."""
        self.console = Console()

    def analyze_historical_data(self, data_file: str) -> dict[str, Any]:
        """
        Analyze historical performance data.

        Args:
            data_file: Path to historical data file

        Returns:
            Analysis results
        """
        try:
            with open(data_file) as f:
                data = json.load(f)

            analysis = {
                "summary": self._analyze_summary_stats(data),
                "trends": self._analyze_trends(data),
                "patterns": self._identify_patterns(data),
                "risk_analysis": self._analyze_risk_metrics(data),
                "recommendations": self._generate_recommendations(data),
            }

            self._display_analysis(analysis)
            return analysis

        except Exception as e:
            logger.exception("Error analyzing historical data: %s", e)
            return {}

    def _analyze_summary_stats(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze summary statistics."""
        if not data:
            return {}

        pnl_values = [d["total_pnl"] for d in data if "total_pnl" in d]
        win_rates = [d["win_rate"] for d in data if "win_rate" in d]
        fill_rates = [d["fill_rate"] for d in data if "fill_rate" in d]

        return {
            "total_data_points": len(data),
            "pnl_stats": {
                "mean": sum(pnl_values) / len(pnl_values) if pnl_values else 0,
                "min": min(pnl_values) if pnl_values else 0,
                "max": max(pnl_values) if pnl_values else 0,
                "final": pnl_values[-1] if pnl_values else 0,
            },
            "win_rate_stats": {
                "mean": sum(win_rates) / len(win_rates) if win_rates else 0,
                "min": min(win_rates) if win_rates else 0,
                "max": max(win_rates) if win_rates else 0,
            },
            "fill_rate_stats": {
                "mean": sum(fill_rates) / len(fill_rates) if fill_rates else 0,
                "min": min(fill_rates) if fill_rates else 0,
                "max": max(fill_rates) if fill_rates else 0,
            },
        }

    def _analyze_trends(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze performance trends."""
        if len(data) < 2:
            return {}

        # Calculate trends for key metrics
        pnl_values = [d["total_pnl"] for d in data if "total_pnl" in d]
        win_rates = [d["win_rate"] for d in data if "win_rate" in d]

        return {
            "pnl_trend": (
                "increasing" if pnl_values[-1] > pnl_values[0] else "decreasing"
            ),
            "pnl_volatility": self._calculate_volatility(pnl_values),
            "win_rate_trend": (
                "improving" if win_rates[-1] > win_rates[0] else "declining"
            ),
            "performance_consistency": self._calculate_consistency(data),
        }

    def _identify_patterns(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Identify performance patterns."""
        patterns = {
            "best_performance_periods": [],
            "worst_performance_periods": [],
            "volatility_spikes": [],
            "consistent_performance_periods": [],
        }

        # Analyze P&L patterns
        pnl_values = [d["total_pnl"] for d in data if "total_pnl" in d]
        if len(pnl_values) >= 10:
            # Find best and worst periods
            window_size = min(10, len(pnl_values) // 4)
            for i in range(len(pnl_values) - window_size + 1):
                window = pnl_values[i : i + window_size]
                window_return = window[-1] - window[0]

                if window_return > 50:  # Good performance threshold
                    patterns["best_performance_periods"].append(
                        {
                            "start_index": i,
                            "end_index": i + window_size - 1,
                            "return": window_return,
                        }
                    )
                elif window_return < -30:  # Poor performance threshold
                    patterns["worst_performance_periods"].append(
                        {
                            "start_index": i,
                            "end_index": i + window_size - 1,
                            "return": window_return,
                        }
                    )

        return patterns

    def _analyze_risk_metrics(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze risk metrics."""
        drawdowns = [d.get("max_drawdown", 0) for d in data]
        sharpe_ratios = [
            d.get("sharpe_ratio", 0) for d in data if d.get("sharpe_ratio", 0) != 0
        ]

        return {
            "max_historical_drawdown": max(drawdowns) if drawdowns else 0,
            "average_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
            "average_sharpe_ratio": (
                sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
            ),
            "risk_score": self._calculate_risk_score(data),
        }

    def _generate_recommendations(self, data: list[dict[str, Any]]) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Analyze recent performance
        recent_data = data[-10:] if len(data) >= 10 else data

        # Check win rate
        recent_win_rates = [d["win_rate"] for d in recent_data if "win_rate" in d]
        avg_win_rate = (
            sum(recent_win_rates) / len(recent_win_rates) if recent_win_rates else 0
        )

        if avg_win_rate < 0.4:
            recommendations.append("Consider tightening spreads to improve win rate")
        elif avg_win_rate > 0.8:
            recommendations.append(
                "Win rate very high - consider widening spreads for better margins"
            )

        # Check fill rate
        recent_fill_rates = [d["fill_rate"] for d in recent_data if "fill_rate" in d]
        avg_fill_rate = (
            sum(recent_fill_rates) / len(recent_fill_rates) if recent_fill_rates else 0
        )

        if avg_fill_rate < 0.3:
            recommendations.append("Low fill rate - consider more aggressive pricing")

        # Check drawdown
        recent_drawdowns = [d.get("max_drawdown", 0) for d in recent_data]
        max_recent_drawdown = max(recent_drawdowns) if recent_drawdowns else 0

        if max_recent_drawdown > 100:
            recommendations.append(
                "High drawdown detected - consider reducing position sizes"
            )

        return recommendations

    def _calculate_volatility(self, values: list[float]) -> float:
        """Calculate volatility of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _calculate_consistency(self, data: list[dict[str, Any]]) -> float:
        """Calculate performance consistency score."""
        pnl_values = [d["total_pnl"] for d in data if "total_pnl" in d]

        if len(pnl_values) < 3:
            return 0.0

        # Calculate returns
        returns = [pnl_values[i] - pnl_values[i - 1] for i in range(1, len(pnl_values))]

        # Calculate consistency as inverse of volatility
        volatility = self._calculate_volatility(returns)
        return 1.0 / (1.0 + volatility) if volatility > 0 else 1.0

    def _calculate_risk_score(self, data: list[dict[str, Any]]) -> float:
        """Calculate overall risk score (0-100)."""
        if not data:
            return 50.0

        # Factors: drawdown, volatility, consistency
        drawdowns = [d.get("max_drawdown", 0) for d in data]
        max_drawdown = max(drawdowns) if drawdowns else 0

        # Normalize scores (lower is better for risk)
        drawdown_score = min(max_drawdown / 200.0, 1.0)  # 200 as max threshold
        consistency_score = 1.0 - self._calculate_consistency(data)

        # Combine scores
        risk_score = (drawdown_score + consistency_score) / 2.0 * 100
        return min(max(risk_score, 0), 100)

    def _display_analysis(self, analysis: dict[str, Any]) -> None:
        """Display analysis results."""
        console.print(
            Panel.fit(
                "[bold blue]Historical Performance Analysis[/bold blue]",
                title="📊 Analysis Results",
            )
        )

        # Summary statistics
        summary = analysis.get("summary", {})
        if summary:
            console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")

            pnl_stats = summary.get("pnl_stats", {})
            console.print(
                f"  • Total Data Points: {summary.get('total_data_points', 0)}"
            )
            console.print(f"  • Final P&L: ${pnl_stats.get('final', 0):.2f}")
            console.print(
                f"  • P&L Range: ${pnl_stats.get('min', 0):.2f} to ${pnl_stats.get('max', 0):.2f}"
            )
            console.print(
                f"  • Average Win Rate: {summary.get('win_rate_stats', {}).get('mean', 0):.1%}"
            )
            console.print(
                f"  • Average Fill Rate: {summary.get('fill_rate_stats', {}).get('mean', 0):.1%}"
            )

        # Risk analysis
        risk = analysis.get("risk_analysis", {})
        if risk:
            console.print("\n[bold red]Risk Analysis:[/bold red]")
            console.print(f"  • Risk Score: {risk.get('risk_score', 0):.1f}/100")
            console.print(
                f"  • Max Drawdown: ${risk.get('max_historical_drawdown', 0):.2f}"
            )
            console.print(
                f"  • Average Sharpe Ratio: {risk.get('average_sharpe_ratio', 0):.2f}"
            )

        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            console.print("\n[bold green]Recommendations:[/bold green]")
            for i, rec in enumerate(recommendations, 1):
                console.print(f"  {i}. {rec}")


async def main():
    """Main function to run performance monitoring examples."""
    console.print(
        Panel.fit(
            "[bold green]Performance Monitoring Scripts[/bold green]\n\n"
            "Choose monitoring type:\n"
            "1. Real-time performance monitoring\n"
            "2. Historical data analysis\n"
            "3. Both (sequential)",
            title="📊 Performance Monitoring",
        )
    )

    choice = console.input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        monitor = RealTimePerformanceMonitor("SUI-PERP")
        await monitor.start_monitoring(duration_minutes=15)

    elif choice == "2":
        analyzer = PerformanceAnalyzer()

        # Try to find recent data file
        exports_dir = Path("data/performance_exports")
        if exports_dir.exists():
            data_files = list(exports_dir.glob("performance_metrics_*.json"))
            if data_files:
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                console.print(f"Analyzing latest data file: {latest_file}")
                analyzer.analyze_historical_data(str(latest_file))
            else:
                console.print(
                    "[yellow]No historical data files found. Run real-time monitoring first.[/yellow]"
                )
        else:
            console.print(
                "[yellow]No performance data directory found. Run real-time monitoring first.[/yellow]"
            )

    elif choice == "3":
        # Run real-time monitoring first
        console.print("[cyan]Starting real-time monitoring...[/cyan]")
        monitor = RealTimePerformanceMonitor("SUI-PERP")
        await monitor.start_monitoring(duration_minutes=10)

        # Then analyze the generated data
        console.print("\n[cyan]Analyzing generated data...[/cyan]")
        analyzer = PerformanceAnalyzer()

        exports_dir = Path("data/performance_exports")
        if exports_dir.exists():
            data_files = list(exports_dir.glob("performance_metrics_*.json"))
            if data_files:
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                analyzer.analyze_historical_data(str(latest_file))

    else:
        console.print("[red]Invalid choice. Please run again with 1, 2, or 3.[/red]")


if __name__ == "__main__":
    asyncio.run(main())
