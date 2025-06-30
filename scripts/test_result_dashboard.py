#!/usr/bin/env python3
"""
Test Result Dashboard
Web-based dashboard for visualizing test execution results and log analysis.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request


class TestResultDatabase:
    """SQLite database for storing test results and metrics"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    container TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL,
                    error_message TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    container TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    timestamp DATETIME NOT NULL,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS log_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    container TEXT NOT NULL,
                    level TEXT NOT NULL,
                    logger TEXT,
                    message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS container_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    container TEXT NOT NULL,
                    state TEXT NOT NULL,
                    health TEXT,
                    uptime_seconds REAL,
                    restart_count INTEGER,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_test_results_timestamp ON test_results(timestamp);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_log_events_timestamp ON log_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_container_status_timestamp ON container_status(timestamp);
            """
            )

    def insert_test_result(
        self,
        test_name: str,
        container: str,
        status: str,
        duration: float = None,
        error_message: str = None,
        timestamp: datetime = None,
    ):
        """Insert test result"""
        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO test_results (test_name, container, status, duration, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (test_name, container, status, duration, error_message, timestamp),
            )

    def insert_performance_metric(
        self,
        container: str,
        metric_name: str,
        value: float,
        unit: str = None,
        context: str = None,
        timestamp: datetime = None,
    ):
        """Insert performance metric"""
        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO performance_metrics (container, metric_name, value, unit, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (container, metric_name, value, unit, timestamp, context),
            )

    def insert_log_event(
        self,
        container: str,
        level: str,
        message: str,
        logger: str = None,
        timestamp: datetime = None,
    ):
        """Insert log event"""
        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO log_events (container, level, logger, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (container, level, logger, message, timestamp),
            )

    def get_test_summary(self, hours: int = 24) -> dict:
        """Get test execution summary"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Overall stats
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_tests,
                    SUM(CASE WHEN status = 'PASSED' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'SKIPPED' THEN 1 ELSE 0 END) as skipped,
                    AVG(duration) as avg_duration,
                    SUM(duration) as total_duration
                FROM test_results
                WHERE timestamp >= ?
            """,
                (since,),
            ).fetchone()

            # Tests by container
            by_container = conn.execute(
                """
                SELECT
                    container,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'PASSED' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    AVG(duration) as avg_duration
                FROM test_results
                WHERE timestamp >= ?
                GROUP BY container
                ORDER BY total DESC
            """,
                (since,),
            ).fetchall()

            # Recent failures
            failures = conn.execute(
                """
                SELECT test_name, container, error_message, timestamp, duration
                FROM test_results
                WHERE timestamp >= ? AND status = 'FAILED'
                ORDER BY timestamp DESC
                LIMIT 10
            """,
                (since,),
            ).fetchall()

            # Slow tests
            slow_tests = conn.execute(
                """
                SELECT test_name, container, duration, timestamp
                FROM test_results
                WHERE timestamp >= ? AND duration > 30
                ORDER BY duration DESC
                LIMIT 10
            """,
                (since,),
            ).fetchall()

            return {
                "summary": dict(stats) if stats else {},
                "by_container": [dict(row) for row in by_container],
                "recent_failures": [dict(row) for row in failures],
                "slow_tests": [dict(row) for row in slow_tests],
            }

    def get_performance_metrics(self, hours: int = 24) -> dict:
        """Get performance metrics summary"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Metrics by type
            metrics = conn.execute(
                """
                SELECT
                    metric_name,
                    container,
                    AVG(value) as avg_value,
                    MAX(value) as max_value,
                    MIN(value) as min_value,
                    COUNT(*) as count,
                    unit
                FROM performance_metrics
                WHERE timestamp >= ?
                GROUP BY metric_name, container, unit
                ORDER BY metric_name, container
            """,
                (since,),
            ).fetchall()

            # Time series data for key metrics
            cpu_series = conn.execute(
                """
                SELECT container, timestamp, value
                FROM performance_metrics
                WHERE timestamp >= ? AND metric_name = 'cpu_usage'
                ORDER BY timestamp
            """,
                (since,),
            ).fetchall()

            memory_series = conn.execute(
                """
                SELECT container, timestamp, value
                FROM performance_metrics
                WHERE timestamp >= ? AND metric_name = 'memory_usage'
                ORDER BY timestamp
            """,
                (since,),
            ).fetchall()

            return {
                "metrics_summary": [dict(row) for row in metrics],
                "cpu_time_series": [dict(row) for row in cpu_series],
                "memory_time_series": [dict(row) for row in memory_series],
            }

    def get_log_summary(self, hours: int = 24) -> dict:
        """Get log events summary"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Log level distribution
            levels = conn.execute(
                """
                SELECT
                    level,
                    container,
                    COUNT(*) as count
                FROM log_events
                WHERE timestamp >= ?
                GROUP BY level, container
                ORDER BY count DESC
            """,
                (since,),
            ).fetchall()

            # Recent errors
            errors = conn.execute(
                """
                SELECT container, message, timestamp
                FROM log_events
                WHERE timestamp >= ? AND level IN ('ERROR', 'CRITICAL')
                ORDER BY timestamp DESC
                LIMIT 20
            """,
                (since,),
            ).fetchall()

            # Log activity timeline
            timeline = conn.execute(
                """
                SELECT
                    datetime(timestamp, 'localtime') as hour,
                    level,
                    COUNT(*) as count
                FROM log_events
                WHERE timestamp >= ?
                GROUP BY hour, level
                ORDER BY hour
            """,
                (since,),
            ).fetchall()

            return {
                "level_distribution": [dict(row) for row in levels],
                "recent_errors": [dict(row) for row in errors],
                "timeline": [dict(row) for row in timeline],
            }


class TestDashboard:
    """Flask web dashboard for test results"""

    def __init__(self, db_path: Path, logs_dir: Path = None):
        self.db = TestResultDatabase(db_path)
        self.logs_dir = logs_dir or Path("/app/logs")
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route("/")
        def dashboard():
            return render_template_string(self.get_dashboard_template())

        @self.app.route("/api/test-summary")
        def api_test_summary():
            hours = request.args.get("hours", 24, type=int)
            return jsonify(self.db.get_test_summary(hours))

        @self.app.route("/api/performance-metrics")
        def api_performance_metrics():
            hours = request.args.get("hours", 24, type=int)
            return jsonify(self.db.get_performance_metrics(hours))

        @self.app.route("/api/log-summary")
        def api_log_summary():
            hours = request.args.get("hours", 24, type=int)
            return jsonify(self.db.get_log_summary(hours))

        @self.app.route("/api/refresh")
        def api_refresh():
            """Refresh data from log files"""
            try:
                self.load_data_from_logs()
                return jsonify({"status": "success", "message": "Data refreshed"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

    def load_data_from_logs(self):
        """Load data from log files into database"""
        # Load test results
        for test_file in self.logs_dir.glob("*-tests.jsonl"):
            container = test_file.stem.replace("-tests", "")

            with open(test_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get("event_type") in ["pass", "fail", "skip"]:
                            status_map = {
                                "pass": "PASSED",
                                "fail": "FAILED",
                                "skip": "SKIPPED",
                            }
                            self.db.insert_test_result(
                                test_name=event["test_name"],
                                container=event["container"],
                                status=status_map[event["event_type"]],
                                duration=event.get("duration"),
                                error_message=event.get("error_message"),
                                timestamp=datetime.fromisoformat(event["timestamp"]),
                            )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        # Load metrics from container monitor files
        for metrics_file in self.logs_dir.glob("*-metrics.jsonl"):
            container = metrics_file.stem.replace("-metrics", "")

            with open(metrics_file) as f:
                for line in f:
                    try:
                        snapshot = json.loads(line)
                        timestamp = datetime.fromtimestamp(snapshot["timestamp"])

                        # Insert various metrics
                        metrics = [
                            ("cpu_usage", snapshot["cpu_percent"], "%"),
                            ("memory_usage", snapshot["memory_percent"], "%"),
                            ("memory_used", snapshot["memory_used_mb"], "MB"),
                            ("disk_usage", snapshot["disk_usage_percent"], "%"),
                            ("process_count", snapshot["process_count"], "count"),
                            ("tcp_connections", snapshot["tcp_connections"], "count"),
                        ]

                        for metric_name, value, unit in metrics:
                            self.db.insert_performance_metric(
                                container=container,
                                metric_name=metric_name,
                                value=value,
                                unit=unit,
                                timestamp=timestamp,
                            )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        # Load log events
        for log_file in self.logs_dir.glob("*-logs.jsonl"):
            container = log_file.stem.replace("-logs", "")

            with open(log_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        self.db.insert_log_event(
                            container=event["container"],
                            level=event["level"],
                            message=event["message"],
                            logger=event.get("logger"),
                            timestamp=datetime.fromisoformat(event["timestamp"]),
                        )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

    def get_dashboard_template(self) -> str:
        """Get HTML template for dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-value {
            font-weight: bold;
            font-size: 1.2em;
        }
        .success { color: #28a745; }
        .danger { color: #dc3545; }
        .warning { color: #ffc107; }
        .info { color: #17a2b8; }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-online { background-color: #28a745; }
        .status-offline { background-color: #dc3545; }
        .status-warning { background-color: #ffc107; }
        .refresh-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 0;
        }
        .refresh-btn:hover {
            background: #0056b3;
        }
        .test-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .test-item {
            padding: 8px;
            border-left: 4px solid #eee;
            margin: 5px 0;
            background: #f8f9fa;
        }
        .test-item.failed {
            border-left-color: #dc3545;
        }
        .test-item.passed {
            border-left-color: #28a745;
        }
        .test-item.slow {
            border-left-color: #ffc107;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Test Results Dashboard</h1>
        <p>Real-time monitoring of test execution and system performance</p>
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
    </div>

    <div class="dashboard">
        <!-- Test Summary Card -->
        <div class="card">
            <h3>📊 Test Summary (24h)</h3>
            <div id="test-summary">Loading...</div>
        </div>

        <!-- Container Status Card -->
        <div class="card">
            <h3>🐳 Container Status</h3>
            <div id="container-status">Loading...</div>
        </div>

        <!-- Performance Metrics Card -->
        <div class="card">
            <h3>⚡ Performance Metrics</h3>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>

        <!-- Recent Failures Card -->
        <div class="card">
            <h3>❌ Recent Test Failures</h3>
            <div id="recent-failures" class="test-list">Loading...</div>
        </div>

        <!-- Slow Tests Card -->
        <div class="card">
            <h3>🐌 Slow Tests (>30s)</h3>
            <div id="slow-tests" class="test-list">Loading...</div>
        </div>

        <!-- Log Activity Card -->
        <div class="card">
            <h3>📝 Log Activity</h3>
            <div class="chart-container">
                <canvas id="log-chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let performanceChart, logChart;

        async function loadTestSummary() {
            try {
                const response = await fetch('/api/test-summary');
                const data = await response.json();

                const summary = data.summary;
                const passRate = summary.total_tests > 0 ?
                    (summary.passed / summary.total_tests * 100).toFixed(1) : 0;

                document.getElementById('test-summary').innerHTML = `
                    <div class="metric">
                        <span>Total Tests</span>
                        <span class="metric-value">${summary.total_tests || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Passed</span>
                        <span class="metric-value success">${summary.passed || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Failed</span>
                        <span class="metric-value danger">${summary.failed || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Skipped</span>
                        <span class="metric-value warning">${summary.skipped || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Pass Rate</span>
                        <span class="metric-value ${passRate >= 90 ? 'success' : passRate >= 70 ? 'warning' : 'danger'}">${passRate}%</span>
                    </div>
                    <div class="metric">
                        <span>Avg Duration</span>
                        <span class="metric-value">${(summary.avg_duration || 0).toFixed(2)}s</span>
                    </div>
                `;

                // Update container status
                let containerHtml = '';
                data.by_container.forEach(container => {
                    const passRate = container.total > 0 ?
                        (container.passed / container.total * 100).toFixed(1) : 0;
                    const statusClass = passRate >= 90 ? 'status-online' :
                                       passRate >= 70 ? 'status-warning' : 'status-offline';

                    containerHtml += `
                        <div class="metric">
                            <span><span class="status-indicator ${statusClass}"></span>${container.container}</span>
                            <span class="metric-value">${passRate}% (${container.passed}/${container.total})</span>
                        </div>
                    `;
                });
                document.getElementById('container-status').innerHTML = containerHtml || 'No data available';

                // Update recent failures
                let failuresHtml = '';
                data.recent_failures.forEach(failure => {
                    failuresHtml += `
                        <div class="test-item failed">
                            <strong>${failure.test_name}</strong><br>
                            <small>${failure.container} • ${new Date(failure.timestamp).toLocaleString()}</small><br>
                            <small>${failure.error_message || 'No error message'}</small>
                        </div>
                    `;
                });
                document.getElementById('recent-failures').innerHTML = failuresHtml || 'No recent failures';

                // Update slow tests
                let slowHtml = '';
                data.slow_tests.forEach(test => {
                    slowHtml += `
                        <div class="test-item slow">
                            <strong>${test.test_name}</strong><br>
                            <small>${test.container} • ${test.duration.toFixed(2)}s • ${new Date(test.timestamp).toLocaleString()}</small>
                        </div>
                    `;
                });
                document.getElementById('slow-tests').innerHTML = slowHtml || 'No slow tests';

            } catch (error) {
                console.error('Error loading test summary:', error);
            }
        }

        async function loadPerformanceMetrics() {
            try {
                const response = await fetch('/api/performance-metrics');
                const data = await response.json();

                // Update or create performance chart
                const ctx = document.getElementById('performance-chart').getContext('2d');

                if (performanceChart) {
                    performanceChart.destroy();
                }

                performanceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [
                            {
                                label: 'CPU Usage (%)',
                                data: data.cpu_time_series.map(d => ({
                                    x: new Date(d.timestamp),
                                    y: d.value
                                })),
                                borderColor: 'rgb(255, 99, 132)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)'
                            },
                            {
                                label: 'Memory Usage (%)',
                                data: data.memory_time_series.map(d => ({
                                    x: new Date(d.timestamp),
                                    y: d.value
                                })),
                                borderColor: 'rgb(54, 162, 235)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'hour'
                                }
                            },
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('Error loading performance metrics:', error);
            }
        }

        async function loadLogSummary() {
            try {
                const response = await fetch('/api/log-summary');
                const data = await response.json();

                // Update or create log chart
                const ctx = document.getElementById('log-chart').getContext('2d');

                if (logChart) {
                    logChart.destroy();
                }

                // Group timeline data by level
                const timelineData = {};
                data.timeline.forEach(entry => {
                    if (!timelineData[entry.level]) {
                        timelineData[entry.level] = {};
                    }
                    timelineData[entry.level][entry.hour] = entry.count;
                });

                const hours = [...new Set(data.timeline.map(d => d.hour))].sort();
                const datasets = Object.keys(timelineData).map(level => ({
                    label: level,
                    data: hours.map(hour => timelineData[level][hour] || 0),
                    backgroundColor: {
                        'ERROR': 'rgba(220, 53, 69, 0.8)',
                        'WARNING': 'rgba(255, 193, 7, 0.8)',
                        'INFO': 'rgba(23, 162, 184, 0.8)',
                        'DEBUG': 'rgba(108, 117, 125, 0.8)'
                    }[level] || 'rgba(0, 0, 0, 0.8)'
                }));

                logChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: hours,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                stacked: true
                            },
                            y: {
                                stacked: true,
                                beginAtZero: true
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('Error loading log summary:', error);
            }
        }

        async function refreshData() {
            const button = document.querySelector('.refresh-btn');
            button.textContent = '🔄 Refreshing...';
            button.disabled = true;

            try {
                await fetch('/api/refresh');
                await loadDashboard();
            } catch (error) {
                console.error('Error refreshing data:', error);
            } finally {
                button.textContent = '🔄 Refresh Data';
                button.disabled = false;
            }
        }

        async function loadDashboard() {
            await Promise.all([
                loadTestSummary(),
                loadPerformanceMetrics(),
                loadLogSummary()
            ]);
        }

        // Load dashboard on page load
        loadDashboard();

        // Auto-refresh every 5 minutes
        setInterval(loadDashboard, 5 * 60 * 1000);
    </script>
</body>
</html>
        """

    def run(self, host="0.0.0.0", port=8080, debug=False):
        """Run the dashboard server"""
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Results Dashboard")
    parser.add_argument(
        "--db-path", type=Path, default="test_results.db", help="Database file path"
    )
    parser.add_argument("--logs-dir", type=Path, help="Logs directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--load-data", action="store_true", help="Load data from logs on startup"
    )

    args = parser.parse_args()

    dashboard = TestDashboard(args.db_path, args.logs_dir)

    if args.load_data:
        print("Loading data from log files...")
        dashboard.load_data_from_logs()
        print("Data loaded successfully")

    print(f"Starting dashboard server on http://{args.host}:{args.port}")
    dashboard.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
