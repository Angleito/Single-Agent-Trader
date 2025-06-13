#!/usr/bin/env python3
"""
Docker Performance Monitor for AI Trading Bot

Real-time monitoring and analysis tool that:
- Monitors Docker container performance metrics
- Tracks memory, CPU, I/O, and network usage
- Maintains performance baseline database
- Provides alerts for performance degradation
- Generates optimization recommendations

Usage:
    python scripts/monitor_test_performance.py --realtime
    python scripts/monitor_test_performance.py --baseline --duration 1h
    python scripts/monitor_test_performance.py --compare-baseline
    python scripts/monitor_test_performance.py --generate-recommendations
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class PerformanceMetrics:
    """Container performance metrics snapshot."""
    timestamp: datetime
    container_name: str
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_percent: float
    network_rx_mb: float
    network_tx_mb: float
    block_read_mb: float
    block_write_mb: float
    pids: int
    restart_count: int
    status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class PerformanceDatabase:
    """SQLite database for storing performance baselines and history."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the performance database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    container_name TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_usage_mb REAL,
                    memory_limit_mb REAL,
                    memory_percent REAL,
                    network_rx_mb REAL,
                    network_tx_mb REAL,
                    block_read_mb REAL,
                    block_write_mb REAL,
                    pids INTEGER,
                    restart_count INTEGER,
                    status TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    container_name TEXT NOT NULL,
                    baseline_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    avg_cpu_percent REAL,
                    avg_memory_mb REAL,
                    max_memory_mb REAL,
                    avg_network_rx_mb REAL,
                    avg_network_tx_mb REAL,
                    avg_block_read_mb REAL,
                    avg_block_write_mb REAL,
                    sample_count INTEGER,
                    duration_seconds REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_container_timestamp 
                ON performance_metrics(container_name, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_baselines_container 
                ON performance_baselines(container_name, baseline_type)
            """)
    
    def insert_metrics(self, metrics: List[PerformanceMetrics]):
        """Insert performance metrics into the database."""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, container_name, cpu_percent, memory_usage_mb, 
                     memory_limit_mb, memory_percent, network_rx_mb, network_tx_mb,
                     block_read_mb, block_write_mb, pids, restart_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp.isoformat(),
                    metric.container_name,
                    metric.cpu_percent,
                    metric.memory_usage_mb,
                    metric.memory_limit_mb,
                    metric.memory_percent,
                    metric.network_rx_mb,
                    metric.network_tx_mb,
                    metric.block_read_mb,
                    metric.block_write_mb,
                    metric.pids,
                    metric.restart_count,
                    metric.status
                ))
            conn.commit()
    
    def create_baseline(
        self, 
        container_name: str, 
        baseline_type: str, 
        metrics: List[PerformanceMetrics],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a performance baseline from metrics."""
        if not metrics:
            return
        
        # Calculate baseline statistics
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_usage_mb for m in metrics]
        network_rx_values = [m.network_rx_mb for m in metrics]
        network_tx_values = [m.network_tx_mb for m in metrics]
        block_read_values = [m.block_read_mb for m in metrics]
        block_write_values = [m.block_write_mb for m in metrics]
        
        duration = (metrics[-1].timestamp - metrics[0].timestamp).total_seconds()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_baselines
                (container_name, baseline_type, created_at, avg_cpu_percent,
                 avg_memory_mb, max_memory_mb, avg_network_rx_mb, avg_network_tx_mb,
                 avg_block_read_mb, avg_block_write_mb, sample_count, duration_seconds, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                container_name,
                baseline_type,
                datetime.now().isoformat(),
                sum(cpu_values) / len(cpu_values),
                sum(memory_values) / len(memory_values),
                max(memory_values),
                sum(network_rx_values) / len(network_rx_values),
                sum(network_tx_values) / len(network_tx_values),
                sum(block_read_values) / len(block_read_values),
                sum(block_write_values) / len(block_write_values),
                len(metrics),
                duration,
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    def get_baseline(self, container_name: str, baseline_type: str = 'normal') -> Optional[Dict[str, Any]]:
        """Get the latest baseline for a container."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM performance_baselines
                WHERE container_name = ? AND baseline_type = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (container_name, baseline_type))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        
        return None
    
    def get_historical_metrics(
        self, 
        container_name: str, 
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PerformanceMetrics]:
        """Get historical metrics for a container."""
        query = """
            SELECT * FROM performance_metrics
            WHERE container_name = ?
        """
        params = [container_name]
        
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        metrics = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                metrics.append(PerformanceMetrics(**data))
        
        return metrics

class PerformanceMonitor:
    """Real-time Docker container performance monitor."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.docker_client = docker.from_env()
        self.db = PerformanceDatabase(project_dir / 'data' / 'performance.db')
        self.metrics_buffer = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 90.0,
            'memory_growth_rate_mb_per_min': 50.0,
            'restart_count_increase': 1
        }
        self.active_alerts = {}
    
    async def collect_metrics(self, container_names: Optional[List[str]] = None) -> List[PerformanceMetrics]:
        """Collect current performance metrics from containers."""
        if container_names is None:
            # Auto-discover trading bot containers
            containers = self.docker_client.containers.list(
                filters={'name': 'trading-bot'}
            ) + self.docker_client.containers.list(
                filters={'name': 'dashboard'}
            )
        else:
            containers = []
            for name in container_names:
                try:
                    containers.append(self.docker_client.containers.get(name))
                except docker.errors.NotFound:
                    logger.warning(f"Container {name} not found")
        
        metrics = []
        timestamp = datetime.now()
        
        for container in containers:
            try:
                # Get container stats
                stats = container.stats(stream=False)
                container.reload()
                
                # Parse CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                
                cpu_percent = 0.0
                if system_delta > 0 and cpu_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * \
                                 len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                
                # Parse memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                # Parse network I/O
                network_rx = sum(net['rx_bytes'] for net in stats['networks'].values())
                network_tx = sum(net['tx_bytes'] for net in stats['networks'].values())
                
                # Parse block I/O
                block_read = sum(
                    io['value'] for io in stats['blkio_stats']['io_service_bytes_recursive']
                    if io['op'] == 'read'
                )
                block_write = sum(
                    io['value'] for io in stats['blkio_stats']['io_service_bytes_recursive']
                    if io['op'] == 'write'
                )
                
                # Get PIDs count
                pids = stats['pids_stats']['current'] if 'pids_stats' in stats else 0
                
                metric = PerformanceMetrics(
                    timestamp=timestamp,
                    container_name=container.name,
                    cpu_percent=cpu_percent,
                    memory_usage_mb=memory_usage / (1024 * 1024),
                    memory_limit_mb=memory_limit / (1024 * 1024),
                    memory_percent=memory_percent,
                    network_rx_mb=network_rx / (1024 * 1024),
                    network_tx_mb=network_tx / (1024 * 1024),
                    block_read_mb=block_read / (1024 * 1024),
                    block_write_mb=block_write / (1024 * 1024),
                    pids=pids,
                    restart_count=container.attrs['RestartCount'],
                    status=container.status
                )
                
                metrics.append(metric)
                
            except Exception as e:
                logger.error(f"Failed to collect metrics for {container.name}: {e}")
        
        return metrics
    
    def check_alerts(self, current_metrics: List[PerformanceMetrics]):
        """Check for performance alerts based on current metrics."""
        alerts = []
        
        for metric in current_metrics:
            container_name = metric.container_name
            
            # CPU alert
            if metric.cpu_percent > self.alert_thresholds['cpu_percent']:
                alert = {
                    'type': 'cpu_high',
                    'container': container_name,
                    'message': f"High CPU usage: {metric.cpu_percent:.1f}%",
                    'severity': 'warning' if metric.cpu_percent < 95 else 'critical',
                    'timestamp': metric.timestamp
                }
                alerts.append(alert)
            
            # Memory alert
            if metric.memory_percent > self.alert_thresholds['memory_percent']:
                alert = {
                    'type': 'memory_high',
                    'container': container_name,
                    'message': f"High memory usage: {metric.memory_percent:.1f}%",
                    'severity': 'warning' if metric.memory_percent < 95 else 'critical',
                    'timestamp': metric.timestamp
                }
                alerts.append(alert)
            
            # Memory growth rate alert
            self._check_memory_growth_alert(metric, alerts)
            
            # Restart count alert
            self._check_restart_alert(metric, alerts)
        
        return alerts
    
    def _check_memory_growth_alert(self, current_metric: PerformanceMetrics, alerts: List[Dict]):
        """Check for rapid memory growth."""
        container_name = current_metric.container_name
        
        # Get recent metrics for this container
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m.container_name == container_name and 
               m.timestamp > current_metric.timestamp - timedelta(minutes=5)
        ]
        
        if len(recent_metrics) < 2:
            return
        
        # Calculate memory growth rate
        oldest_metric = min(recent_metrics, key=lambda m: m.timestamp)
        time_diff_minutes = (current_metric.timestamp - oldest_metric.timestamp).total_seconds() / 60
        memory_diff_mb = current_metric.memory_usage_mb - oldest_metric.memory_usage_mb
        
        if time_diff_minutes > 0:
            growth_rate = memory_diff_mb / time_diff_minutes
            
            if growth_rate > self.alert_thresholds['memory_growth_rate_mb_per_min']:
                alert = {
                    'type': 'memory_growth',
                    'container': container_name,
                    'message': f"Rapid memory growth: {growth_rate:.1f} MB/min",
                    'severity': 'warning',
                    'timestamp': current_metric.timestamp
                }
                alerts.append(alert)
    
    def _check_restart_alert(self, current_metric: PerformanceMetrics, alerts: List[Dict]):
        """Check for container restarts."""
        container_name = current_metric.container_name
        
        # Check if restart count increased
        if container_name in self.active_alerts:
            last_restart_count = self.active_alerts[container_name].get('last_restart_count', 0)
            
            if current_metric.restart_count > last_restart_count:
                alert = {
                    'type': 'container_restart',
                    'container': container_name,
                    'message': f"Container restarted (count: {current_metric.restart_count})",
                    'severity': 'critical',
                    'timestamp': current_metric.timestamp
                }
                alerts.append(alert)
        
        # Update restart count tracking
        if container_name not in self.active_alerts:
            self.active_alerts[container_name] = {}
        self.active_alerts[container_name]['last_restart_count'] = current_metric.restart_count
    
    async def monitor_realtime(self, duration: Optional[int] = None, interval: int = 5):
        """Monitor containers in real-time with live display."""
        console.print("[yellow]Starting real-time performance monitoring...[/yellow]")
        
        start_time = time.time()
        
        def create_layout():
            """Create the monitoring layout."""
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            layout["main"].split_row(
                Layout(name="metrics"),
                Layout(name="alerts", ratio=1)
            )
            return layout
        
        layout = create_layout()
        
        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - start_time) > duration:
                    break
                
                try:
                    # Collect metrics
                    metrics = await self.collect_metrics()
                    
                    if metrics:
                        # Add to buffer
                        self.metrics_buffer.extend(metrics)
                        
                        # Store in database
                        self.db.insert_metrics(metrics)
                        
                        # Check for alerts
                        alerts = self.check_alerts(metrics)
                        
                        # Update layout
                        self._update_realtime_layout(layout, metrics, alerts, current_time - start_time)
                    
                    await asyncio.sleep(interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(interval)
        
        console.print("[green]✓ Real-time monitoring completed[/green]")
    
    def _update_realtime_layout(
        self, 
        layout: Layout, 
        metrics: List[PerformanceMetrics], 
        alerts: List[Dict],
        elapsed_time: float
    ):
        """Update the real-time monitoring layout."""
        # Header
        layout["header"].update(Panel(
            f"[bold]AI Trading Bot - Performance Monitor[/bold] | "
            f"Runtime: {elapsed_time:.0f}s | "
            f"Containers: {len(metrics)} | "
            f"Active Alerts: {len(alerts)}",
            style="blue"
        ))
        
        # Metrics table
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Container", style="cyan")
        metrics_table.add_column("CPU %", style="green")
        metrics_table.add_column("Memory", style="yellow")
        metrics_table.add_column("Network RX/TX", style="blue")
        metrics_table.add_column("Block I/O", style="magenta")
        metrics_table.add_column("Status", style="white")
        
        for metric in sorted(metrics, key=lambda m: m.container_name):
            cpu_style = "red" if metric.cpu_percent > 80 else "green"
            memory_style = "red" if metric.memory_percent > 90 else "yellow"
            
            metrics_table.add_row(
                metric.container_name,
                f"[{cpu_style}]{metric.cpu_percent:.1f}%[/{cpu_style}]",
                f"[{memory_style}]{metric.memory_usage_mb:.1f}MB ({metric.memory_percent:.1f}%)[/{memory_style}]",
                f"{metric.network_rx_mb:.1f}/{metric.network_tx_mb:.1f} MB",
                f"{metric.block_read_mb:.1f}/{metric.block_write_mb:.1f} MB",
                metric.status
            )
        
        layout["metrics"].update(Panel(metrics_table, title="Current Metrics"))
        
        # Alerts panel
        if alerts:
            alerts_text = Text()
            for alert in alerts[-10:]:  # Show last 10 alerts
                severity_style = {
                    'critical': 'bold red',
                    'warning': 'yellow',
                    'info': 'blue'
                }.get(alert['severity'], 'white')
                
                alerts_text.append(
                    f"[{alert['timestamp'].strftime('%H:%M:%S')}] {alert['message']}\n",
                    style=severity_style
                )
        else:
            alerts_text = Text("No active alerts", style="green")
        
        layout["alerts"].update(Panel(alerts_text, title="Recent Alerts"))
        
        # Footer
        layout["footer"].update(Panel(
            "Press Ctrl+C to stop monitoring",
            style="dim"
        ))
    
    async def create_baseline(
        self, 
        duration: int = 3600, 
        baseline_type: str = 'normal',
        containers: Optional[List[str]] = None
    ):
        """Create performance baseline by monitoring for specified duration."""
        console.print(f"[yellow]Creating {baseline_type} baseline (duration: {duration}s)...[/yellow]")
        
        collected_metrics = defaultdict(list)
        start_time = time.time()
        interval = 10  # seconds
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task(f"Collecting baseline data...", total=duration)
            
            while time.time() - start_time < duration:
                try:
                    metrics = await self.collect_metrics(containers)
                    
                    for metric in metrics:
                        collected_metrics[metric.container_name].append(metric)
                    
                    # Update progress
                    elapsed = time.time() - start_time
                    progress.update(task, completed=elapsed)
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Baseline collection error: {e}")
                    await asyncio.sleep(interval)
        
        # Store baselines
        for container_name, metrics in collected_metrics.items():
            if metrics:
                metadata = {
                    'collection_date': datetime.now().isoformat(),
                    'sample_interval_seconds': interval,
                    'total_samples': len(metrics)
                }
                
                self.db.create_baseline(container_name, baseline_type, metrics, metadata)
                console.print(f"[green]✓ Created {baseline_type} baseline for {container_name} ({len(metrics)} samples)[/green]")
    
    def compare_with_baseline(
        self, 
        container_name: str, 
        baseline_type: str = 'normal',
        sample_duration: int = 300
    ) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        console.print(f"[yellow]Comparing {container_name} with {baseline_type} baseline...[/yellow]")
        
        # Get baseline
        baseline = self.db.get_baseline(container_name, baseline_type)
        if not baseline:
            return {'error': f'No {baseline_type} baseline found for {container_name}'}
        
        # Collect current metrics
        start_time = datetime.now()
        current_metrics = []
        
        async def collect_current():
            nonlocal current_metrics
            end_time = start_time + timedelta(seconds=sample_duration)
            
            while datetime.now() < end_time:
                metrics = await self.collect_metrics([container_name])
                current_metrics.extend([m for m in metrics if m.container_name == container_name])
                await asyncio.sleep(5)
        
        # Run collection
        asyncio.run(collect_current())
        
        if not current_metrics:
            return {'error': f'No current metrics collected for {container_name}'}
        
        # Calculate current averages
        current_avg_cpu = sum(m.cpu_percent for m in current_metrics) / len(current_metrics)
        current_avg_memory = sum(m.memory_usage_mb for m in current_metrics) / len(current_metrics)
        current_max_memory = max(m.memory_usage_mb for m in current_metrics)
        
        # Compare with baseline
        comparison = {
            'container_name': container_name,
            'baseline_type': baseline_type,
            'baseline_created': baseline['created_at'],
            'comparison_timestamp': datetime.now().isoformat(),
            'sample_count': len(current_metrics),
            'metrics': {
                'cpu_percent': {
                    'baseline': baseline['avg_cpu_percent'],
                    'current': current_avg_cpu,
                    'difference': current_avg_cpu - baseline['avg_cpu_percent'],
                    'change_percent': ((current_avg_cpu - baseline['avg_cpu_percent']) / baseline['avg_cpu_percent']) * 100
                },
                'memory_mb': {
                    'baseline_avg': baseline['avg_memory_mb'],
                    'baseline_max': baseline['max_memory_mb'],
                    'current_avg': current_avg_memory,
                    'current_max': current_max_memory,
                    'avg_difference': current_avg_memory - baseline['avg_memory_mb'],
                    'avg_change_percent': ((current_avg_memory - baseline['avg_memory_mb']) / baseline['avg_memory_mb']) * 100
                }
            },
            'analysis': self._analyze_performance_comparison(baseline, current_metrics)
        }
        
        return comparison
    
    def _analyze_performance_comparison(
        self, 
        baseline: Dict[str, Any], 
        current_metrics: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Analyze performance comparison and generate insights."""
        current_avg_cpu = sum(m.cpu_percent for m in current_metrics) / len(current_metrics)
        current_avg_memory = sum(m.memory_usage_mb for m in current_metrics) / len(current_metrics)
        
        analysis = {
            'status': 'normal',
            'issues': [],
            'recommendations': []
        }
        
        # CPU analysis
        cpu_change = ((current_avg_cpu - baseline['avg_cpu_percent']) / baseline['avg_cpu_percent']) * 100
        if cpu_change > 50:
            analysis['issues'].append(f"CPU usage increased by {cpu_change:.1f}%")
            analysis['recommendations'].append("Investigate recent code changes or increased workload")
            analysis['status'] = 'degraded'
        
        # Memory analysis
        memory_change = ((current_avg_memory - baseline['avg_memory_mb']) / baseline['avg_memory_mb']) * 100
        if memory_change > 30:
            analysis['issues'].append(f"Memory usage increased by {memory_change:.1f}%")
            analysis['recommendations'].append("Check for memory leaks or increased data processing")
            analysis['status'] = 'degraded'
        
        # Overall status
        if analysis['issues']:
            analysis['status'] = 'degraded' if len(analysis['issues']) < 3 else 'critical'
        
        return analysis
    
    def generate_optimization_recommendations(self, container_name: str) -> Dict[str, Any]:
        """Generate performance optimization recommendations."""
        console.print(f"[yellow]Generating optimization recommendations for {container_name}...[/yellow]")
        
        # Get recent metrics
        recent_metrics = self.db.get_historical_metrics(
            container_name, 
            since=datetime.now() - timedelta(hours=24),
            limit=1000
        )
        
        if not recent_metrics:
            return {'error': 'No historical metrics available'}
        
        recommendations = {
            'container_name': container_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'metrics_analyzed': len(recent_metrics),
            'recommendations': []
        }
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        
        # CPU recommendations
        if avg_cpu > 70:
            recommendations['recommendations'].append({
                'category': 'CPU',
                'priority': 'high',
                'issue': f'High average CPU usage: {avg_cpu:.1f}%',
                'recommendations': [
                    'Consider increasing CPU limits in docker-compose.yml',
                    'Optimize indicator calculations for better performance',
                    'Review log level settings (DEBUG logging is CPU intensive)'
                ]
            })
        
        if max_cpu > 95:
            recommendations['recommendations'].append({
                'category': 'CPU',
                'priority': 'critical',
                'issue': f'CPU spikes detected: {max_cpu:.1f}%',
                'recommendations': [
                    'Investigate CPU spikes during specific operations',
                    'Consider implementing rate limiting for API calls',
                    'Profile code to identify performance bottlenecks'
                ]
            })
        
        # Memory recommendations
        if avg_memory > 800:  # 800MB
            recommendations['recommendations'].append({
                'category': 'Memory',
                'priority': 'medium',
                'issue': f'High memory usage: {avg_memory:.1f}MB average',
                'recommendations': [
                    'Review data retention policies',
                    'Implement periodic cleanup of old data',
                    'Consider increasing memory limits if justified'
                ]
            })
        
        # Memory growth analysis
        if len(recent_metrics) > 100:
            # Check for memory growth trend
            early_metrics = recent_metrics[-100:-50]
            late_metrics = recent_metrics[-50:]
            
            early_avg = sum(m.memory_usage_mb for m in early_metrics) / len(early_metrics)
            late_avg = sum(m.memory_usage_mb for m in late_metrics) / len(late_metrics)
            
            growth_percent = ((late_avg - early_avg) / early_avg) * 100
            
            if growth_percent > 20:
                recommendations['recommendations'].append({
                    'category': 'Memory',
                    'priority': 'high',
                    'issue': f'Memory growth detected: {growth_percent:.1f}% increase',
                    'recommendations': [
                        'Investigate potential memory leaks',
                        'Review data structure usage and cleanup',
                        'Monitor for memory growth patterns'
                    ]
                })
        
        # Container stability
        restart_counts = [m.restart_count for m in recent_metrics]
        if restart_counts and max(restart_counts) > 0:
            recommendations['recommendations'].append({
                'category': 'Stability',
                'priority': 'high',
                'issue': f'Container restarts detected: {max(restart_counts)} restarts',
                'recommendations': [
                    'Check container logs for crash causes',
                    'Review health check configuration',
                    'Investigate resource limits and OOM conditions'
                ]
            })
        
        return recommendations


def parse_duration(duration_str: str) -> int:
    """Parse duration string (e.g., '1h', '30m', '300s') to seconds."""
    if duration_str.endswith('h'):
        return int(duration_str[:-1]) * 3600
    elif duration_str.endswith('m'):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith('s'):
        return int(duration_str[:-1])
    else:
        return int(duration_str)  # Assume seconds


async def main():
    """Main entry point for performance monitoring."""
    parser = argparse.ArgumentParser(description="Docker Performance Monitor for AI Trading Bot")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--realtime', action='store_true', help='Real-time monitoring')
    mode_group.add_argument('--baseline', action='store_true', help='Create performance baseline')
    mode_group.add_argument('--compare-baseline', action='store_true', help='Compare with baseline')
    mode_group.add_argument('--generate-recommendations', action='store_true', help='Generate optimization recommendations')
    
    # Common arguments
    parser.add_argument('--container', type=str, help='Specific container to monitor')
    parser.add_argument('--duration', type=str, default='1h', help='Duration (e.g., 1h, 30m, 300s)')
    parser.add_argument('--baseline-type', type=str, default='normal', help='Baseline type name')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).parent.parent, help='Project directory')
    parser.add_argument('--export', type=Path, help='Export results to file')
    
    args = parser.parse_args()
    
    try:
        monitor = PerformanceMonitor(args.project_dir)
        duration_seconds = parse_duration(args.duration)
        
        containers = [args.container] if args.container else None
        
        if args.realtime:
            await monitor.monitor_realtime(duration_seconds, args.interval)
        
        elif args.baseline:
            await monitor.create_baseline(duration_seconds, args.baseline_type, containers)
        
        elif args.compare_baseline:
            if not args.container:
                console.print("[red]Container name required for baseline comparison[/red]")
                return 1
            
            results = monitor.compare_with_baseline(args.container, args.baseline_type, 300)
            
            # Display results
            if 'error' in results:
                console.print(f"[red]Error: {results['error']}[/red]")
                return 1
            
            # Create comparison table
            table = Table(title=f"Baseline Comparison - {args.container}")
            table.add_column("Metric", style="cyan")
            table.add_column("Baseline", style="green")
            table.add_column("Current", style="yellow")
            table.add_column("Change", style="red")
            
            cpu_metrics = results['metrics']['cpu_percent']
            memory_metrics = results['metrics']['memory_mb']
            
            table.add_row(
                "CPU %",
                f"{cpu_metrics['baseline']:.1f}%",
                f"{cpu_metrics['current']:.1f}%",
                f"{cpu_metrics['change_percent']:+.1f}%"
            )
            
            table.add_row(
                "Memory (Avg)",
                f"{memory_metrics['baseline_avg']:.1f}MB",
                f"{memory_metrics['current_avg']:.1f}MB",
                f"{memory_metrics['avg_change_percent']:+.1f}%"
            )
            
            console.print(table)
            
            # Display analysis
            analysis = results['analysis']
            if analysis['status'] != 'normal':
                console.print(f"\n[bold red]Status: {analysis['status'].upper()}[/bold red]")
                
                if analysis['issues']:
                    console.print("\n[bold]Issues Found:[/bold]")
                    for issue in analysis['issues']:
                        console.print(f"  • {issue}")
                
                if analysis['recommendations']:
                    console.print("\n[bold]Recommendations:[/bold]")
                    for rec in analysis['recommendations']:
                        console.print(f"  • {rec}")
            else:
                console.print("\n[green]✓ Performance is within normal range[/green]")
        
        elif args.generate_recommendations:
            if not args.container:
                console.print("[red]Container name required for recommendations[/red]")
                return 1
            
            recommendations = monitor.generate_optimization_recommendations(args.container)
            
            if 'error' in recommendations:
                console.print(f"[red]Error: {recommendations['error']}[/red]")
                return 1
            
            console.print(f"\n[bold]Optimization Recommendations for {args.container}[/bold]")
            console.print(f"Based on {recommendations['metrics_analyzed']} metrics samples\n")
            
            if not recommendations['recommendations']:
                console.print("[green]✓ No optimization recommendations at this time[/green]")
            else:
                for i, rec in enumerate(recommendations['recommendations'], 1):
                    priority_style = {
                        'critical': 'bold red',
                        'high': 'red',
                        'medium': 'yellow',
                        'low': 'green'
                    }.get(rec['priority'], 'white')
                    
                    console.print(f"[bold]{i}. {rec['category']} [{priority_style}]{rec['priority'].upper()}[/{priority_style}][/bold]")
                    console.print(f"   Issue: {rec['issue']}")
                    console.print("   Recommendations:")
                    for recommendation in rec['recommendations']:
                        console.print(f"     • {recommendation}")
                    console.print()
        
        # Export results if requested
        if args.export and 'results' in locals():
            with open(args.export, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[green]✓ Results exported to {args.export}[/green]")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Monitoring failed: {e}[/red]")
        logger.exception("Performance monitoring failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))