"""
Performance Monitoring Integration for OPTIMIZE Platform

This module monitors the performance impact of security tools on the trading bot
and provides optimization recommendations to minimize trading latency and maximize
security effectiveness.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import psutil
import redis
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()


class PerformanceMetric(Base):
    """Database model for performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    component = Column(String, nullable=False)
    metric_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String)
    metadata = Column(Text)


class PerformanceAlert(Base):
    """Database model for performance alerts."""
    __tablename__ = 'performance_alerts'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    component = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    threshold = Column(Float)
    actual_value = Column(Float)
    severity = Column(String, nullable=False)
    description = Column(Text)
    resolved_at = Column(DateTime)


# Enums and Data Classes
class ComponentType(Enum):
    """Types of monitored components."""
    FALCO = "falco"
    DOCKER_BENCH = "docker_bench"
    TRIVY = "trivy"
    CORRELATION_ENGINE = "correlation_engine"
    ALERT_ORCHESTRATOR = "alert_orchestrator"
    SECURITY_DASHBOARD = "security_dashboard"
    TRADING_BOT = "trading_bot"
    SYSTEM = "system"


class MetricType(Enum):
    """Types of performance metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"


class AlertSeverity(Enum):
    """Performance alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    component: ComponentType
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    unit: str
    enabled: bool = True
    aggregation_window: int = 300  # seconds


@dataclass
class ComponentMetrics:
    """Performance metrics for a component."""
    component: ComponentType
    timestamp: datetime
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_read_mb_per_sec: float = 0.0
    disk_write_mb_per_sec: float = 0.0
    network_recv_mb_per_sec: float = 0.0
    network_sent_mb_per_sec: float = 0.0
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate_percent: float = 0.0
    availability_percent: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'component': self.component.value,
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'disk_read_mb_per_sec': self.disk_read_mb_per_sec,
            'disk_write_mb_per_sec': self.disk_write_mb_per_sec,
            'network_recv_mb_per_sec': self.network_recv_mb_per_sec,
            'network_sent_mb_per_sec': self.network_sent_mb_per_sec,
            'latency_ms': self.latency_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'error_rate_percent': self.error_rate_percent,
            'availability_percent': self.availability_percent
        }


@dataclass
class TradingImpactMetrics:
    """Trading bot impact metrics."""
    timestamp: datetime
    order_latency_ms: float = 0.0
    market_data_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    throughput_impact_percent: float = 0.0
    uptime_percent: float = 100.0
    missed_opportunities: int = 0
    error_count: int = 0
    
    def calculate_impact_score(self) -> float:
        """Calculate overall trading impact score (0-100, lower is better)."""
        # Normalize metrics to 0-100 scale
        latency_score = min(100, self.order_latency_ms / 10)  # 1000ms = 100 points
        throughput_score = self.throughput_impact_percent
        uptime_score = max(0, 100 - self.uptime_percent)
        error_score = min(100, self.error_count * 5)  # 20 errors = 100 points
        
        # Weighted average
        impact_score = (
            latency_score * 0.4 +
            throughput_score * 0.3 +
            uptime_score * 0.2 +
            error_score * 0.1
        )
        
        return min(100, impact_score)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    component: ComponentType
    issue_type: str
    severity: AlertSeverity
    description: str
    recommendation: str
    estimated_improvement: str
    implementation_effort: str
    priority_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)


class ComponentMonitor:
    """Monitors performance of individual security components."""
    
    def __init__(self, component: ComponentType):
        self.component = component
        self.process_info = {}
        self.last_metrics = None
        self.baseline_metrics = None
        self.collection_start_time = time.time()
    
    def collect_metrics(self) -> ComponentMetrics:
        """Collect current performance metrics for the component."""
        try:
            metrics = ComponentMetrics(
                component=self.component,
                timestamp=datetime.utcnow()
            )
            
            # Get process information if available
            process = self._get_component_process()
            if process:
                # CPU usage
                metrics.cpu_usage_percent = process.cpu_percent(interval=0.1)
                
                # Memory usage
                memory_info = process.memory_info()
                metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
                
                # I/O metrics
                try:
                    io_counters = process.io_counters()
                    current_time = time.time()
                    
                    if self.last_metrics:
                        time_diff = current_time - self.last_metrics['timestamp']
                        
                        # Disk I/O rate
                        disk_read_diff = io_counters.read_bytes - self.last_metrics['disk_read_bytes']
                        disk_write_diff = io_counters.write_bytes - self.last_metrics['disk_write_bytes']
                        
                        metrics.disk_read_mb_per_sec = (disk_read_diff / time_diff) / (1024 * 1024)
                        metrics.disk_write_mb_per_sec = (disk_write_diff / time_diff) / (1024 * 1024)
                    
                    # Store current values for next calculation
                    self.last_metrics = {
                        'timestamp': current_time,
                        'disk_read_bytes': io_counters.read_bytes,
                        'disk_write_bytes': io_counters.write_bytes
                    }
                    
                except (psutil.AccessDenied, AttributeError):
                    # Some systems don't provide I/O counters
                    pass
            
            # Component-specific metrics
            metrics = self._collect_component_specific_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {self.component.value}: {e}")
            return ComponentMetrics(component=self.component, timestamp=datetime.utcnow())
    
    def _get_component_process(self) -> Optional[psutil.Process]:
        """Get the process for this component."""
        try:
            # Map components to process names
            process_names = {
                ComponentType.FALCO: ["falco"],
                ComponentType.DOCKER_BENCH: ["docker-bench-security"],
                ComponentType.TRIVY: ["trivy"],
                ComponentType.TRADING_BOT: ["python", "ai-trading-bot"],
                ComponentType.CORRELATION_ENGINE: ["python"],
                ComponentType.ALERT_ORCHESTRATOR: ["python"],
                ComponentType.SECURITY_DASHBOARD: ["python"]
            }
            
            names = process_names.get(self.component, [])
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if any(name in proc.info['name'].lower() for name in names):
                        # Additional filtering for Python processes
                        if self.component in [ComponentType.TRADING_BOT, ComponentType.CORRELATION_ENGINE,
                                            ComponentType.ALERT_ORCHESTRATOR, ComponentType.SECURITY_DASHBOARD]:
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            if self.component.value.replace('_', '-') in cmdline.lower():
                                return proc
                        else:
                            return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding process for {self.component.value}: {e}")
            return None
    
    def _collect_component_specific_metrics(self, metrics: ComponentMetrics) -> ComponentMetrics:
        """Collect component-specific performance metrics."""
        try:
            if self.component == ComponentType.TRADING_BOT:
                # Collect trading-specific metrics
                metrics = self._collect_trading_bot_metrics(metrics)
            elif self.component == ComponentType.FALCO:
                # Collect Falco-specific metrics
                metrics = self._collect_falco_metrics(metrics)
            elif self.component == ComponentType.TRIVY:
                # Collect Trivy-specific metrics
                metrics = self._collect_trivy_metrics(metrics)
            # Add more component-specific collectors as needed
            
        except Exception as e:
            logger.error(f"Error collecting component-specific metrics for {self.component.value}: {e}")
        
        return metrics
    
    def _collect_trading_bot_metrics(self, metrics: ComponentMetrics) -> ComponentMetrics:
        """Collect trading bot specific metrics."""
        # TODO: Integrate with trading bot metrics API
        # For now, use placeholder values
        metrics.latency_ms = 50.0  # Order placement latency
        metrics.throughput_ops_per_sec = 10.0  # Orders per second
        metrics.error_rate_percent = 0.1  # Error rate
        return metrics
    
    def _collect_falco_metrics(self, metrics: ComponentMetrics) -> ComponentMetrics:
        """Collect Falco specific metrics."""
        # TODO: Integrate with Falco metrics endpoint
        metrics.throughput_ops_per_sec = 1000.0  # Events per second
        return metrics
    
    def _collect_trivy_metrics(self, metrics: ComponentMetrics) -> ComponentMetrics:
        """Collect Trivy specific metrics."""
        # TODO: Integrate with Trivy metrics
        return metrics


class TradingImpactAnalyzer:
    """Analyzes the impact of security tools on trading performance."""
    
    def __init__(self):
        self.baseline_metrics = None
        self.impact_history = []
    
    async def analyze_trading_impact(self, security_metrics: Dict[ComponentType, ComponentMetrics]) -> TradingImpactMetrics:
        """Analyze the impact of security tools on trading performance."""
        try:
            impact_metrics = TradingImpactMetrics(timestamp=datetime.utcnow())
            
            # Get trading bot metrics
            trading_metrics = security_metrics.get(ComponentType.TRADING_BOT)
            if trading_metrics:
                impact_metrics.order_latency_ms = trading_metrics.latency_ms
                impact_metrics.error_count = int(trading_metrics.error_rate_percent * 100)
                impact_metrics.uptime_percent = trading_metrics.availability_percent
            
            # Calculate impact from security tools
            security_cpu_total = 0.0
            security_memory_total = 0.0
            
            for component, metrics in security_metrics.items():
                if component != ComponentType.TRADING_BOT and component != ComponentType.SYSTEM:
                    security_cpu_total += metrics.cpu_usage_percent
                    security_memory_total += metrics.memory_usage_mb
            
            # Estimate throughput impact based on resource usage
            impact_metrics.throughput_impact_percent = min(10.0, security_cpu_total * 0.1)
            
            # Calculate market data latency impact
            correlation_metrics = security_metrics.get(ComponentType.CORRELATION_ENGINE)
            if correlation_metrics:
                impact_metrics.market_data_latency_ms = correlation_metrics.latency_ms * 0.1
            
            # Store in history for trend analysis
            self.impact_history.append(impact_metrics)
            
            # Keep only last 1000 entries
            if len(self.impact_history) > 1000:
                self.impact_history = self.impact_history[-1000:]
            
            return impact_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing trading impact: {e}")
            return TradingImpactMetrics(timestamp=datetime.utcnow())
    
    def get_impact_trend(self, hours: int = 24) -> List[TradingImpactMetrics]:
        """Get trading impact trend over specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metrics for metrics in self.impact_history
            if metrics.timestamp >= cutoff_time
        ]


class PerformanceOptimizer:
    """Provides performance optimization recommendations."""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    def analyze_and_recommend(self, 
                            metrics: Dict[ComponentType, ComponentMetrics],
                            trading_impact: TradingImpactMetrics) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations."""
        recommendations = []
        
        try:
            # Analyze each component
            for component, component_metrics in metrics.items():
                component_recommendations = self._analyze_component(component, component_metrics)
                recommendations.extend(component_recommendations)
            
            # Analyze overall system performance
            system_recommendations = self._analyze_system_performance(metrics, trading_impact)
            recommendations.extend(system_recommendations)
            
            # Sort by priority score (highest first)
            recommendations.sort(key=lambda r: r.priority_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
        
        return recommendations
    
    def _analyze_component(self, component: ComponentType, metrics: ComponentMetrics) -> List[OptimizationRecommendation]:
        """Analyze individual component performance."""
        recommendations = []
        
        try:
            # High CPU usage
            if metrics.cpu_usage_percent > 80:
                recommendations.append(OptimizationRecommendation(
                    component=component,
                    issue_type="high_cpu_usage",
                    severity=AlertSeverity.HIGH,
                    description=f"{component.value} is using {metrics.cpu_usage_percent:.1f}% CPU",
                    recommendation="Consider reducing scan frequency or implementing CPU throttling",
                    estimated_improvement="10-30% latency reduction",
                    implementation_effort="Medium",
                    priority_score=85.0
                ))
            
            # High memory usage
            if metrics.memory_usage_mb > 1000:
                recommendations.append(OptimizationRecommendation(
                    component=component,
                    issue_type="high_memory_usage",
                    severity=AlertSeverity.MEDIUM,
                    description=f"{component.value} is using {metrics.memory_usage_mb:.1f}MB memory",
                    recommendation="Review memory allocation and implement garbage collection optimization",
                    estimated_improvement="5-15% memory reduction",
                    implementation_effort="Low",
                    priority_score=60.0
                ))
            
            # High disk I/O
            if metrics.disk_read_mb_per_sec + metrics.disk_write_mb_per_sec > 50:
                recommendations.append(OptimizationRecommendation(
                    component=component,
                    issue_type="high_disk_io",
                    severity=AlertSeverity.MEDIUM,
                    description=f"{component.value} has high disk I/O activity",
                    recommendation="Consider using SSD storage or implementing disk I/O throttling",
                    estimated_improvement="20-40% I/O performance improvement",
                    implementation_effort="High",
                    priority_score=70.0
                ))
            
            # Component-specific recommendations
            if component == ComponentType.FALCO:
                recommendations.extend(self._analyze_falco_performance(metrics))
            elif component == ComponentType.TRIVY:
                recommendations.extend(self._analyze_trivy_performance(metrics))
            elif component == ComponentType.DOCKER_BENCH:
                recommendations.extend(self._analyze_docker_bench_performance(metrics))
            
        except Exception as e:
            logger.error(f"Error analyzing component {component.value}: {e}")
        
        return recommendations
    
    def _analyze_system_performance(self, 
                                  metrics: Dict[ComponentType, ComponentMetrics],
                                  trading_impact: TradingImpactMetrics) -> List[OptimizationRecommendation]:
        """Analyze overall system performance."""
        recommendations = []
        
        try:
            impact_score = trading_impact.calculate_impact_score()
            
            # High trading impact
            if impact_score > 20:
                recommendations.append(OptimizationRecommendation(
                    component=ComponentType.SYSTEM,
                    issue_type="high_trading_impact",
                    severity=AlertSeverity.CRITICAL,
                    description=f"Security tools are causing {impact_score:.1f}% trading impact",
                    recommendation="Implement security tool scheduling during low-activity periods",
                    estimated_improvement="50-80% impact reduction",
                    implementation_effort="Medium",
                    priority_score=95.0
                ))
            
            # Resource contention
            total_cpu = sum(m.cpu_usage_percent for m in metrics.values())
            if total_cpu > 90:
                recommendations.append(OptimizationRecommendation(
                    component=ComponentType.SYSTEM,
                    issue_type="resource_contention",
                    severity=AlertSeverity.HIGH,
                    description=f"Total CPU usage is {total_cpu:.1f}%",
                    recommendation="Implement CPU quotas and process prioritization",
                    estimated_improvement="30-50% latency improvement",
                    implementation_effort="Medium",
                    priority_score=88.0
                ))
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")
        
        return recommendations
    
    def _analyze_falco_performance(self, metrics: ComponentMetrics) -> List[OptimizationRecommendation]:
        """Analyze Falco-specific performance issues."""
        recommendations = []
        
        if metrics.throughput_ops_per_sec > 5000:
            recommendations.append(OptimizationRecommendation(
                component=ComponentType.FALCO,
                issue_type="high_event_rate",
                severity=AlertSeverity.MEDIUM,
                description="Falco is processing a high number of events",
                recommendation="Review and optimize Falco rules to reduce noise",
                estimated_improvement="20-40% CPU reduction",
                implementation_effort="Low",
                priority_score=65.0
            ))
        
        return recommendations
    
    def _analyze_trivy_performance(self, metrics: ComponentMetrics) -> List[OptimizationRecommendation]:
        """Analyze Trivy-specific performance issues."""
        recommendations = []
        
        if metrics.cpu_usage_percent > 50:
            recommendations.append(OptimizationRecommendation(
                component=ComponentType.TRIVY,
                issue_type="scan_frequency",
                severity=AlertSeverity.MEDIUM,
                description="Trivy scans are consuming significant CPU",
                recommendation="Reduce scan frequency or implement incremental scanning",
                estimated_improvement="30-60% CPU reduction",
                implementation_effort="Low",
                priority_score=75.0
            ))
        
        return recommendations
    
    def _analyze_docker_bench_performance(self, metrics: ComponentMetrics) -> List[OptimizationRecommendation]:
        """Analyze Docker Bench-specific performance issues."""
        recommendations = []
        
        # Docker Bench runs periodically, so analyze based on resource usage patterns
        if metrics.memory_usage_mb > 200:
            recommendations.append(OptimizationRecommendation(
                component=ComponentType.DOCKER_BENCH,
                issue_type="memory_optimization",
                severity=AlertSeverity.LOW,
                description="Docker Bench security scans using excessive memory",
                recommendation="Run Docker Bench with memory limits and during off-peak hours",
                estimated_improvement="10-20% memory reduction",
                implementation_effort="Low",
                priority_score=40.0
            ))
        
        return recommendations
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules configuration."""
        # TODO: Load from configuration files
        return {}


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 db_url: str = "sqlite:///performance_metrics.db"):
        self.redis_client = redis.from_url(redis_url)
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Component monitors
        self.component_monitors = {
            component: ComponentMonitor(component)
            for component in ComponentType
        }
        
        self.trading_analyzer = TradingImpactAnalyzer()
        self.optimizer = PerformanceOptimizer()
        
        # Performance thresholds
        self.thresholds = self._load_thresholds()
        
        # Internal state
        self.running = False
        self.metrics_history = {}
        self.active_alerts = {}
        
        # Collection interval
        self.collection_interval = 30  # seconds
    
    async def start(self):
        """Start the performance monitor."""
        self.running = True
        logger.info("Starting Performance Monitor")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._collect_metrics()),
            asyncio.create_task(self._analyze_performance()),
            asyncio.create_task(self._monitor_thresholds()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the performance monitor."""
        self.running = False
        logger.info("Performance Monitor stopped")
    
    async def _collect_metrics(self):
        """Collect performance metrics from all components."""
        while self.running:
            try:
                current_metrics = {}
                
                # Collect metrics from each component
                for component_type, monitor in self.component_monitors.items():
                    metrics = monitor.collect_metrics()
                    current_metrics[component_type] = metrics
                    
                    # Store in database
                    await self._store_metrics(metrics)
                
                # Store in memory for real-time access
                self.metrics_history[datetime.utcnow()] = current_metrics
                
                # Keep only recent history (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = {
                    timestamp: metrics
                    for timestamp, metrics in self.metrics_history.items()
                    if timestamp >= cutoff_time
                }
                
                # Publish metrics for real-time updates
                await self._publish_metrics(current_metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _store_metrics(self, metrics: ComponentMetrics):
        """Store metrics in database."""
        session = self.Session()
        try:
            # Store individual metric values
            metric_entries = [
                PerformanceMetric(
                    id=str(uuid4()),
                    timestamp=metrics.timestamp,
                    component=metrics.component.value,
                    metric_type=MetricType.CPU_USAGE.value,
                    value=metrics.cpu_usage_percent,
                    unit="percent"
                ),
                PerformanceMetric(
                    id=str(uuid4()),
                    timestamp=metrics.timestamp,
                    component=metrics.component.value,
                    metric_type=MetricType.MEMORY_USAGE.value,
                    value=metrics.memory_usage_mb,
                    unit="MB"
                ),
                PerformanceMetric(
                    id=str(uuid4()),
                    timestamp=metrics.timestamp,
                    component=metrics.component.value,
                    metric_type=MetricType.LATENCY.value,
                    value=metrics.latency_ms,
                    unit="ms"
                )
                # Add more metric entries as needed
            ]
            
            for entry in metric_entries:
                session.add(entry)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing metrics for {metrics.component.value}: {e}")
        finally:
            session.close()
    
    async def _publish_metrics(self, metrics: Dict[ComponentType, ComponentMetrics]):
        """Publish metrics for real-time updates."""
        try:
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {
                    component.value: component_metrics.to_dict()
                    for component, component_metrics in metrics.items()
                }
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.publish,
                "performance_metrics",
                json.dumps(metrics_data)
            )
            
        except Exception as e:
            logger.error(f"Error publishing metrics: {e}")
    
    async def _analyze_performance(self):
        """Analyze performance and generate recommendations."""
        while self.running:
            try:
                # Get latest metrics
                if self.metrics_history:
                    latest_timestamp = max(self.metrics_history.keys())
                    latest_metrics = self.metrics_history[latest_timestamp]
                    
                    # Analyze trading impact
                    trading_impact = await self.trading_analyzer.analyze_trading_impact(latest_metrics)
                    
                    # Generate optimization recommendations
                    recommendations = self.optimizer.analyze_and_recommend(latest_metrics, trading_impact)
                    
                    # Publish analysis results
                    analysis_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'trading_impact': {
                            'impact_score': trading_impact.calculate_impact_score(),
                            'order_latency_ms': trading_impact.order_latency_ms,
                            'throughput_impact_percent': trading_impact.throughput_impact_percent,
                            'uptime_percent': trading_impact.uptime_percent
                        },
                        'recommendations': [
                            {
                                'id': rec.id,
                                'component': rec.component.value,
                                'severity': rec.severity.value,
                                'description': rec.description,
                                'recommendation': rec.recommendation,
                                'priority_score': rec.priority_score
                            }
                            for rec in recommendations[:5]  # Top 5 recommendations
                        ]
                    }
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis_client.publish,
                        "performance_analysis",
                        json.dumps(analysis_data)
                    )
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error analyzing performance: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_thresholds(self):
        """Monitor performance thresholds and generate alerts."""
        while self.running:
            try:
                if self.metrics_history:
                    latest_timestamp = max(self.metrics_history.keys())
                    latest_metrics = self.metrics_history[latest_timestamp]
                    
                    for component_type, metrics in latest_metrics.items():
                        await self._check_component_thresholds(metrics)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring thresholds: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_thresholds(self, metrics: ComponentMetrics):
        """Check performance thresholds for a component."""
        try:
            component_thresholds = [
                threshold for threshold in self.thresholds
                if threshold.component == metrics.component and threshold.enabled
            ]
            
            for threshold in component_thresholds:
                value = self._get_metric_value(metrics, threshold.metric_type)
                
                if value is not None:
                    # Check critical threshold
                    if value >= threshold.critical_threshold:
                        await self._create_performance_alert(
                            metrics.component,
                            threshold.metric_type,
                            AlertSeverity.CRITICAL,
                            threshold.critical_threshold,
                            value
                        )
                    # Check warning threshold
                    elif value >= threshold.warning_threshold:
                        await self._create_performance_alert(
                            metrics.component,
                            threshold.metric_type,
                            AlertSeverity.MEDIUM,
                            threshold.warning_threshold,
                            value
                        )
            
        except Exception as e:
            logger.error(f"Error checking thresholds for {metrics.component.value}: {e}")
    
    def _get_metric_value(self, metrics: ComponentMetrics, metric_type: MetricType) -> Optional[float]:
        """Get metric value by type."""
        metric_map = {
            MetricType.CPU_USAGE: metrics.cpu_usage_percent,
            MetricType.MEMORY_USAGE: metrics.memory_usage_mb,
            MetricType.LATENCY: metrics.latency_ms,
            MetricType.THROUGHPUT: metrics.throughput_ops_per_sec,
            MetricType.ERROR_RATE: metrics.error_rate_percent,
            MetricType.AVAILABILITY: metrics.availability_percent
        }
        return metric_map.get(metric_type)
    
    async def _create_performance_alert(self, 
                                      component: ComponentType,
                                      metric_type: MetricType,
                                      severity: AlertSeverity,
                                      threshold: float,
                                      actual_value: float):
        """Create a performance alert."""
        try:
            alert_id = str(uuid4())
            alert_key = f"{component.value}_{metric_type.value}"
            
            # Check if similar alert already exists
            if alert_key in self.active_alerts:
                return
            
            # Store alert in database
            session = self.Session()
            try:
                alert = PerformanceAlert(
                    id=alert_id,
                    timestamp=datetime.utcnow(),
                    component=component.value,
                    alert_type=metric_type.value,
                    threshold=threshold,
                    actual_value=actual_value,
                    severity=severity.value,
                    description=f"{component.value} {metric_type.value} is {actual_value:.2f}, exceeding threshold of {threshold:.2f}"
                )
                session.add(alert)
                session.commit()
                
                # Add to active alerts
                self.active_alerts[alert_key] = alert_id
                
                # Publish alert
                alert_data = {
                    'alert_id': alert_id,
                    'type': 'performance_alert',
                    'component': component.value,
                    'metric_type': metric_type.value,
                    'severity': severity.value,
                    'threshold': threshold,
                    'actual_value': actual_value,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.publish,
                    "performance_alerts",
                    json.dumps(alert_data)
                )
                
                logger.warning(f"Performance alert: {component.value} {metric_type.value} = {actual_value:.2f} (threshold: {threshold:.2f})")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error storing performance alert: {e}")
            finally:
                session.close()
            
        except Exception as e:
            logger.error(f"Error creating performance alert: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old performance data."""
        while self.running:
            try:
                # Clean up metrics older than 7 days
                session = self.Session()
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                # Clean up metrics
                deleted_metrics = session.query(PerformanceMetric).filter(
                    PerformanceMetric.timestamp < cutoff_date
                ).delete()
                
                # Clean up resolved alerts older than 30 days
                alert_cutoff_date = datetime.utcnow() - timedelta(days=30)
                deleted_alerts = session.query(PerformanceAlert).filter(
                    PerformanceAlert.resolved_at < alert_cutoff_date
                ).delete()
                
                session.commit()
                session.close()
                
                if deleted_metrics > 0 or deleted_alerts > 0:
                    logger.info(f"Cleaned up {deleted_metrics} metrics and {deleted_alerts} alerts")
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)
    
    def _load_thresholds(self) -> List[PerformanceThreshold]:
        """Load performance thresholds configuration."""
        # Default thresholds
        thresholds = [
            # CPU thresholds
            PerformanceThreshold(ComponentType.TRADING_BOT, MetricType.CPU_USAGE, 60.0, 80.0, "percent"),
            PerformanceThreshold(ComponentType.FALCO, MetricType.CPU_USAGE, 40.0, 70.0, "percent"),
            PerformanceThreshold(ComponentType.TRIVY, MetricType.CPU_USAGE, 50.0, 80.0, "percent"),
            PerformanceThreshold(ComponentType.CORRELATION_ENGINE, MetricType.CPU_USAGE, 30.0, 60.0, "percent"),
            
            # Memory thresholds
            PerformanceThreshold(ComponentType.TRADING_BOT, MetricType.MEMORY_USAGE, 500.0, 1000.0, "MB"),
            PerformanceThreshold(ComponentType.FALCO, MetricType.MEMORY_USAGE, 200.0, 500.0, "MB"),
            PerformanceThreshold(ComponentType.TRIVY, MetricType.MEMORY_USAGE, 300.0, 800.0, "MB"),
            
            # Latency thresholds
            PerformanceThreshold(ComponentType.TRADING_BOT, MetricType.LATENCY, 100.0, 500.0, "ms"),
            PerformanceThreshold(ComponentType.CORRELATION_ENGINE, MetricType.LATENCY, 200.0, 1000.0, "ms"),
        ]
        
        return thresholds
    
    async def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status."""
        try:
            status = {
                'running': self.running,
                'monitored_components': list(ComponentType),
                'active_alerts': len(self.active_alerts),
                'metrics_collected': len(self.metrics_history),
                'collection_interval': self.collection_interval,
                'last_update': datetime.utcnow().isoformat()
            }
            
            # Add latest metrics summary
            if self.metrics_history:
                latest_timestamp = max(self.metrics_history.keys())
                latest_metrics = self.metrics_history[latest_timestamp]
                
                status['latest_metrics'] = {
                    component.value: {
                        'cpu_usage_percent': metrics.cpu_usage_percent,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'latency_ms': metrics.latency_ms
                    }
                    for component, metrics in latest_metrics.items()
                }
                
                # Calculate trading impact
                trading_impact = await self.trading_analyzer.analyze_trading_impact(latest_metrics)
                status['trading_impact_score'] = trading_impact.calculate_impact_score()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting performance status: {e}")
            return {'error': str(e)}


# Factory function
def create_performance_monitor(redis_url: str = "redis://localhost:6379",
                             db_url: str = "sqlite:///performance_metrics.db") -> PerformanceMonitor:
    """Create a performance monitor instance."""
    return PerformanceMonitor(redis_url, db_url)


if __name__ == "__main__":
    async def main():
        monitor = create_performance_monitor()
        await monitor.start()
    
    asyncio.run(main())