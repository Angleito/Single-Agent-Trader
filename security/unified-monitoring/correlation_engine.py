"""
Security Event Correlation Engine for OPTIMIZE Platform

This module provides advanced security event correlation, pattern detection,
and threat intelligence integration for the unified security monitoring system.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import redis
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()


class SecurityEvent(Base):
    """Database model for security events."""
    __tablename__ = 'security_events'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    raw_data = Column(Text, nullable=False)
    processed_data = Column(Text)
    correlation_id = Column(String)
    status = Column(String, default='new')


class CorrelationRule(Base):
    """Database model for correlation rules."""
    __tablename__ = 'correlation_rules'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    rule_type = Column(String, nullable=False)
    conditions = Column(Text, nullable=False)
    actions = Column(Text, nullable=False)
    enabled = Column(String, default='true')
    created_at = Column(DateTime, default=datetime.utcnow)


# Enums and Data Classes
class EventSeverity(Enum):
    """Security event severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventSource(Enum):
    """Security event sources."""
    FALCO = "falco"
    DOCKER_BENCH = "docker_bench"
    TRIVY = "trivy"
    TRADING_BOT = "trading_bot"
    NETWORK_MONITOR = "network_monitor"
    SYSTEM_MONITOR = "system_monitor"
    CUSTOM = "custom"


class CorrelationType(Enum):
    """Types of event correlation."""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    BEHAVIORAL = "behavioral"
    PATTERN = "pattern"


@dataclass
class NormalizedEvent:
    """Normalized security event structure."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: EventSource = EventSource.CUSTOM
    event_type: str = ""
    severity: EventSeverity = EventSeverity.INFO
    title: str = ""
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    entity: str = ""  # Container, host, service, etc.
    network_info: Dict[str, Any] = field(default_factory=dict)
    process_info: Dict[str, Any] = field(default_factory=dict)
    file_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'event_type': self.event_type,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'raw_data': self.raw_data,
            'metadata': self.metadata,
            'tags': list(self.tags),
            'entity': self.entity,
            'network_info': self.network_info,
            'process_info': self.process_info,
            'file_info': self.file_info
        }


@dataclass
class CorrelationResult:
    """Result of event correlation analysis."""
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_type: CorrelationType = CorrelationType.TEMPORAL
    events: List[NormalizedEvent] = field(default_factory=list)
    confidence_score: float = 0.0
    risk_score: float = 0.0
    pattern_name: str = ""
    description: str = ""
    recommendations: List[str] = field(default_factory=list)
    timeline: List[Tuple[datetime, str]] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class EventNormalizer:
    """Normalizes events from different security sources."""
    
    def __init__(self):
        self.normalization_rules = {
            EventSource.FALCO: self._normalize_falco_event,
            EventSource.DOCKER_BENCH: self._normalize_docker_bench_event,
            EventSource.TRIVY: self._normalize_trivy_event,
            EventSource.TRADING_BOT: self._normalize_trading_bot_event,
        }
    
    def normalize(self, raw_event: Dict[str, Any], source: EventSource) -> NormalizedEvent:
        """Normalize a raw event based on its source."""
        try:
            normalizer = self.normalization_rules.get(source, self._normalize_generic_event)
            return normalizer(raw_event)
        except Exception as e:
            logger.error(f"Error normalizing event from {source}: {e}")
            return self._create_error_event(raw_event, source, str(e))
    
    def _normalize_falco_event(self, raw_event: Dict[str, Any]) -> NormalizedEvent:
        """Normalize Falco runtime security events."""
        event = NormalizedEvent()
        event.source = EventSource.FALCO
        event.timestamp = datetime.fromisoformat(raw_event.get('time', datetime.utcnow().isoformat()))
        event.event_type = raw_event.get('rule', 'unknown_rule')
        event.title = raw_event.get('rule', 'Falco Security Event')
        event.description = raw_event.get('output', '')
        event.raw_data = raw_event
        
        # Map Falco priority to severity
        priority_map = {
            'Emergency': EventSeverity.CRITICAL,
            'Alert': EventSeverity.CRITICAL,
            'Critical': EventSeverity.CRITICAL,
            'Error': EventSeverity.HIGH,
            'Warning': EventSeverity.MEDIUM,
            'Notice': EventSeverity.LOW,
            'Informational': EventSeverity.INFO,
            'Debug': EventSeverity.INFO
        }
        event.severity = priority_map.get(raw_event.get('priority', 'Informational'), EventSeverity.INFO)
        
        # Extract entity information
        output_fields = raw_event.get('output_fields', {})
        event.entity = output_fields.get('container.name', output_fields.get('proc.name', 'unknown'))
        
        # Extract process information
        event.process_info = {
            'pid': output_fields.get('proc.pid'),
            'name': output_fields.get('proc.name'),
            'cmdline': output_fields.get('proc.cmdline'),
            'user': output_fields.get('user.name'),
            'container': output_fields.get('container.name'),
        }
        
        # Extract file information
        event.file_info = {
            'path': output_fields.get('fd.name'),
            'directory': output_fields.get('fd.directory'),
            'type': output_fields.get('fd.type'),
        }
        
        # Extract network information
        event.network_info = {
            'src_ip': output_fields.get('fd.sip'),
            'dst_ip': output_fields.get('fd.cip'),
            'src_port': output_fields.get('fd.sport'),
            'dst_port': output_fields.get('fd.cport'),
            'protocol': output_fields.get('fd.l4proto'),
        }
        
        # Add tags based on rule type
        rule_tags = self._extract_falco_tags(raw_event.get('rule', ''))
        event.tags.update(rule_tags)
        
        return event
    
    def _normalize_docker_bench_event(self, raw_event: Dict[str, Any]) -> NormalizedEvent:
        """Normalize Docker Bench Security events."""
        event = NormalizedEvent()
        event.source = EventSource.DOCKER_BENCH
        event.timestamp = datetime.fromisoformat(raw_event.get('timestamp', datetime.utcnow().isoformat()))
        event.event_type = raw_event.get('check_id', 'unknown_check')
        event.title = f"Docker Security: {raw_event.get('description', 'Security Check')}"
        event.description = raw_event.get('details', '')
        event.raw_data = raw_event
        
        # Map Docker Bench results to severity
        result_map = {
            'FAIL': EventSeverity.HIGH,
            'WARN': EventSeverity.MEDIUM,
            'PASS': EventSeverity.INFO,
            'INFO': EventSeverity.INFO
        }
        event.severity = result_map.get(raw_event.get('result', 'INFO'), EventSeverity.INFO)
        
        # Extract container information
        event.entity = raw_event.get('container_name', raw_event.get('image_name', 'docker_host'))
        
        # Add tags based on check category
        category = raw_event.get('category', '')
        if category:
            event.tags.add(f"category:{category}")
        
        event.tags.add("docker_security")
        event.tags.add("compliance")
        
        return event
    
    def _normalize_trivy_event(self, raw_event: Dict[str, Any]) -> NormalizedEvent:
        """Normalize Trivy vulnerability scanner events."""
        event = NormalizedEvent()
        event.source = EventSource.TRIVY
        event.timestamp = datetime.fromisoformat(raw_event.get('timestamp', datetime.utcnow().isoformat()))
        event.event_type = raw_event.get('type', 'vulnerability')
        
        # Handle different Trivy result types
        if 'vulnerability' in raw_event:
            vuln = raw_event['vulnerability']
            event.title = f"Vulnerability: {vuln.get('VulnerabilityID', 'Unknown')}"
            event.description = vuln.get('Description', vuln.get('Title', ''))
            
            # Map CVSS severity to our severity levels
            severity_map = {
                'CRITICAL': EventSeverity.CRITICAL,
                'HIGH': EventSeverity.HIGH,
                'MEDIUM': EventSeverity.MEDIUM,
                'LOW': EventSeverity.LOW,
                'UNKNOWN': EventSeverity.INFO
            }
            event.severity = severity_map.get(vuln.get('Severity', 'UNKNOWN'), EventSeverity.INFO)
            
            event.tags.add("vulnerability")
            event.tags.add(f"cvss:{vuln.get('CVSS', {}).get('nvd', {}).get('V3Score', 0)}")
            
        elif 'secret' in raw_event:
            secret = raw_event['secret']
            event.title = f"Secret Detected: {secret.get('RuleID', 'Unknown')}"
            event.description = secret.get('Match', '')
            event.severity = EventSeverity.HIGH
            event.tags.add("secret_exposure")
            
        elif 'config' in raw_event:
            config = raw_event['config']
            event.title = f"Config Issue: {config.get('Type', 'Unknown')}"
            event.description = config.get('Description', '')
            event.severity = EventSeverity.MEDIUM
            event.tags.add("misconfiguration")
        
        event.raw_data = raw_event
        event.entity = raw_event.get('target', raw_event.get('image', 'unknown'))
        
        return event
    
    def _normalize_trading_bot_event(self, raw_event: Dict[str, Any]) -> NormalizedEvent:
        """Normalize trading bot security events."""
        event = NormalizedEvent()
        event.source = EventSource.TRADING_BOT
        event.timestamp = datetime.fromisoformat(raw_event.get('timestamp', datetime.utcnow().isoformat()))
        event.event_type = raw_event.get('event_type', 'trading_security')
        event.title = raw_event.get('title', 'Trading Bot Security Event')
        event.description = raw_event.get('description', '')
        event.raw_data = raw_event
        
        # Map trading bot event types to severity
        severity_map = {
            'api_key_exposure': EventSeverity.CRITICAL,
            'unusual_trading_pattern': EventSeverity.HIGH,
            'authentication_failure': EventSeverity.HIGH,
            'rate_limit_exceeded': EventSeverity.MEDIUM,
            'configuration_change': EventSeverity.MEDIUM,
            'health_check_failure': EventSeverity.LOW
        }
        event.severity = severity_map.get(raw_event.get('event_type', ''), EventSeverity.INFO)
        
        event.entity = raw_event.get('service', raw_event.get('component', 'trading_bot'))
        event.tags.add("trading_security")
        
        # Add specific trading context
        if 'exchange' in raw_event:
            event.tags.add(f"exchange:{raw_event['exchange']}")
        if 'symbol' in raw_event:
            event.tags.add(f"symbol:{raw_event['symbol']}")
        
        return event
    
    def _normalize_generic_event(self, raw_event: Dict[str, Any]) -> NormalizedEvent:
        """Normalize generic security events."""
        event = NormalizedEvent()
        event.source = EventSource.CUSTOM
        event.timestamp = datetime.fromisoformat(raw_event.get('timestamp', datetime.utcnow().isoformat()))
        event.event_type = raw_event.get('event_type', 'generic_security')
        event.title = raw_event.get('title', 'Security Event')
        event.description = raw_event.get('description', '')
        event.raw_data = raw_event
        
        # Try to extract severity
        severity_str = raw_event.get('severity', 'info').lower()
        severity_map = {
            'critical': EventSeverity.CRITICAL,
            'high': EventSeverity.HIGH,
            'medium': EventSeverity.MEDIUM,
            'low': EventSeverity.LOW,
            'info': EventSeverity.INFO
        }
        event.severity = severity_map.get(severity_str, EventSeverity.INFO)
        
        event.entity = raw_event.get('entity', raw_event.get('source', 'unknown'))
        
        return event
    
    def _create_error_event(self, raw_event: Dict[str, Any], source: EventSource, error: str) -> NormalizedEvent:
        """Create an event for normalization errors."""
        event = NormalizedEvent()
        event.source = source
        event.event_type = "normalization_error"
        event.severity = EventSeverity.MEDIUM
        event.title = f"Event Normalization Error from {source.value}"
        event.description = f"Failed to normalize event: {error}"
        event.raw_data = raw_event
        event.metadata = {"error": error}
        event.tags.add("normalization_error")
        
        return event
    
    def _extract_falco_tags(self, rule_name: str) -> Set[str]:
        """Extract relevant tags from Falco rule names."""
        tags = set()
        
        rule_tags = {
            'Write below etc': ['filesystem', 'configuration'],
            'Read sensitive file': ['filesystem', 'data_access'],
            'Run shell in container': ['container', 'shell_access'],
            'Network tool launched': ['network', 'reconnaissance'],
            'Privileged container': ['container', 'privilege_escalation'],
            'Container run as root': ['container', 'privilege'],
            'Create files below dev': ['filesystem', 'device_access'],
            'Modify binary dirs': ['filesystem', 'binary_manipulation'],
            'Contact cloud metadata': ['cloud', 'metadata_access'],
        }
        
        for rule_pattern, rule_tags_list in rule_tags.items():
            if rule_pattern.lower() in rule_name.lower():
                tags.update(rule_tags_list)
        
        return tags


class PatternDetector:
    """Detects security patterns and attack chains in correlated events."""
    
    def __init__(self):
        self.attack_patterns = {
            'container_escape': self._detect_container_escape,
            'privilege_escalation': self._detect_privilege_escalation,
            'data_exfiltration': self._detect_data_exfiltration,
            'reconnaissance': self._detect_reconnaissance,
            'persistence': self._detect_persistence,
            'lateral_movement': self._detect_lateral_movement,
            'crypto_mining': self._detect_crypto_mining,
        }
    
    def detect_patterns(self, events: List[NormalizedEvent]) -> List[CorrelationResult]:
        """Detect attack patterns in a set of events."""
        patterns = []
        
        for pattern_name, detector in self.attack_patterns.items():
            try:
                result = detector(events)
                if result and result.confidence_score > 0.5:
                    patterns.append(result)
            except Exception as e:
                logger.error(f"Error in pattern detection {pattern_name}: {e}")
        
        return patterns
    
    def _detect_container_escape(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect container escape attempts."""
        escape_indicators = []
        
        for event in events:
            # Look for privileged operations in containers
            if (event.source == EventSource.FALCO and 
                'container' in event.tags and
                any(indicator in event.description.lower() for indicator in [
                    'privileged', 'mount', '/proc', '/sys', 'breakout', 'escape'
                ])):
                escape_indicators.append(event)
        
        if len(escape_indicators) >= 2:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.BEHAVIORAL
            result.events = escape_indicators
            result.pattern_name = "container_escape"
            result.confidence_score = min(0.9, 0.3 + (len(escape_indicators) * 0.2))
            result.risk_score = 0.9
            result.description = "Potential container escape attempt detected"
            result.recommendations = [
                "Review container security policies",
                "Check for unnecessary privileged containers",
                "Audit container runtime configuration",
                "Implement container security monitoring"
            ]
            return result
        
        return None
    
    def _detect_privilege_escalation(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect privilege escalation attempts."""
        escalation_events = []
        
        for event in events:
            if (event.source == EventSource.FALCO and
                any(indicator in event.description.lower() for indicator in [
                    'sudo', 'su -', 'setuid', 'setgid', 'root', 'privilege'
                ])):
                escalation_events.append(event)
        
        if len(escalation_events) >= 1:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.BEHAVIORAL
            result.events = escalation_events
            result.pattern_name = "privilege_escalation"
            result.confidence_score = min(0.8, 0.4 + (len(escalation_events) * 0.2))
            result.risk_score = 0.8
            result.description = "Privilege escalation activity detected"
            result.recommendations = [
                "Review user permissions and access controls",
                "Audit sudo configuration",
                "Implement principle of least privilege",
                "Monitor privileged account usage"
            ]
            return result
        
        return None
    
    def _detect_data_exfiltration(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect data exfiltration patterns."""
        exfil_events = []
        
        for event in events:
            # Look for suspicious file access and network activity
            if (event.source == EventSource.FALCO and
                (('sensitive' in event.description.lower() and 'read' in event.description.lower()) or
                 ('network' in event.tags and event.network_info.get('dst_port') in [443, 80, 22]))):
                exfil_events.append(event)
        
        if len(exfil_events) >= 2:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.TEMPORAL
            result.events = exfil_events
            result.pattern_name = "data_exfiltration"
            result.confidence_score = min(0.7, 0.3 + (len(exfil_events) * 0.15))
            result.risk_score = 0.9
            result.description = "Potential data exfiltration activity detected"
            result.recommendations = [
                "Review network traffic for unusual patterns",
                "Audit file access logs",
                "Implement data loss prevention (DLP)",
                "Monitor outbound network connections"
            ]
            return result
        
        return None
    
    def _detect_reconnaissance(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect reconnaissance activities."""
        recon_events = []
        
        for event in events:
            if (event.source == EventSource.FALCO and
                any(tool in event.description.lower() for tool in [
                    'nmap', 'netstat', 'ps aux', 'whoami', 'id', 'env'
                ])):
                recon_events.append(event)
        
        if len(recon_events) >= 2:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.BEHAVIORAL
            result.events = recon_events
            result.pattern_name = "reconnaissance"
            result.confidence_score = min(0.6, 0.2 + (len(recon_events) * 0.15))
            result.risk_score = 0.6
            result.description = "Reconnaissance activity detected"
            result.recommendations = [
                "Monitor system information gathering activities",
                "Review process execution logs",
                "Implement network segmentation",
                "Enable detailed audit logging"
            ]
            return result
        
        return None
    
    def _detect_persistence(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect persistence establishment."""
        persistence_events = []
        
        for event in events:
            if (event.source == EventSource.FALCO and
                any(indicator in event.description.lower() for indicator in [
                    'crontab', 'systemd', 'bashrc', '.profile', 'startup'
                ])):
                persistence_events.append(event)
        
        if len(persistence_events) >= 1:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.BEHAVIORAL
            result.events = persistence_events
            result.pattern_name = "persistence"
            result.confidence_score = min(0.7, 0.4 + (len(persistence_events) * 0.2))
            result.risk_score = 0.7
            result.description = "Persistence mechanism detected"
            result.recommendations = [
                "Review system startup configurations",
                "Audit scheduled tasks and cron jobs",
                "Monitor file system changes",
                "Implement file integrity monitoring"
            ]
            return result
        
        return None
    
    def _detect_lateral_movement(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect lateral movement attempts."""
        movement_events = []
        
        for event in events:
            if (event.source == EventSource.FALCO and
                'network' in event.tags and
                event.network_info.get('dst_port') in [22, 3389, 5985, 5986]):
                movement_events.append(event)
        
        if len(movement_events) >= 2:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.SPATIAL
            result.events = movement_events
            result.pattern_name = "lateral_movement"
            result.confidence_score = min(0.8, 0.3 + (len(movement_events) * 0.2))
            result.risk_score = 0.8
            result.description = "Lateral movement activity detected"
            result.recommendations = [
                "Review network access patterns",
                "Implement network segmentation",
                "Monitor remote access protocols",
                "Enable network traffic analysis"
            ]
            return result
        
        return None
    
    def _detect_crypto_mining(self, events: List[NormalizedEvent]) -> Optional[CorrelationResult]:
        """Detect cryptocurrency mining activities."""
        mining_events = []
        
        for event in events:
            if (event.source == EventSource.FALCO and
                any(indicator in event.description.lower() for indicator in [
                    'xmrig', 'cpuminer', 'mining', 'pool', 'stratum'
                ])):
                mining_events.append(event)
        
        if len(mining_events) >= 1:
            result = CorrelationResult()
            result.correlation_type = CorrelationType.BEHAVIORAL
            result.events = mining_events
            result.pattern_name = "crypto_mining"
            result.confidence_score = min(0.9, 0.6 + (len(mining_events) * 0.2))
            result.risk_score = 0.7
            result.description = "Cryptocurrency mining activity detected"
            result.recommendations = [
                "Terminate unauthorized mining processes",
                "Review resource usage patterns",
                "Implement application whitelisting",
                "Monitor network connections to mining pools"
            ]
            return result
        
        return None


class CorrelationEngine:
    """Main security event correlation engine."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db_url: str = "sqlite:///security_events.db"):
        self.redis_client = redis.from_url(redis_url)
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.normalizer = EventNormalizer()
        self.pattern_detector = PatternDetector()
        
        # Configuration
        self.correlation_window = timedelta(minutes=30)
        self.batch_size = 100
        self.processing_interval = 10  # seconds
        
        # Internal state
        self.running = False
        self.event_buffer = []
        self.correlation_cache = {}
    
    async def start(self):
        """Start the correlation engine."""
        self.running = True
        logger.info("Starting Security Event Correlation Engine")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._process_events()),
            asyncio.create_task(self._correlate_events()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the correlation engine."""
        self.running = False
        logger.info("Stopping Security Event Correlation Engine")
    
    async def ingest_event(self, raw_event: Dict[str, Any], source: EventSource):
        """Ingest a raw security event for processing."""
        try:
            # Normalize the event
            normalized_event = self.normalizer.normalize(raw_event, source)
            
            # Store in database
            await self._store_event(normalized_event)
            
            # Add to buffer for correlation
            self.event_buffer.append(normalized_event)
            
            # Publish to Redis for real-time processing
            await self._publish_event(normalized_event)
            
            logger.debug(f"Ingested event {normalized_event.id} from {source.value}")
            
        except Exception as e:
            logger.error(f"Error ingesting event from {source}: {e}")
    
    async def _store_event(self, event: NormalizedEvent):
        """Store normalized event in database."""
        session = self.Session()
        try:
            db_event = SecurityEvent(
                id=event.id,
                timestamp=event.timestamp,
                source=event.source.value,
                event_type=event.event_type,
                severity=event.severity.value,
                raw_data=json.dumps(event.raw_data),
                processed_data=json.dumps(event.to_dict())
            )
            session.add(db_event)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing event {event.id}: {e}")
        finally:
            session.close()
    
    async def _publish_event(self, event: NormalizedEvent):
        """Publish event to Redis for real-time processing."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.publish,
                "security_events",
                json.dumps(event.to_dict())
            )
        except Exception as e:
            logger.error(f"Error publishing event {event.id}: {e}")
    
    async def _process_events(self):
        """Process events in batches."""
        while self.running:
            try:
                if len(self.event_buffer) >= self.batch_size:
                    batch = self.event_buffer[:self.batch_size]
                    self.event_buffer = self.event_buffer[self.batch_size:]
                    
                    await self._process_batch(batch)
                
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                await asyncio.sleep(5)
    
    async def _process_batch(self, events: List[NormalizedEvent]):
        """Process a batch of events for correlation."""
        try:
            # Group events by time windows
            time_windows = self._group_by_time_window(events)
            
            for window_events in time_windows:
                # Detect patterns within the time window
                patterns = self.pattern_detector.detect_patterns(window_events)
                
                for pattern in patterns:
                    await self._handle_correlation_result(pattern)
            
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
    
    def _group_by_time_window(self, events: List[NormalizedEvent]) -> List[List[NormalizedEvent]]:
        """Group events by correlation time windows."""
        windows = []
        current_window = []
        window_start = None
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for event in sorted_events:
            if (window_start is None or 
                event.timestamp - window_start <= self.correlation_window):
                current_window.append(event)
                if window_start is None:
                    window_start = event.timestamp
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [event]
                window_start = event.timestamp
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    async def _correlate_events(self):
        """Correlate events across different time windows."""
        while self.running:
            try:
                # Get recent events from database
                session = self.Session()
                cutoff_time = datetime.utcnow() - self.correlation_window
                
                recent_events = session.query(SecurityEvent).filter(
                    SecurityEvent.timestamp >= cutoff_time
                ).all()
                
                session.close()
                
                # Convert to normalized events
                normalized_events = []
                for db_event in recent_events:
                    try:
                        processed_data = json.loads(db_event.processed_data)
                        event = self._dict_to_normalized_event(processed_data)
                        normalized_events.append(event)
                    except Exception as e:
                        logger.error(f"Error converting DB event {db_event.id}: {e}")
                
                # Detect cross-window patterns
                patterns = self.pattern_detector.detect_patterns(normalized_events)
                
                for pattern in patterns:
                    # Check if we've already processed this pattern
                    pattern_key = f"{pattern.pattern_name}:{pattern.correlation_id}"
                    if pattern_key not in self.correlation_cache:
                        await self._handle_correlation_result(pattern)
                        self.correlation_cache[pattern_key] = time.time()
                
                # Clean cache
                current_time = time.time()
                cache_ttl = 3600  # 1 hour
                self.correlation_cache = {
                    k: v for k, v in self.correlation_cache.items()
                    if current_time - v < cache_ttl
                }
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in event correlation: {e}")
                await asyncio.sleep(30)
    
    def _dict_to_normalized_event(self, data: Dict[str, Any]) -> NormalizedEvent:
        """Convert dictionary back to NormalizedEvent."""
        event = NormalizedEvent()
        event.id = data.get('id', str(uuid4()))
        event.timestamp = datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
        event.source = EventSource(data.get('source', 'custom'))
        event.event_type = data.get('event_type', '')
        event.severity = EventSeverity(data.get('severity', 'info'))
        event.title = data.get('title', '')
        event.description = data.get('description', '')
        event.raw_data = data.get('raw_data', {})
        event.metadata = data.get('metadata', {})
        event.tags = set(data.get('tags', []))
        event.entity = data.get('entity', '')
        event.network_info = data.get('network_info', {})
        event.process_info = data.get('process_info', {})
        event.file_info = data.get('file_info', {})
        
        return event
    
    async def _handle_correlation_result(self, result: CorrelationResult):
        """Handle a correlation result by triggering alerts and responses."""
        try:
            # Log the correlation result
            logger.info(f"Pattern detected: {result.pattern_name} "
                       f"(confidence: {result.confidence_score:.2f}, "
                       f"risk: {result.risk_score:.2f})")
            
            # Publish correlation result for alert processing
            correlation_data = {
                'correlation_id': result.correlation_id,
                'pattern_name': result.pattern_name,
                'confidence_score': result.confidence_score,
                'risk_score': result.risk_score,
                'description': result.description,
                'event_count': len(result.events),
                'recommendations': result.recommendations,
                'timestamp': result.created_at.isoformat()
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.publish,
                "security_correlations",
                json.dumps(correlation_data)
            )
            
        except Exception as e:
            logger.error(f"Error handling correlation result {result.correlation_id}: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data from database and cache."""
        while self.running:
            try:
                # Clean up events older than 30 days
                session = self.Session()
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                deleted_count = session.query(SecurityEvent).filter(
                    SecurityEvent.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                session.close()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old security events")
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def get_correlation_status(self) -> Dict[str, Any]:
        """Get the current status of the correlation engine."""
        session = self.Session()
        try:
            # Get event counts by source
            event_counts = {}
            for source in EventSource:
                count = session.query(SecurityEvent).filter(
                    SecurityEvent.source == source.value,
                    SecurityEvent.timestamp >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                event_counts[source.value] = count
            
            total_events = sum(event_counts.values())
            
            status = {
                'running': self.running,
                'buffer_size': len(self.event_buffer),
                'cache_size': len(self.correlation_cache),
                'events_24h': total_events,
                'events_by_source': event_counts,
                'correlation_window_minutes': int(self.correlation_window.total_seconds() / 60),
                'last_update': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting correlation status: {e}")
            return {'error': str(e)}
        finally:
            session.close()


# Factory function
def create_correlation_engine(redis_url: str = "redis://localhost:6379",
                            db_url: str = "sqlite:///security_events.db") -> CorrelationEngine:
    """Create a correlation engine instance."""
    return CorrelationEngine(redis_url, db_url)


if __name__ == "__main__":
    async def main():
        engine = create_correlation_engine()
        await engine.start()
    
    asyncio.run(main())