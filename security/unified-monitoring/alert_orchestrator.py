"""
Alert Orchestration and Routing System for OPTIMIZE Platform

This module provides intelligent alert routing, de-duplication, escalation,
and automated response capabilities for the unified security monitoring system.
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import aiohttp
import redis
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .correlation_engine import CorrelationResult, EventSeverity

logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()


class Alert(Base):
    """Database model for security alerts."""
    __tablename__ = 'security_alerts'
    
    id = Column(String, primary_key=True)
    correlation_id = Column(String)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_by = Column(String)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    escalated_at = Column(DateTime)
    escalation_level = Column(Integer, default=0)
    metadata = Column(Text)


class AlertRule(Base):
    """Database model for alert routing rules."""
    __tablename__ = 'alert_rules'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    conditions = Column(Text, nullable=False)
    actions = Column(Text, nullable=False)
    enabled = Column(String, default='true')
    priority = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)


# Enums and Data Classes
class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


class AlertType(Enum):
    """Types of security alerts."""
    CORRELATION = "correlation"
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    SYSTEM = "system"


class NotificationChannel(Enum):
    """Available notification channels."""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    SMS = "sms"
    WEBHOOK = "webhook"


@dataclass
class AlertNotification:
    """Alert notification configuration."""
    channel: NotificationChannel
    target: str  # Webhook URL, email address, etc.
    severity_filter: Set[EventSeverity] = field(default_factory=set)
    enabled: bool = True
    retry_count: int = 3
    retry_delay: int = 60  # seconds
    template: Optional[str] = None


@dataclass
class EscalationRule:
    """Alert escalation rule configuration."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    delay_minutes: int = 30
    escalation_level: int = 1
    notifications: List[AlertNotification] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AlertContext:
    """Complete alert context with enriched information."""
    alert_id: str
    correlation_id: Optional[str]
    alert_type: AlertType
    severity: EventSeverity
    title: str
    description: str
    status: AlertStatus
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    affected_entities: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    business_impact: str = ""
    technical_details: Dict[str, Any] = field(default_factory=dict)
    similar_alerts: List[str] = field(default_factory=list)


class AlertDeduplicator:
    """Handles alert de-duplication to reduce noise."""
    
    def __init__(self):
        self.alert_fingerprints = {}
        self.dedup_window = timedelta(minutes=15)
        self.similarity_threshold = 0.8
    
    def generate_fingerprint(self, alert: AlertContext) -> str:
        """Generate a fingerprint for alert de-duplication."""
        # Create a fingerprint based on alert characteristics
        fingerprint_data = {
            'type': alert.alert_type.value,
            'severity': alert.severity.value,
            'entities': sorted(alert.affected_entities),
            'title_keywords': self._extract_keywords(alert.title),
            'description_keywords': self._extract_keywords(alert.description)
        }
        
        # Create a hash of the fingerprint data
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return str(hash(fingerprint_str))
    
    def is_duplicate(self, alert: AlertContext) -> tuple[bool, Optional[str]]:
        """Check if an alert is a duplicate of a recent alert."""
        fingerprint = self.generate_fingerprint(alert)
        current_time = datetime.utcnow()
        
        # Clean up old fingerprints
        self._cleanup_old_fingerprints(current_time)
        
        # Check for exact matches
        if fingerprint in self.alert_fingerprints:
            existing_alert = self.alert_fingerprints[fingerprint]
            if current_time - existing_alert['timestamp'] <= self.dedup_window:
                return True, existing_alert['alert_id']
        
        # Check for similar alerts using fuzzy matching
        for existing_fingerprint, existing_data in self.alert_fingerprints.items():
            if current_time - existing_data['timestamp'] <= self.dedup_window:
                similarity = self._calculate_similarity(alert, existing_data['alert'])
                if similarity >= self.similarity_threshold:
                    return True, existing_data['alert_id']
        
        # Store this alert's fingerprint
        self.alert_fingerprints[fingerprint] = {
            'alert_id': alert.alert_id,
            'alert': alert,
            'timestamp': current_time
        }
        
        return False, None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for fingerprinting."""
        # Simple keyword extraction - in production, use more sophisticated NLP
        words = text.lower().split()
        # Filter out common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return sorted(keywords)
    
    def _calculate_similarity(self, alert1: AlertContext, alert2: AlertContext) -> float:
        """Calculate similarity between two alerts."""
        # Simple similarity calculation based on shared entities and keywords
        entities1 = set(alert1.affected_entities)
        entities2 = set(alert2.affected_entities)
        
        if entities1 and entities2:
            entity_similarity = len(entities1.intersection(entities2)) / len(entities1.union(entities2))
        else:
            entity_similarity = 0
        
        keywords1 = set(self._extract_keywords(alert1.title + " " + alert1.description))
        keywords2 = set(self._extract_keywords(alert2.title + " " + alert2.description))
        
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
        else:
            keyword_similarity = 0
        
        # Weight entities more heavily than keywords
        return (entity_similarity * 0.7) + (keyword_similarity * 0.3)
    
    def _cleanup_old_fingerprints(self, current_time: datetime):
        """Remove old fingerprints outside the deduplication window."""
        cutoff_time = current_time - self.dedup_window
        expired_fingerprints = [
            fp for fp, data in self.alert_fingerprints.items()
            if data['timestamp'] < cutoff_time
        ]
        for fp in expired_fingerprints:
            del self.alert_fingerprints[fp]


class NotificationManager:
    """Manages sending notifications to various channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = aiohttp.ClientSession()
        
        # Email configuration
        self.smtp_server = config.get('email', {}).get('smtp_server')
        self.smtp_port = config.get('email', {}).get('smtp_port', 587)
        self.smtp_username = config.get('email', {}).get('username')
        self.smtp_password = config.get('email', {}).get('password')
        
        # Notification templates
        self.templates = self._load_templates()
    
    async def send_notification(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send a notification for an alert."""
        try:
            # Check severity filter
            if notification.severity_filter and alert.severity not in notification.severity_filter:
                return True  # Filtered out, considered successful
            
            if not notification.enabled:
                return True  # Disabled, considered successful
            
            # Send based on channel
            if notification.channel == NotificationChannel.SLACK:
                return await self._send_slack_notification(notification, alert)
            elif notification.channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(notification, alert)
            elif notification.channel == NotificationChannel.PAGERDUTY:
                return await self._send_pagerduty_notification(notification, alert)
            elif notification.channel == NotificationChannel.TEAMS:
                return await self._send_teams_notification(notification, alert)
            elif notification.channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(notification, alert)
            else:
                logger.warning(f"Unsupported notification channel: {notification.channel}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending notification via {notification.channel}: {e}")
            return False
    
    async def _send_slack_notification(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send Slack notification."""
        try:
            # Create Slack message
            color = self._get_severity_color(alert.severity)
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"🚨 Security Alert: {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Type", "value": alert.alert_type.value, "short": True},
                            {"title": "Status", "value": alert.status.value, "short": True},
                            {"title": "Alert ID", "value": alert.alert_id, "short": True}
                        ],
                        "footer": "OPTIMIZE Security Platform",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Add affected entities if available
            if alert.affected_entities:
                payload["attachments"][0]["fields"].append({
                    "title": "Affected Entities",
                    "value": ", ".join(alert.affected_entities),
                    "short": False
                })
            
            # Add remediation steps if available
            if alert.remediation_steps:
                payload["attachments"][0]["fields"].append({
                    "title": "Recommended Actions",
                    "value": "\n".join([f"• {step}" for step in alert.remediation_steps]),
                    "short": False
                })
            
            async with self.session.post(notification.target, json=payload) as response:
                return response.status == 200
        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_email_notification(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send email notification."""
        try:
            if not all([self.smtp_server, self.smtp_username, self.smtp_password]):
                logger.error("Email configuration incomplete")
                return False
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[OPTIMIZE] Security Alert: {alert.title}"
            msg['From'] = self.smtp_username
            msg['To'] = notification.target
            
            # Create HTML content
            html_content = self._create_email_html(alert)
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email (run in executor to avoid blocking)
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_smtp_email, msg
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _send_smtp_email(self, msg: MIMEMultipart):
        """Send email via SMTP (blocking operation)."""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
    
    async def _send_pagerduty_notification(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send PagerDuty notification."""
        try:
            # PagerDuty Events API v2 format
            payload = {
                "routing_key": notification.target,
                "event_action": "trigger",
                "dedup_key": alert.correlation_id or alert.alert_id,
                "payload": {
                    "summary": f"Security Alert: {alert.title}",
                    "source": "OPTIMIZE Security Platform",
                    "severity": self._map_severity_to_pagerduty(alert.severity),
                    "component": ", ".join(alert.affected_entities) if alert.affected_entities else "Security System",
                    "group": alert.alert_type.value,
                    "custom_details": {
                        "alert_id": alert.alert_id,
                        "correlation_id": alert.correlation_id,
                        "description": alert.description,
                        "business_impact": alert.business_impact,
                        "remediation_steps": alert.remediation_steps
                    }
                }
            }
            
            pagerduty_url = "https://events.pagerduty.com/v2/enqueue"
            async with self.session.post(pagerduty_url, json=payload) as response:
                return response.status == 202
        
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
            return False
    
    async def _send_teams_notification(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send Microsoft Teams notification."""
        try:
            color = self._get_severity_color(alert.severity)
            
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color.replace("#", ""),
                "summary": f"Security Alert: {alert.title}",
                "sections": [
                    {
                        "activityTitle": f"🚨 Security Alert: {alert.title}",
                        "activitySubtitle": f"Severity: {alert.severity.value.upper()}",
                        "text": alert.description,
                        "facts": [
                            {"name": "Alert ID", "value": alert.alert_id},
                            {"name": "Type", "value": alert.alert_type.value},
                            {"name": "Status", "value": alert.status.value},
                            {"name": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}
                        ]
                    }
                ]
            }
            
            # Add affected entities
            if alert.affected_entities:
                payload["sections"][0]["facts"].append({
                    "name": "Affected Entities",
                    "value": ", ".join(alert.affected_entities)
                })
            
            async with self.session.post(notification.target, json=payload) as response:
                return response.status == 200
        
        except Exception as e:
            logger.error(f"Error sending Teams notification: {e}")
            return False
    
    async def _send_webhook_notification(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send generic webhook notification."""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "correlation_id": alert.correlation_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "status": alert.status.value,
                "created_at": alert.created_at.isoformat(),
                "affected_entities": alert.affected_entities,
                "remediation_steps": alert.remediation_steps,
                "business_impact": alert.business_impact,
                "technical_details": alert.technical_details,
                "metadata": alert.metadata
            }
            
            async with self.session.post(notification.target, json=payload) as response:
                return response.status in [200, 201, 202]
        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    def _get_severity_color(self, severity: EventSeverity) -> str:
        """Get color code for severity level."""
        color_map = {
            EventSeverity.CRITICAL: "#e74c3c",
            EventSeverity.HIGH: "#f39c12",
            EventSeverity.MEDIUM: "#3498db",
            EventSeverity.LOW: "#2ecc71",
            EventSeverity.INFO: "#95a5a6"
        }
        return color_map.get(severity, "#95a5a6")
    
    def _map_severity_to_pagerduty(self, severity: EventSeverity) -> str:
        """Map our severity levels to PagerDuty severity levels."""
        mapping = {
            EventSeverity.CRITICAL: "critical",
            EventSeverity.HIGH: "error",
            EventSeverity.MEDIUM: "warning",
            EventSeverity.LOW: "info",
            EventSeverity.INFO: "info"
        }
        return mapping.get(severity, "info")
    
    def _create_email_html(self, alert: AlertContext) -> str:
        """Create HTML email content for alert."""
        color = self._get_severity_color(alert.severity)
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; margin: -20px -20px 20px -20px; }}
                .severity {{ background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }}
                .details {{ margin: 20px 0; }}
                .detail-row {{ margin: 10px 0; }}
                .label {{ font-weight: bold; color: #333; }}
                .value {{ color: #666; }}
                .remediation {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 Security Alert</h1>
                    <h2>{alert.title}</h2>
                </div>
                
                <div class="details">
                    <div class="detail-row">
                        <span class="label">Severity:</span>
                        <span class="severity">{alert.severity.value.upper()}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Alert ID:</span>
                        <span class="value">{alert.alert_id}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Type:</span>
                        <span class="value">{alert.alert_type.value}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Time:</span>
                        <span class="value">{alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}</span>
                    </div>
        """
        
        if alert.affected_entities:
            html += f"""
                    <div class="detail-row">
                        <span class="label">Affected Entities:</span>
                        <span class="value">{", ".join(alert.affected_entities)}</span>
                    </div>
            """
        
        html += f"""
                    <div class="detail-row">
                        <span class="label">Description:</span>
                        <div class="value">{alert.description}</div>
                    </div>
                </div>
        """
        
        if alert.remediation_steps:
            html += f"""
                <div class="remediation">
                    <h3>Recommended Actions:</h3>
                    <ul>
                        {"".join([f"<li>{step}</li>" for step in alert.remediation_steps])}
                    </ul>
                </div>
            """
        
        html += """
                <div class="footer">
                    <p>This alert was generated by the OPTIMIZE Security Platform</p>
                    <p>AI Trading Bot Protection System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _load_templates(self) -> Dict[str, str]:
        """Load notification templates."""
        # TODO: Load from configuration files
        return {}
    
    async def close(self):
        """Close the notification manager."""
        await self.session.close()


class AlertOrchestrator:
    """Main alert orchestration and routing system."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 db_url: str = "sqlite:///security_alerts.db",
                 config: Optional[Dict[str, Any]] = None):
        self.redis_client = redis.from_url(redis_url)
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.config = config or {}
        self.deduplicator = AlertDeduplicator()
        self.notification_manager = NotificationManager(self.config)
        
        # Alert routing rules
        self.routing_rules: List[AlertRule] = []
        self.escalation_rules: List[EscalationRule] = []
        self.notification_channels: List[AlertNotification] = []
        
        # Internal state
        self.running = False
        self.active_alerts = {}
        self.alert_history = {}
        
        # Load configuration
        self._load_configuration()
    
    async def start(self):
        """Start the alert orchestrator."""
        self.running = True
        logger.info("Starting Alert Orchestrator")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._process_correlations()),
            asyncio.create_task(self._handle_escalations()),
            asyncio.create_task(self._cleanup_old_alerts()),
            asyncio.create_task(self._health_monitoring())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the alert orchestrator."""
        self.running = False
        await self.notification_manager.close()
        logger.info("Alert Orchestrator stopped")
    
    async def process_correlation(self, correlation: CorrelationResult):
        """Process a security correlation and generate alerts."""
        try:
            # Create alert context from correlation
            alert = self._create_alert_from_correlation(correlation)
            
            # Check for duplicates
            is_duplicate, duplicate_id = self.deduplicator.is_duplicate(alert)
            if is_duplicate:
                logger.info(f"Alert {alert.alert_id} is duplicate of {duplicate_id}, suppressing")
                await self._update_duplicate_count(duplicate_id)
                return
            
            # Store alert in database
            await self._store_alert(alert)
            
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            
            # Route and send notifications
            await self._route_alert(alert)
            
            # Publish alert for real-time updates
            await self._publish_alert(alert)
            
            logger.info(f"Processed alert {alert.alert_id} for correlation {correlation.correlation_id}")
            
        except Exception as e:
            logger.error(f"Error processing correlation {correlation.correlation_id}: {e}")
    
    def _create_alert_from_correlation(self, correlation: CorrelationResult) -> AlertContext:
        """Create alert context from correlation result."""
        alert = AlertContext(
            alert_id=str(uuid4()),
            correlation_id=correlation.correlation_id,
            alert_type=AlertType.CORRELATION,
            severity=self._map_correlation_severity(correlation),
            title=f"Security Pattern Detected: {correlation.pattern_name}",
            description=correlation.description,
            status=AlertStatus.ACTIVE,
            created_at=correlation.created_at,
            metadata={
                'confidence_score': correlation.confidence_score,
                'risk_score': correlation.risk_score,
                'event_count': len(correlation.events),
                'correlation_type': correlation.correlation_type.value
            },
            affected_entities=list(set([event.entity for event in correlation.events if event.entity])),
            remediation_steps=correlation.recommendations,
            business_impact=self._assess_business_impact(correlation),
            technical_details={
                'pattern_name': correlation.pattern_name,
                'indicators': correlation.indicators,
                'timeline': [(ts.isoformat(), desc) for ts, desc in correlation.timeline]
            }
        )
        
        return alert
    
    def _map_correlation_severity(self, correlation: CorrelationResult) -> EventSeverity:
        """Map correlation risk score to alert severity."""
        risk_score = correlation.risk_score
        
        if risk_score >= 0.9:
            return EventSeverity.CRITICAL
        elif risk_score >= 0.7:
            return EventSeverity.HIGH
        elif risk_score >= 0.5:
            return EventSeverity.MEDIUM
        elif risk_score >= 0.3:
            return EventSeverity.LOW
        else:
            return EventSeverity.INFO
    
    def _assess_business_impact(self, correlation: CorrelationResult) -> str:
        """Assess business impact of a correlation."""
        # TODO: Implement sophisticated business impact assessment
        risk_score = correlation.risk_score
        
        if risk_score >= 0.9:
            return "High - Potential service disruption or data breach"
        elif risk_score >= 0.7:
            return "Medium - May affect trading operations"
        elif risk_score >= 0.5:
            return "Low - Monitoring recommended"
        else:
            return "Minimal - Informational"
    
    async def _store_alert(self, alert: AlertContext):
        """Store alert in database."""
        session = self.Session()
        try:
            db_alert = Alert(
                id=alert.alert_id,
                correlation_id=alert.correlation_id,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                title=alert.title,
                description=alert.description,
                status=alert.status.value,
                created_at=alert.created_at,
                metadata=json.dumps({
                    **alert.metadata,
                    'affected_entities': alert.affected_entities,
                    'remediation_steps': alert.remediation_steps,
                    'business_impact': alert.business_impact,
                    'technical_details': alert.technical_details
                })
            )
            session.add(db_alert)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing alert {alert.alert_id}: {e}")
        finally:
            session.close()
    
    async def _route_alert(self, alert: AlertContext):
        """Route alert to appropriate notification channels."""
        try:
            # Apply routing rules
            applicable_channels = self._get_applicable_channels(alert)
            
            # Send notifications
            notification_tasks = []
            for channel in applicable_channels:
                task = asyncio.create_task(
                    self._send_notification_with_retry(channel, alert)
                )
                notification_tasks.append(task)
            
            # Wait for all notifications to complete
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Log results
            successful = sum(1 for result in results if result is True)
            total = len(results)
            logger.info(f"Sent {successful}/{total} notifications for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error routing alert {alert.alert_id}: {e}")
    
    def _get_applicable_channels(self, alert: AlertContext) -> List[AlertNotification]:
        """Get notification channels applicable to an alert."""
        applicable = []
        
        for channel in self.notification_channels:
            # Check if channel is enabled
            if not channel.enabled:
                continue
            
            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            # TODO: Add more sophisticated routing logic based on alert content,
            # time of day, escalation level, etc.
            
            applicable.append(channel)
        
        return applicable
    
    async def _send_notification_with_retry(self, notification: AlertNotification, alert: AlertContext) -> bool:
        """Send notification with retry logic."""
        for attempt in range(notification.retry_count):
            try:
                success = await self.notification_manager.send_notification(notification, alert)
                if success:
                    return True
                
                if attempt < notification.retry_count - 1:
                    await asyncio.sleep(notification.retry_delay)
            
            except Exception as e:
                logger.error(f"Notification attempt {attempt + 1} failed: {e}")
                if attempt < notification.retry_count - 1:
                    await asyncio.sleep(notification.retry_delay)
        
        logger.error(f"Failed to send notification via {notification.channel} after {notification.retry_count} attempts")
        return False
    
    async def _publish_alert(self, alert: AlertContext):
        """Publish alert for real-time updates."""
        try:
            alert_data = {
                'alert_id': alert.alert_id,
                'type': 'new_alert',
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'affected_entities': alert.affected_entities,
                'timestamp': alert.created_at.isoformat()
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.publish,
                "security_alerts",
                json.dumps(alert_data)
            )
        except Exception as e:
            logger.error(f"Error publishing alert {alert.alert_id}: {e}")
    
    async def _update_duplicate_count(self, original_alert_id: str):
        """Update duplicate count for an existing alert."""
        if original_alert_id in self.active_alerts:
            original_alert = self.active_alerts[original_alert_id]
            duplicate_count = original_alert.metadata.get('duplicate_count', 0) + 1
            original_alert.metadata['duplicate_count'] = duplicate_count
            
            # Update in database
            session = self.Session()
            try:
                db_alert = session.query(Alert).filter(Alert.id == original_alert_id).first()
                if db_alert:
                    metadata = json.loads(db_alert.metadata)
                    metadata['duplicate_count'] = duplicate_count
                    db_alert.metadata = json.dumps(metadata)
                    db_alert.updated_at = datetime.utcnow()
                    session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating duplicate count for alert {original_alert_id}: {e}")
            finally:
                session.close()
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                
                # Update in database
                session = self.Session()
                try:
                    db_alert = session.query(Alert).filter(Alert.id == alert_id).first()
                    if db_alert:
                        db_alert.status = AlertStatus.ACKNOWLEDGED.value
                        db_alert.acknowledged_by = acknowledged_by
                        db_alert.acknowledged_at = datetime.utcnow()
                        db_alert.updated_at = datetime.utcnow()
                        session.commit()
                        
                        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                        return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error acknowledging alert {alert_id}: {e}")
                finally:
                    session.close()
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                # Update in database
                session = self.Session()
                try:
                    db_alert = session.query(Alert).filter(Alert.id == alert_id).first()
                    if db_alert:
                        db_alert.status = AlertStatus.RESOLVED.value
                        db_alert.resolved_at = datetime.utcnow()
                        db_alert.updated_at = datetime.utcnow()
                        session.commit()
                        
                        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                        return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error resolving alert {alert_id}: {e}")
                finally:
                    session.close()
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    # Background Tasks
    
    async def _process_correlations(self):
        """Process correlation events from Redis."""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe('security_correlations')
        
        while self.running:
            try:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    # TODO: Deserialize correlation and process
                    correlation_data = json.loads(message['data'])
                    logger.debug(f"Received correlation: {correlation_data['correlation_id']}")
            
            except Exception as e:
                logger.error(f"Error processing correlations: {e}")
                await asyncio.sleep(5)
    
    async def _handle_escalations(self):
        """Handle alert escalations based on rules."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for alert_id, alert in list(self.active_alerts.items()):
                    # Check if alert needs escalation
                    for rule in self.escalation_rules:
                        if self._should_escalate(alert, rule, current_time):
                            await self._escalate_alert(alert, rule)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error handling escalations: {e}")
                await asyncio.sleep(60)
    
    def _should_escalate(self, alert: AlertContext, rule: EscalationRule, current_time: datetime) -> bool:
        """Check if an alert should be escalated based on a rule."""
        if not rule.enabled:
            return False
        
        # Check if enough time has passed
        time_threshold = alert.created_at + timedelta(minutes=rule.delay_minutes)
        if current_time < time_threshold:
            return False
        
        # Check if alert is still active or acknowledged (not resolved)
        if alert.status in [AlertStatus.RESOLVED]:
            return False
        
        # Check if already escalated to this level
        current_escalation = alert.metadata.get('escalation_level', 0)
        if current_escalation >= rule.escalation_level:
            return False
        
        # TODO: Check other conditions from rule.conditions
        
        return True
    
    async def _escalate_alert(self, alert: AlertContext, rule: EscalationRule):
        """Escalate an alert according to a rule."""
        try:
            alert.status = AlertStatus.ESCALATED
            alert.metadata['escalation_level'] = rule.escalation_level
            alert.metadata['escalated_at'] = datetime.utcnow().isoformat()
            
            # Send escalation notifications
            for notification in rule.notifications:
                await self._send_notification_with_retry(notification, alert)
            
            # Update in database
            session = self.Session()
            try:
                db_alert = session.query(Alert).filter(Alert.id == alert.alert_id).first()
                if db_alert:
                    db_alert.status = AlertStatus.ESCALATED.value
                    db_alert.escalated_at = datetime.utcnow()
                    db_alert.escalation_level = rule.escalation_level
                    db_alert.updated_at = datetime.utcnow()
                    session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating escalated alert {alert.alert_id}: {e}")
            finally:
                session.close()
            
            logger.info(f"Escalated alert {alert.alert_id} to level {rule.escalation_level}")
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert.alert_id}: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        while self.running:
            try:
                # Clean up alerts older than 30 days
                session = self.Session()
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                deleted_count = session.query(Alert).filter(
                    Alert.resolved_at < cutoff_date
                ).delete()
                
                session.commit()
                session.close()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old alerts")
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error cleaning up old alerts: {e}")
                await asyncio.sleep(3600)
    
    async def _health_monitoring(self):
        """Monitor the health of the alert orchestrator."""
        while self.running:
            try:
                # Check component health
                health_status = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'active_alerts': len(self.active_alerts),
                    'notification_channels': len(self.notification_channels),
                    'escalation_rules': len(self.escalation_rules),
                    'deduplication_cache_size': len(self.deduplicator.alert_fingerprints)
                }
                
                # Publish health status
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.publish,
                    "alert_orchestrator_health",
                    json.dumps(health_status)
                )
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    def _load_configuration(self):
        """Load alert orchestrator configuration."""
        # TODO: Load from configuration files
        # For now, use default configuration
        
        # Default notification channels
        if 'notifications' in self.config:
            for channel_config in self.config['notifications']:
                notification = AlertNotification(
                    channel=NotificationChannel(channel_config['channel']),
                    target=channel_config['target'],
                    severity_filter=set([EventSeverity(s) for s in channel_config.get('severity_filter', [])]),
                    enabled=channel_config.get('enabled', True)
                )
                self.notification_channels.append(notification)
        
        # Default escalation rules
        default_escalation = EscalationRule(
            name="Default Escalation",
            delay_minutes=30,
            escalation_level=1,
            notifications=[
                # Add default escalation notifications
            ]
        )
        self.escalation_rules.append(default_escalation)
    
    async def get_alert_status(self) -> Dict[str, Any]:
        """Get the current status of the alert orchestrator."""
        try:
            session = self.Session()
            
            # Get alert counts by status
            alert_counts = {}
            for status in AlertStatus:
                count = session.query(Alert).filter(
                    Alert.status == status.value,
                    Alert.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                alert_counts[status.value] = count
            
            session.close()
            
            status = {
                'running': self.running,
                'active_alerts': len(self.active_alerts),
                'alert_counts_24h': alert_counts,
                'notification_channels': len(self.notification_channels),
                'escalation_rules': len(self.escalation_rules),
                'dedup_cache_size': len(self.deduplicator.alert_fingerprints),
                'last_update': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting alert status: {e}")
            return {'error': str(e)}


# Factory function
def create_alert_orchestrator(redis_url: str = "redis://localhost:6379",
                            db_url: str = "sqlite:///security_alerts.db",
                            config: Optional[Dict[str, Any]] = None) -> AlertOrchestrator:
    """Create an alert orchestrator instance."""
    return AlertOrchestrator(redis_url, db_url, config)


if __name__ == "__main__":
    async def main():
        orchestrator = create_alert_orchestrator()
        await orchestrator.start()
    
    asyncio.run(main())