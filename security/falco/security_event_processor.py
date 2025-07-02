#!/usr/bin/env python3
"""
Falco Security Event Processor for AI Trading Bot
Processes, enriches, and routes security events from Falco monitoring
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from slack_sdk.webhook import WebhookClient
import yaml

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
security_events_total = Counter(
    'falco_security_events_total',
    'Total number of security events processed',
    ['rule', 'priority', 'container', 'source']
)

security_event_processing_duration = Histogram(
    'falco_security_event_processing_duration_seconds',
    'Time spent processing security events',
    ['rule', 'priority']
)

active_security_alerts = Gauge(
    'falco_active_security_alerts',
    'Number of active security alerts',
    ['priority', 'container']
)

trading_bot_security_score = Gauge(
    'falco_trading_bot_security_score',
    'Overall security score for trading bot (0-100)',
    ['container']
)

# Pydantic models
class FalcoEvent(BaseModel):
    """Falco security event model"""
    time: str
    rule: str
    priority: str
    output: str
    output_fields: Dict[str, Any] = Field(default_factory=dict)
    source: str = "falco"
    tags: List[str] = Field(default_factory=list)
    hostname: Optional[str] = None
    uuid: Optional[str] = None

class ProcessedSecurityEvent(BaseModel):
    """Processed and enriched security event"""
    original_event: FalcoEvent
    timestamp: datetime
    severity_score: int = Field(ge=0, le=100)
    threat_category: str
    affected_services: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    correlations: List[str] = Field(default_factory=list)
    trading_impact: str
    containment_status: str = "open"
    investigation_notes: str = ""

class SecurityEventProcessor:
    """Main security event processor class"""
    
    def __init__(self, config_path: str = "/app/processor_config.yaml"):
        self.config = self._load_config(config_path)
        self.slack_client = None
        self.event_history: List[ProcessedSecurityEvent] = []
        self.threat_intelligence = self._load_threat_intelligence()
        
        # Initialize integrations
        self._initialize_slack()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load processor configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully", config_file=config_path)
            return config
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e), config_path=config_path)
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if config file not found"""
        return {
            "severity_scoring": {
                "CRITICAL": 90,
                "EMERGENCY": 100,
                "ALERT": 70,
                "WARNING": 40,
                "NOTICE": 20,
                "INFORMATIONAL": 10
            },
            "threat_categories": {
                "container_escape": "Container Security",
                "privilege_escalation": "Access Control",
                "data_exfiltration": "Data Protection",
                "credential_theft": "Authentication",
                "malware": "Malicious Activity",
                "network_anomaly": "Network Security",
                "file_tampering": "File Integrity"
            },
            "trading_impact_rules": {
                "ai-trading-bot": "HIGH",
                "bluefin-service": "MEDIUM",
                "dashboard-backend": "LOW",
                "mcp-memory": "MEDIUM",
                "mcp-omnisearch": "LOW"
            },
            "auto_containment": {
                "enabled": True,
                "critical_threshold": 90,
                "actions": ["isolate_container", "stop_trading", "alert_admin"]
            }
        }
    
    def _load_threat_intelligence(self) -> Dict[str, Any]:
        """Load threat intelligence data"""
        # In production, this would integrate with threat intelligence feeds
        return {
            "known_attack_patterns": [
                "container escape via cgroup manipulation",
                "docker socket abuse",
                "privilege escalation via setuid",
                "credential harvesting",
                "trading API abuse"
            ],
            "suspicious_processes": [
                "nc", "netcat", "ncat", "socat", "wget", "curl",
                "base64", "python -c", "perl -e", "ruby -e"
            ],
            "trusted_images": [
                "ai-trading-bot",
                "bluefin-service", 
                "mcp-memory-server",
                "mcp-omnisearch-server"
            ]
        }
    
    def _initialize_slack(self):
        """Initialize Slack webhook client"""
        slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook_url:
            self.slack_client = WebhookClient(slack_webhook_url)
            logger.info("Slack integration initialized")
        else:
            logger.warning("Slack webhook URL not configured")
    
    async def process_event(self, event: FalcoEvent) -> ProcessedSecurityEvent:
        """Process and enrich a Falco security event"""
        start_time = datetime.now()
        
        try:
            # Extract container information
            container_name = event.output_fields.get('container.name', 'unknown')
            
            # Calculate severity score
            severity_score = self._calculate_severity_score(event)
            
            # Determine threat category
            threat_category = self._categorize_threat(event)
            
            # Assess trading impact
            trading_impact = self._assess_trading_impact(container_name, event)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(event, severity_score)
            
            # Find correlations with previous events
            correlations = self._find_correlations(event)
            
            # Determine affected services
            affected_services = self._identify_affected_services(event)
            
            # Create processed event
            processed_event = ProcessedSecurityEvent(
                original_event=event,
                timestamp=datetime.fromisoformat(event.time.replace('Z', '+00:00')),
                severity_score=severity_score,
                threat_category=threat_category,
                affected_services=affected_services,
                recommended_actions=recommendations,
                correlations=correlations,
                trading_impact=trading_impact,
                containment_status="open"
            )
            
            # Store in history
            self.event_history.append(processed_event)
            
            # Update metrics
            security_events_total.labels(
                rule=event.rule,
                priority=event.priority,
                container=container_name,
                source=event.source
            ).inc()
            
            # Process duration metric
            duration = (datetime.now() - start_time).total_seconds()
            security_event_processing_duration.labels(
                rule=event.rule,
                priority=event.priority
            ).observe(duration)
            
            # Update security score
            self._update_security_score(container_name, severity_score)
            
            # Handle high-severity events
            if severity_score >= 70:
                await self._handle_high_severity_event(processed_event)
            
            logger.info(
                "Security event processed",
                rule=event.rule,
                priority=event.priority,
                container=container_name,
                severity_score=severity_score,
                threat_category=threat_category
            )
            
            return processed_event
            
        except Exception as e:
            logger.error("Error processing security event", error=str(e), event=event.dict())
            raise
    
    def _calculate_severity_score(self, event: FalcoEvent) -> int:
        """Calculate numerical severity score"""
        base_score = self.config["severity_scoring"].get(event.priority, 50)
        
        # Adjust based on container criticality
        container_name = event.output_fields.get('container.name', '')
        if container_name == 'ai-trading-bot':
            base_score += 10
        elif container_name in ['bluefin-service', 'mcp-memory']:
            base_score += 5
        
        # Adjust based on rule patterns
        rule_adjustments = {
            'escape': +20,
            'privilege': +15,
            'credential': +25,
            'tampering': +10,
            'unauthorized': +5
        }
        
        for pattern, adjustment in rule_adjustments.items():
            if pattern in event.rule.lower():
                base_score += adjustment
                break
        
        return min(100, max(0, base_score))
    
    def _categorize_threat(self, event: FalcoEvent) -> str:
        """Categorize the type of threat"""
        rule_lower = event.rule.lower()
        output_lower = event.output.lower()
        
        categories = {
            'container_escape': ['escape', 'namespace', 'mount', 'cgroup'],
            'privilege_escalation': ['privilege', 'sudo', 'setuid', 'capabilities'],
            'data_exfiltration': ['exfiltration', 'unauthorized access', 'data'],
            'credential_theft': ['credential', 'api_key', 'private_key', 'secret'],
            'malware': ['malicious', 'backdoor', 'trojan', 'virus'],
            'network_anomaly': ['network', 'connection', 'port scanning', 'dns'],
            'file_tampering': ['tampering', 'modification', 'write', 'unlink']
        }
        
        for category, keywords in categories.items():
            if any(keyword in rule_lower or keyword in output_lower for keyword in keywords):
                return self.config["threat_categories"].get(category, category)
        
        return "Unknown"
    
    def _assess_trading_impact(self, container_name: str, event: FalcoEvent) -> str:
        """Assess the impact on trading operations"""
        # Base impact from container importance
        base_impact = self.config["trading_impact_rules"].get(container_name, "LOW")
        
        # Adjust based on event content
        high_impact_indicators = [
            'trading', 'position', 'order', 'balance', 'api_key',
            'private_key', 'exchange', 'wallet', 'credential'
        ]
        
        event_text = (event.rule + " " + event.output).lower()
        
        if any(indicator in event_text for indicator in high_impact_indicators):
            if base_impact == "LOW":
                return "MEDIUM"
            elif base_impact == "MEDIUM":
                return "HIGH"
            else:
                return "CRITICAL"
        
        return base_impact
    
    def _generate_recommendations(self, event: FalcoEvent, severity_score: int) -> List[str]:
        """Generate recommended actions"""
        recommendations = []
        
        # Generic recommendations based on severity
        if severity_score >= 90:
            recommendations.extend([
                "IMMEDIATE: Stop all trading operations",
                "Isolate affected container",
                "Preserve forensic evidence",
                "Initiate incident response"
            ])
        elif severity_score >= 70:
            recommendations.extend([
                "Review container logs immediately",
                "Check for signs of compromise",
                "Verify trading operations integrity"
            ])
        elif severity_score >= 40:
            recommendations.extend([
                "Monitor container for additional activity",
                "Review security configurations"
            ])
        
        # Specific recommendations based on rule type
        rule_lower = event.rule.lower()
        
        if 'credential' in rule_lower or 'api_key' in rule_lower:
            recommendations.extend([
                "Rotate all API keys immediately",
                "Review authentication logs",
                "Check for unauthorized trading activity"
            ])
        
        if 'escape' in rule_lower or 'privilege' in rule_lower:
            recommendations.extend([
                "Review container security settings",
                "Check host system integrity",
                "Verify container isolation"
            ])
        
        if 'network' in rule_lower:
            recommendations.extend([
                "Review network connections",
                "Check firewall rules",
                "Monitor for data exfiltration"
            ])
        
        return recommendations
    
    def _find_correlations(self, event: FalcoEvent) -> List[str]:
        """Find correlations with previous events"""
        correlations = []
        
        # Look for similar events in recent history (last 100 events)
        recent_events = self.event_history[-100:] if len(self.event_history) >= 100 else self.event_history
        
        container_name = event.output_fields.get('container.name', '')
        
        for past_event in recent_events:
            past_container = past_event.original_event.output_fields.get('container.name', '')
            
            # Same container, similar rule
            if (container_name == past_container and 
                self._rules_similar(event.rule, past_event.original_event.rule)):
                correlations.append(f"Similar event in {past_container}: {past_event.original_event.rule}")
            
            # Related containers
            if self._containers_related(container_name, past_container):
                correlations.append(f"Related event in {past_container}: {past_event.original_event.rule}")
        
        return correlations[:5]  # Limit to 5 correlations
    
    def _rules_similar(self, rule1: str, rule2: str) -> bool:
        """Check if two rules are similar"""
        # Simple similarity check based on keywords
        keywords1 = set(rule1.lower().split())
        keywords2 = set(rule2.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords1 -= common_words
        keywords2 -= common_words
        
        if not keywords1 or not keywords2:
            return False
        
        # Calculate similarity
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union) > 0.5
    
    def _containers_related(self, container1: str, container2: str) -> bool:
        """Check if two containers are related"""
        related_groups = [
            ['ai-trading-bot', 'bluefin-service'],
            ['dashboard-backend', 'dashboard-frontend'],
            ['mcp-memory', 'mcp-omnisearch']
        ]
        
        for group in related_groups:
            if container1 in group and container2 in group:
                return True
        
        return False
    
    def _identify_affected_services(self, event: FalcoEvent) -> List[str]:
        """Identify services affected by the security event"""
        affected = []
        
        container_name = event.output_fields.get('container.name', '')
        if container_name:
            affected.append(container_name)
        
        # Add related services based on dependencies
        service_dependencies = {
            'ai-trading-bot': ['bluefin-service', 'mcp-memory', 'mcp-omnisearch'],
            'bluefin-service': ['ai-trading-bot'],
            'dashboard-backend': ['ai-trading-bot'],
            'mcp-memory': ['ai-trading-bot'],
            'mcp-omnisearch': ['ai-trading-bot']
        }
        
        if container_name in service_dependencies:
            # For critical events, include dependent services
            if event.priority in ['CRITICAL', 'EMERGENCY']:
                affected.extend(service_dependencies[container_name])
        
        return list(set(affected))  # Remove duplicates
    
    def _update_security_score(self, container_name: str, severity_score: int):
        """Update security score metrics"""
        # Calculate overall security score (higher is better)
        # Invert severity score (higher severity = lower security)
        security_score = max(0, 100 - severity_score)
        
        trading_bot_security_score.labels(container=container_name).set(security_score)
    
    async def _handle_high_severity_event(self, processed_event: ProcessedSecurityEvent):
        """Handle high-severity events with immediate actions"""
        event = processed_event.original_event
        
        # Send immediate notifications
        if self.slack_client:
            await self._send_slack_alert(processed_event)
        
        # Auto-containment for critical events
        if (processed_event.severity_score >= 90 and 
            self.config.get("auto_containment", {}).get("enabled", False)):
            await self._trigger_auto_containment(processed_event)
        
        # Update active alerts gauge
        active_security_alerts.labels(
            priority=event.priority,
            container=event.output_fields.get('container.name', 'unknown')
        ).inc()
    
    async def _send_slack_alert(self, processed_event: ProcessedSecurityEvent):
        """Send Slack alert for security event"""
        try:
            event = processed_event.original_event
            
            color = {
                'CRITICAL': '#FF0000',
                'EMERGENCY': '#8B0000',
                'ALERT': '#FFA500',
                'WARNING': '#FFFF00'
            }.get(event.priority, '#808080')
            
            message = {
                "text": f"🚨 Security Alert: {event.rule}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Rule",
                                "value": event.rule,
                                "short": True
                            },
                            {
                                "title": "Priority",
                                "value": event.priority,
                                "short": True
                            },
                            {
                                "title": "Container",
                                "value": event.output_fields.get('container.name', 'Unknown'),
                                "short": True
                            },
                            {
                                "title": "Severity Score",
                                "value": f"{processed_event.severity_score}/100",
                                "short": True
                            },
                            {
                                "title": "Trading Impact",
                                "value": processed_event.trading_impact,
                                "short": True
                            },
                            {
                                "title": "Threat Category",
                                "value": processed_event.threat_category,
                                "short": True
                            },
                            {
                                "title": "Details",
                                "value": event.output,
                                "short": False
                            },
                            {
                                "title": "Recommended Actions",
                                "value": "\n".join(f"• {action}" for action in processed_event.recommended_actions[:3]),
                                "short": False
                            }
                        ],
                        "footer": "Falco Security Monitor",
                        "ts": int(processed_event.timestamp.timestamp())
                    }
                ]
            }
            
            response = self.slack_client.send_dict(message)
            if response.status_code == 200:
                logger.info("Slack alert sent successfully", rule=event.rule)
            else:
                logger.error("Failed to send Slack alert", status_code=response.status_code)
                
        except Exception as e:
            logger.error("Error sending Slack alert", error=str(e))
    
    async def _trigger_auto_containment(self, processed_event: ProcessedSecurityEvent):
        """Trigger automatic containment actions"""
        try:
            container_name = processed_event.original_event.output_fields.get('container.name')
            
            if not container_name:
                return
            
            logger.warning(
                "Triggering auto-containment",
                container=container_name,
                severity_score=processed_event.severity_score
            )
            
            # Actions based on configuration
            actions = self.config.get("auto_containment", {}).get("actions", [])
            
            for action in actions:
                if action == "isolate_container":
                    await self._isolate_container(container_name)
                elif action == "stop_trading":
                    await self._emergency_stop_trading()
                elif action == "alert_admin":
                    await self._alert_admin(processed_event)
                    
        except Exception as e:
            logger.error("Error in auto-containment", error=str(e))
    
    async def _isolate_container(self, container_name: str):
        """Isolate a container by removing network access"""
        # In a real implementation, this would use Docker API
        logger.critical("CONTAINER ISOLATION TRIGGERED", container=container_name)
        # docker network disconnect trading-network container_name
    
    async def _emergency_stop_trading(self):
        """Trigger emergency stop of trading operations"""
        logger.critical("EMERGENCY TRADING STOP TRIGGERED")
        # Send stop signal to trading bot
    
    async def _alert_admin(self, processed_event: ProcessedSecurityEvent):
        """Send immediate alert to administrators"""
        logger.critical("ADMIN ALERT TRIGGERED", event=processed_event.dict())

# FastAPI application
app = FastAPI(title="Falco Security Event Processor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = SecurityEventProcessor()

@app.post("/security-event")
async def receive_security_event(request: Request):
    """Receive and process security events from Falco"""
    try:
        event_data = await request.json()
        
        # Parse Falco event
        falco_event = FalcoEvent(**event_data)
        
        # Process the event
        processed_event = await processor.process_event(falco_event)
        
        return {
            "status": "processed",
            "event_id": processed_event.original_event.uuid,
            "severity_score": processed_event.severity_score,
            "threat_category": processed_event.threat_category
        }
        
    except Exception as e:
        logger.error("Error processing security event", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "events_processed": len(processor.event_history),
        "service": "falco-security-processor"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/events")
async def get_recent_events(limit: int = 50):
    """Get recent security events"""
    recent_events = processor.event_history[-limit:] if processor.event_history else []
    return {
        "events": [event.dict() for event in recent_events],
        "total_events": len(processor.event_history)
    }

@app.get("/events/{event_id}")
async def get_event_details(event_id: str):
    """Get details for a specific event"""
    for event in processor.event_history:
        if event.original_event.uuid == event_id:
            return event.dict()
    
    raise HTTPException(status_code=404, detail="Event not found")

@app.post("/events/{event_id}/resolve")
async def resolve_event(event_id: str):
    """Mark an event as resolved"""
    for event in processor.event_history:
        if event.original_event.uuid == event_id:
            event.containment_status = "resolved"
            return {"status": "resolved", "event_id": event_id}
    
    raise HTTPException(status_code=404, detail="Event not found")

if __name__ == "__main__":
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level=log_level.lower(),
        access_log=True
    )