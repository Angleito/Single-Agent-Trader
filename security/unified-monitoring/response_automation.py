"""
Automated Response Workflows for OPTIMIZE Platform

This module provides automated security response capabilities including
incident response, threat containment, evidence collection, and self-healing
infrastructure management.
"""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import aiofiles
import aiohttp
import redis
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .alert_orchestrator import AlertContext, AlertStatus
from .correlation_engine import CorrelationResult, EventSeverity

logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()


class ResponseAction(Base):
    """Database model for response actions."""
    __tablename__ = 'response_actions'
    
    id = Column(String, primary_key=True)
    incident_id = Column(String)
    action_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    executed_by = Column(String)
    action_data = Column(Text)
    result_data = Column(Text)
    error_message = Column(Text)


class SecurityIncident(Base):
    """Database model for security incidents."""
    __tablename__ = 'security_incidents'
    
    id = Column(String, primary_key=True)
    alert_id = Column(String)
    correlation_id = Column(String)
    incident_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)
    assigned_to = Column(String)
    escalated_at = Column(DateTime)
    escalation_level = Column(Integer, default=0)
    metadata = Column(Text)


# Enums and Data Classes
class ResponseActionType(Enum):
    """Types of automated response actions."""
    ISOLATE_CONTAINER = "isolate_container"
    BLOCK_IP_ADDRESS = "block_ip_address"
    SUSPEND_USER = "suspend_user"
    COLLECT_FORENSICS = "collect_forensics"
    QUARANTINE_FILE = "quarantine_file"
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    UPDATE_FIREWALL = "update_firewall"
    ROTATE_CREDENTIALS = "rotate_credentials"
    BACKUP_DATA = "backup_data"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    CREATE_TICKET = "create_ticket"
    PATCH_VULNERABILITY = "patch_vulnerability"
    STOP_TRADING = "stop_trading"
    NETWORK_SEGMENTATION = "network_segmentation"


class ActionStatus(Enum):
    """Response action execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class IncidentStatus(Enum):
    """Security incident status."""
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"


class ResponseSeverity(Enum):
    """Response action severity levels."""
    IMMEDIATE = "immediate"  # Execute immediately
    HIGH = "high"           # Execute within 5 minutes
    MEDIUM = "medium"       # Execute within 30 minutes
    LOW = "low"            # Execute within 2 hours
    SCHEDULED = "scheduled" # Execute during maintenance window


@dataclass
class ResponsePlaybook:
    """Automated response playbook definition."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    severity_threshold: EventSeverity = EventSeverity.MEDIUM
    auto_execute: bool = False
    require_approval: bool = True
    timeout_minutes: int = 60
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches_conditions(self, alert: AlertContext) -> bool:
        """Check if this playbook matches the alert conditions."""
        # TODO: Implement sophisticated condition matching
        if alert.severity.value not in self.trigger_conditions.get('severity', []):
            return False
        
        if alert.alert_type.value not in self.trigger_conditions.get('alert_types', []):
            return False
        
        # Check for specific patterns
        pattern_names = self.trigger_conditions.get('pattern_names', [])
        if pattern_names:
            correlation_pattern = alert.metadata.get('pattern_name', '')
            if correlation_pattern not in pattern_names:
                return False
        
        return True


@dataclass
class ActionResult:
    """Result of a response action execution."""
    action_id: str
    action_type: ResponseActionType
    status: ActionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    output: str = ""
    error_message: str = ""
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ContainerIsolationHandler:
    """Handles container isolation responses."""
    
    async def isolate_container(self, container_name: str) -> ActionResult:
        """Isolate a container from the network."""
        action_id = str(uuid4())
        result = ActionResult(
            action_id=action_id,
            action_type=ResponseActionType.ISOLATE_CONTAINER,
            status=ActionStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Create isolated network if it doesn't exist
            await self._ensure_isolated_network()
            
            # Move container to isolated network
            cmd = [
                "docker", "network", "disconnect", "trading-network", container_name
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Connect to isolated network
                cmd = [
                    "docker", "network", "connect", "isolated-network", container_name
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout2, stderr2 = await process.communicate()
                
                if process.returncode == 0:
                    result.success = True
                    result.status = ActionStatus.COMPLETED
                    result.output = f"Container {container_name} isolated successfully"
                    
                    # Log isolation action
                    isolation_log = {
                        'action': 'container_isolation',
                        'container': container_name,
                        'timestamp': datetime.utcnow().isoformat(),
                        'networks': {
                            'disconnected_from': 'trading-network',
                            'connected_to': 'isolated-network'
                        }
                    }
                    await self._log_action(isolation_log)
                else:
                    result.error_message = f"Failed to connect to isolated network: {stderr2.decode()}"
            else:
                result.error_message = f"Failed to disconnect from trading network: {stderr.decode()}"
            
        except Exception as e:
            result.error_message = f"Error isolating container: {str(e)}"
            logger.error(f"Container isolation failed: {e}")
        
        finally:
            result.completed_at = datetime.utcnow()
            if not result.success:
                result.status = ActionStatus.FAILED
        
        return result
    
    async def _ensure_isolated_network(self):
        """Ensure isolated network exists."""
        try:
            # Check if isolated network exists
            cmd = ["docker", "network", "inspect", "isolated-network"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate()
            
            if process.returncode != 0:
                # Create isolated network
                cmd = [
                    "docker", "network", "create",
                    "--driver", "bridge",
                    "--internal",
                    "isolated-network"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                
                logger.info("Created isolated network for container isolation")
        
        except Exception as e:
            logger.error(f"Error ensuring isolated network: {e}")
    
    async def _log_action(self, action_data: Dict[str, Any]):
        """Log security action for audit trail."""
        try:
            log_file = "/var/log/security/isolation_actions.log"
            async with aiofiles.open(log_file, "a") as f:
                await f.write(json.dumps(action_data) + "\n")
        except Exception as e:
            logger.error(f"Error logging isolation action: {e}")


class ForensicsCollector:
    """Handles forensic evidence collection."""
    
    def __init__(self):
        self.evidence_dir = "/var/log/security/evidence"
    
    async def collect_forensics(self, target: str, incident_id: str) -> ActionResult:
        """Collect forensic evidence for an incident."""
        action_id = str(uuid4())
        result = ActionResult(
            action_id=action_id,
            action_type=ResponseActionType.COLLECT_FORENSICS,
            status=ActionStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            evidence_path = f"{self.evidence_dir}/{incident_id}"
            await self._ensure_evidence_directory(evidence_path)
            
            # Collect container logs
            if await self._is_container(target):
                await self._collect_container_evidence(target, evidence_path)
            
            # Collect system information
            await self._collect_system_evidence(evidence_path)
            
            # Collect network information
            await self._collect_network_evidence(evidence_path)
            
            # Create evidence manifest
            manifest = await self._create_evidence_manifest(evidence_path, incident_id)
            
            result.success = True
            result.status = ActionStatus.COMPLETED
            result.output = f"Forensic evidence collected in {evidence_path}"
            result.artifacts = [evidence_path, f"{evidence_path}/manifest.json"]
            
        except Exception as e:
            result.error_message = f"Error collecting forensics: {str(e)}"
            logger.error(f"Forensics collection failed: {e}")
            result.status = ActionStatus.FAILED
        
        finally:
            result.completed_at = datetime.utcnow()
        
        return result
    
    async def _ensure_evidence_directory(self, path: str):
        """Ensure evidence directory exists."""
        import os
        os.makedirs(path, exist_ok=True)
    
    async def _is_container(self, target: str) -> bool:
        """Check if target is a container."""
        try:
            cmd = ["docker", "inspect", target]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate()
            return process.returncode == 0
        except:
            return False
    
    async def _collect_container_evidence(self, container: str, evidence_path: str):
        """Collect evidence from a container."""
        # Container logs
        cmd = ["docker", "logs", container]
        await self._run_and_save(cmd, f"{evidence_path}/container_logs.txt")
        
        # Container inspect
        cmd = ["docker", "inspect", container]
        await self._run_and_save(cmd, f"{evidence_path}/container_inspect.json")
        
        # Container processes
        cmd = ["docker", "exec", container, "ps", "aux"]
        await self._run_and_save(cmd, f"{evidence_path}/container_processes.txt")
        
        # Container network connections
        cmd = ["docker", "exec", container, "netstat", "-tulpn"]
        await self._run_and_save(cmd, f"{evidence_path}/container_network.txt")
    
    async def _collect_system_evidence(self, evidence_path: str):
        """Collect system-level evidence."""
        # System processes
        cmd = ["ps", "aux"]
        await self._run_and_save(cmd, f"{evidence_path}/system_processes.txt")
        
        # Network connections
        cmd = ["netstat", "-tulpn"]
        await self._run_and_save(cmd, f"{evidence_path}/system_network.txt")
        
        # System logs (last 1000 lines)
        cmd = ["tail", "-n", "1000", "/var/log/syslog"]
        await self._run_and_save(cmd, f"{evidence_path}/system_logs.txt")
        
        # Docker containers
        cmd = ["docker", "ps", "-a"]
        await self._run_and_save(cmd, f"{evidence_path}/docker_containers.txt")
    
    async def _collect_network_evidence(self, evidence_path: str):
        """Collect network-related evidence."""
        # Routing table
        cmd = ["route", "-n"]
        await self._run_and_save(cmd, f"{evidence_path}/routing_table.txt")
        
        # Network interfaces
        cmd = ["ip", "addr", "show"]
        await self._run_and_save(cmd, f"{evidence_path}/network_interfaces.txt")
        
        # Firewall rules
        cmd = ["iptables", "-L", "-n"]
        await self._run_and_save(cmd, f"{evidence_path}/firewall_rules.txt")
    
    async def _run_and_save(self, cmd: List[str], output_file: str):
        """Run command and save output to file."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            async with aiofiles.open(output_file, "w") as f:
                if stdout:
                    await f.write(stdout.decode())
                if stderr:
                    await f.write(f"\nSTDERR:\n{stderr.decode()}")
        
        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}")
    
    async def _create_evidence_manifest(self, evidence_path: str, incident_id: str) -> str:
        """Create evidence collection manifest."""
        import os
        
        manifest = {
            'incident_id': incident_id,
            'collection_timestamp': datetime.utcnow().isoformat(),
            'collector_version': '1.0.0',
            'evidence_files': []
        }
        
        # List all collected files
        for root, dirs, files in os.walk(evidence_path):
            for file in files:
                if file != 'manifest.json':
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, evidence_path)
                    file_size = os.path.getsize(file_path)
                    
                    manifest['evidence_files'].append({
                        'filename': relative_path,
                        'size_bytes': file_size,
                        'collection_time': datetime.utcnow().isoformat()
                    })
        
        manifest_path = f"{evidence_path}/manifest.json"
        async with aiofiles.open(manifest_path, "w") as f:
            await f.write(json.dumps(manifest, indent=2))
        
        return manifest_path


class TradingProtectionHandler:
    """Handles trading bot protection responses."""
    
    async def stop_trading(self, reason: str) -> ActionResult:
        """Stop trading operations for security reasons."""
        action_id = str(uuid4())
        result = ActionResult(
            action_id=action_id,
            action_type=ResponseActionType.STOP_TRADING,
            status=ActionStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Send stop signal to trading bot
            stop_signal = {
                'action': 'emergency_stop',
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'initiated_by': 'security_automation'
            }
            
            # Try to send via API first
            success = await self._send_stop_signal_api(stop_signal)
            
            if not success:
                # Fallback to container signal
                success = await self._send_stop_signal_container()
            
            if success:
                result.success = True
                result.status = ActionStatus.COMPLETED
                result.output = f"Trading stopped successfully. Reason: {reason}"
                
                # Log the emergency stop
                await self._log_emergency_stop(stop_signal)
            else:
                result.error_message = "Failed to stop trading operations"
                result.status = ActionStatus.FAILED
        
        except Exception as e:
            result.error_message = f"Error stopping trading: {str(e)}"
            result.status = ActionStatus.FAILED
            logger.error(f"Trading stop failed: {e}")
        
        finally:
            result.completed_at = datetime.utcnow()
        
        return result
    
    async def _send_stop_signal_api(self, signal_data: Dict[str, Any]) -> bool:
        """Send stop signal via trading bot API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://ai-trading-bot:8000/api/emergency/stop",
                    json=signal_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def _send_stop_signal_container(self) -> bool:
        """Send stop signal via container command."""
        try:
            cmd = ["docker", "exec", "ai-trading-bot", "pkill", "-USR1", "python"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except:
            return False
    
    async def _log_emergency_stop(self, signal_data: Dict[str, Any]):
        """Log emergency stop action."""
        try:
            log_file = "/var/log/security/emergency_stops.log"
            async with aiofiles.open(log_file, "a") as f:
                await f.write(json.dumps(signal_data) + "\n")
        except Exception as e:
            logger.error(f"Error logging emergency stop: {e}")


class NetworkSecurityHandler:
    """Handles network security responses."""
    
    async def block_ip_address(self, ip_address: str, duration_minutes: int = 60) -> ActionResult:
        """Block an IP address using iptables."""
        action_id = str(uuid4())
        result = ActionResult(
            action_id=action_id,
            action_type=ResponseActionType.BLOCK_IP_ADDRESS,
            status=ActionStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Add iptables rule to block IP
            cmd = [
                "iptables", "-I", "INPUT", "-s", ip_address, "-j", "DROP"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result.success = True
                result.status = ActionStatus.COMPLETED
                result.output = f"IP address {ip_address} blocked successfully"
                
                # Schedule removal of the block
                if duration_minutes > 0:
                    asyncio.create_task(
                        self._schedule_ip_unblock(ip_address, duration_minutes)
                    )
                
                # Log the IP block
                block_log = {
                    'action': 'ip_block',
                    'ip_address': ip_address,
                    'duration_minutes': duration_minutes,
                    'timestamp': datetime.utcnow().isoformat()
                }
                await self._log_network_action(block_log)
            else:
                result.error_message = f"Failed to block IP: {stderr.decode()}"
                result.status = ActionStatus.FAILED
        
        except Exception as e:
            result.error_message = f"Error blocking IP: {str(e)}"
            result.status = ActionStatus.FAILED
            logger.error(f"IP blocking failed: {e}")
        
        finally:
            result.completed_at = datetime.utcnow()
        
        return result
    
    async def _schedule_ip_unblock(self, ip_address: str, delay_minutes: int):
        """Schedule IP address unblocking."""
        await asyncio.sleep(delay_minutes * 60)
        
        try:
            cmd = [
                "iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            logger.info(f"Automatically unblocked IP address {ip_address}")
            
            # Log the unblock
            unblock_log = {
                'action': 'ip_unblock',
                'ip_address': ip_address,
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'scheduled_expiration'
            }
            await self._log_network_action(unblock_log)
        
        except Exception as e:
            logger.error(f"Error unblocking IP {ip_address}: {e}")
    
    async def _log_network_action(self, action_data: Dict[str, Any]):
        """Log network security action."""
        try:
            log_file = "/var/log/security/network_actions.log"
            async with aiofiles.open(log_file, "a") as f:
                await f.write(json.dumps(action_data) + "\n")
        except Exception as e:
            logger.error(f"Error logging network action: {e}")


class ResponseAutomation:
    """Main automated response system."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 db_url: str = "sqlite:///security_responses.db"):
        self.redis_client = redis.from_url(redis_url)
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Response handlers
        self.container_handler = ContainerIsolationHandler()
        self.forensics_handler = ForensicsCollector()
        self.trading_handler = TradingProtectionHandler()
        self.network_handler = NetworkSecurityHandler()
        
        # Playbooks
        self.playbooks: List[ResponsePlaybook] = []
        self.load_default_playbooks()
        
        # Internal state
        self.running = False
        self.active_responses = {}
        self.response_queue = asyncio.Queue()
        
        # Configuration
        self.max_concurrent_responses = 5
        self.response_timeout = timedelta(minutes=60)
    
    async def start(self):
        """Start the response automation system."""
        self.running = True
        logger.info("Starting Response Automation System")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._process_alerts()),
            asyncio.create_task(self._execute_responses()),
            asyncio.create_task(self._monitor_active_responses()),
            asyncio.create_task(self._cleanup_old_responses())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the response automation system."""
        self.running = False
        logger.info("Response Automation System stopped")
    
    def load_default_playbooks(self):
        """Load default response playbooks."""
        # Container Anomaly Playbook
        container_anomaly_playbook = ResponsePlaybook(
            name="Container Anomaly Response",
            description="Respond to suspicious container behavior",
            trigger_conditions={
                'severity': ['critical', 'high'],
                'pattern_names': ['container_escape', 'privilege_escalation']
            },
            actions=[
                {
                    'type': ResponseActionType.COLLECT_FORENSICS.value,
                    'target': '{{affected_entities[0]}}',
                    'priority': 1
                },
                {
                    'type': ResponseActionType.ISOLATE_CONTAINER.value,
                    'target': '{{affected_entities[0]}}',
                    'priority': 2
                },
                {
                    'type': ResponseActionType.NOTIFY_STAKEHOLDERS.value,
                    'channels': ['slack', 'email'],
                    'priority': 3
                }
            ],
            auto_execute=False,
            require_approval=True
        )
        self.playbooks.append(container_anomaly_playbook)
        
        # Network Intrusion Playbook
        network_intrusion_playbook = ResponsePlaybook(
            name="Network Intrusion Response",
            description="Respond to network-based attacks",
            trigger_conditions={
                'severity': ['critical', 'high'],
                'pattern_names': ['lateral_movement', 'data_exfiltration']
            },
            actions=[
                {
                    'type': ResponseActionType.BLOCK_IP_ADDRESS.value,
                    'target': '{{source_ip}}',
                    'duration_minutes': 60,
                    'priority': 1
                },
                {
                    'type': ResponseActionType.COLLECT_FORENSICS.value,
                    'target': 'network',
                    'priority': 2
                },
                {
                    'type': ResponseActionType.NETWORK_SEGMENTATION.value,
                    'priority': 3
                }
            ],
            auto_execute=True,
            require_approval=False
        )
        self.playbooks.append(network_intrusion_playbook)
        
        # Critical Security Event Playbook
        critical_event_playbook = ResponsePlaybook(
            name="Critical Security Event Response",
            description="Emergency response for critical security events",
            trigger_conditions={
                'severity': ['critical']
            },
            actions=[
                {
                    'type': ResponseActionType.STOP_TRADING.value,
                    'reason': 'Critical security event detected',
                    'priority': 1
                },
                {
                    'type': ResponseActionType.COLLECT_FORENSICS.value,
                    'target': 'all',
                    'priority': 2
                },
                {
                    'type': ResponseActionType.NOTIFY_STAKEHOLDERS.value,
                    'channels': ['pagerduty', 'slack'],
                    'priority': 3
                },
                {
                    'type': ResponseActionType.CREATE_TICKET.value,
                    'priority': 4
                }
            ],
            auto_execute=True,
            require_approval=False,
            severity_threshold=EventSeverity.CRITICAL
        )
        self.playbooks.append(critical_event_playbook)
    
    async def process_alert(self, alert: AlertContext) -> Optional[str]:
        """Process an alert and determine automated response."""
        try:
            # Find matching playbooks
            matching_playbooks = [
                playbook for playbook in self.playbooks
                if playbook.enabled and playbook.matches_conditions(alert)
            ]
            
            if not matching_playbooks:
                logger.debug(f"No matching playbooks for alert {alert.alert_id}")
                return None
            
            # Select the most appropriate playbook (highest severity threshold)
            selected_playbook = max(
                matching_playbooks,
                key=lambda p: p.severity_threshold.value
            )
            
            # Create incident
            incident_id = await self._create_incident(alert, selected_playbook)
            
            # Queue response actions
            for action_config in selected_playbook.actions:
                await self._queue_response_action(incident_id, alert, action_config, selected_playbook)
            
            logger.info(f"Queued response actions for alert {alert.alert_id} using playbook {selected_playbook.name}")
            return incident_id
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id} for automated response: {e}")
            return None
    
    async def _create_incident(self, alert: AlertContext, playbook: ResponsePlaybook) -> str:
        """Create a security incident."""
        incident_id = str(uuid4())
        
        session = self.Session()
        try:
            incident = SecurityIncident(
                id=incident_id,
                alert_id=alert.alert_id,
                correlation_id=alert.correlation_id,
                incident_type=playbook.name,
                severity=alert.severity.value,
                status=IncidentStatus.NEW.value,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=json.dumps({
                    'playbook_id': playbook.id,
                    'auto_execute': playbook.auto_execute,
                    'require_approval': playbook.require_approval,
                    'alert_metadata': alert.metadata
                })
            )
            
            session.add(incident)
            session.commit()
            
            logger.info(f"Created security incident {incident_id} for alert {alert.alert_id}")
            return incident_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating incident: {e}")
            raise
        finally:
            session.close()
    
    async def _queue_response_action(self, 
                                   incident_id: str,
                                   alert: AlertContext,
                                   action_config: Dict[str, Any],
                                   playbook: ResponsePlaybook):
        """Queue a response action for execution."""
        try:
            action_id = str(uuid4())
            
            # Create response action record
            session = self.Session()
            try:
                action = ResponseAction(
                    id=action_id,
                    incident_id=incident_id,
                    action_type=action_config['type'],
                    status=ActionStatus.PENDING.value,
                    started_at=datetime.utcnow(),
                    executed_by='automation',
                    action_data=json.dumps({
                        'config': action_config,
                        'alert_data': alert.to_dict(),
                        'playbook_id': playbook.id,
                        'auto_execute': playbook.auto_execute,
                        'require_approval': playbook.require_approval
                    })
                )
                
                session.add(action)
                session.commit()
            finally:
                session.close()
            
            # Add to execution queue
            await self.response_queue.put({
                'action_id': action_id,
                'incident_id': incident_id,
                'action_config': action_config,
                'alert': alert,
                'playbook': playbook
            })
            
        except Exception as e:
            logger.error(f"Error queueing response action: {e}")
    
    async def _process_alerts(self):
        """Process incoming alerts from Redis."""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe('security_alerts')
        
        while self.running:
            try:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    alert_data = json.loads(message['data'])
                    
                    # TODO: Reconstruct AlertContext from alert_data
                    # For now, just log the alert
                    logger.debug(f"Received alert for processing: {alert_data.get('alert_id')}")
            
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(5)
    
    async def _execute_responses(self):
        """Execute queued response actions."""
        while self.running:
            try:
                # Respect concurrent execution limit
                if len(self.active_responses) >= self.max_concurrent_responses:
                    await asyncio.sleep(1)
                    continue
                
                # Get next action from queue
                try:
                    action_data = await asyncio.wait_for(
                        self.response_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if approval is required
                if action_data['playbook'].require_approval and not action_data['playbook'].auto_execute:
                    logger.info(f"Action {action_data['action_id']} requires approval, skipping auto-execution")
                    continue
                
                # Execute action
                task = asyncio.create_task(
                    self._execute_action(action_data)
                )
                self.active_responses[action_data['action_id']] = task
                
            except Exception as e:
                logger.error(f"Error in response execution loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_action(self, action_data: Dict[str, Any]):
        """Execute a specific response action."""
        action_id = action_data['action_id']
        action_config = action_data['action_config']
        alert = action_data['alert']
        
        try:
            # Update action status to running
            await self._update_action_status(action_id, ActionStatus.RUNNING)
            
            # Execute based on action type
            action_type = ResponseActionType(action_config['type'])
            
            if action_type == ResponseActionType.ISOLATE_CONTAINER:
                target = self._resolve_template(action_config.get('target', ''), alert)
                result = await self.container_handler.isolate_container(target)
                
            elif action_type == ResponseActionType.COLLECT_FORENSICS:
                target = self._resolve_template(action_config.get('target', ''), alert)
                incident_id = action_data['incident_id']
                result = await self.forensics_handler.collect_forensics(target, incident_id)
                
            elif action_type == ResponseActionType.STOP_TRADING:
                reason = action_config.get('reason', 'Security event')
                result = await self.trading_handler.stop_trading(reason)
                
            elif action_type == ResponseActionType.BLOCK_IP_ADDRESS:
                ip_address = self._resolve_template(action_config.get('target', ''), alert)
                duration = action_config.get('duration_minutes', 60)
                result = await self.network_handler.block_ip_address(ip_address, duration)
                
            else:
                # TODO: Implement other action types
                result = ActionResult(
                    action_id=action_id,
                    action_type=action_type,
                    status=ActionStatus.SKIPPED,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    output=f"Action type {action_type.value} not implemented yet"
                )
            
            # Store result
            await self._store_action_result(action_id, result)
            
        except Exception as e:
            logger.error(f"Error executing action {action_id}: {e}")
            
            # Create error result
            result = ActionResult(
                action_id=action_id,
                action_type=ResponseActionType(action_config['type']),
                status=ActionStatus.FAILED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                error_message=str(e)
            )
            await self._store_action_result(action_id, result)
        
        finally:
            # Remove from active responses
            if action_id in self.active_responses:
                del self.active_responses[action_id]
    
    def _resolve_template(self, template: str, alert: AlertContext) -> str:
        """Resolve template variables in action configuration."""
        # Simple template resolution
        if '{{affected_entities[0]}}' in template and alert.affected_entities:
            return template.replace('{{affected_entities[0]}}', alert.affected_entities[0])
        
        # TODO: Implement more sophisticated template resolution
        return template
    
    async def _update_action_status(self, action_id: str, status: ActionStatus):
        """Update action status in database."""
        session = self.Session()
        try:
            action = session.query(ResponseAction).filter(ResponseAction.id == action_id).first()
            if action:
                action.status = status.value
                if status in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.CANCELLED]:
                    action.completed_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating action status: {e}")
        finally:
            session.close()
    
    async def _store_action_result(self, action_id: str, result: ActionResult):
        """Store action execution result in database."""
        session = self.Session()
        try:
            action = session.query(ResponseAction).filter(ResponseAction.id == action_id).first()
            if action:
                action.status = result.status.value
                action.completed_at = result.completed_at
                action.result_data = json.dumps({
                    'success': result.success,
                    'output': result.output,
                    'error_message': result.error_message,
                    'artifacts': result.artifacts,
                    'metrics': result.metrics
                })
                if result.error_message:
                    action.error_message = result.error_message
                session.commit()
                
                logger.info(f"Stored result for action {action_id}: {result.status.value}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing action result: {e}")
        finally:
            session.close()
    
    async def _monitor_active_responses(self):
        """Monitor active responses for timeouts."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_actions = []
                
                for action_id, task in self.active_responses.items():
                    if task.done():
                        expired_actions.append(action_id)
                    # TODO: Check for timeout based on start time
                
                # Clean up completed/expired actions
                for action_id in expired_actions:
                    if action_id in self.active_responses:
                        del self.active_responses[action_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring active responses: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_old_responses(self):
        """Clean up old response data."""
        while self.running:
            try:
                # Clean up actions older than 30 days
                session = self.Session()
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                deleted_actions = session.query(ResponseAction).filter(
                    ResponseAction.completed_at < cutoff_date
                ).delete()
                
                deleted_incidents = session.query(SecurityIncident).filter(
                    SecurityIncident.resolved_at < cutoff_date
                ).delete()
                
                session.commit()
                session.close()
                
                if deleted_actions > 0 or deleted_incidents > 0:
                    logger.info(f"Cleaned up {deleted_actions} actions and {deleted_incidents} incidents")
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error cleaning up old responses: {e}")
                await asyncio.sleep(3600)
    
    async def get_response_status(self) -> Dict[str, Any]:
        """Get current response automation status."""
        try:
            session = self.Session()
            
            # Get action counts by status
            action_counts = {}
            for status in ActionStatus:
                count = session.query(ResponseAction).filter(
                    ResponseAction.status == status.value,
                    ResponseAction.started_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                action_counts[status.value] = count
            
            # Get incident counts by status
            incident_counts = {}
            for status in IncidentStatus:
                count = session.query(SecurityIncident).filter(
                    SecurityIncident.status == status.value,
                    SecurityIncident.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                incident_counts[status.value] = count
            
            session.close()
            
            status = {
                'running': self.running,
                'active_responses': len(self.active_responses),
                'queued_actions': self.response_queue.qsize(),
                'playbooks_loaded': len(self.playbooks),
                'action_counts_24h': action_counts,
                'incident_counts_24h': incident_counts,
                'last_update': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting response status: {e}")
            return {'error': str(e)}


# Factory function
def create_response_automation(redis_url: str = "redis://localhost:6379",
                             db_url: str = "sqlite:///security_responses.db") -> ResponseAutomation:
    """Create a response automation instance."""
    return ResponseAutomation(redis_url, db_url)


if __name__ == "__main__":
    async def main():
        automation = create_response_automation()
        await automation.start()
    
    asyncio.run(main())