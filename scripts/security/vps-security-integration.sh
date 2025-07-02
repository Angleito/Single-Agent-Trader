#!/bin/bash
# VPS Security Integration Script for Digital Ocean
# Integrates host security tools with Docker container security
# Designed for AI Trading Bot infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECURITY_CONFIG_DIR="$PROJECT_ROOT/security"
INTEGRATION_CONFIG_DIR="$SECURITY_CONFIG_DIR/integration"
LOG_DIR="/var/log/ai-trading-bot-security"
ALERT_CONFIG="$INTEGRATION_CONFIG_DIR/alert-config.json"

# Digital Ocean configuration
DO_VPC_ID=${DO_VPC_ID:-}
DO_FIREWALL_ID=${DO_FIREWALL_ID:-}
DO_SPACES_ENDPOINT=${DO_SPACES_ENDPOINT:-"fra1.digitaloceanspaces.com"}
DO_SPACES_BUCKET=${DO_SPACES_BUCKET:-"ai-trading-bot-security"}

# Service configuration
FAIL2BAN_CONFIG="/etc/fail2ban/jail.d/ai-trading-bot.conf"
UFW_RULES_FILE="$INTEGRATION_CONFIG_DIR/ufw-rules.conf"
DOCKER_SECURITY_CONFIG="/etc/docker/daemon-security.json"

# Functions
print_header() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${BLUE}  $1"
    echo -e "${BLUE}===========================================${NC}\n"
}

print_section() {
    echo -e "\n${PURPLE}--- $1 ---${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Logging function
log_security_event() {
    local level="$1"
    local component="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] [$component] $message" >> "$LOG_DIR/security-integration.log"

    # Send to syslog for centralized logging
    logger -t "ai-trading-bot-security" -p "security.$level" "[$component] $message"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root for system integration"
        print_warning "Run: sudo $0 $*"
        exit 1
    fi
}

# Create necessary directories
setup_directories() {
    print_section "Setting Up Security Directories"

    mkdir -p "$INTEGRATION_CONFIG_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$SECURITY_CONFIG_DIR/backup"
    mkdir -p "$SECURITY_CONFIG_DIR/monitoring"
    mkdir -p "$SECURITY_CONFIG_DIR/automation"

    # Set proper permissions
    chmod 750 "$LOG_DIR"
    chmod 700 "$SECURITY_CONFIG_DIR"

    print_success "Security directories created"
    log_security_event "info" "setup" "Security directories initialized"
}

# Configure UFW firewall integration with Docker
configure_ufw_docker_integration() {
    print_section "Configuring UFW-Docker Integration"

    # Backup existing UFW configuration
    cp /etc/ufw/before.rules /etc/ufw/before.rules.backup 2>/dev/null || true

    # Create UFW rules for Docker integration
    cat > "$UFW_RULES_FILE" << 'EOF'
# UFW Rules for AI Trading Bot Docker Integration
# These rules ensure UFW controls Docker container network access

# Default Docker bridge network restrictions
-A ufw-user-forward -i docker0 -o docker0 -j ACCEPT
-A ufw-user-forward -i docker0 -o eth0 -j ACCEPT
-A ufw-user-forward -i eth0 -o docker0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# AI Trading Bot specific network rules
-A ufw-user-forward -i br-trading-network -o br-trading-network -j ACCEPT
-A ufw-user-forward -i br-trading-network -o eth0 -j ACCEPT
-A ufw-user-forward -i eth0 -o br-trading-network -m state --state RELATED,ESTABLISHED -j ACCEPT

# Block inter-container communication except for allowed networks
-A ufw-user-forward -i docker0 -o docker0 -j DROP
EOF

    # Configure UFW after.rules for Docker
    cat >> /etc/ufw/after.rules << 'EOF'

# AI Trading Bot Docker Integration Rules
*filter
:ufw-user-forward - [0:0]
:DOCKER-USER - [0:0]

# Allow established connections
-A DOCKER-USER -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

# Allow internal Docker networks
-A DOCKER-USER -i docker0 -j ACCEPT
-A DOCKER-USER -i br-+ -j ACCEPT

# Block external access to Docker daemon
-A DOCKER-USER -p tcp --dport 2375 -j DROP
-A DOCKER-USER -p tcp --dport 2376 -j DROP

# Log dropped packets
-A DOCKER-USER -j LOG --log-prefix "[UFW DOCKER BLOCK] "
-A DOCKER-USER -j DROP

COMMIT
EOF

    # Enable UFW Docker integration
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing

    # Allow SSH (custom port)
    ufw allow 2222/tcp comment 'SSH'

    # Allow HTTPS for API calls
    ufw allow out 443/tcp comment 'HTTPS outbound'

    # Allow specific trading bot ports if needed
    # ufw allow 8080/tcp comment 'Dashboard'

    ufw --force enable

    print_success "UFW-Docker integration configured"
    log_security_event "info" "firewall" "UFW-Docker integration enabled"
}

# Configure fail2ban for container logs
configure_fail2ban_integration() {
    print_section "Configuring Fail2ban Container Integration"

    # Create fail2ban configuration for AI trading bot
    cat > "$FAIL2BAN_CONFIG" << 'EOF'
[DEFAULT]
# Ban time: 1 hour for first offense, increases exponentially
bantime = 3600
findtime = 600
maxretry = 3
backend = auto

[ai-trading-bot-auth]
enabled = true
port = ssh,2222
filter = ai-trading-bot-auth
logpath = /var/log/ai-trading-bot-security/auth.log
maxretry = 3
bantime = 3600

[ai-trading-bot-api]
enabled = true
port = 8080,443
filter = ai-trading-bot-api
logpath = /opt/ai-trading-bot/logs/bot.log
maxretry = 5
bantime = 1800

[docker-container-breach]
enabled = true
port = all
filter = docker-container-breach
logpath = /var/log/docker-security.log
maxretry = 1
bantime = 86400
EOF

    # Create custom fail2ban filters
    mkdir -p /etc/fail2ban/filter.d

    # Filter for trading bot authentication failures
    cat > /etc/fail2ban/filter.d/ai-trading-bot-auth.conf << 'EOF'
[Definition]
failregex = ^.*AI-TRADING-BOT.*Authentication failed.*from <HOST>.*$
            ^.*AI-TRADING-BOT.*Invalid API key.*from <HOST>.*$
            ^.*AI-TRADING-BOT.*Unauthorized access.*from <HOST>.*$
ignoreregex =
EOF

    # Filter for API abuse
    cat > /etc/fail2ban/filter.d/ai-trading-bot-api.conf << 'EOF'
[Definition]
failregex = ^.*ERROR.*Rate limit exceeded.*<HOST>.*$
            ^.*WARNING.*Suspicious API activity.*<HOST>.*$
            ^.*ERROR.*Invalid request from.*<HOST>.*$
ignoreregex =
EOF

    # Filter for container security breaches
    cat > /etc/fail2ban/filter.d/docker-container-breach.conf << 'EOF'
[Definition]
failregex = ^.*SECURITY.*Container escape attempt.*<HOST>.*$
            ^.*SECURITY.*Unauthorized container access.*<HOST>.*$
            ^.*SECURITY.*Privilege escalation.*<HOST>.*$
ignoreregex =
EOF

    # Enable and restart fail2ban
    systemctl enable fail2ban
    systemctl restart fail2ban

    print_success "Fail2ban container integration configured"
    log_security_event "info" "fail2ban" "Container log monitoring enabled"
}

# Configure Docker daemon security
configure_docker_security() {
    print_section "Configuring Docker Daemon Security"

    # Backup existing Docker daemon configuration
    cp /etc/docker/daemon.json /etc/docker/daemon.json.backup 2>/dev/null || true

    # Enhanced Docker daemon security configuration
    cat > "$DOCKER_SECURITY_CONFIG" << 'EOF'
{
  "icc": false,
  "userland-proxy": false,
  "disable-legacy-registry": true,
  "live-restore": true,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp-profiles/ai-trading-bot.json",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5",
    "labels": "service={{.Name}},image={{.ImageName}}"
  },
  "userns-remap": "dockremap",
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    },
    "nproc": {
      "Hard": 8192,
      "Name": "nproc",
      "Soft": 4096
    }
  },
  "max-concurrent-downloads": 3,
  "max-concurrent-uploads": 5,
  "debug": false,
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "20GB"
    }
  }
}
EOF

    # Create custom seccomp profile for trading bot
    mkdir -p /etc/docker/seccomp-profiles
    cat > /etc/docker/seccomp-profiles/ai-trading-bot.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_X32"
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "access",
        "bind",
        "brk",
        "chmod",
        "chown",
        "close",
        "connect",
        "dup",
        "dup2",
        "execve",
        "exit",
        "exit_group",
        "fcntl",
        "fstat",
        "futex",
        "getcwd",
        "getdents",
        "getpid",
        "getppid",
        "gettid",
        "gettimeofday",
        "listen",
        "lseek",
        "mmap",
        "mprotect",
        "munmap",
        "nanosleep",
        "open",
        "openat",
        "pipe",
        "pipe2",
        "poll",
        "prctl",
        "read",
        "readv",
        "recv",
        "recvfrom",
        "recvmsg",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "select",
        "send",
        "sendmsg",
        "sendto",
        "setsockopt",
        "shutdown",
        "socket",
        "stat",
        "time",
        "wait4",
        "write",
        "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF

    # Create user namespace mapping for Docker
    if ! id dockremap >/dev/null 2>&1; then
        useradd -r -s /bin/false dockremap
    fi

    echo "dockremap:165536:65536" > /etc/subuid
    echo "dockremap:165536:65536" > /etc/subgid

    # Restart Docker daemon
    systemctl daemon-reload
    systemctl restart docker

    print_success "Docker daemon security configured"
    log_security_event "info" "docker" "Enhanced security configuration applied"
}

# Setup system monitoring integration
setup_monitoring_integration() {
    print_section "Setting Up Security Monitoring Integration"

    # Install security monitoring tools
    apt-get update
    apt-get install -y auditd aide rkhunter osquery

    # Configure auditd for container monitoring
    cat >> /etc/audit/rules.d/ai-trading-bot.rules << 'EOF'
# AI Trading Bot Security Audit Rules
-w /var/lib/docker/ -p wa -k docker_files
-w /etc/docker/ -p wa -k docker_config
-w /opt/ai-trading-bot/ -p wa -k trading_bot_files
-w /home/*/ai-trading-bot/ -p wa -k trading_bot_user_files
-a always,exit -F arch=b64 -S socket -F success=1 -k network_socket
-a always,exit -F arch=b64 -S connect -F success=1 -k network_connect
-a always,exit -F arch=b64 -S execve -F path=/usr/bin/docker -k docker_exec
EOF

    # Configure AIDE for file integrity monitoring
    cat > /etc/aide/aide.conf.d/ai-trading-bot << 'EOF'
# AI Trading Bot AIDE Configuration
/opt/ai-trading-bot NORMAL
/etc/docker NORMAL
/var/lib/docker/containers DATAONLY
!/var/lib/docker/containers/*/logs
!/var/lib/docker/overlay2
!/var/lib/docker/tmp
EOF

    # Initialize AIDE database
    aideinit --yes --force

    # Setup osquery for security monitoring
    cat > /etc/osquery/osquery.conf << 'EOF'
{
  "options": {
    "config_plugin": "filesystem",
    "logger_plugin": "filesystem",
    "logger_path": "/var/log/osquery",
    "disable_logging": "false",
    "log_result_events": "true",
    "schedule_splay_percent": "10",
    "pidfile": "/var/osquery/osquery.pidfile",
    "events_expiry": "3600",
    "database_path": "/var/osquery/osquery.db",
    "verbose": "false",
    "worker_threads": "2",
    "enable_monitor": "true"
  },
  "schedule": {
    "docker_containers": {
      "query": "SELECT * FROM docker_containers;",
      "interval": 60
    },
    "docker_networks": {
      "query": "SELECT * FROM docker_networks;",
      "interval": 300
    },
    "processes": {
      "query": "SELECT name, pid, parent, path, cmdline FROM processes WHERE name IN ('python', 'docker', 'dockerd');",
      "interval": 60
    },
    "network_connections": {
      "query": "SELECT pid, fd, socket, family, protocol, local_address, remote_address, local_port, remote_port FROM process_open_sockets WHERE remote_port IN (443, 80, 8080, 9000);",
      "interval": 60
    },
    "file_changes": {
      "query": "SELECT * FROM file_events WHERE path LIKE '/opt/ai-trading-bot/%' OR path LIKE '/etc/docker/%';",
      "interval": 60
    }
  }
}
EOF

    # Start monitoring services
    systemctl enable auditd aide osqueryd
    systemctl restart auditd osqueryd

    print_success "Security monitoring integration configured"
    log_security_event "info" "monitoring" "Security monitoring services enabled"
}

# Setup Digital Ocean cloud firewall automation
setup_digitalocean_firewall() {
    print_section "Setting Up Digital Ocean Cloud Firewall"

    # Check if doctl is installed
    if ! command -v doctl >/dev/null 2>&1; then
        print_warning "doctl not installed. Installing..."
        wget -O - https://github.com/digitalocean/doctl/releases/download/v1.90.0/doctl-1.90.0-linux-amd64.tar.gz | tar xz
        mv doctl /usr/local/bin/
        chmod +x /usr/local/bin/doctl
    fi

    # Create firewall automation script
    cat > "$SECURITY_CONFIG_DIR/automation/digitalocean-firewall.sh" << 'EOF'
#!/bin/bash
# Digital Ocean Firewall Automation for AI Trading Bot

set -euo pipefail

# Configuration
DO_TOKEN=${DO_TOKEN:-}
FIREWALL_NAME="ai-trading-bot-security"
DROPLET_TAG="ai-trading-bot"

# Authenticate with Digital Ocean
if [[ -n "$DO_TOKEN" ]]; then
    doctl auth init -t "$DO_TOKEN"
else
    echo "Error: DO_TOKEN environment variable not set"
    exit 1
fi

# Create or update firewall rules
create_firewall_rules() {
    local firewall_id=$(doctl compute firewall list --format ID,Name | grep "$FIREWALL_NAME" | cut -d' ' -f1)

    if [[ -z "$firewall_id" ]]; then
        # Create new firewall
        doctl compute firewall create \
            --name "$FIREWALL_NAME" \
            --inbound-rules "protocol:tcp,ports:2222,sources:addresses:0.0.0.0/0,::0/0" \
            --inbound-rules "protocol:tcp,ports:443,sources:addresses:0.0.0.0/0,::0/0" \
            --outbound-rules "protocol:tcp,ports:443,destinations:addresses:0.0.0.0/0,::0/0" \
            --outbound-rules "protocol:tcp,ports:80,destinations:addresses:0.0.0.0/0,::0/0" \
            --outbound-rules "protocol:udp,ports:53,destinations:addresses:0.0.0.0/0,::0/0" \
            --tag-names "$DROPLET_TAG"

        echo "Created new firewall: $FIREWALL_NAME"
    else
        echo "Firewall already exists: $firewall_id"
    fi
}

# Update firewall based on threat intelligence
update_threat_protection() {
    # Get current threat intelligence (this would integrate with threat feeds)
    # For now, we'll block common attack sources
    local malicious_ips=(
        "185.220.100.0/24"  # Tor exit nodes example
        "198.98.51.0/24"    # Known scanner networks
    )

    for ip in "${malicious_ips[@]}"; do
        # Block malicious IPs
        doctl compute firewall add-rules "$firewall_id" \
            --inbound-rules "protocol:tcp,ports:all,sources:addresses:$ip" \
            --priority 1 \
            --action deny || true
    done
}

# Main execution
main() {
    create_firewall_rules
    update_threat_protection

    echo "Digital Ocean firewall automation complete"
}

main "$@"
EOF

    chmod +x "$SECURITY_CONFIG_DIR/automation/digitalocean-firewall.sh"

    print_success "Digital Ocean firewall automation configured"
    log_security_event "info" "cloud-firewall" "Digital Ocean firewall automation setup"
}

# Setup security alert correlation
setup_alert_correlation() {
    print_section "Setting Up Security Alert Correlation"

    # Create alert configuration
    cat > "$ALERT_CONFIG" << 'EOF'
{
  "alert_sources": [
    {
      "name": "fail2ban",
      "log_path": "/var/log/fail2ban.log",
      "patterns": [
        "Ban ",
        "Found ",
        "WARNING"
      ],
      "severity": "medium"
    },
    {
      "name": "ufw",
      "log_path": "/var/log/ufw.log",
      "patterns": [
        "BLOCK",
        "DENY"
      ],
      "severity": "low"
    },
    {
      "name": "docker",
      "log_path": "/var/log/docker-security.log",
      "patterns": [
        "SECURITY",
        "BREACH",
        "UNAUTHORIZED"
      ],
      "severity": "high"
    },
    {
      "name": "auditd",
      "log_path": "/var/log/audit/audit.log",
      "patterns": [
        "AVC",
        "SYSCALL",
        "EXECVE"
      ],
      "severity": "medium"
    }
  ],
  "notification_channels": [
    {
      "name": "webhook",
      "type": "webhook",
      "url": "${SECURITY_WEBHOOK_URL:-}",
      "enabled": true
    },
    {
      "name": "email",
      "type": "email",
      "smtp_server": "${SMTP_SERVER:-}",
      "enabled": false
    }
  ],
  "correlation_rules": [
    {
      "name": "coordinated_attack",
      "conditions": [
        "fail2ban.severity >= medium",
        "ufw.count >= 10",
        "timeframe <= 300"
      ],
      "action": "immediate_alert"
    },
    {
      "name": "container_breach",
      "conditions": [
        "docker.severity == high"
      ],
      "action": "emergency_response"
    }
  ]
}
EOF

    # Create alert correlation engine
    cat > "$SECURITY_CONFIG_DIR/automation/alert-correlator.py" << 'EOF'
#!/usr/bin/env python3
"""
Security Alert Correlation Engine for AI Trading Bot
Correlates security events from multiple sources and triggers appropriate responses
"""

import json
import re
import time
import logging
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

class SecurityAlertCorrelator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.events = defaultdict(list)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/ai-trading-bot-security/correlator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SecurityCorrelator')

    def monitor_logs(self):
        """Monitor security logs for events"""
        for source in self.config['alert_sources']:
            try:
                log_path = Path(source['log_path'])
                if not log_path.exists():
                    continue

                # Tail log file (simplified implementation)
                with open(log_path, 'r') as f:
                    f.seek(0, 2)  # Go to end of file
                    while True:
                        line = f.readline()
                        if not line:
                            time.sleep(1)
                            continue

                        self.process_log_line(source, line.strip())

            except Exception as e:
                self.logger.error(f"Error monitoring {source['name']}: {e}")

    def process_log_line(self, source, line):
        """Process a single log line for security events"""
        for pattern in source['patterns']:
            if re.search(pattern, line, re.IGNORECASE):
                event = {
                    'source': source['name'],
                    'severity': source['severity'],
                    'pattern': pattern,
                    'line': line,
                    'timestamp': datetime.now().isoformat()
                }

                self.events[source['name']].append(event)
                self.logger.info(f"Security event detected: {source['name']} - {pattern}")

                # Check correlation rules
                self.check_correlation_rules()
                break

    def check_correlation_rules(self):
        """Check if events match correlation rules"""
        for rule in self.config['correlation_rules']:
            if self.evaluate_rule(rule):
                self.trigger_response(rule)

    def evaluate_rule(self, rule):
        """Evaluate if a correlation rule is triggered"""
        # Simplified rule evaluation
        # In production, this would be more sophisticated
        return len(self.events) > 0

    def trigger_response(self, rule):
        """Trigger appropriate security response"""
        self.logger.warning(f"Correlation rule triggered: {rule['name']}")

        if rule['action'] == 'immediate_alert':
            self.send_alert(f"Security correlation detected: {rule['name']}", 'warning')
        elif rule['action'] == 'emergency_response':
            self.send_alert(f"EMERGENCY: {rule['name']}", 'critical')
            self.emergency_response()

    def send_alert(self, message, severity):
        """Send security alert to configured channels"""
        for channel in self.config['notification_channels']:
            if not channel['enabled']:
                continue

            try:
                if channel['type'] == 'webhook':
                    self.send_webhook_alert(channel, message, severity)
                elif channel['type'] == 'email':
                    self.send_email_alert(channel, message, severity)

            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel['name']}: {e}")

    def send_webhook_alert(self, channel, message, severity):
        """Send alert via webhook"""
        if not channel.get('url'):
            return

        payload = {
            'text': f"🚨 AI Trading Bot Security Alert [{severity.upper()}]: {message}",
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'host': 'trading-bot-vps'
        }

        requests.post(channel['url'], json=payload, timeout=10)

    def emergency_response(self):
        """Execute emergency security response"""
        self.logger.critical("EXECUTING EMERGENCY SECURITY RESPONSE")

        # Lock down system (example actions)
        # In production, this might include:
        # - Blocking all external access
        # - Stopping containers
        # - Creating security snapshots
        # - Notifying administrators

        pass

if __name__ == '__main__':
    correlator = SecurityAlertCorrelator('/opt/ai-trading-bot/security/integration/alert-config.json')
    correlator.monitor_logs()
EOF

    chmod +x "$SECURITY_CONFIG_DIR/automation/alert-correlator.py"

    print_success "Security alert correlation configured"
    log_security_event "info" "alerts" "Alert correlation system initialized"
}

# Setup backup and disaster recovery integration
setup_backup_integration() {
    print_section "Setting Up Security Backup Integration"

    # Create security backup script
    cat > "$SECURITY_CONFIG_DIR/automation/security-backup.sh" << 'EOF'
#!/bin/bash
# Security Configuration Backup Script
# Backs up all security configurations and logs to Digital Ocean Spaces

set -euo pipefail

# Configuration
BACKUP_DIR="/tmp/security-backup-$(date +%Y%m%d-%H%M%S)"
SECURITY_CONFIG_DIR="/opt/ai-trading-bot/security"
LOG_DIR="/var/log/ai-trading-bot-security"

# Digital Ocean Spaces configuration
DO_SPACES_KEY=${DO_SPACES_KEY:-}
DO_SPACES_SECRET=${DO_SPACES_SECRET:-}
DO_SPACES_ENDPOINT=${DO_SPACES_ENDPOINT:-"fra1.digitaloceanspaces.com"}
DO_SPACES_BUCKET=${DO_SPACES_BUCKET:-"ai-trading-bot-security"}

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup security configurations
echo "Backing up security configurations..."
cp -r "$SECURITY_CONFIG_DIR" "$BACKUP_DIR/"
cp -r /etc/fail2ban/jail.d/ai-trading-bot.conf "$BACKUP_DIR/" 2>/dev/null || true
cp -r /etc/ufw/user*.rules "$BACKUP_DIR/" 2>/dev/null || true
cp -r /etc/docker/daemon-security.json "$BACKUP_DIR/" 2>/dev/null || true

# Backup security logs (last 7 days)
echo "Backing up security logs..."
find "$LOG_DIR" -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || true
find /var/log -name "*fail2ban*" -mtime -7 -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || true
find /var/log -name "*ufw*" -mtime -7 -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || true

# Create archive
cd "$(dirname "$BACKUP_DIR")"
tar czf "$(basename "$BACKUP_DIR").tar.gz" "$(basename "$BACKUP_DIR")"

# Upload to Digital Ocean Spaces
if [[ -n "$DO_SPACES_KEY" && -n "$DO_SPACES_SECRET" ]]; then
    echo "Uploading backup to Digital Ocean Spaces..."

    # Configure AWS CLI for Digital Ocean Spaces
    aws configure set aws_access_key_id "$DO_SPACES_KEY"
    aws configure set aws_secret_access_key "$DO_SPACES_SECRET"
    aws configure set default.region us-east-1

    # Upload backup
    aws s3 cp "$(basename "$BACKUP_DIR").tar.gz" \
        "s3://$DO_SPACES_BUCKET/security-backups/" \
        --endpoint-url "https://$DO_SPACES_ENDPOINT"

    echo "Backup uploaded successfully"
else
    echo "Digital Ocean Spaces credentials not configured - backup saved locally"
fi

# Cleanup
rm -rf "$BACKUP_DIR"
echo "Security backup completed: $(basename "$BACKUP_DIR").tar.gz"
EOF

    chmod +x "$SECURITY_CONFIG_DIR/automation/security-backup.sh"

    # Add to crontab for regular backups
    (crontab -l 2>/dev/null; echo "0 2 * * * $SECURITY_CONFIG_DIR/automation/security-backup.sh >> $LOG_DIR/backup.log 2>&1") | crontab -

    print_success "Security backup integration configured"
    log_security_event "info" "backup" "Security backup automation enabled"
}

# Setup incident response automation
setup_incident_response() {
    print_section "Setting Up Incident Response Automation"

    # Create incident response playbook
    cat > "$SECURITY_CONFIG_DIR/automation/incident-response.sh" << 'EOF'
#!/bin/bash
# Automated Incident Response for AI Trading Bot
# Executes pre-defined security responses based on incident type

set -euo pipefail

INCIDENT_TYPE="$1"
SEVERITY="$2"
DETAILS="${3:-}"

# Logging
LOG_DIR="/var/log/ai-trading-bot-security"
INCIDENT_LOG="$LOG_DIR/incidents.log"

log_incident() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INCIDENT-$SEVERITY] [$INCIDENT_TYPE] $1" | tee -a "$INCIDENT_LOG"
}

# Incident response actions
isolate_container() {
    log_incident "Isolating compromised containers"

    # Stop all trading bot containers
    docker compose -f /opt/ai-trading-bot/docker-compose.yml down || true

    # Create network isolation
    docker network create --internal incident-quarantine || true

    log_incident "Container isolation complete"
}

create_forensic_snapshot() {
    log_incident "Creating forensic snapshot"

    # Create memory dump
    mkdir -p /opt/forensics/$(date +%Y%m%d-%H%M%S)

    # Dump container state
    docker ps -a > /opt/forensics/$(date +%Y%m%d-%H%M%S)/containers.txt
    docker logs ai-trading-bot > /opt/forensics/$(date +%Y%m%d-%H%M%S)/bot-logs.txt 2>&1 || true

    # Copy security logs
    cp -r "$LOG_DIR" /opt/forensics/$(date +%Y%m%d-%H%M%S)/security-logs/

    log_incident "Forensic snapshot created"
}

emergency_shutdown() {
    log_incident "EXECUTING EMERGENCY SHUTDOWN"

    # Stop all containers
    docker stop $(docker ps -q) 2>/dev/null || true

    # Block all network traffic except SSH
    ufw --force reset
    ufw default deny incoming
    ufw default deny outgoing
    ufw allow 2222/tcp
    ufw --force enable

    # Send emergency notification
    curl -X POST "${EMERGENCY_WEBHOOK_URL:-}" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"🚨 EMERGENCY SHUTDOWN: AI Trading Bot system has been locked down due to security incident\",\"severity\":\"critical\"}" \
        2>/dev/null || true

    log_incident "Emergency shutdown complete"
}

# Main incident response logic
case "$INCIDENT_TYPE" in
    "container_breach")
        log_incident "Container breach detected - executing containment"
        isolate_container
        create_forensic_snapshot
        if [[ "$SEVERITY" == "critical" ]]; then
            emergency_shutdown
        fi
        ;;

    "coordinated_attack")
        log_incident "Coordinated attack detected - implementing defense"
        # Enhanced firewall rules
        ufw insert 1 deny from any to any
        ufw allow from $(curl -s ipinfo.io/ip)/32 to any port 2222
        create_forensic_snapshot
        ;;

    "data_exfiltration")
        log_incident "Data exfiltration attempt detected"
        isolate_container
        create_forensic_snapshot
        emergency_shutdown
        ;;

    *)
        log_incident "Unknown incident type: $INCIDENT_TYPE"
        create_forensic_snapshot
        ;;
esac

log_incident "Incident response completed for $INCIDENT_TYPE"
EOF

    chmod +x "$SECURITY_CONFIG_DIR/automation/incident-response.sh"

    print_success "Incident response automation configured"
    log_security_event "info" "incident-response" "Automated incident response system enabled"
}

# Create security integration health check
create_health_check() {
    print_section "Creating Security Integration Health Check"

    cat > "$SECURITY_CONFIG_DIR/monitoring/security-health-check.sh" << 'EOF'
#!/bin/bash
# Security Integration Health Check
# Validates all security components are functioning properly

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

HEALTH_STATUS=0

check_service() {
    local service="$1"
    local description="$2"

    if systemctl is-active --quiet "$service"; then
        echo -e "${GREEN}✓${NC} $description ($service)"
        return 0
    else
        echo -e "${RED}✗${NC} $description ($service) - FAILED"
        return 1
    fi
}

check_file() {
    local file="$1"
    local description="$2"

    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description - FILE MISSING"
        return 1
    fi
}

echo "=== AI Trading Bot Security Integration Health Check ==="
echo

# Check core security services
echo "Core Security Services:"
check_service "fail2ban" "Fail2ban intrusion prevention" || ((HEALTH_STATUS++))
check_service "ufw" "UFW firewall" || ((HEALTH_STATUS++))
check_service "auditd" "Audit daemon" || ((HEALTH_STATUS++))
check_service "docker" "Docker daemon" || ((HEALTH_STATUS++))
echo

# Check monitoring services
echo "Monitoring Services:"
check_service "osqueryd" "OSQuery monitoring" || ((HEALTH_STATUS++))
echo

# Check configuration files
echo "Configuration Files:"
check_file "/etc/fail2ban/jail.d/ai-trading-bot.conf" "Fail2ban configuration" || ((HEALTH_STATUS++))
check_file "/etc/docker/daemon-security.json" "Docker security configuration" || ((HEALTH_STATUS++))
check_file "/opt/ai-trading-bot/security/integration/alert-config.json" "Alert configuration" || ((HEALTH_STATUS++))
echo

# Check log directories
echo "Log Directories:"
if [[ -d "/var/log/ai-trading-bot-security" && -w "/var/log/ai-trading-bot-security" ]]; then
    echo -e "${GREEN}✓${NC} Security log directory"
else
    echo -e "${RED}✗${NC} Security log directory - NOT ACCESSIBLE"
    ((HEALTH_STATUS++))
fi

# Check network security
echo
echo "Network Security:"
if ufw status | grep -q "Status: active"; then
    echo -e "${GREEN}✓${NC} UFW firewall active"
else
    echo -e "${RED}✗${NC} UFW firewall inactive"
    ((HEALTH_STATUS++))
fi

# Check container security
if docker info --format '{{.SecurityOptions}}' | grep -q "seccomp"; then
    echo -e "${GREEN}✓${NC} Docker seccomp profiles enabled"
else
    echo -e "${YELLOW}⚠${NC} Docker seccomp profiles not enabled"
fi

echo
echo "=== Health Check Summary ==="
if [[ $HEALTH_STATUS -eq 0 ]]; then
    echo -e "${GREEN}✓ All security components are healthy${NC}"
    exit 0
else
    echo -e "${RED}✗ $HEALTH_STATUS security components need attention${NC}"
    exit 1
fi
EOF

    chmod +x "$SECURITY_CONFIG_DIR/monitoring/security-health-check.sh"

    # Add health check to cron
    (crontab -l 2>/dev/null; echo "*/15 * * * * $SECURITY_CONFIG_DIR/monitoring/security-health-check.sh >> $LOG_DIR/health-check.log 2>&1") | crontab -

    print_success "Security health check configured"
}

# Main integration setup function
main() {
    print_header "VPS Security Integration for AI Trading Bot"

    # Check prerequisites
    check_root

    # Run integration steps
    setup_directories
    configure_ufw_docker_integration
    configure_fail2ban_integration
    configure_docker_security
    setup_monitoring_integration
    setup_digitalocean_firewall
    setup_alert_correlation
    setup_backup_integration
    setup_incident_response
    create_health_check

    print_header "Security Integration Complete!"
    print_success "VPS security integration has been successfully configured"

    echo -e "\n${BLUE}Next Steps:${NC}"
    echo "1. Configure Digital Ocean API token: export DO_TOKEN=your_token"
    echo "2. Set up security webhook: export SECURITY_WEBHOOK_URL=your_webhook"
    echo "3. Configure backup credentials: export DO_SPACES_KEY and DO_SPACES_SECRET"
    echo "4. Test the integration: $SECURITY_CONFIG_DIR/monitoring/security-health-check.sh"
    echo "5. Review logs: tail -f $LOG_DIR/security-integration.log"

    echo -e "\n${YELLOW}Important Files:${NC}"
    echo "- Security config: $SECURITY_CONFIG_DIR/"
    echo "- Alert config: $ALERT_CONFIG"
    echo "- Security logs: $LOG_DIR/"
    echo "- Health check: $SECURITY_CONFIG_DIR/monitoring/security-health-check.sh"

    log_security_event "info" "integration" "VPS security integration setup completed successfully"
}

# Parse command line arguments
case "${1:-install}" in
    "install")
        main
        ;;
    "health-check")
        if [[ -f "$SECURITY_CONFIG_DIR/monitoring/security-health-check.sh" ]]; then
            "$SECURITY_CONFIG_DIR/monitoring/security-health-check.sh"
        else
            print_error "Health check script not found. Run installation first."
            exit 1
        fi
        ;;
    "backup")
        if [[ -f "$SECURITY_CONFIG_DIR/automation/security-backup.sh" ]]; then
            "$SECURITY_CONFIG_DIR/automation/security-backup.sh"
        else
            print_error "Backup script not found. Run installation first."
            exit 1
        fi
        ;;
    "help"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  install      Install and configure VPS security integration (default)"
        echo "  health-check Run security integration health check"
        echo "  backup       Run security configuration backup"
        echo "  help         Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
