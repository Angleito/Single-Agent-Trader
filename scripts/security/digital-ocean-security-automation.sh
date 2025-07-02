#!/bin/bash
# Digital Ocean Security Automation
# Automates security features specific to Digital Ocean infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECURITY_DIR="$PROJECT_ROOT/security"
LOG_DIR="/var/log/ai-trading-bot-security"

# Digital Ocean configuration
DO_TOKEN=${DO_TOKEN:-}
DO_REGION=${DO_REGION:-"fra1"}
DO_VPC_NAME=${DO_VPC_NAME:-"ai-trading-bot-vpc"}
DO_FIREWALL_NAME=${DO_FIREWALL_NAME:-"ai-trading-bot-firewall"}
DO_LOAD_BALANCER_NAME=${DO_LOAD_BALANCER_NAME:-"ai-trading-bot-lb"}
DO_SPACES_BUCKET=${DO_SPACES_BUCKET:-"ai-trading-bot-security"}
DROPLET_TAG="ai-trading-bot"

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
log_operation() {
    local level="$1"
    local component="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] [$component] $message" >> "$LOG_DIR/digitalocean-automation.log"
}

# Check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check if doctl is installed
    if ! command -v doctl >/dev/null 2>&1; then
        print_warning "Installing doctl CLI..."
        
        # Install doctl
        cd /tmp
        wget -O doctl.tar.gz https://github.com/digitalocean/doctl/releases/download/v1.90.0/doctl-1.90.0-linux-amd64.tar.gz
        tar xf doctl.tar.gz
        sudo mv doctl /usr/local/bin/
        rm doctl.tar.gz
        
        print_success "doctl installed"
    fi
    
    # Check Digital Ocean token
    if [[ -z "$DO_TOKEN" ]]; then
        print_error "DO_TOKEN environment variable not set"
        print_warning "Set your Digital Ocean token: export DO_TOKEN=your_token"
        exit 1
    fi
    
    # Authenticate with Digital Ocean
    doctl auth init -t "$DO_TOKEN" >/dev/null 2>&1
    
    # Verify authentication
    if ! doctl account get >/dev/null 2>&1; then
        print_error "Failed to authenticate with Digital Ocean"
        exit 1
    fi
    
    print_success "Digital Ocean authentication successful"
    log_operation "info" "auth" "Digital Ocean authentication verified"
}

# Create VPC for network isolation
create_vpc() {
    print_section "Creating VPC for Network Isolation"
    
    # Check if VPC already exists
    local existing_vpc=$(doctl vpcs list --format Name,ID | grep "$DO_VPC_NAME" | awk '{print $2}')
    
    if [[ -n "$existing_vpc" ]]; then
        print_success "VPC already exists: $DO_VPC_NAME ($existing_vpc)"
        echo "$existing_vpc" > /tmp/vpc_id
        return 0
    fi
    
    # Create VPC
    local vpc_id=$(doctl vpcs create \
        --name "$DO_VPC_NAME" \
        --region "$DO_REGION" \
        --ip-range "10.0.0.0/16" \
        --format ID \
        --no-header)
    
    if [[ -n "$vpc_id" ]]; then
        print_success "VPC created: $DO_VPC_NAME ($vpc_id)"
        echo "$vpc_id" > /tmp/vpc_id
        log_operation "info" "vpc" "VPC created with ID: $vpc_id"
    else
        print_error "Failed to create VPC"
        exit 1
    fi
}

# Configure cloud firewall
configure_cloud_firewall() {
    print_section "Configuring Cloud Firewall"
    
    # Check if firewall already exists
    local existing_firewall=$(doctl compute firewall list --format Name,ID | grep "$DO_FIREWALL_NAME" | awk '{print $2}')
    
    if [[ -n "$existing_firewall" ]]; then
        print_success "Firewall already exists: $DO_FIREWALL_NAME ($existing_firewall)"
        echo "$existing_firewall" > /tmp/firewall_id
        return 0
    fi
    
    # Create firewall with comprehensive rules
    local firewall_id=$(doctl compute firewall create \
        --name "$DO_FIREWALL_NAME" \
        --inbound-rules "protocol:tcp,ports:2222,sources:addresses:0.0.0.0/0,::0/0" \
        --inbound-rules "protocol:tcp,ports:443,sources:addresses:0.0.0.0/0,::0/0" \
        --inbound-rules "protocol:tcp,ports:80,sources:addresses:0.0.0.0/0,::0/0" \
        --inbound-rules "protocol:icmp,sources:addresses:0.0.0.0/0,::0/0" \
        --outbound-rules "protocol:tcp,ports:443,destinations:addresses:0.0.0.0/0,::0/0" \
        --outbound-rules "protocol:tcp,ports:80,destinations:addresses:0.0.0.0/0,::0/0" \
        --outbound-rules "protocol:tcp,ports:53,destinations:addresses:0.0.0.0/0,::0/0" \
        --outbound-rules "protocol:udp,ports:53,destinations:addresses:0.0.0.0/0,::0/0" \
        --outbound-rules "protocol:tcp,ports:22,destinations:addresses:0.0.0.0/0,::0/0" \
        --outbound-rules "protocol:icmp,destinations:addresses:0.0.0.0/0,::0/0" \
        --format ID \
        --no-header)
    
    if [[ -n "$firewall_id" ]]; then
        print_success "Cloud firewall created: $DO_FIREWALL_NAME ($firewall_id)"
        echo "$firewall_id" > /tmp/firewall_id
        log_operation "info" "firewall" "Cloud firewall created with ID: $firewall_id"
    else
        print_error "Failed to create cloud firewall"
        exit 1
    fi
}

# Setup load balancer with SSL termination
setup_load_balancer() {
    print_section "Setting Up Load Balancer with SSL"
    
    # Check if load balancer already exists
    local existing_lb=$(doctl compute load-balancer list --format Name,ID | grep "$DO_LOAD_BALANCER_NAME" | awk '{print $2}')
    
    if [[ -n "$existing_lb" ]]; then
        print_success "Load balancer already exists: $DO_LOAD_BALANCER_NAME ($existing_lb)"
        return 0
    fi
    
    # Get VPC ID
    local vpc_id
    if [[ -f "/tmp/vpc_id" ]]; then
        vpc_id=$(cat /tmp/vpc_id)
    else
        vpc_id=$(doctl vpcs list --format Name,ID | grep "$DO_VPC_NAME" | awk '{print $2}')
    fi
    
    # Create load balancer configuration
    cat > /tmp/lb-config.json << EOF
{
  "name": "$DO_LOAD_BALANCER_NAME",
  "algorithm": "round_robin",
  "status": "active",
  "region": {
    "name": "$DO_REGION"
  },
  "vpc_uuid": "$vpc_id",
  "tag": "$DROPLET_TAG",
  "health_check": {
    "protocol": "http",
    "port": 8080,
    "path": "/health",
    "check_interval_seconds": 10,
    "response_timeout_seconds": 5,
    "healthy_threshold": 3,
    "unhealthy_threshold": 3
  },
  "sticky_sessions": {
    "type": "cookies",
    "cookie_name": "lb-cookie",
    "cookie_ttl_seconds": 300
  },
  "forwarding_rules": [
    {
      "entry_protocol": "https",
      "entry_port": 443,
      "target_protocol": "http",
      "target_port": 8080,
      "certificate_id": "",
      "tls_passthrough": false
    },
    {
      "entry_protocol": "http",
      "entry_port": 80,
      "target_protocol": "http",
      "target_port": 8080,
      "tls_passthrough": false
    }
  ]
}
EOF

    # Create load balancer
    local lb_id=$(doctl compute load-balancer create \
        --name "$DO_LOAD_BALANCER_NAME" \
        --algorithm round_robin \
        --forwarding-rules "entry_protocol:https,entry_port:443,target_protocol:http,target_port:8080,tls_passthrough:false" \
        --forwarding-rules "entry_protocol:http,entry_port:80,target_protocol:http,target_port:8080" \
        --health-check "protocol:http,port:8080,path:/health,check_interval_seconds:10,response_timeout_seconds:5,healthy_threshold:3,unhealthy_threshold:3" \
        --region "$DO_REGION" \
        --vpc-uuid "$vpc_id" \
        --tag "$DROPLET_TAG" \
        --format ID \
        --no-header 2>/dev/null || echo "")
    
    if [[ -n "$lb_id" ]]; then
        print_success "Load balancer created: $DO_LOAD_BALANCER_NAME ($lb_id)"
        log_operation "info" "load-balancer" "Load balancer created with ID: $lb_id"
    else
        print_warning "Load balancer creation skipped (may require manual setup for SSL certificates)"
    fi
    
    rm -f /tmp/lb-config.json
}

# Setup Digital Ocean Spaces for secure backup
setup_spaces_backup() {
    print_section "Setting Up Digital Ocean Spaces for Backup"
    
    # Check if Spaces bucket exists
    if doctl compute cdn list --format Origin | grep -q "$DO_SPACES_BUCKET"; then
        print_success "Spaces bucket already configured: $DO_SPACES_BUCKET"
        return 0
    fi
    
    # Install AWS CLI for Spaces interaction
    if ! command -v aws >/dev/null 2>&1; then
        print_warning "Installing AWS CLI for Spaces interaction..."
        
        # Install AWS CLI
        cd /tmp
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip >/dev/null 2>&1
        sudo ./aws/install >/dev/null 2>&1
        rm -rf aws awscliv2.zip
        
        print_success "AWS CLI installed"
    fi
    
    # Create Spaces backup configuration
    cat > "$SECURITY_DIR/spaces-backup-config.sh" << 'EOF'
#!/bin/bash
# Digital Ocean Spaces Backup Configuration

# Set these environment variables:
# export DO_SPACES_ACCESS_KEY=your_spaces_access_key
# export DO_SPACES_SECRET_KEY=your_spaces_secret_key
# export DO_SPACES_ENDPOINT=fra1.digitaloceanspaces.com
# export DO_SPACES_BUCKET=ai-trading-bot-security

configure_spaces_client() {
    if [[ -z "$DO_SPACES_ACCESS_KEY" || -z "$DO_SPACES_SECRET_KEY" ]]; then
        echo "Error: Spaces credentials not configured"
        echo "Set DO_SPACES_ACCESS_KEY and DO_SPACES_SECRET_KEY environment variables"
        return 1
    fi
    
    # Configure AWS CLI for Spaces
    aws configure set aws_access_key_id "$DO_SPACES_ACCESS_KEY"
    aws configure set aws_secret_access_key "$DO_SPACES_SECRET_KEY"
    aws configure set default.region us-east-1
    
    return 0
}

backup_to_spaces() {
    local source_dir="$1"
    local backup_name="$2"
    local timestamp=$(date +%Y%m%d-%H%M%S)
    
    if ! configure_spaces_client; then
        return 1
    fi
    
    # Create archive
    tar czf "/tmp/${backup_name}-${timestamp}.tar.gz" -C "$(dirname "$source_dir")" "$(basename "$source_dir")"
    
    # Upload to Spaces
    aws s3 cp "/tmp/${backup_name}-${timestamp}.tar.gz" \
        "s3://$DO_SPACES_BUCKET/backups/${backup_name}/" \
        --endpoint-url "https://$DO_SPACES_ENDPOINT"
    
    # Cleanup local archive
    rm -f "/tmp/${backup_name}-${timestamp}.tar.gz"
    
    echo "Backup completed: ${backup_name}-${timestamp}.tar.gz"
}

# Backup retention policy (keep last 30 days)
cleanup_old_backups() {
    if ! configure_spaces_client; then
        return 1
    fi
    
    local cutoff_date=$(date -d "30 days ago" +%Y%m%d)
    
    # List and delete old backups
    aws s3 ls "s3://$DO_SPACES_BUCKET/backups/" --recursive --endpoint-url "https://$DO_SPACES_ENDPOINT" | \
    while read -r line; do
        local file_date=$(echo "$line" | awk '{print $1}' | tr -d '-')
        local file_path=$(echo "$line" | awk '{print $4}')
        
        if [[ "$file_date" < "$cutoff_date" ]]; then
            aws s3 rm "s3://$DO_SPACES_BUCKET/$file_path" --endpoint-url "https://$DO_SPACES_ENDPOINT"
            echo "Deleted old backup: $file_path"
        fi
    done
}
EOF

    chmod +x "$SECURITY_DIR/spaces-backup-config.sh"
    
    print_success "Digital Ocean Spaces backup configuration created"
    print_warning "Configure Spaces credentials: DO_SPACES_ACCESS_KEY and DO_SPACES_SECRET_KEY"
    log_operation "info" "spaces" "Spaces backup configuration created"
}

# Setup monitoring and alerting
setup_monitoring() {
    print_section "Setting Up Digital Ocean Monitoring"
    
    # Create monitoring automation script
    cat > "$SECURITY_DIR/do-monitoring.py" << 'EOF'
#!/usr/bin/env python3
"""
Digital Ocean Monitoring Integration
Integrates with DO monitoring API and sends alerts
"""

import json
import requests
import time
import logging
from datetime import datetime, timedelta

class DOMonitoringClient:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://api.digitalocean.com/v2"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/ai-trading-bot-security/do-monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DOMonitoring')
    
    def get_droplet_metrics(self, droplet_id, metric_type='cpu', period='5m'):
        """Get droplet metrics from DO monitoring API"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=30)
        
        url = f"{self.base_url}/monitoring/metrics/droplet/{metric_type}"
        params = {
            'host_id': droplet_id,
            'start': start_time.isoformat() + 'Z',
            'end': end_time.isoformat() + 'Z'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return None
    
    def check_security_metrics(self, droplet_id):
        """Check security-related metrics"""
        security_alerts = []
        
        # Check CPU usage (high CPU might indicate crypto mining or attack)
        cpu_metrics = self.get_droplet_metrics(droplet_id, 'cpu')
        if cpu_metrics and 'data' in cpu_metrics:
            recent_cpu = cpu_metrics['data']['result']
            if recent_cpu:
                latest_cpu = float(recent_cpu[0]['values'][-1][1])
                if latest_cpu > 90.0:  # 90% CPU threshold
                    security_alerts.append({
                        'type': 'high_cpu',
                        'severity': 'warning',
                        'value': latest_cpu,
                        'message': f'High CPU usage detected: {latest_cpu:.1f}%'
                    })
        
        # Check network traffic (unusual patterns might indicate data exfiltration)
        network_out = self.get_droplet_metrics(droplet_id, 'bandwidth')
        if network_out and 'data' in network_out:
            # Analyze network patterns here
            pass
        
        # Check memory usage
        memory_metrics = self.get_droplet_metrics(droplet_id, 'memory')
        if memory_metrics and 'data' in memory_metrics:
            recent_memory = memory_metrics['data']['result']
            if recent_memory:
                latest_memory = float(recent_memory[0]['values'][-1][1])
                if latest_memory > 95.0:  # 95% memory threshold
                    security_alerts.append({
                        'type': 'high_memory',
                        'severity': 'warning',
                        'value': latest_memory,
                        'message': f'High memory usage detected: {latest_memory:.1f}%'
                    })
        
        return security_alerts
    
    def send_alert(self, alert, webhook_url=None):
        """Send security alert"""
        self.logger.warning(f"Security alert: {alert['message']}")
        
        if webhook_url:
            payload = {
                'text': f"🚨 DO Security Alert [{alert['severity'].upper()}]: {alert['message']}",
                'severity': alert['severity'],
                'timestamp': datetime.now().isoformat(),
                'droplet_id': alert.get('droplet_id', 'unknown')
            }
            
            try:
                requests.post(webhook_url, json=payload, timeout=10)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to send webhook alert: {e}")
    
    def monitor_droplets(self, webhook_url=None):
        """Monitor all tagged droplets"""
        try:
            # Get droplets with AI trading bot tag
            url = f"{self.base_url}/droplets"
            params = {'tag_name': 'ai-trading-bot'}
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            droplets = response.json().get('droplets', [])
            
            for droplet in droplets:
                droplet_id = droplet['id']
                droplet_name = droplet['name']
                
                self.logger.info(f"Checking security metrics for droplet: {droplet_name} ({droplet_id})")
                
                alerts = self.check_security_metrics(droplet_id)
                
                for alert in alerts:
                    alert['droplet_id'] = droplet_id
                    alert['droplet_name'] = droplet_name
                    self.send_alert(alert, webhook_url)
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to monitor droplets: {e}")

def main():
    import os
    
    token = os.environ.get('DO_TOKEN')
    webhook_url = os.environ.get('SECURITY_WEBHOOK_URL')
    
    if not token:
        print("Error: DO_TOKEN environment variable not set")
        return 1
    
    client = DOMonitoringClient(token)
    
    # Run monitoring loop
    while True:
        try:
            client.monitor_droplets(webhook_url)
            time.sleep(300)  # Check every 5 minutes
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break
        except Exception as e:
            client.logger.error(f"Monitoring error: {e}")
            time.sleep(60)  # Wait 1 minute before retry

if __name__ == '__main__':
    main()
EOF

    chmod +x "$SECURITY_DIR/do-monitoring.py"
    
    # Create systemd service for monitoring
    cat > /tmp/do-monitoring.service << 'EOF'
[Unit]
Description=Digital Ocean Security Monitoring for AI Trading Bot
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Environment=DO_TOKEN=
Environment=SECURITY_WEBHOOK_URL=
ExecStart=/usr/bin/python3 /opt/ai-trading-bot/security/do-monitoring.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/do-monitoring.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    print_success "Digital Ocean monitoring configured"
    print_warning "Enable monitoring service: sudo systemctl enable do-monitoring.service"
    log_operation "info" "monitoring" "DO monitoring service configured"
}

# Setup automatic threat intelligence updates
setup_threat_intelligence() {
    print_section "Setting Up Threat Intelligence Automation"
    
    # Create threat intelligence updater
    cat > "$SECURITY_DIR/threat-intelligence.sh" << 'EOF'
#!/bin/bash
# Threat Intelligence Automation
# Updates firewall rules based on threat intelligence feeds

set -euo pipefail

# Configuration
THREAT_LIST_DIR="/opt/ai-trading-bot/security/threat-lists"
LOG_FILE="/var/log/ai-trading-bot-security/threat-intelligence.log"

# Create directories
mkdir -p "$THREAT_LIST_DIR"

log_threat() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Download threat intelligence feeds
download_threat_feeds() {
    log_threat "Downloading threat intelligence feeds..."
    
    # Spamhaus DROP list
    curl -s "https://www.spamhaus.org/drop/drop.txt" | grep -E "^[0-9]" > "$THREAT_LIST_DIR/spamhaus-drop.txt" || true
    
    # Emerging Threats compromised IPs
    curl -s "https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt" | grep -E "^[0-9]" > "$THREAT_LIST_DIR/emerging-threats.txt" || true
    
    # TOR exit nodes
    curl -s "https://check.torproject.org/torbulkexitlist" | grep -E "^[0-9]" > "$THREAT_LIST_DIR/tor-exit-nodes.txt" || true
    
    log_threat "Threat feeds downloaded"
}

# Update Digital Ocean firewall with threat intelligence
update_cloud_firewall() {
    if [[ -z "${DO_TOKEN:-}" ]]; then
        log_threat "DO_TOKEN not set, skipping cloud firewall update"
        return 0
    fi
    
    log_threat "Updating Digital Ocean firewall with threat intelligence..."
    
    # Get firewall ID
    local firewall_id=$(doctl compute firewall list --format Name,ID | grep "ai-trading-bot-firewall" | awk '{print $2}')
    
    if [[ -z "$firewall_id" ]]; then
        log_threat "Firewall not found, skipping update"
        return 0
    fi
    
    # Combine all threat lists
    cat "$THREAT_LIST_DIR"/*.txt | sort -u > "$THREAT_LIST_DIR/combined-threats.txt"
    
    # Add deny rules for threat IPs (limited to avoid API limits)
    head -50 "$THREAT_LIST_DIR/combined-threats.txt" | while read -r ip; do
        if [[ -n "$ip" && "$ip" != \#* ]]; then
            # Extract IP/CIDR from line
            threat_ip=$(echo "$ip" | awk '{print $1}')
            
            # Add deny rule (catch errors to continue processing)
            doctl compute firewall add-rules "$firewall_id" \
                --inbound-rules "protocol:tcp,ports:all,sources:addresses:$threat_ip" 2>/dev/null || true
        fi
    done
    
    log_threat "Cloud firewall updated with threat intelligence"
}

# Update local firewall rules
update_local_firewall() {
    log_threat "Updating local firewall with threat intelligence..."
    
    # Backup current UFW rules
    cp /etc/ufw/user.rules /etc/ufw/user.rules.backup 2>/dev/null || true
    
    # Add threat IPs to local firewall (sample - in production, use ipset for performance)
    head -20 "$THREAT_LIST_DIR/combined-threats.txt" | while read -r ip; do
        if [[ -n "$ip" && "$ip" != \#* ]]; then
            threat_ip=$(echo "$ip" | awk '{print $1}')
            ufw insert 1 deny from "$threat_ip" to any 2>/dev/null || true
        fi
    done
    
    log_threat "Local firewall updated with threat intelligence"
}

# Main execution
main() {
    log_threat "Starting threat intelligence update"
    
    download_threat_feeds
    update_cloud_firewall
    update_local_firewall
    
    log_threat "Threat intelligence update completed"
}

main "$@"
EOF

    chmod +x "$SECURITY_DIR/threat-intelligence.sh"
    
    # Add to crontab for regular updates
    (crontab -l 2>/dev/null; echo "0 */6 * * * $SECURITY_DIR/threat-intelligence.sh") | crontab -
    
    print_success "Threat intelligence automation configured"
    log_operation "info" "threat-intel" "Threat intelligence automation enabled"
}

# Create infrastructure as code templates
create_terraform_templates() {
    print_section "Creating Terraform Templates"
    
    mkdir -p "$SECURITY_DIR/terraform"
    
    # Main Terraform configuration
    cat > "$SECURITY_DIR/terraform/main.tf" << 'EOF'
# Digital Ocean Infrastructure for AI Trading Bot
# Security-focused infrastructure deployment

terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

# Variables
variable "do_token" {
  description = "Digital Ocean API Token"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "Digital Ocean region"
  type        = string
  default     = "fra1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "ai-trading-bot"
}

# VPC for network isolation
resource "digitalocean_vpc" "trading_bot_vpc" {
  name     = "${var.project_name}-vpc"
  region   = var.region
  ip_range = "10.0.0.0/16"
  
  timeouts {
    delete = "4m"
  }
}

# Cloud Firewall
resource "digitalocean_firewall" "trading_bot_firewall" {
  name = "${var.project_name}-firewall"
  
  droplet_ids = []
  tags        = [var.project_name]
  
  # SSH access (custom port)
  inbound_rule {
    protocol         = "tcp"
    port_range       = "2222"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # HTTPS
  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # HTTP (redirect to HTTPS)
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # ICMP
  inbound_rule {
    protocol         = "icmp"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # Outbound HTTPS
  outbound_rule {
    protocol              = "tcp"
    port_range            = "443"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # Outbound HTTP
  outbound_rule {
    protocol              = "tcp"
    port_range            = "80"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # DNS
  outbound_rule {
    protocol              = "udp"
    port_range            = "53"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  outbound_rule {
    protocol              = "tcp"
    port_range            = "53"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  
  # ICMP outbound
  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# Spaces bucket for secure backups
resource "digitalocean_spaces_bucket" "security_backup" {
  name   = "${var.project_name}-security"
  region = var.region
  
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["PUT", "POST", "GET"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
  
  lifecycle_rule {
    id      = "security-backup-lifecycle"
    enabled = true
    
    expiration {
      days = 90
    }
    
    noncurrent_version_expiration {
      days = 30
    }
  }
  
  versioning {
    enabled = true
  }
}

# Load Balancer
resource "digitalocean_loadbalancer" "trading_bot_lb" {
  name   = "${var.project_name}-lb"
  region = var.region
  vpc_uuid = digitalocean_vpc.trading_bot_vpc.id
  
  size_unit = 1
  
  forwarding_rule {
    entry_protocol  = "https"
    entry_port      = 443
    target_protocol = "http"
    target_port     = 8080
    tls_passthrough = false
  }
  
  forwarding_rule {
    entry_protocol  = "http"
    entry_port      = 80
    target_protocol = "http"
    target_port     = 8080
  }
  
  healthcheck {
    protocol               = "http"
    port                   = 8080
    path                   = "/health"
    check_interval_seconds = 10
    response_timeout_seconds = 5
    unhealthy_threshold    = 3
    healthy_threshold      = 5
  }
  
  droplet_tag = var.project_name
}

# Project for resource organization
resource "digitalocean_project" "trading_bot_project" {
  name        = var.project_name
  description = "AI Trading Bot Infrastructure"
  purpose     = "Web Application"
  environment = "Production"
  
  resources = [
    digitalocean_vpc.trading_bot_vpc.urn,
    digitalocean_spaces_bucket.security_backup.urn,
    digitalocean_loadbalancer.trading_bot_lb.urn
  ]
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = digitalocean_vpc.trading_bot_vpc.id
}

output "firewall_id" {
  description = "Firewall ID"
  value       = digitalocean_firewall.trading_bot_firewall.id
}

output "load_balancer_ip" {
  description = "Load Balancer IP"
  value       = digitalocean_loadbalancer.trading_bot_lb.ip
}

output "spaces_bucket_name" {
  description = "Spaces bucket name"
  value       = digitalocean_spaces_bucket.security_backup.name
}
EOF

    # Terraform variables file
    cat > "$SECURITY_DIR/terraform/terraform.tfvars.example" << 'EOF'
# Copy this file to terraform.tfvars and fill in your values

do_token = "your_digitalocean_api_token_here"
region = "fra1"
project_name = "ai-trading-bot"
EOF

    # Terraform deployment script
    cat > "$SECURITY_DIR/terraform/deploy.sh" << 'EOF'
#!/bin/bash
# Deploy Digital Ocean infrastructure with Terraform

set -euo pipefail

# Check if terraform.tfvars exists
if [[ ! -f "terraform.tfvars" ]]; then
    echo "Error: terraform.tfvars not found"
    echo "Copy terraform.tfvars.example to terraform.tfvars and configure your values"
    exit 1
fi

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply deployment
read -p "Deploy infrastructure? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply tfplan
    echo "Infrastructure deployed successfully!"
else
    echo "Deployment cancelled"
fi

# Cleanup plan file
rm -f tfplan
EOF

    chmod +x "$SECURITY_DIR/terraform/deploy.sh"
    
    print_success "Terraform templates created"
    log_operation "info" "terraform" "Infrastructure as code templates created"
}

# Main automation setup function
main() {
    print_header "Digital Ocean Security Automation Setup"
    
    # Create directories
    mkdir -p "$SECURITY_DIR"
    mkdir -p "$LOG_DIR"
    
    # Run setup steps
    check_prerequisites
    create_vpc
    configure_cloud_firewall
    setup_load_balancer
    setup_spaces_backup
    setup_monitoring
    setup_threat_intelligence
    create_terraform_templates
    
    print_header "Digital Ocean Security Automation Complete!"
    print_success "All Digital Ocean security features have been configured"
    
    echo -e "\n${BLUE}Configuration Summary:${NC}"
    if [[ -f "/tmp/vpc_id" ]]; then
        echo "- VPC ID: $(cat /tmp/vpc_id)"
    fi
    if [[ -f "/tmp/firewall_id" ]]; then
        echo "- Firewall ID: $(cat /tmp/firewall_id)"
    fi
    echo "- Spaces Bucket: $DO_SPACES_BUCKET"
    echo "- Region: $DO_REGION"
    
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo "1. Configure Spaces credentials for backup:"
    echo "   export DO_SPACES_ACCESS_KEY=your_access_key"
    echo "   export DO_SPACES_SECRET_KEY=your_secret_key"
    echo ""
    echo "2. Enable monitoring service:"
    echo "   sudo systemctl enable do-monitoring.service"
    echo "   sudo systemctl start do-monitoring.service"
    echo ""
    echo "3. Apply droplet tags for automation:"
    echo "   doctl compute droplet tag your_droplet_id --tag-names $DROPLET_TAG"
    echo ""
    echo "4. Set up Terraform for infrastructure management:"
    echo "   cd $SECURITY_DIR/terraform"
    echo "   cp terraform.tfvars.example terraform.tfvars"
    echo "   # Edit terraform.tfvars with your values"
    echo "   ./deploy.sh"
    
    echo -e "\n${YELLOW}Important Files:${NC}"
    echo "- Digital Ocean config: $SECURITY_DIR/"
    echo "- Monitoring script: $SECURITY_DIR/do-monitoring.py"
    echo "- Threat intelligence: $SECURITY_DIR/threat-intelligence.sh"
    echo "- Terraform templates: $SECURITY_DIR/terraform/"
    echo "- Backup config: $SECURITY_DIR/spaces-backup-config.sh"
    echo "- Logs: $LOG_DIR/"
    
    log_operation "info" "setup" "Digital Ocean security automation setup completed successfully"
    
    # Cleanup temp files
    rm -f /tmp/vpc_id /tmp/firewall_id
}

# Parse command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "monitor")
        if [[ -f "$SECURITY_DIR/do-monitoring.py" ]]; then
            python3 "$SECURITY_DIR/do-monitoring.py"
        else
            print_error "Monitoring script not found. Run setup first."
            exit 1
        fi
        ;;
    "threat-update")
        if [[ -f "$SECURITY_DIR/threat-intelligence.sh" ]]; then
            "$SECURITY_DIR/threat-intelligence.sh"
        else
            print_error "Threat intelligence script not found. Run setup first."
            exit 1
        fi
        ;;
    "backup")
        if [[ -f "$SECURITY_DIR/spaces-backup-config.sh" ]]; then
            source "$SECURITY_DIR/spaces-backup-config.sh"
            backup_to_spaces "/opt/ai-trading-bot" "trading-bot-config"
        else
            print_error "Backup script not found. Run setup first."
            exit 1
        fi
        ;;
    "help"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  setup         Set up Digital Ocean security automation (default)"
        echo "  monitor       Start monitoring service"
        echo "  threat-update Update threat intelligence feeds"
        echo "  backup        Run backup to Digital Ocean Spaces"
        echo "  help          Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac