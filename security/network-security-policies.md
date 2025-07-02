# AI Trading Bot Network Security Policies
## Comprehensive Network Security Framework for Digital Ocean VPS Deployment

### Executive Summary

This document outlines comprehensive network security policies for the AI Trading Bot deployed on Digital Ocean VPS. The security framework implements defense-in-depth strategies across multiple layers: Docker network isolation, host firewall configuration, cloud firewall policies, and continuous monitoring.

### Architecture Overview

```
Internet
    ↓
[Digital Ocean Cloud Firewall]
    ↓
[VPS Host (Ubuntu 24.04)]
    ↓
[UFW + iptables Host Firewall]
    ↓
[Docker Bridge Network: trading-network]
    ↓
[Container Network Policies]
    ↓
[Application Services]
```

## 1. Docker Network Security Policies

### 1.1 Container Network Isolation

**Network Segmentation Strategy:**
- **Primary Trading Network**: `trading-network` (bridge driver)
- **Isolated service communication** within Docker network
- **No direct container-to-internet access** except through controlled gateways

**Container Network Configuration:**
```yaml
networks:
  trading-network:
    name: trading-network
    driver: bridge
    driver_opts:
      com.docker.network.enable_icc: "false"  # Disable inter-container communication by default
      com.docker.network.enable_ip_masquerade: "true"
      com.docker.network.bridge.enable_ip_forward: "false"
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
          ip_range: 172.20.1.0/24
```

### 1.2 Service-Specific Network Policies

**AI Trading Bot (Primary Service):**
- Internal network communication only
- No direct external access
- Communicates with exchanges through secure proxies

**Dashboard Services:**
- Frontend: Port 3000/8080 (external access required)
- Backend: Port 8000 (external access required for API)
- Nginx: Port 80/443 (external access required)

**MCP Services:**
- Memory Server: Port 8765 (internal only)
- OmniSearch: Port 8767 (internal only)

**Bluefin Service:**
- API Port: 8080 (internal + localhost debugging only)

### 1.3 Container Communication Rules

**Allowed Communication Flows:**
```
ai-trading-bot → bluefin-service:8080
ai-trading-bot → mcp-memory:8765
ai-trading-bot → mcp-omnisearch:8767
dashboard-backend → bluefin-service:8080
dashboard-backend → ai-trading-bot (health checks)
dashboard-frontend → dashboard-backend:8000
```

**Blocked Communication:**
- Direct container-to-internet (except through application proxies)
- Inter-service communication not explicitly required
- Container-to-host communication (except mounted volumes)

## 2. VPS Host Firewall Configuration (UFW + iptables)

### 2.1 UFW Base Configuration

```bash
#!/bin/bash
# UFW Base Security Configuration

# Reset UFW to defaults
ufw --force reset

# Set default policies
ufw default deny incoming
ufw default allow outgoing
ufw default deny forward

# Enable UFW logging
ufw logging on

# SSH Access (modify port as needed)
ufw allow 22/tcp comment 'SSH Access'

# Docker daemon (for Docker API if needed - restrict to specific IPs)
# ufw allow from 10.0.0.0/8 to any port 2376 comment 'Docker API'

# Application ports (restrict to specific sources when possible)
ufw allow 3000/tcp comment 'Dashboard Frontend'
ufw allow 8000/tcp comment 'Dashboard Backend API'
ufw allow 8080/tcp comment 'Nginx Reverse Proxy'

# Rate limiting for HTTP services
ufw limit 3000/tcp
ufw limit 8000/tcp
ufw limit 8080/tcp

# Debug ports (localhost only)
ufw allow from 127.0.0.1 to any port 8765 comment 'MCP Memory (localhost only)'
ufw allow from 127.0.0.1 to any port 8767 comment 'MCP OmniSearch (localhost only)'
ufw allow from 127.0.0.1 to any port 8081 comment 'Bluefin Service Debug (localhost only)'

# Enable UFW
ufw --force enable
```

### 2.2 Advanced iptables Rules for Docker Integration

```bash
#!/bin/bash
# Advanced iptables rules for Docker network security

# Create custom chains for Docker traffic
iptables -t filter -N DOCKER-TRADING
iptables -t filter -N DOCKER-TRADING-ISOLATION

# Insert rules to use custom chains
iptables -t filter -I DOCKER-USER -j DOCKER-TRADING
iptables -t filter -I FORWARD -j DOCKER-TRADING-ISOLATION

# Block inter-container communication by default
iptables -t filter -A DOCKER-TRADING-ISOLATION -i docker0 -o docker0 -j DROP

# Allow specific container-to-container communication
# AI Trading Bot to Bluefin Service
iptables -t filter -A DOCKER-TRADING -s 172.20.1.10 -d 172.20.1.20 -p tcp --dport 8080 -j ACCEPT

# AI Trading Bot to MCP Services
iptables -t filter -A DOCKER-TRADING -s 172.20.1.10 -d 172.20.1.30 -p tcp --dport 8765 -j ACCEPT
iptables -t filter -A DOCKER-TRADING -s 172.20.1.10 -d 172.20.1.31 -p tcp --dport 8767 -j ACCEPT

# Dashboard Backend to Bluefin Service
iptables -t filter -A DOCKER-TRADING -s 172.20.1.40 -d 172.20.1.20 -p tcp --dport 8080 -j ACCEPT

# Rate limiting for external connections
iptables -t filter -A DOCKER-TRADING -p tcp --dport 3000 -m limit --limit 25/min --limit-burst 100 -j ACCEPT
iptables -t filter -A DOCKER-TRADING -p tcp --dport 8000 -m limit --limit 50/min --limit-burst 200 -j ACCEPT

# Drop excessive connections
iptables -t filter -A DOCKER-TRADING -p tcp --dport 3000 -j DROP
iptables -t filter -A DOCKER-TRADING -p tcp --dport 8000 -j DROP

# Log dropped packets
iptables -t filter -A DOCKER-TRADING -j LOG --log-prefix "DOCKER-TRADING-DROP: "
iptables -t filter -A DOCKER-TRADING -j DROP
```

### 2.3 DDoS Protection Rules

```bash
#!/bin/bash
# DDoS Protection and Rate Limiting

# Limit new SSH connections
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m limit --limit 2/min --limit-burst 5 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j DROP

# SYN flood protection
iptables -A INPUT -p tcp --syn -m limit --limit 5/s --limit-burst 10 -j ACCEPT
iptables -A INPUT -p tcp --syn -j DROP

# Ping flood protection
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s --limit-burst 3 -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# Port scan protection
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,FIN SYN,FIN -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,RST SYN,RST -j DROP
iptables -A INPUT -p tcp --tcp-flags FIN,RST FIN,RST -j DROP
iptables -A INPUT -p tcp --tcp-flags ACK,FIN FIN -j DROP
iptables -A INPUT -p tcp --tcp-flags ACK,PSH PSH -j DROP
iptables -A INPUT -p tcp --tcp-flags ACK,URG URG -j DROP
```

## 3. Digital Ocean Cloud Firewall Configuration

### 3.1 Inbound Rules

```yaml
# Digital Ocean Cloud Firewall - Inbound Rules
inbound_rules:
  # SSH Access (restrict to your IP ranges)
  - type: tcp
    ports: "22"
    sources:
      addresses:
        - "YOUR_IP_ADDRESS/32"  # Replace with your IP
        - "OFFICE_IP_RANGE/24"   # Replace with office IP range

  # HTTP/HTTPS for dashboard
  - type: tcp
    ports: "80"
    sources:
      addresses: ["0.0.0.0/0"]

  - type: tcp
    ports: "443"
    sources:
      addresses: ["0.0.0.0/0"]

  # Dashboard Frontend
  - type: tcp
    ports: "3000"
    sources:
      addresses:
        - "YOUR_IP_ADDRESS/32"
        - "TRUSTED_IP_RANGE/24"

  # Dashboard Backend API
  - type: tcp
    ports: "8000"
    sources:
      addresses:
        - "YOUR_IP_ADDRESS/32"
        - "TRUSTED_IP_RANGE/24"

  # Nginx Reverse Proxy
  - type: tcp
    ports: "8080"
    sources:
      addresses:
        - "YOUR_IP_ADDRESS/32"
        - "TRUSTED_IP_RANGE/24"

  # ICMP for monitoring
  - type: icmp
    sources:
      addresses:
        - "MONITORING_IP/32"
```

### 3.2 Outbound Rules

```yaml
# Digital Ocean Cloud Firewall - Outbound Rules
outbound_rules:
  # DNS
  - type: tcp
    ports: "53"
    destinations:
      addresses: ["0.0.0.0/0"]

  - type: udp
    ports: "53"
    destinations:
      addresses: ["0.0.0.0/0"]

  # HTTP/HTTPS for API calls
  - type: tcp
    ports: "80"
    destinations:
      addresses: ["0.0.0.0/0"]

  - type: tcp
    ports: "443"
    destinations:
      addresses: ["0.0.0.0/0"]

  # NTP
  - type: udp
    ports: "123"
    destinations:
      addresses: ["0.0.0.0/0"]

  # SMTP for alerts (if needed)
  - type: tcp
    ports: "587"
    destinations:
      addresses: ["0.0.0.0/0"]
```

### 3.3 Geographic Restrictions

```yaml
# Geographic Access Control
geographic_restrictions:
  allowed_countries:
    - "US"  # United States
    - "CA"  # Canada
    - "GB"  # United Kingdom
    - "DE"  # Germany
    - "JP"  # Japan
    # Add other countries as needed

  blocked_countries:
    - "CN"  # China
    - "RU"  # Russia
    - "KP"  # North Korea
    - "IR"  # Iran
    # Add other high-risk countries
```

## 4. Network Monitoring and Logging

### 4.1 Network Traffic Monitoring

```bash
#!/bin/bash
# Network Monitoring Script

# Install required tools
apt-get update
apt-get install -y nethogs iftop vnstat tcpdump

# Configure vnstat for interface monitoring
vnstat -i eth0 --create
systemctl enable vnstat
systemctl start vnstat

# Setup continuous monitoring
cat > /usr/local/bin/network-monitor.sh << 'EOF'
#!/bin/bash

LOG_FILE="/var/log/trading-bot/network-monitor.log"
ALERT_THRESHOLD_MBPS=100

while true; do
    # Get current network usage
    USAGE=$(vnstat -i eth0 --json | jq '.interfaces[0].traffic.total.rx + .interfaces[0].traffic.total.tx')

    # Check for suspicious activity
    CONNECTIONS=$(netstat -an | grep ESTABLISHED | wc -l)

    # Log current stats
    echo "$(date): Connections: $CONNECTIONS, Usage: $USAGE" >> $LOG_FILE

    # Alert on high connection count
    if [ $CONNECTIONS -gt 50 ]; then
        echo "$(date): HIGH CONNECTION COUNT: $CONNECTIONS" >> $LOG_FILE
        # Send alert (implement notification system)
    fi

    sleep 60
done
EOF

chmod +x /usr/local/bin/network-monitor.sh
```

### 4.2 Intrusion Detection Integration

```bash
#!/bin/bash
# Fail2Ban Configuration for Trading Bot

# Install Fail2Ban
apt-get install -y fail2ban

# Configure Fail2Ban for SSH
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
ignoreip = 127.0.0.1/8 YOUR_IP_ADDRESS/32

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 1800

[trading-bot-api]
enabled = true
port = 8000
filter = trading-bot-api
logpath = /var/log/trading-bot/access.log
maxretry = 10
findtime = 300
bantime = 1800

[dashboard]
enabled = true
port = 3000,8080
filter = dashboard
logpath = /var/log/nginx/access.log
maxretry = 15
findtime = 300
bantime = 900
EOF

# Create custom filters
cat > /etc/fail2ban/filter.d/trading-bot-api.conf << 'EOF'
[Definition]
failregex = ^.*"POST /api/.*" 40[13] .*$
            ^.*"GET /api/.*" 40[13] .*$
ignoreregex =
EOF

cat > /etc/fail2ban/filter.d/dashboard.conf << 'EOF'
[Definition]
failregex = ^<HOST> -.*"(GET|POST).*" (404|403|400) .*$
ignoreregex = ^<HOST> -.*"GET /health.*" 200 .*$
EOF

# Start Fail2Ban
systemctl enable fail2ban
systemctl start fail2ban
```

### 4.3 Security Event Logging

```bash
#!/bin/bash
# Security Event Logging Configuration

# Create log directories
mkdir -p /var/log/trading-bot/security
chmod 750 /var/log/trading-bot/security

# Configure rsyslog for security events
cat >> /etc/rsyslog.conf << 'EOF'
# Trading Bot Security Logs
:msg,contains,"DOCKER-TRADING-DROP" /var/log/trading-bot/security/docker-drops.log
:msg,contains,"UFW BLOCK" /var/log/trading-bot/security/ufw-blocks.log
:msg,contains,"FAIL2BAN" /var/log/trading-bot/security/fail2ban.log
& stop
EOF

# Setup log rotation
cat > /etc/logrotate.d/trading-bot-security << 'EOF'
/var/log/trading-bot/security/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    copytruncate
    notifempty
    postrotate
        systemctl reload rsyslog
    endscript
}
EOF

# Restart rsyslog
systemctl restart rsyslog
```

## 5. Container Security Enhancements

### 5.1 Network Security Labels

```yaml
# Container Security Labels (add to docker-compose.yml)
services:
  ai-trading-bot:
    labels:
      - "traefik.enable=false"
      - "security.network.isolation=high"
      - "security.external.access=false"
      - "security.monitoring=enabled"

  dashboard-backend:
    labels:
      - "security.network.isolation=medium"
      - "security.external.access=true"
      - "security.rate.limit=enabled"
      - "security.monitoring=enabled"

  bluefin-service:
    labels:
      - "security.network.isolation=high"
      - "security.external.access=debug-only"
      - "security.monitoring=enabled"
```

### 5.2 Runtime Security Policies

```yaml
# Runtime Security Policies
security_policies:
  network:
    # Prevent containers from accessing host network
    host_network: false

    # Restrict container network capabilities
    cap_add: []
    cap_drop: ["ALL"]

    # Enable AppArmor/SELinux profiles
    security_opt:
      - "apparmor:docker-trading-bot"
      - "no-new-privileges:true"

  processes:
    # Run as non-root user
    user: "1000:1000"

    # Read-only root filesystem
    read_only: true

    # Limit process capabilities
    cap_drop: ["ALL"]
    cap_add: ["NET_BIND_SERVICE"]
```

## 6. Security Incident Response Procedures

### 6.1 Automated Response System

```bash
#!/bin/bash
# Automated Security Response Script

SCRIPT_DIR="/usr/local/bin"
LOG_DIR="/var/log/trading-bot/security"
ALERT_EMAIL="admin@yourdomain.com"

# Incident detection and response
cat > $SCRIPT_DIR/security-incident-response.sh << 'EOF'
#!/bin/bash

# Function to handle security incidents
handle_incident() {
    local incident_type=$1
    local severity=$2
    local details=$3

    # Log incident
    echo "$(date): INCIDENT [$severity] $incident_type: $details" >> /var/log/trading-bot/security/incidents.log

    case $incident_type in
        "DDOS")
            # Activate DDoS protection
            iptables -A INPUT -m limit --limit 5/min -j ACCEPT
            iptables -A INPUT -j DROP
            ;;
        "BRUTE_FORCE")
            # Extend ban time
            fail2ban-client set sshd bantime 7200
            ;;
        "SUSPICIOUS_TRAFFIC")
            # Increase logging verbosity
            ufw logging full
            ;;
        "CONTAINER_BREACH")
            # Isolate affected container
            docker network disconnect trading-network $4
            ;;
    esac

    # Send alert
    if [ "$severity" = "HIGH" ] || [ "$severity" = "CRITICAL" ]; then
        echo "$details" | mail -s "CRITICAL: Security Incident Detected" $ALERT_EMAIL
    fi
}

# Monitor for incidents
tail -f /var/log/auth.log /var/log/syslog | while read line; do
    if echo "$line" | grep -q "Failed password"; then
        handle_incident "BRUTE_FORCE" "MEDIUM" "$line"
    elif echo "$line" | grep -q "DOCKER-TRADING-DROP"; then
        handle_incident "SUSPICIOUS_TRAFFIC" "LOW" "$line"
    fi
done
EOF

chmod +x $SCRIPT_DIR/security-incident-response.sh
```

### 6.2 Emergency Procedures

```bash
#!/bin/bash
# Emergency Security Procedures

# Emergency shutdown script
cat > /usr/local/bin/emergency-shutdown.sh << 'EOF'
#!/bin/bash

echo "EMERGENCY SHUTDOWN INITIATED" | logger -t TRADING-BOT-SECURITY

# Stop all trading bot containers
docker-compose -f /opt/trading-bot/docker-compose.yml down

# Block all traffic except SSH
iptables -F
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A OUTPUT -p tcp --sport 22 -j ACCEPT

echo "Emergency shutdown complete. SSH access maintained."
EOF

chmod +x /usr/local/bin/emergency-shutdown.sh

# Network isolation script
cat > /usr/local/bin/isolate-containers.sh << 'EOF'
#!/bin/bash

echo "ISOLATING TRADING BOT CONTAINERS" | logger -t TRADING-BOT-SECURITY

# Disconnect all containers from external networks
for container in $(docker ps --format "table {{.Names}}" | grep -E "(trading-bot|bluefin|dashboard)"); do
    docker network disconnect bridge $container 2>/dev/null || true
done

# Create isolated network
docker network create --internal isolated-trading-network

# Reconnect containers to isolated network
docker network connect isolated-trading-network ai-trading-bot
docker network connect isolated-trading-network bluefin-service
docker network connect isolated-trading-network mcp-memory
docker network connect isolated-trading-network mcp-omnisearch

echo "Containers isolated. External network access blocked."
EOF

chmod +x /usr/local/bin/isolate-containers.sh
```

## 7. Security Monitoring Dashboard

### 7.1 Network Security Metrics

```python
# Network Security Monitoring Integration
# Add to dashboard backend

import psutil
import subprocess
from datetime import datetime

class NetworkSecurityMonitor:
    def __init__(self):
        self.metrics = {}

    def get_network_stats(self):
        """Get current network statistics"""
        stats = psutil.net_io_counters()
        connections = len(psutil.net_connections())

        return {
            'bytes_sent': stats.bytes_sent,
            'bytes_recv': stats.bytes_recv,
            'packets_sent': stats.packets_sent,
            'packets_recv': stats.packets_recv,
            'active_connections': connections,
            'timestamp': datetime.now().isoformat()
        }

    def get_firewall_status(self):
        """Check UFW and iptables status"""
        try:
            ufw_status = subprocess.check_output(['ufw', 'status'], text=True)
            return {
                'ufw_active': 'Status: active' in ufw_status,
                'ufw_rules': len(ufw_status.split('\n')) - 5,
                'status': 'active' if 'Status: active' in ufw_status else 'inactive'
            }
        except:
            return {'status': 'error', 'ufw_active': False}

    def get_security_alerts(self):
        """Get recent security alerts from logs"""
        alerts = []
        try:
            with open('/var/log/trading-bot/security/incidents.log', 'r') as f:
                lines = f.readlines()[-10:]  # Last 10 incidents
                for line in lines:
                    if 'INCIDENT' in line:
                        alerts.append({
                            'timestamp': line.split(':')[0],
                            'type': line.split('[')[1].split(']')[0],
                            'message': line.split(': ', 2)[-1].strip()
                        })
        except:
            pass
        return alerts
```

### 7.2 Real-time Security Dashboard

```javascript
// Frontend Security Dashboard Component
const SecurityDashboard = () => {
  const [securityMetrics, setSecurityMetrics] = useState({});
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const fetchSecurityData = async () => {
      try {
        const response = await fetch('/api/security/metrics');
        const data = await response.json();
        setSecurityMetrics(data);

        const alertResponse = await fetch('/api/security/alerts');
        const alertData = await alertResponse.json();
        setAlerts(alertData);
      } catch (error) {
        console.error('Error fetching security data:', error);
      }
    };

    fetchSecurityData();
    const interval = setInterval(fetchSecurityData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="security-dashboard">
      <div className="security-status">
        <h3>Network Security Status</h3>
        <div className="status-indicator">
          <span className={`status ${securityMetrics.firewall?.status === 'active' ? 'active' : 'inactive'}`}>
            Firewall: {securityMetrics.firewall?.status}
          </span>
        </div>
        <div className="metrics">
          <div>Active Connections: {securityMetrics.network?.active_connections}</div>
          <div>Firewall Rules: {securityMetrics.firewall?.ufw_rules}</div>
        </div>
      </div>

      <div className="security-alerts">
        <h3>Recent Security Alerts</h3>
        {alerts.map((alert, index) => (
          <div key={index} className={`alert alert-${alert.type.toLowerCase()}`}>
            <span className="timestamp">{alert.timestamp}</span>
            <span className="type">{alert.type}</span>
            <span className="message">{alert.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

## 8. Implementation Checklist

### 8.1 Initial Setup
- [ ] Configure Docker network with custom bridge
- [ ] Implement UFW firewall rules
- [ ] Set up iptables integration with Docker
- [ ] Configure Digital Ocean Cloud Firewall
- [ ] Install and configure Fail2Ban
- [ ] Set up network monitoring tools

### 8.2 Security Hardening
- [ ] Implement container network isolation
- [ ] Configure rate limiting rules
- [ ] Set up DDoS protection
- [ ] Enable geographic restrictions
- [ ] Configure security event logging
- [ ] Test emergency response procedures

### 8.3 Monitoring Setup
- [ ] Deploy network monitoring scripts
- [ ] Configure security alerts
- [ ] Set up log rotation
- [ ] Integrate security metrics into dashboard
- [ ] Test incident response automation

### 8.4 Validation and Testing
- [ ] Perform penetration testing
- [ ] Validate firewall rules
- [ ] Test rate limiting effectiveness
- [ ] Verify container isolation
- [ ] Confirm monitoring accuracy

## 9. Maintenance and Updates

### 9.1 Regular Security Tasks
- **Daily**: Review security logs and alerts
- **Weekly**: Update firewall rules if needed
- **Monthly**: Review and update IP allowlists
- **Quarterly**: Perform security audit and penetration testing

### 9.2 Security Updates
- Keep all security tools updated (UFW, Fail2Ban, iptables)
- Regular review of container security policies
- Update geographic restrictions based on threat intelligence
- Maintain incident response procedures

---

**Document Version**: 2.0
**Last Updated**: July 2025
**Classification**: Internal Use
**Owner**: Trading Bot Security Team
