#!/bin/bash
# AI Trading Bot - Network Security Setup Script
# Comprehensive network security implementation for Digital Ocean VPS

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/trading-bot-security-setup.log"
TRADING_BOT_DIR="/opt/trading-bot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}AI Trading Bot - Network Security Setup${NC}"
    echo -e "${BLUE}================================================${NC}"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}This script must be run as root${NC}"
        exit 1
    fi
}

check_system() {
    log "Checking system requirements..."

    # Check Ubuntu version
    if ! grep -q "Ubuntu" /etc/os-release; then
        echo -e "${YELLOW}Warning: This script is optimized for Ubuntu. Proceeding anyway...${NC}"
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi

    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}docker-compose is not installed. Please install docker-compose first.${NC}"
        exit 1
    fi

    log "System requirements check passed"
}

install_security_tools() {
    log "Installing security tools..."

    apt-get update
    apt-get install -y \
        ufw \
        fail2ban \
        iptables-persistent \
        netfilter-persistent \
        nethogs \
        iftop \
        vnstat \
        tcpdump \
        jq \
        mailutils \
        logrotate

    log "Security tools installed successfully"
}

configure_ufw() {
    log "Configuring UFW firewall..."

    # Reset UFW to defaults
    ufw --force reset

    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    ufw default deny forward

    # Enable UFW logging
    ufw logging on

    # SSH Access (default port - modify as needed)
    ufw allow 22/tcp comment 'SSH Access'

    # Dashboard ports with rate limiting
    ufw limit 3000/tcp comment 'Dashboard Frontend'
    ufw limit 8000/tcp comment 'Dashboard Backend API'
    ufw limit 8080/tcp comment 'Nginx Reverse Proxy'

    # Debug ports (localhost only)
    ufw allow from 127.0.0.1 to any port 8765 comment 'MCP Memory (localhost only)'
    ufw allow from 127.0.0.1 to any port 8767 comment 'MCP OmniSearch (localhost only)'
    ufw allow from 127.0.0.1 to any port 8081 comment 'Bluefin Service Debug (localhost only)'

    # Enable UFW
    ufw --force enable

    log "UFW firewall configured successfully"
}

configure_iptables_docker() {
    log "Configuring iptables for Docker integration..."

    # Create custom chains for Docker traffic
    iptables -t filter -N DOCKER-TRADING 2>/dev/null || true
    iptables -t filter -N DOCKER-TRADING-ISOLATION 2>/dev/null || true

    # Clear existing rules in custom chains
    iptables -t filter -F DOCKER-TRADING 2>/dev/null || true
    iptables -t filter -F DOCKER-TRADING-ISOLATION 2>/dev/null || true

    # Insert rules to use custom chains
    iptables -t filter -I DOCKER-USER -j DOCKER-TRADING 2>/dev/null || true
    iptables -t filter -I FORWARD -j DOCKER-TRADING-ISOLATION 2>/dev/null || true

    # Block inter-container communication by default
    iptables -t filter -A DOCKER-TRADING-ISOLATION -i docker0 -o docker0 -j DROP

    # Allow specific container-to-container communication
    # These rules will be applied dynamically when containers start

    # Rate limiting for external connections
    iptables -t filter -A DOCKER-TRADING -p tcp --dport 3000 -m limit --limit 25/min --limit-burst 100 -j ACCEPT
    iptables -t filter -A DOCKER-TRADING -p tcp --dport 8000 -m limit --limit 50/min --limit-burst 200 -j ACCEPT
    iptables -t filter -A DOCKER-TRADING -p tcp --dport 8080 -m limit --limit 30/min --limit-burst 150 -j ACCEPT

    # Drop excessive connections
    iptables -t filter -A DOCKER-TRADING -p tcp --dport 3000 -j REJECT --reject-with tcp-reset
    iptables -t filter -A DOCKER-TRADING -p tcp --dport 8000 -j REJECT --reject-with tcp-reset
    iptables -t filter -A DOCKER-TRADING -p tcp --dport 8080 -j REJECT --reject-with tcp-reset

    # Log dropped packets (limited to prevent log spam)
    iptables -t filter -A DOCKER-TRADING -m limit --limit 5/min -j LOG --log-prefix "DOCKER-TRADING-DROP: "
    iptables -t filter -A DOCKER-TRADING -j DROP

    # Save iptables rules
    netfilter-persistent save

    log "iptables Docker integration configured successfully"
}

configure_ddos_protection() {
    log "Configuring DDoS protection..."

    # Limit new SSH connections
    iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m limit --limit 2/min --limit-burst 5 -j ACCEPT
    iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j REJECT --reject-with tcp-reset

    # SYN flood protection
    iptables -A INPUT -p tcp --syn -m limit --limit 10/s --limit-burst 20 -j ACCEPT
    iptables -A INPUT -p tcp --syn -j DROP

    # Ping flood protection
    iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 2/s --limit-burst 5 -j ACCEPT
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

    # Connection tracking limits
    iptables -A INPUT -p tcp -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    iptables -A INPUT -p tcp -m connlimit --connlimit-above 20 --connlimit-mask 32 -j REJECT --reject-with tcp-reset

    # Save iptables rules
    netfilter-persistent save

    log "DDoS protection configured successfully"
}

configure_fail2ban() {
    log "Configuring Fail2Ban..."

    # Create Fail2Ban configuration
    cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
ignoreip = 127.0.0.1/8
backend = systemd

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
findtime = 600

[trading-bot-api]
enabled = true
port = 8000
filter = trading-bot-api
logpath = /var/log/trading-bot/access.log
maxretry = 15
findtime = 300
bantime = 1800

[dashboard]
enabled = true
port = 3000,8080
filter = dashboard
logpath = /var/log/nginx/access.log
maxretry = 20
findtime = 300
bantime = 900

[docker-iptables]
enabled = true
filter = docker-iptables
logpath = /var/log/kern.log
maxretry = 5
findtime = 300
bantime = 7200
action = iptables-allports[name=docker-iptables]
EOF

    # Create custom filters
    mkdir -p /etc/fail2ban/filter.d

    cat > /etc/fail2ban/filter.d/trading-bot-api.conf << 'EOF'
[Definition]
failregex = ^.*"(POST|GET|PUT|DELETE) /api/.*" (400|401|403|404|429|500) .*$
            ^.*\[error\].*client: <HOST>.*$
ignoreregex = ^.*"GET /api/health.*" 200 .*$
              ^.*"GET /health.*" 200 .*$
EOF

    cat > /etc/fail2ban/filter.d/dashboard.conf << 'EOF'
[Definition]
failregex = ^<HOST> -.*"(GET|POST).*" (404|403|400|429) .*$
            ^.*\[error\].*client: <HOST>.*$
ignoreregex = ^<HOST> -.*"GET /(health|favicon\.ico).*" (200|404) .*$
EOF

    cat > /etc/fail2ban/filter.d/docker-iptables.conf << 'EOF'
[Definition]
failregex = .*DOCKER-TRADING-DROP:.*SRC=<HOST>.*
ignoreregex =
EOF

    # Start and enable Fail2Ban
    systemctl enable fail2ban
    systemctl restart fail2ban

    log "Fail2Ban configured successfully"
}

setup_network_monitoring() {
    log "Setting up network monitoring..."

    # Create monitoring directories
    mkdir -p /var/log/trading-bot/security
    mkdir -p /var/log/trading-bot/monitoring
    chmod 750 /var/log/trading-bot/security
    chmod 750 /var/log/trading-bot/monitoring

    # Configure vnstat for interface monitoring
    vnstat -i eth0 --create 2>/dev/null || true
    systemctl enable vnstat
    systemctl start vnstat

    # Create network monitoring script
    cat > /usr/local/bin/network-monitor.sh << 'EOF'
#!/bin/bash

LOG_FILE="/var/log/trading-bot/monitoring/network-monitor.log"
ALERT_THRESHOLD_CONNECTIONS=100
ALERT_EMAIL="${ALERT_EMAIL:-admin@localhost}"

# Function to send alerts
send_alert() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null || logger -t TRADING-BOT-MONITOR "$subject: $message"
}

# Function to check network health
check_network_health() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Get current network connections
    local connections=$(netstat -an | grep -E ':(3000|8000|8080|22) ' | grep ESTABLISHED | wc -l)
    local total_connections=$(netstat -an | grep ESTABLISHED | wc -l)

    # Get network interface stats
    local interface_stats=$(cat /proc/net/dev | grep eth0 | awk '{print $2,$10}')
    local rx_bytes=$(echo $interface_stats | cut -d' ' -f1)
    local tx_bytes=$(echo $interface_stats | cut -d' ' -f2)

    # Log current stats
    echo "$timestamp,connections:$connections,total:$total_connections,rx:$rx_bytes,tx:$tx_bytes" >> "$LOG_FILE"

    # Check for suspicious activity
    if [ "$total_connections" -gt "$ALERT_THRESHOLD_CONNECTIONS" ]; then
        send_alert "HIGH CONNECTION COUNT ALERT" "Total connections: $total_connections exceeds threshold: $ALERT_THRESHOLD_CONNECTIONS"
    fi

    # Check for Docker container network issues
    if ! docker network ls | grep -q trading-network; then
        send_alert "DOCKER NETWORK ERROR" "Trading network is not available"
    fi

    # Check firewall status
    if ! ufw status | grep -q "Status: active"; then
        send_alert "FIREWALL DOWN" "UFW firewall is not active"
    fi
}

# Main monitoring loop
while true; do
    check_network_health
    sleep 60
done
EOF

    chmod +x /usr/local/bin/network-monitor.sh

    # Create systemd service for network monitoring
    cat > /etc/systemd/system/trading-bot-network-monitor.service << 'EOF'
[Unit]
Description=Trading Bot Network Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/network-monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable trading-bot-network-monitor
    systemctl start trading-bot-network-monitor

    log "Network monitoring setup completed"
}

configure_logging() {
    log "Configuring security logging..."

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
/var/log/trading-bot/security/*.log /var/log/trading-bot/monitoring/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    copytruncate
    notifempty
    create 0640 root root
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}
EOF

    # Restart rsyslog
    systemctl restart rsyslog

    log "Security logging configured successfully"
}

create_emergency_scripts() {
    log "Creating emergency response scripts..."

    # Emergency shutdown script
    cat > /usr/local/bin/emergency-shutdown.sh << 'EOF'
#!/bin/bash

echo "EMERGENCY SHUTDOWN INITIATED" | logger -t TRADING-BOT-SECURITY -p crit

# Stop all trading bot containers
if [ -f "/opt/trading-bot/docker-compose.yml" ]; then
    cd /opt/trading-bot
    docker-compose down 2>/dev/null || true
fi

# Block all traffic except SSH and localhost
iptables -F INPUT
iptables -F OUTPUT
iptables -F FORWARD
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# Allow localhost
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow SSH (both directions)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW,ESTABLISHED -j ACCEPT
iptables -A OUTPUT -p tcp --sport 22 -m conntrack --ctstate ESTABLISHED -j ACCEPT

# Allow DNS for emergency operations
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT -p udp --sport 53 -m conntrack --ctstate ESTABLISHED -j ACCEPT
iptables -A INPUT -p tcp --sport 53 -m conntrack --ctstate ESTABLISHED -j ACCEPT

# Save emergency rules
netfilter-persistent save

echo "Emergency shutdown complete. Only SSH and localhost access maintained."
echo "To restore normal operation, run: /usr/local/bin/restore-normal-operation.sh"
EOF

    # Container isolation script
    cat > /usr/local/bin/isolate-containers.sh << 'EOF'
#!/bin/bash

echo "ISOLATING TRADING BOT CONTAINERS" | logger -t TRADING-BOT-SECURITY -p warning

# Create isolated network if it doesn't exist
docker network create --internal isolated-trading-network 2>/dev/null || true

# Get list of trading bot containers
containers=$(docker ps --format "{{.Names}}" | grep -E "(trading-bot|bluefin|dashboard|mcp-)")

if [ -z "$containers" ]; then
    echo "No trading bot containers found"
    exit 0
fi

# Disconnect containers from external networks
for container in $containers; do
    echo "Isolating container: $container"

    # Disconnect from bridge network
    docker network disconnect bridge "$container" 2>/dev/null || true

    # Disconnect from trading-network
    docker network disconnect trading-network "$container" 2>/dev/null || true

    # Connect to isolated network
    docker network connect isolated-trading-network "$container" 2>/dev/null || true
done

echo "Containers isolated. External network access blocked."
echo "To restore normal operation, run: /usr/local/bin/restore-normal-operation.sh"
EOF

    # Restore normal operation script
    cat > /usr/local/bin/restore-normal-operation.sh << 'EOF'
#!/bin/bash

echo "RESTORING NORMAL OPERATION" | logger -t TRADING-BOT-SECURITY -p info

# Restore iptables rules
iptables-restore < /etc/iptables/rules.v4 2>/dev/null || true

# Restart UFW
ufw --force enable

# Restart trading bot services
if [ -f "/opt/trading-bot/docker-compose.yml" ]; then
    cd /opt/trading-bot
    docker-compose up -d
fi

# Restart fail2ban
systemctl restart fail2ban

echo "Normal operation restored."
EOF

    chmod +x /usr/local/bin/emergency-shutdown.sh
    chmod +x /usr/local/bin/isolate-containers.sh
    chmod +x /usr/local/bin/restore-normal-operation.sh

    log "Emergency response scripts created successfully"
}

configure_docker_network() {
    log "Configuring Docker network security..."

    # Create secure Docker daemon configuration
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << 'EOF'
{
  "icc": false,
  "userland-proxy": false,
  "iptables": true,
  "ip-forward": false,
  "ip-masq": true,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true,
  "default-address-pools": [
    {
      "base": "172.20.0.0/16",
      "size": 24
    }
  ]
}
EOF

    # Restart Docker daemon
    systemctl restart docker

    # Wait for Docker to restart
    sleep 5

    # Create secure trading network if it doesn't exist
    if ! docker network ls | grep -q trading-network; then
        docker network create \
            --driver bridge \
            --subnet=172.20.0.0/16 \
            --ip-range=172.20.1.0/24 \
            --gateway=172.20.0.1 \
            --opt com.docker.network.enable_icc=false \
            --opt com.docker.network.enable_ip_masquerade=true \
            --opt com.docker.network.bridge.enable_ip_forward=false \
            trading-network
    fi

    log "Docker network security configured successfully"
}

validate_security_setup() {
    log "Validating security setup..."

    local validation_failed=0

    # Check UFW status
    if ! ufw status | grep -q "Status: active"; then
        echo -e "${RED}FAIL: UFW firewall is not active${NC}"
        validation_failed=1
    else
        echo -e "${GREEN}PASS: UFW firewall is active${NC}"
    fi

    # Check Fail2Ban status
    if ! systemctl is-active --quiet fail2ban; then
        echo -e "${RED}FAIL: Fail2Ban is not running${NC}"
        validation_failed=1
    else
        echo -e "${GREEN}PASS: Fail2Ban is running${NC}"
    fi

    # Check Docker network
    if ! docker network ls | grep -q trading-network; then
        echo -e "${RED}FAIL: Trading network is not configured${NC}"
        validation_failed=1
    else
        echo -e "${GREEN}PASS: Trading network is configured${NC}"
    fi

    # Check monitoring service
    if ! systemctl is-active --quiet trading-bot-network-monitor; then
        echo -e "${RED}FAIL: Network monitor is not running${NC}"
        validation_failed=1
    else
        echo -e "${GREEN}PASS: Network monitor is running${NC}"
    fi

    # Check log directories
    if [ ! -d "/var/log/trading-bot/security" ]; then
        echo -e "${RED}FAIL: Security log directory does not exist${NC}"
        validation_failed=1
    else
        echo -e "${GREEN}PASS: Security log directory exists${NC}"
    fi

    if [ $validation_failed -eq 0 ]; then
        echo -e "${GREEN}All security validations passed!${NC}"
        log "Security setup validation completed successfully"
        return 0
    else
        echo -e "${RED}Some security validations failed!${NC}"
        log "Security setup validation failed"
        return 1
    fi
}

print_summary() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Network Security Setup Complete${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo -e "${GREEN}✓ UFW firewall configured and enabled${NC}"
    echo -e "${GREEN}✓ iptables Docker integration configured${NC}"
    echo -e "${GREEN}✓ DDoS protection implemented${NC}"
    echo -e "${GREEN}✓ Fail2Ban intrusion detection configured${NC}"
    echo -e "${GREEN}✓ Network monitoring enabled${NC}"
    echo -e "${GREEN}✓ Security logging configured${NC}"
    echo -e "${GREEN}✓ Emergency response scripts created${NC}"
    echo -e "${GREEN}✓ Docker network security hardened${NC}"
    echo ""
    echo -e "${YELLOW}Important files created:${NC}"
    echo "  • /usr/local/bin/emergency-shutdown.sh"
    echo "  • /usr/local/bin/isolate-containers.sh"
    echo "  • /usr/local/bin/restore-normal-operation.sh"
    echo "  • /usr/local/bin/network-monitor.sh"
    echo ""
    echo -e "${YELLOW}Log locations:${NC}"
    echo "  • Security logs: /var/log/trading-bot/security/"
    echo "  • Monitor logs: /var/log/trading-bot/monitoring/"
    echo "  • Setup log: $LOG_FILE"
    echo ""
    echo -e "${YELLOW}Services running:${NC}"
    echo "  • UFW firewall"
    echo "  • Fail2Ban intrusion detection"
    echo "  • Network monitor daemon"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Review and customize IP allowlists in /etc/fail2ban/jail.local"
    echo "2. Configure email alerts by setting ALERT_EMAIL environment variable"
    echo "3. Test emergency procedures: /usr/local/bin/emergency-shutdown.sh"
    echo "4. Monitor security logs regularly"
    echo "5. Configure Digital Ocean Cloud Firewall in your DO dashboard"
}

# Main execution
main() {
    print_header

    # Preliminary checks
    check_root
    check_system

    # Create log file
    touch "$LOG_FILE"
    chmod 600 "$LOG_FILE"

    # Main setup steps
    install_security_tools
    configure_ufw
    configure_iptables_docker
    configure_ddos_protection
    configure_fail2ban
    setup_network_monitoring
    configure_logging
    create_emergency_scripts
    configure_docker_network

    # Validation
    if validate_security_setup; then
        print_summary
        log "Network security setup completed successfully"
        exit 0
    else
        echo -e "${RED}Setup completed with errors. Check logs for details.${NC}"
        log "Network security setup completed with errors"
        exit 1
    fi
}

# Run main function
main "$@"
