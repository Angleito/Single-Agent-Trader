#!/bin/bash

# VPS Deployment Script for AI Trading Bot
# Optimized for Bluefin balance functionality and regional restrictions bypass

set -euo pipefail

# Script Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/vps-deployment.log"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
VPS_REGION="${VPS_REGION:-US}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# System requirements check
check_system_requirements() {
    log "Checking system requirements..."

    # Check if running as root for system setup
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This is acceptable for initial VPS setup but not recommended for running the application."
    fi

    # Check Docker installation
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df / | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        error "Insufficient disk space. At least 10GB free space required."
    fi

    # Check available memory (minimum 2GB)
    AVAILABLE_MEM=$(free -m | grep '^Mem:' | awk '{print $2}')
    if [ "$AVAILABLE_MEM" -lt 2048 ]; then
        warn "Low memory detected ($AVAILABLE_MEM MB). Recommended minimum: 2GB"
    fi

    log "System requirements check completed successfully"
}

# Environment validation
validate_environment() {
    log "Validating environment configuration..."

    # Check for required environment file
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        error ".env file not found. Please create .env file with required configuration."
    fi

    # Source environment variables
    source "$PROJECT_ROOT/.env"

    # Validate critical environment variables
    REQUIRED_VARS=(
        "EXCHANGE__BLUEFIN_PRIVATE_KEY"
        "BLUEFIN_SERVICE_API_KEY"
        "LLM__OPENAI_API_KEY"
    )

    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
        fi
    done

    # Validate Bluefin private key format
    if [[ ! "$EXCHANGE__BLUEFIN_PRIVATE_KEY" =~ ^0x[a-fA-F0-9]{64}$ ]]; then
        error "Invalid Bluefin private key format. Must be 64-character hex string starting with 0x"
    fi

    # Validate API key format
    if [[ ${#BLUEFIN_SERVICE_API_KEY} -lt 32 ]]; then
        warn "Bluefin service API key seems short. Ensure it's correctly configured."
    fi

    log "Environment validation completed successfully"
}

# VPS-specific system setup
setup_vps_system() {
    log "Setting up VPS-specific system configuration..."

    # Create necessary directories
    sudo mkdir -p /var/log/vps-trading-bot
    sudo mkdir -p /var/lib/vps-trading-bot/data
    sudo mkdir -p /etc/vps-trading-bot/config
    sudo mkdir -p /var/log/vps-bluefin-service
    sudo mkdir -p /var/lib/vps-bluefin-service/data
    sudo mkdir -p /var/lib/vps-mcp-memory/data
    sudo mkdir -p /var/log/vps-mcp-memory
    sudo mkdir -p /var/log/vps-dashboard
    sudo mkdir -p /var/lib/vps-dashboard/data

    # Set proper permissions
    sudo chown -R 1000:1000 /var/log/vps-*
    sudo chown -R 1000:1000 /var/lib/vps-*
    sudo chown -R 1000:1000 /etc/vps-trading-bot

    # Create systemd service for monitoring
    sudo tee /etc/systemd/system/vps-trading-monitor.service > /dev/null <<EOF
[Unit]
Description=VPS Trading Bot Monitor
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/vps-health-check.sh
User=root

[Install]
WantedBy=multi-user.target
EOF

    # Create systemd timer for periodic health checks
    sudo tee /etc/systemd/system/vps-trading-monitor.timer > /dev/null <<EOF
[Unit]
Description=Run VPS Trading Bot Monitor every 5 minutes
Requires=vps-trading-monitor.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Enable and start the timer
    sudo systemctl daemon-reload
    sudo systemctl enable vps-trading-monitor.timer
    sudo systemctl start vps-trading-monitor.timer

    log "VPS system setup completed"
}

# Create VPS configuration files
create_vps_config() {
    log "Creating VPS-specific configuration files..."

    # Create VPS production config
    cat > "$PROJECT_ROOT/config/vps_production.json" <<EOF
{
  "system": {
    "environment": "vps-production",
    "dry_run": ${SYSTEM__DRY_RUN:-true},
    "enable_websocket_publishing": true,
    "websocket_publish_interval": 2.0,
    "websocket_max_retries": 10,
    "websocket_retry_delay": 10,
    "websocket_timeout": 60,
    "performance_monitoring": true,
    "health_check_interval": 30,
    "geographic_region": "${VPS_REGION}",
    "vps_deployment": true
  },
  "exchange": {
    "exchange_type": "bluefin",
    "bluefin_network": "${EXCHANGE__BLUEFIN_NETWORK:-mainnet}",
    "connection_timeout": 30,
    "read_timeout": 60,
    "max_retries": 5,
    "retry_backoff": 2,
    "rate_limit": 50
  },
  "trading": {
    "symbol": "SUI-PERP",
    "leverage": 5,
    "position_check_interval": 10,
    "market_data_interval": 5,
    "health_check_interval": 30,
    "max_position_size": 0.25,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.04
  },
  "risk": {
    "max_drawdown": 0.15,
    "max_daily_loss": 0.05,
    "max_open_positions": 1,
    "position_size_multiplier": 0.25,
    "emergency_stop_enabled": true
  },
  "logging": {
    "level": "INFO",
    "enable_file_logging": true,
    "log_rotation": true,
    "max_log_size": "100MB",
    "max_log_files": 10,
    "structured_logging": true
  },
  "monitoring": {
    "enabled": true,
    "metrics_interval": 60,
    "health_check_interval": 30,
    "performance_tracking": true,
    "alert_enabled": true,
    "backup_enabled": true,
    "backup_interval": 3600
  }
}
EOF

    # Create Fluent Bit configuration for log aggregation
    mkdir -p "$PROJECT_ROOT/config"
    cat > "$PROJECT_ROOT/config/fluent-bit-vps.conf" <<EOF
[SERVICE]
    Flush         5
    Daemon        off
    Log_Level     info
    Parsers_File  parsers.conf

[INPUT]
    Name              tail
    Path              /var/log/trading/*.log
    Tag               trading.*
    Refresh_Interval  5
    Read_from_Head    true

[INPUT]
    Name              tail
    Path              /var/log/bluefin/*.log
    Tag               bluefin.*
    Refresh_Interval  5
    Read_from_Head    true

[INPUT]
    Name              tail
    Path              /var/log/mcp/*.log
    Tag               mcp.*
    Refresh_Interval  5
    Read_from_Head    true

[INPUT]
    Name              tail
    Path              /var/log/dashboard/*.log
    Tag               dashboard.*
    Refresh_Interval  5
    Read_from_Head    true

[FILTER]
    Name                modify
    Match               *
    Add                 hostname \${HOSTNAME}
    Add                 vps_region ${VPS_REGION}
    Add                 environment vps-production

[OUTPUT]
    Name  stdout
    Match *
    Format json
EOF

    # Create VPS health check script
    sudo tee /usr/local/bin/vps-health-check.sh > /dev/null <<'EOF'
#!/bin/bash

# VPS Health Check Script
set -euo pipefail

DOCKER_COMPOSE_FILE="${PROJECT_ROOT:-/opt/trading-bot}/docker-compose.vps.yml"
LOG_FILE="/var/log/vps-health-check.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if services are running
check_services() {
    local services=("bluefin-service-vps" "ai-trading-bot-vps" "mcp-memory-vps")

    for service in "${services[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "$service"; then
            log "ERROR: Service $service is not running"
            # Attempt to restart the service
            docker-compose -f "$DOCKER_COMPOSE_FILE" up -d "$service" || log "Failed to restart $service"
        else
            log "INFO: Service $service is running"
        fi
    done
}

# Check disk space
check_disk_space() {
    local usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$usage" -gt 90 ]; then
        log "WARNING: Disk usage is ${usage}%"
        # Clean up old logs
        find /var/log/vps-* -name "*.log" -mtime +7 -delete
    fi
}

# Check memory usage
check_memory() {
    local mem_usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
    if [ "$mem_usage" -gt 90 ]; then
        log "WARNING: Memory usage is ${mem_usage}%"
    fi
}

# Main health check
main() {
    log "Starting VPS health check"
    check_services
    check_disk_space
    check_memory
    log "VPS health check completed"
}

main
EOF

    sudo chmod +x /usr/local/bin/vps-health-check.sh

    log "VPS configuration files created successfully"
}

# Network optimization for VPS
optimize_vps_network() {
    log "Optimizing VPS network configuration..."

    # Create Docker network if it doesn't exist
    if ! docker network ls | grep -q "vps-trading-network"; then
        docker network create \
            --driver bridge \
            --subnet=172.20.0.0/16 \
            --opt com.docker.network.bridge.name=vps-trading-br \
            vps-trading-network
    fi

    # Optimize network settings for better performance
    sudo tee -a /etc/sysctl.conf > /dev/null <<EOF

# VPS Trading Bot Network Optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
EOF

    # Apply sysctl settings
    sudo sysctl -p

    log "VPS network optimization completed"
}

# Security hardening for VPS
harden_vps_security() {
    log "Applying VPS security hardening..."

    # Configure firewall rules
    if command -v ufw &> /dev/null; then
        sudo ufw --force enable
        sudo ufw default deny incoming
        sudo ufw default allow outgoing

        # Allow SSH (adjust port as needed)
        sudo ufw allow 22/tcp

        # Allow HTTPS for external services
        sudo ufw allow 443/tcp

        # Allow monitoring dashboard (adjust as needed)
        sudo ufw allow 8000/tcp

        log "UFW firewall configured"
    fi

    # Set up log rotation for application logs
    sudo tee /etc/logrotate.d/vps-trading-bot > /dev/null <<EOF
/var/log/vps-trading-bot/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    postrotate
        /bin/kill -HUP \$(cat /var/run/rsyslogd.pid 2>/dev/null) 2>/dev/null || true
    endscript
}

/var/log/vps-bluefin-service/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}

/var/log/vps-mcp-memory/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

    log "VPS security hardening completed"
}

# Deploy services
deploy_services() {
    log "Deploying VPS services..."

    cd "$PROJECT_ROOT"

    # Build and start services
    docker-compose -f docker-compose.vps.yml build --no-cache
    docker-compose -f docker-compose.vps.yml up -d

    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30

    # Verify service health
    local services=("bluefin-service-vps" "ai-trading-bot-vps" "mcp-memory-vps")
    local max_attempts=30
    local attempt=0

    for service in "${services[@]}"; do
        attempt=0
        while [ $attempt -lt $max_attempts ]; do
            if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$service" | grep -q "healthy\|Up"; then
                log "Service $service is healthy"
                break
            fi

            attempt=$((attempt + 1))
            log "Waiting for $service to become healthy (attempt $attempt/$max_attempts)..."
            sleep 10
        done

        if [ $attempt -eq $max_attempts ]; then
            error "Service $service failed to become healthy within expected time"
        fi
    done

    log "All services deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying VPS deployment..."

    # Check service status
    docker-compose -f "$PROJECT_ROOT/docker-compose.vps.yml" ps

    # Check service logs for errors
    local services=("bluefin-service-vps" "ai-trading-bot-vps" "mcp-memory-vps")

    for service in "${services[@]}"; do
        log "Checking logs for $service..."
        if docker logs "$service" 2>&1 | grep -i "error\|exception\|failed" | tail -5; then
            warn "Found potential issues in $service logs. Please review."
        fi
    done

    # Test service endpoints
    log "Testing service endpoints..."

    # Test Bluefin service health
    if curl -f --connect-timeout 10 http://localhost:8081/health &> /dev/null; then
        log "Bluefin service health endpoint is responding"
    else
        warn "Bluefin service health endpoint is not responding"
    fi

    # Test MCP memory service health
    if curl -f --connect-timeout 10 http://localhost:8765/health &> /dev/null; then
        log "MCP memory service health endpoint is responding"
    else
        warn "MCP memory service health endpoint is not responding"
    fi

    log "Deployment verification completed"
}

# Backup configuration
setup_backup() {
    log "Setting up backup configuration..."

    # Create backup script
    sudo tee /usr/local/bin/vps-backup.sh > /dev/null <<'EOF'
#!/bin/bash

# VPS Trading Bot Backup Script
set -euo pipefail

BACKUP_DIR="/var/backups/vps-trading-bot"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup data directories
tar -czf "$BACKUP_DIR/data_backup_$DATE.tar.gz" \
    /var/lib/vps-trading-bot/data \
    /var/lib/vps-bluefin-service/data \
    /var/lib/vps-mcp-memory/data \
    /etc/vps-trading-bot/config

# Backup logs (last 24 hours)
tar -czf "$BACKUP_DIR/logs_backup_$DATE.tar.gz" \
    --newer-mtime="24 hours ago" \
    /var/log/vps-trading-bot \
    /var/log/vps-bluefin-service \
    /var/log/vps-mcp-memory

# Clean up old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $DATE"
EOF

    sudo chmod +x /usr/local/bin/vps-backup.sh

    # Create cron job for daily backups
    echo "0 2 * * * /usr/local/bin/vps-backup.sh" | sudo crontab -

    log "Backup configuration completed"
}

# Main deployment function
main() {
    log "Starting VPS deployment for AI Trading Bot..."
    log "Deployment environment: $DEPLOYMENT_ENV"
    log "VPS region: $VPS_REGION"

    # Run deployment steps
    check_system_requirements
    validate_environment
    setup_vps_system
    create_vps_config
    optimize_vps_network
    harden_vps_security
    deploy_services
    verify_deployment
    setup_backup

    log "VPS deployment completed successfully!"

    # Display deployment summary
    echo ""
    echo "========================================"
    echo "VPS DEPLOYMENT SUMMARY"
    echo "========================================"
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Region: $VPS_REGION"
    echo "Services deployed:"
    echo "  - Bluefin SDK Service"
    echo "  - AI Trading Bot"
    echo "  - MCP Memory Server"
    echo "  - Monitoring Dashboard"
    echo "  - Log Aggregator"
    echo ""
    echo "Service URLs:"
    echo "  - Trading Bot Logs: /var/log/vps-trading-bot/"
    echo "  - Bluefin Service Health: http://localhost:8081/health"
    echo "  - MCP Memory Health: http://localhost:8765/health"
    echo "  - Monitoring Dashboard: http://localhost:8000"
    echo ""
    echo "Management commands:"
    echo "  - View logs: docker-compose -f docker-compose.vps.yml logs -f"
    echo "  - Restart services: docker-compose -f docker-compose.vps.yml restart"
    echo "  - Health check: /usr/local/bin/vps-health-check.sh"
    echo "  - Manual backup: /usr/local/bin/vps-backup.sh"
    echo ""
    echo "IMPORTANT: Review and update .env file with production values!"
    echo "========================================"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify")
        verify_deployment
        ;;
    "backup")
        setup_backup
        ;;
    "health-check")
        /usr/local/bin/vps-health-check.sh
        ;;
    *)
        echo "Usage: $0 {deploy|verify|backup|health-check}"
        echo "  deploy      - Full VPS deployment (default)"
        echo "  verify      - Verify existing deployment"
        echo "  backup      - Setup backup configuration"
        echo "  health-check - Run health check"
        exit 1
        ;;
esac
