#!/bin/bash

# OPTIMIZE Unified Security Monitoring Platform Deployment Script
# This script deploys the complete unified security monitoring system

set -e

echo "🛡️ OPTIMIZE - Unified Security Monitoring Platform Deployment"
echo "============================================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECURITY_DIR="$PROJECT_ROOT/security"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Default configuration
REDIS_PORT=${REDIS_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8080}
ENVIRONMENT=${ENVIRONMENT:-production}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Functions
log_info() {
    echo "ℹ️  $1"
}

log_success() {
    echo "✅ $1"
}

log_warning() {
    echo "⚠️  $1"
}

log_error() {
    echo "❌ $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

create_directories() {
    log_info "Creating required directories..."

    # Create data directories
    mkdir -p "$PROJECT_ROOT/data/security"
    mkdir -p "$PROJECT_ROOT/data/unified-monitoring"
    mkdir -p "$PROJECT_ROOT/logs/security"
    mkdir -p "$PROJECT_ROOT/logs/unified-monitoring"
    mkdir -p "$PROJECT_ROOT/config/security"

    # Create database directories
    mkdir -p "$PROJECT_ROOT/data/unified-monitoring/databases"
    mkdir -p "$PROJECT_ROOT/data/unified-monitoring/redis"
    mkdir -p "$PROJECT_ROOT/data/unified-monitoring/reports"
    mkdir -p "$PROJECT_ROOT/data/unified-monitoring/evidence"

    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/data/security"
    chmod 755 "$PROJECT_ROOT/data/unified-monitoring"
    chmod 755 "$PROJECT_ROOT/logs/security"
    chmod 755 "$PROJECT_ROOT/logs/unified-monitoring"

    log_success "Directories created"
}

generate_configuration() {
    log_info "Generating configuration files..."

    # Create main configuration file
    cat > "$SCRIPT_DIR/config.json" << EOF
{
    "platform": {
        "environment": "$ENVIRONMENT",
        "log_level": "$LOG_LEVEL",
        "redis_url": "redis://redis:$REDIS_PORT",
        "dashboard_port": $DASHBOARD_PORT
    },
    "correlation_engine": {
        "batch_size": 100,
        "correlation_window_minutes": 30,
        "db_url": "sqlite:///data/unified-monitoring/databases/correlation.db"
    },
    "alert_orchestrator": {
        "max_concurrent_alerts": 50,
        "deduplication_window_minutes": 15,
        "db_url": "sqlite:///data/unified-monitoring/databases/alerts.db"
    },
    "performance_monitor": {
        "collection_interval_seconds": 30,
        "optimization_interval_minutes": 5,
        "db_url": "sqlite:///data/unified-monitoring/databases/performance.db"
    },
    "response_automation": {
        "max_concurrent_responses": 5,
        "response_timeout_minutes": 60,
        "db_url": "sqlite:///data/unified-monitoring/databases/responses.db"
    },
    "executive_reporting": {
        "report_generation_schedule": {
            "daily_summary": "0 8 * * *",
            "weekly_security": "0 9 * * 1",
            "monthly_executive": "0 10 1 * *"
        },
        "db_url": "sqlite:///data/unified-monitoring/databases/reports.db"
    },
    "notifications": [
        {
            "channel": "slack",
            "target": "${SLACK_WEBHOOK_URL:-}",
            "enabled": false,
            "severity_filter": ["critical", "high"]
        },
        {
            "channel": "email",
            "target": "${SECURITY_EMAIL:-security@company.com}",
            "enabled": false,
            "severity_filter": ["critical", "high", "medium"]
        }
    ],
    "security_tools": {
        "falco": {
            "enabled": true,
            "health_check_interval": 60,
            "events_channel": "falco_events"
        },
        "trivy": {
            "enabled": true,
            "scan_interval": 3600,
            "events_channel": "trivy_events"
        },
        "docker_bench": {
            "enabled": true,
            "scan_interval": 86400,
            "events_channel": "docker_bench_events"
        }
    },
    "trading_bot": {
        "api_url": "http://ai-trading-bot:8000",
        "emergency_stop_endpoint": "/api/emergency/stop",
        "health_check_endpoint": "/api/health",
        "events_channel": "trading_bot_events"
    }
}
EOF

    # Create environment file
    cat > "$SCRIPT_DIR/.env" << EOF
# OPTIMIZE Platform Environment Configuration
ENVIRONMENT=$ENVIRONMENT
LOG_LEVEL=$LOG_LEVEL
REDIS_PORT=$REDIS_PORT
DASHBOARD_PORT=$DASHBOARD_PORT

# Optional: Notification Configuration
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-}
SECURITY_EMAIL=${SECURITY_EMAIL:-}
PAGERDUTY_API_KEY=${PAGERDUTY_API_KEY:-}

# Optional: SMTP Configuration for Email Alerts
SMTP_SERVER=${SMTP_SERVER:-}
SMTP_PORT=${SMTP_PORT:-587}
SMTP_USERNAME=${SMTP_USERNAME:-}
SMTP_PASSWORD=${SMTP_PASSWORD:-}

# Database Configuration
DB_ENCRYPTION_KEY=${DB_ENCRYPTION_KEY:-$(openssl rand -hex 32)}

# Security Configuration
API_SECRET_KEY=${API_SECRET_KEY:-$(openssl rand -hex 32)}
SESSION_SECRET_KEY=${SESSION_SECRET_KEY:-$(openssl rand -hex 32)}
EOF

    log_success "Configuration files generated"
}

create_docker_compose() {
    log_info "Creating Docker Compose configuration..."

    cat > "$COMPOSE_FILE" << 'EOF'
version: '3.8'

services:
  # Redis for event streaming and caching
  redis:
    image: redis:7-alpine
    container_name: optimize-redis
    restart: unless-stopped
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - optimize-network

  # Unified Security Orchestrator
  optimize-orchestrator:
    build:
      context: ../..
      dockerfile: security/unified-monitoring/Dockerfile
    container_name: optimize-orchestrator
    restart: unless-stopped
    depends_on:
      redis:
        condition: service_healthy
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-production}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - REDIS_URL=redis://redis:${REDIS_PORT:-6379}
      - CONFIG_PATH=/app/config/config.json
    volumes:
      - ./config.json:/app/config/config.json:ro
      - ../../data/unified-monitoring:/app/data/unified-monitoring
      - ../../logs/unified-monitoring:/app/logs/unified-monitoring
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - optimize-network
      - trading-network
    healthcheck:
      test: ["CMD", "python3", "-c", "import requests; requests.get('http://localhost:8080/api/v1/status')"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Security Dashboard
  optimize-dashboard:
    build:
      context: ../..
      dockerfile: security/unified-monitoring/Dockerfile.dashboard
    container_name: optimize-dashboard
    restart: unless-stopped
    depends_on:
      - optimize-orchestrator
    ports:
      - "${DASHBOARD_PORT:-8080}:8080"
    environment:
      - REDIS_URL=redis://redis:${REDIS_PORT:-6379}
      - ORCHESTRATOR_URL=http://optimize-orchestrator:8080
    networks:
      - optimize-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Falco Integration (if not already running)
  falco:
    image: falcosecurity/falco:latest
    container_name: optimize-falco
    restart: unless-stopped
    privileged: true
    volumes:
      - /var/run/docker.sock:/host/var/run/docker.sock
      - /dev:/host/dev
      - /proc:/host/proc:ro
      - /boot:/host/boot:ro
      - /lib/modules:/host/lib/modules:ro
      - /usr:/host/usr:ro
      - /etc:/host/etc:ro
      - ../falco/falco.yaml:/etc/falco/falco.yaml:ro
      - ../falco/trading_bot_rules.yaml:/etc/falco/rules.d/trading_bot_rules.yaml:ro
    environment:
      - REDIS_URL=redis://redis:${REDIS_PORT:-6379}
    networks:
      - optimize-network
    profiles:
      - security-tools

  # Trivy Scanner (scheduled)
  trivy:
    image: aquasec/trivy:latest
    container_name: optimize-trivy
    restart: "no"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ../../data/unified-monitoring/trivy-cache:/root/.cache/trivy
      - ../trivy/trivy-config.yaml:/etc/trivy/trivy.yaml:ro
    environment:
      - REDIS_URL=redis://redis:${REDIS_PORT:-6379}
    networks:
      - optimize-network
    profiles:
      - security-tools

networks:
  optimize-network:
    name: optimize-network
    driver: bridge
  trading-network:
    name: trading-network
    external: true

volumes:
  redis-data:
    driver: local
EOF

    log_success "Docker Compose configuration created"
}

create_dockerfile() {
    log_info "Creating Dockerfile for the orchestrator..."

    cat > "$SCRIPT_DIR/Dockerfile" << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    iptables \
    netstat-nat \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY security/unified-monitoring/ ./security/unified-monitoring/
COPY bot/ ./bot/

# Create directories
RUN mkdir -p /app/data/unified-monitoring /app/logs/unified-monitoring /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/api/v1/status')" || exit 1

# Start the orchestrator
CMD ["python3", "-m", "security.unified_monitoring.unified_orchestrator"]
EOF

    # Create dashboard Dockerfile
    cat > "$SCRIPT_DIR/Dockerfile.dashboard" << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code
COPY security/unified-monitoring/security_dashboard.py ./
COPY security/unified-monitoring/correlation_engine.py ./

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Start the dashboard
CMD ["python3", "security_dashboard.py"]
EOF

    # Create requirements file
    cat > "$SCRIPT_DIR/requirements.txt" << 'EOF'
# Core dependencies
aiohttp>=3.8.0
aiofiles>=23.0.0
asyncio-mqtt>=0.13.0
redis>=4.5.0
sqlalchemy>=2.0.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
psutil>=5.9.0

# Security and validation
cryptography>=40.0.0
pydantic>=2.0.0

# Templating and reporting
jinja2>=3.1.0
matplotlib>=3.7.0
plotly>=5.14.0

# Networking and HTTP
aiohttp-cors>=0.7.0
aiohttp-session>=2.12.0
requests>=2.31.0

# Optional: For enhanced features
# prometheus-client>=0.16.0
# grafana-api>=1.0.3
EOF

    log_success "Dockerfiles created"
}

create_systemd_service() {
    log_info "Creating systemd service (optional)..."

    cat > "$SCRIPT_DIR/optimize-security.service" << EOF
[Unit]
Description=OPTIMIZE Unified Security Monitoring Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    log_success "Systemd service file created"
}

setup_monitoring_scripts() {
    log_info "Setting up monitoring scripts..."

    # Create health check script
    cat > "$SCRIPT_DIR/health-check.sh" << 'EOF'
#!/bin/bash

# OPTIMIZE Platform Health Check Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_URL="http://localhost:${DASHBOARD_PORT:-8080}"

echo "🏥 OPTIMIZE Platform Health Check"
echo "================================"

# Check if containers are running
echo "📦 Checking container status..."
if ! docker-compose -f "$SCRIPT_DIR/docker-compose.yml" ps | grep -q "Up"; then
    echo "❌ Some containers are not running"
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" ps
    exit 1
fi
echo "✅ All containers are running"

# Check Redis connectivity
echo "📊 Checking Redis connectivity..."
if ! docker exec optimize-redis redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not responding"
    exit 1
fi
echo "✅ Redis is healthy"

# Check dashboard accessibility
echo "🖥️  Checking dashboard accessibility..."
if ! curl -f "$DASHBOARD_URL/api/v1/status" > /dev/null 2>&1; then
    echo "❌ Dashboard is not accessible"
    exit 1
fi
echo "✅ Dashboard is accessible"

# Check orchestrator health
echo "🛡️ Checking orchestrator health..."
ORCHESTRATOR_STATUS=$(curl -s "$DASHBOARD_URL/api/v1/status" | jq -r '.platform.running' 2>/dev/null || echo "false")
if [ "$ORCHESTRATOR_STATUS" != "true" ]; then
    echo "❌ Orchestrator is not healthy"
    exit 1
fi
echo "✅ Orchestrator is healthy"

echo "🎉 All health checks passed!"
EOF

    # Create log monitoring script
    cat > "$SCRIPT_DIR/monitor-logs.sh" << 'EOF'
#!/bin/bash

# OPTIMIZE Platform Log Monitoring Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📋 OPTIMIZE Platform Log Monitor"
echo "==============================="

# Function to show logs for a specific service
show_service_logs() {
    local service=$1
    local lines=${2:-50}

    echo "📄 $service logs (last $lines lines):"
    echo "----------------------------------------"
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" logs --tail "$lines" "$service"
    echo ""
}

# Function to follow all logs
follow_all_logs() {
    echo "📡 Following all platform logs..."
    echo "Press Ctrl+C to stop"
    echo "================================"
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" logs -f
}

# Main menu
case "${1:-menu}" in
    orchestrator)
        show_service_logs "optimize-orchestrator" "${2:-50}"
        ;;
    dashboard)
        show_service_logs "optimize-dashboard" "${2:-50}"
        ;;
    redis)
        show_service_logs "redis" "${2:-50}"
        ;;
    falco)
        show_service_logs "falco" "${2:-50}"
        ;;
    all)
        follow_all_logs
        ;;
    *)
        echo "Usage: $0 {orchestrator|dashboard|redis|falco|all} [lines]"
        echo ""
        echo "Examples:"
        echo "  $0 orchestrator 100  # Show last 100 lines of orchestrator logs"
        echo "  $0 all               # Follow all logs in real-time"
        ;;
esac
EOF

    # Make scripts executable
    chmod +x "$SCRIPT_DIR/health-check.sh"
    chmod +x "$SCRIPT_DIR/monitor-logs.sh"

    log_success "Monitoring scripts created"
}

deploy_platform() {
    log_info "Deploying OPTIMIZE platform..."

    # Build and start services
    cd "$SCRIPT_DIR"

    # Pull base images
    docker-compose pull redis

    # Build custom images
    docker-compose build

    # Start core services
    docker-compose up -d redis

    # Wait for Redis to be healthy
    log_info "Waiting for Redis to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec optimize-redis redis-cli ping > /dev/null 2>&1; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        log_error "Redis failed to start within timeout"
        exit 1
    fi

    # Start orchestrator
    docker-compose up -d optimize-orchestrator

    # Wait for orchestrator to be ready
    log_info "Waiting for orchestrator to be ready..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if docker exec optimize-orchestrator python3 -c "
import requests
try:
    r = requests.get('http://localhost:8080/api/v1/status', timeout=5)
    if r.status_code == 200:
        exit(0)
except:
    pass
exit(1)
        " > /dev/null 2>&1; then
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done

    if [ $timeout -le 0 ]; then
        log_warning "Orchestrator health check timeout, but continuing..."
    fi

    # Start dashboard
    docker-compose up -d optimize-dashboard

    # Optionally start security tools
    if [ "${DEPLOY_SECURITY_TOOLS:-false}" = "true" ]; then
        log_info "Starting security tools..."
        docker-compose --profile security-tools up -d
    fi

    log_success "Platform deployment completed"
}

show_status() {
    log_info "OPTIMIZE Platform Status"
    echo "========================"

    # Show container status
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" ps

    echo ""
    echo "🌐 Access URLs:"
    echo "  Dashboard: http://localhost:${DASHBOARD_PORT:-8080}"
    echo "  Executive: http://localhost:${DASHBOARD_PORT:-8080}/executive"
    echo "  Operations: http://localhost:${DASHBOARD_PORT:-8080}/operations"
    echo "  Technical: http://localhost:${DASHBOARD_PORT:-8080}/technical"
    echo "  API Status: http://localhost:${DASHBOARD_PORT:-8080}/api/v1/status"
    echo ""
    echo "🔧 Management Commands:"
    echo "  Health Check: $SCRIPT_DIR/health-check.sh"
    echo "  View Logs: $SCRIPT_DIR/monitor-logs.sh"
    echo "  Stop Platform: docker-compose -f $SCRIPT_DIR/docker-compose.yml down"
    echo ""
}

cleanup() {
    log_info "Cleaning up deployment artifacts..."

    cd "$SCRIPT_DIR"

    # Stop and remove containers
    docker-compose down --volumes

    # Remove images (optional)
    if [ "${CLEANUP_IMAGES:-false}" = "true" ]; then
        docker-compose down --rmi all
    fi

    log_success "Cleanup completed"
}

# Main deployment logic
main() {
    case "${1:-deploy}" in
        deploy)
            check_prerequisites
            create_directories
            generate_configuration
            create_docker_compose
            create_dockerfile
            create_systemd_service
            setup_monitoring_scripts
            deploy_platform
            show_status
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        health)
            "$SCRIPT_DIR/health-check.sh"
            ;;
        logs)
            "$SCRIPT_DIR/monitor-logs.sh" "${2:-all}"
            ;;
        *)
            echo "Usage: $0 {deploy|status|cleanup|health|logs}"
            echo ""
            echo "Commands:"
            echo "  deploy  - Deploy the complete OPTIMIZE platform"
            echo "  status  - Show platform status and access URLs"
            echo "  cleanup - Stop and remove all platform components"
            echo "  health  - Run health checks"
            echo "  logs    - View platform logs"
            echo ""
            echo "Environment Variables:"
            echo "  ENVIRONMENT=production|development"
            echo "  LOG_LEVEL=DEBUG|INFO|WARNING|ERROR"
            echo "  REDIS_PORT=6379"
            echo "  DASHBOARD_PORT=8080"
            echo "  DEPLOY_SECURITY_TOOLS=true|false"
            echo "  SLACK_WEBHOOK_URL=https://hooks.slack.com/..."
            echo "  SECURITY_EMAIL=security@company.com"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
