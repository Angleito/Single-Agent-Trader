#!/bin/bash
# Enhanced VPS Deployment Script for AI Trading Bot
# Optimized for 1-core VPS with memory constraints and real-time monitoring
# Supports both Coinbase and Bluefin exchanges

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_USER=${DEPLOY_USER:-ubuntu}
DEPLOY_HOST=${DEPLOY_HOST:-}
DEPLOY_PATH=${DEPLOY_PATH:-/home/$DEPLOY_USER/ai-trading-bot}
EXCHANGE_TYPE=${EXCHANGE_TYPE:-coinbase}
ENVIRONMENT=${ENVIRONMENT:-production}
VPS_TYPE=${VPS_TYPE:-1core}  # 1core, 2core, etc.
ENABLE_MONITORING=${ENABLE_MONITORING:-true}
ENABLE_ALERTS=${ENABLE_ALERTS:-true}

# Memory optimization settings for 1-core VPS
MEMORY_LIMIT_BOT="512M"
MEMORY_LIMIT_SERVICE="256M"
MEMORY_LIMIT_DASHBOARD="128M"
CPU_LIMIT="1.0"

# Functions
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
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

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Enhanced prerequisites check for 1-core VPS
check_prerequisites() {
    print_header "Checking Prerequisites for 1-Core VPS"

    if [ -z "$DEPLOY_HOST" ]; then
        print_error "DEPLOY_HOST not set. Please export DEPLOY_HOST=your.vps.ip"
        exit 1
    fi

    # Check SSH access
    print_info "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=5 $DEPLOY_USER@$DEPLOY_HOST "echo 'SSH OK'" > /dev/null 2>&1; then
        print_error "Cannot connect to $DEPLOY_USER@$DEPLOY_HOST"
        exit 1
    fi
    print_success "SSH connection OK"

    # Check VPS resources
    print_info "Checking VPS resources..."
    VPS_MEMORY=$(ssh $DEPLOY_USER@$DEPLOY_HOST "free -m | awk '/^Mem:/{print \$2}'")
    VPS_CORES=$(ssh $DEPLOY_USER@$DEPLOY_HOST "nproc")
    VPS_DISK=$(ssh $DEPLOY_USER@$DEPLOY_HOST "df -h / | awk 'NR==2{print \$4}'")

    print_info "VPS Specs: ${VPS_CORES} cores, ${VPS_MEMORY}MB RAM, ${VPS_DISK} disk available"

    if [ "$VPS_MEMORY" -lt 900 ]; then
        print_warning "Low memory detected (${VPS_MEMORY}MB). Enabling aggressive memory optimization."
        MEMORY_LIMIT_BOT="256M"
        MEMORY_LIMIT_SERVICE="128M"
        MEMORY_LIMIT_DASHBOARD="64M"
    fi

    # Check Docker
    if ! ssh $DEPLOY_USER@$DEPLOY_HOST "docker --version" > /dev/null 2>&1; then
        print_error "Docker not installed on VPS"
        exit 1
    fi
    print_success "Docker installed"

    # Check docker-compose
    if ! ssh $DEPLOY_USER@$DEPLOY_HOST "docker compose version || docker-compose --version" > /dev/null 2>&1; then
        print_error "Docker Compose not installed on VPS"
        exit 1
    fi
    print_success "Docker Compose installed"

    # Check available disk space
    AVAILABLE_SPACE=$(ssh $DEPLOY_USER@$DEPLOY_HOST "df / | awk 'NR==2{print \$4}'")
    if [ "$AVAILABLE_SPACE" -lt 2000000 ]; then  # Less than 2GB
        print_warning "Low disk space detected. Cleanup may be needed."
    fi
}

# Build optimized images for 1-core VPS
build_optimized_images() {
    print_header "Building Memory-Optimized Docker Images"

    # Build slim bot image
    print_info "Building optimized trading bot image..."
    docker build \
        --build-arg EXCHANGE_TYPE=$EXCHANGE_TYPE \
        --build-arg VERSION=$(git describe --tags --always) \
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        --build-arg VCS_REF=$(git rev-parse --short HEAD) \
        --build-arg PYTHON_VERSION=3.12 \
        --build-arg MEMORY_LIMIT=$MEMORY_LIMIT_BOT \
        -t ai-trading-bot:slim \
        -f Dockerfile.slim .

    print_success "Built optimized ai-trading-bot:slim"

    # Build service image if needed
    if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
        print_info "Building Bluefin service image..."
        docker build \
            --build-arg MEMORY_LIMIT=$MEMORY_LIMIT_SERVICE \
            -t bluefin-service:slim \
            -f services/Dockerfile.bluefin .
        print_success "Built bluefin-service:slim"
    fi

    # Build dashboard images
    print_info "Building dashboard images..."
    docker build -t dashboard-backend:slim -f dashboard/backend/Dockerfile dashboard/backend/
    docker build -t dashboard-frontend:slim -f dashboard/frontend/Dockerfile dashboard/frontend/
    print_success "Built dashboard images"
}

# Transfer optimized images
transfer_optimized_images() {
    print_header "Transferring Optimized Images"

    # Create temp directory
    mkdir -p /tmp/ai-trading-bot-deploy

    # Save images with compression
    print_info "Compressing and saving images..."
    docker save ai-trading-bot:slim | gzip -9 > /tmp/ai-trading-bot-deploy/ai-trading-bot-slim.tar.gz
    docker save dashboard-backend:slim | gzip -9 > /tmp/ai-trading-bot-deploy/dashboard-backend-slim.tar.gz

    if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
        docker save bluefin-service:slim | gzip -9 > /tmp/ai-trading-bot-deploy/bluefin-service-slim.tar.gz
    fi

    # Transfer with progress
    print_info "Transferring images to VPS..."
    scp -r /tmp/ai-trading-bot-deploy/*.tar.gz $DEPLOY_USER@$DEPLOY_HOST:/tmp/

    # Load images on VPS
    print_info "Loading images on VPS..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "docker load < /tmp/ai-trading-bot-slim.tar.gz"
    ssh $DEPLOY_USER@$DEPLOY_HOST "docker load < /tmp/dashboard-backend-slim.tar.gz"

    if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
        ssh $DEPLOY_USER@$DEPLOY_HOST "docker load < /tmp/bluefin-service-slim.tar.gz"
    fi

    # Cleanup
    rm -rf /tmp/ai-trading-bot-deploy
    ssh $DEPLOY_USER@$DEPLOY_HOST "rm -f /tmp/*.tar.gz"

    print_success "Images loaded on VPS"
}

# Deploy optimized configuration
deploy_optimized_config() {
    print_header "Deploying Optimized Configuration"

    # Create deployment directory structure
    ssh $DEPLOY_USER@$DEPLOY_HOST "mkdir -p $DEPLOY_PATH/{config,logs,data,scripts,monitoring,backup}"

    # Generate optimized docker-compose file
    print_info "Generating memory-optimized docker-compose.yml..."

    cat > /tmp/docker-compose-1core.yml << EOF
version: '3.8'

services:
  ai-trading-bot:
    image: ai-trading-bot:slim
    container_name: ai-trading-bot
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: $MEMORY_LIMIT_BOT
          cpus: '$CPU_LIMIT'
        reservations:
          memory: 128M
          cpus: '0.5'
    environment:
      - ENABLE_MEMORY_OPTIMIZATION=true
      - MAX_MEMORY_MB=${MEMORY_LIMIT_BOT%M}
      - PYTHONOPTIMIZE=2
      - MALLOC_TRIM_THRESHOLD_=100000
      - LOG_LEVEL=\${LOG_LEVEL:-INFO}
      - SYSTEM__DRY_RUN=\${SYSTEM__DRY_RUN:-true}
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
    ports:
      - "8080:8080"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health', timeout=5)"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 30s

  dashboard-backend:
    image: dashboard-backend:slim
    container_name: dashboard-backend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: $MEMORY_LIMIT_DASHBOARD
          cpus: '0.5'
        reservations:
          memory: 32M
          cpus: '0.1'
    environment:
      - DASHBOARD_HOST=0.0.0.0
      - DASHBOARD_PORT=8000
      - LOG_LEVEL=\${LOG_LEVEL:-INFO}
    volumes:
      - ./logs:/app/logs:ro
      - ./data:/app/data:ro
    ports:
      - "8000:8000"
    networks:
      - trading-network
    depends_on:
      - ai-trading-bot
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

EOF

    # Add Bluefin service if needed
    if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
        cat >> /tmp/docker-compose-1core.yml << EOF
  bluefin-service:
    image: bluefin-service:slim
    container_name: bluefin-service
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: $MEMORY_LIMIT_SERVICE
          cpus: '0.5'
        reservations:
          memory: 64M
          cpus: '0.2'
    environment:
      - BLUEFIN_PRIVATE_KEY=\${EXCHANGE__BLUEFIN_PRIVATE_KEY}
      - BLUEFIN_SERVICE_API_KEY=\${BLUEFIN_SERVICE_API_KEY}
      - EXCHANGE__BLUEFIN_NETWORK=\${EXCHANGE__BLUEFIN_NETWORK:-testnet}
      - SYSTEM__DRY_RUN=\${SYSTEM__DRY_RUN:-true}
      - LOG_LEVEL=\${LOG_LEVEL:-INFO}
    volumes:
      - ./logs/bluefin:/app/logs
    ports:
      - "8082:8080"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3

EOF
    fi

    # Add monitoring if enabled
    if [ "$ENABLE_MONITORING" = "true" ]; then
        cat >> /tmp/docker-compose-1core.yml << EOF
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 128M
          cpus: '0.3'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    networks:
      - trading-network
    profiles:
      - monitoring

EOF
    fi

    # Complete the compose file
    cat >> /tmp/docker-compose-1core.yml << EOF
networks:
  trading-network:
    driver: bridge

volumes:
  prometheus-data:
EOF

    # Copy optimized compose file
    scp /tmp/docker-compose-1core.yml $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml
    rm /tmp/docker-compose-1core.yml

    # Copy configuration files
    scp -r config/* $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/config/ 2>/dev/null || true
    scp -r prompts $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/ 2>/dev/null || true

    # Copy monitoring configurations
    if [ "$ENABLE_MONITORING" = "true" ]; then
        scp scripts/prometheus.yml $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/monitoring/
        scp scripts/vps-monitor.py $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/scripts/
        ssh $DEPLOY_USER@$DEPLOY_HOST "chmod +x $DEPLOY_PATH/scripts/vps-monitor.py"
    fi

    print_success "Optimized configuration deployed"
}

# Setup environment with memory optimization
setup_optimized_environment() {
    print_header "Setting Up Optimized Environment"

    # Select environment file
    if [ "$EXCHANGE_TYPE" = "bluefin" ] && [ -f ".env.bluefin" ]; then
        ENV_FILE=".env.bluefin"
    elif [ -f ".env.$ENVIRONMENT" ]; then
        ENV_FILE=".env.$ENVIRONMENT"
    elif [ -f ".env" ]; then
        ENV_FILE=".env"
    else
        print_error "No .env file found!"
        exit 1
    fi

    # Create optimized environment file
    cp $ENV_FILE /tmp/.env.optimized

    # Add memory optimization settings
    cat >> /tmp/.env.optimized << EOF

# Memory optimization for 1-core VPS
ENABLE_MEMORY_OPTIMIZATION=true
MAX_MEMORY_MB=${MEMORY_LIMIT_BOT%M}
PYTHONOPTIMIZE=2
MALLOC_TRIM_THRESHOLD_=100000

# Reduce concurrent operations
SYSTEM__MAX_CONCURRENT_TASKS=2
SYSTEM__THREAD_POOL_SIZE=2
SYSTEM__WEBSOCKET_QUEUE_SIZE=100
FP_MAX_CONCURRENT_EFFECTS=10

# Conservative trading settings for stability
RISK__MAX_CONCURRENT_TRADES=1
TRADING__ORDER_TIMEOUT_SECONDS=30

EOF

    # Copy optimized environment
    scp /tmp/.env.optimized $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/.env
    rm /tmp/.env.optimized

    # Set secure permissions
    ssh $DEPLOY_USER@$DEPLOY_HOST "chmod 600 $DEPLOY_PATH/.env"
    print_success "Optimized environment configured"
}

# Start services with monitoring
start_optimized_services() {
    print_header "Starting Optimized Services"

    # Stop any existing containers
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose down || true"

    # Start core services first
    print_info "Starting core trading services..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose up -d ai-trading-bot dashboard-backend"

    if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
        print_info "Starting Bluefin service..."
        ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose up -d bluefin-service"
    fi

    # Wait for core services
    print_info "Waiting for services to initialize..."
    sleep 20

    # Start monitoring if enabled
    if [ "$ENABLE_MONITORING" = "true" ]; then
        print_info "Starting monitoring services..."
        ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose --profile monitoring up -d || true"

        # Start real-time monitor in background
        print_info "Starting real-time VPS monitor..."
        ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && nohup python3 scripts/vps-monitor.py --interval 10 > /tmp/vps-monitor.out 2>&1 &"
    fi

    print_success "Services started"

    # Check service status
    print_info "Checking service health..."
    sleep 10
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose ps"
}

# Setup monitoring and alerts
setup_comprehensive_monitoring() {
    if [ "$ENABLE_MONITORING" != "true" ]; then
        return
    fi

    print_header "Setting Up Comprehensive Monitoring"

    # Install required monitoring dependencies
    ssh $DEPLOY_USER@$DEPLOY_HOST "pip3 install psutil requests || sudo apt-get update && sudo apt-get install -y python3-pip && pip3 install psutil requests"

    # Create monitoring scripts
    print_info "Setting up monitoring scripts..."

    # Create systemd service for monitoring
    cat > /tmp/vps-monitor.service << EOF
[Unit]
Description=VPS Monitor for AI Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$DEPLOY_PATH
ExecStart=/usr/bin/python3 $DEPLOY_PATH/scripts/vps-monitor.py --interval 10
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Install monitoring service
    scp /tmp/vps-monitor.service $DEPLOY_USER@$DEPLOY_HOST:/tmp/
    ssh $DEPLOY_USER@$DEPLOY_HOST "sudo mv /tmp/vps-monitor.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable vps-monitor.service"
    rm /tmp/vps-monitor.service

    print_success "Monitoring system configured"
}

# Health check and optimization
perform_health_check() {
    print_header "Performing Comprehensive Health Check"

    print_info "Checking container health..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose ps"

    print_info "Checking resource usage..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "free -h && df -h && top -bn1 | head -10"

    print_info "Checking application logs..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose logs --tail=20 ai-trading-bot"

    # Test API endpoints
    print_info "Testing API endpoints..."
    if ssh $DEPLOY_USER@$DEPLOY_HOST "curl -f http://localhost:8080/health" > /dev/null 2>&1; then
        print_success "Main bot API healthy"
    else
        print_warning "Main bot API not responding"
    fi

    if ssh $DEPLOY_USER@$DEPLOY_HOST "curl -f http://localhost:8000/health" > /dev/null 2>&1; then
        print_success "Dashboard API healthy"
    else
        print_warning "Dashboard API not responding"
    fi
}

# Main deployment flow
main() {
    print_header "AI Trading Bot 1-Core VPS Optimized Deployment"
    echo -e "${PURPLE}Exchange: $EXCHANGE_TYPE${NC}"
    echo -e "${PURPLE}Environment: $ENVIRONMENT${NC}"
    echo -e "${PURPLE}Target: $DEPLOY_USER@$DEPLOY_HOST${NC}"
    echo -e "${PURPLE}Path: $DEPLOY_PATH${NC}"
    echo -e "${PURPLE}VPS Type: $VPS_TYPE${NC}"
    echo -e "${PURPLE}Memory Limits: Bot=$MEMORY_LIMIT_BOT, Service=$MEMORY_LIMIT_SERVICE${NC}"

    # Confirm deployment
    read -p "Continue with optimized 1-core deployment? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi

    # Run deployment steps
    check_prerequisites
    build_optimized_images
    transfer_optimized_images
    deploy_optimized_config
    setup_optimized_environment
    start_optimized_services
    setup_comprehensive_monitoring
    perform_health_check

    print_header "🎉 Optimized Deployment Complete!"
    print_success "AI Trading Bot deployed successfully on 1-core VPS"

    echo -e "\n${GREEN}📊 MONITORING COMMANDS:${NC}"
    echo "Real-time monitor:  ssh $DEPLOY_USER@$DEPLOY_HOST 'cd $DEPLOY_PATH && python3 scripts/vps-monitor.py'"
    echo "View logs:          ssh $DEPLOY_USER@$DEPLOY_HOST 'cd $DEPLOY_PATH && docker compose logs -f ai-trading-bot'"
    echo "Check status:       ssh $DEPLOY_USER@$DEPLOY_HOST 'cd $DEPLOY_PATH && docker compose ps'"
    echo "System resources:   ssh $DEPLOY_USER@$DEPLOY_HOST 'free -h && df -h'"

    echo -e "\n${GREEN}🌐 WEB INTERFACES:${NC}"
    echo "Dashboard:          http://$DEPLOY_HOST:8000"
    echo "Trading Bot API:    http://$DEPLOY_HOST:8080"
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "Prometheus:         http://$DEPLOY_HOST:9090"
    fi

    echo -e "\n${YELLOW}⚠️  NEXT STEPS:${NC}"
    echo "1. Monitor system resources closely for the first 24 hours"
    echo "2. Check trading performance and adjust settings if needed"
    echo "3. Verify all containers remain stable under load"
    echo "4. Consider upgrading VPS if performance issues occur"

    print_warning "\n🔒 SECURITY REMINDERS:"
    print_warning "- Monitor your VPS resources regularly"
    print_warning "- Keep your environment variables secure"
    print_warning "- Test thoroughly before enabling live trading"
    print_warning "- Set up automated backups of your configuration"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            DEPLOY_HOST="$2"
            shift 2
            ;;
        --user)
            DEPLOY_USER="$2"
            shift 2
            ;;
        --path)
            DEPLOY_PATH="$2"
            shift 2
            ;;
        --exchange)
            EXCHANGE_TYPE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --memory-bot)
            MEMORY_LIMIT_BOT="$2"
            shift 2
            ;;
        --memory-service)
            MEMORY_LIMIT_SERVICE="$2"
            shift 2
            ;;
        --cpu-limit)
            CPU_LIMIT="$2"
            shift 2
            ;;
        --enable-monitoring)
            ENABLE_MONITORING=true
            shift
            ;;
        --disable-monitoring)
            ENABLE_MONITORING=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --host HOST              VPS hostname or IP (required)"
            echo "  --user USER              SSH user (default: ubuntu)"
            echo "  --path PATH              Deployment path (default: /home/USER/ai-trading-bot)"
            echo "  --exchange TYPE          Exchange type: coinbase or bluefin (default: coinbase)"
            echo "  --env ENV               Environment: production, staging (default: production)"
            echo "  --memory-bot SIZE        Memory limit for bot container (default: 512M)"
            echo "  --memory-service SIZE    Memory limit for service container (default: 256M)"
            echo "  --cpu-limit CORES        CPU limit for containers (default: 1.0)"
            echo "  --enable-monitoring      Enable comprehensive monitoring (default)"
            echo "  --disable-monitoring     Disable monitoring"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main deployment
main
