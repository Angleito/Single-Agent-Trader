#!/bin/bash
# VPS Deployment Script for AI Trading Bot
# Supports both Coinbase and Bluefin exchanges

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_USER=${DEPLOY_USER:-ubuntu}
DEPLOY_HOST=${DEPLOY_HOST:-}
DEPLOY_PATH=${DEPLOY_PATH:-/home/$DEPLOY_USER/ai-trading-bot}
EXCHANGE_TYPE=${EXCHANGE_TYPE:-coinbase}
ENVIRONMENT=${ENVIRONMENT:-production}

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

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    if [ -z "$DEPLOY_HOST" ]; then
        print_error "DEPLOY_HOST not set. Please export DEPLOY_HOST=your.vps.ip"
        exit 1
    fi
    
    # Check SSH access
    if ! ssh -o ConnectTimeout=5 $DEPLOY_USER@$DEPLOY_HOST "echo 'SSH OK'" > /dev/null 2>&1; then
        print_error "Cannot connect to $DEPLOY_USER@$DEPLOY_HOST"
        exit 1
    fi
    print_success "SSH connection OK"
    
    # Check Docker on VPS
    if ! ssh $DEPLOY_USER@$DEPLOY_HOST "docker --version" > /dev/null 2>&1; then
        print_error "Docker not installed on VPS"
        exit 1
    fi
    print_success "Docker installed on VPS"
    
    # Check docker-compose on VPS
    if ! ssh $DEPLOY_USER@$DEPLOY_HOST "docker compose version || docker-compose --version" > /dev/null 2>&1; then
        print_error "Docker Compose not installed on VPS"
        exit 1
    fi
    print_success "Docker Compose installed on VPS"
}

# Build Docker images locally
build_images() {
    print_header "Building Docker Images"
    
    # Build main bot image
    echo "Building $EXCHANGE_TYPE image..."
    docker build \
        --build-arg EXCHANGE_TYPE=$EXCHANGE_TYPE \
        --build-arg VERSION=$(git describe --tags --always) \
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        --build-arg VCS_REF=$(git rev-parse --short HEAD) \
        -t ai-trading-bot:$EXCHANGE_TYPE-latest \
        -f Dockerfile .
    
    print_success "Built ai-trading-bot:$EXCHANGE_TYPE-latest"
    
    # Build MCP image if needed
    if [ -f "Dockerfile.mcp" ]; then
        echo "Building MCP memory server image..."
        docker build -t ai-trading-bot-mcp:latest -f Dockerfile.mcp .
        print_success "Built ai-trading-bot-mcp:latest"
    fi
}

# Save and transfer images
transfer_images() {
    print_header "Transferring Docker Images"
    
    # Create temp directory for images
    mkdir -p /tmp/ai-trading-bot-deploy
    
    # Save images
    echo "Saving Docker images..."
    docker save ai-trading-bot:$EXCHANGE_TYPE-latest | gzip > /tmp/ai-trading-bot-deploy/ai-trading-bot-$EXCHANGE_TYPE.tar.gz
    
    if docker images | grep -q "ai-trading-bot-mcp"; then
        docker save ai-trading-bot-mcp:latest | gzip > /tmp/ai-trading-bot-deploy/ai-trading-bot-mcp.tar.gz
    fi
    
    # Transfer images to VPS
    echo "Transferring images to VPS..."
    scp /tmp/ai-trading-bot-deploy/*.tar.gz $DEPLOY_USER@$DEPLOY_HOST:/tmp/
    print_success "Images transferred"
    
    # Load images on VPS
    echo "Loading images on VPS..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "docker load < /tmp/ai-trading-bot-$EXCHANGE_TYPE.tar.gz"
    
    if [ -f "/tmp/ai-trading-bot-deploy/ai-trading-bot-mcp.tar.gz" ]; then
        ssh $DEPLOY_USER@$DEPLOY_HOST "docker load < /tmp/ai-trading-bot-mcp.tar.gz"
    fi
    
    # Cleanup
    rm -rf /tmp/ai-trading-bot-deploy
    ssh $DEPLOY_USER@$DEPLOY_HOST "rm -f /tmp/ai-trading-bot*.tar.gz"
    
    print_success "Images loaded on VPS"
}

# Deploy application files
deploy_files() {
    print_header "Deploying Application Files"
    
    # Create deployment directory on VPS
    ssh $DEPLOY_USER@$DEPLOY_HOST "mkdir -p $DEPLOY_PATH/{config,logs,data,scripts,monitoring}"
    
    # Copy docker-compose files
    if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
        scp docker-compose.bluefin.yml $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml
    else
        scp docker-compose.yml $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/
    fi
    
    # Copy configuration files
    scp -r config/* $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/config/ 2>/dev/null || true
    scp -r prompts $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/ 2>/dev/null || true
    
    # Copy scripts
    scp scripts/*.sh $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/scripts/ 2>/dev/null || true
    
    # Copy monitoring config if exists
    if [ -d "monitoring" ]; then
        scp -r monitoring/* $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/monitoring/ 2>/dev/null || true
    fi
    
    print_success "Application files deployed"
}

# Setup environment
setup_environment() {
    print_header "Setting Up Environment"
    
    # Check for .env file
    if [ "$EXCHANGE_TYPE" = "bluefin" ] && [ -f ".env.bluefin" ]; then
        ENV_FILE=".env.bluefin"
    elif [ -f ".env.$ENVIRONMENT" ]; then
        ENV_FILE=".env.$ENVIRONMENT"
    elif [ -f ".env" ]; then
        ENV_FILE=".env"
    else
        print_error "No .env file found!"
        print_warning "Please create .env file with your configuration"
        exit 1
    fi
    
    # Copy environment file
    scp $ENV_FILE $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/.env
    print_success "Environment file deployed"
    
    # Set proper permissions
    ssh $DEPLOY_USER@$DEPLOY_HOST "chmod 600 $DEPLOY_PATH/.env"
}

# Start services
start_services() {
    print_header "Starting Services"
    
    # Navigate to deploy path and start services
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose down || true"
    
    # Start based on profile
    if [ "$MCP_ENABLED" = "true" ]; then
        ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose --profile with-memory up -d"
    else
        ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose up -d"
    fi
    
    print_success "Services started"
    
    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    sleep 10
    
    # Check service status
    ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose ps"
}

# Setup monitoring (optional)
setup_monitoring() {
    if [ "$ENABLE_MONITORING" = "true" ]; then
        print_header "Setting Up Monitoring"
        
        ssh $DEPLOY_USER@$DEPLOY_HOST "cd $DEPLOY_PATH && docker compose --profile monitoring up -d"
        
        print_success "Monitoring services started"
        print_warning "Grafana available at http://$DEPLOY_HOST:3000 (admin/admin)"
        print_warning "Prometheus available at http://$DEPLOY_HOST:9090"
    fi
}

# Setup systemd service (optional)
setup_systemd() {
    if [ "$SETUP_SYSTEMD" = "true" ]; then
        print_header "Setting Up Systemd Service"
        
        # Create systemd service file
        cat > /tmp/ai-trading-bot.service <<EOF
[Unit]
Description=AI Trading Bot ($EXCHANGE_TYPE)
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_PATH
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        
        # Copy and enable service
        scp /tmp/ai-trading-bot.service $DEPLOY_USER@$DEPLOY_HOST:/tmp/
        ssh $DEPLOY_USER@$DEPLOY_HOST "sudo mv /tmp/ai-trading-bot.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable ai-trading-bot.service"
        
        rm /tmp/ai-trading-bot.service
        print_success "Systemd service configured"
    fi
}

# Main deployment flow
main() {
    print_header "AI Trading Bot VPS Deployment"
    echo "Exchange: $EXCHANGE_TYPE"
    echo "Environment: $ENVIRONMENT"
    echo "Target: $DEPLOY_USER@$DEPLOY_HOST"
    echo "Path: $DEPLOY_PATH"
    
    # Confirm deployment
    read -p "Continue with deployment? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi
    
    # Run deployment steps
    check_prerequisites
    build_images
    transfer_images
    deploy_files
    setup_environment
    start_services
    setup_monitoring
    setup_systemd
    
    print_header "Deployment Complete!"
    print_success "AI Trading Bot deployed successfully"
    
    # Show logs command
    echo -e "\nTo view logs:"
    echo "ssh $DEPLOY_USER@$DEPLOY_HOST 'cd $DEPLOY_PATH && docker compose logs -f ai-trading-bot'"
    
    # Show status command
    echo -e "\nTo check status:"
    echo "ssh $DEPLOY_USER@$DEPLOY_HOST 'cd $DEPLOY_PATH && docker compose ps'"
    
    # Security reminder
    print_warning "\nSecurity Reminders:"
    print_warning "- Ensure VPS firewall is configured"
    print_warning "- Use strong SSH keys only (no passwords)"
    print_warning "- Keep your .env file secure"
    print_warning "- Monitor your trading activity regularly"
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
        --with-monitoring)
            ENABLE_MONITORING=true
            shift
            ;;
        --with-systemd)
            SETUP_SYSTEMD=true
            shift
            ;;
        --with-memory)
            MCP_ENABLED=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --host HOST          VPS hostname or IP (required)"
            echo "  --user USER          SSH user (default: ubuntu)"
            echo "  --path PATH          Deployment path (default: /home/USER/ai-trading-bot)"
            echo "  --exchange TYPE      Exchange type: coinbase or bluefin (default: coinbase)"
            echo "  --env ENV           Environment: production, staging (default: production)"
            echo "  --with-monitoring    Enable Prometheus/Grafana monitoring"
            echo "  --with-systemd       Setup systemd service for auto-start"
            echo "  --with-memory        Enable MCP memory system"
            echo "  --help              Show this help message"
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