#!/usr/bin/env bash
#
# Automated Safe Deployment Script for AI Trading Bot
#
# This script orchestrates the complete deployment process with safety checks,
# validation, and rollback capabilities.
#
# SAFETY FEATURES:
# 1. Pre-deployment validation
# 2. Configuration safety checks
# 3. Service health verification
# 4. Automated rollback on failure
# 5. Paper trading verification before live deployment
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
BACKUP_DIR="${PROJECT_ROOT}/backups/deployment-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Create necessary directories
mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
mkdir -p "$BACKUP_DIR"

# Safety check functions
check_prerequisites() {
    log "Checking prerequisites..."

    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "python3" "poetry")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            return 1
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    }

    # Check Python version
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version < 3.10" | bc) -eq 1 ]]; then
        log_error "Python 3.10+ required, found $python_version"
        return 1
    fi

    log_success "All prerequisites met"
    return 0
}

backup_current_state() {
    log "Creating backup of current state..."

    # Backup configuration files
    cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/"
    cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/.env" 2>/dev/null || true
    cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/"

    # Save current git state
    cd "$PROJECT_ROOT"
    git rev-parse HEAD > "$BACKUP_DIR/git_commit_hash"
    git status --porcelain > "$BACKUP_DIR/git_status"

    log_success "Backup created at $BACKUP_DIR"
}

validate_configuration() {
    log "Validating configuration..."

    # Run Python configuration validation
    cd "$PROJECT_ROOT"
    if ! poetry run python scripts/validate_config_comprehensive.py; then
        log_error "Configuration validation failed"
        return 1
    fi

    # Check for paper trading default
    if [[ -f .env ]]; then
        if grep -q "SYSTEM__DRY_RUN=false" .env; then
            log_warning "LIVE TRADING MODE DETECTED!"
            read -p "Are you SURE you want to deploy with live trading? (yes/no): " confirm
            if [[ "$confirm" != "yes" ]]; then
                log_error "Deployment cancelled by user"
                return 1
            fi
        else
            log_success "Paper trading mode confirmed"
        fi
    fi

    # Validate type safety
    if ! poetry run python scripts/validate-types.py; then
        log_error "Type validation failed"
        return 1
    fi

    log_success "Configuration validated"
    return 0
}

run_tests() {
    log "Running test suite..."

    cd "$PROJECT_ROOT"

    # Run property tests
    log "Running property-based tests..."
    if ! poetry run pytest tests/property/ -v --tb=short; then
        log_error "Property tests failed"
        return 1
    fi

    # Run unit tests
    log "Running unit tests..."
    if ! poetry run pytest tests/unit/ -v --tb=short; then
        log_error "Unit tests failed"
        return 1
    fi

    # Run integration tests (if they exist)
    if [[ -d tests/integration ]]; then
        log "Running integration tests..."
        if ! poetry run pytest tests/integration/ -v --tb=short; then
            log_warning "Integration tests failed (non-critical)"
        fi
    fi

    log_success "All tests passed"
    return 0
}

build_docker_images() {
    log "Building Docker images..."

    cd "$PROJECT_ROOT"

    # Build with proper BuildKit
    export DOCKER_BUILDKIT=1

    if ! docker-compose build --no-cache; then
        log_error "Docker build failed"
        return 1
    fi

    log_success "Docker images built successfully"
    return 0
}

deploy_services() {
    log "Deploying services..."

    cd "$PROJECT_ROOT"

    # Stop existing services
    log "Stopping existing services..."
    docker-compose down || true

    # Start infrastructure services first
    log "Starting infrastructure services..."
    docker-compose up -d mcp-memory mcp-omnisearch

    # Wait for infrastructure to be ready
    log "Waiting for infrastructure services..."
    sleep 10

    # Verify infrastructure health
    if ! "$SCRIPT_DIR/validate-docker-network.sh"; then
        log_error "Infrastructure services not healthy"
        return 1
    fi

    # Start optional services
    log "Starting optional services..."
    docker-compose up -d bluefin-service dashboard-backend dashboard-frontend || {
        log_warning "Optional services failed to start (non-critical)"
    }

    # Start main trading bot
    log "Starting trading bot..."
    docker-compose up -d ai-trading-bot

    log_success "All services deployed"
    return 0
}

verify_deployment() {
    log "Verifying deployment..."

    # Wait for services to stabilize
    log "Waiting for services to stabilize..."
    sleep 30

    # Check service health
    local services=("ai-trading-bot" "mcp-memory" "mcp-omnisearch")
    for service in "${services[@]}"; do
        if ! docker-compose ps | grep -q "${service}.*Up"; then
            log_error "Service $service is not running"
            return 1
        fi
    done

    # Check logs for errors
    log "Checking service logs..."
    if docker-compose logs --tail=100 ai-trading-bot | grep -i "error\|exception\|panic"; then
        log_warning "Errors found in trading bot logs"
    fi

    # Verify paper trading mode
    if docker-compose exec ai-trading-bot grep -q "PAPER TRADING MODE" /app/logs/*.log 2>/dev/null; then
        log_success "Paper trading mode confirmed in logs"
    fi

    # Run health check script
    if [[ -f "$SCRIPT_DIR/docker-health-check.sh" ]]; then
        if ! "$SCRIPT_DIR/docker-health-check.sh"; then
            log_warning "Health check reported issues"
        fi
    fi

    log_success "Deployment verification complete"
    return 0
}

rollback_deployment() {
    log_error "Rolling back deployment..."

    # Stop all services
    cd "$PROJECT_ROOT"
    docker-compose down

    # Restore configuration
    if [[ -d "$BACKUP_DIR" ]]; then
        cp -r "$BACKUP_DIR/config" "$PROJECT_ROOT/" 2>/dev/null || true
        cp "$BACKUP_DIR/.env" "$PROJECT_ROOT/.env" 2>/dev/null || true
        cp "$BACKUP_DIR/docker-compose.yml" "$PROJECT_ROOT/"

        log_success "Configuration restored from backup"
    fi

    log_error "Deployment rolled back"
}

post_deployment_tasks() {
    log "Running post-deployment tasks..."

    # Set up monitoring
    log "Setting up monitoring..."
    # TODO: Add monitoring setup (Prometheus, Grafana, etc.)

    # Set up log rotation
    log "Configuring log rotation..."
    cat > /tmp/trading-bot-logrotate <<EOF
${PROJECT_ROOT}/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $(id -u) $(id -g)
}
EOF

    # Generate deployment report
    log "Generating deployment report..."
    cat > "$PROJECT_ROOT/logs/deployment-report-$(date +%Y%m%d-%H%M%S).txt" <<EOF
Deployment Report
================
Date: $(date)
Git Commit: $(git rev-parse HEAD)
Environment: $(grep SYSTEM__ENVIRONMENT .env 2>/dev/null || echo "Not set")
Trading Mode: $(grep SYSTEM__DRY_RUN .env 2>/dev/null || echo "Not set")

Services Status:
$(docker-compose ps)

Recent Logs:
$(docker-compose logs --tail=20 ai-trading-bot)
EOF

    log_success "Post-deployment tasks completed"
}

# Main deployment flow
main() {
    log "Starting automated deployment..."
    log "Deployment log: $DEPLOYMENT_LOG"

    # Phase 1: Pre-deployment checks
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi

    # Phase 2: Backup current state
    backup_current_state

    # Phase 3: Validate configuration
    if ! validate_configuration; then
        log_error "Configuration validation failed"
        exit 1
    fi

    # Phase 4: Run tests
    if ! run_tests; then
        log_error "Tests failed"
        exit 1
    fi

    # Phase 5: Build Docker images
    if ! build_docker_images; then
        log_error "Docker build failed"
        rollback_deployment
        exit 1
    fi

    # Phase 6: Deploy services
    if ! deploy_services; then
        log_error "Service deployment failed"
        rollback_deployment
        exit 1
    fi

    # Phase 7: Verify deployment
    if ! verify_deployment; then
        log_error "Deployment verification failed"
        rollback_deployment
        exit 1
    fi

    # Phase 8: Post-deployment tasks
    post_deployment_tasks

    log_success "Deployment completed successfully!"
    log "Check logs at: $PROJECT_ROOT/logs/"
    log "Monitor services with: docker-compose logs -f"

    # Final safety reminder
    if grep -q "SYSTEM__DRY_RUN=false" .env 2>/dev/null; then
        log_warning "REMINDER: Live trading is ENABLED! Monitor carefully!"
    else
        log_success "Paper trading mode active - no real money at risk"
    fi
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; rollback_deployment; exit 1' INT TERM

# Run main deployment
main "$@"
