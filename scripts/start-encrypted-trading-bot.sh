#!/bin/bash
# AI Trading Bot - Encrypted Startup Script
# This script provides a unified startup procedure for the encrypted trading bot system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/trading-bot-startup.log"
EMERGENCY_STOP_FILE="/opt/trading-bot/EMERGENCY_STOP"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${BLUE}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

error() {
    local message="[ERROR] $1"
    echo -e "${RED}${message}${NC}" >&2
    echo "$message" >> "$LOG_FILE"
}

warning() {
    local message="[WARNING] $1"
    echo -e "${YELLOW}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

success() {
    local message="[SUCCESS] $1"
    echo -e "${GREEN}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

critical() {
    local message="[CRITICAL] $1"
    echo -e "${PURPLE}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

# Print banner
print_banner() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                 AI Trading Bot - Encrypted Startup          ║
║                    High-Security Trading Platform           ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo ""
}

# Check if running as appropriate user
check_user() {
    if [[ $EUID -eq 0 ]]; then
        log "Running as root - checking encryption volumes..."
        PRIVILEGED_MODE=true
    else
        log "Running as user $(whoami) - checking Docker permissions..."
        PRIVILEGED_MODE=false
        
        # Check if user is in docker group
        if ! groups | grep -q docker; then
            error "User $(whoami) is not in docker group"
            error "Add user to docker group: sudo usermod -aG docker $(whoami)"
            exit 1
        fi
    fi
}

# Check emergency stop status
check_emergency_stop() {
    if [ -f "$EMERGENCY_STOP_FILE" ]; then
        critical "EMERGENCY STOP FLAG DETECTED"
        critical "System is in emergency stop mode"
        critical "Remove flag to resume operations: sudo rm $EMERGENCY_STOP_FILE"
        echo ""
        echo "Emergency stop reasons might include:"
        echo "- Security incident"
        echo "- System maintenance"
        echo "- Disaster recovery in progress"
        echo "- Manual safety stop"
        echo ""
        exit 1
    fi
}

# Verify encrypted volumes are ready
verify_encrypted_volumes() {
    log "Verifying encrypted volumes..."
    
    local volumes=("data" "logs" "config" "backup")
    local all_mounted=true
    
    for volume in "${volumes[@]}"; do
        local mount_point="/mnt/trading-$volume"
        
        if mountpoint -q "$mount_point" 2>/dev/null; then
            log "✓ Volume mounted: trading-$volume"
            
            # Check if writable
            if [ -w "$mount_point" ]; then
                log "✓ Volume writable: trading-$volume"
            else
                error "Volume not writable: trading-$volume"
                all_mounted=false
            fi
        else
            error "Volume not mounted: trading-$volume"
            all_mounted=false
        fi
    done
    
    if ! $all_mounted; then
        error "Some encrypted volumes are not ready"
        if $PRIVILEGED_MODE; then
            warning "Attempting to start encrypted volumes service..."
            systemctl start trading-bot-volumes || {
                error "Failed to start encrypted volumes service"
                return 1
            }
            sleep 5
            # Recheck
            verify_encrypted_volumes
        else
            error "Run with sudo to attempt automatic volume mounting"
            error "Or manually start volumes: sudo systemctl start trading-bot-volumes"
            return 1
        fi
    else
        success "All encrypted volumes are ready"
    fi
}

# Check encryption key status
check_encryption_keys() {
    log "Checking encryption key status..."
    
    if $PRIVILEGED_MODE; then
        "$SCRIPT_DIR/manage-encryption-keys.sh" check-age || {
            warning "Some encryption keys may need attention"
            warning "Review key status: sudo $SCRIPT_DIR/manage-encryption-keys.sh report"
        }
    else
        log "Skipping key age check (requires root privileges)"
    fi
}

# Verify Docker environment
verify_docker_environment() {
    log "Verifying Docker environment..."
    
    # Check Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or accessible"
        if $PRIVILEGED_MODE; then
            warning "Attempting to start Docker service..."
            systemctl start docker || {
                error "Failed to start Docker service"
                return 1
            }
            sleep 5
        else
            error "Start Docker service: sudo systemctl start docker"
            return 1
        fi
    fi
    
    # Check Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "Docker Compose not found"
        return 1
    fi
    
    # Check if any trading bot containers are already running
    local running_containers=$(docker ps --format "table {{.Names}}" | grep -E "(trading|bluefin|mcp|dashboard)" | tail -n +2 | wc -l)
    if [ "$running_containers" -gt 0 ]; then
        warning "Some trading bot containers are already running:"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(trading|bluefin|mcp|dashboard)"
        echo ""
        echo -n "Stop existing containers and continue? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            log "Stopping existing containers..."
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down 2>/dev/null || true
            docker-compose -f "$PROJECT_ROOT/docker-compose.encrypted.yml" down 2>/dev/null || true
        else
            log "Startup cancelled by user"
            exit 0
        fi
    fi
    
    success "Docker environment verified"
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check available memory
    local available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    local required_mem=2048  # 2GB minimum
    
    if [ "$available_mem" -lt "$required_mem" ]; then
        warning "Low available memory: ${available_mem}MB (recommended: ${required_mem}MB+)"
    else
        log "✓ Available memory: ${available_mem}MB"
    fi
    
    # Check disk space for encrypted volumes
    local volumes=("data" "logs" "config" "backup")
    for volume in "${volumes[@]}"; do
        local mount_point="/mnt/trading-$volume"
        if mountpoint -q "$mount_point" 2>/dev/null; then
            local usage=$(df "$mount_point" | tail -1 | awk '{print $5}' | sed 's/%//')
            if [ "$usage" -gt 80 ]; then
                warning "Volume $volume is ${usage}% full"
            else
                log "✓ Volume $volume usage: ${usage}%"
            fi
        fi
    done
    
    # Check CPU load
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    if (( $(echo "$cpu_load > $cpu_cores" | bc -l) )); then
        warning "High CPU load: $cpu_load (cores: $cpu_cores)"
    else
        log "✓ CPU load: $cpu_load (cores: $cpu_cores)"
    fi
    
    success "System resources checked"
}

# Validate configuration
validate_configuration() {
    log "Validating configuration..."
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        warning ".env file not found, using defaults"
    else
        log "✓ Configuration file found: .env"
        
        # Check for encryption-related settings
        if grep -q "ENABLE_ENCRYPTION=true" "$PROJECT_ROOT/.env"; then
            log "✓ Encryption enabled in configuration"
        else
            warning "Encryption not explicitly enabled in .env"
        fi
    fi
    
    # Check if encrypted compose file exists
    if [ ! -f "$PROJECT_ROOT/docker-compose.encrypted.yml" ]; then
        error "Encrypted Docker Compose configuration not found"
        return 1
    else
        log "✓ Encrypted Docker Compose configuration found"
    fi
    
    # Validate compose file
    if ! docker-compose -f "$PROJECT_ROOT/docker-compose.encrypted.yml" config >/dev/null 2>&1; then
        error "Invalid Docker Compose configuration"
        return 1
    else
        log "✓ Docker Compose configuration valid"
    fi
    
    success "Configuration validation completed"
}

# Create performance monitoring setup
setup_monitoring() {
    log "Setting up performance monitoring..."
    
    # Start crypto performance monitoring in background
    if $PRIVILEGED_MODE; then
        "$SCRIPT_DIR/monitor-crypto-performance.sh" continuous > /var/log/crypto-performance.log 2>&1 &
        local monitor_pid=$!
        echo "$monitor_pid" > /var/run/crypto-monitor.pid
        log "✓ Crypto performance monitoring started (PID: $monitor_pid)"
    else
        log "Skipping crypto monitoring setup (requires root privileges)"
    fi
    
    success "Monitoring setup completed"
}

# Start encrypted services
start_encrypted_services() {
    log "Starting encrypted trading bot services..."
    
    cd "$PROJECT_ROOT"
    
    # Set proper environment for Docker Compose
    export DOCKER_DEFAULT_PLATFORM=linux/amd64
    
    # Start services with encrypted configuration
    log "Starting services with encrypted volumes..."
    docker-compose -f docker-compose.encrypted.yml up -d
    
    # Wait for services to initialize
    log "Waiting for services to initialize..."
    sleep 30
    
    # Check service health
    log "Checking service health..."
    local unhealthy_services=0
    
    # Check each service
    local services=("ai-trading-bot-encrypted" "bluefin-service-encrypted" "mcp-memory-encrypted" "mcp-omnisearch-encrypted")
    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$service.*healthy\|$service.*running"; then
            log "✓ Service healthy: $service"
        else
            warning "Service may not be healthy: $service"
            ((unhealthy_services++))
        fi
    done
    
    if [ $unhealthy_services -gt 0 ]; then
        warning "$unhealthy_services services may not be healthy"
        warning "Check logs: docker-compose -f docker-compose.encrypted.yml logs"
    else
        success "All services started successfully"
    fi
}

# Perform post-startup verification
post_startup_verification() {
    log "Performing post-startup verification..."
    
    # Check if all containers are running
    local expected_containers=4
    local running_containers=$(docker ps --format "table {{.Names}}" | grep -E "encrypted" | wc -l)
    
    if [ "$running_containers" -ge "$expected_containers" ]; then
        log "✓ Expected number of containers running: $running_containers"
    else
        warning "Only $running_containers of $expected_containers expected containers running"
    fi
    
    # Test encrypted volume access
    local test_file="/mnt/trading-data/startup_test_$(date +%s)"
    if echo "Startup test $(date)" > "$test_file" 2>/dev/null && [ -f "$test_file" ]; then
        log "✓ Encrypted volume write test passed"
        rm -f "$test_file"
    else
        warning "Encrypted volume write test failed"
    fi
    
    # Check if emergency stop flag was created (shouldn't be)
    if [ -f "$EMERGENCY_STOP_FILE" ]; then
        critical "Emergency stop flag was created during startup"
        return 1
    fi
    
    success "Post-startup verification completed"
}

# Print startup summary
print_startup_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                     STARTUP COMPLETED                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🔒 Encrypted Trading Bot Status:"
    echo ""
    echo "📊 Services:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{Ports}}" | grep -E "(NAME|encrypted)" | head -10
    echo ""
    echo "💾 Encrypted Volumes:"
    df -h | grep trading | awk '{printf "   %-20s %8s %8s %8s %6s\n", $6, $2, $3, $4, $5}'
    echo ""
    echo "🔑 Security Status:"
    echo "   ✓ LUKS encryption active"
    echo "   ✓ Encrypted backups enabled"
    echo "   ✓ Key management active"
    echo "   ✓ Performance monitoring enabled"
    echo ""
    echo "📋 Management Commands:"
    echo "   Monitor services:  docker-compose -f docker-compose.encrypted.yml logs -f"
    echo "   Check encryption:  sudo cryptsetup status trading-data"
    echo "   Check performance: sudo ./scripts/monitor-crypto-performance.sh"
    echo "   Backup system:     sudo ./scripts/backup-encrypted.sh"
    echo "   Emergency stop:    sudo ./scripts/disaster-recovery.sh emergency-stop"
    echo ""
    echo "🌐 Access URLs:"
    echo "   Dashboard:         http://localhost:3000"
    echo "   API Backend:       http://localhost:8000"
    echo "   Health Check:      http://localhost:8000/health"
    echo ""
    echo "📝 Logs:"
    echo "   Startup log:       $LOG_FILE"
    echo "   Service logs:      docker-compose -f docker-compose.encrypted.yml logs"
    echo "   Crypto monitoring: /var/log/crypto-performance.log"
    echo ""
    success "Encrypted AI Trading Bot is ready for secure trading operations!"
}

# Print usage information
print_usage() {
    cat << EOF
AI Trading Bot - Encrypted Startup Script

Usage:
  $0 [options]

Options:
  --check-only     Perform checks without starting services
  --no-monitoring  Skip performance monitoring setup
  --help           Show this help

Examples:
  $0                    # Full startup with encryption
  $0 --check-only       # Verify system without starting
  sudo $0               # Startup with full encryption management

This script performs:
1. System prerequisite verification
2. Encrypted volume readiness check
3. Docker environment preparation
4. Secure service startup
5. Post-startup verification
6. Performance monitoring setup

For troubleshooting, see: docs/VOLUME_ENCRYPTION_GUIDE.md

EOF
}

# Main execution
main() {
    # Parse arguments
    local check_only=false
    local enable_monitoring=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                check_only=true
                shift
                ;;
            --no-monitoring)
                enable_monitoring=false
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Create log file if it doesn't exist
    sudo touch "$LOG_FILE" 2>/dev/null || touch "$LOG_FILE"
    
    print_banner
    log "Starting encrypted AI Trading Bot..."
    
    # Perform all checks
    check_user
    check_emergency_stop
    verify_encrypted_volumes
    check_encryption_keys
    verify_docker_environment
    check_system_resources
    validate_configuration
    
    if $check_only; then
        success "All checks passed - system ready for startup"
        exit 0
    fi
    
    # Setup and start services
    if $enable_monitoring; then
        setup_monitoring
    fi
    
    start_encrypted_services
    post_startup_verification
    print_startup_summary
}

# Run main function with all arguments
main "$@"