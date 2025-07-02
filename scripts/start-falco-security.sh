#!/bin/bash

# Falco Security Monitoring Startup Script for AI Trading Bot
# Provides safe deployment and management of Falco runtime security

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FALCO_CONFIG_DIR="$PROJECT_ROOT/security/falco"
LOGS_DIR="$PROJECT_ROOT/logs/falco"
DATA_DIR="$PROJECT_ROOT/data/security"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running as root or with docker permissions
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker not available or insufficient permissions"
        log_error "Please ensure Docker is running and user has docker permissions"
        exit 1
    fi
    
    # Check if trading bot network exists
    if ! docker network ls | grep -q "trading-network"; then
        log_error "Trading network not found"
        log_error "Please start the trading bot first: docker-compose up -d"
        exit 1
    fi
    
    # Check kernel version for eBPF support
    KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
    if [ "$(printf '%s\n' "4.14" "$KERNEL_VERSION" | sort -V | head -n1)" != "4.14" ]; then
        log_warning "Kernel version $KERNEL_VERSION may not support modern eBPF"
        log_warning "Consider upgrading to kernel 4.14+ for optimal performance"
    fi
    
    log_success "Prerequisites check completed"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    # Create directories with proper permissions
    mkdir -p "$LOGS_DIR" "$DATA_DIR"
    mkdir -p "$DATA_DIR/events" "$DATA_DIR/metrics"
    
    # Set permissions (using host user if available)
    if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
        chown -R "$HOST_UID:$HOST_GID" "$LOGS_DIR" "$DATA_DIR" 2>/dev/null || true
    fi
    
    log_success "Directories created successfully"
}

# Validate Falco configuration
validate_config() {
    log_info "Validating Falco configuration..."
    
    # Check if configuration files exist
    local config_files=(
        "$FALCO_CONFIG_DIR/falco.yaml"
        "$FALCO_CONFIG_DIR/trading_bot_rules.yaml"
        "$FALCO_CONFIG_DIR/financial_security_rules.yaml"
        "$FALCO_CONFIG_DIR/container_security_rules.yaml"
    )
    
    for config_file in "${config_files[@]}"; do
        if [ ! -f "$config_file" ]; then
            log_error "Configuration file not found: $config_file"
            exit 1
        fi
    done
    
    # Validate YAML syntax
    for config_file in "${config_files[@]}"; do
        if command -v python3 >/dev/null 2>&1; then
            if ! python3 -c "import yaml; yaml.safe_load(open('$config_file'))" >/dev/null 2>&1; then
                log_error "Invalid YAML syntax in: $config_file"
                exit 1
            fi
        fi
    done
    
    log_success "Configuration validation completed"
}

# Check resource availability
check_resources() {
    log_info "Checking system resources..."
    
    # Check available memory
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 1024 ]; then
        log_warning "Low available memory: ${available_memory}MB"
        log_warning "Falco may impact system performance"
    fi
    
    # Check CPU load
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    if (( $(echo "$cpu_load > 2.0" | bc -l) )); then
        log_warning "High CPU load: $cpu_load"
        log_warning "Consider delaying Falco deployment"
    fi
    
    # Check disk space
    local disk_usage=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log_warning "High disk usage: ${disk_usage}%"
        log_warning "Ensure sufficient space for Falco logs"
    fi
    
    log_success "Resource check completed"
}

# Performance impact assessment
assess_performance_impact() {
    log_info "Assessing performance impact..."
    
    # Get baseline trading bot performance
    if docker ps | grep -q "ai-trading-bot"; then
        local trading_bot_cpu=$(docker stats ai-trading-bot --no-stream --format "table {{.CPUPerc}}" | tail -1 | sed 's/%//')
        local trading_bot_memory=$(docker stats ai-trading-bot --no-stream --format "table {{.MemUsage}}" | tail -1 | awk '{print $1}' | sed 's/MiB//')
        
        log_info "Current trading bot resource usage:"
        log_info "  CPU: ${trading_bot_cpu}%"
        log_info "  Memory: ${trading_bot_memory}MB"
        
        # Store baseline for comparison
        echo "BASELINE_CPU=$trading_bot_cpu" > "$DATA_DIR/performance_baseline.env"
        echo "BASELINE_MEMORY=$trading_bot_memory" >> "$DATA_DIR/performance_baseline.env"
    else
        log_warning "Trading bot not running - cannot assess baseline performance"
    fi
}

# Start Falco services
start_falco() {
    log_info "Starting Falco security monitoring..."
    
    # Start Falco services using docker-compose
    cd "$PROJECT_ROOT"
    
    # Check if already running
    if docker ps | grep -q "falco-security-monitor"; then
        log_warning "Falco is already running"
        return 0
    fi
    
    # Start services in order
    log_info "Starting Falco core service..."
    docker-compose -f docker-compose.falco.yml up -d falco
    
    # Wait for Falco to be ready
    local retry_count=0
    local max_retries=30
    
    while [ $retry_count -lt $max_retries ]; do
        if docker exec falco-security-monitor curl -f -s http://localhost:8765/healthz >/dev/null 2>&1; then
            log_success "Falco core service is ready"
            break
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -eq $max_retries ]; then
            log_error "Falco failed to start within expected time"
            docker logs falco-security-monitor --tail 20
            exit 1
        fi
        
        log_info "Waiting for Falco to be ready... ($retry_count/$max_retries)"
        sleep 2
    done
    
    # Start alert manager
    log_info "Starting AlertManager..."
    docker-compose -f docker-compose.falco.yml up -d falco-alertmanager
    
    # Start security event processor
    log_info "Starting security event processor..."
    docker-compose -f docker-compose.falco.yml up -d falco-security-processor
    
    # Start Prometheus (optional)
    if [ "${ENABLE_PROMETHEUS:-true}" = "true" ]; then
        log_info "Starting Prometheus metrics collection..."
        docker-compose -f docker-compose.falco.yml up -d prometheus-falco
    fi
    
    log_success "Falco security monitoring started successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying Falco deployment..."
    
    # Check service health
    local services=(
        "falco-security-monitor:8765/healthz"
        "falco-alertmanager:9093/-/healthy"
        "falco-security-processor:8080/health"
    )
    
    for service in "${services[@]}"; do
        local container_name="${service%%:*}"
        local health_endpoint="${service#*:}"
        
        if docker ps | grep -q "$container_name"; then
            if docker exec "$container_name" curl -f -s "http://localhost:$health_endpoint" >/dev/null 2>&1; then
                log_success "$container_name is healthy"
            else
                log_error "$container_name health check failed"
                docker logs "$container_name" --tail 10
            fi
        else
            log_error "$container_name is not running"
        fi
    done
    
    # Test event generation
    log_info "Testing security event detection..."
    
    # Generate a test event (safe rule trigger)
    docker exec ai-trading-bot touch /tmp/test_security_event.txt 2>/dev/null || true
    docker exec ai-trading-bot rm -f /tmp/test_security_event.txt 2>/dev/null || true
    
    # Wait for event processing
    sleep 5
    
    # Check if events are being processed
    local event_count=$(curl -s http://localhost:8080/events | jq '.total_events // 0' 2>/dev/null || echo "0")
    if [ "$event_count" -gt 0 ]; then
        log_success "Security event processing is working ($event_count events processed)"
    else
        log_warning "No security events detected yet"
    fi
}

# Monitor performance impact
monitor_performance() {
    log_info "Monitoring performance impact..."
    
    # Wait for stabilization
    sleep 30
    
    # Get current performance metrics
    if docker ps | grep -q "ai-trading-bot"; then
        local current_cpu=$(docker stats ai-trading-bot --no-stream --format "table {{.CPUPerc}}" | tail -1 | sed 's/%//')
        local current_memory=$(docker stats ai-trading-bot --no-stream --format "table {{.MemUsage}}" | tail -1 | awk '{print $1}' | sed 's/MiB//')
        
        # Load baseline if available
        if [ -f "$DATA_DIR/performance_baseline.env" ]; then
            source "$DATA_DIR/performance_baseline.env"
            
            local cpu_impact=$(echo "scale=1; $current_cpu - $BASELINE_CPU" | bc -l 2>/dev/null || echo "N/A")
            local memory_impact=$(echo "scale=1; $current_memory - $BASELINE_MEMORY" | bc -l 2>/dev/null || echo "N/A")
            
            log_info "Performance impact assessment:"
            log_info "  CPU impact: +${cpu_impact}%"
            log_info "  Memory impact: +${memory_impact}MB"
            
            # Check if impact is acceptable
            if [ "$cpu_impact" != "N/A" ] && (( $(echo "$cpu_impact > 10.0" | bc -l) )); then
                log_warning "High CPU impact detected: +${cpu_impact}%"
                log_warning "Consider performance tuning"
            fi
            
            if [ "$memory_impact" != "N/A" ] && (( $(echo "$memory_impact > 100.0" | bc -l) )); then
                log_warning "High memory impact detected: +${memory_impact}MB"
                log_warning "Consider reducing buffer sizes"
            fi
        fi
    fi
    
    # Show Falco resource usage
    log_info "Falco resource usage:"
    docker stats falco-security-monitor --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}"
}

# Create monitoring dashboard URLs
show_monitoring_info() {
    log_info "Falco Security Monitoring Information:"
    echo
    echo "📊 Service Endpoints:"
    echo "  Falco Web UI:        http://localhost:8765"
    echo "  AlertManager:        http://localhost:9093"
    echo "  Security Processor:  http://localhost:8080"
    echo "  Prometheus:          http://localhost:9090"
    echo
    echo "📁 Log Locations:"
    echo "  Falco Logs:          $LOGS_DIR/"
    echo "  Security Events:     $DATA_DIR/events/"
    echo
    echo "🔧 Management Commands:"
    echo "  View events:         curl http://localhost:8080/events"
    echo "  Check status:        docker-compose -f docker-compose.falco.yml ps"
    echo "  View logs:           docker-compose -f docker-compose.falco.yml logs -f"
    echo "  Stop services:       docker-compose -f docker-compose.falco.yml down"
    echo
    echo "⚠️  Security Alerts:"
    echo "  Slack channel:       #trading-security-alerts"
    echo "  Email alerts:        ${SECURITY_EMAIL_TO:-Not configured}"
    echo
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic if needed
}

# Signal handlers
trap cleanup EXIT

# Main execution
main() {
    local command="${1:-start}"
    
    case "$command" in
        "start")
            log_info "Starting Falco Security Monitoring for AI Trading Bot"
            check_prerequisites
            setup_directories
            validate_config
            check_resources
            assess_performance_impact
            start_falco
            verify_deployment
            monitor_performance
            show_monitoring_info
            log_success "Falco security monitoring deployment completed successfully"
            ;;
        
        "stop")
            log_info "Stopping Falco security monitoring..."
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.falco.yml down
            log_success "Falco security monitoring stopped"
            ;;
        
        "restart")
            log_info "Restarting Falco security monitoring..."
            "$0" stop
            sleep 5
            "$0" start
            ;;
        
        "status")
            log_info "Checking Falco service status..."
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.falco.yml ps
            ;;
        
        "logs")
            log_info "Showing Falco logs..."
            cd "$PROJECT_ROOT"
            docker-compose -f docker-compose.falco.yml logs -f "${2:-falco}"
            ;;
        
        "events")
            log_info "Recent security events:"
            curl -s http://localhost:8080/events | jq '.events[] | {time: .timestamp, rule: .original_event.rule, container: .original_event.output_fields.container_name, severity: .severity_score}' 2>/dev/null || echo "Events API not available"
            ;;
        
        "performance")
            log_info "Performance monitoring..."
            monitor_performance
            ;;
        
        "test")
            log_info "Testing security event detection..."
            # Generate test events
            docker exec ai-trading-bot bash -c 'echo "test" > /tmp/security_test && rm /tmp/security_test' 2>/dev/null || true
            sleep 2
            "$0" events
            ;;
        
        "help"|"-h"|"--help")
            echo "Falco Security Monitoring Management Script"
            echo
            echo "Usage: $0 [COMMAND]"
            echo
            echo "Commands:"
            echo "  start       Start Falco security monitoring (default)"
            echo "  stop        Stop Falco security monitoring"
            echo "  restart     Restart Falco security monitoring"
            echo "  status      Show service status"
            echo "  logs        Show service logs"
            echo "  events      Show recent security events"
            echo "  performance Monitor performance impact"
            echo "  test        Test security event detection"
            echo "  help        Show this help message"
            echo
            echo "Environment Variables:"
            echo "  ENABLE_PROMETHEUS   Enable Prometheus metrics (default: true)"
            echo "  SECURITY_EMAIL_TO   Email for security alerts"
            echo "  SLACK_WEBHOOK_URL   Slack webhook for notifications"
            echo
            ;;
        
        *)
            log_error "Unknown command: $command"
            log_error "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"