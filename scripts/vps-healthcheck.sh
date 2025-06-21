#!/bin/bash

# VPS Health Check Script for AI Trading Bot Containers
# This script runs inside Docker containers to perform comprehensive health checks

set -euo pipefail

# Configuration
SCRIPT_NAME="VPS Health Check"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
VPS_DEPLOYMENT="${VPS_DEPLOYMENT:-false}"
SERVICE_NAME="${SERVICE_NAME:-unknown}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME] INFO: $1${NC}"
}

# Basic system checks
check_system_resources() {
    info "Checking system resources..."

    # Check disk space
    local disk_usage=$(df /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
    if [ "$disk_usage" -gt 90 ]; then
        error "Disk usage critical: ${disk_usage}%"
    elif [ "$disk_usage" -gt 80 ]; then
        warn "Disk usage high: ${disk_usage}%"
    fi

    # Check memory usage (if available)
    if command -v free >/dev/null 2>&1; then
        local mem_usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}' 2>/dev/null || echo "0")
        if [ "$mem_usage" -gt 95 ]; then
            error "Memory usage critical: ${mem_usage}%"
        elif [ "$mem_usage" -gt 85 ]; then
            warn "Memory usage high: ${mem_usage}%"
        fi
    fi

    # Check if required directories are writable
    local required_dirs=("/app/logs" "/app/data")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ] && [ ! -w "$dir" ]; then
            error "Directory not writable: $dir"
        fi
    done

    log "System resources check passed"
}

# Network connectivity checks
check_network_connectivity() {
    info "Checking network connectivity..."

    # Check basic DNS resolution
    if command -v nslookup >/dev/null 2>&1; then
        if ! nslookup google.com >/dev/null 2>&1; then
            warn "DNS resolution may be impaired"
        fi
    fi

    # Check external connectivity
    if command -v curl >/dev/null 2>&1; then
        # Test basic HTTP connectivity
        if ! curl -f --connect-timeout 10 --max-time 30 -s https://httpbin.org/status/200 >/dev/null 2>&1; then
            warn "External HTTP connectivity may be impaired"
        fi

        # Test Bluefin API connectivity (if this is Bluefin service)
        if [ "$SERVICE_NAME" = "bluefin-service" ] || [ "$SERVICE_NAME" = "ai-trading-bot" ]; then
            if ! curl -f --connect-timeout 10 --max-time 30 -s https://api.bluefin.io >/dev/null 2>&1; then
                warn "Bluefin API connectivity may be impaired"
            fi
        fi

        # Test OpenAI API connectivity (if this is trading bot)
        if [ "$SERVICE_NAME" = "ai-trading-bot" ]; then
            if [ -n "${LLM__OPENAI_API_KEY:-}" ]; then
                if ! curl -f --connect-timeout 10 --max-time 30 -s -H "Authorization: Bearer $LLM__OPENAI_API_KEY" https://api.openai.com/v1/models >/dev/null 2>&1; then
                    warn "OpenAI API connectivity may be impaired"
                fi
            fi
        fi
    fi

    log "Network connectivity check completed"
}

# Service-specific health checks
check_service_health() {
    info "Checking service-specific health..."

    case "$SERVICE_NAME" in
        "bluefin-service")
            check_bluefin_service_health
            ;;
        "ai-trading-bot")
            check_trading_bot_health
            ;;
        "mcp-memory")
            check_mcp_memory_health
            ;;
        *)
            info "No specific health checks for service: $SERVICE_NAME"
            ;;
    esac
}

# Bluefin service specific checks
check_bluefin_service_health() {
    info "Checking Bluefin service health..."

    # Check if service is listening on expected port
    local port="${PORT:-8080}"
    if command -v netstat >/dev/null 2>&1; then
        if ! netstat -ln | grep ":$port" >/dev/null 2>&1; then
            error "Service not listening on port $port"
        fi
    fi

    # Check health endpoint
    if command -v curl >/dev/null 2>&1; then
        if ! curl -f --connect-timeout 10 --max-time 30 http://localhost:$port/health >/dev/null 2>&1; then
            error "Health endpoint not responding"
        fi
    fi

    # Check environment variables
    local required_vars=("BLUEFIN_PRIVATE_KEY" "BLUEFIN_SERVICE_API_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
        fi
    done

    # Validate private key format
    if [[ ! "${BLUEFIN_PRIVATE_KEY:-}" =~ ^0x[a-fA-F0-9]{64}$ ]]; then
        error "Invalid Bluefin private key format"
    fi

    log "Bluefin service health check passed"
}

# Trading bot specific checks
check_trading_bot_health() {
    info "Checking trading bot health..."

    # Check required environment variables
    local required_vars=("LLM__OPENAI_API_KEY" "EXCHANGE__BLUEFIN_PRIVATE_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
        fi
    done

    # Check if configuration file exists
    local config_file="${CONFIG_FILE:-/app/config/vps_production.json}"
    if [ ! -f "$config_file" ]; then
        error "Configuration file not found: $config_file"
    fi

    # Check log files are being created
    if [ -d "/app/logs" ]; then
        local log_count=$(find /app/logs -name "*.log" -type f 2>/dev/null | wc -l)
        if [ "$log_count" -eq 0 ]; then
            warn "No log files found in /app/logs"
        fi
    fi

    # Check Bluefin service connectivity
    if [ -n "${BLUEFIN_SERVICE_URL:-}" ]; then
        if command -v curl >/dev/null 2>&1; then
            if ! curl -f --connect-timeout 10 --max-time 30 "${BLUEFIN_SERVICE_URL}/health" >/dev/null 2>&1; then
                error "Cannot connect to Bluefin service at $BLUEFIN_SERVICE_URL"
            fi
        fi
    fi

    log "Trading bot health check passed"
}

# MCP memory service specific checks
check_mcp_memory_health() {
    info "Checking MCP memory service health..."

    # Check if service is listening on expected port
    local port="${MCP_SERVER_PORT:-8765}"
    if command -v netstat >/dev/null 2>&1; then
        if ! netstat -ln | grep ":$port" >/dev/null 2>&1; then
            error "MCP memory service not listening on port $port"
        fi
    fi

    # Check health endpoint
    if command -v curl >/dev/null 2>&1; then
        if ! curl -f --connect-timeout 10 --max-time 30 http://localhost:$port/health >/dev/null 2>&1; then
            error "MCP memory health endpoint not responding"
        fi
    fi

    # Check data directory
    if [ ! -d "/app/data" ] || [ ! -w "/app/data" ]; then
        error "MCP memory data directory not accessible"
    fi

    log "MCP memory service health check passed"
}

# Process checks
check_processes() {
    info "Checking processes..."

    # Check if Python processes are running
    if command -v pgrep >/dev/null 2>&1; then
        local python_procs=$(pgrep -f python | wc -l)
        if [ "$python_procs" -eq 0 ]; then
            error "No Python processes found"
        fi
    fi

    # Check for zombie processes
    if command -v ps >/dev/null 2>&1; then
        local zombie_count=$(ps aux | awk '$8 ~ /^Z/ { count++ } END { print count+0 }')
        if [ "$zombie_count" -gt 0 ]; then
            warn "$zombie_count zombie processes detected"
        fi
    fi

    log "Process check completed"
}

# File system checks
check_filesystem() {
    info "Checking filesystem..."

    # Check critical directories exist
    local critical_dirs=("/app" "/app/logs")
    for dir in "${critical_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            error "Critical directory missing: $dir"
        fi
    done

    # Check log rotation is working
    if [ -d "/app/logs" ]; then
        local old_logs=$(find /app/logs -name "*.log.*" -type f 2>/dev/null | wc -l)
        if [ "$old_logs" -gt 100 ]; then
            warn "Many old log files detected ($old_logs), log rotation may not be working"
        fi
    fi

    # Check for core dumps
    local core_dumps=$(find /app -name "core.*" -type f 2>/dev/null | wc -l)
    if [ "$core_dumps" -gt 0 ]; then
        warn "$core_dumps core dump files detected"
    fi

    log "Filesystem check completed"
}

# VPS-specific checks
check_vps_specific() {
    if [ "$VPS_DEPLOYMENT" != "true" ]; then
        return 0
    fi

    info "Performing VPS-specific checks..."

    # Check geographic region if set
    if [ -n "${GEOGRAPHIC_REGION:-}" ]; then
        info "Configured for geographic region: $GEOGRAPHIC_REGION"

        # Test IP geolocation if curl is available
        if command -v curl >/dev/null 2>&1; then
            local detected_region=$(curl -s --connect-timeout 10 --max-time 30 https://ipapi.co/country_code/ 2>/dev/null || echo "UNKNOWN")
            if [ "$detected_region" != "UNKNOWN" ] && [ "$detected_region" != "$GEOGRAPHIC_REGION" ]; then
                warn "Detected region ($detected_region) differs from configured region ($GEOGRAPHIC_REGION)"
            fi
        fi
    fi

    # Check proxy configuration if enabled
    if [ "${PROXY_ENABLED:-false}" = "true" ]; then
        info "Proxy is enabled, checking configuration..."

        if [ -z "${PROXY_HOST:-}" ] || [ -z "${PROXY_PORT:-}" ]; then
            error "Proxy enabled but host/port not configured"
        fi

        # Test proxy connectivity if curl supports it
        if command -v curl >/dev/null 2>&1; then
            if ! curl -f --connect-timeout 10 --proxy "${PROXY_HOST}:${PROXY_PORT}" -s https://httpbin.org/ip >/dev/null 2>&1; then
                warn "Proxy connectivity test failed"
            fi
        fi
    fi

    # Check monitoring configuration
    if [ "${MONITORING__ENABLED:-false}" = "true" ]; then
        info "Monitoring is enabled"

        # Check if monitoring dashboard is accessible
        if [ -n "${ALERT_WEBHOOK_URL:-}" ]; then
            if command -v curl >/dev/null 2>&1; then
                if ! curl -f --connect-timeout 10 --max-time 30 -X POST "$ALERT_WEBHOOK_URL" -H 'Content-type: application/json' --data '{"text":"Health check test"}' >/dev/null 2>&1; then
                    warn "Alert webhook may not be accessible"
                fi
            fi
        fi
    fi

    log "VPS-specific checks completed"
}

# Generate health report
generate_health_report() {
    info "Generating health report..."

    local report_file="/app/logs/health-report-$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "service_name": "$SERVICE_NAME",
    "vps_deployment": $VPS_DEPLOYMENT,
    "health_status": "healthy",
    "checks_performed": [
        "system_resources",
        "network_connectivity",
        "service_health",
        "processes",
        "filesystem",
        "vps_specific"
    ],
    "system_info": {
        "disk_usage": "$(df /app 2>/dev/null | tail -1 | awk '{print $5}' || echo 'unknown')",
        "memory_usage": "$(free | grep Mem | awk '{printf("%.0f%%", $3/$2 * 100.0)}' 2>/dev/null || echo 'unknown')",
        "uptime": "$(uptime -p 2>/dev/null || echo 'unknown')",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}' | sed 's/^[ \t]*//' 2>/dev/null || echo 'unknown')"
    },
    "environment": {
        "geographic_region": "${GEOGRAPHIC_REGION:-unknown}",
        "proxy_enabled": "${PROXY_ENABLED:-false}",
        "monitoring_enabled": "${MONITORING__ENABLED:-false}",
        "log_level": "$LOG_LEVEL"
    }
}
EOF

    info "Health report generated: $report_file"
}

# Main health check function
main() {
    log "Starting comprehensive health check for $SERVICE_NAME..."

    # Perform all health checks
    check_system_resources
    check_network_connectivity
    check_service_health
    check_processes
    check_filesystem
    check_vps_specific

    # Generate report
    generate_health_report

    log "All health checks completed successfully!"

    # Return success
    exit 0
}

# Handle different service types
case "${1:-main}" in
    "main"|"")
        # Detect service name from hostname or environment
        if [ -n "${HOSTNAME:-}" ]; then
            case "$HOSTNAME" in
                *bluefin*) SERVICE_NAME="bluefin-service" ;;
                *trading*) SERVICE_NAME="ai-trading-bot" ;;
                *mcp*) SERVICE_NAME="mcp-memory" ;;
                *) SERVICE_NAME="${HOSTNAME}" ;;
            esac
        fi
        main
        ;;
    "bluefin-service")
        SERVICE_NAME="bluefin-service"
        main
        ;;
    "ai-trading-bot")
        SERVICE_NAME="ai-trading-bot"
        main
        ;;
    "mcp-memory")
        SERVICE_NAME="mcp-memory"
        main
        ;;
    "system-only")
        check_system_resources
        check_filesystem
        ;;
    "network-only")
        check_network_connectivity
        ;;
    *)
        echo "Usage: $0 {main|bluefin-service|ai-trading-bot|mcp-memory|system-only|network-only}"
        echo "  main              - Full health check (default)"
        echo "  bluefin-service   - Bluefin service specific checks"
        echo "  ai-trading-bot    - Trading bot specific checks"
        echo "  mcp-memory        - MCP memory service specific checks"
        echo "  system-only       - Only system resource checks"
        echo "  network-only      - Only network connectivity checks"
        exit 1
        ;;
esac
