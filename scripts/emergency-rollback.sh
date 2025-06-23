#!/bin/bash
#
# Emergency Rollback Script for AI Trading Bot
# Usage: ./emergency-rollback.sh [options]
#
# Options:
#   --auto              Run without confirmation prompts (dangerous!)
#   --dry-run           Show what would be done without executing
#   --partial SERVICE   Rollback only specific service (bot|mcp|all)
#   --skip-backup       Skip backup restoration (use with caution)
#   --from-date DATE    Rollback to specific date (YYYY-MM-DD)
#   --help              Show this help message

set -euo pipefail

# ========================================
# Configuration
# ========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_DIR="${PROJECT_ROOT}/logs/rollback"
INCIDENT_DIR="${PROJECT_ROOT}/logs/incidents"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROLLBACK_LOG="${LOG_DIR}/rollback_${TIMESTAMP}.log"
INCIDENT_REPORT="${INCIDENT_DIR}/incident_${TIMESTAMP}.md"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
AUTO_MODE=false
DRY_RUN=false
PARTIAL_SERVICE="all"
SKIP_BACKUP=false
ROLLBACK_DATE=""

# ========================================
# Helper Functions
# ========================================

log() {
    echo -e "${2:-}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$ROLLBACK_LOG"
}

log_error() {
    log "ERROR: $1" "$RED"
}

log_success() {
    log "SUCCESS: $1" "$GREEN"
}

log_warning() {
    log "WARNING: $1" "$YELLOW"
}

log_info() {
    log "INFO: $1" "$BLUE"
}

confirm() {
    if [ "$AUTO_MODE" = true ]; then
        return 0
    fi

    read -p "$(echo -e "${YELLOW}$1 (y/N): ${NC}")" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

execute_cmd() {
    local cmd="$1"
    local description="${2:-Executing command}"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would execute: $cmd"
        return 0
    fi

    log_info "$description"
    if eval "$cmd" >> "$ROLLBACK_LOG" 2>&1; then
        log_success "$description completed"
        return 0
    else
        log_error "$description failed"
        return 1
    fi
}

setup_directories() {
    mkdir -p "$LOG_DIR" "$INCIDENT_DIR"
    touch "$ROLLBACK_LOG"
}

# ========================================
# Service Management Functions
# ========================================

stop_services() {
    log_warning "============================================"
    log_warning "PHASE 1: EMERGENCY SERVICE SHUTDOWN"
    log_warning "============================================"

    # Stop Docker containers
    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "bot" ]; then
        execute_cmd "docker-compose stop ai-trading-bot" "Stopping AI trading bot container"
        execute_cmd "docker stop ai-trading-bot 2>/dev/null || true" "Force stopping bot container"
    fi

    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "mcp" ]; then
        execute_cmd "docker-compose stop mcp-memory" "Stopping MCP memory server"
        execute_cmd "docker stop mcp-memory 2>/dev/null || true" "Force stopping MCP container"
    fi

    # Kill any hanging Python processes
    execute_cmd "pkill -f 'python.*bot.main' || true" "Killing hanging bot processes"
    execute_cmd "pkill -f 'uvicorn.*memory_server' || true" "Killing hanging MCP processes"

    # Stop systemd services if they exist
    if systemctl is-active --quiet ai-trading-bot 2>/dev/null; then
        execute_cmd "sudo systemctl stop ai-trading-bot" "Stopping systemd bot service"
    fi

    log_success "All services stopped"
}

# ========================================
# Backup Discovery Functions
# ========================================

find_latest_backup() {
    local backup_type="$1"
    local target_date="${2:-}"

    if [ -n "$target_date" ]; then
        # Find backup from specific date
        find "$BACKUP_DIR" -name "*${backup_type}*${target_date}*" -type f | sort -r | head -1
    else
        # Find latest backup
        find "$BACKUP_DIR" -name "*${backup_type}*" -type f | sort -r | head -1
    fi
}

list_available_backups() {
    log_info "Available backups:"
    log_info "=================="

    for type in "config" "env" "database" "state" "docker"; do
        local count=$(find "$BACKUP_DIR" -name "*${type}*" -type f 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            log_info "${type^^}: $count backups found"
            find "$BACKUP_DIR" -name "*${type}*" -type f | sort -r | head -5 | while read -r file; do
                log_info "  - $(basename "$file")"
            done
        fi
    done
}

# ========================================
# Configuration Restoration
# ========================================

restore_configuration() {
    log_warning "============================================"
    log_warning "PHASE 2: CONFIGURATION RESTORATION"
    log_warning "============================================"

    if [ "$SKIP_BACKUP" = true ]; then
        log_warning "Skipping backup restoration as requested"
        return 0
    fi

    # Backup current configuration first
    local current_backup_dir="${BACKUP_DIR}/pre_rollback_${TIMESTAMP}"
    execute_cmd "mkdir -p '$current_backup_dir'" "Creating pre-rollback backup directory"

    # Backup current files
    for file in ".env" "config/production.json" "config/conservative_config.json"; do
        if [ -f "${PROJECT_ROOT}/${file}" ]; then
            execute_cmd "cp '${PROJECT_ROOT}/${file}' '$current_backup_dir/'" "Backing up current $file"
        fi
    done

    # Restore .env file
    local env_backup=$(find_latest_backup "env" "$ROLLBACK_DATE")
    if [ -n "$env_backup" ]; then
        execute_cmd "cp '$env_backup' '${PROJECT_ROOT}/.env'" "Restoring .env from $env_backup"
    else
        log_error "No .env backup found!"
    fi

    # Restore config files
    local config_backup=$(find_latest_backup "config" "$ROLLBACK_DATE")
    if [ -n "$config_backup" ] && [ -f "$config_backup" ]; then
        if [[ "$config_backup" == *.tar.gz ]]; then
            execute_cmd "tar -xzf '$config_backup' -C '${PROJECT_ROOT}'" "Restoring config files from $config_backup"
        else
            execute_cmd "cp '$config_backup' '${PROJECT_ROOT}/config/'" "Restoring config from $config_backup"
        fi
    else
        log_warning "No config backup found"
    fi

    # Restore MCP memory database if applicable
    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "mcp" ]; then
        local db_backup=$(find_latest_backup "database" "$ROLLBACK_DATE")
        if [ -n "$db_backup" ]; then
            execute_cmd "cp '$db_backup' '${PROJECT_ROOT}/data/memories.db'" "Restoring MCP database from $db_backup"
        fi
    fi
}

# ========================================
# Docker Rollback Functions
# ========================================

rollback_docker_containers() {
    log_warning "============================================"
    log_warning "PHASE 3: DOCKER CONTAINER ROLLBACK"
    log_warning "============================================"

    # Get previous docker image tags from backup
    local docker_backup=$(find_latest_backup "docker" "$ROLLBACK_DATE")

    if [ -n "$docker_backup" ] && [ -f "$docker_backup" ]; then
        log_info "Found Docker state backup: $docker_backup"

        # Read previous image tags
        if [ -f "$docker_backup" ]; then
            source "$docker_backup"

            if [ -n "${BOT_IMAGE_TAG:-}" ]; then
                execute_cmd "docker pull $BOT_IMAGE_TAG" "Pulling previous bot image: $BOT_IMAGE_TAG"

                # Update docker-compose.yml or .env with the tag
                if grep -q "AI_TRADING_BOT_TAG" "${PROJECT_ROOT}/.env"; then
                    execute_cmd "sed -i.bak 's/AI_TRADING_BOT_TAG=.*/AI_TRADING_BOT_TAG=${BOT_IMAGE_TAG##*:}/' '${PROJECT_ROOT}/.env'" \
                        "Updating bot image tag in .env"
                fi
            fi

            if [ -n "${MCP_IMAGE_TAG:-}" ] && ([ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "mcp" ]); then
                execute_cmd "docker pull $MCP_IMAGE_TAG" "Pulling previous MCP image: $MCP_IMAGE_TAG"
            fi
        fi
    else
        log_warning "No Docker state backup found - using current images"
    fi

    # Rebuild containers with previous configuration
    execute_cmd "cd '${PROJECT_ROOT}' && docker-compose build --no-cache" "Rebuilding containers"
}

# ========================================
# Service Restart Functions
# ========================================

start_services() {
    log_warning "============================================"
    log_warning "PHASE 4: SERVICE RESTART"
    log_warning "============================================"

    # Start services based on partial rollback selection
    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "mcp" ]; then
        execute_cmd "cd '${PROJECT_ROOT}' && docker-compose up -d mcp-memory" "Starting MCP memory server"
        sleep 5  # Give MCP time to initialize
    fi

    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "bot" ]; then
        execute_cmd "cd '${PROJECT_ROOT}' && docker-compose up -d ai-trading-bot" "Starting AI trading bot"
    fi

    # Restart systemd services if they exist
    if systemctl is-enabled ai-trading-bot 2>/dev/null; then
        execute_cmd "sudo systemctl start ai-trading-bot" "Starting systemd bot service"
    fi

    log_success "Services started"
}

# ========================================
# Health Check Functions
# ========================================

perform_health_checks() {
    log_warning "============================================"
    log_warning "PHASE 5: POST-ROLLBACK HEALTH CHECKS"
    log_warning "============================================"

    local health_status=0

    # Check Docker container status
    log_info "Checking container health..."

    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "bot" ]; then
        if docker ps | grep -q ai-trading-bot; then
            log_success "AI trading bot container is running"

            # Check container logs for errors
            if docker logs --tail 50 ai-trading-bot 2>&1 | grep -i error; then
                log_warning "Errors found in bot container logs"
                health_status=1
            fi
        else
            log_error "AI trading bot container is not running!"
            health_status=1
        fi
    fi

    if [ "$PARTIAL_SERVICE" = "all" ] || [ "$PARTIAL_SERVICE" = "mcp" ]; then
        if docker ps | grep -q mcp-memory; then
            log_success "MCP memory server container is running"

            # Test MCP endpoint
            if command -v curl &> /dev/null; then
                if curl -s -f http://localhost:8765/health > /dev/null 2>&1; then
                    log_success "MCP memory server is responding"
                else
                    log_error "MCP memory server is not responding!"
                    health_status=1
                fi
            fi
        else
            log_error "MCP memory server container is not running!"
            health_status=1
        fi
    fi

    # Check exchange connectivity
    log_info "Testing exchange connectivity..."
    if docker exec ai-trading-bot python -c "
import os
import sys
sys.path.append('/app')
from bot.exchange.factory import ExchangeFactory
from bot.config import Settings
try:
    settings = Settings()
    exchange = ExchangeFactory.create(settings.exchange)
    print('Exchange connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Exchange connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_success "Exchange connectivity verified"
    else
        log_error "Exchange connectivity test failed!"
        health_status=1
    fi

    # Check configuration validity
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        if grep -q "SYSTEM__DRY_RUN=true" "${PROJECT_ROOT}/.env"; then
            log_success "Paper trading mode is ENABLED (safe)"
        else
            log_warning "LIVE TRADING mode is ENABLED - be careful!"
        fi
    fi

    return $health_status
}

# ========================================
# Incident Report Generation
# ========================================

generate_incident_report() {
    log_warning "============================================"
    log_warning "PHASE 6: INCIDENT REPORT GENERATION"
    log_warning "============================================"

    cat > "$INCIDENT_REPORT" << EOF
# Emergency Rollback Incident Report

**Generated**: $(date '+%Y-%m-%d %H:%M:%S %Z')
**Rollback ID**: ROLLBACK_${TIMESTAMP}
**Initiated By**: $(whoami)@$(hostname)
**Script Version**: 1.0.0

## Rollback Summary

- **Type**: ${PARTIAL_SERVICE^^} Service Rollback
- **Mode**: $([ "$DRY_RUN" = true ] && echo "DRY RUN" || echo "LIVE EXECUTION")
- **Target Date**: ${ROLLBACK_DATE:-Latest Available}
- **Duration**: $SECONDS seconds

## Actions Taken

### 1. Service Shutdown
$(grep "Stopping" "$ROLLBACK_LOG" | sed 's/^/- /')

### 2. Backup Restoration
$(grep "Restoring" "$ROLLBACK_LOG" | sed 's/^/- /')

### 3. Docker Rollback
$(grep "Docker" "$ROLLBACK_LOG" | grep -E "(Pulling|Updating)" | sed 's/^/- /')

### 4. Service Restart
$(grep "Starting" "$ROLLBACK_LOG" | sed 's/^/- /')

## Health Check Results

$(grep -E "(SUCCESS|ERROR|WARNING).*health|container|connectivity" "$ROLLBACK_LOG" | tail -20 | sed 's/^/- /')

## Errors Encountered

$(grep "ERROR" "$ROLLBACK_LOG" | sed 's/^/- /' || echo "- No errors recorded")

## Warnings

$(grep "WARNING" "$ROLLBACK_LOG" | grep -v "PHASE" | sed 's/^/- /' || echo "- No warnings recorded")

## Post-Rollback Status

- **Bot Container**: $(docker ps | grep -q ai-trading-bot && echo "RUNNING ✅" || echo "STOPPED ❌")
- **MCP Container**: $(docker ps | grep -q mcp-memory && echo "RUNNING ✅" || echo "STOPPED ❌")
- **Trading Mode**: $(grep "SYSTEM__DRY_RUN" "${PROJECT_ROOT}/.env" | cut -d= -f2 | sed 's/true/PAPER TRADING 🟢/;s/false/LIVE TRADING 🔴/')

## Recommended Actions

1. Review container logs: \`docker-compose logs --tail 100\`
2. Monitor system behavior for next 30 minutes
3. Check trading positions and balances
4. Verify all integrations are functional
5. Document root cause in issue tracker

## Log Files

- **Rollback Log**: $ROLLBACK_LOG
- **Container Logs**: Run \`docker-compose logs\` to view

## Rollback Command Used

\`\`\`bash
$0 $*
\`\`\`

---
*This report was automatically generated by the emergency rollback script.*
EOF

    log_success "Incident report generated: $INCIDENT_REPORT"

    # Display report location and summary
    echo
    log_info "========================================"
    log_info "ROLLBACK COMPLETE"
    log_info "========================================"
    log_info "Incident Report: $INCIDENT_REPORT"
    log_info "Full Log: $ROLLBACK_LOG"
    echo

    # Show quick summary
    if [ "$DRY_RUN" = false ]; then
        tail -20 "$INCIDENT_REPORT" | grep -E "^- \*\*.*:" | sed 's/^- //'
    fi
}

# ========================================
# Argument Parsing
# ========================================

show_help() {
    cat << EOF
Emergency Rollback Script for AI Trading Bot

Usage: $0 [options]

Options:
    --auto              Run without confirmation prompts (dangerous!)
    --dry-run           Show what would be done without executing
    --partial SERVICE   Rollback only specific service (bot|mcp|all)
    --skip-backup       Skip backup restoration (use with caution)
    --from-date DATE    Rollback to specific date (YYYY-MM-DD)
    --help              Show this help message

Examples:
    # Interactive rollback with confirmations
    $0

    # Automated rollback (for CI/CD)
    $0 --auto

    # Test what would happen without making changes
    $0 --dry-run

    # Rollback only the bot service
    $0 --partial bot

    # Rollback to specific date
    $0 --from-date 2024-01-15

Emergency Contacts:
    - DevOps Team: devops@company.com
    - On-Call: +1-555-0123

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --partial)
            PARTIAL_SERVICE="$2"
            if [[ ! "$PARTIAL_SERVICE" =~ ^(bot|mcp|all)$ ]]; then
                log_error "Invalid service: $PARTIAL_SERVICE. Must be bot, mcp, or all."
                exit 1
            fi
            shift 2
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --from-date)
            ROLLBACK_DATE="$2"
            if ! date -d "$ROLLBACK_DATE" &>/dev/null; then
                log_error "Invalid date format: $ROLLBACK_DATE. Use YYYY-MM-DD."
                exit 1
            fi
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ========================================
# Main Execution
# ========================================

main() {
    # Setup
    setup_directories

    # Show banner
    echo -e "${RED}"
    echo "╔══════════════════════════════════════════════╗"
    echo "║        EMERGENCY ROLLBACK INITIATED          ║"
    echo "║                                              ║"
    echo "║  ⚠️  THIS WILL STOP ALL TRADING SERVICES  ⚠️  ║"
    echo "╚══════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Log execution details
    log_info "Rollback initiated by $(whoami) at $(date)"
    log_info "Options: AUTO=$AUTO_MODE, DRY_RUN=$DRY_RUN, PARTIAL=$PARTIAL_SERVICE"
    [ -n "$ROLLBACK_DATE" ] && log_info "Target rollback date: $ROLLBACK_DATE"

    # Show available backups
    if [ "$DRY_RUN" = false ]; then
        list_available_backups
        echo
    fi

    # Confirmation
    if [ "$DRY_RUN" = false ] && [ "$AUTO_MODE" = false ]; then
        if ! confirm "This will stop trading services and rollback to previous state. Continue?"; then
            log_warning "Rollback cancelled by user"
            exit 0
        fi
    fi

    # Execute rollback phases
    local rollback_success=true

    # Phase 1: Stop services
    if ! stop_services; then
        log_error "Failed to stop services cleanly"
        rollback_success=false
    fi

    # Phase 2: Restore configuration
    if ! restore_configuration; then
        log_error "Configuration restoration had issues"
        rollback_success=false
    fi

    # Phase 3: Rollback Docker containers
    if ! rollback_docker_containers; then
        log_error "Docker rollback had issues"
        rollback_success=false
    fi

    # Phase 4: Start services
    if [ "$DRY_RUN" = false ]; then
        if ! start_services; then
            log_error "Failed to start services"
            rollback_success=false
        fi

        # Wait for services to stabilize
        log_info "Waiting for services to stabilize..."
        sleep 10
    fi

    # Phase 5: Health checks
    if [ "$DRY_RUN" = false ]; then
        if ! perform_health_checks; then
            log_error "Health checks failed!"
            rollback_success=false
        fi
    fi

    # Phase 6: Generate report
    generate_incident_report

    # Final status
    if [ "$rollback_success" = true ]; then
        echo -e "${GREEN}"
        echo "╔══════════════════════════════════════════════╗"
        echo "║         ROLLBACK COMPLETED SUCCESSFULLY      ║"
        echo "╚══════════════════════════════════════════════╝"
        echo -e "${NC}"
        exit 0
    else
        echo -e "${RED}"
        echo "╔══════════════════════════════════════════════╗"
        echo "║      ROLLBACK COMPLETED WITH ERRORS          ║"
        echo "║      MANUAL INTERVENTION MAY BE REQUIRED     ║"
        echo "╚══════════════════════════════════════════════╝"
        echo -e "${NC}"
        exit 1
    fi
}

# Trap errors and interrupts
trap 'log_error "Script interrupted! Services may be in inconsistent state."; exit 130' INT TERM

# Execute main function
main "$@"
