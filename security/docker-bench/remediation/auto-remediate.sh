#!/bin/bash

# Automated Docker Security Remediation System
# Automatically fixes common Docker security misconfigurations for AI Trading Bot

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${BASE_DIR}/config"
LOGS_DIR="${BASE_DIR}/logs"
BACKUP_DIR="${BASE_DIR}/backups"

# Load configuration
REMEDIATION_CONFIG="${CONFIG_DIR}/remediation.conf"
if [ -f "$REMEDIATION_CONFIG" ]; then
    source "$REMEDIATION_CONFIG"
fi

# Default settings
DRY_RUN=${DRY_RUN:-false}
BACKUP_ENABLED=${BACKUP_ENABLED:-true}
AUTO_RESTART_CONTAINERS=${AUTO_RESTART_CONTAINERS:-false}
MAX_REMEDIATION_ATTEMPTS=${MAX_REMEDIATION_ATTEMPTS:-3}
TRADING_BOT_CONTAINERS=${TRADING_BOT_CONTAINERS:-"ai-trading-bot bluefin-service dashboard-backend mcp-memory mcp-omnisearch"}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOGS_DIR}/remediation.log"
}

info() { log "INFO" "${BLUE}$1${NC}"; }
success() { log "SUCCESS" "${GREEN}$1${NC}"; }
warning() { log "WARNING" "${YELLOW}$1${NC}"; }
error() { log "ERROR" "${RED}$1${NC}"; }
debug() { log "DEBUG" "${PURPLE}$1${NC}"; }

# Backup function
create_backup() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
        local backup_location="${BACKUP_DIR}/remediation-backup-${backup_timestamp}"
        
        mkdir -p "$backup_location"
        
        # Backup Docker Compose files
        cp -r ../../docker-compose*.yml "$backup_location/" 2>/dev/null || true
        
        # Backup configuration files
        cp -r ../../config "$backup_location/" 2>/dev/null || true
        
        # Backup current container configurations
        for container in $TRADING_BOT_CONTAINERS; do
            if docker ps -q -f name="$container" &> /dev/null; then
                docker inspect "$container" > "${backup_location}/${container}-config.json" 2>/dev/null || true
            fi
        done
        
        success "Backup created at $backup_location"
        echo "$backup_location"
    fi
}

# Rollback function
rollback_changes() {
    local backup_location="$1"
    
    if [ -n "$backup_location" ] && [ -d "$backup_location" ]; then
        warning "Rolling back changes from backup: $backup_location"
        
        # Restore Docker Compose files
        cp "$backup_location"/*.yml ../../ 2>/dev/null || true
        
        # Restore configuration files
        cp -r "$backup_location/config" ../../ 2>/dev/null || true
        
        success "Rollback completed"
    else
        error "No valid backup location provided for rollback"
    fi
}

# Execute command with dry-run support
execute_command() {
    local command="$1"
    local description="$2"
    
    if [ "$DRY_RUN" = "true" ]; then
        info "[DRY RUN] $description"
        info "[DRY RUN] Command: $command"
        return 0
    else
        info "Executing: $description"
        debug "Command: $command"
        eval "$command"
    fi
}

# Remediation functions

# Fix container user permissions
fix_container_user() {
    local container="$1"
    
    info "Checking user configuration for container: $container"
    
    # Check if container is running as root
    local container_user
    container_user=$(docker inspect "$container" --format '{{.Config.User}}' 2>/dev/null || echo "")
    
    if [ -z "$container_user" ] || [ "$container_user" = "root" ] || [ "$container_user" = "0" ]; then
        warning "Container $container is running as root or no user specified"
        
        # Fix: Add non-root user specification to docker-compose
        local compose_file="../../docker-compose.yml"
        if [ -f "$compose_file" ]; then
            # Check if user directive already exists for this service
            if grep -A 20 "^  $container:" "$compose_file" | grep -q "user:"; then
                info "User directive already exists for $container"
            else
                # Add user directive (this would need manual intervention in real scenario)
                warning "Manual intervention required: Add 'user: \"\${HOST_UID:-1000}:\${HOST_GID:-1000}\"' to $container service"
            fi
        fi
    else
        success "Container $container has appropriate user configuration: $container_user"
    fi
}

# Fix privileged containers
fix_privileged_containers() {
    local container="$1"
    
    info "Checking privileged mode for container: $container"
    
    local is_privileged
    is_privileged=$(docker inspect "$container" --format '{{.HostConfig.Privileged}}' 2>/dev/null || echo "false")
    
    if [ "$is_privileged" = "true" ]; then
        error "Container $container is running in privileged mode - CRITICAL SECURITY ISSUE"
        
        # This requires restarting the container with new configuration
        warning "Manual intervention required: Remove privileged mode from $container"
        warning "Add 'privileged: false' to docker-compose.yml and restart container"
        
        return 1
    else
        success "Container $container is not running in privileged mode"
    fi
}

# Fix capability issues
fix_container_capabilities() {
    local container="$1"
    
    info "Checking capabilities for container: $container"
    
    # Get current capabilities
    local cap_add
    local cap_drop
    cap_add=$(docker inspect "$container" --format '{{.HostConfig.CapAdd}}' 2>/dev/null || echo "[]")
    cap_drop=$(docker inspect "$container" --format '{{.HostConfig.CapDrop}}' 2>/dev/null || echo "[]")
    
    # Check for dangerous capabilities
    if echo "$cap_add" | grep -q "SYS_ADMIN\|DAC_OVERRIDE\|SETUID\|SETGID"; then
        warning "Container $container has potentially dangerous capabilities: $cap_add"
        warning "Review and minimize capabilities in docker-compose.yml"
    fi
    
    # Check if ALL capabilities are dropped (recommended)
    if ! echo "$cap_drop" | grep -q "ALL"; then
        warning "Container $container should drop ALL capabilities and only add required ones"
        info "Recommended: Add 'cap_drop: [\"ALL\"]' to $container service"
    else
        success "Container $container properly drops ALL capabilities"
    fi
}

# Fix read-only filesystem
fix_readonly_filesystem() {
    local container="$1"
    
    info "Checking read-only filesystem for container: $container"
    
    local readonly_rootfs
    readonly_rootfs=$(docker inspect "$container" --format '{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null || echo "false")
    
    if [ "$readonly_rootfs" = "false" ]; then
        warning "Container $container does not have read-only root filesystem"
        info "Recommended: Add 'read_only: true' to $container service in docker-compose.yml"
        info "Ensure proper tmpfs mounts for writable directories"
    else
        success "Container $container has read-only root filesystem"
    fi
}

# Fix security options
fix_security_options() {
    local container="$1"
    
    info "Checking security options for container: $container"
    
    local security_opt
    security_opt=$(docker inspect "$container" --format '{{.HostConfig.SecurityOpt}}' 2>/dev/null || echo "[]")
    
    # Check for no-new-privileges
    if ! echo "$security_opt" | grep -q "no-new-privileges:true"; then
        warning "Container $container should have no-new-privileges security option"
        info "Recommended: Add 'no-new-privileges:true' to security_opt"
    fi
    
    # Check for seccomp profile
    if ! echo "$security_opt" | grep -q "seccomp"; then
        info "Container $container could benefit from custom seccomp profile"
        info "Consider adding seccomp profile for enhanced security"
    fi
}

# Fix network security
fix_network_security() {
    local container="$1"
    
    info "Checking network security for container: $container"
    
    # Check network mode
    local network_mode
    network_mode=$(docker inspect "$container" --format '{{.HostConfig.NetworkMode}}' 2>/dev/null || echo "")
    
    if [ "$network_mode" = "host" ]; then
        error "Container $container is using host network mode - SECURITY RISK"
        warning "Change to bridge or custom network for better isolation"
        return 1
    fi
    
    # Check port bindings
    local port_bindings
    port_bindings=$(docker inspect "$container" --format '{{.NetworkSettings.Ports}}' 2>/dev/null || echo "{}")
    
    if echo "$port_bindings" | grep -q "0.0.0.0"; then
        warning "Container $container has ports bound to all interfaces (0.0.0.0)"
        info "Consider binding to localhost (127.0.0.1) for internal services"
    fi
}

# Fix resource limits
fix_resource_limits() {
    local container="$1"
    
    info "Checking resource limits for container: $container"
    
    # Check memory limit
    local memory_limit
    memory_limit=$(docker inspect "$container" --format '{{.HostConfig.Memory}}' 2>/dev/null || echo "0")
    
    if [ "$memory_limit" = "0" ]; then
        warning "Container $container has no memory limit set"
        info "Recommended: Set memory limits to prevent resource exhaustion"
    fi
    
    # Check CPU limits
    local cpu_quota
    cpu_quota=$(docker inspect "$container" --format '{{.HostConfig.CpuQuota}}' 2>/dev/null || echo "0")
    
    if [ "$cpu_quota" = "0" ]; then
        warning "Container $container has no CPU quota set"
        info "Recommended: Set CPU limits for better resource management"
    fi
}

# Fix volume mounts
fix_volume_mounts() {
    local container="$1"
    
    info "Checking volume mounts for container: $container"
    
    # Get mount information
    local mounts
    mounts=$(docker inspect "$container" --format '{{json .Mounts}}' 2>/dev/null || echo "[]")
    
    # Check for Docker socket mount (dangerous)
    if echo "$mounts" | grep -q "/var/run/docker.sock"; then
        error "Container $container has Docker socket mounted - CRITICAL SECURITY RISK"
        warning "Remove Docker socket mount immediately"
        return 1
    fi
    
    # Check for writable system mounts
    if echo "$mounts" | grep -q '"RW":true' && echo "$mounts" | grep -qE '"/etc"|"/proc"|"/sys"'; then
        warning "Container $container has writable system directory mounts"
        info "Change system mounts to read-only where possible"
    fi
    
    # Check file permissions on mounted volumes
    local data_dir="../../data"
    local logs_dir="../../logs"
    
    if [ -d "$data_dir" ]; then
        local data_perms
        data_perms=$(stat -c %a "$data_dir" 2>/dev/null || echo "000")
        if [ "${data_perms:2:1}" = "7" ]; then
            warning "Data directory has world-writable permissions: $data_perms"
            execute_command "chmod 755 '$data_dir'" "Fix data directory permissions"
        fi
    fi
    
    if [ -d "$logs_dir" ]; then
        local logs_perms
        logs_perms=$(stat -c %a "$logs_dir" 2>/dev/null || echo "000")
        if [ "${logs_perms:2:1}" = "7" ]; then
            warning "Logs directory has world-writable permissions: $logs_perms"
            execute_command "chmod 755 '$logs_dir'" "Fix logs directory permissions"
        fi
    fi
}

# Fix environment variable security
fix_environment_security() {
    local container="$1"
    
    info "Checking environment variable security for container: $container"
    
    # Check for exposed secrets in environment variables
    local env_vars
    env_vars=$(docker inspect "$container" --format '{{json .Config.Env}}' 2>/dev/null || echo "[]")
    
    # Look for potential secrets
    if echo "$env_vars" | grep -iE "(password|secret|key|token)" | grep -v "\\*\\*\\*"; then
        warning "Container $container may have exposed secrets in environment variables"
        info "Use Docker secrets or external secret management instead"
    fi
}

# Main remediation orchestrator
remediate_container() {
    local container="$1"
    local issues_found=0
    local critical_issues=0
    
    info "Starting remediation for container: $container"
    
    # Check if container exists and is running
    if ! docker ps -q -f name="$container" &> /dev/null; then
        warning "Container $container is not running - skipping"
        return 0
    fi
    
    # Run all remediation checks
    fix_container_user "$container" || issues_found=$((issues_found + 1))
    fix_privileged_containers "$container" || { issues_found=$((issues_found + 1)); critical_issues=$((critical_issues + 1)); }
    fix_container_capabilities "$container" || issues_found=$((issues_found + 1))
    fix_readonly_filesystem "$container" || issues_found=$((issues_found + 1))
    fix_security_options "$container" || issues_found=$((issues_found + 1))
    fix_network_security "$container" || { issues_found=$((issues_found + 1)); critical_issues=$((critical_issues + 1)); }
    fix_resource_limits "$container" || issues_found=$((issues_found + 1))
    fix_volume_mounts "$container" || { issues_found=$((issues_found + 1)); critical_issues=$((critical_issues + 1)); }
    fix_environment_security "$container" || issues_found=$((issues_found + 1))
    
    if [ $issues_found -eq 0 ]; then
        success "No remediation needed for container: $container"
    else
        warning "Found $issues_found issues for container $container ($critical_issues critical)"
        
        if [ $critical_issues -gt 0 ]; then
            error "Critical security issues found in $container - manual intervention required"
            return 1
        fi
    fi
    
    return 0
}

# System-wide security fixes
fix_system_security() {
    info "Checking system-wide Docker security configuration..."
    
    # Check Docker daemon configuration
    local docker_daemon_config="/etc/docker/daemon.json"
    if [ -f "$docker_daemon_config" ]; then
        # Check for security-related configurations
        if ! grep -q '"live-restore": true' "$docker_daemon_config"; then
            info "Consider enabling live-restore in Docker daemon configuration"
        fi
        
        if ! grep -q '"userland-proxy": false' "$docker_daemon_config"; then
            info "Consider disabling userland-proxy for better performance"
        fi
    else
        warning "Docker daemon configuration file not found"
        info "Consider creating $docker_daemon_config with security configurations"
    fi
    
    # Check for Docker socket permissions
    if [ -S "/var/run/docker.sock" ]; then
        local socket_perms
        socket_perms=$(stat -c %a "/var/run/docker.sock" 2>/dev/null || echo "000")
        if [ "$socket_perms" = "666" ]; then
            warning "Docker socket has permissive permissions: $socket_perms"
            info "Consider restricting Docker socket access"
        fi
    fi
}

# Main execution function
main() {
    local analysis_file="$1"
    local total_issues=0
    local critical_issues=0
    local backup_location=""
    
    info "Starting automated Docker security remediation"
    
    if [ "$DRY_RUN" = "true" ]; then
        warning "Running in DRY RUN mode - no changes will be made"
    fi
    
    # Create backup if enabled
    if [ "$BACKUP_ENABLED" = "true" ] && [ "$DRY_RUN" = "false" ]; then
        backup_location=$(create_backup)
    fi
    
    # Parse analysis file if provided
    if [ -n "$analysis_file" ] && [ -f "$analysis_file" ]; then
        info "Using analysis file: $analysis_file"
        if command -v jq &> /dev/null; then
            total_issues=$(jq -r '.summary.critical_issues + .summary.high_issues + .summary.medium_issues + .summary.low_issues' "$analysis_file" 2>/dev/null || echo "0")
            critical_issues=$(jq -r '.summary.critical_issues' "$analysis_file" 2>/dev/null || echo "0")
        fi
    fi
    
    # System-wide fixes
    fix_system_security
    
    # Container-specific remediation
    local containers_processed=0
    local containers_failed=0
    
    for container in $TRADING_BOT_CONTAINERS; do
        containers_processed=$((containers_processed + 1))
        
        if ! remediate_container "$container"; then
            containers_failed=$((containers_failed + 1))
            error "Remediation failed for container: $container"
        fi
    done
    
    # Summary
    info "Remediation completed:"
    info "  - Containers processed: $containers_processed"
    info "  - Containers failed: $containers_failed"
    info "  - Total issues identified: $total_issues"
    info "  - Critical issues: $critical_issues"
    
    if [ $containers_failed -gt 0 ]; then
        error "Some containers failed remediation - manual intervention required"
        
        # Offer rollback option
        if [ -n "$backup_location" ] && [ "$DRY_RUN" = "false" ]; then
            warning "Backup available for rollback: $backup_location"
        fi
        
        return 1
    else
        success "All containers successfully processed"
        
        # Restart containers if configured and changes were made
        if [ "$AUTO_RESTART_CONTAINERS" = "true" ] && [ "$DRY_RUN" = "false" ] && [ $total_issues -gt 0 ]; then
            info "Restarting containers to apply security changes..."
            execute_command "docker-compose restart" "Restart all containers"
        fi
        
        return 0
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "${1:-}"
fi