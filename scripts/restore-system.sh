#!/bin/bash

# AI Trading Bot System Restore Script
# This script restores system components from backups created by backup-system.sh
# Usage: ./scripts/restore-system.sh <backup_name> [options]

set -euo pipefail

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

log_prompt() {
    echo -e "${CYAN}[PROMPT]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Restore configuration
BACKUP_NAME="${1:-}"
BACKUP_ROOT="${PROJECT_ROOT}/backups"
RESTORE_LOG="${PROJECT_ROOT}/restore_$(date +%Y%m%d_%H%M%S).log"
ROLLBACK_DIR="${PROJECT_ROOT}/.rollback_$(date +%Y%m%d_%H%M%S)"

# Options
DRY_RUN="${DRY_RUN:-false}"
INTERACTIVE="${INTERACTIVE:-true}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
COMPONENTS_TO_RESTORE="${COMPONENTS_TO_RESTORE:-all}"

# Parse command line options
parse_options() {
    while [[ $# -gt 1 ]]; do
        case "$2" in
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --no-interactive)
                INTERACTIVE="false"
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION="true"
                shift
                ;;
            --components)
                COMPONENTS_TO_RESTORE="$3"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
}

# Show usage
show_usage() {
    cat <<EOF
Usage: $0 <backup_name> [options]

Options:
    --dry-run           Show what would be restored without making changes
    --no-interactive    Don't prompt for confirmation
    --skip-validation   Skip backup integrity validation
    --components <list> Comma-separated list of components to restore
                       (config,env,logs,data,mcp_memory,docker_volumes,prompts)

Examples:
    # Restore full backup interactively
    $0 backup_20240115_120000

    # Dry run to see what would be restored
    $0 backup_20240115_120000 --dry-run

    # Restore only config and data
    $0 backup_20240115_120000 --components config,data

    # Non-interactive full restore
    $0 backup_20240115_120000 --no-interactive
EOF
}

# Initialize restore
initialize_restore() {
    # Check for help flag
    if [[ "${BACKUP_NAME}" == "--help" ]] || [[ "${BACKUP_NAME}" == "-h" ]]; then
        show_usage
        exit 0
    fi

    # Start logging
    exec > >(tee -a "${RESTORE_LOG}")
    exec 2>&1

    log_info "AI Trading Bot System Restore"
    log_info "Restore log: ${RESTORE_LOG}"

    # Validate backup name provided
    if [[ -z "${BACKUP_NAME}" ]]; then
        log_error "No backup name provided!"
        show_usage
        exit 1
    fi

    # Determine backup path
    BACKUP_PATH=""
    if [[ -f "${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz" ]]; then
        # Compressed backup
        BACKUP_PATH="${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz"
        BACKUP_TYPE="compressed"
        log_info "Found compressed backup: ${BACKUP_PATH}"
    elif [[ -d "${BACKUP_ROOT}/${BACKUP_NAME}" ]]; then
        # Uncompressed backup
        BACKUP_PATH="${BACKUP_ROOT}/${BACKUP_NAME}"
        BACKUP_TYPE="directory"
        log_info "Found backup directory: ${BACKUP_PATH}"
    else
        log_error "Backup not found: ${BACKUP_NAME}"
        log_error "Searched in: ${BACKUP_ROOT}"

        # List available backups
        log_info "Available backups:"
        if [[ -d "${BACKUP_ROOT}" ]]; then
            ls -la "${BACKUP_ROOT}"
        else
            log_error "Backup directory does not exist: ${BACKUP_ROOT}"
        fi
        exit 1
    fi
}

# Extract compressed backup
extract_backup() {
    if [[ "${BACKUP_TYPE}" == "compressed" ]]; then
        log_info "Extracting compressed backup..."

        TEMP_EXTRACT_DIR="${BACKUP_ROOT}/.temp_extract_$$"
        mkdir -p "${TEMP_EXTRACT_DIR}"

        if tar -xzf "${BACKUP_PATH}" -C "${TEMP_EXTRACT_DIR}"; then
            BACKUP_DIR="${TEMP_EXTRACT_DIR}/${BACKUP_NAME}"
            log_info "✓ Backup extracted successfully"
        else
            log_error "✗ Failed to extract backup"
            exit 1
        fi
    else
        BACKUP_DIR="${BACKUP_PATH}"
    fi

    # Verify backup directory structure
    if [[ ! -f "${BACKUP_DIR}/manifest.json" ]]; then
        log_error "Invalid backup: manifest.json not found"
        exit 1
    fi
}

# Load and display backup manifest
load_manifest() {
    log_info "Loading backup manifest..."

    python3 - <<EOF
import json
import sys
from datetime import datetime

manifest_file = "${BACKUP_DIR}/manifest.json"

try:
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    print("\n${CYAN}=== Backup Information ===${NC}")
    print(f"Backup Name: {manifest['backup_name']}")
    print(f"Backup Date: {manifest['backup_date']}")
    print(f"System: {manifest['system_info']['hostname']}")
    print(f"Git Commit: {manifest['system_info']['git_commit'][:8]}")

    print("\n${CYAN}=== Components Status ===${NC}")
    print(f"{'Component':<20} {'Status':<10} {'Size':<10} {'Notes':<30}")
    print("-" * 70)

    components = manifest.get('components', {})
    for comp, info in components.items():
        status = info.get('status', 'unknown')
        size = info.get('size', '-')
        notes = info.get('error', '-')[:30]

        # Color code status
        if status == 'success':
            status_display = "${GREEN}" + status + "${NC}"
        elif status == 'partial':
            status_display = "${YELLOW}" + status + "${NC}"
        elif status == 'skipped':
            status_display = "${BLUE}" + status + "${NC}"
        else:
            status_display = "${RED}" + status + "${NC}"

        print(f"{comp:<20} {status:<10} {size:<10} {notes:<30}")

    # Save components list
    with open('/tmp/backup_components.txt', 'w') as f:
        f.write(','.join(components.keys()))

except Exception as e:
    print(f"${RED}Error loading manifest: {e}${NC}")
    sys.exit(1)
EOF

    # Read available components
    AVAILABLE_COMPONENTS=$(cat /tmp/backup_components.txt 2>/dev/null || echo "")
    rm -f /tmp/backup_components.txt
}

# Validate backup integrity
validate_backup() {
    if [[ "${SKIP_VALIDATION}" == "true" ]]; then
        log_warn "Skipping backup validation as requested"
        return 0
    fi

    log_info "Validating backup integrity..."

    local validation_passed=true

    # Check manifest
    if [[ ! -f "${BACKUP_DIR}/manifest.json" ]]; then
        log_error "✗ Missing manifest.json"
        validation_passed=false
    fi

    # Check component directories/files
    python3 - <<EOF
import json
import os
import sys

manifest_file = "${BACKUP_DIR}/manifest.json"
backup_dir = "${BACKUP_DIR}"
validation_passed = True

try:
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    components = manifest.get('components', {})

    for comp, info in components.items():
        if info['status'] in ['success', 'partial']:
            # Check if component backup exists
            comp_path = os.path.join(backup_dir, comp)

            if comp == 'logs':
                # Logs are compressed
                if not os.path.exists(os.path.join(comp_path, 'logs.tar.gz')):
                    print(f"${RED}✗ Missing backup for component: {comp}${NC}")
                    validation_passed = False
                else:
                    print(f"${GREEN}✓ Validated: {comp}${NC}")
            elif os.path.exists(comp_path):
                print(f"${GREEN}✓ Validated: {comp}${NC}")
            else:
                print(f"${RED}✗ Missing backup for component: {comp}${NC}")
                validation_passed = False

    sys.exit(0 if validation_passed else 1)

except Exception as e:
    print(f"${RED}Validation error: {e}${NC}")
    sys.exit(1)
EOF

    if [[ $? -ne 0 ]]; then
        validation_passed=false
    fi

    if [[ "${validation_passed}" == "true" ]]; then
        log_info "✓ Backup validation passed"
        return 0
    else
        log_error "✗ Backup validation failed"
        if [[ "${INTERACTIVE}" == "true" ]]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            exit 1
        fi
    fi
}

# Create rollback backup
create_rollback() {
    log_info "Creating rollback backup of current state..."

    mkdir -p "${ROLLBACK_DIR}"

    # Quick backup of critical files only
    local rollback_items=(
        "config"
        "data/paper_trading"
        "data/positions"
        "data/inventory"
        ".env"
    )

    for item in "${rollback_items[@]}"; do
        if [[ -e "${PROJECT_ROOT}/${item}" ]]; then
            local target_dir=$(dirname "${ROLLBACK_DIR}/${item}")
            mkdir -p "${target_dir}"
            cp -r "${PROJECT_ROOT}/${item}" "${target_dir}/" 2>/dev/null || true
        fi
    done

    log_info "✓ Rollback backup created: ${ROLLBACK_DIR}"
}

# Restore component with dry-run support
restore_component() {
    local component="$1"
    local source_path="$2"
    local target_path="$3"
    local restore_method="${4:-copy}"

    log_info "Restoring ${component}..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore ${component} from ${source_path} to ${target_path}"
        return 0
    fi

    case "${restore_method}" in
        "copy")
            if cp -r "${source_path}" "${target_path}"; then
                log_info "✓ ${component} restored"
                return 0
            else
                log_error "✗ Failed to restore ${component}"
                return 1
            fi
            ;;
        "extract")
            if tar -xzf "${source_path}" -C "${target_path}"; then
                log_info "✓ ${component} restored"
                return 0
            else
                log_error "✗ Failed to extract ${component}"
                return 1
            fi
            ;;
        "rsync")
            if rsync -av --delete "${source_path}/" "${target_path}/"; then
                log_info "✓ ${component} restored"
                return 0
            else
                log_error "✗ Failed to sync ${component}"
                return 1
            fi
            ;;
    esac
}

# Restore configuration files
restore_config() {
    if [[ ! -d "${BACKUP_DIR}/config/config" ]]; then
        log_warn "Config backup not found, skipping..."
        return 0
    fi

    restore_component "configuration files" \
        "${BACKUP_DIR}/config/config" \
        "${PROJECT_ROOT}/" \
        "rsync"
}

# Restore environment files
restore_env() {
    if [[ ! -d "${BACKUP_DIR}/env" ]]; then
        log_warn "Environment backup not found, skipping..."
        return 0
    fi

    log_info "Restoring environment configuration..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore environment files"
        return 0
    fi

    # Check if .env exists in backup
    if [[ -f "${BACKUP_DIR}/env/.env" ]]; then
        log_warn "⚠️ Found masked .env file in backup (sensitive data redacted)"
        log_warn "You will need to manually restore API keys and secrets"

        if [[ "${INTERACTIVE}" == "true" ]]; then
            read -p "View masked .env file? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                cat "${BACKUP_DIR}/env/.env"
            fi
        fi

        # Copy to .env.restored for manual merge
        cp "${BACKUP_DIR}/env/.env" "${PROJECT_ROOT}/.env.restored"
        log_info "Masked .env saved to: .env.restored"
        log_info "Please manually merge with your existing .env file"
    fi

    # Restore .env.example if present
    if [[ -f "${BACKUP_DIR}/env/.env.example" ]]; then
        cp "${BACKUP_DIR}/env/.env.example" "${PROJECT_ROOT}/.env.example"
        log_info "✓ .env.example restored"
    fi
}

# Restore logs
restore_logs() {
    if [[ ! -d "${BACKUP_DIR}/logs" ]]; then
        log_warn "Logs backup not found, skipping..."
        return 0
    fi

    log_info "Restoring logs..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore logs"
        return 0
    fi

    # Create logs directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/logs"

    # Extract logs archive
    if [[ -f "${BACKUP_DIR}/logs/logs.tar.gz" ]]; then
        tar -xzf "${BACKUP_DIR}/logs/logs.tar.gz" -C "${PROJECT_ROOT}/"
        log_info "✓ Logs restored"
    else
        log_error "✗ Logs archive not found"
        return 1
    fi
}

# Restore data directory
restore_data() {
    if [[ ! -d "${BACKUP_DIR}/data" ]]; then
        log_warn "Data backup not found, skipping..."
        return 0
    fi

    log_info "Restoring data directory..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore data directory"
        return 0
    fi

    # Ask about paper trading data
    if [[ "${INTERACTIVE}" == "true" ]] && [[ -d "${BACKUP_DIR}/data/paper_trading" ]]; then
        log_prompt "Found paper trading data in backup."
        read -p "Restore paper trading accounts and history? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            RESTORE_PAPER_TRADING="true"
        else
            RESTORE_PAPER_TRADING="false"
        fi
    else
        RESTORE_PAPER_TRADING="true"
    fi

    # Restore data with exclusions if needed
    if [[ "${RESTORE_PAPER_TRADING}" == "false" ]]; then
        rsync -av --delete \
            --exclude='paper_trading/' \
            "${BACKUP_DIR}/data/" "${PROJECT_ROOT}/data/"
    else
        rsync -av --delete "${BACKUP_DIR}/data/" "${PROJECT_ROOT}/data/"
    fi

    log_info "✓ Data directory restored"
}

# Restore MCP memory
restore_mcp_memory() {
    if [[ ! -d "${BACKUP_DIR}/mcp_memory" ]]; then
        log_warn "MCP memory backup not found, skipping..."
        return 0
    fi

    restore_component "MCP memory data" \
        "${BACKUP_DIR}/mcp_memory/mcp_memory" \
        "${PROJECT_ROOT}/data/" \
        "rsync"
}

# Restore prompts
restore_prompts() {
    if [[ ! -d "${BACKUP_DIR}/prompts" ]]; then
        log_warn "Prompts backup not found, skipping..."
        return 0
    fi

    restore_component "LLM prompts" \
        "${BACKUP_DIR}/prompts/prompts" \
        "${PROJECT_ROOT}/" \
        "rsync"
}

# Restore Docker volumes
restore_docker_volumes() {
    if [[ ! -d "${BACKUP_DIR}/docker_volumes" ]]; then
        log_warn "Docker volumes backup not found, skipping..."
        return 0
    fi

    log_info "Restoring Docker volumes..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore Docker volumes"
        return 0
    fi

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not available, skipping volume restore..."
        return 0
    fi

    # List of volumes to restore
    local volumes=(
        "dashboard-logs"
        "dashboard-data"
    )

    for volume in "${volumes[@]}"; do
        local backup_path="${BACKUP_DIR}/docker_volumes/${volume}"

        if [[ -d "${backup_path}" ]]; then
            log_info "Restoring volume: ${volume}"

            # Create volume if it doesn't exist
            docker volume create "${volume}" &> /dev/null || true

            # Create temporary container for restoration
            local temp_container="restore_temp_$(date +%s)"

            # Run Alpine container with volume mounted
            if docker run -d --name "${temp_container}" \
                -v "${volume}:/restore_target" \
                alpine:latest tail -f /dev/null &> /dev/null; then

                # Copy data to container
                if docker cp "${backup_path}/." "${temp_container}:/restore_target/"; then
                    log_info "✓ Volume ${volume} restored"
                else
                    log_error "✗ Failed to restore volume ${volume}"
                fi

                # Clean up container
                docker rm -f "${temp_container}" &> /dev/null
            else
                log_error "✗ Failed to create container for volume ${volume}"
            fi
        else
            log_warn "Backup for volume ${volume} not found"
        fi
    done
}

# Post-restore validation
post_restore_validation() {
    log_info "Performing post-restore validation..."

    local validation_passed=true

    # Check critical files
    local critical_files=(
        "config/production.json"
        "config/development.json"
        "prompts/trade_action.txt"
    )

    for file in "${critical_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
            log_info "✓ Found: ${file}"
        else
            log_warn "⚠ Missing: ${file}"
        fi
    done

    # Check directory permissions
    local dirs_to_check=(
        "logs"
        "data"
        "config"
    )

    for dir in "${dirs_to_check[@]}"; do
        if [[ -d "${PROJECT_ROOT}/${dir}" ]]; then
            if [[ -w "${PROJECT_ROOT}/${dir}" ]]; then
                log_info "✓ Writable: ${dir}/"
            else
                log_error "✗ Not writable: ${dir}/"
                validation_passed=false
            fi
        fi
    done

    if [[ "${validation_passed}" == "true" ]]; then
        log_info "✓ Post-restore validation passed"
    else
        log_error "✗ Post-restore validation failed"
        log_warn "Run ./setup-docker-permissions.sh to fix permissions"
    fi
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."

    # Remove extracted backup if it was compressed
    if [[ "${BACKUP_TYPE}" == "compressed" ]] && [[ -d "${TEMP_EXTRACT_DIR}" ]]; then
        rm -rf "${TEMP_EXTRACT_DIR}"
    fi

    # Remove temporary files
    rm -f /tmp/backup_components.txt
}

# Rollback on failure
rollback() {
    log_error "Restore failed! Starting rollback..."

    if [[ -d "${ROLLBACK_DIR}" ]]; then
        log_info "Restoring from rollback: ${ROLLBACK_DIR}"

        # Restore critical files
        cp -r "${ROLLBACK_DIR}"/* "${PROJECT_ROOT}/" 2>/dev/null || true

        log_info "✓ Rollback completed"
        log_info "Previous state restored from: ${ROLLBACK_DIR}"
    else
        log_error "No rollback available!"
    fi
}

# Main restore process
main() {
    # Parse options
    parse_options "$@"

    # Initialize
    initialize_restore

    # Extract backup if compressed
    extract_backup

    # Load and display manifest
    load_manifest

    # Validate backup
    validate_backup

    # Confirm restoration
    if [[ "${INTERACTIVE}" == "true" ]] && [[ "${DRY_RUN}" == "false" ]]; then
        echo
        log_prompt "Ready to restore from backup: ${BACKUP_NAME}"
        log_prompt "Components to restore: ${COMPONENTS_TO_RESTORE}"
        read -p "Continue with restoration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Restoration cancelled"
            cleanup
            exit 0
        fi
    fi

    # Create rollback backup
    if [[ "${DRY_RUN}" == "false" ]]; then
        create_rollback
    fi

    # Determine components to restore
    if [[ "${COMPONENTS_TO_RESTORE}" == "all" ]]; then
        COMPONENTS_TO_RESTORE="${AVAILABLE_COMPONENTS}"
    fi

    # Convert comma-separated list to array
    IFS=',' read -ra RESTORE_COMPONENTS <<< "${COMPONENTS_TO_RESTORE}"

    # Restore components
    local failed=0
    for component in "${RESTORE_COMPONENTS[@]}"; do
        case "${component}" in
            "config")
                restore_config || ((failed++))
                ;;
            "env")
                restore_env || ((failed++))
                ;;
            "logs")
                restore_logs || ((failed++))
                ;;
            "data")
                restore_data || ((failed++))
                ;;
            "mcp_memory")
                restore_mcp_memory || ((failed++))
                ;;
            "prompts")
                restore_prompts || ((failed++))
                ;;
            "docker_volumes")
                restore_docker_volumes || ((failed++))
                ;;
            *)
                log_warn "Unknown component: ${component}, skipping..."
                ;;
        esac
    done

    # Post-restore validation
    if [[ "${DRY_RUN}" == "false" ]]; then
        post_restore_validation
    fi

    # Cleanup
    cleanup

    # Final status
    echo
    if [[ ${failed} -eq 0 ]]; then
        log_info "✅ Restore completed successfully!"

        if [[ "${DRY_RUN}" == "false" ]]; then
            log_info "Rollback backup saved at: ${ROLLBACK_DIR}"

            # Provide next steps
            echo
            log_info "Next steps:"
            log_info "1. If you have .env.restored, merge it with your .env file"
            log_info "2. Run: ./setup-docker-permissions.sh"
            log_info "3. Restart Docker services: docker-compose restart"
            log_info "4. Check service health: docker-compose ps"
        fi
    else
        log_error "✗ Restore completed with ${failed} errors"

        if [[ "${DRY_RUN}" == "false" ]]; then
            read -p "Rollback to previous state? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rollback
            fi
        fi

        exit 1
    fi
}

# Handle interrupts
trap 'log_error "Restore interrupted!"; cleanup; exit 130' INT TERM

# Run main function
main "$@"
