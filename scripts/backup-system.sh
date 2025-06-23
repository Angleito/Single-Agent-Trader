#!/bin/bash

# AI Trading Bot System Backup Script
# This script backs up all critical components including config, logs, data, and Docker volumes
# Usage: ./scripts/backup-system.sh [backup_name]

set -euo pipefail

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Backup configuration
BACKUP_NAME="${1:-backup_$(date +%Y%m%d_%H%M%S)}"
BACKUP_ROOT="${PROJECT_ROOT}/backups"
BACKUP_DIR="${BACKUP_ROOT}/${BACKUP_NAME}"
BACKUP_MANIFEST="${BACKUP_DIR}/manifest.json"
BACKUP_LOG="${BACKUP_DIR}/backup.log"

# Components to backup - using arrays instead of associative arrays for compatibility
BACKUP_COMPONENTS_NAMES=("config" "logs" "data" "docker_volumes" "mcp_memory" "env" "prompts")
BACKUP_COMPONENTS_DESC=(
    "Configuration files"
    "Trading and system logs"
    "Database and state files"
    "Docker volume data"
    "MCP memory and learning data"
    "Environment configuration"
    "LLM prompts"
)

# Initialize backup
initialize_backup() {
    log_info "Initializing backup: ${BACKUP_NAME}"

    # Create backup directory
    mkdir -p "${BACKUP_DIR}"

    # Start logging
    exec > >(tee -a "${BACKUP_LOG}")
    exec 2>&1

    # Create manifest
    cat > "${BACKUP_MANIFEST}" <<EOF
{
    "backup_name": "${BACKUP_NAME}",
    "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "system_info": {
        "hostname": "$(hostname)",
        "user": "$(whoami)",
        "project_root": "${PROJECT_ROOT}",
        "git_commit": "$(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "git_branch": "$(cd "${PROJECT_ROOT}" && git branch --show-current 2>/dev/null || echo 'unknown')"
    },
    "components": {},
    "docker_status": {}
}
EOF

    log_info "Backup directory created: ${BACKUP_DIR}"
}

# Update manifest with component info
update_manifest() {
    local component="$1"
    local status="$2"
    local size="$3"
    local error="${4:-}"

    python3 - <<EOF
import json
import sys

manifest_file = "${BACKUP_MANIFEST}"
with open(manifest_file, 'r') as f:
    manifest = json.load(f)

manifest['components']['${component}'] = {
    'status': '${status}',
    'size': '${size}',
    'error': '${error}',
    'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
}

with open(manifest_file, 'w') as f:
    json.dump(manifest, f, indent=2)
EOF
}

# Backup configuration files
backup_config() {
    log_info "Backing up configuration files..."
    local component="config"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    # Backup all config files
    if cp -r "${PROJECT_ROOT}/config" "${target_dir}/" 2>/dev/null; then
        # Calculate size
        local size=$(du -sh "${target_dir}" | cut -f1)
        update_manifest "${component}" "success" "${size}"
        log_info "✓ Configuration files backed up (${size})"
    else
        update_manifest "${component}" "failed" "0" "Failed to copy config files"
        log_error "✗ Failed to backup configuration files"
        return 1
    fi
}

# Backup environment files
backup_env() {
    log_info "Backing up environment configuration..."
    local component="env"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    # Backup .env files (with masking of sensitive data)
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        log_info "Masking sensitive data in .env file..."
        sed -E 's/(API_KEY|PRIVATE_KEY|SECRET|PASSWORD|TOKEN)=.*/\1=<REDACTED>/g' \
            "${PROJECT_ROOT}/.env" > "${target_dir}/.env"

        # Store hash of original for verification
        echo "$(sha256sum "${PROJECT_ROOT}/.env" | cut -d' ' -f1)" > "${target_dir}/.env.sha256"
    fi

    # Backup .env.example
    [[ -f "${PROJECT_ROOT}/.env.example" ]] && cp "${PROJECT_ROOT}/.env.example" "${target_dir}/"

    local size=$(du -sh "${target_dir}" | cut -f1)
    update_manifest "${component}" "success" "${size}"
    log_info "✓ Environment configuration backed up (${size})"
}

# Backup logs with compression
backup_logs() {
    log_info "Backing up logs..."
    local component="logs"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    # Create tar archive of logs
    if tar -czf "${target_dir}/logs.tar.gz" -C "${PROJECT_ROOT}" logs/ 2>/dev/null; then
        local size=$(du -sh "${target_dir}/logs.tar.gz" | cut -f1)
        update_manifest "${component}" "success" "${size}"
        log_info "✓ Logs backed up and compressed (${size})"
    else
        update_manifest "${component}" "failed" "0" "Failed to archive logs"
        log_error "✗ Failed to backup logs"
        return 1
    fi
}

# Backup data directory
backup_data() {
    log_info "Backing up data directory..."
    local component="data"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    # Exclude temporary and cache files
    if rsync -av --progress \
        --exclude='*.tmp' \
        --exclude='*_cache/' \
        --exclude='omnisearch_cache/' \
        --exclude='test_fallback_cache/' \
        "${PROJECT_ROOT}/data/" "${target_dir}/" 2>/dev/null; then
        local size=$(du -sh "${target_dir}" | cut -f1)
        update_manifest "${component}" "success" "${size}"
        log_info "✓ Data directory backed up (${size})"
    else
        update_manifest "${component}" "failed" "0" "Failed to sync data directory"
        log_error "✗ Failed to backup data directory"
        return 1
    fi
}

# Backup MCP memory data
backup_mcp_memory() {
    log_info "Backing up MCP memory data..."
    local component="mcp_memory"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    # Check if MCP memory directory exists
    if [[ -d "${PROJECT_ROOT}/data/mcp_memory" ]]; then
        if cp -r "${PROJECT_ROOT}/data/mcp_memory" "${target_dir}/" 2>/dev/null; then
            local size=$(du -sh "${target_dir}" | cut -f1)
            update_manifest "${component}" "success" "${size}"
            log_info "✓ MCP memory data backed up (${size})"
        else
            update_manifest "${component}" "failed" "0" "Failed to copy MCP memory data"
            log_error "✗ Failed to backup MCP memory data"
            return 1
        fi
    else
        update_manifest "${component}" "skipped" "0" "MCP memory directory not found"
        log_warn "⚠ MCP memory directory not found, skipping..."
    fi
}

# Backup prompts
backup_prompts() {
    log_info "Backing up LLM prompts..."
    local component="prompts"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    if cp -r "${PROJECT_ROOT}/prompts" "${target_dir}/" 2>/dev/null; then
        local size=$(du -sh "${target_dir}" | cut -f1)
        update_manifest "${component}" "success" "${size}"
        log_info "✓ LLM prompts backed up (${size})"
    else
        update_manifest "${component}" "failed" "0" "Failed to copy prompts"
        log_error "✗ Failed to backup prompts"
        return 1
    fi
}

# Check Docker status
check_docker_status() {
    log_info "Checking Docker services status..."

    python3 - <<EOF
import json
import subprocess
import sys

manifest_file = "${BACKUP_MANIFEST}"
with open(manifest_file, 'r') as f:
    manifest = json.load(f)

# Get Docker container status
try:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=True
    )

    containers = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            container = json.loads(line)
            name = container.get('Names', '')
            if any(svc in name for svc in ['ai-trading-bot', 'mcp-memory', 'mcp-omnisearch', 'bluefin-service', 'dashboard']):
                containers[name] = {
                    'status': container.get('Status', 'unknown'),
                    'image': container.get('Image', 'unknown'),
                    'created': container.get('CreatedAt', 'unknown')
                }

    manifest['docker_status']['containers'] = containers
    manifest['docker_status']['docker_available'] = True
except Exception as e:
    manifest['docker_status']['docker_available'] = False
    manifest['docker_status']['error'] = str(e)

with open(manifest_file, 'w') as f:
    json.dump(manifest, f, indent=2)
EOF
}

# Backup Docker volumes
backup_docker_volumes() {
    log_info "Backing up Docker volumes..."
    local component="docker_volumes"
    local target_dir="${BACKUP_DIR}/${component}"

    mkdir -p "${target_dir}"

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        update_manifest "${component}" "skipped" "0" "Docker not available"
        log_warn "⚠ Docker not available, skipping volume backup..."
        return 0
    fi

    # List of volumes to backup
    local volumes=(
        "dashboard-logs"
        "dashboard-data"
    )

    local total_size=0
    local failed=0

    for volume in "${volumes[@]}"; do
        log_info "Backing up volume: ${volume}"

        # Check if volume exists
        if docker volume inspect "${volume}" &> /dev/null; then
            # Create temporary container to access volume data
            local temp_container="backup_temp_$(date +%s)"

            # Run Alpine container with volume mounted
            if docker run -d --name "${temp_container}" \
                -v "${volume}:/backup_source:ro" \
                alpine:latest tail -f /dev/null &> /dev/null; then

                # Copy data from container
                if docker cp "${temp_container}:/backup_source" "${target_dir}/${volume}" &> /dev/null; then
                    log_info "✓ Volume ${volume} backed up"
                else
                    log_error "✗ Failed to copy data from volume ${volume}"
                    ((failed++))
                fi

                # Clean up container
                docker rm -f "${temp_container}" &> /dev/null
            else
                log_error "✗ Failed to create container for volume ${volume}"
                ((failed++))
            fi
        else
            log_warn "⚠ Volume ${volume} not found, skipping..."
        fi
    done

    # Update manifest
    if [[ ${failed} -eq 0 ]]; then
        local size=$(du -sh "${target_dir}" 2>/dev/null | cut -f1 || echo "0")
        update_manifest "${component}" "success" "${size}"
        log_info "✓ Docker volumes backed up (${size})"
    else
        update_manifest "${component}" "partial" "unknown" "${failed} volumes failed"
        log_warn "⚠ Docker volume backup completed with ${failed} failures"
    fi
}

# Create backup summary
create_summary() {
    log_info "Creating backup summary..."

    python3 - <<EOF
import json
import sys
from datetime import datetime

manifest_file = "${BACKUP_MANIFEST}"
summary_file = "${BACKUP_DIR}/BACKUP_SUMMARY.md"

with open(manifest_file, 'r') as f:
    manifest = json.load(f)

# Create markdown summary
with open(summary_file, 'w') as f:
    f.write("# AI Trading Bot Backup Summary\n\n")
    f.write(f"**Backup Name:** {manifest['backup_name']}\n")
    f.write(f"**Date:** {manifest['backup_date']}\n")
    f.write(f"**System:** {manifest['system_info']['hostname']}\n")
    f.write(f"**Git Commit:** {manifest['system_info']['git_commit'][:8]}\n\n")

    f.write("## Components Backed Up\n\n")
    f.write("| Component | Status | Size | Notes |\n")
    f.write("|-----------|--------|------|-------|\n")

    for comp, info in manifest['components'].items():
        status_icon = "✅" if info['status'] == 'success' else "⚠️" if info['status'] == 'partial' else "❌"
        notes = info.get('error', '-')
        f.write(f"| {comp} | {status_icon} {info['status']} | {info['size']} | {notes} |\n")

    if manifest.get('docker_status', {}).get('containers'):
        f.write("\n## Docker Services Status at Backup\n\n")
        f.write("| Service | Status | Image |\n")
        f.write("|---------|--------|-------|\n")
        for name, info in manifest['docker_status']['containers'].items():
            f.write(f"| {name} | {info['status']} | {info['image']} |\n")

    f.write("\n## Restoration Instructions\n\n")
    f.write("To restore this backup, run:\n")
    f.write(f"\`\`\`bash\n./scripts/restore-system.sh {manifest['backup_name']}\n\`\`\`\n")

print(f"Backup summary created: {summary_file}")
EOF
}

# Compress backup
compress_backup() {
    log_info "Compressing backup..."

    cd "${BACKUP_ROOT}"
    local archive_name="${BACKUP_NAME}.tar.gz"

    if tar -czf "${archive_name}" "${BACKUP_NAME}/"; then
        local size=$(du -sh "${archive_name}" | cut -f1)
        log_info "✓ Backup compressed: ${archive_name} (${size})"

        # Optionally remove uncompressed backup
        if [[ "${COMPRESS_ONLY:-false}" == "true" ]]; then
            rm -rf "${BACKUP_DIR}"
            log_info "Uncompressed backup removed"
        fi
    else
        log_error "✗ Failed to compress backup"
        return 1
    fi
}

# Main backup process
main() {
    log_info "Starting AI Trading Bot system backup..."
    log_info "Project root: ${PROJECT_ROOT}"

    # Initialize backup
    initialize_backup

    # Check Docker status first
    check_docker_status

    # Backup components
    local components=(
        "backup_config"
        "backup_env"
        "backup_prompts"
        "backup_logs"
        "backup_data"
        "backup_mcp_memory"
        "backup_docker_volumes"
    )

    local failed=0
    for component in "${components[@]}"; do
        if ! ${component}; then
            ((failed++))
        fi
    done

    # Create summary
    create_summary

    # Compress backup
    compress_backup

    # Final status
    if [[ ${failed} -eq 0 ]]; then
        log_info "✅ Backup completed successfully!"
        log_info "Backup location: ${BACKUP_DIR}"
        log_info "Compressed archive: ${BACKUP_ROOT}/${BACKUP_NAME}.tar.gz"
    else
        log_warn "⚠️ Backup completed with ${failed} component failures"
        log_warn "Check ${BACKUP_LOG} for details"
        exit 1
    fi
}

# Handle interrupts
trap 'log_error "Backup interrupted!"; exit 130' INT TERM

# Run main function
main "$@"
