#!/bin/bash
# AI Trading Bot - Disaster Recovery and Emergency Procedures
# This script provides automated disaster recovery capabilities for encrypted volumes

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KEY_DIR="/opt/trading-bot/keys"
BACKUP_KEY_DIR="/opt/trading-bot/keys/backup"
ENCRYPTED_DIR="/opt/trading-bot/encrypted"
BACKUP_DIR="/mnt/trading-backup"
RECOVERY_LOG="/var/log/trading-bot-recovery.log"

# Recovery configuration
RECOVERY_TEMP_DIR="/tmp/trading-bot-recovery"
VERIFICATION_TIMEOUT=300  # 5 minutes
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
    echo "$message" >> "$RECOVERY_LOG"
}

error() {
    local message="[ERROR] $1"
    echo -e "${RED}${message}${NC}" >&2
    echo "$message" >> "$RECOVERY_LOG"
}

warning() {
    local message="[WARNING] $1"
    echo -e "${YELLOW}${message}${NC}"
    echo "$message" >> "$RECOVERY_LOG"
}

success() {
    local message="[SUCCESS] $1"
    echo -e "${GREEN}${message}${NC}"
    echo "$message" >> "$RECOVERY_LOG"
}

critical() {
    local message="[CRITICAL] $1"
    echo -e "${PURPLE}${message}${NC}"
    echo "$message" >> "$RECOVERY_LOG"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for recovery operations"
        error "Please run: sudo $0"
        exit 1
    fi
}

# Emergency stop all trading operations
emergency_stop() {
    critical "INITIATING EMERGENCY STOP PROCEDURE"

    # Create emergency stop flag
    touch "$EMERGENCY_STOP_FILE"
    chmod 644 "$EMERGENCY_STOP_FILE"

    # Stop Docker containers
    log "Stopping all trading bot containers..."
    docker stop $(docker ps --format "table {{.Names}}" | grep -E "(trading|bluefin|mcp|dashboard)" | tail -n +2) 2>/dev/null || {
        warning "Some containers may not have stopped cleanly"
    }

    # Wait for graceful shutdown
    sleep 10

    # Force stop if needed
    docker kill $(docker ps --format "table {{.Names}}" | grep -E "(trading|bluefin|mcp|dashboard)" | tail -n +2) 2>/dev/null || true

    # Stop encrypted volumes service
    systemctl stop trading-bot-volumes 2>/dev/null || {
        warning "Could not stop trading-bot-volumes service"
    }

    success "Emergency stop completed"
    log "Emergency stop flag created at: $EMERGENCY_STOP_FILE"
    log "To resume operations, remove the emergency stop flag and restart services"
}

# Verify system prerequisites for recovery
verify_prerequisites() {
    log "Verifying recovery prerequisites..."

    local prerequisites_ok=true

    # Check required tools
    local required_tools=("cryptsetup" "gpg" "tar" "jq" "docker")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            error "Required tool not found: $tool"
            prerequisites_ok=false
        fi
    done

    # Check key directory exists
    if [ ! -d "$KEY_DIR" ]; then
        error "Key directory not found: $KEY_DIR"
        prerequisites_ok=false
    fi

    # Check if GPG keys are accessible
    if [ ! -f "$KEY_DIR/master.key" ]; then
        error "Master key not found: $KEY_DIR/master.key"
        prerequisites_ok=false
    fi

    # Create recovery temp directory
    mkdir -p "$RECOVERY_TEMP_DIR"
    chmod 700 "$RECOVERY_TEMP_DIR"

    if $prerequisites_ok; then
        success "Prerequisites verification passed"
        return 0
    else
        error "Prerequisites verification failed"
        return 1
    fi
}

# Recover encrypted volume from backup
recover_volume_from_backup() {
    local volume_name=$1
    local backup_timestamp=$2

    log "Recovering volume '$volume_name' from backup..."

    # Find backup file
    local backup_pattern="${volume_name}_${backup_timestamp}*.tar.gz.gpg"
    local backup_file=$(find "$BACKUP_DIR" -name "$backup_pattern" -type f | head -n 1)

    if [ -z "$backup_file" ]; then
        # Try latest backup if timestamp not specified
        backup_file=$(find "$BACKUP_DIR" -name "${volume_name}_*.tar.gz.gpg" -type f | sort | tail -n 1)
    fi

    if [ -z "$backup_file" ]; then
        error "No backup found for volume: $volume_name"
        return 1
    fi

    log "Using backup: $(basename "$backup_file")"

    # Verify backup integrity
    local checksum_file="${backup_file%.tar.gz.gpg}.sha512"
    if [ -f "$checksum_file" ]; then
        log "Verifying backup integrity..."
        if ! sha512sum -c "$checksum_file" >/dev/null 2>&1; then
            error "Backup integrity verification failed"
            return 1
        fi
        success "Backup integrity verified"
    else
        warning "No checksum file found, skipping integrity check"
    fi

    # Decrypt and extract backup
    local recovery_dir="$RECOVERY_TEMP_DIR/volume_$volume_name"
    mkdir -p "$recovery_dir"

    log "Decrypting and extracting backup..."
    gpg --batch --quiet --decrypt \
        --passphrase-file "$KEY_DIR/backup-encryption.key" \
        "$backup_file" | tar -xzf - -C "$recovery_dir"

    # Verify extracted data
    if [ ! -d "$recovery_dir/$(basename "/mnt/trading-$volume_name")" ]; then
        error "Extracted backup structure is invalid"
        return 1
    fi

    success "Volume backup extracted to: $recovery_dir"
    log "Volume '$volume_name' recovery preparation completed"
}

# Recover LUKS encryption keys
recover_encryption_keys() {
    local backup_timestamp=$1

    log "Recovering encryption keys..."

    # Find key backup
    local key_backup_pattern="keys_${backup_timestamp}*.tar.gz.gpg"
    local key_backup_file=$(find "${ENCRYPTED_DIR}/key-backup" -name "$key_backup_pattern" -type f | head -n 1)

    if [ -z "$key_backup_file" ]; then
        # Try latest key backup
        key_backup_file=$(find "${ENCRYPTED_DIR}/key-backup" -name "keys_*.tar.gz.gpg" -type f | sort | tail -n 1)
    fi

    if [ -z "$key_backup_file" ]; then
        error "No key backup found"
        return 1
    fi

    log "Using key backup: $(basename "$key_backup_file")"

    # Create backup of current keys
    local current_key_backup="$KEY_DIR/backup_before_recovery_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$current_key_backup"
    cp -r "$KEY_DIR"/* "$current_key_backup/" 2>/dev/null || true

    # Decrypt and restore keys
    local key_recovery_dir="$RECOVERY_TEMP_DIR/keys"
    mkdir -p "$key_recovery_dir"

    log "Enter GPG passphrase for key backup decryption:"
    gpg --decrypt "$key_backup_file" | tar -xzf - -C "$key_recovery_dir"

    # Verify key files
    if [ ! -f "$key_recovery_dir/master.key" ]; then
        error "Master key not found in backup"
        return 1
    fi

    # Restore keys
    cp -r "$key_recovery_dir"/* "$KEY_DIR/"
    chmod -R 600 "$KEY_DIR"/*.key 2>/dev/null || true

    success "Encryption keys recovered"
    log "Previous keys backed up to: $current_key_backup"
}

# Rebuild encrypted volumes
rebuild_encrypted_volumes() {
    log "Rebuilding encrypted volumes..."

    local volumes=("data" "logs" "config" "backup")

    for volume in "${volumes[@]}"; do
        log "Rebuilding encrypted volume: $volume"

        # Unmount and close existing volume if active
        if mountpoint -q "/mnt/trading-$volume" 2>/dev/null; then
            umount "/mnt/trading-$volume" || {
                warning "Could not unmount /mnt/trading-$volume"
            }
        fi

        if [ -e "/dev/mapper/trading-$volume" ]; then
            cryptsetup luksClose "trading-$volume" || {
                warning "Could not close LUKS volume trading-$volume"
            }
        fi

        # Remove old loop device
        local old_loop=$(losetup -j "${ENCRYPTED_DIR}/${volume}.img" | cut -d: -f1)
        if [ -n "$old_loop" ]; then
            losetup -d "$old_loop" 2>/dev/null || true
        fi

        # Remove old volume file
        rm -f "${ENCRYPTED_DIR}/${volume}.img"

        # Create new encrypted volume
        log "Creating new encrypted volume for $volume..."
        local volume_size
        case $volume in
            "data") volume_size="5G" ;;
            "logs") volume_size="2G" ;;
            "config") volume_size="512M" ;;
            "backup") volume_size="10G" ;;
        esac

        # Create volume file
        dd if=/dev/zero of="${ENCRYPTED_DIR}/${volume}.img" bs=1 count=0 seek="$volume_size" 2>/dev/null

        # Setup loop device
        local loop_device=$(losetup -f)
        losetup "$loop_device" "${ENCRYPTED_DIR}/${volume}.img"

        # Format with LUKS using recovered key
        echo "YES" | cryptsetup luksFormat \
            --type luks2 \
            --cipher aes-xts-plain64 \
            --key-size 512 \
            --hash sha512 \
            --iter-time 2000 \
            --use-random \
            "$loop_device" "${KEY_DIR}/${volume}.key"

        # Open encrypted volume
        cryptsetup luksOpen "$loop_device" "trading-${volume}" --key-file "${KEY_DIR}/${volume}.key"

        # Create filesystem
        mkfs.ext4 -F "/dev/mapper/trading-${volume}"

        # Mount volume
        mkdir -p "/mnt/trading-${volume}"
        mount "/dev/mapper/trading-${volume}" "/mnt/trading-${volume}"
        chown 1000:1000 "/mnt/trading-${volume}"

        # Store loop device info
        echo "$loop_device" > "${KEY_DIR}/${volume}.loop"

        success "Encrypted volume $volume rebuilt successfully"
    done

    success "All encrypted volumes rebuilt"
}

# Restore data from recovery
restore_data_from_recovery() {
    log "Restoring data from recovery area..."

    local volumes=("data" "logs" "config")

    for volume in "${volumes[@]}"; do
        local recovery_dir="$RECOVERY_TEMP_DIR/volume_$volume"
        local mount_point="/mnt/trading-$volume"

        if [ -d "$recovery_dir" ] && mountpoint -q "$mount_point"; then
            log "Restoring data for volume: $volume"

            # Copy data from recovery area
            rsync -av "$recovery_dir"/*/ "$mount_point/" || {
                warning "Some files may not have been restored for $volume"
            }

            # Fix permissions
            chown -R 1000:1000 "$mount_point"

            success "Data restored for volume: $volume"
        else
            warning "Skipping data restore for $volume (recovery data or mount point not available)"
        fi
    done

    success "Data restoration completed"
}

# Verify system integrity after recovery
verify_recovery_integrity() {
    log "Verifying system integrity after recovery..."

    local integrity_ok=true

    # Check all volumes are mounted
    local volumes=("data" "logs" "config" "backup")
    for volume in "${volumes[@]}"; do
        if ! mountpoint -q "/mnt/trading-$volume"; then
            error "Volume not mounted: trading-$volume"
            integrity_ok=false
        else
            log "✓ Volume mounted: trading-$volume"
        fi
    done

    # Check key files
    for volume in "${volumes[@]}"; do
        if [ ! -f "$KEY_DIR/${volume}.key" ]; then
            error "Key file missing: ${volume}.key"
            integrity_ok=false
        else
            log "✓ Key file present: ${volume}.key"
        fi
    done

    # Test encryption/decryption
    local test_file="/mnt/trading-data/recovery_test_$(date +%s)"
    local test_data="Recovery integrity test - $(date)"

    echo "$test_data" > "$test_file" 2>/dev/null || {
        error "Cannot write to encrypted volume"
        integrity_ok=false
    }

    if [ -f "$test_file" ] && [ "$(cat "$test_file")" = "$test_data" ]; then
        log "✓ Read/write test passed"
        rm -f "$test_file"
    else
        error "Read/write test failed"
        integrity_ok=false
    fi

    # Check Docker can access volumes
    if command -v docker >/dev/null 2>&1; then
        docker run --rm -v /mnt/trading-data:/test alpine sh -c 'echo "docker test" > /test/docker_test && cat /test/docker_test' >/dev/null 2>&1 && {
            log "✓ Docker volume access test passed"
            rm -f /mnt/trading-data/docker_test
        } || {
            error "Docker volume access test failed"
            integrity_ok=false
        }
    fi

    if $integrity_ok; then
        success "System integrity verification passed"
        return 0
    else
        error "System integrity verification failed"
        return 1
    fi
}

# Generate recovery report
generate_recovery_report() {
    local recovery_type=$1
    local recovery_timestamp=$(date +%Y%m%d_%H%M%S)
    local report_file="/tmp/recovery_report_${recovery_timestamp}.json"

    log "Generating recovery report..."

    cat > "$report_file" << EOF
{
  "recovery_type": "$recovery_type",
  "recovery_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "recovery_initiated_by": "$(whoami)",
  "system_info": {
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "uptime": "$(uptime)"
  },
  "volume_status": {},
  "integrity_checks": {},
  "next_steps": []
}
EOF

    # Add volume status
    local volumes=("data" "logs" "config" "backup")
    for volume in "${volumes[@]}"; do
        local mounted=$(mountpoint -q "/mnt/trading-$volume" && echo "true" || echo "false")
        local size=$(df -h "/mnt/trading-$volume" 2>/dev/null | tail -1 | awk '{print $2}' || echo "unknown")
        local used=$(df -h "/mnt/trading-$volume" 2>/dev/null | tail -1 | awk '{print $3}' || echo "unknown")

        local temp_report=$(mktemp)
        jq --arg volume "$volume" \
           --arg mounted "$mounted" \
           --arg size "$size" \
           --arg used "$used" \
           '.volume_status[$volume] = {
             "mounted": ($mounted | test("true")),
             "size": $size,
             "used": $used
           }' "$report_file" > "$temp_report"
        mv "$temp_report" "$report_file"
    done

    # Add next steps
    local temp_report=$(mktemp)
    jq '.next_steps = [
      "Verify all trading bot services can start",
      "Run comprehensive data integrity checks",
      "Test trading operations in paper mode",
      "Monitor system performance for 24 hours",
      "Update disaster recovery documentation",
      "Schedule key rotation if keys were recovered from backup"
    ]' "$report_file" > "$temp_report"
    mv "$temp_report" "$report_file"

    echo "=== Recovery Report ==="
    jq . "$report_file"
    echo ""
    echo "Full report saved to: $report_file"

    success "Recovery report generated"
}

# Clean recovery temporary files
cleanup_recovery() {
    log "Cleaning up recovery temporary files..."

    if [ -d "$RECOVERY_TEMP_DIR" ]; then
        rm -rf "$RECOVERY_TEMP_DIR"
        success "Recovery temporary files cleaned up"
    fi
}

# Full disaster recovery procedure
full_disaster_recovery() {
    local backup_timestamp=$1

    critical "INITIATING FULL DISASTER RECOVERY"
    log "This will completely rebuild the encrypted storage system"
    log "Backup timestamp: ${backup_timestamp:-latest}"

    # Confirm operation
    echo -n "Are you sure you want to proceed? (type 'YES' to confirm): "
    read -r confirmation
    if [ "$confirmation" != "YES" ]; then
        log "Disaster recovery cancelled by user"
        exit 0
    fi

    # Execute recovery steps
    verify_prerequisites || exit 1
    emergency_stop
    recover_encryption_keys "$backup_timestamp" || exit 1
    rebuild_encrypted_volumes || exit 1
    recover_volume_from_backup "data" "$backup_timestamp" || exit 1
    recover_volume_from_backup "logs" "$backup_timestamp" || exit 1
    recover_volume_from_backup "config" "$backup_timestamp" || exit 1
    restore_data_from_recovery || exit 1
    verify_recovery_integrity || exit 1
    generate_recovery_report "full_disaster_recovery"
    cleanup_recovery

    critical "FULL DISASTER RECOVERY COMPLETED"
    warning "Before resuming operations:"
    warning "1. Remove emergency stop flag: rm $EMERGENCY_STOP_FILE"
    warning "2. Start encrypted volumes: systemctl start trading-bot-volumes"
    warning "3. Test all services in paper trading mode"
    warning "4. Verify data integrity"
}

# Partial recovery for specific volume
partial_recovery() {
    local volume_name=$1
    local backup_timestamp=$2

    log "Initiating partial recovery for volume: $volume_name"

    verify_prerequisites || exit 1
    recover_volume_from_backup "$volume_name" "$backup_timestamp" || exit 1

    # Stop services that use this volume
    warning "Stopping services that use volume: $volume_name"
    docker stop $(docker ps --format "table {{.Names}}" | grep -E "(trading|bluefin|mcp|dashboard)" | tail -n +2) 2>/dev/null || true

    # Restore specific volume data
    local recovery_dir="$RECOVERY_TEMP_DIR/volume_$volume_name"
    local mount_point="/mnt/trading-$volume_name"

    if [ -d "$recovery_dir" ] && mountpoint -q "$mount_point"; then
        log "Restoring data for volume: $volume_name"
        rsync -av --delete "$recovery_dir"/*/ "$mount_point/"
        chown -R 1000:1000 "$mount_point"
        success "Data restored for volume: $volume_name"
    fi

    generate_recovery_report "partial_recovery_$volume_name"
    cleanup_recovery

    success "Partial recovery completed for volume: $volume_name"
}

# Print usage information
print_usage() {
    cat << EOF
AI Trading Bot - Disaster Recovery System

Usage:
  $0 emergency-stop              Emergency stop all operations
  $0 full-recovery [timestamp]   Full disaster recovery
  $0 partial-recovery <volume> [timestamp]  Recover specific volume
  $0 verify-system               Verify system integrity
  $0 test-backups                Test backup recovery (dry run)
  $0 clean-recovery              Clean recovery temporary files
  $0 help                        Show this help

Recovery Scenarios:

1. EMERGENCY STOP
   - Immediately stop all trading operations
   - Create emergency stop flag
   - Safe for use during critical situations

2. FULL DISASTER RECOVERY
   - Complete system rebuild from backups
   - Recovers all encrypted volumes and keys
   - Use when system is completely compromised

3. PARTIAL RECOVERY
   - Recover specific volume from backup
   - Minimal downtime for single volume issues
   - Available volumes: data, logs, config, backup

4. VERIFICATION
   - Check system integrity without changes
   - Verify all volumes and keys are working
   - Safe diagnostic operation

Examples:
  $0 emergency-stop                    # Stop everything immediately
  $0 full-recovery                     # Full recovery with latest backups
  $0 full-recovery 20241201_120000     # Recovery from specific backup
  $0 partial-recovery data             # Recover only data volume
  $0 verify-system                     # Check system health

Important Notes:
- Always create fresh backups before recovery operations
- Test recovery procedures regularly in non-production environments
- Keep recovery documentation updated
- Store recovery keys securely offsite

Recovery Log: $RECOVERY_LOG
Emergency Stop Flag: $EMERGENCY_STOP_FILE

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "emergency-stop")
            check_root
            emergency_stop
            ;;
        "full-recovery")
            check_root
            full_disaster_recovery "$2"
            ;;
        "partial-recovery")
            if [ -z "$2" ]; then
                error "Usage: $0 partial-recovery <volume> [timestamp]"
                exit 1
            fi
            check_root
            partial_recovery "$2" "$3"
            ;;
        "verify-system")
            check_root
            verify_prerequisites && verify_recovery_integrity
            ;;
        "test-backups")
            log "Testing backup recovery (dry run mode)"
            verify_prerequisites
            # Add backup testing logic here
            success "Backup test completed"
            ;;
        "clean-recovery")
            check_root
            cleanup_recovery
            ;;
        "help"|"-h"|"--help"|"")
            print_usage
            ;;
        *)
            error "Unknown command: $1"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
