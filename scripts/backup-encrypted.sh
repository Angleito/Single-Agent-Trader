#!/bin/bash
# AI Trading Bot - Encrypted Backup System
# This script creates encrypted backups of trading data with integrity verification

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="/mnt/trading-backup"
DATA_DIR="/mnt/trading-data"
LOGS_DIR="/mnt/trading-logs"
CONFIG_DIR="/mnt/trading-config"
KEY_DIR="/opt/trading-bot/keys"

# Backup configuration
BACKUP_PREFIX="trading-bot"
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
COMPRESSION_LEVEL=${BACKUP_COMPRESSION_LEVEL:-6}
ENCRYPTION_CIPHER="AES256"
HASH_ALGORITHM="SHA512"

# Remote backup configuration
ENABLE_OFFSITE_BACKUP=${ENABLE_OFFSITE_BACKUP:-false}
REMOTE_HOST=${BACKUP_REMOTE_HOST:-}
REMOTE_PATH=${BACKUP_REMOTE_PATH:-}
REMOTE_USER=${BACKUP_REMOTE_USER:-}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Generate backup timestamp
get_timestamp() {
    date +'%Y%m%d_%H%M%S'
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if running inside container or on host
    if [ -f /.dockerenv ]; then
        log "Running inside Docker container"
        CONTAINER_MODE=true
    else
        log "Running on host system"
        CONTAINER_MODE=false
    fi

    # Check required tools
    local required_tools=("gpg" "tar" "gzip" "sha512sum")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            error "Required tool not found: $tool"
            exit 1
        fi
    done

    # Check directories exist
    local required_dirs=("$DATA_DIR" "$LOGS_DIR" "$CONFIG_DIR" "$BACKUP_DIR")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            error "Required directory not found: $dir"
            exit 1
        fi
    done

    # Check backup directory is writable
    if [ ! -w "$BACKUP_DIR" ]; then
        error "Backup directory is not writable: $BACKUP_DIR"
        exit 1
    fi

    success "Prerequisites check passed"
}

# Create backup encryption key if not exists
setup_backup_encryption() {
    local backup_key_file="${KEY_DIR}/backup-encryption.key"

    if [ ! -f "$backup_key_file" ]; then
        log "Creating backup encryption key..."

        # Generate random passphrase
        openssl rand -base64 32 > "$backup_key_file"
        chmod 600 "$backup_key_file"

        success "Backup encryption key created"
    else
        log "Using existing backup encryption key"
    fi
}

# Create encrypted backup of directory
backup_directory() {
    local source_dir=$1
    local backup_name=$2
    local timestamp=$3

    log "Creating backup of $source_dir as $backup_name..."

    local temp_dir=$(mktemp -d)
    local archive_name="${backup_name}_${timestamp}"
    local archive_path="${temp_dir}/${archive_name}.tar.gz"
    local encrypted_path="${BACKUP_DIR}/${archive_name}.tar.gz.gpg"
    local checksum_path="${BACKUP_DIR}/${archive_name}.sha512"
    local metadata_path="${BACKUP_DIR}/${archive_name}.metadata"

    # Create compressed archive
    log "Creating compressed archive..."
    tar -czf "$archive_path" -C "$(dirname "$source_dir")" "$(basename "$source_dir")" 2>/dev/null || {
        warning "Some files may have been skipped during archive creation"
    }

    # Calculate checksum before encryption
    local original_checksum=$(sha512sum "$archive_path" | cut -d' ' -f1)

    # Encrypt the archive
    log "Encrypting archive..."
    gpg --batch --yes --quiet \
        --symmetric \
        --cipher-algo "$ENCRYPTION_CIPHER" \
        --compress-algo 2 \
        --s2k-digest-algo "$HASH_ALGORITHM" \
        --passphrase-file "${KEY_DIR}/backup-encryption.key" \
        --output "$encrypted_path" \
        "$archive_path"

    # Create metadata file
    cat > "$metadata_path" << EOF
{
  "backup_name": "$backup_name",
  "source_directory": "$source_dir",
  "timestamp": "$timestamp",
  "created_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "original_size": $(stat -f%z "$archive_path" 2>/dev/null || stat -c%s "$archive_path"),
  "encrypted_size": $(stat -f%z "$encrypted_path" 2>/dev/null || stat -c%s "$encrypted_path"),
  "original_checksum": "$original_checksum",
  "encryption_cipher": "$ENCRYPTION_CIPHER",
  "hash_algorithm": "$HASH_ALGORITHM",
  "compression_level": $COMPRESSION_LEVEL
}
EOF

    # Create checksum of encrypted file
    sha512sum "$encrypted_path" > "$checksum_path"

    # Cleanup temp files
    rm -rf "$temp_dir"

    success "Backup created: $(basename "$encrypted_path")"
    log "Original size: $(stat -f%z "$archive_path" 2>/dev/null || stat -c%s "$archive_path" 2>/dev/null || echo "unknown") bytes"
    log "Encrypted size: $(stat -f%z "$encrypted_path" 2>/dev/null || stat -c%s "$encrypted_path" 2>/dev/null || echo "unknown") bytes"
}

# Verify backup integrity
verify_backup() {
    local encrypted_file=$1
    local checksum_file="${encrypted_file%.*}.sha512"
    local metadata_file="${encrypted_file%.*}.metadata"

    log "Verifying backup: $(basename "$encrypted_file")"

    # Check if files exist
    if [ ! -f "$encrypted_file" ] || [ ! -f "$checksum_file" ]; then
        error "Backup files missing for verification"
        return 1
    fi

    # Verify checksum
    if ! sha512sum -c "$checksum_file" >/dev/null 2>&1; then
        error "Checksum verification failed for $(basename "$encrypted_file")"
        return 1
    fi

    # Test decryption (without extracting)
    if ! gpg --batch --quiet --decrypt \
        --passphrase-file "${KEY_DIR}/backup-encryption.key" \
        "$encrypted_file" >/dev/null 2>&1; then
        error "Decryption test failed for $(basename "$encrypted_file")"
        return 1
    fi

    success "Backup verification passed: $(basename "$encrypted_file")"
    return 0
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days..."

    local deleted_count=0

    # Find and delete old backup files
    find "$BACKUP_DIR" -name "${BACKUP_PREFIX}_*.tar.gz.gpg" -mtime +$RETENTION_DAYS -type f | while read -r file; do
        local base_name="${file%.tar.gz.gpg}"

        # Remove associated files
        rm -f "$file"
        rm -f "${base_name}.sha512"
        rm -f "${base_name}.metadata"

        log "Deleted old backup: $(basename "$file")"
        ((deleted_count++))
    done

    if [ $deleted_count -gt 0 ]; then
        success "Cleaned up $deleted_count old backup sets"
    else
        log "No old backups to clean up"
    fi
}

# Upload to remote storage
upload_to_remote() {
    local backup_file=$1

    if [ "$ENABLE_OFFSITE_BACKUP" != "true" ]; then
        log "Remote backup disabled, skipping upload"
        return 0
    fi

    if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PATH" ] || [ -z "$REMOTE_USER" ]; then
        warning "Remote backup configuration incomplete, skipping upload"
        return 0
    fi

    log "Uploading backup to remote storage..."

    local base_name="${backup_file%.tar.gz.gpg}"
    local files_to_upload=(
        "$backup_file"
        "${base_name}.sha512"
        "${base_name}.metadata"
    )

    # Create remote directory if needed
    ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH" || {
        error "Failed to create remote directory"
        return 1
    }

    # Upload files
    for file in "${files_to_upload[@]}"; do
        if [ -f "$file" ]; then
            log "Uploading $(basename "$file")..."
            scp "$file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" || {
                error "Failed to upload $(basename "$file")"
                return 1
            }
        fi
    done

    success "Remote upload completed"
}

# Create database backup
backup_database() {
    local timestamp=$1

    # Check if database files exist
    if [ -d "$DATA_DIR/mcp_memory" ]; then
        log "Creating database backup..."
        backup_directory "$DATA_DIR/mcp_memory" "mcp_memory" "$timestamp"
    fi

    if [ -f "$DATA_DIR/paper_trading/account.json" ]; then
        log "Creating paper trading backup..."
        backup_directory "$DATA_DIR/paper_trading" "paper_trading" "$timestamp"
    fi
}

# Create configuration backup
backup_configuration() {
    local timestamp=$1

    log "Creating configuration backup..."
    backup_directory "$CONFIG_DIR" "config" "$timestamp"
}

# Create logs backup
backup_logs() {
    local timestamp=$1

    log "Creating logs backup..."

    # Only backup recent logs to save space
    local temp_logs_dir=$(mktemp -d)

    # Copy logs from last 7 days
    find "$LOGS_DIR" -name "*.log" -mtime -7 -exec cp {} "$temp_logs_dir/" \; 2>/dev/null || true
    find "$LOGS_DIR" -name "*.jsonl" -mtime -7 -exec cp {} "$temp_logs_dir/" \; 2>/dev/null || true

    if [ "$(ls -A "$temp_logs_dir")" ]; then
        backup_directory "$temp_logs_dir" "logs" "$timestamp"
    else
        log "No recent logs found to backup"
    fi

    # Cleanup
    rm -rf "$temp_logs_dir"
}

# Generate backup report
generate_backup_report() {
    local timestamp=$1
    local report_file="${BACKUP_DIR}/backup_report_${timestamp}.json"

    log "Generating backup report..."

    local backup_files=($(find "$BACKUP_DIR" -name "*${timestamp}*.tar.gz.gpg" -type f))
    local total_size=0

    echo "{" > "$report_file"
    echo "  \"backup_timestamp\": \"$timestamp\"," >> "$report_file"
    echo "  \"backup_date\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"," >> "$report_file"
    echo "  \"backup_files\": [" >> "$report_file"

    local file_count=0
    for file in "${backup_files[@]}"; do
        if [ $file_count -gt 0 ]; then
            echo "    ," >> "$report_file"
        fi

        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        total_size=$((total_size + size))

        echo "    {" >> "$report_file"
        echo "      \"filename\": \"$(basename "$file")\"," >> "$report_file"
        echo "      \"size\": $size," >> "$report_file"
        echo "      \"path\": \"$file\"" >> "$report_file"
        echo -n "    }" >> "$report_file"

        ((file_count++))
    done

    echo "" >> "$report_file"
    echo "  ]," >> "$report_file"
    echo "  \"total_files\": $file_count," >> "$report_file"
    echo "  \"total_size\": $total_size," >> "$report_file"
    echo "  \"retention_days\": $RETENTION_DAYS" >> "$report_file"
    echo "}" >> "$report_file"

    success "Backup report generated: $(basename "$report_file")"
}

# Main backup function
main() {
    log "Starting encrypted backup process..."

    local timestamp=$(get_timestamp)

    check_prerequisites
    setup_backup_encryption

    # Create backups
    backup_configuration "$timestamp"
    backup_database "$timestamp"
    backup_logs "$timestamp"

    # Verify all backups created for this timestamp
    local backup_files=($(find "$BACKUP_DIR" -name "*${timestamp}*.tar.gz.gpg" -type f))
    local verification_failed=false

    for file in "${backup_files[@]}"; do
        if ! verify_backup "$file"; then
            verification_failed=true
        fi
    done

    if $verification_failed; then
        error "Some backup verifications failed"
        exit 1
    fi

    # Upload to remote storage
    for file in "${backup_files[@]}"; do
        upload_to_remote "$file"
    done

    # Generate report
    generate_backup_report "$timestamp"

    # Cleanup old backups
    cleanup_old_backups

    success "Encrypted backup process completed successfully"
    log "Backup timestamp: $timestamp"
    log "Files created: ${#backup_files[@]}"
}

# Handle script arguments
case "${1:-}" in
    "verify")
        if [ -z "$2" ]; then
            error "Usage: $0 verify <backup_file>"
            exit 1
        fi
        verify_backup "$2"
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    "help"|"-h"|"--help")
        cat << EOF
AI Trading Bot - Encrypted Backup System

Usage:
  $0                    Create encrypted backup
  $0 verify <file>      Verify backup integrity
  $0 cleanup            Clean up old backups
  $0 help               Show this help

Environment Variables:
  RETENTION_DAYS              Days to keep backups (default: 30)
  BACKUP_COMPRESSION_LEVEL    Compression level 1-9 (default: 6)
  ENABLE_OFFSITE_BACKUP       Upload to remote storage (default: false)
  BACKUP_REMOTE_HOST          Remote hostname for offsite backup
  BACKUP_REMOTE_PATH          Remote path for offsite backup
  BACKUP_REMOTE_USER          Remote username for offsite backup

Backup Location: $BACKUP_DIR
Encryption Key: ${KEY_DIR}/backup-encryption.key

EOF
        ;;
    *)
        main "$@"
        ;;
esac
