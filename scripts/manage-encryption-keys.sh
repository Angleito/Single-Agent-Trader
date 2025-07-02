#!/bin/bash
# AI Trading Bot - Encryption Key Management and Rotation
# This script manages LUKS encryption keys with secure rotation procedures

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KEY_DIR="/opt/trading-bot/keys"
BACKUP_KEY_DIR="/opt/trading-bot/keys/backup"
ENCRYPTED_DIR="/opt/trading-bot/encrypted"

# Key rotation settings
KEY_ROTATION_DAYS=${KEY_ROTATION_DAYS:-90}
KEY_BACKUP_RETENTION=${KEY_BACKUP_RETENTION:-365}
SECURITY_AUDIT_LOG="/var/log/trading-bot-key-management.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${BLUE}${message}${NC}"
    echo "$message" >> "$SECURITY_AUDIT_LOG"
}

error() {
    local message="[ERROR] $1"
    echo -e "${RED}${message}${NC}" >&2
    echo "$message" >> "$SECURITY_AUDIT_LOG"
}

warning() {
    local message="[WARNING] $1"
    echo -e "${YELLOW}${message}${NC}"
    echo "$message" >> "$SECURITY_AUDIT_LOG"
}

success() {
    local message="[SUCCESS] $1"
    echo -e "${GREEN}${message}${NC}"
    echo "$message" >> "$SECURITY_AUDIT_LOG"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for key management operations"
        error "Please run: sudo $0"
        exit 1
    fi
}

# Initialize key management system
init_key_management() {
    log "Initializing key management system..."
    
    # Create directories
    mkdir -p "$KEY_DIR"
    mkdir -p "$BACKUP_KEY_DIR"
    mkdir -p "$(dirname "$SECURITY_AUDIT_LOG")"
    
    # Set secure permissions
    chmod 700 "$KEY_DIR"
    chmod 700 "$BACKUP_KEY_DIR"
    chmod 600 "$SECURITY_AUDIT_LOG" 2>/dev/null || touch "$SECURITY_AUDIT_LOG"
    
    # Create key metadata database
    if [ ! -f "$KEY_DIR/key_metadata.json" ]; then
        cat > "$KEY_DIR/key_metadata.json" << 'EOF'
{
  "version": "1.0",
  "created": "",
  "last_rotation": "",
  "keys": {},
  "rotation_policy": {
    "max_age_days": 90,
    "backup_retention_days": 365,
    "require_dual_approval": false
  }
}
EOF
        chmod 600 "$KEY_DIR/key_metadata.json"
    fi
    
    success "Key management system initialized"
}

# Generate secure random key
generate_secure_key() {
    local key_length=${1:-64}
    openssl rand -hex "$key_length"
}

# Create key with metadata
create_key() {
    local key_name=$1
    local key_purpose=$2
    local key_length=${3:-64}
    
    log "Creating key: $key_name"
    
    local key_file="$KEY_DIR/${key_name}.key"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Generate key
    generate_secure_key "$key_length" > "$key_file"
    chmod 600 "$key_file"
    
    # Update metadata
    local temp_metadata=$(mktemp)
    jq --arg name "$key_name" \
       --arg purpose "$key_purpose" \
       --arg created "$timestamp" \
       --arg file "$key_file" \
       '.keys[$name] = {
         "purpose": $purpose,
         "created": $created,
         "last_rotation": $created,
         "file": $file,
         "active": true,
         "rotation_count": 0
       }' "$KEY_DIR/key_metadata.json" > "$temp_metadata"
    
    mv "$temp_metadata" "$KEY_DIR/key_metadata.json"
    chmod 600 "$KEY_DIR/key_metadata.json"
    
    success "Key created: $key_name"
}

# Backup key securely
backup_key() {
    local key_name=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_KEY_DIR/${key_name}_${timestamp}.key.gpg"
    
    log "Creating secure backup of key: $key_name"
    
    # Encrypt key backup with master passphrase
    gpg --batch --yes --quiet \
        --symmetric \
        --cipher-algo AES256 \
        --compress-algo 2 \
        --s2k-digest-algo SHA512 \
        --passphrase-file "$KEY_DIR/master.key" \
        --output "$backup_file" \
        "$KEY_DIR/${key_name}.key"
    
    chmod 600 "$backup_file"
    
    # Create backup metadata
    cat > "${backup_file}.metadata" << EOF
{
  "key_name": "$key_name",
  "backup_timestamp": "$timestamp",
  "backup_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "original_file": "$KEY_DIR/${key_name}.key",
  "backup_file": "$backup_file",
  "encryption_method": "GPG-AES256"
}
EOF
    
    chmod 600 "${backup_file}.metadata"
    
    success "Key backup created: $(basename "$backup_file")"
}

# Rotate LUKS key
rotate_luks_key() {
    local volume_name=$1
    local device_path="/dev/mapper/trading-${volume_name}"
    local old_key_file="$KEY_DIR/${volume_name}.key"
    local new_key_file="$KEY_DIR/${volume_name}_new.key"
    
    log "Rotating LUKS key for volume: $volume_name"
    
    # Check if volume is active
    if [ ! -e "$device_path" ]; then
        error "LUKS volume not active: $volume_name"
        return 1
    fi
    
    # Backup current key
    backup_key "$volume_name"
    
    # Generate new key
    generate_secure_key 64 > "$new_key_file"
    chmod 600 "$new_key_file"
    
    # Get loop device
    local loop_device=$(losetup -j "${ENCRYPTED_DIR}/${volume_name}.img" | cut -d: -f1)
    if [ -z "$loop_device" ]; then
        error "Could not find loop device for $volume_name"
        rm -f "$new_key_file"
        return 1
    fi
    
    # Add new key to LUKS header
    log "Adding new key to LUKS header..."
    cryptsetup luksAddKey "$loop_device" "$new_key_file" --key-file "$old_key_file"
    
    # Verify new key works
    log "Verifying new key..."
    if ! cryptsetup luksOpen --test-passphrase "$loop_device" --key-file "$new_key_file"; then
        error "New key verification failed, removing new key"
        cryptsetup luksRemoveKey "$loop_device" "$new_key_file" --key-file "$old_key_file" 2>/dev/null || true
        rm -f "$new_key_file"
        return 1
    fi
    
    # Remove old key from LUKS header
    log "Removing old key from LUKS header..."
    cryptsetup luksRemoveKey "$loop_device" "$old_key_file" --key-file "$new_key_file"
    
    # Replace old key file with new one
    mv "$new_key_file" "$old_key_file"
    
    # Update metadata
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local temp_metadata=$(mktemp)
    jq --arg name "$volume_name" \
       --arg timestamp "$timestamp" \
       '.keys[$name].last_rotation = $timestamp |
        .keys[$name].rotation_count += 1' \
       "$KEY_DIR/key_metadata.json" > "$temp_metadata"
    
    mv "$temp_metadata" "$KEY_DIR/key_metadata.json"
    chmod 600 "$KEY_DIR/key_metadata.json"
    
    success "LUKS key rotation completed for volume: $volume_name"
}

# Rotate backup encryption key
rotate_backup_key() {
    log "Rotating backup encryption key..."
    
    local old_key_file="$KEY_DIR/backup-encryption.key"
    local new_key_file="$KEY_DIR/backup-encryption_new.key"
    
    # Backup current key
    backup_key "backup-encryption"
    
    # Generate new key
    generate_secure_key 32 > "$new_key_file"
    chmod 600 "$new_key_file"
    
    # Test new key with a sample encryption/decryption
    local test_data="test_encryption_$(date +%s)"
    echo "$test_data" | gpg --batch --quiet --symmetric --passphrase-file "$new_key_file" | \
    gpg --batch --quiet --decrypt --passphrase-file "$new_key_file" > /tmp/key_test
    
    if [ "$(cat /tmp/key_test)" = "$test_data" ]; then
        # Replace old key with new one
        mv "$new_key_file" "$old_key_file"
        rm -f /tmp/key_test
        
        # Update metadata
        local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        local temp_metadata=$(mktemp)
        jq --arg timestamp "$timestamp" \
           '.keys["backup-encryption"].last_rotation = $timestamp |
            .keys["backup-encryption"].rotation_count += 1' \
           "$KEY_DIR/key_metadata.json" > "$temp_metadata"
        
        mv "$temp_metadata" "$KEY_DIR/key_metadata.json"
        chmod 600 "$KEY_DIR/key_metadata.json"
        
        success "Backup encryption key rotation completed"
    else
        error "New backup key verification failed"
        rm -f "$new_key_file" /tmp/key_test
        return 1
    fi
}

# Check key age and recommend rotation
check_key_age() {
    log "Checking key ages for rotation recommendations..."
    
    local current_time=$(date +%s)
    local rotation_needed=false
    
    # Read metadata
    local keys=$(jq -r '.keys | keys[]' "$KEY_DIR/key_metadata.json")
    
    for key_name in $keys; do
        local last_rotation=$(jq -r ".keys[\"$key_name\"].last_rotation" "$KEY_DIR/key_metadata.json")
        local rotation_time=$(date -d "$last_rotation" +%s 2>/dev/null || echo "0")
        local age_days=$(( (current_time - rotation_time) / 86400 ))
        
        if [ $age_days -gt $KEY_ROTATION_DAYS ]; then
            warning "Key '$key_name' is $age_days days old (rotation recommended after $KEY_ROTATION_DAYS days)"
            rotation_needed=true
        else
            log "Key '$key_name' is $age_days days old (OK)"
        fi
    done
    
    if $rotation_needed; then
        warning "Some keys need rotation. Run: $0 rotate-all"
    else
        success "All keys are within rotation policy"
    fi
}

# Rotate all keys
rotate_all_keys() {
    log "Starting rotation of all keys..."
    
    # Rotate LUKS keys
    local volumes=("data" "logs" "config" "backup")
    for volume in "${volumes[@]}"; do
        if [ -f "$KEY_DIR/${volume}.key" ]; then
            rotate_luks_key "$volume"
        fi
    done
    
    # Rotate backup encryption key
    if [ -f "$KEY_DIR/backup-encryption.key" ]; then
        rotate_backup_key
    fi
    
    # Update global rotation timestamp
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local temp_metadata=$(mktemp)
    jq --arg timestamp "$timestamp" \
       '.last_rotation = $timestamp' \
       "$KEY_DIR/key_metadata.json" > "$temp_metadata"
    
    mv "$temp_metadata" "$KEY_DIR/key_metadata.json"
    chmod 600 "$KEY_DIR/key_metadata.json"
    
    success "All key rotation completed"
}

# Generate key status report
generate_key_report() {
    log "Generating key status report..."
    
    local report_file="/tmp/key_status_report_$(date +%Y%m%d_%H%M%S).json"
    local current_time=$(date +%s)
    
    # Create detailed report
    cat > "$report_file" << EOF
{
  "report_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "key_directory": "$KEY_DIR",
  "backup_directory": "$BACKUP_KEY_DIR",
  "rotation_policy_days": $KEY_ROTATION_DAYS,
  "key_status": []
}
EOF
    
    # Add each key status
    local keys=$(jq -r '.keys | keys[]' "$KEY_DIR/key_metadata.json")
    local temp_report=$(mktemp)
    cp "$report_file" "$temp_report"
    
    for key_name in $keys; do
        local key_info=$(jq ".keys[\"$key_name\"]" "$KEY_DIR/key_metadata.json")
        local last_rotation=$(echo "$key_info" | jq -r '.last_rotation')
        local rotation_time=$(date -d "$last_rotation" +%s 2>/dev/null || echo "0")
        local age_days=$(( (current_time - rotation_time) / 86400 ))
        local needs_rotation=$([ $age_days -gt $KEY_ROTATION_DAYS ] && echo "true" || echo "false")
        
        # Add key status to report
        jq --arg name "$key_name" \
           --arg age "$age_days" \
           --arg needs_rotation "$needs_rotation" \
           --argjson key_info "$key_info" \
           '.key_status += [{
             "name": $name,
             "age_days": ($age | tonumber),
             "needs_rotation": ($needs_rotation | test("true")),
             "info": $key_info
           }]' "$temp_report" > "$report_file"
        
        cp "$report_file" "$temp_report"
    done
    
    rm -f "$temp_report"
    
    # Display summary
    echo "=== Key Status Report ==="
    jq -r '.key_status[] | "\(.name): \(.age_days) days old, needs rotation: \(.needs_rotation)"' "$report_file"
    echo ""
    echo "Full report saved to: $report_file"
    
    success "Key status report generated"
}

# Clean old key backups
clean_old_backups() {
    log "Cleaning old key backups..."
    
    local deleted_count=0
    
    # Find and delete old backup files
    find "$BACKUP_KEY_DIR" -name "*.key.gpg" -mtime +$KEY_BACKUP_RETENTION -type f | while read -r file; do
        rm -f "$file"
        rm -f "${file}.metadata"
        log "Deleted old backup: $(basename "$file")"
        ((deleted_count++))
    done
    
    if [ $deleted_count -gt 0 ]; then
        success "Cleaned up $deleted_count old key backups"
    else
        log "No old key backups to clean up"
    fi
}

# Verify key integrity
verify_keys() {
    log "Verifying key integrity..."
    
    local all_ok=true
    local keys=$(jq -r '.keys | keys[]' "$KEY_DIR/key_metadata.json")
    
    for key_name in $keys; do
        local key_file="$KEY_DIR/${key_name}.key"
        
        if [ -f "$key_file" ]; then
            # Check file permissions
            local perms=$(stat -c %a "$key_file")
            if [ "$perms" != "600" ]; then
                error "Key file has incorrect permissions: $key_file ($perms)"
                all_ok=false
            fi
            
            # Check file size (should not be empty)
            local size=$(stat -c %s "$key_file")
            if [ "$size" -lt 10 ]; then
                error "Key file too small: $key_file ($size bytes)"
                all_ok=false
            fi
            
            # For LUKS keys, verify they work
            if [[ "$key_name" =~ ^(data|logs|config|backup)$ ]]; then
                local loop_device=$(losetup -j "${ENCRYPTED_DIR}/${key_name}.img" | cut -d: -f1)
                if [ -n "$loop_device" ]; then
                    if ! cryptsetup luksOpen --test-passphrase "$loop_device" --key-file "$key_file" 2>/dev/null; then
                        error "LUKS key verification failed: $key_name"
                        all_ok=false
                    else
                        log "✓ LUKS key verified: $key_name"
                    fi
                fi
            fi
            
            log "✓ Key file OK: $key_name"
        else
            error "Key file missing: $key_file"
            all_ok=false
        fi
    done
    
    if $all_ok; then
        success "All keys verified successfully"
    else
        error "Some key verification checks failed"
        return 1
    fi
}

# Create systemd timer for automatic key rotation
create_rotation_timer() {
    log "Creating systemd timer for automatic key rotation..."
    
    # Create service file
    cat > /etc/systemd/system/trading-bot-key-rotation.service << EOF
[Unit]
Description=AI Trading Bot Key Rotation
After=trading-bot-volumes.service

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/manage-encryption-keys.sh check-age
ExecStartPost=$SCRIPT_DIR/manage-encryption-keys.sh clean-backups
StandardOutput=journal
StandardError=journal
EOF

    # Create timer file
    cat > /etc/systemd/system/trading-bot-key-rotation.timer << EOF
[Unit]
Description=AI Trading Bot Key Rotation Timer
Requires=trading-bot-key-rotation.service

[Timer]
OnCalendar=weekly
Persistent=true
RandomizedDelaySec=3600

[Install]
WantedBy=timers.target
EOF

    # Enable timer
    systemctl daemon-reload
    systemctl enable trading-bot-key-rotation.timer
    systemctl start trading-bot-key-rotation.timer
    
    success "Automatic key rotation timer created and enabled"
}

# Print usage information
print_usage() {
    cat << EOF
AI Trading Bot - Encryption Key Management

Usage:
  $0 init                    Initialize key management system
  $0 create <name> <purpose> Create new key
  $0 rotate <volume>         Rotate LUKS key for specific volume
  $0 rotate-all              Rotate all keys
  $0 rotate-backup           Rotate backup encryption key
  $0 check-age               Check key ages and rotation needs
  $0 verify                  Verify key integrity
  $0 report                  Generate key status report
  $0 clean-backups           Clean old key backups
  $0 setup-timer             Setup automatic rotation timer
  $0 help                    Show this help

Examples:
  $0 init                    # First-time setup
  $0 check-age               # Check if rotation needed
  $0 rotate data             # Rotate data volume key
  $0 rotate-all              # Rotate all keys (maintenance)

Configuration:
  KEY_ROTATION_DAYS=$KEY_ROTATION_DAYS
  KEY_BACKUP_RETENTION=$KEY_BACKUP_RETENTION
  
Security Audit Log: $SECURITY_AUDIT_LOG

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "init")
            check_root
            init_key_management
            ;;
        "create")
            if [ -z "$2" ] || [ -z "$3" ]; then
                error "Usage: $0 create <name> <purpose>"
                exit 1
            fi
            check_root
            create_key "$2" "$3" "${4:-64}"
            ;;
        "rotate")
            if [ -z "$2" ]; then
                error "Usage: $0 rotate <volume_name>"
                exit 1
            fi
            check_root
            rotate_luks_key "$2"
            ;;
        "rotate-all")
            check_root
            rotate_all_keys
            ;;
        "rotate-backup")
            check_root
            rotate_backup_key
            ;;
        "check-age")
            check_key_age
            ;;
        "verify")
            verify_keys
            ;;
        "report")
            generate_key_report
            ;;
        "clean-backups")
            check_root
            clean_old_backups
            ;;
        "setup-timer")
            check_root
            create_rotation_timer
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