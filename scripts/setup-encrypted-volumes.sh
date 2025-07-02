#!/bin/bash
# AI Trading Bot - Encrypted Volume Setup Script
# This script sets up LUKS encrypted volumes for secure Docker storage

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENCRYPTED_DIR="/opt/trading-bot/encrypted"
KEY_DIR="/opt/trading-bot/keys"
VOLUME_SIZE="10G"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for LUKS operations"
        error "Please run: sudo $0"
        exit 1
    fi
}

# Install required packages
install_dependencies() {
    log "Installing required packages..."

    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y cryptsetup lvm2 parted e2fsprogs
    elif command -v yum >/dev/null 2>&1; then
        yum install -y cryptsetup lvm2 parted e2fsprogs
    elif command -v pacman >/dev/null 2>&1; then
        pacman -S --noconfirm cryptsetup lvm2 parted e2fsprogs
    else
        error "Unsupported package manager. Please install cryptsetup, lvm2, parted, and e2fsprogs manually."
        exit 1
    fi

    success "Dependencies installed successfully"
}

# Create secure directories
create_directories() {
    log "Creating secure directories..."

    mkdir -p "$ENCRYPTED_DIR"
    mkdir -p "$KEY_DIR"
    mkdir -p "${ENCRYPTED_DIR}/data"
    mkdir -p "${ENCRYPTED_DIR}/logs"
    mkdir -p "${ENCRYPTED_DIR}/config"
    mkdir -p "${ENCRYPTED_DIR}/backups"

    # Set secure permissions
    chmod 700 "$KEY_DIR"
    chmod 755 "$ENCRYPTED_DIR"

    success "Directories created with secure permissions"
}

# Generate encryption keys
generate_keys() {
    log "Generating encryption keys..."

    # Generate master key
    openssl rand -hex 64 > "${KEY_DIR}/master.key"
    chmod 600 "${KEY_DIR}/master.key"

    # Generate data volume key
    openssl rand -hex 64 > "${KEY_DIR}/data.key"
    chmod 600 "${KEY_DIR}/data.key"

    # Generate logs volume key
    openssl rand -hex 64 > "${KEY_DIR}/logs.key"
    chmod 600 "${KEY_DIR}/logs.key"

    # Generate config volume key
    openssl rand -hex 64 > "${KEY_DIR}/config.key"
    chmod 600 "${KEY_DIR}/config.key"

    # Generate backup volume key
    openssl rand -hex 64 > "${KEY_DIR}/backup.key"
    chmod 600 "${KEY_DIR}/backup.key"

    success "Encryption keys generated and secured"
}

# Create encrypted volume files
create_volume_files() {
    local volume_name=$1
    local size=$2

    log "Creating encrypted volume: $volume_name (${size})"

    # Create sparse file
    dd if=/dev/zero of="${ENCRYPTED_DIR}/${volume_name}.img" bs=1 count=0 seek="${size}" 2>/dev/null

    # Setup loop device
    local loop_device=$(losetup -f)
    losetup "$loop_device" "${ENCRYPTED_DIR}/${volume_name}.img"

    # Format with LUKS
    echo "YES" | cryptsetup luksFormat \
        --type luks2 \
        --cipher aes-xts-plain64 \
        --key-size 512 \
        --hash sha512 \
        --iter-time 2000 \
        --use-random \
        "$loop_device" "${KEY_DIR}/${volume_name}.key"

    # Open the encrypted volume
    cryptsetup luksOpen "$loop_device" "trading-${volume_name}" --key-file "${KEY_DIR}/${volume_name}.key"

    # Create filesystem
    mkfs.ext4 -F "/dev/mapper/trading-${volume_name}"

    # Create mount point and mount
    mkdir -p "/mnt/trading-${volume_name}"
    mount "/dev/mapper/trading-${volume_name}" "/mnt/trading-${volume_name}"

    # Set permissions for Docker user
    chown 1000:1000 "/mnt/trading-${volume_name}"
    chmod 755 "/mnt/trading-${volume_name}"

    success "Encrypted volume $volume_name created and mounted"

    # Store loop device for cleanup
    echo "$loop_device" > "${KEY_DIR}/${volume_name}.loop"
}

# Create all encrypted volumes
setup_encrypted_volumes() {
    log "Setting up encrypted volumes..."

    create_volume_files "data" "5G"
    create_volume_files "logs" "2G"
    create_volume_files "config" "512M"
    create_volume_files "backup" "10G"

    success "All encrypted volumes created successfully"
}

# Create systemd service for auto-mounting
create_systemd_service() {
    log "Creating systemd service for encrypted volumes..."

    cat > /etc/systemd/system/trading-bot-volumes.service << 'EOF'
[Unit]
Description=AI Trading Bot Encrypted Volumes
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/trading-bot/scripts/mount-encrypted-volumes.sh
ExecStop=/opt/trading-bot/scripts/umount-encrypted-volumes.sh
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable trading-bot-volumes.service

    success "Systemd service created and enabled"
}

# Create mount/unmount scripts
create_mount_scripts() {
    log "Creating mount/unmount scripts..."

    # Create mount script
    cat > /opt/trading-bot/scripts/mount-encrypted-volumes.sh << 'EOF'
#!/bin/bash
# Mount encrypted volumes for AI Trading Bot

set -e

KEY_DIR="/opt/trading-bot/keys"
ENCRYPTED_DIR="/opt/trading-bot/encrypted"

# Function to mount volume
mount_volume() {
    local volume_name=$1

    if [ -f "${KEY_DIR}/${volume_name}.loop" ]; then
        local loop_device=$(cat "${KEY_DIR}/${volume_name}.loop")

        # Setup loop device if not exists
        if ! losetup -l | grep -q "${ENCRYPTED_DIR}/${volume_name}.img"; then
            losetup "$loop_device" "${ENCRYPTED_DIR}/${volume_name}.img" 2>/dev/null || {
                loop_device=$(losetup -f)
                losetup "$loop_device" "${ENCRYPTED_DIR}/${volume_name}.img"
                echo "$loop_device" > "${KEY_DIR}/${volume_name}.loop"
            }
        fi

        # Open LUKS volume if not already open
        if [ ! -e "/dev/mapper/trading-${volume_name}" ]; then
            cryptsetup luksOpen "$loop_device" "trading-${volume_name}" --key-file "${KEY_DIR}/${volume_name}.key"
        fi

        # Mount if not already mounted
        if ! mountpoint -q "/mnt/trading-${volume_name}"; then
            mkdir -p "/mnt/trading-${volume_name}"
            mount "/dev/mapper/trading-${volume_name}" "/mnt/trading-${volume_name}"
            chown 1000:1000 "/mnt/trading-${volume_name}"
        fi

        echo "Volume $volume_name mounted successfully"
    else
        echo "Warning: Loop device file for $volume_name not found"
    fi
}

# Mount all volumes
mount_volume "data"
mount_volume "logs"
mount_volume "config"
mount_volume "backup"

echo "All encrypted volumes mounted"
EOF

    # Create unmount script
    cat > /opt/trading-bot/scripts/umount-encrypted-volumes.sh << 'EOF'
#!/bin/bash
# Unmount encrypted volumes for AI Trading Bot

set -e

KEY_DIR="/opt/trading-bot/keys"

# Function to unmount volume
umount_volume() {
    local volume_name=$1

    # Unmount filesystem
    if mountpoint -q "/mnt/trading-${volume_name}"; then
        umount "/mnt/trading-${volume_name}" || {
            echo "Warning: Failed to unmount /mnt/trading-${volume_name}, forcing..."
            umount -l "/mnt/trading-${volume_name}" 2>/dev/null || true
        }
    fi

    # Close LUKS volume
    if [ -e "/dev/mapper/trading-${volume_name}" ]; then
        cryptsetup luksClose "trading-${volume_name}" || {
            echo "Warning: Failed to close LUKS volume trading-${volume_name}"
        }
    fi

    # Detach loop device
    if [ -f "${KEY_DIR}/${volume_name}.loop" ]; then
        local loop_device=$(cat "${KEY_DIR}/${volume_name}.loop")
        if losetup -l | grep -q "$loop_device"; then
            losetup -d "$loop_device" 2>/dev/null || {
                echo "Warning: Failed to detach loop device $loop_device"
            }
        fi
    fi

    echo "Volume $volume_name unmounted successfully"
}

# Unmount all volumes
umount_volume "data"
umount_volume "logs"
umount_volume "config"
umount_volume "backup"

echo "All encrypted volumes unmounted"
EOF

    # Make scripts executable
    chmod +x /opt/trading-bot/scripts/mount-encrypted-volumes.sh
    chmod +x /opt/trading-bot/scripts/umount-encrypted-volumes.sh

    success "Mount/unmount scripts created"
}

# Create key backup
create_key_backup() {
    log "Creating encrypted key backup..."

    # Create backup directory
    mkdir -p "${ENCRYPTED_DIR}/key-backup"

    # Create encrypted backup of keys
    tar -czf - -C "$KEY_DIR" . | gpg --symmetric --cipher-algo AES256 --compress-algo 2 --s2k-digest-algo SHA512 --output "${ENCRYPTED_DIR}/key-backup/keys-$(date +%Y%m%d).tar.gz.gpg"

    # Set secure permissions
    chmod 600 "${ENCRYPTED_DIR}/key-backup"/*.gpg

    warning "IMPORTANT: Store the GPG passphrase securely!"
    warning "Key backup created at: ${ENCRYPTED_DIR}/key-backup/"

    success "Key backup created and encrypted"
}

# Verify setup
verify_setup() {
    log "Verifying encrypted volume setup..."

    local volumes=("data" "logs" "config" "backup")
    local all_ok=true

    for volume in "${volumes[@]}"; do
        if mountpoint -q "/mnt/trading-${volume}"; then
            echo "✓ Volume $volume is mounted"

            # Test write/read
            echo "test" > "/mnt/trading-${volume}/test.txt"
            if [ "$(cat "/mnt/trading-${volume}/test.txt")" = "test" ]; then
                echo "✓ Volume $volume read/write test passed"
                rm "/mnt/trading-${volume}/test.txt"
            else
                echo "✗ Volume $volume read/write test failed"
                all_ok=false
            fi
        else
            echo "✗ Volume $volume is not mounted"
            all_ok=false
        fi
    done

    if $all_ok; then
        success "All encrypted volumes are working correctly"
    else
        error "Some encrypted volumes have issues"
        exit 1
    fi
}

# Print usage information
print_usage() {
    cat << EOF

AI Trading Bot - Encrypted Volume Setup Complete!

Encrypted volumes created:
- /mnt/trading-data   (5GB)  - Trading data storage
- /mnt/trading-logs   (2GB)  - Log files
- /mnt/trading-config (512MB) - Configuration files
- /mnt/trading-backup (10GB) - Backup storage

Key files location: $KEY_DIR
Volume files location: $ENCRYPTED_DIR

Management commands:
- Start volumes:  systemctl start trading-bot-volumes
- Stop volumes:   systemctl stop trading-bot-volumes
- Status:         systemctl status trading-bot-volumes

Manual mount:     /opt/trading-bot/scripts/mount-encrypted-volumes.sh
Manual unmount:   /opt/trading-bot/scripts/umount-encrypted-volumes.sh

SECURITY NOTES:
1. Key files are stored at $KEY_DIR with 600 permissions
2. Encrypted key backup created at ${ENCRYPTED_DIR}/key-backup/
3. Store GPG passphrase securely for key recovery
4. Regular key rotation recommended every 90 days

Next steps:
1. Update docker-compose.yml to use encrypted volumes
2. Configure backup automation
3. Test recovery procedures

EOF
}

# Main execution
main() {
    log "Starting AI Trading Bot encrypted volume setup..."

    check_root
    install_dependencies
    create_directories
    generate_keys
    setup_encrypted_volumes
    create_systemd_service
    create_mount_scripts
    create_key_backup
    verify_setup

    success "Encrypted volume setup completed successfully!"
    print_usage
}

# Run main function
main "$@"
