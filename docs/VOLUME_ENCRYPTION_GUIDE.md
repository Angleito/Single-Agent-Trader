# AI Trading Bot - Volume Encryption and Security Guide

This guide provides comprehensive instructions for implementing and managing encrypted storage for the AI Trading Bot on Digital Ocean VPS deployments.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Management](#management)
7. [Performance Optimization](#performance-optimization)
8. [Backup and Recovery](#backup-and-recovery)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Monitoring](#monitoring)

## Overview

The volume encryption system provides enterprise-grade security for sensitive trading data using:

- **LUKS2 encryption** with AES-256-XTS cipher
- **Automated key management** with rotation policies
- **Encrypted backups** with integrity verification
- **Performance optimization** for trading workloads
- **Disaster recovery** procedures
- **Docker integration** with secure bind mounts

### Security Features

- ✅ **Data at Rest Encryption**: All sensitive data encrypted with LUKS2
- ✅ **Key Rotation**: Automated key rotation every 90 days
- ✅ **Backup Encryption**: GPG-encrypted backups with integrity checks
- ✅ **Access Controls**: Strict file permissions and user isolation
- ✅ **Audit Logging**: Complete audit trail of key operations
- ✅ **Emergency Procedures**: Rapid emergency stop and recovery

### Performance Features

- ✅ **Hardware Acceleration**: AES-NI and AVX instruction support
- ✅ **I/O Optimization**: Tuned for real-time trading performance
- ✅ **Memory Optimization**: Efficient memory usage for encrypted operations
- ✅ **Cache Optimization**: Optimized read/write patterns

## Quick Start

### 1. Initial Setup

```bash
# Run as root
sudo ./scripts/setup-encrypted-volumes.sh
```

This script will:
- Install required packages (cryptsetup, lvm2, etc.)
- Create encrypted volumes for data, logs, config, and backups
- Generate encryption keys with secure permissions
- Set up systemd service for automatic mounting
- Create performance monitoring scripts

### 2. Start Encrypted Services

```bash
# Start encrypted volumes
sudo systemctl start trading-bot-volumes

# Verify volumes are mounted
df -h | grep trading

# Start Docker services with encryption
docker-compose -f docker-compose.encrypted.yml up -d
```

### 3. Verify Security

```bash
# Check key permissions
sudo ls -la /opt/trading-bot/keys/

# Verify encryption is active
sudo cryptsetup status trading-data
sudo cryptsetup status trading-logs

# Test backup system
sudo ./scripts/backup-encrypted.sh
```

## Architecture

### Encrypted Volume Layout

```
/opt/trading-bot/
├── encrypted/                 # Volume image files
│   ├── data.img              # 5GB - Trading data
│   ├── logs.img              # 2GB - Application logs
│   ├── config.img            # 512MB - Configuration
│   ├── backup.img            # 10GB - Backup storage
│   └── key-backup/           # Encrypted key backups
├── keys/                     # Encryption keys (600 permissions)
│   ├── master.key            # Master encryption key
│   ├── data.key             # Data volume key
│   ├── logs.key             # Logs volume key
│   ├── config.key           # Config volume key
│   ├── backup.key           # Backup volume key
│   ├── backup-encryption.key # Backup encryption key
│   └── key_metadata.json    # Key management metadata
└── scripts/                 # Management scripts
    ├── setup-encrypted-volumes.sh
    ├── manage-encryption-keys.sh
    ├── backup-encrypted.sh
    ├── optimize-encrypted-performance.sh
    └── disaster-recovery.sh
```

### Mount Points

```
/mnt/trading-data/            # Encrypted data storage
/mnt/trading-logs/            # Encrypted log storage
/mnt/trading-config/          # Encrypted configuration
/mnt/trading-backup/          # Encrypted backup storage
```

### Docker Volume Mapping

```yaml
volumes:
  - /mnt/trading-data:/app/data:rw
  - /mnt/trading-logs:/app/logs:rw
  - /mnt/trading-config:/app/config:rw
```

## Installation

### Prerequisites

- Ubuntu 20.04+ or CentOS 8+
- Root access
- At least 20GB free disk space
- Docker and Docker Compose

### Step-by-Step Installation

#### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y cryptsetup lvm2 parted e2fsprogs jq curl
```

#### 2. Run Setup Script

```bash
# Make script executable
chmod +x scripts/setup-encrypted-volumes.sh

# Run setup (as root)
sudo ./scripts/setup-encrypted-volumes.sh
```

#### 3. Verify Installation

```bash
# Check services
sudo systemctl status trading-bot-volumes

# Check mounts
sudo mountpoint /mnt/trading-data
sudo mountpoint /mnt/trading-logs
sudo mountpoint /mnt/trading-config
sudo mountpoint /mnt/trading-backup

# Test encryption
sudo cryptsetup status trading-data
```

#### 4. Set Up Performance Optimization

```bash
# Run performance optimization
sudo ./scripts/optimize-encrypted-performance.sh

# Run benchmark to establish baseline
sudo ./scripts/benchmark-crypto.sh
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Encryption Configuration
ENABLE_ENCRYPTION=true
ENCRYPTION_AT_REST=true
ENCRYPTED_LOGS=true
ENCRYPTED_DATA=true
ENCRYPTED_CONFIG=true

# Key Management
KEY_ROTATION_DAYS=90
KEY_BACKUP_RETENTION=365

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION_LEVEL=6
ENABLE_OFFSITE_BACKUP=false

# Performance Tuning
ENCRYPTION_PERFORMANCE_MODE=trading
ENABLE_HARDWARE_ACCELERATION=true

# Monitoring
ENABLE_ENCRYPTION_MONITORING=true
ENCRYPTION_ALERT_THRESHOLD=95
```

### Docker Compose Configuration

Use the encrypted configuration:

```bash
# Start with encrypted volumes
docker-compose -f docker-compose.encrypted.yml up -d

# Or add to your existing compose
services:
  ai-trading-bot:
    volumes:
      - /mnt/trading-data:/app/data:rw
      - /mnt/trading-logs:/app/logs:rw
      - /mnt/trading-config:/app/config:rw
    environment:
      - ENABLE_ENCRYPTION=true
      - ENCRYPTION_AT_REST=true
```

### Systemd Service Configuration

The installation creates a systemd service for automatic mounting:

```bash
# Service file: /etc/systemd/system/trading-bot-volumes.service
sudo systemctl enable trading-bot-volumes
sudo systemctl start trading-bot-volumes
```

## Management

### Key Management

#### View Key Status

```bash
# Check key ages and rotation needs
sudo ./scripts/manage-encryption-keys.sh check-age

# Generate detailed key report
sudo ./scripts/manage-encryption-keys.sh report

# Verify key integrity
sudo ./scripts/manage-encryption-keys.sh verify
```

#### Key Rotation

```bash
# Rotate all keys (recommended monthly)
sudo ./scripts/manage-encryption-keys.sh rotate-all

# Rotate specific volume key
sudo ./scripts/manage-encryption-keys.sh rotate data

# Rotate backup encryption key
sudo ./scripts/manage-encryption-keys.sh rotate-backup
```

#### Automatic Key Rotation

```bash
# Set up automatic rotation (weekly checks)
sudo ./scripts/manage-encryption-keys.sh setup-timer

# Check timer status
sudo systemctl status trading-bot-key-rotation.timer
```

### Volume Management

#### Manual Mount/Unmount

```bash
# Mount all encrypted volumes
sudo /opt/trading-bot/scripts/mount-encrypted-volumes.sh

# Unmount all encrypted volumes
sudo /opt/trading-bot/scripts/umount-encrypted-volumes.sh

# Check mount status
sudo systemctl status trading-bot-volumes
```

#### Volume Expansion

```bash
# Stop services
docker-compose -f docker-compose.encrypted.yml down

# Expand volume (example: data volume to 10GB)
sudo dd if=/dev/zero bs=1G count=5 >> /opt/trading-bot/encrypted/data.img

# Resize LUKS container
sudo cryptsetup resize trading-data

# Resize filesystem
sudo resize2fs /dev/mapper/trading-data

# Restart services
docker-compose -f docker-compose.encrypted.yml up -d
```

## Performance Optimization

### Automatic Optimization

```bash
# Run full performance optimization
sudo ./scripts/optimize-encrypted-performance.sh

# Monitor performance
sudo ./scripts/monitor-crypto-performance.sh

# Run benchmark
sudo ./scripts/benchmark-crypto.sh
```

### Manual Tuning

#### Kernel Parameters

```bash
# Add to /etc/sysctl.d/99-trading-bot-crypto.conf
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
```

#### I/O Scheduler

```bash
# Set deadline scheduler for better latency
echo deadline > /sys/block/loop*/queue/scheduler

# Optimize read-ahead
echo 128 > /sys/block/loop*/queue/read_ahead_kb
```

#### CPU Features

```bash
# Check AES-NI support
grep aes /proc/cpuinfo

# Check AVX support
grep avx /proc/cpuinfo

# Monitor crypto CPU usage
top -p $(pgrep -d, crypto)
```

### Performance Benchmarks

Expected performance with SSD storage and AES-NI:

- **Sequential Read**: 400-600 MB/s
- **Sequential Write**: 300-500 MB/s
- **Random 4K Read**: 80-120 MB/s
- **Random 4K Write**: 60-100 MB/s
- **Encryption Overhead**: 5-15%

## Backup and Recovery

### Automated Backups

#### Daily Backups

```bash
# Run manual backup
sudo ./scripts/backup-encrypted.sh

# Set up cron job for daily backups
sudo crontab -e
# Add: 0 2 * * * /path/to/scripts/backup-encrypted.sh
```

#### Backup Verification

```bash
# Verify specific backup
sudo ./scripts/backup-encrypted.sh verify /mnt/trading-backup/data_20241201_120000.tar.gz.gpg

# Verify all recent backups
find /mnt/trading-backup -name "*.tar.gz.gpg" -mtime -7 | while read file; do
    sudo ./scripts/backup-encrypted.sh verify "$file"
done
```

#### Offsite Backups

Configure remote backup:

```bash
# Set environment variables
export ENABLE_OFFSITE_BACKUP=true
export BACKUP_REMOTE_HOST=backup.example.com
export BACKUP_REMOTE_USER=trading-bot
export BACKUP_REMOTE_PATH=/backups/trading-bot

# Set up SSH key authentication
ssh-keygen -t ed25519 -f ~/.ssh/trading_backup
ssh-copy-id -i ~/.ssh/trading_backup.pub trading-bot@backup.example.com
```

### Disaster Recovery

#### Emergency Stop

```bash
# Emergency stop all operations
sudo ./scripts/disaster-recovery.sh emergency-stop

# Resume operations
sudo rm /opt/trading-bot/EMERGENCY_STOP
sudo systemctl start trading-bot-volumes
docker-compose -f docker-compose.encrypted.yml up -d
```

#### Full Disaster Recovery

```bash
# Full system recovery from backups
sudo ./scripts/disaster-recovery.sh full-recovery

# Recovery from specific backup timestamp
sudo ./scripts/disaster-recovery.sh full-recovery 20241201_120000
```

#### Partial Recovery

```bash
# Recover specific volume
sudo ./scripts/disaster-recovery.sh partial-recovery data

# Recover with specific timestamp
sudo ./scripts/disaster-recovery.sh partial-recovery logs 20241201_120000
```

### Recovery Testing

Regular recovery testing is essential:

```bash
# Monthly recovery test procedure
1. sudo ./scripts/backup-encrypted.sh
2. sudo ./scripts/disaster-recovery.sh test-backups
3. Verify test results
4. Document any issues
```

## Security Best Practices

### Key Security

1. **Key Storage**
   - Store keys on encrypted filesystem
   - Use hardware security modules (HSM) if available
   - Never store keys in version control
   - Regular key rotation (90 days)

2. **Access Control**
   - Limit root access
   - Use principle of least privilege
   - Monitor key access with audit logs
   - Implement dual approval for key operations

3. **Backup Security**
   - Encrypt all backups
   - Store offsite securely
   - Test recovery procedures regularly
   - Verify backup integrity

### System Security

1. **Operating System**
   - Keep system updated
   - Use SELinux/AppArmor
   - Configure firewall
   - Monitor system logs

2. **Docker Security**
   - Use non-root users
   - Read-only filesystems
   - Minimal capabilities
   - Security scanning

3. **Network Security**
   - VPN for management access
   - Certificate pinning
   - Network segmentation
   - Traffic encryption

### Monitoring and Alerting

1. **Key Monitoring**
   - Key age alerts
   - Failed access attempts
   - Unauthorized key changes
   - Backup failures

2. **Performance Monitoring**
   - Encryption overhead
   - I/O performance
   - CPU utilization
   - Memory usage

3. **Security Monitoring**
   - Failed mount attempts
   - Unusual access patterns
   - System intrusion detection
   - Log analysis

## Troubleshooting

### Common Issues

#### Volume Won't Mount

```bash
# Check loop device
sudo losetup -l | grep trading

# Check LUKS status
sudo cryptsetup status trading-data

# Check key file
sudo ls -la /opt/trading-bot/keys/data.key

# Manual mount
sudo /opt/trading-bot/scripts/mount-encrypted-volumes.sh
```

#### Performance Issues

```bash
# Check I/O stats
sudo iostat -x 1 5

# Check encryption overhead
sudo ./scripts/monitor-crypto-performance.sh 60

# Check CPU usage
sudo top -p $(pgrep -d, crypto)

# Check memory usage
sudo free -h
```

#### Key Issues

```bash
# Verify key integrity
sudo ./scripts/manage-encryption-keys.sh verify

# Check key permissions
sudo ls -la /opt/trading-bot/keys/

# Test key with LUKS
sudo cryptsetup luksOpen --test-passphrase /dev/loop0 --key-file /opt/trading-bot/keys/data.key
```

#### Backup Issues

```bash
# Test backup creation
sudo ./scripts/backup-encrypted.sh

# Verify backup integrity
sudo ./scripts/backup-encrypted.sh verify <backup_file>

# Check backup space
sudo df -h /mnt/trading-backup
```

### Recovery Procedures

#### Corrupted Volume

```bash
# Stop services
docker-compose -f docker-compose.encrypted.yml down

# Check filesystem
sudo fsck.ext4 /dev/mapper/trading-data

# If corruption is severe, recover from backup
sudo ./scripts/disaster-recovery.sh partial-recovery data
```

#### Lost Keys

```bash
# Restore keys from backup
sudo ./scripts/disaster-recovery.sh full-recovery

# Or restore manually
sudo gpg --decrypt /opt/trading-bot/encrypted/key-backup/keys_*.tar.gz.gpg | sudo tar -xzf - -C /opt/trading-bot/keys/
```

#### System Compromise

```bash
# Immediate response
sudo ./scripts/disaster-recovery.sh emergency-stop

# Investigate and assess damage
# Rebuild system if necessary
sudo ./scripts/disaster-recovery.sh full-recovery

# Update all keys after recovery
sudo ./scripts/manage-encryption-keys.sh rotate-all
```

## Monitoring

### Automated Monitoring

```bash
# Set up monitoring
sudo ./scripts/monitor-crypto-performance.sh continuous &

# Check monitoring status
ps aux | grep monitor-crypto-performance
```

### Manual Monitoring

```bash
# Check volume status
sudo df -h | grep trading
sudo cryptsetup status trading-data

# Check key status
sudo ./scripts/manage-encryption-keys.sh report

# Check backup status
sudo ls -la /mnt/trading-backup/ | tail -10

# Check performance
sudo ./scripts/benchmark-crypto.sh
```

### Alerting

Set up alerts for:

- Disk space > 80%
- Failed backup operations
- Key rotation needed
- Performance degradation
- Security events

Example alert script:

```bash
#!/bin/bash
# Alert if encryption volume > 80% full
usage=$(df /mnt/trading-data | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $usage -gt 80 ]; then
    echo "WARNING: Trading data volume is ${usage}% full" | mail -s "Disk Alert" admin@example.com
fi
```

## Advanced Configuration

### High Availability Setup

For production deployments requiring high availability:

1. **RAID Configuration**
   ```bash
   # Set up RAID 1 for redundancy
   sudo mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc

   # Create LUKS on RAID
   sudo cryptsetup luksFormat /dev/md0 /opt/trading-bot/keys/raid.key
   ```

2. **Cluster Setup**
   ```bash
   # Shared storage with LUKS
   # Network-attached storage encryption
   # Distributed key management
   ```

### Compliance Configurations

#### PCI DSS Compliance

```bash
# Enhanced key rotation (monthly)
KEY_ROTATION_DAYS=30

# Strong cipher requirements
LUKS_CIPHER="aes-xts-plain64"
LUKS_KEY_SIZE=512
LUKS_HASH="sha512"

# Audit logging
ENABLE_AUDIT_LOGGING=true
AUDIT_LOG_RETENTION=2555  # 7 years
```

#### SOX Compliance

```bash
# Dual approval for key operations
REQUIRE_DUAL_APPROVAL=true

# Extended retention
KEY_BACKUP_RETENTION=2555  # 7 years
BACKUP_RETENTION_DAYS=2555

# Immutable audit logs
AUDIT_LOG_IMMUTABLE=true
```

## Conclusion

This volume encryption system provides enterprise-grade security for the AI Trading Bot while maintaining high performance for real-time trading operations. Regular maintenance, monitoring, and testing ensure the system remains secure and reliable.

For additional support or questions, refer to the troubleshooting section or contact the development team.

## References

- [LUKS Documentation](https://gitlab.com/cryptsetup/cryptsetup/-/wikis/home)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Linux Encryption Guide](https://wiki.archlinux.org/title/Dm-crypt)
- [Performance Tuning Guide](https://wiki.archlinux.org/title/Improving_performance)
