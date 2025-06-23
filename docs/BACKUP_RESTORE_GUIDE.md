# AI Trading Bot Backup & Restore Guide

This guide covers the backup and restore scripts for the AI Trading Bot system, which provide comprehensive data protection and recovery capabilities.

## Overview

The backup system captures all critical components:
- Configuration files (JSON configs)
- Environment settings (.env files with masked secrets)
- Trading logs (compressed)
- Database state (positions, paper trading, inventory)
- Docker volumes (dashboard data)
- MCP memory data (AI learning history)
- LLM prompts

## Backup Script (`backup-system.sh`)

### Basic Usage

```bash
# Create a backup with automatic timestamp
./scripts/backup-system.sh

# Create a backup with custom name
./scripts/backup-system.sh my_backup_name

# Create compressed-only backup (removes uncompressed files)
COMPRESS_ONLY=true ./scripts/backup-system.sh

# Enable debug output
DEBUG=true ./scripts/backup-system.sh
```

### What Gets Backed Up

1. **Configuration Files** (`config/`)
   - All JSON configuration files
   - Schema definitions
   - Profile configurations

2. **Environment Files**
   - `.env` (with sensitive data masked)
   - `.env.example`
   - SHA256 hash of original for verification

3. **Logs** (`logs/`)
   - All trading logs
   - Decision logs
   - LLM completion logs
   - Compressed into tar.gz archive

4. **Data Directory** (`data/`)
   - Paper trading accounts
   - Position files
   - Inventory state
   - MCP memory data
   - Excludes temporary/cache files

5. **Docker Volumes**
   - dashboard-logs
   - dashboard-data

6. **Prompts** (`prompts/`)
   - LLM prompt templates

### Backup Output

Backups are stored in `./backups/` directory with the following structure:
```
backups/
├── backup_20240115_120000/
│   ├── config/
│   ├── env/
│   ├── logs/
│   ├── data/
│   ├── docker_volumes/
│   ├── mcp_memory/
│   ├── prompts/
│   ├── manifest.json
│   ├── backup.log
│   └── BACKUP_SUMMARY.md
└── backup_20240115_120000.tar.gz
```

### Security Features

- Sensitive data (API keys, passwords) are automatically masked in .env backups
- Original .env file hash is stored for verification
- Read-only access to source files during backup
- Comprehensive logging of all operations

## Restore Script (`restore-system.sh`)

### Basic Usage

```bash
# Interactive restore (prompts for confirmation)
./scripts/restore-system.sh backup_20240115_120000

# Dry run (shows what would be restored)
./scripts/restore-system.sh backup_20240115_120000 --dry-run

# Non-interactive restore
./scripts/restore-system.sh backup_20240115_120000 --no-interactive

# Restore specific components only
./scripts/restore-system.sh backup_20240115_120000 --components config,data

# Skip validation checks
./scripts/restore-system.sh backup_20240115_120000 --skip-validation

# Show help
./scripts/restore-system.sh --help
```

### Restore Options

- `--dry-run`: Preview what would be restored without making changes
- `--no-interactive`: Skip all confirmation prompts
- `--skip-validation`: Skip backup integrity validation
- `--components <list>`: Restore only specific components (comma-separated)

### Available Components

- `config`: Configuration files
- `env`: Environment files
- `logs`: Trading and system logs
- `data`: Database and state files
- `mcp_memory`: MCP learning data
- `docker_volumes`: Docker volume data
- `prompts`: LLM prompts

### Restoration Process

1. **Validation**: Checks backup integrity and manifest
2. **Rollback Creation**: Saves current state for recovery
3. **Component Restoration**: Restores selected components
4. **Post-Validation**: Verifies restored files and permissions
5. **Cleanup**: Removes temporary files

### Rollback Feature

The restore script automatically creates a rollback backup before making changes:
- Stored in `.rollback_TIMESTAMP` directory
- Includes critical configuration and data files
- Can be used to revert if restoration fails

### Handling Environment Files

When restoring environment files:
- Masked .env is saved as `.env.restored`
- You must manually merge API keys and secrets
- Original file hash is available for verification

## Best Practices

### Regular Backups

1. **Before Major Changes**
   ```bash
   ./scripts/backup-system.sh pre_upgrade_backup
   ```

2. **Daily Automated Backups**
   ```bash
   # Add to crontab
   0 2 * * * /path/to/scripts/backup-system.sh daily_$(date +\%Y\%m\%d)
   ```

3. **After Successful Trading Sessions**
   ```bash
   ./scripts/backup-system.sh successful_session_$(date +%Y%m%d)
   ```

### Backup Rotation

```bash
# Keep only last 7 days of backups
find ./backups -name "daily_*.tar.gz" -mtime +7 -delete

# Keep only last 3 manual backups
ls -t ./backups/*.tar.gz | tail -n +4 | xargs rm -f
```

### Testing Restores

1. **Test Restore to Separate Directory**
   ```bash
   # Create test environment
   cp -r /path/to/project /tmp/test_restore
   cd /tmp/test_restore

   # Test restore
   ./scripts/restore-system.sh backup_name --dry-run
   ```

2. **Verify Specific Components**
   ```bash
   # Check what's in a backup
   tar -tzf backups/backup_name.tar.gz | head -20

   # Extract and inspect manifest
   tar -xzf backups/backup_name.tar.gz backup_name/manifest.json -O | jq .
   ```

## Troubleshooting

### Common Issues

1. **Permission Errors During Backup**
   ```bash
   # Fix permissions
   ./setup-docker-permissions.sh

   # Run backup with sudo (not recommended)
   sudo ./scripts/backup-system.sh
   ```

2. **Docker Volumes Not Backing Up**
   ```bash
   # Ensure Docker is running
   docker ps

   # Check volume existence
   docker volume ls
   ```

3. **Restore Validation Failures**
   ```bash
   # Skip validation if you trust the backup
   ./scripts/restore-system.sh backup_name --skip-validation

   # Check backup integrity manually
   tar -tzf backups/backup_name.tar.gz > /dev/null
   echo $?  # Should be 0
   ```

### Recovery Scenarios

1. **Corrupted Configuration**
   ```bash
   # Restore only config files
   ./scripts/restore-system.sh last_good_backup --components config
   ```

2. **Lost Trading History**
   ```bash
   # Restore data and logs
   ./scripts/restore-system.sh backup_name --components data,logs
   ```

3. **Complete System Recovery**
   ```bash
   # Full restore
   ./scripts/restore-system.sh backup_name

   # Fix permissions
   ./setup-docker-permissions.sh

   # Restart services
   docker-compose down
   docker-compose up -d
   ```

### Verification After Restore

```bash
# Check service health
docker-compose ps

# Verify configurations
python scripts/validate_config.py

# Test trading bot startup
docker-compose logs ai-trading-bot --tail 50

# Check file permissions
ls -la logs/ data/ config/
```

## Advanced Usage

### Partial Backups

```bash
# Backup only trading data
tar -czf backups/trading_data_$(date +%Y%m%d).tar.gz \
  data/paper_trading \
  data/positions \
  logs/trades

# Backup only configurations
tar -czf backups/configs_$(date +%Y%m%d).tar.gz \
  config/ \
  .env \
  prompts/
```

### Remote Backups

```bash
# Backup to remote server
./scripts/backup-system.sh remote_backup
scp backups/remote_backup.tar.gz user@backup-server:/backups/

# Backup to S3
./scripts/backup-system.sh s3_backup
aws s3 cp backups/s3_backup.tar.gz s3://my-bucket/trading-bot-backups/
```

### Automated Backup Monitoring

```bash
# Check last backup age
find ./backups -name "*.tar.gz" -mtime -1 | wc -l

# Alert if no recent backup
if [ $(find ./backups -name "*.tar.gz" -mtime -1 | wc -l) -eq 0 ]; then
    echo "WARNING: No backup in last 24 hours!"
fi
```

## Important Notes

1. **Always test restores** in a non-production environment first
2. **Keep multiple backup versions** for different time periods
3. **Store critical backups off-site** for disaster recovery
4. **Document your backup schedule** and retention policy
5. **Verify backup integrity** regularly with test restores
6. **Never restore a backup** without understanding what changed since it was created

## Support

For issues or questions about backup/restore:
1. Check the backup/restore logs in the backup directory
2. Review the manifest.json for backup details
3. Use --dry-run to preview operations
4. Enable DEBUG=true for verbose output
