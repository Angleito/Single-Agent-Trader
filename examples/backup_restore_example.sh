#!/bin/bash

# Example: AI Trading Bot Backup and Restore Workflow
# This script demonstrates common backup/restore scenarios

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}AI Trading Bot - Backup & Restore Examples${NC}\n"

# Example 1: Daily backup before trading
echo -e "${GREEN}Example 1: Daily Backup Before Trading${NC}"
echo "# Create a daily backup with timestamp"
echo "./scripts/backup-system.sh daily_\$(date +%Y%m%d)"
echo

# Example 2: Backup before configuration changes
echo -e "${GREEN}Example 2: Backup Before Configuration Changes${NC}"
echo "# Quick backup before modifying configs"
echo "./scripts/backup-system.sh pre_config_change"
echo "# Make your configuration changes..."
echo "# If something goes wrong, restore:"
echo "./scripts/restore-system.sh pre_config_change --components config"
echo

# Example 3: Test restore in dry-run mode
echo -e "${GREEN}Example 3: Test Restore (Dry Run)${NC}"
echo "# See what would be restored without making changes"
echo "./scripts/restore-system.sh backup_name --dry-run"
echo

# Example 4: Restore after failed upgrade
echo -e "${GREEN}Example 4: Emergency Restore After Failed Upgrade${NC}"
echo "# Stop services first"
echo "docker-compose down"
echo "# Restore from last known good backup"
echo "./scripts/restore-system.sh last_good_backup --no-interactive"
echo "# Fix permissions"
echo "./setup-docker-permissions.sh"
echo "# Restart services"
echo "docker-compose up -d"
echo

# Example 5: Partial restore for specific issues
echo -e "${GREEN}Example 5: Selective Component Restore${NC}"
echo "# Lost trading data? Restore just data and logs"
echo "./scripts/restore-system.sh backup_name --components data,logs"
echo
echo "# Corrupted configs? Restore just configurations"
echo "./scripts/restore-system.sh backup_name --components config,env"
echo
echo "# MCP memory issues? Restore AI learning data"
echo "./scripts/restore-system.sh backup_name --components mcp_memory"
echo

# Example 6: Automated backup with cleanup
echo -e "${GREEN}Example 6: Automated Backup with Rotation${NC}"
echo "# Add to crontab for daily 2 AM backups"
echo "0 2 * * * cd /path/to/project && ./scripts/backup-system.sh daily_\$(date +\\%Y\\%m\\%d) && find ./backups -name 'daily_*.tar.gz' -mtime +7 -delete"
echo

# Example 7: Backup verification
echo -e "${GREEN}Example 7: Verify Backup Integrity${NC}"
echo "# List contents of a backup"
echo "tar -tzf backups/backup_name.tar.gz | head -20"
echo
echo "# Extract and view manifest"
echo "tar -xzf backups/backup_name.tar.gz backup_name/manifest.json -O | python -m json.tool"
echo

# Example 8: Remote backup
echo -e "${GREEN}Example 8: Remote Backup Storage${NC}"
echo "# Create backup and upload to remote server"
echo "./scripts/backup-system.sh remote_backup"
echo "scp backups/remote_backup.tar.gz user@backup-server:/backups/"
echo
echo "# Or upload to S3"
echo "aws s3 cp backups/remote_backup.tar.gz s3://my-bucket/trading-bot-backups/"
echo

# Example 9: Paper trading data management
echo -e "${GREEN}Example 9: Paper Trading Data Backup${NC}"
echo "# Backup paper trading performance before reset"
echo "./scripts/backup-system.sh paper_trading_results_\$(date +%Y%m%d)"
echo "# Reset paper trading"
echo "rm -f data/paper_trading/*.json"
echo "# Initialize fresh paper trading"
echo "docker-compose restart ai-trading-bot"
echo

# Example 10: Full disaster recovery
echo -e "${GREEN}Example 10: Full Disaster Recovery${NC}"
echo "# Download backup from remote storage"
echo "scp user@backup-server:/backups/backup_name.tar.gz ./backups/"
echo "# Or from S3"
echo "aws s3 cp s3://my-bucket/trading-bot-backups/backup_name.tar.gz ./backups/"
echo
echo "# Restore everything"
echo "./scripts/restore-system.sh backup_name"
echo
echo "# Merge environment variables"
echo "# Compare and merge .env.restored with your .env file"
echo "vimdiff .env.restored .env"
echo
echo "# Fix permissions and restart"
echo "./setup-docker-permissions.sh"
echo "docker-compose up -d"
echo

# Show current backup status
echo -e "${YELLOW}Current Backup Status:${NC}"
if [ -d "./backups" ]; then
    echo "Recent backups:"
    ls -lht ./backups/*.tar.gz 2>/dev/null | head -5 || echo "No compressed backups found"
    echo
    echo "Total backup size: $(du -sh ./backups 2>/dev/null | cut -f1 || echo '0')"
    echo "Number of backups: $(ls ./backups/*.tar.gz 2>/dev/null | wc -l || echo '0')"
else
    echo "No backups directory found. Run your first backup:"
    echo "./scripts/backup-system.sh initial_backup"
fi
