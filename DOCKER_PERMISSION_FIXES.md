# Docker Volume Permission Issues - Analysis and Fixes

## Problem Summary

The AI trading bot container was failing during initialization due to directory permission mismatches between the host system and Docker containers.

## Root Cause Analysis

### Issues Identified:

1. **User ID Mismatch**:
   - Host user (macOS): `angel` with UID 501, GID 20 (staff)
   - Container expectation: UID 1000, GID 1000 (botuser)
   - Volume mounts: Host directories owned by 501:20, container tries to write as 1000:1000

2. **Inconsistent Docker Compose Configuration**:
   - `bluefin-service`: Had proper user mapping `"${HOST_UID:-1000}:${HOST_GID:-1000}"`
   - `ai-trading-bot`: Had hardcoded `user: "1000:1000"` (incorrect for macOS host)

3. **Missing Permission Setup**:
   - Documentation referenced `./setup-docker-permissions.sh` but script didn't exist
   - No automated way to create directories with proper ownership

## Fixes Applied

### 1. Fixed Docker Compose User Configuration

**File**: `docker-compose.yml`

**Before**:
```yaml
ai-trading-bot:
  # ... service config ...
  user: "1000:1000"   # Hardcoded - incompatible with macOS host
```

**After**:
```yaml
ai-trading-bot:
  # ... service config ...
  user: "${HOST_UID:-1000}:${HOST_GID:-1000}"  # Dynamic host user mapping
```

### 2. Updated Environment Variables

**File**: `.env`

**Updated**:
```bash
# Docker User Permissions (Fixed for macOS)
HOST_UID=501
HOST_GID=20
```

### 3. Created Permission Setup Script

**File**: `setup-docker-permissions.sh` (NEW)

**Features**:
- Automatically detects host user ID and group ID
- Creates all required directories with proper permissions
- Sets ownership to current host user
- Updates `.env` file with correct `HOST_UID` and `HOST_GID`
- Verifies write permissions with test files
- Provides troubleshooting guidance

### 4. Updated Documentation

**Files**: `docker-compose.yml`, `CLAUDE.md`

**Updated comments and instructions**:
- Corrected user permission documentation
- Updated setup instructions to use new script
- Fixed deployment notes

## Volume Mount Strategy

The fix uses **host user mapping** rather than **fixed container users**:

### Before (Problematic):
```
Host directories: owned by 501:20 (angel:staff)
Container user: runs as 1000:1000 (botuser)
Result: Permission denied when container tries to write
```

### After (Fixed):
```
Host directories: owned by 501:20 (angel:staff)
Container user: runs as 501:20 (mapped to host user)
Result: Container can read/write as host user
```

## Directory Structure

The script sets up these directories with proper permissions:

```
logs/                   # Bot logs
├── bluefin/           # Bluefin service logs
├── mcp/               # MCP server logs
└── trades/            # Trading decision logs

data/                   # Persistent data
├── mcp_memory/        # MCP memory storage
├── omnisearch_cache/  # Search cache
├── orders/            # Order tracking
├── paper_trading/     # Paper trading data
├── positions/         # Position tracking
└── bluefin/           # Bluefin-specific data
```

## Security Implications

### Maintained Security Features:
- Read-only root filesystem (`read_only: true`)
- Dropped capabilities (`cap_drop: ALL`)
- Limited capability additions (only required ones)
- Secure tmpfs mounts for temporary files
- No-new-privileges security option

### Changes Made:
- User mapping allows container to run as host user
- Maintains file system isolation through volume mounts
- Preserves all other security hardening measures

## Usage Instructions

### First-Time Setup:
```bash
# 1. Run permission setup (creates directories, sets ownership)
./setup-docker-permissions.sh

# 2. Start services
docker-compose up -d

# 3. Verify no permission errors
docker-compose logs -f ai-trading-bot
```

### Troubleshooting:
```bash
# If permission errors persist:
docker-compose down
docker-compose build --no-cache
./setup-docker-permissions.sh
docker-compose up -d
```

## Verification

The setup script includes verification steps:
1. Creates test files in each directory
2. Confirms write permissions
3. Reports success/failure for each directory
4. Provides clear next-step instructions

## Platform Compatibility

This fix is designed to work across different platforms:
- **macOS**: Uses actual host UID/GID (501:20 typical)
- **Linux**: Uses actual host UID/GID (1000:1000 typical)
- **Windows**: Should work with Docker Desktop user mapping

The `${HOST_UID:-1000}:${HOST_GID:-1000}` syntax provides fallback values if environment variables aren't set.

## Prevention

To prevent similar issues in the future:
1. Always use `${HOST_UID:-1000}:${HOST_GID:-1000}` for services that need volume write access
2. Run `./setup-docker-permissions.sh` before first deployment
3. Include permission verification in CI/CD pipelines
4. Document user mapping requirements for new services
