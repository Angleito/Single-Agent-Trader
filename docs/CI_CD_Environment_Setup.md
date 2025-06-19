# CI/CD Environment Setup Guide

This guide explains how to set up the AI Trading Bot for CI/CD environments, including handling missing environment files and configuring GitHub Actions.

## Problem Solved

**Issue**: `env file /path/to/.env not found: stat /path/to/.env: no such file or directory`

This occurs in CI/CD environments where `.env` files don't exist, causing Docker Compose to fail.

## Solution Overview

We've implemented a robust solution that:
1. Makes `.env` files optional in docker-compose.yml
2. Provides environment variable defaults
3. Includes GitHub Actions workflow for testing
4. Offers automated CI/CD environment setup

## Changes Made

### 1. Docker Compose Configuration

Updated `docker-compose.yml` to make `.env` files optional:

```yaml
# Before (required .env file)
env_file:
  - .env

# After (optional .env file)
env_file:
  - path: .env
    required: false
```

This allows Docker Compose to run even without a `.env` file present.

### 2. GitHub Actions Workflow

Created `.github/workflows/docker-compose-test.yml` that:
- Tests Docker Compose configuration validation
- Creates mock environment variables for CI/CD
- Validates all service definitions
- Tests with and without `.env` files
- Verifies build contexts and Dockerfiles exist

### 3. CI/CD Setup Script

Created `scripts/setup-ci-env.sh` for automated environment setup:
- Generates `.env` file with safe CI/CD defaults
- Uses mock/placeholder credentials (non-functional)
- Enables paper trading mode only
- Includes comprehensive security settings

## Quick Fix Commands

### For Local Development

```bash
# Copy example environment file
cp .env.example .env

# Or create minimal .env for testing
cat > .env << EOF
EXCHANGE_TYPE=coinbase
SYSTEM__DRY_RUN=true
OPENAI_API_KEY=your-key-here
EOF
```

### For CI/CD Environments

```bash
# Run automated setup script
./scripts/setup-ci-env.sh

# Or manually test configuration
docker-compose -f docker-compose.yml config --quiet
```

### For GitHub Actions

The workflow automatically:
1. Creates a `.env` file with CI/CD safe defaults
2. Tests Docker Compose configuration
3. Validates all services and dependencies
4. Ensures environment variables are properly substituted

## Environment Variables

### Required for CI/CD

```bash
# Minimal required variables
EXCHANGE_TYPE=coinbase
SYSTEM__DRY_RUN=true
SYSTEM__ENVIRONMENT=development
LOG_LEVEL=INFO

# Mock credentials (safe for CI/CD)
OPENAI_API_KEY=sk-test-key-for-ci-cd-only
COINBASE_API_KEY=test-coinbase-key
BLUEFIN_PRIVATE_KEY=test-private-key
```

### Optional Variables

All other variables have sensible defaults defined in the docker-compose.yml:

```yaml
environment:
  - SYSTEM__DRY_RUN=${SYSTEM__DRY_RUN:-true}
  - LOG_LEVEL=${LOG_LEVEL:-INFO}
  - EXCHANGE__BLUEFIN_NETWORK=${EXCHANGE__BLUEFIN_NETWORK:-mainnet}
  # ... more defaults
```

## Security Considerations

### CI/CD Safety Features

1. **Paper Trading Only**: `SYSTEM__DRY_RUN=true` prevents real trading
2. **Mock Credentials**: Non-functional placeholder API keys
3. **Development Environment**: Safe configuration settings
4. **Feature Flags**: Live trading disabled, testing enabled

### Production Deployment

For production deployments:
1. Use real environment variables (not `.env` files)
2. Set `required: false` allows flexibility
3. Use secrets management (GitHub Secrets, Docker Secrets, etc.)
4. Override environment variables at runtime

## Testing the Solution

### Local Testing

```bash
# Test without .env file
rm .env
docker-compose config --quiet

# Test with .env file
cp .env.example .env
docker-compose config --quiet
```

### GitHub Actions Testing

The workflow automatically tests:
- Configuration validation
- Environment variable substitution  
- Service definitions
- Build contexts
- Network configuration

## Troubleshooting

### Common Issues

1. **Permission Denied on Script**
   ```bash
   chmod +x scripts/setup-ci-env.sh
   ```

2. **Missing Build Context**
   - Verify all referenced directories exist
   - Check Dockerfile locations match docker-compose.yml

3. **Network Configuration Issues**
   - Ensure trading-network is properly defined
   - Verify service aliases are correct

### Debug Commands

```bash
# Validate configuration
docker-compose config --quiet

# Show resolved configuration
docker-compose config

# Test specific service
docker-compose config --services

# Check environment variable substitution
docker-compose config | grep -i "DRY_RUN\|LOG_LEVEL"
```

## Integration with Existing Workflows

### GitHub Actions

Add to existing workflows:

```yaml
- name: Setup CI Environment
  run: ./scripts/setup-ci-env.sh

- name: Validate Docker Compose
  run: docker-compose config --quiet
```

### Other CI/CD Platforms

For Jenkins, GitLab CI, etc.:

```bash
# Create environment setup step
script:
  - ./scripts/setup-ci-env.sh
  - docker-compose config --quiet
  - docker-compose up --dry-run
```

## Benefits

1. **Robust CI/CD**: Handles missing environment files gracefully
2. **Secure by Default**: Mock credentials, paper trading only
3. **Flexible Deployment**: Works with or without `.env` files
4. **Automated Testing**: GitHub Actions validates configuration
5. **Easy Setup**: One command to create CI/CD environment

## File Structure

```
.
├── .env.example              # Template environment file
├── .github/
│   └── workflows/
│       └── docker-compose-test.yml  # GitHub Actions workflow
├── scripts/
│   └── setup-ci-env.sh      # Automated CI/CD setup
├── docker-compose.yml       # Updated with optional .env
└── docs/
    └── CI_CD_Environment_Setup.md   # This guide
```

This solution ensures your Docker Compose configuration works reliably across all environments while maintaining security and flexibility.