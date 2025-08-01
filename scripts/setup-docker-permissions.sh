#!/bin/bash
# Docker Permissions Setup Script
# This script sets up proper Docker volume permissions for the AI Trading Bot
# Run this BEFORE starting Docker containers to prevent permission errors

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
        "SETUP") echo -e "${PURPLE}🔧 $2${NC}" ;;
        "VALIDATE") echo -e "${CYAN}🧪 $2${NC}" ;;
    esac
}

echo "🐳 Docker Permissions Setup for AI Trading Bot"
echo "=============================================="
echo ""

# Detect host user ID and group ID
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(whoami)
HOST_GROUP=$(id -gn)

print_status "INFO" "Detected host user: $HOST_UID:$HOST_GID ($HOST_USER:$HOST_GROUP)"

# Detect operating system
OS_TYPE=$(uname -s)
print_status "INFO" "Detected operating system: $OS_TYPE"

# Platform-specific adjustments
case $OS_TYPE in
    "Darwin")
        print_status "INFO" "macOS detected - using standard Unix permissions"
        CHMOD_WRITABLE="775"
        CHMOD_READONLY="755"
        ;;
    "Linux")
        print_status "INFO" "Linux detected - using strict permissions"
        CHMOD_WRITABLE="775"
        CHMOD_READONLY="755"
        ;;
    *)
        print_status "WARNING" "Unknown OS ($OS_TYPE) - using default permissions"
        CHMOD_WRITABLE="775"
        CHMOD_READONLY="755"
        ;;
esac

echo ""
print_status "SETUP" "Creating required directories..."

# Define all required directories
DIRECTORIES=(
    # Main data directories
    "logs"
    "data"
    "tmp"

    # Exchange-specific directories
    "logs/bluefin"
    "logs/trades"
    "logs/mcp"
    "data/paper_trading"
    "data/orders"
    "data/positions"
    "data/bluefin"
    "data/mcp_memory"
    "data/omnisearch_cache"
    "tmp/bluefin"

    # Functional Programming directories
    "logs/fp"
    "logs/fp/migration"
    "logs/fp/performance"
    "logs/fp/validation"
    "data/fp"
    "data/fp/benchmarks"
    "data/fp/migration_reports"
    "data/fp/validation_results"
    "tmp/fp"
    "tmp/fp/testing"
    "tmp/fp/migration"

    # Dashboard directories
    "dashboard/backend/logs"
    "dashboard/backend/data"
)

# Create directories with proper ownership and permissions
CREATED_DIRS=0
FIXED_DIRS=0
EXISTING_DIRS=0

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        chown "$HOST_UID:$HOST_GID" "$dir" 2>/dev/null || true
        chmod "$CHMOD_WRITABLE" "$dir"
        print_status "SUCCESS" "Created directory: $dir"
        CREATED_DIRS=$((CREATED_DIRS + 1))
    else
        # Check if directory has correct ownership
        CURRENT_OWNER=$(stat -f "%u:%g" "$dir" 2>/dev/null || stat -c "%u:%g" "$dir" 2>/dev/null || echo "unknown")
        if [ "$CURRENT_OWNER" != "$HOST_UID:$HOST_GID" ]; then
            chown "$HOST_UID:$HOST_GID" "$dir" 2>/dev/null || true
            chmod "$CHMOD_WRITABLE" "$dir"
            print_status "SETUP" "Fixed ownership for existing directory: $dir ($CURRENT_OWNER → $HOST_UID:$HOST_GID)"
            FIXED_DIRS=$((FIXED_DIRS + 1))
        else
            EXISTING_DIRS=$((EXISTING_DIRS + 1))
        fi
    fi
done

# Create essential files if they don't exist
print_status "SETUP" "Creating essential files..."

# Create empty log files with proper permissions
LOG_FILES=(
    "logs/bot.log"
    "logs/live_trading.log"
    "logs/llm_completions.log"
    "data/paper_trading/account.json"
    "data/paper_trading/trades.json"
    "data/orders/active_orders.json"
    "data/positions/positions.json"

    # Functional Programming log files
    "logs/fp/migration.log"
    "logs/fp/performance.log"
    "logs/fp/validation.log"
    "logs/fp/type_checking.log"
    "data/fp/migration_reports/latest_report.json"
    "data/fp/benchmarks/performance_baseline.json"
    "data/fp/validation_results/latest_validation.json"
)

for log_file in "${LOG_FILES[@]}"; do
    if [ ! -f "$log_file" ]; then
        # Ensure parent directory exists
        mkdir -p "$(dirname "$log_file")"

        # Create appropriate default content based on file type
        case "$log_file" in
            *.json)
                echo "{}" > "$log_file"
                ;;
            *.log)
                touch "$log_file"
                ;;
            *)
                touch "$log_file"
                ;;
        esac

        chown "$HOST_UID:$HOST_GID" "$log_file" 2>/dev/null || true
        chmod 664 "$log_file"
        print_status "SUCCESS" "Created file: $log_file"
    fi
done

echo ""
print_status "SETUP" "Updating .env file with host user permissions..."

# Update or create .env file with HOST_UID and HOST_GID
ENV_FILE=".env"
TEMP_ENV_FILE=".env.tmp"

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    print_status "WARNING" ".env file not found - creating from .env.example"
    if [ -f ".env.example" ]; then
        cp ".env.example" "$ENV_FILE"
        print_status "SUCCESS" "Created .env from .env.example"
    else
        print_status "ERROR" ".env.example not found - creating minimal .env"
        cat > "$ENV_FILE" << EOF
# AI Trading Bot Environment Configuration
# Generated by setup-docker-permissions.sh

# Docker User Permissions (Auto-detected)
HOST_UID=$HOST_UID
HOST_GID=$HOST_GID

# System Configuration
SYSTEM__DRY_RUN=true
SYSTEM__ENVIRONMENT=development

# Exchange Configuration
EXCHANGE__EXCHANGE_TYPE=coinbase

# Add your API keys here:
# LLM__OPENAI_API_KEY=your_openai_api_key_here
# EXCHANGE__CDP_API_KEY_NAME=your_coinbase_api_key_name
# EXCHANGE__CDP_PRIVATE_KEY=your_coinbase_private_key_here
EOF
    fi
fi

# Create a temporary file to update the .env
cp "$ENV_FILE" "$TEMP_ENV_FILE"

# Function to update or add environment variable
update_env_var() {
    local var_name="$1"
    local var_value="$2"
    local temp_file="$3"

    if grep -q "^${var_name}=" "$temp_file"; then
        # Variable exists, update it
        if [[ "$OS_TYPE" == "Darwin" ]]; then
            sed -i '' "s/^${var_name}=.*/${var_name}=${var_value}/" "$temp_file"
        else
            sed -i "s/^${var_name}=.*/${var_name}=${var_value}/" "$temp_file"
        fi
        print_status "SETUP" "Updated $var_name=$var_value in .env"
    elif grep -q "^#.*${var_name}=" "$temp_file"; then
        # Variable exists but commented, uncomment and update
        if [[ "$OS_TYPE" == "Darwin" ]]; then
            sed -i '' "s/^#.*${var_name}=.*/${var_name}=${var_value}/" "$temp_file"
        else
            sed -i "s/^#.*${var_name}=.*/${var_name}=${var_value}/" "$temp_file"
        fi
        print_status "SETUP" "Uncommented and updated $var_name=$var_value in .env"
    else
        # Variable doesn't exist, add it
        echo "" >> "$temp_file"
        echo "# Docker User Permissions (Auto-detected by setup-docker-permissions.sh)" >> "$temp_file"
        echo "${var_name}=${var_value}" >> "$temp_file"
        print_status "SETUP" "Added $var_name=$var_value to .env"
    fi
}

# Update HOST_UID and HOST_GID
update_env_var "HOST_UID" "$HOST_UID" "$TEMP_ENV_FILE"
update_env_var "HOST_GID" "$HOST_GID" "$TEMP_ENV_FILE"

# Move temp file to actual .env
mv "$TEMP_ENV_FILE" "$ENV_FILE"
chown "$HOST_UID:$HOST_GID" "$ENV_FILE" 2>/dev/null || true
chmod 600 "$ENV_FILE"  # Secure permissions for .env file

echo ""
print_status "VALIDATE" "Validating directory permissions..."

# Validate write access to all directories
VALIDATION_ERRORS=0
for dir in "${DIRECTORIES[@]}"; do
    TEST_FILE="$dir/docker-permission-test-$$.txt"
    if echo "test" > "$TEST_FILE" 2>/dev/null; then
        rm -f "$TEST_FILE"
        print_status "SUCCESS" "Write access confirmed for $dir"
    else
        print_status "ERROR" "Cannot write to $dir"
        VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    fi
done

# Test Docker user mapping if Docker is available
echo ""
print_status "VALIDATE" "Testing Docker container user mapping..."

if command -v docker >/dev/null 2>&1; then
    # Test with a simple alpine container
    print_status "INFO" "Running Docker permission test container..."

    TEST_RESULT=$(docker run --rm \
        --user "$HOST_UID:$HOST_GID" \
        -v "$(pwd)/logs:/test/logs" \
        alpine:latest \
        /bin/sh -c "
            whoami 2>/dev/null || echo 'user-$HOST_UID'
            echo 'Docker permission test' > /test/logs/docker-test-$$.txt &&
            cat /test/logs/docker-test-$$.txt &&
            rm /test/logs/docker-test-$$.txt &&
            echo 'SUCCESS: Container can write to host volumes'
        " 2>&1)

    if echo "$TEST_RESULT" | grep -q "SUCCESS: Container can write to host volumes"; then
        print_status "SUCCESS" "Docker container user mapping works correctly"
        CONTAINER_USER=$(echo "$TEST_RESULT" | head -1)
        print_status "INFO" "Container runs as user: $CONTAINER_USER"
    else
        print_status "ERROR" "Docker container cannot write to host volumes"
        print_status "ERROR" "Test output: $TEST_RESULT"
        VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    fi
else
    print_status "WARNING" "Docker not found - skipping container test"
fi

# Test docker-compose configuration
echo ""
print_status "VALIDATE" "Validating docker-compose configuration..."

if command -v docker-compose >/dev/null 2>&1; then
    if docker-compose config >/dev/null 2>&1; then
        print_status "SUCCESS" "docker-compose.yml configuration is valid"

        # Show resolved user configuration
        RESOLVED_USER=$(docker-compose config 2>/dev/null | grep -A 5 "ai-trading-bot:" | grep "user:" | head -1 | sed 's/.*user: //' | tr -d '"')
        if [ -n "$RESOLVED_USER" ]; then
            print_status "INFO" "ai-trading-bot service will run as user: $RESOLVED_USER"
        fi
    else
        print_status "ERROR" "docker-compose.yml configuration is invalid"
        VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    fi
else
    print_status "WARNING" "docker-compose not found - skipping configuration test"
fi

# Summary report
echo ""
echo "📊 Setup Summary"
echo "================"
print_status "INFO" "Directories created: $CREATED_DIRS"
print_status "INFO" "Directories fixed: $FIXED_DIRS"
print_status "INFO" "Directories already correct: $EXISTING_DIRS"
print_status "INFO" "Host user: $HOST_UID:$HOST_GID ($HOST_USER:$HOST_GROUP)"
print_status "INFO" "Operating system: $OS_TYPE"

if [ $VALIDATION_ERRORS -eq 0 ]; then
    echo ""
    print_status "SUCCESS" "All permissions configured successfully! 🎉"
    echo ""
    echo "📋 What was configured:"
    echo "======================"
    echo "• Created $(( CREATED_DIRS + FIXED_DIRS )) directories with proper ownership ($HOST_UID:$HOST_GID)"
    echo "• Updated .env file with HOST_UID and HOST_GID variables"
    echo "• Set directory permissions to $CHMOD_WRITABLE for writable dirs"
    echo "• Validated write access to all required directories"
    echo "• Tested Docker container user mapping"
    echo ""
    echo "🚀 Ready to start services:"
    echo "=========================="
    echo "   # Start all services:"
    echo "   docker-compose up"
    echo ""
    echo "   # Start in background:"
    echo "   docker-compose up -d"
    echo ""
    echo "   # View logs:"
    echo "   docker-compose logs -f ai-trading-bot"
    echo ""
    echo "   # Alternative startup scripts:"
    echo "   ./scripts/start-trading-bot.sh"
    echo ""
    echo "🧮 Functional Programming Support:"
    echo "================================="
    echo "The following FP-specific directories have been configured:"
    echo "• logs/fp/ - Functional programming logs (migration, performance, validation)"
    echo "• data/fp/ - FP benchmarks, migration reports, and validation results"
    echo "• tmp/fp/ - Temporary FP testing and migration data"
    echo ""
    echo "🔧 FP Development Tools:"
    echo "======================="
    echo "   # Run FP-specific code quality checks:"
    echo "   ./scripts/fp-code-quality.sh"
    echo ""
    echo "   # FP migration assistance:"
    echo "   ./scripts/fp-migration-helper.sh"
    echo ""
    echo "   # FP performance benchmarks:"
    echo "   ./scripts/fp-performance-benchmark.sh"
    echo ""
    echo "   # FP validation and testing:"
    echo "   ./scripts/fp-test-runner.sh"
    echo ""
else
    echo ""
    print_status "ERROR" "Setup completed with $VALIDATION_ERRORS validation errors"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "=================="
    echo "1. Ensure you have write permissions to the current directory"
    echo "2. Try running with sudo if permission errors persist:"
    echo "   sudo ./setup-docker-permissions.sh"
    echo "3. Check that Docker is installed and running"
    echo "4. Verify your .env file contains the correct API keys"
    echo ""
    exit 1
fi

# Security reminder
echo ""
print_status "WARNING" "Security Reminders:"
echo "==================="
echo "• .env file contains sensitive API keys - never commit it to version control"
echo "• .env file permissions set to 600 (owner read/write only)"
echo "• Always test with SYSTEM__DRY_RUN=true before live trading"
echo "• Monitor your API key usage and set up alerts"
echo "• Consider using different API keys for development/production"
echo ""
