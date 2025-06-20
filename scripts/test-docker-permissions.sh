#!/bin/bash
# Test Docker User Permissions Fix
# This script validates that Docker containers can write to host volumes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
    esac
}

echo "🧪 Docker User Permissions Test"
echo "==============================="

# Set host user ID and group ID
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

print_status "INFO" "Host user: $HOST_UID:$HOST_GID ($(whoami):$(id -gn))"

# Check if logs directory exists and its permissions
if [ -d "./logs" ]; then
    LOGS_OWNER=$(stat -f "%u:%g" ./logs)
    print_status "INFO" "Logs directory owner: $LOGS_OWNER"
    
    if [ "$LOGS_OWNER" = "$HOST_UID:$HOST_GID" ]; then
        print_status "SUCCESS" "Logs directory permissions match host user"
    else
        print_status "WARNING" "Logs directory has different owner, but Docker user mapping will handle this"
    fi
else
    print_status "ERROR" "Logs directory not found"
    exit 1
fi

# Test write permission with a simple container
print_status "INFO" "Testing write permissions with a test container..."

# Create a simple test using a basic alpine container
docker run --rm \
    --user "$HOST_UID:$HOST_GID" \
    -v "$(pwd)/logs:/test/logs" \
    alpine:latest \
    /bin/sh -c "
        echo 'Docker permission test' > /test/logs/docker-permission-test.txt && 
        echo 'Container can write to host volume' && 
        rm /test/logs/docker-permission-test.txt &&
        echo 'Container can delete from host volume'
    "

if [ $? -eq 0 ]; then
    print_status "SUCCESS" "Docker container can write to host volumes with user $HOST_UID:$HOST_GID"
else
    print_status "ERROR" "Docker container cannot write to host volumes"
    exit 1
fi

# Test environment variable substitution in docker-compose
print_status "INFO" "Testing docker-compose environment variable substitution..."

# Check if docker-compose can resolve the environment variables
if command -v docker-compose >/dev/null 2>&1; then
    # Test config parsing
    docker-compose config > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_status "SUCCESS" "docker-compose configuration is valid"
        
        # Show the resolved user configuration for ai-trading-bot service
        USER_CONFIG=$(docker-compose config | grep -A 20 "ai-trading-bot:" | grep "user:" | head -1 | sed 's/.*user: //')
        print_status "INFO" "ai-trading-bot service will run as user: $USER_CONFIG"
    else
        print_status "ERROR" "docker-compose configuration is invalid"
        exit 1
    fi
else
    print_status "WARNING" "docker-compose not found, skipping configuration test"
fi

echo ""
print_status "SUCCESS" "All permission tests passed! 🎉"
echo ""
echo "📋 Summary:"
echo "==========="
echo "• Host user: $HOST_UID:$HOST_GID ($(whoami):$(id -gn))"
echo "• Docker containers will run as this user to match file permissions"
echo "• Volume mounts will have correct read/write permissions"
echo ""
echo "🚀 Ready to start services:"
echo "   ./scripts/start-trading-bot.sh"
echo "   OR"
echo "   source scripts/setup-user-permissions.sh && docker-compose up -d"