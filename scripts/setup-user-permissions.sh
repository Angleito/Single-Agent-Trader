#!/bin/bash
# Setup User Permissions for Docker Containers
# Source this script to set HOST_UID and HOST_GID environment variables
# Usage: source scripts/setup-user-permissions.sh

# Set host user ID and group ID for proper volume permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

echo "✅ Docker user permissions configured:"
echo "   HOST_UID=$HOST_UID ($(whoami))"
echo "   HOST_GID=$HOST_GID ($(id -gn))"
echo ""
echo "📋 You can now run Docker Compose commands:"
echo "   docker-compose up -d"
echo "   docker-compose logs -f ai-trading-bot"
echo ""
echo "💡 Or use the automated startup script:"
echo "   ./scripts/start-trading-bot.sh"