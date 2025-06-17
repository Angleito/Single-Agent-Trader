#!/bin/bash
# Clean start script for AI Trading Bot

set -e

echo "🧹 Cleaning up existing Docker resources..."

# Stop all containers
echo "Stopping containers..."
docker-compose down -v 2>/dev/null || true

# Remove all stopped containers
echo "Removing stopped containers..."
docker ps -aq | xargs -r docker rm -f 2>/dev/null || true

# Prune system
echo "Pruning Docker system..."
docker system prune -a --volumes -f

# Clear local data directories
echo "Clearing local data directories..."
rm -rf data/* logs/* 2>/dev/null || true
mkdir -p data logs

echo ""
echo "🔨 Building fresh images..."
docker-compose build --no-cache

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "📊 Checking service status..."
sleep 5
docker-compose ps

echo ""
echo "📜 Viewing logs (Ctrl+C to exit)..."
docker-compose logs -f
