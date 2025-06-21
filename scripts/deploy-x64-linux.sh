#!/bin/bash
# X64 Linux Deployment Script
# Ensures consistent platform builds for linux/amd64

set -e

echo "🚀 Starting X64 Linux Deployment..."

# Set platform environment variable
export DOCKER_DEFAULT_PLATFORM=linux/amd64
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "✅ Platform environment set to: $DOCKER_DEFAULT_PLATFORM"

# Validate platform specifications
echo "🔍 Running platform validation..."
./scripts/validate-platforms.sh

# Build all services with explicit platform specification
echo "🏗️  Building all services for linux/amd64..."
docker-compose build --platform linux/amd64 --parallel

# Verify built images
echo "📋 Verifying built images..."
images=$(docker-compose config --services)
for service in $images; do
    image_name=$(docker-compose config | grep -A 5 "^  $service:" | grep "image:" | awk '{print $2}' | head -1)
    if [ -n "$image_name" ]; then
        # Check if image exists
        if docker image inspect "$image_name" >/dev/null 2>&1; then
            arch=$(docker image inspect "$image_name" | jq -r '.[0].Architecture' 2>/dev/null || echo "unknown")
            echo "✅ $service ($image_name): $arch"
        else
            echo "⚠️  $service ($image_name): image not found"
        fi
    fi
done

echo ""
echo "🎯 X64 Linux Deployment Ready!"
echo ""
echo "Next steps:"
echo "1. Deploy with: DOCKER_DEFAULT_PLATFORM=linux/amd64 docker-compose up -d"
echo "2. Monitor logs: docker-compose logs -f"
echo "3. Check health: docker-compose ps"
echo ""
echo "Platform-specific commands:"
echo "  - Build only: docker-compose build --platform linux/amd64"
echo "  - Force rebuild: docker-compose build --platform linux/amd64 --no-cache"
echo "  - Pull images: docker-compose pull --platform linux/amd64"
echo ""
