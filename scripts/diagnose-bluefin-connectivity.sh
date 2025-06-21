#!/bin/bash
# Bluefin Service Docker Network Diagnostics Script
# This script helps diagnose connection issues between the trading bot and bluefin-service

set -e

echo "🔍 BLUEFIN SERVICE NETWORK DIAGNOSTICS"
echo "=========================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running or not accessible"
    exit 1
fi

echo "✅ Docker is running"

# Check if containers are running
echo ""
echo "📦 CONTAINER STATUS:"
echo "--------------------"

if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(bluefin-service|ai-trading-bot)"; then
    echo "✅ Relevant containers found"
else
    echo "❌ No bluefin-service or ai-trading-bot containers running"
    echo "Run: docker-compose up -d"
    exit 1
fi

# Check bluefin-service specifically
echo ""
echo "🔧 BLUEFIN SERVICE DETAILS:"
echo "---------------------------"

if docker ps | grep -q bluefin-service; then
    echo "✅ bluefin-service container is running"

    # Check the service logs for startup
    echo ""
    echo "📋 Recent bluefin-service logs:"
    echo "-------------------------------"
    docker logs --tail 10 bluefin-service 2>/dev/null || echo "❌ Could not read bluefin-service logs"

    # Check if service is listening on port 8080
    echo ""
    echo "🔌 PORT BINDING CHECK:"
    echo "----------------------"

    if docker exec bluefin-service netstat -tlnp 2>/dev/null | grep :8080 || \
       docker exec bluefin-service ss -tlnp 2>/dev/null | grep :8080; then
        echo "✅ bluefin-service is listening on port 8080"
    else
        echo "❌ bluefin-service is NOT listening on port 8080"
        echo "🔍 Available ports:"
        docker exec bluefin-service netstat -tlnp 2>/dev/null || \
        docker exec bluefin-service ss -tlnp 2>/dev/null || \
        echo "Could not check ports"
    fi

    # Test internal connectivity
    echo ""
    echo "🌐 INTERNAL CONNECTIVITY TEST:"
    echo "-------------------------------"

    if docker exec bluefin-service curl -f --connect-timeout 5 http://localhost:8080/health 2>/dev/null; then
        echo "✅ bluefin-service responds to localhost:8080/health"
    else
        echo "❌ bluefin-service does NOT respond to localhost:8080/health"
    fi

    if docker exec bluefin-service curl -f --connect-timeout 5 http://0.0.0.0:8080/health 2>/dev/null; then
        echo "✅ bluefin-service responds to 0.0.0.0:8080/health"
    else
        echo "❌ bluefin-service does NOT respond to 0.0.0.0:8080/health"
    fi

else
    echo "❌ bluefin-service container is not running"
fi

# Test cross-container connectivity
echo ""
echo "🔗 CROSS-CONTAINER CONNECTIVITY:"
echo "---------------------------------"

if docker ps | grep -q ai-trading-bot; then
    echo "✅ ai-trading-bot container is running"

    # Test DNS resolution
    if docker exec ai-trading-bot nslookup bluefin-service 2>/dev/null; then
        echo "✅ DNS resolution: bluefin-service resolves correctly"
    else
        echo "❌ DNS resolution failed for bluefin-service"
    fi

    # Test connectivity from trading bot to bluefin-service
    if docker exec ai-trading-bot curl -f --connect-timeout 10 http://bluefin-service:8080/health 2>/dev/null; then
        echo "✅ ai-trading-bot can connect to bluefin-service:8080/health"
    else
        echo "❌ ai-trading-bot CANNOT connect to bluefin-service:8080/health"
        echo "🔍 This is the main issue causing connection problems"
    fi
else
    echo "⚠️  ai-trading-bot container is not running"
fi

# Check Docker network
echo ""
echo "🔌 DOCKER NETWORK ANALYSIS:"
echo "----------------------------"

NETWORK_NAME="trading-network"
if docker network ls | grep -q $NETWORK_NAME; then
    echo "✅ $NETWORK_NAME exists"

    echo ""
    echo "📋 Network containers:"
    docker network inspect $NETWORK_NAME --format '{{range .Containers}}{{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}' 2>/dev/null || echo "Could not inspect network"

else
    echo "❌ $NETWORK_NAME does not exist"
    echo "Run: docker-compose up to create the network"
fi

# Environment variable check
echo ""
echo "🔧 ENVIRONMENT VARIABLES:"
echo "-------------------------"

echo "bluefin-service environment:"
docker exec bluefin-service env | grep -E "(HOST|PORT|BLUEFIN)" 2>/dev/null || echo "Could not read environment"

echo ""
echo "ai-trading-bot bluefin config:"
docker exec ai-trading-bot env | grep -E "(BLUEFIN_SERVICE|EXCHANGE.*BLUEFIN)" 2>/dev/null || echo "Could not read environment"

# Recommendations
echo ""
echo "💡 RECOMMENDATIONS:"
echo "-------------------"

if docker exec bluefin-service curl -f --connect-timeout 5 http://localhost:8080/health >/dev/null 2>&1; then
    if ! docker exec ai-trading-bot curl -f --connect-timeout 10 http://bluefin-service:8080/health >/dev/null 2>&1; then
        echo "🔧 The bluefin-service is running but not accessible from other containers"
        echo "   - Check that HOST=0.0.0.0 in bluefin-service environment"
        echo "   - Restart the bluefin-service container"
        echo "   - Run: docker-compose restart bluefin-service"
    fi
else
    echo "🔧 The bluefin-service is not responding on port 8080"
    echo "   - Check bluefin-service logs: docker logs bluefin-service"
    echo "   - Verify the service started correctly"
    echo "   - Check for port conflicts"
fi

echo ""
echo "✅ Diagnostics complete!"
echo "To fix issues, run: docker-compose restart bluefin-service"
