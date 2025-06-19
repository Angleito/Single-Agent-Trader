#!/bin/bash
# Docker Network Validation Script
# Validates Docker network configuration and service connectivity

set -e

echo "🔍 Docker Network Configuration Validation"
echo "=========================================="

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

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_status "ERROR" "Docker is not running"
    exit 1
fi
print_status "SUCCESS" "Docker is running"

# Check if trading-network exists
if docker network ls | grep -q "trading-network"; then
    print_status "SUCCESS" "trading-network exists"
    
    # Get network details
    echo ""
    echo "🔍 Network Details:"
    docker network inspect trading-network --format '{{.Name}}: {{.Driver}} ({{range .IPAM.Config}}{{.Subnet}}{{end}})'
    
    # List services on the network
    echo ""
    echo "📡 Services on trading-network:"
    docker network inspect trading-network --format '{{range $key, $value := .Containers}}{{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}' | sort
    
else
    print_status "WARNING" "trading-network does not exist - will be created when services start"
fi

echo ""
echo "🐳 Container Status:"
echo "==================="

# Check bluefin-service
if docker ps --filter "name=bluefin-service" --filter "status=running" | grep -q bluefin-service; then
    print_status "SUCCESS" "bluefin-service is running"
    
    # Test bluefin-service health endpoint
    if docker exec bluefin-service curl -f http://localhost:8080/health >/dev/null 2>&1; then
        print_status "SUCCESS" "bluefin-service health check passed"
    else
        print_status "ERROR" "bluefin-service health check failed"
    fi
else
    print_status "WARNING" "bluefin-service is not running"
fi

# Check ai-trading-bot
if docker ps --filter "name=ai-trading-bot" --filter "status=running" | grep -q ai-trading-bot; then
    print_status "SUCCESS" "ai-trading-bot is running"
else
    print_status "WARNING" "ai-trading-bot is not running"
fi

echo ""
echo "🌐 Network Connectivity Tests:"
echo "=============================="

# Test connectivity from ai-trading-bot to bluefin-service
if docker ps --filter "name=ai-trading-bot" --filter "status=running" | grep -q ai-trading-bot; then
    if docker exec ai-trading-bot ping -c 1 bluefin-service >/dev/null 2>&1; then
        print_status "SUCCESS" "ai-trading-bot can ping bluefin-service"
    else
        print_status "ERROR" "ai-trading-bot cannot ping bluefin-service"
    fi
    
    # Test HTTP connectivity
    if docker exec ai-trading-bot curl -f http://bluefin-service:8080/health >/dev/null 2>&1; then
        print_status "SUCCESS" "ai-trading-bot can reach bluefin-service HTTP endpoint"
    else
        print_status "ERROR" "ai-trading-bot cannot reach bluefin-service HTTP endpoint"
    fi
else
    print_status "INFO" "Skipping connectivity tests - ai-trading-bot not running"
fi

echo ""
echo "📋 Network Configuration Summary:"
echo "================================="

# Check docker-compose.yml network config
if grep -q "trading-network:" docker-compose.yml; then
    print_status "SUCCESS" "docker-compose.yml uses trading-network"
else
    print_status "ERROR" "docker-compose.yml does not use trading-network"
fi

# Check docker-compose.bluefin.yml network config
if [ -f docker-compose.bluefin.yml ]; then
    if grep -q "trading-network:" docker-compose.bluefin.yml; then
        print_status "SUCCESS" "docker-compose.bluefin.yml uses trading-network"
    else
        print_status "ERROR" "docker-compose.bluefin.yml does not use trading-network"
    fi
    
    if grep -q "external: true" docker-compose.bluefin.yml; then
        print_status "SUCCESS" "docker-compose.bluefin.yml uses external network"
    else
        print_status "WARNING" "docker-compose.bluefin.yml does not use external network"
    fi
else
    print_status "INFO" "docker-compose.bluefin.yml not found"
fi

echo ""
echo "🔧 Recommendations:"
echo "=================="

if ! docker network ls | grep -q "trading-network"; then
    print_status "INFO" "Create the trading-network manually: docker network create trading-network"
fi

print_status "INFO" "Start services with: docker-compose up -d"
print_status "INFO" "For Bluefin: docker-compose -f docker-compose.bluefin.yml up -d"
print_status "INFO" "View logs with: docker-compose logs -f [service-name]"

echo ""
print_status "SUCCESS" "Network validation complete!"