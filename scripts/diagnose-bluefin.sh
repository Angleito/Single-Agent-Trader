#!/bin/bash

# Bluefin Integration Diagnostic Script
# This script helps diagnose and fix common Bluefin integration issues

echo "🔍 Bluefin Integration Diagnostic Tool"
echo "======================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker service status
check_docker_service() {
    echo ""
    echo "📋 Checking Docker Service Status..."

    if ! command_exists docker; then
        echo "❌ Docker is not installed"
        return 1
    fi

    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker daemon is not running"
        return 1
    fi

    echo "✅ Docker is running"

    # Check if bluefin-service container exists
    if docker ps -a --format "table {{.Names}}" | grep -q "bluefin-service"; then
        local status=$(docker inspect --format='{{.State.Status}}' bluefin-service 2>/dev/null)
        echo "📦 Bluefin service container status: $status"

        if [ "$status" = "running" ]; then
            echo "✅ Bluefin service is running"

            # Check health
            local health=$(docker inspect --format='{{.State.Health.Status}}' bluefin-service 2>/dev/null)
            if [ "$health" = "healthy" ]; then
                echo "✅ Bluefin service is healthy"
            else
                echo "⚠️ Bluefin service health status: $health"
            fi
        else
            echo "❌ Bluefin service is not running"
        fi
    else
        echo "⚠️ Bluefin service container not found"
    fi
}

# Function to check network connectivity
check_network_connectivity() {
    echo ""
    echo "🌐 Checking Network Connectivity..."

    # Check if containers can communicate
    if docker ps --format "table {{.Names}}" | grep -q "ai-trading-bot"; then
        echo "📡 Testing container-to-container communication..."

        # Try to reach bluefin-service from ai-trading-bot
        if docker exec ai-trading-bot curl -f --connect-timeout 5 http://bluefin-service:8080/health >/dev/null 2>&1; then
            echo "✅ AI trading bot can reach Bluefin service"
        else
            echo "❌ AI trading bot cannot reach Bluefin service"
            echo "  Checking network configuration..."

            # Check if both containers are on the same network
            local ai_networks=$(docker inspect ai-trading-bot --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' 2>/dev/null)
            local bluefin_networks=$(docker inspect bluefin-service --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' 2>/dev/null)

            echo "  AI Trading Bot networks: $ai_networks"
            echo "  Bluefin Service networks: $bluefin_networks"
        fi
    fi

    # Check external connectivity from host
    echo "🌍 Testing external connectivity..."
    if curl -f --connect-timeout 5 http://localhost:8081/health >/dev/null 2>&1; then
        echo "✅ Bluefin service is accessible from host"
    else
        echo "❌ Bluefin service is not accessible from host"
    fi
}

# Function to check environment configuration
check_environment_config() {
    echo ""
    echo "⚙️  Checking Environment Configuration..."

    if [ -f ".env" ]; then
        echo "✅ .env file found"

        # Check critical environment variables
        if grep -q "EXCHANGE__BLUEFIN_PRIVATE_KEY" .env; then
            local key_value=$(grep "EXCHANGE__BLUEFIN_PRIVATE_KEY" .env | cut -d'=' -f2)
            if [ -n "$key_value" ] && [ "$key_value" != "your_bluefin_private_key_here" ]; then
                if [ "$key_value" = "dummy_key_for_testing" ]; then
                    echo "⚠️  Using dummy private key (development mode)"
                else
                    echo "✅ Bluefin private key is configured"
                fi
            else
                echo "❌ Bluefin private key is not set"
            fi
        else
            echo "❌ EXCHANGE__BLUEFIN_PRIVATE_KEY not found in .env"
        fi

        if grep -q "EXCHANGE__EXCHANGE_TYPE=bluefin" .env; then
            echo "✅ Exchange type is set to Bluefin"
        else
            echo "⚠️  Exchange type is not set to Bluefin"
        fi

        if grep -q "LLM__OPENAI_API_KEY" .env; then
            local llm_key=$(grep "LLM__OPENAI_API_KEY" .env | cut -d'=' -f2)
            if [ -n "$llm_key" ] && [ "$llm_key" != "your_openai_api_key_here" ]; then
                echo "✅ OpenAI API key is configured"
            else
                echo "❌ OpenAI API key is not set"
            fi
        else
            echo "❌ LLM__OPENAI_API_KEY not found in .env"
        fi
    else
        echo "❌ .env file not found"
        echo "   Please copy .env.bluefin to .env and configure it"
    fi
}

# Function to check logs for common issues
check_logs() {
    echo ""
    echo "📝 Checking Recent Logs..."

    if docker ps --format "table {{.Names}}" | grep -q "bluefin-service"; then
        echo "🔍 Recent Bluefin service logs:"
        docker logs --tail=10 bluefin-service 2>&1 | while read line; do
            if echo "$line" | grep -q -E "(ERROR|CRITICAL|Failed|Exception)"; then
                echo "❌ $line"
            elif echo "$line" | grep -q -E "(WARNING|Warning)"; then
                echo "⚠️ $line"
            elif echo "$line" | grep -q -E "(INFO|Starting|✅)"; then
                echo "ℹ️ $line"
            else
                echo "   $line"
            fi
        done
    fi

    if docker ps --format "table {{.Names}}" | grep -q "ai-trading-bot"; then
        echo ""
        echo "🔍 Recent AI trading bot logs (Bluefin related):"
        docker logs --tail=20 ai-trading-bot 2>&1 | grep -i bluefin | tail -5 | while read line; do
            if echo "$line" | grep -q -E "(ERROR|CRITICAL|Failed|Exception)"; then
                echo "❌ $line"
            elif echo "$line" | grep -q -E "(WARNING|Warning)"; then
                echo "⚠️ $line"
            else
                echo "ℹ️ $line"
            fi
        done
    fi
}

# Function to provide recommendations
provide_recommendations() {
    echo ""
    echo "💡 Recommendations:"
    echo "=================="

    echo ""
    echo "Quick Fixes:"
    echo "1. 🔧 Use the fixed Bluefin configuration:"
    echo "   docker-compose -f docker-compose.bluefin-fixed.yml up -d"
    echo ""
    echo "2. 📋 Copy Bluefin environment template:"
    echo "   cp .env.bluefin .env"
    echo ""
    echo "3. 🔑 For development/testing, the dummy key should work automatically"
    echo "   For real trading, add your actual Sui private key to .env"
    echo ""
    echo "4. 🌐 Restart services with proper networking:"
    echo "   docker-compose down && docker-compose up -d"
    echo ""
    echo "5. 📊 Monitor logs for issues:"
    echo "   docker-compose logs -f bluefin-service"
    echo ""
    echo "Configuration Profiles:"
    echo "• Development/Testing: Use docker-compose.bluefin-fixed.yml with dummy key"
    echo "• Coinbase Only: Use docker-compose.coinbase.yml"
    echo "• Production Bluefin: Configure real keys in .env and use main docker-compose.yml"
}

# Main execution
echo "🚀 Starting Bluefin integration diagnosis..."

check_docker_service
check_network_connectivity
check_environment_config
check_logs
provide_recommendations

echo ""
echo "✅ Diagnosis complete!"
echo ""
echo "📞 For additional help:"
echo "• Check the integration guide: docs/bluefin_integration.md"
echo "• Review Discord community: https://discord.gg/sui"
echo "• Report issues: https://github.com/your-repo/issues"
