#!/bin/bash

# Dashboard Deployment Script
# This script helps you start the dashboard in different modes

set -e

echo "🚀 Dashboard Deployment Helper"
echo "==============================="

# Check if we're in the dashboard directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the dashboard directory"
    exit 1
fi

# Function to check if main trading system is running
check_trading_system() {
    if docker network inspect trading-network >/dev/null 2>&1; then
        echo "✅ Trading network detected"
        return 0
    else
        echo "⚠️  Trading network not found"
        return 1
    fi
}

# Function to check for port conflicts
check_port_conflicts() {
    local ports=("3000" "8000" "8080")
    local conflicts=()
    
    for port in "${ports[@]}"; do
        if lsof -i :$port >/dev/null 2>&1; then
            conflicts+=($port)
        fi
    done
    
    if [ ${#conflicts[@]} -gt 0 ]; then
        echo "⚠️  Port conflicts detected on: ${conflicts[*]}"
        echo "   Please stop services using these ports or choose a different mode"
        return 1
    fi
    return 0
}

# Function to display menu
show_menu() {
    echo ""
    echo "Select deployment mode:"
    echo "1) Standalone Dashboard (recommended for testing)"
    echo "2) Compatible with Trading System (can connect to main system)"
    echo "3) Pure Standalone (isolated, no trading network)"
    echo "4) Production with Nginx"
    echo "5) Check system status"
    echo "6) Stop all dashboard services"
    echo "7) Exit"
    echo ""
}

# Function to start standalone mode
start_standalone() {
    echo "🔧 Starting dashboard in standalone mode..."
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    docker-compose up
}

# Function to start compatible mode
start_compatible() {
    echo "🔧 Starting dashboard in compatible mode..."
    if check_trading_system; then
        echo "   Will connect to existing trading network"
    else
        echo "   Will create trading network (compatible with main system)"
    fi
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    docker-compose up
}

# Function to start pure standalone mode
start_pure_standalone() {
    echo "🔧 Starting dashboard in pure standalone mode..."
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    docker-compose -f docker-compose.yml -f docker-compose.standalone.yml up
}

# Function to start production mode
start_production() {
    echo "🔧 Starting dashboard in production mode..."
    echo "   Nginx Proxy: http://localhost:8080"
    echo "   Direct Frontend: http://localhost:3001"
    echo "   Backend: http://localhost:8000"
    docker-compose --profile production up
}

# Function to check system status
check_status() {
    echo "📊 System Status Check"
    echo "====================="
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not installed"
        return 1
    fi
    echo "✅ Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose not installed"
        return 1
    fi
    echo "✅ Docker Compose is available"
    
    # Check networks
    echo ""
    echo "Docker Networks:"
    if docker network inspect dashboard-network >/dev/null 2>&1; then
        echo "✅ dashboard-network exists"
    else
        echo "⚠️  dashboard-network not found"
    fi
    
    if check_trading_system; then
        echo "✅ trading-network exists (main system running)"
    else
        echo "⚠️  trading-network not found (main system not running)"
    fi
    
    # Check running containers
    echo ""
    echo "Running Dashboard Containers:"
    docker ps --filter "name=dashboard" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No dashboard containers running"
    
    # Check ports
    echo ""
    echo "Port Status:"
    for port in 3000 8000 8080; do
        if lsof -i :$port >/dev/null 2>&1; then
            echo "❌ Port $port is in use"
        else
            echo "✅ Port $port is available"
        fi
    done
}

# Function to stop all services
stop_services() {
    echo "🛑 Stopping all dashboard services..."
    
    # Stop main compose
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    fi
    
    # Stop standalone compose
    if [ -f "docker-compose.standalone.yml" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.standalone.yml down 2>/dev/null || true
    fi
    
    # Stop production services
    docker-compose --profile production down 2>/dev/null || true
    
    # Clean up any orphaned containers
    docker container prune -f
    
    echo "✅ All dashboard services stopped"
}

# Main script logic
main() {
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                if check_port_conflicts; then
                    start_standalone
                else
                    echo "❌ Cannot start due to port conflicts"
                fi
                break
                ;;
            2)
                if check_port_conflicts; then
                    start_compatible
                else
                    echo "❌ Cannot start due to port conflicts"
                fi
                break
                ;;
            3)
                if check_port_conflicts; then
                    start_pure_standalone
                else
                    echo "❌ Cannot start due to port conflicts"
                fi
                break
                ;;
            4)
                if check_port_conflicts; then
                    start_production
                else
                    echo "❌ Cannot start due to port conflicts"
                fi
                break
                ;;
            5)
                check_status
                echo ""
                read -p "Press Enter to continue..."
                ;;
            6)
                stop_services
                echo ""
                read -p "Press Enter to continue..."
                ;;
            7)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice. Please enter 1-7."
                ;;
        esac
    done
}

# Run main function
main