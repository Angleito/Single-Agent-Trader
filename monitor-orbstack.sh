#!/bin/bash

# OrbStack Monitoring Script for AI Trading Bot
# Monitor the trading bot performance and VuManChu indicators

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

CONTAINER_NAME="ai-trading-bot-orbstack"
COMPOSE_FILE="docker-compose.orbstack.yml"

# Function to clear screen and show header
show_header() {
    clear
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  AI Trading Bot - OrbStack Monitor with VuManChu Indicators${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}📊 Container: $CONTAINER_NAME${NC}"
    echo -e "${CYAN}⏰ Last updated: $(date)${NC}"
    echo ""
}

# Function to check container status
check_container_status() {
    if docker ps --filter name=$CONTAINER_NAME --format '{{.Names}}' | grep -q $CONTAINER_NAME; then
        local status=$(docker inspect --format='{{.State.Status}}' $CONTAINER_NAME 2>/dev/null || echo "not found")
        local health=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "no health check")
        echo -e "${GREEN}✅ Status: $status${NC}"
        if [ "$health" != "no health check" ]; then
            echo -e "${GREEN}🏥 Health: $health${NC}"
        fi
        return 0
    else
        echo -e "${RED}❌ Container not running${NC}"
        return 1
    fi
}

# Function to show resource usage
show_resource_usage() {
    echo -e "${BLUE}📈 Resource Usage:${NC}"
    if docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" $CONTAINER_NAME 2>/dev/null; then
        echo ""
    else
        echo -e "${RED}❌ Unable to get resource stats${NC}"
    fi
}

# Function to show recent logs with VuManChu indicators
show_recent_logs() {
    echo -e "${BLUE}📋 Recent Logs (VuManChu & Trading):${NC}"
    echo -e "${CYAN}─────────────────────────────────────${NC}"
    
    # Show VuManChu indicator logs
    docker logs --tail 10 $CONTAINER_NAME 2>/dev/null | grep -E "(vumanchu|cipher|wavetrend|diamond|yellow_cross)" --color=always || true
    
    echo -e "${CYAN}─────────────────────────────────────${NC}"
    
    # Show general trading logs
    docker logs --tail 5 $CONTAINER_NAME 2>/dev/null | grep -E "(Loop|Trade|Signal|Price)" --color=always || true
}

# Function to show VuManChu indicator performance
show_indicator_performance() {
    echo -e "${BLUE}🎯 VuManChu Indicator Performance:${NC}"
    
    # Check if performance logs exist
    if docker exec $CONTAINER_NAME test -f /app/logs/performance.log 2>/dev/null; then
        echo -e "${GREEN}📊 Indicator calculation metrics:${NC}"
        docker exec $CONTAINER_NAME tail -5 /app/logs/performance.log 2>/dev/null | grep -E "(cipher|wavetrend|calculation)" || echo -e "${YELLOW}No recent indicator performance data${NC}"
    else
        echo -e "${YELLOW}⚠️  Performance log not found${NC}"
    fi
    
    # Show recent signal counts
    echo -e "${GREEN}🔢 Recent signal counts:${NC}"
    docker logs --tail 50 $CONTAINER_NAME 2>/dev/null | grep -o -E "(red_diamond|green_diamond|yellow_cross|moon_diamond|dump_diamond)" | sort | uniq -c | head -5 || echo -e "${YELLOW}No recent signals detected${NC}"
}

# Function to show trading performance
show_trading_performance() {
    echo -e "${BLUE}💰 Trading Performance:${NC}"
    
    # Get latest trading stats from logs
    local latest_performance=$(docker logs --tail 100 $CONTAINER_NAME 2>/dev/null | grep -E "(Paper account|P&L|ROI|equity)" | tail -3)
    
    if [ -n "$latest_performance" ]; then
        echo -e "${GREEN}$latest_performance${NC}"
    else
        echo -e "${YELLOW}⚠️  No recent trading performance data${NC}"
    fi
}

# Function to show interactive menu
show_menu() {
    echo ""
    echo -e "${BLUE}Choose an action:${NC}"
    echo -e "  ${GREEN}1)${NC} Refresh monitor"
    echo -e "  ${GREEN}2)${NC} View full logs"
    echo -e "  ${GREEN}3)${NC} View VuManChu indicator logs only"
    echo -e "  ${GREEN}4)${NC} View trading signals"
    echo -e "  ${GREEN}5)${NC} Restart container"
    echo -e "  ${GREEN}6)${NC} Stop monitoring"
    echo -e "  ${GREEN}7)${NC} Container shell"
    echo -e "  ${GREEN}8)${NC} Export logs"
    echo -e "  ${GREEN}q)${NC} Quit"
    echo ""
    echo -ne "${CYAN}Enter choice: ${NC}"
}

# Function to export logs
export_logs() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local export_dir="./logs/exports"
    
    mkdir -p $export_dir
    
    echo -e "${BLUE}📤 Exporting logs...${NC}"
    
    # Export container logs
    docker logs $CONTAINER_NAME > "$export_dir/container_logs_$timestamp.log" 2>&1
    
    # Export VuManChu specific logs
    docker logs $CONTAINER_NAME 2>/dev/null | grep -E "(vumanchu|cipher|wavetrend)" > "$export_dir/vumanchu_logs_$timestamp.log" || true
    
    # Export performance data if available
    docker exec $CONTAINER_NAME cat /app/logs/performance.log > "$export_dir/performance_$timestamp.log" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Logs exported to $export_dir/${NC}"
    echo -e "   Container logs: ${YELLOW}container_logs_$timestamp.log${NC}"
    echo -e "   VuManChu logs: ${YELLOW}vumanchu_logs_$timestamp.log${NC}"
    echo -e "   Performance: ${YELLOW}performance_$timestamp.log${NC}"
}

# Main monitoring loop
main() {
    local auto_refresh=false
    
    # Check if auto-refresh is requested
    if [ "$1" = "--auto" ] || [ "$1" = "-a" ]; then
        auto_refresh=true
        echo -e "${BLUE}🔄 Auto-refresh mode enabled (Ctrl+C to exit)${NC}"
        sleep 2
    fi
    
    while true; do
        show_header
        
        if check_container_status; then
            echo ""
            show_resource_usage
            echo ""
            show_indicator_performance
            echo ""
            show_trading_performance
            echo ""
            show_recent_logs
        else
            echo -e "${RED}❌ Container is not running. Start with: docker-compose -f $COMPOSE_FILE up -d${NC}"
        fi
        
        if $auto_refresh; then
            echo -e "${CYAN}🔄 Auto-refreshing in 30 seconds... (Ctrl+C to exit)${NC}"
            sleep 30
            continue
        fi
        
        show_menu
        read -r choice
        
        case $choice in
            1)
                continue
                ;;
            2)
                echo -e "${BLUE}📋 Full logs (press 'q' to exit):${NC}"
                docker logs -f $CONTAINER_NAME
                ;;
            3)
                echo -e "${BLUE}🎯 VuManChu indicator logs (press 'q' to exit):${NC}"
                docker logs -f $CONTAINER_NAME | grep -E "(vumanchu|cipher|wavetrend|diamond|yellow_cross)" --line-buffered
                ;;
            4)
                echo -e "${BLUE}📡 Trading signals (press 'q' to exit):${NC}"
                docker logs -f $CONTAINER_NAME | grep -E "(signal|diamond|cross|Trade|Buy|Sell)" --line-buffered
                ;;
            5)
                echo -e "${YELLOW}🔄 Restarting container...${NC}"
                docker-compose -f $COMPOSE_FILE restart ai-trading-bot
                echo -e "${GREEN}✅ Container restarted${NC}"
                sleep 3
                ;;
            6)
                echo -e "${YELLOW}⏹️  Stopping container...${NC}"
                docker-compose -f $COMPOSE_FILE down
                echo -e "${GREEN}✅ Container stopped${NC}"
                break
                ;;
            7)
                echo -e "${BLUE}🐚 Opening container shell...${NC}"
                docker exec -it $CONTAINER_NAME bash || echo -e "${RED}❌ Unable to open shell${NC}"
                ;;
            8)
                export_logs
                echo -e "${CYAN}Press Enter to continue...${NC}"
                read
                ;;
            q|Q)
                echo -e "${GREEN}👋 Goodbye!${NC}"
                break
                ;;
            *)
                echo -e "${RED}❌ Invalid choice${NC}"
                sleep 1
                ;;
        esac
    done
}

# Check if help is requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo -e "${BLUE}AI Trading Bot OrbStack Monitor${NC}"
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo -e "  $0                 Interactive monitoring"
    echo -e "  $0 --auto         Auto-refresh mode"
    echo -e "  $0 --help         Show this help"
    echo ""
    echo -e "${GREEN}Features:${NC}"
    echo -e "  • Container status and health monitoring"
    echo -e "  • Resource usage tracking"
    echo -e "  • VuManChu indicator performance metrics"
    echo -e "  • Trading performance monitoring"
    echo -e "  • Real-time log viewing"
    echo -e "  • Log export functionality"
    exit 0
fi

# Start monitoring
main "$@"