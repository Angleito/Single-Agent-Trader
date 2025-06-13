#!/bin/bash

# OrbStack Deployment Script for AI Trading Bot with VuManChu Indicators
# This script deploys the trading bot to OrbStack with the new indicators

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.orbstack.yml"
SERVICE_NAME="ai-trading-bot"
CONTAINER_NAME="ai-trading-bot-orbstack"

echo -e "${BLUE}🚀 Starting OrbStack deployment of AI Trading Bot with VuManChu Indicators${NC}"

# Check if OrbStack is running
if ! docker system info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker/OrbStack is not running. Please start OrbStack first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ OrbStack is running${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
        echo -e "${YELLOW}📝 Please edit .env with your API keys before continuing${NC}"
        echo -e "${YELLOW}   Press Enter when ready...${NC}"
        read
    else
        echo -e "${RED}❌ .env.example not found. Please create .env manually${NC}"
        exit 1
    fi
fi

# Validate environment variables
echo -e "${BLUE}🔍 Validating environment configuration...${NC}"
source .env

if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}❌ No LLM API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env${NC}"
    exit 1
fi

if [ -z "$CB_API_KEY" ] || [ -z "$CB_API_SECRET" ]; then
    echo -e "${YELLOW}⚠️  Coinbase API credentials not found. Bot will run in simulation mode only.${NC}"
fi

echo -e "${GREEN}✅ Environment configuration validated${NC}"

# Create required directories
echo -e "${BLUE}📁 Creating required directories...${NC}"
mkdir -p logs data config
chmod 755 logs data

# Stop existing containers
echo -e "${BLUE}🛑 Stopping existing containers...${NC}"
docker-compose -f $COMPOSE_FILE down 2>/dev/null || true

# Remove old container if exists
if docker ps -a --format 'table {{.Names}}' | grep -q $CONTAINER_NAME; then
    echo -e "${YELLOW}🗑️  Removing existing container: $CONTAINER_NAME${NC}"
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
fi

# Build the image
echo -e "${BLUE}🔨 Building AI Trading Bot image with VuManChu indicators...${NC}"
docker-compose -f $COMPOSE_FILE build --no-cache $SERVICE_NAME

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build completed successfully${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Start the service
echo -e "${BLUE}🚀 Starting AI Trading Bot in OrbStack...${NC}"
docker-compose -f $COMPOSE_FILE up -d $SERVICE_NAME

# Wait for container to be ready
echo -e "${BLUE}⏳ Waiting for container to be ready...${NC}"
sleep 10

# Check if container is running
if docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -q $CONTAINER_NAME; then
    echo -e "${GREEN}✅ Container is running successfully${NC}"
    
    # Show container status
    echo -e "${BLUE}📊 Container Status:${NC}"
    docker ps --filter name=$CONTAINER_NAME --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    
    # Show recent logs
    echo -e "${BLUE}📋 Recent logs (last 20 lines):${NC}"
    docker logs --tail 20 $CONTAINER_NAME
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}📝 Useful commands:${NC}"
    echo -e "   View logs: ${YELLOW}docker logs -f $CONTAINER_NAME${NC}"
    echo -e "   Stop bot:  ${YELLOW}docker-compose -f $COMPOSE_FILE down${NC}"
    echo -e "   Restart:   ${YELLOW}docker-compose -f $COMPOSE_FILE restart $SERVICE_NAME${NC}"
    echo -e "   Shell:     ${YELLOW}docker exec -it $CONTAINER_NAME bash${NC}"
    echo -e "   Monitor:   ${YELLOW}./monitor-orbstack.sh${NC}"
    
else
    echo -e "${RED}❌ Container failed to start. Checking logs...${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Optional: Start monitoring services
echo -e "${BLUE}🔍 Would you like to start monitoring services? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}🚀 Starting monitoring services...${NC}"
    docker-compose -f $COMPOSE_FILE --profile monitoring up -d
    echo -e "${GREEN}✅ Monitoring services started${NC}"
    echo -e "   Performance metrics: ${YELLOW}http://localhost:9100${NC}"
    echo -e "   Log aggregator: ${YELLOW}http://localhost:2020${NC}"
fi

echo -e "${GREEN}🎯 OrbStack deployment complete! Bot is running with VuManChu Cipher A & B indicators.${NC}"