#!/bin/bash

# Test OrbStack deployment script for AI Trading Bot with VuManChu Indicators
# This script tests the deployment without running the full bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Testing OrbStack deployment for AI Trading Bot with VuManChu Indicators${NC}"

# Check if OrbStack is running
if ! docker system info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker/OrbStack is not running. Please start OrbStack first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ OrbStack is running${NC}"

# Run the integration tests first
echo -e "${BLUE}🔍 Running integration tests...${NC}"
python3 test_vumanchu_integration.py

echo -e "${BLUE}🔨 Building test image...${NC}"

# Create a test Dockerfile that includes dependency installation
cat > Dockerfile.test << 'EOF'
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    setuptools \
    wheel \
    "numpy<2.0" \
    pandas \
    pandas-ta \
    pydantic \
    pydantic-settings \
    python-dotenv

# Copy bot code
COPY bot/ ./bot/

# Test the VuManChu indicators
RUN python3 -c "
try:
    from bot.indicators.vumanchu import VuManChuIndicators
    print('✅ VuManChuIndicators imported successfully')
    
    # Test instantiation
    indicators = VuManChuIndicators()
    print('✅ VuManChuIndicators instantiated successfully')
    
    print('🎯 VuManChu indicators are working in Docker!')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Create a simple health check
RUN echo '#!/bin/bash\necho "VuManChu indicators test passed"' > /app/healthcheck.sh \
    && chmod +x /app/healthcheck.sh

CMD ["python3", "-c", "print('VuManChu indicators ready for deployment!')"]
EOF

# Build test image
docker build -f Dockerfile.test -t ai-trading-bot:test .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Test build completed successfully${NC}"
else
    echo -e "${RED}❌ Test build failed${NC}"
    exit 1
fi

# Run test container
echo -e "${BLUE}🚀 Running test container...${NC}"
docker run --rm ai-trading-bot:test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Test container ran successfully${NC}"
else
    echo -e "${RED}❌ Test container failed${NC}"
    exit 1
fi

# Clean up test artifacts
echo -e "${BLUE}🧹 Cleaning up test artifacts...${NC}"
rm -f Dockerfile.test
docker rmi ai-trading-bot:test >/dev/null 2>&1 || true

echo -e "${GREEN}🎉 OrbStack deployment test completed successfully!${NC}"
echo -e "${BLUE}📝 Next steps:${NC}"
echo -e "   1. Run: ${YELLOW}./deploy-orbstack.sh${NC}"
echo -e "   2. Monitor: ${YELLOW}./monitor-orbstack.sh${NC}"
echo -e "   3. View logs: ${YELLOW}docker logs -f ai-trading-bot-orbstack${NC}"

echo -e "${GREEN}🚀 VuManChu indicators are ready for OrbStack deployment!${NC}"