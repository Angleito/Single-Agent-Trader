#!/bin/bash
# Test script for Bluefin Docker integration

echo "=== Bluefin Docker Integration Test ==="
echo

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Creating from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your Bluefin credentials:"
    echo "   - BLUEFIN_PRIVATE_KEY"
    echo "   - OPENAI_API_KEY"
    echo "   - Set EXCHANGE_TYPE=bluefin"
    exit 1
fi

# Check if required environment variables are set
source .env
if [ -z "$BLUEFIN_PRIVATE_KEY" ] || [ "$BLUEFIN_PRIVATE_KEY" == "your_sui_wallet_private_key_here" ]; then
    echo "❌ BLUEFIN_PRIVATE_KEY not configured in .env"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" == "your_openai_api_key_here" ]; then
    echo "❌ OPENAI_API_KEY not configured in .env"
    exit 1
fi

# Set environment for Bluefin
export EXCHANGE_TYPE=bluefin

echo "✅ Environment variables configured"
echo "   Exchange Type: $EXCHANGE_TYPE"
echo "   Network: ${BLUEFIN_NETWORK:-mainnet}"
echo

# Build Docker image with Bluefin support
echo "Building Docker image with Bluefin support..."
docker-compose build --build-arg EXCHANGE_TYPE=bluefin ai-trading-bot

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"
echo

# Test 1: Basic container health check
echo "Test 1: Starting container with dry-run mode..."
docker-compose run --rm \
    -e EXCHANGE__EXCHANGE_TYPE=bluefin \
    -e SYSTEM__DRY_RUN=true \
    -e TRADING__SYMBOL=ETH-PERP \
    ai-trading-bot python -m bot.main live --dry-run --symbol ETH-PERP --interval 5m &

CONTAINER_PID=$!
sleep 30

# Check if container is still running
if ps -p $CONTAINER_PID > /dev/null; then
    echo "✅ Container is running"
    # Give it more time to initialize
    sleep 20
else
    echo "❌ Container exited unexpectedly"
    exit 1
fi

# Kill the test container
kill $CONTAINER_PID 2>/dev/null

echo
echo "Test 2: Testing Bluefin connection..."
docker-compose run --rm \
    -e EXCHANGE__EXCHANGE_TYPE=bluefin \
    -e SYSTEM__DRY_RUN=true \
    ai-trading-bot python -c "
import asyncio
from bot.exchange.factory import create_exchange
from bot.config import settings

async def test_connection():
    try:
        exchange = create_exchange(exchange_type='bluefin', dry_run=True)
        connected = await exchange.connect()
        print(f'Connection successful: {connected}')
        
        if connected:
            status = exchange.get_connection_status()
            print(f'Exchange: {status[\"exchange\"]}')
            print(f'Network: {status[\"network\"]}')
            print(f'Trading Mode: {status[\"trading_mode\"]}')
            print(f'Blockchain: {status[\"blockchain\"]}')
            
            # Test getting account balance
            balance = await exchange.get_account_balance()
            print(f'Account Balance: \${balance}')
            
            # Test symbol conversion
            from bot.exchange.bluefin import BluefinClient
            client = BluefinClient(dry_run=True)
            perp_symbol = client._convert_symbol('ETH-USD')
            print(f'Symbol Conversion: ETH-USD -> {perp_symbol}')
            
            await exchange.disconnect()
            return True
    except Exception as e:
        print(f'Error: {e}')
        return False

success = asyncio.run(test_connection())
exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo "✅ Bluefin connection test passed"
else
    echo "❌ Bluefin connection test failed"
    exit 1
fi

echo
echo "Test 3: Testing perpetual trading functionality..."
docker-compose run --rm \
    -e EXCHANGE__EXCHANGE_TYPE=bluefin \
    -e SYSTEM__DRY_RUN=true \
    -e TRADING__SYMBOL=BTC-PERP \
    -e TRADING__LEVERAGE=5 \
    ai-trading-bot python -c "
import asyncio
from decimal import Decimal
from bot.exchange.factory import create_exchange
from bot.types import TradeAction

async def test_perpetual_trading():
    try:
        exchange = create_exchange(exchange_type='bluefin', dry_run=True)
        await exchange.connect()
        
        # Test perpetual futures support
        print(f'Supports Futures: {exchange.supports_futures}')
        print(f'Is Decentralized: {exchange.is_decentralized}')
        
        # Test placing a paper trade
        trade_action = TradeAction(
            action='LONG',
            size_pct=10.0,
            leverage=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            confidence=0.85,
            reasoning='Test trade'
        )
        
        current_price = Decimal('45000')  # Mock BTC price
        order = await exchange.execute_trade_action(
            trade_action, 
            'BTC-USD',  # Will be converted to BTC-PERP
            current_price
        )
        
        if order:
            print(f'Order placed: {order.id}')
            print(f'Symbol: {order.symbol}')
            print(f'Side: {order.side}')
            print(f'Quantity: {order.quantity}')
            print(f'Status: {order.status}')
            print('✅ Perpetual trading test passed')
        else:
            print('❌ Failed to place order')
            
        await exchange.disconnect()
        return order is not None
        
    except Exception as e:
        print(f'Error in perpetual trading test: {e}')
        import traceback
        traceback.print_exc()
        return False

success = asyncio.run(test_perpetual_trading())
exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo "✅ Perpetual trading functionality test passed"
else
    echo "❌ Perpetual trading functionality test failed"
    exit 1
fi

echo
echo "=== All Bluefin Docker tests passed! ==="
echo
echo "To run the bot with Bluefin in paper trading mode:"
echo "  docker-compose run -e EXCHANGE__EXCHANGE_TYPE=bluefin ai-trading-bot"
echo
echo "To run in live mode (CAREFUL - REAL MONEY):"
echo "  1. Set SYSTEM__DRY_RUN=false in docker-compose.yml"
echo "  2. docker-compose up ai-trading-bot"