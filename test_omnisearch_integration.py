#!/usr/bin/env python3
"""
Test script to demonstrate OmniSearch integration with the LLM agent.

This script shows how the enhanced LLM agent uses OmniSearch for 
financial market intelligence and sentiment analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced components
from bot.config import settings
from bot.mcp.omnisearch_client import OmniSearchClient
from bot.strategy.llm_agent import LLMAgent
from bot.types import MarketState, Position, IndicatorData, Candle


async def create_mock_market_state() -> MarketState:
    """Create a mock market state for testing."""
    # Create mock OHLCV data
    mock_candles = []
    base_price = 50000.0
    for i in range(10):
        candle = Candle(
            timestamp=datetime.now(timezone.utc),
            open=base_price + i * 100,
            high=base_price + i * 100 + 200,
            low=base_price + i * 100 - 100,
            close=base_price + i * 100 + 50,
            volume=1000.0 + i * 50
        )
        mock_candles.append(candle)
    
    # Create mock indicators
    indicators = IndicatorData(
        rsi=65.0,
        ema_fast=50050.0,
        ema_slow=49950.0,
        cipher_a_dot=1.2,
        cipher_b_wave=15.5,
        cipher_b_money_flow=60.0,
        usdt_dominance=8.5,
        usdc_dominance=2.1,
        stablecoin_dominance=10.6,
        dominance_trend=-0.2,
        dominance_rsi=45.0,
        market_sentiment="BULLISH"
    )
    
    # Create mock position
    position = Position(
        side="FLAT",
        size=0.0,
        entry_price=0.0,
        unrealized_pnl=Decimal("0.0")
    )
    
    return MarketState(
        symbol="BTC-USD",
        interval="5m",
        current_price=Decimal("50125.0"),
        ohlcv_data=mock_candles,
        indicators=indicators,
        current_position=position
    )


async def test_omnisearch_integration():
    """Test the OmniSearch integration with LLM agent."""
    print("🔍 Testing OmniSearch Integration with LLM Agent")
    print("=" * 60)
    
    # Create OmniSearch client (will use fallback mode if service unavailable)
    omnisearch_client = OmniSearchClient(
        server_url="http://localhost:8766",  # Default from config
        api_key=None,  # Will use fallback mode
        enable_cache=True,
        cache_ttl=300
    )
    
    # Test basic client functionality
    print("\n1. Testing OmniSearch Client")
    print("-" * 30)
    
    try:
        connected = await omnisearch_client.connect()
        print(f"   Connection Status: {'✅ Connected' if connected else '⚠️ Fallback mode'}")
        
        # Test health check
        health = await omnisearch_client.health_check()
        print(f"   Client Health: {health}")
        
    except Exception as e:
        print(f"   Connection Error: {e}")
        print("   Continuing with fallback mode...")
    
    # Create LLM agent with OmniSearch integration
    print("\n2. Creating LLM Agent with OmniSearch")
    print("-" * 40)
    
    try:
        llm_agent = LLMAgent(
            model_provider="openai",
            model_name="gpt-3.5-turbo",  # Use a more accessible model for testing
            omnisearch_client=omnisearch_client
        )
        
        # Get agent status
        status = llm_agent.get_status()
        print(f"   LLM Available: {'✅' if status['llm_available'] else '❌'}")
        print(f"   OmniSearch Enabled: {'✅' if status['omnisearch_enabled'] else '❌'}")
        print(f"   OmniSearch Client Available: {'✅' if status['omnisearch_client_available'] else '❌'}")
        
    except Exception as e:
        print(f"   LLM Agent Error: {e}")
        return
    
    # Test market analysis with OmniSearch context
    print("\n3. Testing Market Analysis with Financial Context")
    print("-" * 50)
    
    try:
        # Create mock market state
        market_state = await create_mock_market_state()
        
        print("   Market State:")
        print(f"     Symbol: {market_state.symbol}")
        print(f"     Price: ${market_state.current_price}")
        print(f"     RSI: {market_state.indicators.rsi}")
        print(f"     Position: {market_state.current_position.side}")
        
        # Analyze market with OmniSearch integration
        print("\n   Analyzing market with OmniSearch context...")
        
        # Note: This will likely use fallback mode unless OmniSearch service is running
        trade_action = await llm_agent.analyze_market(market_state)
        
        print(f"\n   🎯 Trade Decision:")
        print(f"     Action: {trade_action.action}")
        print(f"     Size: {trade_action.size_pct}%")
        print(f"     Take Profit: {trade_action.take_profit_pct}%")
        print(f"     Stop Loss: {trade_action.stop_loss_pct}%")
        print(f"     Leverage: {trade_action.leverage}x")
        print(f"     Rationale: {trade_action.rationale}")
        
    except Exception as e:
        print(f"   Market Analysis Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test financial context retrieval directly
    print("\n4. Testing Financial Context Retrieval")
    print("-" * 40)
    
    try:
        market_state = await create_mock_market_state()
        
        print("   Attempting to retrieve financial context...")
        
        # This will test our _get_financial_context method
        if hasattr(llm_agent, '_get_financial_context'):
            financial_context = await llm_agent._get_financial_context(market_state)
            
            print(f"\n   📊 Financial Context Preview:")
            # Show first 500 characters of context
            preview = financial_context[:500] + "..." if len(financial_context) > 500 else financial_context
            print(f"     {preview}")
        else:
            print("   ⚠️ _get_financial_context method not found")
            
    except Exception as e:
        print(f"   Financial Context Error: {e}")
    
    # Cleanup
    print("\n5. Cleanup")
    print("-" * 10)
    
    try:
        await omnisearch_client.disconnect()
        print("   ✅ OmniSearch client disconnected")
    except Exception as e:
        print(f"   Disconnect Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 OmniSearch Integration Test Complete")


async def main():
    """Main test function."""
    try:
        await test_omnisearch_integration()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())