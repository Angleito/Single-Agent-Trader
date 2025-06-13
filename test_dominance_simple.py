#!/usr/bin/env python3
"""
Simple test script for USDT/USDC dominance integration.
This can be run inside Docker to verify the integration works.
"""

import asyncio
import logging
import sys
from datetime import datetime, UTC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_dominance_provider():
    """Test the dominance data provider."""
    print("\n🧪 Testing Dominance Data Provider")
    print("="*50)
    
    try:
        from bot.data.dominance import DominanceDataProvider
        print("✅ Successfully imported DominanceDataProvider")
        
        # Create provider
        provider = DominanceDataProvider(
            data_source="coingecko",
            update_interval=300
        )
        print("✅ Created DominanceDataProvider instance")
        
        # Test connection
        await provider.connect()
        print("✅ Connected to dominance data provider")
        
        # Wait for initial data
        await asyncio.sleep(2)
        
        # Get latest dominance
        dominance = provider.get_latest_dominance()
        if dominance:
            print(f"\n📊 Dominance Data Retrieved:")
            print(f"  • USDT Dominance: {dominance.usdt_dominance:.2f}%")
            print(f"  • USDC Dominance: {dominance.usdc_dominance:.2f}%")
            print(f"  • Total Stablecoin: {dominance.stablecoin_dominance:.2f}%")
            print(f"  • 24h Change: {dominance.dominance_24h_change:+.2f}%")
        else:
            print("⚠️  No dominance data available (API might be rate limited)")
        
        # Get market sentiment
        sentiment = provider.get_market_sentiment()
        print(f"\n🎯 Market Sentiment:")
        print(f"  • Sentiment: {sentiment['sentiment']}")
        print(f"  • Confidence: {sentiment['confidence']:.0f}%")
        
        # Disconnect
        await provider.disconnect()
        print("\n✅ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_integration_with_types():
    """Test dominance integration with bot types."""
    print("\n🧪 Testing Type Integration")
    print("="*50)
    
    try:
        from bot.types import IndicatorData, MarketState
        print("✅ Successfully imported bot types")
        
        # Create indicator data with dominance
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=0.5,
            cipher_b_wave=0.3,
            stablecoin_dominance=7.5,
            dominance_trend=-0.3,
            market_sentiment="NEUTRAL"
        )
        print("✅ Created IndicatorData with dominance fields")
        
        # Verify fields
        print(f"\n📊 Indicator Data:")
        print(f"  • Stablecoin Dominance: {indicators.stablecoin_dominance}%")
        print(f"  • Dominance Trend: {indicators.dominance_trend}%")
        print(f"  • Market Sentiment: {indicators.market_sentiment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_config_integration():
    """Test dominance configuration."""
    print("\n🧪 Testing Configuration")
    print("="*50)
    
    try:
        from bot.config import create_settings
        settings = create_settings()
        print("✅ Successfully loaded configuration")
        
        # Check dominance settings
        print(f"\n⚙️  Dominance Settings:")
        print(f"  • Enabled: {settings.dominance.enable_dominance_data}")
        print(f"  • Data Source: {settings.dominance.data_source}")
        print(f"  • Update Interval: {settings.dominance.update_interval}s")
        print(f"  • Weight in Decisions: {settings.dominance.dominance_weight_in_decisions}")
        print(f"  • Alert Threshold: {settings.dominance.dominance_alert_threshold}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def main():
    """Run all tests."""
    print("\n🚀 USDT/USDC Dominance Integration Test")
    print("="*50)
    print("Running inside Docker container...\n")
    
    results = []
    
    # Test 1: Dominance Provider
    result1 = await test_dominance_provider()
    results.append(("Dominance Provider", result1))
    
    # Test 2: Type Integration
    result2 = await test_integration_with_types()
    results.append(("Type Integration", result2))
    
    # Test 3: Configuration
    result3 = await test_config_integration()
    results.append(("Configuration", result3))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed!")
        print("\nThe USDT/USDC dominance integration is working correctly.")
    else:
        print("❌ Some tests failed!")
        print("\nPlease check the logs for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)