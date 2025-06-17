#!/usr/bin/env python3
"""
Test script to verify the Bluefin market data provider fixes.
Tests the market data provider without the _public_client dependency.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for testing
os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
os.environ['SYSTEM__DRY_RUN'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

print("=" * 60)
print("BLUEFIN MARKET DATA PROVIDER FIX TEST")
print("=" * 60)
print()

async def test_market_data_provider():
    """Test BluefinMarketDataProvider functionality."""
    
    # Test 1: Import and initialization
    print("Test 1: Testing imports and initialization...")
    try:
        from bot.data.bluefin_market import BluefinMarketDataProvider
        
        provider = BluefinMarketDataProvider('ETH-USD', '5m')
        print(f"✅ Provider initialized for {provider.symbol} with {provider.interval} interval")
        print(f"   Candle limit: {provider.candle_limit}")
        
    except Exception as e:
        print(f"❌ Import/initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Connection test
    print("\nTest 2: Testing connection...")
    try:
        await provider.connect()
        print(f"✅ Connected successfully")
        print(f"   Connection status: {provider.is_connected()}")
        
        # Get status info
        status = provider.get_status()
        print("\nProvider Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Historical data fetch
    print("\nTest 3: Testing historical data fetch...")
    try:
        # This should work in dry run mode using mock data
        historical_data = await provider.fetch_historical_data()
        print(f"✅ Fetched {len(historical_data)} historical candles")
        
        if historical_data:
            latest = historical_data[-1]
            print(f"   Latest candle: {latest.timestamp} - Close: ${latest.close}")
        
    except Exception as e:
        print(f"❌ Historical data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Get latest OHLCV
    print("\nTest 4: Testing latest OHLCV...")
    try:
        ohlcv_data = await provider.get_latest_ohlcv(limit=10)
        print(f"✅ Retrieved {len(ohlcv_data)} OHLCV candles")
        
        if ohlcv_data:
            print("   Last 3 candles:")
            for candle in ohlcv_data[-3:]:
                print(f"     {candle.timestamp}: O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close}")
        
    except Exception as e:
        print(f"❌ OHLCV fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Convert to DataFrame
    print("\nTest 5: Testing DataFrame conversion...")
    try:
        df = provider.to_dataframe(limit=5)
        print(f"✅ Created DataFrame with {len(df)} rows")
        print("   Columns:", list(df.columns))
        
        if not df.empty:
            print("   Sample data:")
            print(df.tail(2).to_string())
        
    except Exception as e:
        print(f"❌ DataFrame conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Exchange client integration
    print("\nTest 6: Testing exchange client integration...")
    try:
        from bot.exchange.factory import create_exchange
        
        exchange = create_exchange(exchange_type='bluefin', dry_run=True)
        await exchange.connect()
        
        # Test historical candles method
        candles = await exchange.get_historical_candles('ETH-USD', '5m', 10)
        print(f"✅ Exchange client returned {len(candles)} candles")
        
        if candles:
            print("   Sample candle structure:")
            sample = candles[0] if isinstance(candles[0], dict) else {}
            for key in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                if key in sample:
                    print(f"     {key}: {sample[key]}")
        
        await exchange.disconnect()
        
    except Exception as e:
        print(f"❌ Exchange client integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    await provider.disconnect()
    print("\n✅ Disconnected successfully")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("BluefinMarketDataProvider is working correctly")
    print("The _public_client dependency issue has been resolved")
    print("=" * 60)
    
    return True


async def test_hybrid_provider():
    """Test HybridMarketDataProvider functionality."""
    print("\n" + "=" * 60)
    print("HYBRID MARKET DATA PROVIDER TEST")
    print("=" * 60)
    
    try:
        from bot.data.bluefin_market import HybridMarketDataProvider
        
        # Initialize hybrid provider
        hybrid = HybridMarketDataProvider('ETH-USD', '5m')
        print("✅ HybridMarketDataProvider initialized")
        
        # Test connection (this might fail if Coinbase provider has issues)
        print("\nTesting hybrid connection...")
        try:
            await hybrid.connect()
            print("✅ Hybrid provider connected")
            
            # Test data fetch
            data = await hybrid.get_latest_ohlcv(limit=5)
            print(f"✅ Retrieved {len(data)} candles from hybrid provider")
            
            await hybrid.disconnect()
            print("✅ Hybrid provider disconnected")
            
        except Exception as e:
            print(f"⚠️  Hybrid provider test failed (expected if Coinbase not configured): {e}")
        
    except Exception as e:
        print(f"❌ Hybrid provider test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        from bot.exchange.bluefin_client import BluefinServiceClient
        print("✅ BluefinServiceClient is available")
    except ImportError as e:
        print(f"⚠️  BluefinServiceClient import issue: {e}")
    
    try:
        import aiohttp
        print("✅ aiohttp is available")
    except ImportError:
        print("⚠️  aiohttp not available - some features may not work")
    
    print()
    
    # Run the test
    async def run_tests():
        success = await test_market_data_provider()
        if success:
            await test_hybrid_provider()
        return success
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)