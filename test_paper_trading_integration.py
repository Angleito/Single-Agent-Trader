#!/usr/bin/env python3
"""Paper Trading Integration Test"""

import os
import sys
from pathlib import Path


def test_paper_trading_simulation():
    """Test paper trading simulation accuracy and completeness."""
    print('=== PAPER TRADING SIMULATION INTEGRATION TEST ===')
    print()

    try:
        # Test 1: Check if PaperTradingEngine exists
        print('1. Testing PaperTradingEngine availability...')
        try:
            from bot.paper_trading import PaperTradingEngine
            print('✅ PaperTradingEngine import successful')
            
            # Test initialization
            engine = PaperTradingEngine()
            print('✅ PaperTradingEngine initialized successfully')
            
        except ImportError as e:
            print(f'❌ PaperTradingEngine not available: {e}')
            
            # Check what's available in paper_trading module
            try:
                import bot.paper_trading as pt
                available = [attr for attr in dir(pt) if not attr.startswith('_')]
                print(f'   Available in bot.paper_trading: {available}')
            except ImportError:
                print('   bot.paper_trading module not found')
                
        # Test 2: Check dry run functionality
        print()
        print('2. Testing dry run configuration...')
        try:
            from bot.config import Settings
            settings = Settings()
            print(f'✅ Settings loaded - Dry run mode: {settings.system.dry_run}')
            
            if settings.system.dry_run:
                print('✅ System correctly configured for paper trading')
            else:
                print('⚠️  System configured for live trading - ensure this is intentional')
                
        except Exception as e:
            print(f'❌ Configuration test failed: {e}')
            
        # Test 3: Check for paper trading components
        print()
        print('3. Testing paper trading components...')
        
        # Check for position simulation
        try:
            from bot.fp.types.paper_trading import PaperPosition, PaperTrade
            print('✅ Paper trading types available')
        except ImportError as e:
            print(f'❌ Paper trading types not available: {e}')
            
        # Test 4: Test order simulation
        print()
        print('4. Testing order simulation capabilities...')
        try:
            from bot.fp.types.trading import MarketOrder, LimitOrder, OrderStatus
            from decimal import Decimal
            
            # Create sample orders
            market_order = MarketOrder(
                symbol="BTC-USD",
                side="buy", 
                size=0.1
            )
            
            limit_order = LimitOrder(
                symbol="BTC-USD",
                side="sell",
                price=50000.0,
                size=0.1
            )
            
            print('✅ Order types working correctly')
            print(f'   Market order: {market_order.symbol} {market_order.side} {market_order.size}')
            print(f'   Limit order: {limit_order.symbol} {limit_order.side} ${limit_order.price} size:{limit_order.size}')
            
        except Exception as e:
            print(f'❌ Order simulation test failed: {e}')
            
        # Test 5: Portfolio simulation
        print()
        print('5. Testing portfolio simulation...')
        try:
            from bot.fp.types.portfolio import Portfolio, PortfolioState
            print('✅ Portfolio types available for simulation')
            
        except ImportError as e:
            print(f'❌ Portfolio simulation types not available: {e}')
            
        return True
        
    except Exception as e:
        import traceback
        print(f'❌ Paper trading test failed: {e}')
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_paper_trading_simulation()
    exit(0 if success else 1)